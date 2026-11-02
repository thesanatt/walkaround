"""R27 (11/2, hero_focus4): the privacy blur inside each face or plate's own mask (privacy.mask_blur), the crop gate
for a view mostly blurred (G12) and the marks left out under the blur, the keyframe pair's scale gap (G11), the twin
lamp check (G13) and the small panels' filmed floor (coverage.filmed_area_frac_min_by_panel)."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from walkaround import config, ledger, match, panels, paths, privacy, rules, sam_client
from walkaround import contracts as c
from walkaround.contracts import CoverageState, Keyframe, KeyframeScore, PanelId

from tests.test_privacy import FakeSam, _detection, _plate_track

TH = c.load_thresholds()


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    return root


def _text_frame(h: int = 400, w: int = 600) -> np.ndarray:
    frame = np.full((h, w, 3), 225, dtype=np.uint8)
    cv2.putText(frame, "ABC 1234", (180, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    return frame


def _high_freq(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(gray, cv2.CV_32F)).mean())


# Leaves and their absence --------------------------------------------------------------------------------------------


def test_leaves_on_in_config_and_absent_means_the_old_rules() -> None:
    assert privacy.mask_blur_settings(TH) is not None
    assert match.pair_scale(TH) is not None and match.privacy_blur(TH) is not None and match.twin_mirror(TH) is not None
    old = copy.deepcopy(TH)
    old["privacy"].pop("mask_blur")
    for leaf in ("pair_scale", "privacy_blur", "twin_mirror"):
        old["matching"].pop(leaf)
    assert privacy.mask_blur_settings(old) is None
    assert match.pair_scale(old) is None and match.privacy_blur(old) is None and match.twin_mirror(old) is None
    assert "track_carry_grow_max" not in privacy.carry_params(old) and "carry_nouns" not in privacy.carry_params(old)
    for gate in ("G11", "G12", "G13"):
        assert rules.rule(gate).label
    assert set(("G11", "G12", "G13")) <= set(match.VIEW_GATES)


def test_web_rules_list_the_new_gates() -> None:
    text = (Path(__file__).resolve().parent.parent / "web" / "src" / "lib" / "rules.ts").read_text(encoding="utf-8")
    for gate in ("G11", "G12", "G13"):
        assert f"  {gate}: {{ label: \"{rules.rule(gate).label}\"" in text


# The smooth blur ------------------------------------------------------------------------------------------------------


def test_smooth_blur_inside_the_mask_only_and_text_is_gone() -> None:
    frame = _text_frame()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[180:230, 170:430] = True  # the "plate"
    out, where = privacy.blur_regions(frame, [(mask, 260.0, False)], th=TH)
    assert np.array_equal(out[~where], frame[~where])
    assert np.array_equal(where, mask)  # not dilated when asked not to
    assert _high_freq(out[180:230, 170:430]) < 0.15 * _high_freq(frame[180:230, 170:430])
    # smooth, not blocks: no 2 x 2 tile repeats a single colour everywhere, unlike pixelate
    sub = out[185:225, 180:420].astype(int)
    assert np.abs(np.diff(sub, axis=1)).max() <= 30


def test_dilation_is_small_and_follows_the_object() -> None:
    frame = _text_frame()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[180:230, 170:430] = True
    _, where = privacy.blur_regions(frame, [(mask, 260.0, True)], th=TH)
    grown = int(where.sum()) - int(mask.sum())
    ys, xs = np.nonzero(where)
    mb = privacy.mask_blur_settings(TH)
    reach = 260.0 * mb["dilate_frac"] + mb["dilate_px_at_4k"] * 600 / 3840
    assert 0 < grown <= 2 * (260 + 50) * (reach + 1) + 4 * (reach + 1) ** 2
    assert xs.min() >= 170 - reach - 1 and xs.max() <= 430 + reach + 1


# Track masks: faces not carried, plates carried with a cap -------------------------------------------------------------


def test_face_track_is_not_carried_and_plate_carry_growth_is_capped() -> None:
    face = _plate_track(list(range(10, 20)), [100.0, 50.0, 140.0, 90.0], noun=c.Noun.FACE)
    plate = _plate_track(list(range(10, 20)), [300.0, 200.0, 340.0, 212.0])
    common = dict(fps=15.0, media_size_proxy=(640, 360), media_size_src=(640, 360))
    old = copy.deepcopy(TH)
    old["privacy"].pop("mask_blur")
    assert privacy.track_boxes_at(40, [face], th=old, **common)  # the box rule carried the face 3 s
    assert privacy.track_boxes_at(40, [face], th=TH, **common) == []
    boxes = privacy.track_boxes_at(55, [plate], th=TH, **common)  # 2.4 s after the last sighting
    assert boxes
    w = boxes[0][2] - boxes[0][0]
    assert w <= 40.0 * (1.0 + privacy.mask_blur_settings(TH)["plate_carry_grow_max"]) + 1e-6
    assert privacy.track_boxes_at(15, [plate], th=TH, sightings=False, **common) == []


def _write_lane(runs: Path, run_id: str, role: str, noun: c.Noun, frames: dict[int, np.ndarray], size: tuple[int, int]) -> None:
    """A stored lane: the index with one object per frame and the events file carrying its one_bit masks."""
    from meta_sam_parser._mask_codec import _encode_raster

    sam = paths.sam_dir(run_id, role)
    index = []
    events = []
    for f, mask in sorted(frames.items()):
        ys, xs = np.nonzero(mask)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        local = mask[y1:y2, x1:x2].astype(np.uint8)
        payload = _encode_raster(bytes(local.flatten().tolist()), x2 - x1, y2 - y1)
        index.append({"frame": f, "objects": [{"id": "0", "box": [x1, y1, x2, y2], "c": 0.8, "mask": {"encoding": "one_bit", "x": 0, "y": 0, "w": x2 - x1, "h": y2 - y1, "offset": 0, "length": len(payload)}}]})
        events.append(payload)
    return index, events


def test_track_masks_blur_the_shape_not_the_box(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id, role, size = "r27_masks", "pickup", (640, 360)
    ring = np.zeros((size[1], size[0]), dtype=bool)
    cv2.circle(ring.view(np.uint8), (200, 150), 60, 1, thickness=-1)
    face = _plate_track([10], [140.0, 90.0, 261.0, 211.0], noun=c.Noun.FACE)

    class Masks:
        media_size = size

        def track_mask(self, track: c.PanelTrack, frame: int) -> np.ndarray | None:
            return ring if frame == 10 else None

        def on_car(self, track: c.PanelTrack, settings: dict[str, Any]) -> bool:
            return False

        def car_mask(self, frame: int) -> np.ndarray | None:
            return None

    frame = np.random.default_rng(3).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    out, where = privacy.blur_keyframe_masks(frame, f_proxy=10, tracks=[face], fps=15.0, proxy_size=size, masks=Masks(), plates=[], faces=[], th=TH)
    corner = (slice(92, 100), slice(142, 150))  # inside the box, outside the round mask and its small dilation
    assert not where[corner].any() and np.array_equal(out[corner], frame[corner])
    assert where[150, 200] and where[150, 145]
    _, where_reflection = privacy.blur_keyframe_masks(
        frame, f_proxy=10, tracks=[face], fps=15.0, proxy_size=size, masks=type("M", (Masks,), {"on_car": lambda self, t, s: True})(), plates=[], faces=[], th=TH
    )
    assert np.array_equal(where_reflection, ring)  # a face on the car's glass or paint: inside its mask only


def test_image_check_face_on_the_car_is_not_dilated() -> None:
    frame = np.full((300, 400, 3), 200, dtype=np.uint8)
    face_mask = np.zeros((300, 400), dtype=bool)
    face_mask[100:140, 150:190] = True
    car = np.zeros((150, 200), dtype=bool)  # a proxy-size car mask, half the frame size
    car[25:125, 25:175] = True
    det = _detection(face_mask)
    on_car = privacy.detection_regions([det], frame.shape, car_mask=car, th=TH, faces=True)
    off_car = privacy.detection_regions([det], frame.shape, car_mask=None, th=TH, faces=True)
    assert on_car[0][2] is False and off_car[0][2] is True


def test_blur_keyframe_records_what_it_blurred(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _text_frame(360, 640)
    plate_mask = np.zeros((360, 640), dtype=bool)
    plate_mask[180:230, 170:430] = True
    fake = FakeSam([_detection(plate_mask)], [])
    monkeypatch.setattr(sam_client, "segment_image", fake.segment_image)
    blurred, record = asyncio.run(
        privacy.blur_keyframe("r27_kf", "pickup", frame, f_proxy=5, tracks=[], fps=15.0, sizes=((640, 360), (640, 360)), keep_plate=False, th=TH)
    )
    where = record.blurred_mask()
    assert where is not None and where[200, 300] and not where[20, 20]
    assert _high_freq(blurred[185:225, 180:420]) < 0.15 * _high_freq(frame[185:225, 180:420])
    assert np.array_equal(blurred[~where], frame[~where])
    assert "_blurred" not in record.to_json()
    old = copy.deepcopy(TH)
    old["privacy"].pop("mask_blur")
    _, old_record = asyncio.run(
        privacy.blur_keyframe("r27_kf", "pickup", frame, f_proxy=5, tracks=[], fps=15.0, sizes=((640, 360), (640, 360)), keep_plate=False, th=old)
    )
    assert old_record.blurred_mask() is None


# Keyframe contract -----------------------------------------------------------------------------------------------------


def _kf(kf_id: str, role: str, panel_id: PanelId = PanelId.LEFT_FRONT_DOOR, **extra: Any) -> Keyframe:
    return Keyframe(
        kf_id=kf_id, role=role, panel_id=panel_id, track_id=f"{role}_{panel_id.value}_t1", f_proxy=10, pts=0.5,
        score=KeyframeScore(area=extra.pop("area", 0.1), completeness=1.0, sharpness=1.0, glare=0.0, frontality=1.0, total=0.1),
        aspect_ratio=1.2, box_src=[0.0, 0.0, 100.0, 100.0], front_sign=1, **extra,
    )


def test_keyframe_serializes_as_before_without_the_new_fields() -> None:
    plain = _kf("pickup_left_front_door_kf1", "pickup")
    assert "privacy_blurred_frac" not in plain.to_json() and "blur_png" not in plain.to_json()
    marked = _kf("pickup_left_front_door_kf1", "pickup", privacy_blurred_frac=0.3, blur_png="crops/pickup/x.blur.png")
    again = Keyframe.from_json(marked.to_json())
    assert again.privacy_blurred_frac == 0.3 and again.blur_png == "crops/pickup/x.blur.png"


# Coverage: small panels ------------------------------------------------------------------------------------------------


def test_small_panels_have_their_own_filmed_floor() -> None:
    cov = TH["coverage"]
    lamp = _kf("pickup_headlights_kf1", "pickup", PanelId.HEADLIGHTS, area=0.02)
    door = _kf("pickup_left_front_door_kf1", "pickup", area=0.02)
    assert panels._is_filmed(lamp, cov) and not panels._is_filmed(door, cov)
    old = {k: v for k, v in cov.items() if k != "filmed_area_frac_min_by_panel"}
    assert not panels._is_filmed(lamp, old)


# Gates -------------------------------------------------------------------------------------------------------------------


def _panel(pid: PanelId = PanelId.LEFT_FRONT_DOOR) -> c.Panel:
    return c.Panel(
        panel_id=pid, noun=c.Noun.CAR_DOOR, side=c.Side.LEFT, position=c.Position.FRONT, front_sign=1,
        per_role={r: c.PanelRole(state=CoverageState.FILMED) for r in ("pickup", "return")},
    )


def _gate_inputs(**over: Any) -> dict[str, Any]:
    cal = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=TH)
    base: dict[str, Any] = {
        "mask_iou": 0.9, "residual": None,
        "lane_status": {r: {c.Noun.CAR_DOOR: c.LaneStatus.COMPLETED} for r in ("pickup", "return")},
        "marks_truncated_all": {r: False for r in ("pickup", "return")}, "calibration": cal, "th": TH,
    }
    base.update(over)
    return base


def test_g11_scale_gap_g12_blurred_view_g13_other_lamp() -> None:
    p, r = _kf("p", "pickup"), _kf("r", "return")
    assert match.panel_gates(_panel(), p, r, **_gate_inputs(scale_ratio=1.9)) == []
    assert match.panel_gates(_panel(), p, r, **_gate_inputs(scale_ratio=2.5)) == ["G11"]
    blurred = _kf("p", "pickup", privacy_blurred_frac=0.35)
    assert match.panel_gates(_panel(), blurred, r, **_gate_inputs()) == ["G12"]
    assert match.panel_gates(_panel(), _kf("p", "pickup", privacy_blurred_frac=0.1), r, **_gate_inputs()) == []
    lamps = _panel(PanelId.HEADLIGHTS)
    assert match.panel_gates(lamps, p, r, **_gate_inputs(mirrored=(0.45, 0.91))) == ["G13"]
    assert match.panel_gates(lamps, p, r, **_gate_inputs(mirrored=(0.80, 0.35))) == []
    assert match.panel_gates(_panel(), p, r, **_gate_inputs(mirrored=(0.45, 0.91))) == []  # doors are no twins
    old = copy.deepcopy(TH)
    for leaf in ("pair_scale", "privacy_blur", "twin_mirror"):
        old["matching"].pop(leaf)
    assert match.panel_gates(lamps, blurred, r, **_gate_inputs(th=old, scale_ratio=3.0, mirrored=(0.4, 0.9))) == []


def test_mirror_ious_tell_a_left_lamp_from_a_right_one() -> None:
    lamp = np.zeros((60, 200), dtype=bool)
    pts = np.array([[5, 50], [195, 5], [195, 25], [40, 58]], dtype=np.int32)
    cv2.fillPoly(lamp.view(np.uint8), [pts], 1)
    same = match.mirror_ious(lamp, lamp)
    other = match.mirror_ious(lamp, lamp[:, ::-1])
    assert same[0] > same[1] + 0.1 and other[1] > other[0] + 0.1


def test_pair_order_puts_similar_scales_first() -> None:
    p1, p2 = _kf("pickup_left_front_door_kf1", "pickup"), _kf("pickup_left_front_door_kf2", "pickup")
    r1 = _kf("return_left_front_door_kf1", "return")
    scales = {p1.kf_id: 1900.0, p2.kf_id: 500.0, r1.kf_id: 480.0}
    order = match.keyframe_pair_order([p1, p2], [r1], th=TH, scales=scales)
    assert [a.kf_id for a, _ in order][0] == p2.kf_id
    assert match.scale_ratio(1900.0, 480.0) > 2.0 and match.scale_ratio(None, 3.0) is None


def test_keyframe_scales_use_the_mask_box_and_the_wheel_for_built_fenders(runs: Path) -> None:
    run_id = "r27_scales"
    crops = paths.crops_dir(run_id, "pickup")
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[50:150, 20:380] = 255
    cv2.imwrite(str(crops / "door.mask.png"), mask)
    door = _kf("pickup_left_front_door_kf1", "pickup", crop=c.Crop(path="crops/pickup/door.jpg", sha256="0" * 64, dx=0, dy=0, w=400, h=300, scale=0.5), mask_png="crops/pickup/door.mask.png")
    fender = Keyframe(
        kf_id="pickup_right_rear_fender_kf1", role="pickup", panel_id=PanelId.RIGHT_REAR_FENDER, track_id="pickup_fender_from_wheel_04", f_proxy=7, pts=0.5,
        score=KeyframeScore(area=0.1, completeness=1.0, sharpness=1.0, glare=0.0, frontality=1.0, total=0.1), aspect_ratio=1.0,
        box_src=[0.0, 0.0, 400.0, 200.0], front_sign=1,
    )
    row = lambda box: {"frame": 7, "pts": 0.5, "box": box}  # noqa: E731
    tracks = [
        {"track_id": "pickup_fender_from_wheel_04", "per_frame": [row([0, 0, 200, 100])]},
        {"track_id": "pickup_wheel_04", "per_frame": [row([10, 10, 130, 110])]},
    ]
    paths.tracks_json(run_id).write_text(__import__("json").dumps(tracks), encoding="utf-8")
    got = match.keyframe_scales(run_id, [door, fender])
    assert got[door.kf_id] == pytest.approx(720.0)  # 360 crop px at scale 0.5
    assert got[fender.kf_id] == pytest.approx(240.0)  # the wheel box's 120 proxy px times 2


def test_marks_under_the_blur_are_left_out_with_a_line(runs: Path) -> None:
    run_id = "r27_drops"
    crops = paths.crops_dir(run_id, "pickup")
    blur = np.zeros((300, 400), dtype=np.uint8)
    blur[:150, :] = 255  # the top half of the crop blurred
    cv2.imwrite(str(crops / "k.blur.png"), blur)
    kf = _kf("pickup_left_front_door_kf1", "pickup", crop=c.Crop(path="crops/pickup/k.jpg", sha256="0" * 64, dx=0, dy=0, w=400, h=300, scale=1.0), blur_png="crops/pickup/k.blur.png")

    def mark(mid: str, y0: int, y1: int) -> c.Mark:
        return c.Mark.model_construct(mark_id=mid, kf_id=kf.kf_id, grid_bbox=c.GridBox(x_min=100, y_min=y0, x_max=200, y_max=y1))

    drops = match.privacy_drops(run_id, [mark("m_top", 100, 300), mark("m_low", 700, 900)], {kf.kf_id: kf}, TH)
    assert set(drops) == {"m_top"} and drops["m_top"].share == pytest.approx(1.0)
    words = match.privacy_drop_words(list(drops.values()))
    assert "blurred for privacy" in words and "m_top" in words


def test_keyframe_records_the_blurred_share_of_its_panel(runs: Path) -> None:
    run_id, role = "r27_kf_files", "pickup"
    media = (320, 180)
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    mask = np.zeros((180, 320), dtype=bool)
    mask[40:140, 60:260] = True  # the panel, proxy px
    where = np.zeros((360, 640), dtype=bool)
    where[80:180, 120:520] = True  # the top half of the panel blurred, source px
    record = c.BlurRecord()
    record.set_blurred(where)
    track = _plate_track([12], [60.0, 40.0, 260.0, 140.0])
    track = track.model_copy(update={"noun": c.Noun.CAR_DOOR, "track_id": "pickup_car_door_00"})
    score = KeyframeScore(area=0.3, completeness=1.0, sharpness=1.0, glare=0.0, frontality=1.0, total=0.3)
    kf = panels._keyframe_files(run_id, role, "pickup_left_front_door_kf1", _panel(), track, 12, mask, False, frame, record, score, None, media, TH)
    assert kf.privacy_blurred_frac == pytest.approx(0.5, abs=0.02)
    assert kf.blur_png and (paths.run_dir(run_id) / kf.blur_png).is_file()
    plain = panels._keyframe_files(run_id, role, "pickup_left_front_door_kf2", _panel(), track, 12, mask, False, frame, c.BlurRecord(), score, None, media, TH)
    assert plain.privacy_blurred_frac is None and plain.blur_png is None
