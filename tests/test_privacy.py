"""privacy (M12): pixelation, the plate extent, the image checks through a fake sam_client, the step-7 order, the
plate always pixelated in stored frames, track boxes, deletions with their receipts, the quarantine, the sweeper,
the serving allowlist and the privacy line. LIVE=1 adds one real pair of image checks on the ping pong clip (two
sam_image rows, about $0.005)."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from meta_sam_parser import SegmentationMask
from meta_sam_parser._mask_codec import _encode_raster

from walkaround import config
from walkaround import contracts as c
from walkaround import ledger, paths, privacy, sam_client

REPO = Path(__file__).resolve().parent.parent
TH = c.load_thresholds()
NOW = "2026-10-27T12:00:00Z"


# Fixtures and helpers --------------------------------------------------------------------------------------------------


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    return root


def _noise(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _rect_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def _rotated_rect_mask(shape: tuple[int, int], center: tuple[int, int], size: tuple[int, int], angle: float) -> np.ndarray:
    pts = cv2.boxPoints((center, size, angle)).astype(np.int32)
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas.astype(bool)


def _bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _detection(mask: np.ndarray, object_id: str = "0", confidence: float | None = 0.9) -> sam_client.Detection:
    """A Detection whose one_bit mask is the real wire encoding of the box-local raster."""
    x1, y1, x2, y2 = (int(v) for v in _bbox(mask))
    local = mask[y1:y2, x1:x2].astype(np.uint8)
    payload = _encode_raster(bytes(local.flatten().tolist()), x2 - x1, y2 - y1)
    seg = SegmentationMask(encoding="one_bit", payload=payload, width=x2 - x1, height=y2 - y1)
    return sam_client.Detection(object_id=object_id, frame_index=None, box=sam_client.Box(x1, y1, x2, y2), mask=seg, confidence=confidence)


def _sha(rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


class FakeSam:
    """Stands in for sam_client.segment_image: records what it saw and answers with the given detections."""

    def __init__(self, plates: list[sam_client.Detection], faces: list[sam_client.Detection]) -> None:
        self.plates, self.faces = plates, faces
        self.calls: list[tuple[str, str, str | None, str | None, str]] = []

    async def segment_image(self, image, concept: str, *, run_id, role=None, tag="", include_confidence=True):
        self.calls.append((concept, _sha(image), run_id, role, tag))
        dets = self.plates if concept == c.Noun.LICENSE_PLATE.value else self.faces
        run = sam_client.SegmentationRun(concept=concept, media="image", media_width=int(image.shape[1]), media_height=int(image.shape[0]))
        if dets:
            run.frames[0] = list(dets)
        run.response_id = "resp_plate" if concept == c.Noun.LICENSE_PLATE.value else "resp_face"
        return run


def _changed(a: np.ndarray, b: np.ndarray) -> float:
    return float((a != b).any(axis=2).mean())


def _make_clip(path: Path, *, w: int = 320, h: int = 180, fps: int = 15, seconds: float = 0.5) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}", "-t", str(seconds),
           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(path)]
    subprocess.run(cmd, check=True)


def _plate_track(frames: list[int], box: list[float], *, role: str = "pickup", noun: c.Noun = c.Noun.LICENSE_PLATE) -> c.PanelTrack:
    return c.PanelTrack(
        track_id=f"{role}_{c.noun_slug(noun)}_0", role=role, noun=noun, sam_object_ids=["0"],
        fragments=[c.TrackFragment(object_id="0", first_frame=min(frames), last_frame=max(frames))],
        per_frame=[c.FrameObs(frame=f, pts=f / 15.0, box=list(box), box_area_frac=0.002, in_car_box=True, c=0.7) for f in frames],
    )


# Pixelation ---------------------------------------------------------------------------------------------------------------


def test_pixelate_region_unrecognisable_rest_untouched() -> None:
    rgb = _noise(240, 320, seed=1)
    out = privacy.pixelate(rgb, [[80.0, 60.0, 200.0, 150.0]], block_px=10)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert not np.shares_memory(out, rgb)
    region = _rect_mask((240, 320), (80, 60, 200, 150))
    assert np.array_equal(out[~region], rgb[~region])
    inside_out, inside_in = out[60:150, 80:200], rgb[60:150, 80:200]
    blocks = inside_out.reshape(9, 10, 12, 10, 3)
    assert (blocks == blocks[:, :1, :, :1, :]).all()  # one colour per 10 x 10 block
    assert _changed(inside_out, inside_in) > 0.95
    assert len(np.unique(inside_in.reshape(-1, 3), axis=0)) > 5000
    assert len(np.unique(inside_out.reshape(-1, 3), axis=0)) <= 9 * 12


def test_pixelate_text_is_gone() -> None:
    frame = np.full((200, 400, 3), 230, dtype=np.uint8)
    cv2.putText(frame, "ABC 1234", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (20, 20, 20), 8, cv2.LINE_AA)
    gray_in = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    assert (gray_in < 100).sum() > 2000  # the strokes
    fixed = cv2.cvtColor(privacy.pixelate(frame, [[0.0, 0.0, 400.0, 200.0]], block_px=24, blocks_across=None), cv2.COLOR_RGB2GRAY)
    assert fixed.min() > 20  # no pixel keeps the stroke colour: every one is a block mean
    blocks = fixed[:192, :384].reshape(8, 24, 16, 24)
    assert (blocks == blocks[:, :1, :, :1]).all()  # one value per 24 px block
    grown = cv2.cvtColor(privacy.pixelate(frame, [[0.0, 0.0, 400.0, 200.0]], block_px=24), cv2.COLOR_RGB2GRAY)
    big = grown.reshape(2, 100, 4, 100)
    assert (big == big[:, :1, :, :1]).all()  # by default the 400 px box spans 4 blocks of 100 px (box_blocks_across)


def test_pixelate_dilation_grows_the_region_and_keeps_the_rest() -> None:
    rgb = _noise(240, 320, seed=2)
    out = privacy.pixelate(rgb, [[100.0, 100.0, 120.0, 120.0]], block_px=4, dilate_frac=0.5, dilate_px=6)
    grown = _rect_mask((240, 320), (84, 84, 136, 136))  # 20 px box x 0.5 + 6 px per side
    assert _changed(out[84:136, 84:136], rgb[84:136, 84:136]) > 0.9
    assert np.array_equal(out[~grown], rgb[~grown])
    assert np.array_equal(privacy.pixelate(rgb, [], block_px=4), rgb)
    assert np.array_equal(privacy.pixelate(rgb, [[50.0, 50.0, 50.0, 90.0]], block_px=4), rgb)  # an empty box
    with pytest.raises(ValueError):
        privacy.pixelate(rgb, [[0.0, 0.0, 10.0, 10.0]], block_px=0)


def test_pixelate_far_apart_regions_one_box_each() -> None:
    frame = _noise(2160, 3840, seed=9)
    where = np.zeros((2160, 3840), dtype=bool)
    where[10:200, 10:200] = True  # a face in one corner
    where[1900:2150, 3500:3830] = True  # a plate in the other
    out = frame.copy()
    tracemalloc.start()
    privacy._pixelate_where(out, where, 20)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 100e6  # one union box over the whole 4K frame peaked at about 800 MB
    assert _changed(out[10:200, 10:200], frame[10:200, 10:200]) > 0.95
    assert _changed(out[1900:2150, 3500:3830], frame[1900:2150, 3500:3830]) > 0.95
    assert np.array_equal(out[~where], frame[~where])


def test_dilate_mask_covers_the_elliptical_kernel() -> None:
    mask = _rotated_rect_mask((400, 600), (300, 200), (120, 60), 20)
    grown = privacy._dilate_mask(mask, dilate_frac=0.1, dilate_px=6)
    r = int(math.ceil(max(_bbox(mask)[2] - _bbox(mask)[0], _bbox(mask)[3] - _bbox(mask)[1]) * 0.1 + 6))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ellipse = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    assert not (ellipse & ~grown).any()  # every pixel the kernel reaches is covered
    assert (grown & ~ellipse).sum() < 0.01 * ellipse.sum()  # and barely more
    assert np.array_equal(privacy._dilate_mask(np.zeros((10, 10), dtype=bool), dilate_frac=0.1, dilate_px=6), np.zeros((10, 10), dtype=bool))


# Plate extent -----------------------------------------------------------------------------------------------------------


def test_plate_extent_measures_long_side_and_tilt() -> None:
    frontal = _rotated_rect_mask((400, 600), (300, 200), (240, 120), 25)
    long_px, hw, tilt = privacy.plate_extent(frontal)
    assert abs(long_px - 240) < 240 * 0.03
    assert abs(hw - 0.5) < 0.03
    assert tilt < 12  # acos is steep near h / w = 0.5, so a 1% extent error reads as several degrees
    yawed = _rotated_rect_mask((400, 600), (300, 200), (144, 120), 0)  # h / w 0.833: cos theta = 0.6
    _, hw2, tilt2 = privacy.plate_extent(yawed)
    assert abs(hw2 - 0.833) < 0.03
    assert abs(tilt2 - math.degrees(math.acos(0.6))) < 3
    assert privacy.plate_extent(np.zeros((10, 10), dtype=bool)) is None


def test_plate_extent_read_from_a_pixelated_region_overstates_by_over_20_percent() -> None:
    frame = _noise(720, 1280, seed=3)
    plate = _rotated_rect_mask((720, 1280), (640, 400), (200, 100), 0)
    frame[plate] = (240, 240, 240)
    true_long = privacy.plate_extent(plate)[0]
    blurred = privacy.pixelate(
        frame, [_bbox(plate)], block_px=20,
        dilate_frac=TH["privacy"]["track_box_dilation_frac"], dilate_px=TH["privacy"]["track_box_dilation_px"],
    )
    region = (blurred != frame).any(axis=2)  # what a reader of the blurred frame would call the plate
    assert privacy.plate_extent(region)[0] / true_long > 1.2


# Image checks and the blur order -------------------------------------------------------------------------------------------


async def test_image_checks_record_the_plate_extent_from_the_unblurred_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _noise(720, 1280, seed=4)
    plate = _detection(_rect_mask((720, 1280), (500, 350, 700, 450)), confidence=0.91)
    face = _detection(_rect_mask((720, 1280), (100, 100, 160, 180)), confidence=0.8)
    fake = FakeSam([plate], [face])
    monkeypatch.setattr(sam_client, "segment_image", fake.segment_image)
    plates, faces, record = await privacy.image_checks(frame, run_id="r1", role="pickup", tag="f000003", th=TH)
    assert [call[0] for call in fake.calls] == ["license plate", "face"]
    assert all(call[1] == _sha(frame) for call in fake.calls)  # the unblurred pixels went to SAM
    assert [call[2:] for call in fake.calls] == [("r1", "pickup", "f000003.license_plate"), ("r1", "pickup", "f000003.face")]
    assert plates == [plate] and faces == [face]
    assert record.plate_boxes == [[500.0, 350.0, 700.0, 450.0]]
    assert record.face_boxes == [[100.0, 100.0, 160.0, 180.0]]
    assert record.request_ids == ["resp_plate", "resp_face"]
    assert record.plate_ruler_px is not None and abs(record.plate_ruler_px - 200) < 6
    assert record.plate_ruler_c == 0.91
    assert abs(record.plate_tilt_hw - 0.5) < 0.03


async def test_image_checks_ruler_refuses_low_c_and_border_plates(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _noise(720, 1280, seed=5)
    low_c = _detection(_rect_mask((720, 1280), (500, 350, 700, 450)), confidence=0.3)
    on_border = _detection(_rect_mask((720, 1280), (0, 350, 200, 450)), confidence=0.95)
    fake = FakeSam([low_c, on_border], [])
    monkeypatch.setattr(sam_client, "segment_image", fake.segment_image)
    plates, faces, record = await privacy.image_checks(frame, run_id="r1", role="pickup", tag="t", th=TH)
    assert len(plates) == 2 and faces == []
    assert len(record.plate_boxes) == 2 and record.face_boxes == []
    assert record.plate_ruler_px is None and record.plate_ruler_c is None and record.plate_tilt_hw is None


async def test_blur_keyframe_checks_first_then_pixelates_the_plate_even_under_keep_plate(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _noise(720, 1280, seed=6)
    plate_mask = _rect_mask((720, 1280), (500, 350, 700, 450))
    face_mask = _rect_mask((720, 1280), (100, 100, 160, 180))
    fake = FakeSam([_detection(plate_mask, confidence=0.9)], [_detection(face_mask, confidence=0.8)])
    monkeypatch.setattr(sam_client, "segment_image", fake.segment_image)
    sizes = ((640, 360), (1280, 720))
    before = frame.copy()
    blurred, record = await privacy.blur_keyframe("r1", "pickup", frame, f_proxy=3, tracks=[], fps=15.0, sizes=sizes, keep_plate=False, th=TH)
    assert np.array_equal(frame, before)  # the caller's frame is untouched
    assert all(call[1] == _sha(before) for call in fake.calls)  # SAM saw the frame before any pixel changed
    assert record.plate_ruler_px is not None and abs(record.plate_ruler_px - 200) < 6
    assert _changed(blurred[350:450, 500:700], before[350:450, 500:700]) > 0.9  # the plate mask is pixelated
    assert _changed(blurred[100:180, 100:160], before[100:180, 100:160]) > 0.9  # so is the face
    assert np.array_equal(blurred[600:720, 900:1280], before[600:720, 900:1280])  # far from both masks
    # keep_plate never reaches a stored crop or a Spark input: only the sheet writer's blur_frame call keeps it
    kept, record2 = await privacy.blur_keyframe("r1", "pickup", frame, f_proxy=3, tracks=[], fps=15.0, sizes=sizes, keep_plate=True, th=TH)
    assert np.array_equal(kept, blurred)
    assert _changed(kept[350:450, 500:700], before[350:450, 500:700]) > 0.9
    assert record2.plate_ruler_px == record.plate_ruler_px
    owner = privacy.blur_frame(before, track_boxes_src=[], detections=fake.faces, keep_plate_region=True, th=TH, plate_detections=fake.plates)
    assert np.array_equal(owner[350:450, 500:700], before[350:450, 500:700])  # the owner's variant keeps it
    assert _changed(owner[100:180, 100:160], before[100:180, 100:160]) > 0.9  # faces are always pixelated
    with pytest.raises(ValueError):
        await privacy.blur_keyframe("r1", "pickup", frame, f_proxy=3, tracks=[], fps=15.0, sizes=((640, 360), (1920, 1080)), keep_plate=False, th=TH)


def test_blur_frame_pixelates_track_boxes_scaled_from_the_proxy() -> None:
    frame = _noise(360, 640, seed=7)
    track = _plate_track([0, 1, 2], [10.0, 20.0, 30.0, 40.0])
    boxes = privacy.track_boxes_at(1, [track], fps=15.0, media_size_proxy=(160, 90), media_size_src=(640, 360), th=TH)
    assert boxes == [[40.0, 80.0, 120.0, 160.0]]
    out = privacy.blur_frame(frame, track_boxes_src=boxes, detections=[], keep_plate_region=False, th=TH)
    assert _changed(out[80:160, 40:120], frame[80:160, 40:120]) > 0.9
    assert np.array_equal(out[250:360, 300:640], frame[250:360, 300:640])


def _inside(inner: list[float], outer: list[float], tol: float = 1e-6) -> bool:
    return outer[0] <= inner[0] + tol and outer[1] <= inner[1] + tol and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol


def test_track_boxes_at_holds_the_last_box_and_ignores_other_nouns() -> None:
    plate = _plate_track([0, 1, 2, 3, 10], [10.0, 20.0, 30.0, 40.0])
    car = _plate_track([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.0, 0.0, 100.0, 100.0], noun=c.Noun.CAR)
    kw = dict(fps=15.0, media_size_proxy=(100, 100), media_size_src=(400, 200), th=TH)
    base = [40.0, 40.0, 120.0, 80.0]
    assert privacy.track_boxes_at(3, [plate, car], **kw) == [base]  # a sighting, and its neighbours give the same box
    at5 = privacy.track_boxes_at(5, [plate, car], **kw)  # 2 frames after a sighting, 5 before the next
    assert at5[0] == base and len(at5) == 3  # the held box, then the carries from frame 3 (forward) and frame 10 (back)
    assert all(_inside(base, b) for b in at5)
    assert base in privacy.track_boxes_at(11, [plate, car], **kw)  # frame 10 is within track_box_union_s, and held
    assert privacy.track_boxes_at(5, [car], **kw) == []


def test_track_boxes_at_carries_before_the_first_and_after_the_last_sighting() -> None:
    th = TH
    fps = 15.0
    carry = math.ceil(th["privacy"]["track_carry_s"] * fps - 1e-9)
    bridge = math.ceil(th["privacy"]["track_gap_bridge_s"] * fps)
    plate = _plate_track(list(range(40, 61)), [100.0, 100.0, 140.0, 120.0])
    kw = dict(fps=fps, media_size_proxy=(960, 540), media_size_src=(960, 540), th=th)
    assert carry == 30 and bridge == 8
    before = [privacy.track_boxes_at(40 - d, [plate], **kw) for d in range(1, carry + 1)]
    assert all(len(b) == (2 if d <= 2 else 1) for d, b in enumerate(before, start=1))  # the first sighting is in reach for 2 frames
    assert all(_inside([100.0, 100.0, 140.0, 120.0], b[-1]) for b in before)  # the carried box: an edge-on plate SAM missed
    widths = [b[-1][2] - b[-1][0] for b in before]
    assert widths == sorted(widths) and widths[-1] == pytest.approx(40 * (1 + th["privacy"]["track_carry_grow_per_s"] * carry / fps), abs=0.01)
    assert privacy.track_boxes_at(40 - carry - 1, [plate], **kw) == []  # bounded in time
    after = privacy.track_boxes_at(60 + bridge + 1, [plate], **kw)
    assert len(after) == 1 and _inside([100.0, 100.0, 140.0, 120.0], after[0])  # past the hold, still carried
    assert privacy.track_boxes_at(60 + carry + 1, [plate], **kw) == []


def test_track_boxes_at_extrapolates_a_moving_box_and_takes_flow_boxes_when_given() -> None:
    fps = 15.0
    rows = list(range(30, 45))
    track = _plate_track(rows, [0.0, 0.0, 1.0, 1.0])
    for r in track.per_frame:  # moving right 4 px a frame
        x = 200.0 + 4.0 * (r.frame - 30)
        r.box = [x, 100.0, x + 40.0, 120.0]
    kw = dict(fps=fps, media_size_proxy=(960, 540), media_size_src=(960, 540), th=TH)
    [b] = privacy.track_boxes_at(20, [track], **kw)  # 10 frames before the first sighting
    assert (b[0] + b[2]) / 2 == pytest.approx(220.0 - 40.0, abs=0.01)  # centre moved back 10 x 4 px
    assert (b[1] + b[3]) / 2 == pytest.approx(110.0, abs=0.01)
    flow = {(track.track_id, 20, -1): [10.0, 10.0, 50.0, 30.0]}
    assert privacy.track_boxes_at(20, [track], carried=flow, **kw) == [[10.0, 10.0, 50.0, 30.0]]
    glitch = _plate_track([0, 1, 2, 3, 4], [100.0, 100.0, 160.0, 130.0])
    glitch.per_frame[2].box = [150.0, 120.0, 152.0, 122.0]  # one frame where the box shrank to a corner
    at2 = privacy.track_boxes_at(2, [glitch], **kw)
    assert [100.0, 100.0, 160.0, 130.0] in at2  # the neighbours' box still covers the plate on that frame


# Quarantine, deletions, the sweeper ----------------------------------------------------------------------------------------


def test_quarantine_put_and_read_original_frame(runs: Path, tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    run_id = "r1"
    paths.ensure_run_layout(run_id)
    src = tmp_path / "upload.mp4"
    _make_clip(src)
    dst = privacy.quarantine_put(run_id, "pickup", src)
    assert dst == paths.quarantine_original(run_id, "pickup") and dst.is_file() and not src.exists()
    frame = privacy.read_original_frame(run_id, "pickup", 0.2)
    assert frame.shape == (180, 320, 3) and frame.dtype == np.uint8
    with pytest.raises(FileNotFoundError):
        privacy.read_original_frame(run_id, "return", 0.0)


def test_delete_quarantine_logs_hash_bytes_time_and_removes_everything(runs: Path) -> None:
    run_id = "r1"
    paths.ensure_run_layout(run_id)
    data = bytes(range(256)) * 4
    paths.quarantine_original(run_id, "pickup").write_bytes(data)
    paths.quarantine_temp(run_id, "f1.jpg").write_bytes(b"y" * 10)
    rows = privacy.delete_quarantine(run_id, now=NOW)
    assert {r.target for r in rows} == {"quarantine/pickup.orig", "quarantine/tmp/f1.jpg"}
    row = next(r for r in rows if r.target == "quarantine/pickup.orig")
    assert row.kind == c.DeletionKind.QUARANTINE
    assert row.sha256 == hashlib.sha256(data).hexdigest() and row.bytes == len(data) and row.deleted_at == NOW
    assert not [p for p in (runs / run_id / "quarantine").rglob("*") if p.is_file()]
    deletions = privacy.close_deletions(run_id, rows, keep_plate=True, chosen_by="renter at upload")
    back = c.read_json(paths.deletions_json(run_id), c.Deletions)
    assert back == deletions and back.rows == rows and back.keep_plate and back.keep_plate_chosen_by == "renter at upload"
    assert paths.deletions_json(run_id).read_bytes() == c.json_bytes(deletions)
    assert privacy.delete_quarantine(run_id, now=NOW) == []


async def test_delete_remote_records_receipts_and_failures(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(privacy, "_RETRY_SLEEP_S", 0.0)
    calls: list[str] = []

    async def fake_delete_file(file_id: str) -> dict:
        calls.append(file_id)
        if file_id == "file_bad":
            raise RuntimeError("boom")
        return {"id": file_id, "object": "file", "deleted": True}

    async def fake_delete_response(response_id: str) -> dict:
        calls.append(response_id)
        return {"id": response_id, "status_code": 200, "deleted": True, "body": None}

    monkeypatch.setattr(sam_client, "delete_file", fake_delete_file)
    monkeypatch.setattr(sam_client, "delete_response", fake_delete_response)

    def video(role: str, file_id: str, deleted_at: str | None = None) -> c.VideoRecord:
        proxy = c.ProxyRecord(path=f"media/{role}.proxy.mp4", fps=15.0, width=960, height=540, frames=600, pts_map_path=f"media/{role}.pts.json", sha256="ab" * 32)
        files = c.FilesApiRecord(file_id=file_id, uploaded_at="2026-10-27T11:00:00Z", expires_at="2026-10-27T12:00:00Z", deleted_at=deleted_at)
        return c.VideoRecord(role=role, original_filename=f"{role}.mp4", bytes=1, proxy=proxy, files_api=files)

    videos = [video("pickup", "file_ok"), video("return", "file_bad"), video("topup_1", "file_done", deleted_at="2026-10-27T11:30:00Z")]
    rows = await privacy.delete_remote("r1", videos, response_ids=["resp_1"], now=NOW, retries=3)
    by_target = {r.target: r for r in rows}
    ok = by_target["file_ok"]
    assert ok.kind == c.DeletionKind.FILES_API and ok.api_response == {"id": "file_ok", "object": "file", "deleted": True}
    assert ok.sha256 == "ab" * 32 and ok.deleted_at == NOW
    bad = by_target["file_bad"]
    assert bad.api_response is None and bad.reason.startswith(f"delete request failed at {NOW}; expiry at 2026-10-27T12:00:00Z applies")
    assert "3 tries" in bad.reason and "boom" in bad.reason
    assert calls.count("file_bad") == 3 and "file_done" not in calls
    resp = by_target["resp_1"]
    assert resp.kind == c.DeletionKind.RESPONSE and resp.api_response["deleted"] is True
    calls.clear()
    rows = await privacy.delete_remote("r1", [video("pickup", "file_ok")], response_ids=None, now=NOW, retries=1)
    assert [r.target for r in rows] == ["file_ok"] and calls == ["file_ok"]


async def test_render_blurred_proxy_pixelates_the_track_and_deletes_the_unblurred_proxy(runs: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    from walkaround import video as video_mod

    run_id = "r1"
    paths.ensure_run_layout(run_id)
    rng = np.random.default_rng(8)
    frame = np.full((540, 960, 3), 128, dtype=np.uint8)
    frame[200:320, 300:600] = rng.integers(0, 256, (120, 300, 3), dtype=np.uint8)  # inside the plate track
    frame[380:500, 650:950] = rng.integers(0, 256, (120, 300, 3), dtype=np.uint8)  # an untracked noisy patch
    proxy = paths.proxy_path(run_id, "pickup")
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540", "-r", "15", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(proxy)],
        stdin=subprocess.PIPE,
    )
    assert enc.stdin is not None
    for _ in range(15):
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0
    record = video_mod.proxy_record(run_id, "pickup", proxy)
    c.write_json(paths.video_json(run_id, "pickup"), c.VideoRecord(role="pickup", original_filename="a.mp4", bytes=1, proxy=record))
    track = _plate_track(list(range(15)), [300.0, 200.0, 600.0, 320.0])
    blurred, deletion = await privacy.render_blurred_proxy(run_id, "pickup", [track], fps=15.0, th=TH)
    out_path = paths.run_dir(run_id) / blurred.path
    assert out_path == paths.proxy_blurred_path(run_id, "pickup") and out_path.is_file()
    assert blurred.sha256 == hashlib.sha256(out_path.read_bytes()).hexdigest()
    assert not proxy.exists()
    assert deletion.kind == c.DeletionKind.FILE and deletion.target == "media/pickup.proxy.mp4"
    assert deletion.sha256 == record.sha256 and deletion.bytes and deletion.bytes > 0
    got = video_mod.read_frame_at(out_path, 0.3)
    gray = cv2.cvtColor(got, cv2.COLOR_RGB2GRAY).astype(np.float64)
    assert gray[200:320, 300:600].var() < 0.3 * gray[380:500, 650:950].var()  # the tracked noise became blocks
    assert abs(gray[0:150, 0:250].mean() - 128) < 6  # the flat background is untouched


def test_sweep_deletes_old_runs_and_stale_temps_but_keeps_keep_runs(runs: Path) -> None:
    def make_run(run_id: str, created_at: str, *, keep: bool = False) -> None:
        paths.ensure_run_layout(run_id)
        c.write_json(paths.run_json(run_id), c.RunManifest(run_id=run_id, created_at=created_at, keep=keep))
        paths.quarantine_original(run_id, "pickup").write_bytes(b"z" * 100)

    make_run("old", "2026-10-25T12:00:00Z")
    make_run("kept", "2026-10-25T12:00:00Z", keep=True)
    make_run("listed", "2026-10-25T12:00:00Z")
    make_run("young", "2026-10-27T11:00:00Z")
    stale = paths.quarantine_temp("young", "a.jpg")
    stale.write_bytes(b"s" * 20)
    fresh = paths.quarantine_temp("young", "b.jpg")
    fresh.write_bytes(b"f" * 20)
    now_ts = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    os.utime(stale, (now_ts - 7200, now_ts - 7200))
    os.utime(fresh, (now_ts - 60, now_ts - 60))
    rows = privacy.sweep(now=NOW, retention_hours=24, temp_minutes=30, keep_run_ids={"listed"})
    assert not (runs / "old").exists()
    assert (runs / "kept").exists() and (runs / "listed").exists() and (runs / "young").exists()
    assert not stale.exists() and fresh.exists()
    targets = {r.target: r for r in rows}
    assert targets["old/"].kind == c.DeletionKind.FILE and targets["old/"].bytes and targets["old/"].bytes > 100
    temp = targets["young/quarantine/tmp/a.jpg"]
    assert temp.kind == c.DeletionKind.QUARANTINE and temp.sha256 == hashlib.sha256(b"s" * 20).hexdigest()
    assert len(rows) == 2


# Serving rule and the privacy line ------------------------------------------------------------------------------------------


def test_served_file_allowed_is_an_allowlist() -> None:
    run = c.RunManifest(run_id="r1", created_at=NOW)
    kept = c.RunManifest(run_id="r2", created_at=NOW, keep_plate=True)
    denied = (
        "quarantine/pickup.orig", "quarantine/tmp/x.jpg", "uploads/u1/chunks/000000", "media/pickup.proxy.mp4", "../other/run.json", "/etc/passwd", "",
        "media/.pickup.proxy.mp4.part.mp4",  # the encoder's temp of the unblurred proxy (video._tmp_for)
        "media/pickup.wav", "media/pickup.overlay.mp4", "run.json", "spark.json", "voice.json", "events.ndjson", "events.pending.ndjson",
        "QUARANTINE/pickup.orig", "Quarantine/tmp/x.jpg", "Uploads/u1/chunks/000000",  # APFS resolves these to the real files
        "Media/pickup.proxy.mp4", "media/pickup.PROXY.mp4",
        "sam/pickup/images/f000003.face.resp_1.ndjson", "sam/pickup/car.ndjson",
        "crops/pickup/.k.jpg.tmp", "crops/pickup/k.jpg.part", "crops/k.jpg", "derived/sub/x.json", "sheet/other/x.jpg", "bundle/manifest.json.ots.bak",
        "crops/pickup/./../../quarantine/pickup.orig", "crops\\pickup\\k.jpg",
        "claim/claim_1.jpg", "claim/claim_1.answers.json", "claim/notes.json", "claim/claim_1.png", "seeds.json",  # the upload, answers
        "clips/pickup/staysput.json", "chips/pickup/t1.part.m4a",
    )
    for rel in denied:
        assert privacy.served_file_allowed(run, rel, run_owner=True) is False, rel
    allowed = (
        "deletions.json", "videos/pickup.json", "media/pickup.proxy_blurred.mp4", "media/pickup.pts.json",
        "sam/pickup/car.ndjson.gz", "sam/pickup/car.index.json", "sam/pickup/license_plate.summary.json",
        "crops/pickup/k.jpg", "crops/pickup/k.mask.png", "keyframes/pickup/k.frame.jpg", "keyframes/pickup/odometer.jpg",
        "clips/return/m1.mp4", "chips/pickup/t1.m4a", "claim/claim_1.blurred.jpg", "claim/claim_12.json", "derived/verdicts.json",
        "derived/looks.json", "derived/narration_pieces.json", "crops/pickup/p_m1__r_m1.look.jpg", "open_weights.ndjson",
        "sheet/sheet.html", "sheet/sheet.json", "sheet/evidence.pdf", "sheet/frames/k.jpg",
        "bundle/manifest.json", "bundle/manifest_run.json", "bundle/manifest.json.ots", "bundle/manifest.json.tsr", "bundle/verify.md",
        "Crops/pickup/k.jpg",  # the same blurred crop on a case-insensitive disk
    )
    for rel in allowed:
        assert privacy.served_file_allowed(run, rel, run_owner=False) is True, rel
    for rel in ("sheet/frames/k.jpg", "sheet/evidence.pdf", "crops/pickup/k.jpg", "keyframes/pickup/k.frame.jpg", "clips/return/m1.mp4", "claim/claim_1.blurred.jpg", "SHEET/frames/k.jpg"):
        assert privacy.served_file_allowed(kept, rel, run_owner=True) is True, rel
        assert privacy.served_file_allowed(kept, rel, run_owner=False) is False, rel
    for owner_frame in ("sheet/frames/k.owner.jpg", "sheet/frames/K.OWNER.jpg"):
        assert privacy.served_file_allowed(kept, owner_frame, run_owner=True) is True
        assert privacy.served_file_allowed(kept, owner_frame, run_owner=False) is False
        assert privacy.served_file_allowed(run, owner_frame, run_owner=False) is False  # owner-only on any run
    assert privacy.served_file_allowed(kept, "media/pickup.proxy_blurred.mp4", run_owner=False) is True
    assert privacy.served_file_allowed(kept, "derived/verdicts.json", run_owner=False) is True
    assert privacy.served_file_allowed(kept, "quarantine/pickup.orig", run_owner=True) is False


def test_privacy_line_word_for_word() -> None:
    run = c.RunManifest(run_id="r1", created_at=NOW, responses_store=c.ResponsesStore.FALSE_ACCEPTED)
    line = privacy.privacy_line(run, {"pickup": True, "return": True})
    assert line.startswith(
        "Processed with Meta's Model API in Standard mode, which Meta's documentation says is not used to train Meta "
        "models. Uploads were set to expire after one hour and were deleted by Walkaround when the run ended. "
        "Model requests were sent with storage off. Meta's terms allow it to retain content for abuse review and legal "
        "reasons; Walkaround does not control that. SAM 3.1 saw the unblurred proxy for the length of one segmentation "
        "pass to find the plates and faces it then blurred."
    )
    assert "no plate detected" not in line
    for banned in ("zero data retention", "Meta deleted", "Contributor"):
        assert banned not in line
    refused = c.RunManifest(run_id="r1", created_at=NOW, responses_store=c.ResponsesStore.DELETED_PER_REQUEST)
    line2 = privacy.privacy_line(refused, {"pickup": True, "return": False})
    assert "Model requests were sent with storage off and every stored model response was deleted by Walkaround when the run ended." in line2
    assert line2.endswith("Return walk: no plate detected; nothing was blurred for that reason.")
    assert line.isascii() and line2.isascii()
    failed = c.DeletionRecord(target="file-1", kind=c.DeletionKind.FILES_API, deleted_at=NOW, reason="failed after 3 tries: 500")
    line3 = privacy.privacy_line(run, {"pickup": True}, [failed])
    assert "Walkaround's delete request failed, so that expiry applies." in line3 and "were deleted by Walkaround" not in line3
    done = failed.model_copy(update={"api_response": {"id": "file-1", "deleted": True}, "reason": "run end"})
    assert privacy.privacy_line(run, {"pickup": True}, [done]) == privacy.privacy_line(run, {"pickup": True})


# Block size, carry and the display video (privacy lane, Oct 31) --------------------------------------------------------------


def test_privacy_defaults_match_the_thresholds() -> None:
    # a run sealed before Oct 31 has none of these keys in its snapshot; the code's defaults must be today's values
    for key, value in privacy.PRIVACY_DEFAULTS.items():
        assert TH["privacy"][key] == value, key
    old = {"privacy": {k: v for k, v in TH["privacy"].items() if k not in privacy.PRIVACY_DEFAULTS}}
    assert privacy.privacy_params(old)["track_carry_s"] == privacy.PRIVACY_DEFAULTS["track_carry_s"]


def test_block_for_box_grows_with_the_box() -> None:
    assert privacy.block_for_box([0, 0, 400, 200], block_px=5, blocks_across=4) == 100
    assert privacy.block_for_box([0, 0, 127, 111], block_px=5, blocks_across=4) == 32  # pairC_core1 return, 4.4 s
    assert privacy.block_for_box([0, 0, 12, 6], block_px=5, blocks_across=4) == 5  # never under the floor
    assert privacy.block_for_box([0, 0, 400, 200], block_px=5, blocks_across=None) == 5


def test_pixelate_overlapping_boxes_share_the_coarsest_block() -> None:
    rgb = _noise(400, 600, seed=21)
    big, small, far = [100.0, 100.0, 300.0, 200.0], [280.0, 180.0, 300.0, 196.0], [500.0, 300.0, 516.0, 316.0]
    out = privacy.pixelate(rgb, [big, small, far], block_px=4)
    region = out[100:200, 100:300].reshape(2, 50, 4, 50, 3)
    assert (region == region[:, :1, :, :1]).all()  # one region, 50 px blocks from the 200 px box, small box included
    lone = out[300:316, 500:516].reshape(4, 4, 4, 4, 3)
    assert (lone == lone[:, :1, :, :1]).all()  # the far 16 px box keeps its own 4 px block
    assert len(np.unique(out[300:316, 500:516].reshape(-1, 3), axis=0)) > 4


def _plate_image(text: str, w: int) -> np.ndarray:
    h = w // 2
    img = np.full((h, w, 3), 235, np.uint8)
    scale = w / 260.0
    thick = max(1, int(round(scale * 3)))
    (tw, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, text, ((w - tw) // 2, (h + th_) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), thick, cv2.LINE_AA)
    return img


def test_two_different_plates_look_alike_after_pixelation_at_any_size() -> None:
    """The 5 px display block left a close plate readable on pairC_core1. With the block grown to a quarter of the box,
    two plates with different text keep under a quarter of their raw difference at every size, while the fixed block
    keeps most of it on a close plate."""
    W, H, floor = 1920, 1080, 10  # pixel_block_px_at_4k 20 at 1080p
    ratios_new, ratios_old = [], []
    for w in (80, 160, 320, 640):
        got = []
        for text in ("ABC 1234", "XYZ 9876"):
            frame = np.full((H, W, 3), 128, np.uint8)
            plate = _plate_image(text, w)
            x0, y0 = 600, 300
            frame[y0 : y0 + plate.shape[0], x0 : x0 + w] = plate
            box = [float(x0), float(y0), float(x0 + w), float(y0 + plate.shape[0])]
            cut = (slice(y0, y0 + plate.shape[0]), slice(x0, x0 + w))
            new = privacy.pixelate(frame, [box], block_px=floor, dilate_frac=0.1, dilate_px=floor)
            old = privacy.pixelate(frame, [box], block_px=floor, dilate_frac=0.1, dilate_px=floor, blocks_across=None)
            got.append((frame[cut].astype(float), new[cut].astype(float), old[cut].astype(float)))
        (a0, a1, a2), (b0, b1, b2) = got
        raw = np.abs(a0 - b0).mean()
        ratios_new.append(np.abs(a1 - b1).mean() / raw)
        ratios_old.append(np.abs(a2 - b2).mean() / raw)
    assert max(ratios_new) < 0.25, ratios_new
    assert ratios_old[-1] > 0.6, ratios_old  # a 640 px plate under 10 px blocks still shows its text


def test_blur_frame_grows_the_block_for_a_close_plate_and_a_face() -> None:
    frame = _noise(1080, 1920, seed=22)
    plate_mask = _rect_mask((1080, 1920), (600, 500, 1000, 700))  # a 400 px plate at 1080p
    face_box = [1500.0, 100.0, 1700.0, 340.0]
    out = privacy.blur_frame(frame, track_boxes_src=[face_box], detections=[], keep_plate_region=False, th=TH, plate_detections=[_detection(plate_mask)])
    inner = out[520:680, 620:980]
    assert len(np.unique(inner.reshape(-1, 3), axis=0)) <= 4 * 3  # 100 px blocks: about 4 x 2 over the plate
    face = out[110:330, 1510:1690]
    assert len(np.unique(face.reshape(-1, 3), axis=0)) <= 5 * 5  # 60 px blocks over a 240 px face box
    assert np.array_equal(out[900:1080, 0:300], frame[900:1080, 0:300])


def _pan_frames(offsets: list[int], *, h: int = 270, w: int = 480, seed: int = 11) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.integers(0, 256, (h, w + 400), dtype=np.uint8), (0, 0), 2.0)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    return {f: np.ascontiguousarray(tex[:, off : off + w]) for f, off in enumerate(offsets)}


def test_flow_carry_follows_a_panning_plate_before_its_track_starts() -> None:
    """A camera pan that slows down (10 px a frame, then 2): the plate track starts at frame 20; optical flow carries
    its box back to frame 0 on the plate, where the extrapolated box (2 px a frame) is 80 px off."""
    offsets = [10 * f if f <= 10 else 100 + 2 * (f - 10) for f in range(41)]
    grays = _pan_frames(offsets)
    true = {f: [300.0 - off, 120.0, 340.0 - off, 140.0] for f, off in enumerate(offsets)}
    track = _plate_track(list(range(20, 31)), [0.0, 0.0, 1.0, 1.0])
    for r in track.per_frame:
        r.box = list(true[r.frame])
    carried = privacy.flow_carry(grays, [track], fps=15.0, media_size_proxy=(480, 270), n_frames=41, th=TH)
    assert sorted(f for (_tid, f, side) in carried if side < 0) == list(range(0, 20))
    for f in range(20):
        box = carried[(track.track_id, f, -1)]
        assert _inside(true[f], box), (f, box, true[f])
        assert abs((box[0] + box[2]) / 2 - (true[f][0] + true[f][2]) / 2) < 3.0
    kw = dict(fps=15.0, media_size_proxy=(480, 270), media_size_src=(480, 270), th=TH)
    [guess] = privacy.track_boxes_at(0, [track], **kw)  # no flow: the extrapolated box misses the plate
    assert not _inside(true[0], guess)
    assert privacy.track_boxes_at(0, [track], carried=carried, **kw) == [carried[(track.track_id, 0, -1)]]


def test_flow_carry_goes_on_kinematically_where_frames_are_missing() -> None:
    offsets = [2 * f for f in range(30)]
    grays = _pan_frames(offsets)
    track = _plate_track(list(range(15, 25)), [0.0, 0.0, 1.0, 1.0])
    for r in track.per_frame:
        r.box = [300.0 - offsets[r.frame], 120.0, 340.0 - offsets[r.frame], 140.0]
    del grays[10]  # the flow loses the box at frame 10
    carried = privacy.flow_carry(grays, [track], fps=15.0, media_size_proxy=(480, 270), n_frames=30, th=TH)
    w11 = carried[(track.track_id, 11, -1)][2] - carried[(track.track_id, 11, -1)][0]
    w5 = carried[(track.track_id, 5, -1)][2] - carried[(track.track_id, 5, -1)][0]
    assert w5 > w11  # past the loss it grows at track_carry_grow_per_s
    assert set(privacy.carry_frames_needed([track], fps=15.0, n_frames=30, th=TH)) == set(range(0, 16)) | set(range(24, 30))


async def test_display_video_rederives_byte_for_byte_from_the_original(runs: Path, tmp_path: Path) -> None:
    """The blur stage's display video is a derived file: from the sealed original and the stored lanes,
    rederive_display makes the same bytes again, outside the run, and keeps no unblurred copy."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    from walkaround import video as video_mod

    run_id, role = "r1", "pickup"
    paths.ensure_run_layout(run_id)
    original = tmp_path / "walk.mp4"
    _make_clip(original, w=640, h=360, fps=30, seconds=1.0)
    c.write_json(paths.run_json(run_id), c.RunManifest(run_id=run_id, created_at=NOW, thresholds=TH))
    media = TH["media"]
    proxy = paths.proxy_path(run_id, role)
    video_mod.make_proxy(original, proxy, fps=float(media["proxy_fps"]), max_edge=int(media["proxy_max_edge"]), preset=media["proxy_preset"], crf=int(media["proxy_crf"]))
    record = video_mod.proxy_record(run_id, role, proxy)
    sha = hashlib.sha256(original.read_bytes()).hexdigest()
    walk = c.VideoRecord(role=role, original_filename="walk.mp4", bytes=original.stat().st_size, sha256_server=sha, proxy=record)
    c.write_json(paths.video_json(run_id, role), walk)
    plate = [c.IndexFrame(frame=f, objects=[c.IndexObject(id="1", box=[200 + 6 * f, 150, 280 + 6 * f, 190], c=0.8)]) for f in range(6, 12)]
    c.write_json(paths.lane_index_path(run_id, role, c.Noun.LICENSE_PLATE), plate)
    c.write_json(paths.lane_index_path(run_id, role, c.Noun.FACE), [])
    tracks = privacy.walk_privacy_tracks(run_id, role, th=TH)
    assert [(t.noun, [r.frame for r in t.per_frame]) for t in tracks] == [(c.Noun.LICENSE_PLATE, list(range(6, 12)))]
    blurred, _deletion = await privacy.render_blurred_proxy(run_id, role, tracks, fps=record.fps, th=TH)
    assert not proxy.exists()
    c.write_json(paths.video_json(run_id, role), walk.model_copy(update={"proxy_blurred": blurred}))
    out = tmp_path / "again"
    got = privacy.rederive_display(run_id, role, original, out)
    assert got["proxy_matches_sealed"] and got["matches_sealed"], got
    assert got["blurred_sha256"] == blurred.sha256 and (out / f"{role}.proxy_blurred.mp4").is_file()
    assert sorted(p.name for p in out.iterdir()) == [f"{role}.proxy_blurred.mp4"]  # the unblurred proxy is gone
    first = video_mod.read_frame_at(out / f"{role}.proxy_blurred.mp4", 0.0)
    clean = video_mod.read_frame_at(original, 0.0)
    clean = cv2.resize(clean, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
    assert _changed(first[150:190, 236:316], clean[150:190, 236:316]) > 0.5  # frame 0 is carried from the frame 6 sighting
    other = tmp_path / "other.mp4"
    _make_clip(other, w=640, h=360, fps=30, seconds=0.6)
    with pytest.raises(ValueError, match="not the sealed"):
        privacy.rederive_display(run_id, role, other, out)
    with pytest.raises(ValueError, match="never writes inside the run"):
        privacy.rederive_display(run_id, role, original, paths.run_dir(run_id) / "media")


# LIVE -----------------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="LIVE=1 only: two sam_image calls on one frame, about $0.005")
async def test_live_image_checks_on_the_pingpong_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real sam_client on a real frame; the ledger rows land in runs/ledger.jsonl because the call costs money,
    the run files go to a temporary runs dir."""
    from walkaround import video as video_mod

    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    clip = REPO / "data" / "test" / "pingpong.mp4"
    frame = video_mod.read_frame_at(clip, 1.0)
    run_id = "lane_privacy_smoke"
    plates, faces, record = await privacy.image_checks(frame, run_id=run_id, role="pickup", tag="live", th=TH)
    assert len(record.request_ids) == 2
    assert len(record.plate_boxes) == len(plates) and len(record.face_boxes) == len(faces)
    rows = [r for r in ledger.rows(ledger.LEDGER_PATH) if r.get("run_id") == run_id]
    assert len(rows) >= 2 and all(r["kind"] == "sam_image" and r["usd"] > 0 for r in rows)
    saved = sorted(paths.image_requests_dir(run_id, "pickup").glob("live.*.ndjson"))
    assert len(saved) == 2
    print(f"LIVE image checks: {len(plates)} plates, {len(faces)} faces, ruler px {record.plate_ruler_px}, requests {record.request_ids}")


def test_privacy_line_names_baseten_only_when_a_second_look_went_there() -> None:
    run = c.RunManifest(run_id="r1", created_at=NOW)
    plain = privacy.privacy_line(run, {"pickup": True})
    told = privacy.privacy_line(run, {"pickup": True}, baseten=True)
    assert "Baseten" not in plain and told == plain + " " + privacy.BASETEN_CLAUSE
    assert told.isascii() and "Muse Spark Open on Baseten, a company separate from Meta" in told
