"""Friend run 1 fixes of 11/2 (docs/FRIEND_RUN1.md), each behind a thresholds leaf that frozen runs lack:
panels.uncertain_per_track (a stray undecided track no longer hides a panel with decided tracks in both walks),
panels.part_check (the side of a wheel or fender-lane track, the end of a glass track), dedup.marks_on_panel (a mark
off its own panel mask is moved to the panel under it in the same frame or left out) and matching.second_marks_pass
(one more Spark look where one walk had no mark on a comparable panel). No network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from walkaround import config, geometry, ledger, marks, match, panels, paths, second_pass, spark_client, video, zoom_refine
from walkaround import contracts as c
from walkaround.contracts import LaneStatus, MarkKind, Noun, PanelId, Position, Side

TH = c.load_thresholds()
FRIEND = Path(__file__).resolve().parent.parent / "runs" / "friend_pair1" / "derived" / "tracks.json"


def _without(path: str) -> dict[str, Any]:
    th = json.loads(json.dumps(TH))
    section, leaf = path.split(".")
    del th[section][leaf]
    return th


def _track(tid: str, role: str, noun: Noun, pid: PanelId | None, frames: range, box: Any, side: Side = Side.NONE, position: Position = Position.NONE) -> c.PanelTrack:
    rows = [c.FrameObs(frame=f, pts=f / 15, box=list(box(f)) if callable(box) else list(box), box_area_frac=0.05, in_car_box=True) for f in frames]
    return c.PanelTrack(track_id=tid, role=role, noun=noun, sam_object_ids=["1"], fragments=[], per_frame=rows, panel_id=pid, side=side, position=position)


# 1. Uncertainty per track --------------------------------------------------------------------------------------------


def _door_world() -> list[c.PanelTrack]:
    f = range(0, 20)
    return [
        _track("pickup_car_door_01", "pickup", Noun.CAR_DOOR, PanelId.LEFT_REAR_DOOR, f, [100, 100, 300, 300], Side.LEFT, Position.REAR),
        _track("return_car_door_01", "return", Noun.CAR_DOOR, PanelId.LEFT_REAR_DOOR, f, [100, 100, 300, 300], Side.LEFT, Position.REAR),
        # the door check placed it on the left but not front or rear: a placeholder in the left rear slot
        _track("return_car_door_08", "return", Noun.CAR_DOOR, PanelId.LEFT_REAR_DOOR, range(40, 60), [400, 100, 600, 300], Side.LEFT, Position.NONE),
    ]


def _panel(tracks: list[c.PanelTrack], pid: PanelId, per_track: bool) -> c.Panel:
    built = panels._panels_from(tracks, {}, fps=15.0, merge_gap_s=1.0, fallback_dt=1 / 15, per_track=per_track)
    return next(p for p in built if p.panel_id == pid)


def test_a_stray_undecided_track_no_longer_hides_a_panel_with_decided_tracks_in_both_walks() -> None:
    world = _door_world()
    assert panels.uncertain_per_track(TH) and not panels.uncertain_per_track(_without("panels.uncertain_per_track"))
    on, off = _panel(world, PanelId.LEFT_REAR_DOOR, True), _panel(world, PanelId.LEFT_REAR_DOOR, False)
    assert (on.side_uncertain, on.front_sign) == (False, -1)
    assert (off.side_uncertain, off.front_sign) == (True, None)  # runs frozen before the leaf: one stray track hides it
    # a walk with nothing but the undecided track still leaves the panel uncertain
    only_stray = [world[0], world[2]]
    assert _panel(only_stray, PanelId.LEFT_REAR_DOOR, True).side_uncertain


def test_the_stray_track_is_named_in_coverage_with_the_leaf_only() -> None:
    world = _door_world()
    status = {Noun.CAR: LaneStatus.COMPLETED, Noun.CAR_DOOR: LaneStatus.COMPLETED}

    def reason(th: dict[str, Any]) -> str:
        built = panels._panels_from(world, {}, fps=15.0, merge_gap_s=1.0, fallback_dt=1 / 15, per_track=panels.uncertain_per_track(th))
        cov = panels.coverage_for_role(built, world, [], role="return", fps=15.0, lane_status=status, th=th)
        return cov[PanelId.LEFT_REAR_DOOR].reason

    assert "also return_car_door_08 could be left front door or left rear door" in reason(TH)
    assert "return_car_door_08" not in reason(_without("panels.uncertain_per_track"))


# 3. The part check: wheel sides and glass ends ---------------------------------------------------------------------


def _wheel_world(*, light_dx: float, door_side: Side, hood_dx: float = 0.0) -> list[c.PanelTrack]:
    f = range(0, 12)
    wheel = [500, 400, 600, 500]  # d = 100
    return [
        _track("pickup_wheel_00", "pickup", Noun.WHEEL, PanelId.RIGHT_FRONT_WHEEL, f, wheel, Side.RIGHT, Position.FRONT),
        _track("pickup_headlight_00", "pickup", Noun.HEADLIGHT, PanelId.HEADLIGHTS, f, [550 + light_dx - 20, 350, 550 + light_dx + 20, 370]),
        _track("pickup_car_door_00", "pickup", Noun.CAR_DOOR, PanelId.LEFT_FRONT_DOOR if door_side is Side.LEFT else PanelId.RIGHT_FRONT_DOOR, f, [610, 250, 900, 480], door_side, Position.FRONT),
        _track("pickup_hood_00", "pickup", Noun.HOOD, PanelId.HOOD, f, [550 + hood_dx - 150, 250, 550 + hood_dx + 150, 380]),
    ]


def test_a_wheel_with_the_headlight_and_a_left_door_beside_it_is_the_left_wheel() -> None:
    world = _wheel_world(light_dx=-120.0, door_side=Side.LEFT, hood_dx=-30.0)  # the front image-left: the left side
    got = panels.part_check(world, TH)
    slot, rec = got["pickup_wheel_00"]
    assert slot == PanelId.LEFT_FRONT_WHEEL and rec.decided and rec.kind == "side" and rec.was == PanelId.RIGHT_FRONT_WHEEL
    assert rec.votes == {"lid_side": 0, "light_side": -1, "door_side": -1}  # the hood over the wheel says nothing
    panels.apply_part_check(world, TH)
    assert (world[0].panel_id, world[0].side, world[0].position) == (PanelId.LEFT_FRONT_WHEEL, Side.LEFT, Position.FRONT)
    assert world[0].part_check == rec


def test_split_side_cues_leave_the_wheel_side_uncertain_in_its_slot() -> None:
    world = _wheel_world(light_dx=-120.0, door_side=Side.RIGHT)  # the light says left, the door right
    panels.apply_part_check(world, TH)
    w = world[0]
    assert w.panel_id == PanelId.RIGHT_FRONT_WHEEL and w.side is Side.NONE and not w.part_check.decided  # type: ignore[union-attr]
    assert set(panels.candidate_slots(w)) == {PanelId.LEFT_FRONT_WHEEL, PanelId.RIGHT_FRONT_WHEEL} and panels.undecided(w)


def _glass_world(*, tail: bool, hood: bool) -> list[c.PanelTrack]:
    f = range(0, 12)
    out = [_track("return_windshield_00", "return", Noun.WINDSHIELD, PanelId.WINDSHIELD, f, [300, 100, 700, 250])]
    if tail:
        out.append(_track("return_tail_light_00", "return", Noun.TAIL_LIGHT, PanelId.TAIL_LIGHTS, f, [250, 320, 330, 360]))
        out.append(_track("return_bumper_00", "return", Noun.BUMPER, PanelId.REAR_BUMPER, f, [250, 380, 750, 450], Side.NONE, Position.REAR))
    if hood:
        out.append(_track("return_hood_00", "return", Noun.HOOD, PanelId.HOOD, f, [300, 260, 700, 400]))
    return out


def test_a_windshield_track_with_the_tail_lights_and_rear_bumper_beside_it_is_the_rear_window() -> None:
    world = _glass_world(tail=True, hood=False)
    panels.apply_part_check(world, TH)
    g = world[0]
    assert g.panel_id == PanelId.REAR_WINDOW and g.part_check is not None and g.part_check.kind == "end" and g.part_check.decided
    assert g.part_check.votes["lights"] == -1 and g.part_check.votes["bumper"] == -1


def test_glass_with_both_ends_beside_it_is_left_undecided_between_windshield_and_rear_window() -> None:
    world = _glass_world(tail=True, hood=True)
    panels.apply_part_check(world, TH)
    g = world[0]
    assert g.part_check is not None and not g.part_check.decided and g.panel_id == PanelId.WINDSHIELD
    assert panels.candidate_slots(g) == [PanelId.WINDSHIELD, PanelId.REAR_WINDOW]


def test_without_the_leaf_no_part_check_runs_and_tracks_serialize_as_before() -> None:
    th = _without("panels.part_check")
    world = _wheel_world(light_dx=-120.0, door_side=Side.LEFT)
    assert panels.part_check(world, th) == {}
    panels.apply_part_check(world, th)
    assert world[0].panel_id == PanelId.RIGHT_FRONT_WHEEL and world[0].part_check is None
    assert "part_check" not in json.loads(c.canonical_json(world[0]))


@pytest.mark.skipif(not FRIEND.is_file(), reason="runs/friend_pair1 is not in this checkout")
def test_on_friend_pair1_the_left_front_wheels_and_the_rear_seen_windshields_move() -> None:
    tracks = c.read_json_list(FRIEND, c.PanelTrack)
    moved = {}
    for role in ("pickup", "return"):
        for tid, (slot, rec) in panels.part_check([t for t in tracks if t.role == role], TH).items():
            if slot != rec.was:
                moved[tid] = slot
    assert moved == {
        "pickup_wheel_00": PanelId.LEFT_FRONT_WHEEL,
        "pickup_wheel_04": PanelId.LEFT_FRONT_WHEEL,
        "pickup_wheel_05": PanelId.LEFT_FRONT_WHEEL,
        "pickup_windshield_01": PanelId.REAR_WINDOW,
        "return_wheel_02": PanelId.RIGHT_FRONT_WHEEL,
        "return_wheel_03": PanelId.LEFT_FRONT_WHEEL,
        "return_windshield_00": PanelId.REAR_WINDOW,
    }


# 2. Marks on their own panel ---------------------------------------------------------------------------------------


def _write_mask(root: Path, rel: str, mask: np.ndarray) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / rel), mask.astype(np.uint8) * 255)


def _kf_with_mask(root: Path, kf_id: str, pid: PanelId, crop: c.Crop, mask: np.ndarray, f: int = 10) -> c.Keyframe:
    rel = f"crops/pickup/{kf_id}.mask.png"
    _write_mask(root, rel, mask)
    return c.Keyframe(
        kf_id=kf_id, role="pickup", panel_id=pid, track_id=f"t_{kf_id}", f_proxy=f, pts=f / 15,
        score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=1.5,
        box_src=[0, 0, 100, 100], crop=crop, mask_png=rel, moments=geometry.moments_from_mask(mask), front_sign=1, marks_status=c.MarksStatus.COMPLETE,
    )


def _spark_mark(mid: str, kf: c.Keyframe, point: list[float]) -> c.Mark:
    box = [point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5]
    return c.Mark(
        mark_id=mid, role="pickup", panel_id=kf.panel_id, kf_id=kf.kf_id, source=c.MarkSource.SPARK_BOX, type=MarkKind.SCRATCH,
        confidence=c.Confidence.MEDIUM, size_class=c.SizeClass.S, bbox_crop=box, point_crop=point,
        bbox_src=geometry.box_crop_to_src(box, kf.crop), point_src=geometry.crop_to_src(point, kf.crop), size_px_src=40.0,  # type: ignore[arg-type]
    )


def test_a_mark_off_its_panel_moves_to_the_panel_under_it_in_the_same_frame_or_is_left_out(tmp_path: Path) -> None:
    hood_mask = np.zeros((200, 400), bool)
    hood_mask[:, :200] = True  # the hood is the crop's left half
    hood = _kf_with_mask(tmp_path, "pickup_hood_kf1", PanelId.HOOD, c.Crop(path="crops/pickup/pickup_hood_kf1.jpg", w=400, h=200, scale=1.0, dx=0.0, dy=0.0), hood_mask)
    fender_mask = np.zeros((200, 200), bool)
    fender_mask[:100, :] = True  # the fender's crop starts at source x 250: its top half
    fender = _kf_with_mask(tmp_path, "pickup_left_front_fender_kf1", PanelId.LEFT_FRONT_FENDER, c.Crop(path="crops/pickup/f.jpg", w=200, h=200, scale=1.0, dx=250.0, dy=0.0), fender_mask)
    on = _spark_mark("pickup_hood_kf1_m1", hood, [100.0, 100.0])
    over = _spark_mark("pickup_hood_kf1_m2", hood, [300.0, 50.0])  # off the hood, on the fender (crop point 50, 50)
    nowhere = _spark_mark("pickup_hood_kf1_m3", hood, [300.0, 180.0])  # off the hood and off the fender
    ms, ws = marks.whiten_marks([on, over, nowhere], hood, th=TH)
    kept, whitened, left = marks.on_panel(ms, ws, [hood, fender], tmp_path, th=TH, masks={})
    assert [m.mark_id for m in kept] == ["pickup_hood_kf1_m1", "pickup_hood_kf1_m2"] and [m.mark_id for m in left] == ["pickup_hood_kf1_m3"]
    moved = kept[1]
    assert moved.panel_id == PanelId.LEFT_FRONT_FENDER and moved.kf_id == fender.kf_id and moved.point_crop == [50.0, 50.0]
    assert moved.bbox_crop == [45.0, 45.0, 55.0, 55.0] and moved.w is not None and whitened[moved.mark_id].panel_id == PanelId.LEFT_FRONT_FENDER
    op = moved.off_panel
    assert op is not None and (op.from_panel, op.from_kf, op.moved_to_kf) == (PanelId.HOOD, hood.kf_id, fender.kf_id) and op.off_frac == pytest.approx(0.25, abs=0.003)
    assert left[0].off_panel is not None and left[0].off_panel.moved_to_kf is None and left[0].off_panel.off_frac > 0.025
    assert kept[0] == ms[0] and "pickup_hood_kf1_m3" not in whitened
    # how_checked names both, on the panels they concern
    notes = match.off_panel_notes([moved, left[0]])
    words = match.how_checked(c.PanelMatch(panel_id=PanelId.HOOD, state=c.MatchState.COMPARABLE, off_panel=notes[PanelId.HOOD]))
    assert words is not None and "moved: pickup_hood_kf1_m2 sits 25.1% of the crop's long side off the hood and on the left front fender" in words
    assert "left out: pickup_hood_kf1_m3" in words and "so it was not matched" in words
    there = match.how_checked(c.PanelMatch(panel_id=PanelId.LEFT_FRONT_FENDER, state=c.MatchState.COMPARABLE, off_panel=notes[PanelId.LEFT_FRONT_FENDER]))
    assert there is not None and there.startswith("moved here: pickup_hood_kf1_m2 was found on the hood keyframe")
    # without the leaf nothing moves and nothing is left out
    th = _without("dedup.marks_on_panel")
    assert marks.on_panel(ms, ws, [hood, fender], tmp_path, th=th, masks={}) == (ms, ws, [])
    assert not match.marks_on_panel(th) and match.marks_on_panel(TH)


def test_a_mark_within_the_claim_radius_of_its_mask_stays(tmp_path: Path) -> None:
    mask = np.zeros((200, 400), bool)
    mask[:, :200] = True
    kf = _kf_with_mask(tmp_path, "pickup_hood_kf1", PanelId.HOOD, c.Crop(path="crops/pickup/h.jpg", w=400, h=200, scale=1.0, dx=0.0, dy=0.0), mask)
    near = _spark_mark("pickup_hood_kf1_m1", kf, [208.0, 100.0])  # 8 px off, under 0.025 x 400 = 10 px
    kept, _w, left = marks.on_panel([near], {}, [kf], tmp_path, th=TH, masks={})
    assert kept == [near] and left == []


# 4. The second marks pass ------------------------------------------------------------------------------------------


def _pass_world(root: Path | None = None) -> tuple[dict[str, c.Keyframe], dict[str, c.Mark], c.PanelMatch]:
    mask = np.zeros((300, 400), bool)
    mask[50:250, 50:350] = True
    kfs = {}
    for role in ("pickup", "return"):
        kf_id = f"{role}_left_rear_fender_kf1"
        crop = c.Crop(path=f"crops/{role}/{kf_id}.jpg", sha256=("a" if role == "pickup" else "b") * 64, w=400, h=300, scale=1.0, dx=0.0, dy=0.0)
        kf = c.Keyframe(
            kf_id=kf_id, role=role, panel_id=PanelId.LEFT_REAR_FENDER, track_id=f"{role}_t", f_proxy=5, pts=0.3,
            score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=1.5,
            box_src=[0, 0, 400, 300], crop=crop, moments=geometry.moments_from_mask(mask), front_sign=-1, marks_status=c.MarksStatus.COMPLETE,
        )
        kfs[kf_id] = kf
    r = kfs["return_left_rear_fender_kf1"]
    m = c.Mark(
        mark_id="return_left_rear_fender_kf1_m1", role="return", panel_id=PanelId.LEFT_REAR_FENDER, kf_id=r.kf_id, source=c.MarkSource.SPARK_BOX,
        type=MarkKind.SCRATCH, confidence=c.Confidence.MEDIUM, size_class=c.SizeClass.S, bbox_crop=[240.0, 120.0, 260.0, 130.0], point_crop=[250.0, 125.0],
        bbox_src=[240.0, 120.0, 260.0, 130.0], point_src=[250.0, 125.0], size_px_src=20.0,
    )
    [m], _ = marks.whiten_marks([m], r, th=TH)
    pm = c.PanelMatch(panel_id=PanelId.LEFT_REAR_FENDER, state=c.MatchState.COMPARABLE, kf_pair=["pickup_left_rear_fender_kf1", r.kf_id], unmatched_return=[m.mark_id])
    return kfs, {m.mark_id: m}, pm


def _answer(boxes: list[list[int]], conf: str = "low", rid: str = "resp_pass_1") -> c.SparkRequestRecord:
    items = [{"type": "scratch", "point": {"x": (b[0] + b[2]) // 2, "y": (b[1] + b[3]) // 2}, "bbox": {"x_min": b[0], "y_min": b[1], "x_max": b[2], "y_max": b[3]}, "size_class": "S", "confidence": conf, "note": "thin line"} for b in boxes]
    text = json.dumps({"marks": items, "conditions": {"wet": False, "dirty": False, "glare": False, "dark": False}, "landmarks": [], "neighbour_car_visible": False})
    return c.SparkRequestRecord(request_id=rid, purpose=c.SparkPurpose.MARKS, model="muse-spark-1.3", schema_name="marks_v1", raw_text=text, parsed=json.loads(text), valid=True)


def test_the_pass_plans_one_look_on_the_silent_walk_around_the_mapped_mark_and_only_there() -> None:
    s = second_pass.settings(TH)
    assert s is not None and s["max_asks"] == 12 and s["window"]["out_w"] == 1280
    kfs, ms, pm = _pass_world()
    got = second_pass.plan([pm], ms, kfs, s)
    assert len(got) == 1
    kf, other, expected, win = got[0]
    assert kf.kf_id == "pickup_left_rear_fender_kf1" and other.mark_id == "return_left_rear_fender_kf1_m1"
    assert expected == pytest.approx([250.0, 125.0], abs=0.01)  # same moments both walks: the same spot
    assert win[0] <= 240 and win[2] >= 260 and win[1] <= 120 and win[3] >= 130
    assert second_pass.plan([pm.model_copy(update={"state": c.MatchState.NOT_COMPARABLE})], ms, kfs, s) == []
    both = pm.model_copy(update={"unmatched_pickup": ["pickup_left_rear_fender_kf1_m1"]})
    assert second_pass.plan([both], ms, kfs, s) == []  # both walks have marks: nothing silent
    assert second_pass.settings(_without("matching.second_marks_pass")) is None
    assert second_pass.settings(_without("dedup.spark_zoom_refine")) is None  # it borrows the zoom window


def test_the_box_nearest_the_spot_becomes_a_capped_mark_recorded_as_a_second_pass() -> None:
    s = second_pass.settings(TH)
    kfs, ms, pm = _pass_world()
    kf, other, expected, win = second_pass.plan([pm], ms, kfs, s)[0]  # type: ignore[arg-type]
    ww, wh = win[2] - win[0], win[3] - win[1]

    def grid(b: list[float]) -> list[int]:
        return [round((b[0] - win[0]) * 1000 / ww), round((b[1] - win[1]) * 1000 / wh), round((b[2] - win[0]) * 1000 / ww), round((b[3] - win[1]) * 1000 / wh)]

    rec = _answer([grid([243.0, 121.0, 259.0, 129.0]), grid([win[0] + 2.0, win[1] + 2.0, win[0] + 8.0, win[1] + 8.0])])
    ask = second_pass.SecondPassAsk(kf_id=kf.kf_id, role="pickup", panel_id=kf.panel_id, other_mark=other.mark_id, crop_path=kf.crop.path, crop_sha256=kf.crop.sha256, crop_w=400, crop_h=300, window=win, expected_crop=expected, out_w=1280, out_h=720, record=rec)  # type: ignore[union-attr]
    [m] = second_pass.marks_from_asks([ask], kfs, th=TH)
    assert m.mark_id == "pickup_left_rear_fender_kf1_s1" and m.role == "pickup" and m.source == c.MarkSource.SPARK_BOX
    assert abs(m.point_crop[0] - 251) < 2 and abs(m.point_crop[1] - 125) < 2 and m.w is not None  # type: ignore[index]
    assert "K12" in m.caps and m.grid_bbox is None  # low confidence keeps its cap; no grid of the crop to claim
    assert m.second_pass == c.SecondPass(other_mark=other.mark_id, window_crop=win, expected_crop=expected, boxes=2, spark_request_id="resp_pass_1")
    empty = ask.model_copy(update={"record": _answer([])})
    assert second_pass.marks_from_asks([empty], kfs, th=TH) == []
    twice = second_pass.marks_from_asks([ask, ask.model_copy(update={"other_mark": "x"})], kfs, th=TH)
    assert len(twice) == 1  # the same mark found from two spots is one mark


@pytest.fixture
def pass_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    run_id = "sp1"
    paths.ensure_run_layout(run_id)
    kfs, ms, pm = _pass_world()
    for kf in kfs.values():
        video.write_jpeg(np.full((300, 400, 3), 90, np.uint8), paths.run_dir(run_id) / kf.crop.path, quality=90)  # type: ignore[union-attr]
    c.write_json(paths.keyframes_json(run_id), list(kfs.values()))
    c.write_json(paths.marks_json(run_id), list(ms.values()))
    calls: list[tuple[int, int]] = []

    def fake_find_marks(rgb: np.ndarray, panel_id: PanelId, **kw: Any) -> spark_client.MarksResult:
        h, w = rgb.shape[:2]
        calls.append((w, h))
        return spark_client.marks_result_from_record(_answer([[480, 470, 520, 530]]), orig_width=w, orig_height=h, landmarks=[])

    monkeypatch.setattr(spark_client, "find_marks", fake_find_marks)
    monkeypatch.setattr(match, "derive_match", lambda run_id, th: ([pm], [], None))
    return {"run_id": run_id, "calls": calls}


def test_derive_asks_once_and_apply_replays_with_no_call(pass_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = pass_run["run_id"]
    asked = asyncio.run(second_pass.derive_second_pass(run_id, th=TH))
    assert pass_run["calls"] == [(1280, 720)] and [role for role, _rec in asked] == ["pickup"]
    rows = c.read_json_list(paths.marks_json(run_id), c.Mark)
    assert [m.mark_id for m in rows] == ["return_left_rear_fender_kf1_m1", "pickup_left_rear_fender_kf1_s1"]
    before = paths.marks_json(run_id).read_bytes()
    assert asyncio.run(second_pass.derive_second_pass(run_id, th=TH)) == []  # stored: not asked again
    assert pass_run["calls"] == [(1280, 720)] and paths.marks_json(run_id).read_bytes() == before

    def no_spark(*a: Any, **k: Any) -> Any:
        raise AssertionError("apply must not call Spark")

    monkeypatch.setattr(spark_client, "find_marks", no_spark)
    base = [m for m in rows if m.second_pass is None]
    c.write_json(paths.marks_json(run_id), base)  # what rebuild_marks leaves before the match step
    assert second_pass.apply_second_pass(run_id, th=TH) == 1 and paths.marks_json(run_id).read_bytes() == before
    assert second_pass.apply_second_pass(run_id, th=TH) == 1 and paths.marks_json(run_id).read_bytes() == before
    words = match.second_pass_words(run_id, rows, TH)
    assert words[PanelId.LEFT_REAR_FENDER] == ["looked again at pickup_left_rear_fender_kf1 around return_left_rear_fender_kf1_m1: found pickup_left_rear_fender_kf1_s1"]


def test_without_the_leaf_the_pass_neither_asks_nor_touches_marks(pass_run: dict[str, Any]) -> None:
    run_id = pass_run["run_id"]
    th = _without("matching.second_marks_pass")
    before = paths.marks_json(run_id).read_bytes()
    assert asyncio.run(second_pass.derive_second_pass(run_id, th=th)) == [] and pass_run["calls"] == []
    assert second_pass.apply_second_pass(run_id, th=th) == 0 and paths.marks_json(run_id).read_bytes() == before
    assert not paths.second_pass_json(run_id).exists() and match.second_pass_words(run_id, [], th) == {}


def test_a_second_pass_box_off_the_panel_mask_is_not_used(tmp_path: Path) -> None:
    s = second_pass.settings(TH)
    assert s is not None and s["on_panel"]
    kfs, ms, pm = _pass_world()
    kf, other, expected, win = second_pass.plan([pm], ms, kfs, s)[0]  # type: ignore[arg-type]
    mask = np.zeros((300, 400), bool)
    mask[50:250, 50:200] = True  # the panel ends at x 200: the mapped spot (250, 125) is 50 px off it
    _write_mask(tmp_path, "crops/pickup/m.png", mask)
    kfs[kf.kf_id] = kf.model_copy(update={"mask_png": "crops/pickup/m.png"})
    ww, wh = win[2] - win[0], win[3] - win[1]
    box = [243.0, 121.0, 259.0, 129.0]
    rec = _answer([[round((box[0] - win[0]) * 1000 / ww), round((box[1] - win[1]) * 1000 / wh), round((box[2] - win[0]) * 1000 / ww), round((box[3] - win[1]) * 1000 / wh)]])
    ask = second_pass.SecondPassAsk(kf_id=kf.kf_id, role="pickup", panel_id=kf.panel_id, other_mark=other.mark_id, crop_path=kf.crop.path, crop_sha256=kf.crop.sha256, crop_w=400, crop_h=300, window=win, expected_crop=expected, out_w=1280, out_h=720, record=rec)  # type: ignore[union-attr]
    [(_a, got, words)] = second_pass.outcomes([ask], kfs, th=TH, root=tmp_path)
    assert got is None and words.startswith("a box 12.") and words.endswith("off the panel, not used")
    old = json.loads(json.dumps(TH))
    del old["matching"]["second_marks_pass"]["on_panel"]  # friend_pair2's frozen leaf: the box is used, as it was
    assert len(second_pass.marks_from_asks([ask], kfs, th=old, root=tmp_path)) == 1
