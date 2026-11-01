"""Lead 11/1, 2 m laps: the door's glass edge (panels.door_glass_edge), fenders built from wheels
(panels.fender_from_wheels), the view-matched return keyframe (panels.view_match_return) and the seams sentence
(matching.seams_not_marks). Each is off when the run's thresholds lack its leaf."""

from __future__ import annotations

import copy
import hashlib

import numpy as np

from walkaround import contracts as c
from walkaround import geometry, match, panels, spark_client
from walkaround.contracts import Noun, PanelId, Position, Side

TH = c.load_thresholds()
W, H, FPS = 960, 540, 15.0


def frozen(*leaves: str) -> dict:
    """TH without the named leaves (section.leaf), as a run frozen before them."""
    th = copy.deepcopy(TH)
    for leaf in leaves:
        sec, name = leaf.split(".")
        th[sec].pop(name, None)
    return th


def track(role: str, noun: Noun, rows: list[tuple[int, list[float]]], *, track_id: str, side: Side = Side.NONE, position: Position = Position.NONE, pid: PanelId | None = None) -> c.PanelTrack:
    per_frame = [
        c.FrameObs(
            frame=f, pts=f / FPS, box=[float(v) for v in box], box_area_frac=(box[2] - box[0]) * (box[3] - box[1]) / float(W * H),
            in_car_box=True, c=0.8, border_frac=panels.border_frac(box, (W, H)),
        )
        for f, box in rows
    ]
    return c.PanelTrack(
        track_id=track_id, role=role, noun=noun, sam_object_ids=["0"],
        fragments=[c.TrackFragment(object_id="0", first_frame=rows[0][0], last_frame=rows[-1][0])],
        c_present_share=1.0, per_frame=per_frame, panel_id=pid, side=side, position=position,
    )


# panels.door_glass_edge ------------------------------------------------------------------------------------------------


def door_at_the_top() -> c.PanelTrack:
    rows = [(f, [100.0, 60.0, 400.0, 400.0]) for f in range(10)]  # whole, small
    rows += [(f, [40.0, 0.0, 800.0, 520.0]) for f in range(10, 20)]  # the 2 m stop: glass cut by the top edge
    return track("pickup", Noun.CAR_DOOR, rows, track_id="pickup_car_door_00")


def test_the_top_edge_of_a_door_box_is_left_out_only_with_the_leaf() -> None:
    door = door_at_the_top()
    row = door.per_frame[15]
    assert row.border_frac > TH["panels"]["border_frac_keyframe_max"]
    assert panels.pass_border_frac(door, row, frozen("panels.door_glass_edge")) == row.border_frac
    assert panels.pass_border_frac(door, row, TH) == 0.0
    assert 15 not in panels.keyframe_candidates(door, frozen("panels.door_glass_edge"))
    assert panels.keyframe_candidates(door, TH)[0] >= 10  # the big frontal frames now rank first


def test_the_glass_edge_never_frees_a_door_cut_at_the_side_or_another_noun() -> None:
    side_cut = track("pickup", Noun.CAR_DOOR, [(0, [0.0, 0.0, 500.0, 400.0])], track_id="pickup_car_door_01")
    assert panels.pass_border_frac(side_cut, side_cut.per_frame[0], TH) > TH["panels"]["border_frac_keyframe_max"]
    hood = track("pickup", Noun.HOOD, [(0, [40.0, 0.0, 800.0, 520.0])], track_id="pickup_hood_00")
    assert panels.pass_border_frac(hood, hood.per_frame[0], TH) == hood.per_frame[0].border_frac


# panels.fender_from_wheels -----------------------------------------------------------------------------------------------


def wheel(track_id: str, x0: float, *, side: Side, position: Position, pid: PanelId | None, frames: range = range(20)) -> c.PanelTrack:
    return track("pickup", Noun.WHEEL, [(f, [x0, 330.0, x0 + 160.0, 490.0]) for f in frames], track_id=track_id, side=side, position=position, pid=pid)


def test_settings_are_none_on_a_frozen_run_and_nothing_changes() -> None:
    th = frozen("panels.fender_from_wheels")
    assert panels.fender_settings(th) is None
    lane_fender = track("pickup", Noun.FENDER, [(0, [10.0, 300.0, 60.0, 330.0])], track_id="pickup_fender_00", side=Side.LEFT, position=Position.FRONT, pid=PanelId.LEFT_FRONT_FENDER)
    tracks = [lane_fender, wheel("pickup_wheel_00", 200, side=Side.LEFT, position=Position.FRONT, pid=PanelId.LEFT_FRONT_WHEEL)]
    out_panels, out_tracks = panels.apply_wheel_fenders("pickup", [], tracks, {}, (W, H), FPS, th)
    assert out_tracks is tracks and out_panels == []


def test_a_decided_wheel_gives_its_fender_and_the_lane_fender_leaves() -> None:
    cfg = panels.fender_settings(TH)
    assert cfg is not None
    lane_fender = track("pickup", Noun.FENDER, [(0, [10.0, 300.0, 60.0, 330.0])], track_id="pickup_fender_00", side=Side.LEFT, position=Position.FRONT, pid=PanelId.LEFT_FRONT_FENDER)
    w = wheel("pickup_wheel_00", 200, side=Side.LEFT, position=Position.FRONT, pid=PanelId.LEFT_FRONT_WHEEL)
    canon = [c.Panel(panel_id=pid, noun=panels.PANEL_NOUN[pid]) for pid in PanelId]
    out_panels, out_tracks = panels.apply_wheel_fenders("pickup", canon, [lane_fender, w], {f: [0, 0, W, H] for f in range(20)}, (W, H), FPS, TH)
    ids = [t.track_id for t in out_tracks]
    assert "pickup_fender_00" not in ids and "pickup_fender_from_wheel_00" in ids
    built = next(t for t in out_tracks if t.track_id == "pickup_fender_from_wheel_00")
    assert panels.is_wheel_fender(built) and panels.wheel_id_of(built) == "pickup_wheel_00"
    assert built.panel_id is PanelId.LEFT_FRONT_FENDER and len(built.per_frame) == 20
    x1, y1, x2, y2 = built.per_frame[0].box
    assert x1 < 200 and x2 > 360 and y1 < 330 and y2 == 410  # widened, over the wheel, down to its centre
    fp = {p.panel_id: p for p in out_panels}[PanelId.LEFT_FRONT_FENDER]
    assert fp.front_sign == -1 and not fp.side_uncertain and fp.per_role["pickup"].track_ids == ["pickup_fender_from_wheel_00"]
    assert panels.keyframe_candidates(built, TH)  # its estimated box never shuts it out of the box pass


def test_an_undecided_wheel_takes_its_side_from_the_doors_beside_it() -> None:
    w = wheel("pickup_wheel_03", 200, side=Side.NONE, position=Position.FRONT, pid=PanelId.LEFT_FRONT_WHEEL)
    assert set(panels.candidate_slots(w)) == {PanelId.LEFT_FRONT_WHEEL, PanelId.RIGHT_FRONT_WHEEL}
    door = track("pickup", Noun.CAR_DOOR, [(f, [380.0, 120.0, 700.0, 470.0]) for f in range(20)], track_id="pickup_car_door_00", side=Side.RIGHT, position=Position.FRONT, pid=PanelId.RIGHT_FRONT_DOOR)
    cfg = panels.fender_settings(TH)
    built = panels.fender_tracks("pickup", [w, door], {}, (W, H), cfg)
    assert [t.panel_id for t in built] == [PanelId.RIGHT_FRONT_FENDER]
    far = track("pickup", Noun.CAR_DOOR, [(f, [800.0, 120.0, 950.0, 470.0]) for f in range(20)], track_id="pickup_car_door_01", side=Side.RIGHT, position=Position.FRONT, pid=PanelId.RIGHT_FRONT_DOOR)
    assert panels.fender_tracks("pickup", [w, far], {}, (W, H), cfg) == []  # too far from the wheel to vote
    assert panels.fender_tracks("pickup", [w], {}, (W, H), cfg) == []  # no door, no side, no fender


def test_the_fender_mask_is_the_car_near_the_wheel_less_the_named_parts() -> None:
    cfg = panels.fender_settings(TH)
    car = np.zeros((H, W), dtype=bool)
    car[150:500, 50:900] = True
    door = np.zeros_like(car)
    door[150:480, 450:900] = True
    wheel_m = np.zeros_like(car)
    wheel_m[330:490, 200:360] = True
    hood = np.zeros_like(car)
    hood[150:200, 50:450] = True
    mask = panels.wheel_fender_mask([200, 330, 360, 490], car, [door, wheel_m, hood], cfg)
    assert mask is not None
    assert not (mask & (door | wheel_m | hood)).any()
    assert mask[300, 280] and not mask[450, 100]  # over the arch yes; below the centre far ahead of the wheel no
    ys, _xs = np.nonzero(mask)
    assert ys.min() >= int(330 - cfg["above_d"] * 160) - 1  # the band over the wheel, not the roof
    assert panels.wheel_fender_mask([200, 330, 360, 490], np.zeros_like(car), [], cfg) is None


# panels.view_match_return --------------------------------------------------------------------------------------------------


def _rect(h0: int, h1: int, w0: int, w1: int) -> np.ndarray:
    m = np.zeros((200, 300), dtype=bool)
    m[h0:h1, w0:w1] = True
    return m


def _score(total: float, sharp: float = 1.0) -> c.KeyframeScore:
    return c.KeyframeScore(area=0.1, completeness=1.0, sharpness=sharp, glare=0.0, frontality=1.0, total=total)


def test_the_best_matched_return_view_goes_first_only_when_it_beats_the_head() -> None:
    ref = _rect(50, 150, 50, 250)  # wide panel at pickup
    refs = [(ref, geometry.moments_from_mask(ref), True)]
    narrow = _rect(20, 180, 120, 180) | _rect(140, 180, 120, 280)  # an L: no affine map makes it the rectangle
    wide = _rect(55, 150, 45, 245)
    ordered = [(1, _score(0.9)), (2, _score(0.8)), (3, _score(0.7)), (4, _score(0.2))]
    masks = {1: (narrow, True), 2: (narrow, True), 3: (narrow, True), 4: (wide, True)}
    out, ious = panels.view_matched_first(ordered, masks, refs, -1, top_k=3, th=TH)
    assert [f for f, _ in out] == [4, 1, 2, 3] and ious[4] > max(ious[1], ious[2], ious[3])
    same, _ = panels.view_matched_first(ordered, masks, refs, -1, top_k=4, th=TH)
    assert same == ordered  # already among the first top_k
    other_cut = {**masks, 4: (wide, False)}
    kept, ious2 = panels.view_matched_first(ordered, other_cut, refs, -1, top_k=3, th=TH)
    assert kept == ordered and ious2[4] == -1.0  # a different window cut would fail G9
    blurry = [(1, _score(0.9)), (2, _score(0.8)), (3, _score(0.7)), (4, _score(0.2, sharp=0.1))]
    assert panels.view_matched_first(blurry, masks, refs, -1, top_k=3, th=TH)[0] == blurry  # G3 would fail it
    assert panels.view_matched_first(ordered, masks, refs, None, top_k=3, th=TH)[0] == ordered  # G6: no front sign


def test_view_match_is_off_on_a_frozen_run() -> None:
    assert panels.view_match_return(TH) and not panels.view_match_return(frozen("panels.view_match_return"))


# matching.seams_not_marks ----------------------------------------------------------------------------------------------------


def test_the_seams_sentence_is_sent_only_with_the_leaf() -> None:
    old = spark_client.marks_prompt(PanelId.LEFT_FRONT_DOOR, ["handle"], fixtures=True)
    assert spark_client.marks_prompt(PanelId.LEFT_FRONT_DOOR, ["handle"], fixtures=True, seams=False) == old
    new = spark_client.marks_prompt(PanelId.LEFT_FRONT_DOOR, ["handle"], fixtures=True, seams=True)
    assert spark_client.SEAMS_SENTENCE in new and new.replace(spark_client.SEAMS_SENTENCE, "") == old
    assert hashlib.sha256(new.encode()).hexdigest() != hashlib.sha256(old.encode()).hexdigest()
    assert not match.seams_not_marks(frozen("matching.seams_not_marks"))
    assert new.isascii()
