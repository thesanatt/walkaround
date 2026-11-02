"""Doors 11/1: the door check (panels.door_check) files each car door track by side and front or rear from the lids,
mirrors, wheels and the other doors the walk tracked; the edge-on floor (panels.edge_on_floor) keeps slivers out of
the keyframes and coverage says when a panel was only seen edge-on. Both are off when the run's thresholds lack them."""

from __future__ import annotations

import copy

import numpy as np

from walkaround import claim, panels
from walkaround import contracts as c
from walkaround.contracts import CoverageState, LaneStatus, Noun, PanelId, Position, Side

from .test_panels import _pair_c_vote
from .test_walk_2m import FPS, TH, frozen, track

CFG = TH["panels"]["door_check"]


def _boxes(box: list[float], frames: range) -> dict[int, list[float]]:
    return {f: box for f in frames}


# the cues ---------------------------------------------------------------------------------------------------------------

DOOR = [300.0, 100.0, 600.0, 420.0]  # 300 wide, 320 tall


def test_a_mirror_at_the_door_edge_marks_the_front_door_and_a_door_width_away_the_rear() -> None:
    door = _boxes(DOOR, range(10))
    near = {f: [[590.0, 180.0, 640.0, 220.0]] for f in range(10)}  # centre 615: 0.05 door widths past the edge
    far = {f: [[860.0, 180.0, 900.0, 220.0]] for f in range(10)}  # centre 880: 0.93 door widths away
    above = {f: [[590.0, -200.0, 640.0, -150.0]] for f in range(10)}  # outside the door's height band
    assert panels.door_position_cues(door, near, {}, CFG, min_frames=5)[0]["mirror"] == 1
    votes, numbers = panels.door_position_cues(door, far, {}, CFG, min_frames=5)
    assert votes["mirror"] == -1 and numbers["mirror_gap_door_widths"] > 0.9
    assert panels.door_position_cues(door, above, {}, CFG, min_frames=5)[0]["mirror"] == 0
    assert panels.door_position_cues(_boxes(DOOR, range(3)), near, {}, CFG, min_frames=5)[0]["mirror"] == 0  # too few frames


def test_a_wheel_under_the_door_marks_the_rear_door_and_a_clear_door_says_nothing() -> None:
    door = _boxes(DOOR, range(10))
    under = {f: [[250.0, 350.0, 400.0, 500.0]] for f in range(10)}  # two thirds of the wheel under the door
    clear = {f: [[100.0, 350.0, 250.0, 500.0], [650.0, 350.0, 800.0, 500.0]] for f in range(10)}
    assert panels.door_position_cues(door, {}, under, CFG, min_frames=5)[0]["wheel"] == -1
    votes, numbers = panels.door_position_cues(door, {}, clear, CFG, min_frames=5)
    assert votes["wheel"] == 0 and numbers["wheel_under_frames"] == 0  # a rear door's own wheel may be out of the frame


def test_the_hood_and_the_trunk_lid_tell_the_side() -> None:
    door = _boxes(DOOR, range(10))
    hood_right = {f: [([700.0, 200.0, 900.0, 300.0], 1)] for f in range(10)}
    trunk_right = {f: [([700.0, 200.0, 900.0, 300.0], -1)] for f in range(10)}
    assert panels.door_side_cues(door, hood_right, {}, CFG, min_frames=5)[0]["lid_side"] == 1  # front image-right: right side
    assert panels.door_side_cues(door, trunk_right, {}, CFG, min_frames=5)[0]["lid_side"] == -1
    mirror_left = {f: [[150.0, 180.0, 190.0, 220.0]] for f in range(10)}  # outside the box, toward image-left
    assert panels.door_side_cues(door, {}, mirror_left, CFG, min_frames=5)[0]["mirror_side"] == -1
    sliver = _boxes([300.0, 100.0, 380.0, 420.0], range(10))  # an edge-on door: the mirror's direction says nothing
    assert panels.door_side_cues(sliver, {}, mirror_left, CFG, min_frames=5)[0]["mirror_side"] == 0


def test_the_decision_needs_the_margin_or_an_unopposed_prior() -> None:
    w = CFG["weights"]
    assert panels.door_decision({"mirror": 1}, panels.DOOR_POSITION_CUES, -1, w, 2) == (1, 2.0, True)
    assert panels.door_decision({"walk": 1}, panels.DOOR_POSITION_CUES, 1, w, 2) == (1, 1.0, True)
    assert panels.door_decision({}, panels.DOOR_POSITION_CUES, -1, w, 2) == (-1, 0.0, True)
    assert panels.door_decision({"walk": 1}, panels.DOOR_POSITION_CUES, -1, w, 2) == (0, 1.0, False)  # a cue against
    assert panels.door_decision({"mirror": 1, "wheel": -1}, panels.DOOR_POSITION_CUES, 1, w, 2) == (0, 0.0, False)


# the door check on a walk -----------------------------------------------------------------------------------------------


def test_pair_c_left_doors_seen_from_the_rear_corner_leave_the_right_side() -> None:
    _, before = _pair_c_vote(frozen("panels.door_check"))
    _, after = _pair_c_vote(TH)
    was = {t.track_id: t for t in before}
    moved = {t.track_id: t.panel_id for t in after if t.noun is Noun.CAR_DOOR and t.panel_id != was[t.track_id].panel_id}
    assert moved == {"pickup_car_door_04": PanelId.LEFT_REAR_DOOR, "pickup_car_door_05": PanelId.LEFT_FRONT_DOOR}
    by_id = {t.track_id: t for t in after}
    d5 = by_id["pickup_car_door_05"].door_check
    assert d5.was is PanelId.RIGHT_REAR_DOOR and d5.votes["lid_side"] == -1 and d5.side_decided and d5.position_decided
    assert by_id["pickup_car_door_01"].panel_id is PanelId.RIGHT_FRONT_DOOR and by_id["pickup_car_door_01"].door_check.votes["mirror"] == 1
    assert by_id["pickup_car_door_03"].door_check.votes["wheel"] == -1 and by_id["pickup_car_door_03"].door_check.votes["mirror"] == -1
    assert all(t.door_check is None for t in before)


def _side_walk(rear_mirror_gap: float, wheel_under_front: bool) -> list[c.PanelTrack]:
    """A right side seen front to rear: the hood at the image right, the front door then the rear door."""
    frames = range(20)
    hood = track("pickup", Noun.HOOD, [(f, [820.0, 150.0, 950.0, 260.0]) for f in frames], track_id="pickup_hood_00", pid=PanelId.HOOD)
    front = track("pickup", Noun.CAR_DOOR, [(f, [480.0, 100.0, 760.0, 420.0]) for f in frames], track_id="pickup_car_door_00",
                  side=Side.RIGHT, position=Position.FRONT, pid=PanelId.RIGHT_FRONT_DOOR)
    rear = track("pickup", Noun.CAR_DOOR, [(f, [200.0, 100.0, 480.0, 420.0]) for f in frames], track_id="pickup_car_door_01",
                 side=Side.RIGHT, position=Position.REAR, pid=PanelId.RIGHT_REAR_DOOR)
    mx = 760.0 + rear_mirror_gap * 280.0
    mirror = track("pickup", Noun.SIDE_MIRROR, [(f, [mx - 20, 180.0, mx + 20, 220.0]) for f in frames], track_id="pickup_side_mirror_00", pid=PanelId.MIRRORS)
    wheel_x = 600.0 if wheel_under_front else 150.0
    wheel = track("pickup", Noun.WHEEL, [(f, [wheel_x, 360.0, wheel_x + 120, 480.0]) for f in frames], track_id="pickup_wheel_00",
                  side=Side.RIGHT, position=Position.REAR, pid=PanelId.RIGHT_REAR_WHEEL)
    return [hood, front, rear, mirror, wheel]


def test_agreeing_cues_keep_the_doors_and_disagreeing_cues_leave_front_or_rear_uncertain() -> None:
    role = _side_walk(rear_mirror_gap=0.0, wheel_under_front=False)
    ctx = panels._RoleContext(role, None, 2.0)
    panels.apply_door_check(ctx, role, TH)
    assert [t.panel_id for t in role if t.noun is Noun.CAR_DOOR] == [PanelId.RIGHT_FRONT_DOOR, PanelId.RIGHT_REAR_DOOR]
    split = _side_walk(rear_mirror_gap=0.0, wheel_under_front=True)  # the mirror says front, a wheel under it says rear
    ctx = panels._RoleContext(split, None, 2.0)
    panels.apply_door_check(ctx, split, TH)
    front = split[1]
    assert front.side is Side.RIGHT and front.position is Position.NONE and not front.door_check.position_decided
    assert panels.candidate_slots(front) == [PanelId.RIGHT_FRONT_DOOR, PanelId.RIGHT_REAR_DOOR]
    pnl = panels._panels_from(split, {}, fps=FPS, merge_gap_s=1.0, fallback_dt=1 / FPS)
    cov = panels.coverage_for_role(pnl, split, [], role="pickup", fps=FPS, lane_status={n: LaneStatus.COMPLETED for n in Noun}, th=TH)
    assert cov[PanelId.RIGHT_FRONT_DOOR].reason.startswith("front or rear uncertain: pickup_car_door_00 could be right front door or right rear door")


def test_a_door_check_record_is_left_out_of_the_json_when_absent() -> None:
    t = _side_walk(0.0, False)[1]
    plain = c.canonical_json(t)
    assert "door_check" not in plain and "edge_on_frames" not in plain
    checked = t.model_copy(update={"door_check": c.DoorCheck(was=PanelId.RIGHT_FRONT_DOOR), "edge_on_frames": [3]})
    assert "door_check" in c.canonical_json(checked) and "edge_on_frames" in c.canonical_json(checked)


def test_a_door_box_holding_a_lamp_is_an_end_view_never_a_door_candidate() -> None:
    door = track("pickup", Noun.CAR_DOOR, [(f, [200.0, 100.0, 600.0, 400.0]) for f in range(10)], track_id="pickup_car_door_00",
                 side=Side.RIGHT, position=Position.FRONT, pid=PanelId.RIGHT_FRONT_DOOR)
    lamp = track("pickup", Noun.TAIL_LIGHT, [(f, [220.0, 150.0, 300.0, 200.0]) for f in range(5)], track_id="pickup_tail_light_00", pid=PanelId.TAIL_LIGHTS)
    assert panels.door_end_views([door, lamp], TH) == {"pickup_car_door_00": {0, 1, 2, 3, 4}}
    assert panels.door_end_views([door, lamp], frozen("panels.door_check")) == {}
    pnl = [c.Panel(panel_id=PanelId.RIGHT_FRONT_DOOR, noun=Noun.CAR_DOOR)]
    plan = panels._keyframe_plan(pnl, [door, lamp], TH)
    assert plan and all(f >= 5 for f, _ in plan[0][1])


# the edge-on floor ------------------------------------------------------------------------------------------------------


def _raw(aspect: float) -> c.KeyframeScore:
    return c.KeyframeScore(area=0.1, completeness=1.0, sharpness=10.0, glare=0.0, frontality=aspect, total=1.0)


def test_slivers_under_the_floor_are_found_per_noun_and_only_with_the_leaf() -> None:
    door = track("pickup", Noun.CAR_DOOR, [(0, DOOR)], track_id="pickup_car_door_00")
    wheel = track("pickup", Noun.WHEEL, [(1, DOOR)], track_id="pickup_wheel_00")
    scored = [(0, _raw(0.33)), (1, _raw(0.38)), (2, _raw(0.9))]
    owner = {0: door, 1: wheel, 2: door}
    assert panels.edge_on_candidates(scored, owner, TH) == {0}  # the wheel's floor is lower (oblique rims stay)
    assert panels.edge_on_candidates(scored, owner, frozen("panels.edge_on_floor")) == set()
    assert panels.edge_on_floor(TH, Noun.HOOD) is None


def test_a_panel_seen_only_edge_on_says_so_in_coverage() -> None:
    t = track("pickup", Noun.CAR_DOOR, [(f, [300.0, 100.0, 380.0, 420.0]) for f in range(30)], track_id="pickup_car_door_00",
              side=Side.LEFT, position=Position.FRONT, pid=PanelId.LEFT_FRONT_DOOR)
    for r in t.per_frame[:3]:
        r.mask_decoded = True
    pnl = panels._panels_from([t], {}, fps=FPS, merge_gap_s=1.0, fallback_dt=1 / FPS)
    status = {n: LaneStatus.COMPLETED for n in Noun}
    plain = panels.coverage_for_role(pnl, [t], [], role="pickup", fps=FPS, lane_status=status, th=TH)[PanelId.LEFT_FRONT_DOOR]
    assert plain.state is CoverageState.GLIMPSED and "edge-on" not in plain.reason
    t.edge_on_frames = [0, 1, 2]
    edge = panels.coverage_for_role(pnl, [t], [], role="pickup", fps=FPS, lane_status=status, th=TH)[PanelId.LEFT_FRONT_DOOR]
    assert edge.state is CoverageState.GLIMPSED and edge.reason.startswith("only seen edge-on: 2.0 s visible") and edge.best_kf is None
    t.edge_on_frames = [0, 1]  # one candidate was not a sliver: the usual words
    assert "edge-on" not in panels.coverage_for_role(pnl, [t], [], role="pickup", fps=FPS, lane_status=status, th=TH)[PanelId.LEFT_FRONT_DOOR].reason


# the claim photo --------------------------------------------------------------------------------------------------------


class _BoxMasks(claim._Masks):
    def get(self, concept: str, index: int, d: claim.SamDetection) -> np.ndarray:
        m = np.zeros((self.size[1], self.size[0]), bool)
        b = [int(v) for v in d.box]
        m[b[1] : b[3], b[0] : b[2]] = True
        return m


def _answer(concept: str, *boxes: list[float]) -> claim.SamAnswer:
    return claim.SamAnswer(concept=concept, detections=[claim.SamDetection(object_id=str(i), box=b, c=0.9) for i, b in enumerate(boxes)])


def test_the_photo_door_check_reads_the_vote_s_mirror_and_wheel() -> None:
    masks = _BoxMasks((800, 600))
    door = [200.0, 100.0, 500.0, 450.0]
    front = {"car door": _answer("car door", door), "side mirror": _answer("side mirror", [480.0, 180.0, 540.0, 230.0])}
    pos, record = claim.photo_door_check(front, Noun.CAR_DOOR, (350.0, 300.0), masks, TH)
    assert pos is Position.FRONT and record["votes"]["mirror"] == 1 and record["position"] == "front"
    rear = {"car door": _answer("car door", door), "wheel": _answer("wheel", [420.0, 380.0, 560.0, 520.0])}
    assert claim.photo_door_check(rear, Noun.CAR_DOOR, (350.0, 300.0), masks, TH)[0] is Position.REAR
    both = {**front, "wheel": rear["wheel"]}
    pos, record = claim.photo_door_check(both, Noun.CAR_DOOR, (350.0, 300.0), masks, TH)
    assert pos is None and record["decided"] is False
    assert claim.photo_door_check(front, Noun.HOOD, (350.0, 300.0), masks, TH) is None
    assert claim.photo_door_check(front, Noun.CAR_DOOR, (350.0, 300.0), masks, frozen("panels.door_check")) is None


def test_a_frozen_run_votes_its_doors_as_before() -> None:
    old = copy.deepcopy(TH)
    for leaf in ("door_check", "edge_on_floor"):
        old["panels"].pop(leaf)
    role = _side_walk(rear_mirror_gap=0.0, wheel_under_front=True)
    ctx = panels._RoleContext(role, None, 2.0)
    panels.apply_door_check(ctx, role, old)
    assert [(t.panel_id, t.door_check) for t in role if t.noun is Noun.CAR_DOOR] == [(PanelId.RIGHT_FRONT_DOOR, None), (PanelId.RIGHT_REAR_DOOR, None)]
