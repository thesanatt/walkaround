"""Offline tests for walkaround.panels (M5): a scripted walk around a rectangle car, the merge rules one by one, the
side vote in both walk directions, the keyframe passes, the coverage states, the top-up plan, the timeline, and the
10/26 ping pong fixtures replayed through sam_wire into derive() with a real DecodePool. No network."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from meta_sam_parser import build_object_prompt

from walkaround import contracts as c
from walkaround import panels, paths, sam_wire, video
from walkaround.contracts import LANE_NOUNS, CoverageState, LaneStatus, MergeRule, Noun, PanelId, Position, Side, StateBasis

ROOT = Path(__file__).resolve().parent
TEXT_FIXTURE = ROOT / "fixtures" / "sam" / "pingpong_text.events.ndjson"
BOX_FIXTURE = ROOT / "fixtures" / "sam" / "pingpong_box.events.ndjson"
CLIP = ROOT.parent / "data" / "test" / "pingpong.mp4"
TH = c.load_thresholds()

# ----------------------------------------------------------------------------------------------------------------
# The scripted car. u is the perimeter coordinate in metres, counterclockwise from the front centre: the front face
# (0 to 1.8), the left side (1.8 to 6.3), the rear face (6.3 to 8.1), the right side (8.1 to 12.6). The camera faces
# the car and walks along the perimeter at 0.5 m/s; a point at u projects to x = 480 + (u - s) * PX_PER_M in a
# 960x540 proxy frame when the camera stands at s. Walking counterclockwise the scene drifts image-left.
# ----------------------------------------------------------------------------------------------------------------

W, H = 960, 540
FPS = 15.0
PX_PER_M = 200.0
STEP_M = 1.0 / 30.0
WALK_START, WALK_END = 0.9, 12.3
SPLIT_AT = 4.4  # where the left rear door's SAM id changes

# noun, instance key, u centre, width m, y top px, y bottom px, expected canonical slot
LAYOUT: list[tuple[Noun, str, float, float, int, int, PanelId | None]] = [
    (Noun.BUMPER, "front", 0.9, 1.8, 380, 460, PanelId.FRONT_BUMPER),
    (Noun.HEADLIGHT, "right_a", 0.3, 0.5, 300, 360, PanelId.HEADLIGHTS),
    (Noun.HEADLIGHT, "left", 1.5, 0.5, 300, 360, PanelId.HEADLIGHTS),
    (Noun.HOOD, "hood", 0.9, 1.2, 200, 300, PanelId.HOOD),
    (Noun.WINDSHIELD, "windshield", 0.9, 1.5, 100, 190, PanelId.WINDSHIELD),
    (Noun.FENDER, "left_front", 2.4, 1.0, 300, 400, PanelId.LEFT_FRONT_FENDER),
    (Noun.WHEEL, "left_front", 2.5, 0.65, 400, 530, PanelId.LEFT_FRONT_WHEEL),
    (Noun.SIDE_MIRROR, "left", 3.1, 0.3, 220, 280, PanelId.MIRRORS),
    (Noun.CAR_DOOR, "left_front", 3.6, 1.0, 250, 430, PanelId.LEFT_FRONT_DOOR),
    (Noun.CAR_WINDOW, "left_front", 3.6, 0.9, 150, 250, None),
    (Noun.CAR_DOOR, "left_rear", 4.6, 1.0, 250, 430, PanelId.LEFT_REAR_DOOR),
    (Noun.CAR_WINDOW, "left_rear", 4.6, 0.9, 150, 250, None),
    (Noun.ROOF, "roof", 4.1, 2.0, 80, 150, PanelId.ROOF),
    (Noun.WHEEL, "left_rear", 5.6, 0.65, 400, 530, PanelId.LEFT_REAR_WHEEL),
    (Noun.FENDER, "left_rear", 5.8, 1.0, 300, 400, PanelId.LEFT_REAR_FENDER),
    (Noun.TAIL_LIGHT, "left", 6.6, 0.4, 300, 360, PanelId.TAIL_LIGHTS),
    (Noun.BUMPER, "rear", 7.2, 1.8, 380, 460, PanelId.REAR_BUMPER),
    (Noun.TRUNK_LID, "trunk", 7.2, 1.4, 220, 340, PanelId.TRUNK_LID),
    (Noun.REAR_WINDOW, "rear_window", 7.2, 1.3, 100, 200, PanelId.REAR_WINDOW),
    (Noun.TAIL_LIGHT, "right", 7.8, 0.4, 300, 360, PanelId.TAIL_LIGHTS),
    (Noun.FENDER, "right_rear", 8.6, 1.0, 300, 400, PanelId.RIGHT_REAR_FENDER),
    (Noun.WHEEL, "right_rear", 8.8, 0.65, 400, 530, PanelId.RIGHT_REAR_WHEEL),
    (Noun.CAR_DOOR, "right_rear", 9.8, 1.0, 250, 430, PanelId.RIGHT_REAR_DOOR),
    (Noun.CAR_DOOR, "right_front", 10.8, 1.0, 250, 430, PanelId.RIGHT_FRONT_DOOR),
    (Noun.SIDE_MIRROR, "right", 11.3, 0.3, 220, 280, PanelId.MIRRORS),
    (Noun.WHEEL, "right_front", 11.9, 0.65, 400, 530, PanelId.RIGHT_FRONT_WHEEL),
    (Noun.FENDER, "right_front", 12.0, 1.0, 300, 400, PanelId.RIGHT_FRONT_FENDER),
    (Noun.HEADLIGHT, "right_b", 12.9, 0.5, 300, 360, PanelId.HEADLIGHTS),
]
EXPECTED_SLOT = {(noun, key): pid for noun, key, _u, _w, _y0, _y1, pid in LAYOUT}


def _box_at(u: float, width_m: float, y0: int, y1: int, s: float) -> list[int] | None:
    x0 = 480.0 + (u - width_m / 2 - s) * PX_PER_M
    x1 = 480.0 + (u + width_m / 2 - s) * PX_PER_M
    cx0, cx1 = max(0.0, x0), min(float(W), x1)
    if cx1 - cx0 < 20:
        return None
    return [int(round(cx0)), y0, int(round(cx1)), y1]


def scripted_walk(
    *,
    reverse: bool = False,
    split_gap_frames: int = 7,
    coupe: bool = False,
    roof: bool = True,
    c_none: tuple[tuple[Noun, str], ...] = (),
    c_default: float = 0.8,
) -> tuple[dict[Noun, list[c.IndexFrame]], dict[tuple[Noun, str], str], list[float]]:
    """Lanes of IndexFrames per noun for one walk, the SAM id of every instance, and the pts list."""
    n = int(round((WALK_END - WALK_START) / STEP_M)) + 1
    positions = [WALK_START + i * STEP_M for i in range(n)]
    if reverse:
        positions.reverse()
    lanes: dict[Noun, list[c.IndexFrame]] = {noun: [] for noun in LANE_NOUNS}
    ids: dict[tuple[Noun, str], str] = {}
    half_gap = (split_gap_frames / 2.0) * STEP_M
    for i, s in enumerate(positions):
        objs: dict[Noun, list[c.IndexObject]] = {}
        boxes: list[list[int]] = []
        for noun, key, u, wm, y0, y1, _pid in LAYOUT:
            if coupe and noun is Noun.CAR_DOOR and key == "right_rear":
                continue
            if not roof and noun is Noun.ROOF:
                continue
            box = _box_at(u, wm, y0, y1, s)
            if box is None:
                continue
            id_key = key
            if noun is Noun.CAR_DOOR and key == "left_rear" and split_gap_frames > 0:
                if abs(s - SPLIT_AT) < half_gap:
                    continue
                id_key = "left_rear_a" if s < SPLIT_AT else "left_rear_b"
            oid = ids.setdefault((noun, id_key), str(len([k for k in ids if k[0] is noun])))
            conf = None if (noun, key) in c_none else c_default
            objs.setdefault(noun, []).append(c.IndexObject(id=oid, box=box, c=conf))
            boxes.append(box)
        if not boxes:
            continue
        car = [
            max(0, min(b[0] for b in boxes) - 10),
            max(0, min(b[1] for b in boxes) - 10),
            min(W, max(b[2] for b in boxes) + 10),
            min(H, max(b[3] for b in boxes) + 10),
        ]
        objs[Noun.CAR] = [c.IndexObject(id="0", box=car, c=0.9)]
        for noun, items in objs.items():
            lanes[noun].append(c.IndexFrame(frame=i, objects=items))
    pts = [i / FPS for i in range(n)]
    return lanes, ids, pts


def tracks_for_walk(role: str, lanes: dict[Noun, list[c.IndexFrame]], pts: list[float], th: dict[str, Any] = TH) -> list[c.PanelTrack]:
    """filter -> fragments -> merge -> tracks for every noun of one walk, as derive() does it."""
    car_boxes = panels.car_boxes_by_frame(lanes[Noun.CAR])
    out: list[c.PanelTrack] = []
    for noun in LANE_NOUNS:
        kept = lanes[noun] if noun in c.PRIVACY_NOUNS else panels.filter_objects(lanes[noun], noun, car_boxes, (W, H), th)
        frags = panels.build_fragments(kept, fps=FPS, th=th)
        groups = panels.merge_fragments(frags, kept, fps=FPS, th=th)
        out.extend(panels.build_tracks(role, noun, groups, kept, pts, car_boxes, (W, H), th=th))
    return out


def track_of(tracks: list[c.PanelTrack], role: str, noun: Noun, object_id: str) -> c.PanelTrack:
    for t in tracks:
        if t.role == role and t.noun is noun and object_id in t.sam_object_ids:
            return t
    raise AssertionError(f"no {role} {noun.value} track with id {object_id}")


def _track(role: str, noun: Noun, rows: list[tuple[int, list[float], float | None]], *, track_id: str | None = None, fps: float = FPS) -> c.PanelTrack:
    """A hand-made PanelTrack from (frame, box, c) rows on a 960x540 frame, one fragment."""
    frame_area = float(W * H)
    per_frame = [
        c.FrameObs(
            frame=f,
            pts=f / fps,
            box=[float(v) for v in box],
            box_area_frac=(box[2] - box[0]) * (box[3] - box[1]) / frame_area,
            in_car_box=True,
            c=conf,
            border_frac=panels.border_frac(box, (W, H)),
        )
        for f, box, conf in rows
    ]
    n_c = sum(1 for r in per_frame if r.c is not None)
    return c.PanelTrack(
        track_id=track_id or f"{role}_{c.noun_slug(noun)}_{rows[0][0]:03d}",
        role=role,
        noun=noun,
        sam_object_ids=["0"],
        fragments=[c.TrackFragment(object_id="0", first_frame=rows[0][0], last_frame=rows[-1][0])],
        c_present_share=n_c / len(per_frame),
        per_frame=per_frame,
    )


def _moving(role: str, noun: Noun, frames: range, x0: float, dx: float, *, w: float = 200, y0: float = 250, h: float = 180, conf: float | None = 0.8) -> c.PanelTrack:
    rows = [(f, [x0 + dx * (f - frames[0]), y0, x0 + dx * (f - frames[0]) + w, y0 + h], conf) for f in frames]
    return _track(role, noun, rows)


def _keyframe(role: str, pid: PanelId, track_id: str, f: int, *, completeness: float = 0.95, area: float = 0.05, sharpness: float = 0.9, kf_id: str | None = None) -> c.Keyframe:
    total = area * completeness * sharpness
    return c.Keyframe(
        kf_id=kf_id or f"{role}_{pid.value}_kf{f}",
        role=role,
        panel_id=pid,
        track_id=track_id,
        f_proxy=f,
        pts=f / FPS,
        score=c.KeyframeScore(area=area, completeness=completeness, sharpness=sharpness, glare=0.0, frontality=1.0, total=total),
        aspect_ratio=1.2,
        box_src=[0.0, 0.0, 10.0, 10.0],
    )


@pytest.fixture(scope="module")
def pickup_walk() -> tuple[dict[Noun, list[c.IndexFrame]], dict[tuple[Noun, str], str], list[float]]:
    return scripted_walk()


@pytest.fixture(scope="module")
def pickup_tracks(pickup_walk) -> list[c.PanelTrack]:
    lanes, _ids, pts = pickup_walk
    return tracks_for_walk("pickup", lanes, pts)


@pytest.fixture(scope="module")
def voted(pickup_tracks) -> tuple[list[c.Panel], list[c.PanelTrack], int | None]:
    direction = panels.walk_direction(pickup_tracks, TH)
    voted_panels, tracks = panels.vote_sides(pickup_tracks, direction, TH, fps=FPS)
    return voted_panels, tracks, direction


# ----------------------------------------------------------------------------------------------------------------
# 5.1 box rules
# ----------------------------------------------------------------------------------------------------------------


def test_car_boxes_by_frame_keeps_the_largest_box_per_frame() -> None:
    frames = [
        c.IndexFrame(frame=3, objects=[c.IndexObject(id="0", box=[0, 0, 10, 10]), c.IndexObject(id="1", box=[5, 5, 100, 100])]),
        c.IndexFrame(frame=4, objects=[]),
    ]
    assert panels.car_boxes_by_frame(frames) == {3: [5, 5, 100, 100]}


def test_border_frac_counts_the_perimeter_on_the_frame_edge() -> None:
    assert panels.border_frac([100, 100, 200, 200], (960, 540)) == 0.0
    assert panels.border_frac([0, 100, 100, 200], (960, 540)) == pytest.approx(0.25)
    assert panels.border_frac([0, 0, 100, 100], (960, 540)) == pytest.approx(0.5)
    assert panels.border_frac([0, 0, 960, 540], (960, 540)) == pytest.approx(1.0)
    assert panels.border_frac([0, 0, 0, 0], (960, 540)) == 0.0


def test_filter_objects_applies_containment_and_the_per_noun_floor() -> None:
    car = {0: [100, 100, 600, 400]}
    big_inside = c.IndexObject(id="0", box=[150, 150, 350, 300], c=0.5)  # 30000 px, 5.8%
    big_outside = c.IndexObject(id="1", box=[500, 150, 700, 300], c=0.5)  # half outside the car box
    small = c.IndexObject(id="2", box=[200, 200, 260, 250], c=None)  # 3000 px, 0.58%
    frames = [c.IndexFrame(frame=0, objects=[big_inside, big_outside, small]), c.IndexFrame(frame=1, objects=[big_inside])]
    doors = panels.filter_objects(frames, Noun.CAR_DOOR, car, (W, H), TH)
    assert [o.id for o in doors[0].objects] == ["0"]
    assert len(doors) == 1  # frame 1 has no car box and is dropped
    mirrors = panels.filter_objects(frames, Noun.SIDE_MIRROR, car, (W, H), TH)
    assert [o.id for o in mirrors[0].objects] == ["0", "2"]  # small nouns use the 0.5% floor; c None is kept


def test_build_fragments_splits_long_gaps_and_drops_short_runs() -> None:
    frames = [c.IndexFrame(frame=f, objects=[c.IndexObject(id="7", box=[0, 0, 10, 10])]) for f in [*range(0, 20), *range(40, 60), 90, 91]]
    frags = panels.build_fragments(frames, fps=FPS, th=TH)
    assert [(fr.first_frame, fr.last_frame) for fr in frags] == [(0, 19), (40, 59)]  # a 2-frame run is under 8
    assert panels.min_fragment_frames(FPS, TH) == 8
    assert panels.split_gap_frames(FPS, TH) == pytest.approx(7.5)


def _frag_frames(oid: str, frames: range, x0: float, dx: float, w: float = 200.0, h: float = 180.0) -> list[c.IndexFrame]:
    return [c.IndexFrame(frame=f, objects=[c.IndexObject(id=oid, box=[int(x0 + dx * (f - frames[0])), 250, int(x0 + dx * (f - frames[0]) + w), int(250 + h)])]) for f in frames]


def test_merge_rules_one_by_one() -> None:
    a = c.TrackFragment(object_id="1", first_frame=0, last_frame=29)
    base = _frag_frames("1", range(0, 30), 700.0, -6.0)
    lookup = panels._box_lookup(base)

    def merge_with(b_frames: list[c.IndexFrame], first: int, last: int) -> c.MergeRecord | None:
        b = c.TrackFragment(object_id="2", first_frame=first, last_frame=last)
        return panels.merge_test(a, b, panels._box_lookup(base + b_frames), fps=FPS, th=TH)

    # the continuation predicted by the median velocity: 700 - 6 * 36 = 484 at frame 36
    good = merge_with(_frag_frames("2", range(36, 60), 484.0, -6.0), 36, 59)
    assert good is not None and good.rule is MergeRule.VELOCITY_GAP
    assert good.gap_s == pytest.approx(7 / FPS) and good.pred_dist < 0.05 and good.area_ratio == pytest.approx(1.0)
    # gap over 1.0 s
    assert merge_with(_frag_frames("2", range(46, 70), 424.0, -6.0), 46, 69) is None
    # predicted centroid off by more than half a box width
    assert merge_with(_frag_frames("2", range(36, 60), 300.0, -6.0), 36, 59) is None
    # area ratio outside [0.6, 1.6]
    assert merge_with(_frag_frames("2", range(36, 60), 484.0, -6.0, w=400.0, h=360.0), 36, 59) is None
    # aspect ratio off by more than 20% (same area, different shape)
    assert merge_with(_frag_frames("2", range(36, 60), 484.0, -6.0, w=270.0, h=133.3), 36, 59) is None
    # overlapping fragments never merge: two instances on screen at once
    assert panels.merge_test(a, c.TrackFragment(object_id="2", first_frame=20, last_frame=40), lookup, fps=FPS, th=TH) is None
    assert panels._box_lookup([]) == {}


# ----------------------------------------------------------------------------------------------------------------
# The scripted walk: fragments, merges, tracks
# ----------------------------------------------------------------------------------------------------------------


def test_walk_tracks_merge_the_split_door_and_nothing_else(pickup_walk, pickup_tracks) -> None:
    lanes, ids, _pts = pickup_walk
    doors = [t for t in pickup_tracks if t.noun is Noun.CAR_DOOR]
    assert len(doors) == 4
    rear = track_of(pickup_tracks, "pickup", Noun.CAR_DOOR, ids[(Noun.CAR_DOOR, "left_rear_a")])
    assert rear.sam_object_ids == [ids[(Noun.CAR_DOOR, "left_rear_a")], ids[(Noun.CAR_DOOR, "left_rear_b")]]
    assert len(rear.fragments) == 2 and len(rear.merges) == 1
    merge = rear.merges[0]
    assert merge.rule is MergeRule.VELOCITY_GAP
    assert merge.gap_s == pytest.approx(8 / FPS)
    assert merge.pred_dist is not None and merge.pred_dist < 0.05
    assert merge.area_ratio == pytest.approx(1.0, abs=0.02)
    assert all(len(t.merges) == 0 for t in pickup_tracks if t is not rear)
    frames = [r.frame for r in rear.per_frame]
    assert frames == sorted(set(frames)) and rear.c_present_share == 1.0
    assert all(r.in_car_box for r in rear.per_frame)
    # every instance of the layout became exactly one track
    for (noun, key), oid in ids.items():
        if key in ("left_rear_a", "left_rear_b"):
            continue
        assert track_of(pickup_tracks, "pickup", noun, oid).noun is noun
    assert sum(1 for t in pickup_tracks if t.noun is Noun.CAR) == 1
    assert len(lanes[Noun.CAR]) == len(set(r.frame for t in pickup_tracks for r in t.per_frame if t.noun is Noun.CAR))


def test_walk_tracks_are_pure_and_deterministic(pickup_walk) -> None:
    lanes, _ids, pts = pickup_walk
    a = tracks_for_walk("pickup", lanes, pts)
    b = tracks_for_walk("pickup", lanes, pts)
    assert c.canonical_json(a) == c.canonical_json(b)
    ids = [t.track_id for t in a]
    assert len(set(ids)) == len(ids)
    for noun in LANE_NOUNS:
        mine = [t for t in a if t.noun is noun]
        assert [t.track_id for t in mine] == [f"pickup_{c.noun_slug(noun)}_{i:02d}" for i in range(len(mine))]
        assert [t.per_frame[0].frame for t in mine] == sorted(t.per_frame[0].frame for t in mine)


def test_walk_direction_follows_the_scene_drift(pickup_tracks) -> None:
    assert panels.walk_direction(pickup_tracks, TH) == -1
    lanes, _ids, pts = scripted_walk(reverse=True)
    assert panels.walk_direction(tracks_for_walk("return", lanes, pts), TH) == 1
    still = [_moving("pickup", Noun.CAR_DOOR, range(0, 30), 300.0, 0.5)]
    assert panels.walk_direction(still, TH) is None
    # the phase-correlation backup is a sign only and stays off until the flag is on
    assert panels.walk_direction(still, TH, backup=1) is None
    th_on = copy.deepcopy(TH)
    th_on["panels"]["direction_phase_corr"] = True
    assert panels.walk_direction(still, th_on, backup=1) == 1
    assert panels.walk_direction(still, th_on, backup=None) is None


def test_phase_correlation_sign_reads_the_shift_direction() -> None:
    rng = np.random.default_rng(7)
    base = rng.random((120, 160)).astype(np.float32)
    import cv2

    base = cv2.GaussianBlur(base, (0, 0), 2)
    right = np.roll(base, 8, axis=1)
    assert panels.phase_correlation_sign(base, right, TH) == 1
    assert panels.phase_correlation_sign(right, base, TH) == -1
    assert panels.phase_correlation_sign(base, np.roll(base, 1, axis=1), TH) is None
    assert panels.phase_correlation_sign(base, base[:60], TH) is None


# ----------------------------------------------------------------------------------------------------------------
# 5.2 the side vote and canonical names
# ----------------------------------------------------------------------------------------------------------------


def test_side_vote_names_every_panel_of_the_walk(pickup_walk, voted) -> None:
    _lanes, ids, _pts = pickup_walk
    voted_panels, tracks, direction = voted
    assert direction == -1
    by_id = {t.track_id: t for t in tracks}
    for (noun, key), oid in ids.items():
        layout_key = key.replace("_a", "").replace("_b", "") if noun is Noun.CAR_DOOR else key
        expected = EXPECTED_SLOT[(noun, layout_key)]
        t = track_of(tracks, "pickup", noun, oid)
        assert t.panel_id is expected, f"{noun.value} {key}: got {t.panel_id}, want {expected}"
    for t in tracks:
        if t.noun in (Noun.CAR, Noun.CAR_WINDOW):
            assert t.panel_id is None and t.side is Side.NONE
    assert len(voted_panels) == len(PanelId)
    assert [p.panel_id for p in voted_panels] == list(PanelId)
    with_tracks = {p.panel_id for p in voted_panels if p.instance_from is c.InstanceFrom.TRACKS}
    assert with_tracks == set(PanelId)  # the scripted sedan shows every slot
    for p in voted_panels:
        assert not p.side_uncertain, p.panel_id
        if p.side is Side.LEFT:
            assert p.front_sign == -1
        elif p.side is Side.RIGHT:
            assert p.front_sign == 1
        if p.panel_id in (PanelId.HOOD, PanelId.FRONT_BUMPER, PanelId.HEADLIGHTS, PanelId.WINDSHIELD):
            assert p.front_sign == 1
        if p.panel_id in (PanelId.TRUNK_LID, PanelId.REAR_BUMPER, PanelId.TAIL_LIGHTS, PanelId.REAR_WINDOW):
            assert p.front_sign == -1
        if p.panel_id in (PanelId.ROOF, PanelId.MIRRORS):
            assert p.front_sign is None
        assert "pickup" in p.per_role and p.per_role["pickup"].seconds_visible > 0
    headlights = next(p for p in voted_panels if p.panel_id is PanelId.HEADLIGHTS)
    assert len(headlights.per_role["pickup"].track_ids) == 3  # the right headlight is seen at both ends of the walk
    door = next(p for p in voted_panels if p.panel_id is PanelId.LEFT_FRONT_DOOR)
    assert door.votes.R1 == -2.0 and door.votes.R2 == -1.0 and door.votes.R3 == -1.0 and door.votes.R4 == -1.0
    wheel = next(p for p in voted_panels if p.panel_id is PanelId.RIGHT_REAR_WHEEL)
    assert wheel.votes.R1 == 0.0 and wheel.votes.R2 + wheel.votes.R3 + wheel.votes.R4 == 3.0
    assert by_id[door.per_role["pickup"].track_ids[0]].position is Position.FRONT


def test_side_vote_agrees_on_the_reverse_walk() -> None:
    lanes, ids, pts = scripted_walk(reverse=True)
    tracks = tracks_for_walk("return", lanes, pts)
    direction = panels.walk_direction(tracks, TH)
    assert direction == 1
    voted_panels, voted_tracks = panels.vote_sides(tracks, direction, TH, fps=FPS)
    for (noun, key), oid in ids.items():
        layout_key = key.replace("_a", "").replace("_b", "") if noun is Noun.CAR_DOOR else key
        assert track_of(voted_tracks, "return", noun, oid).panel_id is EXPECTED_SLOT[(noun, layout_key)], (noun, key)
    assert not any(p.side_uncertain for p in voted_panels)


def test_vote_sides_uses_each_roles_own_direction(pickup_tracks) -> None:
    lanes, ids, pts = scripted_walk(reverse=True)
    both = pickup_tracks + tracks_for_walk("return", lanes, pts)
    by_role = {"pickup": -1, "return": 1}
    voted_panels, tracks = panels.vote_sides(both, -1, TH, fps=FPS, direction_by_role=by_role)
    for (noun, key), oid in ids.items():
        layout_key = key.replace("_a", "").replace("_b", "") if noun is Noun.CAR_DOOR else key
        assert track_of(tracks, "return", noun, oid).panel_id is EXPECTED_SLOT[(noun, layout_key)]
    door = next(p for p in voted_panels if p.panel_id is PanelId.LEFT_FRONT_DOOR)
    assert set(door.per_role) == {"pickup", "return"}
    assert door.votes.R3 == -2.0  # one R3 vote per walk, both left
    # the wrong direction for the return walk flips R3 and costs the margin on R3 alone, never the side
    wrong_panels, _ = panels.vote_sides(both, -1, TH, fps=FPS)
    wrong_door = next(p for p in wrong_panels if p.panel_id is PanelId.LEFT_FRONT_DOOR)
    assert wrong_door.votes.R3 == 0.0 and wrong_door.side is Side.LEFT


def test_side_vote_tie_prints_side_uncertain() -> None:
    lone = [_moving("pickup", Noun.CAR_DOOR, range(0, 40), 700.0, -6.0)]
    voted_panels, tracks = panels.vote_sides(lone, -1, TH, fps=FPS)
    assert tracks[0].panel_id is PanelId.LEFT_FRONT_DOOR
    panel = next(p for p in voted_panels if p.panel_id is PanelId.LEFT_FRONT_DOOR)
    assert panel.side_uncertain and panel.front_sign is None
    assert panel.votes == c.SideVotes()
    assert panel.instance_from is c.InstanceFrom.TRACKS
    canonical = [p for p in voted_panels if p.instance_from is c.InstanceFrom.CANONICAL]
    assert len(canonical) == len(PanelId) - 1 and all(p.front_sign is None and not p.per_role for p in canonical)


def test_bumper_goes_front_or_rear_by_the_light_it_shares_frames_with() -> None:
    bumper = _moving("pickup", Noun.BUMPER, range(0, 40), 300.0, -4.0, w=360, y0=380, h=80)
    head = _moving("pickup", Noun.HEADLIGHT, range(0, 40), 350.0, -4.0, w=100, y0=300, h=60)
    tail = _moving("pickup", Noun.TAIL_LIGHT, range(60, 100), 350.0, -4.0, w=80, y0=300, h=60)
    _, tracks = panels.vote_sides([bumper, head, tail], -1, TH, fps=FPS)
    assert tracks[0].panel_id is PanelId.FRONT_BUMPER and tracks[0].position is Position.FRONT
    rear = _moving("pickup", Noun.BUMPER, range(60, 100), 300.0, -4.0, w=360, y0=380, h=80)
    _, tracks = panels.vote_sides([rear, head, tail], -1, TH, fps=FPS)
    assert tracks[0].panel_id is PanelId.REAR_BUMPER and tracks[0].position is Position.REAR
    alone, tracks = panels.vote_sides([bumper], -1, TH, fps=FPS)
    assert tracks[0].panel_id is PanelId.FRONT_BUMPER
    assert next(p for p in alone if p.panel_id is PanelId.FRONT_BUMPER).side_uncertain


def test_coupe_keeps_its_three_doors_named() -> None:
    lanes, ids, pts = scripted_walk(coupe=True)
    tracks = tracks_for_walk("pickup", lanes, pts)
    voted_panels, tracks = panels.vote_sides(tracks, -1, TH, fps=FPS)
    assert track_of(tracks, "pickup", Noun.CAR_DOOR, ids[(Noun.CAR_DOOR, "right_front")]).panel_id is PanelId.RIGHT_FRONT_DOOR
    rear = next(p for p in voted_panels if p.panel_id is PanelId.RIGHT_REAR_DOOR)
    assert rear.instance_from is c.InstanceFrom.CANONICAL and not rear.per_role


# ----------------------------------------------------------------------------------------------------------------
# The anchor pass
# ----------------------------------------------------------------------------------------------------------------


def test_needs_anchor_pass_and_absorb_bridge_a_long_gap() -> None:
    lanes, ids, pts = scripted_walk(split_gap_frames=25)
    tracks = tracks_for_walk("pickup", lanes, pts)
    a = track_of(tracks, "pickup", Noun.CAR_DOOR, ids[(Noun.CAR_DOOR, "left_rear_a")])
    b = track_of(tracks, "pickup", Noun.CAR_DOOR, ids[(Noun.CAR_DOOR, "left_rear_b")])
    assert a is not b and not a.merges and not b.merges  # 26 frames is 1.73 s, over the merge gap
    doors = [t for t in tracks if t.noun is Noun.CAR_DOOR]
    kept = panels.filter_objects(lanes[Noun.CAR_DOOR], Noun.CAR_DOOR, panels.car_boxes_by_frame(lanes[Noun.CAR]), (W, H), TH)
    assert panels.needs_anchor_pass(doors, kept, TH)  # per noun before the vote
    _, voted_tracks = panels.vote_sides(tracks, -1, TH, fps=FPS)
    assert panels.fragmented_panels(voted_tracks, TH) == [PanelId.LEFT_REAR_DOOR]
    assert panels.needs_anchor_pass([t for t in voted_tracks if t.noun is Noun.CAR_DOOR], kept, TH)
    whole, _ids2, _pts2 = scripted_walk(split_gap_frames=0)
    whole_tracks = tracks_for_walk("pickup", whole, _pts2)
    _, whole_voted = panels.vote_sides(whole_tracks, -1, TH, fps=FPS)
    assert panels.fragmented_panels(whole_voted, TH) == []
    assert not panels.needs_anchor_pass([], [], TH)

    # the anchor lane: object "1" tracked over the whole window on a clip that starts 30 frames before a
    first, last = a.per_frame[0].frame, b.per_frame[-1].frame
    offset = first - 30
    u, wm = 4.6, 1.0
    anchor_frames = []
    for f in range(first, last + 1):
        s = WALK_START + f * STEP_M
        box = _box_at(u, wm, 250, 430, s)
        if box is not None:
            anchor_frames.append(c.IndexFrame(frame=f - offset, objects=[c.IndexObject(id="1", box=box, c=0.9)]))
    merged = panels.absorb_anchor([a, b], anchor_frames, frame_offset=offset, th=TH, media_size=(W, H), pts=pts, fps=FPS, request_id="resp_anchor")
    assert len(merged) == 1
    m = merged[0]
    assert m.track_id == a.track_id
    assert set(m.sam_object_ids) >= {a.sam_object_ids[0], b.sam_object_ids[0]}
    assert any(r.rule is MergeRule.ANCHOR_ABSORB for r in m.merges)
    assert m.anchor_reprompts and m.anchor_reprompts[0].request_id == "resp_anchor"
    frames = [r.frame for r in m.per_frame]
    assert frames == list(range(first, last + 1))  # the gap is filled from the anchor
    assert all(r.in_car_box for r in m.per_frame)
    assert m.c_present_share == 1.0
    # a track the anchor does not overlap comes back untouched
    other = track_of(tracks, "pickup", Noun.CAR_DOOR, ids[(Noun.CAR_DOOR, "right_front")])
    out = panels.absorb_anchor([a, b, other], anchor_frames, frame_offset=offset, th=TH, media_size=(W, H), pts=pts, fps=FPS)
    assert len(out) == 2 and any(t.track_id == other.track_id and not t.merges for t in out)


def test_anchor_prompts_build_a_valid_object_prompt(pickup_tracks) -> None:
    door = next(t for t in pickup_tracks if t.noun is Noun.CAR_DOOR)
    frames = [r.frame for r in door.per_frame][5:40:8]  # five frames; only three may be used
    objects = panels.anchor_prompts(door, frames, None, (W, H), TH)
    assert len(objects) == 1 and len(objects[0].prompts) == 3
    for fp in objects[0].prompts:
        assert fp.box is not None and 0 <= fp.box.left < fp.box.right <= W and 0 <= fp.box.top < fp.box.bottom <= H
        assert fp.points == ()
    text = build_object_prompt(size=(W, H), objects=objects)
    assert text.startswith("<") and "|>" in text
    window = np.zeros((H, W), dtype=bool)
    row = door.per_frame[5]
    x1, y1, x2, y2 = (int(v) for v in row.box)
    window[y1 + 10 : y1 + 40, x1 + 10 : x1 + 60] = True
    with_points = panels.anchor_prompts(door, [row.frame], {row.frame: window}, (W, H), TH, frame_offset=row.frame - 2)
    fp = with_points[0].prompts[0]
    assert fp.frame == 2 and len(fp.points) == 1 and fp.points[0].label == "negative"
    assert x1 + 10 <= fp.points[0].x < x1 + 60 and y1 + 10 <= fp.points[0].y < y1 + 40
    assert panels.anchor_prompts(door, [10 ** 6], None, (W, H), TH) == []


# ----------------------------------------------------------------------------------------------------------------
# 5.3 keyframes
# ----------------------------------------------------------------------------------------------------------------


def test_keyframe_candidates_prefer_big_centred_frontal_boxes_and_skip_border_frames() -> None:
    rows: list[tuple[int, list[float], float | None]] = []
    for f in range(0, 30):
        rows.append((f, [300.0, 250.0, 500.0, 430.0], 0.8))  # steady 200x180
    rows.append((30, [0.0, 250.0, 200.0, 430.0], 0.8))  # touches the left edge: border_frac 180/760 = 0.24
    rows.append((31, [250.0, 250.0, 550.0, 430.0], 0.8))  # the widest box, best aspect
    rows.append((32, [250.0, 250.0, 550.0, 430.0], 0.8))
    track = _track("pickup", Noun.CAR_DOOR, rows)
    cands = panels.keyframe_candidates(track, TH)
    assert len(cands) == TH["panels"]["keyframe_box_pass_candidates"] == 9
    assert cands[:2] == [31, 32] and 30 not in cands
    assert panels.box_pass_score(track, 31) > panels.box_pass_score(track, 0) > 0
    assert panels.box_pass_score(track, 999) == 0.0
    none_in_car = track.model_copy(deep=True)
    for r in none_in_car.per_frame:
        r.in_car_box = False
    assert panels.keyframe_candidates(none_in_car, TH) == []


def test_select_keyframes_keeps_the_top_three_half_a_second_apart() -> None:
    def s(total: float) -> c.KeyframeScore:
        return c.KeyframeScore(area=0.1, completeness=1.0, sharpness=1.0, glare=0.0, frontality=1.0, total=total)

    scored = [(10, s(0.9)), (12, s(0.85)), (14, s(0.8)), (30, s(0.7)), (31, s(0.75)), (60, s(0.5)), (70, s(0.0))]
    chosen = panels.select_keyframes(scored, fps=FPS, th=TH)
    assert chosen == [10, 31, 60]
    for i, f in enumerate(chosen):
        for g in chosen[i + 1 :]:
            assert abs(f - g) >= TH["panels"]["keyframe_min_gap_s"] * FPS
    assert panels.select_keyframes([(5, s(0.0))], fps=FPS, th=TH) == []
    assert panels.select_keyframes([], fps=FPS, th=TH) == []


def _panel_image() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    rgb = np.full((H, W, 3), 120, dtype=np.uint8)
    rgb[250:430, 300:500] = rng.integers(60, 200, size=(180, 200, 3), dtype=np.uint8)  # a textured door
    mask = np.zeros((H, W), dtype=bool)
    mask[250:430, 300:500] = True
    return rgb, mask


def test_score_keyframe_measures_the_five_factors() -> None:
    rgb, mask = _panel_image()
    track = _track("pickup", Noun.CAR_DOOR, [(0, [300.0, 250.0, 500.0, 430.0], 0.8)])
    score, out_mask, subtracted = panels.score_keyframe(track, 0, mask=mask, car_mask=None, window_mask=None, frame_rgb=rgb, exclude_masks=[], th=TH)
    assert score.area == pytest.approx(180 * 200 / (W * H))
    assert score.completeness == 1.0 and score.glare == 0.0 and score.sharpness > 0
    assert score.frontality == pytest.approx(200 / 180)
    assert not subtracted and out_mask.sum() == mask.sum()
    # a window on the top half is subtracted and recorded
    window = np.zeros_like(mask)
    window[250:340, 300:500] = True
    score2, out2, subtracted2 = panels.score_keyframe(track, 0, mask=mask, car_mask=None, window_mask=window, frame_rgb=rgb, exclude_masks=[], th=TH)
    assert subtracted2 and out2.sum() == 90 * 200 and score2.area == pytest.approx(90 * 200 / (W * H))
    # glare: a bright white patch inside the mask
    glare_rgb = rgb.copy()
    glare_rgb[250:340, 300:500] = 255
    score3, _, _ = panels.score_keyframe(track, 0, mask=mask, car_mask=None, window_mask=None, frame_rgb=glare_rgb, exclude_masks=[], th=TH)
    assert score3.glare == pytest.approx(0.5)
    # a plate inside the panel is excluded from the sharpness measure
    plate = np.zeros_like(mask)
    plate[300:330, 350:450] = True
    score4, _, _ = panels.score_keyframe(track, 0, mask=mask, car_mask=None, window_mask=None, frame_rgb=rgb, exclude_masks=[plate], th=TH)
    assert score4.sharpness != score.sharpness
    # the car-mask containment test: a car mask covering only a corner zeroes the area
    car = np.zeros_like(mask)
    car[250:300, 300:350] = True
    score5, _, _ = panels.score_keyframe(track, 0, mask=mask, car_mask=car, window_mask=None, frame_rgb=rgb, exclude_masks=[], th=TH)
    assert score5.area == 0.0 and score5.total == 0.0
    # touching the frame edge lowers completeness
    edge_mask = np.zeros_like(mask)
    edge_mask[250:430, 0:200] = True
    score6, _, _ = panels.score_keyframe(track, 0, mask=edge_mask, car_mask=None, window_mask=None, frame_rgb=rgb, exclude_masks=[], th=TH)
    assert score6.completeness == pytest.approx(1.0 - panels.mask_border_frac(edge_mask)) and 0.0 < score6.completeness < 1.0
    # an empty mask scores zero everywhere
    score7, _, _ = panels.score_keyframe(track, 0, mask=np.zeros_like(mask), car_mask=None, window_mask=None, frame_rgb=rgb, exclude_masks=[], th=TH)
    assert score7.total == 0.0


def test_normalise_scores_divides_sharpness_and_frontality_by_the_candidates_max() -> None:
    raw = [
        (1, c.KeyframeScore(area=0.05, completeness=1.0, sharpness=400.0, glare=0.0, frontality=1.2, total=0.0)),
        (9, c.KeyframeScore(area=0.05, completeness=0.9, sharpness=200.0, glare=0.5, frontality=0.6, total=0.0)),
    ]
    normed = dict(panels.normalise_scores(raw, TH))
    assert normed[1].sharpness == 1.0 and normed[1].frontality == 1.0 and normed[1].total == pytest.approx(0.05)
    assert normed[9].sharpness == 0.5 and normed[9].frontality == 0.5 and normed[9].total == pytest.approx(0.05 * 0.9 * 0.5 * 0.5 * 0.5)
    assert panels.normalise_scores([], TH) == []


# ----------------------------------------------------------------------------------------------------------------
# 5.4 coverage and the top-up plan
# ----------------------------------------------------------------------------------------------------------------


def _lane_status(**overrides: LaneStatus) -> dict[Noun, LaneStatus]:
    out = {noun: LaneStatus.COMPLETED for noun in LANE_NOUNS}
    for key, status in overrides.items():
        out[Noun(key.replace("_", " "))] = status
    return out


def test_coverage_states_on_the_scripted_pair(pickup_tracks) -> None:
    ret_lanes, ret_ids, ret_pts = scripted_walk(reverse=True, coupe=True, roof=False, c_none=((Noun.WHEEL, "left_front"),))
    ret_lanes[Noun.FENDER] = []  # the fender lane failed at return
    ret_tracks = tracks_for_walk("return", ret_lanes, ret_pts)
    both = pickup_tracks + ret_tracks
    voted_panels, tracks = panels.vote_sides(both, None, TH, fps=FPS, direction_by_role={"pickup": -1, "return": 1})
    door_track = next(t for t in tracks if t.role == "pickup" and t.panel_id is PanelId.LEFT_FRONT_DOOR)
    kfs = [
        _keyframe("pickup", PanelId.LEFT_FRONT_DOOR, door_track.track_id, door_track.per_frame[10].frame, kf_id="pickup_left_front_door_kf1"),
        _keyframe("pickup", PanelId.LEFT_FRONT_DOOR, door_track.track_id, door_track.per_frame[30].frame, completeness=0.5, kf_id="pickup_left_front_door_kf2"),
        _keyframe("pickup", PanelId.HOOD, "pickup_hood_00", 3, area=0.01, kf_id="pickup_hood_kf1"),  # too small to count as filmed
    ]
    pickup = panels.coverage_for_role(voted_panels, tracks, kfs, role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    assert set(pickup) == set(PanelId)
    assert pickup[PanelId.LEFT_FRONT_DOOR].state is CoverageState.FILMED
    assert pickup[PanelId.LEFT_FRONT_DOOR].best_kf == "pickup_left_front_door_kf1"
    assert pickup[PanelId.LEFT_FRONT_DOOR].seconds == pytest.approx(len(door_track.per_frame) / FPS)
    assert pickup[PanelId.HOOD].state is CoverageState.GLIMPSED and pickup[PanelId.HOOD].best_kf == "pickup_hood_kf1"
    assert pickup[PanelId.HOOD].state_basis is StateBasis.C
    assert pickup[PanelId.ROOF].state is CoverageState.GLIMPSED
    assert pickup[PanelId.RIGHT_REAR_DOOR].state is CoverageState.GLIMPSED  # the pickup walk saw it
    assert pickup[PanelId.LEFT_REAR_DOOR].state is CoverageState.GLIMPSED and pickup[PanelId.LEFT_REAR_DOOR].seconds > 2.0

    status = _lane_status(roof=LaneStatus.ZERO_MATCHES, fender=LaneStatus.FAILED)
    ret = panels.coverage_for_role(voted_panels, tracks, kfs, role="return", fps=FPS, lane_status=status, th=TH)
    assert ret[PanelId.ROOF].state is CoverageState.NOT_FILMED and ret[PanelId.ROOF].seconds == 0.0  # a zero-match lane
    for pid in (PanelId.LEFT_FRONT_FENDER, PanelId.RIGHT_REAR_FENDER):
        assert ret[pid].state is CoverageState.NOT_RUN and "failed" in ret[pid].reason  # never not_filmed
    assert ret[PanelId.RIGHT_REAR_DOOR].state is CoverageState.NOT_FILMED  # seen at pickup, missing from this walk
    assert "no track for right_rear_door in this walk" in ret[PanelId.RIGHT_REAR_DOOR].reason
    assert ret[PanelId.LEFT_FRONT_WHEEL].state is CoverageState.GLIMPSED
    assert ret[PanelId.LEFT_FRONT_WHEEL].state_basis is StateBasis.AREA_ONLY  # c absent on every frame
    assert "area-only" in ret[PanelId.LEFT_FRONT_WHEEL].reason
    assert ret[PanelId.LEFT_REAR_WHEEL].state_basis is StateBasis.C
    assert ret[PanelId.LEFT_FRONT_DOOR].state is CoverageState.GLIMPSED and ret[PanelId.LEFT_FRONT_DOOR].best_kf is None
    assert all(v.state is not CoverageState.NOT_SEEN for v in ret.values())


def test_coverage_not_seen_is_a_slot_with_no_track_in_either_walk() -> None:
    lanes, _ids, pts = scripted_walk(coupe=True)
    tracks = tracks_for_walk("pickup", lanes, pts)
    voted_panels, tracks = panels.vote_sides(tracks, -1, TH, fps=FPS)
    cov = panels.coverage_for_role(voted_panels, tracks, [], role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    assert cov[PanelId.RIGHT_REAR_DOOR].state is CoverageState.NOT_SEEN
    assert cov[PanelId.RIGHT_FRONT_DOOR].state is CoverageState.GLIMPSED
    assert "not seen" not in cov[PanelId.RIGHT_FRONT_DOOR].reason.lower()
    # the same slot with the door lane never run is not_run, never not_seen or not_filmed
    down = panels.coverage_for_role(voted_panels, [t for t in tracks if t.noun is not Noun.CAR_DOOR], [], role="pickup", fps=FPS, lane_status=_lane_status(car_door=LaneStatus.INCOMPLETE), th=TH)
    assert down[PanelId.RIGHT_REAR_DOOR].state is CoverageState.NOT_RUN
    missing = panels.coverage_for_role(voted_panels, [t for t in tracks if t.noun is not Noun.CAR_DOOR], [], role="pickup", fps=FPS, lane_status={}, th=TH)
    assert missing[PanelId.LEFT_FRONT_DOOR].state is CoverageState.NOT_RUN


def test_coverage_with_a_failed_car_lane_is_not_run_everywhere(pickup_tracks) -> None:
    voted_panels, tracks = panels.vote_sides(pickup_tracks, -1, TH, fps=FPS)
    cov = panels.coverage_for_role(voted_panels, tracks, [], role="pickup", fps=FPS, lane_status=_lane_status(car=LaneStatus.FAILED), th=TH)
    assert {v.state for v in cov.values()} == {CoverageState.NOT_RUN}
    assert all("car lane failed" in v.reason for v in cov.values())


def test_glimpsed_compares_the_median_c_with_the_nouns_25th_percentile() -> None:
    doors = [_moving("pickup", Noun.CAR_DOOR, range(k * 100, k * 100 + 40), 700.0, -6.0, conf=conf) for k, conf in enumerate((0.5, 0.6, 0.7, 0.8))]
    for k, t in enumerate(doors):
        t.panel_id = [PanelId.LEFT_FRONT_DOOR, PanelId.LEFT_REAR_DOOR, PanelId.RIGHT_FRONT_DOOR, PanelId.RIGHT_REAR_DOOR][k]
    voted_panels, _ = panels.vote_sides([], None, TH)
    cov = panels.coverage_for_role(voted_panels, doors, [], role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    assert cov[PanelId.LEFT_FRONT_DOOR].state is CoverageState.NOT_FILMED  # 0.5 sits under the 25th percentile (0.575)
    assert cov[PanelId.LEFT_FRONT_DOOR].state_basis is StateBasis.C
    for pid in (PanelId.LEFT_REAR_DOOR, PanelId.RIGHT_FRONT_DOOR, PanelId.RIGHT_REAR_DOOR):
        assert cov[pid].state is CoverageState.GLIMPSED
    # one track with a constant c (the text-lane case of 10/26) is at its own percentile, so it is glimpsed
    single = panels.coverage_for_role(voted_panels, doors[:1], [], role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    assert single[PanelId.LEFT_FRONT_DOOR].state is CoverageState.GLIMPSED
    # under the area floor with c absent the area-only basis says not_filmed
    tiny = _moving("pickup", Noun.CAR_DOOR, range(0, 40), 700.0, -6.0, w=60, h=60, conf=None)
    tiny.panel_id = PanelId.LEFT_FRONT_DOOR
    low = panels.coverage_for_role(voted_panels, [tiny], [], role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    assert low[PanelId.LEFT_FRONT_DOOR].state is CoverageState.NOT_FILMED and low[PanelId.LEFT_FRONT_DOOR].state_basis is StateBasis.AREA_ONLY


def test_plan_topup_orders_wheels_then_lower_bumpers_then_the_rest() -> None:
    states = {pid: c.PanelCoverage(state=CoverageState.FILMED) for pid in PanelId}
    states[PanelId.HOOD] = c.PanelCoverage(state=CoverageState.NOT_FILMED)
    states[PanelId.REAR_BUMPER] = c.PanelCoverage(state=CoverageState.GLIMPSED, seconds=0.8)
    states[PanelId.LEFT_REAR_WHEEL] = c.PanelCoverage(state=CoverageState.NOT_RUN)
    states[PanelId.RIGHT_FRONT_WHEEL] = c.PanelCoverage(state=CoverageState.NOT_FILMED)
    states[PanelId.TAIL_LIGHTS] = c.PanelCoverage(state=CoverageState.NOT_SEEN)
    cov = panels.plan_topup(c.Coverage(per_role={"pickup": states}), TH)
    assert cov.missing == [PanelId.LEFT_REAR_WHEEL, PanelId.RIGHT_FRONT_WHEEL, PanelId.REAR_BUMPER, PanelId.HOOD, PanelId.TAIL_LIGHTS]
    assert cov.nouns_to_rerun == [Noun.WHEEL, Noun.BUMPER, Noun.HOOD, Noun.TAIL_LIGHT, Noun.CAR, Noun.LICENSE_PLATE, Noun.FACE]
    assert "rear bumper (glimpsed, 0.8 s)" in cov.topup_text and "7 of 16 nouns" in cov.topup_text
    assert cov.odometer.asked and "odometer" in cov.topup_text
    again = panels.plan_topup(cov, TH)
    assert again.odometer.asked and again.missing == cov.missing
    full = panels.plan_topup(c.Coverage(per_role={"pickup": {pid: c.PanelCoverage(state=CoverageState.FILMED) for pid in PanelId}}), TH)
    assert full.missing == [] and full.nouns_to_rerun == [Noun.CAR, Noun.LICENSE_PLATE, Noun.FACE]
    empty = panels.plan_topup(c.Coverage(), TH)
    assert empty.missing == [] and empty.nouns_to_rerun == [Noun.CAR, Noun.LICENSE_PLATE, Noun.FACE]
    # the latest role decides: a top-up that filmed the hood clears it
    cov.per_role["topup_1"] = {pid: c.PanelCoverage(state=CoverageState.FILMED) for pid in PanelId}
    assert panels.plan_topup(cov, TH).missing == []


def test_panel_timeline_prints_windows_and_merged_fragment_counts(pickup_walk, voted) -> None:
    voted_panels, tracks, _ = voted
    rows = panels.panel_timeline(voted_panels, tracks, role="pickup")
    by_pid = {r["panel_id"]: r for r in rows}
    rear = by_pid[PanelId.LEFT_REAR_DOOR]
    assert rear["fragments"] == 2 and rear["tracks"] == 1 and len(rear["windows"]) == 1  # the 0.5 s gap is bridged
    assert rear["line"].startswith("left rear door: 0:") and rear["line"].endswith(", same door, fragments 2 to 1")
    front = by_pid[PanelId.LEFT_FRONT_DOOR]
    assert front["fragments"] == 1 and "fragments" not in front["line"]
    assert front["first_s"] < front["last_s"] and front["seconds_visible"] > 0
    heads = by_pid[PanelId.HEADLIGHTS]
    assert heads["tracks"] == 3 and len(heads["windows"]) == 2  # seen at the start and again at the end
    assert panels.panel_timeline(voted_panels, tracks, role="return") == []
    assert panels.timeline_line(voted_panels[0], "return", tracks) is None
    assert panels._mmss(65.4) == "1:05"


# ----------------------------------------------------------------------------------------------------------------
# The 10/26 fixtures through sam_wire and derive()
# ----------------------------------------------------------------------------------------------------------------


def _fixture_lanes() -> dict[str, tuple[c.LaneRecord, list[c.IndexFrame]]]:
    car_frames, car_out = sam_wire.index_from_events(sam_wire.read_events(BOX_FIXTURE), media="video")
    door_frames, door_out = sam_wire.index_from_events(sam_wire.read_events(TEXT_FIXTURE), media="video")
    assert car_out.status is LaneStatus.COMPLETED and door_out.status is LaneStatus.COMPLETED

    def record(noun: Noun, status: LaneStatus) -> c.LaneRecord:
        return c.LaneRecord(lane_id=c.noun_slug(noun), role="pickup", noun=noun, prompt_text=noun.value, media_size=[1280, 720], status=status, frames_sent=141)

    return {
        "car": (record(Noun.CAR, LaneStatus.COMPLETED), car_frames),
        "car_door": (record(Noun.CAR_DOOR, LaneStatus.COMPLETED), door_frames),
        "fender": (record(Noun.FENDER, LaneStatus.ZERO_MATCHES), []),
        "wheel": (record(Noun.WHEEL, LaneStatus.FAILED), []),
    }


@pytest.fixture(scope="module")
def fixture_th() -> dict[str, Any]:
    th = copy.deepcopy(TH)
    th["panels"]["area_floor_default"] = 0.0001  # the ping pong ball is 0.3% of the frame
    return th


def test_fixture_replay_builds_one_track_with_a_constant_c(fixture_th) -> None:
    lanes = _fixture_lanes()
    car_boxes = panels.car_boxes_by_frame(lanes["car"][1])
    assert len(car_boxes) == 134 and 80 not in car_boxes
    kept = panels.filter_objects(lanes["car_door"][1], Noun.CAR_DOOR, car_boxes, (1280, 720), fixture_th)
    assert len(kept) == 134
    assert panels.filter_objects(lanes["car_door"][1], Noun.CAR_DOOR, car_boxes, (1280, 720), TH) == []  # under the 2% floor
    frags = panels.build_fragments(kept, fps=24.0, th=fixture_th)
    assert [(f.object_id, f.first_frame, f.last_frame) for f in frags] == [("0", 0, 136)]  # one-frame holes never split
    groups = panels.merge_fragments(frags, kept, fps=24.0, th=fixture_th)
    pts = [i / 24.0 for i in range(141)]
    tracks = panels.build_tracks("pickup", Noun.CAR_DOOR, groups, kept, pts, car_boxes, (1280, 720), th=fixture_th)
    assert len(tracks) == 1
    t = tracks[0]
    assert t.track_id == "pickup_car_door_00" and len(t.per_frame) == 134 and t.c_present_share == 1.0
    assert {r.c for r in t.per_frame} == {0.7299578189849854}  # the text-prompt constant of 10/26
    assert all(r.in_car_box and not r.mask_decoded for r in t.per_frame)
    touching = [r.frame for r in t.per_frame if r.border_frac > 0]
    assert touching == [108]  # the ball touches the bottom edge once (box y2 = 720)
    assert next(r for r in t.per_frame if r.frame == 108).border_frac > TH["panels"]["border_frac_keyframe_max"]
    assert panels.walk_direction(tracks, fixture_th) is None  # median 1.0 px per frame, under the 2 px floor
    slow = copy.deepcopy(fixture_th)
    slow["panels"]["direction_min_px_per_frame"] = 0.5
    assert panels.walk_direction(tracks, slow) == 1  # the ball drifts image-right (607 to 1020 px)
    cands = panels.keyframe_candidates(t, fixture_th)
    assert len(cands) == 9 and len(set(cands)) == 9 and 108 not in cands
    assert panels.lane_status_from_lanes(lanes) == {
        Noun.CAR: LaneStatus.COMPLETED,
        Noun.CAR_DOOR: LaneStatus.COMPLETED,
        Noun.FENDER: LaneStatus.ZERO_MATCHES,
        Noun.WHEEL: LaneStatus.FAILED,
    }


async def test_derive_on_the_fixture_pair_writes_the_four_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_th) -> None:
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    run_id, role = "01TESTPANELS", "pickup"
    shutil.copy(CLIP, paths.quarantine_original(run_id, role))
    shutil.copy(BOX_FIXTURE, paths.lane_events_path(run_id, role, Noun.CAR))
    shutil.copy(TEXT_FIXTURE, paths.lane_events_path(run_id, role, Noun.CAR_DOOR))
    lanes = _fixture_lanes()
    pts = video.pts_map(CLIP)
    assert len(pts) == 141
    blurred_calls: list[int] = []

    async def fake_blur(run_id_: str, role_: str, rgb: np.ndarray, *, f_proxy: int, tracks: list[c.PanelTrack], fps: float, sizes, keep_plate: bool, th) -> tuple[np.ndarray, c.BlurRecord]:
        assert run_id_ == run_id and role_ == role and rgb.shape == (720, 1280, 3) and not keep_plate
        assert sizes == ((1280, 720), (1280, 720)) and fps == 24.0
        blurred_calls.append(f_proxy)
        return rgb, c.BlurRecord(request_ids=[f"img_{f_proxy}"])

    with sam_wire.DecodePool(1) as pool:
        tracks, voted_panels, keyframes, coverage = await panels.derive(
            run_id, role, lanes=lanes, pts=pts, media_size=(1280, 720), fps=24.0, decode=pool, blur_keyframe=fake_blur, th=fixture_th
        )
        decoded, cpu_s = pool.stats()
    assert decoded >= 9 and cpu_s >= 0.0
    door = next(t for t in tracks if t.noun is Noun.CAR_DOOR)
    assert door.panel_id is PanelId.LEFT_FRONT_DOOR  # a lone door with no mirror, anchor or wheel: side uncertain
    assert next(p for p in voted_panels if p.panel_id is PanelId.LEFT_FRONT_DOOR).side_uncertain
    assert sum(1 for r in door.per_frame if r.mask_decoded) == 9
    assert all(r.area_frac is not None and r.complete is not None for r in door.per_frame if r.mask_decoded)
    kfs = [k for k in keyframes if k.panel_id is PanelId.LEFT_FRONT_DOOR]
    assert 1 <= len(kfs) <= 3 and len(kfs) == len(blurred_calls)
    for i, a in enumerate(kfs):
        for b in kfs[i + 1 :]:
            assert abs(a.f_proxy - b.f_proxy) >= 12  # 0.5 s at 24 fps
    assert [k.kf_id for k in kfs] == [f"pickup_left_front_door_kf{n + 1}" for n in range(len(kfs))]
    best = kfs[0]
    assert best.score.total > 0 and best.score.completeness == 1.0 and best.score.sharpness == 1.0
    assert best.crop is not None and best.crop.sha256 and (paths.run_dir(run_id) / best.crop.path).is_file()
    assert 0 < best.crop.w <= 1280 and 0 < best.crop.h <= 1280 and best.crop.scale > 0
    assert video.probe(paths.run_dir(run_id) / best.crop.path).width == best.crop.w
    assert best.mask_png and (paths.run_dir(run_id) / best.mask_png).is_file()
    assert best.moments is not None and best.moments.mu20 > 0
    assert best.front_sign is None and best.blur.request_ids == [f"img_{best.f_proxy}"]
    assert best.window_subtracted is False
    assert paths.keyframe_frame_path(run_id, role, best.kf_id).is_file()
    assert coverage[PanelId.LEFT_FRONT_DOOR].state is CoverageState.GLIMPSED  # the ball is far under 3% of the frame
    assert coverage[PanelId.LEFT_FRONT_DOOR].state_basis is StateBasis.C and coverage[PanelId.LEFT_FRONT_DOOR].best_kf == best.kf_id
    assert coverage[PanelId.LEFT_FRONT_DOOR].seconds == pytest.approx(134 / 24.0)
    for pid in (PanelId.LEFT_FRONT_FENDER, PanelId.RIGHT_REAR_FENDER):
        assert coverage[pid].state is CoverageState.NOT_FILMED  # a zero-match lane
    assert coverage[PanelId.LEFT_FRONT_WHEEL].state is CoverageState.NOT_RUN  # a failed lane
    assert coverage[PanelId.HOOD].state is CoverageState.NOT_RUN  # a lane that never ran
    assert coverage[PanelId.RIGHT_FRONT_DOOR].state is CoverageState.NOT_SEEN  # the door noun was seen, this slot never
    # the four files parse back through the contracts and rebuild byte for byte
    read_tracks = c.read_json_list(paths.tracks_json(run_id), c.PanelTrack)
    read_panels = c.read_json_list(paths.panels_json(run_id), c.Panel)
    read_kfs = c.read_json_list(paths.keyframes_json(run_id), c.Keyframe)
    read_cov = c.read_json(paths.coverage_json(run_id), c.Coverage)
    assert c.canonical_json(read_tracks) == c.canonical_json(tracks)
    assert c.canonical_json(read_panels) == c.canonical_json(voted_panels)
    assert c.canonical_json(read_kfs) == c.canonical_json(keyframes)
    assert read_cov.per_role["pickup"][PanelId.LEFT_FRONT_DOOR].state is CoverageState.GLIMPSED
    assert read_cov.missing[0] in c.WHEEL_PANELS and PanelId.LEFT_FRONT_DOOR in read_cov.missing
    assert set(read_cov.nouns_to_rerun) >= {Noun.CAR, Noun.LICENSE_PLATE, Noun.FACE}
    door_panel = next(p for p in read_panels if p.panel_id is PanelId.LEFT_FRONT_DOOR)
    assert door_panel.per_role["pickup"].keyframe_ids == [k.kf_id for k in kfs]
    assert door_panel.per_role["pickup"].state is CoverageState.GLIMPSED
    # a second derive of the same role rebuilds the same tracks and panels
    with sam_wire.DecodePool(1) as pool:
        tracks2, panels2, _k2, _c2 = await panels.derive(
            run_id, role, lanes=lanes, pts=pts, media_size=(1280, 720), fps=24.0, decode=pool, blur_keyframe=fake_blur, th=fixture_th
        )
    assert c.canonical_json(tracks2) == c.canonical_json(tracks) and c.canonical_json(panels2) == c.canonical_json(voted_panels)
