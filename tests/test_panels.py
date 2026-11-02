"""Offline tests for walkaround.panels (M5): a scripted walk around a rectangle car, the merge rules one by one, the
side vote in both walk directions, the keyframe passes, the coverage states, the top-up plan, the timeline, and the
10/26 ping pong fixtures replayed through sam_wire into derive() with a real DecodePool. No network."""

from __future__ import annotations

import asyncio
import copy
import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from meta_sam_parser import build_object_prompt

from walkaround import contracts as c
from walkaround import fastmask, panels, paths, privacy, sam_wire, video
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
    """Every lane of one walk to tracks, through the same build_role_tracks derive() calls."""
    tracks, _kept = panels.build_role_tracks(role, lanes, pts=pts, media_size=(W, H), fps=FPS, th=th)
    return tracks


def _filed(track: c.PanelTrack, pid: PanelId) -> c.PanelTrack:
    """A hand-made track filed into a slot with the side and position a decided vote gives it."""
    side, position = panels._panel_side_position(pid)
    track.panel_id, track.side, track.position = pid, side, position
    return track


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


def test_short_plate_and_face_runs_reach_tracks_and_the_blur_boxes() -> None:
    frames_by_noun: dict[Noun, list[c.IndexFrame]] = {noun: [] for noun in LANE_NOUNS}
    for f in range(60):
        frames_by_noun[Noun.CAR].append(c.IndexFrame(frame=f, objects=[c.IndexObject(id="0", box=[100, 100, 900, 500], c=0.9)]))
        faces = []
        if 20 <= f < 27:  # a passer-by outside the car box for 7 frames, under the 8-frame fragment minimum
            faces.append(c.IndexObject(id="3", box=[10, 10, 60, 70], c=0.8))
        if 40 <= f < 43:  # three frames
            faces.append(c.IndexObject(id="4", box=[700, 20, 740, 70], c=0.8))
        if faces:
            frames_by_noun[Noun.FACE].append(c.IndexFrame(frame=f, objects=faces))
        plate_id = "1" if 30 <= f < 50 else "2" if 50 <= f < 55 else None  # a 5-frame id after an id switch
        if plate_id:
            frames_by_noun[Noun.LICENSE_PLATE].append(c.IndexFrame(frame=f, objects=[c.IndexObject(id=plate_id, box=[400, 400, 480, 430], c=0.8)]))
    pts = [i / FPS for i in range(60)]
    tracks, kept = panels.build_role_tracks("pickup", frames_by_noun, pts=pts, media_size=(W, H), fps=FPS, th=TH)
    faces = [t for t in tracks if t.noun is Noun.FACE]
    plates = [t for t in tracks if t.noun is Noun.LICENSE_PLATE]
    assert [(t.sam_object_ids, [r.frame for r in t.per_frame]) for t in faces] == [(["3"], list(range(20, 27))), (["4"], [40, 41, 42])]
    assert [(t.sam_object_ids, len(t.per_frame)) for t in plates] == [(["1"], 20), (["2"], 5)]
    assert not any(t.merges for t in faces + plates) and kept[Noun.FACE] == frames_by_noun[Noun.FACE]
    for f in (20, 23, 26, 40, 42, 52, 54):
        assert privacy.track_boxes_at(f, tracks, fps=FPS, media_size_proxy=(W, H), media_size_src=(W, H), th=TH), f
    # fragments still record where an id dropped out for longer than the 5.1 gap
    gap = {Noun.FACE: [c.IndexFrame(frame=f, objects=[c.IndexObject(id="5", box=[10, 10, 60, 70])]) for f in (1, 2, 30)]}
    split = panels.build_role_tracks("pickup", gap, pts=[i / FPS for i in range(40)], media_size=(W, H), fps=FPS, th=TH)[0]
    assert [(fr.first_frame, fr.last_frame) for fr in split[0].fragments] == [(1, 2), (30, 30)]


def test_a_duplicate_sam_id_on_one_door_is_suppressed() -> None:
    lanes, ids, pts = scripted_walk()
    front = ids[(Noun.CAR_DOOR, "left_front")]
    n = 0
    for fr in lanes[Noun.CAR_DOOR]:
        o = next((o for o in fr.objects if o.id == front), None)
        if o is not None and o.box[0] > 200 and n < 20:  # a second id on the same door, box IoU about 0.8
            x1, y1, x2, y2 = o.box
            fr.objects.append(c.IndexObject(id="99", box=[x1 + 12, y1 + 6, x2 + 12, y2 + 6], c=0.8))
            n += 1
    assert n == 20
    tracks = tracks_for_walk("pickup", lanes, pts)
    doors = [t for t in tracks if t.noun is Noun.CAR_DOOR]
    assert len(doors) == 4 and not any("99" in t.sam_object_ids for t in doors)
    _, voted_tracks = panels.vote_sides(tracks, -1, TH, fps=FPS)
    rear = [t.sam_object_ids for t in voted_tracks if t.panel_id is PanelId.LEFT_REAR_DOOR]
    assert rear == [[ids[(Noun.CAR_DOOR, "left_rear_a")], ids[(Noun.CAR_DOOR, "left_rear_b")]]]
    touching = [c.IndexFrame(frame=0, objects=[c.IndexObject(id="1", box=[0, 0, 100, 100]), c.IndexObject(id="2", box=[100, 0, 200, 100])])]
    assert len(panels.suppress_duplicates(touching, TH)[0].objects) == 2


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
    assert tracks[0].panel_id is PanelId.LEFT_FRONT_DOOR  # a placeholder: left and front first
    assert tracks[0].side is Side.NONE and tracks[0].position is Position.NONE and panels.undecided(tracks[0])
    doors = [PanelId.LEFT_FRONT_DOOR, PanelId.LEFT_REAR_DOOR, PanelId.RIGHT_FRONT_DOOR, PanelId.RIGHT_REAR_DOOR]
    assert panels.candidate_slots(tracks[0]) == doors
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
    assert tracks[0].panel_id is PanelId.FRONT_BUMPER and tracks[0].position is Position.NONE
    assert panels.candidate_slots(tracks[0]) == [PanelId.FRONT_BUMPER, PanelId.REAR_BUMPER]
    placeholder = next(p for p in alone if p.panel_id is PanelId.FRONT_BUMPER)
    assert placeholder.side_uncertain and placeholder.front_sign is None


def test_a_tie_in_one_walk_prints_every_candidate_slot_as_side_uncertain(pickup_tracks) -> None:
    door = _moving("return", Noun.CAR_DOOR, range(0, 60), 300.0, 0.5)  # 4 s of the right front door, no mirror, no drift
    door.track_id = "return_car_door_00"
    dirs = {"pickup": -1, "return": panels.walk_direction([door], TH)}
    assert dirs["return"] is None
    voted_panels, tracks = panels.vote_sides(pickup_tracks + [door], None, TH, fps=FPS, direction_by_role=dirs)
    rt = next(t for t in tracks if t.role == "return")
    assert rt.panel_id is PanelId.LEFT_FRONT_DOOR and rt.side is Side.NONE and rt.position is Position.NONE
    doors = [PanelId.LEFT_FRONT_DOOR, PanelId.LEFT_REAR_DOOR, PanelId.RIGHT_FRONT_DOOR, PanelId.RIGHT_REAR_DOOR]
    ret = panels.coverage_for_role(voted_panels, tracks, [], role="return", fps=FPS, lane_status=_lane_status(), th=TH)
    for pid in doors:
        assert ret[pid].state is CoverageState.GLIMPSED, pid  # never not_filmed while the undecided door may be it
        assert "side uncertain" in ret[pid].reason and "return_car_door_00" in ret[pid].reason
        assert ret[pid].seconds == pytest.approx(60 / FPS)
    assert "frames filed under left front door" in ret[PanelId.RIGHT_FRONT_DOOR].reason
    pick = panels.coverage_for_role(voted_panels, tracks, [], role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    assert pick[PanelId.LEFT_FRONT_DOOR].state is CoverageState.GLIMPSED
    assert "side uncertain" not in pick[PanelId.LEFT_FRONT_DOOR].reason  # the pickup's own door is decided
    pickup_door = next(t for t in tracks if t.role == "pickup" and t.panel_id is PanelId.LEFT_FRONT_DOOR)
    assert pickup_door.side is Side.LEFT and pickup_door.position is Position.FRONT


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
    # with car boxes the anchor rows outside the car say so; without media_size the call refuses to guess a frame
    car_boxes = {f + offset: [0, 0, 10, 10] for f in range(0, 400)}
    outside = panels.absorb_anchor([a, b], anchor_frames, frame_offset=offset, th=TH, media_size=(W, H), pts=pts, fps=FPS, car_boxes=car_boxes)
    gap_rows = [r for r in outside[0].per_frame if a.per_frame[-1].frame < r.frame < b.per_frame[0].frame]
    assert gap_rows and not any(r.in_car_box for r in gap_rows)
    with pytest.raises(TypeError):
        panels.absorb_anchor([a, b], anchor_frames, frame_offset=offset, th=TH)  # type: ignore[call-arg]


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


def test_car_lane_states_reach_only_the_slots_without_a_track(pickup_tracks) -> None:
    empty_panels, _ = panels.vote_sides([], None, TH)
    for car in (LaneStatus.FAILED, LaneStatus.INCOMPLETE, None):
        status = _lane_status()
        if car is None:
            del status[Noun.CAR]  # no car summary, or one that could not be read
        else:
            status[Noun.CAR] = car
        cov = panels.coverage_for_role(empty_panels, [], [], role="pickup", fps=FPS, lane_status=status, th=TH)
        assert {v.state for v in cov.values()} == {CoverageState.NOT_RUN}, car
        assert all(v.reason.startswith(f"car lane {car.value if car else 'not run'}") for v in cov.values())
    # an incomplete car lane still let these tracks through on the frames it returned: they keep their states
    voted_panels, tracks = panels.vote_sides(pickup_tracks, -1, TH, fps=FPS)
    door = next(t for t in tracks if t.panel_id is PanelId.LEFT_FRONT_DOOR)
    kf = _keyframe("pickup", PanelId.LEFT_FRONT_DOOR, door.track_id, door.per_frame[10].frame, kf_id="kf1")
    no_wheels = [t for t in tracks if t.noun is not Noun.WHEEL]
    cov = panels.coverage_for_role(voted_panels, no_wheels, [kf], role="pickup", fps=FPS, lane_status=_lane_status(car=LaneStatus.INCOMPLETE), th=TH)
    assert cov[PanelId.LEFT_FRONT_DOOR].state is CoverageState.FILMED and cov[PanelId.LEFT_FRONT_DOOR].best_kf == "kf1"
    assert cov[PanelId.HOOD].state is CoverageState.GLIMPSED
    assert cov[PanelId.LEFT_FRONT_WHEEL].state is CoverageState.NOT_RUN and "car lane incomplete" in cov[PanelId.LEFT_FRONT_WHEEL].reason


def test_a_tracked_slot_is_glimpsed_whatever_its_detector_score() -> None:
    slots = [PanelId.LEFT_FRONT_DOOR, PanelId.LEFT_REAR_DOOR, PanelId.RIGHT_FRONT_DOOR, PanelId.RIGHT_REAR_DOOR]
    doors = [
        _filed(_moving("pickup", Noun.CAR_DOOR, range(k * 100, k * 100 + 40), 700.0, -6.0, conf=conf), slots[k])
        for k, conf in enumerate((0.5, 0.6, 0.7, 0.8))
    ]
    voted_panels, _ = panels.vote_sides([], None, TH)
    cov = panels.coverage_for_role(voted_panels, doors, [], role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    for pid in slots:
        assert cov[pid].state is CoverageState.GLIMPSED and cov[pid].state_basis is StateBasis.C  # 5.4: not_filmed means no track
        assert "percentile" not in cov[pid].reason and "median c" not in cov[pid].reason  # c is relative, never printed
    assert "under this run's lower quartile for car door" in cov[PanelId.LEFT_FRONT_DOOR].reason  # 0.5 sits under 0.575
    assert "at or above this run's lower quartile" in cov[PanelId.RIGHT_REAR_DOOR].reason
    # one track with a constant c (the text-lane case of 10/26) is at its own percentile
    single = panels.coverage_for_role(voted_panels, doors[:1], [], role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    assert single[PanelId.LEFT_FRONT_DOOR].state is CoverageState.GLIMPSED and "at or above" in single[PanelId.LEFT_FRONT_DOOR].reason
    # c absent on every frame: the area-only basis, and a box under its floor still leaves the slot glimpsed
    tiny = _filed(_moving("pickup", Noun.CAR_DOOR, range(0, 40), 700.0, -6.0, w=60, h=60, conf=None), PanelId.LEFT_FRONT_DOOR)
    low = panels.coverage_for_role(voted_panels, [tiny], [], role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    assert low[PanelId.LEFT_FRONT_DOOR].state is CoverageState.GLIMPSED and low[PanelId.LEFT_FRONT_DOOR].state_basis is StateBasis.AREA_ONLY
    assert "area-only basis: box area stays under 1% of the frame" in low[PanelId.LEFT_FRONT_DOOR].reason


def test_plan_topup_orders_wheels_then_lower_bumpers_then_the_rest() -> None:
    states = {pid: c.PanelCoverage(state=CoverageState.FILMED) for pid in PanelId}
    states[PanelId.HOOD] = c.PanelCoverage(state=CoverageState.NOT_FILMED)
    states[PanelId.REAR_BUMPER] = c.PanelCoverage(state=CoverageState.GLIMPSED, seconds=0.8)
    states[PanelId.LEFT_REAR_WHEEL] = c.PanelCoverage(state=CoverageState.NOT_RUN)
    states[PanelId.RIGHT_FRONT_WHEEL] = c.PanelCoverage(state=CoverageState.NOT_FILMED)
    states[PanelId.TAIL_LIGHTS] = c.PanelCoverage(state=CoverageState.NOT_SEEN)
    cov = panels.plan_topup(c.Coverage(per_role={"pickup": states}), TH)
    assert cov.missing == [PanelId.LEFT_REAR_WHEEL, PanelId.RIGHT_FRONT_WHEEL, PanelId.REAR_BUMPER, PanelId.HOOD, PanelId.TAIL_LIGHTS]
    # a missing wheel brings the side anchors along so the top-up track can be voted a side
    assert cov.nouns_to_rerun == [
        Noun.WHEEL, Noun.BUMPER, Noun.HOOD, Noun.TAIL_LIGHT, Noun.HEADLIGHT, Noun.SIDE_MIRROR, Noun.CAR, Noun.LICENSE_PLATE, Noun.FACE,
    ]
    assert "rear bumper (glimpsed, 0.8 s)" in cov.topup_text and "9 of 16 nouns" in cov.topup_text
    assert cov.odometer.asked and "odometer" in cov.topup_text
    again = panels.plan_topup(cov, TH)
    assert again.odometer.asked and again.missing == cov.missing
    full = panels.plan_topup(c.Coverage(per_role={"pickup": {pid: c.PanelCoverage(state=CoverageState.FILMED) for pid in PanelId}}), TH)
    assert full.missing == [] and full.nouns_to_rerun == [Noun.CAR, Noun.LICENSE_PLATE, Noun.FACE]
    empty = panels.plan_topup(c.Coverage(), TH)
    assert empty.missing == [] and empty.nouns_to_rerun == [Noun.CAR, Noun.LICENSE_PLATE, Noun.FACE]
    # a top-up that filmed everything clears the list
    cov.per_role["topup_1"] = {pid: c.PanelCoverage(state=CoverageState.FILMED) for pid in PanelId}
    assert panels.plan_topup(cov, TH).missing == []


def test_plan_topup_reads_the_walk_and_every_topup_in_number_order() -> None:
    pickup = {pid: c.PanelCoverage(state=CoverageState.FILMED) for pid in PanelId}
    pickup[PanelId.RIGHT_REAR_WHEEL] = c.PanelCoverage(state=CoverageState.GLIMPSED, seconds=0.8)
    pickup[PanelId.HOOD] = c.PanelCoverage(state=CoverageState.NOT_FILMED)
    # a top-up re-runs only some nouns, so every other slot is not_run there and must not come back as missing
    t2 = {pid: c.PanelCoverage(state=CoverageState.NOT_RUN) for pid in PanelId}
    t2[PanelId.RIGHT_REAR_WHEEL] = c.PanelCoverage(state=CoverageState.GLIMPSED, seconds=1.5)
    t10 = {pid: c.PanelCoverage(state=CoverageState.NOT_RUN) for pid in PanelId}
    t10[PanelId.HOOD] = c.PanelCoverage(state=CoverageState.FILMED)
    cov = panels.plan_topup(c.Coverage(per_role={"pickup": pickup, "topup_2": t2, "topup_10": t10}), TH)
    assert cov.missing == [PanelId.RIGHT_REAR_WHEEL]
    assert "right rear wheel (glimpsed, 1.5 s)" in cov.topup_text  # the latest role whose wheel lane ran
    assert panels.coverage_roles(["topup_10", "pickup", "topup_2", "claim_1"]) == ["pickup", "topup_2", "topup_10"]
    assert panels.coverage_roles(["pickup", "return", "topup_1"]) == ["return", "topup_1"]
    roles = ["claim_1", "topup_10", "return", "topup_2", "pickup"]
    assert sorted(roles, key=panels._role_order) == ["pickup", "return", "topup_2", "topup_10", "claim_1"]


def test_plan_topup_is_idempotent_and_a_rebuild_matches_what_derive_wrote() -> None:
    states = {pid: c.PanelCoverage(state=CoverageState.FILMED) for pid in PanelId}
    states[PanelId.HOOD] = c.PanelCoverage(state=CoverageState.NOT_FILMED)
    written = panels.plan_topup(c.Coverage(per_role={"pickup": states}), TH)  # what derive writes on a fresh run
    assert written.topup_text.endswith("Also film the odometer once.") and written.odometer.asked
    assert c.json_bytes(panels.plan_topup(written, TH)) == c.json_bytes(written)
    for asked in (True, False):  # the rebuild may hand back either flag; the text depends only on filmed
        rebuilt = panels.plan_topup(c.Coverage(per_role={"pickup": states}, odometer=c.Odometer(asked=asked), rerun_seconds=written.rerun_seconds), TH)
        assert c.json_bytes(rebuilt) == c.json_bytes(written)
    done = panels.plan_topup(c.Coverage(per_role={"pickup": states}, odometer=c.Odometer(asked=True, filmed=True)), TH)
    assert "odometer" not in done.topup_text


def test_plan_topup_keeps_a_coupes_absent_rear_door_off_the_missing_list() -> None:
    lanes, _ids, pts = scripted_walk(coupe=True)
    voted_panels, tracks = panels.vote_sides(tracks_for_walk("pickup", lanes, pts), -1, TH, fps=FPS)
    cov = panels.coverage_for_role(voted_panels, tracks, [], role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    assert cov[PanelId.RIGHT_REAR_DOOR].state is CoverageState.NOT_SEEN
    plan = panels.plan_topup(c.Coverage(per_role={"pickup": cov}), TH)
    assert PanelId.RIGHT_REAR_DOOR not in plan.missing and PanelId.RIGHT_FRONT_DOOR in plan.missing
    assert "If this car has them, also film: right rear door." in plan.topup_text
    assert "right rear door (" not in plan.topup_text
    # a whole side never seen stays on the list: the renter may simply have skipped it
    states = {pid: c.PanelCoverage(state=CoverageState.FILMED) for pid in PanelId}
    for pid in PanelId:
        if panels._panel_side_position(pid)[0] is Side.RIGHT:
            states[pid] = c.PanelCoverage(state=CoverageState.NOT_SEEN)
    skipped = panels.plan_topup(c.Coverage(per_role={"pickup": states}), TH)
    assert PanelId.RIGHT_REAR_DOOR in skipped.missing and "If this car has" not in skipped.topup_text


def test_a_topup_track_takes_the_one_slot_it_was_asked_for_and_the_loop_converges() -> None:
    pickup = {pid: c.PanelCoverage(state=CoverageState.FILMED) for pid in PanelId}
    pickup[PanelId.RIGHT_REAR_WHEEL] = c.PanelCoverage(state=CoverageState.NOT_FILMED)
    plan0 = panels.plan_topup(c.Coverage(per_role={"pickup": pickup}), TH)
    assert plan0.missing == [PanelId.RIGHT_REAR_WHEEL]
    assert plan0.nouns_to_rerun == [Noun.WHEEL, Noun.HEADLIGHT, Noun.TAIL_LIGHT, Noun.SIDE_MIRROR, Noun.CAR, Noun.LICENSE_PLATE, Noun.FACE]
    wheel = _moving("topup_1", Noun.WHEEL, range(0, 60), 700.0, -6.0, w=140, y0=350, h=140)  # a close-up, no light in frame
    wheel.track_id = "topup_1_wheel_00"
    _, plain = panels.vote_sides([wheel], -1, TH, fps=FPS)
    assert plain[0].panel_id is PanelId.LEFT_FRONT_WHEEL and panels.undecided(plain[0])  # the vote alone cannot tell
    voted_panels, tracks = panels.vote_sides([wheel], -1, TH, fps=FPS, expected_by_role={"topup_1": plan0.missing})
    t = tracks[0]
    assert t.panel_id is PanelId.RIGHT_REAR_WHEEL and t.side is Side.RIGHT and t.position is Position.REAR and not panels.undecided(t)
    assert not next(p for p in voted_panels if p.panel_id is PanelId.RIGHT_REAR_WHEEL).side_uncertain
    _, again = panels.vote_sides(tracks, -1, TH, fps=FPS, frozen_roles={"topup_1"})  # a later derive keeps the vote
    assert again[0].panel_id is PanelId.RIGHT_REAR_WHEEL
    # two undecided wheels on the one asked slot could be two different wheels: neither takes it
    other = _moving("topup_1", Noun.WHEEL, range(0, 60), 300.0, -6.0, w=140, y0=350, h=140)
    other.track_id = "topup_1_wheel_01"
    _, two = panels.vote_sides([wheel, other], -1, TH, fps=FPS, expected_by_role={"topup_1": plan0.missing})
    assert all(panels.undecided(t) and t.panel_id is PanelId.RIGHT_REAR_WHEEL for t in two)  # filed under the asked slot
    status = {Noun.WHEEL: LaneStatus.COMPLETED, Noun.CAR: LaneStatus.COMPLETED, Noun.LICENSE_PLATE: LaneStatus.COMPLETED, Noun.FACE: LaneStatus.COMPLETED}
    kf = _keyframe("topup_1", PanelId.RIGHT_REAR_WHEEL, t.track_id, 30, kf_id="topup_1_right_rear_wheel_kf1")
    top = panels.coverage_for_role(voted_panels, tracks, [kf], role="topup_1", fps=FPS, lane_status=status, th=TH)
    assert top[PanelId.RIGHT_REAR_WHEEL].state is CoverageState.FILMED
    assert top[PanelId.HOOD].state is CoverageState.NOT_RUN  # the hood lane did not run on the top-up clip
    plan1 = panels.plan_topup(c.Coverage(per_role={"pickup": pickup, "topup_1": top}, odometer=plan0.odometer), TH)
    assert plan1.missing == [] and plan1.topup_text.startswith("Every panel is filmed in the pickup walk.")


def test_expected_slots_are_the_missing_list_of_the_walk_before_a_topup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voted) -> None:
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    run_id = "01TESTEXPECTED"
    for noun in LANE_NOUNS:
        c.write_json(paths.lane_summary_path(run_id, "pickup", noun), _lane_record(noun, LaneStatus.COMPLETED))
    _panels, tracks, _direction = voted
    kfs = [
        _keyframe("pickup", t.panel_id, t.track_id, t.per_frame[len(t.per_frame) // 2].frame, kf_id=f"{t.track_id}_kf1")
        for t in tracks
        if t.panel_id is not None and t.panel_id is not PanelId.RIGHT_REAR_WHEEL
    ]
    assert panels._expected_slots(run_id, "topup_1", tracks, kfs, FPS, TH) == [PanelId.RIGHT_REAR_WHEEL]
    assert panels._expected_slots(run_id, "return", tracks, kfs, FPS, TH) == []


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


def _lane_record(noun: Noun, status: LaneStatus) -> c.LaneRecord:
    return c.LaneRecord(lane_id=c.noun_slug(noun), role="pickup", noun=noun, prompt_text=noun.value, media_size=[1280, 720], status=status, frames_sent=141)


def _fixture_lanes() -> dict[str, tuple[c.LaneRecord, list[c.IndexFrame]]]:
    car_frames, car_out = sam_wire.index_from_events(sam_wire.read_events(BOX_FIXTURE), media="video")
    door_frames, door_out = sam_wire.index_from_events(sam_wire.read_events(TEXT_FIXTURE), media="video")
    assert car_out.status is LaneStatus.COMPLETED and door_out.status is LaneStatus.COMPLETED
    return {
        "car": (_lane_record(Noun.CAR, LaneStatus.COMPLETED), car_frames),
        "car_door": (_lane_record(Noun.CAR_DOOR, LaneStatus.COMPLETED), door_frames),
        "fender": (_lane_record(Noun.FENDER, LaneStatus.ZERO_MATCHES), []),
        "wheel": (_lane_record(Noun.WHEEL, LaneStatus.FAILED), []),
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


def _fixture_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_id: str, role: str = "pickup") -> tuple[dict[str, Any], list[float]]:
    """A run dir with the clip as the original and the two 10/26 lanes on disk; the lanes and the pts map."""
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    shutil.copy(CLIP, paths.quarantine_original(run_id, role))
    shutil.copy(BOX_FIXTURE, paths.lane_events_path(run_id, role, Noun.CAR))
    shutil.copy(TEXT_FIXTURE, paths.lane_events_path(run_id, role, Noun.CAR_DOOR))
    pts = video.pts_map(CLIP)
    assert len(pts) == 141
    return _fixture_lanes(), pts


async def _passthrough_blur(run_id_: str, role_: str, rgb: np.ndarray, *, f_proxy: int, tracks: list[c.PanelTrack], fps: float, sizes, keep_plate: bool, th) -> tuple[np.ndarray, c.BlurRecord]:
    assert not keep_plate
    return rgb, c.BlurRecord(request_ids=[f"img_{f_proxy}"])


async def test_derive_on_the_fixture_pair_writes_the_four_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_th) -> None:
    run_id, role = "01TESTPANELS", "pickup"
    lanes, pts = _fixture_run(tmp_path, monkeypatch, run_id, role)
    # the renter chose keep_plate: stored crops and frames must still get a blurred plate
    c.write_json(paths.run_json(run_id), c.RunManifest(run_id=run_id, created_at="2026-10-27T15:04:05Z", keep_plate=True))
    blurred_calls: list[int] = []

    async def fake_blur(run_id_: str, role_: str, rgb: np.ndarray, *, f_proxy: int, tracks: list[c.PanelTrack], fps: float, sizes, keep_plate: bool, th) -> tuple[np.ndarray, c.BlurRecord]:
        assert run_id_ == run_id and role_ == role and rgb.shape == (720, 1280, 3) and keep_plate is False
        assert sizes == ((1280, 720), (1280, 720)) and fps == 24.0
        blurred_calls.append(f_proxy)
        return rgb, c.BlurRecord(request_ids=[f"img_{f_proxy}"])

    with sam_wire.DecodePool(1) as pool:
        tracks, voted_panels, keyframes, coverage = await panels.derive(
            run_id, role, lanes=lanes, pts=pts, media_size=(1280, 720), fps=24.0, decode=pool, blur_keyframe=fake_blur, th=fixture_th
        )
        decoded, _cpu_s = pool.stats()
    assert decoded == 0  # fastmask read every fixture mask in process; the pool is only the fallback
    door = next(t for t in tracks if t.noun is Noun.CAR_DOOR)
    assert door.panel_id is PanelId.LEFT_FRONT_DOOR  # a lone door with no mirror, anchor or wheel: a placeholder slot
    assert door.side is Side.NONE and door.position is Position.NONE
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
    door_cov = coverage[PanelId.LEFT_FRONT_DOOR]
    assert door_cov.state is CoverageState.GLIMPSED and "side uncertain" in door_cov.reason  # the ball is far under 3%
    assert door_cov.state_basis is StateBasis.C and door_cov.best_kf == best.kf_id
    assert door_cov.seconds == pytest.approx(134 / 24.0)
    for pid in (PanelId.LEFT_FRONT_FENDER, PanelId.RIGHT_REAR_FENDER):
        assert coverage[pid].state is CoverageState.NOT_FILMED  # a zero-match lane
    assert coverage[PanelId.LEFT_FRONT_WHEEL].state is CoverageState.NOT_RUN  # a failed lane
    assert coverage[PanelId.HOOD].state is CoverageState.NOT_RUN  # a lane that never ran
    other_door = coverage[PanelId.RIGHT_FRONT_DOOR]  # the undecided door may be any door: never not_seen or not_filmed
    assert other_door.state is CoverageState.GLIMPSED and other_door.best_kf == best.kf_id
    # the four files parse back through the contracts, sorted by role and id
    read_tracks = c.read_json_list(paths.tracks_json(run_id), c.PanelTrack)
    read_panels = c.read_json_list(paths.panels_json(run_id), c.Panel)
    read_kfs = c.read_json_list(paths.keyframes_json(run_id), c.Keyframe)
    read_cov = c.read_json(paths.coverage_json(run_id), c.Coverage)
    assert c.canonical_json(read_tracks) == c.canonical_json(tracks)
    assert [t.track_id for t in read_tracks] == sorted(t.track_id for t in read_tracks)
    assert c.canonical_json(read_panels) == c.canonical_json(voted_panels)
    assert c.canonical_json(read_kfs) == c.canonical_json(keyframes)
    assert read_cov.per_role["pickup"][PanelId.LEFT_FRONT_DOOR].state is CoverageState.GLIMPSED
    assert read_cov.missing[0] in c.WHEEL_PANELS and PanelId.LEFT_FRONT_DOOR in read_cov.missing
    assert set(read_cov.nouns_to_rerun) >= {Noun.CAR, Noun.LICENSE_PLATE, Noun.FACE}
    door_panel = next(p for p in read_panels if p.panel_id is PanelId.LEFT_FRONT_DOOR)
    assert door_panel.per_role["pickup"].keyframe_ids == [k.kf_id for k in kfs]
    assert door_panel.per_role["pickup"].state is CoverageState.GLIMPSED
    # a second derive of the same role (a resume or the stage retry) rewrites all four files byte for byte
    before = {p: p.read_bytes() for p in (paths.tracks_json(run_id), paths.panels_json(run_id), paths.keyframes_json(run_id), paths.coverage_json(run_id))}
    with sam_wire.DecodePool(1) as pool:
        await panels.derive(run_id, role, lanes=lanes, pts=pts, media_size=(1280, 720), fps=24.0, decode=pool, blur_keyframe=fake_blur, th=fixture_th)
    assert {p: p.read_bytes() for p in before} == before


async def test_derive_keeps_the_event_loop_free_and_reads_each_candidate_frame_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_th) -> None:
    run_id, role = "01TESTLOOP", "pickup"
    lanes, pts = _fixture_run(tmp_path, monkeypatch, run_id, role)
    requested: list[float] = []
    singles: list[float] = []
    real_iter, real_read = video.iter_frames_at, video.read_frame_at

    def counting_iter(path: Path, times: list[float]):
        requested.extend(times)
        yield from real_iter(path, times)

    def counting_read(path: Path, t: float) -> np.ndarray:
        singles.append(t)
        return real_read(path, t)

    monkeypatch.setattr(video, "iter_frames_at", counting_iter)
    monkeypatch.setattr(video, "read_frame_at", counting_read)
    lags: list[float] = []

    async def ticker() -> None:
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            lags.append(now - last - 0.005)
            last = now

    with sam_wire.DecodePool(1) as pool:
        task = asyncio.create_task(ticker())
        await asyncio.sleep(0.02)
        _tracks, _panels, keyframes, _cov = await panels.derive(
            run_id, role, lanes=lanes, pts=pts, media_size=(1280, 720), fps=24.0, decode=pool, blur_keyframe=_passthrough_blur, th=fixture_th
        )
        task.cancel()
    assert keyframes
    assert len(requested) == len(set(requested)) == 9 and singles == []  # each candidate read once, reused for its crop
    assert lags and max(lags) < 0.3, max(lags)


@pytest.mark.parametrize("fast", [True, False])
async def test_derive_decodes_each_mask_once_and_keeps_two_keyframes_over_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_th, fast: bool) -> None:
    run_id, role = "01TESTBUDGET", "pickup"
    lanes, pts = _fixture_run(tmp_path, monkeypatch, run_id, role)
    shutil.copy(TEXT_FIXTURE, paths.lane_events_path(run_id, role, Noun.BUMPER))  # a second panel on the same frames
    lanes["bumper"] = (_lane_record(Noun.BUMPER, LaneStatus.COMPLETED), lanes["car_door"][1])
    th = copy.deepcopy(fixture_th)
    th["panels"]["keyframe_decode_budget_s_per_video"] = -1.0  # always over budget
    th["panels"]["fast_mask_decode"] = fast
    pool_calls: list[tuple[str, list[int]]] = []
    fast_calls: list[tuple[str, int, int | None]] = []
    real_fast = fastmask.LaneMasks.decode

    def counting_fast(self: fastmask.LaneMasks, lane_path: Path, refs: list[c.MaskRef], *, rows: int | None = None) -> list[np.ndarray]:
        fast_calls.extend((Path(lane_path).name, ref.payload_offset, rows) for ref in refs)
        return real_fast(self, lane_path, refs, rows=rows)

    monkeypatch.setattr(fastmask.LaneMasks, "decode", counting_fast)

    class CountingPool:
        def __init__(self, pool: sam_wire.DecodePool) -> None:
            self.pool = pool

        def decode(self, lane_path: Path, refs: list[c.MaskRef]) -> list[np.ndarray]:
            pool_calls.append((Path(lane_path).name, [ref.payload_offset for ref in refs]))
            return self.pool.decode(lane_path, refs)

        def stats(self) -> tuple[int, float]:
            return self.pool.stats()

    blurred: list[int] = []

    async def counting_blur(run_id_: str, role_: str, rgb: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, c.BlurRecord]:
        blurred.append(kwargs["f_proxy"])
        await asyncio.sleep(0.01)  # both panels are in flight at once
        return rgb, c.BlurRecord(request_ids=[f"img_{kwargs['f_proxy']}"])

    with sam_wire.DecodePool(1) as pool:
        _tracks, _panels, keyframes, _cov = await panels.derive(
            run_id, role, lanes=lanes, pts=pts, media_size=(1280, 720), fps=24.0, decode=CountingPool(pool), blur_keyframe=counting_blur, th=th
        )
    if fast:
        assert pool_calls == []  # the pool only takes what fastmask refuses
        decoded = [(name, offset) for name, offset, _rows in fast_calls]
    else:
        assert fast_calls == []
        assert sorted(name for name, _ in pool_calls if name != "car.ndjson") == ["bumper.ndjson", "car_door.ndjson"]  # one call per lane
        decoded = [(name, offset) for name, offsets in pool_calls for offset in offsets]
    assert len(decoded) == len(set(decoded))  # no mask decoded twice, the car masks the two panels share included
    car = [offset for name, offset in decoded if name == "car.ndjson"]
    assert sum(1 for name, _ in decoded if name == "car_door.ndjson") == 9 and sum(1 for name, _ in decoded if name == "bumper.ndjson") == 9
    assert 1 <= len(car) < 9  # the containment test reads the car mask only at the candidates the panels kept
    by_panel: dict[PanelId, int] = {}
    for k in keyframes:
        by_panel[k.panel_id] = by_panel.get(k.panel_id, 0) + 1
    assert set(by_panel) == {PanelId.LEFT_FRONT_DOOR, PanelId.FRONT_BUMPER} and all(1 <= n <= 2 for n in by_panel.values())
    assert len(blurred) == len(set(blurred))  # a frame both panels keep is blurred once, and both carry that answer
    shared = {k.f_proxy for k in keyframes if k.panel_id is PanelId.LEFT_FRONT_DOOR} & {k.f_proxy for k in keyframes if k.panel_id is PanelId.FRONT_BUMPER}
    assert shared and all(k.blur.request_ids == [f"img_{k.f_proxy}"] for k in keyframes)


async def test_a_failing_blur_still_leaves_the_plate_and_face_tracks_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_th) -> None:
    run_id, role = "01TESTFAILCLOSED", "pickup"
    lanes, pts = _fixture_run(tmp_path, monkeypatch, run_id, role)
    lanes["face"] = (
        _lane_record(Noun.FACE, LaneStatus.COMPLETED),
        [c.IndexFrame(frame=f, objects=[c.IndexObject(id="0", box=[10, 10, 50, 60], c=0.8)]) for f in range(30, 33)],
    )

    async def broken_blur(*args: Any, **kwargs: Any) -> tuple[np.ndarray, c.BlurRecord]:
        raise RuntimeError("the image check request failed")

    with sam_wire.DecodePool(1) as pool:
        with pytest.raises(panels.KeyframeError, match="image check"):
            await panels.derive(run_id, role, lanes=lanes, pts=pts, media_size=(1280, 720), fps=24.0, decode=pool, blur_keyframe=broken_blur, th=fixture_th)
    on_disk = c.read_json_list(paths.tracks_json(run_id), c.PanelTrack)
    assert [[r.frame for r in t.per_frame] for t in on_disk if t.noun is Noun.FACE] == [[30, 31, 32]]
    assert not paths.keyframes_json(run_id).is_file() and not paths.coverage_json(run_id).is_file()
    assert not list(paths.run_dir(run_id).glob("**/*.jpg"))  # nothing unblurred was written


async def test_one_failed_blur_hands_its_slot_to_the_next_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_th) -> None:
    run_id, role = "01TESTFLAKY", "pickup"
    lanes, pts = _fixture_run(tmp_path, monkeypatch, run_id, role)
    calls: list[int] = []

    async def flaky_blur(run_id_: str, role_: str, rgb: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, c.BlurRecord]:
        calls.append(kwargs["f_proxy"])
        if len(calls) == 1:
            raise RuntimeError("one image check timed out")
        return rgb, c.BlurRecord()

    with sam_wire.DecodePool(1) as pool:
        _tracks, _panels, keyframes, coverage = await panels.derive(
            run_id, role, lanes=lanes, pts=pts, media_size=(1280, 720), fps=24.0, decode=pool, blur_keyframe=flaky_blur, th=fixture_th
        )
    frames = [k.f_proxy for k in keyframes]
    assert keyframes and calls[0] not in frames and frames == calls[1:]
    assert keyframes[0].kf_id == "pickup_left_front_door_kf1" and coverage[PanelId.LEFT_FRONT_DOOR].best_kf is not None


async def test_derive_hands_a_topup_its_request_and_freezes_the_walk_it_tops_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    run_id = "01TESTTOPUP"

    async def filmed_keyframes(run_id_: str, role: str, panels_: list[c.Panel], role_tracks: list[c.PanelTrack], *rest: Any) -> list[c.Keyframe]:
        return [_keyframe(role, t.panel_id, t.track_id, t.per_frame[0].frame, kf_id=f"{t.track_id}_kf1") for t in role_tracks if t.panel_id is not None]

    monkeypatch.setattr(panels, "_derive_keyframes", filmed_keyframes)  # this test is about the vote wiring, not masks
    walk, ids, pts = scripted_walk()
    missing_id = ids[(Noun.WHEEL, "right_rear")]
    walk[Noun.WHEEL] = [c.IndexFrame(frame=fr.frame, objects=[o for o in fr.objects if o.id != missing_id]) for fr in walk[Noun.WHEEL]]
    walk[Noun.WHEEL] = [fr for fr in walk[Noun.WHEEL] if fr.objects]
    for noun in LANE_NOUNS:
        c.write_json(paths.lane_summary_path(run_id, "pickup", noun), _lane_record(noun, LaneStatus.COMPLETED))
    lanes = {c.noun_slug(n): (_lane_record(n, LaneStatus.COMPLETED), walk[n]) for n in LANE_NOUNS}
    common = dict(pts=pts, media_size=(W, H), fps=FPS, decode=None, blur_keyframe=_passthrough_blur, th=TH)
    pickup_tracks, _p, _k, _cov = await panels.derive(run_id, "pickup", lanes=lanes, **common)
    assert c.read_json(paths.coverage_json(run_id), c.Coverage).missing == [PanelId.RIGHT_REAR_WHEEL]
    # the top-up clip: one wheel close-up with no light in frame, and only the nouns the plan re-runs
    frames = range(0, 60)
    wheel = [c.IndexFrame(frame=f, objects=[c.IndexObject(id="0", box=[700 - 6 * f, 350, 840 - 6 * f, 490], c=0.8)]) for f in frames]
    car = [c.IndexFrame(frame=f, objects=[c.IndexObject(id="0", box=[0, 100, W, H], c=0.9)]) for f in frames]
    top_lanes = {
        "wheel": (_lane_record(Noun.WHEEL, LaneStatus.COMPLETED), wheel),
        "car": (_lane_record(Noun.CAR, LaneStatus.COMPLETED), car),
        "license_plate": (_lane_record(Noun.LICENSE_PLATE, LaneStatus.ZERO_MATCHES), []),
        "face": (_lane_record(Noun.FACE, LaneStatus.ZERO_MATCHES), []),
    }
    tracks, _p, _k, top_cov = await panels.derive(run_id, "topup_1", lanes=top_lanes, **{**common, "pts": [f / FPS for f in frames]})
    top_wheel = next(t for t in tracks if t.role == "topup_1" and t.noun is Noun.WHEEL)
    assert top_wheel.panel_id is PanelId.RIGHT_REAR_WHEEL and not panels.undecided(top_wheel)
    assert top_cov[PanelId.RIGHT_REAR_WHEEL].state is CoverageState.FILMED and top_cov[PanelId.HOOD].state is CoverageState.NOT_RUN
    final = c.read_json(paths.coverage_json(run_id), c.Coverage)
    assert final.missing == [] and set(final.per_role) == {"pickup", "topup_1"}
    frozen = {t.track_id: t.panel_id for t in tracks if t.role == "pickup"}
    assert frozen == {t.track_id: t.panel_id for t in pickup_tracks}


async def test_a_crash_in_the_panel_path_still_leaves_the_face_track_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_th) -> None:
    run_id, role = "01TESTCRASH", "pickup"
    lanes, pts = _fixture_run(tmp_path, monkeypatch, run_id, role)
    lanes["face"] = (_lane_record(Noun.FACE, LaneStatus.COMPLETED), [c.IndexFrame(frame=7, objects=[c.IndexObject(id="0", box=[10, 10, 50, 60])])])

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("a bug in the panel path")

    monkeypatch.setattr(panels, "_role_panel_tracks", broken)
    with pytest.raises(ValueError):
        await panels.derive(run_id, role, lanes=lanes, pts=pts, media_size=(1280, 720), fps=24.0, decode=None, blur_keyframe=_passthrough_blur, th=fixture_th)
    on_disk = c.read_json_list(paths.tracks_json(run_id), c.PanelTrack)
    assert [(t.noun, [r.frame for r in t.per_frame]) for t in on_disk] == [(Noun.FACE, [7])]


# ----------------------------------------------------------------------------------------------------------------
# The twin check: which lid is the hood (panels lane 11/1)
# ----------------------------------------------------------------------------------------------------------------

TWIN_FIXTURE = ROOT / "fixtures" / "twin" / "pairC_core1_lane_boxes.json.gz"  # boxes only, no pixel or mask


def _pair_c_tracks(th: dict[str, Any]) -> list[c.PanelTrack]:
    """Both walks of the 10/31 live run on uCO3D pair C, rebuilt from the stored boxes of all 16 lanes the way derive()
    builds tracks."""
    data = json.loads(gzip.decompress(TWIN_FIXTURE.read_bytes()))
    tracks: list[c.PanelTrack] = []
    for role, walk in data["walks"].items():
        lanes = {
            Noun(noun): [c.IndexFrame(frame=f, objects=[c.IndexObject(id=oid, box=box, c=conf) for oid, box, conf in objs]) for f, objs in frames]
            for noun, frames in walk["lanes"].items()
        }
        built, _ = panels.build_role_tracks(role, lanes, pts=walk["pts"], media_size=tuple(walk["media_size"]), fps=walk["fps"], th=th)
        tracks += built
    return tracks


def _pair_c_vote(th: dict[str, Any], tracks: list[c.PanelTrack] | None = None, **kw: Any) -> tuple[list[c.Panel], list[c.PanelTrack]]:
    tracks = tracks if tracks is not None else _pair_c_tracks(th)
    dirs = {r: panels.walk_direction([t for t in tracks if t.role == r], th) for r in ("pickup", "return")}
    return panels.vote_sides(tracks, None, th, fps=15.0, direction_by_role=dirs, **kw)


def test_twin_check_files_the_pair_c_lids_by_the_end_they_show() -> None:
    voted_panels, voted = _pair_c_vote(TH)
    lids = {t.track_id: t for t in voted if t.noun in panels.LID_NOUNS}
    assert {tid: t.panel_id for tid, t in lids.items()} == {
        "pickup_hood_00": PanelId.HOOD,
        "pickup_hood_01": PanelId.TRUNK_LID,  # the hood lane on the trunk lid, frames 155 to 295
        "pickup_trunk_lid_00": PanelId.TRUNK_LID,
        "return_hood_00": PanelId.TRUNK_LID,  # the hood lane on the trunk lid, frames 0 to 103
        "return_hood_01": PanelId.HOOD,
        "return_trunk_lid_00": PanelId.TRUNK_LID,
        "return_trunk_lid_01": PanelId.HOOD,  # the trunk-lid lane on the hood, frames 184 to 303
    }
    assert all(t.twin_check is not None and t.twin_check.decided and abs(t.twin_check.score) >= 3 for t in lids.values())
    rear = lids["return_hood_00"].twin_check
    assert rear.lid == ["return_hood_00", "return_trunk_lid_00"] and rear.votes["plate"] == -1 and rear.votes["mirror"] == -1
    assert rear.numbers["plate_on_lid_share"] > 0.9 and rear.numbers["mirror_gap_lid_sizes"] > 2.5
    front = lids["return_trunk_lid_01"].twin_check
    assert front.votes["mirror"] == 1 and front.votes["glass"] == 1 and front.numbers["plate_on_lid_share"] == 0.0
    by_id = {p.panel_id: p for p in voted_panels}
    assert (by_id[PanelId.HOOD].front_sign, by_id[PanelId.TRUNK_LID].front_sign) == (1, -1)
    assert not by_id[PanelId.HOOD].side_uncertain and not by_id[PanelId.TRUNK_LID].side_uncertain
    assert by_id[PanelId.HOOD].per_role["return"].track_ids == ["return_hood_01", "return_trunk_lid_01"]
    assert by_id[PanelId.TRUNK_LID].per_role["pickup"].track_ids == ["pickup_hood_01", "pickup_trunk_lid_00"]


def test_lights_and_glass_take_the_end_of_the_lid_they_sit_beside() -> None:
    twin_only = copy.deepcopy(TH)
    twin_only["panels"].pop("door_check", None)  # the door check's moves are test_door_check_* below
    old = copy.deepcopy(twin_only)
    del old["panels"]["twin_check"]
    _, before = _pair_c_vote(old)
    _, after = _pair_c_vote(twin_only)
    was = {t.track_id: t.panel_id for t in before}
    moved = {t.track_id: t.panel_id for t in after if t.noun in panels.NEIGHBOUR_SLOTS and t.panel_id != was[t.track_id]}
    assert moved == {
        "pickup_headlight_03": PanelId.TAIL_LIGHTS,  # the headlight lane on a tail light at the rear quarter
        "pickup_windshield_02": PanelId.REAR_WINDOW,  # the windshield lane on the rear window
        "return_headlight_00": PanelId.TAIL_LIGHTS,
        "return_headlight_01": PanelId.TAIL_LIGHTS,
        "return_tail_light_03": PanelId.HEADLIGHTS,
        "return_tail_light_04": PanelId.HEADLIGHTS,
        "return_windshield_00": PanelId.REAR_WINDOW,
        "return_windshield_01": PanelId.REAR_WINDOW,
    }
    by_id = {t.track_id: t for t in after}
    assert by_id["return_windshield_00"].twin_check.votes == {"next_to_lid": -1} and by_id["return_windshield_00"].twin_check.lid == ["return_hood_00", "return_trunk_lid_00"]
    mud_flap = by_id["return_tail_light_02"]  # the tail light lane's orange mud flap by the front wheel: beside the hood now and then only
    assert mud_flap.panel_id is PanelId.TAIL_LIGHTS and mud_flap.twin_check is None
    # the placed lids also anchor the side vote: two placeholders and one door land on the side the walk shows
    lateral = {t.track_id: (t.panel_id, t.side) for t in after if t.panel_id != was[t.track_id] and t.noun in panels.LATERAL_NOUNS}
    assert lateral == {
        "pickup_wheel_02": (PanelId.RIGHT_REAR_WHEEL, Side.RIGHT),
        "pickup_fender_01": (PanelId.RIGHT_REAR_FENDER, Side.RIGHT),
        "return_car_door_00": (PanelId.LEFT_REAR_DOOR, Side.LEFT),  # the first frames show the rear and the left side
    }


def test_a_run_frozen_before_the_twin_check_keeps_the_lane_nouns() -> None:
    old = copy.deepcopy(TH)
    del old["panels"]["twin_check"]
    _, voted = _pair_c_vote(old)
    lids = [t for t in voted if t.noun in panels.LID_NOUNS]
    assert lids and all(t.panel_id is panels.SINGLE_SLOT[t.noun] and t.twin_check is None for t in lids)


def test_a_frozen_walk_keeps_its_twin_check_and_a_revote_without_it_drops_the_record() -> None:
    _, first = _pair_c_vote(TH)
    old = copy.deepcopy(TH)
    del old["panels"]["twin_check"]
    _, again = _pair_c_vote(old, tracks=first, frozen_roles={"pickup"})
    stored = {t.track_id: (t.panel_id, t.twin_check) for t in first if t.role == "pickup"}
    assert {t.track_id: (t.panel_id, t.twin_check) for t in again if t.role == "pickup"} == stored
    assert all(t.twin_check is None and t.panel_id is panels.SINGLE_SLOT[t.noun] for t in again if t.role == "return" and t.noun in panels.LID_NOUNS)


def test_twin_cues_read_the_plate_the_mirror_and_the_neighbours() -> None:
    cfg = TH["panels"]["twin_check"]
    lid = {f: [100.0, 100.0, 400.0, 300.0] for f in range(10)}
    rear = {
        Noun.LICENSE_PLATE: {f: [[200, 220, 280, 260]] for f in range(10)},  # on the lid
        Noun.SIDE_MIRROR: {f: [[1200, 120, 1240, 160]] for f in range(10)},  # 820 px off: 2.7 lid sizes
        Noun.TAIL_LIGHT: {f: [[80, 220, 110, 260]] for f in range(10)},  # touching the lid
        Noun.WINDSHIELD: {f: [[120, 40, 380, 98]] for f in range(10)},  # both glass lanes see the same glass
        Noun.REAR_WINDOW: {f: [[120, 40, 380, 98]] for f in range(10)},
    }
    votes, numbers = panels.twin_cues(lid, rear, cfg, min_frames=5)
    assert votes == {"plate": -1, "mirror": -1, "lights": -1, "glass": 0}
    assert numbers["plate_on_lid_share"] == 1.0 and numbers["mirror_gap_lid_sizes"] == pytest.approx(820 / 300)
    front = {
        Noun.SIDE_MIRROR: {f: [[410, 90, 440, 120]] for f in range(10)},  # beside the lid's corner
        Noun.HEADLIGHT: {f: [[110, 300, 160, 340]] for f in range(10)},  # at its front edge
        Noun.LICENSE_PLATE: {f: [[220, 380, 300, 410]] for f in range(10)},  # on the bumper, below it
    }
    votes, _ = panels.twin_cues(lid, front, cfg, min_frames=5)
    assert votes == {"plate": 0, "mirror": 1, "lights": 1, "glass": 0}
    votes, _ = panels.twin_cues({0: lid[0]}, rear, cfg, min_frames=5)  # one frame of a walk is not evidence
    assert votes == {"plate": 0, "mirror": 0, "lights": 0, "glass": 0}


def test_twin_decision_needs_the_margin_or_an_unopposed_lane() -> None:
    cfg = TH["panels"]["twin_check"]
    silent = {"plate": 0, "mirror": 0, "lights": 0, "glass": 0}
    assert cfg["margin_min"] == 3  # review 11/1: one strong cue, or the two weak ones, never turns a lid alone
    assert panels.twin_decision({**silent, "plate": -1, "lights": -1}, [Noun.HOOD], cfg) == (-1, -3.0)  # plate and lights move a hood-lane lid
    assert panels.twin_decision({**silent, "plate": -1}, [Noun.HOOD], cfg) == (None, -2.0)  # the plate alone against the lane: split
    assert panels.twin_decision({**silent, "mirror": 1}, [Noun.TRUNK_LID], cfg) == (None, 2.0)  # a mirror alone against it: split
    assert panels.twin_decision({**silent, "lights": 1, "glass": 1}, [Noun.TRUNK_LID], cfg) == (None, 2.0)  # the twin-prone weak cues: split
    assert panels.twin_decision({**silent, "plate": -1}, [Noun.TRUNK_LID], cfg) == (-1, -2.0)  # the same cue on the lane's side
    assert panels.twin_decision({**silent, "lights": 1}, [Noun.HOOD], cfg) == (1, 1.0)  # weak, but on the lane's side
    assert panels.twin_decision(silent, [Noun.TRUNK_LID], cfg) == (-1, 0.0)  # nothing against the lane
    assert panels.twin_decision({**silent, "lights": -1}, [Noun.HOOD], cfg) == (None, -1.0)  # weak and against it: split
    assert panels.twin_decision(silent, [Noun.HOOD, Noun.TRUNK_LID], cfg) == (None, 0.0)  # both lanes on one lid, no cue
    assert panels.twin_decision({"plate": -1, "mirror": 1, "lights": 1, "glass": 0}, [Noun.HOOD, Noun.TRUNK_LID], cfg) == (None, 1.0)


# The review of 11/1: a right trunk lid seen from the rear, its plate on the bumper (below the lid box), no mirror in
# view, and SAM's headlight and windshield lanes on the tail lights and the rear window (both seen on pair C).
_REAR = range(100, 160)
_LID = [300, 200, 600, 330]
_PLATE_ON_BUMPER = [410, 360, 490, 390]
_LIGHT_BESIDE = [250, 260, 300, 300]
_GLASS_ABOVE = [320, 140, 580, 198]


def _rear_lid_scene() -> list[c.PanelTrack]:
    return [
        _track("pickup", Noun.TRUNK_LID, [(f, _LID, 0.8) for f in _REAR], track_id="pickup_trunk_lid_00"),
        _track("pickup", Noun.LICENSE_PLATE, [(f, _PLATE_ON_BUMPER, 0.8) for f in _REAR], track_id="pickup_license_plate_00"),
    ]


def test_the_light_and_glass_lanes_alone_split_a_trunk_lid_instead_of_turning_it_into_the_hood() -> None:
    scene = _rear_lid_scene() + [
        _track("pickup", Noun.HEADLIGHT, [(f, _LIGHT_BESIDE, 0.8) for f in _REAR], track_id="pickup_headlight_00"),
        _track("pickup", Noun.WINDSHIELD, [(f, _GLASS_ABOVE, 0.8) for f in _REAR], track_id="pickup_windshield_00"),
    ]
    slot, record = panels.twin_check(scene, TH)["pickup_trunk_lid_00"]
    assert record.votes == {"plate": 0, "mirror": 0, "lights": 1, "glass": 1} and record.score == 2.0
    assert slot is None and not record.decided  # split: side uncertain, not compared (margin 2 filed it as the hood)
    voted_panels, voted = panels.vote_sides(scene, None, TH, fps=FPS)
    by_id = {p.panel_id: p for p in voted_panels}
    assert by_id[PanelId.TRUNK_LID].side_uncertain and by_id[PanelId.TRUNK_LID].front_sign is None
    assert "pickup" not in by_id[PanelId.HOOD].per_role  # nothing is filed under the hood
    assert next(t for t in voted if t.track_id == "pickup_trunk_lid_00").panel_id is PanelId.TRUNK_LID  # its lane's slot, as a placeholder


def test_a_mirror_alone_beside_a_trunk_lid_splits_it() -> None:
    mirror = [840, 200, 860, 230]  # 0.8 lid sizes off a 300 px lid on a rear-quarter view
    scene = _rear_lid_scene() + [_track("pickup", Noun.SIDE_MIRROR, [(f, mirror, 0.8) for f in _REAR], track_id="pickup_side_mirror_00")]
    slot, record = panels.twin_check(scene, TH)["pickup_trunk_lid_00"]
    assert record.votes["mirror"] == 1 and record.score == 2.0 and slot is None and not record.decided
    old = copy.deepcopy(TH)
    old["panels"]["twin_check"]["margin_min"] = 2  # the rule before the review: the mirror alone made it the hood
    assert panels.twin_check(scene, old)["pickup_trunk_lid_00"][0] is PanelId.HOOD


def test_a_split_twin_check_leaves_both_lids_side_uncertain() -> None:
    frames = range(0, 30)
    hood = _track("pickup", Noun.HOOD, [(f, [300, 200, 600, 400], 0.8) for f in frames], track_id="pickup_hood_00")
    trunk = _track("pickup", Noun.TRUNK_LID, [(f, [302, 201, 600, 400], 0.8) for f in frames], track_id="pickup_trunk_lid_00")
    voted_panels, voted = panels.vote_sides([hood, trunk], None, TH, fps=FPS)
    for t in voted:
        assert t.twin_check is not None and not t.twin_check.decided and t.twin_check.lid == ["pickup_hood_00", "pickup_trunk_lid_00"]
        assert t.panel_id is panels.SINGLE_SLOT[t.noun] and panels.candidate_slots(t) == [PanelId.HOOD, PanelId.TRUNK_LID]
    by_id = {p.panel_id: p for p in voted_panels}
    assert by_id[PanelId.HOOD].side_uncertain and by_id[PanelId.TRUNK_LID].side_uncertain  # G6: neither is compared
    assert by_id[PanelId.HOOD].front_sign is None and by_id[PanelId.TRUNK_LID].front_sign is None
    cov = panels.coverage_for_role(voted_panels, voted, [], role="pickup", fps=FPS, lane_status=_lane_status(), th=TH)
    for pid in (PanelId.HOOD, PanelId.TRUNK_LID):
        assert cov[pid].state is CoverageState.GLIMPSED and cov[pid].reason.startswith("front or rear uncertain: pickup_hood_00, pickup_trunk_lid_00 could be hood or trunk lid")


def test_a_relabelled_lid_decodes_its_masks_from_its_own_lane() -> None:
    trunk_on_hood = _filed(_track("return", Noun.TRUNK_LID, [(f, [300, 200, 600, 400], 0.8) for f in range(5)], track_id="return_trunk_lid_01"), PanelId.HOOD)
    hood_panel = c.Panel(panel_id=PanelId.HOOD, noun=Noun.HOOD)
    wanted = panels._wanted_masks([(hood_panel, [(2, trunk_on_hood)])], {Noun.CAR_WINDOW: {}, Noun.LICENSE_PLATE: {}, Noun.FACE: {}})
    assert wanted == {Noun.TRUNK_LID: [(2, "0")]}
