"""Panels (M5): SAM lanes to tracks, canonical panel names, keyframes, coverage and the top-up plan.

Covers docs/ARCHITECTURE.md 5.1 to 5.4. Every per-frame rule runs on boxes and c. Masks are decoded only for keyframe
candidates, in process through fastmask (the DecodePool the caller hands to derive() takes only what fastmask refuses),
and a candidate's car mask only when its panel is about to keep it. Nothing here reads a mark, calls Spark or opens the
network. Every function except derive() is a pure function of its arguments and the threshold dict th
(contracts.load_thresholds()); derive() is the one function that touches the run directory.

Orientation conventions used by the side vote (5.2):
- direction = sign of the image x motion of the scene (panel centroids drift that way as the camera walks).
- front_sign = the image x direction of the car's front as seen on a lateral panel. Facing the left (US driver)
  side the front appears image-left, so front_sign -1 is left and +1 is right.
- Walking front-to-rear along a side, the scene drifts toward the front, so front_sign = order x drift where order is
  +1 front-to-rear and -1 rear-to-front. R2 uses the panel's own drift, R3 the walk-level direction.
- End-on panels (hood, windshield, front bumper, headlights; trunk lid, rear window, rear bumper, tail lights) get a
  fixed front_sign (+1 front-facing, -1 rear-facing) so both walks whiten them the same way; roof and mirrors None.
- A track whose side vote ties keeps side none, and one whose position no rule decides keeps position none (a bumper
  too). It is filed under a placeholder among its candidate slots so its frames still get keyframes, its panel is
  side_uncertain, and coverage prints every candidate slot as side uncertain, never not_filmed.

Plate and face tracks skip the car filter and every 5.1 length and merge rule: the blur reads them from tracks.json, and
a face seen for three frames must still be pixelated.
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Collection, Sequence

import numpy as np

from . import contracts as c
from . import fastmask, paths, sam_wire
from .config import redact
from .contracts import (
    LANE_NOUNS,
    LOWER_BUMPERS,
    PANEL_NOUN,
    PANEL_NOUNS,
    PRIVACY_NOUNS,
    WHEEL_PANELS,
    CoverageState,
    InstanceFrom,
    LaneStatus,
    MergeRule,
    Noun,
    PanelId,
    Position,
    Side,
    StateBasis,
    noun_slug,
)

if TYPE_CHECKING:
    from meta_sam_parser import PromptObject

Box = Sequence[float]
log = logging.getLogger(__name__)

FRONT_ANCHORS: tuple[Noun, ...] = (Noun.HEADLIGHT, Noun.WINDSHIELD, Noun.HOOD)
REAR_ANCHORS: tuple[Noun, ...] = (Noun.TAIL_LIGHT, Noun.REAR_WINDOW, Noun.TRUNK_LID)
LATERAL_NOUNS: tuple[Noun, ...] = (Noun.CAR_DOOR, Noun.FENDER, Noun.WHEEL)
UNNAMED_NOUNS: tuple[Noun, ...] = (Noun.CAR, Noun.CAR_WINDOW, Noun.LICENSE_PLATE, Noun.FACE, Noun.DOOR_HANDLE)
DIRECTION_NOUNS: tuple[Noun, ...] = tuple(n for n in PANEL_NOUNS if n is not Noun.CAR)
END_ON_FRONT_SIGN: dict[PanelId, int] = {
    PanelId.HOOD: 1,
    PanelId.WINDSHIELD: 1,
    PanelId.FRONT_BUMPER: 1,
    PanelId.HEADLIGHTS: 1,
    PanelId.TRUNK_LID: -1,
    PanelId.REAR_WINDOW: -1,
    PanelId.REAR_BUMPER: -1,
    PanelId.TAIL_LIGHTS: -1,
}
GROUP_SLOTS: tuple[PanelId, ...] = (PanelId.HEADLIGHTS, PanelId.TAIL_LIGHTS, PanelId.MIRRORS)  # several instances per slot
SINGLE_SLOT: dict[Noun, PanelId] = {
    Noun.HOOD: PanelId.HOOD,
    Noun.ROOF: PanelId.ROOF,
    Noun.TRUNK_LID: PanelId.TRUNK_LID,
    Noun.HEADLIGHT: PanelId.HEADLIGHTS,
    Noun.TAIL_LIGHT: PanelId.TAIL_LIGHTS,
    Noun.SIDE_MIRROR: PanelId.MIRRORS,
    Noun.WINDSHIELD: PanelId.WINDSHIELD,
    Noun.REAR_WINDOW: PanelId.REAR_WINDOW,
}
# A two-door car has no rear doors, and SAM may not call a rear quarter panel a fender (5.2 item 4).
MAY_BE_ABSENT: tuple[PanelId, ...] = (
    PanelId.LEFT_REAR_DOOR,
    PanelId.RIGHT_REAR_DOOR,
    PanelId.LEFT_REAR_FENDER,
    PanelId.RIGHT_REAR_FENDER,
)
SIDE_ANCHOR_NOUNS: tuple[Noun, ...] = (Noun.HEADLIGHT, Noun.TAIL_LIGHT, Noun.SIDE_MIRROR)


# ----------------------------------------------------------------------------------------------------------------
# Box helpers
# ----------------------------------------------------------------------------------------------------------------


def _area(box: Box) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _width(box: Box) -> float:
    return max(0.0, float(box[2]) - float(box[0]))


def _height(box: Box) -> float:
    return max(0.0, float(box[3]) - float(box[1]))


def _aspect(box: Box) -> float:
    h = _height(box)
    return _width(box) / h if h > 0 else 0.0


def _centroid(box: Box) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _intersection(a: Box, b: Box) -> float:
    w = min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))
    h = min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    return w * h if w > 0 and h > 0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Share of inner's area that lies inside outer (0 for an empty inner box)."""
    area = _area(inner)
    return _intersection(inner, outer) / area if area > 0 else 0.0


def box_iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _id_key(object_id: str) -> tuple[int, str]:
    return (len(object_id), object_id)


def _box_lookup(frames: list[c.IndexFrame]) -> dict[tuple[int, str], c.IndexObject]:
    out: dict[tuple[int, str], c.IndexObject] = {}
    for fr in frames:
        for obj in fr.objects:
            out.setdefault((fr.frame, obj.id), obj)
    return out


def _pts_at(pts: Sequence[float], frame: int, fallback_fps: float) -> float:
    if 0 <= frame < len(pts):
        return float(pts[frame])
    if len(pts) >= 2:
        dt = statistics.median(pts[i + 1] - pts[i] for i in range(len(pts) - 1))
        return float(pts[-1]) + (frame - (len(pts) - 1)) * dt
    return frame / fallback_fps


def _object_at(track: c.PanelTrack, frame: int) -> str | None:
    """The SAM object id a track used at a frame, from its fragments."""
    for frag in track.fragments:
        if frag.first_frame <= frame <= frag.last_frame:
            return frag.object_id
    return None


# ----------------------------------------------------------------------------------------------------------------
# 5.1 filters and fragments
# ----------------------------------------------------------------------------------------------------------------


def car_boxes_by_frame(car_frames: list[c.IndexFrame]) -> dict[int, list[int]]:
    """The largest "car" box per frame."""
    out: dict[int, list[int]] = {}
    for fr in car_frames:
        best: tuple[float, list[int]] | None = None
        for obj in fr.objects:
            area = _area(obj.box)
            if best is None or area > best[0]:
                best = (area, [int(v) for v in obj.box])
        if best is not None and best[0] > 0:
            out[fr.frame] = best[1]
    return out


def border_frac(box: Box, media_size: tuple[int, int]) -> float:
    """Share of the box perimeter that lies on the frame edge (half-open right and bottom)."""
    w_img, h_img = media_size
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    perimeter = 2.0 * (w + h)
    if perimeter <= 0:
        return 0.0
    edge = 0.0
    if x1 <= 0:
        edge += h
    if x2 >= w_img:
        edge += h
    if y1 <= 0:
        edge += w
    if y2 >= h_img:
        edge += w
    return min(1.0, edge / perimeter)


def area_floor(noun: Noun, th: dict[str, Any]) -> float:
    p = th["panels"]
    return float(p["area_floor_small"]) if noun.value in p["small_nouns"] else float(p["area_floor_default"])


def filter_objects(
    frames: list[c.IndexFrame],
    noun: Noun,
    car_boxes: dict[int, list[int]],
    media_size: tuple[int, int],
    th: dict[str, Any],
) -> list[c.IndexFrame]:
    """Keep an object when >= car_box_containment_min of its box lies in the car box and it clears the area floor."""
    floor = area_floor(noun, th)
    cmin = float(th["panels"]["car_box_containment_min"])
    frame_area = float(media_size[0] * media_size[1])
    out: list[c.IndexFrame] = []
    for fr in frames:
        car = car_boxes.get(fr.frame)
        if car is None:
            continue
        kept = [o for o in fr.objects if _area(o.box) / frame_area >= floor and containment(o.box, car) >= cmin]
        if kept:
            out.append(c.IndexFrame(frame=fr.frame, objects=kept))
    return out


def split_gap_frames(fps: float, th: dict[str, Any]) -> float:
    """A fragment splits where the frame gap exceeds max(0.5 s, 2.5 frames), in frames."""
    p = th["panels"]
    return max(float(p["fragment_gap_split_s"]) * fps, float(p["fragment_gap_split_frames"]))


def min_fragment_frames(fps: float, th: dict[str, Any]) -> int:
    p = th["panels"]
    return max(int(p["fragment_min_frames"]), int(round(fps * float(p["fragment_min_s"]))))


def build_fragments(frames: list[c.IndexFrame], *, fps: float, th: dict[str, Any]) -> list[c.TrackFragment]:
    """Per object id over consecutive frames; split at a long gap; drop short fragments."""
    by_id: dict[str, list[int]] = {}
    for fr in frames:
        for obj in fr.objects:
            by_id.setdefault(obj.id, []).append(fr.frame)
    split = split_gap_frames(fps, th)
    keep = min_fragment_frames(fps, th)
    runs: list[tuple[str, list[int]]] = []
    for oid, frame_list in by_id.items():
        ordered = sorted(set(frame_list))
        run = [ordered[0]]
        for f in ordered[1:]:
            if f - run[-1] > split:
                runs.append((oid, run))
                run = [f]
            else:
                run.append(f)
        runs.append((oid, run))
    frags = [c.TrackFragment(object_id=oid, first_frame=r[0], last_frame=r[-1]) for oid, r in runs if len(r) >= keep]
    return sorted(frags, key=lambda fr: (fr.first_frame, fr.last_frame, _id_key(fr.object_id)))


def _fragment_rows(frag: c.TrackFragment, lookup: dict[tuple[int, str], c.IndexObject]) -> list[tuple[int, Box]]:
    rows = []
    for f in range(frag.first_frame, frag.last_frame + 1):
        obj = lookup.get((f, frag.object_id))
        if obj is not None:
            rows.append((f, obj.box))
    return rows


def _median_velocity(rows: list[tuple[int, Box]]) -> tuple[float, float]:
    vx: list[float] = []
    vy: list[float] = []
    for (f1, b1), (f2, b2) in zip(rows, rows[1:]):
        df = f2 - f1
        if df <= 0:
            continue
        c1, c2 = _centroid(b1), _centroid(b2)
        vx.append((c2[0] - c1[0]) / df)
        vy.append((c2[1] - c1[1]) / df)
    if not vx:
        return (0.0, 0.0)
    return (statistics.median(vx), statistics.median(vy))


def merge_test(
    a: c.TrackFragment,
    b: c.TrackFragment,
    lookup: dict[tuple[int, str], c.IndexObject],
    *,
    fps: float,
    th: dict[str, Any],
) -> c.MergeRecord | None:
    """The 5.1 item 4 test for merging B onto A; the MergeRecord with its numbers, or None."""
    p = th["panels"]
    gap_frames = b.first_frame - a.last_frame
    if gap_frames < 1:
        return None
    gap_s = gap_frames / fps
    if gap_s > float(p["merge_gap_max_s"]):
        return None
    a_rows = _fragment_rows(a, lookup)
    b_obj = lookup.get((b.first_frame, b.object_id))
    if not a_rows or b_obj is None:
        return None
    last_box = a_rows[-1][1]
    vx, vy = _median_velocity(a_rows)
    cx, cy = _centroid(last_box)
    pred = (cx + vx * gap_frames, cy + vy * gap_frames)
    bx, by = _centroid(b_obj.box)
    pred_dist = math.hypot(pred[0] - bx, pred[1] - by) / max(_width(last_box), 1.0)
    if pred_dist > float(p["merge_pred_dist_box_widths"]):
        return None
    area_a = _area(last_box)
    if area_a <= 0:
        return None
    area_ratio = _area(b_obj.box) / area_a
    lo, hi = p["merge_area_ratio"]
    if not float(lo) <= area_ratio <= float(hi):
        return None
    asp_a, asp_b = _aspect(last_box), _aspect(b_obj.box)
    if asp_a <= 0 or asp_b <= 0 or max(asp_a, asp_b) / min(asp_a, asp_b) > 1.0 + float(p["merge_aspect_tolerance"]):
        return None
    return c.MergeRecord(rule=MergeRule.VELOCITY_GAP, gap_s=gap_s, pred_dist=pred_dist, area_ratio=area_ratio)


def merge_fragments(
    fragments: list[c.TrackFragment],
    frames: list[c.IndexFrame],
    *,
    fps: float,
    th: dict[str, Any],
) -> list[tuple[list[c.TrackFragment], list[c.MergeRecord]]]:
    """Greedy in time order: each fragment joins the group whose tail predicts it best, else starts a group."""
    lookup = _box_lookup(frames)
    groups: list[tuple[list[c.TrackFragment], list[c.MergeRecord]]] = []
    for frag in sorted(fragments, key=lambda fr: (fr.first_frame, fr.last_frame, _id_key(fr.object_id))):
        best: tuple[int, c.MergeRecord] | None = None
        for gi, (frags, _) in enumerate(groups):
            rec = merge_test(frags[-1], frag, lookup, fps=fps, th=th)
            if rec is not None and (best is None or (rec.pred_dist or 0.0) < (best[1].pred_dist or 0.0)):
                best = (gi, rec)
        if best is None:
            groups.append(([frag], []))
        else:
            groups[best[0]][0].append(frag)
            groups[best[0]][1].append(best[1])
    return groups


def build_tracks(
    role: str,
    noun: Noun,
    groups: list[tuple[list[c.TrackFragment], list[c.MergeRecord]]],
    frames: list[c.IndexFrame],
    pts: Sequence[float],
    car_boxes: dict[int, list[int]],
    media_size: tuple[int, int],
    *,
    th: dict[str, Any],
) -> list[c.PanelTrack]:
    """One PanelTrack per merged group with its per_frame rows; panel_id stays None until the vote."""
    lookup = _box_lookup(frames)
    cmin = float(th["panels"]["car_box_containment_min"])
    fallback_fps = float(th["media"]["proxy_fps"])
    frame_area = float(media_size[0] * media_size[1])
    ordered = sorted(groups, key=lambda g: (g[0][0].first_frame, g[0][0].last_frame, _id_key(g[0][0].object_id)))
    tracks: list[c.PanelTrack] = []
    for i, (frags, merges) in enumerate(ordered):
        rows: dict[int, c.FrameObs] = {}
        ids: list[str] = []
        for frag in sorted(frags, key=lambda fr: (fr.first_frame, fr.last_frame)):
            if frag.object_id not in ids:
                ids.append(frag.object_id)
            for f, box in _fragment_rows(frag, lookup):
                if f in rows:
                    continue
                obj = lookup[(f, frag.object_id)]
                fbox = [float(v) for v in box]
                car = car_boxes.get(f)
                rows[f] = c.FrameObs(
                    frame=f,
                    pts=_pts_at(pts, f, fallback_fps),
                    box=fbox,
                    box_area_frac=_area(fbox) / frame_area,
                    in_car_box=car is not None and containment(fbox, car) >= cmin,
                    c=obj.c,
                    border_frac=border_frac(fbox, media_size),
                )
        per_frame = [rows[f] for f in sorted(rows)]
        n_c = sum(1 for r in per_frame if r.c is not None)
        tracks.append(
            c.PanelTrack(
                track_id=f"{role}_{noun_slug(noun)}_{i:02d}",
                role=role,
                noun=noun,
                sam_object_ids=ids,
                fragments=sorted(frags, key=lambda fr: (fr.first_frame, fr.last_frame)),
                merges=list(merges),
                c_present_share=n_c / len(per_frame) if per_frame else 0.0,
                per_frame=per_frame,
            )
        )
    return tracks


def privacy_tracks(
    role: str,
    noun: Noun,
    frames: list[c.IndexFrame],
    pts: Sequence[float],
    car_boxes: dict[int, list[int]],
    media_size: tuple[int, int],
    *,
    fps: float,
    th: dict[str, Any],
) -> list[c.PanelTrack]:
    """Plate and face tracks for the blur: one track per SAM object id with every frame it was seen on. No car filter,
    area floor, minimum length or merge; fragments still split at the 5.1 gap so the record shows where the id dropped
    out."""
    by_id: dict[str, list[int]] = {}
    for fr in frames:
        for obj in fr.objects:
            by_id.setdefault(obj.id, []).append(fr.frame)
    split = split_gap_frames(fps, th)
    groups: list[tuple[list[c.TrackFragment], list[c.MergeRecord]]] = []
    for oid in sorted(by_id, key=_id_key):
        ordered = sorted(set(by_id[oid]))
        frags: list[c.TrackFragment] = []
        start = prev = ordered[0]
        for f in ordered[1:]:
            if f - prev > split:
                frags.append(c.TrackFragment(object_id=oid, first_frame=start, last_frame=prev))
                start = f
            prev = f
        frags.append(c.TrackFragment(object_id=oid, first_frame=start, last_frame=prev))
        groups.append((frags, []))
    return build_tracks(role, noun, groups, frames, pts, car_boxes, media_size, th=th)


def suppress_duplicates(frames: list[c.IndexFrame], th: dict[str, Any]) -> list[c.IndexFrame]:
    """Drop a box that repeats a longer-lived id's box in the same frame (box IoU >= anchor_absorb_iou_min). SAM can
    hand one panel two ids, and the second would otherwise be voted into the neighbouring slot."""
    # TODO(D1, E2/E3): count how often the hosted tracker emits same-frame duplicates; drop this pass if it never does.
    iou_min = float(th["panels"]["anchor_absorb_iou_min"])
    life: dict[str, int] = {}
    for fr in frames:
        for obj in fr.objects:
            life[obj.id] = life.get(obj.id, 0) + 1
    out: list[c.IndexFrame] = []
    for fr in frames:
        keep: list[c.IndexObject] = []
        for obj in sorted(fr.objects, key=lambda o: (-life[o.id], _id_key(o.id))):
            if all(box_iou(obj.box, k.box) < iou_min for k in keep):
                keep.append(obj)
        if len(keep) == len(fr.objects):
            out.append(fr)
        elif keep:
            ids = {o.id for o in keep}
            out.append(c.IndexFrame(frame=fr.frame, objects=[o for o in fr.objects if o.id in ids]))
    return out


def _role_privacy_tracks(
    role: str,
    frames_by_noun: dict[Noun, list[c.IndexFrame]],
    *,
    pts: Sequence[float],
    media_size: tuple[int, int],
    fps: float,
    th: dict[str, Any],
) -> list[c.PanelTrack]:
    car_boxes = car_boxes_by_frame(frames_by_noun.get(Noun.CAR, []))
    return [
        t
        for noun in PRIVACY_NOUNS
        for t in privacy_tracks(role, noun, frames_by_noun.get(noun, []), pts, car_boxes, media_size, fps=fps, th=th)
    ]


def _role_panel_tracks(
    role: str,
    frames_by_noun: dict[Noun, list[c.IndexFrame]],
    *,
    pts: Sequence[float],
    media_size: tuple[int, int],
    fps: float,
    th: dict[str, Any],
) -> tuple[list[c.PanelTrack], dict[Noun, list[c.IndexFrame]]]:
    car_boxes = car_boxes_by_frame(frames_by_noun.get(Noun.CAR, []))
    kept: dict[Noun, list[c.IndexFrame]] = {noun: frames_by_noun.get(noun, []) for noun in PRIVACY_NOUNS}
    tracks: list[c.PanelTrack] = []
    for noun in LANE_NOUNS:
        if noun in PRIVACY_NOUNS:
            continue
        kept[noun] = filter_objects(frames_by_noun.get(noun, []), noun, car_boxes, media_size, th)
        if noun is not Noun.CAR:
            kept[noun] = suppress_duplicates(kept[noun], th)
        frags = build_fragments(kept[noun], fps=fps, th=th)
        groups = merge_fragments(frags, kept[noun], fps=fps, th=th)
        tracks.extend(build_tracks(role, noun, groups, kept[noun], pts, car_boxes, media_size, th=th))
    return tracks, kept


def build_role_tracks(
    role: str,
    frames_by_noun: dict[Noun, list[c.IndexFrame]],
    *,
    pts: Sequence[float],
    media_size: tuple[int, int],
    fps: float,
    th: dict[str, Any],
) -> tuple[list[c.PanelTrack], dict[Noun, list[c.IndexFrame]]]:
    """Every lane of one walk to PanelTracks: privacy_tracks for plate and face (first), then the car filter, duplicate
    suppression, fragments and merges for the panel nouns. Returns the tracks and the frames each noun kept."""
    common = dict(pts=pts, media_size=media_size, fps=fps, th=th)
    tracks, kept = _role_panel_tracks(role, frames_by_noun, **common)
    return _role_privacy_tracks(role, frames_by_noun, **common) + tracks, kept


def _longest_share(tracks: list[c.PanelTrack], visible: set[int]) -> float:
    """Frames of the longest track over the frames where any track is visible."""
    longest = max((len({r.frame for r in t.per_frame}) for t in tracks), default=0)
    return longest / len(visible) if visible else 1.0


def fragmented_panels(tracks: list[c.PanelTrack], th: dict[str, Any]) -> list[PanelId]:
    """Voted panels whose longest track covers under anchor_pass_coverage_below of the frames their tracks span.
    The group slots (headlights, tail lights, mirrors) hold one track per instance and are never fragmented."""
    below = float(th["panels"]["anchor_pass_coverage_below"])
    by_panel: dict[PanelId, list[c.PanelTrack]] = {}
    for t in tracks:
        if t.panel_id is not None and t.panel_id not in GROUP_SLOTS:
            by_panel.setdefault(t.panel_id, []).append(t)
    out = []
    for pid, mine in by_panel.items():
        visible = {r.frame for t in mine for r in t.per_frame}
        if _longest_share(mine, visible) < below:
            out.append(pid)
    return sorted(out, key=lambda pid: pid.value)


def needs_anchor_pass(tracks: list[c.PanelTrack], frames: list[c.IndexFrame], th: dict[str, Any]) -> bool:
    """True when a panel is still fragmented: its longest track covers under anchor_pass_coverage_below of the
    frames where any of its tracks is visible. Before the vote (no panel ids yet) the test runs per noun over
    frames, the frames where any fragment of the noun is visible."""
    if any(t.panel_id is not None for t in tracks):
        return bool(fragmented_panels(tracks, th))
    visible = {fr.frame for fr in frames if fr.objects}
    if not visible:
        return False
    return _longest_share(tracks, visible) < float(th["panels"]["anchor_pass_coverage_below"])


def _prompt_box(box: Box, media_size: tuple[int, int], *, pad_px: int):
    """sam_client.prompt_box: integer edges, padded, clamped half-open (section 5 step 6)."""
    from .sam_client import prompt_box

    return prompt_box([float(v) for v in box], media_size, pad_px=pad_px)


def _mask_centroid_in_box(mask: np.ndarray, box: Box) -> tuple[float, float] | None:
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    h, w = mask.shape[:2]
    sub = mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
    if sub.size == 0 or not sub.any():
        return None
    ys, xs = np.nonzero(sub)
    return (float(xs.mean()) + max(0, x1), float(ys.mean()) + max(0, y1))


def anchor_prompts(
    track: c.PanelTrack,
    kf_frames: list[int],
    window_masks: dict[int, np.ndarray] | None,
    media_size: tuple[int, int],
    th: dict[str, Any],
    *,
    frame_offset: int = 0,
    object_id: str = "1",
) -> list[PromptObject]:
    """One PromptObject with a FramePrompt per chosen frame (at most 3); a negative point at the glass centroid
    when window_masks holds that frame. The seam point is backlog B5. Frames are clip-relative via frame_offset."""
    from meta_sam_parser import FramePrompt, PromptObject, PromptPoint

    rows = {r.frame: r for r in track.per_frame}
    chosen = [f for f in kf_frames if f in rows][: int(th["panels"]["anchor_max_frames_per_panel"])]
    prompts = []
    for f in chosen:
        box = _prompt_box(rows[f].box, media_size, pad_px=int(th["panels"]["prompt_box_pad_px"]))
        points: list[PromptPoint] = []
        if window_masks is not None and f in window_masks:
            pt = _mask_centroid_in_box(window_masks[f], rows[f].box)
            if pt is not None:
                x = min(media_size[0] - 1, max(0, int(round(pt[0]))))
                y = min(media_size[1] - 1, max(0, int(round(pt[1]))))
                points.append(PromptPoint(x=x, y=y, label="negative"))
        prompts.append(FramePrompt(frame=f - frame_offset, box=box, points=tuple(points)))
    return [PromptObject(id=object_id, prompts=tuple(prompts))] if prompts else []


def absorb_anchor(
    tracks: list[c.PanelTrack],
    anchor_frames: list[c.IndexFrame],
    *,
    frame_offset: int,
    th: dict[str, Any],
    media_size: tuple[int, int],
    pts: Sequence[float] | None = None,
    fps: float | None = None,
    car_boxes: dict[int, list[int]] | None = None,
    request_id: str = "",
    prompts: list[c.AnchorReprompt] | None = None,
) -> list[c.PanelTrack]:
    """Merge the tracks whose boxes agree with an anchor object (median box IoU >= anchor_absorb_iou_min on shared
    frames) into one track; clip frames map back by frame_offset. Tracks that match no anchor are returned as is.
    media_size (the proxy's display size) is required for box_area_frac and border_frac. in_car_box comes from
    car_boxes when given, else True: the anchor was prompted on a panel inside the car."""
    iou_min = float(th["panels"]["anchor_absorb_iou_min"])
    cmin = float(th["panels"]["car_box_containment_min"])
    fallback_dt = 1.0 / float(th["media"]["proxy_fps"])
    by_anchor: dict[str, dict[int, c.IndexObject]] = {}
    for fr in anchor_frames:
        for obj in fr.objects:
            by_anchor.setdefault(obj.id, {})[fr.frame + frame_offset] = obj
    remaining = list(tracks)
    out: list[c.PanelTrack] = []
    for anchor_id in sorted(by_anchor, key=_id_key):
        a_rows = by_anchor[anchor_id]
        matched = []
        for t in remaining:
            ious = [box_iou(r.box, a_rows[r.frame].box) for r in t.per_frame if r.frame in a_rows]
            if ious and statistics.median(ious) >= iou_min:
                matched.append(t)
        if not matched:
            continue
        for t in matched:
            remaining.remove(t)
        out.append(
            _absorbed_track(
                matched, anchor_id, a_rows, media_size, pts, fps, request_id, prompts or [],
                car_boxes=car_boxes, cmin=cmin, fallback_dt=fallback_dt,
            )
        )
    out.extend(remaining)
    return sorted(out, key=lambda t: (t.per_frame[0].frame if t.per_frame else 0, t.track_id))


def _absorbed_track(
    matched: list[c.PanelTrack],
    anchor_id: str,
    a_rows: dict[int, c.IndexObject],
    media_size: tuple[int, int],
    pts: Sequence[float] | None,
    fps: float | None,
    request_id: str,
    prompts: list[c.AnchorReprompt],
    *,
    car_boxes: dict[int, list[int]] | None,
    cmin: float,
    fallback_dt: float,
) -> c.PanelTrack:
    matched = sorted(matched, key=lambda t: t.per_frame[0].frame if t.per_frame else 0)
    base = matched[0]
    rows: dict[int, c.FrameObs] = {}
    ids: list[str] = []
    frags: list[c.TrackFragment] = []
    merges: list[c.MergeRecord] = list(base.merges)
    prev_last: int | None = None
    for t in matched:
        for oid in t.sam_object_ids:
            if oid not in ids:
                ids.append(oid)
        frags.extend(t.fragments)
        for r in t.per_frame:
            rows.setdefault(r.frame, r)
        if prev_last is not None and t.per_frame:
            gap_frames = t.per_frame[0].frame - prev_last
            dt = 1.0 / fps if fps else _row_dt(base.per_frame, fallback_dt)
            merges.append(c.MergeRecord(rule=MergeRule.ANCHOR_ABSORB, gap_s=max(0.0, gap_frames * dt)))
        if t.per_frame:
            prev_last = t.per_frame[-1].frame
    frame_area = float(media_size[0] * media_size[1])
    for f in sorted(a_rows):
        if f in rows:
            continue
        obj = a_rows[f]
        fbox = [float(v) for v in obj.box]
        car = car_boxes.get(f) if car_boxes is not None else None
        rows[f] = c.FrameObs(
            frame=f,
            pts=_pts_at(pts, f, fps or 1.0 / _row_dt(base.per_frame, fallback_dt)) if pts is not None else _extrapolate_pts(base.per_frame, f, fallback_dt),
            box=fbox,
            box_area_frac=_area(fbox) / frame_area if frame_area > 0 else 0.0,
            in_car_box=True if car_boxes is None else car is not None and containment(fbox, car) >= cmin,
            c=obj.c,
            border_frac=border_frac(fbox, media_size),
        )
    anchor_frames_sorted = sorted(a_rows)
    anchor_tag = f"anchor/{anchor_id}"
    if anchor_tag not in ids:
        ids.append(anchor_tag)
    frags.append(c.TrackFragment(object_id=anchor_tag, first_frame=anchor_frames_sorted[0], last_frame=anchor_frames_sorted[-1]))
    per_frame = [rows[f] for f in sorted(rows)]
    n_c = sum(1 for r in per_frame if r.c is not None)
    reprompts = list(base.anchor_reprompts) + list(prompts)
    if request_id and not prompts:
        first = anchor_frames_sorted[0]
        reprompts.append(c.AnchorReprompt(request_id=request_id, frame=first, box=[int(v) for v in a_rows[first].box]))
    return base.model_copy(
        update={
            "sam_object_ids": ids,
            "fragments": sorted(frags, key=lambda fr: (fr.first_frame, fr.last_frame)),
            "merges": merges,
            "anchor_reprompts": reprompts,
            "c_present_share": n_c / len(per_frame) if per_frame else 0.0,
            "per_frame": per_frame,
        }
    )


def _row_dt(rows: list[c.FrameObs], fallback_dt: float) -> float:
    deltas = [(b.pts - a.pts) / (b.frame - a.frame) for a, b in zip(rows, rows[1:]) if b.frame > a.frame]
    return statistics.median(deltas) if deltas else fallback_dt


def _extrapolate_pts(rows: list[c.FrameObs], frame: int, fallback_dt: float) -> float:
    if not rows:
        return 0.0
    dt = _row_dt(rows, fallback_dt)
    return rows[0].pts + (frame - rows[0].frame) * dt


# ----------------------------------------------------------------------------------------------------------------
# 5.2 walk direction and the side vote
# ----------------------------------------------------------------------------------------------------------------


def _centroid_dx(track: c.PanelTrack) -> list[float]:
    out = []
    for a, b in zip(track.per_frame, track.per_frame[1:]):
        df = b.frame - a.frame
        if df > 0:
            out.append((_centroid(b.box)[0] - _centroid(a.box)[0]) / df)
    return out


def walk_direction(tracks: list[c.PanelTrack], th: dict[str, Any], *, backup: int | None = None) -> int | None:
    """Sign of the median per-frame x displacement of panel centroids; None under direction_min_px_per_frame.
    backup is a phase_correlation_sign the caller measured; it is used only when direction_phase_corr is on."""
    floor = float(th["panels"]["direction_min_px_per_frame"])
    dxs = [d for t in tracks if t.noun in DIRECTION_NOUNS for d in _centroid_dx(t)]
    if dxs:
        med = statistics.median(dxs)
        if abs(med) >= floor:
            return _sign(med)
    if th["panels"].get("direction_phase_corr") and backup in (1, -1):
        return backup
    return None


def phase_correlation_sign(prev_gray: np.ndarray, next_gray: np.ndarray, th: dict[str, Any]) -> int | None:
    """The scene's x shift sign between two frames by phase correlation; None under the per-frame floor."""
    import cv2

    a = np.asarray(prev_gray, dtype=np.float32)
    b = np.asarray(next_gray, dtype=np.float32)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    if a.shape != b.shape:
        return None
    (dx, _dy), _response = cv2.phaseCorrelate(a, b)
    if abs(dx) < float(th["panels"]["direction_min_px_per_frame"]):
        return None
    return _sign(dx)


def _own_drift_sign(track: c.PanelTrack, floor: float) -> int:
    dxs = _centroid_dx(track)
    if not dxs:
        return 0
    med = statistics.median(dxs)
    return _sign(med) if abs(med) >= floor else 0


class _RoleContext:
    """The anchors, mirrors and wheels of one role, with the shared-frame geometry the side rules read."""

    def __init__(self, tracks: list[c.PanelTrack], direction: int | None, floor: float, ends: dict[str, int] | None = None) -> None:
        self.tracks = tracks
        self.direction = direction
        self.floor = floor
        self.by_noun: dict[Noun, list[c.PanelTrack]] = {}
        for t in tracks:
            noun = t.noun
            if ends is not None and t.track_id in ends:  # a lid the twin check placed anchors its end; a split one none
                if not ends[t.track_id]:
                    continue
                noun = Noun.HOOD if ends[t.track_id] > 0 else Noun.TRUNK_LID
            self.by_noun.setdefault(noun, []).append(t)
        self.cent: dict[str, dict[int, tuple[float, float]]] = {
            t.track_id: {r.frame: _centroid(r.box) for r in t.per_frame} for t in tracks
        }
        anchors: list[tuple[int, str]] = []
        for noun in FRONT_ANCHORS:
            for t in self.by_noun.get(noun, []):
                anchors.extend((r.frame, "F") for r in t.per_frame)
        for noun in REAR_ANCHORS:
            for t in self.by_noun.get(noun, []):
                anchors.extend((r.frame, "R") for r in t.per_frame)
        self.anchors = sorted(set(anchors))

    def frames(self, t: c.PanelTrack) -> set[int]:
        return set(self.cent[t.track_id])

    def shared(self, a: c.PanelTrack, b: c.PanelTrack) -> list[int]:
        return sorted(self.frames(a) & self.frames(b))

    def mean_dx(self, a: c.PanelTrack, b: c.PanelTrack, frames: list[int]) -> float:
        """Mean of (b.cx - a.cx) over frames."""
        ca, cb = self.cent[a.track_id], self.cent[b.track_id]
        return float(np.mean([cb[f][0] - ca[f][0] for f in frames])) if frames else 0.0

    def window(self, t: c.PanelTrack) -> tuple[int, int]:
        fr = self.cent[t.track_id]
        return (min(fr), max(fr)) if fr else (0, 0)

    def mid(self, t: c.PanelTrack) -> float:
        first, last = self.window(t)
        return (first + last) / 2.0

    def anchor_sign_on_shared(self, t: c.PanelTrack) -> tuple[int, int]:
        """(front_sign vote, shared frame count) from front and rear anchors sharing frames with t."""
        best: tuple[int, int] = (0, 0)
        for noun, polarity in ((n, 1) for n in FRONT_ANCHORS):
            best = self._better(best, t, noun, polarity)
        for noun in REAR_ANCHORS:
            best = self._better(best, t, noun, -1)
        return best

    def _better(self, best: tuple[int, int], t: c.PanelTrack, noun: Noun, polarity: int) -> tuple[int, int]:
        for a in self.by_noun.get(noun, []):
            sh = self.shared(t, a)
            if len(sh) > best[1]:
                vote = polarity * _sign(self.mean_dx(t, a, sh))
                if vote != 0:
                    best = (vote, len(sh))
        return best

    def adjacent_order(self, t: c.PanelTrack) -> int:
        """+1 when the walk passes t front-to-rear, -1 rear-to-front, 0 unknown, from the anchors outside t's window."""
        first, last = self.window(t)
        before = [k for f, k in self.anchors if f < first]
        after = [k for f, k in self.anchors if f > last]
        kb = before[-1] if before else None
        ka = after[0] if after else None
        if kb == "F" and ka == "R":
            return 1
        if kb == "R" and ka == "F":
            return -1
        if kb is None and ka is not None:
            return 1 if ka == "R" else -1
        if ka is None and kb is not None:
            return 1 if kb == "F" else -1
        return 0

    def nearest_anchor_kind(self, t: c.PanelTrack) -> str | None:
        first, last = self.window(t)
        best: tuple[float, str] | None = None
        for f, k in self.anchors:
            dist = 0.0 if first <= f <= last else min(abs(f - first), abs(f - last))
            if best is None or dist < best[0]:
                best = (dist, k)
        return best[1] if best else None

    def nearest_wheel(self, t: c.PanelTrack) -> c.PanelTrack | None:
        wheels = self.by_noun.get(Noun.WHEEL, [])
        if not wheels:
            return None
        shared_counts = [(len(self.shared(t, w)), w) for w in wheels]
        with_shared = [(n, w) for n, w in shared_counts if n > 0]
        if with_shared:
            return min(with_shared, key=lambda nw: abs(self.mean_dx(t, nw[1], self.shared(t, nw[1]))))[1]
        return min(wheels, key=lambda w: abs(self.mid(w) - self.mid(t)))

    def wheel_light_vote(self, wheel: c.PanelTrack) -> tuple[int, Position]:
        """R4 on a wheel: the headlight it shares frames with marks the front, a tail light the rear."""
        best: tuple[int, int, Position] = (0, 0, Position.NONE)
        for noun, polarity, pos in ((Noun.HEADLIGHT, 1, Position.FRONT), (Noun.TAIL_LIGHT, -1, Position.REAR)):
            for light in self.by_noun.get(noun, []):
                sh = self.shared(wheel, light)
                if len(sh) > best[1]:
                    vote = polarity * _sign(self.mean_dx(wheel, light, sh))
                    if vote != 0:
                        best = (vote, len(sh), pos)
        return best[0], best[2]

    def mirror_rule(self, door: c.PanelTrack) -> tuple[int, Position]:
        """R1 on a door: the mirror sharing the most frames marks the front; the nearest door to it is the front door."""
        mirrors = self.by_noun.get(Noun.SIDE_MIRROR, [])
        best: tuple[int, c.PanelTrack] | None = None
        for m in mirrors:
            n = len(self.shared(door, m))
            if n > 0 and (best is None or n > best[0]):
                best = (n, m)
        if best is None:
            return 0, Position.NONE
        mirror = best[1]
        sh = self.shared(door, mirror)
        vote = _sign(self.mean_dx(door, mirror, sh))
        my_dist = abs(self.mean_dx(door, mirror, sh))
        position = Position.FRONT
        for other in self.by_noun.get(Noun.CAR_DOOR, []):
            if other.track_id == door.track_id:
                continue
            osh = self.shared(other, mirror)
            if osh and abs(self.mean_dx(other, mirror, osh)) < my_dist and self.shared(other, door):
                position = Position.REAR
        return vote, position

    def pairwise_position(self, t: c.PanelTrack, side: Side, front_sign: int, sides: dict[str, Side]) -> Position:
        """Among same-noun tracks on the same side sharing frames, the one farther toward the front is front."""
        if front_sign == 0:
            return Position.NONE
        peers = [
            o
            for o in self.by_noun.get(t.noun, [])
            if o.track_id != t.track_id and sides.get(o.track_id) == side and self.shared(t, o)
        ]
        if not peers:
            return Position.NONE
        rel = [self.mean_dx(t, o, self.shared(t, o)) * front_sign for o in peers]
        if all(r < 0 for r in rel):
            return Position.FRONT
        if all(r > 0 for r in rel):
            return Position.REAR
        return Position.FRONT if abs(min(rel)) >= abs(max(rel)) else Position.REAR


def _side_from_votes(votes: c.SideVotes, margin_min: float) -> tuple[Side, bool, int]:
    """(the side the votes lean to, undecided, front_sign); a total under the margin leaves the side undecided."""
    total = votes.R1 + votes.R2 + votes.R3 + votes.R4
    lean = Side.RIGHT if total > 0 else Side.LEFT
    if total == 0 or abs(total) < margin_min:
        return lean, True, 0
    return lean, False, _sign(total)


def candidate_slots(track: c.PanelTrack) -> list[PanelId]:
    """The slots a voted track may be: its own, or every slot an undecided side or position allows."""
    if track.panel_id is None:
        return []
    if track.noun in LATERAL_NOUNS:
        sides = [track.side] if track.side is not Side.NONE else [Side.LEFT, Side.RIGHT]
        positions = [track.position] if track.position is not Position.NONE else [Position.FRONT, Position.REAR]
        return [_lateral_panel_id(track.noun, s, q) for s in sides for q in positions]
    if track.noun is Noun.BUMPER and track.position is Position.NONE:
        return [PanelId.FRONT_BUMPER, PanelId.REAR_BUMPER]
    if track.twin_check is not None and not track.twin_check.decided:
        return [PanelId.HOOD, PanelId.TRUNK_LID]
    if track.part_check is not None and track.part_check.kind == "end" and not track.part_check.decided:
        return [PanelId.WINDSHIELD, PanelId.REAR_WINDOW]
    return [track.panel_id]


def undecided(track: c.PanelTrack) -> bool:
    """True for a voted track the side vote could not place in a single slot."""
    return len(candidate_slots(track)) > 1


def _take_slot(track: c.PanelTrack, pid: PanelId) -> None:
    side, position = _panel_side_position(pid)
    track.panel_id, track.side, track.position = pid, side, position


def _lateral_panel_id(noun: Noun, side: Side, position: Position) -> PanelId:
    suffix = {Noun.CAR_DOOR: "door", Noun.FENDER: "fender", Noun.WHEEL: "wheel"}[noun]
    return PanelId(f"{side.value}_{position.value}_{suffix}")


def _panel_side_position(pid: PanelId) -> tuple[Side, Position]:
    parts = pid.value.split("_")
    side = Side(parts[0]) if parts[0] in ("left", "right") else Side.NONE
    position = Position.NONE
    if pid in LOWER_BUMPERS or side is not Side.NONE:
        position = Position.FRONT if "front" in parts else Position.REAR if "rear" in parts else Position.NONE
    return side, position


def _side_votes_for(ctx: _RoleContext, t: c.PanelTrack, weights: dict[str, float]) -> tuple[c.SideVotes, dict[str, Position]]:
    """The four signed votes (positive = right) for one lateral track plus the positions the rules suggest."""
    r1 = r2 = r3 = r4 = 0
    positions: dict[str, Position] = {}
    if t.noun is Noun.CAR_DOOR:
        r1, pos1 = ctx.mirror_rule(t)
        if pos1 is not Position.NONE:
            positions["R1"] = pos1
    vote, _n = ctx.anchor_sign_on_shared(t)
    if vote != 0:
        r2 = vote
    else:
        order = ctx.adjacent_order(t)
        drift = _own_drift_sign(t, ctx.floor)
        r2 = order * drift
    if ctx.direction is not None:
        r3 = ctx.adjacent_order(t) * ctx.direction
    if t.noun is Noun.WHEEL:
        r4, pos4 = ctx.wheel_light_vote(t)
        if pos4 is not Position.NONE:
            positions["R4"] = pos4
    else:
        wheel = ctx.nearest_wheel(t)
        if wheel is not None:
            r4, pos4 = ctx.wheel_light_vote(wheel)
            if pos4 is not Position.NONE:
                positions["R4"] = pos4
    votes = c.SideVotes(
        R1=float(weights["R1"]) * r1,
        R2=float(weights["R2"]) * r2,
        R3=float(weights["R3"]) * r3,
        R4=float(weights["R4"]) * r4,
    )
    return votes, positions


def _bumper_position(ctx: _RoleContext, t: c.PanelTrack) -> Position:
    heads = sum(len(ctx.shared(t, h)) for h in ctx.by_noun.get(Noun.HEADLIGHT, []))
    tails = sum(len(ctx.shared(t, h)) for h in ctx.by_noun.get(Noun.TAIL_LIGHT, []))
    if heads > tails:
        return Position.FRONT
    if tails > heads:
        return Position.REAR
    kind = ctx.nearest_anchor_kind(t)
    return Position.FRONT if kind == "F" else Position.REAR if kind == "R" else Position.NONE


def _sum_votes(votes: list[c.SideVotes]) -> c.SideVotes:
    return c.SideVotes(
        R1=sum(v.R1 for v in votes), R2=sum(v.R2 for v in votes), R3=sum(v.R3 for v in votes), R4=sum(v.R4 for v in votes)
    )


def _windows(tracks: list[c.PanelTrack], merge_gap_s: float) -> list[list[float]]:
    spans: list[list[float]] = []
    for t in tracks:
        rows = {r.frame: r.pts for r in t.per_frame}
        for frag in t.fragments:
            inside = [rows[f] for f in range(frag.first_frame, frag.last_frame + 1) if f in rows]
            if inside:
                spans.append([min(inside), max(inside)])
    spans.sort()
    merged: list[list[float]] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(list(s))
    return merged


def _seconds_visible(tracks: list[c.PanelTrack], fps: float | None, fallback_dt: float) -> float:
    frames = {r.frame for t in tracks for r in t.per_frame if r.in_car_box}
    if not frames:
        return 0.0
    if fps:
        return len(frames) / fps
    rows = [r for t in tracks for r in t.per_frame]
    return len(frames) * _row_dt(sorted(rows, key=lambda r: r.frame), fallback_dt)


def _lateral_position(
    ctx: _RoleContext,
    t: c.PanelTrack,
    side: Side,
    front_sign: int,
    sides: dict[str, Side],
    suggested: dict[str, Position],
) -> Position:
    """Pairwise order among decided same-side peers, then R1, then R4, then the nearest anchor; NONE when none decides."""
    position = ctx.pairwise_position(t, side, front_sign, sides) if side is not Side.NONE else Position.NONE
    for rule in ("R1", "R4"):
        if position is Position.NONE:
            position = suggested.get(rule, Position.NONE)
    if position is Position.NONE:
        kind = ctx.nearest_anchor_kind(t)
        position = Position.FRONT if kind == "F" else Position.REAR if kind == "R" else Position.NONE
    return position


def _vote_role(
    ctx: _RoleContext,
    role_tracks: list[c.PanelTrack],
    weights: dict[str, float],
    margin_min: float,
    expected: Collection[PanelId],
    votes_by_track: dict[str, c.SideVotes],
) -> None:
    """Vote one role's tracks in place (panel_id, side, position) and record each lateral track's SideVotes."""
    sides: dict[str, Side] = {}
    leans: dict[str, Side] = {}
    signs: dict[str, int] = {}
    suggested: dict[str, dict[str, Position]] = {}
    for t in role_tracks:
        if t.noun in LATERAL_NOUNS:
            votes, positions = _side_votes_for(ctx, t, weights)
            lean, unsure, fs = _side_from_votes(votes, margin_min)
            votes_by_track[t.track_id] = votes
            sides[t.track_id] = Side.NONE if unsure else lean
            leans[t.track_id] = lean
            signs[t.track_id] = fs
            suggested[t.track_id] = positions
    held: set[PanelId] = set()
    pending: list[tuple[c.PanelTrack, list[PanelId]]] = []
    for t in role_tracks:
        if t.noun in LATERAL_NOUNS:
            side = sides[t.track_id]
            position = _lateral_position(ctx, t, side, signs[t.track_id], sides, suggested[t.track_id])
            t.side, t.position = side, position
            if side is not Side.NONE:
                order = [side]
            else:
                order = [Side.RIGHT, Side.LEFT] if leans[t.track_id] is Side.RIGHT else [Side.LEFT, Side.RIGHT]
            spots = [position] if position is not Position.NONE else [Position.FRONT, Position.REAR]
            cands = [_lateral_panel_id(t.noun, s, q) for s in order for q in spots]
        elif t.noun is Noun.BUMPER:
            votes_by_track[t.track_id] = c.SideVotes()
            position = _bumper_position(ctx, t)
            t.side, t.position = Side.NONE, position
            by_position = {Position.FRONT: [PanelId.FRONT_BUMPER], Position.REAR: [PanelId.REAR_BUMPER]}
            cands = by_position.get(position, [PanelId.FRONT_BUMPER, PanelId.REAR_BUMPER])
        elif t.noun in SINGLE_SLOT:
            votes_by_track[t.track_id] = c.SideVotes()
            t.side, t.position, t.panel_id = Side.NONE, Position.NONE, SINGLE_SLOT[t.noun]
            continue
        else:
            t.panel_id, t.side, t.position = None, Side.NONE, Position.NONE
            continue
        if len(cands) == 1:
            t.panel_id = cands[0]
            held.add(cands[0])
        else:
            pending.append((t, cands))
    claims: dict[PanelId, list[c.PanelTrack]] = {}
    for t, cands in pending:
        asked = [pid for pid in cands if pid in expected]
        if len(asked) == 1 and asked[0] not in held:
            claims.setdefault(asked[0], []).append(t)
    taken: set[str] = set()
    for pid, claimants in claims.items():
        if len(claimants) == 1:  # two tracks on one asked slot could be two panels: neither takes it
            _take_slot(claimants[0], pid)
            taken.add(claimants[0].track_id)
            log.info("%s: %s filed as %s, the one slot among its candidates the top-up asked for", claimants[0].role, claimants[0].track_id, pid.value)
    for t, cands in pending:
        if t.track_id in taken:
            continue
        asked = [pid for pid in cands if pid in expected]
        free = [pid for pid in (asked or cands) if pid not in held]
        t.panel_id = (free or asked or cands)[0]


def _panels_from(
    tracks: list[c.PanelTrack],
    votes_by_track: dict[str, c.SideVotes],
    *,
    fps: float | None,
    merge_gap_s: float,
    fallback_dt: float,
    per_track: bool = False,
) -> list[c.Panel]:
    """Every canonical PanelId with its tracks per role, summed votes, side_uncertain and front_sign. With per_track
    (thresholds panels.uncertain_per_track) the panel is side_uncertain only when some walk holds nothing but undecided
    tracks in the slot: a stray undecided track beside decided ones is listed in coverage and takes no keyframe there
    (_keyframe_plan), so it no longer hides the panel."""
    panels: list[c.Panel] = []
    for pid in PanelId:
        mine = [t for t in tracks if t.panel_id == pid]
        noun = PANEL_NOUN[pid]
        side, position = _panel_side_position(pid)
        if per_track:
            unc = any(all(undecided(t) for t in mine if t.role == role) for role in {t.role for t in mine})
        else:
            unc = any(undecided(t) for t in mine)
        if not mine or unc:
            front_sign = None
        elif noun in LATERAL_NOUNS:
            front_sign = 1 if side is Side.RIGHT else -1
        else:
            front_sign = END_ON_FRONT_SIGN.get(pid)
        per_role: dict[str, c.PanelRole] = {}
        for role in sorted({t.role for t in mine}, key=_role_order):
            rts = [t for t in mine if t.role == role]
            per_role[role] = c.PanelRole(
                track_ids=[t.track_id for t in rts],
                windows=_windows(rts, merge_gap_s),
                seconds_visible=_seconds_visible(rts, fps, fallback_dt),
            )
        panels.append(
            c.Panel(
                panel_id=pid,
                noun=noun,
                side=side,
                position=position,
                instance_from=InstanceFrom.TRACKS if mine else InstanceFrom.CANONICAL,
                votes=_sum_votes([votes_by_track.get(t.track_id, c.SideVotes()) for t in mine]),
                side_uncertain=unc,
                front_sign=front_sign,
                per_role=per_role,
            )
        )
    return panels


def vote_sides(
    tracks: list[c.PanelTrack],
    direction: int | None,
    th: dict[str, Any],
    *,
    fps: float | None = None,
    direction_by_role: dict[str, int | None] | None = None,
    frozen_roles: Collection[str] = (),
    expected_by_role: dict[str, Collection[PanelId]] | None = None,
) -> tuple[list[c.Panel], list[c.PanelTrack]]:
    """R1 to R4 per lateral track with side_vote_weights (SideVotes signed, positive for right); bumpers go front or rear
    by the light they share frames with; every canonical PanelId comes back. A side under side_vote_margin_min, or a
    position no rule decides, stays Side.NONE or Position.NONE: the track is filed under a placeholder candidate slot
    (an asked slot, else left and front first, skipping slots a decided track of the role holds) and its panel is
    side_uncertain. expected_by_role names a top-up's asked slots: an undecided track that is the only claimant of the
    one asked slot among its candidates takes it. frozen_roles keep their stored panel_id, side and position (their
    votes are recomputed for the sums). direction_by_role overrides direction per role (walks may go opposite ways).
    After the vote, the twin check files the lids, lights and glass and the door check (thresholds panels.door_check)
    the doors of every role that is not frozen."""
    p = th["panels"]
    weights = p["side_vote_weights"]
    margin_min = float(p["side_vote_margin_min"])
    floor = float(p["direction_min_px_per_frame"])
    out_tracks = [t.model_copy(deep=True) for t in tracks]
    votes_by_track: dict[str, c.SideVotes] = {}
    by_role: dict[str, list[c.PanelTrack]] = {}
    for t in out_tracks:
        by_role.setdefault(t.role, []).append(t)
    for role, role_tracks in by_role.items():
        role_direction = direction if direction_by_role is None else direction_by_role.get(role, direction)
        stored = {t.track_id: (t.panel_id, t.side, t.position) for t in role_tracks}
        expected = (expected_by_role or {}).get(role, ())
        twin = twin_check(role_tracks, th)
        # only the lids move the side vote's anchors: lights and glass placed by them changed wheels and fenders both
        # ways on pair C (panels lane notes in docs/CONTRACTS.md)
        lid_ids = {t.track_id for t in role_tracks if t.noun in LID_NOUNS}
        ends = {tid: LID_END[PANEL_NOUN[slot]] if slot is not None else 0 for tid, (slot, _record) in twin.items() if tid in lid_ids}
        ctx = _RoleContext(role_tracks, role_direction, floor, ends)
        _vote_role(ctx, role_tracks, weights, margin_min, expected, votes_by_track)
        if role in frozen_roles:
            for t in role_tracks:
                t.panel_id, t.side, t.position = stored[t.track_id]
        else:
            apply_twin_check(role_tracks, th, result=twin)
            apply_door_check(ctx, role_tracks, th)
            apply_part_check(role_tracks, th)
    panels = _panels_from(
        out_tracks,
        votes_by_track,
        fps=fps,
        merge_gap_s=float(p["merge_gap_max_s"]),
        fallback_dt=1.0 / float(th["media"]["proxy_fps"]),
        per_track=uncertain_per_track(th),
    )
    return panels, out_tracks


def uncertain_per_track(th: dict[str, Any] | None) -> bool:
    """thresholds panels.uncertain_per_track (lexi run 1, 11/2): side or position uncertainty is judged per track, not
    per slot. Off when absent, so runs frozen before it keep their panels."""
    return bool((th or {}).get("panels", {}).get("uncertain_per_track", False))


def _role_order(role: str) -> tuple[int, int, str]:
    """pickup, return, the top-ups by number, then the claims by number."""
    if role in ("pickup", "return"):
        return (0 if role == "pickup" else 1, 0, role)
    prefix, _, number = role.partition("_")
    return ({"topup": 2, "claim": 3}.get(prefix, 4), int(number) if number.isdigit() else 0, role)


# ----------------------------------------------------------------------------------------------------------------
# 5.2 the twin check: which lid is the hood
# ----------------------------------------------------------------------------------------------------------------
#
# The hood and the trunk lid can look alike (pair C's white Ambassador): SAM 3.1's "hood" lane then holds the trunk
# lid whenever the camera is at the rear, and its "trunk lid" lane the hood at the front, and the match compared a
# trunk with a hood. So the lane noun is only the prior. The tracks of both lanes that show one lid (boxes overlapping
# on shared frames) are judged together from what the walk already tracked, each cue voting +1 for the front, -1 for
# the rear or 0:
#   plate   a license plate box lying on the lid box marks the rear (rear plates sit on the trunk lid or right under
#           it; a front plate sits on the bumper, below the grille);
#   mirror  side mirrors sit beside the hood's rear corners: near the lid marks the front, far from it the rear;
#   lights  headlights or tail lights next to the lid, when one lane sees them clearly more often than the other;
#   glass   the windshield or the rear window next to the lid, the same way.
# Plate and mirror weigh 2 and lights and glass 1 (thresholds panels.twin_check); a sum reaching the margin decides.
# Under it the lane noun stands when all the lid's tracks come from one lane and no cue points the other way;
# otherwise the evidence is split, the tracks keep their lane's slot as a placeholder and the panel is side_uncertain.

LID_NOUNS: tuple[Noun, ...] = (Noun.HOOD, Noun.TRUNK_LID)
LID_END: dict[Noun, int] = {Noun.HOOD: 1, Noun.TRUNK_LID: -1}
END_LID: dict[int, PanelId] = {1: PanelId.HOOD, -1: PanelId.TRUNK_LID}
# Lights and glass have twins too (pair C: the headlight lane on the tail lights, the windshield lane on the rear
# window). Once a lid is placed, a light or glass track takes the end of the placed lid it sits next to.
NEIGHBOUR_SLOTS: dict[Noun, dict[int, PanelId]] = {
    Noun.HEADLIGHT: {1: PanelId.HEADLIGHTS, -1: PanelId.TAIL_LIGHTS},
    Noun.TAIL_LIGHT: {1: PanelId.HEADLIGHTS, -1: PanelId.TAIL_LIGHTS},
    Noun.WINDSHIELD: {1: PanelId.WINDSHIELD, -1: PanelId.REAR_WINDOW},
    Noun.REAR_WINDOW: {1: PanelId.WINDSHIELD, -1: PanelId.REAR_WINDOW},
}
TWIN_NOUNS: tuple[Noun, ...] = (*LID_NOUNS, *NEIGHBOUR_SLOTS)
TWIN_CUES: dict[str, tuple[Noun, Noun]] = {"lights": (Noun.HEADLIGHT, Noun.TAIL_LIGHT), "glass": (Noun.WINDSHIELD, Noun.REAR_WINDOW)}
TWIN_OBSERVED: tuple[Noun, ...] = (Noun.LICENSE_PLATE, Noun.SIDE_MIRROR, Noun.HEADLIGHT, Noun.TAIL_LIGHT, Noun.WINDSHIELD, Noun.REAR_WINDOW)


def twin_settings(th: dict[str, Any]) -> dict[str, Any] | None:
    """thresholds panels.twin_check, or None when the run's thresholds predate it (the check is then off)."""
    cfg = th.get("panels", {}).get("twin_check")
    return cfg if isinstance(cfg, dict) else None


def _box_gap(a: Box, b: Box) -> float:
    """Distance between two boxes, 0 when they touch or overlap."""
    dx = max(float(b[0]) - float(a[2]), float(a[0]) - float(b[2]), 0.0)
    dy = max(float(b[1]) - float(a[3]), float(a[1]) - float(b[3]), 0.0)
    return math.hypot(dx, dy)


def _point_gap(point: tuple[float, float], box: Box) -> float:
    """Distance from a point to a box, 0 inside it."""
    dx = max(float(box[0]) - point[0], point[0] - float(box[2]), 0.0)
    dy = max(float(box[1]) - point[1], point[1] - float(box[3]), 0.0)
    return math.hypot(dx, dy)


def _lid_size(box: Box) -> float:
    return max(_width(box), _height(box), 1.0)


def twin_cues(
    lid: dict[int, Box],
    seen: dict[Noun, dict[int, list[Box]]],
    cfg: dict[str, Any],
    *,
    min_frames: int,
) -> tuple[dict[str, int], dict[str, float]]:
    """The four votes (+1 front, -1 rear, 0 silent) on one lid, given its box per frame and the boxes of the plate,
    mirror, light and glass lanes per frame, plus the numbers behind them. A cue votes only on at least min_frames
    frames of evidence (1 for a single photo)."""
    frames = sorted(lid)
    votes: dict[str, int] = {}
    numbers: dict[str, float] = {"lid_frames": float(len(frames))}
    plates = seen.get(Noun.LICENSE_PLATE, {})
    with_plate = [f for f in frames if plates.get(f)]
    on_lid = sum(1 for f in with_plate if any(containment(p, lid[f]) >= float(cfg["plate_inside_min"]) for p in plates[f]))
    share = on_lid / len(with_plate) if with_plate else 0.0
    numbers.update(plate_frames=float(len(with_plate)), plate_on_lid_share=share)
    votes["plate"] = -1 if len(with_plate) >= min_frames and share >= float(cfg["plate_share_min"]) else 0
    mirrors = seen.get(Noun.SIDE_MIRROR, {})
    gaps = [min(_point_gap(_centroid(m), lid[f]) for m in mirrors[f]) / _lid_size(lid[f]) for f in frames if mirrors.get(f)]
    numbers["mirror_frames"] = float(len(gaps))
    votes["mirror"] = 0
    if gaps and len(gaps) >= min_frames:
        gap = statistics.median(gaps)
        numbers["mirror_gap_lid_sizes"] = gap
        votes["mirror"] = 1 if gap <= float(cfg["mirror_near_lid_sizes"]) else -1 if gap >= float(cfg["mirror_far_lid_sizes"]) else 0
    near = float(cfg["adjacent_lid_sizes"])
    margin = float(cfg["presence_margin"])
    for cue, (front, rear) in TWIN_CUES.items():
        shares = []
        for noun in (front, rear):
            boxes = seen.get(noun, {})
            hits = sum(1 for f in frames if any(_box_gap(b, lid[f]) <= near * _lid_size(lid[f]) for b in boxes.get(f, [])))
            shares.append(hits / len(frames) if frames else 0.0)
        diff = shares[0] - shares[1]
        numbers.update({f"{cue}_front_share": shares[0], f"{cue}_rear_share": shares[1]})
        votes[cue] = 0 if len(frames) < min_frames else 1 if diff >= margin else -1 if diff <= -margin else 0
    return votes, numbers


def twin_decision(votes: dict[str, int], nouns: Collection[Noun], cfg: dict[str, Any]) -> tuple[int | None, float]:
    """(+1 hood, -1 trunk lid or None when split, the weighted score) for one lid shown by tracks of nouns."""
    weights = cfg["weights"]
    score = float(sum(float(weights.get(cue, 0)) * vote for cue, vote in votes.items()))
    margin = float(cfg["margin_min"])
    if score >= margin:
        return 1, score
    if score <= -margin:
        return -1, score
    ends = {LID_END[n] for n in nouns if n in LID_END}
    if len(ends) == 1:
        end = ends.pop()
        if score == 0 or (score > 0) == (end > 0):
            return end, score
    return None, score


def lid_groups(lids: list[c.PanelTrack], iou_min: float) -> list[list[c.PanelTrack]]:
    """Hood and trunk-lid tracks grouped by the lid they show: two tracks whose boxes overlap (median box IoU >= iou_min
    on their shared frames) show the same one. Tracks of one lane that never share a frame stay apart: SAM's lanes
    keep an id from one lid to the other across a gap."""
    ordered = sorted(lids, key=lambda t: t.track_id)
    parent = list(range(len(ordered)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = [{r.frame: r.box for r in t.per_frame} for t in ordered]
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            shared = sorted(set(boxes[i]) & set(boxes[j]))
            if shared and statistics.median(box_iou(boxes[i][f], boxes[j][f]) for f in shared) >= iou_min:
                parent[root(j)] = root(i)
    groups: dict[int, list[c.PanelTrack]] = {}
    for i, t in enumerate(ordered):
        groups.setdefault(root(i), []).append(t)
    return [groups[k] for k in sorted(groups)]


def _seen_boxes(tracks: list[c.PanelTrack]) -> dict[Noun, dict[int, list[Box]]]:
    seen: dict[Noun, dict[int, list[Box]]] = {}
    for t in tracks:
        if t.noun in TWIN_OBSERVED:
            for r in t.per_frame:
                seen.setdefault(t.noun, {}).setdefault(r.frame, []).append(r.box)
    return seen


def twin_check(role_tracks: list[c.PanelTrack], th: dict[str, Any]) -> dict[str, tuple[PanelId | None, c.TwinCheck]]:
    """{track id: (the slot or None when split, its TwinCheck)} for every hood and trunk-lid track of one walk, and for
    the light and glass tracks the placed lids decide; empty when thresholds panels.twin_check is absent."""
    cfg = twin_settings(th)
    lids = [t for t in role_tracks if t.noun in LID_NOUNS and t.per_frame]
    if cfg is None or not lids:
        return {}
    seen = _seen_boxes(role_tracks)
    out: dict[str, tuple[PanelId | None, c.TwinCheck]] = {}
    for group in lid_groups(lids, float(th["panels"]["anchor_absorb_iou_min"])):
        lid: dict[int, Box] = {}
        for t in group:
            for r in t.per_frame:
                lid.setdefault(r.frame, r.box)
        votes, numbers = twin_cues(lid, seen, cfg, min_frames=int(cfg["min_frames"]))
        end, score = twin_decision(votes, [t.noun for t in group], cfg)
        record = c.TwinCheck(lid=[t.track_id for t in group], votes=votes, numbers=numbers, score=score, decided=end is not None)
        for t in group:
            out[t.track_id] = (END_LID[end] if end is not None else None, record)
    out.update(_ends_next_to_lids(role_tracks, out, cfg))
    return out


def _ends_next_to_lids(
    role_tracks: list[c.PanelTrack], lids: dict[str, tuple[PanelId | None, c.TwinCheck]], cfg: dict[str, Any]
) -> dict[str, tuple[PanelId | None, c.TwinCheck]]:
    """Each light and glass track next to a placed lid (box gap <= adjacent_lid_sizes lid sizes) on at least min_frames
    frames and on neighbour_share_min of its own frames takes the end of that lid when one end has neighbour_majority
    times the other's frames: headlights and the windshield sit at the hood, tail lights and the rear window at the
    trunk lid. A part is attached to its lid, so it is beside it whenever both show; a track beside a lid only now and
    then is not that lid's part (pair C's "tail light" lane also tracked an orange mud flap by the front wheel, beside
    the hood on 57 of its 152 frames). Without such evidence the lane noun stands and no record is kept; a light or
    glass track is never left side uncertain."""
    near = float(cfg["adjacent_lid_sizes"])
    majority = float(cfg.get("neighbour_majority", 2.0))
    share_min = float(cfg.get("neighbour_share_min", 0.5))
    by_frame: dict[int, list[tuple[Box, int, list[str]]]] = {}
    for t in role_tracks:
        found = lids.get(t.track_id)
        if t.noun in LID_NOUNS and found is not None and found[0] is not None:
            end = LID_END[PANEL_NOUN[found[0]]]
            for r in t.per_frame:
                by_frame.setdefault(r.frame, []).append((r.box, end, found[1].lid))
    out: dict[str, tuple[PanelId | None, c.TwinCheck]] = {}
    for t in role_tracks:
        slots = NEIGHBOUR_SLOTS.get(t.noun)
        if slots is None:
            continue
        frames = {1: 0, -1: 0}
        beside: list[str] = []
        for r in t.per_frame:
            for box, end, lid_ids in by_frame.get(r.frame, ()):
                if _box_gap(r.box, box) <= near * _lid_size(box):
                    frames[end] += 1
                    beside.extend(i for i in lid_ids if i not in beside)
                    break
        if frames[1] + frames[-1] < max(int(cfg["min_frames"]), share_min * len(t.per_frame)):
            continue
        end = 1 if frames[1] >= majority * frames[-1] else -1 if frames[-1] >= majority * frames[1] else 0
        if not end:
            continue
        numbers = {"front_lid_frames": float(frames[1]), "rear_lid_frames": float(frames[-1]), "frames": float(len(t.per_frame))}
        record = c.TwinCheck(lid=sorted(beside), votes={"next_to_lid": end}, numbers=numbers, score=float(frames[1] - frames[-1]), decided=True)
        out[t.track_id] = (slots[end], record)
    return out


def apply_twin_check(
    role_tracks: list[c.PanelTrack], th: dict[str, Any], *, result: dict[str, tuple[PanelId | None, c.TwinCheck]] | None = None
) -> None:
    """File every hood and trunk-lid track of one voted walk under the lid the twin check found, and every light and
    glass track it placed under its end, in place; a split lid keeps its lane's slot as a placeholder and carries
    twin_check.decided false (candidate_slots then lists both). result is twin_check(role_tracks, th) when the caller
    already has it."""
    result = twin_check(role_tracks, th) if result is None else result
    for t in role_tracks:
        if t.noun not in TWIN_NOUNS:
            continue
        found = result.get(t.track_id)
        t.twin_check = found[1] if found is not None else None  # a re-vote without the check drops an old record
        if found is None:
            continue
        slot, record = found
        t.panel_id = slot if slot is not None else SINGLE_SLOT[t.noun]
        if slot is not None and slot is not SINGLE_SLOT[t.noun]:
            log.info("%s: %s (%s lane) filed as %s by the twin check, score %+.0f", t.role, t.track_id, t.noun.value, slot.value, record.score)


# ----------------------------------------------------------------------------------------------------------------
# 5.2 the door check: which side and which door (11/1, thresholds panels.door_check)
# ----------------------------------------------------------------------------------------------------------------
#
# The side vote files doors by the sum of R1 to R4, and on pair C it filed the left doors seen from the rear corner at
# the end of the pickup walk as the right doors (the light and glass lanes there hold the other end's parts), so the
# right rear door's keyframe showed the left front door with its mirror. After the side vote and the twin check, each
# door track is judged from what the walk tracked, each cue +1, -1 or 0:
#   side      lid_side: a hood or trunk lid the twin check decided, on shared frames: the hood lies toward the front, the
#             trunk lid toward the rear, and the front shows image-right on the right side; mirror_side: a side mirror
#             outside the door box on frames where the door is not edge-on lies toward the front.
#   position  mirror: the side mirror sits at the front door's front edge and a door width or more from the rear door;
#             wheel: the rear door's lower edge runs over the rear wheel (a door clear of the wheels says nothing);
#             pair: of two same-side doors sharing frames, the one toward the front; walk: of two same-side doors, the
#             one the walk reaches first when it passes that side front to rear.
# A weighted sum reaching margin_min decides; under it the side vote's answer stands unless a cue points the other way,
# and then the side or the position is left undecided (a placeholder slot, side_uncertain).

DOOR_SIDE_CUES: tuple[str, ...] = ("lid_side", "mirror_side")
DOOR_POSITION_CUES: tuple[str, ...] = ("mirror", "wheel", "pair", "walk")


def door_settings(th: dict[str, Any] | None) -> dict[str, Any] | None:
    """thresholds panels.door_check, or None when the run's thresholds predate it (the check is then off)."""
    cfg = (th or {}).get("panels", {}).get("door_check")
    return cfg if isinstance(cfg, dict) else None


def _mirror_near(door: Box, mirrors: Sequence[Box], cfg: dict[str, Any]) -> tuple[float, int] | None:
    """(gap in door widths, +1 when the mirror lies image-right of the door centre) for the nearest mirror whose centre
    sits in the door's height band, or None."""
    w = max(_width(door), 1.0)
    band = float(cfg["mirror_band_door_heights"]) * _height(door)
    cx = _centroid(door)[0]
    best: tuple[float, int] | None = None
    for m in mirrors:
        mx, my = _centroid(m)
        if not float(door[1]) - band <= my <= float(door[3]):
            continue
        gap = max(float(door[0]) - mx, mx - float(door[2]), 0.0) / w
        if best is None or gap < best[0]:
            best = (gap, 1 if mx > cx else -1)
    return best


def _wheel_overlap(door: Box, wheel: Box) -> float:
    """Share of the wheel's width that lies under the door's horizontal span."""
    inter = max(0.0, min(float(door[2]), float(wheel[2])) - max(float(door[0]), float(wheel[0])))
    return inter / max(_width(wheel), 1.0)


def _majority(signs: list[int], share_min: float, min_frames: int) -> tuple[int, float]:
    """(+1 or -1 when at least share_min of the signs agree on at least min_frames frames, else 0; that share)."""
    if len(signs) < min_frames or not signs:
        return 0, 0.0
    plus = sum(1 for s in signs if s > 0) / len(signs)
    if plus >= share_min:
        return 1, plus
    if 1.0 - plus >= share_min:
        return -1, 1.0 - plus
    return 0, max(plus, 1.0 - plus)


def door_side_cues(
    door: dict[int, Box],
    lids: dict[int, list[tuple[Box, int]]],
    mirrors: dict[int, list[Box]],
    cfg: dict[str, Any],
    *,
    min_frames: int,
) -> tuple[dict[str, int], dict[str, float]]:
    """The two side votes (+1 right, -1 left, 0 silent) on one door given its box per frame, the decided lids per frame
    as (box, +1 hood or -1 trunk lid) and the mirror boxes per frame, plus the numbers behind them. The car's front shows
    image-right on its right side (front_sign +1), so the image direction of the front is the side."""
    share_min = float(cfg["dir_share_min"])
    lid_signs = [end * _sign(_centroid(box)[0] - _centroid(door[f])[0]) for f in sorted(door) for box, end in lids.get(f, [])]
    lid_signs = [s for s in lid_signs if s]
    votes: dict[str, int] = {}
    numbers: dict[str, float] = {}
    votes["lid_side"], share = _majority(lid_signs, share_min, min_frames)
    numbers.update(lid_frames=float(len(lid_signs)), lid_side_share=share)
    mirror_signs: list[int] = []
    for f in sorted(door):
        if _aspect(door[f]) < float(cfg["side_view_aspect_min"]):
            continue
        near = _mirror_near(door[f], mirrors.get(f, []), cfg)
        if near is not None and near[0] >= float(cfg["mirror_side_gap_min"]):
            mirror_signs.append(near[1])
    votes["mirror_side"], share = _majority(mirror_signs, share_min, min_frames)
    numbers.update(mirror_side_frames=float(len(mirror_signs)), mirror_side_share=share)
    return votes, numbers


def door_position_cues(
    door: dict[int, Box], mirrors: dict[int, list[Box]], wheels: dict[int, list[Box]], cfg: dict[str, Any], *, min_frames: int
) -> tuple[dict[str, int], dict[str, float]]:
    """The mirror and wheel votes (+1 front door, -1 rear door, 0 silent) on one door given its box per frame and the
    mirror and wheel boxes per frame, plus the numbers behind them (a claim photo is one frame, min_frames 1)."""
    votes: dict[str, int] = {}
    numbers: dict[str, float] = {}
    gaps = [near[0] for f in sorted(door) if (near := _mirror_near(door[f], mirrors.get(f, []), cfg)) is not None]
    numbers["mirror_frames"] = float(len(gaps))
    votes["mirror"] = 0
    if gaps and len(gaps) >= min_frames:
        gap = statistics.median(gaps)
        numbers["mirror_gap_door_widths"] = gap
        votes["mirror"] = 1 if gap <= float(cfg["mirror_near_door_widths"]) else -1 if gap >= float(cfg["mirror_far_door_widths"]) else 0
    lap_min = float(cfg["wheel_overlap_min"])
    with_wheel = [f for f in sorted(door) if wheels.get(f)]
    under = sum(1 for f in with_wheel if max(_wheel_overlap(door[f], w) for w in wheels[f]) >= lap_min)
    numbers.update(wheel_frames=float(len(with_wheel)), wheel_under_frames=float(under))
    # only ever rear: a door clear of the wheels may still be a rear door whose wheel is out of the frame
    rear = len(with_wheel) >= min_frames and under / len(with_wheel) >= float(cfg["wheel_rear_share_min"])
    votes["wheel"] = -1 if rear else 0
    return votes, numbers


def door_decision(votes: dict[str, int], cues: Collection[str], prior: int, weights: dict[str, Any], margin: float) -> tuple[int, float, bool]:
    """(+1, -1 or 0 for undecided; the weighted score; decided) for one door's side or position. A score reaching
    margin decides; under it the prior (the side vote's answer, 0 when it had none) stands unless a cue points the
    other way, which leaves it undecided."""
    score = float(sum(float(weights.get(k, 0)) * votes.get(k, 0) for k in cues))
    if score >= margin:
        return 1, score, True
    if score <= -margin:
        return -1, score, True
    against = any(votes.get(k, 0) and votes[k] != prior for k in cues)
    if prior and not against:
        return prior, score, True
    return 0, score, False


def _door_peers(ctx: _RoleContext, t: c.PanelTrack, doors: list[c.PanelTrack], sides: dict[str, int], cfg: dict[str, Any]) -> list[c.PanelTrack]:
    """The other doors filed on t's side that share at least min_frames frames with it and are not the same door
    (median box IoU on the shared frames under same_door_iou_min)."""
    boxes = {r.frame: r.box for r in t.per_frame}
    peers = []
    for o in doors:
        if o.track_id == t.track_id or not sides.get(t.track_id) or sides.get(o.track_id) != sides[t.track_id]:
            continue
        sh = ctx.shared(t, o)
        if len(sh) < int(cfg["min_frames"]):
            continue
        ob = {r.frame: r.box for r in o.per_frame}
        if statistics.median(box_iou(boxes[f], ob[f]) for f in sh) < float(cfg["same_door_iou_min"]):
            peers.append(o)
    return peers


def _pair_and_walk(ctx: _RoleContext, t: c.PanelTrack, peers: list[c.PanelTrack], front_sign: int) -> tuple[dict[str, int], dict[str, float]]:
    votes = {"pair": 0, "walk": 0}
    numbers: dict[str, float] = {"peers": float(len(peers))}
    if not peers:
        return votes, numbers
    if front_sign:
        rel = [ctx.mean_dx(t, o, ctx.shared(t, o)) * front_sign for o in peers]
        numbers["pair_toward_front_px"] = float(min(rel)) if all(r > 0 for r in rel) else float(max(rel))
        votes["pair"] = -1 if all(r > 0 for r in rel) else 1 if all(r < 0 for r in rel) else 0
    order = ctx.adjacent_order(t)
    first = ctx.window(t)[0]
    starts = [ctx.window(o)[0] for o in peers]
    numbers["walk_order"] = float(order)
    if order and all(first < s for s in starts):
        votes["walk"] = order
    elif order and all(first > s for s in starts):
        votes["walk"] = -order
    return votes, numbers


def _seen_by_frame(tracks: list[c.PanelTrack], noun: Noun) -> dict[int, list[Box]]:
    out: dict[int, list[Box]] = {}
    for t in tracks:
        if t.noun is noun and not is_wheel_fender(t):
            for r in t.per_frame:
                out.setdefault(r.frame, []).append(r.box)
    return out


def door_check(ctx: _RoleContext, role_tracks: list[c.PanelTrack], th: dict[str, Any]) -> dict[str, tuple[Side, Position, c.DoorCheck]]:
    """{track id: (side, position, its DoorCheck)} for every car door track of one voted walk, NONE where the cues left
    it undecided; empty when thresholds panels.door_check is absent. Reads the side vote's answer and the placed lids
    from the tracks (run it after the twin check is applied)."""
    cfg = door_settings(th)
    doors = [t for t in role_tracks if t.noun is Noun.CAR_DOOR and t.per_frame]
    if cfg is None or not doors:
        return {}
    min_frames = int(cfg["min_frames"])
    weights, margin = cfg["weights"], float(cfg["margin_min"])
    mirrors = _seen_by_frame(role_tracks, Noun.SIDE_MIRROR)
    wheels = _seen_by_frame(role_tracks, Noun.WHEEL)
    lids: dict[int, list[tuple[Box, int]]] = {}
    for t in role_tracks:
        placed = t.noun in LID_NOUNS and t.panel_id in (PanelId.HOOD, PanelId.TRUNK_LID) and (t.twin_check is None or t.twin_check.decided)
        if placed and t.panel_id is not None:
            for r in t.per_frame:
                lids.setdefault(r.frame, []).append((r.box, LID_END[PANEL_NOUN[t.panel_id]]))
    sides: dict[str, int] = {}
    found: dict[str, tuple[dict[str, int], dict[str, float], float, bool]] = {}
    for t in doors:
        box = {r.frame: r.box for r in t.per_frame}
        prior = 1 if t.side is Side.RIGHT else -1 if t.side is Side.LEFT else 0
        votes, numbers = door_side_cues(box, lids, mirrors, cfg, min_frames=min_frames)
        side, score, decided = door_decision(votes, DOOR_SIDE_CUES, prior, weights, margin)
        sides[t.track_id] = side
        found[t.track_id] = (votes, numbers, score, decided)
    out: dict[str, tuple[Side, Position, c.DoorCheck]] = {}
    for t in doors:
        box = {r.frame: r.box for r in t.per_frame}
        votes, numbers, side_score, side_decided = found[t.track_id]
        pv, pn = door_position_cues(box, mirrors, wheels, cfg, min_frames=min_frames)
        wv, wn = _pair_and_walk(ctx, t, _door_peers(ctx, t, doors, sides, cfg), sides[t.track_id])
        votes = {**votes, **pv, **wv}
        numbers = {**numbers, **pn, **wn}
        prior = 1 if t.position is Position.FRONT else -1 if t.position is Position.REAR else 0
        pos, pos_score, pos_decided = door_decision(votes, DOOR_POSITION_CUES, prior, weights, margin)
        side = Side.RIGHT if sides[t.track_id] > 0 else Side.LEFT if sides[t.track_id] < 0 else Side.NONE
        position = Position.FRONT if pos > 0 else Position.REAR if pos < 0 else Position.NONE
        record = c.DoorCheck(
            was=t.panel_id, votes=votes, numbers=numbers, side_score=side_score, position_score=pos_score,
            side_decided=side_decided and side is not Side.NONE, position_decided=pos_decided and position is not Position.NONE,
        )
        out[t.track_id] = (side, position, record)
    return out


def apply_door_check(ctx: _RoleContext, role_tracks: list[c.PanelTrack], th: dict[str, Any]) -> None:
    """File every car door track of one voted walk under the side and door the door check found, in place; an
    undecided side or position keeps a placeholder slot among its candidates (the one the side vote gave when it is
    one of them) and the panel is side_uncertain."""
    result = door_check(ctx, role_tracks, th)
    for t in role_tracks:
        if t.noun is not Noun.CAR_DOOR:
            continue
        got = result.get(t.track_id)
        t.door_check = got[2] if got is not None else None
        if got is None:
            continue
        side, position, record = got
        lean = t.side if t.side is not Side.NONE else Side.LEFT
        sides = [side] if side is not Side.NONE else [lean, Side.RIGHT if lean is Side.LEFT else Side.LEFT]
        positions = [position] if position is not Position.NONE else [Position.FRONT, Position.REAR]
        cands = [_lateral_panel_id(Noun.CAR_DOOR, s, q) for s in sides for q in positions]
        t.side, t.position = side, position
        t.panel_id = t.panel_id if t.panel_id in cands else cands[0]
        if t.panel_id != record.was:
            log.info("%s: %s filed as %s by the door check (was %s)", t.role, t.track_id, t.panel_id.value, record.was.value if record.was else None)


# ----------------------------------------------------------------------------------------------------------------
# 5.2 the part check: which side a wheel is on, which end a glass panel is at (lexi run 1, 11/2, panels.part_check)
# ----------------------------------------------------------------------------------------------------------------
#
# On lexi_pair1 both walks start and end at the front-left corner and the side vote filed the left front wheel (and so
# its built fender, and the fender view that held mark M3) as the right front in both walks, and the windshield lane
# tracks seen behind the car stayed the windshield (the return's windshield keyframes showed the rear window). After the
# door check, in its style, each wheel track (and fender-lane track) is judged for its side and each windshield or
# rear-window track for its end, from what the walk already tracked:
#   side  lid_side: a lid the twin check placed, on shared frames (the hood is toward the front, the trunk lid toward
#         the rear, the front shows image-right on the right side); light_side: a headlight or tail light the same way;
#         door_side: decided doors beside the wheel (the gap rule of fender_from_wheels).
#   end   lights, bumper, lid: a placed headlight or tail light, front or rear bumper, hood or trunk lid beside the glass
#         on clearly more of its frames than the other end's; mirror: a side mirror beside the glass marks the front
#         (never the rear: a rear window may have a mirror far off in the frame).
# door_decision decides: a weighted sum reaching margin_min; under it the slot the vote gave stands unless a cue points
# the other way, then the side or end is left undecided (a placeholder slot, side_uncertain).

PART_SIDE_CUES: tuple[str, ...] = ("lid_side", "light_side", "door_side")
PART_END_CUES: tuple[str, ...] = ("lights", "bumper", "lid", "mirror")
GLASS_NOUNS: tuple[Noun, ...] = (Noun.WINDSHIELD, Noun.REAR_WINDOW)
GLASS_END: dict[PanelId, int] = {PanelId.WINDSHIELD: 1, PanelId.REAR_WINDOW: -1}
LIGHT_END: dict[PanelId, int] = {PanelId.HEADLIGHTS: 1, PanelId.TAIL_LIGHTS: -1}


def part_settings(th: dict[str, Any] | None) -> dict[str, Any] | None:
    """thresholds panels.part_check, or None when the run's thresholds predate it (the check is then off)."""
    cfg = (th or {}).get("panels", {}).get("part_check")
    return cfg if isinstance(cfg, dict) else None


def _ends_by_frame(role_tracks: list[c.PanelTrack], ends: dict[PanelId, int]) -> dict[int, list[tuple[Box, int]]]:
    """(box, end) per frame for every track filed under one of ends' slots and not split by the twin check."""
    out: dict[int, list[tuple[Box, int]]] = {}
    for t in role_tracks:
        if t.panel_id in ends and (t.twin_check is None or t.twin_check.decided) and not is_wheel_fender(t):
            for r in t.per_frame:
                out.setdefault(r.frame, []).append((r.box, ends[t.panel_id]))
    return out


def _bumper_ends(role_tracks: list[c.PanelTrack]) -> dict[int, list[tuple[Box, int]]]:
    out: dict[int, list[tuple[Box, int]]] = {}
    for t in role_tracks:
        end = 1 if t.position is Position.FRONT else -1 if t.position is Position.REAR else 0
        if t.noun is Noun.BUMPER and end:
            for r in t.per_frame:
                out.setdefault(r.frame, []).append((r.box, end))
    return out


def part_side_cues(
    part: dict[int, Box],
    lids: dict[int, list[tuple[Box, int]]],
    lights: dict[int, list[tuple[Box, int]]],
    doors: list[tuple[dict[int, Box], int]],
    cfg: dict[str, Any],
    *,
    min_frames: int,
) -> tuple[dict[str, int], dict[str, float]]:
    """The three side votes (+1 right, -1 left, 0 silent) on one wheel or fender given its box per frame, the placed lids
    and lights per frame as (box, +1 front or -1 rear), and the decided doors as (box per frame, +1 right or -1 left). A
    lid or light votes on a frame only when its centre lies offset_min_d part sizes or more to one side."""
    share_min = float(cfg["dir_share_min"])
    offset_min = float(cfg["offset_min_d"])
    votes: dict[str, int] = {}
    numbers: dict[str, float] = {}
    for cue, ends in (("lid_side", lids), ("light_side", lights)):
        # a lid or light within offset_min_d part sizes of the part's centre sits over it (the hood over a front wheel):
        # that offset is depth, not direction, so only frames where it lies clearly to one side vote
        signs = []
        for f in sorted(part):
            d = max(_width(part[f]), _height(part[f]), 1.0)
            for box, end in ends.get(f, []):
                dx = _centroid(box)[0] - _centroid(part[f])[0]
                if abs(dx) >= offset_min * d:
                    signs.append(end * _sign(dx))
        votes[cue], share = _majority(signs, share_min, min_frames)
        numbers.update({f"{cue}_frames": float(len(signs)), f"{cue}_share": share})
    gap_d = float(cfg["door_gap_d"])
    door_signs: list[int] = []
    for f in sorted(part):
        x1, y1, x2, _y2, d, _cx, cy = _wheel_geom(part[f])
        for boxes, side in doors:
            b = boxes.get(f)
            if b is None:
                continue
            gap = max(float(b[0]) - x2, x1 - float(b[2]), 0.0)
            if gap <= gap_d * d and float(b[3]) >= y1 - 0.5 * d and float(b[1]) <= cy:
                door_signs.append(side)
    votes["door_side"], share = _majority(door_signs, share_min, min_frames)
    numbers.update(door_side_frames=float(len(door_signs)), door_side_share=share)
    return votes, numbers


def glass_end_cues(
    glass: dict[int, Box],
    near: dict[str, dict[int, list[tuple[Box, int]]]],
    mirrors: dict[int, list[Box]],
    cfg: dict[str, Any],
    *,
    min_frames: int,
) -> tuple[dict[str, int], dict[str, float]]:
    """The four end votes (+1 front, -1 rear, 0 silent) on one glass track given its box per frame, the placed lights,
    bumpers and lids per frame as (box, end), and the mirror boxes per frame. A part is beside the glass when the box gap
    is at most adjacent_sizes glass sizes; a cue votes when one end is beside it on presence_margin more of the glass
    frames than the other end."""
    frames = sorted(glass)
    adj = float(cfg["adjacent_sizes"])
    margin = float(cfg["presence_margin"])
    votes: dict[str, int] = {}
    numbers: dict[str, float] = {"glass_frames": float(len(frames))}
    for cue in ("lights", "bumper", "lid"):
        hits = {1: 0, -1: 0}
        for f in frames:
            size = _lid_size(glass[f])
            seen = {end for box, end in near[cue].get(f, []) if _box_gap(box, glass[f]) <= adj * size}
            for end in seen:
                hits[end] += 1
        front, rear = (hits[1] / len(frames), hits[-1] / len(frames)) if frames else (0.0, 0.0)
        numbers.update({f"{cue}_front_share": front, f"{cue}_rear_share": rear})
        enough = hits[1] + hits[-1] >= min_frames
        votes[cue] = 0 if not enough else 1 if front - rear >= margin else -1 if rear - front >= margin else 0
    beside = sum(1 for f in frames if any(_box_gap(m, glass[f]) <= float(cfg["mirror_near_sizes"]) * _lid_size(glass[f]) for m in mirrors.get(f, [])))
    share = beside / len(frames) if frames else 0.0
    numbers.update(mirror_frames=float(beside), mirror_share=share)
    votes["mirror"] = 1 if beside >= min_frames and share >= margin else 0
    return votes, numbers


def part_check(role_tracks: list[c.PanelTrack], th: dict[str, Any]) -> dict[str, tuple[PanelId, c.PartCheck]]:
    """{track id: (its slot, its PartCheck)} for every wheel, fender-lane and glass track of one voted walk; a slot whose
    side or end stayed undecided is the placeholder (the one the vote gave). Empty when thresholds panels.part_check is
    absent. Run it after the twin check and the door check are applied: it reads the placed lids, lights and doors."""
    cfg = part_settings(th)
    if cfg is None:
        return {}
    min_frames = int(cfg["min_frames"])
    weights, margin = cfg["weights"], float(cfg["margin_min"])
    lids = _ends_by_frame(role_tracks, {PanelId.HOOD: 1, PanelId.TRUNK_LID: -1})
    lights = _ends_by_frame(role_tracks, LIGHT_END)
    doors = [({r.frame: r.box for r in t.per_frame}, 1 if t.side is Side.RIGHT else -1) for t in role_tracks if t.noun is Noun.CAR_DOOR and t.side is not Side.NONE]
    near = {"lights": lights, "bumper": _bumper_ends(role_tracks), "lid": lids}
    mirrors = _seen_by_frame(role_tracks, Noun.SIDE_MIRROR)
    out: dict[str, tuple[PanelId, c.PartCheck]] = {}
    for t in role_tracks:
        if not t.per_frame or t.panel_id is None:
            continue
        box = {r.frame: r.box for r in t.per_frame}
        if t.noun in (Noun.WHEEL, Noun.FENDER) and not is_wheel_fender(t):
            votes, numbers = part_side_cues(box, lids, lights, doors, cfg, min_frames=min_frames)
            prior = 1 if t.side is Side.RIGHT else -1 if t.side is Side.LEFT else 0
            side, score, decided = door_decision(votes, PART_SIDE_CUES, prior, weights, margin)
            got = Side.RIGHT if side > 0 else Side.LEFT if side < 0 else Side.NONE
            if got is Side.NONE:
                slot = t.panel_id
            elif t.position is Position.NONE:
                lean = _panel_side_position(t.panel_id)[1]
                slot = _lateral_panel_id(t.noun, got, lean if lean is not Position.NONE else Position.FRONT)
            else:
                slot = _lateral_panel_id(t.noun, got, t.position)
            out[t.track_id] = (slot, c.PartCheck(kind="side", was=t.panel_id, votes=votes, numbers=numbers, score=score, decided=decided and side != 0))
        elif t.noun in GLASS_NOUNS and t.panel_id in GLASS_END:
            votes, numbers = glass_end_cues(box, near, mirrors, cfg, min_frames=min_frames)
            end, score, decided = door_decision(votes, PART_END_CUES, GLASS_END[t.panel_id], weights, margin)
            slot = {1: PanelId.WINDSHIELD, -1: PanelId.REAR_WINDOW}.get(end, t.panel_id)
            out[t.track_id] = (slot, c.PartCheck(kind="end", was=t.panel_id, votes=votes, numbers=numbers, score=score, decided=decided and end != 0))
    return out


def apply_part_check(role_tracks: list[c.PanelTrack], th: dict[str, Any]) -> None:
    """File every wheel, fender-lane and glass track of one voted walk under the slot the part check found, in place. A
    wheel or fender whose side stayed undecided keeps its slot as a placeholder with side NONE (candidate_slots lists
    both sides); an undecided glass track keeps its slot and carries part_check.decided false (candidate_slots lists
    the windshield and the rear window)."""
    result = part_check(role_tracks, th)
    for t in role_tracks:
        if not (t.noun in (Noun.WHEEL, Noun.FENDER, *GLASS_NOUNS)) or is_wheel_fender(t):
            continue
        got = result.get(t.track_id)
        t.part_check = got[1] if got is not None else None
        if got is None:
            continue
        slot, record = got
        if t.noun in LATERAL_NOUNS:
            side, position = _panel_side_position(slot)
            t.side = side if record.decided else Side.NONE
        t.panel_id = slot
        if slot != record.was:
            log.info("%s: %s filed as %s by the part check (was %s)", t.role, t.track_id, slot.value, record.was.value if record.was else None)


# ----------------------------------------------------------------------------------------------------------------
# Fenders and rear quarters from the wheels (lead 11/1, thresholds panels.fender_from_wheels)
# ----------------------------------------------------------------------------------------------------------------
#
# SAM 3.1 has no noun for the panel over a wheel: on stored frames "fender", "car fender", "front fender", "quarter
# panel" and "rear quarter panel" came back empty on the synthetic sedan, and on pair C "fender" boxed only the arch lip
# (0.3% to 0.9% of the frame), so no fender was ever filmed. With the leaf on, the fender lane's own tracks are left out
# and every wheel track gives one fender track after the side vote: its slot from the wheel's side and position (the
# side from the decided doors beside the wheel when the wheel's own is undecided), its boxes from the wheel's, and at
# each keyframe candidate its mask is what is left of the car mask near the wheel once every part SAM did name is cut
# away (the connected piece over the wheel arch). Built tracks never take part in the side vote.

FENDER_CUT_NOUNS: tuple[Noun, ...] = (
    Noun.CAR_DOOR, Noun.BUMPER, Noun.WHEEL, Noun.HOOD, Noun.TRUNK_LID, Noun.ROOF, Noun.HEADLIGHT, Noun.TAIL_LIGHT,
    Noun.SIDE_MIRROR, Noun.WINDSHIELD, Noun.REAR_WINDOW, Noun.CAR_WINDOW, Noun.LICENSE_PLATE,
)
_FROM_WHEEL = "_fender_from_wheel_"


def fender_settings(th: dict[str, Any] | None) -> dict[str, Any] | None:
    """thresholds panels.fender_from_wheels, or None when the run's thresholds predate it (fenders then come from the
    fender lane as before)."""
    cfg = (th or {}).get("panels", {}).get("fender_from_wheels")
    return cfg if isinstance(cfg, dict) else None


def is_wheel_fender(track: c.PanelTrack) -> bool:
    """A fender track built from a wheel track (its id is <role>_fender_from_wheel_<n>)."""
    return track.noun is Noun.FENDER and _FROM_WHEEL in track.track_id


def wheel_id_of(track: c.PanelTrack) -> str:
    """The wheel track a built fender track came from."""
    return track.track_id.replace(_FROM_WHEEL, "_wheel_", 1)


def _wheel_geom(box: Box) -> tuple[float, float, float, float, float, float, float]:
    """x1, y1, x2, y2, the wheel's size d (the longer box side, as the bottom is often cut), centre x and y."""
    x1, y1, x2, y2 = (float(v) for v in box)
    d = max(x2 - x1, y2 - y1, 1.0)
    return x1, y1, x2, y2, d, (x1 + x2) / 2.0, (y1 + y2) / 2.0


def fender_box(wheel_box: Box, car_box: Box | None, media_size: tuple[int, int], cfg: dict[str, Any]) -> list[int] | None:
    """The box pass box of the fender over a wheel: the wheel widened by box_side_d x d each way, from box_above_d x d
    over its top down to its centre, inside the car box and the frame."""
    x1, y1, x2, y2, d, _cx, cy = _wheel_geom(wheel_box)
    side, above = float(cfg["box_side_d"]), float(cfg["box_above_d"])
    b = [x1 - side * d, y1 - above * d, x2 + side * d, cy]
    if car_box is not None:
        b = [max(b[0], float(car_box[0])), max(b[1], float(car_box[1])), min(b[2], float(car_box[2])), min(b[3], float(car_box[3]))]
    w_img, h_img = media_size
    out = [int(round(max(0.0, b[0]))), int(round(max(0.0, b[1]))), int(round(min(float(w_img), b[2]))), int(round(min(float(h_img), b[3])))]
    return out if out[2] > out[0] and out[3] > out[1] else None


def _door_side(wheel: c.PanelTrack, doors: list[c.PanelTrack], cfg: dict[str, Any]) -> Side:
    """The side the decided doors beside a wheel vote for: one vote per shared frame where a door box is within
    door_gap_d x d of the wheel box and reaches down past the wheel's top; NONE under min_frames or on a tie."""
    rows = {r.frame: r for r in wheel.per_frame if r.in_car_box}
    gap_d = float(cfg["door_gap_d"])
    votes = {Side.LEFT: 0, Side.RIGHT: 0}
    for door in doors:
        for r in door.per_frame:
            w = rows.get(r.frame)
            if w is None or door.side not in votes:
                continue
            x1, y1, x2, _y2, d, _cx, cy = _wheel_geom(w.box)
            gap = max(float(r.box[0]) - x2, x1 - float(r.box[2]), 0.0)
            if gap <= gap_d * d and float(r.box[3]) >= y1 - 0.5 * d and float(r.box[1]) <= cy:
                votes[door.side] += 1
    left, right = votes[Side.LEFT], votes[Side.RIGHT]
    if max(left, right) < int(cfg["min_frames"]) or left == right:
        return Side.NONE
    return Side.LEFT if left > right else Side.RIGHT


def fender_tracks(
    role: str, tracks: list[c.PanelTrack], car_boxes: dict[int, list[int]], media_size: tuple[int, int], cfg: dict[str, Any]
) -> list[c.PanelTrack]:
    """One fender track per voted wheel track of the role whose slot is known: side and position from the wheel's
    candidate slots when they agree, else the side from the doors beside it; boxes from fender_box. A wheel whose
    side or position stays unknown, or with fewer than min_frames rows, gives none."""
    role_tracks = [t for t in tracks if t.role == role]
    doors = [t for t in role_tracks if t.noun is Noun.CAR_DOOR and not undecided(t) and t.side is not Side.NONE]
    w_img, h_img = media_size
    out: list[c.PanelTrack] = []
    for wheel in sorted((t for t in role_tracks if t.noun is Noun.WHEEL), key=lambda t: t.track_id):
        slots = candidate_slots(wheel)
        if not slots:
            continue
        sides = {_panel_side_position(s)[0] for s in slots}
        positions = {_panel_side_position(s)[1] for s in slots}
        side = next(iter(sides)) if len(sides) == 1 else _door_side(wheel, doors, cfg)
        position = next(iter(positions)) if len(positions) == 1 else Position.NONE
        if side is Side.NONE or position is Position.NONE:
            continue
        rows: list[c.FrameObs] = []
        for r in wheel.per_frame:
            if not r.in_car_box:
                continue
            box = fender_box(r.box, car_boxes.get(r.frame), media_size, cfg)
            if box is None:
                continue
            rows.append(
                c.FrameObs(
                    frame=r.frame, pts=r.pts, box=box, box_area_frac=round(_area(box) / float(w_img * h_img), 6), in_car_box=True,
                    border_frac=round(border_frac(box, media_size), 6),
                )
            )
        if len(rows) < int(cfg["min_frames"]):
            continue
        pid = _lateral_panel_id(Noun.FENDER, side, position)
        out.append(
            c.PanelTrack(
                track_id=wheel.track_id.replace("_wheel_", _FROM_WHEEL, 1), role=role, noun=Noun.FENDER, sam_object_ids=[],
                fragments=[], per_frame=rows, panel_id=pid, side=side, position=position,
            )
        )
    return out


def apply_wheel_fenders(
    role: str,
    panels: list[c.Panel],
    tracks: list[c.PanelTrack],
    car_boxes: dict[int, list[int]],
    media_size: tuple[int, int],
    fps: float,
    th: dict[str, Any],
) -> tuple[list[c.Panel], list[c.PanelTrack]]:
    """After the vote: the role's fender lane tracks out, its wheel fenders in, and the four fender panels rebuilt from
    every role's fender tracks (votes zero: a built track casts none)."""
    cfg = fender_settings(th)
    if cfg is None:
        return panels, tracks
    kept = [t for t in tracks if not (t.role == role and t.noun is Noun.FENDER)]
    out = _sorted_tracks(kept + fender_tracks(role, kept, car_boxes, media_size, cfg))
    rebuilt = _panels_from(
        [t for t in out if t.noun is Noun.FENDER], {}, fps=fps, merge_gap_s=float(th["panels"]["merge_gap_max_s"]),
        fallback_dt=1.0 / float(th["media"]["proxy_fps"]), per_track=uncertain_per_track(th),
    )
    by_pid = {p.panel_id: p for p in rebuilt if p.noun is Noun.FENDER}
    return [by_pid.get(p.panel_id, p) if p.noun is Noun.FENDER else p for p in panels], out


def wheel_fender_mask(
    wheel_box: Box, car_mask: np.ndarray, cut_masks: list[np.ndarray], cfg: dict[str, Any]
) -> np.ndarray | None:
    """The fender over a wheel on one proxy frame: the car mask inside a region around the wheel (reach_d x d either
    side of its centre from above_d x d over its top down to its centre, below that only below_side_d x d beyond its
    sides, so the sill under the doors stays out), less every named part, opened by open_px, then the connected piece
    with the most pixels just over the wheel's top. None when nothing is there."""
    import cv2

    h, w = car_mask.shape[:2]
    x1, y1, x2, _y2, d, cx, cy = _wheel_geom(wheel_box)
    reach, above, below_side = float(cfg["reach_d"]), float(cfg["above_d"]), float(cfg["below_side_d"])
    region = np.zeros((h, w), dtype=bool)
    top, mid = max(0, int(round(y1 - above * d))), min(h, max(0, int(round(cy))))
    region[top:mid, max(0, int(round(cx - reach * d))) : min(w, int(round(cx + reach * d)))] = True
    region[mid:, max(0, int(round(x1 - below_side * d))) : min(w, int(round(x2 + below_side * d)))] = True
    rest = np.asarray(car_mask, dtype=bool) & region
    for m in cut_masks:
        rest &= ~np.asarray(m, dtype=bool)
    k = int(cfg["open_px"])
    if k > 1:
        rest = cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0
    n, labels, _stats, _c = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    zone = labels[max(0, int(round(y1 - 0.4 * d))) : max(0, int(round(y1 + 0.1 * d))) + 1, max(0, int(round(x1))) : min(w, int(round(x2)) + 1)]
    counts = np.bincount(zone.ravel(), minlength=n)
    counts[0] = 0
    if int(counts.max()) == 0:
        return None
    return labels == int(np.argmax(counts))


# ----------------------------------------------------------------------------------------------------------------
# 5.3 keyframes
# ----------------------------------------------------------------------------------------------------------------


def door_glass_edge(th: dict[str, Any] | None) -> bool:
    """thresholds panels.door_glass_edge (lead 11/1, 2 m laps): a door box that reaches the top of the frame is cut
    there by its window glass, which SAM's door mask includes, so the box pass leaves the top edge out of a door's
    border share; the mask pass then measures completeness after the car windows are cut out. Off when absent."""
    return bool((th or {}).get("panels", {}).get("door_glass_edge", False))


def pass_border_frac(track: c.PanelTrack, row: c.FrameObs, th: dict[str, Any] | None = None) -> float:
    """The border share the box pass uses: border_frac, less the top edge for a door under panels.door_glass_edge."""
    if track.noun is not Noun.CAR_DOOR or not door_glass_edge(th) or float(row.box[1]) > 0:
        return row.border_frac
    w, h = max(0.0, float(row.box[2]) - float(row.box[0])), max(0.0, float(row.box[3]) - float(row.box[1]))
    return max(0.0, row.border_frac - (w / (2.0 * (w + h)) if w + h > 0 else 0.0))


def box_pass_score(track: c.PanelTrack, frame: int, th: dict[str, Any] | None = None) -> float:
    """box area x (1 - border_frac) x aspect over the track's max aspect, for one frame of a track."""
    rows = {r.frame: r for r in track.per_frame}
    row = rows.get(frame)
    if row is None:
        return 0.0
    max_aspect = max((_aspect(r.box) for r in track.per_frame), default=0.0) or 1.0
    return row.box_area_frac * (1.0 - pass_border_frac(track, row, th)) * (_aspect(row.box) / max_aspect)


def keyframe_candidates(track: c.PanelTrack, th: dict[str, Any]) -> list[int]:
    """The box pass: top keyframe_box_pass_candidates frames by box_pass_score; border_frac over the max excluded."""
    p = th["panels"]
    limit = 1.0 if is_wheel_fender(track) else float(p["border_frac_keyframe_max"])  # a built fender's mask pass judges its border
    rows = [r for r in track.per_frame if r.in_car_box and pass_border_frac(track, r, th) <= limit]
    if not rows:
        return []
    max_aspect = max((_aspect(r.box) for r in track.per_frame), default=0.0) or 1.0
    ranked = sorted(
        rows,
        key=lambda r: (-(r.box_area_frac * (1.0 - pass_border_frac(track, r, th)) * (_aspect(r.box) / max_aspect)), r.frame),
    )
    return [r.frame for r in ranked[: int(p["keyframe_box_pass_candidates"])]]


def _fit_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a bool mask to (H, W) when the sizes differ."""
    m = np.asarray(mask).astype(bool)
    if m.shape[:2] == tuple(shape):
        return m
    import cv2

    resized = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def mask_border_frac(mask: np.ndarray) -> float:
    """Share of the mask's boundary pixels that lie on the frame edge."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return 0.0
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    boundary = m & ~inner
    edge = np.zeros_like(m)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    n_boundary = int(boundary.sum())
    return float((boundary & edge).sum() / n_boundary) if n_boundary else 0.0


def _mask_aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0
    return float(xs.max() - xs.min() + 1) / float(ys.max() - ys.min() + 1)


def score_keyframe(
    track: c.PanelTrack,
    f_proxy: int,
    *,
    mask: np.ndarray,
    car_mask: np.ndarray | None,
    window_mask: np.ndarray | None,
    frame_rgb: np.ndarray,
    exclude_masks: list[np.ndarray],
    th: dict[str, Any],
) -> tuple[c.KeyframeScore, np.ndarray, bool]:
    """The mask pass on one candidate: raw factors (sharpness and frontality are normalised to the candidates' max by
    normalise_scores), the panel mask after window subtraction, and window_subtracted. A mask failing the car-mask
    containment test scores area 0 so it drops out."""
    import cv2

    p = th["panels"]
    shape = (int(frame_rgb.shape[0]), int(frame_rgb.shape[1]))
    m = _fit_mask(mask, shape)
    window_subtracted = False
    if window_mask is not None:
        wm = _fit_mask(window_mask, shape)
        if (m & wm).any():
            m = m & ~wm
            window_subtracted = True
    total_px = int(m.sum())
    if total_px == 0:
        return c.KeyframeScore(area=0.0, completeness=0.0, sharpness=0.0, glare=0.0, frontality=0.0, total=0.0), m, window_subtracted
    contained = True
    if car_mask is not None:
        cm = _fit_mask(car_mask, shape)
        contained = (m & cm).sum() / total_px >= float(p["car_mask_containment_min"])
    area = total_px / float(shape[0] * shape[1]) if contained else 0.0
    completeness = 1.0 - mask_border_frac(m)
    gray = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    valid = m.copy()
    for ex in exclude_masks:
        valid &= ~_fit_mask(ex, shape)
    sharpness = float(lap[valid].var()) if valid.any() else 0.0
    hsv = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2HSV)
    glare_px = (hsv[..., 2] > int(p["glare_v_min"])) & (hsv[..., 1] < int(p["glare_s_max"])) & m
    glare = float(glare_px.sum() / total_px)
    frontality = _mask_aspect(m)
    total = area * completeness * sharpness * (1.0 - glare) * frontality
    score = c.KeyframeScore(area=area, completeness=completeness, sharpness=sharpness, glare=glare, frontality=frontality, total=total)
    return score, m, window_subtracted


def normalise_scores(scored: list[tuple[int, c.KeyframeScore]], th: dict[str, Any]) -> list[tuple[int, c.KeyframeScore]]:
    """Sharpness and frontality over the candidates' max; total = the product with keyframe_score_exponents."""
    if not scored:
        return []
    e = th["panels"]["keyframe_score_exponents"]
    max_sharp = max(s.sharpness for _, s in scored) or 1.0
    max_front = max(s.frontality for _, s in scored) or 1.0
    out = []
    for f, s in scored:
        sharp = s.sharpness / max_sharp
        front = s.frontality / max_front
        total = (
            s.area ** float(e["area"])
            * s.completeness ** float(e["completeness"])
            * sharp ** float(e["sharpness"])
            * max(0.0, 1.0 - s.glare) ** float(e["one_minus_glare"])
            * front ** float(e["frontality"])
        )
        out.append((f, c.KeyframeScore(area=s.area, completeness=s.completeness, sharpness=sharp, glare=s.glare, frontality=front, total=total)))
    return out


def select_keyframes(scored: list[tuple[int, c.KeyframeScore]], *, fps: float, th: dict[str, Any]) -> list[int]:
    """Top keyframe_top_k by total, at least keyframe_min_gap_s apart, best first; zero totals never qualify."""
    p = th["panels"]
    min_gap = float(p["keyframe_min_gap_s"]) * fps
    chosen: list[int] = []
    for f, s in sorted(scored, key=lambda fs: (-fs[1].total, fs[0])):
        if s.total <= 0:
            continue
        if all(abs(f - g) >= min_gap for g in chosen):
            chosen.append(f)
        if len(chosen) >= int(p["keyframe_top_k"]):
            break
    return chosen


def edge_on_floor(th: dict[str, Any] | None, noun: Noun) -> float | None:
    """thresholds panels.edge_on_floor for one noun: the narrowest panel mask (width over height, after the window cut)
    a keyframe may show; None when the leaf is absent (runs frozen before 11/1) or the noun has no floor."""
    floors = (th or {}).get("panels", {}).get("edge_on_floor")
    value = floors.get(noun.value) if isinstance(floors, dict) else None
    return float(value) if value is not None else None


def edge_on_candidates(scored: list[tuple[int, c.KeyframeScore]], track_of: dict[int, c.PanelTrack], th: dict[str, Any]) -> set[int]:
    """The scored candidates seen near edge-on: raw frontality (the mask's width over height, before normalise_scores)
    under the floor of the candidate track's noun."""
    out: set[int] = set()
    for f, s in scored:
        floor = edge_on_floor(th, track_of[f].noun)
        if floor is not None and s.frontality < floor:
            out.add(f)
    return out


# ----------------------------------------------------------------------------------------------------------------
# 5.4 coverage and the top-up plan
# ----------------------------------------------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _is_filmed(kf: c.Keyframe, cov: dict[str, Any]) -> bool:
    s = kf.score
    return (
        s.completeness >= float(cov["filmed_completeness_min"])
        and s.area >= float(cov["filmed_area_frac_min"])
        and s.sharpness >= float(cov["filmed_sharpness_rel_min"])
    )


def coverage_for_role(
    panels: list[c.Panel],
    tracks: list[c.PanelTrack],
    keyframes: list[c.Keyframe],
    *,
    role: str,
    fps: float,
    lane_status: dict[Noun, LaneStatus],
    th: dict[str, Any],
) -> dict[PanelId, c.PanelCoverage]:
    """The 5.4 states for one role. filmed needs a qualifying keyframe. A slot with a decided track and no such
    keyframe is glimpsed (state_basis area_only when c is missing on most frames); a slot an undecided track may be is
    glimpsed with a side-uncertain reason. A slot with a track never prints not_filmed: 5.4 keeps that for no track
    after 5.1 on a completed or zero-match lane. not_run is a failed, incomplete or missing lane, or any slot without a
    track when the car lane did not complete (the car filter did not cover every frame). not_seen is a slot with no
    decided track in any walk while the noun was seen elsewhere in this walk."""
    cov = th["coverage"]
    role_tracks = [t for t in tracks if t.role == role]
    c_by_noun: dict[Noun, list[float]] = {}
    for t in tracks:
        for r in t.per_frame:
            if r.c is not None:
                c_by_noun.setdefault(t.noun, []).append(r.c)
    pct = {n: _percentile(v, float(cov["glimpsed_c_percentile"])) for n, v in c_by_noun.items()}
    car_status = lane_status.get(Noun.CAR)
    car_ok = car_status in (LaneStatus.COMPLETED, LaneStatus.ZERO_MATCHES)
    unsure = {t.track_id for t in role_tracks if undecided(t)}
    sure_elsewhere = {t.panel_id for t in tracks if t.role != role and t.panel_id is not None and not undecided(t)}
    per_track = uncertain_per_track(th)
    out: dict[PanelId, c.PanelCoverage] = {}
    for panel in panels:
        pid, noun = panel.panel_id, panel.noun
        mine = [t for t in role_tracks if t.panel_id == pid and t.track_id not in unsure]
        maybe = [t for t in role_tracks if t.track_id in unsure and pid in candidate_slots(t)]
        status = lane_status.get(noun)
        if mine:
            kfs = [k for k in keyframes if k.role == role and k.panel_id == pid and k.track_id not in unsure]
            out[pid] = _tracked_coverage(mine, kfs, noun=noun, pct=pct, fps=fps, cov=cov)
            if maybe and per_track:
                out[pid].reason += "; " + _stray_words(maybe)
            continue
        if maybe:
            maybe_ids = {t.track_id for t in maybe}
            out[pid] = _uncertain_coverage(maybe, [k for k in keyframes if k.role == role and k.track_id in maybe_ids], fps=fps)
            continue
        if not car_ok:
            state = CoverageState.NOT_RUN
            reason = f"car lane {car_status.value if car_status else 'not run'}: the car filter did not cover every frame, so no track here says nothing"
        elif status is None or status in (LaneStatus.FAILED, LaneStatus.INCOMPLETE):
            state = CoverageState.NOT_RUN
            reason = f"{noun.value} lane {status.value if status else 'not run'}"
        elif pid in sure_elsewhere or not any(t.noun == noun for t in role_tracks):
            state = CoverageState.NOT_FILMED
            reason = f"{noun.value} lane {status.value}, no track for {pid.value} in this walk"
        else:
            state = CoverageState.NOT_SEEN
            reason = f"{noun.value} seen in this walk, no track for the {pid.value} slot in any walk"
        out[pid] = c.PanelCoverage(state=state, reason=reason)
    return out


def _seconds_in_car(tracks: list[c.PanelTrack], fps: float) -> float:
    return len({r.frame for t in tracks for r in t.per_frame if r.in_car_box}) / fps if fps else 0.0


def _tracked_coverage(
    mine: list[c.PanelTrack],
    kfs: list[c.Keyframe],
    *,
    noun: Noun,
    pct: dict[Noun, float],
    fps: float,
    cov: dict[str, Any],
) -> c.PanelCoverage:
    """filmed by a qualifying keyframe, else glimpsed; the c test only words the reason (c is never printed)."""
    seconds = _seconds_in_car(mine, fps)
    best_kf = max(kfs, key=lambda k: (k.score.total, k.kf_id), default=None)
    best_id = best_kf.kf_id if best_kf else None
    filmed = [k for k in kfs if _is_filmed(k, cov)]
    if filmed:
        k = max(filmed, key=lambda k: (k.score.total, k.kf_id))
        reason = f"keyframe {k.kf_id}: completeness {k.score.completeness:.2f}, area {k.score.area:.3f}, sharpness {k.score.sharpness:.2f}"
        return c.PanelCoverage(state=CoverageState.FILMED, seconds=seconds, best_kf=best_id, reason=reason)
    if not kfs and _only_edge_on(mine):
        reason = (
            f"only seen edge-on: {seconds:.1f} s visible, and every candidate view of it was too narrow (under the "
            "edge-on floor), so no keyframe was kept; film it from the side"
        )
        return c.PanelCoverage(state=CoverageState.GLIMPSED, seconds=seconds, best_kf=None, reason=reason)
    lead = f"{seconds:.1f} s visible, no keyframe whole, large and sharp enough to count as filmed"
    best_track = max(mine, key=lambda t: (len(t.per_frame), t.track_id))
    if best_track.c_present_share >= float(cov["c_present_floor"]):
        present = [r.c for r in best_track.per_frame if r.c is not None]
        above = bool(present) and statistics.median(present) >= pct.get(noun, 0.0)
        # TODO(D1, E3): c is one detector score per text-prompt track (API notes 10/26), so this quartile only words the
        # reason; E3 decides whether c should gate any state.
        reason = f"{lead}; detector score {'at or above' if above else 'under'} this run's lower quartile for {noun.value}"
        basis = StateBasis.C
    else:
        floor = float(cov["area_only_box_area_frac_min"])
        reaches = any(r.box_area_frac >= floor for t in mine for r in t.per_frame if r.in_car_box)
        reason = f"{lead}; detector score missing on most frames, area-only basis: box area {'reaches' if reaches else 'stays under'} {floor:.0%} of the frame"
        basis = StateBasis.AREA_ONLY
    return c.PanelCoverage(state=CoverageState.GLIMPSED, state_basis=basis, seconds=seconds, best_kf=best_id, reason=reason)


def _only_edge_on(tracks: list[c.PanelTrack]) -> bool:
    """True when the tracks' keyframe candidates were all under panels.edge_on_floor (every decoded row is one)."""
    edge = {(t.track_id, f) for t in tracks for f in t.edge_on_frames}
    decoded = {(t.track_id, r.frame) for t in tracks for r in t.per_frame if r.mask_decoded}
    return bool(edge) and decoded <= edge


def _uncertain_coverage(maybe: list[c.PanelTrack], kfs: list[c.Keyframe], *, fps: float) -> c.PanelCoverage:
    """glimpsed for a slot an undecided track may be, naming the track, its candidates and where its frames are filed."""
    seconds = _seconds_in_car(maybe, fps)
    best_kf = max(kfs, key=lambda k: (k.score.total, k.kf_id), default=None)
    slots: list[PanelId] = []
    for t in maybe:
        slots.extend(s for s in candidate_slots(t) if s not in slots)
    filed = sorted({t.panel_id for t in maybe if t.panel_id is not None}, key=lambda pid: pid.value)
    # a door the door check placed on a side but not front or rear says so (runs before it keep their words)
    ends_only = all(t.noun in LID_NOUNS or t.noun in GLASS_NOUNS or (t.door_check is not None and t.side is not Side.NONE) for t in maybe)
    lead = "front or rear uncertain" if ends_only else "side uncertain"
    reason = (
        f"{lead}: {', '.join(t.track_id for t in maybe)} could be {' or '.join(display_name(s) for s in slots)}; "
        f"{seconds:.1f} s visible, frames filed under {', '.join(display_name(pid) for pid in filed)}"
    )
    return c.PanelCoverage(state=CoverageState.GLIMPSED, seconds=seconds, best_kf=best_kf.kf_id if best_kf else None, reason=reason)


def _stray_words(maybe: list[c.PanelTrack]) -> str:
    """The undecided tracks that could also be a slot decided tracks already hold (thresholds panels.uncertain_per_track):
    named, never used for this panel's keyframes."""
    words = []
    for t in sorted(maybe, key=lambda t: t.track_id):
        words.append(f"{t.track_id} could be {' or '.join(display_name(s) for s in candidate_slots(t))}")
    return f"also {'; '.join(words)} (not used here: the decided tracks name this panel)"


def display_name(pid: PanelId) -> str:
    return pid.value.replace("_", " ")


def coverage_roles(roles: Collection[str]) -> list[str]:
    """The roles a top-up plan reads, in order: the latest walk (return over pickup) and every top-up."""
    ordered = sorted(roles, key=_role_order)
    walks = [r for r in ordered if r in ("pickup", "return")]
    topups = [r for r in ordered if r.startswith("topup_")]
    return (walks[-1:] + topups) or ordered[-1:]


def _shown_state(per_role: dict[str, dict[PanelId, c.PanelCoverage]], roles: list[str], pid: PanelId) -> c.PanelCoverage:
    """What a missing slot prints: its state in the latest role whose lane for it ran, else in the walk."""
    for r in reversed(roles):
        st = per_role[r].get(pid)
        if st is not None and st.state is not CoverageState.NOT_RUN:
            return st
    return per_role[roles[0]].get(pid) or c.PanelCoverage(state=CoverageState.NOT_RUN)


def _side_seen(per_role: dict[str, dict[PanelId, c.PanelCoverage]], roles: list[str], side: Side) -> bool:
    """True when some door, fender or wheel on that side was filmed or glimpsed in roles."""
    for pid in PanelId:
        if _panel_side_position(pid)[0] is not side:
            continue
        for r in roles:
            st = per_role[r].get(pid)
            if st is not None and st.state in (CoverageState.FILMED, CoverageState.GLIMPSED):
                return True
    return False


def topup_missing(
    per_role: dict[str, dict[PanelId, c.PanelCoverage]], th: dict[str, Any]
) -> tuple[list[PanelId], list[PanelId]]:
    """(missing, maybe_absent) over the latest walk and its top-ups: a slot filmed in any of them is done. missing runs
    wheels first, then lower bumpers, then the rest. maybe_absent holds rear doors and rear fenders never seen while the
    rest of their side was (a two-door car, or a quarter panel SAM does not call a fender): 16.22 keeps them off the
    missing list, and the plan asks for them only if the car has them. A whole side never seen stays missing."""
    if not per_role:
        return [], []
    roles = coverage_roles(per_role)
    filmed = {pid for r in roles for pid, st in per_role[r].items() if st.state is CoverageState.FILMED}
    open_slots = [pid for pid in PanelId if pid not in filmed and any(pid in per_role[r] for r in roles)]
    absent = [
        pid
        for pid in open_slots
        if pid in MAY_BE_ABSENT
        and _shown_state(per_role, roles, pid).state is CoverageState.NOT_SEEN
        and _side_seen(per_role, roles, _panel_side_position(pid)[0])
    ]
    rest = [pid for pid in open_slots if pid not in absent]
    groups = {
        "wheels": [pid for pid in rest if pid in WHEEL_PANELS],
        "lower_bumpers": [pid for pid in rest if pid in LOWER_BUMPERS],
    }
    groups["rest"] = [pid for pid in rest if pid not in WHEEL_PANELS and pid not in LOWER_BUMPERS]
    ordered: list[PanelId] = []
    for key in th["coverage"]["topup_order"]:
        ordered.extend(groups.get(key, []))
    return ordered, absent


def plan_topup(coverage: c.Coverage, th: dict[str, Any]) -> c.Coverage:
    """missing per topup_missing; topup_text; nouns_to_rerun = the missing nouns, then the side anchors (headlight, tail
    light, side mirror) when a door, fender or wheel is missing so a top-up track has something to vote its side on,
    then car, license plate and face. The odometer line stays while the odometer is not filmed, so the plan is a pure
    function of the coverage state: plan_topup(plan_topup(x)) == plan_topup(x)."""
    cov = coverage.model_copy(deep=True)
    always = [Noun(x) for x in th["nouns"]["topup_always"]]
    if not cov.per_role:
        cov.missing = []
        cov.nouns_to_rerun = list(always)
        cov.topup_text = ""
        return cov
    roles = coverage_roles(cov.per_role)
    ordered, absent = topup_missing(cov.per_role, th)
    cov.missing = ordered
    nouns: list[Noun] = []
    for pid in ordered + absent:
        if PANEL_NOUN[pid] not in nouns:
            nouns.append(PANEL_NOUN[pid])
    if any(n in LATERAL_NOUNS for n in nouns):
        nouns += [n for n in SIDE_ANCHOR_NOUNS if n not in nouns]
    cov.nouns_to_rerun = nouns + [n for n in always if n not in nouns]
    parts = []
    for pid in ordered:
        st = _shown_state(cov.per_role, roles, pid)
        tag = st.state.value.replace("_", " ")
        if st.state is CoverageState.GLIMPSED:
            tag = f"{tag}, {st.seconds:.1f} s"
        parts.append(f"{display_name(pid)} ({tag})")
    if ordered:
        text = "Film these again, in this order: " + "; ".join(parts) + "."
        text += f" The app then re-runs {len(cov.nouns_to_rerun)} of {len(LANE_NOUNS)} nouns: " + ", ".join(n.value for n in cov.nouns_to_rerun) + "."
    elif absent:
        text = f"Every panel seen in the {roles[0]} walk is filmed."
    else:
        text = f"Every panel is filmed in the {roles[0]} walk."
    if absent:
        text += " If this car has them, also film: " + ", ".join(display_name(pid) for pid in absent) + "."
    if th["coverage"].get("odometer_ask_once", True) and not cov.odometer.filmed:
        cov.odometer.asked = True
        text += " Also film the odometer once."
    cov.topup_text = text
    return cov


def _mmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60}:{s % 60:02d}"


def timeline_line(panel: c.Panel, role: str, tracks: list[c.PanelTrack]) -> str | None:
    """'left rear door: 0:08 to 0:14 and 0:31 to 0:35, same door, fragments 5 to 1' for one panel and role."""
    pr = panel.per_role.get(role)
    if pr is None or not pr.windows:
        return None
    spans = " and ".join(f"{_mmss(a)} to {_mmss(b)}" for a, b in pr.windows)
    mine = [t for t in tracks if t.role == role and t.track_id in pr.track_ids]
    fragments = sum(len(t.fragments) for t in mine)
    name = display_name(panel.panel_id)
    noun_word = name.split()[-1]
    tail = f", same {noun_word}, fragments {fragments} to {len(mine)}" if fragments > len(mine) else ""
    return f"{name}: {spans}{tail}"


def panel_timeline(panels: list[c.Panel], tracks: list[c.PanelTrack], *, role: str) -> list[dict[str, Any]]:
    """First and last visible seconds per canonical panel in a role, with fragment and track counts."""
    out: list[dict[str, Any]] = []
    for panel in panels:
        pr = panel.per_role.get(role)
        if pr is None or not pr.windows:
            continue
        mine = [t for t in tracks if t.role == role and t.track_id in pr.track_ids]
        out.append(
            {
                "panel_id": panel.panel_id,
                "first_s": pr.windows[0][0],
                "last_s": pr.windows[-1][1],
                "windows": pr.windows,
                "seconds_visible": pr.seconds_visible,
                "fragments": sum(len(t.fragments) for t in mine),
                "tracks": len(mine),
                "line": timeline_line(panel, role, tracks),
            }
        )
    return out


# ----------------------------------------------------------------------------------------------------------------
# derive: the glue
# ----------------------------------------------------------------------------------------------------------------


class KeyframeError(RuntimeError):
    """No keyframe of a role could be read or blurred: coverage would otherwise blame the footage for an outage."""


def _geometry_mod():
    from . import geometry

    return geometry


def _video_mod():
    from . import video

    return video


def _lane_path(run_id: str, role: str, record: c.LaneRecord | None, noun: Noun) -> Path | None:
    if record is None:
        return None
    if record.lane_path:
        return paths.run_dir(run_id) / record.lane_path
    gz = paths.lane_events_gz_path(run_id, role, noun)
    if gz.is_file():
        return gz
    plain = paths.lane_events_path(run_id, role, noun)
    return plain if plain.is_file() else None


def _read_list(path: Path, model: type) -> list[Any]:
    return c.read_json_list(path, model) if path.is_file() else []


def _lane_status_of_role(run_id: str, role: str) -> dict[Noun, LaneStatus]:
    """Status per noun from the lane summaries on disk; an unreadable summary counts as missing (so not_run)."""
    out: dict[Noun, LaneStatus] = {}
    for noun in LANE_NOUNS:
        p = paths.lane_summary_path(run_id, role, noun)
        if p.is_file():
            try:
                out[noun] = c.read_json(p, c.LaneRecord).status
            except Exception as exc:
                log.warning("%s: unreadable lane summary %s (%s); its slots print not_run", run_id, p.name, exc)
    return out


def lane_status_from_lanes(lanes: dict[str, tuple[c.LaneRecord, list[c.IndexFrame]]]) -> dict[Noun, LaneStatus]:
    out: dict[Noun, LaneStatus] = {}
    for noun in LANE_NOUNS:
        entry = lanes.get(noun_slug(noun))
        if entry is not None and entry[0] is not None:
            out[noun] = entry[0].status
    return out


def _mask_to_crop(mask: np.ndarray, crop: c.Crop) -> np.ndarray:
    """Cut the source-pixel mask to the crop box and resize it to (crop.h, crop.w)."""
    import cv2

    h_src, w_src = mask.shape[:2]
    x0, y0 = int(math.floor(crop.dx)), int(math.floor(crop.dy))
    x1 = int(math.ceil(crop.dx + crop.w / crop.scale))
    y1 = int(math.ceil(crop.dy + crop.h / crop.scale))
    sub = mask[max(0, y0) : min(h_src, y1), max(0, x0) : min(w_src, x1)]
    if sub.size == 0:
        return np.zeros((crop.h, crop.w), dtype=bool)
    return cv2.resize(sub.astype(np.uint8), (crop.w, crop.h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _scale_box(box: Box, from_size: tuple[int, int], to_size: tuple[int, int]) -> list[float]:
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return [float(box[0]) * sx, float(box[1]) * sy, float(box[2]) * sx, float(box[3]) * sy]


def _sorted_tracks(tracks: list[c.PanelTrack]) -> list[c.PanelTrack]:
    return sorted(tracks, key=lambda t: (_role_order(t.role), t.track_id))


def _sorted_keyframes(keyframes: list[c.Keyframe]) -> list[c.Keyframe]:
    return sorted(keyframes, key=lambda k: (_role_order(k.role), k.kf_id))


def _voted(tracks: list[c.PanelTrack]) -> bool:
    """True when every track the vote names carries a panel_id (an earlier derive finished its vote)."""
    return all(t.panel_id is not None for t in tracks if t.noun in LATERAL_NOUNS or t.noun is Noun.BUMPER or t.noun in SINGLE_SLOT)


def _expected_slots(
    run_id: str, role: str, prior_tracks: list[c.PanelTrack], prior_keyframes: list[c.Keyframe], fps: float, th: dict[str, Any]
) -> list[PanelId]:
    """What a top-up role was asked to film: the missing list of the walk and the top-ups before it."""
    if not role.startswith("topup_"):
        return []
    before = sorted({t.role for t in prior_tracks if _role_order(t.role) < _role_order(role)}, key=_role_order)
    if not before:
        return []
    canon = [c.Panel(panel_id=pid, noun=PANEL_NOUN[pid]) for pid in PanelId]
    states = {
        r: coverage_for_role(canon, prior_tracks, prior_keyframes, role=r, fps=fps, lane_status=_lane_status_of_role(run_id, r), th=th)
        for r in before
    }
    return topup_missing(states, th)[0]


def _vote_with_prior(
    run_id: str,
    role: str,
    new_tracks: list[c.PanelTrack],
    prior_tracks: list[c.PanelTrack],
    prior_keyframes: list[c.Keyframe],
    fps: float,
    th: dict[str, Any],
) -> tuple[list[c.Panel], list[c.PanelTrack]]:
    """vote_sides over every role: this role fresh (with its top-up request), the others frozen once voted."""
    direction_by_role: dict[str, int | None] = {role: walk_direction(new_tracks, th)}
    for other in sorted({t.role for t in prior_tracks}, key=_role_order):
        direction_by_role[other] = walk_direction([t for t in prior_tracks if t.role == other], th)
    frozen = {r for r in direction_by_role if r != role and _voted([t for t in prior_tracks if t.role == r])}
    expected = _expected_slots(run_id, role, prior_tracks, prior_keyframes, fps, th)
    panels, tracks = vote_sides(
        prior_tracks + new_tracks,
        direction_by_role[role],
        th,
        fps=fps,
        direction_by_role=direction_by_role,
        frozen_roles=frozen,
        expected_by_role={role: expected} if expected else None,
    )
    return panels, _sorted_tracks(tracks)


def _coverage_all(
    run_id: str,
    role: str,
    panels: list[c.Panel],
    tracks: list[c.PanelTrack],
    keyframes: list[c.Keyframe],
    prior_tracks: list[c.PanelTrack],
    prior_coverage: c.Coverage,
    lane_status: dict[Noun, LaneStatus],
    fps: float,
    th: dict[str, Any],
) -> tuple[c.Coverage, dict[PanelId, c.PanelCoverage]]:
    """Coverage of every role, copied onto the panels' per_role entries, then the top-up plan."""
    cov_role = coverage_for_role(panels, tracks, keyframes, role=role, fps=fps, lane_status=lane_status, th=th)
    per_role: dict[str, dict[PanelId, c.PanelCoverage]] = {}
    for other in sorted({t.role for t in prior_tracks} | set(prior_coverage.per_role), key=_role_order):
        if other != role:
            status = _lane_status_of_role(run_id, other)
            per_role[other] = coverage_for_role(panels, tracks, keyframes, role=other, fps=fps, lane_status=status, th=th)
    per_role[role] = cov_role
    for panel in panels:
        for r, states in per_role.items():
            pc = states[panel.panel_id]
            pr = panel.per_role.get(r) or c.PanelRole()
            pr.state, pr.state_basis, pr.reason = pc.state, pc.state_basis, pc.reason
            pr.keyframe_ids = [k.kf_id for k in keyframes if k.role == r and k.panel_id == panel.panel_id]
            panel.per_role[r] = pr
    coverage = c.Coverage(per_role=per_role, odometer=prior_coverage.odometer, rerun_seconds=prior_coverage.rerun_seconds)
    return plan_topup(coverage, th), cov_role


def _write_derived(
    run_id: str, tracks: list[c.PanelTrack], panels: list[c.Panel], keyframes: list[c.Keyframe], coverage: c.Coverage
) -> None:
    c.write_json(paths.tracks_json(run_id), tracks)
    c.write_json(paths.panels_json(run_id), panels)
    c.write_json(paths.keyframes_json(run_id), keyframes)
    c.write_json(paths.coverage_json(run_id), coverage)


async def derive(
    run_id: str,
    role: str,
    *,
    lanes: dict[str, tuple[c.LaneRecord, list[c.IndexFrame]]],
    pts: Sequence[float],
    media_size: tuple[int, int],
    fps: float,
    decode: Any,
    blur_keyframe: Callable[..., Awaitable[tuple[np.ndarray, c.BlurRecord]]],
    th: dict[str, Any],
) -> tuple[list[c.PanelTrack], list[c.Panel], list[c.Keyframe], dict[PanelId, c.PanelCoverage]]:
    """Filter, fragments, merge, votes, keyframes (decoding only the candidates), blurred crops, moments, coverage.

    This role's plate and face tracks reach tracks.json before anything that can fail, so the blur stage never renders
    the display proxy from a list that lacks them. Then tracks.json, panels.json, keyframes.json and coverage.json are
    written through contracts.write_json, keeping the entries of the other roles (their votes frozen) and sorted by role
    and id. Heavy work runs in worker threads so the event loop keeps serving SSE. Returns everything written plus this
    role's coverage."""
    records = {noun: (lanes.get(noun_slug(noun)) or (None, []))[0] for noun in LANE_NOUNS}
    frames_by_noun = {noun: (lanes.get(noun_slug(noun)) or (None, []))[1] for noun in LANE_NOUNS}
    common = dict(pts=pts, media_size=media_size, fps=fps, th=th)
    privacy_new = await asyncio.to_thread(_role_privacy_tracks, role, frames_by_noun, **common)
    on_disk = await asyncio.to_thread(_read_list, paths.tracks_json(run_id), c.PanelTrack)
    prior_tracks = [t for t in on_disk if t.role != role]
    early = prior_tracks + [t for t in on_disk if t.role == role and t.noun not in PRIVACY_NOUNS] + privacy_new
    await asyncio.to_thread(c.write_json, paths.tracks_json(run_id), _sorted_tracks(early))
    panel_new, kept = await asyncio.to_thread(_role_panel_tracks, role, frames_by_noun, **common)
    new_tracks = privacy_new + panel_new
    prior_keyframes = [k for k in await asyncio.to_thread(_read_list, paths.keyframes_json(run_id), c.Keyframe) if k.role != role]
    cov_path = paths.coverage_json(run_id)
    prior_coverage = await asyncio.to_thread(c.read_json, cov_path, c.Coverage) if cov_path.is_file() else c.Coverage()
    built = [t for t in prior_tracks if is_wheel_fender(t)]  # lead 11/1: fenders built from wheels never vote
    voters = [t for t in prior_tracks if not is_wheel_fender(t)]
    panels, tracks = await asyncio.to_thread(_vote_with_prior, run_id, role, new_tracks, voters, prior_keyframes, fps, th)
    if fender_settings(th) is not None:
        panels, tracks = apply_wheel_fenders(role, panels, tracks + built, car_boxes_by_frame(kept.get(Noun.CAR, [])), media_size, fps, th)
    role_tracks = [t for t in tracks if t.role == role]
    keyframes = await _derive_keyframes(
        run_id, role, panels, role_tracks, kept, records, pts, media_size, fps, decode, blur_keyframe, th, prior_keyframes
    )
    all_keyframes = _sorted_keyframes(prior_keyframes + keyframes)
    coverage, cov_role = await asyncio.to_thread(
        _coverage_all, run_id, role, panels, tracks, all_keyframes, prior_tracks, prior_coverage,
        lane_status_from_lanes(lanes), fps, th,
    )
    await asyncio.to_thread(_write_derived, run_id, tracks, panels, all_keyframes, coverage)
    return tracks, panels, all_keyframes, cov_role


def door_end_views(role_tracks: list[c.PanelTrack], th: dict[str, Any]) -> dict[str, set[int]]:
    """{door track id: frames whose door box holds at least panels.door_check end_light_inside_min of a headlight or tail
    light box}. A side door never holds a lamp; such a box is a tailgate or a corner seen end-on (SAM's door lane on
    the NHTSA Golf's hatch), so those frames are never door keyframe candidates. Empty when the leaf is absent."""
    cfg = door_settings(th)
    share_min = cfg.get("end_light_inside_min") if cfg is not None else None
    if share_min is None:
        return {}
    lights: dict[int, list[Box]] = {}
    for t in role_tracks:
        if t.noun in (Noun.HEADLIGHT, Noun.TAIL_LIGHT):
            for r in t.per_frame:
                lights.setdefault(r.frame, []).append(r.box)
    out: dict[str, set[int]] = {}
    for t in role_tracks:
        if t.noun is Noun.CAR_DOOR:
            hit = {r.frame for r in t.per_frame if any(containment(b, r.box) >= float(share_min) for b in lights.get(r.frame, ()))}
            if hit:
                out[t.track_id] = hit
    return out


def _keyframe_plan(
    panels: list[c.Panel], role_tracks: list[c.PanelTrack], th: dict[str, Any]
) -> list[tuple[c.Panel, list[tuple[int, c.PanelTrack]]]]:
    """The box pass per panel: up to keyframe_box_pass_candidates (frame, track) pairs, one per frame. A placeholder
    track never takes keyframes from a slot a decided track of the role holds."""
    limit = int(th["panels"]["keyframe_box_pass_candidates"])
    end_views = door_end_views(role_tracks, th)
    plan = []
    for panel in panels:
        if panel.noun in UNNAMED_NOUNS:
            continue
        mine = [t for t in role_tracks if t.panel_id == panel.panel_id]
        pool = [t for t in mine if not undecided(t)] or mine
        ranked = sorted(
            ((f, t) for t in pool for f in keyframe_candidates(t, th) if f not in end_views.get(t.track_id, ())),
            key=lambda ft: (-box_pass_score(ft[1], ft[0], th), ft[0]),
        )
        cands: list[tuple[int, c.PanelTrack]] = []
        for f, t in ranked:
            if len(cands) < limit and all(f != g for g, _ in cands):
                cands.append((f, t))
        if cands:
            plan.append((panel, cands))
    return plan


def _wanted_masks(
    plan: list[tuple[c.Panel, list[tuple[int, c.PanelTrack]]]],
    kept_at: dict[Noun, dict[int, list[c.IndexObject]]],
    car_ids: dict[int, str] | None = None,
) -> dict[Noun, list[tuple[int, str]]]:
    """Every (frame, id) the scoring pass decodes per lane: the panel at each candidate, from the lane of the track it
    came from (the twin check can file a trunk-lid lane track under the hood), and the car windows (doors), plates and
    faces in those frames. Shared frames are decoded once. The car masks are not here: the containment test decodes
    them only for the candidates a panel is about to keep (_CarTest)."""
    wanted: dict[Noun, set[tuple[int, str]]] = {}
    for _panel, cands in plan:
        for f, t in cands:
            oid = _object_at(t, f)
            if oid is not None:
                wanted.setdefault(t.noun, set()).add((f, oid))
            companions = (*PRIVACY_NOUNS, Noun.CAR_WINDOW) if t.noun is Noun.CAR_DOOR else PRIVACY_NOUNS
            if is_wheel_fender(t) and car_ids is not None:  # the car and every named part at the frame (lead 11/1)
                companions = tuple(dict.fromkeys((*PRIVACY_NOUNS, *FENDER_CUT_NOUNS)))
                if f in car_ids:
                    wanted.setdefault(Noun.CAR, set()).add((f, car_ids[f]))
            for noun in companions:
                wanted.setdefault(noun, set()).update((f, o.id) for o in kept_at[noun].get(f, []))
    return {noun: sorted(pairs) for noun, pairs in wanted.items() if pairs}


def _decode_lane(
    decode: Any, lane_path: Path | None, pairs: list[tuple[int, str]], lookup: dict[tuple[int, str], c.IndexObject]
) -> dict[tuple[int, str], np.ndarray]:
    """Rasters for (frame, id) pairs of one lane in one DecodePool call; a bad payload falls back to one call per mask,
    so it costs only its own frame."""
    if lane_path is None:
        return {}
    items = [(key, lookup[key].mask) for key in pairs if key in lookup and lookup[key].mask is not None]
    if not items:
        return {}
    try:
        return {key: raster for (key, _), raster in zip(items, decode.decode(lane_path, [ref for _, ref in items]))}
    except Exception as exc:
        log.warning("decoding %d masks of %s failed (%s); retrying one at a time", len(items), lane_path.name, exc)
    out: dict[tuple[int, str], np.ndarray] = {}
    for key, ref in items:
        try:
            out[key] = decode.decode(lane_path, [ref])[0]
        except Exception as exc:
            log.warning("mask of id %s at frame %d of %s dropped: %s", key[1], key[0], lane_path.name, exc)
    return out


def _decode_cpu(decode: Any) -> float | None:
    stats = getattr(decode, "stats", None)
    return float(stats()[1]) if callable(stats) else None


class _Masks:
    """The keyframe pass's decoder. One_bit masks go through fastmask in the calling thread (each lane's text read
    once) unless thresholds panels.fast_mask_decode is false; a payload it refuses, and everything when it is off, go
    to the caller's DecodePool. cpu() is the CPU both spent, the figure the decode budget compares."""

    def __init__(self, pool: Any, th: dict[str, Any]) -> None:
        self.pool = pool
        self.fast = fastmask.LaneMasks() if th["panels"].get("fast_mask_decode", True) and fastmask.available() else None
        self._pool_cpu0 = _decode_cpu(pool) if pool is not None else None
        self._lock = threading.Lock()

    @property
    def usable(self) -> bool:
        return self.fast is not None or self.pool is not None

    def lane(
        self, lane_path: Path | None, pairs: list[tuple[int, str]], lookup: dict[tuple[int, str], c.IndexObject], *, rows: int | None = None
    ) -> dict[tuple[int, str], np.ndarray]:
        """Rasters for (frame, id) pairs of one lane; a mask no decoder could read is dropped, costing only its frame."""
        if self.fast is None:
            return _decode_lane(self.pool, lane_path, pairs, lookup) if self.pool is not None else {}
        if lane_path is None:
            return {}
        out: dict[tuple[int, str], np.ndarray] = {}
        for key in pairs:
            obj = lookup.get(key)
            if obj is None or obj.mask is None:
                continue
            try:
                with self._lock:
                    out[key] = self.fast.decode(lane_path, [obj.mask], rows=rows)[0]
                continue
            except Exception as exc:
                why = exc
            if self.pool is not None:
                got = _decode_lane(self.pool, lane_path, [key], lookup)
                if key in got:
                    out[key] = got[key]
                    continue
            log.warning("mask of id %s at frame %d of %s dropped: %s", key[1], key[0], lane_path.name, why)
        return out

    def cpu(self) -> float:
        spent = self.fast.stats()[1] if self.fast is not None else 0.0
        pool_now = _decode_cpu(self.pool) if self.pool is not None else None
        if pool_now is not None and self._pool_cpu0 is not None:
            spent += pool_now - self._pool_cpu0
        return spent


class _CarTest:
    """score_keyframe's car-mask containment test, run only on a candidate a panel is about to keep. The panel mask
    (window already cut) must have car_mask_containment_min of its pixels inside the car mask. The car raster is
    decoded only down to the panel's last row: every pixel the test reads lies above it, so the answer is the one a
    whole decode gives. Each frame's car mask is decoded once (again in full when a later panel needs more rows)."""

    def __init__(self, masks: _Masks, lane_path: Path | None, lookup: dict[tuple[int, str], c.IndexObject], car_ids: dict[int, str], media_size: tuple[int, int], th: dict[str, Any]) -> None:
        self.masks, self.lane_path, self.lookup, self.car_ids = masks, lane_path, lookup, car_ids
        self.media_size = media_size
        self.min_share = float(th["panels"]["car_mask_containment_min"])
        self.decoded = 0
        self._cache: dict[int, tuple[int | None, np.ndarray | None]] = {}
        self._lock = threading.Lock()  # panels run concurrently and call passes() from worker threads

    def car_mask(self, f: int, last_row: int) -> tuple[np.ndarray | None, bool]:
        """(the car mask at proxy size or None when there is none to test against, False when the car raster lies
        wholly under the panel's last row and so holds none of its pixels)."""
        with self._lock:
            return self._car_mask(f, last_row)

    def _car_mask(self, f: int, last_row: int) -> tuple[np.ndarray | None, bool]:
        oid = self.car_ids.get(f)
        obj = self.lookup.get((f, oid)) if oid is not None else None
        if obj is None or obj.mask is None:
            return None, True
        top = int(obj.box[1]) + obj.mask.y
        need = last_row - top + 1
        if need <= 0:
            return None, False
        rows = None if need >= obj.mask.h else need
        cached = self._cache.get(f)
        if cached is not None and (cached[0] is None or (rows is not None and cached[0] >= rows)):
            return cached[1], True
        if cached is not None:
            rows = None  # a second, deeper look at the same frame: decode it whole once
        got = self.masks.lane(self.lane_path, [(f, obj.id)], self.lookup, rows=rows)
        raster = got.get((f, obj.id))
        self.decoded += 1
        pasted = sam_wire.paste_mask(np.asarray(raster, dtype=bool), list(obj.box), obj.mask, self.media_size) if raster is not None else None
        self._cache[f] = (rows, pasted)
        return pasted, True

    def passes(self, f: int, mask: np.ndarray, last_row: int) -> bool:
        car, reachable = self.car_mask(f, last_row)
        if not reachable:
            return False
        if car is None:
            return True  # no car mask decoded at this frame: score_keyframe skips the test too
        total = int(mask.sum())
        return total > 0 and float((mask & _fit_mask(car, mask.shape[:2])).sum()) / total >= self.min_share


def _mask_at(
    rasters: dict[tuple[Noun, int, str], np.ndarray],
    noun: Noun,
    frame: int,
    objects: Sequence[c.IndexObject],
    media_size: tuple[int, int],
) -> np.ndarray | None:
    """The union of the decoded masks of objects at a frame as a bool (H, W) proxy frame; None when none decoded."""
    out: np.ndarray | None = None
    for obj in objects:
        raster = rasters.get((noun, frame, obj.id))
        if raster is None or obj.mask is None:
            continue
        pasted = sam_wire.paste_mask(np.asarray(raster, dtype=bool), list(obj.box), obj.mask, media_size)
        out = pasted if out is None else out | pasted
    return out


def _read_frames(video: Any, src: Path, times: list[float]) -> dict[float, np.ndarray]:
    """Source frames at times in one decode pass; if the pass fails, one read per time, dropping only what fails."""
    out: dict[float, np.ndarray] = {}
    try:
        for t, rgb in video.iter_frames_at(src, times):
            out[t] = rgb
        return out
    except Exception as exc:
        log.warning("reading %d frames of %s failed (%s); reading them one at a time", len(times), src.name, exc)
    for t in times:
        if t in out:
            continue
        try:
            out[t] = video.read_frame_at(src, t)
        except Exception as exc:
            log.warning("frame at %.3f s of %s dropped: %s", t, src.name, exc)
    return out


def _score_panel(
    panel: c.Panel,
    cands: list[tuple[int, c.PanelTrack]],
    rasters: dict[tuple[Noun, int, str], np.ndarray],
    lookups: dict[Noun, dict[tuple[int, str], c.IndexObject]],
    kept_at: dict[Noun, dict[int, list[c.IndexObject]]],
    video: Any,
    src: Path,
    pts: Sequence[float],
    media_size: tuple[int, int],
    th: dict[str, Any],
    built: dict[str, Any] | None = None,
) -> tuple[list[tuple[int, c.KeyframeScore]], dict[int, tuple[c.PanelTrack, np.ndarray, bool, np.ndarray, int]]]:
    """The mask pass on one panel's candidates, each source frame read once: [(frame, raw score)] and per frame the
    track, the panel mask after window subtraction, window_subtracted, the unblurred frame (kept for the blur) and the
    panel mask's last proxy row. The car-mask containment test is left to _CarTest: it only zeroes a failing area, so
    the normalised totals of the others do not depend on it."""
    fallback_fps = float(th["media"]["proxy_fps"])
    times = {f: float(_pts_at(pts, f, fallback_fps)) for f, _ in cands}
    frames = _read_frames(video, src, sorted(set(times.values())))
    scored: list[tuple[int, c.KeyframeScore]] = []
    details: dict[int, tuple[c.PanelTrack, np.ndarray, bool, np.ndarray, int]] = {}
    for f, t in cands:
        rgb = frames.get(times[f])
        if is_wheel_fender(t):
            mask = _built_fender_mask(t, f, rasters, lookups, kept_at, media_size, built)
        else:
            oid = _object_at(t, f)
            obj = lookups[t.noun].get((f, oid)) if oid is not None else None
            mask = _mask_at(rasters, t.noun, f, [obj] if obj is not None else [], media_size)
        if rgb is None or mask is None:
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        window = _mask_at(rasters, Noun.CAR_WINDOW, f, kept_at[Noun.CAR_WINDOW].get(f, []), media_size) if t.noun is Noun.CAR_DOOR else None
        exclude = [m for n in PRIVACY_NOUNS if (m := _mask_at(rasters, n, f, kept_at[n].get(f, []), media_size)) is not None]
        try:
            score, m2, ws = score_keyframe(t, f, mask=mask, car_mask=None, window_mask=window, frame_rgb=rgb, exclude_masks=exclude, th=th)
        except Exception as exc:
            log.warning("%s frame %d: scoring failed (%s)", panel.panel_id.value, f, exc)
            continue
        scored.append((f, score))
        details[f] = (t, m2, ws, rgb, int(rows[-1]) if rows.size else -1)
    return scored, details


def view_match_return(th: dict[str, Any] | None) -> bool:
    """thresholds panels.view_match_return (lead 11/1): the return walk puts first the candidate whose view best
    matches a pickup keyframe of the panel, measured the way G4 measures it (the whitened mask IoU) with the same
    window cut (G9) and a sharp enough frame (G3), when it beats what the score order alone would keep. Off when
    absent."""
    return bool((th or {}).get("panels", {}).get("view_match_return", False))


def view_matched_first(
    ordered: list[tuple[int, c.KeyframeScore]],
    masks: dict[int, tuple[np.ndarray, bool]],
    refs: list[tuple[np.ndarray, c.Moments, bool]],
    front_sign: int | None,
    *,
    top_k: int,
    th: dict[str, Any],
) -> tuple[list[tuple[int, c.KeyframeScore]], dict[int, float]]:
    """ordered (best score first) with the best view-matched candidate moved to the front when its IoU with a pickup
    keyframe beats the best of the first top_k; and each candidate's best IoU (-1 when it cannot pair: no score, too
    blurry for G3, or no pickup keyframe with the same window cut)."""
    geometry = _geometry_mod()
    ious: dict[int, float] = {}
    if front_sign is None or not refs:
        return ordered, ious
    sharp_min = float(th["matching"]["sharpness_rel_min"])
    for f, score in ordered:
        mask, ws = masks[f]
        best = -1.0
        if score.total > 0 and score.sharpness >= sharp_min and int(mask.sum()) >= 3:
            mom = geometry.moments_from_mask(mask)
            for ref_mask, ref_mom, ref_ws in refs:
                if ref_ws == ws:
                    best = max(best, float(geometry.whitened_mask_iou(mask, mom, front_sign, ref_mask, ref_mom, front_sign)))
        ious[f] = round(best, 4)
    if not ordered:
        return ordered, ious
    head = max((ious[f] for f, _ in ordered[:top_k]), default=-1.0)
    f_best = max(ordered, key=lambda fs: (ious[fs[0]], -fs[0]))[0]
    if ious[f_best] <= head:
        return ordered, ious
    return [fs for fs in ordered if fs[0] == f_best] + [fs for fs in ordered if fs[0] != f_best], ious


def _built_fender_mask(
    t: c.PanelTrack,
    f: int,
    rasters: dict[tuple[Noun, int, str], np.ndarray],
    lookups: dict[Noun, dict[tuple[int, str], c.IndexObject]],
    kept_at: dict[Noun, dict[int, list[c.IndexObject]]],
    media_size: tuple[int, int],
    built: dict[str, Any] | None,
) -> np.ndarray | None:
    """A wheel fender's mask at a candidate (wheel_fender_mask); the row's box, area and border then follow the mask, so
    the crop is cut around the fender itself."""
    if not built or built.get("cfg") is None:
        return None
    wheel_box = built["wheel_boxes"].get(t.track_id, {}).get(f)
    car_id = built["car_ids"].get(f)
    car_obj = lookups[Noun.CAR].get((f, car_id)) if car_id is not None else None
    car = _mask_at(rasters, Noun.CAR, f, [car_obj] if car_obj is not None else [], media_size)
    if wheel_box is None or car is None:
        return None
    cuts = [m for n in FENDER_CUT_NOUNS if (m := _mask_at(rasters, n, f, kept_at.get(n, {}).get(f, []), media_size)) is not None]
    mask = wheel_fender_mask(wheel_box, car, cuts, built["cfg"])
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    for r in t.per_frame:
        if r.frame == f:
            r.box = box
            r.box_area_frac = round(_area(box) / float(media_size[0] * media_size[1]), 6)
            r.border_frac = round(border_frac(box, media_size), 6)
    return mask


def _keyframe_files(
    run_id: str,
    role: str,
    kf_id: str,
    panel: c.Panel,
    track: c.PanelTrack,
    f: int,
    mask: np.ndarray,
    window_subtracted: bool,
    blurred: np.ndarray,
    blur: c.BlurRecord,
    score: c.KeyframeScore,
    car_box: list[int] | None,
    media_size: tuple[int, int],
    th: dict[str, Any],
) -> c.Keyframe:
    """Crop, crop mask PNG, moments and the car-box frame, every pixel cut from the blurred frame; the Keyframe."""
    p = th["panels"]
    geometry, video = _geometry_mod(), _video_mod()
    quality = int(th["spark"]["jpeg_quality"])
    h_src, w_src = int(blurred.shape[0]), int(blurred.shape[1])
    row = next(r for r in track.per_frame if r.frame == f)
    box_src = _scale_box(row.box, media_size, (w_src, h_src))
    crop = geometry.crop_geometry(box_src, (w_src, h_src), pad_frac=float(p["crop_pad_frac"]), long_edge=int(p["crop_long_edge_px"]))
    crop_file = paths.crop_path(run_id, role, kf_id)
    sha = video.write_jpeg(video.cut_crop(blurred, crop), crop_file, quality=quality)
    crop = crop.model_copy(update={"path": paths.rel(run_id, crop_file), "sha256": sha})
    mask_crop = _mask_to_crop(_fit_mask(mask, (h_src, w_src)), crop)
    mask_file = paths.crop_mask_png_path(run_id, role, kf_id)
    video.write_png_mask(mask_crop, mask_file)
    moments = geometry.moments_from_mask(mask_crop) if int(mask_crop.sum()) >= 3 else None
    if car_box is not None:
        cb = [int(round(v)) for v in _scale_box(car_box, media_size, (w_src, h_src))]
        frame_crop = blurred[max(0, cb[1]) : min(h_src, cb[3]), max(0, cb[0]) : min(w_src, cb[2])]
        if frame_crop.size:
            video.write_jpeg(frame_crop, paths.keyframe_frame_path(run_id, role, kf_id), quality=quality)
    return c.Keyframe(
        kf_id=kf_id,
        role=role,
        panel_id=panel.panel_id,
        track_id=track.track_id,
        f_proxy=f,
        pts=row.pts,
        score=score,
        aspect_ratio=_mask_aspect(mask),
        box_src=box_src,
        crop=crop,
        mask_png=paths.rel(run_id, mask_file),
        moments=moments,
        front_sign=panel.front_sign,
        window_subtracted=window_subtracted,
        blur=blur,
    )


async def _derive_keyframes(
    run_id: str,
    role: str,
    panels: list[c.Panel],
    role_tracks: list[c.PanelTrack],
    kept: dict[Noun, list[c.IndexFrame]],
    records: dict[Noun, c.LaneRecord | None],
    pts: Sequence[float],
    media_size: tuple[int, int],
    fps: float,
    decode: Any,
    blur_keyframe: Callable[..., Awaitable[tuple[np.ndarray, c.BlurRecord]]],
    th: dict[str, Any],
    prior_keyframes: list[c.Keyframe] | None = None,
) -> list[c.Keyframe]:
    """The box pass, one decode per mask the scoring pass needs (panel, car windows on doors, plates, faces at each
    candidate), the mask pass on each candidate's source frame (read once, off the event loop), then per panel in
    score order: the car-mask containment test (the car mask decoded only now, only down to the panel's last row),
    the blur and the crop, until the panel has its keyframes. A failed read, decode, score or blur drops that frame and
    the next candidate takes its place; KeyframeError when no candidate could be scored or every blur failed. Over the
    decode budget (section 14 item 34) each panel keeps keyframe_top_k_under_budget. Up to
    keyframe_panels_in_flight panels run at once (each blur is two image requests); a frame two panels keep is
    blurred once and both keyframes carry that one answer."""
    p = th["panels"]
    plan = _keyframe_plan(panels, role_tracks, th)
    if not plan:
        return []
    masks = _Masks(decode, th)
    if not masks.usable:
        raise KeyframeError(f"{role}: no DecodePool and the fast decoder is off, so no keyframe mask can be decoded")
    video = _video_mod()
    src = paths.quarantine_original(run_id, role)
    lookups = {noun: _box_lookup(kept[noun]) for noun in LANE_NOUNS}
    lane_paths = {noun: _lane_path(run_id, role, records[noun], noun) for noun in LANE_NOUNS}
    fcfg = fender_settings(th)
    at_nouns = (Noun.CAR_WINDOW, *PRIVACY_NOUNS, *(FENDER_CUT_NOUNS if fcfg is not None else ()))
    kept_at = {noun: {fr.frame: fr.objects for fr in kept.get(noun, [])} for noun in dict.fromkeys(at_nouns)}
    car_ids = {fr.frame: max(fr.objects, key=lambda o: _area(o.box)).id for fr in kept[Noun.CAR] if fr.objects}
    by_id = {t.track_id: t for t in role_tracks}
    wheel_boxes = {
        t.track_id: {r.frame: r.box for r in by_id[wheel_id_of(t)].per_frame}
        for t in role_tracks if is_wheel_fender(t) and wheel_id_of(t) in by_id
    }
    car_boxes = car_boxes_by_frame(kept[Noun.CAR])
    wall0 = time.perf_counter()
    rasters: dict[tuple[Noun, int, str], np.ndarray] = {}
    for noun, pairs in _wanted_masks(plan, kept_at, car_ids if fcfg is not None else None).items():
        got = await asyncio.to_thread(masks.lane, lane_paths[noun], pairs, lookups[noun])
        rasters.update({(noun, f, oid): raster for (f, oid), raster in got.items()})
    spent = masks.cpu()
    view_refs: dict[PanelId, list[tuple[np.ndarray, c.Moments, bool]]] = {}
    if role == "return" and view_match_return(th):
        view_refs = await asyncio.to_thread(_view_refs, run_id, prior_keyframes or [])
    top_k = int(p["keyframe_top_k"])
    if spent > float(p["keyframe_decode_budget_s_per_video"]):
        top_k = int(p["keyframe_top_k_under_budget"])
        # TODO(integrator): section 14 item 34 also wants a ledger flag here; panels writes no ledger rows by contract.
        log.warning("%s %s: %.1f s of mask decoding is over the budget; top %d keyframes per panel", run_id, role, spent, top_k)
    run = _KeyframeRun(
        run_id=run_id, role=role, role_tracks=role_tracks, lookups=lookups, kept_at=kept_at, car_boxes=car_boxes,
        rasters=rasters, video=video, src=src, pts=pts, media_size=media_size, fps=fps, blur_keyframe=blur_keyframe, th=th,
        built={"wheel_boxes": wheel_boxes, "car_ids": car_ids, "cfg": fcfg}, view_refs=view_refs,
        top_k=top_k, car_test=_CarTest(masks, lane_paths[Noun.CAR], lookups[Noun.CAR], car_ids, media_size, th),
    )
    gate = asyncio.Semaphore(max(1, int(p.get("keyframe_panels_in_flight", 1))))

    async def one(panel: c.Panel, cands: list[tuple[int, c.PanelTrack]]) -> list[c.Keyframe]:
        async with gate:
            return await run.panel(panel, cands)

    tasks = [asyncio.ensure_future(one(panel, cands)) for panel, cands in plan]
    try:
        found = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:  # a bug in one panel must not leave the others writing crops after derive gave up
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    keyframes = [kf for kfs in found for kf in kfs]
    log.info(
        "%s %s: %d keyframes from %d candidates in %.1f s; mask decoding %.1f s CPU, %d car masks for the containment test",
        run_id, role, len(keyframes), run.n_scored, time.perf_counter() - wall0, masks.cpu(), run.car_test.decoded,
    )
    if not run.n_scored:
        raise KeyframeError(f"{role}: no keyframe candidate could be read and scored; last error: {run.last_error or 'none'}")
    if run.tried and run.failed == run.tried:
        raise KeyframeError(f"{role}: all {run.failed} keyframes failed; last error: {run.last_error}")
    return keyframes


def _view_refs(run_id: str, keyframes: list[c.Keyframe]) -> dict[PanelId, list[tuple[np.ndarray, c.Moments, bool]]]:
    """Each pickup keyframe's crop mask, moments and window cut, per panel (panels.view_match_return)."""
    import cv2

    out: dict[PanelId, list[tuple[np.ndarray, c.Moments, bool]]] = {}
    for kf in sorted(keyframes, key=lambda k: k.kf_id):
        if kf.role != "pickup" or not kf.mask_png or kf.moments is None:
            continue
        m = cv2.imread(str(paths.run_dir(run_id) / kf.mask_png), cv2.IMREAD_GRAYSCALE)
        if m is not None and (m > 0).any():
            out.setdefault(kf.panel_id, []).append((m > 0, kf.moments, kf.window_subtracted))
    return out


class _KeyframeRun:
    """The per-panel half of _derive_keyframes: score, then in score order test, blur and crop."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)
        self.n_scored = self.tried = self.failed = 0
        self.last_error = ""
        self._blurs: dict[int, asyncio.Future] = {}

    async def blur(self, f: int, rgb: np.ndarray) -> tuple[np.ndarray, c.BlurRecord]:
        """One blur per frame per derive: a second panel keeping the same frame waits for the first answer; a failure is
        not kept, so the next panel asks again."""
        pending = self._blurs.get(f)
        if pending is not None:
            blurred, record = await asyncio.shield(pending)
            return blurred, record.model_copy(deep=True)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._blurs[f] = future
        h_src, w_src = int(rgb.shape[0]), int(rgb.shape[1])
        try:
            # stored crops and frames always carry a blurred plate; only the sheet writer keeps it for the run owner
            # under keep_plate (section 8 items 5 and 6)
            result = await self.blur_keyframe(
                self.run_id, self.role, rgb, f_proxy=f, tracks=self.role_tracks, fps=self.fps,
                sizes=(self.media_size, (w_src, h_src)), keep_plate=False, th=self.th,
            )
        except asyncio.CancelledError:
            del self._blurs[f]
            future.cancel()
            raise
        except Exception as exc:
            del self._blurs[f]
            future.set_exception(exc)
            future.exception()  # retrieved here, so a frame nobody else waited for logs no warning
            raise
        future.set_result(result)
        return result

    async def panel(self, panel: c.Panel, cands: list[tuple[int, c.PanelTrack]]) -> list[c.Keyframe]:
        th = self.th
        try:
            scored, details = await asyncio.to_thread(
                _score_panel, panel, cands, self.rasters, self.lookups, self.kept_at, self.video, self.src, self.pts, self.media_size, th,
                self.built,
            )
        except Exception as exc:
            self.last_error = redact(f"{type(exc).__name__}: {exc}")
            log.warning("%s %s %s: keyframe pass failed (%s)", self.run_id, self.role, panel.panel_id.value, self.last_error)
            return []
        self.n_scored += len(scored)
        for f, score in scored:
            _note_decoded(details[f][0], f, score, th)
        edge = edge_on_candidates(scored, {f: details[f][0] for f, _ in scored}, th)
        if edge:  # panels.edge_on_floor: a sliver never becomes a keyframe; coverage reads edge_on_frames
            for f in sorted(edge):
                track = details[f][0]
                track.edge_on_frames = sorted({*track.edge_on_frames, f})
            scored = [(f, s) for f, s in scored if f not in edge]
            log.info("%s %s %s: %d of %d candidates seen edge-on, not kept", self.run_id, self.role, panel.panel_id.value, len(edge), len(edge) + len(scored))
        min_gap = float(th["panels"]["keyframe_min_gap_s"]) * self.fps
        taken: list[int] = []
        out: list[c.Keyframe] = []
        ordered = sorted(normalise_scores(scored, th), key=lambda fs: (-fs[1].total, fs[0]))
        refs = (getattr(self, "view_refs", None) or {}).get(panel.panel_id)
        if refs:
            ordered, _ious = view_matched_first(
                ordered, {f: (details[f][1], details[f][2]) for f, _ in ordered}, refs, panel.front_sign, top_k=self.top_k, th=th
            )
        for f, score in ordered:
            if len(taken) >= self.top_k:
                break
            if score.total <= 0 or any(abs(f - g) < min_gap for g in taken):
                continue
            track, mask, ws, rgb, last_row = details[f]
            try:
                inside = await asyncio.to_thread(self.car_test.passes, f, mask, last_row)
            except Exception as exc:  # a car mask that cannot be tested drops only this candidate
                inside = False
                log.warning("%s %s %s frame %d: car-mask test failed (%s)", self.run_id, self.role, panel.panel_id.value, f, exc)
            if not inside:
                _note_outside_car(track, f)
                continue
            self.tried += 1
            try:
                blurred, blur = await self.blur(f, rgb)
                kf = await asyncio.to_thread(
                    _keyframe_files, self.run_id, self.role, f"{self.role}_{panel.panel_id.value}_kf{len(taken) + 1}", panel,
                    track, f, mask, ws, blurred, blur, score, self.car_boxes.get(f), self.media_size, th,
                )
            except Exception as exc:
                self.failed += 1
                self.last_error = redact(f"{type(exc).__name__}: {exc}")
                log.warning("%s %s %s frame %d: keyframe dropped (%s)", self.run_id, self.role, panel.panel_id.value, f, self.last_error)
                continue
            taken.append(f)
            out.append(kf)
        details.clear()
        return out


def _note_outside_car(track: c.PanelTrack, frame: int) -> None:
    """A candidate whose mask failed the car-mask test scores area 0, as score_keyframe's test would have made it."""
    for r in track.per_frame:
        if r.frame == frame:
            r.area_frac = 0.0
            return


def _note_decoded(track: c.PanelTrack, frame: int, score: c.KeyframeScore, th: dict[str, Any]) -> None:
    """Mark the per_frame row of a decoded candidate: mask_decoded, area_frac and complete. area_frac is the mask's share
    of the frame before the car-mask test, which only the candidates a panel reached in score order took (a failure
    sets it to 0)."""
    for r in track.per_frame:
        if r.frame == frame:
            r.mask_decoded = True
            r.area_frac = score.area
            r.complete = score.completeness >= float(th["coverage"]["filmed_completeness_min"])
            return
