"""Panels (M5): SAM lanes to tracks, canonical panel names, keyframes, coverage and the top-up plan.

Covers docs/ARCHITECTURE.md 5.1 to 5.4. Every per-frame rule runs on boxes and c. Masks are decoded only for keyframe
candidates, through the DecodePool the caller hands to derive(). Nothing here reads a mark, calls Spark or opens the
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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Collection, Sequence

import numpy as np

from . import contracts as c
from . import paths, sam_wire
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

    def __init__(self, tracks: list[c.PanelTrack], direction: int | None, floor: float) -> None:
        self.tracks = tracks
        self.direction = direction
        self.floor = floor
        self.by_noun: dict[Noun, list[c.PanelTrack]] = {}
        for t in tracks:
            self.by_noun.setdefault(t.noun, []).append(t)
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
) -> list[c.Panel]:
    """Every canonical PanelId with its tracks per role, summed votes, side_uncertain and front_sign."""
    panels: list[c.Panel] = []
    for pid in PanelId:
        mine = [t for t in tracks if t.panel_id == pid]
        noun = PANEL_NOUN[pid]
        side, position = _panel_side_position(pid)
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
    votes are recomputed for the sums). direction_by_role overrides direction per role (walks may go opposite ways)."""
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
        _vote_role(_RoleContext(role_tracks, role_direction, floor), role_tracks, weights, margin_min, expected, votes_by_track)
        if role in frozen_roles:
            for t in role_tracks:
                t.panel_id, t.side, t.position = stored[t.track_id]
    panels = _panels_from(
        out_tracks,
        votes_by_track,
        fps=fps,
        merge_gap_s=float(p["merge_gap_max_s"]),
        fallback_dt=1.0 / float(th["media"]["proxy_fps"]),
    )
    return panels, out_tracks


def _role_order(role: str) -> tuple[int, int, str]:
    """pickup, return, the top-ups by number, then the claims by number."""
    if role in ("pickup", "return"):
        return (0 if role == "pickup" else 1, 0, role)
    prefix, _, number = role.partition("_")
    return ({"topup": 2, "claim": 3}.get(prefix, 4), int(number) if number.isdigit() else 0, role)


# ----------------------------------------------------------------------------------------------------------------
# 5.3 keyframes
# ----------------------------------------------------------------------------------------------------------------


def box_pass_score(track: c.PanelTrack, frame: int) -> float:
    """box area x (1 - border_frac) x aspect over the track's max aspect, for one frame of a track."""
    rows = {r.frame: r for r in track.per_frame}
    row = rows.get(frame)
    if row is None:
        return 0.0
    max_aspect = max((_aspect(r.box) for r in track.per_frame), default=0.0) or 1.0
    return row.box_area_frac * (1.0 - row.border_frac) * (_aspect(row.box) / max_aspect)


def keyframe_candidates(track: c.PanelTrack, th: dict[str, Any]) -> list[int]:
    """The box pass: top keyframe_box_pass_candidates frames by box_pass_score; border_frac over the max excluded."""
    p = th["panels"]
    rows = [r for r in track.per_frame if r.in_car_box and r.border_frac <= float(p["border_frac_keyframe_max"])]
    if not rows:
        return []
    max_aspect = max((_aspect(r.box) for r in track.per_frame), default=0.0) or 1.0
    ranked = sorted(
        rows,
        key=lambda r: (-(r.box_area_frac * (1.0 - r.border_frac) * (_aspect(r.box) / max_aspect)), r.frame),
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
    out: dict[PanelId, c.PanelCoverage] = {}
    for panel in panels:
        pid, noun = panel.panel_id, panel.noun
        mine = [t for t in role_tracks if t.panel_id == pid and t.track_id not in unsure]
        maybe = [t for t in role_tracks if t.track_id in unsure and pid in candidate_slots(t)]
        status = lane_status.get(noun)
        if mine:
            kfs = [k for k in keyframes if k.role == role and k.panel_id == pid and k.track_id not in unsure]
            out[pid] = _tracked_coverage(mine, kfs, noun=noun, pct=pct, fps=fps, cov=cov)
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


def _uncertain_coverage(maybe: list[c.PanelTrack], kfs: list[c.Keyframe], *, fps: float) -> c.PanelCoverage:
    """glimpsed for a slot an undecided track may be, naming the track, its candidates and where its frames are filed."""
    seconds = _seconds_in_car(maybe, fps)
    best_kf = max(kfs, key=lambda k: (k.score.total, k.kf_id), default=None)
    slots: list[PanelId] = []
    for t in maybe:
        slots.extend(s for s in candidate_slots(t) if s not in slots)
    filed = sorted({t.panel_id for t in maybe if t.panel_id is not None}, key=lambda pid: pid.value)
    reason = (
        f"side uncertain: {', '.join(t.track_id for t in maybe)} could be {' or '.join(display_name(s) for s in slots)}; "
        f"{seconds:.1f} s visible, frames filed under {', '.join(display_name(pid) for pid in filed)}"
    )
    return c.PanelCoverage(state=CoverageState.GLIMPSED, seconds=seconds, best_kf=best_kf.kf_id if best_kf else None, reason=reason)


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
    panels, tracks = await asyncio.to_thread(_vote_with_prior, run_id, role, new_tracks, prior_tracks, prior_keyframes, fps, th)
    role_tracks = [t for t in tracks if t.role == role]
    keyframes = await _derive_keyframes(
        run_id, role, panels, role_tracks, kept, records, pts, media_size, fps, decode, blur_keyframe, th
    )
    all_keyframes = _sorted_keyframes(prior_keyframes + keyframes)
    coverage, cov_role = await asyncio.to_thread(
        _coverage_all, run_id, role, panels, tracks, all_keyframes, prior_tracks, prior_coverage,
        lane_status_from_lanes(lanes), fps, th,
    )
    await asyncio.to_thread(_write_derived, run_id, tracks, panels, all_keyframes, coverage)
    return tracks, panels, all_keyframes, cov_role


def _keyframe_plan(
    panels: list[c.Panel], role_tracks: list[c.PanelTrack], th: dict[str, Any]
) -> list[tuple[c.Panel, list[tuple[int, c.PanelTrack]]]]:
    """The box pass per panel: up to keyframe_box_pass_candidates (frame, track) pairs, one per frame. A placeholder
    track never takes keyframes from a slot a decided track of the role holds."""
    limit = int(th["panels"]["keyframe_box_pass_candidates"])
    plan = []
    for panel in panels:
        if panel.noun in UNNAMED_NOUNS:
            continue
        mine = [t for t in role_tracks if t.panel_id == panel.panel_id]
        pool = [t for t in mine if not undecided(t)] or mine
        ranked = sorted(((f, t) for t in pool for f in keyframe_candidates(t, th)), key=lambda ft: (-box_pass_score(ft[1], ft[0]), ft[0]))
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
    car_ids: dict[int, str],
) -> dict[Noun, list[tuple[int, str]]]:
    """Every (frame, id) to decode per lane: the panel at each candidate, the car there, and the car windows (doors),
    plates and faces in those frames. Shared frames are decoded once."""
    wanted: dict[Noun, set[tuple[int, str]]] = {}
    for panel, cands in plan:
        for f, t in cands:
            oid = _object_at(t, f)
            if oid is not None:
                wanted.setdefault(panel.noun, set()).add((f, oid))
            if f in car_ids:
                wanted.setdefault(Noun.CAR, set()).add((f, car_ids[f]))
            companions = (*PRIVACY_NOUNS, Noun.CAR_WINDOW) if panel.noun is Noun.CAR_DOOR else PRIVACY_NOUNS
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
    car_ids: dict[int, str],
    video: Any,
    src: Path,
    pts: Sequence[float],
    media_size: tuple[int, int],
    th: dict[str, Any],
) -> tuple[list[tuple[int, c.KeyframeScore]], dict[int, tuple[c.PanelTrack, np.ndarray, bool, np.ndarray]]]:
    """The mask pass on one panel's candidates, each source frame read once: [(frame, raw score)] and per frame the
    track, the panel mask after window subtraction, window_subtracted and the unblurred frame (kept for the blur)."""
    fallback_fps = float(th["media"]["proxy_fps"])
    times = {f: float(_pts_at(pts, f, fallback_fps)) for f, _ in cands}
    frames = _read_frames(video, src, sorted(set(times.values())))
    scored: list[tuple[int, c.KeyframeScore]] = []
    details: dict[int, tuple[c.PanelTrack, np.ndarray, bool, np.ndarray]] = {}
    for f, t in cands:
        rgb = frames.get(times[f])
        oid = _object_at(t, f)
        obj = lookups[panel.noun].get((f, oid)) if oid is not None else None
        mask = _mask_at(rasters, panel.noun, f, [obj] if obj is not None else [], media_size)
        if rgb is None or mask is None:
            continue
        car_obj = lookups[Noun.CAR].get((f, car_ids[f])) if f in car_ids else None
        car_mask = _mask_at(rasters, Noun.CAR, f, [car_obj] if car_obj is not None else [], media_size)
        window = _mask_at(rasters, Noun.CAR_WINDOW, f, kept_at[Noun.CAR_WINDOW].get(f, []), media_size) if panel.noun is Noun.CAR_DOOR else None
        exclude = [m for n in PRIVACY_NOUNS if (m := _mask_at(rasters, n, f, kept_at[n].get(f, []), media_size)) is not None]
        try:
            score, m2, ws = score_keyframe(t, f, mask=mask, car_mask=car_mask, window_mask=window, frame_rgb=rgb, exclude_masks=exclude, th=th)
        except Exception as exc:
            log.warning("%s frame %d: scoring failed (%s)", panel.panel_id.value, f, exc)
            continue
        scored.append((f, score))
        details[f] = (t, m2, ws, rgb)
    return scored, details


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
) -> list[c.Keyframe]:
    """The box pass, one decode call per lane for every candidate's masks, the mask pass on each candidate's source
    frame (read once, off the event loop), then blur and crop the best frames. A failed read, decode, score or blur
    drops that frame and the next candidate takes its place; KeyframeError when no candidate could be scored or every
    blur failed. Over the decode budget (section 14 item 34) each panel keeps keyframe_top_k_under_budget."""
    p = th["panels"]
    plan = _keyframe_plan(panels, role_tracks, th)
    if not plan:
        return []
    if decode is None:
        raise KeyframeError(f"{role}: no DecodePool, so no keyframe mask can be decoded")
    video = _video_mod()
    src = paths.quarantine_original(run_id, role)
    lookups = {noun: _box_lookup(kept[noun]) for noun in LANE_NOUNS}
    lane_paths = {noun: _lane_path(run_id, role, records[noun], noun) for noun in LANE_NOUNS}
    kept_at = {noun: {fr.frame: fr.objects for fr in kept[noun]} for noun in (Noun.CAR_WINDOW, *PRIVACY_NOUNS)}
    car_ids = {fr.frame: max(fr.objects, key=lambda o: _area(o.box)).id for fr in kept[Noun.CAR] if fr.objects}
    car_boxes = car_boxes_by_frame(kept[Noun.CAR])
    cpu0, wall0 = _decode_cpu(decode), time.perf_counter()
    rasters: dict[tuple[Noun, int, str], np.ndarray] = {}
    for noun, pairs in _wanted_masks(plan, kept_at, car_ids).items():
        got = await asyncio.to_thread(_decode_lane, decode, lane_paths[noun], pairs, lookups[noun])
        rasters.update({(noun, f, oid): raster for (f, oid), raster in got.items()})
    cpu1 = _decode_cpu(decode)
    spent = cpu1 - cpu0 if cpu0 is not None and cpu1 is not None else time.perf_counter() - wall0
    top_k = int(p["keyframe_top_k"])
    if spent > float(p["keyframe_decode_budget_s_per_video"]):
        top_k = int(p["keyframe_top_k_under_budget"])
        # TODO(integrator): section 14 item 34 also wants a ledger flag here; panels writes no ledger rows by contract.
        log.warning("%s %s: %.1f s of mask decoding is over the budget; top %d keyframes per panel", run_id, role, spent, top_k)
    min_gap = float(p["keyframe_min_gap_s"]) * fps
    keyframes: list[c.Keyframe] = []
    n_scored = tried = failed = 0
    last_error = ""
    for panel, cands in plan:
        try:
            scored, details = await asyncio.to_thread(
                _score_panel, panel, cands, rasters, lookups, kept_at, car_ids, video, src, pts, media_size, th
            )
        except Exception as exc:
            last_error = redact(f"{type(exc).__name__}: {exc}")
            log.warning("%s %s %s: keyframe pass failed (%s)", run_id, role, panel.panel_id.value, last_error)
            continue
        n_scored += len(scored)
        for f, score in scored:
            _note_decoded(details[f][0], f, score, th)
        taken: list[int] = []
        for f, score in sorted(normalise_scores(scored, th), key=lambda fs: (-fs[1].total, fs[0])):
            if len(taken) >= top_k:
                break
            if score.total <= 0 or any(abs(f - g) < min_gap for g in taken):
                continue
            tried += 1
            track, mask, ws, rgb = details[f]
            try:
                h_src, w_src = int(rgb.shape[0]), int(rgb.shape[1])
                # stored crops and frames always carry a blurred plate; only the sheet writer keeps it for the run
                # owner under keep_plate (section 8 items 5 and 6)
                blurred, blur = await blur_keyframe(
                    run_id, role, rgb, f_proxy=f, tracks=role_tracks, fps=fps, sizes=(media_size, (w_src, h_src)),
                    keep_plate=False, th=th,
                )
                kf = await asyncio.to_thread(
                    _keyframe_files, run_id, role, f"{role}_{panel.panel_id.value}_kf{len(taken) + 1}", panel, track, f,
                    mask, ws, blurred, blur, score, car_boxes.get(f), media_size, th,
                )
            except Exception as exc:
                failed += 1
                last_error = redact(f"{type(exc).__name__}: {exc}")
                log.warning("%s %s %s frame %d: keyframe dropped (%s)", run_id, role, panel.panel_id.value, f, last_error)
                continue
            taken.append(f)
            keyframes.append(kf)
        details.clear()
    if not n_scored:
        raise KeyframeError(f"{role}: no keyframe candidate could be read and scored; last error: {last_error or 'none'}")
    if tried and failed == tried:
        raise KeyframeError(f"{role}: all {failed} keyframes failed; last error: {last_error}")
    return keyframes


def _note_decoded(track: c.PanelTrack, frame: int, score: c.KeyframeScore, th: dict[str, Any]) -> None:
    """Mark the per_frame row of a decoded candidate: mask_decoded, area_frac and complete."""
    for r in track.per_frame:
        if r.frame == frame:
            r.mask_decoded = True
            r.area_frac = score.area
            r.complete = score.completeness >= float(th["coverage"]["filmed_completeness_min"])
            return
