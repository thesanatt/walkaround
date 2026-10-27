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
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

import numpy as np

from . import contracts as c
from . import paths
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
    """sam_client.prompt_box when it exists, else the same rule: integer edges, padded, clamped half-open."""
    try:
        from . import sam_client

        fn = getattr(sam_client, "prompt_box", None)
    except Exception:  # pragma: no cover - the stub may not import in every environment
        fn = None
    if fn is not None:
        return fn([float(v) for v in box], media_size, pad_px=pad_px)
    from meta_sam_parser import PromptBox

    w, h = media_size
    left = max(0, int(math.floor(float(box[0]))) - pad_px)
    top = max(0, int(math.floor(float(box[1]))) - pad_px)
    right = min(w, int(math.ceil(float(box[2]))) + pad_px)
    bottom = min(h, int(math.ceil(float(box[3]))) + pad_px)
    if right <= left:
        right = min(w, left + 1)
        left = max(0, right - 1)
    if bottom <= top:
        bottom = min(h, top + 1)
        top = max(0, bottom - 1)
    return PromptBox(left=left, top=top, right=right, bottom=bottom)


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
    media_size: tuple[int, int] | None = None,
    pts: Sequence[float] | None = None,
    fps: float | None = None,
    request_id: str = "",
    prompts: list[c.AnchorReprompt] | None = None,
) -> list[c.PanelTrack]:
    """Merge the tracks whose boxes agree with an anchor object (median box IoU >= anchor_absorb_iou_min on shared
    frames) into one track; clip frames map back by frame_offset. Tracks that match no anchor are returned as is."""
    iou_min = float(th["panels"]["anchor_absorb_iou_min"])
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
        out.append(_absorbed_track(matched, anchor_id, a_rows, media_size, pts, fps, request_id, prompts or []))
    out.extend(remaining)
    return sorted(out, key=lambda t: (t.per_frame[0].frame if t.per_frame else 0, t.track_id))


def _absorbed_track(
    matched: list[c.PanelTrack],
    anchor_id: str,
    a_rows: dict[int, c.IndexObject],
    media_size: tuple[int, int] | None,
    pts: Sequence[float] | None,
    fps: float | None,
    request_id: str,
    prompts: list[c.AnchorReprompt],
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
            dt = 1.0 / fps if fps else _row_dt(base.per_frame)
            merges.append(c.MergeRecord(rule=MergeRule.ANCHOR_ABSORB, gap_s=max(0.0, gap_frames * dt)))
        if t.per_frame:
            prev_last = t.per_frame[-1].frame
    sample = next((r for r in base.per_frame if r.box_area_frac > 0), None)
    frame_area = float(media_size[0] * media_size[1]) if media_size else (_area(sample.box) / sample.box_area_frac if sample else 1.0)
    size = media_size or _infer_media_size(sample)
    for f in sorted(a_rows):
        if f in rows:
            continue
        obj = a_rows[f]
        fbox = [float(v) for v in obj.box]
        rows[f] = c.FrameObs(
            frame=f,
            pts=_pts_at(pts, f, fps or 1.0 / _row_dt(base.per_frame)) if pts is not None else _extrapolate_pts(base.per_frame, f),
            box=fbox,
            box_area_frac=_area(fbox) / frame_area if frame_area > 0 else 0.0,
            in_car_box=True,
            c=obj.c,
            border_frac=border_frac(fbox, size) if size else 0.0,
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


def _row_dt(rows: list[c.FrameObs]) -> float:
    deltas = [(b.pts - a.pts) / (b.frame - a.frame) for a, b in zip(rows, rows[1:]) if b.frame > a.frame]
    return statistics.median(deltas) if deltas else 1.0 / 15.0


def _extrapolate_pts(rows: list[c.FrameObs], frame: int) -> float:
    if not rows:
        return 0.0
    dt = _row_dt(rows)
    return rows[0].pts + (frame - rows[0].frame) * dt


def _infer_media_size(sample: c.FrameObs | None) -> tuple[int, int] | None:
    if sample is None or sample.box_area_frac <= 0:
        return None
    frame_area = _area(sample.box) / sample.box_area_frac
    side = int(round(math.sqrt(frame_area * 16.0 / 9.0)))
    return (side, int(round(frame_area / side))) if side > 0 else None


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
            ca = self.cent[t.track_id]
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
    total = votes.R1 + votes.R2 + votes.R3 + votes.R4
    if total == 0 or abs(total) < margin_min:
        return (Side.RIGHT if total > 0 else Side.LEFT), True, 0
    return (Side.RIGHT if total > 0 else Side.LEFT), False, _sign(total)


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


def _seconds_visible(tracks: list[c.PanelTrack], fps: float | None) -> float:
    frames = {r.frame for t in tracks for r in t.per_frame if r.in_car_box}
    if not frames:
        return 0.0
    if fps:
        return len(frames) / fps
    rows = [r for t in tracks for r in t.per_frame]
    return len(frames) * _row_dt(sorted(rows, key=lambda r: r.frame))


def vote_sides(
    tracks: list[c.PanelTrack],
    direction: int | None,
    th: dict[str, Any],
    *,
    fps: float | None = None,
    direction_by_role: dict[str, int | None] | None = None,
) -> tuple[list[c.Panel], list[c.PanelTrack]]:
    """R1 to R4 per lateral track with side_vote_weights; margin under side_vote_margin_min gives side_uncertain.
    Bumpers go front or rear by the light they share frames with. Every canonical PanelId comes back (instance_from
    canonical when no track); the tracks come back with panel_id, side and position filled. SideVotes hold each
    rule's signed weighted vote, positive for right. side_uncertain also covers a lateral panel or bumper whose
    position no rule could decide (it is then filed under front). direction applies to every role unless
    direction_by_role names a role's own walk direction (the two walks may go opposite ways)."""
    p = th["panels"]
    weights = p["side_vote_weights"]
    margin_min = float(p["side_vote_margin_min"])
    floor = float(p["direction_min_px_per_frame"])
    out_tracks = [t.model_copy(deep=True) for t in tracks]
    votes_by_track: dict[str, c.SideVotes] = {}
    uncertain: dict[str, bool] = {}
    by_role: dict[str, list[c.PanelTrack]] = {}
    for t in out_tracks:
        by_role.setdefault(t.role, []).append(t)
    for role, role_tracks in by_role.items():
        role_direction = direction if direction_by_role is None else direction_by_role.get(role, direction)
        ctx = _RoleContext(role_tracks, role_direction, floor)
        sides: dict[str, Side] = {}
        signs: dict[str, int] = {}
        suggested: dict[str, dict[str, Position]] = {}
        for t in role_tracks:
            if t.noun not in LATERAL_NOUNS:
                continue
            votes, positions = _side_votes_for(ctx, t, weights)
            side, unc, fs = _side_from_votes(votes, margin_min)
            votes_by_track[t.track_id] = votes
            uncertain[t.track_id] = unc
            sides[t.track_id] = side
            signs[t.track_id] = fs
            suggested[t.track_id] = positions
        for t in role_tracks:
            if t.noun in LATERAL_NOUNS:
                side = sides[t.track_id]
                position = ctx.pairwise_position(t, side, signs[t.track_id], sides)
                if position is Position.NONE:
                    position = suggested[t.track_id].get("R1", Position.NONE)
                if position is Position.NONE:
                    position = suggested[t.track_id].get("R4", Position.NONE)
                if position is Position.NONE:
                    kind = ctx.nearest_anchor_kind(t)
                    position = Position.FRONT if kind == "F" else Position.REAR if kind == "R" else Position.NONE
                if position is Position.NONE:
                    position = Position.FRONT
                    uncertain[t.track_id] = True
                t.side, t.position = side, position
                t.panel_id = _lateral_panel_id(t.noun, side, position)
            elif t.noun is Noun.BUMPER:
                position = _bumper_position(ctx, t)
                votes_by_track[t.track_id] = c.SideVotes()
                uncertain[t.track_id] = position is Position.NONE
                if position is Position.NONE:
                    position = Position.FRONT
                t.side, t.position = Side.NONE, position
                t.panel_id = PanelId.FRONT_BUMPER if position is Position.FRONT else PanelId.REAR_BUMPER
            elif t.noun in SINGLE_SLOT:
                votes_by_track[t.track_id] = c.SideVotes()
                uncertain[t.track_id] = False
                t.side, t.position = Side.NONE, Position.NONE
                t.panel_id = SINGLE_SLOT[t.noun]
            else:
                t.panel_id, t.side, t.position = None, Side.NONE, Position.NONE
    panels: list[c.Panel] = []
    merge_gap_s = float(p["merge_gap_max_s"])
    for pid in PanelId:
        mine = [t for t in out_tracks if t.panel_id == pid]
        noun = PANEL_NOUN[pid]
        side, position = _panel_side_position(pid)
        unc = any(uncertain.get(t.track_id, False) for t in mine)
        if not mine:
            front_sign = None
        elif noun in LATERAL_NOUNS:
            front_sign = None if unc else (1 if side is Side.RIGHT else -1)
        else:
            front_sign = END_ON_FRONT_SIGN.get(pid)
        per_role: dict[str, c.PanelRole] = {}
        for role in sorted({t.role for t in mine}, key=_role_order):
            rts = [t for t in mine if t.role == role]
            per_role[role] = c.PanelRole(
                track_ids=[t.track_id for t in rts],
                windows=_windows(rts, merge_gap_s),
                seconds_visible=_seconds_visible(rts, fps),
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
    return panels, out_tracks


def _role_order(role: str) -> tuple[int, str]:
    return ({"pickup": 0, "return": 1}.get(role, 2), role)


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
    """The 5.4 states for one role. filmed needs a qualifying keyframe; glimpsed compares the track's median c with
    the run's 25th percentile for the noun (area_only basis under c_present_floor); not_filmed is a completed or
    zero-match lane with no usable track for the slot; not_run a failed, incomplete or missing lane, or any panel
    when the car lane itself failed (nothing passed the car filter); not_seen a slot with no track in any walk
    while the noun was seen elsewhere in this walk."""
    cov = th["coverage"]
    role_tracks = [t for t in tracks if t.role == role]
    c_by_noun: dict[Noun, list[float]] = {}
    for t in tracks:
        for r in t.per_frame:
            if r.c is not None:
                c_by_noun.setdefault(t.noun, []).append(r.c)
    pct = {n: _percentile(v, float(cov["glimpsed_c_percentile"])) for n, v in c_by_noun.items()}
    car_status = lane_status.get(Noun.CAR)
    car_down = car_status in (LaneStatus.FAILED, LaneStatus.INCOMPLETE)
    out: dict[PanelId, c.PanelCoverage] = {}
    for panel in panels:
        pid, noun = panel.panel_id, panel.noun
        mine = [t for t in role_tracks if t.panel_id == pid]
        kfs = [k for k in keyframes if k.role == role and k.panel_id == pid]
        best_kf = max(kfs, key=lambda k: (k.score.total, k.kf_id), default=None)
        seconds = len({r.frame for t in mine for r in t.per_frame if r.in_car_box}) / fps if fps else 0.0
        basis = StateBasis.C
        status = lane_status.get(noun)
        if car_down:
            state = CoverageState.NOT_RUN
            reason = f"car lane {car_status.value}: the car filter could not run, so no panel of this walk was read"
        elif mine:
            filmed = [k for k in kfs if _is_filmed(k, cov)]
            if filmed:
                k = max(filmed, key=lambda k: (k.score.total, k.kf_id))
                state = CoverageState.FILMED
                reason = f"keyframe {k.kf_id}: completeness {k.score.completeness:.2f}, area {k.score.area:.3f}, sharpness {k.score.sharpness:.2f}"
            else:
                best_track = max(mine, key=lambda t: (len(t.per_frame), t.track_id))
                if best_track.c_present_share >= float(cov["c_present_floor"]):
                    present = [r.c for r in best_track.per_frame if r.c is not None]
                    med = statistics.median(present) if present else 0.0
                    glimpsed = bool(present) and med >= pct.get(noun, 0.0)
                    reason = f"median c {med:.3f} against the run's {int(cov['glimpsed_c_percentile'])}th percentile {pct.get(noun, 0.0):.3f} for {noun.value}"
                else:
                    basis = StateBasis.AREA_ONLY
                    floor = float(cov["area_only_box_area_frac_min"])
                    glimpsed = any(r.box_area_frac >= floor for t in mine for r in t.per_frame if r.in_car_box)
                    reason = f"c present on {best_track.c_present_share:.0%} of frames, area-only basis: box area {'reaches' if glimpsed else 'never reaches'} {floor:.0%}"
                state = CoverageState.GLIMPSED if glimpsed else CoverageState.NOT_FILMED
                if glimpsed:
                    reason = f"{seconds:.1f} s visible; " + reason
        elif status is None or status in (LaneStatus.FAILED, LaneStatus.INCOMPLETE):
            state = CoverageState.NOT_RUN
            reason = f"{noun.value} lane {status.value if status else 'not run'}"
        else:
            noun_in_role = any(t.noun == noun for t in role_tracks)
            seen_elsewhere = any(t.panel_id == pid for t in tracks if t.role != role)
            if seen_elsewhere or not noun_in_role:
                state = CoverageState.NOT_FILMED
                reason = f"{noun.value} lane {status.value}, no track for {pid.value} in this walk"
            else:
                state = CoverageState.NOT_SEEN
                reason = f"{noun.value} seen in this walk, no track for the {pid.value} slot in any walk"
        out[pid] = c.PanelCoverage(
            state=state, state_basis=basis, seconds=seconds, best_kf=best_kf.kf_id if best_kf else None, reason=reason
        )
    return out


def display_name(pid: PanelId) -> str:
    return pid.value.replace("_", " ")


def plan_topup(coverage: c.Coverage, th: dict[str, Any]) -> c.Coverage:
    """missing = every panel not filmed in the latest role, wheels first, then lower bumpers, then the rest;
    nouns_to_rerun = their nouns plus car, license plate and face; the odometer is asked once."""
    cov = coverage.model_copy(deep=True)
    always = [Noun(x) for x in th["nouns"]["topup_always"]]
    if not cov.per_role:
        cov.missing = []
        cov.nouns_to_rerun = list(always)
        cov.topup_text = ""
        return cov
    role = sorted(cov.per_role, key=_role_order)[-1]
    states = cov.per_role[role]
    not_filmed = [pid for pid in PanelId if pid in states and states[pid].state is not CoverageState.FILMED]
    groups = {
        "wheels": [pid for pid in not_filmed if pid in WHEEL_PANELS],
        "lower_bumpers": [pid for pid in not_filmed if pid in LOWER_BUMPERS],
    }
    groups["rest"] = [pid for pid in not_filmed if pid not in groups["wheels"] and pid not in groups["lower_bumpers"]]
    ordered: list[PanelId] = []
    for key in th["coverage"]["topup_order"]:
        ordered.extend(groups.get(key, []))
    cov.missing = ordered
    nouns: list[Noun] = []
    for pid in ordered:
        if PANEL_NOUN[pid] not in nouns:
            nouns.append(PANEL_NOUN[pid])
    cov.nouns_to_rerun = nouns + [n for n in always if n not in nouns]
    parts = []
    for pid in ordered:
        st = states[pid]
        tag = st.state.value.replace("_", " ")
        if st.state is CoverageState.GLIMPSED:
            tag = f"{tag}, {st.seconds:.1f} s"
        parts.append(f"{display_name(pid)} ({tag})")
    if ordered:
        text = "Film these again, in this order: " + "; ".join(parts) + "."
        text += f" The app then re-runs {len(cov.nouns_to_rerun)} of {len(LANE_NOUNS)} nouns: " + ", ".join(n.value for n in cov.nouns_to_rerun) + "."
    else:
        text = f"Every panel is filmed in the {role} walk."
    if th["coverage"].get("odometer_ask_once", True) and not cov.odometer.asked:
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


def _geometry_mod():
    from . import geometry

    return geometry


def _video_mod():
    from . import video

    return video


def _paste(raster: np.ndarray, box: Sequence[int], ref: c.MaskRef, media_size: tuple[int, int]) -> np.ndarray:
    """A bool (H, W) frame with the box-local raster at (box x1 + ref.x, box y1 + ref.y)."""
    w, h = media_size
    out = np.zeros((h, w), dtype=bool)
    r = np.asarray(raster).astype(bool).reshape(ref.h, ref.w)
    x0, y0 = int(box[0]) + ref.x, int(box[1]) + ref.y
    x1, y1 = min(w, x0 + ref.w), min(h, y0 + ref.h)
    if x1 <= max(0, x0) or y1 <= max(0, y0):
        return out
    out[max(0, y0) : y1, max(0, x0) : x1] = r[max(0, -y0) : y1 - y0, max(0, -x0) : x1 - x0]
    return out


def _decode_frames(
    decode: Any,
    lane_path: Path | None,
    wanted: list[tuple[int, str]],
    lookup: dict[tuple[int, str], c.IndexObject],
    media_size: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Decode the masks of (frame, id) pairs through the pool; frames whose objects share a frame are unioned."""
    if lane_path is None:
        return {}
    pairs = [(f, oid, lookup[(f, oid)]) for f, oid in wanted if (f, oid) in lookup and lookup[(f, oid)].mask is not None]
    if not pairs:
        return {}
    rasters = decode.decode(lane_path, [obj.mask for _, _, obj in pairs])
    out: dict[int, np.ndarray] = {}
    for (f, _oid, obj), raster in zip(pairs, rasters):
        pasted = _paste(raster, obj.box, obj.mask, media_size)  # type: ignore[arg-type]
        out[f] = out[f] | pasted if f in out else pasted
    return out


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
    out: dict[Noun, LaneStatus] = {}
    for noun in LANE_NOUNS:
        p = paths.lane_summary_path(run_id, role, noun)
        if p.is_file():
            try:
                out[noun] = c.read_json(p, c.LaneRecord).status
            except Exception:
                continue
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
    Writes tracks.json, panels.json, keyframes.json and coverage.json through contracts.write_json, keeping the
    entries of the other roles already on disk. Returns everything written plus this role's coverage."""
    records = {noun: (lanes.get(noun_slug(noun)) or (None, []))[0] for noun in LANE_NOUNS}
    frames_by_noun = {noun: (lanes.get(noun_slug(noun)) or (None, []))[1] for noun in LANE_NOUNS}
    car_boxes = car_boxes_by_frame(frames_by_noun[Noun.CAR])
    kept: dict[Noun, list[c.IndexFrame]] = {}
    new_tracks: list[c.PanelTrack] = []
    for noun in LANE_NOUNS:
        frames = frames_by_noun[noun]
        kept[noun] = frames if noun in PRIVACY_NOUNS else filter_objects(frames, noun, car_boxes, media_size, th)
        frags = build_fragments(kept[noun], fps=fps, th=th)
        groups = merge_fragments(frags, kept[noun], fps=fps, th=th)
        new_tracks.extend(build_tracks(role, noun, groups, kept[noun], pts, car_boxes, media_size, th=th))
    prior_tracks = [t for t in _read_list(paths.tracks_json(run_id), c.PanelTrack) if t.role != role]
    prior_keyframes = [k for k in _read_list(paths.keyframes_json(run_id), c.Keyframe) if k.role != role]
    prior_coverage = c.read_json(paths.coverage_json(run_id), c.Coverage) if paths.coverage_json(run_id).is_file() else c.Coverage()
    direction_by_role: dict[str, int | None] = {role: walk_direction(new_tracks, th)}
    for other in sorted({t.role for t in prior_tracks}, key=_role_order):
        direction_by_role[other] = walk_direction([t for t in prior_tracks if t.role == other], th)
    panels, tracks = vote_sides(
        prior_tracks + new_tracks, direction_by_role[role], th, fps=fps, direction_by_role=direction_by_role
    )
    role_tracks = [t for t in tracks if t.role == role]
    keep_plate = False
    if paths.run_json(run_id).is_file():
        try:
            keep_plate = c.read_json(paths.run_json(run_id), c.RunManifest).keep_plate
        except Exception:
            keep_plate = False
    keyframes = await _derive_keyframes(
        run_id, role, panels, role_tracks, kept, records, pts, media_size, fps, decode, blur_keyframe, keep_plate, th
    )
    all_keyframes = prior_keyframes + keyframes
    lane_status = lane_status_from_lanes(lanes)
    cov_role = coverage_for_role(panels, tracks, all_keyframes, role=role, fps=fps, lane_status=lane_status, th=th)
    per_role: dict[str, dict[PanelId, c.PanelCoverage]] = {}
    for other in sorted({t.role for t in prior_tracks} | set(prior_coverage.per_role), key=_role_order):
        if other == role:
            continue
        status_other = _lane_status_of_role(run_id, other)
        per_role[other] = coverage_for_role(panels, tracks, all_keyframes, role=other, fps=fps, lane_status=status_other, th=th)
    per_role[role] = cov_role
    for panel in panels:
        for r, states in per_role.items():
            pc = states[panel.panel_id]
            pr = panel.per_role.get(r) or c.PanelRole()
            pr.state, pr.state_basis, pr.reason = pc.state, pc.state_basis, pc.reason
            pr.keyframe_ids = [k.kf_id for k in all_keyframes if k.role == r and k.panel_id == panel.panel_id]
            panel.per_role[r] = pr
    coverage = c.Coverage(per_role=per_role, odometer=prior_coverage.odometer, rerun_seconds=prior_coverage.rerun_seconds)
    coverage = plan_topup(coverage, th)
    c.write_json(paths.tracks_json(run_id), tracks)
    c.write_json(paths.panels_json(run_id), panels)
    c.write_json(paths.keyframes_json(run_id), all_keyframes)
    c.write_json(paths.coverage_json(run_id), coverage)
    return tracks, panels, all_keyframes, cov_role


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
    keep_plate: bool,
    th: dict[str, Any],
) -> list[c.Keyframe]:
    p = th["panels"]
    geometry, video = _geometry_mod(), _video_mod()
    src_path = paths.quarantine_original(run_id, role)
    lookups = {noun: _box_lookup(kept[noun]) for noun in LANE_NOUNS}
    lane_paths = {noun: _lane_path(run_id, role, records[noun], noun) for noun in LANE_NOUNS}
    car_boxes = car_boxes_by_frame(kept[Noun.CAR])
    car_ids = {fr.frame: max(fr.objects, key=lambda o: _area(o.box)).id for fr in kept[Noun.CAR] if fr.objects}
    keyframes: list[c.Keyframe] = []
    fallback_fps = float(th["media"]["proxy_fps"])
    for panel in panels:
        mine = [t for t in role_tracks if t.panel_id == panel.panel_id]
        if not mine or panel.noun in UNNAMED_NOUNS:
            continue
        cands = sorted(
            ((f, t) for t in mine for f in keyframe_candidates(t, th)),
            key=lambda ft: (-box_pass_score(ft[1], ft[0]), ft[0]),
        )[: int(p["keyframe_box_pass_candidates"])]
        if not cands:
            continue
        wanted = [(f, _object_at(t, f)) for f, t in cands]
        panel_masks = _decode_frames(decode, lane_paths[panel.noun], [(f, o) for f, o in wanted if o], lookups[panel.noun], media_size)
        frames = [f for f, _ in cands]
        car_masks = _decode_frames(decode, lane_paths[Noun.CAR], [(f, car_ids[f]) for f in frames if f in car_ids], lookups[Noun.CAR], media_size)
        window_masks: dict[int, np.ndarray] = {}
        if panel.noun is Noun.CAR_DOOR:
            win_pairs = [(fr.frame, o.id) for fr in kept[Noun.CAR_WINDOW] if fr.frame in frames for o in fr.objects]
            window_masks = _decode_frames(decode, lane_paths[Noun.CAR_WINDOW], win_pairs, lookups[Noun.CAR_WINDOW], media_size)
        exclude: dict[int, list[np.ndarray]] = {f: [] for f in frames}
        for noun in PRIVACY_NOUNS:
            pairs = [(fr.frame, o.id) for fr in kept[noun] if fr.frame in frames for o in fr.objects]
            for f, m in _decode_frames(decode, lane_paths[noun], pairs, lookups[noun], media_size).items():
                exclude[f].append(m)
        scored: list[tuple[int, c.KeyframeScore]] = []
        details: dict[int, tuple[c.PanelTrack, np.ndarray, bool]] = {}
        for f, t in cands:
            mask = panel_masks.get(f)
            if mask is None:
                continue
            rgb = video.read_frame_at(src_path, _pts_at(pts, f, fallback_fps))
            score, m2, ws = score_keyframe(
                t, f, mask=mask, car_mask=car_masks.get(f), window_mask=window_masks.get(f), frame_rgb=rgb,
                exclude_masks=exclude.get(f, []), th=th,
            )
            scored.append((f, score))
            details[f] = (t, m2, ws)
            _note_decoded(t, f, score, th)
        normed = normalise_scores(scored, th)
        by_frame = dict(normed)
        for n, f in enumerate(select_keyframes(normed, fps=fps, th=th)):
            t, m2, ws = details[f]
            rgb = video.read_frame_at(src_path, _pts_at(pts, f, fallback_fps))
            h_src, w_src = int(rgb.shape[0]), int(rgb.shape[1])
            blurred, blur = await blur_keyframe(
                run_id, role, rgb, f_proxy=f, tracks=role_tracks, fps=fps, sizes=(media_size, (w_src, h_src)),
                keep_plate=keep_plate, th=th,
            )
            kf_id = f"{role}_{panel.panel_id.value}_kf{n + 1}"
            row = next(r for r in t.per_frame if r.frame == f)
            box_src = _scale_box(row.box, media_size, (w_src, h_src))
            crop = geometry.crop_geometry(box_src, (w_src, h_src), pad_frac=float(p["crop_pad_frac"]), long_edge=int(p["crop_long_edge_px"]))
            crop_file = paths.crop_path(run_id, role, kf_id)
            sha = video.write_jpeg(video.cut_crop(blurred, crop), crop_file, quality=int(th["spark"]["jpeg_quality"]))
            crop = crop.model_copy(update={"path": paths.rel(run_id, crop_file), "sha256": sha})
            mask_src = _fit_mask(m2, (h_src, w_src))
            mask_crop = _mask_to_crop(mask_src, crop)
            mask_file = paths.crop_mask_png_path(run_id, role, kf_id)
            video.write_png_mask(mask_crop, mask_file)
            moments = geometry.moments_from_mask(mask_crop) if mask_crop.any() else None
            car_box = car_boxes.get(f)
            if car_box is not None:
                cb = [int(round(v)) for v in _scale_box(car_box, media_size, (w_src, h_src))]
                frame_crop = blurred[max(0, cb[1]) : min(h_src, cb[3]), max(0, cb[0]) : min(w_src, cb[2])]
                if frame_crop.size:
                    video.write_jpeg(frame_crop, paths.keyframe_frame_path(run_id, role, kf_id), quality=int(th["spark"]["jpeg_quality"]))
            keyframes.append(
                c.Keyframe(
                    kf_id=kf_id,
                    role=role,
                    panel_id=panel.panel_id,
                    track_id=t.track_id,
                    f_proxy=f,
                    pts=row.pts,
                    score=by_frame[f],
                    aspect_ratio=_mask_aspect(m2),
                    box_src=box_src,
                    crop=crop,
                    mask_png=paths.rel(run_id, mask_file),
                    moments=moments,
                    front_sign=panel.front_sign,
                    window_subtracted=ws,
                    blur=blur,
                )
            )
    return keyframes


def _note_decoded(track: c.PanelTrack, frame: int, score: c.KeyframeScore, th: dict[str, Any]) -> None:
    """Mark the per_frame row of a decoded candidate: mask_decoded, area_frac and complete."""
    for r in track.per_frame:
        if r.frame == frame:
            r.mask_decoded = True
            r.area_frac = score.area
            r.complete = score.completeness >= float(th["coverage"]["filmed_completeness_min"])
            return
