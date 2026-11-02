"""Marks: Spark's grid output mapped to crop, source and whitened coordinates, then deduplicated within one walk
(docs/ARCHITECTURE.md 5.6 and 5.7, docs/CONTRACTS.md M6).

Mapping chain: grid (0..1000 over the sent image) -> sent pixels (spark_client.grid_to_pixels) -> crop pixels
(geometry.sent_to_crop) -> source pixels (geometry.crop_to_src) -> whitened units (geometry.whiten with the
keyframe's own moments). Caps set here are booleans for match.verdict(): K2 single frame, K3 tiny, K4 uncertain
type, K8 count mode only, K9 droplets or spray, K12 low confidence. No verdict is decided here and no mark of the
other walk is ever read; the only model calls are through spark_client, which writes the ledger rows. Each role's
view spread is kept in derived/sigma_view.json and the pooled sigma_view goes to calibration.json.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np

from . import geometry, paths
from .contracts import (
    CAPPED_MARK_KINDS,
    LOWER_BUMPERS,
    WHEEL_PANELS,
    Calibration,
    Confidence,
    Crop,
    FrameUsed,
    GridPoint,
    InputRef,
    Keyframe,
    Mark,
    MarkKind,
    MarkSource,
    MarksStatus,
    OffPanel,
    PanelId,
    SizeClass,
    SparkMark,
    SparkMarksAnswer,
    SparkPurpose,
    SparkRequestRecord,
    WhitenedMark,
    read_json,
    read_json_list,
    write_json,
)

if TYPE_CHECKING:
    from .spark_client import CountResult, MarksResult

Box4 = list[float]
Point2 = list[float]
Pos = tuple[float, float]

PSEUDO_RAISE_TOL = 1e-5  # above the 6-place rounding of write_json: a smaller tau_pre gap is no pseudo-pair raise

OUTLINE_RGB = (255, 0, 255)  # magenta: not a car paint colour

_FAMILY: dict[PanelId, str] = {
    **{p: "door" for p in PanelId if p.value.endswith("_door")},
    **{p: "bumper" for p in LOWER_BUMPERS},
    **{p: "wheel" for p in WHEEL_PANELS},
    PanelId.HOOD: "hood",
}


def landmarks_for(panel_id: PanelId, th: dict) -> list[str]:
    """The closed landmark list of the panel's family (door, bumper, wheel, hood), at most landmarks_max names."""
    family = _FAMILY.get(PanelId(panel_id))
    if family is None:
        return []
    names = list(th["spark"]["landmarks"].get(family, []))
    return names[: int(th["spark"]["landmarks_max"])]


def draw_outline(crop_rgb: np.ndarray, bbox_crop: Box4, *, width_px: int = 2) -> np.ndarray:
    """A copy of the crop with a thin rectangle around bbox_crop, for the second look."""
    out = np.ascontiguousarray(np.asarray(crop_rgb).copy())
    h, w = out.shape[:2]
    x1 = min(max(int(round(bbox_crop[0])), 0), w - 1)
    y1 = min(max(int(round(bbox_crop[1])), 0), h - 1)
    x2 = min(max(int(round(bbox_crop[2])) - 1, x1), w - 1)
    y2 = min(max(int(round(bbox_crop[3])) - 1, y1), h - 1)
    cv2.rectangle(out, (x1, y1), (x2, y2), OUTLINE_RGB, int(width_px))
    return out


# Mapping -------------------------------------------------------------------------------------------------------------


def _sent_size(result: Any) -> tuple[int, int]:
    return int(result.sent_width), int(result.sent_height)


def _grid_to_crop(x: int, y: int, result: Any, crop: Crop) -> Point2:
    from . import spark_client

    sw, sh = _sent_size(result)
    px, py = spark_client.grid_to_pixels(int(x), int(y), sw, sh)
    return geometry.sent_to_crop((px, py), (sw, sh), (crop.w, crop.h))


def map_spark_mark(sm: SparkMark, result: "MarksResult", crop: Crop) -> tuple[Box4, Point2, Box4, Point2]:
    """(bbox_crop, point_crop, bbox_src, point_src) for one marks_v1 mark. The bbox is half-open on the right and
    bottom, so the far corner is one sent pixel past the last pixel Spark named."""
    from . import spark_client

    sw, sh = _sent_size(result)
    point_crop = _grid_to_crop(sm.point.x, sm.point.y, result, crop)
    x1, y1 = spark_client.grid_to_pixels(sm.bbox.x_min, sm.bbox.y_min, sw, sh)
    x2, y2 = spark_client.grid_to_pixels(sm.bbox.x_max, sm.bbox.y_max, sw, sh)
    c1 = geometry.sent_to_crop((x1, y1), (sw, sh), (crop.w, crop.h))
    c2 = geometry.sent_to_crop((max(x2, x1) + 1, max(y2, y1) + 1), (sw, sh), (crop.w, crop.h))
    bbox_crop = [c1[0], c1[1], c2[0], c2[1]]
    return bbox_crop, point_crop, geometry.box_crop_to_src(bbox_crop, crop), geometry.crop_to_src(point_crop, crop)


def _request_id(result: Any) -> str | None:
    record = getattr(result, "record", None)
    return getattr(record, "request_id", None) if record is not None else None


def _long_side(box: Box4) -> float:
    return max(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


def marks_from_answer(answer: SparkMarksAnswer, result: "MarksResult", kf: Keyframe, crop: Crop, *, th: dict) -> list[Mark]:
    """Spark box marks of one crop as Mark rows (source spark_box) with K3 (px), K4 and K12 set; ids <kf_id>_m<n>."""
    small_px = float(th["dedup"]["small_mark_px"])
    request_id = _request_id(result)
    marks: list[Mark] = []
    for n, sm in enumerate(answer.marks, 1):
        bbox_crop, point_crop, bbox_src, point_src = map_spark_mark(sm, result, crop)
        size_px = _long_side(bbox_src)
        caps: list[str] = []
        if size_px < small_px:
            caps.append("K3")
        if sm.type in CAPPED_MARK_KINDS:
            caps.append("K4")
        if sm.confidence == Confidence.LOW:
            caps.append("K12")
        marks.append(
            Mark(
                mark_id=f"{kf.kf_id}_m{n}",
                role=kf.role,
                panel_id=kf.panel_id,
                kf_id=kf.kf_id,
                source=MarkSource.SPARK_BOX,
                type=sm.type,
                confidence=sm.confidence,
                size_class=sm.size_class,
                description=sm.note,
                grid_point=sm.point,
                grid_bbox=sm.bbox,
                bbox_crop=bbox_crop,
                point_crop=point_crop,
                bbox_src=bbox_src,
                point_src=point_src,
                size_px_src=size_px,
                spark_request_id=request_id,
                caps=caps,
            )
        )
    return marks


# Whitening ----------------------------------------------------------------------------------------------------------


def _with_cap(caps: list[str], cap: str, on: bool) -> list[str]:
    present = cap in caps
    if on and not present:
        return [*caps, cap]
    if not on and present:
        return [c for c in caps if c != cap]
    return list(caps)


def _size_units_moment(bbox_crop: Box4 | None, kf: Keyframe) -> float | None:
    """Long side of the bbox in whitened units: the longer of the two whitened edge vectors."""
    if bbox_crop is None:
        return None
    w = geometry.whitening_matrix(kf.moments, kf.front_sign)
    bw, bh = float(bbox_crop[2]) - float(bbox_crop[0]), float(bbox_crop[3]) - float(bbox_crop[1])
    return float(max(np.linalg.norm(w @ np.array([bw, 0.0])), np.linalg.norm(w @ np.array([0.0, bh]))))


def whiten_marks(marks: list[Mark], kf: Keyframe, *, th: dict, mask: np.ndarray | None = None) -> tuple[list[Mark], dict[str, WhitenedMark]]:
    """Fill w, r and size_units with the keyframe's own moments (wheel coordinates on a wheel panel) and set K3 by
    units. Marks whose keyframe has no moments or front_sign come back unchanged with no whitened row, and so do
    the marks of a wheel keyframe without its mask: match reads every wheel w as (radius, angle), so a wheel mark
    is never placed in the moment frame."""
    wheel = kf.panel_id in WHEEL_PANELS
    if kf.moments is None or kf.front_sign is None or (wheel and (mask is None or not np.asarray(mask).any())):
        return list(marks), {}
    small_units = float(th["dedup"]["small_mark_units"])
    radius_px = geometry.inscribed_radius(mask) if wheel else 0.0
    out: list[Mark] = []
    whitened: dict[str, WhitenedMark] = {}
    for m in marks:
        if m.kf_id != kf.kf_id or m.point_crop is None:
            out.append(m)
            continue
        if wheel:
            u, v = geometry.wheel_coords(m.point_crop, mask)
            r = u
            size_units = None if m.bbox_crop is None else _long_side(m.bbox_crop) / radius_px
            frame = FrameUsed.WHEEL
        else:
            u, v = geometry.whiten((m.point_crop[0], m.point_crop[1]), kf.moments, kf.front_sign)
            r = geometry.radius((u, v))
            size_units = _size_units_moment(m.bbox_crop, kf)
            frame = FrameUsed.MOMENT
        tiny_px = m.size_px_src is not None and m.size_px_src < float(th["dedup"]["small_mark_px"])
        tiny_units = size_units is not None and size_units < small_units
        caps = _with_cap(m.caps, "K3", tiny_px or tiny_units)
        updated = m.model_copy(update={"w": [u, v], "r": r, "size_units": size_units, "caps": caps})
        out.append(updated)
        whitened[m.mark_id] = WhitenedMark(mark_id=m.mark_id, role=m.role, panel_id=m.panel_id, kf_id=m.kf_id, u=u, v=v, r=r, frame_used=frame)
    return out, whitened


def whitened_from_marks(marks: list[Mark]) -> dict[str, WhitenedMark]:
    """Rebuild the whitened rows from marks.json rows that carry w and r."""
    out: dict[str, WhitenedMark] = {}
    for m in marks:
        if m.w is None or m.r is None:
            continue
        frame = FrameUsed.WHEEL if m.panel_id in WHEEL_PANELS else FrameUsed.MOMENT
        out[m.mark_id] = WhitenedMark(mark_id=m.mark_id, role=m.role, panel_id=m.panel_id, kf_id=m.kf_id, u=m.w[0], v=m.w[1], r=m.r, frame_used=frame)
    return out


# Count mode ----------------------------------------------------------------------------------------------------------


def count_union(box_marks: list[Mark], count: "CountResult", result: Any, kf: Keyframe, crop: Crop, whitened: dict[str, WhitenedMark], *, th: dict) -> list[Mark]:
    """Count points farther than count_mode_new_point_min_units from every box mark become spark_count marks with
    K8 (type unsure and confidence low, so K4 and K12 follow). Positions are compared in the moment frame; on a
    wheel panel the new marks carry no w, so whiten_marks places them in the wheel frame."""
    if kf.moments is None or kf.front_sign is None:
        return []
    wheel = kf.panel_id in WHEEL_PANELS
    min_units = float(th["dedup"]["count_mode_new_point_min_units"])
    existing: list[tuple[float, float]] = []
    for m in box_marks:
        wm = whitened.get(m.mark_id)
        if wm is not None and wm.frame_used == FrameUsed.MOMENT:
            existing.append((wm.u, wm.v))
        elif m.point_crop is not None:
            existing.append(geometry.whiten((m.point_crop[0], m.point_crop[1]), kf.moments, kf.front_sign))
    out: list[Mark] = []
    for n, gp in enumerate(count.points, 1):
        point_crop = _grid_to_crop(gp.x, gp.y, result, crop)
        u, v = geometry.whiten((point_crop[0], point_crop[1]), kf.moments, kf.front_sign)
        if any(math.hypot(u - eu, v - ev) <= min_units for eu, ev in existing):
            continue
        existing.append((u, v))
        out.append(
            Mark(
                mark_id=f"{kf.kf_id}_c{n}",
                role=kf.role,
                panel_id=kf.panel_id,
                kf_id=kf.kf_id,
                source=MarkSource.SPARK_COUNT,
                type=MarkKind.UNSURE,
                confidence=Confidence.LOW,
                size_class=SizeClass.S,
                grid_point=GridPoint(x=gp.x, y=gp.y),
                point_crop=point_crop,
                point_src=geometry.crop_to_src(point_crop, crop),
                w=None if wheel else [u, v],
                r=None if wheel else geometry.radius((u, v)),
                spark_request_id=_request_id(count),
                caps=["K4", "K8", "K12"],
            )
        )
    return out


# Within-walk dedup ------------------------------------------------------------------------------------------------


def _pos(m: Mark, whitened: dict[str, WhitenedMark]) -> Pos | None:
    wm = whitened.get(m.mark_id)
    if wm is not None:
        return wm.u, wm.v
    if m.w is not None:
        return float(m.w[0]), float(m.w[1])
    return None


def _dist(a: Pos, b: Pos) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _plain_offset(p: Pos, ref: Pos) -> tuple[float, float]:
    return p[0] - ref[0], p[1] - ref[1]


def _metric(panel_id: PanelId) -> Callable[[Pos, Pos], float]:
    """The panel's distance: plain on (u, v), the wrapped polar distance of match on a wheel's (radius, angle)."""
    return geometry.wheel_distance if panel_id in WHEEL_PANELS else _dist


def _offset(panel_id: PanelId) -> Callable[[Pos, Pos], tuple[float, float]]:
    return geometry.wheel_offset if panel_id in WHEEL_PANELS else _plain_offset


def _kf_quality(kf: Keyframe | None) -> float:
    return kf.score.total if kf is not None else 0.0


def _cluster_across_keyframes(marks: list[Mark], pos: dict[str, Pos], radius_units: float, dist: Callable[[Pos, Pos], float] = _dist) -> list[list[Mark]]:
    """Single linkage under radius_units with at most one mark per keyframe in a cluster; input order kept."""
    parent = list(range(len(marks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    kfs: list[set[str]] = [{m.kf_id} for m in marks]
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            if marks[i].kf_id == marks[j].kf_id:
                continue
            d = dist(pos[marks[i].mark_id], pos[marks[j].mark_id])
            if d <= radius_units:
                pairs.append((d, i, j))
    for _, i, j in sorted(pairs):
        ri, rj = find(i), find(j)
        if ri == rj or kfs[ri] & kfs[rj]:
            continue
        parent[rj] = ri
        kfs[ri] |= kfs[rj]
    groups: dict[int, list[Mark]] = {}
    for i, m in enumerate(marks):
        groups.setdefault(find(i), []).append(m)
    return [groups[k] for k in sorted(groups, key=lambda k: min(marks.index(m) for m in groups[k]))]


def _medoid(cluster: list[Mark], pos: dict[str, Pos], kf_by_id: dict[str, Keyframe], dist: Callable[[Pos, Pos], float] = _dist) -> Mark:
    def key(m: Mark) -> tuple[float, float, str]:
        total = sum(dist(pos[m.mark_id], pos[o.mark_id]) for o in cluster if o is not m)
        return (total, -_kf_quality(kf_by_id.get(m.kf_id)), m.mark_id)

    return min(cluster, key=key)


def _spread(cluster: list[Mark], medoid: Mark, pos: dict[str, Pos], offset: Callable[[Pos, Pos], tuple[float, float]]) -> tuple[float, int]:
    """(sum of squared distances to the cluster mean, k - 1). Taken about the mean, not the medoid: the medoid is
    one of the noisy views, so the spread about it reads about 1.4 x the per-view sigma with two views and 1.15 x
    with three, and 5.10's 2.5 sqrt(2) factor assumes the per-view sigma."""
    offsets = np.array([offset(pos[m.mark_id], pos[medoid.mark_id]) for m in cluster], dtype=np.float64)
    centred = offsets - offsets.mean(axis=0)
    return float((centred**2).sum()), len(cluster) - 1


def _keyframe_complete_and_sharp(kf: Keyframe, th: dict) -> bool:
    return (
        kf.score.completeness >= float(th["coverage"]["filmed_completeness_min"])
        and kf.score.sharpness >= float(th["matching"]["sharpness_rel_min"])
        and kf.marks_status == MarksStatus.COMPLETE
    )


def _units_map(kf: Keyframe, masks: dict[str, np.ndarray] | None) -> tuple[np.ndarray, float | None] | None:
    """(2x2 map from crop pixels to panel units, the wheel's inscribed radius in px or None) for the collinear
    rule: the whitening matrix on a moment-frame panel, 1 / inscribed radius on a wheel (the unit of wheel
    size_units). None when the frame is not available (no moments, or a wheel without its mask)."""
    if kf.panel_id in WHEEL_PANELS:
        mask = (masks or {}).get(kf.kf_id)
        if mask is None or not np.asarray(mask).any():
            return None
        radius_px = geometry.inscribed_radius(mask)
        return np.eye(2) / radius_px, radius_px
    if kf.moments is None or kf.front_sign is None:
        return None
    return geometry.whitening_matrix(kf.moments, kf.front_sign), None


def _axis(m: Mark, lin: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """(bbox centre, unit direction of the bbox long axis, half length, aspect) in panel units, or None."""
    if m.bbox_crop is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in m.bbox_crop)
    e1 = lin @ np.array([x2 - x1, 0.0])
    e2 = lin @ np.array([0.0, y2 - y1])
    n1, n2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    long_, short = (e1, n2) if n1 >= n2 else (e2, n1)
    length = max(n1, n2)
    if length <= 0.0:
        return None
    centre = lin @ np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    return centre, long_ / length, length / 2.0, (length / short if short > 0 else math.inf)


def _segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Smallest distance between two 2-D segments (the gap between two collinear boxes)."""

    def point_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        t = 0.0 if float(ab @ ab) == 0.0 else min(max(float((p - a) @ ab) / float(ab @ ab), 0.0), 1.0)
        return float(np.linalg.norm(p - (a + t * ab)))

    def cross(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    d1, d2 = p1 - p0, q1 - q0
    s1, s2 = cross(d1, q0 - p0), cross(d1, q1 - p0)
    t1, t2 = cross(d2, p0 - q0), cross(d2, p1 - q0)
    if s1 * s2 < 0 and t1 * t2 < 0:
        return 0.0
    return min(point_seg(p0, q0, q1), point_seg(p1, q0, q1), point_seg(q0, p0, p1), point_seg(q1, p0, p1))


def _merge_collinear_pair(a: Mark, b: Mark, kf: Keyframe, th: dict, *, mask: np.ndarray | None = None, radius_px: float | None = None) -> Mark:
    """One mark from two pieces of the same keyframe: the union bbox, placed again in the panel's own frame (wheel
    coordinates when the wheel mask and its inscribed radius are given, else the keyframe's moments)."""
    bbox = [min(a.bbox_crop[0], b.bbox_crop[0]), min(a.bbox_crop[1], b.bbox_crop[1]), max(a.bbox_crop[2], b.bbox_crop[2]), max(a.bbox_crop[3], b.bbox_crop[3])]
    point = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    crop = kf.crop
    bbox_src = geometry.box_crop_to_src(bbox, crop) if crop is not None else None
    point_src = geometry.crop_to_src(point, crop) if crop is not None else None
    if mask is not None and radius_px:
        u, v = geometry.wheel_coords(point, mask)
        r = u
        size_units: float | None = _long_side(bbox) / radius_px
    else:
        u, v = geometry.whiten((point[0], point[1]), kf.moments, kf.front_sign)
        r = geometry.radius((u, v))
        size_units = _size_units_moment(bbox, kf)
    keep, other = (a, b) if (a.size_units or 0.0) >= (b.size_units or 0.0) else (b, a)
    size_px = _long_side(bbox_src) if bbox_src is not None else None
    tiny = (size_units is not None and size_units < float(th["dedup"]["small_mark_units"])) or (size_px is not None and size_px < float(th["dedup"]["small_mark_px"]))
    caps = _with_cap(sorted({*keep.caps, *other.caps} - {"K3"}), "K3", tiny)
    return keep.model_copy(
        update={
            "bbox_crop": bbox,
            "point_crop": point,
            "bbox_src": bbox_src,
            "point_src": point_src,
            "grid_point": None,
            "grid_bbox": None,
            "w": [u, v],
            "r": r,
            "size_units": size_units,
            "size_px_src": size_px,
            "merged_from": sorted({*keep.merged_from, *other.merged_from, other.mark_id}),
            "caps": caps,
            "single_frame": "K2" in caps,
        }
    )


def collinear_merge(marks: list[Mark], kf_by_id: dict[str, Keyframe], *, th: dict, masks: dict[str, np.ndarray] | None = None) -> list[Mark]:
    """5.6: two elongated boxes of one type on one keyframe whose long axes are within collinear_angle_deg and whose
    gap is under collinear_gap_units become one mark (a scratch Spark split in two). Gap and axes are measured in
    the panel's own units: whitened on a moment-frame panel, crop pixels over the inscribed radius on a wheel
    (masks maps kf_id to the wheel mask; a wheel keyframe without one is left alone)."""
    angle_max = math.radians(float(th["dedup"]["collinear_angle_deg"]))
    gap_max = float(th["dedup"]["collinear_gap_units"])
    frames: dict[str, tuple[np.ndarray, float | None] | None] = {}
    out = list(marks)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                kf = kf_by_id.get(a.kf_id)
                if kf is None or a.kf_id != b.kf_id or a.type != b.type or a.w is None or b.w is None:
                    continue
                if kf.kf_id not in frames:
                    frames[kf.kf_id] = _units_map(kf, masks)
                frame = frames[kf.kf_id]
                if frame is None:
                    continue
                lin, radius_px = frame
                aa, ab = _axis(a, lin), _axis(b, lin)
                if aa is None or ab is None or aa[3] < 1.5 or ab[3] < 1.5:
                    continue
                if math.acos(min(abs(float(aa[1] @ ab[1])), 1.0)) > angle_max:
                    continue
                gap = _segment_distance(aa[0] - aa[1] * aa[2], aa[0] + aa[1] * aa[2], ab[0] - ab[1] * ab[2], ab[0] + ab[1] * ab[2])
                if gap > gap_max:
                    continue
                mask = (masks or {}).get(kf.kf_id) if radius_px is not None else None
                merged = _merge_collinear_pair(a, b, kf, th, mask=mask, radius_px=radius_px)
                out = [m for k, m in enumerate(out) if k not in (i, j)]
                out.insert(i, merged)
                changed = True
                break
            if changed:
                break
    return out


def _groups(marks: list[Mark], pos: dict[str, Pos], *, role: str, panel_id: PanelId, th: dict, dist: Callable[[Pos, Pos], float] = _dist) -> list[Mark]:
    """>= group_min_marks marks of one type within group_radius_units (single linkage) share a cluster_id."""
    radius_units = float(th["dedup"]["group_radius_units"])
    min_marks = int(th["dedup"]["group_min_marks"])
    out = list(marks)
    n_groups = 0
    for kind in sorted({m.type for m in marks}, key=lambda k: k.value):
        idx = [i for i, m in enumerate(out) if m.type == kind and m.mark_id in pos]
        parent = {i: i for i in idx}

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = idx[x], idx[y]
                if dist(pos[out[i].mark_id], pos[out[j].mark_id]) <= radius_units:
                    parent[find(j)] = find(i)
        comps: dict[int, list[int]] = {}
        for i in idx:
            comps.setdefault(find(i), []).append(i)
        for root in sorted(comps, key=lambda r: min(comps[r])):
            members = comps[root]
            if len(members) < min_marks:
                continue
            n_groups += 1
            cid = f"{role}_{panel_id.value}_{kind.value}_g{n_groups}"
            for i in members:
                out[i] = out[i].model_copy(update={"cluster_id": cid})
    return out


def _panel_caps_k9(marks: list[Mark], pos: dict[str, Pos], *, panel_id: PanelId, th: dict) -> bool:
    """Droplet-like (many tiny marks) or road-spray-like (same-type marks in the bottom band of a door or bumper)."""
    d = th["dedup"]
    tiny = [m for m in marks if m.size_units is not None and m.size_units < float(d["droplet_max_size_units"])]
    if len(tiny) >= int(d["droplet_min_marks"]):
        return True
    family = _FAMILY.get(panel_id)
    if family not in ("door", "bumper"):
        return False
    # A uniform rectangle spans +/- sqrt(3) whitened units on each axis; its bottom band starts there.
    v_min = math.sqrt(3.0) * (1.0 - 2.0 * float(d["road_spray_band_bottom_frac"]))
    by_type: dict[MarkKind, int] = {}
    for m in marks:
        p = pos.get(m.mark_id)
        if p is not None and p[1] >= v_min:
            by_type[m.type] = by_type.get(m.type, 0) + 1
    return any(n >= int(d["road_spray_min_marks"]) for n in by_type.values())


def _dedup(marks: list[Mark], whitened: dict[str, WhitenedMark], keyframes: list[Keyframe], *, th: dict, masks: dict[str, np.ndarray] | None = None) -> tuple[list[Mark], float, int]:
    """dedup_within_walk plus the raw spread: (marks, sum of squared distances to the cluster means, sum of k - 1)."""
    kf_by_id = {kf.kf_id: kf for kf in keyframes}
    pos: dict[str, Pos] = {}
    for m in marks:
        p = _pos(m, whitened)
        if p is not None:
            pos[m.mark_id] = p
    sq_sum = 0.0
    dof = 0
    out: list[Mark] = []
    panels: dict[tuple[str, PanelId], list[Mark]] = {}
    for m in marks:
        panels.setdefault((m.role, m.panel_id), []).append(m)
    for (role, panel_id), group in panels.items():
        dist, offset = _metric(panel_id), _offset(panel_id)
        placed = [m for m in group if m.mark_id in pos]
        out.extend(m for m in group if m.mark_id not in pos)
        panel_kfs = [kf for kf in keyframes if kf.role == role and kf.panel_id == panel_id]
        kept: list[Mark] = []
        for cluster in _cluster_across_keyframes(placed, pos, float(th["dedup"]["cluster_radius_units"]), dist):
            medoid = _medoid(cluster, pos, kf_by_id, dist)
            others = [m for m in cluster if m is not medoid]
            sq, k = _spread(cluster, medoid, pos, offset)
            sq_sum += sq
            dof += k
            update: dict[str, Any] = {}
            if others:
                update["merged_from"] = sorted({*medoid.merged_from, *(o.mark_id for o in others), *(x for o in others for x in o.merged_from)})
            elif len(panel_kfs) >= 3 and all(_keyframe_complete_and_sharp(kf, th) for kf in panel_kfs if kf.kf_id != medoid.kf_id):
                update["single_frame"] = True
                update["caps"] = _with_cap(medoid.caps, "K2", True)
            kept.append(medoid.model_copy(update=update) if update else medoid)
        kept = collinear_merge(kept, kf_by_id, th=th, masks=masks)
        for m in kept:
            if m.w is not None:
                pos[m.mark_id] = (float(m.w[0]), float(m.w[1]))
        kept = _groups(kept, pos, role=role, panel_id=panel_id, th=th, dist=dist)
        if _panel_caps_k9(kept, pos, panel_id=panel_id, th=th):
            kept = [m.model_copy(update={"caps": _with_cap(m.caps, "K9", True)}) for m in kept]
        out.extend(kept)
    return out, sq_sum, dof


def _sigma_from(sq_sum: float, dof: int) -> float | None:
    """Per-axis sigma from a pooled 2-D sum of squares: each cluster gives 2 (k - 1) degrees of freedom."""
    return math.sqrt(sq_sum / (2.0 * dof)) if dof else None


def dedup_within_walk(marks: list[Mark], whitened: dict[str, WhitenedMark], keyframes: list[Keyframe], *, th: dict, masks: dict[str, np.ndarray] | None = None) -> tuple[list[Mark], float | None]:
    """5.6 per (role, panel): clusters across keyframes within cluster_radius_units keep the medoid (merged_from),
    sigma_view is the pooled per-axis spread of the members about their cluster means, K2 marks a mark seen in 1
    of 3 keyframes while the others were complete and sharp, the collinear merge joins split boxes, groups get a
    cluster_id, droplet-like or road-spray-like panels get K9. Wheel panels use match's wrapped polar distance
    throughout; masks (kf_id to wheel mask) lets the collinear merge run on wheels. Marks without a whitened
    position pass through."""
    out, sq_sum, dof = _dedup(marks, whitened, keyframes, th=th, masks=masks)
    return out, _sigma_from(sq_sum, dof)


# Crop helper and the derive entry ----------------------------------------------------------------------------------


def crop_from_proxy_box(box_proxy: Box4, proxy_size: tuple[int, int], src_size: tuple[int, int], *, th: dict) -> Crop:
    """5.3: proxy box x the proxy ratio, crop_pad_frac of padding, long edge crop_long_edge_px; path left ""."""
    box_src = geometry.proxy_to_src(box_proxy, proxy_size, src_size)
    return geometry.crop_geometry(box_src, src_size, pad_frac=float(th["panels"]["crop_pad_frac"]), long_edge=int(th["panels"]["crop_long_edge_px"]))


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


def _read_list(path: Path, model: Any) -> list[Any]:
    return read_json_list(path, model) if path.is_file() and path.stat().st_size > 0 else []


def _sigma_view_path(run_id: str) -> Path:
    """derived/sigma_view.json: each role's own view spread {role: {dof, sigma_view}}, replaced when the role reruns."""
    return paths.derived_dir(run_id) / "sigma_view.json"


def _read_stats(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() and path.stat().st_size > 0 else {}


def _pooled_sigma_view(stats: dict[str, dict[str, Any]]) -> float | None:
    """Every role's sigma_view pooled by its degrees of freedom; the order of the roles does not matter."""
    rows = [(int(s["dof"]), float(s["sigma_view"])) for _, s in sorted(stats.items()) if s.get("dof") and s.get("sigma_view") is not None]
    dof = sum(k for k, _ in rows)
    return math.sqrt(sum(k * sv * sv for k, sv in rows) / dof) if dof else None


def _refresh_calibration(run_id: str, sigma_view: float | None, *, th: dict) -> Calibration:
    """calibration.json with the pooled sigma_view. sigma_frame and sigma_spark keep their stored values, else the
    measured thresholds leaves. A tau_pre that a pseudo pair raised above what the stored sigmas give is kept as a
    floor: the rows that raised it are not stored here, and calibrate(pseudo_pair_d=None) would drop the raise."""
    from . import match

    path = paths.calibration_json(run_id)
    existing = read_json(path, Calibration) if path.is_file() and path.stat().st_size > 0 else None
    measured_frame, measured_spark = match.measured_sigmas(th)
    sigma_frame = existing.sigma_frame if existing is not None and existing.sigma_frame is not None else measured_frame
    sigma_spark = existing.sigma_spark if existing is not None and existing.sigma_spark is not None else measured_spark
    cal = match.calibrate(sigma_frame=sigma_frame, sigma_spark=sigma_spark, sigma_view=sigma_view, pseudo_pair_d=None, th=th)
    if existing is not None:
        own = match.calibrate(sigma_frame=existing.sigma_frame, sigma_spark=existing.sigma_spark, sigma_view=existing.sigma_view, pseudo_pair_d=None, th=th)
        if existing.tau_pre > own.tau_pre + PSEUDO_RAISE_TOL and existing.tau_pre > cal.tau_pre:
            tau_pre = existing.tau_pre
            cal = cal.model_copy(update={"tau_pre": tau_pre, "tau_eye": float(th["matching"]["tau_eye_factor"]) * tau_pre, "calibrated": cal.calibrated and existing.calibrated})
    write_json(path, cal)
    return cal


def _record_sigma_view(run_id: str, role: str, sq_sum: float, dof: int, *, th: dict) -> Calibration:
    """Store this role's spread (replacing an earlier run of the role, never pooling it with itself), pool every
    role from the rounded values on disk and refresh calibration.json, so the same marks write the same bytes."""
    path = _sigma_view_path(run_id)
    stats = _read_stats(path)
    stats[role] = {"dof": int(dof), "sigma_view": _sigma_from(sq_sum, dof)}
    write_json(path, stats)
    return _refresh_calibration(run_id, _pooled_sigma_view(_read_stats(path)), th=th)


def _reusable_records(records: list[SparkRequestRecord], keyframes: list[Keyframe]) -> dict[tuple[str, PanelId], SparkRequestRecord]:
    """Valid marks_v1 answers keyed by (crop sha256, panel): the question is the panel's prompt about those crop bytes,
    so a run asks it once (a later record of the same key wins). The panel is that of the keyframe whose crop the
    record names in input_refs."""
    panel_by_crop = {kf.crop.path: kf.panel_id for kf in keyframes if kf.crop is not None}
    out: dict[tuple[str, PanelId], SparkRequestRecord] = {}
    for rec in records:
        if rec.purpose != SparkPurpose.MARKS or not rec.valid or not rec.input_refs:
            continue
        panel = panel_by_crop.get(rec.input_refs[0].path)
        if panel is not None:
            out[(rec.input_refs[0].sha256, panel)] = rec
    return out


def _wheel_mask(root: Path, kf: Keyframe) -> np.ndarray | None:
    return _read_mask(root / kf.mask_png) if kf.panel_id in WHEEL_PANELS and kf.mask_png else None


def _box_marks(root: Path, kf: Keyframe, result: Any, *, th: dict, halves_index: Any = None, mask: np.ndarray | None = None) -> list[Mark]:
    """The crop's Spark box marks: the full-crop answer, then with dedup.spark_halves_pass on the stored halves boxes that
    are not the same mark as one already kept (halves_pass.halves_marks)."""
    from . import halves_pass

    kf_marks = marks_from_answer(result.answer, result, kf, kf.crop, th=th)
    if halves_index is not None and halves_pass.settings(th) is not None:
        kf_marks.extend(halves_pass.halves_marks(kf_marks, kf, halves_index, th=th, mask=mask if mask is not None else _wheel_mask(root, kf)))
    return kf_marks


def _marks_of(root: Path, results: list[tuple[Keyframe, Any, Any]], *, th: dict, sam_index: Any = None, zoom_index: Any = None, halves_index: Any = None) -> tuple[list[Mark], list[Keyframe], float, int]:
    """One role's marks from its keyframes' Spark results (None when Spark never saw the keyframe): marks_status and
    conditions on each keyframe, mapping, caps, the zoom move of each box mark when dedup.spark_zoom_refine is on (from
    the stored answers in zoom_index), then its SAM box when dedup.sam_box_refine is on (sam_index), whitening, the
    count union, then the within-walk dedup. With dedup.spark_halves_pass on, each crop's halves boxes (halves_index)
    join its full-crop boxes before the zoom move (and with its after_refine key, a halves mark that lands on a kept
    mark once moved is dropped; with its merge_patches key, a halves mark close to a full-crop mark once both are moved
    gives that mark its box, halves_pass.merge_after_refine)."""
    from . import box_refine, halves_pass, zoom_refine

    refine = box_refine.settings(th) is not None and sam_index is not None
    zoom = zoom_refine.settings(th) is not None and zoom_index is not None
    marks: list[Mark] = []
    whitened: dict[str, WhitenedMark] = {}
    masks: dict[str, np.ndarray] = {}
    updated: list[Keyframe] = []
    for kf, result, count in results:
        if result is None:
            updated.append(kf.model_copy(update={"marks_status": MarksStatus.NOT_RUN}))
            continue
        answer = result.answer
        status = MarksStatus.TRUNCATED if answer is None or getattr(result, "truncated", False) else MarksStatus.COMPLETE
        kf_new = kf.model_copy(update={"marks_status": status, "conditions": answer.conditions if answer is not None else kf.conditions})
        updated.append(kf_new)
        if answer is None:
            continue
        mask = _wheel_mask(root, kf)
        if mask is not None:
            masks[kf.kf_id] = mask
        kf_marks = _box_marks(root, kf_new, result, th=th, halves_index=halves_index, mask=mask)
        if zoom:
            kf_marks = zoom_refine.refine_marks(kf_marks, kf_new, zoom_index, th=th)
        if refine:
            kf_marks = box_refine.refine_marks(kf_marks, kf_new, sam_index, th=th)
        if halves_index is not None:
            kf_marks = halves_pass.merge_after_refine(kf_marks, kf_new, th=th, mask=mask)
        kf_marks, kf_whitened = whiten_marks(kf_marks, kf_new, th=th, mask=mask)
        if count is not None and not getattr(count, "truncated", False):
            extra = count_union(kf_marks, count, result, kf_new, kf.crop, kf_whitened, th=th)
            if extra:
                extra, extra_whitened = whiten_marks(extra, kf_new, th=th, mask=mask)
                kf_marks.extend(extra)
                kf_whitened.update(extra_whitened)
        marks.extend(kf_marks)
        whitened.update(kf_whitened)
    marks, whitened, left_out = on_panel(marks, whitened, updated, root, th=th, masks=masks)
    marks, sq_sum, dof = _dedup(marks, whitened, updated, th=th, masks=masks)
    return [*marks, *left_out], updated, sq_sum, dof


# On the panel (lexi run 1, 11/2, thresholds dedup.marks_on_panel) -------------------------------------------------------


def on_panel_frac(th: dict | None) -> float | None:
    """The claim path's on_panel_frac when thresholds dedup.marks_on_panel is on, else None (off when absent, so runs
    frozen before it rebuild unchanged)."""
    cfg = (th or {}).get("dedup", {}).get("marks_on_panel")
    if not (isinstance(cfg, dict) and cfg.get("enabled", True)):
        return None
    from .claim import ON_PANEL_FRAC

    return float((th or {}).get("claim", {}).get("on_panel_frac", ON_PANEL_FRAC))


def _mask_distance(mask: np.ndarray, point: Point2) -> float:
    """Distance in pixels from a point to the nearest mask pixel (0 on the mask); inf for an empty mask."""
    if not mask.any():
        return math.inf
    h, w = mask.shape[:2]
    x, y = min(w - 1, max(0, int(math.floor(float(point[0]))))), min(h - 1, max(0, int(math.floor(float(point[1])))))
    if mask[y, x]:
        return 0.0
    ys, xs = np.nonzero(mask)
    return float(np.min(np.hypot(xs + 0.5 - float(point[0]), ys + 0.5 - float(point[1]))))


def _off(mask: np.ndarray, point: Point2, crop: Crop, frac: float) -> tuple[bool, float]:
    """(the point sits on the panel: within max(3 px, frac of the crop's long side) of the mask, the claim path's rule;
    its distance over the crop's long side)."""
    long_side = float(max(crop.w, crop.h))
    d = _mask_distance(mask, point)
    return d <= max(3.0, frac * long_side), (d / long_side if math.isfinite(d) else 1.0)


def on_panel(
    marks: list[Mark], whitened: dict[str, WhitenedMark], keyframes: list[Keyframe], root: Path, *, th: dict, masks: dict[str, np.ndarray]
) -> tuple[list[Mark], dict[str, WhitenedMark], list[Mark]]:
    """Every Spark mark whose point sits off its keyframe's panel mask: moved to a keyframe of the same walk cut from the
    same frame whose panel mask holds the point (the nearest, then the lower kf id), its box and point mapped through
    the source frame and whitened there; else left out (returned apart, off_panel set, moved_to_kf None). A mark whose
    keyframe has no mask stays. Returns (the marks for the dedup, their whitened rows, the marks left out)."""
    frac = on_panel_frac(th)
    if frac is None:
        return marks, whitened, []
    cache: dict[str, np.ndarray | None] = {}

    def mask_of(kf: Keyframe) -> np.ndarray | None:
        if kf.kf_id not in cache:
            cache[kf.kf_id] = _read_mask(root / kf.mask_png) if kf.mask_png else None
        return cache[kf.kf_id]

    kf_by_id = {kf.kf_id: kf for kf in keyframes}
    kept: list[Mark] = []
    left: list[Mark] = []
    out_whitened = dict(whitened)
    for m in marks:
        kf = kf_by_id.get(m.kf_id)
        own = mask_of(kf) if kf is not None and kf.crop is not None else None
        if m.source == MarkSource.NARRATION_SEED or m.point_crop is None or own is None or kf is None or kf.crop is None:
            kept.append(m)
            continue
        on, off_frac = _off(own, m.point_crop, kf.crop, frac)
        if on:
            kept.append(m)
            continue
        best: tuple[float, str, Keyframe, Point2] | None = None
        for other in keyframes:
            if other.role != m.role or other.f_proxy != kf.f_proxy or other.panel_id == m.panel_id or other.crop is None or m.point_src is None:
                continue
            p = geometry.src_to_crop(m.point_src, other.crop)
            if not (0.0 <= p[0] < other.crop.w and 0.0 <= p[1] < other.crop.h):
                continue
            om = mask_of(other)
            if om is None:
                continue
            ok, f = _off(om, p, other.crop, frac)
            if ok and (best is None or (f, other.kf_id) < best[:2]):
                best = (f, other.kf_id, other, p)
        out_whitened.pop(m.mark_id, None)
        if best is None:
            left.append(m.model_copy(update={"off_panel": OffPanel(from_panel=m.panel_id, from_kf=m.kf_id, off_frac=round(off_frac, 6))}))
            continue
        _f, _id, target, p = best
        box = None
        if m.bbox_src is not None:
            b = geometry.box_src_to_crop(m.bbox_src, target.crop)  # type: ignore[arg-type]
            box = [max(0.0, b[0]), max(0.0, b[1]), min(float(target.crop.w), b[2]), min(float(target.crop.h), b[3])]  # type: ignore[union-attr]
        record = OffPanel(from_panel=m.panel_id, from_kf=m.kf_id, off_frac=round(off_frac, 6), moved_to_kf=target.kf_id)
        moved = m.model_copy(update={"panel_id": target.panel_id, "kf_id": target.kf_id, "point_crop": [float(p[0]), float(p[1])], "bbox_crop": box, "w": None, "r": None, "size_units": None, "off_panel": record})
        wheel_mask = mask_of(target) if target.panel_id in WHEEL_PANELS else None
        if wheel_mask is not None:
            masks[target.kf_id] = wheel_mask
        [moved], got = whiten_marks([moved], target, th=th, mask=wheel_mask)
        out_whitened.update(got)
        kept.append(moved)
    return kept, out_whitened, left


def _write_role_marks(run_id: str, role: str, marks: list[Mark]) -> None:
    marks_path = paths.marks_json(run_id)
    others = [m for m in _read_list(marks_path, Mark) if m.role != role]
    write_json(marks_path, [*others, *marks])


async def derive_marks(run_id: str, role: str, keyframes: list[Keyframe], *, th: dict, count_mode: bool = False) -> tuple[list[Mark], list[Keyframe], list[SparkRequestRecord]]:
    """Per keyframe of the role: Spark marks on the blurred crop (and count mode when asked), conditions and
    marks_status on the keyframe, mapping, caps, whitening, dedup; writes marks.json (this role's rows replaced),
    appends to spark.json, refreshes keyframes.json rows, this role's row of sigma_view.json and the pooled
    sigma_view in calibration.json.

    The same question about the same crop bytes is asked once per run: a keyframe whose crop sha256 and panel match a
    valid answer already in spark.json reads that answer back (no call, no new record), and two keyframes of this call
    with the same crop bytes share one call. Only a walk byte-identical to another (a self-pair) or a resumed run ever
    meets this; rebuild_marks applies the same rule. Count mode always asks."""
    from . import match, spark_client

    root = paths.run_dir(run_id)
    effort = str(th["spark"]["effort"])
    fixtures = match.fixtures_not_damage(th)  # lead decision 4 of 11/1: the prompt's fixture sentence, leaf on only
    seams = match.seams_not_marks(th)  # lead 11/1: the seams sentence, leaf on only
    semaphore = asyncio.Semaphore(int(th["orchestrator"]["spark_semaphore"]))
    kfs = [kf for kf in keyframes if kf.role == role]
    spark_path = paths.spark_json(run_id)
    known = _read_list(spark_path, SparkRequestRecord)
    reuse = {} if count_mode else _reusable_records(known, [*_read_list(paths.keyframes_json(run_id), Keyframe), *kfs])
    asked: dict[tuple[str, PanelId], asyncio.Future[Any]] = {}

    async def ask(kf: Keyframe, crop_rgb: np.ndarray, input_ref: InputRef | None) -> tuple[Any, Any]:
        async with semaphore:
            result = await asyncio.to_thread(
                spark_client.find_marks, crop_rgb, kf.panel_id, landmarks=landmarks_for(kf.panel_id, th), effort=effort, run_id=run_id, input_ref=input_ref, fixtures=fixtures, seams=seams,
                colour=match.colour_on_paint(th), panel_conditions=match.conditions_on_panel(th),
            )
            count = None
            if count_mode:
                count = await asyncio.to_thread(spark_client.count_marks, crop_rgb, kf.panel_id, effort=effort, run_id=run_id, input_ref=input_ref)
        return result, count

    async def one(kf: Keyframe) -> tuple[Keyframe, Any, Any, bool]:
        if kf.crop is None:
            return kf, None, None, False
        crop_rgb = _read_rgb(root / kf.crop.path)
        input_ref = InputRef(path=kf.crop.path, sha256=kf.crop.sha256) if kf.crop.sha256 else None
        key = (kf.crop.sha256, kf.panel_id) if kf.crop.sha256 and not count_mode else None
        if key is not None and key in reuse:
            h, w = crop_rgb.shape[:2]
            return kf, spark_client.marks_result_from_record(reuse[key], orig_width=w, orig_height=h, landmarks=landmarks_for(kf.panel_id, th)), None, True
        if key is not None and key in asked:
            result, count = await asked[key]
            return kf, result, count, True
        shared = asyncio.get_running_loop().create_future() if key is not None else None
        if key is not None:
            asked[key] = shared
        try:
            result, count = await ask(kf, crop_rgb, input_ref)
        except BaseException as exc:
            if shared is not None:
                shared.set_exception(exc)
                shared.exception()  # retrieved here; a twin awaiting it raises the same error
            raise
        if shared is not None:
            shared.set_result((result, count))
        return kf, result, count, False

    results = await asyncio.gather(*(one(kf) for kf in kfs))
    records: list[SparkRequestRecord] = []
    for _kf, result, count, reused in results:
        if result is not None and not reused:
            records.append(result.record)
            if count is not None:
                records.append(count.record)
    answered = [(kf, result) for kf, result, _count, _reused in results]
    halves_index = await _halves_boxes(run_id, answered, th=th)
    zoom_index = await _zoom_boxes(run_id, answered, th=th, halves_index=halves_index)
    sam_index = await _sam_boxes(run_id, answered, th=th, zoom_index=zoom_index, halves_index=halves_index)
    marks, updated, sq_sum, dof = _marks_of(root, [(kf, result, count) for kf, result, count, _ in results], th=th, sam_index=sam_index, zoom_index=zoom_index, halves_index=halves_index)

    _write_role_marks(run_id, role, marks)
    write_json(spark_path, [*_read_list(spark_path, SparkRequestRecord), *records])
    kf_path = paths.keyframes_json(run_id)
    by_id = {kf.kf_id: kf for kf in updated}
    stored = _read_list(kf_path, Keyframe)
    if stored:
        write_json(kf_path, [by_id.get(kf.kf_id, kf) for kf in stored])
    _record_sigma_view(run_id, role, sq_sum, dof, th=th)
    return marks, updated, records


async def _halves_boxes(run_id: str, results: list[tuple[Keyframe, Any]], *, th: dict) -> Any:
    """With dedup.spark_halves_pass on: two Muse Spark marks_v1 calls per answered crop, one on each enlarged half (LIVE),
    appended to halves_marks.json; a (crop, half) with a stored answer is not asked again. Returns the halves_pass.Index
    of every stored answer, or None when the leaf is off."""
    from . import halves_pass

    s = halves_pass.settings(th)
    if s is None:
        return None
    root = paths.run_dir(run_id)
    stored = halves_pass.read_asks(run_id)
    known = halves_pass.Index(stored)
    jobs = []
    for kf, result in results:
        if kf.crop is None or getattr(result, "answer", None) is None:
            continue
        todo = [(half, win) for half, win in halves_pass.plan(kf, s) if not known.has(kf.crop, win)]
        if todo:
            jobs.append((kf, root / kf.crop.path, todo))
    if jobs:
        asked = await halves_pass.ask_missing(run_id, jobs, s, th=th, read_rgb=_read_rgb)
        stored = [*halves_pass.read_asks(run_id), *asked]
        halves_pass.write_asks(run_id, stored)
    return halves_pass.Index(stored)


async def _zoom_boxes(run_id: str, results: list[tuple[Keyframe, Any]], *, th: dict, halves_index: Any = None) -> Any:
    """With dedup.spark_zoom_refine on: one Muse Spark marks_v1 call per box mark of each answered crop, on a window
    around its box enlarged to the zoom size (LIVE), appended to zoom_marks.json; a (crop, window) with a stored answer
    is not asked again. Returns the zoom_refine.Index of every stored answer, or None when the leaf is off."""
    from . import zoom_refine

    s = zoom_refine.settings(th)
    if s is None:
        return None
    root = paths.run_dir(run_id)
    stored = zoom_refine.read_asks(run_id)
    known = zoom_refine.Index(stored)
    jobs = []
    for kf, result in results:
        answer = getattr(result, "answer", None)
        if kf.crop is None or answer is None:
            continue
        seen: set[tuple[int, ...]] = set()
        todo = []
        for m, win in zoom_refine.plan(_box_marks(root, kf, result, th=th, halves_index=halves_index), kf, s):
            if not known.has(kf.crop, win) and tuple(win) not in seen:
                seen.add(tuple(win))
                todo.append((m, win))
        if todo:
            jobs.append((kf, root / kf.crop.path, todo))
    if jobs:
        asked = await zoom_refine.ask_missing(run_id, jobs, s, th=th, read_rgb=_read_rgb)
        stored = [*zoom_refine.read_asks(run_id), *asked]
        zoom_refine.write_asks(run_id, stored)
    return zoom_refine.Index(stored)


async def _sam_boxes(run_id: str, results: list[tuple[Keyframe, Any]], *, th: dict, zoom_index: Any = None, halves_index: Any = None) -> Any:
    """With dedup.sam_box_refine on: SAM 3.1 image requests for the box marks of each answered crop, all of a crop's
    boxes in one request (LIVE), appended to sam_marks.json; a box whose (crop, prompt) already has a stored answer is
    not asked again. With zoom_index, each box is first moved by its stored zoom answer, so SAM is prompted with the
    moved box. Returns the box_refine.Index of every stored answer, or None when the leaf is off."""
    from . import box_refine, zoom_refine

    s = box_refine.settings(th)
    if s is None:
        return None
    root = paths.run_dir(run_id)
    stored = box_refine.read_asks(run_id)
    known = box_refine.Index(stored)
    jobs = []
    for kf, result in results:
        answer = getattr(result, "answer", None)
        if kf.crop is None or answer is None:
            continue
        provisional = _box_marks(root, kf, result, th=th, halves_index=halves_index)
        if zoom_index is not None:
            provisional = zoom_refine.refine_marks(provisional, kf, zoom_index, th=th)
        for chunk in box_refine.plan(provisional, kf, s):
            todo = [o for o in chunk if not known.has(kf.crop, o.prompt)]
            if todo:
                jobs.append((kf, [o.model_copy(update={"object_id": j + 1}) for j, o in enumerate(todo)]))
    if jobs:
        semaphore = asyncio.Semaphore(int(th["orchestrator"]["sam_image_semaphore"]))

        async def one(kf: Keyframe, objects: list[Any]) -> Any:
            async with semaphore:
                return await box_refine.ask(run_id, kf, root / kf.crop.path, objects)  # type: ignore[union-attr]

        asked = await asyncio.gather(*(one(kf, objs) for kf, objs in jobs))
        stored = [*box_refine.read_asks(run_id), *asked]
        box_refine.write_asks(run_id, stored)
    return box_refine.Index(stored)


def rebuild_marks(run_id: str, role: str, *, th: dict) -> list[Mark]:
    """This role's marks.json rows, its sigma_view row and calibration.json recomputed from spark.json and keyframes.json
    with no model call (section 7 item 3). Each keyframe Spark answered for reads back the answer derive_marks used: the
    valid answer for its (crop sha256, panel), else the last record asked for its crop. With dedup.spark_zoom_refine on,
    each box mark's zoom answer is read back from zoom_marks.json, and with dedup.sam_box_refine on its SAM answer from
    sam_marks.json (a box with no stored answer keeps the box it had and says so). With dedup.spark_halves_pass on, each
    crop's halves boxes are read back from halves_marks.json. keyframes.json, spark.json, halves_marks.json,
    zoom_marks.json and sam_marks.json are frozen facts and are not written. Box mode only: count-mode
    points are not replayed."""
    from . import spark_client

    root = paths.run_dir(run_id)
    stored = _read_list(paths.keyframes_json(run_id), Keyframe)
    records = _read_list(paths.spark_json(run_id), SparkRequestRecord)
    valid = _reusable_records(records, stored)
    last_by_crop = {rec.input_refs[0].path: rec for rec in records if rec.purpose == SparkPurpose.MARKS and rec.input_refs}
    results: list[tuple[Keyframe, Any, Any]] = []
    for kf in (k for k in stored if k.role == role):
        record = None
        if kf.crop is not None and kf.marks_status != MarksStatus.NOT_RUN:
            record = valid.get((kf.crop.sha256, kf.panel_id)) if kf.crop.sha256 else None
            record = record or last_by_crop.get(kf.crop.path)
        if record is None:
            results.append((kf, None, None))
            continue
        h, w = _read_rgb(root / kf.crop.path).shape[:2]
        results.append((kf, spark_client.marks_result_from_record(record, orig_width=w, orig_height=h, landmarks=landmarks_for(kf.panel_id, th)), None))
    from . import box_refine, zoom_refine

    from . import halves_pass

    halves_index = halves_pass.Index(halves_pass.read_asks(run_id)) if halves_pass.settings(th) is not None else None
    zoom_index = zoom_refine.Index(zoom_refine.read_asks(run_id)) if zoom_refine.settings(th) is not None else None
    sam_index = box_refine.Index(box_refine.read_asks(run_id)) if box_refine.settings(th) is not None else None
    marks, _updated, sq_sum, dof = _marks_of(root, results, th=th, sam_index=sam_index, zoom_index=zoom_index, halves_index=halves_index)
    _write_role_marks(run_id, role, marks)
    _record_sigma_view(run_id, role, sq_sum, dof, th=th)
    return marks
