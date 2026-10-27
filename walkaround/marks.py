"""Marks: Spark's grid output mapped to crop, source and whitened coordinates, then deduplicated within one walk
(docs/ARCHITECTURE.md 5.6 and 5.7, docs/CONTRACTS.md M6).

Mapping chain: grid (0..1000 over the sent image) -> sent pixels (spark_client.grid_to_pixels) -> crop pixels
(geometry.sent_to_crop) -> source pixels (geometry.crop_to_src) -> whitened units (geometry.whiten with the
keyframe's own moments). Caps set here are booleans for match.verdict(): K2 single frame, K3 tiny, K4 uncertain
type, K8 count mode only, K9 droplets or spray, K12 low confidence. No verdict is decided here and no mark of the
other walk is ever read; the only model calls are through spark_client, which writes the ledger rows.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    PanelId,
    SizeClass,
    SparkMark,
    SparkMarksAnswer,
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
    """Fill w, r and size_units with the keyframe's own moments (wheel coordinates when a wheel mask is given) and
    set K3 by units. Marks whose keyframe has no moments or front_sign come back unchanged with no whitened row."""
    if kf.moments is None or kf.front_sign is None:
        return list(marks), {}
    small_units = float(th["dedup"]["small_mark_units"])
    wheel = kf.panel_id in WHEEL_PANELS and mask is not None
    out: list[Mark] = []
    whitened: dict[str, WhitenedMark] = {}
    for m in marks:
        if m.kf_id != kf.kf_id or m.point_crop is None:
            out.append(m)
            continue
        if wheel:
            u, v = geometry.wheel_coords(m.point_crop, mask)
            r = u
            size_units = None if m.bbox_crop is None else _long_side(m.bbox_crop) / geometry.inscribed_radius(mask)
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
    K8 (type unsure and confidence low, so K4 and K12 follow). Positions are compared in the moment frame."""
    if kf.moments is None or kf.front_sign is None:
        return []
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
                w=[u, v],
                r=geometry.radius((u, v)),
                spark_request_id=_request_id(count),
                caps=["K4", "K8", "K12"],
            )
        )
    return out


# Within-walk dedup ------------------------------------------------------------------------------------------------


def _pos(m: Mark, whitened: dict[str, WhitenedMark]) -> tuple[float, float] | None:
    wm = whitened.get(m.mark_id)
    if wm is not None:
        return wm.u, wm.v
    if m.w is not None:
        return float(m.w[0]), float(m.w[1])
    return None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _kf_quality(kf: Keyframe | None) -> float:
    return kf.score.total if kf is not None else 0.0


def _cluster_across_keyframes(marks: list[Mark], pos: dict[str, tuple[float, float]], radius_units: float) -> list[list[Mark]]:
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
            d = _dist(pos[marks[i].mark_id], pos[marks[j].mark_id])
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


def _medoid(cluster: list[Mark], pos: dict[str, tuple[float, float]], kf_by_id: dict[str, Keyframe]) -> Mark:
    def key(m: Mark) -> tuple[float, float, str]:
        total = sum(_dist(pos[m.mark_id], pos[o.mark_id]) for o in cluster if o is not m)
        return (total, -_kf_quality(kf_by_id.get(m.kf_id)), m.mark_id)

    return min(cluster, key=key)


def _keyframe_complete_and_sharp(kf: Keyframe, th: dict) -> bool:
    return (
        kf.score.completeness >= float(th["coverage"]["filmed_completeness_min"])
        and kf.score.sharpness >= float(th["matching"]["sharpness_rel_min"])
        and kf.marks_status == MarksStatus.COMPLETE
    )


def _axis(m: Mark, kf: Keyframe) -> tuple[np.ndarray, float, float] | None:
    """(unit direction of the bbox long axis in whitened coordinates, half length, aspect) or None."""
    if m.bbox_crop is None or kf.moments is None or kf.front_sign is None:
        return None
    w = geometry.whitening_matrix(kf.moments, kf.front_sign)
    e1 = w @ np.array([float(m.bbox_crop[2]) - float(m.bbox_crop[0]), 0.0])
    e2 = w @ np.array([0.0, float(m.bbox_crop[3]) - float(m.bbox_crop[1])])
    n1, n2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    long_, short = (e1, n2) if n1 >= n2 else (e2, n1)
    length = max(n1, n2)
    if length <= 0.0:
        return None
    return long_ / length, length / 2.0, (length / short if short > 0 else math.inf)


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


def _merge_collinear_pair(a: Mark, b: Mark, kf: Keyframe, th: dict) -> Mark:
    """One mark from two pieces of the same keyframe: the union bbox, re-whitened with the keyframe's moments."""
    bbox = [min(a.bbox_crop[0], b.bbox_crop[0]), min(a.bbox_crop[1], b.bbox_crop[1]), max(a.bbox_crop[2], b.bbox_crop[2]), max(a.bbox_crop[3], b.bbox_crop[3])]
    point = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    crop = kf.crop
    bbox_src = geometry.box_crop_to_src(bbox, crop) if crop is not None else None
    point_src = geometry.crop_to_src(point, crop) if crop is not None else None
    u, v = geometry.whiten((point[0], point[1]), kf.moments, kf.front_sign)
    keep, other = (a, b) if (a.size_units or 0.0) >= (b.size_units or 0.0) else (b, a)
    size_px = _long_side(bbox_src) if bbox_src is not None else None
    merged = keep.model_copy(
        update={
            "bbox_crop": bbox,
            "point_crop": point,
            "bbox_src": bbox_src,
            "point_src": point_src,
            "grid_point": None,
            "grid_bbox": None,
            "w": [u, v],
            "r": geometry.radius((u, v)),
            "size_px_src": size_px,
            "merged_from": sorted({*keep.merged_from, *other.merged_from, other.mark_id}),
            "caps": sorted({*keep.caps, *other.caps} - {"K3"}),
        }
    )
    size_units = _size_units_moment(bbox, kf)
    tiny = (size_units is not None and size_units < float(th["dedup"]["small_mark_units"])) or (size_px is not None and size_px < float(th["dedup"]["small_mark_px"]))
    return merged.model_copy(update={"size_units": size_units, "caps": _with_cap(merged.caps, "K3", tiny)})


def collinear_merge(marks: list[Mark], kf_by_id: dict[str, Keyframe], *, th: dict) -> list[Mark]:
    """5.6: two elongated boxes of one type on one keyframe whose long axes are within collinear_angle_deg and whose
    gap is under collinear_gap_units become one mark (a scratch Spark split in two)."""
    angle_max = math.radians(float(th["dedup"]["collinear_angle_deg"]))
    gap_max = float(th["dedup"]["collinear_gap_units"])
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
                aa, ab = _axis(a, kf), _axis(b, kf)
                if aa is None or ab is None or aa[2] < 1.5 or ab[2] < 1.5:
                    continue
                cosang = abs(float(aa[0] @ ab[0]))
                if math.acos(min(cosang, 1.0)) > angle_max:
                    continue
                pa, pb = np.array(a.w, dtype=np.float64), np.array(b.w, dtype=np.float64)
                gap = _segment_distance(pa - aa[0] * aa[1], pa + aa[0] * aa[1], pb - ab[0] * ab[1], pb + ab[0] * ab[1])
                if gap > gap_max:
                    continue
                merged = _merge_collinear_pair(a, b, kf, th)
                out = [m for k, m in enumerate(out) if k not in (i, j)]
                out.insert(i, merged)
                changed = True
                break
            if changed:
                break
    return out


def _groups(marks: list[Mark], pos: dict[str, tuple[float, float]], *, role: str, panel_id: PanelId, th: dict) -> list[Mark]:
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
                if _dist(pos[out[i].mark_id], pos[out[j].mark_id]) <= radius_units:
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


def _panel_caps_k9(marks: list[Mark], pos: dict[str, tuple[float, float]], *, panel_id: PanelId, th: dict) -> bool:
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


def dedup_within_walk(marks: list[Mark], whitened: dict[str, WhitenedMark], keyframes: list[Keyframe], *, th: dict) -> tuple[list[Mark], float | None]:
    """5.6 per (role, panel): clusters across keyframes within cluster_radius_units keep the medoid (merged_from),
    sigma_view is the per-axis spread of the members about their medoids, K2 marks a mark seen in 1 of 3
    keyframes while the others were complete and sharp, the collinear merge joins split boxes, groups get a
    cluster_id, droplet-like or road-spray-like panels get K9. Marks without a whitened position pass through."""
    kf_by_id = {kf.kf_id: kf for kf in keyframes}
    pos: dict[str, tuple[float, float]] = {}
    for m in marks:
        p = _pos(m, whitened)
        if p is not None:
            pos[m.mark_id] = p
    sq_sum = 0.0
    n_members = 0
    out: list[Mark] = []
    panels: dict[tuple[str, PanelId], list[Mark]] = {}
    for m in marks:
        panels.setdefault((m.role, m.panel_id), []).append(m)
    for (role, panel_id), group in panels.items():
        placed = [m for m in group if m.mark_id in pos]
        out.extend(m for m in group if m.mark_id not in pos)
        panel_kfs = [kf for kf in keyframes if kf.role == role and kf.panel_id == panel_id]
        kept: list[Mark] = []
        for cluster in _cluster_across_keyframes(placed, pos, float(th["dedup"]["cluster_radius_units"])):
            medoid = _medoid(cluster, pos, kf_by_id)
            others = [m for m in cluster if m is not medoid]
            for o in others:
                sq_sum += _dist(pos[medoid.mark_id], pos[o.mark_id]) ** 2
                n_members += 1
            update: dict[str, Any] = {}
            if others:
                update["merged_from"] = sorted({*medoid.merged_from, *(o.mark_id for o in others), *(x for o in others for x in o.merged_from)})
            elif len(panel_kfs) >= 3 and all(_keyframe_complete_and_sharp(kf, th) for kf in panel_kfs if kf.kf_id != medoid.kf_id):
                update["single_frame"] = True
                update["caps"] = _with_cap(medoid.caps, "K2", True)
            kept.append(medoid.model_copy(update=update) if update else medoid)
        kept = collinear_merge(kept, kf_by_id, th=th)
        for m in kept:
            if m.w is not None:
                pos[m.mark_id] = (float(m.w[0]), float(m.w[1]))
        kept = _groups(kept, pos, role=role, panel_id=panel_id, th=th)
        if _panel_caps_k9(kept, pos, panel_id=panel_id, th=th):
            kept = [m.model_copy(update={"caps": _with_cap(m.caps, "K9", True)}) for m in kept]
        out.extend(kept)
    sigma_view = math.sqrt(sq_sum / (2.0 * n_members)) if n_members else None
    return out, sigma_view


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


def _merge_sigma_view(run_id: str, sigma_view: float | None, *, th: dict) -> None:
    """Fold this walk's sigma_view into derived/calibration.json so match.derive_match can read it."""
    from . import match

    path = paths.calibration_json(run_id)
    existing = read_json(path, Calibration) if path.is_file() and path.stat().st_size > 0 else None
    pooled = sigma_view
    if existing is not None and existing.sigma_view is not None and sigma_view is not None:
        pooled = math.sqrt((existing.sigma_view ** 2 + sigma_view ** 2) / 2.0)
    elif existing is not None and sigma_view is None:
        pooled = existing.sigma_view
    measured_frame, measured_spark = match.measured_sigmas(th)
    sigma_frame = existing.sigma_frame if existing is not None and existing.sigma_frame is not None else measured_frame
    sigma_spark = existing.sigma_spark if existing is not None and existing.sigma_spark is not None else measured_spark
    cal = match.calibrate(sigma_frame=sigma_frame, sigma_spark=sigma_spark, sigma_view=pooled, pseudo_pair_d=None, th=th)
    write_json(path, cal)


async def derive_marks(run_id: str, role: str, keyframes: list[Keyframe], *, th: dict, count_mode: bool = False) -> tuple[list[Mark], list[Keyframe], list[SparkRequestRecord]]:
    """Per keyframe of the role: Spark marks on the blurred crop (and count mode when asked), conditions and
    marks_status on the keyframe, mapping, caps, whitening, dedup; writes marks.json (this role's rows replaced),
    appends to spark.json, refreshes keyframes.json rows and the sigma_view in calibration.json."""
    from . import spark_client

    root = paths.run_dir(run_id)
    effort = str(th["spark"]["effort"])
    semaphore = asyncio.Semaphore(int(th["orchestrator"]["spark_semaphore"]))
    kfs = [kf for kf in keyframes if kf.role == role]

    async def one(kf: Keyframe) -> tuple[Keyframe, Any, Any]:
        if kf.crop is None:
            return kf, None, None
        crop_rgb = _read_rgb(root / kf.crop.path)
        input_ref = InputRef(path=kf.crop.path, sha256=kf.crop.sha256) if kf.crop.sha256 else None
        async with semaphore:
            result = await asyncio.to_thread(
                spark_client.find_marks, crop_rgb, kf.panel_id, landmarks=landmarks_for(kf.panel_id, th), effort=effort, run_id=run_id, input_ref=input_ref
            )
            count = None
            if count_mode:
                count = await asyncio.to_thread(spark_client.count_marks, crop_rgb, kf.panel_id, effort=effort, run_id=run_id, input_ref=input_ref)
        return kf, result, count

    results = await asyncio.gather(*(one(kf) for kf in kfs))
    marks: list[Mark] = []
    whitened: dict[str, WhitenedMark] = {}
    updated: list[Keyframe] = []
    records: list[SparkRequestRecord] = []
    for kf, result, count in results:
        if result is None:
            updated.append(kf.model_copy(update={"marks_status": MarksStatus.NOT_RUN}))
            continue
        records.append(result.record)
        if count is not None:
            records.append(count.record)
        answer = result.answer
        status = MarksStatus.TRUNCATED if answer is None or getattr(result, "truncated", False) else MarksStatus.COMPLETE
        kf_new = kf.model_copy(update={"marks_status": status, "conditions": answer.conditions if answer is not None else kf.conditions})
        updated.append(kf_new)
        if answer is None:
            continue
        mask = _read_mask(root / kf.mask_png) if kf.panel_id in WHEEL_PANELS and kf.mask_png else None
        kf_marks = marks_from_answer(answer, result, kf_new, kf.crop, th=th)
        kf_marks, kf_whitened = whiten_marks(kf_marks, kf_new, th=th, mask=mask)
        if count is not None and not getattr(count, "truncated", False):
            extra = count_union(kf_marks, count, result, kf_new, kf.crop, kf_whitened, th=th)
            if extra:
                extra, extra_whitened = whiten_marks(extra, kf_new, th=th, mask=mask)
                kf_marks.extend(extra)
                kf_whitened.update(extra_whitened)
        marks.extend(kf_marks)
        whitened.update(kf_whitened)
    marks, sigma_view = dedup_within_walk(marks, whitened, updated, th=th)

    marks_path = paths.marks_json(run_id)
    others = [m for m in _read_list(marks_path, Mark) if m.role != role]
    write_json(marks_path, [*others, *marks])
    spark_path = paths.spark_json(run_id)
    write_json(spark_path, [*_read_list(spark_path, SparkRequestRecord), *records])
    kf_path = paths.keyframes_json(run_id)
    by_id = {kf.kf_id: kf for kf in updated}
    stored = _read_list(kf_path, Keyframe)
    if stored:
        write_json(kf_path, [by_id.get(kf.kf_id, kf) for kf in stored])
    _merge_sigma_view(run_id, sigma_view, th=th)
    return marks, updated, records
