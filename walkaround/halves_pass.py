"""A recall pass on two zoomed halves of each crop (11/2; thresholds dedup.spark_halves_pass, off when absent).

On lexi_pair1 Muse Spark boxed 22 of the 32 real marks Sanat found in the crops he checked; the pink square M4 was never
boxed in either walk. Round 10 found that the same marks_v1 question on an about 2x zoom of each half of a crop reaches
marks the full crop misses. Here each crop Spark answered gets two more marks_v1 calls, one per half along its long
side (span_frac of it each, so the halves share a band), each half enlarged scale times, with the run's prompt
sentences and effort and no landmarks. Every returned box is mapped back to crop pixels; a box that is the same mark as
one already kept on that crop (the full crop's boxes first, then the halves in order) is dropped: its centre inside a
kept box or a kept centre inside it, IoU above dup_iou, or within dedup.cluster_radius_units of a kept mark in the
panel frame. The rest become Spark box marks of the crop (ids <kf_id>_h<n>, caps K3, K4 and K12 as for the full crop)
and then go through the zoom refine, the SAM step, whitening and the within-walk dedup like any other box.

The answers are frozen in runs/<id>/halves_marks.json and rebuild replays them with no call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import geometry, paths, zoom_refine
from .contracts import (
    CAPPED_MARK_KINDS,
    Confidence,
    Contract,
    Crop,
    GridBox,
    GridPoint,
    HalvesFind,
    HalvesMerge,
    IntBox4,
    Keyframe,
    Mark,
    MarkSource,
    PanelId,
    Sha256,
    SparkRequestRecord,
    read_json_list,
    write_json,
)

GRID = 1000


def settings(th: dict | None) -> dict[str, Any] | None:
    """thresholds dedup.spark_halves_pass, or None when the run's thresholds predate it or it is off."""
    cfg = (th or {}).get("dedup", {}).get("spark_halves_pass")
    return cfg if isinstance(cfg, dict) and cfg.get("enabled", True) else None


class HalvesAsk(Contract):
    """One marks_v1 call on one enlarged half of a keyframe crop, and Spark's record."""

    kf_id: str
    role: str
    panel_id: PanelId
    half: int  # 0 the left or top half, 1 the right or bottom
    crop_path: str
    crop_sha256: Sha256 | None = None
    crop_w: int
    crop_h: int
    window: IntBox4  # crop pixels, half-open
    out_w: int
    out_h: int
    record: SparkRequestRecord


def windows(crop: Crop, s: dict[str, Any]) -> list[list[int]]:
    """Pure: the two halves along the crop's long side, span_frac of it each, in crop pixels (half-open)."""
    w, h = int(crop.w), int(crop.h)
    if w >= h:
        span = int(round(float(s["span_frac"]) * w))
        return [[0, 0, span, h], [w - span, 0, w, h]]
    span = int(round(float(s["span_frac"]) * h))
    return [[0, 0, w, span], [0, h - span, w, h]]


def out_size(win: list[int], s: dict[str, Any]) -> tuple[int, int]:
    k = float(s["scale"])
    return int(round((win[2] - win[0]) * k)), int(round((win[3] - win[1]) * k))


def plan(kf: Keyframe, s: dict[str, Any]) -> list[tuple[int, list[int]]]:
    return [] if kf.crop is None else list(enumerate(windows(kf.crop, s)))


async def ask(run_id: str, kf: Keyframe, crop_rgb: np.ndarray, half: int, win: list[int], s: dict[str, Any], *, th: dict) -> HalvesAsk:
    """LIVE: one marks_v1 call on the enlarged half, with the run's prompt sentences and effort and no landmarks."""
    from . import match, spark_client

    assert kf.crop is not None
    ow, oh = out_size(win, s)
    rgb = zoom_refine.zoom_image(crop_rgb, win, {"out_w": ow, "out_h": oh})
    result = await asyncio.to_thread(
        spark_client.find_marks, rgb, kf.panel_id, landmarks=[], effort=str(th["spark"]["effort"]), run_id=run_id,
        fixtures=match.fixtures_not_damage(th), seams=match.seams_not_marks(th),
    )
    return HalvesAsk(
        kf_id=kf.kf_id, role=kf.role, panel_id=kf.panel_id, half=half, crop_path=kf.crop.path, crop_sha256=kf.crop.sha256,
        crop_w=kf.crop.w, crop_h=kf.crop.h, window=win, out_w=ow, out_h=oh, record=result.record,
    )


def read_asks(run_id: str) -> list[HalvesAsk]:
    path = paths.halves_marks_json(run_id)
    return read_json_list(path, HalvesAsk) if path.is_file() else []


def write_asks(run_id: str, asks: list[HalvesAsk]) -> None:
    write_json(paths.halves_marks_json(run_id), asks)


def _key(sha: str | None, path: str, win: list[int]) -> tuple[str, int, int, int, int]:
    return (sha or path, *(int(v) for v in win))  # type: ignore[return-value]


class Index:
    """Stored halves answers by (crop sha256, window): a later answer of the same key wins."""

    def __init__(self, asks: list[HalvesAsk]) -> None:
        self._by: dict[tuple, HalvesAsk] = {_key(a.crop_sha256, a.crop_path, a.window): a for a in asks}

    def get(self, crop: Crop, win: list[int]) -> HalvesAsk | None:
        return self._by.get(_key(crop.sha256, crop.path, win))

    def has(self, crop: Crop, win: list[int]) -> bool:
        return _key(crop.sha256, crop.path, win) in self._by


def _centre(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _inside(p: tuple[float, float], b: list[float]) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _grid(v: float, dim: int) -> int:
    return int(min(GRID, max(0, round(v * GRID / dim))))


def same_mark(box: list[float], w: list[float] | None, kept: list[tuple[list[float], list[float] | None]], *, dup_iou: float, radius: float, dist: Callable[[Any, Any], float]) -> bool:
    """Pure: is this box the same mark as one already kept on the crop."""
    c = _centre(box)
    for kb, kw in kept:
        if _inside(c, kb) or _inside(_centre(kb), box) or _iou(box, kb) > dup_iou:
            return True
        if w is not None and kw is not None and dist((w[0], w[1]), (kw[0], kw[1])) <= radius:
            return True
    return False


def halves_marks(base: list[Mark], kf: Keyframe, index: Index, *, th: dict, mask: np.ndarray | None = None) -> list[Mark]:
    """Pure: the crop's halves boxes that are not the same mark as a box already kept (base, then earlier halves boxes),
    as Spark box marks with Mark.halves set, in half and answer order."""
    from .marks import _metric, whiten_marks

    s = settings(th)
    if s is None or kf.crop is None:
        return []
    crop = kf.crop
    small_px = float(th["dedup"]["small_mark_px"])
    radius = float(th["dedup"]["cluster_radius_units"])
    dist = _metric(kf.panel_id)
    placed, _ = whiten_marks([m for m in base if m.kf_id == kf.kf_id and m.bbox_crop is not None], kf, th=th, mask=mask)
    kept: list[tuple[list[float], list[float] | None]] = [([float(v) for v in m.bbox_crop or []], m.w) for m in placed]
    out: list[Mark] = []
    for half, win in plan(kf, s):
        a = index.get(crop, win)
        boxes = zoom_refine.zoom_boxes(a) if a is not None else None  # type: ignore[arg-type]
        if not boxes:
            continue
        for box, sm in boxes:
            box = [max(0.0, box[0]), max(0.0, box[1]), min(float(crop.w), box[2]), min(float(crop.h), box[3])]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            bbox_src = geometry.box_crop_to_src(box, crop)
            size_px = max(bbox_src[2] - bbox_src[0], bbox_src[3] - bbox_src[1])
            caps = [c for c, on in (("K3", size_px < small_px), ("K4", sm.type in CAPPED_MARK_KINDS), ("K12", sm.confidence == Confidence.LOW)) if on]
            mark = Mark(
                mark_id=f"{kf.kf_id}_h{len(out) + 1}", role=kf.role, panel_id=kf.panel_id, kf_id=kf.kf_id, source=MarkSource.SPARK_BOX,
                type=sm.type, confidence=sm.confidence, size_class=sm.size_class, description=sm.note,
                grid_point=GridPoint(x=_grid(point[0], crop.w), y=_grid(point[1], crop.h)),
                grid_bbox=GridBox(x_min=_grid(box[0], crop.w), y_min=_grid(box[1], crop.h), x_max=_grid(box[2], crop.w), y_max=_grid(box[3], crop.h)),
                bbox_crop=[round(v, 3) for v in box], point_crop=[round(v, 3) for v in point], bbox_src=bbox_src,
                point_src=geometry.crop_to_src(point, crop), size_px_src=size_px, spark_request_id=a.record.request_id, caps=caps,  # type: ignore[union-attr]
                halves=HalvesFind(half=half, window_crop=list(win), boxes=len(boxes), spark_request_id=a.record.request_id),  # type: ignore[union-attr]
            )
            [w_mark], _ = whiten_marks([mark], kf, th=th, mask=mask)
            if same_mark(mark.bbox_crop or [], w_mark.w, kept, dup_iou=float(s["dup_iou"]), radius=radius, dist=dist):
                continue
            kept.append((list(mark.bbox_crop or []), w_mark.w))
            out.append(mark)
    return out


def drop_after_refine(marks: list[Mark], kf: Keyframe, *, th: dict, mask: np.ndarray | None = None) -> list[Mark]:
    """Pure: with the leaf's after_refine key, the crop's marks once the zoom move and the SAM step have run, less each
    halves mark whose box is now the same mark (same_mark) as a full-crop mark or an earlier halves mark. A full-crop box
    a patch beside its mark and a halves box on it look like two marks until both are moved."""
    from .marks import _metric, whiten_marks

    s = settings(th)
    if s is None or not s.get("after_refine"):
        return list(marks)
    radius = float(th["dedup"]["cluster_radius_units"])
    dist = _metric(kf.panel_id)
    mine = [m for m in marks if m.kf_id == kf.kf_id and m.bbox_crop is not None]
    placed = {m.mark_id: m for m in whiten_marks(mine, kf, th=th, mask=mask)[0]}
    kept = [([float(v) for v in m.bbox_crop or []], placed[m.mark_id].w) for m in mine if m.halves is None]
    out = []
    for m in marks:
        if m.halves is None or m.mark_id not in placed:
            out.append(m)
            continue
        box, w = [float(v) for v in m.bbox_crop or []], placed[m.mark_id].w
        if same_mark(box, w, kept, dup_iou=float(s["dup_iou"]), radius=radius, dist=dist):
            continue
        kept.append((box, w))
        out.append(m)
    return out


def _gap(a: list[float], b: list[float], gx: float, gy: float) -> tuple[float, bool]:
    """(centre gap in patches, the boxes overlap)."""
    (ax, ay), (bx, by) = _centre(a), _centre(b)
    overlap = min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
    return float(np.hypot((ax - bx) / gx, (ay - by) / gy)), overlap


def _take_box(full: Mark, h: Mark, gap: float, overlap: bool, crop: Crop, th: dict) -> Mark:
    """The full-crop mark with the halves mark's box (point, source box, size, K3 by px, and the zoom and SAM records that
    placed that box); type, confidence, note, id and the other caps stay the full crop's."""
    size_px = float(h.size_px_src or 0.0)
    tiny = size_px < float(th["dedup"]["small_mark_px"])
    caps = [c for c in full.caps if c != "K3"]
    if tiny:
        caps.append("K3")
    rec = HalvesMerge(
        halves_mark_id=h.mark_id, half=h.halves.half if h.halves is not None else 0, full_bbox_crop=list(full.bbox_crop or []),
        gap_patches=round(gap, 3), overlap=overlap, spark_request_id=h.spark_request_id,
    )
    return full.model_copy(update={
        "bbox_crop": h.bbox_crop, "point_crop": h.point_crop, "bbox_src": h.bbox_src, "point_src": h.point_src, "size_px_src": h.size_px_src,
        "caps": caps, "zoom_refine": h.zoom_refine, "box_refine": h.box_refine, "halves_merge": rec,
    })


def merge_after_refine(marks: list[Mark], kf: Keyframe, *, th: dict, mask: np.ndarray | None = None) -> list[Mark]:
    """Pure: with the leaf's merge_patches key, the crop's marks once the zoom move and the SAM step have run, with each
    halves mark that sits close to a full-crop mark (boxes overlap, or centres within merge_patches vision patches of the
    full crop) folded into it: one mark, the halves box, the full crop's type and confidence (closest pairs first, one
    halves box to a full-crop mark; a second halves box close to a full-crop mark is dropped). A halves mark with no
    full-crop mark close stays only if it is not the same mark (same_mark) as one kept and, with lone_uncapped, carries
    no cap (K3, K4, K12). Two candidates a patch apart for one mark make the match's ratio test refuse both."""
    from .marks import _metric, whiten_marks

    s = settings(th)
    if s is None or s.get("merge_patches") is None or kf.crop is None:
        return drop_after_refine(marks, kf, th=th, mask=mask)
    reach = float(s["merge_patches"])
    gx, gy = zoom_refine._patch_px(kf.crop, {"patch_sent_px": float(s.get("patch_sent_px", 28))})
    mine = [m for m in marks if m.kf_id == kf.kf_id and m.bbox_crop is not None]
    full = [m for m in mine if m.halves is None and m.source == MarkSource.SPARK_BOX]
    halves = [m for m in mine if m.halves is not None]
    pairs = []
    for i, h in enumerate(halves):
        for j, f in enumerate(full):
            gap, overlap = _gap([float(v) for v in h.bbox_crop or []], [float(v) for v in f.bbox_crop or []], gx, gy)
            if overlap or gap <= reach:
                pairs.append((gap, i, j, overlap))
    pairs.sort()
    near: set[int] = set()
    took: dict[str, Mark] = {}
    used: set[int] = set()
    for gap, i, j, overlap in pairs:
        near.add(i)
        if j in used or any(t.halves_merge is not None and t.halves_merge.halves_mark_id == halves[i].mark_id for t in took.values()):
            continue
        used.add(j)
        took[full[j].mark_id] = _take_box(full[j], halves[i], gap, overlap, kf.crop, th)
    radius = float(th["dedup"]["cluster_radius_units"])
    dist = _metric(kf.panel_id)
    settled = [took.get(m.mark_id, m) for m in mine if m.halves is None]
    placed = {m.mark_id: m for m in whiten_marks([*settled, *[h for i, h in enumerate(halves) if i not in near]], kf, th=th, mask=mask)[0]}
    kept = [([float(v) for v in m.bbox_crop or []], placed[m.mark_id].w if m.mark_id in placed else None) for m in settled]
    lone_ok: set[str] = set()
    for i, h in enumerate(halves):
        if i in near or h.mark_id not in placed or (s.get("lone_uncapped") and h.caps):
            continue
        box, w = [float(v) for v in h.bbox_crop or []], placed[h.mark_id].w
        if same_mark(box, w, kept, dup_iou=float(s["dup_iou"]), radius=radius, dist=dist):
            continue
        kept.append((box, w))
        lone_ok.add(h.mark_id)
    out = []
    for m in marks:
        if m.kf_id != kf.kf_id or m.bbox_crop is None:
            out.append(m)
        elif m.halves is None:
            out.append(took.get(m.mark_id, m))
        elif m.mark_id in lone_ok:
            out.append(m)
    return out


async def ask_missing(run_id: str, jobs: list[tuple[Keyframe, Path, list[tuple[int, list[int]]]]], s: dict[str, Any], *, th: dict, read_rgb: Any) -> list[HalvesAsk]:
    """LIVE: one call per (keyframe, half) under orchestrator.spark_semaphore; each crop is read once."""
    semaphore = asyncio.Semaphore(int(th["orchestrator"]["spark_semaphore"]))

    async def one(kf: Keyframe, rgb: np.ndarray, half: int, win: list[int]) -> HalvesAsk:
        async with semaphore:
            return await ask(run_id, kf, rgb, half, win, s, th=th)

    calls = []
    for kf, crop_path, todo in jobs:
        rgb = read_rgb(crop_path)
        calls.extend(one(kf, rgb, half, win) for half, win in todo)
    return list(await asyncio.gather(*calls))


__all__ = ["HalvesAsk", "Index", "ask", "ask_missing", "drop_after_refine", "halves_marks", "merge_after_refine", "out_size", "plan", "read_asks", "same_mark", "settings", "windows", "write_asks"]
