"""Spark mark boxes moved onto the mark by a zoomed second look (zoom refine of 11/1; thresholds dedup.spark_zoom_refine,
off when absent).

Muse Spark's box often sits about one vision patch (28 sent px) beside the mark. Here each Spark box mark gets one more
marks_v1 call on a window cut around its box (short side max(min_px, window_patches patches, the box plus one patch a
side), long side to out_w:out_h, shifted to stay inside the crop) and enlarged to out_w x out_h. The returned box
nearest Spark's box centre gives the new centre; the mark keeps Spark's box size and type, moved there and shifted back
inside the crop. Scored on Sanat's 48 hand-moved boxes of pairC_core3 (walkaround/docs/evidence/zoom_refine_1101/):
median IoU with his box 0.149 to 0.353, centre on his mark 23 to 34 of 48, median centre offset 26.3 to 11.0 crop px.
The zoomed box itself is about a quarter of his box's area, so its size is not used.

The move runs right after the Spark answer is mapped and before the SAM tighten (box_refine), so SAM is prompted with
the moved box. Spark's own box and the zoom's box stay on the row (Mark.zoom_refine). The answers are frozen in
zoom_marks.json and rebuild replays them with no call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import geometry, paths
from .contracts import Contract, Crop, IntBox4, Keyframe, Mark, MarkSource, PanelId, Sha256, SparkRequestRecord, ZoomRefine, read_json_list, write_json


def settings(th: dict | None) -> dict[str, Any] | None:
    """thresholds dedup.spark_zoom_refine, or None when the run's thresholds predate it (Spark's boxes stay as they are)."""
    cfg = (th or {}).get("dedup", {}).get("spark_zoom_refine")
    return cfg if isinstance(cfg, dict) and cfg.get("enabled", True) else None


class ZoomAsk(Contract):
    """One zoomed marks_v1 call: the window of one keyframe crop that was enlarged and sent, and Spark's record."""

    kf_id: str
    role: str
    mark_id: str
    panel_id: PanelId
    crop_path: str
    crop_sha256: Sha256 | None = None
    crop_w: int
    crop_h: int
    window: IntBox4  # crop pixels, half-open
    out_w: int
    out_h: int
    record: SparkRequestRecord


def refinable(m: Mark) -> bool:
    return m.source == MarkSource.SPARK_BOX and m.bbox_crop is not None


def _patch_px(crop: Crop, s: dict[str, Any]) -> tuple[float, float]:
    from . import spark_client

    sw, sh = spark_client.sent_size(crop.w, crop.h)
    return float(s["patch_sent_px"]) * crop.w / sw, float(s["patch_sent_px"]) * crop.h / sh


def window(box: list[float], crop: Crop, s: dict[str, Any]) -> list[int]:
    """Pure: the window cut around a Spark box, in crop pixels (half-open), inside the crop."""
    gx, gy = _patch_px(crop, s)
    x1, y1, x2, y2 = (float(v) for v in box)
    lo, k = float(s["min_px"]), float(s["window_patches"])
    h = max(lo, k * gy, (y2 - y1) + 2 * gy)
    w = max(lo, k * gx, (x2 - x1) + 2 * gx)
    aspect = float(s["out_w"]) / float(s["out_h"])
    if w < h * aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(float(crop.w), w), min(float(crop.h), h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(0.0, cx - w / 2), crop.w - w)
    top = min(max(0.0, cy - h / 2), crop.h - h)
    return [int(round(left)), int(round(top)), int(round(left + w)), int(round(top + h))]


def zoom_image(crop_rgb: np.ndarray, win: list[int], s: dict[str, Any]) -> np.ndarray:
    """The window of the crop enlarged to out_w x out_h (per axis, so a window clamped by a small crop is stretched)."""
    x1, y1, x2, y2 = win
    img = Image.fromarray(np.ascontiguousarray(np.asarray(crop_rgb)[y1:y2, x1:x2, :3]))
    return np.asarray(img.resize((int(s["out_w"]), int(s["out_h"])), Image.Resampling.LANCZOS))


def plan(marks: list[Mark], kf: Keyframe, s: dict[str, Any]) -> list[tuple[Mark, list[int]]]:
    """The keyframe's Spark box marks with the window each is asked on, in mark order."""
    if kf.crop is None:
        return []
    return [(m, window(list(m.bbox_crop or []), kf.crop, s)) for m in marks if m.kf_id == kf.kf_id and refinable(m)]


async def ask(run_id: str, kf: Keyframe, crop_rgb: np.ndarray, m: Mark, win: list[int], s: dict[str, Any], *, th: dict) -> ZoomAsk:
    """LIVE: one marks_v1 call on the enlarged window, with the run's prompt sentences and effort and no landmarks."""
    from . import match, spark_client

    assert kf.crop is not None
    rgb = zoom_image(crop_rgb, win, s)
    result = await asyncio.to_thread(
        spark_client.find_marks, rgb, kf.panel_id, landmarks=[], effort=str(th["spark"]["effort"]), run_id=run_id,
        fixtures=match.fixtures_not_damage(th), seams=match.seams_not_marks(th),
    )
    return ZoomAsk(
        kf_id=kf.kf_id, role=kf.role, mark_id=m.mark_id, panel_id=kf.panel_id, crop_path=kf.crop.path, crop_sha256=kf.crop.sha256,
        crop_w=kf.crop.w, crop_h=kf.crop.h, window=win, out_w=int(s["out_w"]), out_h=int(s["out_h"]), record=result.record,
    )


def read_asks(run_id: str) -> list[ZoomAsk]:
    path = paths.zoom_marks_json(run_id)
    return read_json_list(path, ZoomAsk) if path.is_file() else []


def write_asks(run_id: str, asks: list[ZoomAsk]) -> None:
    write_json(paths.zoom_marks_json(run_id), asks)


def _key(sha: str | None, path: str, win: list[int]) -> tuple[str, int, int, int, int]:
    return (sha or path, *(int(v) for v in win))  # type: ignore[return-value]


class Index:
    """Stored zoom answers by (crop sha256, window): a later answer of the same key wins, as with spark.json."""

    def __init__(self, asks: list[ZoomAsk]) -> None:
        self._by: dict[tuple, ZoomAsk] = {_key(a.crop_sha256, a.crop_path, a.window): a for a in asks}

    def get(self, crop: Crop, win: list[int]) -> ZoomAsk | None:
        return self._by.get(_key(crop.sha256, crop.path, win))

    def has(self, crop: Crop, win: list[int]) -> bool:
        return _key(crop.sha256, crop.path, win) in self._by


def zoom_boxes(a: ZoomAsk) -> list[tuple[list[float], Any]] | None:
    """Every box of a stored zoom answer in crop pixels with its Spark mark, the half-open rule of marks.map_spark_mark;
    None when the answer was cut short or invalid."""
    from . import spark_client

    res = spark_client.marks_result_from_record(a.record, orig_width=a.out_w, orig_height=a.out_h, landmarks=[])
    if res.answer is None or res.truncated:
        return None
    x0, y0, x1, y1 = a.window
    size = (x1 - x0, y1 - y0)
    sent = (res.sent_width, res.sent_height)
    out = []
    for sm in res.answer.marks:
        a1, b1 = spark_client.grid_to_pixels(sm.bbox.x_min, sm.bbox.y_min, *sent)
        a2, b2 = spark_client.grid_to_pixels(sm.bbox.x_max, sm.bbox.y_max, *sent)
        c1 = geometry.sent_to_crop((a1, b1), sent, size)
        c2 = geometry.sent_to_crop((max(a2, a1) + 1, max(b2, b1) + 1), sent, size)
        out.append(([c1[0] + x0, c1[1] + y0, c2[0] + x0, c2[1] + y0], sm))
    return out


def _centre(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def decide(m: Mark, a: ZoomAsk | None, win: list[int], crop: Crop) -> ZoomRefine:
    """Pure: Spark's box moved so its centre is the centre of the zoom box nearest Spark's box centre, then shifted back
    inside the crop; Spark's box with the reason when there is no usable zoom answer."""
    spark = [float(v) for v in m.bbox_crop or []]
    base = {"spark_bbox_crop": spark, "spark_point_crop": m.point_crop, "window_crop": list(win)}
    if a is None:
        return ZoomRefine(used="spark", note="no zoom answer is stored for this box", **base)
    rid = a.record.request_id
    boxes = zoom_boxes(a)
    if boxes is None:
        return ZoomRefine(used="spark", spark_zoom_request_id=rid, note="the zoomed look was cut short or not valid", **base)
    if not boxes:
        return ZoomRefine(used="spark", spark_zoom_request_id=rid, zoom_boxes=0, note="the zoomed look found no mark near the box", **base)
    sx, sy = _centre(spark)
    zb, sm = min(boxes, key=lambda bs: (_centre(bs[0])[0] - sx) ** 2 + (_centre(bs[0])[1] - sy) ** 2)
    zx, zy = _centre(zb)
    moved = [spark[0] + zx - sx, spark[1] + zy - sy, spark[2] + zx - sx, spark[3] + zy - sy]
    dx = -min(0.0, moved[0]) - max(0.0, moved[2] - crop.w)
    dy = -min(0.0, moved[1]) - max(0.0, moved[3] - crop.h)
    moved = [moved[0] + dx, moved[1] + dy, moved[2] + dx, moved[3] + dy]
    return ZoomRefine(
        used="zoom", zoom_bbox_crop=[round(v, 3) for v in zb], zoom_type=sm.type, zoom_boxes=len(boxes), spark_zoom_request_id=rid,
        moved_bbox_crop=[round(v, 3) for v in moved], **base,
    )


def apply(m: Mark, r: ZoomRefine, crop: Crop, th: dict) -> Mark:
    """The mark with the box it uses: Spark's box moved by the zoom (point at its centre, source box and size from it,
    K3 by px) or Spark's box unchanged; zoom_refine on the row either way. Type, confidence and note stay Spark's."""
    if r.used != "zoom" or r.moved_bbox_crop is None:
        return m.model_copy(update={"zoom_refine": r})
    b = list(r.moved_bbox_crop)
    point = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
    bbox_src = geometry.box_crop_to_src(b, crop)
    size_px = max(bbox_src[2] - bbox_src[0], bbox_src[3] - bbox_src[1])
    tiny = size_px < float(th["dedup"]["small_mark_px"])
    caps = list(m.caps)
    if tiny and "K3" not in caps:
        caps.append("K3")
    elif not tiny and "K3" in caps:
        caps = [c for c in caps if c != "K3"]
    return m.model_copy(update={"bbox_crop": b, "point_crop": point, "bbox_src": bbox_src, "point_src": geometry.crop_to_src(point, crop), "size_px_src": size_px, "caps": caps, "zoom_refine": r})


def refine_marks(marks: list[Mark], kf: Keyframe, index: Index, *, th: dict) -> list[Mark]:
    """Pure: the keyframe's Spark box marks with the box each uses, from the stored zoom answers (no call)."""
    s = settings(th)
    if s is None or kf.crop is None:
        return list(marks)
    out = []
    for m in marks:
        if m.kf_id != kf.kf_id or not refinable(m):
            out.append(m)
            continue
        win = window(list(m.bbox_crop or []), kf.crop, s)
        out.append(apply(m, decide(m, index.get(kf.crop, win), win, kf.crop), kf.crop, th))
    return out


async def ask_missing(run_id: str, jobs: list[tuple[Keyframe, Path, list[tuple[Mark, list[int]]]]], s: dict[str, Any], *, th: dict, read_rgb: Any) -> list[ZoomAsk]:
    """LIVE: one zoom call per (keyframe, mark, window) under orchestrator.spark_semaphore; each crop is read once."""
    semaphore = asyncio.Semaphore(int(th["orchestrator"]["spark_semaphore"]))

    async def one(kf: Keyframe, rgb: np.ndarray, m: Mark, win: list[int]) -> ZoomAsk:
        async with semaphore:
            return await ask(run_id, kf, rgb, m, win, s, th=th)

    calls = []
    for kf, crop_path, todo in jobs:
        rgb = read_rgb(crop_path)
        calls.extend(one(kf, rgb, m, win) for m, win in todo)
    return list(await asyncio.gather(*calls))


__all__ = ["Index", "ZoomAsk", "apply", "ask", "ask_missing", "decide", "plan", "read_asks", "refinable", "refine_marks", "settings", "window", "write_asks", "zoom_boxes", "zoom_image"]
