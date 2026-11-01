"""Tight mark boxes from SAM 3.1 (box fix of 11/1; thresholds dedup.sam_box_refine, off when absent).

Muse Spark's mark boxes often sit a little off the mark: on drawn targets with exact positions its boxes miss by whole
steps of about 28 sent pixels (one vision patch), and on the synthetic car the median centre offset is about 14 crop
px (walkaround/docs/evidence/box_fix_1101/). Here each Spark box goes to SAM 3.1's image endpoint as a box prompt on
the same crop, all box marks of one crop in one request (up to max_objects objects; the masks of a nine-object request
matched the one-object requests exactly). SAM's mask gives the mark a tight box when the mask is small and sits on
Spark's box; otherwise Spark's box stays and the row says why. Spark's box and point are kept on the row
(Mark.box_refine), so both boxes are in marks.json; the mark's own boxes, point and size are those of the box used, so
matching, sizes, the looks and the UI all read the refined box. The answers are frozen in sam_marks.json and rebuild
replays them with no call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from . import geometry, paths
from .contracts import BoxRefine, Contract, Crop, IntBox4, Keyframe, Mark, MarkSource, Sha256, read_json_list, write_json


def settings(th: dict | None) -> dict[str, Any] | None:
    """thresholds dedup.sam_box_refine, or None when the run's thresholds predate it (Spark's boxes are used as before)."""
    cfg = (th or {}).get("dedup", {}).get("sam_box_refine")
    return cfg if isinstance(cfg, dict) and cfg.get("enabled", True) else None


class MarkObject(Contract):
    mark_id: str
    object_id: int
    prompt: IntBox4  # the box prompt sent, crop pixels, half-open


class BoxAsk(Contract):
    """One SAM image request on one keyframe crop: the box prompts of that crop's marks and SAM's answer."""

    kf_id: str
    role: str
    crop_path: str
    crop_sha256: Sha256 | None = None
    crop_w: int
    crop_h: int
    objects: list[MarkObject] = Field(default_factory=list)
    answer: Any = None  # claim.SamAnswer as a plain dict (masks kept as their wire payload)


def _key(sha: str | None, path: str, prompt: list[int]) -> tuple[str, int, int, int, int]:
    return (sha or path, *(int(v) for v in prompt))  # type: ignore[return-value]


def prompt_of(m: Mark, crop: Crop, s: dict[str, Any]) -> list[int]:
    from . import sam_client

    p = sam_client.prompt_box(list(m.bbox_crop or []), (crop.w, crop.h), pad_px=int(s["pad_px"]))
    return [int(p.left), int(p.top), int(p.right), int(p.bottom)]


def refinable(m: Mark) -> bool:
    return m.source == MarkSource.SPARK_BOX and m.bbox_crop is not None


def plan(marks: list[Mark], kf: Keyframe, s: dict[str, Any]) -> list[list[MarkObject]]:
    """The box prompts of one crop's Spark box marks, in mark order, max_objects to a request; ids from 1 per request."""
    if kf.crop is None:
        return []
    objs = [(m.mark_id, prompt_of(m, kf.crop, s)) for m in marks if m.kf_id == kf.kf_id and refinable(m)]
    size = max(1, int(s["max_objects"]))
    chunks = [objs[i : i + size] for i in range(0, len(objs), size)]
    return [[MarkObject(mark_id=mid, object_id=j + 1, prompt=p) for j, (mid, p) in enumerate(chunk)] for chunk in chunks]


def concept(objects: list[MarkObject], crop: Crop) -> str:
    from meta_sam_parser import FramePrompt, PromptBox, PromptObject

    from . import sam_client

    prompts = [PromptObject(id=o.object_id, prompts=[FramePrompt(frame=0, box=PromptBox(*o.prompt))]) for o in objects]
    return sam_client.object_prompt(prompts, (crop.w, crop.h))


async def ask(run_id: str, kf: Keyframe, crop_path: Path, objects: list[MarkObject]) -> BoxAsk:
    """LIVE: one SAM image request with the crop's box prompts. A request that did not complete is stored as such, so
    its marks keep Spark's box and say so."""
    from . import sam_client
    from .claim import SamAnswer, sam_answer_from_run

    assert kf.crop is not None
    text = concept(objects, kf.crop)
    try:
        run = await sam_client.segment_image(crop_path, text, run_id=run_id, role=kf.role, tag=f"marks_{kf.kf_id}")
        answer = sam_answer_from_run(run)
    except sam_client.SamRequestError as exc:
        status = exc.run.outcome if exc.run is not None else "failed"
        answer = SamAnswer(concept=text, status=status if status != "completed" else "failed", request_id=getattr(exc.run, "response_id", None), error=str(exc)[:300])
    return BoxAsk(kf_id=kf.kf_id, role=kf.role, crop_path=kf.crop.path, crop_sha256=kf.crop.sha256, crop_w=kf.crop.w, crop_h=kf.crop.h, objects=objects, answer=answer.model_dump(mode="json"))


def read_asks(run_id: str) -> list[BoxAsk]:
    path = paths.sam_marks_json(run_id)
    return read_json_list(path, BoxAsk) if path.is_file() else []


def write_asks(run_id: str, asks: list[BoxAsk]) -> None:
    write_json(paths.sam_marks_json(run_id), asks)


class Index:
    """Stored answers by (crop sha256, prompt box): a later answer of the same key wins, as with spark.json."""

    def __init__(self, asks: list[BoxAsk]) -> None:
        from .claim import SamAnswer

        self._by: dict[tuple, tuple[Any, Any]] = {}
        for a in asks:
            answer = SamAnswer.model_validate(a.answer) if a.answer is not None else None
            dets = {d.object_id: d for d in (answer.detections if answer is not None else [])}
            for o in a.objects:
                self._by[_key(a.crop_sha256, a.crop_path, o.prompt)] = (answer, dets.get(str(o.object_id)))

    def get(self, crop: Crop, prompt: list[int]) -> tuple[Any, Any] | None:
        return self._by.get(_key(crop.sha256, crop.path, prompt))

    def has(self, crop: Crop, prompt: list[int]) -> bool:
        return _key(crop.sha256, crop.path, prompt) in self._by


def _area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: list[float], b: list[float]) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def decide(m: Mark, answer: Any, det: Any, crop: Crop, s: dict[str, Any]) -> BoxRefine:
    """Pure: which box the mark uses. SAM's tight box when its mask has at least min_mask_px pixels, covers at most
    max_crop_frac of the crop, does not just fill the prompt (a tight box inside the prompt box and at least
    fills_box_min of its area is SAM segmenting the box itself, as on blank paint), is at most max_area_ratio times
    Spark's box, and at least min_inside of it lies inside Spark's box grown by max(margin_px, margin_frac x its long
    side); else Spark's box, with the reason."""
    from . import privacy
    from .claim import _detection

    spark = [float(v) for v in m.bbox_crop or []]
    base = {"spark_bbox_crop": spark, "spark_point_crop": m.point_crop}
    if answer is None:
        return BoxRefine(used="spark", note="no SAM answer is stored for this box", **base)
    rid = getattr(answer, "request_id", None)
    if answer.status != "completed":
        return BoxRefine(used="spark", sam_request_id=rid, note="the SAM check did not complete", **base)
    if det is None:
        return BoxRefine(used="spark", sam_request_id=rid, note="SAM found nothing in the box", **base)
    mask = privacy.detection_mask(_detection(det), (crop.w, crop.h))
    px = int(mask.sum())
    if px < int(s["min_mask_px"]):
        return BoxRefine(used="spark", sam_request_id=rid, sam_c=det.c, mask_px=px, note="SAM's mask is empty", **base)
    ys, xs = np.nonzero(mask)
    tight = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    margin = max(float(s["margin_px"]), float(s["margin_frac"]) * max(spark[2] - spark[0], spark[3] - spark[1]))
    grown = [spark[0] - margin, spark[1] - margin, spark[2] + margin, spark[3] + margin]
    inside = _inter(tight, grown) / max(_area(tight), 1e-9)
    ratio = _area(tight) / max(_area(spark), 1e-9)
    frac = px / float(crop.w * crop.h)
    rec = {**base, "sam_bbox_crop": tight, "sam_c": det.c, "mask_px": px, "inside": round(inside, 6), "area_ratio": round(ratio, 6), "sam_request_id": rid}
    if frac > float(s["max_crop_frac"]):
        return BoxRefine(used="spark", note=f"SAM's mask covers {100 * frac:.1f}% of the crop, more than a mark", **rec)
    prompt = [float(v) for v in prompt_of(m, crop, s)]
    within = tight[0] >= prompt[0] - 1 and tight[1] >= prompt[1] - 1 and tight[2] <= prompt[2] + 1 and tight[3] <= prompt[3] + 1
    if within and _area(tight) >= float(s["fills_box_min"]) * _area(prompt):
        return BoxRefine(used="spark", note="SAM's mask fills Spark's box, so it found no edge of a mark inside it", **rec)
    if ratio > float(s["max_area_ratio"]):
        return BoxRefine(used="spark", note=f"SAM's box is {ratio:.1f} times Spark's box, so it holds more than the mark", **rec)
    if inside < float(s["min_inside"]):
        return BoxRefine(used="spark", note=f"only {100 * inside:.0f}% of SAM's box sits on Spark's box", **rec)
    return BoxRefine(used="sam", **rec)


def apply(m: Mark, r: BoxRefine, crop: Crop, th: dict) -> Mark:
    """The mark with the box it uses: SAM's tight box (point at its centre, source box, size and K3 by px from it) or
    Spark's box unchanged; box_refine on the row either way."""
    if r.used != "sam" or r.sam_bbox_crop is None:
        return m.model_copy(update={"box_refine": r})
    b = list(r.sam_bbox_crop)
    point = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
    bbox_src = geometry.box_crop_to_src(b, crop)
    size_px = max(bbox_src[2] - bbox_src[0], bbox_src[3] - bbox_src[1])
    tiny = size_px < float(th["dedup"]["small_mark_px"])
    caps = list(m.caps)
    if tiny and "K3" not in caps:
        caps.append("K3")
    elif not tiny and "K3" in caps:
        caps = [c for c in caps if c != "K3"]
    return m.model_copy(update={"bbox_crop": b, "point_crop": point, "bbox_src": bbox_src, "point_src": geometry.crop_to_src(point, crop), "size_px_src": size_px, "caps": caps, "box_refine": r})


def refine_marks(marks: list[Mark], kf: Keyframe, index: Index, *, th: dict) -> list[Mark]:
    """Pure: the keyframe's Spark box marks with the box each uses, from the stored answers (no call)."""
    s = settings(th)
    if s is None or kf.crop is None:
        return list(marks)
    out = []
    for m in marks:
        if m.kf_id != kf.kf_id or not refinable(m):
            out.append(m)
            continue
        hit = index.get(kf.crop, prompt_of(m, kf.crop, s))
        answer, det = hit if hit is not None else (None, None)
        out.append(apply(m, decide(m, answer, det, kf.crop, s), kf.crop, th))
    return out


__all__ = ["BoxAsk", "Index", "MarkObject", "apply", "ask", "concept", "decide", "plan", "prompt_of", "read_asks", "refinable", "refine_marks", "settings", "write_asks"]
