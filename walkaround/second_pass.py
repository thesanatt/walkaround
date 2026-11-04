"""The second marks pass for silent misses (friend run 1, 11/2; thresholds matching.second_marks_pass, off when absent).

On friend_pair1 Muse Spark returned no mark on a panel in one walk while the other walk's crops held one there, on views
that passed every gate. Here, once the panels are gated, each panel that is comparable (every gate passed on its
keyframe pair) with no mark in one walk and at least one in the other gets one more marks_v1 call per other-walk mark:
on that walk's compared crop, a window around where the other walk's mark maps through the panel frame (whitened in
its own keyframe, the handle residual applied as the match applies it, unwhitened in this crop; the zoom refine's
window and enlargement, dedup.spark_zoom_refine), with the run's prompt sentences and effort. The returned box nearest
the mapped spot becomes a mark of this walk with Spark's type, confidence and size and the caps every Spark mark gets
(K3, K4, K12 here; K2, K15, K16, K17 and the rest downstream), and Mark.second_pass records it. Nothing is lowered or
raised by the pass itself: the match decides, and a pair inside tau_pre still goes to the second look. At most
max_asks calls per run, surest other-walk marks first. Wheels are left out (their marks sit in wheel coordinates).

The answers are frozen in runs/<id>/second_pass.json; the match stage and rebuild add their marks to marks.json with no
call (apply_second_pass), after the narration seeds and before the match.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from . import geometry, paths, zoom_refine
from .contracts import (
    CAPPED_MARK_KINDS,
    WHEEL_PANELS,
    Confidence,
    Contract,
    IntBox4,
    Keyframe,
    Mark,
    MarkSource,
    MatchState,
    PanelId,
    PanelMatch,
    Point2,
    SecondPass,
    Sha256,
    SparkRequestRecord,
    read_json_list,
    write_json,
)

ROLES = ("pickup", "return")
CONFIDENCE_RANK = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}


def settings(th: dict | None) -> dict[str, Any] | None:
    """thresholds matching.second_marks_pass with the zoom refine's window settings under "window", or None when the
    leaf is absent or off, or the run has no dedup.spark_zoom_refine (the window it borrows)."""
    cfg = (th or {}).get("matching", {}).get("second_marks_pass")
    zoom = zoom_refine.settings(th)
    if not (isinstance(cfg, dict) and cfg.get("enabled", True)) or zoom is None:
        return None
    return {**cfg, "window": zoom}


class SecondPassAsk(Contract):
    """One second-pass marks_v1 call: the walk's compared crop, the window enlarged and sent, the other walk's mark and
    where it maps in this crop, and Spark's record."""

    kf_id: str
    role: str
    panel_id: PanelId
    other_mark: str
    crop_path: str
    crop_sha256: Sha256 | None = None
    crop_w: int
    crop_h: int
    window: IntBox4  # crop pixels, half-open
    expected_crop: Point2
    out_w: int
    out_h: int
    record: SparkRequestRecord


def read_asks(run_id: str) -> list[SecondPassAsk]:
    path = paths.second_pass_json(run_id)
    return read_json_list(path, SecondPassAsk) if path.is_file() else []


def write_asks(run_id: str, asks: list[SecondPassAsk]) -> None:
    write_json(paths.second_pass_json(run_id), asks)


def _panel_mark_ids(pm: PanelMatch) -> dict[str, list[str]]:
    return {
        "pickup": [p.pickup_mark for p in pm.pairs] + list(pm.unmatched_pickup),
        "return": [p.return_mark for p in pm.pairs] + list(pm.unmatched_return),
    }


def plan(matches: list[PanelMatch], marks: dict[str, Mark], keyframes: dict[str, Keyframe], s: dict[str, Any]) -> list[tuple[Keyframe, Mark, Point2, list[int]]]:
    """Pure: (the silent walk's compared keyframe, the other walk's mark, where it maps in that crop, the window) for
    every comparable panel with no mark in one walk and marks in the other, surest other-walk marks first (then panel
    and mark id), one per (keyframe, window), at most max_asks."""
    from .match import counter_spot

    out: list[tuple[tuple[int, str], tuple[Keyframe, Mark, Point2, list[int]]]] = []
    for pm in matches:
        if pm.state != MatchState.COMPARABLE or pm.kf_pair is None or pm.panel_id in WHEEL_PANELS:
            continue
        ids = _panel_mark_ids(pm)
        for silent, other in (("pickup", "return"), ("return", "pickup")):
            if ids[silent] or not ids[other]:
                continue
            kf = keyframes.get(pm.kf_pair[0 if silent == "pickup" else 1])
            if kf is None or kf.crop is None or kf.moments is None or kf.front_sign is None:
                continue
            for mid in ids[other]:
                m = marks.get(mid)
                if m is None or m.source == MarkSource.NARRATION_SEED or m.w is None:
                    continue
                spot = counter_spot(m, keyframes.get(m.kf_id), kf, pm.handle_residual, 0.0)
                if spot is None:
                    continue
                (u, v), box = spot
                cx, cy = geometry.unwhiten((u, v), kf.moments, kf.front_sign)
                win = zoom_refine.window(list(box), kf.crop, s["window"])
                out.append(((CONFIDENCE_RANK.get(m.confidence, 3), f"{pm.panel_id.value} {m.mark_id}"), (kf, m, [round(float(cx), 3), round(float(cy), 3)], win)))
    seen: set[tuple[str, tuple[int, ...]]] = set()
    picked = []
    for _key, item in sorted(out, key=lambda kv: kv[0]):
        k = (item[0].kf_id, tuple(item[3]))
        if k in seen:
            continue
        seen.add(k)
        picked.append(item)
    return picked[: int(s.get("max_asks", 12))]


async def ask(run_id: str, kf: Keyframe, crop_rgb: np.ndarray, other: Mark, expected: Point2, win: list[int], s: dict[str, Any], *, th: dict) -> SecondPassAsk:
    """LIVE: one marks_v1 call on the enlarged window, with the run's prompt sentences and effort and no landmarks."""
    from . import match, spark_client

    assert kf.crop is not None
    rgb = zoom_refine.zoom_image(crop_rgb, win, s["window"])
    result = await asyncio.to_thread(
        spark_client.find_marks, rgb, kf.panel_id, landmarks=[], effort=str(th["spark"]["effort"]), run_id=run_id,
        fixtures=match.fixtures_not_damage(th), seams=match.seams_not_marks(th), colour=match.colour_on_paint(th), panel_conditions=match.conditions_on_panel(th),
    )
    return SecondPassAsk(
        kf_id=kf.kf_id, role=kf.role, panel_id=kf.panel_id, other_mark=other.mark_id, crop_path=kf.crop.path, crop_sha256=kf.crop.sha256,
        crop_w=kf.crop.w, crop_h=kf.crop.h, window=win, expected_crop=expected, out_w=int(s["window"]["out_w"]), out_h=int(s["window"]["out_h"]),
        record=result.record,
    )


def _nearest(boxes: list[tuple[list[float], Any]], p: Point2) -> tuple[list[float], Any]:
    return min(boxes, key=lambda bs: ((bs[0][0] + bs[0][2]) / 2 - p[0]) ** 2 + ((bs[0][1] + bs[0][3]) / 2 - p[1]) ** 2)


def outcomes(asks: list[SecondPassAsk], keyframes: dict[str, Keyframe], *, th: dict, root: Path | None = None) -> list[tuple[SecondPassAsk, Mark | None, str]]:
    """Pure but for reading keyframe masks under root: (ask, the mark it gave or None, what it found in words) per
    stored ask. One mark per ask whose answer holds a box (the box nearest the mapped spot, in crop pixels), ids
    <kf_id>_s<n> in ask order per keyframe, caps K3 (px, then units), K4 and K12, whitened in its keyframe. A mark within
    dedup.cluster_radius_units of an earlier second-pass mark of the same keyframe is the same mark and is not added.
    With the leaf's on_panel key, dedup.marks_on_panel on and root given, a box whose centre lies off the keyframe's
    panel mask (the on-panel rule of the marks stage) is not used: the pass looks for a mark of this panel."""
    from .marks import _off, _read_mask, on_panel_frac, whiten_marks

    small_px = float(th["dedup"]["small_mark_px"])
    radius = float(th["dedup"]["cluster_radius_units"])
    s = settings(th) or {}
    frac = on_panel_frac(th) if s.get("on_panel") and root is not None else None
    count: dict[str, int] = {}
    out: list[tuple[SecondPassAsk, Mark | None, str]] = []
    kept: list[Mark] = []
    for a in asks:
        count[a.kf_id] = count.get(a.kf_id, 0) + 1
        kf = keyframes.get(a.kf_id)
        if kf is None or kf.crop is None or kf.panel_id in WHEEL_PANELS:
            out.append((a, None, "keyframe not found"))
            continue
        boxes = zoom_refine.zoom_boxes(a)  # type: ignore[arg-type]
        if not boxes:
            out.append((a, None, "no mark there" if boxes is not None else "the answer was cut short or not valid"))
            continue
        box, sm = _nearest(boxes, a.expected_crop)
        box = [max(0.0, box[0]), max(0.0, box[1]), min(float(kf.crop.w), box[2]), min(float(kf.crop.h), box[3])]
        point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        bbox_src = geometry.box_crop_to_src(box, kf.crop)
        size_px = max(bbox_src[2] - bbox_src[0], bbox_src[3] - bbox_src[1])
        caps = [c for c, on in (("K3", size_px < small_px), ("K4", sm.type in CAPPED_MARK_KINDS), ("K12", sm.confidence == Confidence.LOW)) if on]
        mark = Mark(
            mark_id=f"{a.kf_id}_s{count[a.kf_id]}", role=kf.role, panel_id=kf.panel_id, kf_id=kf.kf_id, source=MarkSource.SPARK_BOX,
            type=sm.type, confidence=sm.confidence, size_class=sm.size_class, description=sm.note,
            bbox_crop=[round(v, 3) for v in box], point_crop=[round(v, 3) for v in point], bbox_src=bbox_src,
            point_src=geometry.crop_to_src(point, kf.crop), size_px_src=size_px, spark_request_id=a.record.request_id, caps=caps,
            second_pass=SecondPass(other_mark=a.other_mark, window_crop=list(a.window), expected_crop=list(a.expected_crop), boxes=len(boxes), spark_request_id=a.record.request_id),
        )
        if frac is not None and kf.mask_png:
            mask = _read_mask(root / kf.mask_png)  # type: ignore[operator]
            if mask is not None:
                on, off_frac = _off(mask, point, kf.crop, frac)
                if not on:
                    out.append((a, None, f"a box {off_frac * 100:.1f}% of the crop's long side off the panel, not used"))
                    continue
        [mark], _w = whiten_marks([mark], kf, th=th)
        same = next((o for o in kept if o.kf_id == mark.kf_id and o.w is not None and mark.w is not None and float(np.hypot(o.w[0] - mark.w[0], o.w[1] - mark.w[1])) <= radius), None)
        if same is not None:
            out.append((a, None, f"the same mark as {same.mark_id}"))
            continue
        kept.append(mark)
        out.append((a, mark, f"found {mark.mark_id}"))
    return out


def marks_from_asks(asks: list[SecondPassAsk], keyframes: dict[str, Keyframe], *, th: dict, root: Path | None = None) -> list[Mark]:
    """The marks of outcomes(), in ask order."""
    return [m for _a, m, _w in outcomes(asks, keyframes, th=th, root=root) if m is not None]


def outcome_words(run_id: str, *, th: dict) -> dict[PanelId, list[str]]:
    """The second marks pass per panel in words, from the stored asks: the crop looked at, the other walk's mark whose
    spot it was, and what was found."""
    if settings(th) is None:
        return {}
    keyframes = {kf.kf_id: kf for kf in read_json_list(paths.keyframes_json(run_id), Keyframe)} if paths.keyframes_json(run_id).is_file() else {}
    out: dict[PanelId, list[str]] = {}
    for a, _m, words in outcomes(read_asks(run_id), keyframes, th=th, root=paths.run_dir(run_id)):
        out.setdefault(a.panel_id, []).append(f"looked again at {a.kf_id} around {a.other_mark}: {words}")
    return out


def apply_second_pass(run_id: str, *, th: dict) -> int:
    """No call: marks.json without any earlier second-pass row, plus the marks of every stored ask. Leaves the file
    untouched when the leaf is off, and rewrites it only when its rows change. Returns the number of second-pass marks."""
    from .marks import _read_list as read_marks

    if settings(th) is None:
        return 0
    path = paths.marks_json(run_id)
    if not path.is_file():
        return 0
    marks = read_marks(path, Mark)
    base = [m for m in marks if m.second_pass is None]
    keyframes = {kf.kf_id: kf for kf in read_json_list(paths.keyframes_json(run_id), Keyframe)} if paths.keyframes_json(run_id).is_file() else {}
    added = marks_from_asks(read_asks(run_id), keyframes, th=th, root=paths.run_dir(run_id))
    rows = [*base, *added]
    if [m.model_dump(mode="json") for m in rows] != [m.model_dump(mode="json") for m in marks]:
        write_json(path, rows)
    return len(added)


async def derive_second_pass(run_id: str, *, th: dict) -> list[tuple[str, SparkRequestRecord]]:
    """LIVE: gate and match the base marks (no look), plan the asks, send the ones not stored yet (at most max_asks per
    run, stored ones counted), append them to second_pass.json, then apply_second_pass. Returns (role, record) for each
    new call. Nothing happens when the leaf is off."""
    from . import match
    from .marks import _read_list as read_marks
    from .marks import _read_rgb

    s = settings(th)
    if s is None:
        return []
    path = paths.marks_json(run_id)
    if not path.is_file():
        return []
    marks = read_marks(path, Mark)
    base = [m for m in marks if m.second_pass is None]
    if len(base) != len(marks):
        write_json(path, base)
    matches, _verdicts, _cal = await asyncio.to_thread(match.derive_match, run_id, th=th)
    keyframes = {kf.kf_id: kf for kf in read_json_list(paths.keyframes_json(run_id), Keyframe)}
    stored = read_asks(run_id)
    known = {(a.kf_id, a.other_mark, tuple(a.window)) for a in stored}
    room = max(0, int(s.get("max_asks", 12)) - len(stored))
    todo = [p for p in plan(matches, {m.mark_id: m for m in base}, keyframes, s) if (p[0].kf_id, p[1].mark_id, tuple(p[3])) not in known][:room]
    new: list[SecondPassAsk] = []
    if todo:
        semaphore = asyncio.Semaphore(int(th["orchestrator"]["spark_semaphore"]))
        root = paths.run_dir(run_id)
        crops: dict[str, np.ndarray] = {}

        async def one(kf: Keyframe, other: Mark, expected: Point2, win: list[int]) -> SecondPassAsk:
            async with semaphore:
                return await ask(run_id, kf, crops[kf.kf_id], other, expected, win, s, th=th)

        for kf, _m, _p, _w in todo:
            if kf.kf_id not in crops:
                crops[kf.kf_id] = _read_rgb(root / kf.crop.path)  # type: ignore[union-attr]
        new = list(await asyncio.gather(*(one(*p) for p in todo)))
        write_asks(run_id, [*stored, *new])
    apply_second_pass(run_id, th=th)
    return [(a.role, a.record) for a in new]


__all__ = ["SecondPassAsk", "apply_second_pass", "ask", "derive_second_pass", "marks_from_asks", "outcome_words", "outcomes", "plan", "read_asks", "settings", "write_asks"]
