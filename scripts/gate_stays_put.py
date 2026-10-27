"""D1 gate row E5, Stays Put: one real mark box tracked as its own object on a mark-centered 1008 px clip and on
the panel-crop clip, a reflection control from the same walk, drift in whitened units through geometry.py, the sign
test, "scuff" as a text prompt once, and the carried-moments check on 5 frames of one clip.

    LIVE=1 python scripts/gate_stays_put.py --src WALK.mp4 --pts 12.4 --mark X1 Y1 X2 Y2 --panel X1 Y1 X2 Y2 \
        [--panel-lane LANE.ndjson.gz --lane-fps 15 --lane-size W H] [--panel-noun "car door"] [--label demo_mark] \
        [--control] [--panned] [--backtrack] [--scuff-text] [--direction 1] [--sigma-frame 0.02] --out DIR
    LIVE=1 python scripts/gate_stays_put.py --plan clips.json --out DIR
    python scripts/gate_stays_put.py --fixture tests/fixtures/sam/pingpong_box.events.ndjson \
        [--text-fixture tests/fixtures/sam/pingpong_text.events.ndjson] [--cut-clip --clip data/test/pingpong.mp4] --out DIR

--plan takes a JSON list; each entry holds the single-clip arguments by name (src, pts, mark, panel, panel_lane,
lane_fps, lane_size, label, control, panned, backtrack, scuff_text, role, clip_types). Every entry runs as a
mark-centered clip and as a panel-crop clip (5.9 item 1 and section 16 item 7). Without LIVE=1 and without
--fixture the script prints what it would send and exits 2. Every call goes through walkaround.sam_client, which
writes the ledger rows; lane names start with "D1 E5". Every number printed also lands in <out>/gate_stays_put.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (str(REPO), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from meta_sam_parser import FramePrompt, PromptObject  # noqa: E402

import gate_sam  # noqa: E402
from walkaround import contracts, ledger  # noqa: E402
from walkaround.contracts import ClipRecord, IndexFrame, LaneRecord, LaneStatus, MarkTrack, Moments, Noun, StaysPutVerdict, TrackPoint  # noqa: E402

Box4 = list[float]
FIXTURE_BOX = REPO / "tests" / "fixtures" / "sam" / "pingpong_box.events.ndjson"
FIXTURE_TEXT = REPO / "tests" / "fixtures" / "sam" / "pingpong_text.events.ndjson"
TEST_CLIP = REPO / "data" / "test" / "pingpong.mp4"
MARK_OBJECT = "1"
PANEL_OBJECT = "2"
CLIP_TYPES = ("mark", "panel")
CHECK_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)  # the 5 frames of the carried-moments check


# ------------------------------------------------------------------------------------------------------------------
# Small geometry
# ------------------------------------------------------------------------------------------------------------------


def box_center(box: Box4) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_size(box: Box4) -> tuple[float, float]:
    return (box[2] - box[0], box[3] - box[1])


def box_aspect(box: Box4) -> float:
    w, h = box_size(box)
    return w / h if h > 0 else 0.0


def square_window(cx: float, cy: float, side: float) -> Box4:
    half = side / 2.0
    return [cx - half, cy - half, cx + half, cy + half]


def box_to_clip(box_src: Box4, window: Box4, size: tuple[int, int]) -> Box4:
    """Source pixels -> clip pixels of a size (w, h) clip cut from window."""
    fx, fy = size[0] / (window[2] - window[0]), size[1] / (window[3] - window[1])
    return [(box_src[0] - window[0]) * fx, (box_src[1] - window[1]) * fy, (box_src[2] - window[0]) * fx, (box_src[3] - window[1]) * fy]


def point_to_src(p_clip: tuple[float, float], window: Box4, size: tuple[int, int]) -> tuple[float, float]:
    sx, sy = (window[2] - window[0]) / size[0], (window[3] - window[1]) / size[1]
    return (window[0] + p_clip[0] * sx, window[1] + p_clip[1] * sy)


def contains(box: Box4 | None, p: tuple[float, float]) -> bool:
    return box is not None and box[0] <= p[0] < box[2] and box[1] <= p[1] < box[3]


def nearest_index(times: list[float], t: float) -> int:
    return int(min(range(len(times)), key=lambda i: abs(times[i] - t)))


# ------------------------------------------------------------------------------------------------------------------
# Panel boxes per clip frame and the windows (5.9 item 1)
# ------------------------------------------------------------------------------------------------------------------


def largest_box(frame: IndexFrame | None, object_id: str | None = None) -> list[int] | None:
    if frame is None:
        return None
    objs = [o for o in frame.objects if object_id is None or o.id == object_id]
    if not objs:
        return None
    best = max(objs, key=lambda o: (o.box[2] - o.box[0]) * (o.box[3] - o.box[1]))
    return list(best.box)


def panel_boxes_from_lane(
    times: list[float],
    frame_indices: list[int],
    lane_frames: list[IndexFrame],
    *,
    lane_fps: float,
    lane_size: tuple[int, int],
    src_size: tuple[int, int],
    object_id: str | None = None,
    bridge_s: float = 0.5,
) -> list[Box4 | None]:
    """The panel box per source frame, interpolated between the two nearest proxy frames of a SAM lane (5.9 item 1).
    Boxes come back in source pixels; a gap wider than bridge_s on both sides gives None."""
    by_frame = {fr.frame: largest_box(fr, object_id) for fr in lane_frames}
    by_frame = {k: v for k, v in by_frame.items() if v is not None}
    sx, sy = src_size[0] / lane_size[0], src_size[1] / lane_size[1]
    t0 = times[0] if times else 0.0
    bridge = int(round(bridge_s * lane_fps))
    out: list[Box4 | None] = []
    for f in frame_indices:
        p = (times[f] - t0) * lane_fps
        lo, hi = int(math.floor(p)), int(math.floor(p)) + 1
        a, b = by_frame.get(lo), by_frame.get(hi)
        if a is not None and b is not None:
            w = p - lo
            box = [a[i] * (1 - w) + b[i] * w for i in range(4)]
        elif a is not None or b is not None:
            box = list(a if a is not None else b)  # type: ignore[arg-type]
        else:
            near = [k for k in by_frame if abs(k - p) <= bridge]
            box = list(by_frame[min(near, key=lambda k: abs(k - p))]) if near else None
        out.append([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy] if box is not None else None)
    return out


def panel_boxes_linear(n: int, center: int, panel: Box4, start: Box4 | None = None, end: Box4 | None = None) -> list[Box4]:
    """The panel box per clip frame from hand-measured boxes at the start, the centre and the end (linear between)."""
    start = start or panel
    end = end or panel
    out: list[Box4] = []
    for f in range(n):
        if f <= center:
            w = f / center if center > 0 else 1.0
            a, b = start, panel
        else:
            w = (f - center) / (n - 1 - center) if n - 1 > center else 1.0
            a, b = panel, end
        out.append([a[i] * (1 - w) + b[i] * w for i in range(4)])
    return out


def windows_for(clip_type: str, *, mark_box: Box4, panel_boxes: list[Box4 | None], center: int, th: dict) -> list[Box4]:
    """mark: a square of 0.3 x the panel width (min 400 px) centred on the mark, moving with the panel box;
    panel: the panel box padded by crop_pad_frac, made square, centred on the panel."""
    sp = th["staysput"]
    pad = th["panels"]["crop_pad_frac"]
    ref = panel_boxes[center] if center < len(panel_boxes) else None
    mcx, mcy = box_center(mark_box)
    if ref is None:
        side = max(sp["window_min_px"], 3 * max(box_size(mark_box)))
        return [square_window(mcx, mcy, side) for _ in panel_boxes]
    rcx, rcy = box_center(ref)
    out: list[Box4] = []
    last: Box4 | None = None
    for box in panel_boxes:
        box = box if box is not None else last if last is not None else ref
        last = box
        pw, ph = box_size(box)
        pcx, pcy = box_center(box)
        if clip_type == "mark":
            side = max(sp["window_frac_of_panel_width"] * pw, sp["window_min_px"])
            out.append(square_window(mcx + (pcx - rcx), mcy + (pcy - rcy), side))
        else:
            side = max(pw, ph) * (1.0 + 2.0 * pad)
            out.append(square_window(pcx, pcy, side))
    return out


# ------------------------------------------------------------------------------------------------------------------
# Analysis (5.9 item 5, on boxes and carried moments)
# ------------------------------------------------------------------------------------------------------------------


def _sign(v: float) -> int:
    return 1 if v > 0 else -1 if v < 0 else 0


def analyse(
    track_frames: list[IndexFrame],
    mark_id: str,
    *,
    windows: list[Box4],
    clip_size: tuple[int, int],
    panel_boxes_src: list[Box4 | None],
    kf_moments: Moments,
    kf_panel_box: Box4,
    src_size: tuple[int, int],
    front_sign: int,
    direction: int | None,
    th: dict,
    sigma_frame: float | None = None,
    mask_inside: dict[int, bool] | None = None,
) -> tuple[dict[str, Any], list[TrackPoint]]:
    """b1, b2, b_pass, panned, drift against carried moments, the sign test and the verdict of 5.9 item 5."""
    sp = th["staysput"]
    by_frame = {fr.frame: fr for fr in track_frames}
    points: list[TrackPoint] = []
    rows: list[dict[str, Any]] = []
    for f, window in enumerate(windows):
        obj = next((o for o in by_frame[f].objects if o.id == mark_id), None) if f in by_frame else None
        if obj is None:
            continue
        box_clip = [float(v) for v in obj.box]
        c_src = point_to_src(box_center(box_clip), window, clip_size)
        panel = panel_boxes_src[f] if f < len(panel_boxes_src) else None
        inside = contains(panel, c_src)
        if mask_inside is not None and f in mask_inside:
            inside = inside and mask_inside[f]
        w: tuple[float, float] | None = None
        u_rel: float | None = None
        if panel is not None:
            pw, _ = box_size(panel)
            pcx, _ = box_center(panel)
            u_rel = (c_src[0] - pcx) / pw if pw > 0 else None
            try:
                w = gate_sam.whiten(c_src, gate_sam.carry_moments(kf_moments, kf_panel_box, panel), front_sign)
            except ValueError:
                w = None
        points.append(TrackPoint(f=f, box_clip=box_clip, c=obj.c, inside_panel=inside, w=list(w) if w else None, u_rel=u_rel))
        rows.append({"f": f, "c_src": c_src, "inside": inside, "w": w, "u": u_rel, "panel": panel})
    n_frames = len(windows)
    with_box = len(rows)
    box_share = with_box / n_frames if n_frames else 0.0
    inside_share = (sum(1 for r in rows if r["inside"]) / with_box) if with_box else 0.0
    ws = np.array([r["w"] for r in rows if r["w"] is not None and r["inside"]], dtype=float).reshape(-1, 2)
    drift = None
    if len(ws) >= 2:
        med = np.median(ws, axis=0)
        drift = float(math.sqrt(float(np.mean(np.sum((ws - med) ** 2, axis=1)))))
    panels = [(f, b) for f, b in enumerate(panel_boxes_src) if b is not None]
    b1 = b2 = None
    if len(panels) >= 2:
        (_, first), (_, last) = panels[0], panels[-1]
        b1 = math.dist(box_center(first), box_center(last)) / src_size[0]
        ar_f, ar_l = box_aspect(first), box_aspect(last)
        b2 = abs(math.log(ar_l / ar_f)) if ar_f > 0 and ar_l > 0 else None
    b_pass = (b1 is not None and b1 >= sp["b1_floor"]) or (b2 is not None and b2 >= sp["b2_floor"])
    panned = bool(b1 is not None and b1 < sp["b1_floor"] and b2 is not None and b2 >= sp["b2_floor"])
    gap = int(sp["sign_pair_gap_frames"])
    by_f = {r["f"]: r for r in rows}
    pairs = agree = 0
    skipped_no_direction = 0
    for f, r in by_f.items():
        s = by_f.get(f + gap)
        if s is None or r["u"] is None or s["u"] is None or r["panel"] is None or s["panel"] is None:
            continue
        dx = box_center(s["panel"])[0] - box_center(r["panel"])[0]
        if abs(dx) / gap > sp["sign_panel_dx_min_px_per_frame"]:
            s_cam = -_sign(dx)
        elif direction in (1, -1):
            s_cam = int(direction)
        else:
            skipped_no_direction += 1
            continue
        du = s["u"] - r["u"]
        pairs += 1
        agree += int(du != 0 and _sign(du) == s_cam)
    agreement = agree / pairs if pairs else None
    tau_fix = max(sp["tau_fix_floor"], sp["tau_fix_sigma_factor"] * sigma_frame) if sigma_frame else sp["tau_fix_floor"]
    sign_confirmed = agreement is not None and agreement > sp["sign_agreement_min"] and pairs >= sp["sign_pairs_min"]
    if box_share < sp["no_track_box_share_below"]:
        verdict = StaysPutVerdict.NO_TRACK
    elif not b_pass or drift is None:
        verdict = StaysPutVerdict.INCONCLUSIVE
    elif drift < tau_fix:
        verdict = StaysPutVerdict.FIXED
    elif drift > sp["sliding_factor"] * tau_fix:
        verdict = StaysPutVerdict.SLIDING
    else:
        verdict = StaysPutVerdict.INCONCLUSIVE
    cs = [r for r in points if r.c is not None]
    metrics = {
        "frames_total": n_frames,
        "frames_with_box": with_box,
        "box_share": round(box_share, 4),
        "inside_panel_share": round(inside_share, 4),
        "frames_whitened": int(len(ws)),
        "drift_rms_units": round(drift, 4) if drift is not None else None,
        "b1": round(b1, 4) if b1 is not None else None,
        "b2": round(b2, 4) if b2 is not None else None,
        "b_pass": bool(b_pass),
        "panned": panned,
        "sign_pairs": pairs,
        "sign_agreement": round(agreement, 4) if agreement is not None else None,
        "sign_confirmed": bool(sign_confirmed),
        "sign_pairs_skipped_no_direction": skipped_no_direction,
        "tau_fix": round(tau_fix, 4),
        "sigma_frame": sigma_frame,
        "verdict": verdict.value,
        "c_present_share": round(len(cs) / with_box, 4) if with_box else 0.0,
        "c_varies_per_frame": len({round(p.c, 6) for p in cs}) > 1 if cs else None,
        "c_median": round(statistics.median(p.c for p in cs), 4) if cs else None,
    }
    return metrics, points


def mark_track(label: str, role: str, metrics: dict[str, Any], points: list[TrackPoint], *, clip: ClipRecord | None, record: LaneRecord | None, prompt_text: str, is_control: bool) -> MarkTrack:
    return MarkTrack(
        mark_id=label,
        role=role,
        clip=clip,
        request_id=record.request_id if record else None,
        prompt_text=prompt_text,
        track=points,
        b1=metrics["b1"],
        b2=metrics["b2"],
        b_pass=metrics["b_pass"],
        panned=metrics["panned"],
        drift_rms=metrics["drift_rms_units"],
        sign_agreement=metrics["sign_agreement"],
        pairs_n=metrics["sign_pairs"],
        verdict=StaysPutVerdict(metrics["verdict"]),
        is_control=is_control,
    )


def carried_moments_check(
    lane: gate_sam.Lane,
    panel_id: str,
    mark_id: str,
    *,
    windows: list[Box4],
    clip_size: tuple[int, int],
    panel_boxes_src: list[Box4 | None],
    kf_moments: Moments,
    kf_panel_box: Box4,
    front_sign: int,
    th: dict,
) -> dict[str, Any]:
    """5.5: the keyframe moments carried by the box affine vs a full decode of the panel mask on 5 clip frames."""
    n = len(windows)
    frames = sorted({int(round(fr * (n - 1))) for fr in CHECK_FRACTIONS})
    rows = []
    for f in frames:
        panel_obj = next((o for o in lane.objects_at(f) if o.id == panel_id), None)
        mark_obj = next((o for o in lane.objects_at(f) if o.id == mark_id), None)
        box = panel_boxes_src[f] if f < len(panel_boxes_src) else None
        if panel_obj is None or mark_obj is None or box is None:
            rows.append({"f": f, "note": "panel mask, mark box or panel box missing"})
            continue
        try:
            mask = gate_sam.full_mask(lane, f, panel_id, clip_size)
            if mask is None or int(mask.sum()) < 3:
                rows.append({"f": f, "note": "no decodable panel mask"})
                continue
            decoded = gate_sam.moments_from_mask(mask)
            carried_src = gate_sam.carry_moments(kf_moments, kf_panel_box, box)
            carried = gate_sam.carry_moments(carried_src, windows[f], [0.0, 0.0, float(clip_size[0]), float(clip_size[1])])
            p = box_center([float(v) for v in mark_obj.box])
            w_dec = gate_sam.whiten(p, decoded, front_sign)
            w_car = gate_sam.whiten(p, carried, front_sign)
            rows.append(
                {
                    "f": f,
                    "mask_pixels": int(mask.sum()),
                    "centroid_px_diff": round(math.dist((decoded.cx, decoded.cy), (carried.cx, carried.cy)), 2),
                    "w_decoded": [round(v, 4) for v in w_dec],
                    "w_carried": [round(v, 4) for v in w_car],
                    "residual_units": round(math.dist(w_dec, w_car), 4),
                }
            )
        except Exception as exc:  # noqa: BLE001 - the number is the finding; a bad mask must not stop the gate
            rows.append({"f": f, "error": f"{type(exc).__name__}: {exc}"})
    residuals = [r["residual_units"] for r in rows if "residual_units" in r]
    limit = th["geometry"]["carried_moments_residual_max_units"]
    return {
        "frames": rows,
        "frames_checked": len(residuals),
        "residual_max_units": max(residuals) if residuals else None,
        "residual_mean_units": round(sum(residuals) / len(residuals), 4) if residuals else None,
        "limit_units": limit,
        "within_limit": (max(residuals) <= limit) if residuals else None,
    }


def scuff_report(text_lane: gate_sam.Lane, box_lane: gate_sam.Lane, *, mark_id: str, prompt_frame: int) -> dict[str, Any]:
    """What the text prompt "scuff" returned against the box-prompted track of the same clip."""
    shared = 0
    hits = 0
    iou_at_prompt = None
    for fr in box_lane.frames:
        mark = next((o for o in fr.objects if o.id == mark_id), None)
        if mark is None:
            continue
        objs = text_lane.objects_at(fr.frame)
        if not objs:
            continue
        shared += 1
        best = max(gate_sam.box_iou([float(v) for v in o.box], [float(v) for v in mark.box]) for o in objs)
        hits += int(best > 0.3)
        if fr.frame == prompt_frame:
            iou_at_prompt = round(best, 4)
    return {
        "status": text_lane.record.status.value,
        "ids": text_lane.record.ids,
        "frames_with_records": text_lane.record.frames_returned,
        "frames_shared_with_mark_track": shared,
        "frames_overlapping_mark_iou_over_0_3": hits,
        "overlap_share": round(hits / shared, 4) if shared else None,
        "iou_at_prompt_frame": iou_at_prompt,
        "dollars": text_lane.record.dollars,
    }


# ------------------------------------------------------------------------------------------------------------------
# Evaluation (the E5 pass, kill and the numbers the memo takes)
# ------------------------------------------------------------------------------------------------------------------


def clip_passes(metrics: dict[str, Any], th: dict) -> bool:
    sp = th["staysput"]
    return bool(
        metrics["box_share"] >= sp["d1_pass_box_share"]
        and metrics["inside_panel_share"] >= sp["d1_pass_box_share"]
        and metrics["drift_rms_units"] is not None
        and metrics["drift_rms_units"] < sp["d1_pass_drift_max"]
    )


def control_passes(metrics: dict[str, Any], th: dict) -> bool:
    sp = th["staysput"]
    return bool(
        metrics["drift_rms_units"] is not None
        and metrics["drift_rms_units"] > sp["d1_reflection_drift_min"]
        and metrics["sign_agreement"] is not None
        and metrics["sign_agreement"] > sp["sign_agreement_min"]
        and metrics["sign_pairs"] >= sp["sign_pairs_min"]
    )


def evaluate(clips: list[dict[str, Any]], th: dict) -> dict[str, Any]:
    sp = th["staysput"]
    demo = [c for c in clips if not c["is_control"] and not c["panned_pass"] and not c["backtrack"]]
    controls = [c for c in clips if c["is_control"]]
    panned = [c for c in clips if c["panned_pass"]]
    labels = sorted({c["label"] for c in demo})
    per_label = {}
    for label in labels:
        rows = [c for c in demo if c["label"] == label]
        per_label[label] = {ct: next((c["pass"] for c in rows if c["clip_type"] == ct), None) for ct in CLIP_TYPES}
    any_pass = any(v for d in per_label.values() for v in d.values())
    kill = bool(labels) and all(not any(v for v in d.values()) for d in per_label.values())
    inconclusive_panned = sum(1 for c in panned if c["metrics"]["verdict"] in (StaysPutVerdict.INCONCLUSIVE.value, StaysPutVerdict.NO_TRACK.value))
    b1s = [c["metrics"]["b1"] for c in clips if c["metrics"]["b1"] is not None]
    b2s = [c["metrics"]["b2"] for c in clips if c["metrics"]["b2"] is not None]
    sigma = next((c["metrics"]["sigma_frame"] for c in clips if c["metrics"].get("sigma_frame")), None)
    return {
        "demo_clips": len(demo),
        "demo_pass_by_label": per_label,
        "demo_pass_any": any_pass,
        "demo_pass_both_types_any_label": any(all(v for v in d.values()) for d in per_label.values()) if per_label else False,
        "kill": kill,
        "control_clips": len(controls),
        "control_pass": any(c["pass"] for c in controls) if controls else None,
        "panned_clips": len(panned),
        "panned_inconclusive_share": round(inconclusive_panned / len(panned), 4) if panned else None,
        "panned_share_over_half": (inconclusive_panned / len(panned) > sp["panned_inconclusive_share_caption_threshold"]) if panned else None,
        "tau_fix": round(max(sp["tau_fix_floor"], sp["tau_fix_sigma_factor"] * sigma), 4) if sigma else sp["tau_fix_floor"],
        "b1_observed": {"min": round(min(b1s), 4), "max": round(max(b1s), 4)} if b1s else None,
        "b2_observed": {"min": round(min(b2s), 4), "max": round(max(b2s), 4)} if b2s else None,
        "b1_floor": sp["b1_floor"],
        "b2_floor": sp["b2_floor"],
        "verdicts": {c["label"] + "/" + c["clip_type"]: c["metrics"]["verdict"] for c in clips},
    }


# ------------------------------------------------------------------------------------------------------------------
# Live path
# ------------------------------------------------------------------------------------------------------------------


def _video():
    video = gate_sam.optional_module("video")
    if video is None or not gate_sam.implemented(video, "cut_clip"):
        raise RuntimeError("walkaround.video.cut_clip is not implemented yet")
    return video


def _sam_client():
    sam_client = gate_sam.optional_module("sam_client")
    if sam_client is None or not gate_sam.implemented(sam_client, "segment_clip"):
        raise RuntimeError("walkaround.sam_client.segment_clip is not implemented yet")
    return sam_client


async def keyframe_mask(frame_rgb: np.ndarray, noun: str, mark_box: Box4, *, run_id: str, role: str, label: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    """The panel mask on the keyframe from one SAM image request: the detection whose box holds the mark."""
    sam_client = _sam_client()
    run = await sam_client.segment_image(frame_rgb, noun, run_id=run_id, role=role, tag=f"D1 E5 panel kf {label}")
    dets = [d for dets in run.frames.values() for d in dets]
    centre = box_center(mark_box)
    holding = [d for d in dets if d.box.left <= centre[0] < d.box.right and d.box.top <= centre[1] < d.box.bottom]
    best = max(holding, key=lambda d: d.confidence or 0.0, default=None)
    info = {"request_id": run.response_id, "matches": len(dets), "holding_mark": len(holding), "best_c": best.confidence if best else None, "dollars": run.usd}
    if best is None:
        return None, info
    return sam_client.full_mask(best, (frame_rgb.shape[1], frame_rgb.shape[0])), info


async def blur_boxes_for(frame_rgb: np.ndarray, *, run_id: str, role: str, label: str, th: dict) -> tuple[list[Box4], dict[str, Any]]:
    """Plate and face boxes from the two image checks on the unblurred prompt frame (step 7 order), dilated."""
    privacy = gate_sam.optional_module("privacy")
    if privacy is None or not gate_sam.implemented(privacy, "image_checks"):
        return [], {"note": "privacy.image_checks not implemented; clip stored unblurred only if no plate was in frame"}
    plates, faces, record = await privacy.image_checks(frame_rgb, run_id=run_id, role=role, tag=f"D1 E5 {label}", th=th)
    grow = th["staysput"]["clip_blur_dilation_frac"]
    boxes: list[Box4] = []
    for b in list(record.plate_boxes) + list(record.face_boxes):
        w, h = box_size(b)
        boxes.append([b[0] - grow * w / 2, b[1] - grow * h / 2, b[2] + grow * w / 2, b[3] + grow * h / 2])
    info = {"plates": len(plates), "faces": len(faces), "request_ids": record.request_ids, "plate_ruler_px": record.plate_ruler_px, "plate_ruler_c": record.plate_ruler_c}
    return boxes, info


async def run_clip(
    src: Path,
    *,
    pts_center: float,
    window_src: Box4,
    mark_box_src: Box4,
    out: Path,
    panel_box_src: Box4 | None = None,
    panel_boxes_src: list[Box4 | None] | None = None,
    clip_type: str = "mark",
    label: str = "mark",
    role: str = "pickup",
    run_id: str | None = None,
    th: dict | None = None,
    kf_moments: Moments | None = None,
    kf_panel_box: Box4 | None = None,
    front_sign: int = 1,
    direction: int | None = None,
    sigma_frame: float | None = None,
    blur_boxes: list[Box4] | None = None,
    is_control: bool = False,
    text_prompt: str | None = None,
) -> MarkTrack:
    """One clip: cut (blurred), prompt the mark as object 1 at frame 45 (and the panel as object 2 on a panel-crop
    clip), track, analyse. window_src is the window at the keyframe; it moves with panel_boxes_src when given."""
    th = th or contracts.load_thresholds()
    video = _video()
    sam_client = _sam_client()
    run_id = run_id or f"d1_{gate_sam.utc_ts()}_E5"
    size = int(th["media"]["clip_size_px"])
    n = int(th["media"]["clip_frames"])
    prompt_frame = int(th["media"]["clip_prompt_frame"])
    times = video.pts_map(Path(src))
    centre = nearest_index(times, pts_center)
    f0 = centre - n // 2
    if f0 < 0 or f0 + n > len(times):
        raise ValueError(f"{Path(src).name}: {n} frames around {pts_center:.3f} s run past the clip ends")
    boxes = panel_boxes_src if panel_boxes_src is not None else [panel_box_src] * n
    if any(b is not None for b in boxes):
        windows = windows_for(clip_type, mark_box=mark_box_src, panel_boxes=boxes, center=prompt_frame, th=th)
        if clip_type == "mark":
            # keep the caller's window size when it is larger than the rule's (a tape strip or a long scratch)
            side = max(window_src[2] - window_src[0], windows[prompt_frame][2] - windows[prompt_frame][0])
            windows = [square_window(*box_center(w), side) for w in windows]
    else:
        windows = [list(window_src) for _ in range(n)]
    out.mkdir(parents=True, exist_ok=True)
    clip_path = out / "clips" / f"{role}_{label}_{clip_type}.mp4"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    blur = [list(blur_boxes)] * n if blur_boxes else None
    t0 = time.monotonic()
    clip = video.cut_clip(Path(src), clip_path, pts_center=times[centre], frames=n, windows_src=windows, size=size, crf=int(th["media"]["clip_crf"]), blur_boxes=blur, block_px=int(th["privacy"]["pixel_block_px_at_4k"]))
    cut_s = round(time.monotonic() - t0, 3)
    clip_size = (size, size)
    objects = [PromptObject(id=int(MARK_OBJECT), prompts=[FramePrompt(frame=prompt_frame, box=sam_client.prompt_box(box_to_clip(mark_box_src, windows[prompt_frame], clip_size), clip_size, pad_px=int(th["panels"]["prompt_box_pad_px"])))])]
    panel_at_prompt = boxes[prompt_frame] if prompt_frame < len(boxes) else None
    if clip_type == "panel" and panel_at_prompt is not None:
        objects.append(PromptObject(id=int(PANEL_OBJECT), prompts=[FramePrompt(frame=prompt_frame, box=sam_client.prompt_box(box_to_clip(panel_at_prompt, windows[prompt_frame], clip_size), clip_size))]))
    prompt_text = text_prompt or sam_client.object_prompt(objects, (size, size))
    metadata = sam_client.sam_metadata(mask_encoding=None, propagation_direction=None if text_prompt else "both")
    lane_name = f"D1 E5 {label} {clip_type}" + (" scuff text" if text_prompt else "")
    record, frames = await sam_client.segment_clip(clip_path, prompt_text, run_id=run_id, role=role, lane=lane_name, media_size=(size, size), metadata=metadata, frames_sent=n)
    lane = gate_sam.Lane(record=record, frames=frames, reader="sam_wire", events_path=Path(record.lane_path) if record.lane_path else None)
    if lane.events_path is not None and lane.events_path.exists():
        lane.text = gate_sam.lane_text(gate_sam.read_events(lane.events_path))
    moments = kf_moments or gate_sam.rect_moments(panel_at_prompt or window_src)
    kf_box = kf_panel_box or panel_at_prompt or list(window_src)
    src_info = gate_sam.probe_media(Path(src))
    metrics, points = analyse(frames, MARK_OBJECT, windows=windows, clip_size=clip_size, panel_boxes_src=boxes, kf_moments=moments, kf_panel_box=kf_box, src_size=(src_info["width"], src_info["height"]), front_sign=front_sign, direction=direction, th=th, sigma_frame=sigma_frame)
    metrics["cut_s"] = cut_s
    metrics["lane_status"] = record.status.value
    metrics["first_delta_s"] = record.first_delta_s
    metrics["completed_s"] = record.completed_s
    metrics["dollars"] = record.dollars
    metrics["retries_429"] = record.retries
    metrics["video_frames_processed"] = record.video_frames_processed
    track = mark_track(label, role, metrics, points, clip=clip, record=record, prompt_text=prompt_text, is_control=is_control)
    track_json = out / f"{role}_{label}_{clip_type}.mark_track.json"
    contracts.write_json(track_json, track)
    extra = {"metrics": metrics, "windows": windows, "panel_boxes_src": boxes, "lane": lane, "track_path": str(track_json)}
    setattr(track, "_gate", extra)  # the runner reads the measurements without re-parsing the lane
    return track


async def run_entry(entry: dict[str, Any], *, out: Path, run_prefix: str, th: dict) -> dict[str, Any]:
    """One plan entry: the keyframe mask, blur boxes, both clip types, the carried-moments check, "scuff" once."""
    video = _video()
    src = Path(entry["src"])
    label = str(entry.get("label") or "mark")
    role = str(entry.get("role") or "pickup")
    run_id = f"{run_prefix}_E5_{label}"
    th_media = th["media"]
    n = int(th_media["clip_frames"])
    prompt_frame = int(th_media["clip_prompt_frame"])
    mark = [float(v) for v in entry["mark"]]
    panel = [float(v) for v in entry["panel"]] if entry.get("panel") else None
    info = gate_sam.probe_media(src)
    src_size = (info["width"], info["height"])
    times = video.pts_map(src)
    centre = nearest_index(times, float(entry["pts"]))
    f0 = centre - n // 2
    frame_indices = list(range(f0, f0 + n))
    panel_boxes: list[Box4 | None]
    panel_source = "static"
    if entry.get("panel_lane"):
        lane_events = gate_sam.read_events(Path(entry["panel_lane"]))
        lane = gate_sam.lane_from_events(lane_events, lane_id="panel lane", role=role, noun=None, prompt_text=str(entry.get("panel_noun") or "car door"))
        lane_size = tuple(entry.get("lane_size") or gate_sam.media_size_from_text(lane.text) or (960, 540))
        panel_boxes = panel_boxes_from_lane(times, frame_indices, lane.frames, lane_fps=float(entry.get("lane_fps") or th_media["proxy_fps"]), lane_size=(int(lane_size[0]), int(lane_size[1])), src_size=src_size, object_id=entry.get("panel_object_id"))
        panel_source = f"lane {entry['panel_lane']}"
        if panel_boxes[prompt_frame] is None and panel is not None:
            panel_boxes[prompt_frame] = panel
    elif panel is not None:
        panel_boxes = panel_boxes_linear(n, prompt_frame, panel, entry.get("panel_start"), entry.get("panel_end"))
        panel_source = "linear between the hand-measured boxes"
    else:
        panel_boxes = [None] * n
    frame_rgb = gate_sam.frame_at(src, times[centre])
    blur_boxes, blur_info = await blur_boxes_for(frame_rgb, run_id=run_id, role=role, label=label, th=th)
    kf_box = panel_boxes[prompt_frame] or panel
    kf_moments: Moments | None = None
    kf_info: dict[str, Any] = {"source": "box moments (no mask)"}
    if kf_box is not None and entry.get("panel_noun", "car door"):
        mask, kf_info = await keyframe_mask(frame_rgb, str(entry.get("panel_noun") or "car door"), mark, run_id=run_id, role=role, label=label)
        if mask is not None:
            kf_moments = gate_sam.moments_from_mask(mask)
            kf_info["source"] = "SAM image mask on the prompt frame"
            ys, xs = np.nonzero(mask)
            kf_box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    if kf_moments is None and kf_box is not None:
        kf_moments = gate_sam.rect_moments(kf_box)
    clips: list[dict[str, Any]] = []
    tracks: dict[str, MarkTrack] = {}
    sp = th["staysput"]
    side = max(sp["window_frac_of_panel_width"] * (box_size(kf_box)[0] if kf_box else 0.0), sp["window_min_px"])
    window_kf = square_window(*box_center(mark), side)
    for clip_type in entry.get("clip_types") or list(CLIP_TYPES):
        if clip_type == "panel" and kf_box is None:
            clips.append({"label": label, "clip_type": clip_type, "note": "no panel box; panel-crop clip skipped"})
            continue
        try:
            track = await run_clip(src, pts_center=times[centre], window_src=window_kf, mark_box_src=mark, out=out, panel_box_src=kf_box, panel_boxes_src=panel_boxes, clip_type=clip_type, label=label, role=role, run_id=run_id, th=th, kf_moments=kf_moments, kf_panel_box=kf_box, front_sign=int(entry.get("front_sign") or 1), direction=entry.get("direction"), sigma_frame=entry.get("sigma_frame"), blur_boxes=blur_boxes, is_control=bool(entry.get("control")))
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the row (section 14 item 35)
            clips.append({"label": label, "clip_type": clip_type, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "is_control": bool(entry.get("control")), "panned_pass": bool(entry.get("panned")), "backtrack": bool(entry.get("backtrack")), "pass": False, "metrics": {"verdict": StaysPutVerdict.NOT_RUN.value, "b1": None, "b2": None, "sigma_frame": None}})
            continue
        tracks[clip_type] = track
        extra = getattr(track, "_gate")
        metrics = extra["metrics"]
        row = {
            "label": label,
            "clip_type": clip_type,
            "role": role,
            "is_control": bool(entry.get("control")),
            "panned_pass": bool(entry.get("panned")),
            "backtrack": bool(entry.get("backtrack")),
            "clip": track.clip.model_dump() if track.clip else None,
            "request_id": track.request_id,
            "metrics": metrics,
            "pass": control_passes(metrics, th) if entry.get("control") else clip_passes(metrics, th),
            "track_path": extra["track_path"],
            "panel_boxes_source": panel_source,
            "keyframe_mask": kf_info,
            "blur": blur_info,
        }
        if clip_type == "panel":
            size_px = int(th_media["clip_size_px"])
            row["carried_moments_check"] = carried_moments_check(extra["lane"], PANEL_OBJECT, MARK_OBJECT, windows=extra["windows"], clip_size=(size_px, size_px), panel_boxes_src=extra["panel_boxes_src"], kf_moments=kf_moments or gate_sam.rect_moments(kf_box), kf_panel_box=kf_box, front_sign=int(entry.get("front_sign") or 1), th=th)
        clips.append(row)
    result: dict[str, Any] = {"label": label, "clips": clips}
    if entry.get("scuff_text") and "mark" in tracks:
        try:
            text_track = await run_clip(src, pts_center=times[centre], window_src=window_kf, mark_box_src=mark, out=out, panel_box_src=kf_box, panel_boxes_src=panel_boxes, clip_type="mark", label=label, role=role, run_id=run_id, th=th, kf_moments=kf_moments, kf_panel_box=kf_box, blur_boxes=blur_boxes, text_prompt=str(th["nouns"]["d1_text_prompts"][0]))
            result["scuff_text"] = scuff_report(getattr(text_track, "_gate")["lane"], getattr(tracks["mark"], "_gate")["lane"], mark_id=MARK_OBJECT, prompt_frame=prompt_frame)
        except Exception as exc:  # noqa: BLE001
            result["scuff_text"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return result


async def run_live(args: argparse.Namespace, th: dict) -> dict[str, Any]:
    out = Path(args.out)
    prefix = args.run_prefix or f"d1_{gate_sam.utc_ts()}"
    entries = plan_entries(args)
    results: dict[str, Any] = {"mode": "live", "run_prefix": prefix, "entries": [], "clips": []}
    for entry in entries:
        spend = gate_sam.spend_for_prefix(prefix)
        if args.budget is not None and spend >= args.budget:
            results["entries"].append({"label": entry.get("label"), "skipped": f"budget {args.budget} reached (spent {spend})"})
            continue
        row = await run_entry(entry, out=out, run_prefix=prefix, th=th)
        results["entries"].append(row)
        results["clips"].extend(c for c in row["clips"] if "metrics" in c)
    results["evaluation"] = evaluate(results["clips"], th)
    results["spend_usd"] = gate_sam.spend_for_prefix(prefix)
    return results


def plan_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.plan:
        entries = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise SystemExit("--plan must hold a JSON list of clip entries")
        return entries
    entry: dict[str, Any] = {
        "src": args.src,
        "pts": args.pts,
        "mark": args.mark,
        "panel": args.panel,
        "label": args.label,
        "role": args.role,
        "control": args.control,
        "panned": args.panned,
        "backtrack": args.backtrack,
        "scuff_text": args.scuff_text,
        "panel_noun": args.panel_noun,
        "direction": args.direction,
        "sigma_frame": args.sigma_frame,
    }
    if args.panel_lane:
        entry["panel_lane"] = args.panel_lane
        entry["lane_fps"] = args.lane_fps
        entry["lane_size"] = args.lane_size
    return [entry]


# ------------------------------------------------------------------------------------------------------------------
# Fixture path (no call): the 10/26 box-prompt stream as the mark track, a synthetic moving panel box
# ------------------------------------------------------------------------------------------------------------------


def synthetic_panel_boxes(n: int, media_size: tuple[int, int], *, px_per_frame: float = 1.2) -> list[Box4]:
    """A door-sized box that drifts right at px_per_frame (the camera walks), so b1 clears its floor."""
    w, h = media_size
    pw, ph = 0.55 * w, 0.75 * h
    x0, y0 = 0.2 * w, 0.15 * h
    return [[x0 + px_per_frame * f, y0, x0 + px_per_frame * f + pw, y0 + ph] for f in range(n)]


def run_fixture(args: argparse.Namespace, th: dict) -> dict[str, Any]:
    out = Path(args.out)
    fixture = Path(args.fixture)
    events = gate_sam.read_events(fixture)
    lane = gate_sam.lane_from_events(events, lane_id="D1 E5 fixture mark", role="pickup", noun=None, prompt_text="object prompt", kind="object", events_path=fixture)
    size_list = gate_sam.media_size_from_text(lane.text) or [1280, 720]
    media_size = (int(size_list[0]), int(size_list[1]))
    n = max((fr.frame for fr in lane.frames), default=0) + 1
    n = max(n, int(lane.record.video_frames_processed or 0))
    prompt_frame = 0  # the probe prompted at frame 0
    mark_id = lane.record.ids[0] if lane.record.ids else MARK_OBJECT
    windows = [[0.0, 0.0, float(media_size[0]), float(media_size[1])] for _ in range(n)]
    panel_boxes: list[Box4 | None] = list(synthetic_panel_boxes(n, media_size))
    kf_box = panel_boxes[prompt_frame]
    kf_moments = gate_sam.rect_moments(kf_box)  # type: ignore[arg-type]
    clips: list[dict[str, Any]] = []
    for clip_type, direction in (("mark", 1), ("panel", None)):
        # the clip is the whole source here, so clip pixels equal source pixels (the window is the full frame)
        metrics, points = analyse(lane.frames, mark_id, windows=windows, clip_size=media_size, panel_boxes_src=panel_boxes, kf_moments=kf_moments, kf_panel_box=kf_box, src_size=media_size, front_sign=1, direction=direction, th=th, sigma_frame=args.sigma_frame)
        track = mark_track("fixture_mark", "pickup", metrics, points, clip=None, record=lane.record, prompt_text="<0f>1<|box;...|> (fixture)", is_control=False)
        track_path = out / f"fixture_{clip_type}.mark_track.json"
        contracts.write_json(track_path, track)
        clips.append({"label": "fixture_mark", "clip_type": clip_type, "role": "pickup", "is_control": False, "panned_pass": False, "backtrack": False, "clip": None, "request_id": lane.record.request_id, "metrics": metrics, "pass": clip_passes(metrics, th), "track_path": str(track_path), "panel_boxes_source": "synthetic moving box (fixture)", "keyframe_mask": {"source": "box moments (fixture)"}, "note": "the ping pong ball stands in for the mark; the panel box is synthetic, so the verdict says nothing about a car"})
    results: dict[str, Any] = {"mode": "fixture", "run_prefix": None, "fixture": {"path": str(fixture), "sha256": gate_sam.sha256_file(fixture), "frames": n, "media_size": list(media_size), "ids": lane.record.ids, "status": lane.record.status.value}, "entries": [{"label": "fixture_mark", "clips": clips}], "clips": clips, "spend_usd": 0.0}
    # the carried-moments check needs a panel mask; the ball mask stands in for it on the fixture so the code runs
    ball_lane = gate_sam.Lane(record=lane.record, frames=lane.frames, text=lane.text, masks=lane.masks, reader=lane.reader, events_path=lane.events_path)
    ball_boxes: list[Box4 | None] = [largest_box(next((fr for fr in lane.frames if fr.frame == f), None), mark_id) for f in range(n)]
    ball_boxes = [[float(v) for v in b] if b else None for b in ball_boxes]
    kf_ball = ball_boxes[prompt_frame]
    if kf_ball is not None:
        ball_mask = gate_sam.full_mask(ball_lane, prompt_frame, mark_id, media_size)
        kf_m = gate_sam.moments_from_mask(ball_mask) if ball_mask is not None and ball_mask.sum() >= 3 else gate_sam.rect_moments(kf_ball)
        results["carried_moments_check"] = {**carried_moments_check(ball_lane, mark_id, mark_id, windows=windows, clip_size=media_size, panel_boxes_src=ball_boxes, kf_moments=kf_m, kf_panel_box=kf_ball, front_sign=1, th=th), "note": "ball mask as the panel stand-in (fixture)"}
    if args.text_fixture:
        text_events = gate_sam.read_events(Path(args.text_fixture))
        text_lane = gate_sam.lane_from_events(text_events, lane_id="D1 E5 scuff text fixture", role="pickup", noun=None, prompt_text="scuff", kind="text", events_path=Path(args.text_fixture))
        results["scuff_text"] = {**scuff_report(text_lane, lane, mark_id=mark_id, prompt_frame=prompt_frame), "note": "the text fixture's prompt was 'ping pong ball', standing in for 'scuff'"}
    if args.cut_clip:
        results["cut_clip_check"] = cut_clip_check(Path(args.clip) if args.clip else TEST_CLIP, out, th)
    results["evaluation"] = evaluate(clips, th)
    return results


def cut_clip_check(clip_src: Path, out: Path, th: dict) -> dict[str, Any]:
    """Pre-flight for tomorrow: cut one 91-frame 1008 px clip with a moving window from the test clip, no call."""
    if not clip_src.exists():
        return {"note": f"clip not present: {clip_src}"}
    try:
        video = _video()
    except RuntimeError as exc:
        return {"error": str(exc)}
    info = gate_sam.probe_media(clip_src)
    n = int(th["media"]["clip_frames"])
    size = int(th["media"]["clip_size_px"])
    times = video.pts_map(clip_src)
    centre = len(times) // 2
    side = max(th["staysput"]["window_min_px"], 0.3 * info["width"])
    windows = [square_window(info["width"] / 2 + 1.5 * (f - n // 2), info["height"] / 2, side) for f in range(n)]
    t0 = time.monotonic()
    try:
        record = video.cut_clip(clip_src, out / "clips" / "fixture_cut_check.mp4", pts_center=times[centre], frames=n, windows_src=windows, size=size, crf=int(th["media"]["clip_crf"]))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"path": record.path, "sha256": record.sha256, "w": record.w, "h": record.h, "frames": record.frames, "f0_src": record.f0_src, "window_px": record.window_px, "cut_s": round(time.monotonic() - t0, 3), "ok": record.w == size and record.h == size and record.frames == n}


# ------------------------------------------------------------------------------------------------------------------
# Summary and CLI
# ------------------------------------------------------------------------------------------------------------------


def _fmt(value: Any, places: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def summary_markdown(results: dict[str, Any]) -> str:
    ev = results.get("evaluation", {})
    lines = [f"# Stays Put gate ({results['mode']})", ""]
    if results.get("fixture"):
        fx = results["fixture"]
        lines.append(f"- fixture {fx['path']} ({fx['frames']} frames, {fx['media_size'][0]}x{fx['media_size'][1]}, ids {fx['ids']}, {fx['status']}); the panel box is synthetic")
    lines += [
        f"- demo clips passing (box >= 80%, inside >= 80%, drift < 0.08): any {_fmt(ev.get('demo_pass_any'))}, both clip types on one mark {_fmt(ev.get('demo_pass_both_types_any_label'))}; kill {_fmt(ev.get('kill'))}",
        f"- reflection control (drift > 0.16, sign > 80% on >= 8 pairs): {_fmt(ev.get('control_pass'))} ({ev.get('control_clips', 0)} clips)",
        f"- panned pass: {ev.get('panned_clips', 0)} clips, inconclusive share {_fmt(ev.get('panned_inconclusive_share'), 2)}",
        f"- tau_fix {_fmt(ev.get('tau_fix'), 3)}; b1 observed {ev.get('b1_observed')} (floor {ev.get('b1_floor')}); b2 observed {ev.get('b2_observed')} (floor {ev.get('b2_floor')})",
        "",
        "| clip | type | boxes | inside | drift | b1 | b2 | panned | sign | pairs | verdict | pass | $ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in results.get("clips", []):
        m = c.get("metrics", {})
        lines.append(
            f"| {c['label']}{' (control)' if c.get('is_control') else ''} | {c['clip_type']} | {_fmt(m.get('box_share'), 2)} | {_fmt(m.get('inside_panel_share'), 2)} | "
            f"{_fmt(m.get('drift_rms_units'))} | {_fmt(m.get('b1'))} | {_fmt(m.get('b2'))} | {_fmt(m.get('panned'))} | {_fmt(m.get('sign_agreement'), 2)} | "
            f"{m.get('sign_pairs', '-')} | {m.get('verdict')} | {_fmt(c.get('pass'))} | {_fmt(m.get('dollars'), 4)} |"
        )
    for entry in results.get("entries", []):
        for c in entry.get("clips", []):
            if "error" in c:
                lines.append(f"- {c['label']} {c['clip_type']}: {c['error']}")
            cm = c.get("carried_moments_check")
            if cm:
                lines.append(f"- carried moments on {c['label']} {c['clip_type']}: {cm.get('frames_checked')} frames, max residual {_fmt(cm.get('residual_max_units'), 4)} units (limit {cm.get('limit_units')}), within {_fmt(cm.get('within_limit'))}")
        st = entry.get("scuff_text")
        if st:
            lines.append(f"- scuff text prompt on {entry['label']}: {st.get('status') or st.get('error')}, ids {st.get('ids')}, overlap with the tracked mark {_fmt(st.get('overlap_share'), 2)} of {st.get('frames_shared_with_mark_track')} frames")
        if entry.get("skipped"):
            lines.append(f"- {entry.get('label')}: skipped ({entry['skipped']})")
    cm = results.get("carried_moments_check")
    if cm:
        lines.append(f"- carried moments (fixture): {cm.get('frames_checked')} frames, max residual {_fmt(cm.get('residual_max_units'), 4)} units, within {_fmt(cm.get('within_limit'))}")
    st = results.get("scuff_text")
    if st and "status" in st:
        lines.append(f"- scuff text prompt (fixture): {st['status']}, ids {st['ids']}, overlap share {_fmt(st.get('overlap_share'), 2)}")
    cc = results.get("cut_clip_check")
    if cc:
        lines.append(f"- cut_clip check: {cc.get('w')}x{cc.get('h')} x {cc.get('frames')} frames in {_fmt(cc.get('cut_s'), 1)} s, ok {_fmt(cc.get('ok'))}" if "w" in cc else f"- cut_clip check: {cc.get('note') or cc.get('error')}")
    lines += ["", f"Spend on this gate (ledger rows with the run prefix): ${results.get('spend_usd', 0.0):.4f}", ""]
    return "\n".join(lines)


def print_plan(args: argparse.Namespace, th: dict) -> None:
    entries = plan_entries(args)
    per_clip = max(th["media"]["clip_frames"], th["cost"]["sam_video_min_frames"]) * th["cost"]["sam_video_usd_per_frame"]
    print("gate_stays_put would send (set LIVE=1 to run):")
    for entry in entries:
        types = entry.get("clip_types") or list(CLIP_TYPES)
        n_calls = len(types) + (1 if entry.get("scuff_text") else 0)
        print(f"  {entry.get('label')}: {entry.get('src')} at {entry.get('pts')} s, mark {entry.get('mark')}, {n_calls} clip request(s) x ${per_clip:.4f} plus 3 image requests ($0.0075)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", help="the walk video (4K source)")
    parser.add_argument("--pts", type=float, help="keyframe time in seconds")
    parser.add_argument("--mark", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"), help="mark box in source pixels")
    parser.add_argument("--panel", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"), help="panel box in source pixels at the keyframe")
    parser.add_argument("--panel-lane", help="a SAM lane (ndjson or ndjson.gz) whose boxes move the window")
    parser.add_argument("--lane-fps", type=float, default=15.0, help="proxy fps of --panel-lane (default 15)")
    parser.add_argument("--lane-size", nargs=2, type=int, metavar=("W", "H"), help="proxy size of --panel-lane")
    parser.add_argument("--panel-noun", default="car door", help="noun of the keyframe mask image request (default 'car door')")
    parser.add_argument("--label", default="demo_mark", help="clip label (demo_mark, tape, reflection, ...)")
    parser.add_argument("--role", default="pickup")
    parser.add_argument("--control", action="store_true", help="this clip is the reflection control")
    parser.add_argument("--panned", action="store_true", help="this clip is from the deliberately panned pass")
    parser.add_argument("--backtrack", action="store_true", help="this clip is from the backtracking take")
    parser.add_argument("--scuff-text", action="store_true", help="also send 'scuff' as a text prompt once on the mark clip")
    parser.add_argument("--direction", type=int, choices=(1, -1), default=None, help="walk direction sign for the sign test when the panel barely moves")
    parser.add_argument("--sigma-frame", type=float, default=None, help="sigma_frame from E8, for tau_fix")
    parser.add_argument("--plan", help="JSON list of clip entries (see the module docstring)")
    parser.add_argument("--budget", type=float, default=None, help="stop starting entries past this many dollars on the run prefix")
    parser.add_argument("--run-prefix", default=None, help="ledger run_id prefix (default d1_<timestamp>)")
    parser.add_argument("--fixture", help="saved object-prompt events to analyse as the mark track (no call)")
    parser.add_argument("--text-fixture", help="saved text-prompt events standing in for the 'scuff' lane (fixture mode)")
    parser.add_argument("--cut-clip", action="store_true", help="fixture mode: also cut one 1008 px clip from --clip to check the ffmpeg path")
    parser.add_argument("--clip", help="the clip for --cut-clip (default data/test/pingpong.mp4)")
    parser.add_argument("--out", required=True, help="output directory")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    th = contracts.load_thresholds()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.fixture:
        ledger.LEDGER_PATH = out / "ledger.dryrun.jsonl"  # a fixture run never writes a spend row
        results = run_fixture(args, th)
    else:
        results = asyncio.run(run_live(args, th))
    results["thresholds_config_hash"] = contracts.config_hash(th)
    results["written_at"] = gate_sam.utc_ts()
    gate_sam.write_json(out / "gate_stays_put.json", results)
    (out / "gate_summary.md").write_text(summary_markdown(results), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    th = contracts.load_thresholds()
    if not args.fixture:
        if not args.plan and not (args.src and args.pts is not None and args.mark):
            print("gate_stays_put: --src, --pts and --mark (or --plan, or --fixture) are required", file=sys.stderr)
            return 2
        if not gate_sam.live_allowed():
            print_plan(args, th)
            return 2
    run(args)
    print((Path(args.out) / "gate_summary.md").read_text(encoding="utf-8"))
    print(f"wrote {Path(args.out) / 'gate_stays_put.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
