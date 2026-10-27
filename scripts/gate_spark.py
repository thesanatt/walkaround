"""D1 gate rows on Muse Spark 1.3: E6 (jitter of 5 repeated grounding calls per crop at low and medium, Jaccard of
the mark sets, false marks on a clean crop, the on-purpose truncation call, the second look), E7 (count mode union),
E11 (claim-first probes: Spark naming plus the SAM vote, the grayscale and drawn-box shapes, the plate and wheel
rulers, the door whitened across a walk frame and a close photo) and E18 (note_prefill_v1 on synthetic agreements).

    LIVE=1 python scripts/gate_spark.py IMAGE_OR_VIDEO --panel left_front_door [--crop X1 Y1 X2 Y2] [--t 12.0] \
        [--effort low] [--n 5] --out DIR
    LIVE=1 python scripts/gate_spark.py --crops crops.json --efforts low medium --n 5 --count --truncate 600 \
        --second-look A.jpg B.jpg --second-look-control A.jpg C.jpg --claim P1.jpg P2.jpg P3.jpg \
        [--ruler-frame WALK.mp4 --ruler-t 12.0 --tire-code 225/60R16] [--door-pair WALK_FRAME.jpg CLOSE.jpg] \
        [--note-prefill-synthetic] --out DIR
    python scripts/gate_spark.py --fixture tests/fixtures/d1/spark_fixture.json --out DIR

crops.json is a list of {name, source, t, crop [x1 y1 x2 y2], panel (a PanelId), clean (bool), targets [[u v] ...] in
crop pixels, mask (optional one-bit PNG in crop pixels)}. A video source takes one full-resolution frame through
walkaround.video.read_frame_at. Without LIVE=1 and without --fixture the script prints what it would send and exits
2. Every call goes through walkaround.spark_client (and sam_client for the votes and rulers), which write the ledger
rows; the run ids start with the D1 prefix and the E row. Every number printed also lands in <out>/gate_spark.json.
The fixture path drives the real spark_client code through a canned client (set_client), writes its ledger rows
to <out>/ledger.dryrun.jsonl and fakes the SAM image requests box-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (str(REPO), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import gate_sam  # noqa: E402
from walkaround import contracts, ledger, spark_client  # noqa: E402
from walkaround.config import SPARK_MODEL  # noqa: E402
from walkaround.contracts import CAPPED_MARK_KINDS, PANEL_NOUNS, LookVerdict, Noun, PanelId, SparkPurpose, SparkStatus  # noqa: E402
from walkaround.spark_client import MarksResult  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MATCH_RADIUS_FRACTION = 0.1  # of the crop's long edge
FIXTURE = REPO / "tests" / "fixtures" / "d1" / "spark_fixture.json"
NOTE_FIELDS = ("panel", "pickup_date", "return_date", "amount", "agreement_number")
SEGMENT_IMAGE: Callable[..., Any] | None = None  # the fixture path installs a box-only fake here


# ------------------------------------------------------------------------------------------------------------------
# Images and crops
# ------------------------------------------------------------------------------------------------------------------


def load_frame(path: Path, t_seconds: float | None) -> np.ndarray:
    """RGB uint8 array from an image file, or one frame of a video at t_seconds."""
    path = Path(path)
    if path.suffix.lower() in VIDEO_SUFFIXES or t_seconds is not None:
        return gate_sam.frame_at(path, t_seconds or 0.0)
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def crop_frame(frame: np.ndarray, box: list[int] | None) -> tuple[np.ndarray, list[int]]:
    height, width = frame.shape[:2]
    if box is None:
        return frame, [0, 0, width, height]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"crop {box} is empty inside a {width}x{height} frame")
    return np.ascontiguousarray(frame[y1:y2, x1:x2]), [x1, y1, x2, y2]


def panel_id_of(text: str) -> PanelId:
    """A PanelId from its value or a loose noun ('car door' -> left_front_door for the gate)."""
    try:
        return PanelId(text)
    except ValueError:
        pass
    slug = contracts.noun_slug(text)
    loose = {"car_door": PanelId.LEFT_FRONT_DOOR, "door": PanelId.LEFT_FRONT_DOOR, "bumper": PanelId.FRONT_BUMPER, "fender": PanelId.LEFT_FRONT_FENDER, "wheel": PanelId.LEFT_FRONT_WHEEL, "headlight": PanelId.HEADLIGHTS, "tail_light": PanelId.TAIL_LIGHTS, "side_mirror": PanelId.MIRRORS}
    if slug in loose:
        return loose[slug]
    return PanelId(slug)


def landmarks_for(panel_id: PanelId, th: dict) -> list[str]:
    marks_mod = gate_sam.optional_module("marks")
    if marks_mod is not None and gate_sam.implemented(marks_mod, "landmarks_for"):
        return list(marks_mod.landmarks_for(panel_id, th))
    table = th["spark"]["landmarks"]
    family = "door" if panel_id.value.endswith("_door") else "bumper" if panel_id.value.endswith("_bumper") else "wheel" if panel_id.value.endswith("_wheel") else "hood" if panel_id is PanelId.HOOD else None
    return list(table.get(family, []))[: th["spark"]["landmarks_max"]] if family else []


def synthetic_crop(name: str, specks: list[tuple[int, int]], *, size: tuple[int, int] = (1280, 900), seed: int = 7) -> np.ndarray:
    """A flat painted panel with dark specks at the given crop pixels (the dry run's stand-in for a real crop)."""
    rng = np.random.default_rng(seed + len(name))
    w, h = size
    base = np.full((h, w, 3), (168, 172, 178), dtype=np.uint8)
    noise = rng.integers(-6, 7, size=(h, w, 1), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for x, y in specks:
        cv2.circle(img, (int(x), int(y)), 9, (60, 58, 55), -1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------------------------------------------------------
# Mark rows, overlays, jitter and Jaccard (E6)
# ------------------------------------------------------------------------------------------------------------------


def mark_rows(result: MarksResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if result.answer is None:
        return rows
    for n, mark in enumerate(result.answer.marks, 1):
        px, py = spark_client.point_to_crop_pixels(mark.point, result)
        bx = spark_client.box_to_crop_pixels(mark.bbox, result)
        rows.append(
            {
                "n": n,
                "type": mark.type.value,
                "size_class": mark.size_class.value,
                "confidence": mark.confidence.value,
                "grid": [mark.point.x, mark.point.y],
                "px": [int(px), int(py)],
                "bbox_px": [int(v) for v in bx],
                "capped": mark.type in CAPPED_MARK_KINDS,
                "note": mark.note,
            }
        )
    return rows


def draw_overlay(crop_rgb: np.ndarray, runs: list[list[dict[str, Any]]], targets: list[list[int]] | None = None) -> np.ndarray:
    """BGR image: run 1 as numbered green circles, later runs thin orange, hand targets as blue crosses."""
    bgr = cv2.cvtColor(np.ascontiguousarray(crop_rgb), cv2.COLOR_RGB2BGR)
    long_edge = max(bgr.shape[:2])
    base = max(6, long_edge // 100)
    radius = {"S": base, "M": int(base * 1.6), "L": int(base * 2.4)}
    font_scale = max(0.5, long_edge / 1600)
    for run_index, rows in enumerate(runs):
        color = (40, 220, 60) if run_index == 0 else (0, 165, 255)
        for row in rows:
            x, y = row["px"]
            r = radius.get(row["size_class"], base)
            cv2.circle(bgr, (x, y), r, color, 2 if run_index == 0 else 1, cv2.LINE_AA)
            if run_index == 0:
                origin = (x + r + 2, y - 2)
                cv2.putText(bgr, str(row["n"]), origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(bgr, str(row["n"]), origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    for t in targets or []:
        cv2.drawMarker(bgr, (int(t[0]), int(t[1])), (255, 80, 0), cv2.MARKER_CROSS, base * 2, 2, cv2.LINE_AA)
    return bgr


def match_pairs(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], radius: float) -> list[tuple[int, int, float]]:
    """Greedy nearest pairs within radius, each mark used once: (i, j, distance)."""
    candidates = sorted((math.dist(a["px"], b["px"]), i, j) for i, a in enumerate(rows_a) for j, b in enumerate(rows_b) if math.dist(a["px"], b["px"]) <= radius)
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for d, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, d))
    return pairs


def stability(runs: list[list[dict[str, Any]]], crop_shape: tuple[int, ...]) -> dict[str, Any]:
    """Position stability across repeated calls on the same crop (the original gate number, kept)."""
    long_edge = max(crop_shape[:2])
    radius = MATCH_RADIUS_FRACTION * long_edge
    all_distances: list[float] = []
    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            distances = [d for _, _, d in match_pairs(runs[i], runs[j], radius)]
            pairs.append({"runs": [i + 1, j + 1], "matched": len(distances), "possible": max(len(runs[i]), len(runs[j])), "mean_px": round(sum(distances) / len(distances), 2) if distances else None})
            all_distances.extend(distances)
    mean_px = round(sum(all_distances) / len(all_distances), 2) if all_distances else None
    return {
        "counts": [len(rows) for rows in runs],
        "radius_px": round(radius, 1),
        "matched_pairs": len(all_distances),
        "possible_pairs": sum(p["possible"] for p in pairs),
        "mean_pairwise_px": mean_px,
        "mean_pairwise_fraction_of_long_edge": round(mean_px / long_edge, 4) if mean_px is not None else None,
        "pairs": pairs,
    }


def clusters_of(runs: list[list[dict[str, Any]]], radius: float) -> list[list[tuple[int, int]]]:
    """Greedy clusters across runs: each later mark joins the nearest cluster centroid within radius, once per run."""
    clusters: list[list[tuple[int, int]]] = []
    centroids: list[tuple[float, float]] = []
    for r, rows in enumerate(runs):
        taken: set[int] = set()
        order = sorted(((math.dist(row["px"], c), i, k) for i, row in enumerate(rows) for k, c in enumerate(centroids) if math.dist(row["px"], c) <= radius))
        matched: set[int] = set()
        for _, i, k in order:
            if i in matched or k in taken:
                continue
            matched.add(i)
            taken.add(k)
            clusters[k].append((r, i))
            pts = [runs[rr][ii]["px"] for rr, ii in clusters[k]]
            centroids[k] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        for i, row in enumerate(rows):
            if i not in matched:
                clusters.append([(r, i)])
                centroids.append((float(row["px"][0]), float(row["px"][1])))
    return clusters


def jitter(runs: list[list[dict[str, Any]]], crop_shape: tuple[int, ...], moments: contracts.Moments | None = None) -> dict[str, Any]:
    """sigma of the repeated positions in px (RMS radial deviation from the cluster centroid), as a fraction of the
    crop width and in whitened units (through the panel moments, a rectangle's when no mask exists); median Jaccard."""
    h, w = crop_shape[:2]
    radius = MATCH_RADIUS_FRACTION * max(h, w)
    moments = moments or gate_sam.rect_moments([0.0, 0.0, float(w), float(h)])
    groups = clusters_of(runs, radius)
    dev_px: list[float] = []
    dev_units: list[float] = []
    for group in groups:
        if len(group) < 2:
            continue
        pts = np.array([runs[r][i]["px"] for r, i in group], dtype=float)
        centre = pts.mean(axis=0)
        dev_px.extend(float(np.linalg.norm(p - centre)) for p in pts)
        ws = np.array([gate_sam.whiten((float(p[0]), float(p[1])), moments, 1) for p in pts], dtype=float)
        wc = ws.mean(axis=0)
        dev_units.extend(float(np.linalg.norm(x - wc)) for x in ws)
    sigma_px = math.sqrt(sum(d * d for d in dev_px) / len(dev_px)) if dev_px else None
    sigma_units = math.sqrt(sum(d * d for d in dev_units) / len(dev_units)) if dev_units else None
    jaccards = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            m = len(match_pairs(runs[i], runs[j], radius))
            union = len(runs[i]) + len(runs[j]) - m
            jaccards.append(round(m / union, 4) if union else 1.0)
    return {
        "counts": [len(r) for r in runs],
        "clusters": len(groups),
        "clusters_seen_in_every_run": sum(1 for g in groups if len(g) == len(runs)),
        "sigma_px": round(sigma_px, 2) if sigma_px is not None else None,
        "sigma_frac_of_width": round(sigma_px / w, 4) if sigma_px is not None else None,
        "sigma_units": round(sigma_units, 4) if sigma_units is not None else None,
        "units_basis": "mask moments" if moments is not None and moments.mu11 != 0 or (moments is not None and "mask" in str(getattr(moments, "note", ""))) else "rectangle moments of the crop",
        "jaccard_pairs": jaccards,
        "jaccard_median": round(statistics.median(jaccards), 4) if jaccards else None,
        "radius_px": round(radius, 1),
    }


def target_error(runs: list[list[dict[str, Any]]], targets: list[list[int]], radius: float) -> dict[str, Any]:
    """Error on hand-measured targets: the nearest mark per run, and how many runs hit within radius."""
    rows = []
    for t in targets:
        nearest = []
        for run in runs:
            d = min((math.dist(t, m["px"]) for m in run), default=None)
            nearest.append(round(d, 1) if d is not None else None)
        hits = sum(1 for d in nearest if d is not None and d <= radius)
        rows.append({"target": list(t), "nearest_px_by_run": nearest, "hit_runs": hits})
    hit = sum(1 for r in rows if r["hit_runs"] > 0)
    return {"targets": rows, "targets_hit_in_any_run": hit, "targets_total": len(rows), "recall": round(hit / len(rows), 4) if rows else None, "radius_px": round(radius, 1)}


def false_marks(runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Marks on a crop labelled clean: every type, and the uncapped types that would print as damage."""
    return {
        "per_run_all": [len(r) for r in runs],
        "per_run_uncapped": [sum(1 for m in r if not m["capped"]) for r in runs],
        "mean_uncapped": round(sum(sum(1 for m in r if not m["capped"]) for r in runs) / len(runs), 2) if runs else None,
    }


def usage_line(results: list[MarksResult]) -> dict[str, Any]:
    recs = [r.record for r in results]
    if not recs:
        return {}
    return {
        "calls": len(recs),
        "input_tokens_mean": round(sum(r.usage.input for r in recs) / len(recs), 1),
        "output_tokens_mean": round(sum(r.usage.output for r in recs) / len(recs), 1),
        "reasoning_tokens_mean": round(sum(r.usage.reasoning for r in recs) / len(recs), 1),
        "cached_tokens_mean": round(sum(r.usage.cached for r in recs) / len(recs), 1),
        "ms_mean": round(sum(r.ms for r in recs) / len(recs), 1),
        "ms_max": max(r.ms for r in recs),
        "dollars_mean": round(sum(r.dollars for r in recs) / len(recs), 6),
        "dollars_total": round(sum(r.dollars for r in recs), 6),
        "statuses": [r.status.value for r in recs],
        "incomplete_at_4000": sum(1 for r in recs if r.status is SparkStatus.INCOMPLETE and r.incomplete_reason == "max_output_tokens" and r.max_output_tokens == spark_client.MAX_OUTPUT_TOKENS),
        "truncated_after_retry": sum(1 for r in results if r.truncated),
        "invalid": sum(1 for r in recs if not r.valid),
        "retries": sum(len(r.attempts) - 1 for r in results),
    }


# ------------------------------------------------------------------------------------------------------------------
# E6: the crop sweep, the truncation probe and the second look
# ------------------------------------------------------------------------------------------------------------------


def run_crop(spec: dict[str, Any], crop_rgb: np.ndarray, *, efforts: list[str], n: int, run_id: str, th: dict, count: bool, out: Path) -> dict[str, Any]:
    panel_id = panel_id_of(str(spec.get("panel") or "left_front_door"))
    landmarks = landmarks_for(panel_id, th)
    moments = None
    if spec.get("mask"):
        mask = np.asarray(Image.open(spec["mask"]).convert("L")) > 127
        moments = gate_sam.moments_from_mask(mask)
    targets = [list(map(int, t)) for t in spec.get("targets") or []]
    radius = MATCH_RADIUS_FRACTION * max(crop_rgb.shape[:2])
    name = str(spec["name"])
    row: dict[str, Any] = {"name": name, "panel": panel_id.value, "clean": bool(spec.get("clean")), "crop_size": [int(crop_rgb.shape[1]), int(crop_rgb.shape[0])], "landmarks": landmarks, "efforts": {}}
    Image.fromarray(crop_rgb).save(out / f"crop_{name}.png")
    box_rows_low: list[dict[str, Any]] = []
    for effort in efforts:
        if FIXTURE_STATE is not None:
            FIXTURE_STATE["crop"] = name
        results = [spark_client.find_marks(crop_rgb, panel_id, landmarks=landmarks, effort=effort, run_id=f"{run_id}_E6_{name}_{effort}") for _ in range(max(1, n))]
        runs = [mark_rows(r) for r in results]
        if effort == efforts[0]:
            box_rows_low = runs[0]
        cv2.imwrite(str(out / f"overlay_{name}_{effort}.png"), draw_overlay(crop_rgb, runs, targets))
        row["efforts"][effort] = {
            "runs": [{"marks": rows, "status": r.record.status.value, "valid": r.record.valid, "truncated": r.truncated, "incomplete_reason": r.record.incomplete_reason, "request_id": r.record.request_id, "usage": r.record.usage.model_dump(), "ms": r.record.ms, "dollars": r.record.dollars, "conditions": r.answer.conditions.model_dump() if r.answer else None, "landmarks": [lm.model_dump() for lm in r.answer.landmarks] if r.answer else [], "neighbour_car_visible": r.answer.neighbour_car_visible if r.answer else None} for r, rows in zip(results, runs)],
            "stability": stability(runs, crop_rgb.shape) if len(runs) > 1 else None,
            "jitter": jitter(runs, crop_rgb.shape, moments) if len(runs) > 1 else None,
            "targets": target_error(runs, targets, radius) if targets else None,
            "false_marks": false_marks(runs) if spec.get("clean") else None,
            "usage": usage_line(results),
        }
    if count:
        row["count_mode"] = count_probe(crop_rgb, panel_id, box_rows_low, targets, moments=moments, run_id=f"{run_id}_E7_{name}", th=th)
    return row


def count_probe(crop_rgb: np.ndarray, panel_id: PanelId, box_rows: list[dict[str, Any]], targets: list[list[int]], *, moments: contracts.Moments | None, run_id: str, th: dict) -> dict[str, Any]:
    """E7: count_v1 points farther than 0.1 units from every box-mode mark, and whether one hits a missed target."""
    h, w = crop_rgb.shape[:2]
    moments = moments or gate_sam.rect_moments([0.0, 0.0, float(w), float(h)])
    result = spark_client.count_marks(crop_rgb, panel_id, effort=th["spark"]["effort"], run_id=run_id)
    sent_w, sent_h = result.record.parsed and (None, None) or (None, None)
    # count points stay on the grid; map them like marks through the same resize (prepare_image's scale)
    _, sent_w, sent_h, scale = spark_client.prepare_image(crop_rgb)
    points_px = []
    for p in result.points:
        sx, sy = spark_client.grid_to_pixels(p.x, p.y, sent_w, sent_h)
        points_px.append([int(min(max(round(sx / scale), 0), w - 1)), int(min(max(round(sy / scale), 0), h - 1))])
    box_w = [gate_sam.whiten((float(m["px"][0]), float(m["px"][1])), moments, 1) for m in box_rows]
    min_units = th["dedup"]["count_mode_new_point_min_units"]
    new_points = []
    for p in points_px:
        wp = gate_sam.whiten((float(p[0]), float(p[1])), moments, 1)
        d = min((math.dist(wp, b) for b in box_w), default=None)
        if d is None or d > min_units:
            new_points.append({"px": p, "nearest_box_mark_units": round(d, 4) if d is not None else None})
    radius = MATCH_RADIUS_FRACTION * max(h, w)
    missed = [t for t in targets if not any(math.dist(t, m["px"]) <= radius for m in box_rows)]
    recovered = [t for t in missed if any(math.dist(t, p["px"]) <= radius for p in new_points)]
    return {
        "status": result.record.status.value,
        "valid": result.record.valid,
        "count": result.count,
        "points": points_px,
        "new_points": new_points,
        "box_marks": len(box_rows),
        "targets_missed_by_box_pass": missed,
        "targets_recovered_by_count": recovered,
        "usage": result.record.usage.model_dump(),
        "dollars": result.record.dollars,
        "ms": result.record.ms,
        "request_id": result.record.request_id,
    }


def truncation_probe(crop_rgb: np.ndarray, panel_id: PanelId, *, max_output_tokens: int, run_id: str, th: dict) -> dict[str, Any]:
    """One marks_v1 call on purpose at a small output budget to see the truncation signal (no retry)."""
    landmarks = landmarks_for(panel_id, th)
    data_url, *_ = spark_client.prepare_image(crop_rgb)
    parts = [{"type": "input_text", "text": spark_client.marks_prompt(panel_id, landmarks)}, {"type": "input_image", "image_url": data_url}]
    record = spark_client.ask(parts, purpose=SparkPurpose.MARKS, schema_name="marks_v1", schema=spark_client.marks_schema(landmarks), effort=th["spark"]["effort"], max_output_tokens=max_output_tokens, run_id=run_id, input_refs=[])
    return {
        "max_output_tokens": max_output_tokens,
        "status": record.status.value,
        "incomplete_reason": record.incomplete_reason,
        "valid": record.valid,
        "raw_chars": len(record.raw_text),
        "usage": record.usage.model_dump(),
        "ms": record.ms,
        "dollars": record.dollars,
        "request_id": record.request_id,
        "signal_seen": record.status is SparkStatus.INCOMPLETE and record.incomplete_reason == "max_output_tokens",
    }


def second_look_probe(a: np.ndarray, b: np.ndarray, *, panel_id: PanelId, run_id: str, expect: str) -> dict[str, Any]:
    result = spark_client.second_look(a, b, panel_id=panel_id, run_id=run_id)
    return {
        "verdict": result.verdict.value,
        "reason": result.reason,
        "valid_json": bool(result.record.valid),
        "status": result.record.status.value,
        "expected": expect,
        "ok": (result.verdict in (LookVerdict.UNSURE, LookVerdict.DIFFERENT)) if expect == "different" else (result.verdict is LookVerdict.SAME),
        "usage": result.record.usage.model_dump(),
        "dollars": result.record.dollars,
        "ms": result.record.ms,
        "request_id": result.record.request_id,
    }


# ------------------------------------------------------------------------------------------------------------------
# E11: claim naming, the SAM vote, the three photo shapes, rulers, the door across views
# ------------------------------------------------------------------------------------------------------------------


async def segment_image(image: np.ndarray, concept: str, *, run_id: str, role: str, tag: str) -> Any:
    if SEGMENT_IMAGE is not None:
        return await SEGMENT_IMAGE(image, concept, run_id=run_id, role=role, tag=tag)
    sam_client = gate_sam.optional_module("sam_client")
    if sam_client is None or not gate_sam.implemented(sam_client, "segment_image"):
        raise RuntimeError("walkaround.sam_client.segment_image is not implemented yet")
    return await sam_client.segment_image(image, concept, run_id=run_id, role=role, tag=tag)


def detections(run: Any) -> list[Any]:
    return [d for dets in run.frames.values() for d in dets]


def det_box(det: Any) -> list[float]:
    return [float(det.box.left), float(det.box.top), float(det.box.right), float(det.box.bottom)]


def det_mask(det: Any, media_size: tuple[int, int]) -> np.ndarray | None:
    """The frame-sized mask, or None when the detection carries no mask (the box then stands in)."""
    if det.mask is None:
        return None
    sam_client = gate_sam.optional_module("sam_client")
    try:
        return sam_client.full_mask(det, media_size)
    except Exception:  # noqa: BLE001
        return None


def touches_border(box: list[float], media_size: tuple[int, int], margin: float = 1.0) -> bool:
    return box[0] <= margin or box[1] <= margin or box[2] >= media_size[0] - margin or box[3] >= media_size[1] - margin


async def sam_votes(image: np.ndarray, centre: tuple[float, float], *, run_id: str, th: dict) -> dict[str, Any]:
    """14 image requests scored mask-contains-centre x area x c (c absent counts as the median c)."""
    nouns = [n for n in PANEL_NOUNS if n is not Noun.CAR]
    media_size = (image.shape[1], image.shape[0])
    sem = asyncio.Semaphore(int(th["privacy"]["image_checks_in_flight"]))

    async def one(noun: Noun) -> tuple[Noun, Any]:
        async with sem:
            return noun, await segment_image(image, noun.value, run_id=run_id, role="claim", tag=f"D1 E11 vote {noun.value}")

    runs = await asyncio.gather(*(one(n) for n in nouns))
    cs = [d.confidence for _, run in runs for d in detections(run) if d.confidence is not None]
    median_c = statistics.median(cs) if cs else 0.5
    area_total = float(media_size[0] * media_size[1])
    scores = []
    for noun, run in runs:
        best = 0.0
        for d in detections(run):
            box = det_box(d)
            mask = det_mask(d, media_size)
            cx, cy = int(min(max(centre[0], 0), media_size[0] - 1)), int(min(max(centre[1], 0), media_size[1] - 1))
            holds = bool(mask[cy, cx]) if mask is not None else (box[0] <= centre[0] < box[2] and box[1] <= centre[1] < box[3])
            area = (float(mask.sum()) if mask is not None else (box[2] - box[0]) * (box[3] - box[1])) / area_total
            c = d.confidence if d.confidence is not None else median_c
            best = max(best, (1.0 if holds else 0.0) * area * c)
        scores.append({"noun": noun.value, "matches": len(detections(run)), "score": round(best, 6), "dollars": run.usd})
    scores.sort(key=lambda s: -s["score"])
    winner = scores[0]["noun"] if scores and scores[0]["score"] > 0 else None
    return {"winner": winner, "scores": scores, "dollars": round(sum(s["dollars"] for s in scores), 4)}


def grayscale(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.stack([gray, gray, gray], axis=-1)


def drawn_box(rgb: np.ndarray, frac: float = 0.4) -> tuple[np.ndarray, list[int]]:
    """An orange rectangle around the central region, like a renter's annotated claim photo (test only)."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = int(w * (0.5 - frac / 2)), int(h * (0.5 - frac / 2)), int(w * (0.5 + frac / 2)), int(h * (0.5 + frac / 2))
    out = np.ascontiguousarray(rgb).copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 120, 0), max(3, w // 200))
    return out, [x1, y1, x2, y2]


def scanner_style(rgb: np.ndarray, th: dict) -> tuple[bool, float]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = float(hsv[:, :, 1].mean() / 255.0)
    return sat < th["claim"]["scanner_style_saturation_max"], round(sat, 4)


def verify_drawn_box(rgb: np.ndarray, box: list[int], th: dict) -> dict[str, Any]:
    """The perimeter rule of 5.11: >= 30% of perimeter pixels with S > 0.5 and <= 10% inside."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1] / 255.0
    x1, y1, x2, y2 = box
    band = max(2, (x2 - x1) // 100)
    perim = np.concatenate([sat[y1 : y1 + band, x1:x2].ravel(), sat[max(y2 - band, 0) : y2, x1:x2].ravel(), sat[y1:y2, x1 : x1 + band].ravel(), sat[y1:y2, max(x2 - band, 0) : x2].ravel()])
    inside = sat[y1 + band : y2 - band, x1 + band : x2 - band].ravel()
    p_share = float((perim > th["claim"]["drawn_box_saturation_min"]).mean()) if perim.size else 0.0
    i_share = float((inside > th["claim"]["drawn_box_saturation_min"]).mean()) if inside.size else 0.0
    return {"perimeter_saturated_share": round(p_share, 4), "inside_saturated_share": round(i_share, 4), "verified": p_share >= th["claim"]["drawn_box_perimeter_saturated_min"] and i_share <= th["claim"]["drawn_box_inside_saturated_max"]}


async def claim_probe(path: Path, index: int, *, run_id: str, th: dict, out: Path) -> dict[str, Any]:
    """Spark naming plus the SAM vote on the photo, then the grayscale and drawn-box shapes (labelled test only)."""
    rgb = load_frame(path, None)
    h, w = rgb.shape[:2]
    shapes: dict[str, Any] = {}
    plain = spark_client.claim_name(rgb, run_id=f"{run_id}_E11_claim{index}")
    votes = await sam_votes(rgb, (w / 2.0, h / 2.0), run_id=f"{run_id}_E11_claim{index}", th=th)
    shapes["plain"] = {"spark_noun": plain.noun.value if plain.noun else None, "spark_panel": plain.panel.value if plain.panel else None, "side": plain.side.value, "position": plain.position.value, "front_direction": plain.front_direction, "context": plain.context, "valid": plain.record.valid, "dollars": plain.record.dollars, "ms": plain.record.ms, "sam_vote": votes, "agreed": bool(plain.noun and votes["winner"] == plain.noun.value)}
    gray = grayscale(rgb)
    g = spark_client.claim_name(gray, run_id=f"{run_id}_E11_claim{index}_gray")
    is_scan, sat = scanner_style(gray, th)
    shapes["grayscale"] = {"spark_noun": g.noun.value if g.noun else None, "side": g.side.value, "valid": g.record.valid, "scanner_style": is_scan, "mean_saturation": sat, "dollars": g.record.dollars}
    boxed, box = drawn_box(rgb)
    Image.fromarray(boxed).save(out / f"claim{index}_drawn_box.png")
    b = spark_client.claim_name(boxed, run_id=f"{run_id}_E11_claim{index}_box")
    cb = spark_client.claim_box(boxed, run_id=f"{run_id}_E11_claim{index}_box")
    found = None
    if cb.bbox is not None:
        _, sw, sh, scale = spark_client.prepare_image(boxed)
        x1, y1, x2, y2 = spark_client.grid_box_to_pixels(cb.bbox, sw, sh)
        found = [int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)]
    shapes["drawn_box"] = {"spark_noun": b.noun.value if b.noun else None, "valid": b.record.valid, "drawn": box, "spark_box": found, "box_iou": round(gate_sam.box_iou([float(v) for v in box], [float(v) for v in found]), 4) if found else None, "verified": verify_drawn_box(boxed, found, th) if found else None, "dollars": round(b.record.dollars + cb.record.dollars, 6)}
    return {"path": str(path), "size": [w, h], "shapes": shapes, "agreed": shapes["plain"]["agreed"]}


def tire_diameter_in(code: str) -> float | None:
    m = re.match(r"^\s*(\d{3})\s*/\s*(\d{2,3})\s*[A-Z]?\s*R?\s*(\d{2}(?:\.\d)?)\s*$", code.upper())
    if not m:
        return None
    sw, ar, rd = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return 2.0 * (sw * ar / 2540.0 + rd / 2.0)


def extent_px(det: Any, media_size: tuple[int, int]) -> tuple[float, float, float, str]:
    """(long side px, apparent h/w, tilt deg, basis) from the mask's min-area rectangle, or the box when no mask."""
    privacy = gate_sam.optional_module("privacy")
    mask = det_mask(det, media_size)
    if mask is not None and mask.any():
        if privacy is not None and gate_sam.implemented(privacy, "plate_extent"):
            ext = privacy.plate_extent(mask)
        else:
            ext = gate_sam._plate_extent(mask)  # noqa: SLF001 - the same rule, the gate's local copy
        if ext is not None:
            return float(ext[0]), float(ext[1]), float(ext[2]), "mask"
    box = det_box(det)
    w, h = box[2] - box[0], box[3] - box[1]
    return max(w, h), (min(w, h) / max(w, h) if max(w, h) else 0.0), 0.0, "box"


async def ruler_probe(frame: np.ndarray, *, tire_code: str | None, run_id: str, th: dict) -> dict[str, Any]:
    """Plate and wheel pixels per inch on one unblurred 4K frame, and the plate after pixelation (step 7 order)."""
    r = th["ruler"]
    media_size = (frame.shape[1], frame.shape[0])
    plate_run = await segment_image(frame, Noun.LICENSE_PLATE.value, run_id=f"{run_id}_E11_ruler", role="claim", tag="D1 E11 ruler plate")
    wheel_run = await segment_image(frame, Noun.WHEEL.value, run_id=f"{run_id}_E11_ruler", role="claim", tag="D1 E11 ruler wheel")
    out: dict[str, Any] = {"plate": None, "wheel": None, "dollars": round(plate_run.usd + wheel_run.usd, 4)}
    plates = [d for d in detections(plate_run) if (d.confidence or 0.0) >= r["plate_c_min"] and not touches_border(det_box(d), media_size)]
    best_plate = max(plates, key=lambda d: d.confidence or 0.0, default=None)
    ppi_plate = None
    if best_plate is not None:
        long_px, hw, tilt, basis = extent_px(best_plate, media_size)
        ppi_plate = long_px / r["plate_long_side_in"]
        out["plate"] = {"long_px": round(long_px, 1), "c": best_plate.confidence, "apparent_hw": round(hw, 4), "tilt_deg": round(tilt, 1), "basis": basis, "ppi": round(ppi_plate, 2), "qualifies": long_px >= r["plate_min_px"] and (best_plate.confidence or 0.0) >= r["plate_c_min"]}
    wheels = [d for d in detections(wheel_run) if (d.confidence or 0.0) >= r["wheel_c_min"] and not touches_border(det_box(d), media_size)]
    best_wheel = max(wheels, key=lambda d: (d.box.right - d.box.left) * (d.box.bottom - d.box.top), default=None)
    ppi_wheel = None
    diameter = tire_diameter_in(tire_code) if tire_code else None
    if best_wheel is not None:
        long_px, _, _, basis = extent_px(best_wheel, media_size)
        ppi_wheel = long_px / diameter if diameter else None
        out["wheel"] = {"major_axis_px": round(long_px, 1), "c": best_wheel.confidence, "basis": basis, "tire_code": tire_code, "diameter_in": round(diameter, 2) if diameter else None, "ppi": round(ppi_wheel, 2) if ppi_wheel else None, "qualifies": long_px >= r["wheel_min_px"]}
    if ppi_plate and ppi_wheel:
        ratio = abs(math.log(ppi_plate / ppi_wheel))
        out["ppi_agreement"] = {"abs_log_ratio": round(ratio, 4), "within_15pct": ratio <= math.log(1.0 + r["d1_plate_wheel_ppi_agreement"])}
    if best_plate is not None:
        privacy = gate_sam.optional_module("privacy")
        box = det_box(best_plate)
        scale = max(media_size) / 3840.0
        block = max(2, int(round(th["privacy"]["pixel_block_px_at_4k"] * scale)))
        if privacy is not None and gate_sam.implemented(privacy, "pixelate"):
            blurred = privacy.pixelate(frame, [box], block_px=block, dilate_frac=th["privacy"]["track_box_dilation_frac"], dilate_px=int(th["privacy"]["track_box_dilation_px"] * scale))
        else:
            blurred = frame.copy()
            x1, y1, x2, y2 = (int(v) for v in box)
            blurred[y1:y2, x1:x2] = cv2.resize(cv2.resize(frame[y1:y2, x1:x2], (max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)), interpolation=cv2.INTER_AREA), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        after_run = await segment_image(blurred, Noun.LICENSE_PLATE.value, run_id=f"{run_id}_E11_ruler", role="claim", tag="D1 E11 ruler plate pixelated")
        after = max(detections(after_run), key=lambda d: d.confidence or 0.0, default=None)
        if after is not None:
            long_after, _, _, basis_after = extent_px(after, media_size)
            out["plate_after_pixelation"] = {"long_px": round(long_after, 1), "c": after.confidence, "basis": basis_after, "relative_change": round(abs(long_after - out["plate"]["long_px"]) / out["plate"]["long_px"], 4), "over_20pct": abs(long_after - out["plate"]["long_px"]) / out["plate"]["long_px"] > 0.2}
        else:
            out["plate_after_pixelation"] = {"matches": 0, "note": "no plate found on the pixelated frame (the expected outcome for a blurred plate)"}
        out["dollars"] = round(out["dollars"] + after_run.usd, 4)
    return out


async def door_pair_probe(walk_frame: np.ndarray, close_photo: np.ndarray, *, run_id: str) -> dict[str, Any]:
    """The same door whitened from a walk frame and from the close photo: the handle's whitened residual (r_claim)."""
    rows = {}
    for name, img in (("walk", walk_frame), ("close", close_photo)):
        media_size = (img.shape[1], img.shape[0])
        door_run = await segment_image(img, Noun.CAR_DOOR.value, run_id=f"{run_id}_E11_door", role="claim", tag=f"D1 E11 door {name}")
        handle_run = await segment_image(img, Noun.DOOR_HANDLE.value, run_id=f"{run_id}_E11_door", role="claim", tag=f"D1 E11 handle {name}")
        handles = detections(handle_run)
        doors = detections(door_run)
        if not handles or not doors:
            rows[name] = {"note": "door or handle not found", "doors": len(doors), "handles": len(handles)}
            continue
        handle = max(handles, key=lambda d: d.confidence or 0.0)
        hc = ((handle.box.left + handle.box.right) / 2.0, (handle.box.top + handle.box.bottom) / 2.0)
        holding = [d for d in doors if d.box.left <= hc[0] < d.box.right and d.box.top <= hc[1] < d.box.bottom]
        door = max(holding or doors, key=lambda d: (d.box.right - d.box.left) * (d.box.bottom - d.box.top))
        mask = det_mask(door, media_size)
        moments = gate_sam.moments_from_mask(mask) if mask is not None and mask.sum() >= 3 else gate_sam.rect_moments(det_box(door))
        w = gate_sam.whiten(hc, moments, 1)
        rows[name] = {"handle_w": [round(v, 4) for v in w], "door_c": door.confidence, "handle_c": handle.confidence, "basis": "mask" if mask is not None else "box", "dollars": round(door_run.usd + handle_run.usd, 4)}
    if "handle_w" in rows.get("walk", {}) and "handle_w" in rows.get("close", {}):
        res = math.dist(rows["walk"]["handle_w"], rows["close"]["handle_w"])
        rows["residual_units"] = round(res, 4)
        rows["r_claim_suggested"] = round(max(0.1, res), 4)
    return rows


# ------------------------------------------------------------------------------------------------------------------
# E18: note_prefill_v1 on synthetic rental agreements
# ------------------------------------------------------------------------------------------------------------------


SYNTHETIC_AGREEMENTS = [
    {"panel": "left front door", "pickup_date": "2026-10-20", "return_date": "2026-10-27", "amount": "440.00", "agreement_number": "RA-5583921"},
    {"panel": "rear bumper", "pickup_date": "2026-11-01", "return_date": "2026-11-03", "amount": "185.50", "agreement_number": "RA-7710043"},
]


def render_agreement_pdf(fields: dict[str, str], dst: Path) -> Path:
    """A one-page synthetic rental agreement rendered by PIL (ours; no real company named)."""
    img = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(img)
    lines = [
        "RENTAL AGREEMENT (synthetic test document)",
        "",
        f"Agreement number: {fields['agreement_number']}",
        f"Pickup date: {fields['pickup_date']}",
        f"Return date: {fields['return_date']}",
        "",
        "Damage charge summary",
        f"Panel: {fields['panel']}",
        f"Amount charged: ${fields['amount']}",
        "",
        "This document was generated for a software test and names no real renter or company.",
    ]
    y = 120
    for line in lines:
        draw.text((100, y), line, fill="black")
        y += 44
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "PDF", resolution=150.0)
    return dst


def normalise_field(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower().replace("$", ""))


def note_prefill_probe(out: Path, *, run_id: str, th: dict) -> dict[str, Any]:
    rows = []
    for i, fields in enumerate(SYNTHETIC_AGREEMENTS, 1):
        pdf = render_agreement_pdf(fields, out / f"synthetic_agreement_{i}.pdf")
        claim_text = f"They charged me ${fields['amount']} for a scratch on the {fields['panel']}. Agreement {fields['agreement_number']}."
        record = spark_client.note_prefill(claim_text, pdf, run_id=f"{run_id}_E18_agreement{i}")
        parsed = record.parsed or {}
        agree = {k: normalise_field(parsed.get(k)) == normalise_field(fields[k]) for k in NOTE_FIELDS}
        rows.append({"agreement": i, "expected": fields, "parsed": {k: parsed.get(k) for k in NOTE_FIELDS}, "agree": agree, "agreed_fields": sum(agree.values()), "valid": record.valid, "status": record.status.value, "usage": record.usage.model_dump(), "dollars": record.dollars, "ms": record.ms, "request_id": record.request_id})
    return {"agreements": rows, "all_fields_on_all": all(r["agreed_fields"] == len(NOTE_FIELDS) for r in rows), "dollars": round(sum(r["dollars"] for r in rows), 6)}


# ------------------------------------------------------------------------------------------------------------------
# Evaluation (E6, E7, E11, E18 pass lines)
# ------------------------------------------------------------------------------------------------------------------


def evaluate(results: dict[str, Any], th: dict) -> dict[str, Any]:
    sp = th["spark"]
    crops = results.get("crops", [])
    efforts = results.get("efforts", [])
    per_effort: dict[str, Any] = {}
    for effort in efforts:
        sig = [c["efforts"][effort]["jitter"]["sigma_frac_of_width"] for c in crops if not c["clean"] and c["efforts"].get(effort, {}).get("jitter") and c["efforts"][effort]["jitter"]["sigma_frac_of_width"] is not None]
        units = [c["efforts"][effort]["jitter"]["sigma_units"] for c in crops if not c["clean"] and c["efforts"].get(effort, {}).get("jitter") and c["efforts"][effort]["jitter"]["sigma_units"] is not None]
        jac = [c["efforts"][effort]["jitter"]["jaccard_median"] for c in crops if not c["clean"] and c["efforts"].get(effort, {}).get("jitter") and c["efforts"][effort]["jitter"]["jaccard_median"] is not None]
        usage = [c["efforts"][effort]["usage"] for c in crops if c["efforts"].get(effort)]
        per_effort[effort] = {
            "sigma_frac_max": max(sig) if sig else None,
            "sigma_units_max": max(units) if units else None,
            "sigma_pass": (max(sig) <= sp["d1_sigma_pass_frac_of_crop_width"]) if sig else None,
            "jaccard_median_min": min(jac) if jac else None,
            "jaccard_pass": (min(jac) >= sp["d1_jaccard_pass"]) if jac else None,
            "incomplete_at_4000": sum(u.get("incomplete_at_4000", 0) for u in usage),
            "truncated_after_retry": sum(u.get("truncated_after_retry", 0) for u in usage),
            "invalid": sum(u.get("invalid", 0) for u in usage),
            "input_tokens_mean": round(statistics.mean(u["input_tokens_mean"] for u in usage if u), 1) if usage else None,
            "output_tokens_mean": round(statistics.mean(u["output_tokens_mean"] for u in usage if u), 1) if usage else None,
            "reasoning_tokens_mean": round(statistics.mean(u["reasoning_tokens_mean"] for u in usage if u), 1) if usage else None,
            "ms_mean": round(statistics.mean(u["ms_mean"] for u in usage if u), 1) if usage else None,
            "dollars_mean": round(statistics.mean(u["dollars_mean"] for u in usage if u), 6) if usage else None,
        }
    low = per_effort.get("low", {})
    medium = per_effort.get("medium", {})
    effort_default = None
    tau_pre_cap_hit = None
    if low.get("sigma_frac_max") is not None:
        s_low = low["sigma_frac_max"]
        if s_low <= sp["d1_sigma_pass_frac_of_crop_width"]:
            effort_default, tau_pre_cap_hit = "low", False
        elif s_low <= 0.03 and medium.get("sigma_frac_max") is not None and medium["sigma_frac_max"] <= sp["d1_sigma_pass_frac_of_crop_width"]:
            effort_default, tau_pre_cap_hit = "medium", False
        elif s_low > 0.03 and (medium.get("sigma_frac_max") is None or medium["sigma_frac_max"] > 0.03):
            effort_default, tau_pre_cap_hit = "low", True
        else:
            effort_default, tau_pre_cap_hit = "low", False
    clean = [c for c in crops if c["clean"]]
    false_uncapped = [c["efforts"][e]["false_marks"]["mean_uncapped"] for c in clean for e in c["efforts"] if c["efforts"][e].get("false_marks")]
    looks = results.get("second_look", {})
    count_rows = [c.get("count_mode") for c in crops if c.get("count_mode")]
    claims = results.get("claims", [])
    ruler = results.get("ruler") or {}
    note = results.get("note_prefill") or {}
    trunc = results.get("truncation") or {}
    sigma_units_low = low.get("sigma_units_max")
    return {
        "per_effort": per_effort,
        "effort_default": effort_default,
        "tau_pre_over_cap": tau_pre_cap_hit,
        "sigma_spark_units": sigma_units_low,
        "tau_pre0_from_sigma_spark": round(max(th["matching"]["tau_pre0_floor"], th["matching"]["tau_pre0_sigma_factor"] * sigma_units_low), 4) if sigma_units_low is not None else None,
        "false_marks_uncapped_mean_on_clean": max(false_uncapped) if false_uncapped else None,
        "no_truncation_at_4000": all(e.get("incomplete_at_4000", 0) == 0 and e.get("truncated_after_retry", 0) == 0 for e in per_effort.values()) if per_effort else None,
        "truncation_signal_seen": trunc.get("signal_seen"),
        "truncation_reason": trunc.get("incomplete_reason"),
        "second_look_valid": all(v["valid_json"] for v in looks.values()) if looks else None,
        "second_look_control_ok": looks.get("control", {}).get("ok") if "control" in looks else None,
        "second_look_pair_verdict": looks.get("pair", {}).get("verdict"),
        "count_mode_recovered_any": any(r.get("targets_recovered_by_count") for r in count_rows) if count_rows else None,
        "count_mode_new_points": sum(len(r.get("new_points", [])) for r in count_rows) if count_rows else None,
        "claims_agreed": sum(1 for c in claims if c.get("agreed")) if claims else None,
        "claims_total": len(claims),
        "plate_qualifies": (ruler.get("plate") or {}).get("qualifies"),
        "wheel_qualifies": (ruler.get("wheel") or {}).get("qualifies"),
        "ppi_within_15pct": (ruler.get("ppi_agreement") or {}).get("within_15pct"),
        "plate_pixelated_over_20pct": (ruler.get("plate_after_pixelation") or {}).get("over_20pct"),
        "door_pair_residual_units": (results.get("door_pair") or {}).get("residual_units"),
        "note_prefill_all_fields": note.get("all_fields_on_all") if note else None,
        "note_prefill_fields": [r["agreed_fields"] for r in note.get("agreements", [])] if note else None,
    }


# ------------------------------------------------------------------------------------------------------------------
# Fixture client (dry run): the real spark_client code over canned answers
# ------------------------------------------------------------------------------------------------------------------

FIXTURE_STATE: dict[str, Any] | None = None


class _FixtureResponse:
    def __init__(self, data: dict[str, Any], text: str) -> None:
        self._data = data
        self.output_text = text

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)


class _Responses:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def create(self, **request: Any) -> Any:
        state = self.state
        state["calls"] = state.get("calls", 0) + 1
        fx = state["fixture"]
        name = request["text"]["format"]["name"]
        budget = int(request.get("max_output_tokens") or 0)
        effort = (request.get("reasoning") or {}).get("effort", "low")
        usage = {"input_tokens": fx["usage"]["input_tokens"], "input_tokens_details": {"cached_tokens": 0}, "output_tokens": fx["usage"]["output_tokens"], "output_tokens_details": {"reasoning_tokens": fx["usage"]["reasoning_tokens"]}}
        rid = f"fixture-{state['calls']:04d}"
        if budget and budget < 1000:
            data = {"id": rid, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "usage": {**usage, "output_tokens": budget}, "output": []}
            return _FixtureResponse(data, "")
        answer = self._answer(name, effort)
        text = json.dumps(answer)
        data = {"id": rid, "status": "completed", "incomplete_details": None, "usage": usage, "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        if request.get("stream"):
            return iter([{"type": "response.output_text.delta", "delta": text}, {"type": "response.completed", "response": data}])
        return _FixtureResponse(data, text)

    def _answer(self, name: str, effort: str) -> dict[str, Any]:
        fx = self.state["fixture"]
        rng: random.Random = self.state["rng"]
        crop = self.state.get("crop", "mark_door")
        if name == "marks_v1":
            base = json.loads(json.dumps(fx["marks_v1"].get(crop) or fx["marks_v1"]["mark_door"]))
            amp = fx.get("jitter_grid", 8) * (0.5 if effort == "medium" else 1.0)
            for m in base["marks"]:
                dx, dy = rng.gauss(0, amp), rng.gauss(0, amp)
                m["point"] = {"x": int(min(max(m["point"]["x"] + dx, 0), 1000)), "y": int(min(max(m["point"]["y"] + dy, 0), 1000))}
                m["bbox"] = {k: int(min(max(v + (dx if "x" in k else dy), 0), 1000)) for k, v in m["bbox"].items()}
            calls = self.state["calls"]
            if base["marks"] and calls % 4 == 0:
                base["marks"] = base["marks"][:-1]  # every fourth call misses one mark, so Jaccard is below 1
            return base
        if name == "count_v1":
            return fx["count_v1"].get(crop) or {"points": [], "count": 0}
        if name == "look_v1":
            key = self.state.get("look", "pair")
            return fx["look_v1"][key]
        if name in fx:
            return fx[name]
        raise KeyError(f"no fixture answer for schema {name}")


class FixtureClient:
    def __init__(self, fixture: dict[str, Any], seed: int = 11) -> None:
        self.state: dict[str, Any] = {"fixture": fixture, "rng": random.Random(seed), "calls": 0}
        self.responses = _Responses(self.state)


def install_fixture(fixture: dict[str, Any], out: Path) -> dict[str, Any]:
    """Point spark_client at the canned client, the ledger at the dry-run file and SAM at a box-only fake."""
    global FIXTURE_STATE, SEGMENT_IMAGE
    client = FixtureClient(fixture)
    spark_client.set_client(client)
    FIXTURE_STATE = client.state
    ledger.LEDGER_PATH = out / "ledger.dryrun.jsonl"
    sam_client = gate_sam.optional_module("sam_client")
    boxes = fixture.get("sam_boxes", {})

    async def fake_segment_image(image: Any, concept: str, *, run_id: str, role: str | None = None, tag: str = "") -> Any:
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        dets = []
        for i, (box, c) in enumerate(boxes.get(concept, [])):
            x1, y1, x2, y2 = (box[0] * w, box[1] * h, box[2] * w, box[3] * h)
            dets.append(sam_client.Detection(object_id=str(i), frame_index=None, box=sam_client.Box(x1, y1, x2, y2), mask=None, confidence=c))
        return sam_client.SegmentationRun(concept=concept, media="image", media_width=w, media_height=h, frames={0: dets}, usd=0.0, response_id=f"fixture-sam-{concept}")

    SEGMENT_IMAGE = fake_segment_image
    return client.state


# ------------------------------------------------------------------------------------------------------------------
# Runs
# ------------------------------------------------------------------------------------------------------------------


def crop_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.crops:
        specs = json.loads(Path(args.crops).read_text(encoding="utf-8"))
        if not isinstance(specs, list):
            raise SystemExit("--crops must hold a JSON list")
        return specs
    if args.source:
        return [{"name": Path(args.source).stem, "source": args.source, "t": args.t, "crop": args.crop, "panel": args.panel or "left_front_door", "clean": args.clean, "targets": []}]
    return []


def run_gate(args: argparse.Namespace, th: dict, *, specs: list[dict[str, Any]], crops_rgb: dict[str, np.ndarray], claims: list[Path], ruler_frame: np.ndarray | None, door_pair: tuple[np.ndarray, np.ndarray] | None, out: Path, prefix: str, mode: str) -> dict[str, Any]:
    efforts = list(args.efforts or [args.effort])
    results: dict[str, Any] = {"mode": mode, "run_prefix": prefix, "model": SPARK_MODEL, "efforts": efforts, "n": args.n, "crops": [], "spend_usd": 0.0}
    for spec in specs:
        spend = gate_sam.spend_for_prefix(prefix)
        if args.budget is not None and spend >= args.budget:
            results["crops"].append({"name": spec["name"], "skipped": f"budget {args.budget} reached (spent {spend})", "clean": bool(spec.get("clean")), "efforts": {}})
            continue
        results["crops"].append(run_crop(spec, crops_rgb[spec["name"]], efforts=efforts, n=args.n, run_id=prefix, th=th, count=bool(args.count), out=out))
    first = next((s for s in specs if not s.get("clean")), specs[0] if specs else None)
    if args.truncate and first is not None:
        if FIXTURE_STATE is not None:
            FIXTURE_STATE["crop"] = first["name"]
        results["truncation"] = truncation_probe(crops_rgb[first["name"]], panel_id_of(str(first.get("panel") or "left_front_door")), max_output_tokens=int(args.truncate), run_id=f"{prefix}_E6_truncate", th=th)
    looks: dict[str, Any] = {}
    panel_id = panel_id_of(str(first.get("panel") or "left_front_door")) if first else PanelId.LEFT_FRONT_DOOR
    for key, pair, expect in (("pair", args.second_look, "same"), ("control", args.second_look_control, "different")):
        if not pair:
            continue
        if FIXTURE_STATE is not None:
            FIXTURE_STATE["look"] = key
        a, b = (load_frame(Path(p), None) if not isinstance(p, np.ndarray) else p for p in pair)
        looks[key] = second_look_probe(a, b, panel_id=panel_id, run_id=f"{prefix}_E6_look_{key}", expect=expect)
    if looks:
        results["second_look"] = looks
    if claims:
        results["claims"] = [asyncio.run(claim_probe(p, i, run_id=prefix, th=th, out=out)) for i, p in enumerate(claims, 1)]
    if ruler_frame is not None:
        results["ruler"] = asyncio.run(ruler_probe(ruler_frame, tire_code=args.tire_code, run_id=prefix, th=th))
    if door_pair is not None:
        results["door_pair"] = asyncio.run(door_pair_probe(door_pair[0], door_pair[1], run_id=prefix))
    if args.note_prefill_synthetic:
        results["note_prefill"] = note_prefill_probe(out, run_id=prefix, th=th)
    results["evaluation"] = evaluate(results, th)
    results["spend_usd"] = gate_sam.spend_for_prefix(prefix)
    return results


def run_live(args: argparse.Namespace, th: dict) -> dict[str, Any]:
    out = Path(args.out)
    prefix = args.run_prefix or f"d1_{gate_sam.utc_ts()}"
    specs = crop_specs(args)
    crops_rgb: dict[str, np.ndarray] = {}
    for spec in specs:
        frame = load_frame(Path(spec["source"]), spec.get("t"))
        crop, box = crop_frame(frame, spec.get("crop"))
        spec["crop"] = box
        crops_rgb[spec["name"]] = crop
    claims = [Path(p) for p in (args.claim or [])]
    ruler_frame = load_frame(Path(args.ruler_frame), args.ruler_t) if args.ruler_frame else None
    door_pair = (load_frame(Path(args.door_pair[0]), None), load_frame(Path(args.door_pair[1]), None)) if args.door_pair else None
    return run_gate(args, th, specs=specs, crops_rgb=crops_rgb, claims=claims, ruler_frame=ruler_frame, door_pair=door_pair, out=out, prefix=prefix, mode="live")


def run_fixture(args: argparse.Namespace, th: dict) -> dict[str, Any]:
    out = Path(args.out)
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    install_fixture(fixture, out)
    specs = []
    crops_rgb: dict[str, np.ndarray] = {}
    for spec in fixture["crops"]:
        crop = synthetic_crop(spec["name"], [tuple(t) for t in spec.get("targets", [])])
        specs.append({**spec, "source": "synthetic", "crop": [0, 0, crop.shape[1], crop.shape[0]]})
        crops_rgb[spec["name"]] = crop
    args.efforts = args.efforts or ["low", "medium"]
    args.n = args.n if args.n and args.n > 1 else 5
    args.count = True
    args.truncate = args.truncate or 600
    first = crops_rgb[specs[0]["name"]]
    args.second_look = [first, first]
    args.second_look_control = [first, synthetic_crop("reflection", [(640, 450)], seed=99)]
    args.note_prefill_synthetic = True
    claim_img = out / "synthetic_claim.png"
    Image.fromarray(synthetic_crop("claim", [(500, 400), (520, 420)], size=(1600, 1200))).save(claim_img)
    ruler = synthetic_crop("ruler", [], size=(1920, 1080))
    door_pair = (synthetic_crop("walk", [], size=(1920, 1080)), synthetic_crop("close", [], size=(1600, 1200)))
    args.tire_code = args.tire_code or "225/60R16"
    results = run_gate(args, th, specs=specs, crops_rgb=crops_rgb, claims=[claim_img], ruler_frame=ruler, door_pair=door_pair, out=out, prefix="d1_fixture", mode="fixture")
    results["fixture"] = {"path": str(args.fixture), "sha256": gate_sam.sha256_file(Path(args.fixture)), "note": fixture.get("note")}
    results["dryrun_ledger"] = str(ledger.LEDGER_PATH)
    return results


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
    lines = [f"# Spark gate ({results['mode']})", "", f"- model {results['model']}, efforts {results['efforts']}, {results['n']} calls per crop and effort"]
    for effort, e in ev.get("per_effort", {}).items():
        lines.append(f"- {effort}: sigma max {_fmt(e['sigma_frac_max'], 4)} of crop width ({_fmt(e['sigma_units_max'], 4)} units) pass {_fmt(e['sigma_pass'])}; Jaccard median min {_fmt(e['jaccard_median_min'], 3)} pass {_fmt(e['jaccard_pass'])}; incomplete at 4000: {e['incomplete_at_4000']}; invalid {e['invalid']}; tokens in {_fmt(e['input_tokens_mean'], 0)} out {_fmt(e['output_tokens_mean'], 0)} reasoning {_fmt(e['reasoning_tokens_mean'], 0)}; {_fmt(e['ms_mean'], 0)} ms; ${_fmt(e['dollars_mean'], 5)} per call")
    lines += [
        f"- effort default: {ev.get('effort_default')}; tau_pre over the 0.4 cap: {_fmt(ev.get('tau_pre_over_cap'))}; tau_pre0 from sigma_spark: {_fmt(ev.get('tau_pre0_from_sigma_spark'), 3)}",
        f"- false uncapped marks on the clean crop (mean per call): {_fmt(ev.get('false_marks_uncapped_mean_on_clean'), 2)}",
        f"- truncation probe: signal seen {_fmt(ev.get('truncation_signal_seen'))} (reason {ev.get('truncation_reason')})",
        f"- second look: valid JSON {_fmt(ev.get('second_look_valid'))}; control says unsure or different {_fmt(ev.get('second_look_control_ok'))}; pair verdict {ev.get('second_look_pair_verdict')}",
        f"- count mode: new points {ev.get('count_mode_new_points')}, recovered a target the box pass missed {_fmt(ev.get('count_mode_recovered_any'))}",
        f"- claims: naming agreed {ev.get('claims_agreed')} of {ev.get('claims_total')}; plate qualifies {_fmt(ev.get('plate_qualifies'))}; wheel qualifies {_fmt(ev.get('wheel_qualifies'))}; ppi within 15% {_fmt(ev.get('ppi_within_15pct'))}; plate after pixelation over 20% off {_fmt(ev.get('plate_pixelated_over_20pct'))}; door pair residual {_fmt(ev.get('door_pair_residual_units'), 4)} units",
        f"- note prefill: all 5 fields on both agreements {_fmt(ev.get('note_prefill_all_fields'))} (fields agreed {ev.get('note_prefill_fields')})",
        "",
        "| crop | effort | counts | sigma px | sigma frac | sigma units | Jaccard | targets hit | false uncapped |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for crop in results.get("crops", []):
        for effort, e in crop.get("efforts", {}).items():
            j = e.get("jitter") or {}
            t = e.get("targets") or {}
            f = e.get("false_marks") or {}
            lines.append(f"| {crop['name']} | {effort} | {j.get('counts')} | {_fmt(j.get('sigma_px'), 1)} | {_fmt(j.get('sigma_frac_of_width'), 4)} | {_fmt(j.get('sigma_units'), 4)} | {_fmt(j.get('jaccard_median'), 3)} | {t.get('targets_hit_in_any_run', '-')}/{t.get('targets_total', '-')} | {_fmt(f.get('mean_uncapped'), 2)} |")
        if crop.get("skipped"):
            lines.append(f"| {crop['name']} | skipped: {crop['skipped']} | | | | | | | |")
    lines += ["", f"Spend on this gate (ledger rows with the run prefix): ${results.get('spend_usd', 0.0):.4f}", ""]
    return "\n".join(lines)


def print_plan(args: argparse.Namespace, th: dict) -> None:
    specs = crop_specs(args)
    efforts = list(args.efforts or [args.effort])
    calls = len(specs) * len(efforts) * max(1, args.n)
    extra = (1 if args.truncate else 0) + (1 if args.second_look else 0) + (1 if args.second_look_control else 0) + len(args.claim or []) * 3 + (2 if args.note_prefill_synthetic else 0) + (len(specs) if args.count else 0)
    print("gate_spark would send (set LIVE=1 to run):")
    print(f"  {len(specs)} crop(s) x {efforts} x {args.n} = {calls} marks_v1 calls at about ${th['cost'].get('spark_call_estimate_usd', 0.01):.3f} each (estimate), plus {extra} other Spark calls")
    if args.claim:
        print(f"  {len(args.claim)} claim photo(s) x 14 SAM image requests x $0.0025")
    if args.ruler_frame:
        print("  3 SAM image requests for the rulers")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=str, help="image file, or a video (then --t picks the frame)")
    parser.add_argument("--panel", help="PanelId of the crop, e.g. left_front_door (a noun like 'car door' is accepted)")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="crop box in frame pixels")
    parser.add_argument("--t", type=float, default=None, help="seconds into the video")
    parser.add_argument("--clean", action="store_true", help="the single crop is a clean panel (false-mark count)")
    parser.add_argument("--effort", default="low", choices=spark_client.EFFORTS)
    parser.add_argument("--efforts", nargs="+", choices=spark_client.EFFORTS, help="run every crop at each effort (E6: low medium)")
    parser.add_argument("--n", type=int, default=1, help="repeat each call n times and report stability (E6: 5)")
    parser.add_argument("--crops", help="JSON list of crop specs (see the module docstring)")
    parser.add_argument("--count", action="store_true", help="E7: count_v1 on every crop and the union against the box pass")
    parser.add_argument("--truncate", type=int, default=None, help="E6: one marks_v1 call on purpose at this max_output_tokens (600)")
    parser.add_argument("--second-look", nargs=2, metavar=("A", "B"), help="E6: look_v1 on two crops of the same mark")
    parser.add_argument("--second-look-control", nargs=2, metavar=("A", "C"), help="E6: look_v1 on a mark and a reflection (must not say same)")
    parser.add_argument("--claim", nargs="+", help="E11: staged claim photos (naming plus the SAM vote, grayscale and drawn-box shapes)")
    parser.add_argument("--ruler-frame", help="E11: a 4K walk frame (video with --ruler-t, or an image) for the plate and wheel rulers")
    parser.add_argument("--ruler-t", type=float, default=None)
    parser.add_argument("--tire-code", default=None, help="E11: the sidewall code for the wheel ruler, e.g. 225/60R16")
    parser.add_argument("--door-pair", nargs=2, metavar=("WALK_FRAME", "CLOSE_PHOTO"), help="E11: the same door in a walk frame and a close photo")
    parser.add_argument("--note-prefill-synthetic", action="store_true", help="E18: two synthetic agreements through note_prefill_v1")
    parser.add_argument("--budget", type=float, default=None, help="stop starting crops past this many dollars on the run prefix")
    parser.add_argument("--run-prefix", default=None, help="ledger run_id prefix (default d1_<timestamp>)")
    parser.add_argument("--fixture", help="canned answers JSON for a dry run through the real spark_client code (no call)")
    parser.add_argument("--out", required=True, help="output directory")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    th = contracts.load_thresholds()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = run_fixture(args, th) if args.fixture else run_live(args, th)
    results["thresholds_config_hash"] = contracts.config_hash(th)
    results["written_at"] = gate_sam.utc_ts()
    gate_sam.write_json(out / "gate_spark.json", results)
    (out / "gate_summary.md").write_text(summary_markdown(results), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    th = contracts.load_thresholds()
    if not args.fixture:
        if not args.source and not args.crops and not args.claim and not args.note_prefill_synthetic and not args.ruler_frame:
            print("gate_spark: a SOURCE, --crops, --claim, --ruler-frame or --note-prefill-synthetic is required (or --fixture)", file=sys.stderr)
            return 2
        if not gate_sam.live_allowed():
            print_plan(args, th)
            return 2
    run(args)
    print((Path(args.out) / "gate_summary.md").read_text(encoding="utf-8"))
    print(f"wrote {Path(args.out) / 'gate_spark.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
