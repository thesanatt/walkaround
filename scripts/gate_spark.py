"""Gate run: Muse Spark mark finding on one panel crop.

python scripts/gate_spark.py IMAGE_OR_VIDEO --panel "car door" [--crop X1 Y1 X2 Y2] [--t 12.0] [--effort low]
    [--n 3] --out runs/<name>

A video source (or --t) takes one full-resolution frame through walkaround.video.read_frame_at. The crop box is in
pixels of that frame, half open. Writes crop.png, overlay.png (numbered circles on the original crop), marks.json and
gate_summary.md into --out. With --n > 1 the same crop goes to the model n times and the summary reports how stable
the positions are: the mean pairwise distance of matched marks, in crop pixels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from walkaround import spark_client  # noqa: E402
from walkaround.config import SPARK_MODEL  # noqa: E402
from walkaround.spark_client import MarksResult, mark_to_crop_pixels  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MATCH_RADIUS_FRACTION = 0.1  # of the crop's long edge


def load_frame(path: Path, t_seconds: float | None) -> np.ndarray:
    """RGB uint8 array from an image file, or one frame of a video at t_seconds."""
    if path.suffix.lower() in VIDEO_SUFFIXES or t_seconds is not None:
        from walkaround.video import read_frame_at

        return read_frame_at(path, t_seconds or 0.0)
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def crop_frame(frame: np.ndarray, box: list[int] | None) -> tuple[np.ndarray, list[int]]:
    height, width = frame.shape[:2]
    if box is None:
        return frame, [0, 0, width, height]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"crop {box} is empty inside a {width}x{height} frame")
    return np.ascontiguousarray(frame[y1:y2, x1:x2]), [x1, y1, x2, y2]


def mark_rows(result: MarksResult) -> list[dict[str, Any]]:
    rows = []
    for n, mark in enumerate(result.marks, 1):
        px, py = mark_to_crop_pixels(mark, result)
        rows.append(
            {
                "n": n,
                "kind": mark.kind,
                "size": mark.size,
                "confidence": mark.confidence,
                "grid": [mark.x, mark.y],
                "px": [px, py],
                "description": mark.description,
            }
        )
    return rows


def draw_overlay(crop_rgb: np.ndarray, runs: list[list[dict[str, Any]]]) -> np.ndarray:
    """BGR image: run 1 as numbered green circles, later runs as thin orange circles."""
    bgr = cv2.cvtColor(np.ascontiguousarray(crop_rgb), cv2.COLOR_RGB2BGR)
    long_edge = max(bgr.shape[:2])
    base = max(6, long_edge // 100)
    radius = {"small": base, "medium": int(base * 1.6), "large": int(base * 2.4)}
    font_scale = max(0.5, long_edge / 1600)
    for run_index, rows in enumerate(runs):
        color = (40, 220, 60) if run_index == 0 else (0, 165, 255)
        for row in rows:
            x, y = row["px"]
            r = radius[row["size"]]
            cv2.circle(bgr, (x, y), r, color, 2 if run_index == 0 else 1, cv2.LINE_AA)
            if run_index == 0:
                origin = (x + r + 2, y - 2)
                cv2.putText(bgr, str(row["n"]), origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(bgr, str(row["n"]), origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def match_pairs(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], radius: float) -> list[float]:
    """Greedy nearest pairs within radius, each mark used once. Returns the pair distances."""
    candidates = []
    for i, a in enumerate(rows_a):
        for j, b in enumerate(rows_b):
            d = math.dist(a["px"], b["px"])
            if d <= radius:
                candidates.append((d, i, j))
    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    distances = []
    for d, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        distances.append(d)
    return distances


def stability(runs: list[list[dict[str, Any]]], crop_shape: tuple[int, ...]) -> dict[str, Any]:
    """Position stability across repeated calls on the same crop."""
    long_edge = max(crop_shape[:2])
    radius = MATCH_RADIUS_FRACTION * long_edge
    all_distances: list[float] = []
    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            distances = match_pairs(runs[i], runs[j], radius)
            pairs.append(
                {
                    "runs": [i + 1, j + 1],
                    "matched": len(distances),
                    "possible": max(len(runs[i]), len(runs[j])),
                    "mean_px": round(sum(distances) / len(distances), 2) if distances else None,
                }
            )
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


def summary_markdown(info: dict[str, Any], results: list[MarksResult], runs: list[list[dict[str, Any]]], stab: dict[str, Any] | None) -> str:
    first = results[0]
    cond = first.condition
    lines = [
        f"# Spark gate: {info['panel']}",
        "",
        f"- Source: {info['source']}" + (f" (frame at {info['t']} s)" if info["t"] is not None else ""),
        f"- Crop: {info['crop']} -> {info['crop_size'][0]}x{info['crop_size'][1]} px, sent as "
        f"{first.sent_width}x{first.sent_height} (scale {first.scale:.3f})",
        f"- Model: {SPARK_MODEL}, effort {info['effort']}, {len(results)} call(s)",
        f"- Panel condition (run 1): glare {_yn(cond.get('glare'))}, wet {_yn(cond.get('wet'))}, "
        f"dirty {_yn(cond.get('dirty'))}" + (f"; notes: {cond['notes']}" if cond.get("notes") else ""),
        "",
        "## Marks (run 1)",
        "",
    ]
    if runs[0]:
        lines += ["| # | kind | size | conf | grid x,y | crop px x,y | description |", "|---|---|---|---|---|---|---|"]
        for row in runs[0]:
            lines.append(
                f"| {row['n']} | {row['kind']} | {row['size']} | {row['confidence']:.2f} | {row['grid'][0]},{row['grid'][1]} "
                f"| {row['px'][0]},{row['px'][1]} | {row['description']} |"
            )
    else:
        lines.append("No marks reported.")
    lines += [
        "",
        "## Usage",
        "",
        "| run | marks | input | cached | output | reasoning | seconds | usd |",
        "|---|---|---|---|---|---|---|---|",
    ]
    totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "seconds": 0.0, "usd": 0.0}
    for n, result in enumerate(results, 1):
        u = result.usage
        lines.append(
            f"| {n} | {len(result.marks)} | {u.get('input_tokens', 0)} | {u.get('cached_tokens', 0)} | "
            f"{u.get('output_tokens', 0)} | {u.get('reasoning_tokens', 0)} | {result.seconds:.2f} | {result.usd:.6f} |"
        )
        for key in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
            totals[key] += u.get(key, 0)
        totals["seconds"] += result.seconds
        totals["usd"] += result.usd
    lines.append(
        f"| total | | {totals['input_tokens']} | {totals['cached_tokens']} | {totals['output_tokens']} | "
        f"{totals['reasoning_tokens']} | {totals['seconds']:.2f} | {totals['usd']:.6f} |"
    )
    if stab is not None:
        lines += [
            "",
            f"## Stability ({len(results)} runs)",
            "",
            f"- Mark counts per run: {', '.join(str(c) for c in stab['counts'])}",
            f"- Matched pairs within {stab['radius_px']} px: {stab['matched_pairs']} of {stab['possible_pairs']} possible",
            "- Mean pairwise distance of matched marks: "
            + (
                f"{stab['mean_pairwise_px']} px ({100 * stab['mean_pairwise_fraction_of_long_edge']:.2f}% of the long edge)"
                if stab["mean_pairwise_px"] is not None
                else "no matched pairs"
            ),
            "",
            "| runs | matched | possible | mean px |",
            "|---|---|---|---|",
        ]
        for pair in stab["pairs"]:
            mean_px = pair["mean_px"] if pair["mean_px"] is not None else "-"
            lines.append(f"| {pair['runs'][0]} vs {pair['runs'][1]} | {pair['matched']} | {pair['possible']} | {mean_px} |")
    return "\n".join(lines) + "\n"


def _yn(value: Any) -> str:
    return "yes" if value else "no"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spark mark-finding gate on one panel crop.")
    parser.add_argument("source", type=Path, help="image file, or a video (then --t picks the frame)")
    parser.add_argument("--panel", required=True, help="panel noun, e.g. 'car door'")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="crop box in frame pixels")
    parser.add_argument("--t", type=float, default=None, help="seconds into the video")
    parser.add_argument("--effort", default="low", choices=spark_client.EFFORTS)
    parser.add_argument("--n", type=int, default=1, help="repeat the call n times and report stability")
    parser.add_argument("--out", type=Path, required=True, help="output directory, e.g. runs/gate_spark_door")
    args = parser.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.source, args.t)
    crop, box = crop_frame(frame, args.crop)
    Image.fromarray(crop).save(out / "crop.png")

    results = [
        spark_client.find_marks(crop, args.panel, effort=args.effort, run_id=out.name) for _ in range(max(1, args.n))
    ]
    runs = [mark_rows(result) for result in results]
    stab = stability(runs, crop.shape) if len(results) > 1 else None

    cv2.imwrite(str(out / "overlay.png"), draw_overlay(crop, runs))
    info = {
        "source": str(args.source),
        "t": args.t,
        "panel": args.panel,
        "crop": box,
        "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
        "effort": args.effort,
        "model": SPARK_MODEL,
    }
    payload = {
        **info,
        "sent_size": [results[0].sent_width, results[0].sent_height],
        "scale": results[0].scale,
        "runs": [
            {
                "marks": rows,
                "condition": result.condition,
                "usage": result.usage,
                "usd": result.usd,
                "seconds": round(result.seconds, 3),
                "raw": result.raw,
            }
            for result, rows in zip(results, runs)
        ],
        "stability": stab,
    }
    (out / "marks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = summary_markdown(info, results, runs, stab)
    (out / "gate_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"wrote {out / 'marks.json'}, {out / 'overlay.png'}, {out / 'gate_summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
