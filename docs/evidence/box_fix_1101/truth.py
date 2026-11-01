"""Ground truth in crop pixels for the box placement study (no call). Synthetic runs: the render's per-frame
visible_bbox_xyxy of every planted mark, projected into each keyframe crop. Pair C: D1's hand-measured target points."""

from __future__ import annotations

import json
import math
from pathlib import Path

from walkaround import paths
from walkaround.contracts import Keyframe, Mark, read_json_list

HERE = Path(__file__).resolve().parent
SYNTH = HERE.parents[3] / "synthetic" / "out"
RUNS = {"synth_pair1": "pair1", "synth_pair2": "pair2", "synth_pair2_r8": "pair2", "synth_pair2_r9": "pair2"}
D1_DIR = paths.REPO_DIR / "runs" / "d1" / "uco3d_pairC"
STUDY = HERE.parent / "marks_effort_1101" / "answers.jsonl"
_GT: dict[str, dict] = {}


def gt(pair: str) -> dict:
    if pair not in _GT:
        _GT[pair] = json.loads((SYNTH / pair / "ground_truth.json").read_text())
    return _GT[pair]


def to_crop(b: list[float], kf: Keyframe) -> list[float]:
    c = kf.crop
    return [(b[0] - c.dx) * c.scale, (b[1] - c.dy) * c.scale, (b[2] - c.dx) * c.scale, (b[3] - c.dy) * c.scale]


def truth_in_crop(pair: str, kf: Keyframe) -> list[dict]:
    """Planted marks visible on the keyframe's source frame and inside its crop, boxes in crop px."""
    g = gt(pair)
    f = int(round(kf.pts * g["fps"]))
    c = kf.crop
    out = []
    for m in g["marks"]:
        rows = m["frames"].get(kf.role) or []
        row = rows[f] if f < len(rows) and rows[f]["frame"] == f else next((r for r in rows if r["frame"] == f), None)
        if not row or not row.get("visible") or not row.get("visible_bbox_xyxy"):
            continue
        b = to_crop(row["visible_bbox_xyxy"], kf)
        if b[2] > 0 and b[0] < c.w and b[3] > 0 and b[1] < c.h:
            out.append({"id": m["id"], "kind": m["kind"], "box": b, "damage": not m["id"].startswith("control")})
    return out


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def hit(box: list[float], g: list[float], grow: float = 1.5) -> bool:
    """The synthetic scorer's rule: the box centre inside the mark box grown 1.5x, or IoU > 0.1."""
    cx, cy = centre(box)
    gx, gy = centre(g)
    if abs(cx - gx) <= grow / 2 * (g[2] - g[0]) and abs(cy - gy) <= grow / 2 * (g[3] - g[1]):
        return True
    return iou(box, g) > 0.1


def near(box: list[float], g: list[float], px: float = 12.0) -> bool:
    """Tolerant rule (R10's sensitivity): the mark box grown by a quarter of its size or 12 crop px a side."""
    gx, gy = max(0.25 * (g[2] - g[0]), px), max(0.25 * (g[3] - g[1]), px)
    cx, cy = centre(box)
    return (g[0] - gx <= cx <= g[2] + gx and g[1] - gy <= cy <= g[3] + gy) or iou(box, g) > 0.1


def target_hit(t: list[float], box: list[float], long_edge: int) -> bool:
    """D1's real-crop rule: the target inside the box grown a quarter (at least 10 px), or within 0.04 of the long edge
    of the box centre."""
    w, h = box[2] - box[0], box[3] - box[1]
    gx, gy = max(10.0, 0.25 * w), max(10.0, 0.25 * h)
    if box[0] - gx <= t[0] <= box[2] + gx and box[1] - gy <= t[1] <= box[3] + gy:
        return True
    return math.dist(t, centre(box)) <= 0.04 * long_edge


def synth_items(runs: list[str] | None = None) -> list[dict]:
    """One item per keyframe crop that holds a planted mark or a stored Spark box: crop, truth, the run's marks."""
    out = []
    for run, pair in RUNS.items():
        if runs and run not in runs:
            continue
        kfs = [k for k in read_json_list(paths.keyframes_json(run), Keyframe) if k.crop is not None and k.crop.path]
        marks = read_json_list(paths.marks_json(run), Mark)
        for k in kfs:
            own = [m for m in marks if m.kf_id == k.kf_id and m.bbox_crop is not None]
            truth = truth_in_crop(pair, k)
            if not own and not truth:
                continue
            out.append({"run": run, "pair": pair, "kf": k, "truth": truth, "marks": own,
                        "path": str(paths.run_dir(run) / k.crop.path), "w": k.crop.w, "h": k.crop.h})
    return out


def pairc_specs() -> list[dict]:
    specs = json.loads((D1_DIR / "e6_redo_crops.json").read_text())
    return [{"name": s["name"], "panel": s["panel"], "path": str(D1_DIR / "spark_redo" / f"crop_{s['name']}.png"),
             "targets": s["targets"]} for s in specs]
