"""Zoom refine pick, nearest box (as now) against nearest damage-typed box when Spark typed damage (prefer_kind), scored
on Sanat's hand boxes with the stored zoom answers: pairC_core3 (R14, 48 moved) and synth_pair2_r9 (R17, 41 moved).
Product code (zoom_refine.refine_marks), no call. Box before the SAM step (SAM answers exist only for the old boxes).

Usage: .venv/bin/python docs/evidence/hero_1102/zoom_kind.py   -> zoom_kind.json
"""

from __future__ import annotations

import copy
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from walkaround import contracts as c, zoom_refine  # noqa: E402


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def stats(boxes: dict, hand: dict) -> dict:
    ious = [iou(boxes[k], hand[k]) for k in hand]
    offs = [((centre(boxes[k])[0] - centre(hand[k])[0]) ** 2 + (centre(boxes[k])[1] - centre(hand[k])[1]) ** 2) ** 0.5 for k in hand]
    on = sum(hand[k][0] <= centre(boxes[k])[0] <= hand[k][2] and hand[k][1] <= centre(boxes[k])[1] <= hand[k][3] for k in hand)
    return {"n": len(hand), "median_iou": round(st.median(ious), 3), "mean_iou": round(st.mean(ious), 3), "median_offset_px": round(st.median(offs), 1), "on_mark": on, "iou_ge_05": sum(i >= 0.5 for i in ious)}


def compare(a: dict, b: dict, hand: dict) -> dict:
    w = sum(iou(b[k], hand[k]) > iou(a[k], hand[k]) + 1e-9 for k in hand)
    l_ = sum(iou(b[k], hand[k]) < iou(a[k], hand[k]) - 1e-9 for k in hand)
    changed = sorted(k for k in hand if a[k] != b[k])
    return {"wins": w, "losses": l_, "ties": len(hand) - w - l_, "boxes_changed": len(changed), "changed": changed}


def th_with(prefer: bool) -> dict:
    th = c.load_thresholds()
    th = copy.deepcopy(th)
    th["dedup"]["spark_zoom_refine"] = {**th["dedup"]["spark_zoom_refine"], "prefer_kind": prefer}
    return th


def pair_c() -> dict:
    ev = REPO / "docs" / "evidence" / "zoom_refine_1101"
    study = json.loads((ev / "study.json").read_text())
    answers = json.loads((ev / "answers.json").read_text())
    run_marks = {m["mark_id"]: c.Mark.model_validate(m) for m in json.loads((REPO / "runs" / "pairC_core3" / "derived" / "marks.json").read_text())}
    kfs = {k["kf_id"]: c.Keyframe.model_validate(k) for k in json.loads((REPO / "runs" / "pairC_core3" / "derived" / "keyframes.json").read_text())}
    out = {}
    for prefer in (False, True):
        th = th_with(prefer)
        s = zoom_refine.settings(th)
        boxes, hand = {}, {}
        for row in study["rows"]:
            if row["status"] != "moved":
                continue
            m = run_marks[row["mark_id"]].model_copy(update={"bbox_crop": list(row["spark"])})
            kf = kfs[m.kf_id]
            e = answers[f"{row['mark_id']}#w1"]
            win = zoom_refine.window(list(row["spark"]), kf.crop, s)
            assert win == e["window"], row["mark_id"]
            a = zoom_refine.ZoomAsk(kf_id=kf.kf_id, role=m.role, mark_id=m.mark_id, panel_id=m.panel_id, crop_path=kf.crop.path, crop_sha256=kf.crop.sha256, crop_w=kf.crop.w, crop_h=kf.crop.h, window=win, out_w=1280, out_h=720, record=c.SparkRequestRecord.model_validate(e["record"]))
            r = zoom_refine.refine_marks([m], kf, zoom_refine.Index([a]), th=th)[0]
            boxes[m.mark_id] = list(r.bbox_crop)
            hand[m.mark_id] = row["hand"]
        out[prefer] = (boxes, hand)
    return out


def synth() -> dict:
    mod = load_module(REPO / "docs" / "evidence" / "zoom_refine_synth_1102" / "study.py", "zoom_synth_study")
    truth, _marks, _kfs = mod.load()
    hand = {k: v["box"] for k, v in truth.items() if v["status"] == "moved"}
    out = {}
    for prefer in (False, True):
        zoomed, _ = mod.arm_marks("zoom", th_with(prefer))
        out[prefer] = ({m.mark_id: list(m.bbox_crop) for m in zoomed if m.mark_id in hand}, hand)
    return out


result = {}
for name, fn in (("pairC_core3", pair_c), ("synth_pair2_r9", synth)):
    arms = fn()
    (a, hand), (b, _) = arms[False], arms[True]
    result[name] = {"nearest": stats(a, hand), "prefer_kind": stats(b, hand), "prefer_vs_nearest": compare(a, b, hand)}
(HERE / "zoom_kind.json").write_text(json.dumps(result, indent=1) + "\n")
for k, v in result.items():
    print(k, json.dumps({x: v[x] for x in ("nearest", "prefer_kind")}), {x: y for x, y in v["prefer_vs_nearest"].items() if x != "changed"})
