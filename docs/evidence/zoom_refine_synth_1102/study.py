"""The product's zoom refine on the 47 marks of runs/synth_pair2_r9, scored against Sanat's hand corrections
(eval/box_corrections/synth_pair2_r9.json: 2 correct, 41 moved, 4 not a mark). The run predates both the zoom refine
and the SAM box refine, so its boxes are Spark's own.

Same code path as a new run, called offline on the stored crops with today's thresholds:
  zoom_refine.plan / ask_missing (LIVE, one marks_v1 call per box mark) / refine_marks
  box_refine.plan / ask (LIVE, one SAM 3.1 request per crop chunk) / refine_marks
The marks are the run's own rows of derived/marks.json (their boxes equal the product_box he corrected).
Answers are stored in the run file formats: zoom_answers.json (zoom_marks.json rows), sam_spark.json and sam_zoom.json
(sam_marks.json rows; the SAM step on Spark's box and on the moved box, each asked as a fresh run would).

  LIVE=1 python study.py ask_zoom   # asks only what zoom_answers.json lacks; spend guard on the ET day
  LIVE=1 python study.py ask_sam    # both SAM arms, only what their files lack
  python study.py score             # no call; writes study.json and prints the tables
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import math
import os
import statistics as st
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from walkaround import box_refine, zoom_refine
from walkaround.contracts import Keyframe, Mark, load_thresholds, read_json_list, write_json
from walkaround.marks import _read_rgb

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN = REPO / "runs" / "synth_pair2_r9"
TRUTH = REPO / "eval" / "box_corrections" / "synth_pair2_r9.json"
ZOOM = HERE / "zoom_answers.json"
SAM = {"spark": HERE / "sam_spark.json", "zoom": HERE / "sam_zoom.json"}
OUT = HERE / "study.json"
RUN_ID = "zoom_refine_synth_1102"
ZOOM_PER_CALL = 0.007
SAM_PER_REQUEST = 0.0025
DAY_STOP_USD = float(os.environ.get("DAY_STOP_USD", "49.50"))


def load() -> tuple[dict, list[Mark], dict[str, Keyframe]]:
    truth = json.loads(TRUTH.read_text())["marks"]
    marks = [Mark.model_validate(m) for m in json.loads((RUN / "derived" / "marks.json").read_text()) if m["mark_id"] in truth]
    kfs = {k["kf_id"]: Keyframe.model_validate(k) for k in json.loads((RUN / "derived" / "keyframes.json").read_text())}
    assert len(marks) == len(truth) == 47
    for m in marks:
        assert [float(v) for v in truth[m.mark_id]["product_box"]] == list(m.bbox_crop or []), m.mark_id
    return truth, marks, kfs


def day_spent() -> float:
    et = ZoneInfo("America/New_York")
    day = dt.datetime.now(et).date()
    total = 0.0
    for line in (REPO / "runs" / "ledger.jsonl").open():
        row = json.loads(line)
        try:
            t = dt.datetime.fromisoformat(str(row.get("ts", "")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if t.tzinfo is not None and t.astimezone(et).date() == day and row.get("engine") == "meta-model-api":
            total += float(row.get("usd") or 0)
    return total


def guard(n: int, per: float, what: str) -> bool:
    spent = day_spent()
    print(f"Meta day so far ${spent:.4f}; {n} {what} wanted (about ${n * per:.2f}); stop at ${DAY_STOP_USD:.2f}")
    if spent + n * per > DAY_STOP_USD:
        print("would cross the stop; not asking")
        return False
    return os.environ.get("LIVE") == "1" and n > 0


def zoom_asks() -> list[zoom_refine.ZoomAsk]:
    return read_json_list(ZOOM, zoom_refine.ZoomAsk) if ZOOM.exists() else []


def sam_asks(arm: str) -> list[box_refine.BoxAsk]:
    return read_json_list(SAM[arm], box_refine.BoxAsk) if SAM[arm].exists() else []


def kf_ids(marks: list[Mark]) -> list[str]:
    return sorted({m.kf_id for m in marks})


async def ask_zoom() -> None:
    """As marks._zoom_boxes: each keyframe's box marks planned, a (crop, window) with a stored answer not asked again."""
    th = load_thresholds()
    s = zoom_refine.settings(th)
    assert s is not None
    _truth, marks, kfs = load()
    stored = zoom_asks()
    known = zoom_refine.Index(stored)
    jobs = []
    for kf_id in kf_ids(marks):
        kf = kfs[kf_id]
        seen: set[tuple[int, ...]] = set()
        todo = []
        for m, win in zoom_refine.plan(marks, kf, s):
            if not known.has(kf.crop, win) and tuple(win) not in seen:
                seen.add(tuple(win))
                todo.append((m, win))
        if todo:
            jobs.append((kf, RUN / kf.crop.path, todo))
    if not guard(sum(len(t) for _k, _p, t in jobs), ZOOM_PER_CALL, "zoom calls"):
        return
    asked = await zoom_refine.ask_missing(RUN_ID, jobs, s, th=th, read_rgb=_read_rgb)
    write_json(ZOOM, [*stored, *asked])
    print(f"asked {len(asked)}; Meta day now ${day_spent():.4f}")


def arm_marks(arm: str, th: dict) -> tuple[list[Mark], dict[str, Keyframe]]:
    """The 47 marks with the box each uses before the SAM step: Spark's (arm spark) or moved by the zoom (arm zoom)."""
    _truth, marks, kfs = load()
    if arm == "spark":
        return marks, kfs
    index = zoom_refine.Index(zoom_asks())
    out: list[Mark] = []
    for kf_id in kf_ids(marks):
        out.extend(zoom_refine.refine_marks([m for m in marks if m.kf_id == kf_id], kfs[kf_id], index, th=th))
    return out, kfs


async def ask_sam() -> None:
    """As marks._sam_boxes, once on Spark's boxes and once on the zoom-moved boxes, each its own store."""
    th = load_thresholds()
    s = box_refine.settings(th)
    assert s is not None
    for arm in ("spark", "zoom"):
        marks, kfs = arm_marks(arm, th)
        stored = sam_asks(arm)
        known = box_refine.Index(stored)
        jobs = []
        for kf_id in kf_ids(marks):
            kf = kfs[kf_id]
            for chunk in box_refine.plan([m for m in marks if m.kf_id == kf_id], kf, s):
                todo = [o for o in chunk if not known.has(kf.crop, o.prompt)]
                if todo:
                    jobs.append((kf, [o.model_copy(update={"object_id": j + 1}) for j, o in enumerate(todo)]))
        if not guard(len(jobs), SAM_PER_REQUEST, f"SAM requests ({arm} boxes)"):
            continue
        sem = asyncio.Semaphore(int(th["orchestrator"]["sam_image_semaphore"]))

        async def one(kf: Keyframe, objs: list) -> box_refine.BoxAsk:
            async with sem:
                return await box_refine.ask(RUN_ID, kf, RUN / kf.crop.path, objs)

        asked = await asyncio.gather(*(one(kf, objs) for kf, objs in jobs))
        write_json(SAM[arm], [*stored, *asked])
        print(f"asked {len(asked)} ({arm}); Meta day now ${day_spent():.4f}")


# scoring --------------------------------------------------------------------------------------------------------------


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre(b) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def dist(a, b) -> float:
    (ax, ay), (bx, by) = centre(a), centre(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def on_mark(box, truth) -> bool:
    cx, cy = centre(box)
    return truth[0] <= cx <= truth[2] and truth[1] <= cy <= truth[3]


def inside(pt, win) -> bool:
    return win[0] <= pt[0] <= win[2] and win[1] <= pt[1] <= win[3]


def sam_arm(marks: list[Mark], kfs: dict[str, Keyframe], arm: str, th: dict) -> dict[str, Mark]:
    index = box_refine.Index(sam_asks(arm))
    out = {}
    for kf_id in kf_ids(marks):
        for m in box_refine.refine_marks([m for m in marks if m.kf_id == kf_id], kfs[kf_id], index, th=th):
            out[m.mark_id] = m
    return out


def stats(boxes: dict[str, list[float]], truth: dict, ids: list[str]) -> dict:
    ious = [iou(boxes[i], truth[i]["box"]) for i in ids]
    ds = [dist(boxes[i], truth[i]["box"]) for i in ids]
    return {
        "n": len(ids),
        "median_iou": round(st.median(ious), 3),
        "mean_iou": round(st.mean(ious), 3),
        "median_centre_px": round(st.median(ds), 1),
        "on_mark": sum(on_mark(boxes[i], truth[i]["box"]) for i in ids),
        "iou_ge_05": sum(iou(boxes[i], truth[i]["box"]) >= 0.5 for i in ids),
    }


def wlt(a: dict, b: dict, truth: dict, ids: list[str]) -> list[int]:
    """IoU wins, losses, ties of arm b against arm a (ties within 0.001)."""
    d = [iou(b[i], truth[i]["box"]) - iou(a[i], truth[i]["box"]) for i in ids]
    return [sum(x > 1e-3 for x in d), sum(x < -1e-3 for x in d), sum(abs(x) <= 1e-3 for x in d)]


def sign_p(w: int, l: int) -> float:
    """Two-sided sign test on IoU wins and losses (ties dropped)."""
    n, k = w + l, min(w, l)
    return round(min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n), 8)


def score() -> dict:
    th = load_thresholds()
    truth, marks, kfs = load()
    zoomed, _ = arm_marks("zoom", th)
    zoomed_by = {m.mark_id: m for m in zoomed}
    spark_sam = sam_arm(marks, kfs, "spark", th)
    zoom_sam = sam_arm(zoomed, kfs, "zoom", th)
    arms = {
        "spark": {m.mark_id: list(m.bbox_crop) for m in marks},
        "spark_sam": {k: list(m.bbox_crop) for k, m in spark_sam.items()},
        "zoom": {k: list(m.bbox_crop) for k, m in zoomed_by.items()},
        "zoom_sam": {k: list(m.bbox_crop) for k, m in zoom_sam.items()},
    }
    moved = sorted(k for k, v in truth.items() if v["status"] == "moved")
    real = sorted(k for k, v in truth.items() if v["status"] in ("moved", "correct"))
    table = {a: stats(b, truth, moved) for a, b in arms.items()}
    table_real = {a: stats(b, truth, real) for a, b in arms.items()}
    table["zoom"]["zoom_used"] = sum(zoomed_by[k].zoom_refine.used == "zoom" for k in moved)
    table["spark_sam"]["sam_used"] = sum(spark_sam[k].box_refine.used == "sam" for k in moved)
    table["zoom_sam"]["sam_used"] = sum(zoom_sam[k].box_refine.used == "sam" for k in moved)
    vs = {}
    for name, (a, b) in {"zoom_vs_spark": ("spark", "zoom"), "zoom_sam_vs_spark": ("spark", "zoom_sam"), "spark_sam_vs_spark": ("spark", "spark_sam"), "zoom_sam_vs_zoom": ("zoom", "zoom_sam")}.items():
        w, l, t = wlt(arms[a], arms[b], truth, moved)
        vs[name] = {"w_l_t": [w, l, t], "sign_p": sign_p(w, l)}
    rows = []
    for k in sorted(truth):
        t, z = truth[k], zoomed_by[k].zoom_refine
        hand = t["box"]
        row = {
            "mark_id": k, "status": t["status"], "type": t["type"], "size_class": t["size_class"], "hand": hand, "crop_wh": [t["crop_w"], t["crop_h"]],
            "window": z.window_crop, "zoom_used": z.used, "zoom_boxes": z.zoom_boxes, "zoom_type": z.zoom_type, "zoom_box": z.zoom_bbox_crop, "zoom_note": z.note,
            "spark_sam_used": spark_sam[k].box_refine.used, "zoom_sam_used": zoom_sam[k].box_refine.used,
            "spark_sam_note": spark_sam[k].box_refine.note, "zoom_sam_note": zoom_sam[k].box_refine.note,
            "boxes": {a: [round(v, 2) for v in b[k]] for a, b in arms.items()},
        }
        if hand is not None:
            row["hand_centre_in_window"] = inside(centre(hand), z.window_crop)
            row["iou"] = {a: round(iou(b[k], hand), 3) for a, b in arms.items()}
            row["centre_px"] = {a: round(dist(b[k], hand), 1) for a, b in arms.items()}
        row["moved_px"] = round(dist(arms["zoom"][k], arms["spark"][k]), 1)
        rows.append(row)
    by = {r["mark_id"]: r for r in rows}
    window_hits = sum(by[k]["hand_centre_in_window"] for k in moved)
    not_mark = [
        {"mark_id": r["mark_id"], "spark_type": r["type"], "zoom_used": r["zoom_used"], "zoom_boxes": r["zoom_boxes"], "zoom_type": r["zoom_type"], "moved_px": r["moved_px"],
         "zoom_sam_used": r["zoom_sam_used"], "spark_sam_used": r["spark_sam_used"]}
        for r in rows if r["status"] == "not_mark"
    ]
    correct = [{"mark_id": r["mark_id"], "iou": r["iou"], "zoom_used": r["zoom_used"], "moved_px": r["moved_px"]} for r in rows if r["status"] == "correct"]
    no_box_moved = sum(by[k]["zoom_boxes"] == 0 for k in moved)
    out = {
        "run": "synth_pair2_r9", "truth_sha256": hashlib.sha256(TRUTH.read_bytes()).hexdigest(), "ledger_run_id": RUN_ID, "zoom_settings": zoom_refine.settings(th), "table_moved": table, "table_moved_and_correct": table_real,
        "vs": vs, "hand_centre_in_window_moved": window_hits, "zoom_found_nothing_moved": no_box_moved, "not_mark": not_mark, "correct": correct, "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'arm':10} {'n':>3} {'IoU':>6} {'mean':>6} {'ctr px':>7} {'on':>4} {'>=.5':>5} used")
    for a, v in table.items():
        print(f"{a:10} {v['n']:>3} {v['median_iou']:>6} {v['mean_iou']:>6} {v['median_centre_px']:>7} {v['on_mark']:>4} {v['iou_ge_05']:>5} {v.get('zoom_used', v.get('sam_used', ''))}")
    print("moved and correct (43):")
    for a, v in table_real.items():
        print(f"{a:10} {v['n']:>3} {v['median_iou']:>6} {v['mean_iou']:>6} {v['median_centre_px']:>7} {v['on_mark']:>4} {v['iou_ge_05']:>5}")
    print("vs", json.dumps(vs))
    print(f"his centre inside the zoom window: {window_hits}/{len(moved)}; zoom found nothing on {no_box_moved}/{len(moved)}")
    for x in not_mark:
        print("not_mark", x)
    for x in correct:
        print("correct", x)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "ask_zoom":
        asyncio.run(ask_zoom())
    elif cmd == "ask_sam":
        asyncio.run(ask_sam())
    else:
        score()
