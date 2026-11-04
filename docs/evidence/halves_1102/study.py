"""Halves recall pass on stored crops (11/2), scored against Sanat's box-fixer labels. No new run.

Arms, per crop he checked (crops_seen of eval/box_corrections/<run>.json):
  full    the run's own Spark answer on the full crop, as he labeled it (correct, moved = found; not_mark = false box;
          his "missed" boxes = misses).
  halves  full plus the product's halves pass (walkaround/halves_pass.py): two marks_v1 calls, one per enlarged half,
          each box mapped back, dropped when it is the same mark as a box already kept, the rest moved by the zoom
          refine and tightened by the SAM step exactly as a new run does (zoom_refine.plan / ask_missing / refine_marks,
          box_refine.plan / ask / refine_marks).
Runs: friend_pair1 (every answered crop is asked, so the offline replay can run the whole pair), synth_pair2_r9 and
pairC_core3 (crops_seen only). Thresholds: the run's own, with today's dedup.spark_zoom_refine and dedup.sam_box_refine
where the run predates them, plus dedup.spark_halves_pass (LEAF below).

Scoring rule (fixed before any answer came back): a new box (its final box, after the zoom move and SAM) hits one of
his boxes g when IoU >= 0.1, or its centre lies inside g grown by max(10 px, a quarter of g's side) on each side, or
g's centre lies inside it. Each new box takes the nearest box it hits (real marks first: correct, moved, missed; then
his not_mark boxes; then the run's unlabeled boxes). A new box on a missed mark finds it; on a mark already found it is
a duplicate; on a not_mark box or on nothing it is a false box; on an unlabeled box it is unknown. Strict variant:
centre inside g grown by 10 px only.

  LIVE=1 python study.py ask_halves | ask_zoom | ask_sam   (spend guard on the ET day, ledger run id halves_1102)
  python study.py score
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from walkaround import box_refine, halves_pass, zoom_refine
from walkaround.contracts import Keyframe, Mark, SparkRequestRecord, load_thresholds, read_json_list, write_json
from walkaround.marks import _box_marks, _reusable_records, _wheel_mask, _read_rgb, landmarks_for

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = ["friend_pair1", "synth_pair2_r9", "pairC_core3"]
ALL_CROPS = {"friend_pair1"}
SKIP_CROPS = {"pairC_core3": {"return_windshield_kf1"}}  # in crops_seen since 11/2 09:58Z, its two boxes unlabeled
RUN_ID = "halves_1102"
LEAF = {"enabled": True, "span_frac": 0.55, "scale": 2.0, "dup_iou": 0.3}
DAY_STOP_USD = float(os.environ.get("DAY_STOP_USD", "22.00"))
PER = {"halves": 0.0055, "zoom": 0.0065, "sam": 0.0025}
FILES = {k: HERE / f"{k}_answers.json" for k in ("halves", "zoom", "sam")}


def th_for(run: str) -> dict:
    th = json.loads(json.dumps(json.loads((REPO / "runs" / run / "run.json").read_text())["thresholds"]))
    today = load_thresholds()
    for leaf in ("spark_zoom_refine", "sam_box_refine"):
        th["dedup"].setdefault(leaf, today["dedup"][leaf])
    th["dedup"]["spark_halves_pass"] = dict(LEAF)
    return th


def truth(run: str) -> dict:
    return json.loads((REPO / "eval" / "box_corrections" / f"{run}.json").read_text())


def keyframes(run: str) -> dict[str, Keyframe]:
    return {k.kf_id: k for k in read_json_list(REPO / "runs" / run / "derived" / "keyframes.json", Keyframe)}


def results(run: str, th: dict) -> dict[str, object]:
    """kf_id -> the stored full-crop MarksResult, chosen as rebuild_marks chooses it."""
    from walkaround import spark_client

    root = REPO / "runs" / run
    kfs = list(keyframes(run).values())
    recs = read_json_list(root / "spark.json", SparkRequestRecord)
    valid = _reusable_records(recs, kfs)
    last = {r.input_refs[0].path: r for r in recs if r.purpose.value == "marks" and r.input_refs}
    out = {}
    for kf in kfs:
        if kf.crop is None or kf.marks_status is not None and kf.marks_status.value == "not_run":
            continue
        rec = (valid.get((kf.crop.sha256, kf.panel_id)) if kf.crop.sha256 else None) or last.get(kf.crop.path)
        if rec is None:
            continue
        res = spark_client.marks_result_from_record(rec, orig_width=kf.crop.w, orig_height=kf.crop.h, landmarks=landmarks_for(kf.panel_id, th))
        if res.answer is not None:
            out[kf.kf_id] = res
    return out


def eval_crops(run: str) -> list[str]:
    return [k for k in truth(run)["crops_seen"] if k not in SKIP_CROPS.get(run, set())]


def ask_crops(run: str, res: dict) -> list[str]:
    out = sorted(res) if run in ALL_CROPS else [k for k in eval_crops(run) if k in res]
    pilot = os.environ.get("PILOT")
    return [k for k in out if k == pilot] if pilot else out


def stored(kind: str) -> list:
    model = {"halves": halves_pass.HalvesAsk, "zoom": zoom_refine.ZoomAsk, "sam": box_refine.BoxAsk}[kind]
    return read_json_list(FILES[kind], model) if FILES[kind].exists() else []


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


def guard(n: int, kind: str) -> bool:
    spent = day_spent()
    print(f"Meta day ${spent:.4f}; {n} {kind} wanted (about ${n * PER[kind]:.2f}); stop at ${DAY_STOP_USD:.2f}")
    if spent + n * PER[kind] > DAY_STOP_USD:
        print("would cross the stop; not asking")
        return False
    return os.environ.get("LIVE") == "1" and n > 0


def new_marks(run: str, th: dict, res: dict, kfs: dict, crops: list[str]) -> list[Mark]:
    """The halves marks kept on each crop (product union), before the zoom move."""
    root = REPO / "runs" / run
    index = halves_pass.Index(stored("halves"))
    out = []
    for kid in crops:
        if kid in res:
            out.extend(m for m in _box_marks(root, kfs[kid], res[kid], th=th, halves_index=index) if m.halves is not None)
    return out


def zoom_index(run: str) -> zoom_refine.Index:
    return zoom_refine.Index([*zoom_refine.read_asks(run), *stored("zoom")]) if (REPO / "runs" / run / "zoom_marks.json").exists() else zoom_refine.Index(stored("zoom"))


def sam_index(run: str) -> box_refine.Index:
    return box_refine.Index([*box_refine.read_asks(run), *stored("sam")])


def moved(run: str, th: dict, marks: list[Mark], kfs: dict) -> list[Mark]:
    idx = zoom_index(run)
    out = []
    for kid in sorted({m.kf_id for m in marks}):
        out.extend(zoom_refine.refine_marks([m for m in marks if m.kf_id == kid], kfs[kid], idx, th=th))
    return out


def final(run: str, th: dict, marks: list[Mark], kfs: dict) -> list[Mark]:
    idx = sam_index(run)
    out = []
    for kid in sorted({m.kf_id for m in marks}):
        out.extend(box_refine.refine_marks([m for m in marks if m.kf_id == kid], kfs[kid], idx, th=th))
    return out


async def ask_halves() -> None:
    have = halves_pass.Index(stored("halves"))
    jobs, n = [], 0
    for run in RUNS:
        th = th_for(run)
        s = halves_pass.settings(th)
        res, kfs = results(run, th), keyframes(run)
        for kid in ask_crops(run, res):
            kf = kfs[kid]
            todo = [(h, w) for h, w in halves_pass.plan(kf, s) if not have.has(kf.crop, w)]
            if todo:
                jobs.append((run, th, s, kf, todo))
                n += len(todo)
    if not guard(n, "halves"):
        return
    asked = []
    for run in RUNS:
        mine = [(kf, REPO / "runs" / run / kf.crop.path, todo) for r, _th, _s, kf, todo in jobs if r == run]
        if mine:
            th = th_for(run)
            got = await halves_pass.ask_missing(RUN_ID, mine, halves_pass.settings(th), th=th, read_rgb=_read_rgb)
            write_json(FILES["halves"], [*stored("halves"), *got])
            asked += got
    print(f"asked {len(asked)}; Meta day now ${day_spent():.4f}")


async def ask_zoom() -> None:
    jobs, n = {}, 0
    for run in RUNS:
        th = th_for(run)
        s = zoom_refine.settings(th)
        res, kfs = results(run, th), keyframes(run)
        idx = zoom_index(run)
        marks = new_marks(run, th, res, kfs, ask_crops(run, res))
        for kid in sorted({m.kf_id for m in marks}):
            kf = kfs[kid]
            seen, todo = set(), []
            for m, win in zoom_refine.plan(marks, kf, s):
                if not idx.has(kf.crop, win) and tuple(win) not in seen:
                    seen.add(tuple(win))
                    todo.append((m, win))
            if todo:
                jobs.setdefault(run, []).append((kf, REPO / "runs" / run / kf.crop.path, todo))
                n += len(todo)
    if not guard(n, "zoom"):
        return
    for run, mine in jobs.items():
        th = th_for(run)
        asked = await zoom_refine.ask_missing(RUN_ID, mine, zoom_refine.settings(th), th=th, read_rgb=_read_rgb)
        write_json(FILES["zoom"], [*stored("zoom"), *asked])
    print(f"Meta day now ${day_spent():.4f}")


async def ask_sam() -> None:
    jobs, n = [], 0
    for run in RUNS:
        th = th_for(run)
        s = box_refine.settings(th)
        res, kfs = results(run, th), keyframes(run)
        idx = sam_index(run)
        marks = moved(run, th, new_marks(run, th, res, kfs, ask_crops(run, res)), kfs)
        for kid in sorted({m.kf_id for m in marks}):
            kf = kfs[kid]
            for chunk in box_refine.plan([m for m in marks if m.kf_id == kid], kf, s):
                todo = [o for o in chunk if not idx.has(kf.crop, o.prompt)]
                if todo:
                    jobs.append((run, kf, [o.model_copy(update={"object_id": j + 1}) for j, o in enumerate(todo)]))
    if not guard(len(jobs), "sam"):
        return
    sem = asyncio.Semaphore(4)

    async def one(run: str, kf: Keyframe, objs: list) -> box_refine.BoxAsk:
        async with sem:
            return await box_refine.ask(RUN_ID, kf, REPO / "runs" / run / kf.crop.path, objs)

    asked = await asyncio.gather(*(one(*j) for j in jobs))
    write_json(FILES["sam"], [*stored("sam"), *asked])
    print(f"asked {len(asked)}; Meta day now ${day_spent():.4f}")


# scoring --------------------------------------------------------------------------------------------------------------


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def grown(g, strict: bool):
    if strict:
        mx = my = 10.0
    else:
        mx, my = max(10.0, 0.25 * (g[2] - g[0])), max(10.0, 0.25 * (g[3] - g[1]))
    return [g[0] - mx, g[1] - my, g[2] + mx, g[3] + my]


def inside(p, b) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def hits(b, g, strict: bool = False) -> bool:
    if strict:
        return inside(centre(b), grown(g, True))
    return iou(b, g) >= 0.1 or inside(centre(b), grown(g, False)) or inside(centre(g), b)


def gt_boxes(run: str) -> dict[str, list[dict]]:
    """Per crop: his boxes with a kind (found, missed, not_mark, unlabeled) and an id."""
    t = truth(run)
    by: dict[str, list[dict]] = {}
    for mid, v in t["marks"].items():
        kind = {"correct": "found", "moved": "found", "not_mark": "not_mark"}[v["status"]]
        box = v["box"] if v.get("box") is not None else v["product_box"]  # a not_mark row keeps the box he rejected
        by.setdefault(v["kf_id"], []).append({"id": mid, "kind": kind, "box": box, "status": v["status"]})
    for v in t["missed"]:
        by.setdefault(v["kf_id"], []).append({"id": v["id"], "kind": "missed", "box": v["box"], "status": "missed"})
    labeled = set(t["marks"])
    for m in json.loads((REPO / "runs" / run / "derived" / "marks.json").read_text()):
        if m["kf_id"] in t["crops_seen"] and m["mark_id"] not in labeled and m.get("bbox_crop"):
            by.setdefault(m["kf_id"], []).append({"id": m["mark_id"], "kind": "unlabeled", "box": m["bbox_crop"], "status": "unlabeled"})
    return by


def score_run(run: str, strict: bool = False, stage: str = "final") -> dict:
    th = th_for(run)
    res, kfs = results(run, th), keyframes(run)
    crops = eval_crops(run)
    gt = gt_boxes(run)
    base = new_marks(run, th, res, kfs, crops)
    marks = {"halves": base, "moved": moved(run, th, base, kfs)}
    marks["final"] = final(run, th, marks["moved"], kfs)
    real = [g for k in crops for g in gt.get(k, []) if g["kind"] in ("found", "missed")]
    out = {"crops": len(crops), "real": len(real), "found_full": sum(g["kind"] == "found" for g in real), "missed_full": sum(g["kind"] == "missed" for g in real),
           "false_full": sum(g["kind"] == "not_mark" for k in crops for g in gt.get(k, [])), "new_boxes": len(marks[stage])}
    took: dict[str, str] = {}
    rows = []
    order = {"found": 0, "missed": 0, "not_mark": 1, "unlabeled": 2}
    for m in marks[stage]:
        b = list(m.bbox_crop)
        cands = [g for g in gt.get(m.kf_id, []) if hits(b, g["box"], strict)]
        cands.sort(key=lambda g: (order[g["kind"]], (centre(b)[0] - centre(g["box"])[0]) ** 2 + (centre(b)[1] - centre(g["box"])[1]) ** 2))
        g = cands[0] if cands else None
        if g is None:
            what = "false"
        elif g["kind"] == "missed" and g["id"] not in took:
            what = "finds_missed"
        elif g["kind"] in ("found", "missed"):
            what = "duplicate"
        elif g["kind"] == "not_mark":
            what = "false_on_not_mark"
        else:
            what = "unknown"
        if g is not None and g["kind"] in ("found", "missed"):
            took.setdefault(g["id"], m.mark_id)
        rows.append({"mark_id": m.mark_id, "kf_id": m.kf_id, "type": m.type.value, "conf": m.confidence.value, "caps": m.caps, "box": [round(v, 1) for v in b],
                     "what": what, "gt": g["id"] if g else None, "gt_status": g["status"] if g else None})
    n = lambda w: sum(r["what"] == w for r in rows)
    out.update({"finds_missed": n("finds_missed"), "duplicate": n("duplicate"), "false_new": n("false") + n("false_on_not_mark"), "false_on_not_mark": n("false_on_not_mark"), "unknown": n("unknown")})
    out["found_halves"] = out["found_full"] + out["finds_missed"]
    out["missed_halves"] = out["missed_full"] - out["finds_missed"]
    out["false_halves"] = out["false_full"] + out["false_new"]
    capped = {"dirt", "reflection", "unsure"}
    out["finds_missed_uncapped"] = sum(r["what"] == "finds_missed" and r["type"] not in capped and r["conf"] != "low" for r in rows)
    out["false_new_uncapped"] = sum(r["what"] in ("false", "false_on_not_mark") and r["type"] not in capped and r["conf"] != "low" for r in rows)
    out["missed_found_ids"] = sorted(r["gt"] for r in rows if r["what"] == "finds_missed")
    out["rows"] = rows
    return out


def score() -> dict:
    allout = {}
    for run in RUNS:
        allout[run] = {"main": score_run(run), "strict": {k: v for k, v in score_run(run, strict=True).items() if k != "rows"},
                       "raw_halves_box": {k: v for k, v in score_run(run, stage="halves").items() if k != "rows"}}
    spend = {k: round(sum(float(getattr(a, "record", None).dollars or 0) for a in stored(k) if getattr(a, "record", None) is not None), 4) for k in ("halves", "zoom")}
    allout["calls"] = {"halves": len(stored("halves")), "zoom": len(stored("zoom")), "sam": len(stored("sam")), "spark_dollars": spend}
    (HERE / "study.json").write_text(json.dumps(allout, indent=1) + "\n")
    print("| run | crops | real | full: found / missed / false | halves: found / missed / false | new boxes: finds missed / duplicate / false / unknown |")
    for run in RUNS:
        for v in ("main", "strict", "raw_halves_box"):
            o = allout[run][v]
            print(f"| {run} {v} | {o['crops']} | {o['real']} | {o['found_full']} / {o['missed_full']} / {o['false_full']} | {o['found_halves']} / {o['missed_halves']} / {o['false_halves']} | "
                  f"{o['finds_missed']} / {o['duplicate']} / {o['false_new']} / {o['unknown']} | uncapped finds {o['finds_missed_uncapped']} false {o['false_new_uncapped']}")
    print(json.dumps(allout["calls"]))
    return allout


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "ask_halves":
        asyncio.run(ask_halves())
    elif cmd == "ask_zoom":
        asyncio.run(ask_zoom())
    elif cmd == "ask_sam":
        asyncio.run(ask_sam())
    else:
        score()
