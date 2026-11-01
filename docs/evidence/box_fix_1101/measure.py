"""Before and after the SAM box refinement, against ground truth (box fix 11/1).

Synthetic: the stored Spark box marks of synth_pair1, synth_pair2, synth_pair2_r8 and synth_pair2_r9 on their own
crops, scored against the render's per-frame truth boxes. Pair C: the D1 curated crops (runs/d1/uco3d_pairC/spark_redo/)
with D1's hand-measured target points; the boxes are Spark's low-effort answers stored by D1's gate (gate_spark.json)
and by the R10 study (answers.jsonl, full crop, low).

Every box goes to SAM 3.1 through the product's own code (box_refine.plan, concept, decide and apply with the
thresholds leaf dedup.sam_box_refine): all boxes of one crop answer in one request. Answers are kept in
sam_answers.json, so scoring runs with no call.

  LIVE=1 python measure.py batch near     # crops holding a box near a planted mark, and pair C (the handful of 11/1)
  LIVE=1 python measure.py batch all      # every crop with a stored box (after midnight ET)
  python measure.py score                 # writes measure.json and prints the table

A spend guard stops before DAY_STOP_USD for the ET day (default 48.70) counting the ledger's rows for that day.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth as T  # noqa: E402

from walkaround import box_refine, paths, sam_client  # noqa: E402
from walkaround.claim import SamAnswer, sam_answer_from_run  # noqa: E402
from walkaround.contracts import Crop, Keyframe, Mark, MarkSource, load_thresholds  # noqa: E402

HERE = Path(__file__).resolve().parent
ANSWERS = HERE / "sam_answers.json"
OUT = HERE / "measure.json"
RUN_ID = "box_fix_1101"
PER_REQUEST = 0.0025
DAY_STOP_USD = float(os.environ.get("DAY_STOP_USD", "48.70"))
TH = load_thresholds()
S = box_refine.settings(TH)
GATE = T.D1_DIR / "spark_redo" / "gate_spark.json"


# items: one crop and its Spark boxes ----------------------------------------------------------------------------------


def _mark(i: int, kf_id: str, box: list[float]) -> Mark:
    return Mark(mark_id=f"{kf_id}_b{i}", role="pickup", panel_id="hood", kf_id=kf_id, source=MarkSource.SPARK_BOX, type="unsure", confidence="low",
                size_class="S", bbox_crop=[float(v) for v in box], point_crop=[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def synth_items() -> list[dict]:
    out = []
    for it in T.synth_items():
        if not it["marks"]:
            continue
        kf: Keyframe = it["kf"]
        near = any(T.near(m.bbox_crop, g["box"]) for m in it["marks"] for g in it["truth"])
        out.append({"id": f"{it['run']}:{kf.kf_id}", "set": "synth", "crop": kf.crop, "kf": kf, "path": it["path"], "truth": it["truth"],
                    "marks": list(it["marks"]), "near": near})
    return out


def pairc_items() -> list[dict]:
    specs = {s["name"]: s for s in T.pairc_specs()}
    gate = json.loads(GATE.read_text())
    out = []
    for cr in gate["crops"]:
        s = specs[cr["name"]]
        w, h = Image.open(s["path"]).size
        crop = Crop(path=s["path"], sha256=None, w=w, h=h, scale=1.0, dx=0.0, dy=0.0)
        for r, run in enumerate(cr["efforts"]["low"]["runs"]):
            boxes = [[float(v) for v in m["bbox_px"]] for m in run.get("marks") or [] if m.get("bbox_px")]  # D1 mapped them to crop px
            out.append(_pairc(f"pairC:{cr['name']}:gate{r}", crop, s, boxes))
    for line in T.STUDY.read_text().splitlines():
        a = json.loads(line)
        name = a["crop_id"].removeprefix("pairC_")
        if a["arm"] == "base" and a["effort"] == "low" and name in specs:
            s = specs[name]
            w, h = Image.open(s["path"]).size
            crop = Crop(path=s["path"], sha256=None, w=w, h=h, scale=1.0, dx=0.0, dy=0.0)
            out.append(_pairc(f"pairC:{name}:study{a['rep']}", crop, s, [b["box"] for b in a["boxes"]]))
    return out


def _pairc(ident: str, crop: Crop, s: dict, boxes: list[list[float]]) -> dict:
    kf_id = ident.replace(":", "_")
    kf = Keyframe.model_construct(kf_id=kf_id, role="pickup", crop=crop)
    return {"id": ident, "set": "pairc", "crop": crop, "kf": kf, "path": s["path"], "targets": s["targets"],
            "marks": [_mark(i, kf_id, b) for i, b in enumerate(boxes)], "near": True}


def items() -> list[dict]:
    return [*synth_items(), *pairc_items()]


# asking --------------------------------------------------------------------------------------------------------------


def _key(it: dict, objs: list) -> str:
    return it["id"] + "|" + ";".join(",".join(str(v) for v in o.prompt) for o in objs)


def day_spent() -> float:
    day = (dt.datetime.now(dt.timezone(dt.timedelta(hours=-5)))).strftime("%Y-%m-%d")
    total = 0.0
    with open(paths.REPO_DIR / "runs" / "ledger.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            if str(row.get("ts", "")).startswith(day):
                total += float(row.get("usd") or 0)
    return total


async def batch(which: str) -> None:
    stored = json.loads(ANSWERS.read_text()) if ANSWERS.exists() else {}
    jobs = []
    for it in items():
        if which == "near" and not it["near"]:
            continue
        for chunk in box_refine.plan(it["marks"], it["kf"], S):
            if _key(it, chunk) not in stored:
                jobs.append((it, chunk))
    spent = day_spent()
    budget = int(max(0.0, DAY_STOP_USD - spent) // PER_REQUEST)
    print(f"day so far ${spent:.4f}; {len(jobs)} requests wanted, {min(len(jobs), budget)} within the ${DAY_STOP_USD:.2f} stop")
    if os.environ.get("LIVE") != "1":
        return
    jobs = jobs[:budget]
    sem = asyncio.Semaphore(8)

    async def one(it: dict, chunk: list) -> None:
        text = box_refine.concept(chunk, it["crop"])
        async with sem:
            try:
                run = await sam_client.segment_image(Path(it["path"]), text, run_id=RUN_ID, tag="box_fix_measure")
                ans = sam_answer_from_run(run)
            except sam_client.SamRequestError as exc:
                ans = SamAnswer(concept=text, status="failed", error=str(exc)[:300])
        stored[_key(it, chunk)] = {"objects": [o.model_dump(mode="json") for o in chunk], "answer": ans.model_dump(mode="json")}

    await asyncio.gather(*(one(*j) for j in jobs))
    ANSWERS.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n")
    print(f"asked {len(jobs)}; day now ${day_spent():.4f}")


# scoring ------------------------------------------------------------------------------------------------------------


def refined(it: dict, stored: dict) -> list[tuple[Mark, Mark] | None]:
    """(before, after) per mark of the item; None when no stored answer covers the mark."""
    out: list[tuple[Mark, Mark] | None] = []
    asks = []
    for chunk in box_refine.plan(it["marks"], it["kf"], S):
        row = stored.get(_key(it, chunk))
        if row is not None:
            asks.append(box_refine.BoxAsk(kf_id=it["kf"].kf_id, role="pickup", crop_path=it["crop"].path, crop_sha256=it["crop"].sha256, crop_w=it["crop"].w, crop_h=it["crop"].h,
                                          objects=row["objects"], answer=row["answer"]))
    index = box_refine.Index(asks)
    for m in it["marks"]:
        hit = index.get(it["crop"], box_refine.prompt_of(m, it["crop"], S))
        if hit is None:
            out.append(None)
            continue
        r = box_refine.decide(m, hit[0], hit[1], it["crop"], S)
        after = box_refine.apply(m, r, it["crop"], TH) if it["set"] == "synth" else m.model_copy(update={"bbox_crop": r.sam_bbox_crop if r.used == "sam" else m.bbox_crop, "box_refine": r})
        out.append((m, after))
    return out


def _synth_scores(boxes: list[list[float]], truth: list[dict]) -> dict:
    dmg = [g for g in truth if g["damage"]]
    found, false, offs, ious = set(), 0, [], []
    for b in boxes:
        ids = [g for g in dmg if T.hit(b, g["box"])]
        if not ids:
            false += 1
            continue
        for g in ids:
            found.add(g["id"])
            offs.append(math.dist(T.centre(b), T.centre(g["box"])))
            ious.append(T.iou(b, g["box"]))
    return {"found": found, "eligible": {g["id"] for g in dmg}, "false": false, "offs": offs, "ious": ious}


def _med(v: list[float]) -> float | None:
    return round(st.median(v), 3) if v else None


def score() -> dict:
    stored = json.loads(ANSWERS.read_text()) if ANSWERS.exists() else {}
    rows = []
    tot = {"synth": {"before": {"hits": 0, "eligible": 0, "false": 0, "offs": [], "ious": []}, "after": {"hits": 0, "eligible": 0, "false": 0, "offs": [], "ious": []}},
           "pairc": {"before": {"hits": 0, "eligible": 0, "unmatched": 0, "dists": []}, "after": {"hits": 0, "eligible": 0, "unmatched": 0, "dists": []}}}
    paired = {"offs_before": [], "offs_after": [], "ious_before": [], "ious_after": []}
    used = {"sam": 0, "spark": 0}
    notes: dict[str, int] = {}
    crops = 0
    for it in items():
        pairs = refined(it, stored)
        if not pairs or any(p is None for p in pairs):
            continue  # scored only where every box of the crop has its answer
        crops += 1
        before = [p[0].bbox_crop for p in pairs]  # type: ignore[index]
        after = [p[1].bbox_crop for p in pairs]  # type: ignore[index]
        for _b, a in pairs:  # type: ignore[misc]
            used[a.box_refine.used] += 1
            if a.box_refine.used == "spark":
                key = a.box_refine.note.split(" (")[0]
                key = "mask too big for a mark" if ("covers" in key or "times Spark" in key) else ("mask off Spark's box" if "sits on" in key else key)
                notes[key] = notes.get(key, 0) + 1
        if it["set"] == "synth":
            for side, boxes in (("before", before), ("after", after)):
                s = _synth_scores(boxes, it["truth"])
                t = tot["synth"][side]
                t["hits"] += len(s["found"])
                t["eligible"] += len(s["eligible"])
                t["false"] += s["false"]
                t["offs"] += s["offs"]
                t["ious"] += s["ious"]
            for b, a in zip(before, after):
                near = [g for g in it["truth"] if g["damage"] and (T.near(b, g["box"]) or T.near(a, g["box"]))]
                if near:
                    g = min(near, key=lambda g: math.dist(T.centre(b), T.centre(g["box"])))
                    paired["offs_before"].append(math.dist(T.centre(b), T.centre(g["box"])))
                    paired["offs_after"].append(math.dist(T.centre(a), T.centre(g["box"])))
                    paired["ious_before"].append(T.iou(b, g["box"]))
                    paired["ious_after"].append(T.iou(a, g["box"]))
                    rows.append({"item": it["id"], "truth": g["id"], "before": [round(v, 1) for v in b], "after": [round(v, 1) for v in a],
                                 "off_before": round(paired["offs_before"][-1], 1), "off_after": round(paired["offs_after"][-1], 1),
                                 "iou_before": round(paired["ious_before"][-1], 3), "iou_after": round(paired["ious_after"][-1], 3)})
        else:
            long_edge = max(it["crop"].w, it["crop"].h)
            for side, boxes in (("before", before), ("after", after)):
                t = tot["pairc"][side]
                t["eligible"] += len(it["targets"])
                t["hits"] += sum(any(T.target_hit(tg, b, long_edge) for b in boxes) for tg in it["targets"])
                for b in boxes:
                    hits = [tg for tg in it["targets"] if T.target_hit(tg, b, long_edge)]
                    if hits:
                        t["dists"].append(min(math.dist(T.centre(b), tg) for tg in hits))
                    else:
                        t["unmatched"] += 1
            for b, a in zip(before, after):
                if any(T.target_hit(tg, b, long_edge) or T.target_hit(tg, a, long_edge) for tg in it["targets"]):
                    rows.append({"item": it["id"], "before": [round(v, 1) for v in b], "after": [round(v, 1) for v in a],
                                 "dist_before": round(min(math.dist(T.centre(b), tg) for tg in it["targets"]), 1),
                                 "dist_after": round(min(math.dist(T.centre(a), tg) for tg in it["targets"]), 1)})
    summary = {
        "crops scored": crops,
        "boxes using SAM's box / Spark's box": f"{used['sam']} / {used['spark']}",
        "why Spark's box was kept": notes,
        "synthetic": {side: {"planted marks hit": f"{t['hits']}/{t['eligible']}", "false boxes": t["false"], "median centre offset px (hits)": _med(t["offs"]),
                             "median IoU (hits)": _med(t["ious"])} for side, t in tot["synth"].items()},
        "synthetic, same boxes near a planted mark": {"n": len(paired["offs_before"]), "median centre offset px before / after": [_med(paired["offs_before"]), _med(paired["offs_after"])],
                                                      "median IoU before / after": [_med(paired["ious_before"]), _med(paired["ious_after"])],
                                                      "closer / farther / same": [sum(a < b - 0.5 for a, b in zip(paired["offs_after"], paired["offs_before"])),
                                                                                  sum(a > b + 0.5 for a, b in zip(paired["offs_after"], paired["offs_before"])),
                                                                                  sum(abs(a - b) <= 0.5 for a, b in zip(paired["offs_after"], paired["offs_before"]))]},
        "pair C (D1 targets)": {side: {"targets hit": f"{t['hits']}/{t['eligible']}", "boxes on no target": t["unmatched"], "median distance to the target px (boxes on a target)": _med(t["dists"])}
                                for side, t in tot["pairc"].items()},
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    if sys.argv[1:2] == ["batch"]:
        asyncio.run(batch(sys.argv[2] if len(sys.argv) > 2 else "near"))
    else:
        score()
