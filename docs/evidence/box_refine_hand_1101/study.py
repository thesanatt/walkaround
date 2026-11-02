"""SAM box refine variants against Sanat's hand-corrected boxes on pairC_core3 (eval/box_corrections/pairC_core3.json).

Variants per corrected Spark box mark, on the same crop the product sends:
  a   Spark's box as is (the product box of pairC_core3, frozen before the refine leaf)
  b   the current refine: Spark's box + pad_px as a box prompt, box_refine.decide with the thresholds leaf
  c   Spark's box grown by one Spark patch (28 sent px, mapped to crop px per axis) on every side, as a box prompt
  cu  c shifted up by the measured median (UP_PX crop px)
  d   Spark's own point as a single positive point prompt
All of a crop's prompts go in one request (up to 16 objects; more objects split into a second request). Objects in one
request answer independently (box fix 11/1: a nine-object request matched the one-object requests).

  LIVE=1 python study.py ask     # asks only what answers.json lacks; spend guard on the ET day
  python study.py score          # no call; writes study.json and prints the table
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import statistics as st
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint

from walkaround import box_refine, privacy, sam_client
from walkaround.claim import SamAnswer, _detection, sam_answer_from_run
from walkaround.contracts import Keyframe, Mark, load_thresholds

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN = REPO / "runs" / "pairC_core3"
TRUTH = REPO / "eval" / "box_corrections" / "pairC_core3.json"
ANSWERS = HERE / "answers.json"
OUT = HERE / "study.json"
RUN_ID = "box_refine_hand_1101"
PER_REQUEST = 0.0025
DAY_STOP_USD = float(os.environ.get("DAY_STOP_USD", "49.50"))
PATCH_SENT_PX = 28
UP_PX = 4  # median dy of his 48 moved boxes is -4.0 crop px (centre of his box minus centre of Spark's)
MAX_OBJECTS = 16
VARIANTS = ["b", "c", "cu", "d"]
TH = load_thresholds()
S = box_refine.settings(TH)


def load() -> tuple[dict, dict[str, Mark], dict[str, Keyframe]]:
    truth = json.loads(TRUTH.read_text())["marks"]
    marks = {m["mark_id"]: Mark.model_validate(m) for m in json.loads((RUN / "derived" / "marks.json").read_text()) if m["mark_id"] in truth}
    kfs = {k["kf_id"]: Keyframe.model_validate(k) for k in json.loads((RUN / "derived" / "keyframes.json").read_text())}
    return truth, marks, kfs


def patch_px(crop) -> tuple[float, float]:
    from walkaround import spark_client

    sw, sh = spark_client.sent_size(crop.w, crop.h)
    return PATCH_SENT_PX * crop.w / sw, PATCH_SENT_PX * crop.h / sh


def prompts(m: Mark, kf: Keyframe) -> dict[str, dict]:
    crop = kf.crop
    b = box_refine.prompt_of(m, crop, S)
    gx, gy = patch_px(crop)
    x1, y1, x2, y2 = m.bbox_crop
    grown = sam_client.prompt_box([x1 - gx, y1 - gy, x2 + gx, y2 + gy], (crop.w, crop.h), pad_px=int(S["pad_px"]))
    up = sam_client.prompt_box([x1 - gx, y1 - gy - UP_PX, x2 + gx, y2 + gy - UP_PX], (crop.w, crop.h), pad_px=int(S["pad_px"]))
    px, py = m.point_crop
    pt = [min(crop.w - 1, max(0, int(round(px)))), min(crop.h - 1, max(0, int(round(py))))]
    return {
        "b": {"box": b},
        "c": {"box": [grown.left, grown.top, grown.right, grown.bottom]},
        "cu": {"box": [up.left, up.top, up.right, up.bottom]},
        "d": {"point": pt},
    }


def requests() -> list[dict]:
    truth, marks, kfs = load()
    out = []
    for kf_id in sorted({v["kf_id"] for v in truth.values()}):
        kf = kfs[kf_id]
        objs = []
        for mid in sorted(k for k, v in truth.items() if v["kf_id"] == kf_id):
            for var, p in prompts(marks[mid], kf).items():
                objs.append({"mark_id": mid, "variant": var, **p})
        for i in range(0, len(objs), MAX_OBJECTS):
            chunk = [dict(o, object_id=j + 1) for j, o in enumerate(objs[i : i + MAX_OBJECTS])]
            out.append({"key": f"{kf_id}#{i // MAX_OBJECTS}", "kf_id": kf_id, "path": str(RUN / kf.crop.path), "w": kf.crop.w, "h": kf.crop.h, "objects": chunk})
    return out


def concept(req: dict) -> str:
    objs = []
    for o in req["objects"]:
        fp = FramePrompt(frame=0, box=PromptBox(*o["box"])) if "box" in o else FramePrompt(frame=0, points=[PromptPoint(*o["point"])])
        objs.append(PromptObject(id=o["object_id"], prompts=[fp]))
    return sam_client.object_prompt(objs, (req["w"], req["h"]))


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


async def ask() -> None:
    stored = json.loads(ANSWERS.read_text()) if ANSWERS.exists() else {}
    jobs = [r for r in requests() if r["key"] not in stored or stored[r["key"]]["answer"]["status"] != "completed"]
    spent = day_spent()
    print(f"Meta day so far ${spent:.4f}; {len(jobs)} requests wanted (${len(jobs) * PER_REQUEST:.4f}); stop at ${DAY_STOP_USD:.2f}")
    if spent + len(jobs) * PER_REQUEST > DAY_STOP_USD:
        print("would cross the stop; not asking")
        return
    if os.environ.get("LIVE") != "1":
        return
    sem = asyncio.Semaphore(6)

    async def one(req: dict) -> None:
        text = concept(req)
        async with sem:
            try:
                run = await sam_client.segment_image(Path(req["path"]), text, run_id=RUN_ID, tag="box_refine_hand")
                ans = sam_answer_from_run(run)
            except sam_client.SamRequestError as exc:
                ans = SamAnswer(concept=text, status="failed", error=str(exc)[:300])
        stored[req["key"]] = {"kf_id": req["kf_id"], "objects": req["objects"], "answer": ans.model_dump(mode="json")}

    await asyncio.gather(*(one(r) for r in jobs))
    ANSWERS.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n")
    print(f"asked {len(jobs)}; Meta day now ${day_spent():.4f}")


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


def tight_box(det, crop) -> tuple[list[float] | None, int, float]:
    import numpy as np

    mask = privacy.detection_mask(_detection(det), (crop.w, crop.h))
    px = int(mask.sum())
    if px == 0:
        return None, 0, 0.0
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], px, px / float(crop.w * crop.h)


def gate_wide(m: Mark, prompt: list[float], tight, px: int, frac: float, s: dict) -> tuple[bool, str]:
    """The product's rules with the prompt as the frame: a mask that fills the prompt or sits off it is not a mark."""
    spark = m.bbox_crop
    if tight is None or px < int(s["min_mask_px"]):
        return False, "empty"
    if frac > float(s["max_crop_frac"]):
        return False, "too big for the crop"
    area = lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])  # noqa: E731
    within = tight[0] >= prompt[0] - 1 and tight[1] >= prompt[1] - 1 and tight[2] <= prompt[2] + 1 and tight[3] <= prompt[3] + 1
    if within and area(tight) >= float(s["fills_box_min"]) * area(prompt):
        return False, "fills the prompt"
    if area(tight) / max(area(spark), 1e-9) > float(s["max_area_ratio"]):
        return False, "over max_area_ratio x Spark's box"
    inter = area([max(tight[0], prompt[0]), max(tight[1], prompt[1]), min(tight[2], prompt[2]), min(tight[3], prompt[3])])
    if inter / max(area(tight), 1e-9) < float(s["min_inside"]):
        return False, "off the prompt"
    return True, "sam"


def score() -> dict:
    truth, marks, kfs = load()
    stored = json.loads(ANSWERS.read_text())
    got: dict[tuple[str, str], tuple] = {}
    for req in stored.values():
        ans = SamAnswer.model_validate(req["answer"])
        dets = {d.object_id: d for d in ans.detections}
        for o in req["objects"]:
            got[(o["mark_id"], o["variant"])] = (ans, dets.get(str(o["object_id"])), o)
    rows = []
    for mid, t in sorted(truth.items()):
        m, kf = marks[mid], kfs[truth[mid]["kf_id"]]
        crop = kf.crop
        row = {"mark_id": mid, "status": t["status"], "size_class": t["size_class"], "hand": t["box"], "a": {"box": list(m.bbox_crop), "used": "spark"}}
        for var in VARIANTS:
            ans, det, o = got[(mid, var)]
            tight, px, frac = tight_box(det, crop) if det is not None else (None, 0, 0.0)
            if var == "b":
                r = box_refine.decide(m, ans, det, crop, S)
                used = r.used == "sam"
                why = "sam" if used else r.note
            else:
                prompt = o.get("box") or list(m.bbox_crop)
                used, why = gate_wide(m, [float(v) for v in prompt], tight, px, frac, S)
            row[var] = {"raw": tight, "c": det.c if det is not None else None, "mask_px": px, "used": "sam" if used else "spark", "why": why, "box": tight if used else list(m.bbox_crop)}
        rows.append(row)

    def stats(var: str, key: str) -> dict:
        moved = [r for r in rows if r["status"] == "moved"]
        boxes = [(r[var][key], r["hand"]) for r in moved]
        have = [(b, h) for b, h in boxes if b is not None]
        ious = [iou(b, h) for b, h in boxes if b is not None] + [0.0] * (len(boxes) - len(have))
        ds = [dist(b, h) for b, h in have]
        return {
            "n": len(boxes),
            "with_box": len(have),
            "median_iou": round(st.median(ious), 3),
            "median_centre_px": round(st.median(ds), 1) if ds else None,
            "on_mark": sum(on_mark(b, h) for b, h in have),
            "iou_ge_05": sum(iou(b, h) >= 0.5 for b, h in have),
            "sam_used": sum(r[var].get("used") == "sam" for r in moved),
        }

    table = {"a": stats("a", "box")}
    for var in VARIANTS:
        table[var] = stats(var, "box")
        table[var + "_raw"] = stats(var, "raw")
    not_mark = {var: [(r["mark_id"], r[var]["used"], r[var]["why"], r[var]["c"]) for r in rows if r["status"] == "not_mark"] for var in VARIANTS}
    out = {"run": "pairC_core3", "up_px": UP_PX, "patch_sent_px": PATCH_SENT_PX, "table": table, "not_mark": not_mark, "rows": rows}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'variant':8} {'n':>3} {'box':>4} {'IoU':>6} {'ctr px':>7} {'on':>4} {'IoU>=.5':>8} {'sam':>4}")
    for k, v in table.items():
        print(f"{k:8} {v['n']:>3} {v['with_box']:>4} {v['median_iou']:>6} {str(v['median_centre_px']):>7} {v['on_mark']:>4} {v['iou_ge_05']:>8} {v['sam_used']:>4}")
    for var, lst in not_mark.items():
        print("not_mark", var, [(u, w) for _, u, w, _ in lst])
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "ask":
        asyncio.run(ask())
    else:
        score()
