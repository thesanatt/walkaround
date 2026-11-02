"""Zoom refine of Spark's mark boxes against Sanat's hand-corrected boxes on pairC_core3 (eval/box_corrections/pairC_core3.json).

For each of his 53 corrected Spark box marks (48 moved, 5 not a mark), on the crop the product sent:
  w1  a window around Spark's box: short side max(96 px, 3 Spark patches, Spark's box + one patch a side), long side
      to 16:9, centred on Spark's box and shifted (then clamped) to stay inside the crop
  w2  the same window doubled on each axis, same centre
Each window is enlarged to 1280x720 (the size most product crops are sent at, and the canvas R11.2 measured Spark best
on) and sent to Muse Spark with the product's marks_v1 prompt (effort low, fixtures sentence on as in today's
thresholds, no landmarks). The zoom box is the returned box whose centre is nearest Spark's box centre, mapped back to
crop pixels per axis; Spark's type stays on the mark.

  LIVE=1 python study.py ask [w1|w2]   # asks only what answers.json lacks; spend guard on the ET day
  python study.py score                # no call; writes study.json and prints the table
  LIVE=1 python study.py ask_sam       # the product's SAM leaf on each w1 recentred box (sam_answers.json)
  python study.py score_sam            # no call; adds then_sam to study.json
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

import numpy as np
from PIL import Image

from walkaround import geometry, spark_client
from walkaround.contracts import Keyframe, Mark, SparkRequestRecord

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN = REPO / "runs" / "pairC_core3"
TRUTH = REPO / "eval" / "box_corrections" / "pairC_core3.json"
ANSWERS = HERE / "answers.json"
OUT = HERE / "study.json"
RUN_ID = "zoom_refine_hand_1101"
PER_CALL = 0.007  # a little over the median marks call (0.0047) for the 1280x720 canvas
DAY_STOP_USD = float(os.environ.get("DAY_STOP_USD", "49.50"))
PATCH_SENT_PX = 28
MIN_PX = 96
OUT_W, OUT_H = 1280, 720
WINDOWS = {"w1": 1.0, "w2": 2.0}


def load() -> tuple[dict, dict[str, Mark], dict[str, Keyframe]]:
    truth = json.loads(TRUTH.read_text())["marks"]
    marks = {m["mark_id"]: Mark.model_validate(m) for m in json.loads((RUN / "derived" / "marks.json").read_text()) if m["mark_id"] in truth}
    kfs = {k["kf_id"]: Keyframe.model_validate(k) for k in json.loads((RUN / "derived" / "keyframes.json").read_text())}
    return truth, marks, kfs


def patch_px(crop) -> tuple[float, float]:
    sw, sh = spark_client.sent_size(crop.w, crop.h)
    return PATCH_SENT_PX * crop.w / sw, PATCH_SENT_PX * crop.h / sh


def window(box: list[float], crop, k: float) -> list[int]:
    gx, gy = patch_px(crop)
    x1, y1, x2, y2 = box
    h = max(MIN_PX, 3 * gy, (y2 - y1) + 2 * gy)
    w = max(MIN_PX, 3 * gx, (x2 - x1) + 2 * gx)
    if w < h * OUT_W / OUT_H:
        w = h * OUT_W / OUT_H
    else:
        h = w * OUT_H / OUT_W
    w, h = min(crop.w, w * k), min(crop.h, h * k)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(0.0, cx - w / 2), crop.w - w)
    top = min(max(0.0, cy - h / 2), crop.h - h)
    return [int(round(left)), int(round(top)), int(round(left + w)), int(round(top + h))]


def zoom_image(crop_rgb: np.ndarray, win: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = win
    img = Image.fromarray(np.ascontiguousarray(crop_rgb[y1:y2, x1:x2, :3]))
    return np.asarray(img.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS))


def jobs() -> list[dict]:
    truth, marks, kfs = load()
    out = []
    for mid in sorted(truth):
        m = marks[mid]
        kf = kfs[truth[mid]["kf_id"]]
        for name, k in WINDOWS.items():
            out.append({"key": f"{mid}#{name}", "mark_id": mid, "window_name": name, "kf_id": kf.kf_id, "panel_id": kf.panel_id.value, "path": str(RUN / kf.crop.path), "window": window(list(m.bbox_crop), kf.crop, k)})
    return out


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


async def ask(which: list[str]) -> None:
    from walkaround.contracts import PanelId, load_thresholds
    from walkaround.marks import _read_rgb
    from walkaround.match import fixtures_not_damage, seams_not_marks

    th = load_thresholds()
    stored = json.loads(ANSWERS.read_text()) if ANSWERS.exists() else {}
    todo = [j for j in jobs() if j["window_name"] in which and (j["key"] not in stored or not stored[j["key"]]["record"].get("valid"))]
    spent = day_spent()
    print(f"Meta day so far ${spent:.4f}; {len(todo)} calls wanted (about ${len(todo) * PER_CALL:.2f}); stop at ${DAY_STOP_USD:.2f}")
    if spent + len(todo) * PER_CALL > DAY_STOP_USD:
        print("would cross the stop; not asking")
        return
    if os.environ.get("LIVE") != "1":
        return
    sem = asyncio.Semaphore(6)

    async def one(j: dict) -> None:
        x1, y1, x2, y2 = j["window"]
        rgb = zoom_image(_read_rgb(Path(j["path"])), j["window"])
        async with sem:
            res = await asyncio.to_thread(
                spark_client.find_marks, rgb, PanelId(j["panel_id"]), landmarks=[], effort=str(th["spark"]["effort"]), run_id=RUN_ID,
                fixtures=fixtures_not_damage(th), seams=seams_not_marks(th),
            )
        stored[j["key"]] = {**{k: j[k] for k in ("mark_id", "window_name", "kf_id", "panel_id", "window")}, "sent": [res.sent_width, res.sent_height], "record": res.record.model_dump(mode="json")}

    await asyncio.gather(*(one(j) for j in todo))
    ANSWERS.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n")
    print(f"asked {len(todo)}; Meta day now ${day_spent():.4f}")


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


def zoom_boxes(entry: dict) -> list[dict]:
    """Every box of one zoom answer in crop pixels, with the same half-open rule as marks.map_spark_mark."""
    rec = SparkRequestRecord.model_validate(entry["record"])
    res = spark_client.marks_result_from_record(rec, orig_width=OUT_W, orig_height=OUT_H, landmarks=[])
    if res.answer is None:
        return []
    x0, y0, x1, y1 = entry["window"]
    ww, wh = x1 - x0, y1 - y0
    sw, sh = res.sent_width, res.sent_height
    out = []
    for sm in res.answer.marks:
        a1, b1 = spark_client.grid_to_pixels(sm.bbox.x_min, sm.bbox.y_min, sw, sh)
        a2, b2 = spark_client.grid_to_pixels(sm.bbox.x_max, sm.bbox.y_max, sw, sh)
        c1 = geometry.sent_to_crop((a1, b1), (sw, sh), (ww, wh))
        c2 = geometry.sent_to_crop((max(a2, a1) + 1, max(b2, b1) + 1), (sw, sh), (ww, wh))
        out.append({"box": [round(c1[0] + x0, 2), round(c1[1] + y0, 2), round(c2[0] + x0, 2), round(c2[1] + y0, 2)], "type": sm.type.value, "confidence": sm.confidence.value, "note": sm.note})
    return out


def recentre(spark: list[float], box: list[float], wh: tuple[int, int] | None = None) -> list[float]:
    """Spark's box, same size, moved so its centre is the zoom box's centre, then shifted back inside the crop."""
    (sx, sy), (zx, zy) = centre(spark), centre(box)
    b = [spark[0] + zx - sx, spark[1] + zy - sy, spark[2] + zx - sx, spark[3] + zy - sy]
    if wh is not None:
        dx = -min(0.0, b[0]) - max(0.0, b[2] - wh[0])
        dy = -min(0.0, b[1]) - max(0.0, b[3] - wh[1])
        b = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
    return b


def nearest(boxes: list[dict], spark: list[float]) -> dict | None:
    return min(boxes, key=lambda b: dist(b["box"], spark)) if boxes else None


def score() -> dict:
    truth, marks, kfs = load()
    stored = json.loads(ANSWERS.read_text())
    rows = []
    for mid, t in sorted(truth.items()):
        spark = [float(v) for v in marks[mid].bbox_crop]
        crop = kfs[t["kf_id"]].crop
        row = {"mark_id": mid, "status": t["status"], "size_class": t["size_class"], "type": marks[mid].type.value, "hand": t["box"], "spark": spark, "crop_wh": [crop.w, crop.h]}
        for name in WINDOWS:
            e = stored.get(f"{mid}#{name}")
            boxes = zoom_boxes(e) if e else []
            pick = nearest(boxes, spark)
            row[name] = {"window": e["window"] if e else None, "n_boxes": len(boxes), "valid": bool(e and e["record"].get("valid")), "pick": pick, "boxes": boxes}
        a, b = row["w1"]["pick"], row["w2"]["pick"]
        agree = a is not None and b is not None and dist(a["box"], b["box"]) <= max(12.0, 0.5 * max(a["box"][2] - a["box"][0], a["box"][3] - a["box"][1]))
        row["agree"] = agree
        rows.append(row)

    def box_of(r: dict, arm: str) -> tuple[list[float] | None, bool]:
        """(box, zoom used) for one arm; arms fall back to Spark's box when the zoom gave none."""
        if arm == "spark":
            return r["spark"], False
        if arm in WINDOWS:
            p = r[arm]["pick"]
            return (p["box"], True) if p else (r["spark"], False)
        if arm == "w1_raw":
            p = r["w1"]["pick"]
            return (p["box"] if p else None), p is not None
        if arm == "w1_if_agree":
            return (r["w1"]["pick"]["box"], True) if r["agree"] else (r["spark"], False)
        if arm == "w2_then_w1":
            p = r["w1"]["pick"] or r["w2"]["pick"]
            return (p["box"], True) if p else (r["spark"], False)
        if arm.endswith("_centre"):
            src = arm[: -len("_centre")]
            box, used = box_of(r, src)
            return (recentre(r["spark"], box, tuple(r["crop_wh"])), used) if used else (r["spark"], False)
        raise KeyError(arm)

    def stats(arm: str) -> dict:
        moved = [r for r in rows if r["status"] == "moved"]
        pairs = [(box_of(r, arm), r["hand"]) for r in moved]
        have = [(b, h) for (b, _u), h in pairs if b is not None]
        ious = [iou(b, h) for b, h in have] + [0.0] * (len(pairs) - len(have))
        ds = [dist(b, h) for b, h in have]
        return {
            "n": len(pairs),
            "with_box": len(have),
            "median_iou": round(st.median(ious), 3),
            "mean_iou": round(st.mean(ious), 3),
            "median_centre_px": round(st.median(ds), 1) if ds else None,
            "on_mark": sum(on_mark(b, h) for b, h in have),
            "iou_ge_05": sum(iou(b, h) >= 0.5 for b, h in have),
            "zoom_used": sum(u for (_b, u), _h in pairs),
        }

    arms = ["spark", "w1", "w2", "w1_raw", "w1_if_agree", "w2_then_w1", "w1_centre", "w2_centre", "w1_if_agree_centre"]
    table = {arm: stats(arm) for arm in arms}
    not_mark = [
        {"mark_id": r["mark_id"], **{n: {"n_boxes": r[n]["n_boxes"], "pick_type": (r[n]["pick"] or {}).get("type"), "pick_conf": (r[n]["pick"] or {}).get("confidence"), "on_spark": bool(r[n]["pick"]) and iou(r[n]["pick"]["box"], r["spark"]) > 0} for n in WINDOWS}, "agree": r["agree"]}
        for r in rows
        if r["status"] == "not_mark"
    ]
    moved = [r for r in rows if r["status"] == "moved"]
    pick_types = {n: {} for n in WINDOWS}
    for r in moved:
        for n in WINDOWS:
            ty = (r[n]["pick"] or {}).get("type", "none")
            pick_types[n][ty] = pick_types[n].get(ty, 0) + 1
    out = {"run": "pairC_core3", "min_px": MIN_PX, "out": [OUT_W, OUT_H], "table": table, "not_mark": not_mark, "moved_pick_types": pick_types, "rows": rows}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'arm':20} {'n':>3} {'box':>4} {'IoU':>6} {'mean':>6} {'ctr px':>7} {'on':>4} {'IoU>=.5':>8} {'zoom':>5}")
    for k, v in table.items():
        print(f"{k:20} {v['n']:>3} {v['with_box']:>4} {v['median_iou']:>6} {v['mean_iou']:>6} {str(v['median_centre_px']):>7} {v['on_mark']:>4} {v['iou_ge_05']:>8} {v['zoom_used']:>5}")
    print("moved pick types", pick_types)
    for nm in not_mark:
        print("not_mark", nm)
    return out


# zoom then SAM (the product's SAM leaf on the moved box) -------------------------------------------------------------

SAM_ANSWERS = HERE / "sam_answers.json"
SAM_PER_REQUEST = 0.0025


def moved_marks() -> tuple[dict[str, Mark], dict[str, Keyframe]]:
    """Each corrected mark with Spark's box moved to the w1 zoom centre (marks the zoom found nothing for are left out)."""
    truth, marks, kfs = load()
    stored = json.loads(ANSWERS.read_text())
    out = {}
    for mid in sorted(truth):
        m = marks[mid]
        pick = nearest(zoom_boxes(stored[f"{mid}#w1"]), [float(v) for v in m.bbox_crop])
        if pick is None:
            continue
        crop = kfs[m.kf_id].crop
        b = recentre([float(v) for v in m.bbox_crop], pick["box"])
        dx = -min(0.0, b[0]) - max(0.0, b[2] - crop.w)
        dy = -min(0.0, b[1]) - max(0.0, b[3] - crop.h)
        b = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
        out[mid] = m.model_copy(update={"bbox_crop": b, "point_crop": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]})
    return out, kfs


async def ask_sam() -> None:
    from walkaround import box_refine
    from walkaround.contracts import load_thresholds

    s = box_refine.settings(load_thresholds())
    moved, kfs = moved_marks()
    stored = json.loads(SAM_ANSWERS.read_text()) if SAM_ANSWERS.exists() else []
    if stored:
        print("sam answers already stored")
        return
    jobs_ = []
    for kf_id in sorted({m.kf_id for m in moved.values()}):
        kf = kfs[kf_id]
        for chunk in box_refine.plan([m for m in moved.values() if m.kf_id == kf_id], kf, s):
            jobs_.append((kf, chunk))
    spent = day_spent()
    print(f"Meta day so far ${spent:.4f}; {len(jobs_)} SAM requests wanted (${len(jobs_) * SAM_PER_REQUEST:.4f}); stop at ${DAY_STOP_USD:.2f}")
    if spent + len(jobs_) * SAM_PER_REQUEST > DAY_STOP_USD or os.environ.get("LIVE") != "1":
        return
    sem = asyncio.Semaphore(6)

    async def one(kf, objs):
        async with sem:
            return await box_refine.ask(RUN_ID, kf, RUN / kf.crop.path, objs)

    asks = await asyncio.gather(*(one(kf, objs) for kf, objs in jobs_))
    SAM_ANSWERS.write_text(json.dumps([a.model_dump(mode="json") for a in asks], indent=1) + "\n")
    print(f"asked {len(asks)}; Meta day now ${day_spent():.4f}")


def score_sam() -> dict:
    from walkaround import box_refine
    from walkaround.contracts import load_thresholds

    th = load_thresholds()
    s = box_refine.settings(th)
    truth, _marks, _kfs = load()
    moved, kfs = moved_marks()
    index = box_refine.Index([box_refine.BoxAsk.model_validate(a) for a in json.loads(SAM_ANSWERS.read_text())])
    study = json.loads(OUT.read_text())
    rows = {r["mark_id"]: r for r in study["rows"]}
    res = {"zoom_centre": [], "zoom_centre_sam": []}
    used = 0
    for mid, t in sorted(truth.items()):
        if t["status"] != "moved":
            continue
        if mid in moved:
            m, kf = moved[mid], kfs[moved[mid].kf_id]
            zb = list(m.bbox_crop)
            hit = index.get(kf.crop, box_refine.prompt_of(m, kf.crop, s))
            ans, det = hit if hit else (None, None)
            r = box_refine.decide(m, ans, det, kf.crop, s)
            sb = list(r.sam_bbox_crop) if r.used == "sam" else zb
            used += r.used == "sam"
        else:
            zb = sb = rows[mid]["spark"]
        res["zoom_centre"].append((zb, t["box"]))
        res["zoom_centre_sam"].append((sb, t["box"]))
    table = {}
    for k, pairs in res.items():
        table[k] = {"n": len(pairs), "median_iou": round(st.median(iou(b, h) for b, h in pairs), 3), "mean_iou": round(st.mean(iou(b, h) for b, h in pairs), 3), "median_centre_px": round(st.median(dist(b, h) for b, h in pairs), 1),
                    "on_mark": sum(on_mark(b, h) for b, h in pairs), "iou_ge_05": sum(iou(b, h) >= 0.5 for b, h in pairs)}
    table["zoom_centre_sam"]["sam_used"] = used
    study["then_sam"] = table
    OUT.write_text(json.dumps(study, indent=1) + "\n")
    for k, v in table.items():
        print(k, v)
    return table


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "ask":
        asyncio.run(ask(sys.argv[2:] or list(WINDOWS)))
    elif cmd == "ask_sam":
        asyncio.run(ask_sam())
    elif cmd == "score_sam":
        score_sam()
    else:
        score()
