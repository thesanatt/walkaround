"""Effort study on stored crops (11/1): marks_v1 at reasoning effort low, medium and high, and one variant, a second
pass on an about 2x zoom of each half of the crop (unioned with the low pass), on

- synth_pair2_r9: one stored keyframe crop per (walk, panel) among the crops that hold a planted mark or a false box of
  the run (rule in select_synth), scored against the render's ground truth (synthetic/out/pair2/ground_truth.json,
  read only). A box hits a mark when its centre falls inside the mark's box grown to 1.5 times its width and height,
  or IoU > 0.1 (the synthetic scorer's rule). A false box hits no damage mark (a box on the sticker control is false).
- pair C (uCO3D, CC BY 4.0): D1's curated crops in runs/d1/uco3d_pairC/spark_redo/ with their hand-measured target
  points (runs/d1/uco3d_pairC/e6_redo_crops.json). A target is hit when it lies inside a box grown by a quarter of its
  width and height (at least 10 px) or within 0.04 of the crop's long edge of the box centre. A box that hits no
  target is "unmatched": the targets are the known marks, not every mark, so an unmatched box may be a real chip.

The prompt and schema are the product's (marks_v1, fixtures sentence as synth_pair2_r9's thresholds say, seams off).
Answers append to answers.jsonl beside this file as they arrive (resumable); `score` writes study.json and prints the
table. Spends Meta credit with LIVE=1: the study stops before its own ledger rows (run id marks_effort_1101) pass
STUDY_CAP_USD or the Nov 1 total passes DAY_STOP_USD.

  LIVE=1 python study.py pilot | batch ARM EFFORT REP | score | select
"""

import hashlib
import json
import math
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from walkaround import config, marks, match, paths, spark_client
from walkaround.contracts import InputRef, Keyframe, Mark, PanelId, read_json_list

HERE = Path(__file__).resolve().parent
RUN = "synth_pair2_r9"
GT_PATH = HERE.parents[3] / "synthetic" / "out" / "pair2" / "ground_truth.json"
D1_DIR = paths.REPO_DIR / "runs" / "d1" / "uco3d_pairC"
ANSWERS = HERE / "answers.jsonl"
OUT = HERE / "study.json"
LEDGER_RUN_ID = "marks_effort_1101"
STUDY_CAP_USD = float(os.environ.get("STUDY_CAP_USD", "3.00"))
DAY_STOP_USD = float(os.environ.get("DAY_STOP_USD", "49.07"))  # 46.07 at the start plus 3.00
DAY = "2026-11-01"
DAMAGE_TYPES = {"scratch", "scuff", "chip", "dent", "crack"}
_lock = threading.Lock()


# crops and truth

def gt_in_crop(gt: dict, kf: Keyframe) -> list[dict]:
    f = int(round(kf.pts * gt["fps"]))
    c = kf.crop
    x0, y0, x1, y1 = c.dx, c.dy, c.dx + c.w / c.scale, c.dy + c.h / c.scale
    out = []
    for m in gt["marks"]:
        rows = m["frames"].get(kf.role) or []
        row = rows[f] if f < len(rows) and rows[f]["frame"] == f else next((r for r in rows if r["frame"] == f), None)
        if not row or not row.get("visible") or not row.get("visible_bbox_xyxy"):
            continue
        b = row["visible_bbox_xyxy"]
        if b[2] > x0 and b[0] < x1 and b[3] > y0 and b[1] < y1:
            long_crop = max(b[2] - b[0], b[3] - b[1]) * c.scale
            out.append({"id": m["id"], "kind": m["kind"], "bbox_src": b, "long_crop_px": round(long_crop, 1), "damage": not m["id"].startswith("control")})
    return out


def hit(box: list[float], g: list[float], grow: float = 1.5) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    gw, gh = g[2] - g[0], g[3] - g[1]
    gx, gy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    if abs(cx - gx) <= grow / 2 * gw and abs(cy - gy) <= grow / 2 * gh:
        return True
    return iou(box, g) > 0.1


def hit_tolerant(box: list[float], g: list[float], scale: float) -> bool:
    """Sensitivity rule only: the mark's box grown by a quarter of its size or 12 crop px per side, whichever is more,
    so a box lying along a thin scratch counts; IoU > 0.1 still counts."""
    gx, gy = max(0.25 * (g[2] - g[0]), 12 / scale), max(0.25 * (g[3] - g[1]), 12 / scale)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (g[0] - gx <= cx <= g[2] + gx and g[1] - gy <= cy <= g[3] + gy) or iou(box, g) > 0.1


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select_synth(gt: dict) -> list[dict]:
    """Every stored crop of the run with a planted mark or a false box of the run, then one per (walk, panel): first a
    keyframe showing a planted damage mark at least 40 crop px long, then the most false boxes of the run, then the
    longest planted mark, then the lowest keyframe id."""
    kfs = [k for k in read_json_list(paths.keyframes_json(RUN), Keyframe) if k.crop is not None and k.crop.path]
    run_marks = read_json_list(paths.marks_json(RUN), Mark)
    best: dict[tuple[str, str], tuple] = {}
    for k in kfs:
        truth = gt_in_crop(gt, k)
        boxes = [m for m in run_marks if m.kf_id == k.kf_id]
        false = [m for m in boxes if not any(hit(m.bbox_src, g["bbox_src"]) for g in truth if g["damage"])]
        if not truth and not false:
            continue
        biggest = max((g["long_crop_px"] for g in truth if g["damage"]), default=0.0)
        key = (k.role, str(PanelId(k.panel_id).value))
        rank = (biggest >= 40, len(false), biggest, [-ord(ch) for ch in k.kf_id])
        if key not in best or rank > best[key][0]:
            best[key] = (rank, k, truth, boxes)
    out = []
    for (_role, _panel), (_rank, k, truth, boxes) in sorted(best.items()):
        out.append({"crop_id": k.kf_id, "set": "synth", "panel": str(PanelId(k.panel_id).value), "path": str(paths.run_dir(RUN) / k.crop.path), "kf": k, "truth": truth,
                    "run_boxes_src": [m.bbox_src for m in boxes]})
    return out


def select_pairc() -> list[dict]:
    specs = json.loads((D1_DIR / "e6_redo_crops.json").read_text())
    return [{"crop_id": "pairC_" + s["name"], "set": "real", "panel": s["panel"], "path": str(D1_DIR / "spark_redo" / f"crop_{s['name']}.png"), "targets": s["targets"]} for s in specs]


def crops_all() -> list[dict]:
    gt = json.loads(GT_PATH.read_text())
    return [*select_synth(gt), *select_pairc()]


# spend guard

def ledger_totals() -> tuple[float, float]:
    day = study = 0.0
    with open(paths.REPO_DIR / "runs" / "ledger.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            if str(row.get("ts", "")).startswith(DAY):
                day += float(row.get("usd") or 0)
            if row.get("run_id") == LEDGER_RUN_ID:
                study += float(row.get("usd") or 0)
    return day, study


class Guard:
    def __init__(self, per_call_estimate: float):
        self.day, self.study = ledger_totals()
        self.est = per_call_estimate
        self.pending = 0.0

    def take(self) -> bool:
        with _lock:
            if self.study + self.pending + self.est > STUDY_CAP_USD or self.day + self.pending + self.est > DAY_STOP_USD:
                return False
            self.pending += self.est
            return True

    def settle(self, dollars: float) -> None:
        with _lock:
            self.pending -= self.est
            self.study += dollars
            self.day += dollars


# calls

def read_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def halves(rgb: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    """Two halves along the long axis, each 55% of it (a 10% shared band), scaled up 2x: (image, x offset, y offset)."""
    h, w = rgb.shape[:2]
    out = []
    if w >= h:
        span = int(round(0.55 * w))
        for x0 in (0, w - span):
            out.append((rgb[:, x0:x0 + span], x0, 0))
    else:
        span = int(round(0.55 * h))
        for y0 in (0, h - span):
            out.append((rgb[y0:y0 + span, :], 0, y0))
    return [(cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC), x0, y0) for img, x0, y0 in out]


def boxes_of(res, w: int, h: int, zoom: float = 1.0, ox: int = 0, oy: int = 0) -> list[dict]:
    out = []
    for m in (res.answer.marks if res.answer else []):
        x1, y1, x2, y2 = spark_client.grid_box_to_pixels(m.bbox, w, h)
        out.append({"type": m.type.value, "conf": m.confidence.value, "size": m.size_class.value, "note": m.note[:80],
                    "box": [round(x1 / zoom + ox, 1), round(y1 / zoom + oy, 1), round(x2 / zoom + ox, 1), round(y2 / zoom + oy, 1)]})
    return out


def ask(crop: dict, arm: str, effort: str, rep: int, th: dict, guard: Guard) -> list[dict]:
    rgb = read_rgb(crop["path"])
    panel = PanelId(crop["panel"])
    rel = os.path.relpath(crop["path"], paths.REPO_DIR)
    ref = InputRef(path=rel, sha256=hashlib.sha256(Path(crop["path"]).read_bytes()).hexdigest())
    images = [(rgb, 1.0, 0, 0, None)] if arm == "base" else [(img, 2.0, x0, y0, i) for i, (img, x0, y0) in enumerate(halves(rgb))]
    rows = []
    for img, zoom, ox, oy, half in images:
        if not guard.take():
            rows.append({"crop_id": crop["crop_id"], "arm": arm, "effort": effort, "rep": rep, "half": half, "skipped": "budget"})
            continue
        dollars = 0.0
        try:
            res = spark_client.find_marks(img, panel, landmarks=marks.landmarks_for(panel, th), effort=effort, run_id=LEDGER_RUN_ID, input_ref=ref,
                                          fixtures=match.fixtures_not_damage(th), seams=match.seams_not_marks(th))
            dollars = sum(a.dollars for a in (res.attempts or [res.record]))
            ih, iw = img.shape[:2]
            row = {"crop_id": crop["crop_id"], "arm": arm, "effort": effort, "rep": rep, "half": half, "valid": res.answer is not None, "truncated": bool(res.truncated),
                   "dollars": round(dollars, 6), "attempts": len(res.attempts or [res.record]), "request_id": res.record.request_id,
                   "boxes": boxes_of(res, iw, ih, zoom, ox, oy)}
        except Exception as exc:  # recorded, never raised past the batch; no key ever appears in these messages
            row = {"crop_id": crop["crop_id"], "arm": arm, "effort": effort, "rep": rep, "half": half, "error": config.redact(str(exc))[:200]}
        finally:
            guard.settle(dollars)
        rows.append(row)
        with _lock, open(ANSWERS, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return rows


def done_keys() -> set:
    if not ANSWERS.exists():
        return set()
    keys = set()
    for line in ANSWERS.read_text().splitlines():
        r = json.loads(line)
        if "boxes" in r:
            keys.add((r["crop_id"], r["arm"], r["effort"], r["rep"], r["half"]))
    return keys


def run_batch(crops: list[dict], arm: str, effort: str, rep: int, per_call: float) -> None:
    th = json.loads((paths.run_dir(RUN) / "run.json").read_text())["thresholds"]
    done = done_keys()
    todo = [c for c in crops if not all((c["crop_id"], arm, effort, rep, h) in done for h in ([None] if arm == "base" else [0, 1]))]
    guard = Guard(per_call)
    print(f"batch {arm} {effort} rep{rep}: {len(todo)} crops; Nov 1 total ${guard.day:.2f}, study ${guard.study:.3f}")
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda c: ask(c, arm, effort, rep, th, guard), todo))
    day, study = ledger_totals()
    print(f"  after: Nov 1 total ${day:.2f}, study ${study:.3f}")


# scoring

def load_answers() -> dict:
    by = {}
    for line in ANSWERS.read_text().splitlines():
        r = json.loads(line)
        if "boxes" in r:
            by[(r["crop_id"], r["arm"], r["effort"], r["rep"], r["half"])] = r
    return by


def dedup_union(base: list[dict], extra: list[dict]) -> list[dict]:
    kept = list(base)
    for b in extra:
        cx, cy = (b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2
        if any(iou(b["box"], k["box"]) > 0.3 or (k["box"][0] <= cx <= k["box"][2] and k["box"][1] <= cy <= k["box"][3]) for k in kept):
            continue
        kept.append(b)
    return kept


def to_src(kf: Keyframe, box: list[float]) -> list[float]:
    c = kf.crop
    return [c.dx + box[0] / c.scale, c.dy + box[1] / c.scale, c.dx + box[2] / c.scale, c.dy + box[3] / c.scale]


def target_hit(t: list[float], box: list[float], long_edge: int) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    gx, gy = max(10.0, 0.25 * w), max(10.0, 0.25 * h)
    if box[0] - gx <= t[0] <= box[2] + gx and box[1] - gy <= t[1] <= box[3] + gy:
        return True
    return math.dist(t, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)) <= 0.04 * long_edge


def score_answer(crop: dict, boxes: list[dict], tolerant: bool = False) -> dict:
    if crop["set"] == "synth":
        kf = crop["kf"]
        hit = (lambda b, g, grow=1.5: hit_tolerant(b, g, kf.crop.scale) if grow == 1.5 else globals()["hit"](b, g, grow)) if tolerant else globals()["hit"]
        dmg = [g for g in crop["truth"] if g["damage"]]
        found, false, sticker, beside, ious, offs, false_dmg = set(), 0, 0, 0, [], [], 0
        for b in boxes:
            src = to_src(kf, b["box"])
            ids = [g for g in dmg if hit(src, g["bbox_src"])]
            if ids:
                for g in ids:
                    found.add(g["id"])
                    ious.append(iou(src, g["bbox_src"]))
                    gx, gy = (g["bbox_src"][0] + g["bbox_src"][2]) / 2, (g["bbox_src"][1] + g["bbox_src"][3]) / 2
                    offs.append(math.dist(((src[0] + src[2]) / 2, (src[1] + src[3]) / 2), (gx, gy)) * kf.crop.scale)
                continue
            false += 1
            false_dmg += b["type"] in DAMAGE_TYPES
            if any(hit(src, g["bbox_src"]) for g in crop["truth"] if not g["damage"]):
                sticker += 1
            elif any(hit(src, g["bbox_src"], grow=4.0) for g in dmg):
                beside += 1
        large = {g["id"] for g in dmg if g["long_crop_px"] >= 40}
        return {"shown": len(dmg), "hits": len(found), "shown_large": len(large), "hits_large": len(found & large), "boxes": len(boxes), "false": false,
                "false_damage_typed": false_dmg, "sticker": sticker, "beside": beside, "ious": ious, "offs": offs, "found": sorted(found)}
    h, w = cv2.imread(crop["path"]).shape[:2]
    long_edge = max(h, w)
    found, unmatched, offs, lenient = set(), 0, [], set()
    for b in boxes:
        idx = [i for i, t in enumerate(crop["targets"]) if target_hit(t, b["box"], long_edge)]
        found.update(idx)
        unmatched += not idx
    for i, t in enumerate(crop["targets"]):
        centres = [((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2) for b in boxes]
        d = min((math.dist(t, c) for c in centres), default=None)
        if i in found and d is not None:
            offs.append(d)
        if d is not None and d <= 0.1 * long_edge:
            lenient.add(i)
    return {"shown": len(crop["targets"]), "hits": len(found), "lenient_hits": len(lenient), "boxes": len(boxes), "false": unmatched, "offs": offs, "found": sorted(found)}


def arm_rows(by: dict, crops: list[dict]) -> dict:
    """Per (arm label, rep): the boxes of each crop. Labels: low, medium, high (base), zoom_only (the two halves),
    low+zoom (the low base answer unioned with the zoom pass of the same repeat)."""
    out = {}
    reps = sorted({k[3] for k in by})
    for rep in reps:
        for effort in ("low", "medium", "high"):
            rows = {c["crop_id"]: by[(c["crop_id"], "base", effort, rep, None)]["boxes"] for c in crops if (c["crop_id"], "base", effort, rep, None) in by}
            if len(rows) == len(crops):
                out[(effort, rep)] = rows
        z = {}
        for c in crops:
            a, b = by.get((c["crop_id"], "zoom", "low", rep, 0)), by.get((c["crop_id"], "zoom", "low", rep, 1))
            if a and b:
                z[c["crop_id"]] = dedup_union(a["boxes"], b["boxes"])
        if len(z) == len(crops):
            out[("zoom_only", rep)] = z
            if ("low", rep) in out:
                out[("low+zoom", rep)] = {cid: dedup_union(out[("low", rep)][cid], z[cid]) for cid in z}
    return out


def score() -> dict:
    crops = crops_all()
    by = load_answers()
    arms = arm_rows(by, crops)
    result_views = {}
    for view in ("all_boxes", "damage_typed", "all_boxes_tolerant", "damage_typed_tolerant"):
        keep = (lambda b: b["type"] in DAMAGE_TYPES) if view.startswith("damage") else (lambda b: True)
        view_arms = {k: {cid: [b for b in boxes if keep(b)] for cid, boxes in v.items()} for k, v in arms.items()}
        result_views[view] = score_table(view_arms, by, crops, tolerant=view.endswith("tolerant"))
    result = {"run": RUN, "crops": [{k: v for k, v in c.items() if k not in ("kf", "path", "run_boxes_src")} | {"path": os.path.relpath(c["path"], paths.REPO_DIR)} for c in crops],
              "rules": __doc__.split("\n\n")[1], "run_stored_answer_on_synth": run_stored(crops), "views": result_views,
              "spend": dict(zip(("nov1_total", "study"), ledger_totals()))}
    OUT.write_text(json.dumps(result, indent=1))
    for view, table in result_views.items():
        print(f"\n{view}")
        print_table(table)
    print("run's stored answer on the synth crops:", result["run_stored_answer_on_synth"], "spend:", {k: round(v, 3) for k, v in result["spend"].items()})
    return result


def score_table(arms: dict, by: dict, crops: list[dict], tolerant: bool = False) -> dict:
    table = {}
    for label in ("low", "medium", "high", "zoom_only", "low+zoom"):
        reps = sorted(r for (l, r) in arms if l == label)
        if not reps:
            continue
        agg = {s: {"answers": 0, "shown": 0, "hits": 0, "shown_large": 0, "hits_large": 0, "boxes": 0, "false": 0, "false_damage_typed": 0, "sticker": 0, "beside": 0,
                   "lenient_hits": 0, "ious": [], "offs": []} for s in ("synth", "real")}
        per_crop = {}
        for rep in reps:
            for c in crops:
                s = score_answer(c, arms[(label, rep)][c["crop_id"]], tolerant)
                a = agg[c["set"]]
                a["answers"] += 1
                for k, v in s.items():
                    if k in ("found",):
                        continue
                    if isinstance(v, list):
                        a[k].extend(v)
                    else:
                        a[k] += v
                per_crop.setdefault(c["crop_id"], []).append(s["found"])
        for a in agg.values():
            a["median_iou"] = round(statistics.median(a["ious"]), 3) if a["ious"] else None
            a["median_offset_px"] = round(statistics.median(a["offs"]), 1) if a["offs"] else None
            del a["ious"], a["offs"]
        cost = [r.get("dollars", 0) for k, r in by.items() if (k[1] == "base" and k[2] == label) or (label == "zoom_only" and k[1] == "zoom")]
        table[label] = {"reps": reps, "synth": agg["synth"], "real": agg["real"], "per_crop_found": per_crop,
                        "dollars_per_call": round(statistics.mean(cost), 5) if cost else None, "calls": len(cost)}
    return table


def run_stored(crops: list[dict]) -> dict:
    """The run's own stored answers on the same synth crops (low, one answer each, made during the run)."""
    run_agg = {"shown": 0, "hits": 0, "false": 0, "boxes": 0}
    for c in crops:
        if c["set"] != "synth":
            continue
        kf = c["kf"]
        boxes = [{"type": "scratch", "box": [(b[0] - kf.crop.dx) * kf.crop.scale, (b[1] - kf.crop.dy) * kf.crop.scale, (b[2] - kf.crop.dx) * kf.crop.scale, (b[3] - kf.crop.dy) * kf.crop.scale]} for b in c["run_boxes_src"]]
        s = score_answer(c, boxes)
        for k in run_agg:
            run_agg[k] += s[k]
    return run_agg


def print_table(table: dict) -> None:
    print("| arm | answers synth/real | synth hits | synth large hits | synth false boxes | beside | sticker | IoU med | offset px | real hits | real lenient | real unmatched | $/call |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, t in table.items():
        s, r = t["synth"], t["real"]
        print(f"| {label} x{len(t['reps'])} | {s['answers']}/{r['answers']} | {s['hits']}/{s['shown']} | {s['hits_large']}/{s['shown_large']} | {s['false']} ({s['false_damage_typed']} damage-typed) | {s['beside']} | {s['sticker']} | {s['median_iou']} | {s['median_offset_px']} | {r['hits']}/{r['shown']} | {r['lenient_hits']}/{r['shown']} | {r['false']} | {t['dollars_per_call']} |")


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "select"
    if cmd == "select":
        for c in crops_all():
            if c["set"] == "synth":
                print(c["crop_id"], [(g["id"], g["long_crop_px"]) for g in c["truth"]], "run boxes", len(c["run_boxes_src"]))
            else:
                print(c["crop_id"], len(c["targets"]), "targets")
        return 0
    if cmd == "score":
        score()
        return 0
    if os.environ.get("LIVE") != "1":
        sys.stderr.write("set LIVE=1 to spend API credit\n")
        return 2
    config.load_env()
    crops = crops_all()
    if cmd == "pilot":
        pick = [c for c in crops if c["crop_id"] in ("pickup_right_front_door_kf1", "pickup_right_front_door_kf2")][:1] or crops[:1]
        for effort in ("low", "medium", "high"):
            run_batch(pick, "base", effort, 0, 0.05)
        return 0
    if cmd == "batch":
        arm, effort, rep, per_call = argv[2], argv[3], int(argv[4]), float(argv[5])
        run_batch(crops, arm, effort, rep, per_call)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
