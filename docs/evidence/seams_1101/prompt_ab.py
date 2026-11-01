"""The marks prompt synth_pair2 ran with (fixtures sentence on) vs the same prompt plus spark_client.SEAMS_SENTENCE, on
every stored keyframe crop of synth_pair2, scored against the render's ground truth (synthetic/out/pair2, CC0, exact
boxes per frame). REPEATS answers per prompt per crop, interleaved, effort as in a run. A box hits a ground-truth mark
when its centre falls inside the mark's box grown to 1.5 times its width and height, or IoU > 0.1 (the synthetic
scorer's rule); every other box is a false mark. Writes prompt_ab.json beside this file. Spends credit: LIVE=1."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from walkaround import config, marks, paths, spark_client
from walkaround.contracts import InputRef, Keyframe, read_json_list

RUN = "synth_pair2"
GT = Path(__file__).resolve().parents[4] / "synthetic" / "out" / "pair2" / "ground_truth.json"
REPEATS = int(os.environ.get("REPEATS", "2"))
OUT = Path(__file__).with_name("prompt_ab.json")


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
            out.append({"id": m["id"], "kind": m["kind"], "bbox_src": b})
    return out


def hit(box: list[float], g: list[float]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    gw, gh = g[2] - g[0], g[3] - g[1]
    gx, gy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    if abs(cx - gx) <= 0.75 * gw and abs(cy - gy) <= 0.75 * gh:
        return True
    ix = max(0.0, min(box[2], g[2]) - max(box[0], g[0]))
    iy = max(0.0, min(box[3], g[3]) - max(box[1], g[1]))
    inter = ix * iy
    union = (box[2] - box[0]) * (box[3] - box[1]) + gw * gh - inter
    return union > 0 and inter / union > 0.1


def main() -> int:
    if os.environ.get("LIVE") != "1":
        sys.stderr.write("set LIVE=1 to spend API credit\n")
        return 2
    config.load_env()
    th = json.loads((paths.run_dir(RUN) / "run.json").read_text())["thresholds"]
    gt = json.loads(GT.read_text())
    kfs = [k for k in read_json_list(paths.keyframes_json(RUN), Keyframe) if k.crop is not None and k.crop.path]
    jobs = []
    for kf in sorted(kfs, key=lambda k: k.kf_id):
        for rep in range(REPEATS):
            for arm in ("old", "seams"):
                jobs.append((kf, rep, arm))

    def one(job: tuple) -> dict:
        kf, rep, arm = job
        crop = cv2.cvtColor(cv2.imread(str(paths.run_dir(RUN) / kf.crop.path)), cv2.COLOR_BGR2RGB)
        h, w = crop.shape[:2]
        res = spark_client.find_marks(
            crop, kf.panel_id, landmarks=marks.landmarks_for(kf.panel_id, th), effort=str(th["spark"]["effort"]),
            run_id="seams_ab_1101", input_ref=InputRef(path=kf.crop.path, sha256=kf.crop.sha256), fixtures=True, seams=arm == "seams",
        )
        boxes = []
        for m in (res.answer.marks if res.answer else []):
            b = m.bbox
            x1, y1 = spark_client.grid_to_pixels(b.x_min, b.y_min, w, h)
            x2, y2 = spark_client.grid_to_pixels(b.x_max, b.y_max, w, h)
            c = kf.crop
            boxes.append({"type": m.type.value, "bbox_src": [c.dx + x1 / c.scale, c.dy + y1 / c.scale, c.dx + x2 / c.scale, c.dy + y2 / c.scale]})
        return {"kf_id": kf.kf_id, "rep": rep, "arm": arm, "valid": res.answer is not None, "boxes": boxes}

    with ThreadPoolExecutor(6) as pool:
        answers = list(pool.map(one, jobs))
    truth = {kf.kf_id: gt_in_crop(gt, kf) for kf in kfs}
    summary = {}
    for arm in ("old", "seams"):
        rows = [a for a in answers if a["arm"] == arm]
        n_box = n_false = n_hit = 0
        found = set()
        for a in rows:
            for b in a["boxes"]:
                n_box += 1
                ids = [g["id"] for g in truth[a["kf_id"]] if hit(b["bbox_src"], g["bbox_src"])]
                b["hits"] = ids
                if ids:
                    n_hit += 1
                    found.update((a["kf_id"], a["rep"], i) for i in ids)
                else:
                    n_false += 1
        damage = {(k, r, i) for k, r, i in found if not i.startswith("control")}
        summary[arm] = {
            "answers": len(rows), "valid": sum(a["valid"] for a in rows), "boxes": n_box, "boxes_on_gt": n_hit, "false_boxes": n_false,
            "damage_finds": len(damage), "sticker_finds": len(found) - len(damage),
            "gt_marks_shown": sum(len(v) for v in truth.values()) * REPEATS,
        }
    OUT.write_text(json.dumps({"run": RUN, "repeats": REPEATS, "crops": len(kfs), "summary": summary, "truth": truth, "answers": answers}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
