"""Halves merge (11/2), offline on R18's stored answers: no model call, $0.

Per crop Sanat checked, the product's own steps as a new run takes them (marks._box_marks with the halves boxes, the zoom
move, the SAM step), then the variant's last step (halves_pass.merge_after_refine or drop_after_refine). Full-crop boxes
are moved with stored answers where they exist: lexi_pair1 its own run's, synth_pair2_r9 R17's (zoom_refine_synth_1102),
pairC_core3 R14's w1 window and R13's SAM answers on the moved boxes (zoom_refine_1101). Halves boxes use R18's answers.

Scoring is R18's (halves_1102/study.py): full-crop marks by his labels (correct, moved = found; not_mark = false box;
his missed boxes = misses); a kept halves box by where its final box lands (IoU >= 0.1, or centre inside his box grown
by max(10 px, a quarter of a side), or his centre inside it). A merged mark keeps its full-crop label, and if that label
is not_mark and its new box hits a missed mark it finds it. Also: for merged marks he labeled found, is the final box on
his box with the full crop's own moved box, and with the halves box.

  python study.py   (writes study.json, prints the table)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("halves_1102_study", HERE.parent / "halves_1102" / "study.py")
r18 = importlib.util.module_from_spec(_spec)
sys.modules["halves_1102_study"] = r18
_spec.loader.exec_module(r18)

from walkaround import box_refine, halves_pass, zoom_refine  # noqa: E402
from walkaround import contracts as c  # noqa: E402
from walkaround.marks import _box_marks, _wheel_mask  # noqa: E402

REPO = r18.REPO
VARIANTS = {
    "halves": {},  # R18's union
    "halves2": {"after_refine": True},  # R18's second drop
    "merge10": {"merge_patches": 1.0, "patch_sent_px": 28, "lone_uncapped": True},
    "merge15": {"merge_patches": 1.5, "patch_sent_px": 28, "lone_uncapped": True},
    "merge15_keep": {"merge_patches": 1.5, "patch_sent_px": 28},  # lone halves boxes kept, capped as before (not the proposal)
}


def extra_zoom(run: str) -> list:
    if run == "synth_pair2_r9":
        return c.read_json_list(REPO / "docs/evidence/zoom_refine_synth_1102/zoom_answers.json", zoom_refine.ZoomAsk)
    if run == "pairC_core3":
        kfs = r18.keyframes(run)
        rows = json.loads((REPO / "docs/evidence/zoom_refine_1101/answers.json").read_text())
        out = []
        for r in rows.values():
            if r["window_name"] != "w1":
                continue
            kf = kfs[r["kf_id"]]
            out.append(zoom_refine.ZoomAsk(kf_id=kf.kf_id, role=kf.role, mark_id=r["mark_id"], panel_id=kf.panel_id, crop_path=kf.crop.path, crop_sha256=kf.crop.sha256,
                                           crop_w=kf.crop.w, crop_h=kf.crop.h, window=r["window"], out_w=r["sent"][0], out_h=r["sent"][1], record=r["record"]))
        return out
    return []


def extra_sam(run: str) -> list:
    if run == "synth_pair2_r9":
        return c.read_json_list(REPO / "docs/evidence/zoom_refine_synth_1102/sam_zoom.json", box_refine.BoxAsk)
    if run == "pairC_core3":
        return c.read_json_list(REPO / "docs/evidence/zoom_refine_1101/sam_answers.json", box_refine.BoxAsk)
    return []


def th_for(run: str, variant: str) -> dict:
    th = r18.th_for(run)
    th["dedup"]["spark_halves_pass"] = {**r18.LEAF, **VARIANTS[variant]}
    return th


def crop_marks(run: str, variant: str) -> tuple[dict[str, list[c.Mark]], dict]:
    """kf_id -> the crop's marks after the variant's last step, and counts of full-crop boxes moved."""
    th = th_for(run, variant)
    root = REPO / "runs" / run
    res, kfs = r18.results(run, th), r18.keyframes(run)
    hidx = halves_pass.Index(r18.stored("halves"))
    zidx = zoom_refine.Index([*(zoom_refine.read_asks(run) if (root / "zoom_marks.json").exists() else []), *r18.stored("zoom"), *extra_zoom(run)])
    sidx = box_refine.Index([*box_refine.read_asks(run), *r18.stored("sam"), *extra_sam(run)])
    out, moved = {}, {"full": 0, "zoom": 0, "sam": 0}
    for kid in r18.eval_crops(run):
        if kid not in res:
            continue
        kf = kfs[kid]
        mask = _wheel_mask(root, kf)
        ms = _box_marks(root, kf, res[kid], th=th, halves_index=hidx, mask=mask)
        ms = zoom_refine.refine_marks(ms, kf, zidx, th=th)
        ms = box_refine.refine_marks(ms, kf, sidx, th=th)
        for m in ms:
            if m.halves is None and m.source == c.MarkSource.SPARK_BOX:
                moved["full"] += 1
                moved["zoom"] += m.zoom_refine is not None and m.zoom_refine.used == "zoom"
                moved["sam"] += m.box_refine is not None and m.box_refine.used == "sam"
        if VARIANTS[variant]:
            ms = halves_pass.merge_after_refine(ms, kf, th=th, mask=mask)
        out[kid] = ms
    return out, moved


def score(run: str, variant: str) -> dict:
    gt = r18.gt_boxes(run)
    label = {g["id"]: g for k in gt for g in gt[k]}
    by, moved = crop_marks(run, variant)
    crops = r18.eval_crops(run)
    real = [g for k in crops for g in gt.get(k, []) if g["kind"] in ("found", "missed")]
    o = {"crops": len(crops), "real": len(real), "found_full": sum(g["kind"] == "found" for g in real), "missed_full": sum(g["kind"] == "missed" for g in real),
         "false_full": sum(g["kind"] == "not_mark" for k in crops for g in gt.get(k, [])), "full_moved": moved}
    order = {"found": 0, "missed": 0, "not_mark": 1, "unlabeled": 2}
    took: dict[str, str] = {}
    rows, merged = [], []
    for kid, ms in by.items():
        for m in ms:
            b = list(m.bbox_crop or [])
            if m.halves is None:
                g = label.get(m.mark_id)
                if m.halves_merge is None:
                    continue
                own = g is not None and g["kind"] == "found"
                merged.append({"mark_id": m.mark_id, "kf_id": kid, "halves": m.halves_merge.halves_mark_id, "gap": m.halves_merge.gap_patches, "overlap": m.halves_merge.overlap,
                               "label": g["status"] if g else None, "on_his_box_full": own and r18.hits(m.halves_merge.full_bbox_crop, g["box"]), "on_his_box_halves": own and r18.hits(b, g["box"])})
                if g is not None and g["kind"] == "not_mark":
                    hit = [x for x in gt.get(kid, []) if x["kind"] == "missed" and r18.hits(b, x["box"]) and x["id"] not in took]
                    if hit:
                        took[hit[0]["id"]] = m.mark_id
                        rows.append({"mark_id": m.mark_id, "what": "finds_missed", "gt": hit[0]["id"], "type": m.type.value, "conf": m.confidence.value, "caps": m.caps})
                continue
            cands = [g for g in gt.get(kid, []) if r18.hits(b, g["box"])]
            cands.sort(key=lambda g: (order[g["kind"]], (r18.centre(b)[0] - r18.centre(g["box"])[0]) ** 2 + (r18.centre(b)[1] - r18.centre(g["box"])[1]) ** 2))
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
            rows.append({"mark_id": m.mark_id, "kf_id": kid, "what": what, "gt": g["id"] if g else None, "type": m.type.value, "conf": m.confidence.value, "caps": m.caps})
    n = lambda w: sum(r["what"] == w for r in rows)
    o.update({"finds_missed": n("finds_missed"), "duplicate": n("duplicate"), "false_new": n("false") + n("false_on_not_mark"), "unknown": n("unknown"), "kept_halves": sum(1 for r in rows if r["mark_id"].rsplit("_", 1)[-1].startswith("h"))})
    o["found"] = o["found_full"] + o["finds_missed"]
    o["missed"] = o["missed_full"] - o["finds_missed"]
    o["false"] = o["false_full"] + o["false_new"]
    o["merged"] = len(merged)
    o["merged_found_on_box_full"] = sum(x["on_his_box_full"] for x in merged)
    o["merged_found_on_box_halves"] = sum(x["on_his_box_halves"] for x in merged)
    o["merged_found"] = sum(x["label"] in ("correct", "moved") for x in merged)
    o["missed_found_ids"] = sorted(r["gt"] for r in rows if r["what"] == "finds_missed")
    o["rows"], o["merged_rows"] = rows, merged
    return o


def main() -> None:
    out = {run: {v: score(run, v) for v in VARIANTS} for run in r18.RUNS}
    (HERE / "study.json").write_text(json.dumps(out, indent=1) + "\n")
    print("| run | variant | found / missed / false boxes | kept halves: find / dup / false / unknown | merged (he labeled found: on his box full -> halves) | full boxes moved zoom/sam of n |")
    for run in r18.RUNS:
        o0 = out[run]["halves"]
        print(f"| {run} | full crop | {o0['found_full']} / {o0['missed_full']} / {o0['false_full']} | | | |")
        for v, o in out[run].items():
            print(f"| {run} | {v} | {o['found']} / {o['missed']} / {o['false']} | {o['finds_missed']} / {o['duplicate']} / {o['false_new']} / {o['unknown']} | "
                  f"{o['merged']} ({o['merged_found']}: {o['merged_found_on_box_full']} -> {o['merged_found_on_box_halves']}) | {o['full_moved']['zoom']}/{o['full_moved']['sam']} of {o['full_moved']['full']} | {o['missed_found_ids']}")


if __name__ == "__main__":
    main()
