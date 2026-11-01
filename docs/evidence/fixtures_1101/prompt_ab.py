"""Old marks_v1 prompt vs the one with the fixture sentence, on the crop where the eval's false alarm came from
(eval_px_two_takes, return_rear_bumper_kf1). Five calls each, interleaved, effort low as in a run. A mark counts as the
sensor dot when its box overlaps the first run's chip box (return_rear_bumper_kf1_m1) in source pixels. Writes
prompt_ab.json beside this file. Spends credit: LIVE=1 is required."""

import json
import os
import sys
from pathlib import Path

import cv2

from walkaround import config, marks, paths, spark_client
from walkaround.contracts import InputRef, Keyframe, Mark, PanelId, load_thresholds, read_json_list

RUN, KF, DOT = "eval_px_two_takes", "return_rear_bumper_kf1", "return_rear_bumper_kf1_m1"
N = 5


def main() -> int:
    if os.environ.get("LIVE") != "1":
        sys.stderr.write("set LIVE=1 to spend API credit\n")
        return 2
    config.load_env()
    th = load_thresholds()
    kf = next(k for k in read_json_list(paths.keyframes_json(RUN), Keyframe) if k.kf_id == KF)
    dot = next(m for m in read_json_list(paths.marks_json(RUN), Mark) if m.mark_id == DOT)
    crop = cv2.cvtColor(cv2.imread(str(paths.run_dir(RUN) / kf.crop.path)), cv2.COLOR_BGR2RGB)
    h, w = crop.shape[:2]
    names = marks.landmarks_for(PanelId.REAR_BUMPER, th)
    ref = InputRef(path=kf.crop.path, sha256=kf.crop.sha256)
    d = dot.bbox_src

    def src_box(b: dict) -> list[float]:
        x1, y1 = spark_client.grid_to_pixels(b["x_min"], b["y_min"], w, h)
        x2, y2 = spark_client.grid_to_pixels(b["x_max"], b["y_max"], w, h)
        c = kf.crop
        return [c.dx + x1 / c.scale, c.dy + y1 / c.scale, c.dx + x2 / c.scale, c.dy + y2 / c.scale]

    rows = []
    for i in range(N):
        for fixtures in (False, True):
            r = spark_client.find_marks(crop, PanelId.REAR_BUMPER, landmarks=names, effort="low", run_id="fixture_ab_1101", input_ref=ref, fixtures=fixtures)
            got = []
            for m in (r.answer.marks if r.answer else []):
                b = src_box(m.bbox.model_dump())
                on_dot = b[0] < d[2] and d[0] < b[2] and b[1] < d[3] and d[1] < b[3]
                got.append({"type": m.type.value, "confidence": m.confidence.value, "note": m.note, "bbox_src": [round(v) for v in b], "on_dot": on_dot})
            rows.append({"prompt": "fixtures" if fixtures else "old", "i": i, "request_id": r.record.request_id, "valid": r.record.valid, "ms": r.record.ms, "dollars": r.record.dollars, "marks": got})
    summary = {p: {"answers": sum(1 for r in rows if r["prompt"] == p and r["valid"]), "dot_boxed": sum(1 for r in rows if r["prompt"] == p and any(m["on_dot"] for m in r["marks"])), "marks": sum(len(r["marks"]) for r in rows if r["prompt"] == p)} for p in ("old", "fixtures")}
    out = {"run": RUN, "keyframe": KF, "dot_mark": DOT, "dot_bbox_src": [round(v) for v in d], "n_each": N, "summary": summary, "dollars": round(sum(r["dollars"] for r in rows), 6), "rows": rows}
    (Path(__file__).parent / "prompt_ab.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "dollars": out["dollars"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
