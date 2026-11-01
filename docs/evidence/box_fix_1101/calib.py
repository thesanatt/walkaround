"""Spark's grid on drawn targets with exact positions (LIVE=1 to call; about $0.005 a call). Four canvases at the
aspects the product sends (2.9:1, 16:9, 1:1, 9:16 portrait), five light chips and one thin scratch each, asked with the
product's marks_v1 call; every box goes through marks.map_spark_mark, the product's own grid to crop map. Writes
calib.json; without LIVE=1 it re-scores the stored answers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from walkaround import marks, spark_client
from walkaround.contracts import Crop, PanelId

HERE = Path(__file__).resolve().parent
OUT = HERE / "calib.json"
RUN_ID = "box_fix_1101"
SIZES = [(1280, 441), (1280, 720), (1100, 1100), (720, 1280)]
CHIPS = [(0.18, 0.22), (0.82, 0.22), (0.5, 0.5), (0.18, 0.78), (0.82, 0.78)]
SCRATCH = ((0.35, 0.66), (0.62, 0.68))


def canvas(w: int, h: int) -> tuple[np.ndarray, list[list[float]]]:
    rng = np.random.default_rng(w * 7 + h)
    img = np.empty((h, w, 3), np.float32)
    yy = np.linspace(0, 1, h)[:, None]
    img[...] = (np.array([22, 34, 58]) + 30 * yy)[..., None, :].reshape(h, 1, 3)  # dark blue paint, lighter low
    img += rng.normal(0, 2.0, img.shape)
    truth = []
    r = max(4, round(min(w, h) * 0.008))
    for fx, fy in CHIPS:
        cx, cy = fx * w, fy * h
        cv2.circle(img, (round(cx), round(cy)), r, (170, 172, 168), -1, lineType=cv2.LINE_AA)
        truth.append({"kind": "chip", "box": [cx - r, cy - r, cx + r, cy + r]})
    (ax, ay), (bx, by) = SCRATCH
    cv2.line(img, (round(ax * w), round(ay * h)), (round(bx * w), round(by * h)), (190, 195, 200), 2, lineType=cv2.LINE_AA)
    truth.append({"kind": "scratch", "box": [ax * w, min(ay, by) * h - 1, bx * w, max(ay, by) * h + 1]})
    return np.clip(img, 0, 255).astype(np.uint8), truth


def main() -> int:
    live = os.environ.get("LIVE") == "1"
    stored = json.loads(OUT.read_text()) if OUT.exists() else {}
    rows = []
    for w, h in SIZES:
        key = f"{w}x{h}"
        img, truth = canvas(w, h)
        if os.environ.get("CANVAS") == "1":  # CANVAS=1 writes the canvas to look at (not kept in the repo)
            cv2.imwrite(str(HERE / f"calib_{key}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if key not in stored.get("answers", {}):
            if not live:
                continue
            res = spark_client.find_marks(img, PanelId.HOOD, landmarks=[], effort="low", run_id=RUN_ID)
            stored.setdefault("answers", {})[key] = {"sent": [res.sent_width, res.sent_height], "scale": res.scale,
                                                    "marks": [m.model_dump(mode="json") for m in (res.answer.marks if res.answer else [])]}
        ans = stored["answers"][key]
        crop = Crop(path="", w=w, h=h, scale=1.0, dx=0.0, dy=0.0)
        res = spark_client.MarksResult(answer=None, truncated=False, record=None, sent_width=ans["sent"][0], sent_height=ans["sent"][1],
                                       scale=ans["scale"], orig_width=w, orig_height=h)  # type: ignore[arg-type]
        from walkaround.contracts import SparkMark
        for sm in (SparkMark.model_validate(m) for m in ans["marks"]):
            box, point, _, _ = marks.map_spark_mark(sm, res, crop)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            t = min(truth, key=lambda t: (cx - (t["box"][0] + t["box"][2]) / 2) ** 2 + (cy - (t["box"][1] + t["box"][3]) / 2) ** 2)
            tx, ty = (t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2
            rows.append({"canvas": key, "kind": t["kind"], "truth_centre": [round(tx, 1), round(ty, 1)], "box": [round(v, 1) for v in box],
                         "dx": round(cx - tx, 1), "dy": round(cy - ty, 1), "dx_grid": round(1000 * (cx - tx) / w, 1), "dy_grid": round(1000 * (cy - ty) / h, 1)})
    stored["rows"] = rows
    OUT.write_text(json.dumps(stored, indent=1) + "\n")
    for r in rows:
        print(r["canvas"], r["kind"], r["truth_centre"], "dx", r["dx"], "dy", r["dy"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
