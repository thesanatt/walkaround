"""Second calibration round (LIVE=1 to call): eight chips at seeded random positions per canvas, so no symmetric
layout lets Spark copy one coordinate across a row; four aspects, two layouts each. Same product call and map as
calib.py. Writes calib2.json; without LIVE=1 it re-scores the stored answers."""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

from walkaround import marks, spark_client
from walkaround.contracts import Crop, PanelId, SparkMark

HERE = Path(__file__).resolve().parent
OUT = HERE / "calib2.json"
RUN_ID = "box_fix_1101"
SIZES = [(1280, 441), (1280, 720), (1100, 1100), (720, 1280)]


def canvas(w: int, h: int, seed: int) -> tuple[np.ndarray, list[list[float]]]:
    rng = np.random.default_rng(1000 * seed + w + 3 * h)
    img = np.empty((h, w, 3), np.float32)
    yy = np.linspace(0, 1, h)[:, None]
    img[...] = (np.array([22, 34, 58]) + 30 * yy)[..., None, :].reshape(h, 1, 3)
    img += rng.normal(0, 2.0, img.shape)
    r = max(4, round(min(w, h) * 0.008))
    pts: list[list[float]] = []
    while len(pts) < 8:
        p = [float(rng.uniform(0.07, 0.93)) * w, float(rng.uniform(0.07, 0.93)) * h]
        if all(abs(p[0] - q[0]) > 0.12 * w or abs(p[1] - q[1]) > 0.12 * h for q in pts):
            pts.append(p)
    for cx, cy in pts:
        cv2.circle(img, (round(cx), round(cy)), r, (170, 172, 168), -1, lineType=cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8), pts


def main() -> int:
    live = os.environ.get("LIVE") == "1"
    stored = json.loads(OUT.read_text()) if OUT.exists() else {}
    rows = []
    for seed in (1, 2):
        for w, h in SIZES:
            key = f"{w}x{h}_s{seed}"
            img, pts = canvas(w, h, seed)
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
            for sm in (SparkMark.model_validate(m) for m in ans["marks"]):
                box, _, _, _ = marks.map_spark_mark(sm, res, crop)
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                t = min(pts, key=lambda t: (cx - t[0]) ** 2 + (cy - t[1]) ** 2)
                rows.append({"canvas": key, "aspect": round(w / h, 2), "truth": [round(t[0], 1), round(t[1], 1)], "dx": round(cx - t[0], 1), "dy": round(cy - t[1], 1),
                             "gx": round(t[0] / w, 3), "gy": round(t[1] / h, 3)})
    stored["rows"] = rows
    summ = {}
    for a in sorted({r["aspect"] for r in rows}):
        rs = [r for r in rows if r["aspect"] == a]
        summ[str(a)] = {"n": len(rs), "dx_med": st.median(r["dx"] for r in rs), "dy_med": st.median(r["dy"] for r in rs),
                        "abs_med": round(st.median((r["dx"] ** 2 + r["dy"] ** 2) ** 0.5 for r in rs), 1),
                        "far_over_20px": sum((r["dx"] ** 2 + r["dy"] ** 2) ** 0.5 > 20 for r in rs)}
    stored["summary"] = summ
    OUT.write_text(json.dumps(stored, indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
