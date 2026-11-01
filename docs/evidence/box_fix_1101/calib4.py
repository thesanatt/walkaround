"""Fourth calibration round (LIVE=1 to call): calib2's 2.9:1, 1:1 and 9:16 canvases stretched to exactly 1280x720
(each axis scaled on its own) before the product call; boxes mapped back per axis. Writes calib4.json; without LIVE=1
it re-scores the stored answers."""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib2 as C2  # noqa: E402

from walkaround import spark_client  # noqa: E402
from walkaround.contracts import PanelId, SparkMark  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "calib4.json"
RUN_ID = "box_fix_1101"
SIZES = [(1280, 441), (1100, 1100), (720, 1280)]
TARGET = (1280, 720)


def main() -> int:
    live = os.environ.get("LIVE") == "1"
    stored = json.loads(OUT.read_text()) if OUT.exists() else {}
    rows = []
    for seed in (1, 2):
        for w, h in SIZES:
            key = f"{w}x{h}_s{seed}"
            img, pts = C2.canvas(w, h, seed)
            if key not in stored.get("answers", {}):
                if not live:
                    continue
                big = cv2.resize(img, TARGET, interpolation=cv2.INTER_AREA if w * h > TARGET[0] * TARGET[1] else cv2.INTER_CUBIC)
                res = spark_client.find_marks(big, PanelId.HOOD, landmarks=[], effort="low", run_id=RUN_ID)
                stored.setdefault("answers", {})[key] = {"sent": [res.sent_width, res.sent_height], "marks": [m.model_dump(mode="json") for m in (res.answer.marks if res.answer else [])]}
            for sm in (SparkMark.model_validate(m) for m in stored["answers"][key]["marks"]):
                cx = (sm.bbox.x_min + sm.bbox.x_max) / 2 * w / 1000
                cy = (sm.bbox.y_min + sm.bbox.y_max) / 2 * h / 1000
                t = min(pts, key=lambda t: (cx - t[0]) ** 2 + (cy - t[1]) ** 2)
                rows.append({"canvas": key, "aspect": round(w / h, 2), "truth": [round(t[0], 1), round(t[1], 1)], "dx": round(cx - t[0], 1), "dy": round(cy - t[1], 1)})
    stored["rows"] = rows
    summ = {}
    for a in sorted({r["aspect"] for r in rows}):
        rs = [r for r in rows if r["aspect"] == a]
        summ[str(a)] = {"n": len(rs), "dx_med": st.median(r["dx"] for r in rs), "dy_med": st.median(r["dy"] for r in rs),
                        "abs_med": round(st.median((r["dx"] ** 2 + r["dy"] ** 2) ** 0.5 for r in rs), 1),
                        "far_over_20px": sum((r["dx"] ** 2 + r["dy"] ** 2) ** 0.5 > 20 for r in rs)}
    stored["summary"] = summ
    OUT.write_text(json.dumps(stored, indent=1) + "\n")
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
