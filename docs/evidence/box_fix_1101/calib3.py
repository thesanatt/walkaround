"""Third calibration round (LIVE=1 to call): calib2's 2.9:1, 1:1 and 9:16 canvases letterboxed, centred, onto a 16:9
canvas of their mean colour before the product call (prepare_image then sends at most 1280x720); boxes mapped back
by removing the pad. Writes calib3.json; without LIVE=1 it re-scores the stored answers."""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib2 as C2  # noqa: E402

from walkaround import spark_client  # noqa: E402
from walkaround.contracts import PanelId, SparkMark  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "calib3.json"
RUN_ID = "box_fix_1101"
SIZES = [(1280, 441), (1100, 1100), (720, 1280)]


def letterbox(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    cw, ch = max(w, int(np.ceil(h * 16 / 9))), max(h, int(np.ceil(w * 9 / 16)))
    ox, oy = (cw - w) // 2, (ch - h) // 2
    out = np.empty((ch, cw, 3), np.uint8)
    out[...] = img.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    out[oy:oy + h, ox:ox + w] = img
    return out, ox, oy


def main() -> int:
    live = os.environ.get("LIVE") == "1"
    stored = json.loads(OUT.read_text()) if OUT.exists() else {}
    rows = []
    for seed in (1, 2):
        for w, h in SIZES:
            key = f"{w}x{h}_s{seed}"
            img, pts = C2.canvas(w, h, seed)
            box_img, ox, oy = letterbox(img)
            if key not in stored.get("answers", {}):
                if not live:
                    continue
                res = spark_client.find_marks(box_img, PanelId.HOOD, landmarks=[], effort="low", run_id=RUN_ID)
                stored.setdefault("answers", {})[key] = {"canvas": [box_img.shape[1], box_img.shape[0]], "pad": [ox, oy], "sent": [res.sent_width, res.sent_height],
                                                        "marks": [m.model_dump(mode="json") for m in (res.answer.marks if res.answer else [])]}
            ans = stored["answers"][key]
            cw, chh = ans["canvas"]
            for sm in (SparkMark.model_validate(m) for m in ans["marks"]):
                cx = (sm.bbox.x_min + sm.bbox.x_max) / 2 * cw / 1000 - ox
                cy = (sm.bbox.y_min + sm.bbox.y_max) / 2 * chh / 1000 - oy
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
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
