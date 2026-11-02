"""Overlay sheet: 12 of his moved boxes (the first 12 by mark id), each tile cut around his box from the product crop.
Red is Spark's box, green his box, blue Spark's box moved by the small-window zoom. No call.

  python overlays.py   # writes overlays.jpg next to this file
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[2] / "runs" / "pairC_core3"
TILE = 320


def main() -> None:
    study = json.loads((HERE / "study.json").read_text())
    kfs = {k["kf_id"]: k for k in json.loads((RUN / "derived" / "keyframes.json").read_text())}
    rows = [r for r in study["rows"] if r["status"] == "moved"][:12]
    sheet = Image.new("RGB", (4 * TILE, 3 * TILE), "white")
    for i, r in enumerate(rows):
        kf = kfs[r["mark_id"].rsplit("_m", 1)[0]]
        img = Image.open(RUN / kf["crop"]["path"]).convert("RGB")
        p = r["w1"]["pick"]
        moved = r["spark"]
        if p:
            (sx, sy), (zx, zy) = ((r["spark"][0] + r["spark"][2]) / 2, (r["spark"][1] + r["spark"][3]) / 2), ((p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2)
            moved = [r["spark"][0] + zx - sx, r["spark"][1] + zy - sy, r["spark"][2] + zx - sx, r["spark"][3] + zy - sy]
        h = r["hand"]
        cx, cy = (h[0] + h[2]) / 2, (h[1] + h[3]) / 2
        half = max(60.0, 0.9 * max(h[2] - h[0], h[3] - h[1], 2 * abs(cx - (r["spark"][0] + r["spark"][2]) / 2) + 40, 2 * abs(cy - (r["spark"][1] + r["spark"][3]) / 2) + 40))
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        tile = img.crop(box).resize((TILE, TILE), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(tile)
        s = TILE / (box[2] - box[0])
        for b, colour in ((r["spark"], (220, 40, 40)), (h, (40, 200, 60)), (moved, (40, 90, 230))):
            d.rectangle([(b[0] - box[0]) * s, (b[1] - box[1]) * s, (b[2] - box[0]) * s, (b[3] - box[1]) * s], outline=colour, width=3)
        d.text((6, 4), r["mark_id"], fill=(255, 255, 255))
        sheet.paste(tile, ((i % 4) * TILE, (i // 4) * TILE))
    sheet.save(HERE / "overlays.jpg", quality=80)


if __name__ == "__main__":
    main()
