"""Before and after overlays for the box fix (no call): truth green (boxes) or green crosses (pair C's D1 targets),
Spark's box orange (before), the box the mark uses after the SAM step cyan (after), SAM's mask tinted cyan. Written
to <workspace>/docs/box_fix/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure as M  # noqa: E402
import overlay as O  # noqa: E402

from walkaround import privacy  # noqa: E402
from walkaround.claim import SamDetection, _detection  # noqa: E402

OUT = Path(__file__).resolve().parents[4] / "docs" / "box_fix"
PICKS = {
    "synth_pair2:pickup_right_front_door_kf1": "synth_20cm_scratch",
    "synth_pair2_r9:return_hood_kf2": "synth_5cm_scratch",
    "synth_pair2:return_hood_kf1": "synth_20mm_chip_beside",
    "pairC:pickup_trunk_chips:study0": "pairC_pickup_trunk_chips",
    "pairC:return_trunk_chips:study0": "pairC_return_trunk_chips",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stored = json.loads(M.ANSWERS.read_text())
    for it in M.items():
        name = PICKS.get(it["id"])
        if not name:
            continue
        pairs = [p for p in M.refined(it, stored) if p is not None]
        rgb = O.read_rgb(it["path"])
        before = [p[0].bbox_crop for p in pairs]
        after = [p[1].bbox_crop for p in pairs]
        masks = np.zeros(rgb.shape[:2], bool)
        for chunk in M.box_refine.plan(it["marks"], it["kf"], M.S):
            row = stored.get(M._key(it, chunk))
            for d in (row or {}).get("answer", {}).get("detections", []):
                masks |= privacy.detection_mask(_detection(SamDetection.model_validate(d)), (it["crop"].w, it["crop"].h))
        truth = [g["box"] for g in it.get("truth", []) if g["damage"]]
        targets = it.get("targets", [])
        if it["set"] == "synth":  # around the truth the boxes sit on
            near = [b for b in before if any(M.T.near(b, g) for g in truth)] or before
            hit = [g for g in truth if any(M.T.near(b, g) for b in near)]
            xs = [v for b in [*near, *hit] for v in (b[0], b[2])]
            ys = [v for b in [*near, *hit] for v in (b[1], b[3])]
        else:
            xs = [p[0] for p in targets]
            ys = [p[1] for p in targets]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(60.0, 0.7 * (max(xs) - min(xs)), 0.7 * (max(ys) - min(ys)) * 4 / 3)
        win = [cx - half, cy - half * 0.75, cx + half, cy + half * 0.75]
        zoom = max(1, min(4, int(900 // (2 * half))))
        O.save(str(OUT / f"{name}_before.png"), O.draw(rgb, truth=truth, spark=before, targets=targets, window=win, zoom=zoom))
        O.save(str(OUT / f"{name}_after.png"), O.draw(rgb, truth=truth, sam=after, targets=targets, window=win, zoom=zoom, mask=masks))
        print(name, len(before), "boxes, window", [round(v) for v in win], "zoom", zoom)
    return 0


if __name__ == "__main__":
    sys.exit(main())
