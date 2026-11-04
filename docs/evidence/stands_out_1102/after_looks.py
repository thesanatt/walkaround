"""K20 on R22's after-looks replays (docs/evidence/hero_1102/replay_rr_<run>_all+reg_looks.json): every line labelled
new or not seen at return there, its own contrast on the source run's keyframe crop and its panel share. Pixels only.

Usage: .venv/bin/python docs/evidence/stands_out_1102/after_looks.py  -> after_looks.json here
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from walkaround import contracts as c  # noqa: E402
from walkaround import match, paths, register  # noqa: E402

cfg = match.stands_out_config(c.load_thresholds())
out = []
for run in ("hero_focus1", "friend_pair2", "pairC_r16", "eval_nhtsa15759_r16"):
    rows = json.loads((HERE.parent / "hero_1102" / f"replay_rr_{run}_all+reg_looks.json").read_text())["verdicts"]
    kfs = {k["kf_id"]: k for k in json.loads((paths.run_dir(run) / "derived" / "keyframes.json").read_text())}
    for r in rows:
        if r["label"] not in ("new_or_unseen", "not_seen_at_return"):
            continue
        for mid, box, note in zip(r["mark_ids"], r["boxes"], r["notes"]):
            k = kfs[re.sub(r"_[ms]\d+$", "", mid)]
            img = cv2.imread(str(paths.run_dir(run) / k["crop"]["path"]))
            mask = cv2.imread(str(paths.run_dir(run) / k["mask_png"]), cv2.IMREAD_GRAYSCALE) > 0
            share = round((box[2] - box[0]) * (box[3] - box[1]) / float(mask.sum()), 3)
            con = None if share > cfg["max_panel_share"] else register.own_contrast(img, tuple(box), cfg)
            res = "area" if share > cfg["max_panel_share"] else ("stands_out" if con is not None and con >= cfg["min_contrast"] else "faint")
            out.append({"run": run, "label": r["label"], "mark_id": mid, "note": note, "panel_share": share, "contrast": None if con is None else round(con, 2), "k20": res})
            print(run, r["label"], mid, share, None if con is None else round(con, 2), res, note[:40])
(HERE / "after_looks.json").write_text(json.dumps(out, indent=1) + "\n")
