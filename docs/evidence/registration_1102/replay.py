"""Offline verdict replay (no model call) of a stored run with and without matching.registration, on scratch copies.

The run is copied with its frozen facts and rebuilt in place by the product (orchestrator._rebuild_in_place) with the
run's own frozen thresholds (base) or those plus the registration leaf (reg). Stored second looks and counter-looks
replay; a pair that registration makes and that was never asked has no stored look, so its caps stay (a live run
would ask it). Output: replay_<run>_<variant>.json with every verdict and every panel's registration records.

Usage: replay.py RUN_ID SCRATCH VARIANT [LEAF_JSON]   VARIANT is base or reg (LEAF_JSON overrides the leaf)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from walkaround import contracts as c  # noqa: E402
from walkaround import paths, pipeline  # noqa: E402
from walkaround.orchestrator import _runs_root  # noqa: E402

RUN, OUT, VARIANT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
LEAF = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
orch = pipeline.make_orchestrator()
run = orch.load_run(RUN)
th = json.loads(json.dumps(orch.run_th(run)))
if VARIANT != "base":
    th["matching"]["registration"] = LEAF if LEAF is not None else c.load_thresholds()["matching"]["registration"]
target = OUT / RUN
if target.exists():
    shutil.rmtree(target)
never = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", "events.pending.ndjson", "bundle", "clips", "chips")
shutil.copytree(paths.run_dir(RUN), target, ignore=shutil.ignore_patterns(*never))
(OUT / "ledger.jsonl").write_text("")
with _runs_root(OUT):
    orch._rebuild_in_place(run, th)
    vs = c.read_json_list(paths.verdicts_json(RUN), c.Verdict)
    pms = c.read_json_list(paths.matches_json(RUN), c.PanelMatch)
    ms = {m.mark_id: m for m in c.read_json_list(paths.marks_json(RUN), c.Mark)}
rows = []
for v in vs:
    rows.append(
        {
            "verdict_id": v.verdict_id,
            "panel": v.panel_id.value,
            "label": v.label.value,
            "rule_ids": v.trace.rule_ids,
            "mark_ids": v.mark_ids,
            "types": [f"{ms[i].type.value}/{ms[i].confidence.value}" for i in v.mark_ids if i in ms],
            "d": v.distance_units,
            "registered": bool(v.trace.thresholds.get("registered")),
        }
    )
panels = [
    {
        "panel": pm.panel_id.value,
        "state": pm.state.value,
        "kf_pair": pm.kf_pair,
        "fits": [f.model_dump(mode="json", exclude={"H"}) for f in pm.registration],
        "marks": [r.model_dump(mode="json") for r in pm.registration_marks],
    }
    for pm in pms
]
summary: dict[str, int] = {}
for r in rows:
    summary[r["label"]] = summary.get(r["label"], 0) + 1
(HERE / f"replay_{RUN}_{VARIANT}.json").write_text(json.dumps({"run": RUN, "variant": VARIANT, "summary": summary, "verdicts": rows, "panels": panels}, indent=1) + "\n")
print(json.dumps({"run": RUN, "variant": VARIANT, **summary}))
