"""Offline verdict replay (no model call) of stored runs with and without matching.stands_out (K20), on scratch copies,
by the product's rebuild (orchestrator._rebuild_in_place) with each run's frozen thresholds. base is the frozen
thresholds; k20 adds the leaf as config/thresholds.json has it. Every label change is listed with the mark's contrast.

Usage: .venv/bin/python docs/evidence/stands_out_1102/replay.py SCRATCH_DIR RUN_ID [RUN_ID ...]  -> replay.json here
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

OUT, RUNS = Path(sys.argv[1]), sys.argv[2:]
LEAF = c.load_thresholds()["matching"]["stands_out"]
NEVER = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", "events.pending.ndjson", "bundle", "clips", "chips")
orch = pipeline.make_orchestrator()


def rebuild(run_id: str, variant: str) -> dict:
    run = orch.load_run(run_id)
    th = json.loads(json.dumps(orch.run_th(run)))
    if variant == "k20":
        th["matching"]["stands_out"] = dict(LEAF)
    root = OUT / variant
    target = root / run_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(paths.run_dir(run_id), target, ignore=shutil.ignore_patterns(*NEVER))
    (root / "ledger.jsonl").write_text("")
    with _runs_root(root):
        orch._rebuild_in_place(run, th)
        vs = c.read_json_list(paths.verdicts_json(run_id), c.Verdict)
        pms = c.read_json_list(paths.matches_json(run_id), c.PanelMatch)
    rows = {(v.panel_id.value, *v.mark_ids): (v.label.value, v.trace.rule_ids) for v in vs}
    assert len(rows) == len(vs)
    tested = {r.mark_id: r.model_dump() for pm in pms for r in pm.own_contrast}
    summary: dict[str, int] = {}
    for v in vs:
        summary[v.label.value] = summary.get(v.label.value, 0) + 1
    return {"rows": rows, "tested": tested, "summary": summary}


report = {}
for run_id in RUNS:
    base, k20 = rebuild(run_id, "base"), rebuild(run_id, "k20")
    changed = []
    for key in sorted(set(base["rows"]) | set(k20["rows"])):
        a, b = base["rows"].get(key), k20["rows"].get(key)
        if a != b:
            changed.append({"panel": key[0], "mark_ids": list(key[1:]), "base": a, "k20": b, "contrast": [k20["tested"].get(i) for i in key[1:]]})
    one_walk = [r for r in k20["tested"].values()]
    report[run_id] = {"base": base["summary"], "k20": k20["summary"], "changed": changed, "tested": one_walk}
    print(json.dumps({"run": run_id, "base": base["summary"], "k20": k20["summary"], "changed": len(changed), "tested": len(one_walk)}))
(HERE / "replay.json").write_text(json.dumps(report, indent=1) + "\n")
