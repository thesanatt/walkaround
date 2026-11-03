"""Offline verdict replay (no model call) of stored runs with and without the R29 leaves (matching.geometric_already_there,
matching.glimpsed_compare), on scratch copies, by the product's rebuild (orchestrator._rebuild_in_place). Two bases:
frozen (each run's frozen thresholds) and current (frozen with every matching leaf of config/thresholds.json except the
two). r29 adds the two leaves to the base. Every line that changed is listed. Stored looks replay; a look never asked
gives no answer, so a new look (the L8 look) lowers nothing here.

Usage: .venv/bin/python docs/evidence/r29_1102/replay.py SCRATCH_DIR RUN_ID [RUN_ID ...]  -> replay.json here
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
NOW = c.load_thresholds()["matching"]
LEAVES = ("geometric_already_there", "glimpsed_compare")
NEVER = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", "events.pending.ndjson", "bundle", "clips", "chips", "sheet")
orch = pipeline.make_orchestrator()


def thresholds(run_id: str, base: str, r29: bool) -> dict:
    th = json.loads(json.dumps(orch.run_th(orch.load_run(run_id))))
    if base == "current":
        for k, v in NOW.items():
            if k not in LEAVES:
                th["matching"][k] = json.loads(json.dumps(v))
    for leaf in LEAVES:
        th["matching"].pop(leaf, None)
        if r29:
            th["matching"][leaf] = json.loads(json.dumps(NOW[leaf]))
    return th


def rebuild(run_id: str, base: str, r29: bool) -> dict:
    run = orch.load_run(run_id)
    th = thresholds(run_id, base, r29)
    root = OUT / f"{base}_{'r29' if r29 else 'base'}"
    target = root / run_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(paths.run_dir(run_id), target, ignore=shutil.ignore_patterns(*NEVER))
    (root / "ledger.jsonl").write_text("")
    with _runs_root(root):
        orch._rebuild_in_place(run, th)
        vs = c.read_json_list(paths.verdicts_json(run_id), c.Verdict)
        pms = c.read_json_list(paths.matches_json(run_id), c.PanelMatch)
    rows = {"|".join([v.panel_id.value, *v.mark_ids]): (v.label.value, v.trace.rule_ids if v.trace else []) for v in vs}
    summary: dict[str, int] = {}
    for v in vs:
        summary[v.label.value] = summary.get(v.label.value, 0) + 1
    geo = [r.model_dump(exclude_none=True) for pm in pms for r in pm.registration_marks if r.used == "already_there"]
    spots = [s.model_dump(exclude_none=True) for pm in pms for s in pm.spots]
    return {"rows": rows, "summary": summary, "geometric": geo, "spots": spots}


report = {}
for run_id in RUNS:
    report[run_id] = {}
    for base in ("frozen", "current"):
        a, b = rebuild(run_id, base, False), rebuild(run_id, base, True)
        changed = []
        for key in sorted(set(a["rows"]) | set(b["rows"])):
            if a["rows"].get(key) != b["rows"].get(key):
                changed.append({"line": key, "base": a["rows"].get(key), "r29": b["rows"].get(key)})
        report[run_id][base] = {"summary_base": a["summary"], "summary_r29": b["summary"], "changed": changed, "geometric": b["geometric"], "spots": b["spots"]}
        print(run_id, base, a["summary"], "->", b["summary"], f"{len(changed)} changed")
        for ch in changed:
            print("   ", ch["line"], ch["base"], "->", ch["r29"])
(HERE / "replay.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
