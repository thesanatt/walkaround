"""Offline verdict replay (no model call) of stored runs with and without the R28 leaves (matching.same_walk_copies,
matching.faint_left_off, matching.side_from_track, matching.debris_by_eye any_view), on scratch copies, by the
product's rebuild (orchestrator._rebuild_in_place) with each run's frozen thresholds. base is the frozen thresholds;
r28 adds the four leaves as config/thresholds.json has them. Every line that changed is listed; "lost" is a base line
labelled already there, new or not seen at return that r28 does not give the same label.

Usage: .venv/bin/python docs/evidence/r28_1102/replay.py SCRATCH_DIR RUN_ID [RUN_ID ...]  -> replay.json here
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
LEAVES = ("same_walk_copies", "faint_left_off", "side_from_track", "debris_by_eye")
DECIDED = {"already_there", "new_or_unseen", "not_seen_at_return"}
NEVER = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", "events.pending.ndjson", "bundle", "clips", "chips", "sheet")
orch = pipeline.make_orchestrator()


def rebuild(run_id: str, variant: str) -> dict:
    run = orch.load_run(run_id)
    th = json.loads(json.dumps(orch.run_th(run)))
    if variant == "r28":
        for leaf in LEAVES:
            th["matching"][leaf] = json.loads(json.dumps(NOW[leaf]))
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
    rows = {"|".join([v.panel_id.value, *v.mark_ids]): (v.label.value, v.trace.rule_ids if v.trace else []) for v in vs}
    summary: dict[str, int] = {}
    for v in vs:
        summary[v.label.value] = summary.get(v.label.value, 0) + 1
    extra = {
        "copies": [n.model_dump(exclude_none=True) for pm in pms for n in pm.copies],
        "left_off": [i for pm in pms for i in pm.left_off],
        "sides": {k: s for pm in pms for k, s in pm.sides.items()},
        "by_eye": {pm.panel_id.value: [len(b.mark_ids) for b in pm.by_eye if b.kind == "debris"] for pm in pms if any(b.kind == "debris" for b in pm.by_eye)},
    }
    return {"rows": rows, "summary": summary, **extra}


report = {}
for run_id in RUNS:
    base, r28 = rebuild(run_id, "base"), rebuild(run_id, "r28")
    changed, lost = [], []
    for key in sorted(set(base["rows"]) | set(r28["rows"])):
        a, b = base["rows"].get(key), r28["rows"].get(key)
        if a != b:
            changed.append({"line": key, "base": a, "r28": b})
            if a is not None and a[0] in DECIDED and (b is None or b[0] != a[0]):
                lost.append(key)
    report[run_id] = {"base": base["summary"], "r28": r28["summary"], "changed": changed, "lost": lost, **{k: r28[k] for k in ("copies", "left_off", "sides", "by_eye")}, "base_by_eye": base["by_eye"]}
    print(json.dumps({"run": run_id, "base": base["summary"], "r28": r28["summary"], "changed": len(changed), "lost": lost, "copies": len(r28["copies"]), "left_off": len(r28["left_off"])}))
(HERE / "replay.json").write_text(json.dumps(report, indent=1) + "\n")
