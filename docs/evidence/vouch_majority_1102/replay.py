"""Offline verdict replay (no model call) of stored runs under each run's frozen thresholds and with the two lead
decisions of R24 added, on scratch copies, by the product's rebuild (orchestrator._rebuild_in_place). Looks are
replayed from each run's looks.json; a majority look with fewer stored answers than it would ask counts the ones there.

Variants: base (frozen), vouch (+ matching.registration_vouches), majority (+ matching.look_majority), both, and with
--current the matching leaves config/thresholds.json has and the frozen file lacks (K20 on hero_focus2) added to each.

Usage: .venv/bin/python docs/evidence/vouch_majority_1102/replay.py SCRATCH_DIR RUN_ID [RUN_ID ...] [--out NAME]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from walkaround import contracts as c  # noqa: E402
from walkaround import paths, pipeline  # noqa: E402
from walkaround.orchestrator import _runs_root  # noqa: E402

sys.path.insert(0, str(HERE))
from truth import truth_rows  # noqa: E402

NEVER = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", "events.pending.ndjson", "bundle", "clips", "chips", "videos", "media")
CURRENT = c.load_thresholds()


def variant_th(frozen: dict, variant: str, current: bool) -> dict:
    th = json.loads(json.dumps(frozen))
    if current:
        for k, v in CURRENT["matching"].items():
            if k not in th["matching"] and k not in ("registration_vouches", "look_majority"):
                th["matching"][k] = json.loads(json.dumps(v))
    if variant in ("vouch", "both"):
        th["matching"]["registration_vouches"] = dict(CURRENT["matching"]["registration_vouches"])
    if variant in ("majority", "both"):
        th["matching"]["look_majority"] = dict(CURRENT["matching"]["look_majority"])
    return th


def rebuild(orch, scratch: Path, run_id: str, variant: str, current: bool) -> dict:
    run = orch.load_run(run_id)
    th = variant_th(orch.run_th(run), variant, current)
    root = scratch / variant
    target = root / run_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(paths.run_dir(run_id), target, ignore=shutil.ignore_patterns(*NEVER))
    (root / "ledger.jsonl").write_text("")
    with _runs_root(root):
        orch._rebuild_in_place(run, th)
        vs = c.read_json_list(paths.verdicts_json(run_id), c.Verdict)
        pms = c.read_json_list(paths.matches_json(run_id), c.PanelMatch)
    rows = {"|".join([v.panel_id.value, *v.mark_ids]): [v.label.value, v.trace.rule_ids] for v in vs}
    summary: dict[str, int] = {}
    for v in vs:
        summary[v.label.value] = summary.get(v.label.value, 0) + 1
    vouches = [{"panel": pm.panel_id.value, "kf_pair": t.kf_pair, "mask_iou": t.mask_iou, **t.vouch.model_dump(mode="json")} for pm in pms for t in pm.kf_pairs_tried if t.vouch is not None]
    return {"rows": rows, "summary": summary, "vouches": vouches, "truth": truth_rows(run_id, vs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scratch", type=Path)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--variants", default="base,vouch")
    ap.add_argument("--current", action="store_true")
    ap.add_argument("--out", default="replay.json")
    args = ap.parse_args()
    orch = pipeline.make_orchestrator()
    report = {}
    for run_id in args.runs:
        out = {v: rebuild(orch, args.scratch, run_id, v, args.current) for v in args.variants.split(",")}
        base = out[args.variants.split(",")[0]]
        for v, res in out.items():
            res["changed"] = [{"row": k, "base": base["rows"].get(k), v: res["rows"].get(k)} for k in sorted(set(base["rows"]) | set(res["rows"])) if base["rows"].get(k) != res["rows"].get(k)]
            print(json.dumps({"run": run_id, "variant": v, "summary": res["summary"], "changed": len(res["changed"]), "truth": [t["label"] for t in res["truth"]] or None}))
        report[run_id] = out
    (HERE / args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
