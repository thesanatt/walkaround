"""Rebuild a kept run into a scratch dir with its frozen thresholds plus today's 11/1 (second) leaves, no network.
Prints every panel that failed G4, its tries, the reshape record and its verdict."""
import json, socket, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root


def _refuse(*a, **k):
    raise RuntimeError("network refused in replay")


socket.socket.connect = _refuse  # type: ignore[assignment]
from walkaround import contracts as c, paths, pipeline  # noqa: E402

run_id, out = sys.argv[1], Path(sys.argv[2])
orch = pipeline.make_orchestrator()
run = c.read_json(paths.runs_root() / run_id / "run.json", c.RunManifest)
th = json.loads(json.dumps(orch.run_th(run)))
cur = c.load_thresholds()["matching"]
for k in ("tight_pair_k5", "reshaped_panel", "reshape_iou_min", "reshape_aspect_tol", "reshape_open_px", "reshape_blob_share_min"):
    th["matching"][k] = cur[k]
result, where = pipeline.rebuild(run_id, out=out, thresholds=th, orch=orch)
d = out / run_id / "derived"
matches = json.loads((d / "matches.json").read_text())
verdicts = json.loads((d / "verdicts.json").read_text())
from walkaround import match  # noqa: E402

for pm in matches:
    tries = pm.get("kf_pairs_tried") or []
    if "G4" not in pm["reasons"] and not pm.get("reshape"):
        continue
    print(f"== {pm['panel_id']} state {pm['state']} reasons {pm['reasons']}")
    print("   how:", match.how_checked(c.PanelMatch.model_validate(pm)))
    for v in verdicts:
        if v["panel_id"] == pm["panel_id"]:
            print(f"   verdict {v['verdict_id']} {v['label']} {v['rule_id']} {v['trace']['rule_ids']} marks {v['mark_ids']}")
stored = {v["verdict_id"]: (v["label"], v["rule_id"]) for v in json.loads((paths.runs_root() / run_id / "derived" / "verdicts.json").read_text())}
new = {v["verdict_id"]: (v["label"], v["rule_id"]) for v in verdicts}
print("verdicts changed vs stored:", {k: (stored.get(k), new.get(k)) for k in sorted(set(stored) | set(new)) if stored.get(k) != new.get(k)})
