"""Ask the one-walk counter-look (K16) live on a non-kept copy of a stored run whose frozen thresholds get the two new
leaves (matching.one_walk_counter_look, matching.counter_looks_max), then replay it with no call and compare bytes.

Usage: LIVE=1 .venv/bin/python docs/evidence/counter_look_1101/counter_on_copy.py SOURCE_RUN COPY_RUN OUT_DIR
The copy is keep false (the sweeper deletes it); OUT_DIR gets its matches.json, verdicts.json, looks.json, the
look JPEGs as sent and the provenance lines. Only the counter-look and any pair look the copy's match asks are paid."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root
from walkaround import contracts as c, match, paths, second_look  # noqa: E402

src, dst, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
root = paths.runs_root()
if (root / dst).exists():
    sys.exit(f"{dst} exists; pick a new copy id")
shutil.copytree(root / src, root / dst)
for name in ("looks.json",):
    (root / dst / "derived" / name).unlink(missing_ok=True)
(root / dst / "open_weights.ndjson").unlink(missing_ok=True)
manifest = c.read_json(root / dst / "run.json", c.RunManifest)
th = json.loads(json.dumps(manifest.thresholds))
cur = c.load_thresholds()["matching"]
for leaf in ("one_walk_counter_look", "counter_looks_max"):
    th["matching"][leaf] = cur[leaf]
c.write_json(root / dst / "run.json", manifest.model_copy(update={"run_id": dst, "keep": False, "thresholds": th}))

look = second_look.make_second_look(dst, th=th)
_, verdicts, _ = match.derive_match(dst, th=th, second_look=look)
d = root / dst / "derived"
live = {n: (d / n).read_bytes() for n in ("matches.json", "verdicts.json")}
match.derive_match(dst, th=th, second_look=second_look.replay_second_look(dst, th=th))
same = all((d / n).read_bytes() == b for n, b in live.items())

out.mkdir(parents=True, exist_ok=True)
for n in ("matches.json", "verdicts.json", "looks.json"):
    if (d / n).is_file():
        shutil.copy2(d / n, out / f"{dst}_{n}")
if (root / dst / "open_weights.ndjson").is_file():
    shutil.copy2(root / dst / "open_weights.ndjson", out / f"{dst}_open_weights.ndjson")
rows = json.loads((d / "looks.json").read_text()) if (d / "looks.json").is_file() else []
for row in rows:
    for ref in row.get("input_refs", []):
        shutil.copy2(root / dst / ref["path"], out / Path(ref["path"]).name)
print("replay identical:", same)
for v in verdicts:
    if v.label != c.VerdictLabel.NOT_COMPARABLE:
        print(v.verdict_id, v.label.value, v.trace.rule_ids, v.mark_ids, v.counter_note)
for row in rows:
    print(row.get("kind"), row["pickup_mark"], row["return_mark"], row["verdict"], row["engine"], row["model"], row.get("key_slot"), row["request_id"], row["latency_ms"], row["reason"])
