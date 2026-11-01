"""Ask the two-question counter-look (lead decision 5 of 11/1) live on a non-kept copy of a stored run whose frozen
thresholds get matching.counter_confirms_mark, then replay it with no call and compare bytes.

Usage: LIVE=1 .venv/bin/python docs/evidence/confirm_1101/confirm_on_copy.py SOURCE_RUN COPY_RUN OUT_DIR
The copy keeps the source's looks.json, so a pair look whose question is unchanged is reused, not paid; the stored
one-question counter rows are a different question (kind "counter"), so each counter-look is asked again as
"counter2". OUT_DIR gets the copy's verdicts.json and looks.json, the look JPEGs as sent and a summary."""
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
shutil.copytree(root / src, root / dst, ignore=shutil.ignore_patterns("videos", "media", "clips", "sam", "sheet", "bundle"))
(root / dst / "open_weights.ndjson").unlink(missing_ok=True)
manifest = c.read_json(root / dst / "run.json", c.RunManifest)
th = json.loads(json.dumps(manifest.thresholds))
th["matching"]["counter_confirms_mark"] = True
c.write_json(root / dst / "run.json", manifest.model_copy(update={"run_id": dst, "keep": False, "thresholds": th}))
before = {v["verdict_id"]: v for v in json.loads((root / src / "derived" / "verdicts.json").read_text())}

look = second_look.make_second_look(dst, th=th)
_, verdicts, _ = match.derive_match(dst, th=th, second_look=look)
d = root / dst / "derived"
live = {n: (d / n).read_bytes() for n in ("matches.json", "verdicts.json")}
match.derive_match(dst, th=th, second_look=second_look.replay_second_look(dst, th=th))
same = all((d / n).read_bytes() == b for n, b in live.items())

out.mkdir(parents=True, exist_ok=True)
for n in ("verdicts.json", "looks.json"):
    shutil.copy2(d / n, out / f"{dst}_{n}")
if (root / dst / "open_weights.ndjson").is_file():
    shutil.copy2(root / dst / "open_weights.ndjson", out / f"{dst}_open_weights.ndjson")
rows = json.loads((d / "looks.json").read_text())
for row in rows:
    if row.get("kind"):
        for ref in row.get("input_refs", []):
            shutil.copy2(root / dst / ref["path"], out / f"{dst}__{Path(ref['path']).name}")
summary = {
    "replay_identical": same,
    "counter2": [
        {k: row.get(k) for k in ("mark_id", "mark_seen", "verdict", "engine", "model", "key_slot", "request_id", "latency_ms", "reason")}
        for row in rows
        if row.get("kind") == "counter2"
    ],
    "changed": [
        {"verdict_id": v.verdict_id, "marks": v.mark_ids, "before": [before[v.verdict_id]["label"], before[v.verdict_id]["trace"]["rule_ids"]], "after": [v.label.value, v.trace.rule_ids], "note": v.counter_note}
        for v in verdicts
        if v.verdict_id in before and (before[v.verdict_id]["label"] != v.label.value or before[v.verdict_id]["trace"]["rule_ids"] != v.trace.rule_ids)
    ],
}
(out / f"{dst}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
