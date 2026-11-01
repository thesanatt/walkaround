"""Offline, no call: replay a stored run's match with matching.counter_confirms_mark added to its frozen thresholds,
in a scratch runs folder, and list who the counter-look asks and the labels before and after.

Usage: .venv/bin/python docs/evidence/confirm_1101/offline_replay.py RUN [RUN ...]
Sockets are refused, so any attempt to call a model fails loudly. The stored counter rows are kind "counter" (the
one-question look), and a stored row keeps its kind on replay, so part 1 has no stored answer."""
import collections
import json
import shutil
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root
from walkaround import config, match, paths, second_look  # noqa: E402
from walkaround import contracts as c  # noqa: E402


def refuse(*a, **k):
    raise OSError("offline replay: no network")


socket.socket.connect = refuse  # type: ignore[method-assign]
src_root = paths.runs_root()
out = {}
for run in sys.argv[1:]:
    tmp = Path(tempfile.mkdtemp(prefix="confirm_offline_"))
    paths.RUNS_DIR = config.RUNS_DIR = tmp
    shutil.copytree(src_root / run, tmp / run, ignore=shutil.ignore_patterns("videos", "media", "clips", "sam", "sheet", "bundle"))
    manifest = c.read_json(tmp / run / "run.json", c.RunManifest)
    th = json.loads(json.dumps(manifest.thresholds))
    before = json.loads((tmp / run / "derived" / "verdicts.json").read_text())
    th["matching"]["counter_confirms_mark"] = True
    asked = []
    replay = second_look.replay_second_look(run, th=th)

    def counter(ask, _r=replay):
        asked.append((ask.mark_id, ask.confirm_mark))
        return _r.counter(ask)

    _, after, _ = match.derive_match(run, th=th, second_look=replay, counter_look=counter)
    rows = json.loads((tmp / run / "derived" / "looks.json").read_text())
    out[run] = {
        "asked": asked,
        "stored_counter_rows": collections.Counter(r.get("kind") for r in rows if r.get("kind")),
        "labels_before": dict(collections.Counter(v["label"] for v in before)),
        "labels_after": dict(collections.Counter(v.label.value for v in after)),
        "same_labels": [(v["verdict_id"], v["label"]) for v in before] == [(v.verdict_id, v.label.value) for v in after],
    }
    shutil.rmtree(tmp)
print(json.dumps(out, indent=1, default=str))
