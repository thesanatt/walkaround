"""The product's marks stage run again on a scratch copy of a stored run with the Nov 2 prompt switches, so the new
marks answers (reask_<run>.json, already paid for) go through the zoom refine and the SAM step as a new run's would.
LIVE: no marks call (derive_marks reads the stored answer for each crop sha256 and panel), one zoom call per box and the
SAM requests, logged under run id rr_<run>. The copy keeps the run's keyframes, crops and masks; stored looks,
counter-looks and second-pass asks are removed (keyed by mark ids that now name other marks). The rest is then
rebuilt offline by replay.py's variants on this copy (ROOT_RUN=rr_<run>).

Usage: LIVE=1 .venv/bin/python docs/evidence/hero_1102/refine_replay.py RUN_ID SCRATCH_DIR
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from walkaround import config, marks as marks_mod, paths  # noqa: E402
from walkaround import contracts as c  # noqa: E402
from walkaround.orchestrator import _runs_root  # noqa: E402

RUN, OUT = sys.argv[1], Path(sys.argv[2])
NEW = f"rr_{RUN}"
LEAVES = {"colour_on_paint": {"enabled": True}, "conditions_on_panel": {"enabled": True}}

if os.environ.get("LIVE") != "1":
    sys.exit("set LIVE=1 to make the zoom and SAM calls")
config.load_env()
th = json.loads((REPO / "runs" / RUN / "run.json").read_text())["thresholds"]
th["matching"].update(LEAVES)
th["dedup"]["spark_zoom_refine"]["prefer_kind"] = True
target = OUT / NEW
if target.exists():
    shutil.rmtree(target)
never = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", "events.pending.ndjson", "bundle", "clips", "chips")
shutil.copytree(REPO / "runs" / RUN, target, ignore=shutil.ignore_patterns(*never))
new = c.read_json_list(HERE / f"reask_{RUN}.json", c.SparkRequestRecord)
by_ref = {(r.input_refs[0].path, r.input_refs[0].sha256): r for r in new if r.valid and r.input_refs}
old = c.read_json_list(target / "spark.json", c.SparkRequestRecord)
kept = [r for r in old if not (r.purpose == c.SparkPurpose.MARKS and r.input_refs and (r.input_refs[0].path, r.input_refs[0].sha256) in by_ref)]
c.write_json(target / "spark.json", [*kept, *by_ref.values()])
for name in ("looks.json", "second_pass.json", "zoom_marks.json", "sam_marks.json"):
    for f in (target / name, target / "derived" / name):
        if f.exists():
            f.unlink()
run_json = json.loads((target / "run.json").read_text())
run_json["run_id"] = NEW
run_json["thresholds"] = th
(target / "run.json").write_text(json.dumps(run_json))
with _runs_root(OUT):
    kfs = c.read_json_list(paths.keyframes_json(NEW), c.Keyframe)
    for role in ("pickup", "return"):
        out, _, _ = asyncio.run(marks_mod.derive_marks(NEW, role, kfs, th=th))
        kfs = c.read_json_list(paths.keyframes_json(NEW), c.Keyframe)
        print(f"{NEW} {role}: {len(out)} marks")
print("done", NEW)
