"""Offline verdict replay (no model call) of a stored run with the Nov 2 switches, on scratch copies, by the product's
rebuild (orchestrator._rebuild_in_place) with the run's frozen thresholds plus the variant's leaves.

Variants: base (frozen thresholds), glass (matching.glass_by_eye), debris (matching.debris_by_eye), zoom
(dedup.spark_zoom_refine.prefer_kind), prompt (matching.colour_on_paint and conditions_on_panel, with the marks answers
of reask_<run>.json in spark.json and their conditions on keyframes.json), all (every one), reg (matching.registration,
as config has it for new runs; the r16 runs predate it), k19 (matching.inside_other_box) and control (the marks answers
of reask_old_<run>.json, the old prompt asked again the same day, with no prompt switch: the sampling control for
prompt), sw (matching.same_walk_register). Variants join with "+"; all+X adds X to all, and all+control swaps prompt for control. In prompt, control and
all the
stored second looks, counter-looks and second-pass asks are removed: they are keyed by mark ids that now name other
marks. So a new tight pair has no look (no K7 from a look, K4 and K12 cap as when never asked) and a new one-walk mark
no counter-look (no K16, K17): those lines are what a live run would start from, before its looks. New boxes have no
stored zoom or SAM answer, so they keep Spark's box.

With SRC=<refine_replay scratch root> and RUN_ID rr_<run>, the copy made by refine_replay.py (new answers, zoom and SAM
answers for the new boxes) is the source and no answer is swapped.

LOOKS=1 (LIVE) then asks the second looks and counter-looks a new run would ask, on that copy, before the verdicts are
read (output replay_<run>_<variant>_looks.json).

Usage: [SRC=DIR] [LOOKS=1] .venv/bin/python docs/evidence/hero_1102/replay.py RUN_ID VARIANT SCRATCH_DIR   -> replay_<run>_<variant>.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from walkaround import contracts as c  # noqa: E402
from walkaround import paths, pipeline, spark_client  # noqa: E402
from walkaround.orchestrator import _runs_root  # noqa: E402

RUN, VARIANT, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
LEAVES = {
    "glass": {"matching": {"glass_by_eye": {"enabled": True, "panels": ["windshield", "rear_window"]}}},
    "debris": {"matching": {"debris_by_eye": {"enabled": True, "types": ["dirt"]}}},
    "zoom": {"dedup": {"spark_zoom_refine": {"prefer_kind": True}}},
    "prompt": {"matching": {"colour_on_paint": {"enabled": True}, "conditions_on_panel": {"enabled": True}}},
    "reg": {"matching": {"registration": {"enabled": True}}},
    "k19": {"matching": {"inside_other_box": {"enabled": True}}},
    "sw": {"matching": {"same_walk_register": {"enabled": True}}},
    "control": {},
}
ALL = ["glass", "debris", "zoom", "prompt", "k19", "sw"]
parts = ALL if VARIANT == "all" else ([] if VARIANT == "base" else VARIANT.split("+"))
if VARIANT.startswith("all+"):
    parts = ALL + VARIANT.split("+")[1:]
if "control" in parts and "prompt" in parts:
    parts.remove("prompt")  # control: the old prompt asked again, so the same answers sampling without the switches


def merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge(dst[k], v)
        else:
            dst[k] = v


orch = pipeline.make_orchestrator()
run = c.read_json(Path(os.environ["SRC"]) / RUN / "run.json", c.RunManifest) if os.environ.get("SRC") else orch.load_run(RUN)
th = json.loads(json.dumps(orch.run_th(run)))
for p in parts:
    merge(th, json.loads(json.dumps(LEAVES[p])))
target = OUT / RUN
if target.exists():
    shutil.rmtree(target)
never = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", "events.pending.ndjson", "bundle", "clips", "chips")
SRC = os.environ.get("SRC")  # a refine_replay.py scratch root: RUN is rr_<run> there, its answers already in place
shutil.copytree(Path(SRC) / RUN if SRC else paths.run_dir(RUN), target, ignore=shutil.ignore_patterns(*never))
(OUT / "ledger.jsonl").write_text("")
if ("prompt" in parts or "control" in parts) and not SRC:
    new = c.read_json_list(HERE / (f"reask_{RUN}.json" if "prompt" in parts else f"reask_old_{RUN}.json"), c.SparkRequestRecord)
    by_ref = {(r.input_refs[0].path, r.input_refs[0].sha256): r for r in new if r.valid and r.input_refs}
    old = c.read_json_list(target / "spark.json", c.SparkRequestRecord)
    kept = [r for r in old if not (r.purpose == c.SparkPurpose.MARKS and r.input_refs and (r.input_refs[0].path, r.input_refs[0].sha256) in by_ref)]
    c.write_json(target / "spark.json", [*kept, *by_ref.values()])
    kfs = c.read_json_list(target / "derived" / "keyframes.json", c.Keyframe)
    out_kfs = []
    for kf in kfs:
        rec = by_ref.get((kf.crop.path, kf.crop.sha256)) if kf.crop is not None else None
        if rec is not None:
            ans = spark_client.marks_result_from_record(rec, orig_width=kf.crop.w, orig_height=kf.crop.h, landmarks=[]).answer
            kf = kf.model_copy(update={"conditions": ans.conditions, "marks_status": c.MarksStatus.COMPLETE})
        out_kfs.append(kf)
    c.write_json(target / "derived" / "keyframes.json", out_kfs)
    for name in ("looks.json", "second_pass.json"):
        for f in (target / name, target / "derived" / name):
            if f.exists():
                f.unlink()
with _runs_root(OUT):
    orch._rebuild_in_place(run, th)
    if os.environ.get("LOOKS") == "1":  # LIVE: the second looks and counter-looks a new run would ask, logged under RUN
        from walkaround import config, match, second_look

        config.load_env()
        match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))
    vs = c.read_json_list(paths.verdicts_json(RUN), c.Verdict)
    pms = c.read_json_list(paths.matches_json(RUN), c.PanelMatch)
    ms = {m.mark_id: m for m in c.read_json_list(paths.marks_json(RUN), c.Mark)}
rows = [
    {
        "verdict_id": v.verdict_id, "panel": v.panel_id.value, "label": v.label.value, "rule_ids": v.trace.rule_ids, "mark_ids": v.mark_ids,
        "types": [f"{ms[i].type.value}/{ms[i].confidence.value}" for i in v.mark_ids if i in ms],
        "notes": [ms[i].description for i in v.mark_ids if i in ms], "boxes": [ms[i].bbox_crop for i in v.mark_ids if i in ms], "d": v.distance_units,
    }
    for v in vs
]
by_eye = [{"panel": pm.panel_id.value, "kind": b.kind, "text": b.text, "n": len(b.mark_ids), "mark_ids": b.mark_ids} for pm in pms for b in pm.by_eye]
summary: dict[str, int] = {}
for r in rows:
    summary[r["label"]] = summary.get(r["label"], 0) + 1
summary["by_eye_lines"] = len(by_eye)
(HERE / f"replay_{RUN}_{VARIANT}{'_looks' if os.environ.get('LOOKS') == '1' else ''}.json").write_text(json.dumps({"run": RUN, "variant": VARIANT, "summary": summary, "verdicts": rows, "by_eye": by_eye}, indent=1) + "\n")
print(json.dumps({"run": RUN, "variant": VARIANT, **summary}))
