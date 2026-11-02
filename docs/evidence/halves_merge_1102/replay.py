"""Offline verdict replay (no model call) of a stored pair with a halves variant, on a scratch copy.

R18's replay (halves_1102/replay.py) with the variants of study.py. The stored halves, zoom and SAM answers go into the
copy as the run's frozen facts, then the product rebuild runs in place (marks, narration, Stays Put, match with the
stored second looks, sheet). A pair or counter-look with no stored answer replays as not asked (caps stay), so a new
tight pair can only end at needs your eye here; a live run would ask it.

  base     as R18: lexi_pair1 with R16's panels.uncertain_per_track and dedup.marks_on_panel; the other two with today's
           zoom refine and SAM step leaves.
  VARIANT  base plus dedup.spark_halves_pass with study.VARIANTS[VARIANT].
  _r17     synth_pair2_r9 only: R17's zoom and SAM answers for its full-crop marks go in too.
  _r14     pairC_core3 only: R14's w1 zoom answers and R13's SAM answers on the moved boxes go in too.

Usage: replay.py RUN_ID SCRATCH VARIANT   (one line per verdict; JSON summary on the last line)
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study  # noqa: E402

r18 = study.r18
from walkaround import box_refine, halves_pass, panels, paths, pipeline, zoom_refine  # noqa: E402
from walkaround import contracts as c  # noqa: E402
from walkaround.orchestrator import _runs_root  # noqa: E402

RUN, OUT, VARIANT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
EXTRA = VARIANT.endswith(("_r17", "_r14"))
NAME = VARIANT
VARIANT = VARIANT.removesuffix("_r17").removesuffix("_r14")
current = c.load_thresholds()
orch = pipeline.make_orchestrator()
run = orch.load_run(RUN)
th = json.loads(json.dumps(orch.run_th(run)))
lexi = RUN.startswith("lexi")
if lexi:
    th["panels"]["uncertain_per_track"] = True
    th["dedup"]["marks_on_panel"] = current["dedup"]["marks_on_panel"]
else:
    for leaf in ("spark_zoom_refine", "sam_box_refine"):
        th["dedup"].setdefault(leaf, current["dedup"][leaf])
if VARIANT != "base":
    th["dedup"]["spark_halves_pass"] = {**r18.LEAF, **study.VARIANTS[VARIANT]}
target = OUT / RUN
if target.exists():
    shutil.rmtree(target)
never = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", "events.pending.ndjson")
shutil.copytree(paths.run_dir(RUN), target, ignore=shutil.ignore_patterns(*never))
(OUT / "ledger.jsonl").write_text("")
with _runs_root(OUT):
    kf_paths = {k.crop.path for k in c.read_json_list(paths.keyframes_json(RUN), c.Keyframe) if k.crop is not None}
    if VARIANT != "base":
        halves_pass.write_asks(RUN, [a for a in r18.stored("halves") if a.crop_path in kf_paths])
        zoom_refine.write_asks(RUN, [*zoom_refine.read_asks(RUN), *[a for a in r18.stored("zoom") if a.crop_path in kf_paths]])
        box_refine.write_asks(RUN, [*box_refine.read_asks(RUN), *[a for a in r18.stored("sam") if a.crop_path in kf_paths]])
    if EXTRA:
        zoom_refine.write_asks(RUN, [*zoom_refine.read_asks(RUN), *study.extra_zoom(RUN)])
        box_refine.write_asks(RUN, [*box_refine.read_asks(RUN), *study.extra_sam(RUN)])
    if lexi:  # replay_1102 c1: panels re-derived from the stored tracks with per-track uncertainty
        tracks = c.read_json_list(paths.tracks_json(RUN), c.PanelTrack)
        old = {p.panel_id: p for p in c.read_json_list(paths.panels_json(RUN), c.Panel)}
        new = panels._panels_from(tracks, {}, fps=run.fps or 15.0, merge_gap_s=float(th["panels"]["merge_gap_max_s"]), fallback_dt=1.0 / float(th["media"]["proxy_fps"]), per_track=True)
        out = [old[p.panel_id].model_copy(update={"side_uncertain": p.side_uncertain, "front_sign": p.front_sign}) for p in new]
        c.write_json(paths.panels_json(RUN), out)
        sign = {p.panel_id: p.front_sign for p in out}
        kfs = [k.model_copy(update={"front_sign": sign[k.panel_id]}) if k.front_sign is None and sign[k.panel_id] is not None else k for k in c.read_json_list(paths.keyframes_json(RUN), c.Keyframe)]
        c.write_json(paths.keyframes_json(RUN), kfs)
    orch._rebuild_in_place(run, th)
    vs = c.read_json_list(paths.verdicts_json(RUN), c.Verdict)
    ms = {m.mark_id: m for m in c.read_json_list(paths.marks_json(RUN), c.Mark)}
summary: dict[str, int] = {}
for v in vs:
    tag = "H" if any(ms.get(i) is not None and ms[i].halves is not None for i in v.mark_ids) else ("M" if any(ms.get(i) is not None and ms[i].halves_merge is not None for i in v.mark_ids) else "-")
    types = ",".join(f"{ms[i].type.value}/{ms[i].confidence.value}" for i in v.mark_ids if i in ms)
    print("V", tag, v.panel_id.value, v.label.value, v.rule_id, ",".join(v.trace.rule_ids), "notes:" + ",".join(v.trace.note_ids), " ".join(v.mark_ids), types,
          "" if v.distance_units is None else round(v.distance_units, 3))
    summary[v.label.value] = summary.get(v.label.value, 0) + 1
print(json.dumps({"run": RUN, "variant": NAME, "labels": summary, "halves_marks": sum(1 for m in ms.values() if m.halves is not None),
                  "merged_marks": sum(1 for m in ms.values() if m.halves_merge is not None), "marks": len(ms)}))
