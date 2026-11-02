"""Offline replay (no model call) of a stored pair with and without the halves pass, on a scratch copy.

The study's stored answers are put into the copy as the run's own frozen facts: halves_answers.json rows of this run's
crops as halves_marks.json, and the study's zoom and SAM rows appended to zoom_marks.json and sam_marks.json. Then the
product rebuild runs in place on the copy (marks, narration, Stays Put, match with the stored second looks, sheet).
A pair or counter-look with no stored answer replays as not asked (no K7, caps stay), so a new tight pair can only
end at needs your eye here; a live run would ask it.

  base   lexi_pair1: the run's thresholds plus R16's panels.uncertain_per_track and dedup.marks_on_panel (replay_1102
         c12). synth_pair2_r9 and pairC_core3: the run's thresholds plus today's zoom refine and SAM step leaves (their
         full-crop boxes have no stored zoom or SAM answer, so they keep Spark's box, as the frozen run did).
  halves the same plus dedup.spark_halves_pass (halves stored for every answered lexi crop; for the other two runs only
         on the crops Sanat checked).

Usage: replay.py RUN_ID SCRATCH VARIANT   (one line per verdict; JSON summary on the last line)
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study  # noqa: E402

from walkaround import box_refine, halves_pass, panels, paths, pipeline, zoom_refine  # noqa: E402
from walkaround import contracts as c  # noqa: E402
from walkaround.orchestrator import _runs_root  # noqa: E402

RUN, OUT, VARIANT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
R17 = VARIANT.endswith("_r17")  # synth only: R17's stored zoom and SAM answers for its 47 full-crop marks go in too
VARIANT = VARIANT.removesuffix("_r17")
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
if VARIANT in ("halves", "halves2"):
    th["dedup"]["spark_halves_pass"] = dict(study.LEAF)
if VARIANT == "halves2":  # the same answers; a halves mark that lands on a kept mark after the zoom move and SAM is dropped
    th["dedup"]["spark_halves_pass"]["after_refine"] = True
target = OUT / RUN
if target.exists():
    shutil.rmtree(target)
never = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", "events.pending.ndjson")
shutil.copytree(paths.run_dir(RUN), target, ignore=shutil.ignore_patterns(*never))
(OUT / "ledger.jsonl").write_text("")
with _runs_root(OUT):
    kf_paths = {k.crop.path for k in c.read_json_list(paths.keyframes_json(RUN), c.Keyframe) if k.crop is not None}
    if VARIANT in ("halves", "halves2"):
        halves_pass.write_asks(RUN, [a for a in study.stored("halves") if a.crop_path in kf_paths])
        zoom_refine.write_asks(RUN, [*zoom_refine.read_asks(RUN), *[a for a in study.stored("zoom") if a.crop_path in kf_paths]])
        box_refine.write_asks(RUN, [*box_refine.read_asks(RUN), *[a for a in study.stored("sam") if a.crop_path in kf_paths]])
    if R17:
        r17 = HERE.parent / "zoom_refine_synth_1102"
        zoom_refine.write_asks(RUN, [*zoom_refine.read_asks(RUN), *c.read_json_list(r17 / "zoom_answers.json", zoom_refine.ZoomAsk)])
        box_refine.write_asks(RUN, [*box_refine.read_asks(RUN), *c.read_json_list(r17 / "sam_zoom.json", box_refine.BoxAsk)])
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
    tag = "H" if any(ms.get(i) is not None and ms[i].halves is not None for i in v.mark_ids) else "-"
    types = ",".join(f"{ms[i].type.value}/{ms[i].confidence.value}" for i in v.mark_ids if i in ms)
    print("V", tag, v.panel_id.value, v.label.value, v.rule_id, ",".join(v.trace.rule_ids), "notes:" + ",".join(v.trace.note_ids), " ".join(v.mark_ids), types,
          "" if v.distance_units is None else round(v.distance_units, 3))
    summary[v.label.value] = summary.get(v.label.value, 0) + 1
n_h = sum(1 for m in ms.values() if m.halves is not None)
print(json.dumps({"run": RUN, "variant": VARIANT + ("_r17" if R17 else ""), "labels": summary, "halves_marks": n_h, "marks": len(ms)}))
