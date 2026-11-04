"""Offline replay of a stored pair run with the 11/2 leaves (no model call): the run is copied into a scratch root, and
for the leaves that act after the frozen facts the product code is run on the copy.
  c1  panels.uncertain_per_track: panels.json re-derived from the stored tracks with panels._panels_from, keyframes take
      the new front_sign (as derive sets it), marks are re-whitened by rebuild_marks.
  c2  dedup.marks_on_panel: rebuild_marks with the leaf (marks.on_panel), then the match with the stored second looks.
  base, c1, c2, c12 (both). The part check (c3) re-files tracks, so it needs a re-derive of the keyframes and new Spark
  answers, and the second pass (c4) needs Spark calls: both are measured on the live re-run.
Usage: replay_1102.py RUN_ID SCRATCH VARIANT  (prints one line per verdict; JSON summary on the last line)"""
import json, shutil, sys
from pathlib import Path
from walkaround import contracts as c, panels, paths, pipeline
from walkaround.orchestrator import _runs_root

RUN, OUT, VARIANT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
current = c.load_thresholds()
orch = pipeline.make_orchestrator()
run = orch.load_run(RUN)
th = json.loads(json.dumps(orch.run_th(run)))
if VARIANT in ("c1", "c12"):
    th["panels"]["uncertain_per_track"] = True
if VARIANT in ("c2", "c12"):
    th["dedup"]["marks_on_panel"] = current["dedup"]["marks_on_panel"]
target = OUT / RUN
if target.exists():
    shutil.rmtree(target)
never = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", "events.pending.ndjson")
shutil.copytree(paths.run_dir(RUN), target, ignore=shutil.ignore_patterns(*never))
(OUT / "ledger.jsonl").write_text("")
with _runs_root(OUT):
    if VARIANT in ("c1", "c12"):
        tracks = c.read_json_list(paths.tracks_json(RUN), c.PanelTrack)
        old = {p.panel_id: p for p in c.read_json_list(paths.panels_json(RUN), c.Panel)}
        fps = run.fps or 15.0
        new = panels._panels_from(tracks, {}, fps=fps, merge_gap_s=float(th["panels"]["merge_gap_max_s"]), fallback_dt=1.0 / float(th["media"]["proxy_fps"]), per_track=True)
        out = []
        for p in new:  # keep everything the stored panel holds (votes, per-role coverage) but the two fields the leaf owns
            o = old[p.panel_id]
            out.append(o.model_copy(update={"side_uncertain": p.side_uncertain, "front_sign": p.front_sign}))
        c.write_json(paths.panels_json(RUN), out)
        sign = {p.panel_id: p.front_sign for p in out}
        kfs = [k.model_copy(update={"front_sign": sign[k.panel_id]}) if k.front_sign is None and sign[k.panel_id] is not None else k for k in c.read_json_list(paths.keyframes_json(RUN), c.Keyframe)]
        c.write_json(paths.keyframes_json(RUN), kfs)
        for p in out:
            if p.side_uncertain != old[p.panel_id].side_uncertain:
                print("PANEL", p.panel_id.value, "side_uncertain", old[p.panel_id].side_uncertain, "->", p.side_uncertain, "front_sign", p.front_sign)
    rd = orch._rebuild_in_place(run, th)
    vs = c.read_json_list(paths.verdicts_json(RUN), c.Verdict)
    ms = {m.mark_id: m for m in c.read_json_list(paths.marks_json(RUN), c.Mark)}
    looks = json.loads(paths.derived_dir(RUN).joinpath("looks.json").read_text()) if paths.derived_dir(RUN).joinpath("looks.json").is_file() else []
summary = {}
for v in vs:
    print("V", v.panel_id.value, v.label.value, ",".join(v.trace.rule_ids), " ".join(v.mark_ids), "" if v.distance_units is None else round(v.distance_units, 3))
    summary[v.label.value] = summary.get(v.label.value, 0) + 1
off = [m.mark_id for m in ms.values() if m.off_panel is not None]
print("OFF", " ".join(f"{m}->{ms[m].panel_id.value if ms[m].off_panel.moved_to_kf else 'left_out'}" for m in off))
print(json.dumps({"variant": VARIANT, "labels": summary, "off_panel": len(off), "moved": sum(1 for m in off if ms[m].off_panel.moved_to_kf)}))
