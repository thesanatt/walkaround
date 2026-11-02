"""The part check (panels.part_check) on a stored run's tracks, per walk: every wheel, fender-lane and glass track with its
votes and the slot it would take (MOVED) or keep undecided (UNDEC). No call. Usage: partcheck_stored.py RUN_ID"""
import json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from walkaround import panels, contracts as c
run = sys.argv[1]
root = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "runs" / run)
th = c.load_thresholds()
tracks = c.read_json_list(__import__("pathlib").Path(root + "/derived/tracks.json"), c.PanelTrack)
for role in ("pickup", "return"):
    rt = [t.model_copy(deep=True) for t in tracks if t.role == role]
    res = panels.part_check(rt, th)
    for tid, (slot, rec) in sorted(res.items()):
        t = next(x for x in rt if x.track_id == tid)
        fr = [r.frame for r in t.per_frame]
        flag = "MOVED" if slot != rec.was else ("UNDEC" if not rec.decided else "")
        print(role[:3], tid, f"{min(fr)}-{max(fr)}", (rec.was.value if rec.was else None), "->", slot.value, rec.votes, round(rec.score, 1), flag)
