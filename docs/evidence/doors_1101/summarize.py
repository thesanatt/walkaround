"""Door keyframes, door check decisions and door coverage of the stored runs and their scratch re-derives.
Usage: summarize.py SCRATCH_BASE SCRATCH_DOORS OUT_JSON RUN_ID..."""
import json, sys
from pathlib import Path
from walkaround import contracts as c
from walkaround.contracts import Noun
BASE, DOORS, OUT, RUNS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
LATERAL = ('door', 'wheel', 'fender')
def kfs(root):
    out = []
    for k in c.read_json_list(root / 'derived/keyframes.json', c.Keyframe):
        if any(s in k.panel_id.value for s in LATERAL):
            out.append({'kf': k.kf_id, 'track': k.track_id, 'frame': k.f_proxy, 'aspect': round(k.aspect_ratio, 2),
                        'crop': f'{k.crop.w}x{k.crop.h}' if k.crop else None})
    return out
def doors(root):
    out = []
    for t in c.read_json_list(root / 'derived/tracks.json', c.PanelTrack):
        if t.noun is Noun.CAR_DOOR:
            d = t.door_check
            fr = [r.frame for r in t.per_frame]
            out.append({'track': t.track_id, 'frames': f'{min(fr)}-{max(fr)}', 'slot': t.panel_id.value, 'side': t.side.value,
                        'position': t.position.value, 'was': d.was.value if d and d.was else None,
                        'votes': {k: v for k, v in d.votes.items() if v} if d else None,
                        'side_score': d.side_score if d else None, 'position_score': d.position_score if d else None,
                        'edge_on_frames': t.edge_on_frames})
    return out
def cov(root):
    cv = c.read_json(root / 'derived/coverage.json', c.Coverage)
    return {role: {pid.value: f'{pc.state.value}: {pc.reason}' for pid, pc in per.items() if any(s in pid.value for s in LATERAL)}
            for role, per in cv.per_role.items()}
report = {}
for run in RUNS:
    report[run] = {'stored': {'keyframes': kfs(Path('runs') / run)}, 'base': {'keyframes': kfs(BASE / run)},
                   'doors': {'keyframes': kfs(DOORS / run), 'door_tracks': doors(DOORS / run), 'coverage': cov(DOORS / run)}}
OUT.write_text(json.dumps(report, indent=1) + '\n')
for run, r in report.items():
    print('==', run)
    for v in ('stored', 'base', 'doors'):
        print(' ', v, '; '.join(f"{k['kf'].replace(k['kf'].split('_')[0] + '_', '')} {k['track'].split('_')[-1]}@{k['frame']} a{k['aspect']}" for k in r[v]['keyframes'] if 'door' in k['kf']))
