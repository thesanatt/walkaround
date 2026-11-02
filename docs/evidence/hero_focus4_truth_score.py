# Box-level check of hero_focus4 against the hand labels on hero_focus1 (crops from the same frame). No call.
# .venv/bin/python docs/evidence/hero_focus4_truth_score.py
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'runs'
def kfs(r):
    k = json.load(open(R/r/'derived/keyframes.json'))
    return {x['kf_id']: x for x in (k if isinstance(k, list) else k.get('keyframes', k))}
A, B = kfs('hero_focus1'), kfs('hero_focus4')
marks = json.load(open(R/'hero_focus4/derived/marks.json'))
marks = marks if isinstance(marks, list) else marks.get('marks', marks)
verd = json.load(open(R/'hero_focus4/derived/verdicts.json'))
lab = {mid: (v['verdict_id'], v['label']) for v in verd for mid in v['mark_ids']}
sb = json.load(open(ROOT / 'eval/box_corrections/hero_focus1.json'))
seen = set(sb['crops_seen'])
def to_src(kf, b):
    c = kf['crop']; s = c['scale']
    return [c['dx'] + b[0]/s, c['dy'] + b[1]/s, c['dx'] + b[2]/s, c['dy'] + b[3]/s]
def to_crop(kf, b):
    c = kf['crop']; s = c['scale']
    return [(b[0]-c['dx'])*s, (b[1]-c['dy'])*s, (b[2]-c['dx'])*s, (b[3]-c['dy'])*s]
def hit(a, b, pad=6):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])
rows = []
tot = dict(true=0, found=0, h4=0, on_true=0, on_not=0, other=0)
for kid, kf in sorted(B.items()):
    pair = [a for a in A.values() if a['role'] == kf['role'] and a['panel_id'] == kf['panel_id'] and abs(a['pts'] - kf['pts']) < 0.01 and a['kf_id'] in seen]
    if not pair: continue
    akf = pair[0]
    same = akf['crop']['sha256'] == kf['crop']['sha256']
    true, nots = [], []
    for mid, m in sb['marks'].items():
        if m['kf_id'] != akf['kf_id']: continue
        bb = m['box'] or m['product_box']
        if not bb: continue
        c = to_crop(kf, to_src(akf, bb))
        (true if m['status'] in ('correct', 'moved') else nots).append((mid, c))
    for m in sb['missed']:
        if m['kf_id'] == akf['kf_id']:
            true.append((m['id'], to_crop(kf, to_src(akf, m['box']))))
    h4 = [m for m in marks if m['kf_id'] == kid]
    found = [t for t in true if any(hit(t[1], m['bbox_crop']) for m in h4)]
    det = []
    for m in h4:
        if any(hit(m['bbox_crop'], t[1]) for t in true): k = 'real'; tot['on_true'] += 1
        elif any(hit(m['bbox_crop'], n[1]) for n in nots): k = 'not_mark'; tot['on_not'] += 1
        else: k = 'unlabeled'; tot['other'] += 1
        det.append((m['mark_id'].split('_')[-1], k, lab.get(m['mark_id'], ('-', '-'))[1]))
    tot['true'] += len(true); tot['found'] += len(found); tot['h4'] += len(h4)
    print(f"{kid} (hf1 {akf['kf_id']}, {'same crop' if same else 'same frame'}): Sanat real {len(true)}, hf4 boxed {len(found)}; hf4 boxes: {det}")
print(tot)
