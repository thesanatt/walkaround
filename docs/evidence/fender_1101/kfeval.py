"""The view gates (G3, G4, G9, plus side_uncertain for G6) on every keyframe pair of the named panels of a (scratch)
run, and filmed slots per walk. Usage: kfeval.py ROOT RUN_ID [PANEL,PANEL,...]"""
import json, sys, itertools
from pathlib import Path
from walkaround import paths, match, geometry
from walkaround import contracts as c
from walkaround.orchestrator import _runs_root
ROOT, RUN = Path(sys.argv[1]), sys.argv[2]
PANELS = sys.argv[3].split(',') if len(sys.argv) > 3 else None
with _runs_root(ROOT):
    d = paths.derived_dir(RUN)
    K = [c.Keyframe.model_validate(k) for k in json.loads((d / 'keyframes.json').read_text())]
    cov = json.loads((d / 'coverage.json').read_text())['per_role']
    P = {p['panel_id']: p for p in json.loads((d / 'panels.json').read_text())}
    th = json.loads((paths.run_dir(RUN) / 'run.json').read_text())['thresholds']
    m = th['matching']
    pids = PANELS or sorted({k.panel_id.value for k in K} | {p for r in cov.values() for p in r})
    nfilm = {r: sum(1 for v in cov[r].values() if v['state'] == 'filmed') for r in cov}
    print('filmed slots', nfilm)
    comparable_view = 0
    for pid in pids:
        st = {r: cov[r].get(pid, {}).get('state', '-')[:4] for r in ('pickup', 'return')}
        kp = [k for k in K if k.role == 'pickup' and k.panel_id.value == pid]
        kr = [k for k in K if k.role == 'return' and k.panel_id.value == pid]
        best = None
        for a, b in itertools.product(kp, kr):
            iou = match._mask_iou(RUN, a, b)
            g = []
            if a.score.sharpness < m['sharpness_rel_min'] or b.score.sharpness < m['sharpness_rel_min']: g.append('G3')
            if iou is None or iou < m['mask_iou_min']: g.append('G4')
            if a.window_subtracted != b.window_subtracted: g.append('G9')
            key = (len(g), -(iou or 0))
            if best is None or key < best[0]: best = (key, a.f_proxy, b.f_proxy, round(iou or 0, 2), g)
        unc = P.get(pid, {}).get('side_uncertain')
        if best and not best[4] and not unc and all(s == 'film' for s in st.values()): comparable_view += 1
        print(f"{pid:22s} {st['pickup']}/{st['return']} kf {[k.f_proxy for k in kp]}/{[k.f_proxy for k in kr]} unc={unc} best {best[1:] if best else '-'}")
    print('panels passing G1,G3,G4,G6,G9 on some pair:', comparable_view)
