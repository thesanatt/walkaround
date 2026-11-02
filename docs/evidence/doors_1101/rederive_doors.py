"""Offline re-derive of a stored run's tracks, panels, keyframes and coverage into a scratch root, from the stored lanes
and the walk source files, with the panels leaves of config/thresholds.json the run's frozen thresholds lack. No call:
the keyframe blur is a stub, so the scratch copy is for measuring only and is never kept or shown.
VARIANT base leaves out panels.door_check and panels.edge_on_floor (today's code before this change); doors keeps them.
Usage: rederive_doors.py RUN_ID SCRATCH_ROOT VARIANT PICKUP_SRC RETURN_SRC"""
import asyncio, json, shutil, sys, logging
from pathlib import Path
from walkaround import pipeline, paths
from walkaround import contracts as c
from walkaround.orchestrator import _runs_root
logging.basicConfig(level=logging.WARNING)
SRC, OUT, VARIANT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
ORIG = dict(zip(('pickup', 'return'), sys.argv[4:6]))
orch = pipeline.make_orchestrator()
target = OUT / SRC
if target.exists(): shutil.rmtree(target)
skip = ('crops', 'keyframes', 'derived', 'clips', 'chips', 'bundle', 'sheet', 'claim', 'quarantine')
shutil.copytree(paths.run_dir(SRC), target, ignore=lambda d, names: [n for n in names if Path(d) == paths.run_dir(SRC) and n in skip])
(target / 'derived').mkdir(); (target / 'quarantine').mkdir()
for role, src in ORIG.items():
    (target / 'quarantine' / f'{role}.orig').symlink_to(Path(src).resolve())
rj = json.loads((target / 'run.json').read_text())
current = c.load_thresholds()['panels']
mine = ('door_check', 'edge_on_floor')
for leaf, value in current.items():
    if leaf not in rj['thresholds']['panels'] and (VARIANT == 'doors' or leaf not in mine):
        rj['thresholds']['panels'][leaf] = value
(target / 'run.json').write_text(json.dumps(rj))
async def stub_blur(run_id, role, rgb, **kw):
    return rgb, c.BlurRecord()
async def main():
    with _runs_root(OUT):
        orch._stored_blur = lambda run_id=None: stub_blur
        run = orch.load_run(SRC)
        for role in ORIG:
            await orch._derive_role(run, role)
asyncio.run(main())
print('done', target)
