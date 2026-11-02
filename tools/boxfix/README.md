# Box fix
Start: `.venv/bin/python tools/boxfix/serve.py [more_run_ids]` then open http://127.0.0.1:8790/ (default runs: pairC_core3, eval_nhtsa15759_r6, synth_pair2_r9; `--only` drops them, `--port` changes the port).
Drag inside the box to move it, drag a corner square to resize. Every change saves to eval/box_corrections/<run_id>.json.
Keys: C correct (next), M moved (set by any drag), N not a mark (next), D missed one (then drag a new box), R reset, Left/Right back and next.
Z zooms on the box, G shows ground truth on synthetic runs, Delete removes a selected missed box, Esc cancels drawing.
Report: `.venv/bin/python tools/boxfix/analyze.py` prints counts, offsets and IoU, writes summary.json and the 12 worst moved boxes to eval/box_corrections/worst/.
