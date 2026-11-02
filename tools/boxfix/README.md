# Box fix
Start: `.venv/bin/python tools/boxfix/serve.py [more_run_ids]` then open http://127.0.0.1:8790/ (default runs: pairC_core3, eval_nhtsa15759_r6, synth_pair2_r9; `--only` drops them, `--port` changes the port).
Two tabs at the top: Boxes (http://127.0.0.1:8790/) and Verdicts (http://127.0.0.1:8790/verdicts).

## Boxes
Drag inside the box to move it, drag a corner square to resize. Every change saves to eval/box_corrections/<run_id>.json.
One box per separate mark; a tight cluster of specks counts as one.
Keys: C correct (next), M moved (set by any drag), N not a mark (next), D missed one (then drag a new box), R reset, Left/Right back and next.
Z zooms on the box, G shows ground truth on synthetic runs, Delete removes a selected missed box, Esc cancels drawing.

## Verdicts
One item per row of the run's derived/verdicts.json. The pickup crop and the return crop sit side by side with this verdict's mark boxes in yellow (numbered when a side has more than one) and the product's other marks on that crop in thin white. The product's verdict is at the top in plain words: already there, new since pickup, needs your eye, not seen at return, not comparable. The line under it says why (rule names, what a needs your eye was lowered from, the match distance).
Keys: Y right (next), W wrong (then 1 to 5 picks what it should say, and moves on), S not sure (next), R reset, Left/Right back and next, F other frame of the panel (frames with this verdict's marks come first, then the paired keyframe, then the panel's other views), Z zooms on the marks.
Every change saves to eval/verdict_checks/<run_id>.json (atomic), one row per verdict id with the status, the product's label, the right label when wrong, the mark ids, rule ids and config hash.

## Report
`.venv/bin/python tools/boxfix/analyze.py` prints both parts.
Boxes: counts, offsets and IoU, writes eval/box_corrections/summary.json and the 12 worst moved boxes to eval/box_corrections/worst/.
Verdicts: per run and overall, checked, right, wrong and not sure by the product's label, wrong by the right label, and every product "new since pickup" (the costliest error) and "already there" a person called wrong; checks on verdicts that changed after a re-derive are flagged stale. Writes eval/verdict_checks/summary.json.
`--verdicts-only` skips the box part (which rewrites its summary and worst images).
