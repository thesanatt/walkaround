# Box fix
Start: `.venv/bin/python tools/boxfix/serve.py [more_run_ids]` then open http://127.0.0.1:8790/ (default runs: pairC_core3, eval_nhtsa15759_r6, synth_pair2_r9; `--only` drops them, `--port` changes the port).
Two tabs at the top: Boxes (http://127.0.0.1:8790/) and Verdicts (http://127.0.0.1:8790/verdicts).

## Boxes
Drag inside the box to move it, drag a corner square to resize. Every change saves to eval/box_corrections/<run_id>.json.
One box per separate mark; a tight cluster of specks counts as one.
Keys: C correct (next), M moved (set by any drag), N not a mark (next), D missed one (then drag a new box), R reset, Left/Right back and next.
Z zooms on the box, G shows ground truth on synthetic runs, Delete removes a selected missed box, Esc cancels drawing.

## Verdicts
One item per row of the run's derived/verdicts.json. The main view is two photos, Pickup on the left and Return on the right, with this verdict's mark boxes in yellow (numbered when a side has more than one). Above them is one plain sentence with the panel name and the verdict (already there, new since pickup, not seen at return, needs your eye, not comparable).
Three big buttons: Right (Y), Wrong (W), Can't tell (S). Wrong shows the five verdicts in plain words to pick the right one (keys 1 to 5), then moves on. Other keys: R reset, Left/Right back and next, I details, Z zoom on the marks (zoom button is inside details).
Details (closed by default, key I) shows the rule ids and the why line, the match distance, keyframe ids, the other-frame browser (F steps through frames with this verdict's marks first, then the paired keyframe, then the panel's other views), the mark list per side, the product's other marks in thin white, and the run picker.
Every change saves to eval/verdict_checks/<run_id>.json (atomic), one row per verdict id with the status, the product's label, the right label when wrong, the mark ids, rule ids and config hash. The saved format only gained the fields below.
Notes (both tabs): the box under the buttons saves as "note" on that item's row (under "pending" until you judge the item; Reset keeps it) and shows again when you come back. Keys are off while you type; Esc leaves the box.
Missed marks (Verdicts): Draw missed mark or D, then drag on the Pickup or Return photo. Each pink box saves with the verdict under "missed" as side, kf_id, box in that crop's pixels and an optional note; click one to select it, add a note, remove it with Delete or its x. analyze.py lists every note and counts these per side.

## Report
`.venv/bin/python tools/boxfix/analyze.py` prints both parts.
Boxes: counts, offsets and IoU, writes eval/box_corrections/summary.json and the 12 worst moved boxes to eval/box_corrections/worst/.
Verdicts: per run and overall, checked, right, wrong and not sure by the product's label, wrong by the right label, and every product "new since pickup" (the costliest error) and "already there" a person called wrong; checks on verdicts that changed after a re-derive are flagged stale. Writes eval/verdict_checks/summary.json.
`--verdicts-only` skips the box part (which rewrites its summary and worst images).
