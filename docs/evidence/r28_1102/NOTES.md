# R28 replays (Mon Nov 2, 2026, no model call, $0)

`replay.py SCRATCH lexi_pair2 pairC_r16 eval_nhtsa15759_r16 hero_focus5` rebuilds each stored run on a scratch copy
with its frozen thresholds (base) and with the frozen thresholds plus the four R28 leaves as config/thresholds.json has
them (r28): matching.same_walk_copies, matching.faint_left_off, matching.side_from_track and matching.debris_by_eye with
any_view. Stored looks are replayed; nothing is asked. Full output: replay.json.

Already there / new / not seen at return / needs your eye / not comparable:

| run | base | r28 | lines changed | decided labels lost |
|---|---|---|---|---|
| lexi_pair2 | 2 / 0 / 0 / 11 / 9 | 2 / 0 / 0 / 7 / 9 | 5 | none |
| pairC_r16 | 2 / 0 / 1 / 17 / 12 | 2 / 0 / 1 / 12 / 12 | 6 | none |
| eval_nhtsa15759_r16 | 0 / 1 / 0 / 1 / 14 | same | 1 | none |
| hero_focus5 | 5 / 1 / 0 / 12 / 7 | 5 / 1 / 0 / 5 / 6 | 10 | none |

What changed:
- lexi_pair2 and pairC_r16 were frozen before debris_by_eye and stands_out, so the r28 variant brings the whole debris
  leaf: 4 and 5 one-walk dirt lines (all K4, K15) fold into debris lines (lexi hood 1 and rear window 3; pair C right
  rear fender 1 and trunk lid 4). No copies and nothing left off (no K20 in those runs).
- Mirrors: lexi_pair2's pickup mirror keyframes come from two tracks, one right and one left, so the pair tried fails
  G13 instead of G6 (still not comparable, also G1, G3, G4). pairC_r16 and eval_nhtsa15759_r16 lose G6 and stay not
  comparable on G1 (and G3, G4 for pair C).
- hero_focus5: see docs/inspect_hero_focus5b/NOTES.md in the workspace (the rederived run hero_focus5b gives the same
  17 lines).

Every line labelled already there, new or not seen at return in base keeps its label in r28 on all four runs.
