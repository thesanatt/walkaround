# Halves merge on R18's stored answers (11/2): not adopted

Question (R18 open item 1): when a halves box and a full-crop box sit close once both have been moved by the zoom
refine and the SAM step, keep one mark instead of two. Does that stop the match's ratio test refusing real pairs, and
with a cap filter on the halves boxes that have no full-crop neighbour, does it find more real marks on lexi without
losing a right verdict or adding a false "new" or "already there"? Answer: the merge works where the two boxes are
close, but with the filter lexi finds no more marks (22 of 32, as the full crop alone), and M2's "already there" is
still lost because its halves box sits 2.34 patches from the full-crop box. The rule for adoption is not met, so the
merge is not switched on and no live run was made. No model call, $0, no LEDGER row.

## What the merge does (walkaround/halves_pass.py merge_after_refine; leaf key merge_patches, absent from config)

- After the zoom move and the SAM step of every box on the crop: a halves box is close to a full-crop box when the two
  boxes overlap or their centres are within merge_patches vision patches of the full crop (28 sent px; 28 crop px on
  these crops). Closest pairs first, one halves box per full-crop mark.
- A close pair becomes one mark: the full-crop mark's id, type, confidence and note, with the halves box (point,
  source box, size, K3 by size, and the zoom and SAM records that placed it). `Mark.halves_merge` keeps the halves
  mark's id, the full-crop mark's own moved box, the gap in patches and whether they overlapped. A second halves box
  close to the same full-crop mark is dropped.
- A halves box with no full-crop box close: dropped when it is the same mark as one kept (R18's same_mark), and with
  lone_uncapped dropped when it carries any cap (K3 small, K4 dirt, reflection or unsure, K12 low confidence).
- Variants scored: merge10 (1.0 patch, lone_uncapped), merge15 (1.5 patches, lone_uncapped). merge15_keep (1.5
  patches, lone boxes kept capped as in R18) is there for comparison only; it is not the proposal.
- Full-crop boxes are moved with stored answers as a new run would move them: lexi_pair1 its own run's zoom and SAM
  answers, synth_pair2_r9 R17's (zoom_refine_synth_1102), pairC_core3 R14's w1 window and R13's SAM answers on the moved
  boxes (zoom_refine_1101; 47 of 60 full-crop boxes moved by the zoom, 36 by SAM). Halves boxes use R18's answers.

## Recall against his labels (study.py, study.json; R18's rule, which reproduces R18's numbers for its two variants)

found / missed / false boxes, per crop he checked:

| run | real marks | full crop | + halves (R18) | + halves, R18 after_refine | merge 1.0, lone uncapped | merge 1.5, lone uncapped | merge 1.5, lone kept |
|---|---|---|---|---|---|---|---|
| lexi_pair1 (22 crops) | 32 | 22 / 10 / 13 | 25 / 7 / 41 | 25 / 7 / 40 | 22 / 10 / 13 | 22 / 10 / 13 | 25 / 7 / 39 |
| pairC_core3 (19 crops) | 48 | 48 / 0 / 5 | 48 / 0 / 53 | 48 / 0 / 51 | 48 / 0 / 5 | 48 / 0 / 5 | 48 / 0 / 51 |
| synth_pair2_r9 (36 crops) | 44 | 43 / 1 / 4 | 43 / 1 / 10 | 43 / 1 / 9 | 43 / 1 / 8 | 43 / 1 / 8 | 43 / 1 / 9 |

- Merged marks (merge 1.0 / 1.5): lexi 6 / 6, pairC 5 / 7, synth 23 / 24. Of the merged marks he labeled real, the
  final box is on his box (R18's hit rule) with the full crop's own moved box / with the halves box: lexi 5 / 5 of 5,
  pairC 6 / 7 of 7 at 1.5, synth 22 / 23 of 23 at 1.5. So the halves box is as well placed or a little better; on
  these crops the zoom and SAM had already brought most full-crop boxes onto the mark.
- R18's 3 lexi finds are all low confidence (K12), and they sit 13 to 28 patches from any full-crop box, so the merge
  never touches them and the cap filter drops them. That is why found stays 22. Any reach gives the same, so a reach
  between 1 and 1.5 patches cannot change it.
- Lone halves boxes kept with lone_uncapped: lexi 1 (on M2's strip, a duplicate of a found mark), pairC 1 (duplicate),
  synth 9 (5 duplicates, 4 false boxes).

## Verdict replay (replay.py; product rebuild on scratch copies with the stored answers, no call)

already there / new / not seen at return / needs your eye. A new pair with no stored second look keeps its caps.

| run | base | merge 1.0 | merge 1.5 | merge 1.5, lone kept |
|---|---|---|---|---|
| lexi_pair1 (R16 c12 leaves) | 2 / 0 / 0 / 6 | 1 / 0 / 0 / 9 | 1 / 0 / 0 / 9 | 1 / 0 / 0 / 31 |
| pairC_core3 (frozen full boxes) | 1 / 0 / 0 / 11 | 1 / 0 / 0 / 10 | 1 / 0 / 0 / 10 | 1 / 0 / 0 / 16 |
| pairC_core3 (R14 and R13 answers in) | 1 / 0 / 0 / 11 | 1 / 0 / 0 / 11 | 1 / 0 / 0 / 11 | 1 / 0 / 0 / 17 |
| synth_pair2_r9 (frozen full boxes) | 2 / 1 / 0 / 14 | 2 / 2 / 1 / 16 | 1 / 3 / 2 / 18 | 1 / 3 / 2 / 23 |
| synth_pair2_r9 (R17's answers in) | 0 / 1 / 2 / 20 | 0 / 1 / 2 / 22 | 0 / 1 / 2 / 22 | 0 / 1 / 2 / 23 |

- lexi: the merge keeps M1 (the pickup fender box and the halves box overlap; one mark, pairs as already there).
  M2 is lost: Spark's pickup box sits 65 crop px (2.34 patches) above the strip, the halves box h1 is on it (scratch,
  medium, so uncapped), both stay, and neither pairs with the return's strip box. merge10 and merge15 give the same
  lines on lexi.
- pairC: the trunk-lid "already there" stays in every variant. Decided lines unchanged.
- synth with frozen full boxes (Spark's raw boxes, not how a new run looks): merge 1.0 loses the second left front
  door chip pair's right "already there" (the return box takes a halves box and leaves the pair; needs your eye now),
  adds a false "not seen at return" (pickup kf3 chip) and a second "new" line for the real new scratch, and turns the
  right front fender pair into a right "already there" (as R18's union did). merge 1.5 breaks both chip pairs: a false
  "new" on the return's old chip and two false "not seen at return". With R17's answers in (the full boxes moved as a
  new run moves them) both merges keep base's decided lines and add 2 needs your eye.

Lexi per mark (replay, not live; the live re-run was conditional on adoption and was not made):

| mark | truth | base | merge 1.0 and 1.5, lone uncapped |
|---|---|---|---|
| M1 left rear fender strip | already there | already there | already there (merged mark) |
| M2 left rear door strip | already there | already there | needs your eye (two pickup boxes 2.34 patches apart, neither paired) |
| M3 hood edge strip | already there | needs your eye (no look stored) | same |
| M4 right rear fender pink square | already there | silent | silent (its halves boxes are typed reflection, K4, so dropped) |
| M5 trunk pink triangle | new | not comparable | same |
| M6 left rear door lower triangle | new | needs your eye (L2, K4 K12 K15) | same |
| M7 right rear door triangle | new | not comparable | same |

## Decision

Rule: real marks found rise on lexi, no right verdict lost, no new false "new" or "already there" on any of the
three. Found on lexi stays 22 of 32 at both reaches (the filter drops all 3 finds). M2's right "already there" is lost
at both reaches. On synth with frozen boxes both reaches lose a right "already there" and merge 1.5 adds a false
"new" (with the boxes moved as a new run would, synth keeps its lines). Not met, so the leaf key stays out of
config/thresholds.json and no live run was made. What the merge did show: within 1.5 patches it cures the crowding
(M1 back, lexi false boxes 41 to 13, pairC 53 to 5) at no loss of placement. What it cannot fix: a full-crop box more
than 2 patches off its mark (M2), and the filter that removes the false boxes also removes the low-confidence finds.

Files: study.py (recall), replay.py (verdicts), study.json, replay_<run>_<variant>.txt. Re-run with
`python docs/evidence/halves_merge_1102/study.py` and `python docs/evidence/halves_merge_1102/replay.py RUN SCRATCH
VARIANT` for each line of the tables (VARIANT base, merge10, merge15, merge15_keep, with _r17 or _r14 where named).
