# Marks effort study, Nov 1 (stored crops only, no new run)

Question: does Muse Spark find more real marks at reasoning effort medium or high than at low, or with a second pass
on an about 2x zoom of each half of the crop, without more false boxes? Answer: no. Nothing in the product changed.

## What was asked

- 28 stored crops. 25 from synth_pair2_r9: of the 56 crops that hold a planted mark or one of the run's false boxes,
  one per (walk, panel) (rule in `select_synth`). 3 from pair C's D1 set (runs/d1/uco3d_pairC/spark_redo/): the
  pickup and return trunk chips and the hood dent, with D1's hand-measured target points (9 targets).
- The product's marks_v1 prompt and schema, fixtures sentence on, seams sentence off (synth_pair2_r9's thresholds).
- Effort low, medium and high, 2 repeats each. Zoom: each crop cut into two halves along its long side (55% each, a
  10% shared band), each half scaled up 2x and asked at low, 2 repeats. "low + zoom" is the variant as asked (the
  low answer plus the zoom boxes that do not overlap it); "zoom only" is the two halves alone.
- Scoring, synthetic: a box hits a planted mark when its centre is inside the mark's box grown to 1.5x, or IoU > 0.1
  (the synthetic scorer's rule). A false box hits no planted damage mark (the sticker control counts as false).
  Real: a target is hit when it lies inside a box grown by a quarter of its size (at least 10 px) or within 0.04 of
  the long edge of the box centre. A box that hits no target is "unmatched"; the targets are the known marks, so an
  unmatched box may still be an unlisted chip.
- 283 ledger rows under run id marks_effort_1101, $1.93. Answers: answers.jsonl; scores: study.json.

## Numbers (2 repeats; synth 50 answers, real 6 per arm)

All boxes:

| arm | synth hits | synth hits, marks >= 40 px | synth false boxes | real hits | real unmatched | $ per crop |
|---|---|---|---|---|---|---|
| low (product) | 3/70 | 3/38 | 52 | 9/18 | 31 | 0.0047 |
| medium | 5/70 | 4/38 | 45 | 6/18 | 26 | 0.0087 |
| high | 2/70 | 2/38 | 45 | 6/18 | 29 | 0.0107 |
| zoom only | 19/70 | 11/38 | 59 | 16/18 | 49 | 0.0104 |
| low + zoom | 19/70 | 11/38 | 110 | 18/18 | 79 | 0.0151 |

Damage-typed boxes only (scratch, scuff, chip, dent, crack; the product caps dirt, reflection and unsure):

| arm | synth hits | synth false boxes | real hits | real unmatched |
|---|---|---|---|---|
| low | 2/70 | 30 | 3/18 | 2 |
| medium | 3/70 | 30 | 4/18 | 2 |
| high | 1/70 | 24 | 4/18 | 4 |
| zoom only | 12/70 | 30 | 0/18 | 3 |
| low + zoom | 12/70 | 60 | 3/18 | 5 |

Box placement on synthetic hits (all boxes): median IoU with the planted mark and median centre offset in crop px:
low 0.24 / 20, medium 0.20 / 23, high 0.18 / 24, zoom only 0.31 / 13. Boxes that miss at 1.5x but land within 4x of
a planted mark ("beside"): low 9, medium 3, high 4, zoom only 9, low + zoom 18.

Sensitivity, chosen after looking at an overlay: the planted 20 cm scratch is about 3 px tall in the box truth, so
the 1.5x rule scores a box lying along it as false. With the mark's box grown by at least 12 crop px per side, all
boxes: low 5/70 hits and 50 false, zoom only 35/70 and 43 false, low + zoom 36/70 and 92 false. The headline keeps
the scorer's rule.

## What the answers show

- Effort: medium and high do not find more. Real hits drop (9 to 6 of 18), synthetic moves within noise (3, 5, 2 of
  70), false boxes drop a little (52 to 45). Medium's two hits on the 20 cm scratch (two crops) were both typed reflection.
- Zoom finds what the full crop misses on the synthetic car: the 12 cm dent (typed dent), the 1 cm chip on the left
  rear door and quarter (typed chip on 3 of 4 crops), and boxes along the 20 cm scratch. On pair C its boxes sit on more of the
  trunk chips, but every one of those boxes is typed dirt or unsure, and the hood dent, typed chip by low in both
  repeats, is typed dirt by zoom in both. So on real footage zoom adds no damage-typed find.
- The variant as asked (a second pass, unioned) doubles false boxes (52 to 110 synthetic, 31 to 79 unmatched real).

## Decision

The rule was: adopt only if real hits rise on both the synthetic and the real crops without more false boxes.
Higher effort does not raise real hits. Low + zoom raises hits on both but doubles false boxes. Zoom only raises
false boxes when every box counts, and drops real hits to 0 of 18 when only damage-typed boxes count. No threshold,
prompt or switch changed.

Limits: one synthetic render (dark blue paint, clean light), 3 real crops, 2 repeats, unmatched real boxes not
reviewed one by one, the sticker counted as false. Files: study.py (`LIVE=1 python study.py batch ARM EFFORT REP
PER_CALL_USD`, then `python study.py score`; no call without LIVE=1).
