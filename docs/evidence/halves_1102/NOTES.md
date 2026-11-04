# Halves recall pass on stored crops (11/2): not adopted

Question: if Muse Spark's marks call on the full crop is joined by the same call on two zoomed halves of the crop
(R10's variant), does it find more real marks without adding false verdicts? Answer: a little more, at a high cost
in false boxes. The replay also loses correct "already there" lines, so the leaf is not adopted.
`dedup.spark_halves_pass` is not in config/thresholds.json, so new runs do not use it and every stored run rebuilds
unchanged.

## What was asked (study.py, ledger run id halves_1102, $2.26)

- The product code, walkaround/halves_pass.py: two marks_v1 calls per crop, one per half along the long side (55% of
  it each, a 10% shared band), each half enlarged 2x, with the run's prompt sentences and effort and no landmarks.
  Each box is mapped back to crop pixels. A box that is the same mark as one already kept on the crop (full-crop boxes
  first) is dropped: its centre inside a kept box, a kept centre inside it, IoU over 0.3, or within
  dedup.cluster_radius_units (0.2) in the panel frame. The rest go through the zoom refine and the SAM step as in a new
  run (zoom_refine.plan / ask_missing / refine_marks, box_refine.plan / ask / refine_marks).
- Crops: every answered crop of friend_pair1 (74, so the replay can run the whole pair), plus the crops Sanat checked on
  synth_pair2_r9 (36) and pairC_core3 (19; return_windshield_kf1 is left out because its two boxes are unlabeled).
  258 halves calls ($1.24), 171 zoom calls on the new boxes ($0.85), 70 SAM requests ($0.18). Meta day Nov 2 ET:
  $17.63 before, $19.89 after.
- Ground truth: his box-fixer labels (eval/box_corrections/<run>.json). Full crop: correct and moved count as
  found, not_mark counts as a false box, his "missed" boxes are misses. A new box (its final box after the zoom move and
  SAM) hits one of his boxes when IoU >= 0.1, or its centre is inside his box grown by max(10 px, a quarter of a side),
  or his box's centre is inside it. This rule was fixed before any answer came back. If it hits a missed mark it finds
  that mark. If it hits a mark already found it is a duplicate. On a not_mark box or on nothing it is a false box.

## Recall (study.json; per crop, as he labeled them)

| run | crops | real marks | full crop: found / missed / false boxes | full + halves: found / missed / false boxes | new boxes: find a miss / duplicate / false |
|---|---|---|---|---|---|
| friend_pair1 | 22 | 32 | 22 / 10 / 13 | 25 / 7 / 41 | 3 / 7 / 28 |
| pairC_core3 | 19 | 48 | 48 / 0 / 5 | 48 / 0 / 53 | 0 / 21 / 48 |
| synth_pair2_r9 | 36 | 44 | 43 / 1 / 4 | 43 / 1 / 10 | 0 / 39 / 6 |

- The strict rule (centre within 10 px of his box) gives the same finds: 3, 0, 0. Without the zoom move and SAM, the
  raw halves boxes also find 3, 0 and 0.
- All 3 friend finds are low confidence (2 on return_left_rear_fender_kf2: scratch and scuff; 1 at the left edge of
  return_right_rear_door_kf1: reflection). So K12 caps each one, and none can be a decided line alone. No friend false
  box is damage-typed above low. On synth 5 of the 6 false boxes are damage-typed and medium or high.
- Pair C: his labels have no misses on these crops, so the pass can only add boxes there. The 48 false boxes are
  mostly dirt on the debris-covered hood and trunk (54 of 69 new boxes typed dirt). Some may be debris he did not
  mark as missed.
- M4 (friend's pink square, not on a crop he checked): the halves boxed it on all three right rear fender crops
  (pickup kf1 "small pink reflection spot" medium, pickup kf2 and return kf1 low, all typed reflection). I made that
  call by eye against docs/FRIEND_RUN1.md's location (just ahead of the fuel door's top corner, by the rear door's back
  edge). The same crops also got 3 boxes on the shirt's reflection.
- after_refine (a second drop once the zoom move and SAM have run, against the full-crop marks' final boxes; no extra
  call): on friend it keeps the 3 finds, drops 5 of 7 duplicates and 1 false box, and 27 false boxes stay.

## Verdicts (replay.py; product rebuild on a scratch copy with the stored answers, no call)

A new pair or one-walk mark with no stored second look or counter-look replays as not asked. Caps then stay, so a
new pair can only end at needs your eye here.

| run (base) | base: already there / new / not seen / needs your eye | + halves | + halves, after_refine |
|---|---|---|---|
| friend_pair1 (R16 c12 leaves) | 2 / 0 / 0 / 6 | 0 / 0 / 0 / 38 | 1 / 0 / 0 / 31 |
| pairC_core3 (+ zoom and SAM leaves) | 1 / 0 / 0 / 11 | 1 / 0 / 0 / 18 | 1 / 0 / 0 / 17 |
| synth_pair2_r9 (+ zoom and SAM leaves) | 2 / 1 / 0 / 14 | 2 / 2 / 1 / 27 | 2 / 2 / 1 / 24 |
| synth_pair2_r9 with R17's zoom and SAM answers for its full-crop marks | 0 / 1 / 2 / 20 | 0 / 2 / 2 / 31 | 0 / 1 / 2 / 22 |

- friend: M1 and M2, both right "already there", drop to needs your eye with the halves. A halves box lands on the
  strip beside the full crop's box (on M2 the pickup's full box sits about 70 crop px above the strip). Two candidates
  then fail the match's ratio test, so neither pairs. after_refine brings M1 back but not M2: the zoom move does not
  bring M2's full box onto the strip (R16 open item 1).
- synth (frozen boxes): the halves box on the left front door chip pairs with the return's chip box. That leaves the
  pickup's own box for the same chip as a false "not seen at return". A second box on the new 10 cm scratch gives a
  second "new" line for it. The right front fender pair becomes a right "already there" (base: K7). With the full
  boxes moved as a new run would move them (R17's answers), the first union adds a false "new" on the return's old
  chip, and after_refine removes it, so the decided lines equal base there.
- Friend per mark (replay, R16 c12 leaves):

| mark | truth | base | + halves | + halves, after_refine |
|---|---|---|---|---|
| M1 left rear fender strip | already there | already there | needs your eye (ratio test) | already there |
| M2 left rear door strip | already there | already there | needs your eye (ratio test) | needs your eye (ratio test) |
| M3 hood edge strip | already there | needs your eye (pair inside tau_pre, K4 K5 K12, no look stored) | same | same |
| M4 right rear fender pink square | already there | silent | needs your eye: 3 one-walk reflection lines; the pickup kf2 and return kf1 boxes sit d 0.338 apart, but a second pickup box of the square fails the ratio test | same |
| M5 trunk pink triangle | new | not comparable | not comparable (4 false boxes on the return trunk crop) | same |
| M6 left rear door lower triangle | new | needs your eye (L2, K4 K12 K15) | same | same |
| M7 right rear door triangle | new | not comparable | not comparable | same |

  after_refine also makes a second hood pair, two low boxes on the hood's far side away from M3, at d 0.195. A live
  second look saying "same" there would turn it into "already there" for a spot that is not one of the 7 marks.

## Decision

Real marks found rise 22 to 25 of 32 on friend (all three capped) and 48 to 48 on pair C, while false boxes go 13 to
41 and 5 to 53. The replay loses M2's right "already there" in both variants, and the first variant also loses M1
and adds a false "not seen at return" on synth. So the rule for adoption is not met. No live re-run was made. The
code stays in the product with the leaf absent, so these replays can be repeated. What would have to change first:
a duplicate that sits a patch beside the full crop's box should be merged rather than kept as a second candidate,
and the extra one-walk boxes need a type or confidence filter before they reach the sheet.

Files: study.py, replay.py, halves_answers.json, zoom_answers.json, sam_answers.json, study.json,
replay_<run>_<variant>.txt.
