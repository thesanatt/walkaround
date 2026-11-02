# Zoom refine against Sanat's hand boxes (Nov 1, 2026, 10:50 to 11:00 PM EST)

Ground truth: his 53 corrections of pairC_core3 (eval/box_corrections/pairC_core3.json), 48 moved and 5 not a mark.
Spark's box sits a median 26 px from his box, about one 28 px patch. The SAM variants of R13 did not move it
(docs/evidence/box_refine_hand_1101/).

## What was asked

For each of the 53 marks, on the crop the product sent:
- w1: a window around Spark's box. Short side max(96 px, 3 Spark patches, Spark's box plus one patch a side), long
  side to 16:9, centred on Spark's box and shifted to stay inside the crop. Median window 171 x 96 crop px.
- w2: the same window doubled on each axis. Median 342 x 192.

Each window was enlarged to 1280x720 (most product crops are about that size, and R11.2 found Spark best on that
canvas) and sent to Muse Spark 1.3 with the product's marks_v1 prompt, effort low, the fixtures sentence on as in
today's thresholds, no landmarks. The zoom box is the returned box whose centre is nearest Spark's box centre, mapped
back to crop pixels per axis. His mark centre falls inside w1 for 46 of 48 and inside w2 for 48 of 48.

106 Spark calls (53 per window), all completed, $0.6904. Then 17 SAM 3.1 requests ($0.0425) for the product's SAM
step on the w1 moved boxes. Ledger run id zoom_refine_hand_1101, $0.7329 in all; the Nov 1 ET Meta day went from
$48.29 to $49.02.

## Numbers (48 moved boxes)

| arm | median IoU | mean IoU | median centre offset, px | on his mark | IoU >= 0.5 |
|---|---|---|---|---|---|
| Spark's box | 0.149 | 0.210 | 26.3 | 23 | 6 |
| w1 zoom box as drawn | 0.192 | 0.206 | 11.0 | 34 | 5 |
| w2 zoom box as drawn | 0.238 | 0.232 | 15.0 | 33 | 5 |
| w1 box only where w1 and w2 agree | 0.237 | 0.244 | 15.2 | 30 | 7 |
| Spark's box moved to the w1 zoom centre | 0.353 | 0.308 | 11.0 | 34 | 12 |
| Spark's box moved to the w2 zoom centre | 0.250 | 0.270 | 15.0 | 33 | 7 |
| w1 move, then the SAM step | 0.377 | 0.327 | 9.8 | 34 | 12 |

"On his mark" is the box centre inside his box. Marks the zoom returned nothing for keep Spark's box (w1: 5 of 48;
w2: 2 of 48). The moved box is shifted back inside the crop (one mark ran off the top of an 856x380 crop).

- The zoom finds the mark. w1 puts the centre on his mark for 11 more marks and loses none (23 to 34).
- The zoom box as drawn is too small. Its median area is 0.26 of his box (Spark's is 0.53), and in the overlays it
  tends to box only the dark core of the debris. As drawn it wins 20 and loses 18 on IoU, and its mean IoU is no better than Spark's.
- Moving Spark's box (its own size) to the zoom centre wins 26, loses 8, ties 14 on IoU. A bootstrap over the 48
  (5000 draws, seed 0) gives a 95% interval of +0.04 to +0.29 on the median IoU gain. This arm has no tuned number;
  it does what his corrections did, which was mostly moving the box.
- The median IoU is jumpy on 48 marks: clamping one mark inside its crop moved it from 0.309 to 0.353 (the 24th and
  25th values are 0.331 and 0.374). The mean and the counts move less.
- The bigger window (w2) is worse than w1 on every count. Requiring w1 and w2 to agree drops 16 of the w1 moves and
  loses ground. So the 2-window variant is not worth its second call.
- The SAM step after the move adds a little (0.353 to 0.377, 11.0 to 9.8 px); it uses SAM's box on 34 of 48.

## His 5 "not a mark" boxes

| mark | w1 | w2 |
|---|---|---|
| pickup_mirrors_kf1_m1 | no box | no box |
| pickup_rear_window_kf1_m1 | crack, medium, on Spark's box | crack, medium, on Spark's box |
| pickup_right_front_door_kf1_m1 | scratch, medium, on Spark's box | scratch, low, on Spark's box |
| pickup_right_front_door_kf1_m2 | scratch, low, moved off Spark's box | dirt, medium |
| return_headlights_kf1_m1 | unsure, low, moved off Spark's box | unsure, low |

The zoom does not flag them: it returns no box on 1 of 5, against 5 of 48 real marks. With the step on, 4 of the 5
would be moved and 1 keeps Spark's box. Nothing here is a reason to drop or downgrade a mark.

## Product change

Spark's box moved to the w1 zoom centre beats Spark's own box on median IoU (0.149 to 0.353) and on his mark (23 to
34), so it is on for new runs behind thresholds dedup.spark_zoom_refine (walkaround/zoom_refine.py). It runs before
the SAM step, which is then prompted with the moved box. The zoom's own box and type are kept on the row
(Mark.zoom_refine) but not used: the mark keeps Spark's type, as asked, and Spark's box size.
tests/test_zoom_refine.py replays answers.json through the product code and gets the moved-box row above.

Cost: one Spark call per Spark box mark, about $0.0065 each here. pairC_core3 has 94 box marks, so about $0.60 a run.

## Files

study.py (ask, score, ask_sam, score_sam), answers.json (106 Spark records with their windows), sam_answers.json (17
SAM requests), study.json (per-mark rows and the tables), overlays.py and overlays.jpg (12 marks: red Spark, green
his box, blue Spark's box moved by w1).
