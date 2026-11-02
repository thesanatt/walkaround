# Zoom refine on synth_pair2_r9 against Sanat's hand boxes (Nov 2, 2026, about 6:00 AM EST)

Ground truth: his corrections of all 47 marks of runs/synth_pair2_r9 (eval/box_corrections/synth_pair2_r9.json,
sha256 0da77d6c...; 2 correct, 41 moved, 4 not a mark; his one added "missed" box is not scored). The run predates both
the zoom refine and the SAM box refine, so every box in it is Spark's own. Spark's box sits a median 56 px from his
box (pair C: 26 px), in no set direction (median dx +1 px, dy +3 px). His boxes are small: median 25 x 12 crop px,
39 of 41 size S.

## What was run

study.py calls the product code on the stored crops with today's thresholds, as a new run would:
- zoom_refine.plan and zoom_refine.ask_missing: one Muse Spark marks_v1 call per mark on the window around its box
  (median 171 x 96 crop px) enlarged to 1280x720, effort low. 47 calls, all valid ($0.2139).
- zoom_refine.refine_marks: Spark's box moved to the centre of the nearest zoom box, kept inside the crop.
- box_refine.plan, ask and refine_marks (the SAM step): once on Spark's boxes and once on the moved boxes, each asked
  fresh as its own run would. 36 SAM 3.1 image requests per arm, 72 in all ($0.1800).

Ledger run id zoom_refine_synth_1102, $0.3939 in all; the Nov 2 ET Meta day went from $17.23 to $17.63. No file under
runs/synth_pair2_r9 was written.

## Numbers (41 moved boxes)

| arm | median IoU | mean IoU | median centre offset, px | on his mark | IoU >= 0.5 | used |
|---|---|---|---|---|---|---|
| Spark's box (the run) | 0.000 | 0.043 | 56.1 | 2 | 1 | |
| Spark's box, then the SAM step | 0.000 | 0.054 | 56.1 | 5 | 1 | SAM's box 34 |
| zoom move (product) | 0.242 | 0.306 | 9.9 | 22 | 12 | zoom 27 |
| zoom move, then the SAM step (new runs) | 0.458 | 0.348 | 4.2 | 23 | 16 | SAM's box 39 |

IoU wins, losses, ties against Spark's box: zoom move 23, 1, 17 (sign test p = 3e-6); zoom then SAM 24, 2, 15
(p = 1e-5); SAM alone 6, 2, 33 (p = 0.29). The SAM step after the zoom against the zoom alone: 20, 6, 15 (p = 0.009).

- The window is the limit. His mark centre is inside the zoom window for 27 of 41 (pair C: 46 of 48). Where it is, the
  zoom moved the box on 25 and 21 of those land on his mark (median offset after the move 5.2 px over all 27 moves).
  Where it is not (14), the zoom found nothing on 12 and moved 2, both still off the mark. So 14 of 41 keep Spark's
  box (pair C: 5 of 48).
- On Spark's raw boxes the SAM step does almost nothing (median IoU 0.000 either way): it tightens around whatever is
  inside a box that is two patches off. After the move it adds more than on pair C (median IoU 0.242 to 0.458, offset
  9.9 to 4.2 px; pair C 0.353 to 0.377, 11.0 to 9.8 px).
- Over the 43 real marks (moved plus correct) the zoom then SAM arm is 0.458 median IoU, 24 on his mark.

## His 2 correct boxes

Both were moved by the zoom: pickup_left_rear_door_kf1_m1 by 6 px (IoU with his box 1.0 to 0.444, 0.682 after SAM)
and return_right_front_fender_kf1_m1, a thin 49 x 8 px scratch, by 12 px (1.0 to 0.0, 0.072 after SAM). On a box
already on the mark, any move costs; on a thin box a small vertical move costs it all.

## His 4 not-a-mark boxes

| mark | Spark said | zoom | box moved | SAM step after the zoom |
|---|---|---|---|---|
| pickup_right_rear_door_kf3_m2 | reflection | no box | no | SAM's box |
| pickup_tail_lights_kf1_m1 | scratch | 2 boxes, nearest a reflection | 28 px | SAM's box |
| return_hood_kf2_m3 | chip | 1 box, chip | 44 px | kept the moved box |
| return_right_rear_door_kf1_m1 | reflection | 1 box, scratch | 91 px | SAM's box |

The zoom returns no box on 1 of 4 against 14 of 41 real marks, so, as on pair C (1 of 5 against 5 of 48), it gives no
not-a-mark signal. 3 of the 4 are moved. The zoom's own type (reflection on the tail light) is stored on the row but
the mark keeps Spark's type.

## Against pair C (R14, his 48 moved boxes of pairC_core3)

| arm | pair C median IoU | synth median IoU | pair C offset px | synth offset px | pair C on mark | synth on mark |
|---|---|---|---|---|---|---|
| Spark's box | 0.149 | 0.000 | 26.3 | 56.1 | 23/48 | 2/41 |
| Spark's box, then SAM (R13 b) | 0.130 | 0.000 | 27.1 | 56.1 | 23/48 | 5/41 |
| zoom move | 0.353 | 0.242 | 11.0 | 9.9 | 34/48 | 22/41 |
| zoom move, then SAM | 0.377 | 0.458 | 9.8 | 4.2 | 34/48 | 23/41 |

The zoom helps on both; the gain is bigger here because Spark started further off, and the share left on Spark's box
is bigger here for the same reason.

## Files

study.py (ask_zoom, ask_sam, score), zoom_answers.json (47 zoom_marks.json rows), sam_spark.json and sam_zoom.json
(36 sam_marks.json rows each), study.json (tables, per-mark rows with every arm's box, IoU and offset).
