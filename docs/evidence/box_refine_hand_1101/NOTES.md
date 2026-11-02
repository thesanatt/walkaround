# SAM box refine variants against hand-corrected boxes (pairC_core3, 11/1)

Ground truth: Sanat's 53 corrections of pairC_core3's Spark boxes with tools/boxfix
(eval/box_corrections/pairC_core3.json): 48 moved, 5 not a mark, 0 correct as drawn. pairC_core3 was frozen before
the refine leaf, so its mark boxes are Spark's boxes as Spark gave them.

Variants, all on the crop the product sends (study.py):

- a: Spark's box as is.
- b: the current refine. Spark's box plus pad_px as a box prompt, then box_refine.decide with the thresholds leaf.
- c: Spark's box grown by one Spark patch (28 sent px; every crop here is sent at its own size, so 28 crop px) on
  every side, as a box prompt. Same rules as b, with the grown box as the prompt.
- cu: c shifted up 4 px, the median vertical offset of his 48 moved boxes (measured on the same boxes, so this is
  the best case for it).
- d: Spark's own point as one positive point prompt. Same rules (it is 1.6 px from the box centre at the median).

One request per crop, four objects per mark, up to 16 objects a request: 23 requests over the 19 crops, $0.0575,
ledger run id box_refine_hand_1101. Answers are in answers.json; `python study.py score` rescores with no call.

## Numbers (48 moved boxes)

"On his mark" means the box centre falls inside his corrected box. When a rule rejects SAM's box, the mark keeps
Spark's box, the same as the product.

| variant | median IoU | median centre offset, px | on his mark | IoU >= 0.5 | SAM's box used |
|---|---|---|---|---|---|
| a Spark's box | 0.149 | 26.3 | 23/48 | 6 | n/a |
| b current refine | 0.130 | 27.1 | 23/48 | 5 | 15 |
| c grown one patch | 0.109 | 28.4 | 20/48 | 4 | 6 |
| cu grown, up 4 px | 0.083 | 28.1 | 19/48 | 3 | 9 |
| d Spark's point | 0.149 | 26.3 | 23/48 | 6 | 0 |

SAM's own box with no rules (a mark with no mask counts as IoU 0):

| prompt | got a mask | on his mark | centre offset on those, px |
|---|---|---|---|
| b | 22/48 | 12 | 26.2 |
| c | 27/48 | 13 | 37.0 |
| cu | 24/48 | 12 | 40.1 |
| d | 38/48 | 1 | 257.3 |

Where b used SAM's box (15), IoU got better on 6, worse on 3 and stayed about the same on 6 (a change under 0.02).
By size: S boxes (40) 0.096 to 0.111, M boxes (7) 0.502 to 0.403.

## The 5 boxes he marked "not a mark"

- b: SAM's box on 2 (both on the right front door), Spark's kept on 3 (masks of 2.0%, 5.3% and 44.5% of the crop).
- c: SAM's box on 2, kept on 3. cu: SAM's box on 3. d: none (4 point masks too big for the crop, the fifth over 3 times Spark's box).
- No variant flags them. SAM returns a mask for all five with b, so "SAM found nothing" is no sign of a false mark.

## What it means

On pair C most marks are small leaf debris, and Spark's box sits about a patch beside the leaf on blank paint. With
b, SAM finds nothing in 26 of 48 boxes and the mark keeps Spark's box. Growing the box lets the leaf in, but SAM
then takes the larger thing next to it (the wiper, a cluster of leaves, a panel edge): 13 masks are over 3 times
Spark's box and 7 are too big for the crop, and of the 6 it keeps, 4 are worse and none better. A point mask is far bigger than a mark on 37 of 48
(over 3 times Spark's box or too big for the crop) and empty on 10, as the R11 probe found. Shifting up 4 px does not help (19/48 on the mark against 20/48).

Neither c nor d beats b, so the product keeps b. b against a is a wash on pair C (same 23/48 on the mark, median IoU
0.149 to 0.130), while R11 measured a gain on the synthetic car, so the leaf stays on.

What the data points to instead (R11.4 item 1): Spark's box plus its eight one-patch shifts in one request, then a
rule that picks a shift. With his boxes as the truth this can now be scored: about 40 requests on these 19 crops
(nine objects a mark, 16 a request), about $0.10.
