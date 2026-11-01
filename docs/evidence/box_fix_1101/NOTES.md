# Mark boxes sitting off the marks (Nov 1)

Sanat saw the squares around marks in the new UI sitting a little off the marks. I walked every step a mark box
goes through, measured against exact ground truth, fixed the two transform bugs I found, and added a SAM 3.1 step that
tightens each Spark box (thresholds `dedup.sam_box_refine`, on for new runs, off for runs frozen before it).

## Where the offset comes from

Not from our transforms. It comes from Muse Spark itself: its boxes miss by whole steps of about 28 sent pixels, which
looks like it picks the neighbouring vision patch.

The chain, with what I checked:

1. Spark's 0 to 1000 grid to sent pixels: `round(v x side / 1000)`, x by the width and y by the height. On drawn chips
   at exact positions (calib.py, calib2.py) a 1280x720 canvas comes back within 2 grid units for 13 of 16 chips. The
   misses are whole steps: on 1100x1100 every miss is -25 or +26 grid units (27.5 px), on 720x1280 -38 or -76 in x
   (27 and 55 px). The mapping is exact wherever Spark is.
2. Sent to crop pixels: per axis in the product (`geometry.sent_to_crop`). `spark_client.point_to_crop_pixels` and
   `box_to_crop_pixels` used one scale taken from the width for both axes, up to 0.7 px off at the bottom of a
   700x1500 crop. Only D1's gate script uses them. Fixed.
3. Crop to source: `dx + u / scale`, and `video.cut_crop` uses the same pixel-centre rule. A dot cut at three crop
   geometries maps back within half a crop pixel (test).
4. Source to keyframe and proxy: crops are cut from the blurred source frame. Ground truth projected through each
   stored crop lands on the planted marks (I looked at four crops: the 10 mm chip and 5 cm scratch sit exactly in
   their truth boxes). Proxies are 960x540, scaled per axis.
5. Web: every overlay (the mark card's zoomed frame, the crop figure, the match pane, the match cut, the claim page)
   draws `bbox_crop` in the crop's own SVG space, with the image at 0,0 at the crop's stored size, so CSS scaling,
   letterboxing and devicePixelRatio move image and box together. No `object-fit` anywhere. All 1042 stored crop
   JPEGs match their stored size. The card's padding around the box is symmetric, so the centre stays put (test).
6. SAM rasters: SAM sends an object larger than 1024 px with its raster scaled to 1024 on the long side, and
   `privacy.detection_mask` pasted it unscaled. 6 of 101 stored image detections are like that (pair C claims, boxes
   1101 px wide, so masks came out 7% short). Fixed. Rebuilding both claim runs with and without the fix gives the
   same 22 files.

Bias on ground truth (bias.py, no call), Spark centre minus truth centre in crop px:

| set | n | median dx | median dy | median distance |
|---|---|---|---|---|
| synthetic, every box near a planted mark (4 runs plus the R10 study) | 36 | +1.2 | -1.5 | 15.7 |
| pair C, D1 crops (R10 study answers) | 22 | -5.8 | -4.5 | 23.2 |

No consistent sign by position (thirds of the crop) or by crop aspect. With a 150 px window (133 box and truth
pairs) the offsets still peak at zero. Spark's own point is no closer than its box centre (median 13.8 vs 13.9 px).

Changing the image shape we send (calib3.py letterboxed to 16:9, calib4.py stretched to 1280x720) gives mixed
results. Chips within 6 px on canvases that are not 16:9: native 7/48, letterboxed 20/48, stretched 26/48 (native
16:9 got 12/16). Letterboxing fixed 2.9:1 but not 1:1, and stretching fixed 1:1 and 9:16 but not 2.9:1. Not adopted;
it needs a test on real crops.

## The SAM step

Each Spark box goes to SAM 3.1's image endpoint as a box prompt on its own crop, with all of a crop's boxes in one
request. A nine-object request gave the same masks and c as one-object requests, so a crop costs $0.0025, not a mark.
SAM's tight box is used when the mask is small, doesn't just fill the prompt box, and sits on Spark's box (rules in
`box_refine.decide`). Otherwise Spark's box stays and the mark says why. Spark's box and point stay on the row
(`Mark.box_refine`). Matching, sizes, the looks and the UI read the box used. Answers are stored in
`runs/<id>/sam_marks.json`, and rebuild replays them.

Probes on 8 marks (sam_probe.py, sam_probe9.py):

- A point prompt returns the whole car or panel. Not usable.
- A grown box (one patch a side) returns a mask filling the grown box. Not usable.
- Spark's own box helps when the box overlaps the mark (5 cm scratch IoU 0.15 to 0.41). When the box sits beside
  the mark, SAM segments blank paint in the box.
- Spark's box and its eight one-patch shifts in one request: on 5 of 8 marks one shift sits right on the mark (10 mm
  chip IoU 0.65 at 1.2 px, 5 cm scratch 0.68 at 3.0 px, dent 5.9 px, pair C chips 10.9 px, hood dent 4.7 px). But
  neither c nor how full the mask is picks that shift out. Not adopted.

## Before and after (measure.py, the near subset of 11/1)

22 synthetic crops holding a box near a planted mark, plus 21 pair C answer sets (D1's gate, 5 runs a crop, and the
R10 study at low). That's 44 requests, $0.11. Of 186 boxes, 139 use SAM's box. Spark's box is kept on 47: 24 masks
were too big, and 23 just filled Spark's box.

| | before | after |
|---|---|---|
| synthetic planted marks hit (scorer's 1.5x rule) | 10/37 | 12/37 |
| synthetic false boxes | 21 | 19 |
| synthetic median centre offset on hits, crop px | 12.0 | 6.0 |
| synthetic median IoU on hits | 0.24 | 0.36 |
| same 19 boxes near a planted mark: median offset / IoU | 13.9 / 0.11 | 9.5 / 0.21 (11 closer, 3 farther, 5 same) |
| pair C D1 targets hit | 25/63 | 25/63 |
| pair C median distance to the target, px | 22.1 | 21.0 |

Overlays in `<workspace>/docs/box_fix/`. Truth is green, Spark's box orange (before), the box used cyan (after),
SAM's mask tinted.

- synth_20cm_scratch: the mask traces the scratch and the box becomes the truth box (offset 11.5 to 0.2 px).
- synth_5cm_scratch: it covers the scratch but runs on to the right, where Spark's box was.
- synth_20mm_chip_beside: Spark's box sits one patch above the chip, and SAM fills the box with blank paint. Kept.
- pairC_pickup_trunk_chips and pairC_return_trunk_chips: Spark's boxes sit beside the chip clusters, and SAM mostly
  fills them. One return box tightened onto a small dark chip.

So on real footage the step helps little. Spark's boxes are off by a patch, and SAM on Spark's box can't move them.

## Files

truth.py (ground truth in crop px), bias.py and bias.json, calib.py to calib4.py with their json (CANVAS=1 writes
calib.py's canvases),
sam_probe.py, sam_probe9.py and their answers, measure.py, sam_answers.json and measure.json, overlays.py,
overlay.py. LIVE=1 makes calls. Everything else re-scores the stored answers. Ledger run id box_fix_1101.

After midnight ET, for the rest of the synthetic crops (84 requests, about $0.21):
`LIVE=1 .venv/bin/python docs/evidence/box_fix_1101/measure.py batch all`, then `... measure.py score`.
