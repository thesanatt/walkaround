# Registration of the two walks' crops (11/2, R21)

Offline, no model call, $0, no LEDGER row. Product code: walkaround/register.py and match.registration_step, behind
thresholds matching.registration (on for new runs; absent from every frozen run, so they rebuild unchanged).

## What it does

1. Alignment of a panel's pickup crop onto its return crop (register.align):
   - outline fit: ECC on the two SAM masks, affine coarse to fine from the moment-frame map, then a homography from
     that affine (half-size masks; under a pixel lost on the synth truth);
   - SIFT on the outline band (8% of sqrt(mask area) either side) and on compact strong structure inside the mask
     (handles, trim, badges); open paint and every mark box of either walk are cut out; matches with a 0.8 ratio
     test, mutual, gated to 25 px (or 3% of the diagonal) of either outline fit, RANSAC homography;
   - the feature fit is used when it has 12 inliers spanning 15% of the panel box and keeps mask IoU 0.85; otherwise
     the outline homography is used and the affine kept beside it. Their disagreement at a point is that point's
     uncertainty;
   - refused when the outline fit's mask IoU (where the two crops overlap) is under 0.90, or when the edges, ungated,
     agree on a strong motion that puts the panel more than the gate from the outline fit (a mask cut short in one
     walk can be fitted by squeezing the other; the synthetic test caught it).
2. Presence (register.presence): the mark's box is carried into the other crop. On high-passed Lab residuals (the
   box's own scale) the patch around it is slid over a window running from the homography's spot to the affine's spot
   plus 10 px, then the box's saliency (how much it stands out from a ring around it, in ring spreads) is compared at
   the found spot. present: NCC >= 0.6 and 25% of the saliency kept, or NCC >= 0.45, 50% kept and the box itself
   correlates >= 0.5. absent: NCC < 0.5, no spot keeps 15%, own saliency >= 6. Otherwise, or with an unsure map
   (over 50 px), a spot off the other mask or crop, or a box that barely stands out in its own crop: cannot tell.
3. In the match (match.registration_step, compared non-wheel panels only): a one-walk mark that is present pairs with
   an unmatched mark of the other walk found within 12 px or half the box (paired, measured there), else is capped at
   needs your eye (K18); a moment pair outside tau_pre whose spots coincide is measured at the registered spot
   (pair_kept, d_moment kept). Absent and cannot tell change nothing. Every fit and test is on PanelMatch.registration
   and registration_marks; how_checked prints them.

## Files

- study.py: `align RUN...` (per pair fits, synth landmark errors: align_<run>.json), `truthbox RUN...` (presence on
  true boxes: truthbox.json), `presence RUN...` (product boxes on the frozen match), `overlays` (the images in
  docs/registration/ of the workspace).
- replay.py: product rebuild of a scratch copy, base (frozen thresholds) or reg (plus the leaf): replay_<run>_<v>.json.
- compare.py: base against reg with truth: compare.json.
- hand_labels.json: M2 and M6 boxes I drew on 2x zooms of the lexi_pair2 left rear door crops.

## Numbers (final code)

Alignment of the chosen keyframe pair of every compared non-wheel panel (align_<run>.json):

| run | compared | accepted | by features |
|---|---|---|---|
| synth_pair2_r9 | 8 | 7 | 3 |
| synth_pair2_r8 | 9 | 7 | 0 |
| lexi_pair1 | 6 | 2 | 0 |
| lexi_pair2 | 9 | 4 | 1 |
| pairC_core3 | 2 | 0 | 0 |
| pairC_r16 | 5 | 4 | 1 |
| eval_nhtsa15759_r16 | 4 | 1 | 1 |

25 of 43. Every refusal (27 of 77 pairs counting the mark keyframes) is the outline IoU under 0.90: SAM's two masks of
the same panel differ (sill in one walk, crop edge, rear window against a side window on lexi_pair1). Synth truth: 25
generator marks on the panel carried by accepted fits land a median 8.2 px (max 35.1 px) from their true spot, all
inside the search window.

Presence on true boxes (truthbox.json; 190 views, 48 on refused fits; the rest):

| set | views (marks) | present | absent | cannot tell |
|---|---|---|---|---|
| synth, old marks (in both walks) | 100 (7) | 80 | 0 | 20 |
| synth, the new scratch tested in pickup | 12 (1) | 0 | 12 | 0 |
| lexi_pair1, his boxes of old marks | 20 | 2 | 0 | 18 |
| lexi_pair2 hand, M2 (old) | 4 | 4 | 0 | 0 |
| lexi_pair2 hand, M6 (new) | 2 | 0 | 1 | 1 |
| pairC_core3, his boxes (no change truth) | 4 | 0 | 0 | 4 |

Wrong calls: 0. The thresholds were set on the synth and lexi_pair2 rows above, so this is not a held-out score. Two
things it showed: NCC alone does not separate present from absent (M2 0.47 to 0.58, M6 0.43), the saliency does
(M2 keeps 65 to 100%, M6 5%); and a clearcoat scratch can all but vanish under the other walk's light (synth
old_scratch_05cm: 8.5 in pickup, 0.75 at return), which is why absent is strict and changes nothing.

Product boxes in the replays: 80 marks tested, 5 present, 2 absent, 73 cannot tell (29 boxes that barely stand out in
their own crop, mostly Spark boxes beside the mark; 22 refused fits; 12 off the mask or crop; 10 unsure maps). 4
capped, 0 paired, 0 pairs kept.

Verdicts, base against reg (compare.json): the only label change is synth_pair2_r8 pickup_left_front_door_kf2_m2, a
false "not seen at return" on a box with no generator mark, now needs your eye (L4 L1 K18). K18 also joins 3 marks
already at needs your eye (pairC_core3 1, pairC_r16 2). False "new" or "not seen at return": synth_pair2_r8 3 to 2,
the rest unchanged; false "already there": unchanged (synth_pair2_r9 1, synth_pair2_r8 2, 0 elsewhere). No new false
line, no right line lost. Lexi M1 to M7 are unchanged in both runs: M1's fender fit is refused (IoU 0.85), M2's pair
sits inside tau_pre and was lowered by the second look (K7; Spark's pickup box is 70 px above the strip), M3's hood
map is unsure by 67 to 115 px, M4 has no box, M5 and M7 are on panels not compared, M6's spot is below the pickup
door mask.

Time: derive_match 0.2 s to about 3 s a run with the leaf (synth_pair2_r8 2.96 s, lexi_pair2 2.94 s). Two rebuilds
of a run with the leaf give the same bytes (synth_pair2_r8 matches.json b44d721ab71e in two processes).
