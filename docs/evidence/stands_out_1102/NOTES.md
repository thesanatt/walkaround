# K20: a one-walk mark must stand out in its own crop (Nov 2, 2026, round 23)

Lead decision 2 of Nov 2 evening: a mark seen in one walk only must stand out from its own surroundings in plain
pixels to be labelled new or not seen at return; a box that does not stand out caps at needs your eye with a new rule
id. Thresholds leaf matching.stands_out (off when absent), rule K20.

## The case

hero_focus2, rear bumper: pickup_rear_bumper_kf2_m2, Muse Spark's "blue tape strip on bumper" box at [775, 261, 811,
283] on pickup_rear_bumper_kf2, came out not seen at return (L1). The box sits on clean paint above the right strip
(looked at by eye with the boxes drawn on, a scratch folder, not kept: the crop shows the plate area). Registration
could not test it: the rear bumper fit was refused (mask IoU 0.89 after the warp), so no contrast was measured. The
counter-look said it saw a scuff there.

## The measure

register.own_contrast: the presence test's own contrast (contrast_a), on the mark's own crop alone, so it needs no
fit. The Lab image minus its Gaussian blur (sigma 6 px or the box's long side, at most 120), as a Delta E magnitude;
the box's 90th percentile minus the median of a 10 px ring around it, over the ring's spread (1.4826 MAD, at least 1).
Near the crop edge the ring is clipped. Under min_contrast 3.0 (register.DEFAULTS saliency_min, the presence test's own
"too faint to test") is faint and caps K20.

Values (ring spreads) on the marks that matter, from replay.json and after_looks.json:

| mark | what it is | contrast |
|---|---|---|
| hero_focus2 pickup_rear_bumper_kf2_m2 | stray box on clean paint, false not seen at return | 0.95 |
| hero_focus2 return_rear_bumper_kf1_m2 | box 110 px beside its strip (K17 in the run) | 0.48 |
| hero_focus2 return_right_rear_door_kf1_m1 | the new strip, right | 28.24 |
| hero_focus2 pickup_left_front_door_kf1_m1 | the removed strip | 15.09 |
| hero_focus2 pickup_rear_bumper_kf2_m1 | a bumper strip (paired) | 24.07 |
| pairC_r16 pickup_right_front_door_kf1_m1 | box on a barely visible speck, same-car pair, not seen at return | 0.96 |
| eval_nhtsa15759_r16 return_left_front_door_kf1_m1 | the crush, new, right | 0.87 (box 0.68 of the door) |
| hero_focus1 after looks, pickup_right_rear_wheel_kf1_m2 | rim edge, false not seen at return | 5.77 |

The crush is why a size bound is needed: its box covers 0.53 to 0.68 of the door mask, so its ring is tape, door edge
and background, and no ring width or mask restriction tried brought it over 2.1 (variants: ring 10 px, half the short
side, half the long side, each with the ring cut to the panel mask). A box over max_panel_share 0.10 of its keyframe's
panel mask is an area and is not tested. Strips and stray boxes cover 0.002 to 0.003.

## Replays (replay.py, no model call; each run's frozen thresholds with and without the leaf)

already there / new / not seen at return / needs your eye / not comparable:

| run | base | with K20 | lines changed |
|---|---|---|---|
| hero_focus2 | 7 / 1 / 1 / 11 / 8 | 7 / 1 / 0 / 12 / 8 | 7: the stray bumper box L1 to needs your eye; 6 lines already capped gain K20 |
| lexi_pair2 | 2 / 0 / 0 / 11 / 9 | 2 / 0 / 0 / 11 / 9 | 5 capped lines gain K20 |
| pairC_r16 | 2 / 0 / 1 / 17 / 12 | 2 / 0 / 0 / 18 / 12 | 10: the speck L1 to needs your eye; 9 capped lines gain K20 |
| eval_nhtsa15759_r16 | 0 / 1 / 0 / 1 / 14 | same | 0 (the crush is an area) |

No right label lost: every already there is a pair, which K20 does not test; hero's new strip stands out by 28; the
crush stays new. On hero_focus2 return_rear_bumper_kf1_m2 shows K20 in place of K17: a K20 mark is capped before the
counter-look stage, so the rebuild does not ask it again.

R22's after-looks replays (after_looks.py, pixels only on the source crops): of the lines labelled new or not seen
there, hero_focus1's rim edge stands out (5.77, K20 does not fix it; wheel marks still get no counter-look) and the
NHTSA crush is an area. Nothing right lost there either.

## Files

replay.py and replay.json (the four runs), after_looks.py and after_looks.json.

## Live run hero_focus3 (same pair, code 0dd0a81, every leaf on, $6.58, 8.1 min, rebuild identical)

| truth mark | right verdict | hero_focus2 | hero_focus3 |
|---|---|---|---|
| rear bumper strip, left | already there | already there | not comparable (G4) |
| rear bumper strip, right | already there | already there | not comparable (G4) |
| front left door strip, kept | already there | already there | already there |
| front left door strip, removed | not seen at return | needs your eye (K15) | needs your eye (K15, stands out by 16.9) |
| right rear door strip, new | new since pickup | new since pickup | needs your eye (K17, stands out by 28.2) |
| front left door scrape | already there | already there | already there |

2 of 6 right (hero_focus2 5 of 6). No new, no not seen at return, so no false one; 6 already there, 7 needs your eye,
9 not comparable, 8 check-by-eye lines (debris 6, glass 2).

Both losses are run to run variance, not K20:
- rear bumper: the pickup walk got one rear bumper keyframe, its SAM mask cut short (crop 1280 x 324; the other
  bumper crops of both runs are 484 to 527 tall), so the whitened mask IoU was 0.67 and 0.68 on the two pairs tried,
  under G4's 0.70. hero_focus2 had two pickup keyframes and compared at 0.78.
- new strip: the two-question counter-look (Spark Open) did not say yes to a damage mark in the box, so K17. R22's
  after-looks replay saw the same answer once ("a light blue-white horizontal reflection band").

K20 in the live run: 4 lines carry it (a "small thin scratch" box on the front left door at 1.01, near the scrape;
leaves on the right rear door window seal at 1.05; a speck at 0.43; a tail light speck at 1.31), each with another
cap as well, so it moved no label here. Every truth strip it tested stands out by 16.9 to 28.2.
