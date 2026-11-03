# R29 notes (Nov 2 night): Sanat's hero_focus5b checks, the four fender and wheel panels, the replays

No model call, $0. Scratch images in /tmp/hf5c (not kept).

## Why four fender and wheel panels had no keyframe in one walk (coverage.json of hero_focus5b)

| panel | pickup | return |
|---|---|---|
| left_rear_fender, left_rear_wheel | no track | filmed |
| right_front_fender, right_front_wheel | filmed | no track |

Not coverage thresholds: neither walk had a track for these slots at all.

- Left rear, pickup: panel assignment. The pickup walk opens on the left side with both left wheels in frame
  (pickup_wheel_01 frames 0 to 232 is the front wheel, image-left; pickup_wheel_00 frames 0 to 78 and pickup_wheel_02
  frames 124 to 146 are the rear wheel, image-right). The side vote put wheel_01 on the right, so the two were not
  same-side peers and the pairwise order never ran; R4 then gave front to both, since both share frames with the left
  headlight. The part check moved wheel_01 to the left and kept its position. All three tracks ended up
  left_front_wheel, the keyframes came from wheel_01 (the real front wheel), and no left_rear_wheel or left_rear_fender
  track was left. Same on hero_focus3 and hero_focus4 (same footage), and on pairC_r16's return (its front right wheel
  was a second right_rear_wheel; checked on frames 100 and 150). Fix: panels.wheel_order_after_side.
- Right front, return: tracking. SAM 3.1's wheel lane never detected the right front wheel in the return walk. Object 4
  follows the left front wheel to frame 182, then nothing but a background car's wheel (object 9) until object 4 comes
  back on the right rear wheel at frame 330. The right front wheel is whole and sharp at frames 290 to 345 (frame 340
  looked at). The fender is built from the wheel, so it went too. Not fixed here: the fix is a re-ask (an image-mode
  "wheel" detect on frames where a side's door is seen and no wheel is tracked, then a box-prompted video lane), which
  needs live SAM calls and a new run.

Neither fix can show in a rederive: tracks, keyframes and crops are the stored run's, and a new keyframe would need a
Spark call. A new live run gets the left rear keyframes.

## Sanat's five needs-your-eye rows under decision 1

| row | mark | presence at its spot | fit | R29 |
|---|---|---|---|---|
| mirrors_v1 | return_mirrors_kf2_m3 | present, NCC 0.85 | outline, mask IoU 0.97, 3.1 px at the mark | already there (L8) |
| right_rear_wheel_v1 | return_right_rear_wheel_kf3_m2 | present, NCC 0.95 | edges, 110 of 136 points over 0.48, 0.9 px, IoU 0.98 | already there (L8), wheel step |
| left_front_door_v4 | return_left_front_door_kf1_m3 | cannot tell, NCC 0.15 (its pickup half is faint, 1.5) | edges, 24 of 45 | unchanged (K15) |
| right_rear_door_v1 | pickup_right_rear_door_kf1_s1 | cannot tell, NCC 0.81, contrast at the spot 0.27 against 5.2 | edges, 44 of 69 | unchanged |
| left_front_door_v3 | pickup_left_front_door_kf1_m1 | absent, NCC 0.35 | edges | unchanged (K15) |

Six pickup hood marks typed dirt were also present at their spots (all leaves Sanat labelled not a mark); types_excluded
keeps dirt and reflections capped so they stay in the debris line.

## Replays (replay.py, replay.json; frozen = each run's frozen thresholds, current = frozen plus every current
matching leaf except the two; r29 adds the two leaves)

| run | base | changed lines | decided labels gained or lost |
|---|---|---|---|
| hero_focus5 | frozen | right rear wheel to already there (L8); roof drops G1, still not comparable (G4, G6) | +1 already there (right) |
| hero_focus5 | current | the same plus mirrors to already there (L8) | +2 already there (right) |
| lexi_pair2 | both | tail lights (no marks) compared, no line; mirrors, trunk lid, windshield drop G1, still not comparable | none |
| pairC_r16 | frozen | headlights pickup m1 not comparable to needs your eye (K15, K4; its spot is on the return view); tail lights compared, no line; mirrors and rear window drop G1, still not comparable | none |
| pairC_r16 | current | headlights m1 to no line (left off as faint); tail lights no line; mirrors drop G1 | none |
| eval_nhtsa15759_r16 | both | tail lights compared, no line; headlights, rear bumper, roof drop G1, still not comparable | none |

No new false new, not seen at return or already there. Note on the current base for NHTSA: the working tree's
stands_out max_panel_share is 1.0 (not this round's edit), so the crush box is tested for K20 and left off as faint; the
crush "new" is gone in both current columns. With 0.10 (committed) it stays new.
