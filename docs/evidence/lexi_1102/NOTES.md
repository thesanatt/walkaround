# Lexi run 1 fixes (11/2): evidence

Four thresholds leaves from the lexi_pair1 diagnosis (docs/LEXI_RUN1.md in the workspace): panels.uncertain_per_track,
panels.part_check, dedup.marks_on_panel, matching.second_marks_pass (details: walkaround/docs/CONTRACTS.md, "Lexi run 1
fixes (2026-11-02)"). Runs frozen before them lack the leaves and rebuild identical (lexi_pair1, pairC_core3,
synth_pair2_r9, eval_nhtsa15759_r6, pairC_r9).

## Files

- replay_1102.py: offline replay of a stored pair run (no call). c1 re-derives panels.json from the stored tracks with
  the per-track rule and gives keyframes the new front sign; c2 rebuilds marks with the on-panel rule; then the match
  with the stored second looks. base reproduces the stored matches.json, verdicts.json and marks.json byte for byte.
  The part check and the second pass need new keyframes or Spark answers, so they are measured on live runs.
- replay_<run>_<variant>.txt: one line per verdict, the off-panel marks, a JSON summary.
- partcheck_stored.py and partcheck_stored.txt: the part check on four runs' stored tracks (no call).

## Offline (no call)

| run | variant | already there | needs your eye | not comparable | marks off panel (left out) |
|---|---|---|---|---|---|
| lexi_pair1 | base | 1 | 10 | 13 | 0 |
| lexi_pair1 | c1 per track | 2 | 13 | 11 | 0 |
| lexi_pair1 | c2 on panel | 1 | 3 | 13 | 14 |
| lexi_pair1 | c1 + c2 | 2 | 6 | 11 | 14 |
| pairC_core3 | base | 1 | 11 | 18 | 0 |
| pairC_core3 | c1 + c2 | 1 | 10 | 18 | 13 |
| eval_nhtsa15759_r6 | base | 0 (1 new) | 5 | 11 | 0 |
| eval_nhtsa15759_r6 | c1 + c2 | 0 (1 new) | 5 | 11 | 0 |

- lexi_pair1 c1: the left rear door, right front door and hood stop being side uncertain; M2 becomes already there
  (d 0.199, tau_pre_r 0.45); M3's two hood boxes pair inside tau_pre (no stored second look, so held at needs your eye).
- No off-panel mark moved on these runs: no keyframe of another panel was cut from the same frame where the point lay
  on its mask. All were left out.
- pairC_core3: of the 13 marks left out, 2 are ones Sanat called "not a mark" (eval/box_corrections), 2 are rear-window
  marks he moved, and his own corrected boxes also sit off the rear-window mask (0.047 and 0.069 of the crop's long
  side), so they are not rear-window marks. The one line lost is a capped one-walk rear-bumper line (L2, K4, K15).
- Part check on stored tracks (partcheck_stored.txt): lexi_pair1 pickup wheel_00, _04, _05 right front to left front;
  return wheel_02 left front to right front, wheel_03 right front to left front; pickup windshield_01 and return
  windshield_00 (frames behind the car) to the rear window. pairC_core3 return wheel_00 left rear to right rear (R15
  said it is the right rear by its frames) and its fender-lane track with it. eval_nhtsa15759_r6 pickup windshield_00
  (frames 68 to 143, the hatch) to the rear window. synth_pair2_r9 one of two "left front" wheels per walk to right
  front, one windshield track per walk to the rear window. The hood over a front wheel sits 0.0 to 0.33 wheel sizes off
  the wheel's centre (depth, not direction), so a lid or light votes only at 0.5 wheel sizes or more; without that
  floor lexi_pair1 pickup wheel_00 split instead of moving.

## Live: lexi_pair2 (same videos, every leaf on, code 26998bb)

$4.89 (sam_video $3.86, sam_image $0.52, Spark marks $0.33, zoom $0.13, second pass 9 calls $0.03, Spark Open looks
$0.014, voice $0.004). Comparable panels 15 (run 1: 8). Labels: 2 already there, 11 needs your eye, 9 not comparable.
Verify: rebuild identical.

Per mark (truth from docs/LEXI_RUN1.md; mark ids are this run's):

| mark | truth | lexi_pair1 | offline c1 + c2 | lexi_pair2 |
|---|---|---|---|---|
| M1 left rear fender strip | already there | already there | already there | already there (pair d under tau_pre, the look said same) |
| M2 left rear door strip | already there | not comparable (G4 G6, a stray door track) | already there | needs your eye (K7): the panel compares (IoU 0.87) and the pair is inside tau_pre (0.251 vs 0.446), but the pickup box sits about 70 crop px above the strip and the look said different |
| M3 left front fender top, hood edge | already there | not comparable on the hood; its fender view filed as the right front fender in a cross-side pair | hood pair inside tau_pre, held at needs your eye | needs your eye on the hood (pickup box on the strip, L1 with K12 K15; the second pass found nothing in the return hood crop); the fender is now filed left in both walks but not comparable (G3, G4 0.52) |
| M4 right rear fender pink square | already there | silent (no mark either walk) | silent | silent: the panel compares (0.885) but Spark found no mark in either walk, so the second pass (one walk silent, the other not) does not run |
| M5 trunk pink triangle | new | not comparable | not comparable | not comparable (G1 G4 G6) |
| M6 left rear door lower triangle | new | not comparable | needs your eye (L2 capped K4 K12 K15) | needs your eye (L2 capped K4 K12 K15, "yellow spot on lower trim") |
| M7 right rear door triangle | new | not comparable (G1) | not comparable (G1) | not comparable (G1: the pickup never shows the door whole) |

Real marks among the needs-your-eye lines: run 1 0 of 10; lexi_pair2 3 of 11 (M2, M3, M6). The other 8: two specks paired
on the left rear door belt trim, a reflection on the right front door trim, a chrome reflection moved from the front
bumper keyframe onto the hood (same frame, on the hood mask), a speck on the hood, and four glass specks and reflections
on the rear window.

The second pass asked 9 times: once on the right front door, three times on the return hood (around M3's pickup box,
another hood speck and the moved chrome reflection), five times on the pickup rear window. One box came back that paired:
a "small white blurry spot" with the return's "small bright nick at glass edge", already there after the look said same.
That box sits 3.3% of the crop's long side off the rear-window mask (on the body by the glass corner), so the pass broke
the on-panel rule. Fix (commit 084b9b0): with the leaf's new on_panel key a returned box off the panel mask is not used.
lexi_pair2 lacks the key and keeps its files; rebuilt into a scratch copy with the key (no call): 1 already there (M1),
12 needs your eye, the rear-window line reads "a box 3.3% of the crop's long side off the panel, not used".

## Live: eval_nhtsa15759_r16 and pairC_r16 (the same sources as the frozen runs, every leaf on, code 084b9b0)

Both are new SAM and Spark passes, and the frozen runs predate several leaves that are on for every new run (the door
check and edge-on floor, the zoom refine, and for pairC_core3 more), so the comparison is "nothing right is lost", not a
clean attribution. Both verify rebuild identical.

| run | cost | comparable | already there | new | not seen at return | needs your eye | not comparable |
|---|---|---|---|---|---|---|---|
| eval_nhtsa15759_r6 (frozen) | | 5 | 0 | 1 | 0 | 5 | 11 |
| eval_nhtsa15759_r16 | $3.37 | 6 | 0 | 1 | 0 | 1 | 14 |
| pairC_core3 (frozen) | | 3 | 1 | 0 | 0 | 11 | 18 |
| pairC_r16 | $4.05 | 6 | 2 | 0 | 1 | 17 | 12 |

- NHTSA: the crushed left front door is still new (L2, dent high), the struck left rear door still "outline changed in
  place" (L7). The three one-walk crush lines and the windshield reflection line are gone. The second pass looked once,
  at the pickup left front door around the crash damage: no mark there (the door was whole before the crash).
- pair C: the trunk-lid pair is still already there; the right front door, right rear door and right rear fender now
  compare (the door check and the part check), with a second already-there pair on the right rear fender. The six extra
  needs-your-eye lines are dark specks on the white doors and fender, all capped. One new uncapped label: right front
  door "not seen at return" on a pickup chip; the pickup crop shows dark specks there and the return crop clean paint,
  and the counter-look said different, so it is what the crops show (debris gone between the takes). 13 marks off their
  panel: one moved (right rear fender to the right rear wheel, same frame), 12 left out. The second pass looked once,
  on the return right rear door: no mark there.

## Limits

- The re-run is a new SAM and Spark pass, so some differences against run 1 are run-to-run noise (pickup box offsets,
  which specks Spark names), not the leaves.
- The on-panel move needs a keyframe of the other panel from the same frame; tonight none existed, so every off-panel
  mark was left out. Moving into a panel tracked in that frame without a keyframe there would need its mask decoded from
  the lane and is not built.
- The second pass covers one walk silent and the other not; M4 (silent in both) needs a different trigger.
