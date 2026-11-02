# Doors 11/1: which door a door track is, and no edge-on keyframes

Sanat found it checking verdicts on pairC_core3: crops/pickup/pickup_right_rear_door_kf1.jpg showed a front door with
its side mirror, almost edge-on, while the return walk's right rear door crop showed the real rear door. The match had
marked those panels not comparable, but the sheet and the UI still named them wrong. Two changes, each behind its own
thresholds leaf (off when a run's frozen thresholds lack it, so frozen runs rebuild unchanged):

- `panels.door_check`: after the side vote and the twin check, each car door track is filed by side and front or rear
  from cues the walk already tracked (decided lids, side mirrors, wheels, the other doors, the walk order). Cues that
  disagree leave the side or the position undecided, so the panel is side_uncertain. A door frame whose box holds half
  of a headlight or tail light box is an end view (a tailgate) and never a door keyframe candidate.
- `panels.edge_on_floor`: a keyframe candidate whose panel mask is narrower than the floor for its noun (width over
  height) is seen near edge-on and never becomes a keyframe. When every candidate of a panel is under it, coverage says
  "only seen edge-on" instead of picking a sliver.

No model call, $0. Files here: rederive_doors.py (re-derives a stored run's tracks, panels, keyframes and coverage
from its lanes and source videos into a scratch root, stub blur, so nothing is kept or shown), summarize.py,
doors.json (door, wheel and fender keyframes of the stored run, of today's code without the two leaves (base) and with
them (doors), plus every door track's decision with its cues and the door coverage lines).

## How the cues were set (stored tracks, no call)

Measured on every door track of pairC_core3, synth_pair2_r9, eval_nhtsa15759_r6 and eval_px_two_takes_r7, with the
door identity checked by eye on rendered frames:

- Mirror gap (nearest mirror centre in the door's height band, in door widths, median over frames): front doors
  0.00 to 0.05, rear doors 0.54 to 2.18 (one rear door at 0.22 stays silent). Near <= 0.15 votes front, far >= 0.4 rear.
- Wheel under the door (a wheel with >= 0.3 of its width under the door's span): rear doors on 0.54 to 1.00 of the
  frames with a wheel, front doors on 0.00 to 0.33. >= 0.5 votes rear. A door clear of the wheels votes nothing (its
  own wheel may be out of the frame; the scripted test walk in tests/test_panels.py has such a rear door).
- Side: a lid the twin check decided lies toward its end of the car, so the image direction of the front gives the side
  (front image-right on the right side). This holds in side, corner and end views alike. A mirror outside the door box
  gives the same only when the door is not edge-on (box aspect >= 0.5): in an end view the mirror sticks out sideways.
- Keyframe mask aspect (width over height after the window cut), crops looked at by eye: door slivers 0.21 to 0.36,
  the narrowest usable oblique door 0.42 (eval_px_two_takes_r7); an edge-on tire 0.32, oblique wheels with the rim
  readable 0.39 to 0.43; fenders all 0.78 or more. Floors: door 0.40, wheel 0.36, fender 0.40.

## Door decisions (36 door tracks over the three runs; 5 change slot, 3 stay undecided)

| run, track | frames | was | now | cues (+1 right or front, -1 left or rear) |
|---|---|---|---|---|
| pairC_core3 pickup_car_door_04 | 262-295 | right front | left rear | lid_side -1 (trunk lid image-right of it), pair -1, walk -1 |
| pairC_core3 pickup_car_door_05 | 275-295 | right rear | left front | lid_side -1, pair +1, walk +1 |
| synth_pair2_r9 pickup_car_door_06 | 556-572 | right rear | right front | lid_side +1, mirror +1 (0.05 door widths), pair -1, walk +1 |
| synth_pair2_r9 return_car_door_06 | 557-571 | right rear | right front | the same cues |
| eval_nhtsa15759_r6 return_car_door_04 | 355-429 | right front | right rear | lid_side +1, mirror_side +1, mirror -1 (4.7 door widths) |

Every other door keeps its slot and its cues agree with it, for example pairC_core3 pickup_car_door_01 (right front:
lid_side +1, mirror +1 at 0.00, pair +1, walk +1) and _03 (right rear: lid_side +1, mirror_side +1, mirror -1 at 0.95,
wheel -1 on 63 of 106 frames, pair -1, walk -1). synth_pair2_r9 pickup_car_door_02, _03 and return_car_door_03 were
side-undecided before and stay so (no cue fires). All decisions with their numbers are in doors.json.

## Door keyframes before and after (what each crop shows, looked at by eye)

pairC_core3 (stored run against the re-derive with every current panels leaf on):

| keyframe | before | after |
|---|---|---|
| pickup left front door | car_door_00 f4, aspect 0.21: the left front door edge-on from the front | none; coverage "only seen edge-on: 2.4 s visible" |
| pickup right front door | car_door_04 f288, 0.32: the left doors edge-on from the rear-left corner | car_door_01 f141, 1.10: the right front door with its mirror |
| pickup right rear door | car_door_05 f288, 0.32: the left front door and its mirror edge-on | car_door_03 f152 (1.24) and f142 (1.18): the right rear door over the rear wheel |
| pickup left rear door | none | none; only seen edge-on (2.3 s) |
| return left front door | car_door_04 f296, 0.33: the left front door edge-on | none; only seen edge-on (1.9 s) |
| return left rear door | car_door_00 f0 (0.28) and car_door_03 f303 (0.27): slivers | none; only seen edge-on (3.7 s) |
| return right front door | car_door_02 f115, f127, f107: the right front door | car_door_02 f135 (0.91), f119 (0.82): the same door |
| return right rear door | car_door_01 f104, f115, f123: the right rear door | car_door_01 f118 (0.98), f109 (0.94): the same door |

The base re-derive (today's code without the two leaves) already takes the pickup right doors from car_door_01 and
_03 (door_glass_edge lets their frontal frames in), but keeps all four slivers and files car_door_04 and _05 right.

synth_pair2_r9: pickup right_rear_door_kf3 (car_door_06 f562, 0.33) and return right_rear_door_kf1 (car_door_06 f561,
0.33) showed the right front door edge-on from the front-right corner with its mirror. Both are gone; car_door_06 is
now the right front door and all its candidates are under the floor. The other 17 door keyframes keep their frames (rear
doors over the wheel, front doors with the mirror).

eval_nhtsa15759_r6: pickup left_rear_door_kf1 (car_door_01 f68) showed the rear hatch with the NHTSA sign; that frame's
door box holds the left tail light, so it is out, and kf1 is now f6, the left rear door over the wheel. pickup
right_front_door kf1 to kf3 (car_door_02 f121, f132, f110) showed the hatch; all 12 of car_door_02's frames hold a tail
light, so the panel now takes car_door_04 f143, f155, f192, the right front door with its mirror. Still wrong: the
return walk's car_door_04 (f368, f376, aspect 0.56) is a strip at the rear corner with a tail light, not a door. It is
now filed right rear instead of right front (mirror 4.7 door widths away); its frames hold under half of the lamp box
and it is not narrow enough for the floor. That walk's hatch is also filed as the hood, which is what lid_side read.

## Checks

- `walkaround verify` prints "rebuild identical" for pairC_core3, synth_pair2_r9, eval_nhtsa15759_r6 and pairC_r9
  (frozen without the leaves). pairC_core3_claim2 differs in matches.json, verdicts.json and sheet.json, and does so at
  HEAD without this change too (checked from a clean worktree), so it predates it.
- tests/test_doors_1101.py (12): the cues, the decision rule, pair C's left doors leaving the right side, disagreeing
  cues leaving front or rear uncertain in coverage, the JSON leaving the new fields out when absent, the lamp end view,
  the floor per noun, the "only seen edge-on" coverage line, the claim photo's door check, a frozen run unchanged.
