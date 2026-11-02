# hero_focus4 against the truth (Nov 2, 2026)

The run: `runs/hero_focus4`, the sample `samples/hero_focus4`. Live on Nov 2, 5:14 to 5:21 PM EST, code 672a167,
$6.38 (Meta Model API $6.32, Baseten $0.07; rows in `ledger_rows_hero_focus4.jsonl`). `walkaround verify hero_focus4`
prints "rebuild identical".

The car: a friend's white Ford Focus, filmed with the owner's permission. The marks are blue tape strips we put on as
stand-ins, plus one real scrape. Pickup walk IMG_2678.mov (52.9 s), return walk IMG_2679.mov (34.3 s), shot 4 minutes
apart. Two photos taken before the pickup walk show the pickup state.

| Panel | At pickup | At return | Right label |
|---|---|---|---|
| rear bumper | 2 blue strips | the same 2 | already there x2 |
| front left door | 2 blue strips | 1 of them | already there x1, not seen at return x1 |
| right rear door | none | 1 blue strip | new since pickup x1 |
| front left door, lower edge | a real scrape | the same | already there |

## Mark by mark

Each verdict was tied to its strip or scrape by drawing every box of these panels on its crop and looking.

| Truth mark | Right label | hero_focus4 | Score |
|---|---|---|---|
| rear bumper strip A (middle) | already there | already there (rear_bumper_v1, about 5 cm apart); Muse Spark Open said same, 2 of 2 | right |
| rear bumper strip B (right, beside the dealer sticker) | already there | needs your eye (K16): Muse Spark boxed it at pickup only on the second pickup keyframe and at return only on the first return keyframe, so no keyframe pair held it in both walks; each one-walk box went to a counter-look, which said same, so it was capped (rear_bumper_v3, rear_bumper_v7) | miss, safe |
| front left door strip, kept (lower, by the handle) | already there | already there (left_front_door_v1, about 5 cm); same, 2 of 2 | right |
| front left door strip, removed (upper) | not seen at return | needs your eye (K15: Muse Spark flagged the pickup view of the door dirty) (left_front_door_v3) | miss, safe |
| right rear door strip, new | new since pickup | new or unseen (right_rear_door_v2, L5); the counter-look said different, 2 of 2, so the label stands | right |
| front left door scrape | already there | already there on its short right piece (left_front_door_v2, about 4 cm; same, 2 of 2). The long left piece was boxed in both walks but not joined: two needs your eye lines (left_front_door_v4, v5, K15) | right, in part |

4 of 6 right. No false "new" and no false "not seen at return": both misses ended at needs your eye.

Other lines on these panels, all needs your eye: strip A boxed again on the second pickup keyframe (K16, a second
line for a mark already joined); a pickup box on clean paint right of strip B (K15, and K20: the box does not stand
out); two boxes on leaves in the trunk seam (K4, typed dirt); a tiny pickup speck on the right rear door (K12, K20, K3).

Elsewhere on the car: left front fender already there (a thin scratch by the wheel arch, in both walks); hood already
there (a small yellowish speck in both walks, typed dirt). Neither is in the truth list. 10 panels not comparable.
Totals: already there 5, new or unseen 1, not seen at return 0, needs your eye 10, not comparable 10.

## Box by box, against hand labels

The same two videos were run earlier as hero_focus1, and every crop of that run was labelled by hand: each Muse Spark
box marked right, moved or not a mark, and the marks it missed drawn in (`eval/box_corrections/hero_focus1.json`).
hero_focus4 picked the same frame for 37 crops (19 of them byte for byte the same crop). On those 37 crops
(`hero_focus4_truth_score.py`, no call):

- The hand labels hold 45 real marks; hero_focus4's Muse Spark boxed 14 of them.
- Of its 44 boxes there, 14 sit on a real mark, 10 on a spot labelled not a mark (rear window glass, mirror, wheel,
  roof, hood) and 20 on spots left unlabelled. None of the 10 led to an already there, new or not seen line: each
  ended not comparable or in no line.
- On the truth panels: the pickup front left door crop (same bytes) 4 of 4 (both strips, both scrape pieces), the
  return right rear door crop (same bytes) 1 of 1 (the new strip), the pickup rear bumper frame at 49.8 s 2 of 2. The
  second pickup front left door keyframe got no box at all (0 of 5); the first carried the door.

One more real mark from the hand labels: a faint vertical scratch at the rear bumper's lower left corner, boxed in
hero_focus1's pickup crop and drawn as missed in its return crop. hero_focus4 boxed it in neither walk, so it has no
line. Those crops are 1 to 2 frames apart from hero_focus4's, so they are not in the counts above.
