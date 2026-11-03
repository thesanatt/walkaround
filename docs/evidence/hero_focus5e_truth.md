# hero_focus5e against the truth (Nov 3, 2026)

The run: `runs/hero_focus5e`, the sample `samples/hero_focus5e`. Its verdicts were recomputed on Nov 3 with
`walkaround rederive hero_focus5 hero_focus5e` (code d3ce7d9) from the model answers that run hero_focus5 stored on
Nov 2; no new model calls, $0. Its sheet, run.json and manifest_run.json carry the same line. Its 14 verdicts are the
same as the earlier recompute hero_focus5d's (code 818a086), apart from the config hash in each trace; only the sheet
changed (the one-line result on page 1, and the return walk's gaps listed under return). hero_focus5 itself ran
live on Nov 2, 6:42 to 6:55 PM EST, code 8a529ff, $6.52 (Meta Model API $6.44, Baseten $0.08; rows in
`ledger_rows_hero_focus5.jsonl`). `walkaround verify hero_focus5e` prints "rebuild identical".

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

Each verdict was tied to its strip or scrape by looking at every row of the run, both walks' crops side by side.

| Truth mark | Right label | hero_focus5e | Score |
|---|---|---|---|
| rear bumper strip A (middle) | already there | already there (rear_bumper_v1); its copy on the second pickup keyframe merged, no second line | right |
| rear bumper strip B (right, beside the dealer sticker) | already there | already there (rear_bumper_v2); its copy on the second pickup keyframe merged, no second line | right |
| front left door strip, kept (lower, by the handle) | already there | already there (left_front_door_v1) | right |
| front left door strip, removed (upper) | not seen at return | needs your eye (left_front_door_v3, K15: Muse Spark flagged the pickup view of the door dirty) | miss, safe |
| right rear door strip, new | new since pickup | new or unseen (right_rear_door_v2) | right |
| front left door scrape | already there | already there on its short right piece (left_front_door_v2). The long left piece is not joined: its pickup box is left off as faint (K20, contrast 1.5) and its return box is needs your eye (left_front_door_v4, K15) | right, in part |

5 of 6 right, one of them in part. No false "new" and no false "not seen at return": the one miss ended at needs your eye.

Other lines on these panels: a tiny unsure speck on the right rear door at pickup, needs your eye (K3, K4, K12). The
leaves in the trunk seam fold into the rear bumper's debris line.

Elsewhere on the car: hood already there (the same spot boxed in both walks; not in the truth list); mirrors already
there (L8: a scuff boxed at return only, and lining up the two mirror crops on a strong fit finds the same patch at
pickup). The check covers the car body only, so each wheel gets one line, "wheels: check by eye; rims and tires are
not checked mark by mark", and no verdict. 4 panels not comparable. Totals: already there 6, new or unseen 1, not
seen at return 0, needs your eye 3, not comparable 4; 14 rows.

## Box by box, against hand labels

The same two videos were run earlier as hero_focus1, and every crop of that run was labelled by hand: each Muse Spark
box marked right, moved or not a mark, and the marks it missed drawn in (`eval/box_corrections/hero_focus1.json`).
hero_focus5e (hero_focus5's crops and boxes) picked the same frame for 49 crops (21 of them byte for byte the same
crop). On those 49 crops (`hero_focus5e_truth_score.py`, no call):

- The hand labels hold 53 real marks; Muse Spark boxed 17 of them.
- Of its 78 boxes there, 17 sit on a real mark, 13 on a spot labelled not a mark (rear window glass 6, windshield 2,
  wheel 2, mirror 2, roof 1) and 48 on spots left unlabelled. None of the 13 led to an already there, new or not
  seen line: each ended not comparable, in no line, or on a check-by-eye line (glass, wheels).
- On the truth panels: the pickup front left door crop 4 of 4 (both strips, both scrape pieces), the return right
  rear door crop (same bytes) 1 of 1 (the new strip), the pickup rear bumper frame at 49.8 s 2 of 2 (strips A and B,
  the copies that merged). The second pickup front left door keyframe got no box at all (0 of 5); the first carried
  the door.

One more real mark from the hand labels: a faint vertical scratch at the rear bumper's lower left corner. Muse Spark
boxed it at pickup (pickup_rear_bumper_kf1_m3), but its contrast (1.7) is under the faint rule, so it is left off the
list and shows only in "how this run was checked". That crop is not one of the 49, so it is not in the counts above.
