# The one-walk counter-look, live (Nov 1, 2026, about 2:20 AM EST, code f24ae02)

Rule K16 (docs/CONTRACTS.md, "Lead decision 3"): a mark seen in one walk only that would be "new or unseen" or "not
seen at return", with no cap, is shown to the second look beside the other walk's crop of the same panel, the spot
where it would sit outlined. Same or unsure lowers it to "needs your eye"; different keeps it.

1. eval_nhtsa15759_r6 (a kept eval run, LIVE=1): the crushed left front door's return mark
   (return_left_front_door_kf1_m1). Muse Spark Open, meta-models/muse-spark-open-1026 on the hackathon key, 14063 ms,
   chatcmpl-9a473468a256415cade759e5ef8ce097, said different: "Second image box shows clean undamaged door with smooth
   paint and GTI badge, with no crushing or denting seen in the first post-test image." The mark stays "new or unseen".
   Files: eval_nhtsa15759_r6_counter_looks.json, eval_nhtsa15759_r6_open_weights_counter.ndjson and the two JPEGs as
   sent (return_left_front_door_kf1_m1__counter.look.jpg, return_left_front_door_kf1_m1__spot.look.jpg). $0.0030.
2. k16copy_px_two_takes: the eval's one false alarm. The live re-run eval_px_two_takes_r6 never compared the rear
   bumper (its return view counted as glimpsed), so counter_on_copy.py made a non-kept copy of eval_px_two_takes with
   its frozen thresholds plus the two new leaves and ran the match with the look live. The return mark
   return_rear_bumper_kf1_m1 (a parking-sensor dot by eye) was asked; Muse Spark Open, same model and key, 4030 ms,
   chatcmpl-39e93e03d60b42b8aa40f2bec4db1716, said different: "The magenta box in the second pickup image shows intact
   glossy black paint with no chip or scuff matching the mark outlined in the first image, indicating the damage was
   not present at pickup." The mark stays "new or unseen"; the false alarm stands. The replay of the copy's match gave
   matches.json and verdicts.json back byte for byte. $0.0036. The copy is keep false and was moved out of runs/ so the
   eval would not read it.
