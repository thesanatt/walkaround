# Factory fixtures are not damage, live (Nov 1, 2026, about 3:05 AM EST)

Lead decision 4 (docs/CONTRACTS.md, "Lead decision 4"): the marks prompt says not to box factory fixtures, and code
leaves out of matching a mark whose box sits on a fixture point the run has. Both behind matching.fixtures_not_damage.

1. eval_px_two_takes_r7 (a kept eval run, LIVE=1, $1.6968): the rear bumper was compared this time (both walks filmed).
   Spark boxed three return marks there: a "scuff" on a smooth highlight above the bumper strip, a camera reflection and
   a white speck. None sits on the first run's dot. It flagged glare on the return view, so all three are "needs your
   eye" (K15); nothing came out new (0 of 3). No fixture landmark came back on any bumper crop, so the code part dropped
   nothing. Rebuild identical.
2. prompt_ab.py and prompt_ab.json: the first run's crop (eval_px_two_takes, return_rear_bumper_kf1), five marks_v1
   calls with the old prompt and five with the fixture sentence, interleaved, effort low, $0.0489. Every one of the ten
   answers boxed a medium chip or scuff within 75 source pixels of the first run's dot; 2 of 5 overlapped its box under
   each prompt. On this crop the sentence did not stop Spark from calling the dot a chip.
3. Why the code part could not help here: spark.landmarks_max 4 cuts the bumper's landmark list to the four plate
   corners, so Spark is never asked to point at a sensor, and no run records a SAM door-handle point yet.

So the false alarm is not fixed. The re-run did not reach it because Spark answered differently on a slightly different
crop and flagged glare.
