# Fenders and the 2 m lap (Nov 1)

Why no fender or rear quarter was ever filmed on synth_pair1, synth_pair2 or pairC_core3, and what changed.

## SAM's answers

- The fender lane ("fender") came back zero_matches in both walks of synth_pair1, synth_pair2 and eval_nhtsa15759_r6.
  On pairC_core3 it tracked, but only the arch lip: keyframe masks 0.3% to 0.9% of the frame, under the filmed floor
  of 3%, so every fender slot stayed glimpsed or not filmed.
- Image requests on stored frames (ledger note fender_noun_1101, 91 requests, $0.23): "fender",
  "car fender", "front fender", "quarter panel", "rear quarter panel", "rear fender", "rear wing", "front wing" found
  nothing on the synthetic sedan. "wheel arch" found the arch strip (1.6% to 3.4%). "side panel" and "car body panel"
  found the front fender on one frame but also whole sides, doors, the hood or the bumper corner on others.
- First batch (56 requests, results as printed at the time; masks as share of the frame): on the four synthetic
  fender and quarter stops (source frames 210, 444, 798, 1032) and the door stop (327), "fender", "car fender",
  "front fender", "quarter panel" and "rear quarter panel" returned no mask at all; "wheel arch" 0.016 to 0.034 on the
  four stops; "side panel" 0.21 on frame 210 (exactly the front fender), 0.107 on 444 (the bumper corner), none on 798
  and 1032, the two doors on 327; "car body panel" 0.45 to 0.50 whole-side masks, or the hood on 1032. On pair C,
  "fender" 0.005 to 0.007 (the arch lip), "quarter panel" 0.13 on the rear door, "side panel" split the side into rear
  door, front door and front fender.
- Second batch: noun_test.json (35 requests).
- One "side panel" video lane on synth_pair2's pickup (ledger run id fender_lane_1101, $0.12): 3 tracks, each wandering
  over doors, bumper and quarters; nothing during the front left fender stop. Not usable as a lane.

## Doors at 2 m

At the 2 s stop beside each door the door's box reaches the top of the frame (SAM's door mask includes the glass, cut
by the frame), border share 0.29 to 0.59, so the box pass dropped every frontal frame and the keyframes were far-oblique
corner views. That is why the doors failed G4 (0.58 to 0.63) and G9 on synth_pair2.

## Offline measurement

rederive.py re-derives a stored run's keyframes from its lanes and source files into a scratch root (no call, stub blur);
kfeval.py checks the view gates on every keyframe pair. The base re-derive reproduced the stored keyframes exactly.
Numbers are in walkaround/docs/CONTRACTS.md, "Lead 11/1, 2 m laps".
