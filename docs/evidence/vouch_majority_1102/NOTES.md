# R24: the registration vouches for G4, and the second look by majority (Nov 2, 2026, round 24)

The same hero footage gave 5 of 6 right in hero_focus2 and 2 of 6 in hero_focus3. Both losses came from model answers
that vary from call to call: SAM cut the pickup rear bumper mask short (whitened mask IoU 0.67, under G4's 0.70), and
one Spark Open counter-look said there was no damage in the new strip's box (K17). Two lead decisions, each behind a
thresholds leaf that is absent from every frozen run.

## Decision 1: the registration vouches for the views (matching.registration_vouches)

G4 is there to make sure both walks' views show the same region of the panel. It checks that by comparing SAM's two
outlines, so a mask that SAM cut short fails it even when the crops plainly show the same bumper. Now, when no keyframe
pair of a panel passes every gate, the first tried pair whose only failing gate is G4 (on a measured IoU, with the
frames agreeing) is compared if its registration (R21's register.align, pickup crop onto return crop) is strong:

- accepted by register.align, and fitted on matched edges (an outline-only fit rests on the two masks G4 doubts);
- at least 40 agreeing points, spread over at least 0.30 of the panel's box;
- median reprojection at most 2.0 px, mask IoU after the warp at least 0.85 (the registration's own floor).

A native pass is preferred: the vouch runs only after the keyframe-pair loop found none. A weak or refused fit leaves
G4 and is recorded on the try (KeyframePairTry.vouch, left out of the JSON when not asked). The how-checked line says
which: "G4 passed on the registration instead: 110 of 179 matched points agree over 0.67 of the panel, reprojection
1.3 px", or "the registration did not vouch: ...".

The floors sit well above R21's acceptance (12 inliers, spread 0.15, 3 px). They were set by looking at the fits below,
so this is not a held-out score.

Every G4 pair the rule asked about, 11 stored runs (replay_vouch_hf3.json, replay_vouch_others.json):

| run | pairs asked | vouched |
|---|---|---|
| hero_focus3 rear bumper | pickup kf1 with return kf2 (IoU 0.67) | yes: features, 110 of 179 agree, spread 0.665, 1.32 px, IoU after warp 0.858 |
| hero_focus3 right front fender | 4 pairs (IoU 0.28 to 0.63) | kf3 with return kf2: 68 of 93, spread 0.551, 1.78 px, 0.888; the 3 others refused |
| hero_focus3 right rear fender | 1 (0.69) | refused (outline, IoU after warp 0.83) |
| hero_focus1, hero_focus2, lexi_pair1, lexi_pair2, pairC_r16, pairC_core3, eval_nhtsa15759_r16, synth_pair2_r9 | 33 | none: every fit refused by register.align (outline fits, IoU after warp 0.07 to 0.87) |

(synth_pair2_r8 has no G4-only pair.) The only accepted fits on a G4 pair are the two on hero_focus3; one more of its
fits, right front fender kf2 with kf2 (78 agree, spread 0.21), would fail the spread floor, but that pair was never
asked because an earlier one vouched.

### Offline replay of hero_focus3 (no call; stored answers only)

Frozen thresholds with and without the leaf (replay_vouch_hf3.json):

| truth mark | right | base | with the vouch |
|---|---|---|---|
| rear bumper strip A | already there | not comparable (G4) | already there |
| rear bumper strip B | already there | not comparable (G4) | needs your eye (K20) |
| front left door strip, kept | already there | already there | already there |
| front left door strip, removed | not seen at return | needs your eye (K15) | needs your eye (K15) |
| right rear door strip, new | new | needs your eye (K17) | needs your eye (K17) |
| front left door scrape | already there | already there | already there |

Strip A pairs at d 0.42 against tau_pre_r 0.61. The cut mask skews the moment frame mostly up and down (the two copies
are 0.01 apart in u and 0.41 in v), so the pair still forms, with less margin.

Strip B cannot come out already there in this run: in the pickup crop Muse Spark never boxed it. Its one pickup box sits
on clean paint to the right of the strip (seen with the boxes drawn), and K20 caps that box. In the return walk, the
box near strip B on return kf2 is capped by K18 (the registration finds the same patch at pickup). The return kf1 box
right on strip B is carried onto return kf2 by an outline fit that misses it, so it is not merged and comes out new.
Offline that is a false new, because the stored run never asked a counter-look about it. Live (below) the counter-look
said "same" twice and it is needs your eye (K16). So on this panel the vouch leans on the counter-look.

Right front fender: one pair already there (pickup kf2 m1 with return kf1 m1, "thin horizontal scratch"; both live looks
said same) and three lines capped by K20. That panel holds no truth mark, so the truth file cannot say whether the pair
is right.

Other runs: no label changes on hero_focus1, hero_focus2, lexi_pair1, lexi_pair2, pairC_r16, pairC_core3,
eval_nhtsa15759_r16, synth_pair2_r8 or synth_pair2_r9, because no fit vouched.

## Decision 2: the second look by majority (matching.look_majority)

The pair look (K7) and the two-question counter-look (K17, with K16 from the same answers) are asked up to 3 times.
Asking stops once 2 answers take the same side of every yes-or-no part. The label follows the majority:

- pair look: same keeps the pair only when same outnumbers different and unsure together. Otherwise it is different
  when different outnumbers unsure, else unsure. A 1 to 1 split is unsure.
- counter-look, each part on its own over the same answers: part 1 is yes only when yes outnumbers no and unsure, and
  part 2 is different only when different outnumbers same and unsure. Only (yes, different) keeps the label, as before.
- An ask with no usable answer (every engine failed, or the look could not be built) ends the asking, and the answers
  before it stand. A model still only lowers.
- Every answer is its own looks.json row (ask 1 and 2 on the repeats, left out on the first), and rebuild replays them
  in order. A resumed run reuses each stored answer by its question hash and ask. Notes name the count when more than
  one answer stands behind it: "saw damage in the box (2 of 3 answers)".
- One-question counter rows (runs before counter_confirms_mark) are asked once.

### Live, on the stored crops (live.py; live_hero_focus2.json, live_hero_focus3.json)

Each run's match stage was derived again on a scratch copy under its frozen thresholds, plus the matching leaves
config/thresholds.json has and the frozen file lacks (K20 on hero_focus2, and both R24 leaves on each), by the
product's rebuild with the second look live. Each look's first answer is the stored one wherever the question is the
same, so only the repeats and the looks the new rules newly ask were calls. Calls went to Spark Open on Baseten with
the hackathon key only: the account key slot was left out and hosted Muse Spark was blocked. 26 calls, $0.0848, ledger
run ids r24_hero_focus2 and r24_hero_focus3, Nov 2, 5:00 to 5:03 PM EST. Then the copy was rebuilt with the looks
replayed: matches.json and verdicts.json identical to the live pass on both runs.

Looks where the answers split:

| run | look | answers in order | majority |
|---|---|---|---|
| hero_focus3 | counter-look, right rear door new strip | stored (no, different), then (yes, different), (yes, different) | yes 2 of 3, different 3 of 3: new stands |
| hero_focus3 | pair look, front left door scrape | stored same, then different, same | same 2 of 3: already there |
| hero_focus2 | pair look, hood scratch | stored same, then different, same | same 2 of 3 |

Of the 19 looks asked (8 on hero_focus2, 11 on hero_focus3), 3 split; the other 16 agreed twice. The two hero_focus3 splits are the cases the decision is
for. With one answer, the new strip lost its label in hero_focus3. The scrape would have lost its label in a run whose
only answer was the second one.

New looks on hero_focus3's newly compared panels: rear bumper strip A pair (same, same), rear bumper debris pair (same,
same), right front fender pair (same, same), and the counter-look on the return kf1 strip B box (yes and same, twice: K16).

### The truth tables under the new rules (re-derived from stored answers plus the new looks, not a new run)

| truth mark | right verdict | hero_focus2 | hero_focus3 |
|---|---|---|---|
| rear bumper strip A | already there | already there | already there (was not comparable) |
| rear bumper strip B | already there | already there | needs your eye (K20 on the stray pickup box; strip B never boxed at pickup) |
| front left door strip, kept | already there | already there | already there |
| front left door strip, removed | not seen at return | needs your eye (K15) | needs your eye (K15) |
| right rear door strip, new | new since pickup | new since pickup | new since pickup (was K17) |
| front left door scrape | already there | already there | already there (2 of 3) |

hero_focus2 5 of 6 (as stored), hero_focus3 4 of 6 (was 2 of 6). No false new or not seen at return in either:
hero_focus2's stray-box not seen at return is capped by K20, as in R23's replay. Counts, already there / new / not seen
/ needs your eye / not comparable: hero_focus2 7 / 1 / 0 / 12 / 8, hero_focus3 9 / 1 / 0 / 13 / 7.

## No right label lost (replay, no call)

replay_both.json (frozen thresholds) and replay_current.json (frozen plus every matching leaf config has), base
against both leaves: lexi_pair2, pairC_r16 and eval_nhtsa15759_r16 give the same label on every line (0 changed
lines). With one stored answer per look, the majority gives that answer. hero_focus2 does not change offline either.

## Cost and time

A look now costs 1 to 3 Spark Open calls, usually 2. Spark Open ran about $0.0033 a call here, so a hero-sized run's
looks go from about $0.02 to about $0.08. Time: Baseten at 12 requests a minute (orchestrator.baseten_rpm) adds about
5 s per repeat. The vouch's fit is shared with the registration step's cache, so it costs nothing on a panel the
registration step would fit anyway; on a G4-only panel it is one more fit (under 3 s).

## Files

replay.py (offline variants, scratch copies, the product's rebuild), truth.py (the six marks by mark id per run),
live.py (the live majority on stored crops; DRY=1 counts calls with a fixed answer and no network), and the JSON
outputs named above.
