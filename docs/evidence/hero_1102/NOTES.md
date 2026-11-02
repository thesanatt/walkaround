# hero_focus1: each truth mark through the run, the fixes, and the re-run (Nov 2, 2026)

Run: runs/hero_focus1 (white Ford Focus, overcast daylight, 1x, about 2 m, 1080p; pickup IMG_2678, return IMG_2679;
code 29bc7fe; $6.24). Truth: docs/HERO_FOCUS1_TRUTH.md in the workspace (2 blue strips on the rear bumper kept, 2 on
the front left door of which 1 removed, 1 new on the right rear door, a real scrape low on the front left door).
Crops were looked at by eye with the run's boxes drawn on (/private/tmp scratch, not kept: they show the plate area).

## 1. Each truth mark through hero_focus1

| mark | tracked, compared | crops that show it | Spark (marks_v1, effort low, fixtures sentence) | after Spark | verdict |
|---|---|---|---|---|---|
| rear bumper strip, left | pickup_bumper_01 / return_bumper_01, compared on P kf3 / R kf2 (mask IoU 0.81) | all 5 (P kf1 to kf3, R kf1, kf2), plain | no box in any crop | nothing to match | silent |
| rear bumper strip, right | same | all 5 | no box | nothing | silent |
| front left door strip, kept (mid door) | car_door_01 both walks, P kf1 / R kf1 (0.93) | P kf1, kf2, R kf1, kf2 | pickup: no box; return: kf1 m3 and kf2 m3 "small blue rectangular mark", unsure/low (K4, K12), merged | one-walk return mark; registration fit by features, presence present in P kf1 (K18) | L2 capped by K4 K12 K15 K18: needs your eye (v4) |
| front left door strip, removed (below the mirror) | same | P kf1, kf2 (gone at return, right) | no box in the pickup | no pickup mark, so no L1 | silent |
| right rear door strip, new | car_door_04 / car_door_06, P kf1 / R kf1 (0.97) | R kf1, plain and sharp on clean paint | no box in either crop | no mark in either walk, so the second marks pass does not run | silent |
| front left door scrape (real) | as the door | P kf1, kf2, R kf1, kf2 | P kf1 m1 scratch/medium, R kf1 m1 scratch/medium (both on the scrape after the zoom move) | pair d 0.094, under tau_pre_r 0.666; second look (Spark Open): same | L3 already there (v1), right |

So the line read as "the kept strip is already there" is the scrape's pair; the kept strip itself got needs your eye.

Side lines on the door: v2 is the scrape's small dash (P kf1 m2 / R kf1 m2), a tight pair the second look called
different (K7). v3 is the scrape seen again in P kf2: Spark's box sat 15 px under the scrape, the zoomed look returned
6 boxes and the nearest to Spark's centre was a faint dash (typed unsure) beside the scrape, while the zoom's own "dark
wavy paint scratch" box sat on it. Moved onto the dash, the box whitened 0.33 panel units from P kf1 m1 (merge radius
0.2), stayed a separate mark, found no partner and came out L1 not seen at return, held only by K15. In clean light it
would have been a false "gone" on real damage. v5 is grime under the mirror (dirt, K4).

Causes, ranked:
1. The marks prompt names only scratch, scuff, chip, dent and crack. Spark boxed none of the 4 pickup strips and the
   new one; the one it boxed it typed unsure. (Not the seams sentence: matching.seams_not_marks is off in all runs.)
   This one cause takes 4 of the 5 strip lines. A control re-ask with the old prompt (section 3) misses them again.
2. Each strip shows in 2 or 3 keyframes per walk, and on the rear bumper the SAM mask changes shape between them, so the
   moment frame moves: once the strips are boxed (section 3), one strip sits 0.4 to 0.9 units apart across the three
   pickup keyframes, over the 0.2 merge radius. The copies stay apart, the ratio test refuses them, and each copy is
   K2 (single frame).
3. K15: Spark flagged all four front left door crops dirty (37 of 70 crops dirty in the run). It caps every one-walk
   line there, so even a boxed removed strip would end at needs your eye.
4. The zoom refine takes the nearest zoom box even when Spark typed damage and a damage-typed zoom box sits on the mark
   (the v3 copy).

The 37 needs-your-eye lines: glass 11 (windshield 1, rear window 10: tree reflections, leaves), front bumper 9 (8 typed
dirt on views flagged dirty, 1 unsure), hood 4, front left door 4 (above), tail lights 4, front left fender 3, left
front wheel 1, rear bumper 1 (a faint corner line in both walks, K12 K2).

## 2. The fixes (thresholds leaves, absent from every frozen run; config turns them on for new runs)

- matching.glass_by_eye {"enabled", "panels": windshield, rear_window} (lead decision): glass marks are left out of
  matching, the panel gets no verdict and one line "glass: check by eye; reflections make marks on glass unreliable",
  which keeps the mark ids. Side windows are inside door crops whose masks cut the window out (window_subtracted), so
  the on-panel rule already leaves those boxes out.
- matching.debris_by_eye {"enabled", "types": dirt} (lead decision): on a compared panel, needs-your-eye lines whose
  every mark is typed dirt and whose views (the ones K15 reads) are flagged fold into one line "debris on this panel;
  check by eye". A line with any damage type or unsure stays.
- matching.colour_on_paint: one prompt sentence, a stripe or patch of a clearly different colour on the paint (paint
  transfer, tape, a sticker) is a mark, typed scuff; the K17 counter-look counts one in part 1. Paint transfer from
  another car is a common rental charge and looks just like this.
- matching.conditions_on_panel: the prompt asks for wet, dirty, glare and dark of the named panel, not the whole crop.
- matching.inside_other_box (K19, new cap): a one-walk mark whose whitened box lies at least half inside a box the other
  walk drew on its compared keyframe is needs your eye; the other walk did not show clean paint there. Added after the
  NHTSA replay below (a crush box drawn over tape). A centre test capped the crush's own "new" (its centre fell on a
  thin tape box), so the share decides.
- matching.same_walk_register: before the match, each walk's marks of its other keyframes are carried onto its compared
  keyframe through a registration fit of the two crops (register.align; within one walk the light and view barely
  change). A box landing on a mark already there is that mark seen again and leaves matching, and that mark loses K2;
  any other is re-placed and whitened in the compared keyframe. Refused fits change nothing.
- dedup.spark_zoom_refine.prefer_kind: when Spark typed damage, the nearest damage-typed zoom box is taken if there is
  one. On Sanat's hand boxes (zoom_kind.json): pair C 1 of 48 boxes changed (a win), synth 0 of 41.

Not changed: K15 itself. Dropping its "own view flagged: could be dirt" clause for a mark Spark typed damage with high
confidence would let the removed strip reach "not seen at return", but it loosens a cap; that is a lead call.

## 3. Measurements

Prompt re-ask on the stored crops (reask.py, ledger run id prompt_reask_1102, 289 calls new prompt and 289 old prompt
as the same-day control, $2.85, all valid). On the 11 crops that hold the hero's strips, new prompt: every strip boxed in
every crop that shows it, all scuff/high ("light blue tape piece"), the scrape still scratch/medium, nothing on the
pickup right rear door; old prompt again: no strip boxed, one "small blue rectangular mark" unsure at return.

View flags K15 reads (flags.json, crops flagged wet, dirty or glare), control to new prompt: hero 38 to 31 of 70, lexi_pair2
46 to 38 of 78, pairC_r16 37 to 31 of 63, eval_nhtsa15759_r16 34 to 34 of 78.

Offline replays (replay.py on scratch copies; base reproduces each run's labels, rule ids and marks): already there /
new / not seen at return / needs your eye, plus check-by-eye lines.

| run | base (the run) | glass | debris | zoom | glass + debris + zoom |
|---|---|---|---|---|---|
| hero_focus1 | 5 / 0 / 0 / 37 | 3 / 0 / 0 / 26, 2 lines | 5 / 0 / 0 / 16, 6 lines | 5 / 0 / 0 / 37 | 3 / 0 / 0 / 12, 7 lines |
| lexi_pair2 | 2 / 0 / 0 / 11 | 1 / 0 / 0 / 7, 2 | 2 / 0 / 0 / 7, 2 | same as base | |
| pairC_r16 | 2 / 0 / 1 / 17 | 2 / 0 / 1 / 17, 2 | 2 / 0 / 1 / 12, 2 | same as base | |
| eval_nhtsa15759_r16 | 0 / 1 / 0 / 1 | 0 / 1 / 0 / 1, 2 | same as base | same as base | |

Glass removed only glass lines (hero's 2 glass "already there" were reflection and leaf pairs; lexi's was a rear window
speck). Debris folded only dirt-typed lines. The zoom pick changes hero's v3 copy onto the scrape, where it merges with
P kf1 m1; the scrape's three-mark group then has two marks and the dash pair splits into two capped one-walk lines, so
the count stays 37.

With the new answers, the zoom refine and SAM step re-run for their boxes (refine_replay.py, ledger run ids rr_<run>,
$2.67, no marks call) and every switch on plus registration (as new runs have it): before the looks, and with the live
second looks and counter-looks a new run would ask (Spark Open on Baseten, a few cents):

| run | stored run | all switches, before looks | all switches, after live looks |
|---|---|---|---|
| hero_focus1 | 5 / 0 / 0 / 37 | 5 / 2 / 1 / 5, 7 lines | 6 / 0 / 1 / 7, 6 lines |
| lexi_pair2 | 2 / 0 / 0 / 11 | 1 / 0 / 0 / 9, 2 | 2 / 0 / 0 / 8, 2 |
| pairC_r16 | 2 / 0 / 1 / 17 | 0 / 0 / 0 / 12, 5 | 0 / 0 / 0 / 12, 5 |
| eval_nhtsa15759_r16 | 0 / 1 / 0 / 1 | 1 / 1 / 3 / 14, 2 | 1 / 1 / 0 / 17, 2 |

Without the same-walk step (before looks): hero 4 / 3 / 1 / 6 (one bumper strip pair lost, one more false new),
lexi 0 / 0 / 0 / 11 (its door pair lost), NHTSA 1 / 1 / 3 / 20, pair C unchanged.

Per run, after looks:
- hero: kept door strip already there, both bumper strips already there, the scrape and its dash already there; the
  removed strip needs your eye (K15, pickup view flagged dirty); the new right rear door strip needs your eye (K17: Spark
  Open read the boxed strip as "a light blue-white horizontal reflection band along the body crease"; the box sits on
  the strip); a bumper box 110 px beside its strip (the zoom window missed it) capped K17; one false not seen at return
  on the right rear wheel rim (pickup-only box; wheel marks get no counter-look; from this answer sample, not a switch).
- lexi_pair2 against its stored run: M1 already there (both), M2 needs your eye K7 (both), M3 and M6 needs your eye
  (both), M4 silent to a capped pickup-only box, M5 and M7 not comparable (both).
- pairC_r16: the trunk-lid pair and the right rear fender pair still form, d unchanged; the fresh second look said
  different on both (the stored look said same), so needs your eye K7. Its not-seen line on pickup door specks went
  from L1 to capped (unsure/low in the new answer).
- eval_nhtsa15759_r16: the crushed left front door still new, a tape stripe pair already there; the new prompt boxes
  the crash-test calibration tape everywhere, so 3 tape stripes the return answer missed came out not seen at return
  before the looks (2 more were capped by K19 under the return's crush box); the counter-looks capped all 3. Needs your
  eye 1 to 17, all tape or crush lines.

Caveats: one answer sample per arm; pair C and NHTSA truth is by eye; the replays do not run the second marks pass.

## 4. Live re-run hero_focus2

Same pair, `LIVE=1 walkaround run ... --run-id hero_focus2 --keep` (hero.sh step 5; the sample step was not run), code
7e2f5ee, every leaf on, $6.75, 3:55 to 4:04 PM EST (8 min 20 s), rebuild identical.

| truth mark | right verdict | hero_focus1 | hero_focus2 |
|---|---|---|---|
| rear bumper strip, left | already there | silent (no box) | already there (pickup kf2 m1, its kf1 copy merged by the same-walk step; return kf1 m1; d 0.478; look same) |
| rear bumper strip, right | already there | silent (no box) | already there (pickup kf1 m2 carried onto kf2, return kf2 m2 carried onto kf1; d 0.366; look same) |
| front left door strip, kept | already there | needs your eye (return only, unsure/low; K4 K12 K15 K18) | already there (scuff/high both walks, d 0.099; look same) |
| front left door strip, removed | not seen at return | silent (no box) | needs your eye (pickup only, scuff/high, K15: pickup view flagged dirty) |
| right rear door strip, new | new since pickup | silent (no box) | new since pickup (L5 against a pickup speck; counter-look: a mark in the box, not at the pickup spot) |
| front left door scrape | already there | already there | already there (and its small dash, already there) |

Right: 1 of 6 to 5 of 6 (the strips 0 of 5 to 4 of 5). Needs your eye 37 to 11, with 5 check-by-eye lines (windshield
3 boxes, rear window 16, hood debris 11, front bumper debris 10, left front wheel 1).

One new wrong line: rear bumper "not seen at return" on pickup kf2 m2, a box on clean paint above the right strip (the
strip in that crop got no box; the zoom did not move it; the cross-walk fit was refused at mask IoU 0.89 against 0.90).
The counter-look said it saw "a small light-colored scuff" in the box and clean paint at return, so the label stood.
Its return twin (a box right of the strip) was capped by K17. The 11 needs-your-eye lines: the removed strip; 4 on the
rear bumper (3 leaves typed dirt, views not flagged, so not folded; the twin above); 2 trunk lid dirt; and one each on
the left rear door, right rear door (the speck the new strip's L5 pair named), hood, front bumper (K19).

Next, not done: a one-walk mark whose own box does not stand out in its crop (register.saliency under saliency_min,
computable with no fit) should not keep new or gone; that would have capped the stray bumper box. And K15's own-view
clause for a mark typed damage with high confidence (the removed strip) is a lead call.

## Files

reask.py, reask_<run>.json (new prompt), reask_old_<run>.json (control), refine_replay.py, replay.py,
replay_<run>_<variant>[_looks].json, zoom_kind.py and zoom_kind.json, flags.json.
