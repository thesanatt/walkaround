# Eval

Labels and numbers for Walkaround's eval (docs/ARCHITECTURE.md section 10). `scripts/eval_run.py` reads the runs and
the labels in this folder and writes `eval/results.json`; any eval number in the project README is copied from that
file. Nothing in this folder calls a model.

## Pairs

Only footage we have rights to. The organizers ruled on Oct 31 that this covers evaluation data too, so nothing under
data/eval_internal/ is used here, and `eval_run.py` refuses a labels file whose footage points there.

| Pair | Kind | Table | Split | Walks | What changed |
|---|---|---|---|---|---|
| uco3d_pairC | cross walk | B | report | uCO3D 7336-29949-54766, then 50152-5241-46685: one white Ambassador filmed twice in different light | nothing |
| nhtsa15759 | before and after | D | report | NHTSA test 15759 documentation video, 32 to 52 s, then 112 to 142 s | the two left doors (the struck side) |
| px4228872_front | pseudo pair | C | fit | Pexels 4228872, 13.79 to 29.0 s, then 29.0 to 44.875 s | nothing (one clip split in two) |
| px4228872_cut | pseudo pair | C | report | Pexels 4228872, 0 to 13.79 s, then 13.79 to 44.875 s | nothing (one clip split in two) |
| px_two_takes | pseudo pair | C | report | Pexels 4228872 from 0 to 13.79 s, then Pexels 4281322 | nothing (two takes from one shoot) |

Each pair's numbers stand alone and are never pooled across tables (docs/DATA_HUNT.md 4.1). Thresholds may be fit on
the "fit" pair; the "report" pairs show what that gives (section 10 item 3).

The two time splits are the weakest noise test: same pass, same light, seconds apart. DATA_HUNT 4.1 names two other
table C pairs, 4281322 against the left side of 4210869 and 4228872 against 4281029. This build uses only clips 4228872
and 4281322, so those two are not here. TODO after the first live runs: if px4228872_cut and px_two_takes compare 0
panels, ask the lead to add the DATA_HUNT pairs (4210869 needs the person in the driver's window blurred).

Credits, word for word from docs/DATA_HUNT.md 3.8. The eval uses Pexels clips 4228872 and 4281322 and NHTSA test
15759 only:

- "Car videos: uCO3D (Liu et al., Meta, 2025), CC BY 4.0, https://github.com/facebookresearch/uco3d . License:
  https://creativecommons.org/licenses/by/4.0/ . We cropped, re-encoded and blurred plates in these clips."
- "Stock video: WeStarMoney Rec on Pexels, https://www.pexels.com/@westarmoney/ , Pexels License,
  https://www.pexels.com/license/ . Clips 4228872, 4281322, 4210869, 4281335 and 4281029. No endorsement by any brand
  shown is implied."
- "Crash test photos and video: NHTSA Vehicle Crash Test Database, tests 15759, 15637 and 15641,
  https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results/get-multimedia-files/15759 (same path
  for the others), public information under NHTSA's terms of use. NHTSA does not endorse this project. Plates and faces
  blurred."

## Making the numbers

```
.venv/bin/python scripts/pseudo_pair.py list       # pairs, sources, what is on disk, a cost estimate per run
.venv/bin/python scripts/pseudo_pair.py cut        # the cut walks into data/derived/ (ffmpeg only)
LIVE=1 .venv/bin/walkaround run <pickup> <return> --run-id eval_<pair>
.venv/bin/python scripts/eval_run.py               # tables on stdout, eval/results.json
.venv/bin/python scripts/eval_run.py --snapshot    # copy each run's JSON to eval/snapshots/, then score from there
.venv/bin/python scripts/eval_run.py --from-snapshots
.venv/bin/python scripts/pseudo_pair.py calibrate eval_px4228872_front
```

`eval_run.py` prints the exact command for anything missing: the run of a pair, the mark labels, and for pair C an
optional second run for run-to-run stability. It finds a pair's runs by the sha256 of both originals
(videos/<role>.json), by the labels' run_id, or by the names `eval_<pair>` and `eval_<pair>_<n>`; `--run PAIR=RUN_ID`
picks one. A refused labels file stops it from writing results.json at all, so a stale file is never half updated.

`pseudo_pair.py calibrate` finds the run's pair from its id or from the sha256 of both walks and refuses a before and
after pair (nhtsa15759) even without `--pair`.

Spend. `pseudo_pair.py list` prints a cost per run from D1's measured rates (E13, runs/d1/uco3d_pairC/D1_MEMO.md):
lanes at $0.0496 per second of video plus about $2.35 per pair for keyframe checks, Stays Put and Spark. That gives
$3.78 to $4.83 per pseudo or NHTSA pair and $4.33 for pair C, about $21 for all five, plus about $4.33 if pair C gets
its optional second run (estimates; each run's ledger rows are the real number). `walkaround run` checks only the run's
own $8 ceiling, not the $20 daily card ceiling, so do not start all of them in one day without Sanat's yes. In order:
uco3d_pairC, px4228872_front and nhtsa15759 (about $13), then px4228872_cut and px_two_takes on another day.

Keeping a run. `walkaround run` makes runs with keep false, and the app's sweeper deletes those 24 hours after creation,
first when `walkaround serve` starts and then every hour. A snapshot holds no crops, so a deleted run can still be
scored but no longer labeled. Label on the day of the run, and start the app as
`RUN_RETENTION_HOURS=720 .venv/bin/walkaround serve` (30 days) until `walkaround run` gets a way to set keep.

## What eval_run.py reports

Per pair, from the run (runs/<id>/ or eval/snapshots/<id>/) and eval/labels/<pair>.json:

- **False "new" where nothing changed.** On a pair labeled nothing_changed: new_or_unseen verdicts among the
  verdicts that involve a return mark, on comparable panels, and how many comparable panels got one. This is the error
  that would bill the renter.
- **Panel calls.** A comparable panel is called changed when any verdict on it is new_or_unseen, needs your eye when
  any is needs_your_eye and none is new, else unchanged. A call is right when it agrees with the panel's changed label.
  A panel the reshaped-panel rule calls "outline changed in place" (L7, runs made after its leaf, Nov 1) counts as a
  needs-your-eye panel call, not as not comparable.
  Needs-your-eye panels, not-comparable panels (with their gate ids) and panels with no keyframe are counted apart.
- **Needs your eye** share of the verdicts on comparable panels, and **re-shoot match**: pickup marks on unchanged
  comparable panels that got already_there.
- **Coverage state**: each labeled coverage against coverage.json, in all and per panel. not_seen reads as not_filmed;
  a lane that did not run is left out and counted. A side swap is a left and right pair whose states traded places.
- **Mark precision**: real calls over the system marks that have a call. **Mark recall**: real labeled marks that a
  system mark finds within 0.25 units (`eval.recall_radius_units`), counted only where the run has a keyframe of that
  panel in that walk, so recall stays conditional on the panel being filmed; also per size class, using the size the
  labeler set when it differs from the system's. Recall counts every real mark the labeler confirmed or added. A first
  pass with the system's marks hidden lowers the bias, but a mark the labeler only noticed in a system box still counts
  as found, so recall leans optimistic.
- With mark labels: **false new** (labeled pre-existing return marks called new), **false pre-existing** (labeled new
  return marks called already there), mark verdicts right, how many needs-your-eye verdicts with a return mark the
  labeler resolved as already there (a pickup-only one, a missing mark held back by a cap, can never resolve that way and
  is counted apart), and the **labeler-override rate**: system marks called not real plus marks added by hand, over the
  system marks with a call plus the marks added.
- **Stays Put agreement** where mark_tracks.json has tracks: a real mark should stay fixed, a mark labeled a reflection
  should slide, dirt on the paint stays fixed. Inconclusive and no-track clips and the controls are counted apart.
- **Run to run**, when a pair has two runs: SAM object ids per lane always; coverage states, best keyframe moments,
  marks found again within 0.25 units and verdict counts per panel only when both runs froze the same thresholds
  (config hash), since a threshold change is not model variance.
- **Cost and wall clock per walk** from runs/ledger.jsonl: a SAM row carries its walk; a Spark row's walk comes from
  its spark.response event or its crop in spark.json; Voice rows stay unattributed. Dollars per second of video, wall
  clock from the run's event log (first event to the end, last upload to the end, seconds per stage), p50 first delta
  and parse CPU per lane, Spark tokens per call.
- The run's calibration (tau_pre, tau_eye and the sigmas), the door-handle residual (median and max length over
  comparable panels, from matches.json), the **chance-match probability** of section 5.10 (pi x tau_pre^2 / 12 per
  pickup mark, times the pickup marks per comparable panel; computed, not measured), c_present_share per noun,
  **would-be-billable** counts at 10, 20 and 25.4 mm (new_or_unseen verdicts with a ruler size, the only ones that could
  be billed; needs_your_eye on a line of its own; already_there and not_seen_at_return never count), and on a pseudo
  pair the noise floor: "N of M pickup marks matched, same car, one clip split in two (noise floor)", or "two takes from
  one shoot", with the tau_pre the pair implies.
- **Method page lines** (results.json `method_lines`): false new, re-shoot match, the needs-your-eye share with how
  many the labeler resolved, and the noise floor, one line per pair with N, marked when the labels are not yet
  reviewed. The sheet's method page can copy them as they are.

Counts under 20 print as "7 of 9"; from 20 up as "41 of 53 (77%)". The run's frozen thresholds drive calibration; the
eval radii come from config/thresholds.json (`eval`). What is not computed, and why, is listed at the end of the output
(`not_computed`).

## Labels: walkaround_labels_v1

One file per pair, eval/labels/<pair_id>.json. `eval_run.py` refuses a file that breaks this and names every problem.

| Field | Value |
|---|---|
| schema | "walkaround_labels_v1" |
| pair_id | letters, digits, _ and -; the file name without .json (a second browser download saved as "<pair> (1).json" is refused until renamed) |
| status | who labeled it and whether anyone reviewed it, e.g. "labeled by the developer from the frames, not yet reviewed" |
| labeler, labeled_on | optional name and YYYY-MM-DD |
| kind | cross_walk, pseudo or before_after |
| table | A to E, the table of docs/DATA_HUNT.md 4.1 |
| split | fit or report |
| nothing_changed | true, false or null for the pair as a whole |
| footage.pickup, footage.return | path (repo-relative walk file, never under data/eval_internal/), sha256 of that file or null, t0 and t1 (the span cut from the source), source, credit |
| run_id | the run the mark labels were made on, or null when the file has panel labels only |
| panels | an object keyed by panel id (walkaround/contracts.py PanelId); a missing panel is unlabeled |
| panels.<id>.changed | true, false or null (cannot tell from the frames) |
| panels.<id>.coverage | pickup and return, each filmed, glimpsed, not_filmed or null |
| panels.<id>.marks_checked | pickup and return, true once every system mark on that walk's crops has a call and the labeler looked for missed marks |
| panels.<id>.note | free text |
| marks | mark labels on run_id's crops (below) |
| reference_marks | run-independent marks (below) |
| notes | free text |

Coverage words, for the labeler (the system's own bar, docs/ARCHITECTURE.md 5.4 and `coverage` in thresholds.json):

- **filmed**: at least 85% of the panel is in the frame, in focus, at 3% of the frame or more, in at least one frame.
- **glimpsed**: less of the panel is seen, or all of it but smaller than that, blurred or at a steep angle.
- **not_filmed**: none of it is seen.
- null: too close to call. Leave it null rather than guess.

A **mark label** is either a system mark or one the labeler added:

- `{"mark_id", "source": "system", "role", "panel_id", "kf_id", "real", "new", "not_real_reason", "size_class",
  "note"}`: real is true, false or null (not judged); new is true when a return mark was not there at pickup, false when
  it was, null when not said (on a panel labeled unchanged it reads as false); not_real_reason is reflection, dirt or
  other; size_class (optional) is the labeler's S, M or L where the system's was wrong.
- `{"mark_id": "hand_<n>", "source": "hand", "role", "panel_id", "kf_id", "point_crop": [x, y], "type", "size_class",
  "real": true, "new", "note"}`: a mark the system missed, at a pixel of that keyframe's blurred crop; type from the
  marks_v1 list, size S, M or L. A hand mark within 0.25 units of a system mark called real is that mark found again
  (a first pass with the system's marks hidden) and counts once.

A **reference mark** does not depend on a run: `{"ref_id", "panel_id", "role", "t", "point_src": [x, y], "type",
"size_class", "real", "new", "not_real_reason", "group", "note"}`. t is a 15 fps proxy slot of the walk file (k/15 s,
to 6 places, as the run's pts map stores it) and point_src is a pixel of the frame that slot shows (what
video.read_frame_at returns at t). It counts only on a run that cut a keyframe of that panel at that very slot, because
only then does the crop come from the same source frame. A run picks its own keyframe slots, so on most runs a
reference mark does not count; it then tells the labeler where to look (the labeling page lists it under its panel and
rings it on a crop cut at its slot). Marks that share a group are one mark; a group counts at most once per walk.

On the labeled run, the labeler's calls are used as they are, and a system mark with no call stays unjudged even on a
checked panel. On another run of the same walks, a labeled mark is placed in the panel frame (geometry.whiten with its
keyframe's moments; wheel coordinates on a wheel, from the crop's panel mask), exactly as marks.py places a Spark mark.
A system mark within 0.35 units of a labeled mark (`eval.false_mark_radius_units`) takes its call; one with nothing near
it on a panel whose marks were checked is a false mark; anything else stays unjudged.

## The labeling page

web/public/eval/label.html is one static file with no build step; Vite copies public/ into dist/, and the app serves
dist/eval/label.html at /eval/label.html (TODO for the integrator: the app still answers that path with the web app
shell; it needs its own route, see the lane report). Start the app and open the run:

```
RUN_RETENTION_HOURS=720 .venv/bin/walkaround serve
open 'http://127.0.0.1:8000/eval/label.html?run=eval_uco3d_pairC&pair=uco3d_pairC'
mv "$(ls -t ~/Downloads/uco3d_pairC*.json | head -1)" eval/labels/uco3d_pairC.json
```

The page reads only what the app already serves to the run's owner: run.json, derived/keyframes.json, marks.json,
coverage.json, matches.json, videos/<role>.json, the blurred crops, the blurred keyframe frames and the blurred walks.
It saves nothing on the server. In order:

1. Load the pair's current labels file. Its panel labels and reference marks carry over; its mark labels carry over
   only when they were made on this same run. If they were made on another run, the page says so and the first press
   of Download warns that the download leaves them out (eval_run.py reads one file per pair); keep the old file, or
   press Download again to save without them.
2. First pass: tick "hide the system's marks" and add every mark you see. Click an empty spot on a crop to add one,
   with the type and size from the bar; with the keyboard, focus a crop, press Enter, move the mark with the arrow
   keys (Shift for bigger steps), Delete removes it.
3. Show the system's marks and judge each box: click it to cycle real, not real and not judged, or use the buttons.
   Say why a mark is not real, fix a real mark's size if needed, and on the return walk say whether it is new since
   pickup or was there at pickup.
4. Set changed and coverage per panel, then tick "all marks checked" for each walk you finished. The box stays off
   until every system mark of that walk's crops has a call and goes off again if one goes back to not judged; a walk
   with no crop of the panel cannot be checked.
5. Download and move the newest download into eval/labels/ (the `mv` above). The status line of a downloaded file
   names the labeler and says it was not reviewed.

## The starting labels

The five files here are marked "labeled by the developer from the frames, not yet reviewed". They were made on Oct 31
from frames pulled with ffmpeg (contact sheets every 0.5 to 2.5 s, full frames where a mark is), read by eye. They hold
panel labels and, for pair C, reference marks; no mark labels yet, because no eval run existed then.

- uco3d_pairC: nothing changed. Chips above and left of the trunk plate and a chip under the right front door window
  are in both walks; a leaf (return only) and a twig (pickup only) on the trunk lid are not marks. The left side is
  only seen at the corners, so its panels are not labeled changed or unchanged. The reference marks were first read on
  source frames that no 15 fps slot shows; on Oct 31 they moved to the slots 16.4 s and 9.533333 s (pickup) and
  1.533333 s and 8.0 s (return), with each point carried to the slot's frame by template matching and checked by eye.
- nhtsa15759: the two left doors changed. The left rear fender is left unlabeled (its seam with the door moved; the
  panel looks the same at this resolution). Everything else both walks show is unchanged; the right side appears only
  before. The tape grid and the PRE-TEST and POST-TEST placards are on the car in both walks and are not marks.
- px4228872_front, px4228872_cut, px_two_takes: one continuous shoot, so nothing changed anywhere. Black paint with
  strong reflections; a mark Spark reports here is most likely a reflection or dust.

Coverage was first read as "the whole panel" for filmed. Under the 85% wording above, the panels labeled glimpsed only
because the frame cut them were read again (pair C's pickup windshield, px_two_takes' return left doors); none changed.

## Snapshots

runs/ is not in the repo. `--snapshot` copies what the metrics read (run.json, videos/*.json, derived/*.json,
spark.json, the lane summaries, the panel masks, the timing events and the run's ledger rows) into
eval/snapshots/<run_id>/; it copies no frame or crop image and no voice.json. It also leaves out Spark's answer text
(the eval reads only which crop a call was on), the renter's spoken words on narration_seed marks, the original file
name, and the camera time and device; the scores read the same redacted copy from runs/, so `--from-snapshots` gives
the same results.json bytes from the repo alone. results.json holds no clock time and lists the sha256 of every file
it read, as read. Each snapshot's snapshot.json names the contracts.py it was made with; a snapshot that no longer
reads after a contracts change says so. Snapshot the friend's walk only with her written OK.

## Limits

- Tens of marks at most, not the 60 to 100 the strategy hoped for. Under N = 20 the counts print without a percentage.
- No licensed walk shows new damage at rental scale; the one known change is a crash test.
- No night, rain or multi-day pair, and no walk filmed panned or backwards on purpose.
- The starting labels are one reading of the frames by one person and have not been reviewed.
- Reference marks count only on a keyframe at their exact slot, so most of the truth comes from labeling each run.
