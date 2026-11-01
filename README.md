# Walkaround

Walkaround turns the pickup and return videos you take of a rental car into a dated, sealed, panel-by-panel evidence
sheet. SAM 3.1 tracks every panel, Muse Spark 1.3 finds marks on blurred crops, plain code matches the marks across the
two walks and writes every label, and the sheet is hashed and timestamped.

Built Oct 26 to Nov 4, 2026 for the Meta Global AI Developer Hackathon (Indie Developer track, one person). It is
evidence support, not legal advice.

## Quick start: the sample pair, no key

You need [uv](https://docs.astral.sh/uv/) (it fetches Python 3.13 itself) or pip with Python 3.13 or newer. No API key,
no Node, no ffmpeg.

```
git clone <this repo's URL> walkaround
cd walkaround
uv venv --python 3.13
uv pip install -e ".[dev]"
.venv/bin/walkaround serve
```

With pip instead of uv (Python 3.13 or newer as `python3`), the two install lines are:

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The first start in a new environment can take 20 to 30 seconds.

Open http://127.0.0.1:8000 and press "open the sample pair (precomputed)". On start the app copied
`samples/pairC_core3/` into `runs/`; the page replays that run's stored events with fixed delays under a "replayed from
a run on Nov 1, 2026" badge, and no model is called. The "sheet" link at the top of the run page opens the evidence
sheet, `evidence.pdf`, `bundle.zip` and `sheet.json`.

The same from a terminal:

```
curl -s http://127.0.0.1:8000/api/sample
curl -s "http://127.0.0.1:8000/api/runs/pairC_core3/replay?mode=instant" | head -c 400
curl -s -o evidence.pdf http://127.0.0.1:8000/api/runs/pairC_core3/sheet.pdf
```

The offline tests need ffmpeg on PATH (the suite makes short synthetic clips with it) and take about two minutes:
`.venv/bin/python -m pytest -q`. The tests that call a model run only with `LIVE=1`.

What the sample shows, honestly: 3 of the 21 panels were compared (trunk lid, rear bumper, right rear wheel) and 18
were not, each with the rules that stopped it (G1, one walk did not film it, on 15; G4, the two masks differ, on 7;
G3 and G6 on 3 each; G9 on 2; G8 on 1; a panel can fail several). The rear bumper was compared on its second keyframe
pair, since the first failed G4 (mask IoU just under 0.7), and the page says so under the frames. On the trunk lid, the
chips the narration names ("trunk lid two small chips by the edge") pair across the two walks 0.09 units apart (about
3 cm), inside the "already there" distance (L3). Muse Spark typed both marks dirt, with low confidence on the return
mark. Since Nov 1 that is a note on a pair in the same place in both walks, not a cap (K4 and K12), and the pair goes
to the second look. Muse Spark Open answered on Baseten (`meta-models/muse-spark-open-1026`, hackathon key, 6.8 s):
same. So the label is "already there". Its reason reads "Both outlined boxes show the same plain white paint area with
no distinct defect, in identical position relative to the nearby rust spots": the look found the two crops alike, not a
chip in each. The other 11 verdicts on the compared panels are "needs your eye". No size was printed: every mark near a
ruler met a geometric refusal (on this car the plate box is the dataset's grey patch, read as a plate turned more than
30 degrees). The page and the sheet show both frames for every panel, so you can look yourself.

## Live mode, with your own Meta Model API key

```
cp .env.example .env          # your key in MODEL_API_KEY; for the second look, BASETEN_HACKATHON_API_KEY or BASETEN_API_KEY
LIVE=1 .venv/bin/walkaround run PICKUP.mp4 RETURN.mp4
```

`walkaround run` needs ffmpeg and ffprobe on PATH. It seals both files, runs every stage and prints each event as one
JSON line, then `run.json`. Without `LIVE=1` it prints what it would do and spends nothing. One video (`walkaround run
PICKUP.mp4`) is a pickup-only check. Then `.venv/bin/walkaround serve` shows the run at `/run/<run_id>`, and
`.venv/bin/walkaround verify <run_id>` rebuilds every derived file and the sheet from the stored model answers, with no
network call, and prints "rebuild identical". The page also takes uploads from a phone; it hashes each file in the
browser while it uploads. The claim page (`/claim`) places a photo from a damage bill against a finished run's pickup
walk.

The sample pair cost $3.3055 in 321 model calls and took 4 min 58 s end to end (its ledger rows in
`docs/evidence/ledger_rows.jsonl`, its event log in `samples/pairC_core3/events.ndjson.gz`). Pass `--tire-code` with
the size printed on the tire's sidewall (for example 225/60R16) so the wheel ruler can state sizes.

Optional: `uv pip install -e ".[seal,pdf]"` and `.venv/bin/playwright install chromium` add the OpenTimestamps stamp and
the evidence PDF. Without them the seal keeps its RFC 3161 token and the run page offers a PDF retry. The second look
tries `BASETEN_HACKATHON_API_KEY` first (it may call only `meta-models/muse-spark-open-1026`), then `BASETEN_API_KEY`
(`meta-models/muse-spark-open`); with neither it falls back to hosted Muse Spark 1.3, and `derived/looks.json` says so.

## What happens in a run

1. Seal: the browser hashes each video while it uploads, and the server hashes it again on arrival and gets an RFC 3161
   timestamp (and an OpenTimestamps stamp) before any model sees it.
2. SAM 3.1 runs 16 text-prompted lanes per walk on a 960 px, 15 fps proxy. Code votes the tracks into 21 panels with
   side and position and lists each panel as filmed, glimpsed, not filmed or not seen. A hood and a trunk lid can look
   alike, so a twin check files each lid by what the walk shows next to it: a plate on it, mirrors near it, which
   lights and glass sit beside it.
3. Plates and faces are pixelated (SAM 3.1 tracks plus an image check on every keyframe) before any crop goes to Spark.
   Each block grows with the box (a plate spans at most 4 blocks), and each track is carried 3 s before its first
   sighting and after its last. On the display video the carried box follows the plate with optical flow and also
   covers its path back to the sighting it left, since on a small box the flow can follow the ground instead.
4. Muse Spark 1.3 marks each blurred panel crop. Muse Voice turns the renter's narration into turns, and code pins each
   turn that names a panel to the seconds SAM shows that panel.
5. Stays Put: SAM 3.1 follows each mark through its own short clip, with up to 12 rigid neighbours (other marks, the
   lights, the plate, the door handle) in the same request; paint moves with its neighbours and a reflection slides
   against them. It is not calibrated yet, so it records the tracks and changes no label.
6. Code matches marks across the walks in a moment-whitened panel frame and writes each label: already there, new or
   unseen, needs your eye, not seen at return, or not comparable. Gates G1 and G3 to G10 make a panel not comparable;
   caps K1 to K15 lower a label. A wet, dirty or glare flag from Spark is a model's judgment, so it no longer blocks a
   panel (gate G2 did until Oct 31): it caps a mark seen in one walk only (K15). In the same way, Spark calling a mark
   dirt or being unsure of it (K4, K12) no longer lowers a pair in the same place in both walks by itself: that pair
   goes to the second look. When a panel's chosen keyframe pair fails a gate of the pair, the next pair is tried. The
   ids show on the page and the sheet. The second look can only lower a label.
7. The sheet (HTML, PDF, JSON) shows both frames, the hashes, the coverage list, the deletions and the privacy line. A
   closing manifest seals it.

## The Meta models

| Model | Where it runs | Step | Why this model |
|---|---|---|---|
| SAM 3.1 (`sam-3.1`) | Meta Model API, Responses API, streamed | 16 lanes per walk (car, license plate, face, car door, bumper, wheel, hood, trunk lid, fender, roof, headlight, tail light, side mirror, windshield, rear window, car window); plate and face checks on keyframes; Stays Put clips | One noun phrase returns every instance with a stable id and a mask on each frame of a shaky phone walk, so panels, coverage and the blur come from tracks |
| Muse Spark 1.3 (`muse-spark-1.3`) | Meta Model API, Responses API with a JSON schema | Marks on one blurred crop at a time (boxes in the 0 to 1000 grid, type, size, crop condition); the panel in a claim photo; a spoken note turned into a box | Grounded, typed boxes from a single image with no training; it never sees both walks and never writes a label |
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Meta Model API, `/v1/asr/transcribe`, ENDPOINTING mode | The renter's narration | Turn start and end times let code pin "trunk lid two small chips" to the seconds SAM shows the trunk lid |
| Muse Spark Open (`meta-models/muse-spark-open-1026` on the hackathon key, `meta-models/muse-spark-open` on an account key) | Baseten Model APIs, OpenAI-compatible Chat Completions | The second look: two outlined crops, same, different or unsure, for each pair inside the "already there" distance that no other cap lowered (Spark's own type and confidence wait for it) | The one model step that compares the two walks runs on open weights, every call is logged with the model id that answered, and an answer can only lower a label to "needs your eye" |

## Open-weights provenance

Every Muse Spark Open request appends one line to `runs/<run_id>/open_weights.ndjson`: the model id Baseten returned,
which key slot asked (`hackathon` or `account`, never the key), both request ids, the latency, the pair of marks, the
verdict, the reason and whether the answer passed the schema. No image bytes and no keys. The app serves it at
`/api/runs/<run_id>/files/open_weights.ndjson` and links it from the run page. A claim's look writes its lines to
`claim/<claim_id>.open_weights.ndjson` instead, since the run's own file is sealed when the run ends; the app serves
it at `/api/runs/<run_id>/files/claim/<claim_id>.open_weights.ndjson`, and the claim's own paragraph says when its
crops went to Baseten.

The sample run asked one second look, on the trunk chips, and logged this line
(`samples/pairC_core3/open_weights.ndjson`, copied to `docs/evidence/second_look/open_weights_pairC_core3.ndjson`):

```
{"baseten_request_id": "54ca701a19590cc0c7e508fa7f5cf1af1eb20", "engine": "baseten", "error": null, "http_status": 200, "key_slot": "hackathon", "latency_ms": 6829, "model": "meta-models/muse-spark-open-1026", "model_requested": "meta-models/muse-spark-open-1026", "panel_id": "trunk_lid", "pickup_mark": "pickup_trunk_lid_kf1_m1", "purpose": "second_look", "reason": "Both outlined boxes show the same plain white paint area with no distinct defect, in identical position relative to the nearby rust spots, with no visible difference between pickup and return.", "request_id": "chatcmpl-195cb5e99811403ba8467eb240a18d90", "return_mark": "return_trunk_lid_kf1_m2", "run_id": "pairC_core3", "status": "completed", "ts": "2026-11-01T05:24:23Z", "valid": true, "verdict": "same"}
```

The same three questions on the same car's trunk lid, the true pair and two controls, were asked on Oct 31 on the
hackathon key: all three answered by `meta-models/muse-spark-open-1026` with HTTP 200 in 3.3 to 7.9 s, same on the
true pair and different on both controls, as stored (`docs/evidence/second_look/open_weights_hackathon_1031.ndjson`).
Two lines of the earlier run on the account key (`docs/evidence/second_look/open_weights.ndjson`, three lines, all
answered by `meta-models/muse-spark-open`):

```
{"baseten_request_id": "54ca701a193e1872479fc91d261cb85931e62", "engine": "baseten", "error": null, "http_status": 200, "latency_ms": 2649, "model": "meta-models/muse-spark-open", "model_requested": "meta-models/muse-spark-open", "panel_id": "trunk_lid", "pickup_mark": "p_paw", "purpose": "second_look", "reason": "Both boxed spots on the trunk lid show a small dark discoloration of similar size and shape at the same relative location with no conflicting appearance.", "request_id": "chatcmpl-6b4c9f62ad5041a89d27b9fe742b2946", "return_mark": "r_paw", "run_id": "b11_fix_smoke_1031", "status": "completed", "ts": "2026-10-31T21:33:15Z", "valid": true, "verdict": "same"}
{"baseten_request_id": "54ca701a193e1adc87a35b54c36fe367002e0", "engine": "baseten", "error": null, "http_status": 200, "latency_ms": 2375, "model": "meta-models/muse-spark-open", "model_requested": "meta-models/muse-spark-open", "panel_id": "trunk_lid", "pickup_mark": "p_paw", "purpose": "second_look", "reason": "First boxed spot is a small dark chip/speck on the paint while second boxed spot is a larger leaf-shaped debris, clearly not the same persistent mark.", "request_id": "chatcmpl-4f77182a7ff84a21bc7518bec14b7214", "return_mark": "r_leaf", "run_id": "b11_fix_smoke_1031", "status": "completed", "ts": "2026-10-31T21:33:25Z", "valid": true, "verdict": "different"}
```

To ask the same three questions again with your own Baseten key (about $0.003 each):

```
LIVE=1 .venv/bin/python scripts/second_look_again.py
```

It reads `docs/evidence/second_look/looks.json`, finds both outlined crops of each look by SHA-256, asks Spark Open and
prints the new answer beside the stored one. A model answer can differ between calls, which is one reason it can only
lower a label. Inside a run, `walkaround verify` replays the stored looks from `derived/looks.json` with no call.

## Measured

Every number here is copied from a file in this repo.

The sample pair, run pairC_core3 on Nov 1: `docs/evidence/ledger_rows.jsonl` (its rows of `runs/ledger.jsonl`).

| What | Calls | Dollars | Time |
|---|---|---|---|
| SAM 3.1 lanes, 16 per walk (296 and 304 frames) | 32 | 1.9200 | p50 13.1 s a lane |
| SAM 3.1 Stays Put clips, 89 frames each | 32 | 0.5696 | p50 18.0 s |
| SAM 3.1 image checks for plates and faces (keyframes and clip prompt frames) | 184 | 0.4600 | p50 1.5 s on keyframes |
| Muse Spark 1.3 marks, one blurred crop a call | 69 | 0.3408 | p50 4.8 s |
| Muse Spark 1.3 voice seeds | 2 | 0.0098 | |
| Muse Voice Transcribe, 19.8 s of audio | 1 | 0.0009 | 5.0 s |
| Muse Spark Open second look on the hackathon key (Baseten list price, hackathon credits) | 1 | 0.0043 | 6.8 s |
| The whole run | 321 | 3.3055 | 4 min 58 s |

Stage times on that run (`samples/pairC_core3/events.ndjson.gz`): seal 4 s, proxy and audio 5 s, upload 42 s, lanes
30 s for both walks, panels and keyframes 42 s, display blur 4 s, marks 33 s, narration 10 s, Stays Put 108 s, match
7 s (the second look was 6.8 s of it), sheet and close 13 s. The runs before it on the same pair: pairC_core2 on Oct 31
night, $3.3083 in 325 calls and 4 min 39 s, with no pair left "already there" and no look
(`docs/evidence/ledger_rows_pairC_core2.jsonl`); pairC_core1 on Oct 31 evening, $3.2299 in 298 calls
(`docs/evidence/ledger_rows_pairC_core1.jsonl`), 6 min 49 s from the first event to run.completed
(`docs/evidence/events_pairC_core1.ndjson.gz`), and no panel compared.

Claims on a copy of the sample run (the sample itself is public and takes no claim): three photos cut from the return
walk at 2.1 s around the trunk chips, plate pixelated, four claims (the renter's one-tap pick reuses every answer), 54
calls, $0.1591 (`docs/evidence/claim_pairC_core3/`). The first stopped at E2 (Spark named the trunk lid, the SAM vote
the hood); with the tap, and on a wider crop, the claim mark could not be placed on the pickup frame (E5). On a crop
centred on the chips, the claim mark paired with the pickup mark on the chips 0.26 units away, inside the claim's widened
"already there" distance, and the label was "needs your eye": Spark called the claim mark a chip and the pickup mark
dirt (K5), and at that time K5 kept the second look away. After the rule change of Nov 1 (K5 on a pair that close waits
for the second look, like K4 and K12), the same photo went to a second copy of the run carrying the new rule
(`docs/evidence/claim_pairC_core3_claim2/`, 19 calls, $0.0567): the claim mark paired with the same pickup mark (d
0.734 inside 0.769), the second look ran on Muse Spark Open on the hackathon key (6.9 s, HTTP 200) and said
different ("a distinct dark linear scratch/mark ... absent in the corresponding clean white area"), so the claim is
"needs your eye" through K7.

The D1 gate on the same pair: `docs/evidence/D1_MEMO.md` (a copy of `runs/d1/uco3d_pairC/D1_MEMO.md`).

- SAM 3.1 at 15 fps: p50 completion 14.0 s for a 304-frame lane, 16 lanes at once accepted, no request lost to a 429,
  and frames billed equal to frames sent on every lane (E2).
- The same "car door" request three times: mean best mask IoU 0.968 to 0.991 between repeats (E2.variance).
- The repo's streaming reader of SAM's output agrees with meta-sam-parser on a lane (boxes, ids, frames) and parsed it
  in 0.013 s where the parser took 34.4 s (E2.oracle).
- Inside the windows where the plate lane saw a plate, its tracks covered every frame (E9). Outside them it can miss:
  see Privacy.
- `store: false` echoed by SAM 3.1 on every lane and by Muse Spark 1.3 on 49 of 49 answers (E12).
- Muse Spark 1.3 box jitter on real marks: 2.0% of crop width on the trunk chips and 2.4% on a hood dent; it found 6 of
  8 hand-marked chips (lead note 3).
- Claim photos: Spark's panel name agreed with the SAM vote on 2 of 3 (E11).
- Muse Voice on our narrated copy of the pickup walk: 3 turns, and every turn that names a panel landed inside that
  panel's SAM window (lead note 8).

The eval (`eval/results.json`, from the run snapshots in `eval/snapshots/`; each pair stands alone, nothing pooled;
panel labels by the developer from the frames, not yet reviewed; no mark labels yet, so no precision or recall):

| Pair (table) | Run | What it shows |
|---|---|---|
| uCO3D pair C, cross walk, nothing changed (B) | pairC_core3 | False "new": 0 of 6 return-mark verdicts. Needs your eye: 11 of 12 verdicts on compared panels. Re-shoot match: 1 of 7 pickup marks found again as "already there". Coverage state right: 26 of 39 (67%) panel views |
| NHTSA test 15759, before and after the side impact (D) | eval_nhtsa15759 | Changed panels called new: 1 of 1 compared (the left front door's crush, L2). The left rear door, also struck, was not compared: G4 on all 3 keyframe pairs tried (mask IoU 0.699 on two, the panel axes disagreeing on the third; the floor is 0.7). Panel calls right: 3 of 3. Coverage state right: 22 of 35 (63%) |
| Pexels 4228872 split at 29 s, nothing changed (C, fit) | eval_px4228872_front | 1 panel compared (front bumper, no marks on either frame), 9 not comparable (G1 on all 9). False "new": 0 of 0. Coverage state right: 8 of 12 |
| Pexels 4228872 split at its scene cut, nothing changed (C) | eval_px4228872_cut | No panel compared: 16 not comparable (G1 on all 16; the two halves barely overlap), 6 not compared. False "new": 0 of 0. Coverage state right: 14 of 26 (54%) |
| Pexels 4228872 and 4281322, two takes from one shoot, nothing changed (C) | eval_px_two_takes | 1 panel compared (rear bumper), 9 not comparable (G1 7, G4 4, G6 2, G9 1). False "new": 1 of 1 return-mark verdicts: a return mark Spark typed chip, paired with a pickup mark beyond tau_eye (L5); by my eye both are parking-sensor dots, not damage (not a reviewed label). Panel calls right: 0 of 1. Coverage state right: 23 of 34 (68%) |

The five runs cost $3.3055, $3.4017, $1.7397, $2.5255 and $1.6754 (the snapshots' `ledger.jsonl`). The two-takes row
is the one false alarm in the eval so far: a "new or unseen" call on a car where nothing changed.

The reshaped-panel rule of Nov 1 (a panel that fails only the mask IoU gate on every keyframe pair, with its box shape
unchanged and the outline difference in one place, is "needs your eye", rule L7) came after the NHTSA run, which keeps
its frozen rules. Replayed offline with the rule on (`docs/evidence/replay_1101/`), the struck left rear door becomes
"needs your eye" (L7: box aspect ratios 0.988 and 0.984, 0.4% apart; the largest blob holds 64% of the outline
difference) and nothing else changes; on pairC_core3 no verdict changes.

## Footage and data

Only footage we have rights to is in this repo or behind any number in it.

- Car videos: uCO3D (Liu et al., Meta, 2025), CC BY 4.0, https://github.com/facebookresearch/uco3d . License:
  https://creativecommons.org/licenses/by/4.0/ . We cropped, re-encoded and blurred plates in these clips. On screen:
  "Footage: uCO3D (Meta), CC BY 4.0, modified". Paper: arXiv:2501.07574. The sample pair is uCO3D pair C
  (7336-29949-54766 as the pickup, 50152-5241-46685 as the return), one Hindustan Ambassador filmed twice. It is a
  dataset car, not a rental.
- Narration on the sample's pickup walk: our own script, spoken by Piper TTS (MIT) with its en_US-ljspeech-high voice,
  trained on the public-domain LJ Speech dataset, and laid onto the uCO3D clip by us. The dataset clips have no audio.
- Stock video, used in the eval (labels and run eval_px4228872_front): WeStarMoney Rec on Pexels, https://www.pexels.com/@westarmoney/ , Pexels
  License, https://www.pexels.com/license/ . Clips 4228872 and 4281322. No endorsement by any brand shown is implied.
- Crash test video, used in the eval (labels and run eval_nhtsa15759): NHTSA Vehicle Crash Test Database, test 15759,
  https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results/get-multimedia-files/15759 , public
  information under NHTSA's terms of use. NHTSA does not endorse this project.

The source clips are not in the repo (`data/` is ignored); each label file in `eval/labels/` names its footage.

A run made after Nov 1 with a footage credit prints that credit and the license link on page 1 of its evidence sheet,
in the footer of every page (so in the PDF too) and in the letter. The sample's sheet does: its run was made on Nov 1
with the uCO3D credit above and our narration line, and the walk player on the run page shows the same credit.

## Privacy

- Faces are always pixelated. Plates are pixelated unless the renter ticks "this is the car I rented"; then the plate
  stays only on that renter's own sheet and PDF and never on a public page.
- The app serves a fixed allowlist of run files (`served_file_allowed` in `walkaround/privacy.py`). Originals, unblurred
  proxies, the audio track, raw model answers and the quarantine are never served.
- Unblurred originals and proxies are deleted when a run ends; the sheet prints when, and `deletions.json` and the run
  page list each file's SHA-256. Model API uploads are set to expire after an hour and are deleted at run end, and every
  request is sent with storage off.
- Processed with Meta's Model API in Standard mode, which Meta's documentation says is not used to train Meta models.
  Meta's terms allow it to retain content for abuse review and legal reasons; Walkaround does not control that.
- The second look sends two blurred crops to Muse Spark Open on Baseten, a company separate from Meta; no video goes
  there. A claim's paragraph says so when its look went there. We have not confirmed Baseten's retention for Model
  APIs, only what we send.
- The sample in `samples/`: every file is the run's own bytes, except `run.json` (the replay flag); no blur was added
  by hand. We looked at all 106 images, the five clips, every second frame of both display videos where a plate track
  or its 3 s carry runs and every third frame elsewhere, and every frame at full size where a plate turns edge-on.
  Plates sit under blocks of about a quarter of the plate or larger. The spot the Oct 31 sample left open, the front
  strip plate edge-on at 12.3 to 12.6 s of the return walk, is under blocks on this run (the 3 s carry and the path
  cover of Nov 1).

## Limitations

- One live run so far has printed "already there" after a second look (the sample, above), and that look's own reason
  says it saw the same plain paint in both crops, not a chip in each. No live run has stated a size, and no claim has
  reached its second look (the one compared claim was lowered by K5 first).
- Twin panels: on this car SAM 3.1's hood lane holds the trunk lid when the camera is at the rear and the other way
  round. The twin check put both walks' lids right on the sample run, but it reads cues (a plate on the lid, mirrors
  near the hood) that a car with its plate on the bumper or its mirrors on the doors would not give. One such cue
  alone, or the light and glass lanes alone, does not turn a lid against its lane: the lid is left uncertain and not
  compared.
- A wet, dirty or glare flag from Spark caps a mark seen in one walk only (K15); on this weathered car 38 of 69 crops
  carry one. Since Nov 1 a flagged walk crop no longer refuses a size (the size would carry a note), but on this car
  every mark near a ruler met a geometric refusal, mostly the plate box read as turned past 30 degrees.
- Stays Put is not calibrated, so it changes no label. Its neighbour test separated a reflection from paint on a dark
  car, but on this white car's trunk lid too few neighbours stayed tracked. Since Nov 1 marks paired across the walks
  and marks the narration named get their clips first, so both trunk chips got one, but SAM 3.1 followed the chip on
  58 and 42 of 89 frames, under the 80% it needs (no track).
- Muse Voice sometimes merges several sentences into one long turn: the same narration came back as 3, 2 and 4 turns
  on three runs, the longest 11.2 s and 16.1 s (`docs/evidence/voice_turns_pairC.json`). It mishears some words
  ("like scratch" for "light scratch"); the speaker markers it writes into transcripts are left out of the words shown.
- The eval has three runs and panel labels only (not yet reviewed, no mark labels), and there is no real renter's walk:
  every car here comes from a dataset or a crash test.
- The sample cannot be re-verified from the repo: `walkaround verify` needs the run's stored model answers
  (`spark.json`, `voice.json`), which the app never serves, and says so. Verify works on your own runs.
- Built and tested on macOS with Python 3.13 (the quick start also ran on 3.14); not tried on Windows.

## Built Oct 26 to Nov 4, 2026

Everything in this repo was written for the Meta Global AI Developer Hackathon between Oct 26 and Nov 4, 2026. The
pieces that existed before and are used as published: the `@meta-sam` packages (`@meta-sam/parser`,
`@meta-sam/graphics`, `@meta-sam/video`, `@meta-sam/react`) for drawing SAM output over the video in the browser,
`meta-sam-parser` (Python) to decode SAM masks and as the reference the repo's own wire reader is checked against, and
patterns from Meta's model cookbook (streaming a SAM request and closing its output, the Muse Voice file upload,
grounding prompts in the 0 to 1000 grid). Other dependencies are open-source packages listed in `pyproject.toml` and
`web/package.json`.

Layout: `walkaround/` is the Python package (pipeline, model clients, matching, privacy, seal, sheet, app), `web/` the
browser app (its build in `web/dist/` is committed so a clone runs without Node), `config/thresholds.json` every
threshold with its source, `samples/` the sample pair, `scripts/` the gate, eval and sample tools, `eval/` labels and
results, `docs/CONTRACTS.md` the module contracts, `docs/evidence/` the files the numbers above come from.

## License

The code is MIT licensed (`LICENSE`). Dependencies keep their own licenses (`meta-sam-parser` is under the SAM License).
The footage in `samples/` is a modified uCO3D clip under CC BY 4.0 with the credit above, not under MIT.
