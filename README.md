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
`samples/pairC_core2/` into `runs/`; the page replays that run's stored events with fixed delays under a "replayed from
a run on Oct 31, 2026" badge, and no model is called. The "sheet" link at the top of the run page opens the evidence
sheet, `evidence.pdf`, `bundle.zip` and `sheet.json`.

The same from a terminal:

```
curl -s http://127.0.0.1:8000/api/sample
curl -s "http://127.0.0.1:8000/api/runs/pairC_core2/replay?mode=instant" | head -c 400
curl -s -o evidence.pdf http://127.0.0.1:8000/api/runs/pairC_core2/sheet.pdf
```

The offline tests need ffmpeg on PATH (the suite makes short synthetic clips with it) and take about two minutes:
`.venv/bin/python -m pytest -q`. The tests that call a model run only with `LIVE=1`.

What the sample shows, honestly: 3 of the 21 panels were compared (trunk lid, rear bumper, right rear wheel) and 18
were not, each with the rules that stopped it (G1, one walk did not film it, on 15; G4, the two masks differ, on 8;
G3, G6 and G9 on 2 each; G8 on 1; a panel can fail several). On the trunk lid, the chips the narration names ("trunk
lid two small chips by the edge") pair across the two walks 0.04 units apart (about 1.5 cm). Code's distance rule says
"already there" (L3), but Muse Spark typed both marks dirt with low confidence, so K4 and K12 lower the label to
"needs your eye", and the sheet adds that both views were flagged for dirt and glare while the mark sits in the same
place in both walks. No second look was asked (it runs only on a pair code is about to call "already there") and no
size was printed (the crops carry dirt or glare flags, and the one clean rear-bumper view had its plate turned past 30
degrees). The page and the sheet show both frames for every panel, so you can look yourself.

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

The sample pair cost $3.3083 in 325 model calls and took 4 min 39 s end to end (its ledger rows in
`docs/evidence/ledger_rows.jsonl`, its event log in `samples/pairC_core2/events.ndjson.gz`). Pass `--tire-code` with
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
   Each block grows with the box (a plate spans at most 4 blocks), and each track is carried 2 s before its first
   sighting and after its last.
4. Muse Spark 1.3 marks each blurred panel crop. Muse Voice turns the renter's narration into turns, and code pins each
   turn that names a panel to the seconds SAM shows that panel.
5. Stays Put: SAM 3.1 follows each mark through its own short clip, with up to 12 rigid neighbours (other marks, the
   lights, the plate, the door handle) in the same request; paint moves with its neighbours and a reflection slides
   against them. It is not calibrated yet, so it records the tracks and changes no label.
6. Code matches marks across the walks in a moment-whitened panel frame and writes each label: already there, new or
   unseen, needs your eye, not seen at return, or not comparable. Gates G1 and G3 to G10 make a panel not comparable;
   caps K1 to K15 lower a label. A wet, dirty or glare flag from Spark is a model's judgment, so it no longer blocks a
   panel (gate G2 did until Oct 31): it caps a mark seen in one walk only (K15). The ids show on the page and the
   sheet. The second look can only lower a label.
7. The sheet (HTML, PDF, JSON) shows both frames, the hashes, the coverage list, the deletions and the privacy line. A
   closing manifest seals it.

## The Meta models

| Model | Where it runs | Step | Why this model |
|---|---|---|---|
| SAM 3.1 (`sam-3.1`) | Meta Model API, Responses API, streamed | 16 lanes per walk (car, license plate, face, car door, bumper, wheel, hood, trunk lid, fender, roof, headlight, tail light, side mirror, windshield, rear window, car window); plate and face checks on keyframes; Stays Put clips | One noun phrase returns every instance with a stable id and a mask on each frame of a shaky phone walk, so panels, coverage and the blur come from tracks |
| Muse Spark 1.3 (`muse-spark-1.3`) | Meta Model API, Responses API with a JSON schema | Marks on one blurred crop at a time (boxes in the 0 to 1000 grid, type, size, crop condition); the panel in a claim photo; a spoken note turned into a box | Grounded, typed boxes from a single image with no training; it never sees both walks and never writes a label |
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Meta Model API, `/v1/asr/transcribe`, ENDPOINTING mode | The renter's narration | Turn start and end times let code pin "trunk lid two small chips" to the seconds SAM shows the trunk lid |
| Muse Spark Open (`meta-models/muse-spark-open-1026` on the hackathon key, `meta-models/muse-spark-open` on an account key) | Baseten Model APIs, OpenAI-compatible Chat Completions | The second look: two outlined crops, same, different or unsure, for each pair code already called "already there" | The one model step that compares the two walks runs on open weights, every call is logged with the model id that answered, and an answer can only lower a label to "needs your eye" |

## Open-weights provenance

Every Muse Spark Open request appends one line to `runs/<run_id>/open_weights.ndjson`: the model id Baseten returned,
which key slot asked (`hackathon` or `account`, never the key), both request ids, the latency, the pair of marks, the
verdict, the reason and whether the answer passed the schema. No image bytes and no keys. The app serves it at
`/api/runs/<run_id>/files/open_weights.ndjson` and links it from the run page. A claim's look writes its lines to
`claim/<claim_id>.open_weights.ndjson` instead, since the run's own file is sealed when the run ends.

The sample run asked no second look: its one pair (the trunk chips) was already lowered to "needs your eye" by its own
caps. The same three questions on the same car's trunk lid, the true pair and two controls, were asked on Oct 31 on the
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

The sample pair, run pairC_core2 on Oct 31: `docs/evidence/ledger_rows.jsonl` (its rows of `runs/ledger.jsonl`).

| What | Calls | Dollars | Time |
|---|---|---|---|
| SAM 3.1 lanes, 16 per walk (296 and 304 frames) | 32 | 1.9200 | p50 13.0 s a lane |
| SAM 3.1 Stays Put clips, 89 frames each | 32 | 0.5696 | p50 14.3 s |
| SAM 3.1 image checks for plates and faces (keyframes and clip prompt frames) | 188 | 0.4700 | p50 1.6 s on keyframes |
| Muse Spark 1.3 marks, one blurred crop a call | 70 | 0.3371 | p50 4.5 s |
| Muse Spark 1.3 voice seeds | 2 | 0.0107 | |
| Muse Voice Transcribe, 19.8 s of audio | 1 | 0.00095 | 5.1 s |
| The whole run | 325 | 3.3083 | 4 min 39 s |
| Muse Spark Open second looks on the hackathon key (`docs/evidence/second_look`) | 3 | 0.0093 | 3.3 to 7.9 s |

Stage times on that run: lanes 33 s for both walks, panels and keyframes 48 s, display blur 4 s, marks 35 s, narration
13 s, Stays Put 105 s, match 0.1 s, sheet and close 12 s. The run before it on the same pair and the code before Oct 31
night (pairC_core1, `docs/evidence/ledger_rows_pairC_core1.jsonl`): $3.2299 in 298 calls, 6 min 49 s, panels 3 min 13
s, and no panel compared. That file also holds one claim photo placed against that run (17 calls, $0.0465).

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

The eval harness: `eval/results.json` holds five labeled pairs on licensed footage and no metrics yet, because no eval
run has been made. It lists the command each pair needs.

## Footage and data

Only footage we have rights to is in this repo or behind any number in it.

- Car videos: uCO3D (Liu et al., Meta, 2025), CC BY 4.0, https://github.com/facebookresearch/uco3d . License:
  https://creativecommons.org/licenses/by/4.0/ . We cropped, re-encoded and blurred plates in these clips. On screen:
  "Footage: uCO3D (Meta), CC BY 4.0, modified". Paper: arXiv:2501.07574. The sample pair is uCO3D pair C
  (7336-29949-54766 as the pickup, 50152-5241-46685 as the return), one Hindustan Ambassador filmed twice. It is a
  dataset car, not a rental.
- Narration on the sample's pickup walk: our own script, spoken by Piper TTS (MIT) with its en_US-ljspeech-high voice,
  trained on the public-domain LJ Speech dataset, and laid onto the uCO3D clip by us. The dataset clips have no audio.
- Stock video, used in the eval labels: WeStarMoney Rec on Pexels, https://www.pexels.com/@westarmoney/ , Pexels
  License, https://www.pexels.com/license/ . Clips 4228872 and 4281322. No endorsement by any brand shown is implied.
- Crash test video, used in the eval labels: NHTSA Vehicle Crash Test Database, test 15759,
  https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results/get-multimedia-files/15759 , public
  information under NHTSA's terms of use. NHTSA does not endorse this project.

The source clips are not in the repo (`data/` is ignored); each label file in `eval/labels/` names its footage.

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
  there. We have not confirmed Baseten's retention for Model APIs, only what we send.
- The sample in `samples/`: every file is the run's own bytes, except `run.json` (footage credit and replay flag); no
  blur was added by hand. We looked at all 105 images, the four clips, every second frame of both display videos where
  a plate track or its 2 s carry runs and every third frame elsewhere, zooming where a plate turned edge-on. Plates sit
  under blocks of about a quarter of the plate or larger. One spot to know about: on the return walk at 12.3 to 12.6 s the front strip plate shows edge-on, about 45
  by 8 px on the 960 px video, more than 2 s before SAM 3.1's plate lane finds it and before the dataset's own grey patch
  covers it; no character is readable there.

## Limitations

- No live run has yet printed "already there", asked a second look inside a run, or stated a size. On the sample pair
  the chips pair sits inside the "already there" distance, but Spark's own "dirt, low confidence" on both marks lowers
  it (above); on a cleaner car or a clearer chip the same rule would say "already there".
- Twin panels: on this car SAM 3.1's hood lane holds the trunk lid when the camera is at the rear and the other way
  round. The twin check put both walks' lids right on the sample run, but it reads cues (a plate on the lid, mirrors
  near the hood) that a car with its plate on the bumper or its mirrors on the doors would not give.
- A wet, dirty or glare flag from Spark caps a mark seen in one walk only (K15); on this weathered car 38 of 70 crops
  carry one, and the ruler refuses a size on any flagged crop.
- Stays Put is not calibrated, so it changes no label. Its neighbour test separated a reflection from paint on a dark
  car, but on this white car's trunk lid too few neighbours stayed tracked, and the trunk chips got no clip (16 clips per
  walk, by mark confidence).
- Muse Voice merges sentences said less than about 3 s apart into one long turn and mishears some words ("like scrat"
  for "light scratch"); the speaker markers it writes into transcripts are left out of the words shown.
- No eval numbers yet, and no real renter's walk: every car here comes from a dataset.
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
