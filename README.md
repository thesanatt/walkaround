# Walkaround

Walkaround turns the pickup and return videos you take of a rental car into a dated, sealed, panel-by-panel evidence
sheet. SAM 3.1 tracks every panel, Muse Spark 1.3 finds marks on blurred crops, plain code matches the marks across the
two walks and writes every label, and the sheet is hashed and timestamped.

Built Oct 26 to Nov 4, 2026 for the Meta Global AI Developer Hackathon (Indie Developer track, one person). It is
evidence support, not legal advice.

## Quick start: the sample pair, no key

You need Python 3.13 and [uv](https://docs.astral.sh/uv/) or pip. No API key, no Node, no ffmpeg.

```
git clone <this repo's URL> walkaround
cd walkaround
uv venv --python 3.13
uv pip install -e ".[dev]"
.venv/bin/walkaround serve
```

With pip instead of uv: `python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]"`, then the same last line.

Open http://127.0.0.1:8000 and press "open the sample pair (precomputed)". On start the app copied
`samples/pairC_core1/` into `runs/`; the page replays that run's stored events with fixed delays under a "replayed from
a run on Oct 31, 2026" badge, and no model is called. The "sheet" link at the top of the run page opens the evidence
sheet, `evidence.pdf`, `bundle.zip` and `sheet.json`.

The same from a terminal:

```
curl -s http://127.0.0.1:8000/api/sample
curl -s "http://127.0.0.1:8000/api/runs/pairC_core1/replay?mode=instant" | head -c 400
curl -s -o evidence.pdf http://127.0.0.1:8000/api/runs/pairC_core1/sheet.pdf
```

The offline tests need ffmpeg on PATH (the suite makes short synthetic clips with it) and take about two minutes:
`.venv/bin/python -m pytest -q`. The tests that call a model run only with `LIVE=1`.

What the sample shows, honestly: on this pair the match called nothing. All 21 panels came out "not comparable",
because Muse Spark flagged dirt, glare or water on 33 of 61 crops of this weathered white car (rule G2), and on the
return walk SAM 3.1's hood and trunk-lid tracks swapped, so the trunk-lid pair compared a trunk with a hood (rule G4).
The page and the sheet say which rule stopped each panel and show both frames, so you can look yourself.

## Live mode, with your own Meta Model API key

```
cp .env.example .env          # put your key in MODEL_API_KEY; uncomment BASETEN_API_KEY for the second look
LIVE=1 .venv/bin/walkaround run PICKUP.mp4 RETURN.mp4
```

`walkaround run` needs ffmpeg and ffprobe on PATH. It seals both files, runs every stage and prints each event as one
JSON line, then `run.json`. Without `LIVE=1` it prints what it would do and spends nothing. One video (`walkaround run
PICKUP.mp4`) is a pickup-only check. Then `.venv/bin/walkaround serve` shows the run at `/run/<run_id>`, and
`.venv/bin/walkaround verify <run_id>` rebuilds every derived file and the sheet from the stored model answers, with no
network call, and prints "rebuild identical". The page also takes uploads from a phone; it hashes each file in the
browser while it uploads.

The sample pair cost $3.2299 in 298 model calls and took 6 min 49 s end to end (its ledger rows in
`docs/evidence/ledger_rows.jsonl`, its event log in `samples/pairC_core1/events.ndjson.gz`).

Optional: `uv pip install -e ".[seal,pdf]"` and `.venv/bin/playwright install chromium` add the OpenTimestamps stamp and
the evidence PDF. Without them the seal keeps its RFC 3161 token and the run page offers a PDF retry. Without
`BASETEN_API_KEY` the second look falls back to hosted Muse Spark 1.3, and `derived/looks.json` says so.

## What happens in a run

1. Seal: the browser hashes each video while it uploads, and the server hashes it again on arrival and gets an RFC 3161
   timestamp (and an OpenTimestamps stamp) before any model sees it.
2. SAM 3.1 runs 16 text-prompted lanes per walk on a 960 px, 15 fps proxy. Code votes the tracks into 21 panels with
   side and position and lists each panel as filmed, glimpsed, not filmed or not seen.
3. Plates and faces are pixelated (SAM 3.1 tracks plus an image check on every keyframe) before any crop goes to Spark.
4. Muse Spark 1.3 marks each blurred panel crop. Muse Voice turns the renter's narration into turns, and code pins each
   turn that names a panel to the seconds SAM shows that panel.
5. Stays Put: SAM 3.1 follows each mark through its own short clip. It is not calibrated yet, so it records the tracks
   and changes no label.
6. Code matches marks across the walks in a moment-whitened panel frame and writes each label: already there, new or
   unseen, needs your eye, not seen at return, or not comparable. Gates G1 to G9 make a panel not comparable and caps K1
   to K14 lower a label; the ids show on the page and the sheet. The second look can only lower a label.
7. The sheet (HTML, PDF, JSON) shows both frames, the hashes, the coverage list, the deletions and the privacy line. A
   closing manifest seals it.

## The Meta models

| Model | Where it runs | Step | Why this model |
|---|---|---|---|
| SAM 3.1 (`sam-3.1`) | Meta Model API, Responses API, streamed | 16 lanes per walk (car, license plate, face, car door, bumper, wheel, hood, trunk lid, fender, roof, headlight, tail light, side mirror, windshield, rear window, car window); plate and face checks on kept frames; Stays Put clips | One noun phrase returns every instance with a stable id and a mask on each frame of a shaky phone walk, so panels, coverage and the blur come from tracks |
| Muse Spark 1.3 (`muse-spark-1.3`) | Meta Model API, Responses API with a JSON schema | Marks on one blurred crop at a time (boxes in the 0 to 1000 grid, type, size, crop condition); the panel in a claim photo; a spoken note turned into a box | Grounded, typed boxes from a single image with no training; it never sees both walks and never writes a label |
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Meta Model API, `/v1/asr/transcribe`, ENDPOINTING mode | The renter's narration | Turn start and end times let code pin "trunk lid two small chips" to the seconds SAM shows the trunk lid |
| Muse Spark Open (`meta-models/muse-spark-open`) | Baseten Model APIs, OpenAI-compatible Chat Completions | The second look: two outlined crops, same, different or unsure, for each pair code already called "already there" | The one model step that compares the two walks runs on open weights, every call is logged with the model id that answered, and an answer can only lower a label to "needs your eye" |

## Open-weights provenance

Every Muse Spark Open request appends one line to `runs/<run_id>/open_weights.ndjson`: the model id Baseten returned,
both request ids, the latency, the pair of marks, the verdict, the reason and whether the answer passed the schema. No
image bytes and no keys. The app serves it at `/api/runs/<run_id>/files/open_weights.ndjson` and links it from the run
page.

The sample run asked no second look, since no panel was comparable and so there was no "already there" pair to check.
These two lines come from a smoke run on the same car's trunk lid, the true pair and one control
(`docs/evidence/second_look/open_weights.ndjson`, three lines, all answered by `meta-models/muse-spark-open`):

```
{"baseten_request_id": "54ca701a193e1872479fc91d261cb85931e62", "engine": "baseten", "error": null, "http_status": 200, "latency_ms": 2649, "model": "meta-models/muse-spark-open", "model_requested": "meta-models/muse-spark-open", "panel_id": "trunk_lid", "pickup_mark": "p_paw", "purpose": "second_look", "reason": "Both boxed spots on the trunk lid show a small dark discoloration of similar size and shape at the same relative location with no conflicting appearance.", "request_id": "chatcmpl-6b4c9f62ad5041a89d27b9fe742b2946", "return_mark": "r_paw", "run_id": "b11_fix_smoke_1031", "status": "completed", "ts": "2026-10-31T21:33:15Z", "valid": true, "verdict": "same"}
{"baseten_request_id": "54ca701a193e1adc87a35b54c36fe367002e0", "engine": "baseten", "error": null, "http_status": 200, "latency_ms": 2375, "model": "meta-models/muse-spark-open", "model_requested": "meta-models/muse-spark-open", "panel_id": "trunk_lid", "pickup_mark": "p_paw", "purpose": "second_look", "reason": "First boxed spot is a small dark chip/speck on the paint while second boxed spot is a larger leaf-shaped debris, clearly not the same persistent mark.", "request_id": "chatcmpl-4f77182a7ff84a21bc7518bec14b7214", "return_mark": "r_leaf", "run_id": "b11_fix_smoke_1031", "status": "completed", "ts": "2026-10-31T21:33:25Z", "valid": true, "verdict": "different"}
```

To ask the same three questions again with your own Baseten key (about $0.004 each):

```
LIVE=1 .venv/bin/python scripts/second_look_again.py
```

It reads `docs/evidence/second_look/looks.json`, finds both outlined crops of each look by SHA-256, asks Spark Open and
prints the new answer beside the stored one. A model answer can differ between calls, which is one reason it can only
lower a label. Inside a run, `walkaround verify` replays the stored looks from `derived/looks.json` with no call.

## Measured

Every number here is copied from a file in this repo.

The sample pair, run pairC_core1 on Oct 31: `docs/evidence/ledger_rows.jsonl` (its rows of `runs/ledger.jsonl`).

| What | Calls | Dollars | Time |
|---|---|---|---|
| SAM 3.1 lanes, 16 per walk (296 and 304 frames) | 32 | 1.9200 | p50 12.7 s a lane |
| SAM 3.1 Stays Put clips, 89 frames each | 32 | 0.5696 | p50 19.1 s |
| SAM 3.1 image checks for plates and faces | 170 | 0.4250 | p50 1.3 s |
| Muse Spark 1.3 marks, one blurred crop a call | 61 | 0.3065 | p50 4.4 s |
| Muse Spark 1.3 voice seeds | 2 | 0.0078 | |
| Muse Voice Transcribe, 19.8 s of audio | 1 | 0.00095 | 4.2 s |
| The whole run | 298 | 3.2299 | |
| One claim photo placed against the run | 17 | 0.0465 | |
| Muse Spark Open second looks (`docs/evidence/second_look`) | 3 | 0.0111 | 2.4 to 4.4 s |

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
- Unblurred originals and proxies are deleted when a run ends, each with its SHA-256 on the sheet. Model API uploads
  are set to expire after an hour and are deleted at run end, and every request is sent with storage off.
- Processed with Meta's Model API in Standard mode, which Meta's documentation says is not used to train Meta models.
  Meta's terms allow it to retain content for abuse review and legal reasons; Walkaround does not control that.
- The second look sends two blurred crops to Muse Spark Open on Baseten, a company separate from Meta; no video goes
  there. We have not confirmed Baseten's retention for Model APIs, only what we send.
- The sample in `samples/`: we looked at every image and at the video frames around each plate. The run's 5 px blocks on
  the 960 px display videos left a close plate legible on a few frames, so the sample's two display videos are
  pixelated again with 16 px blocks on every plate box, plus five boxes drawn by hand at 12.3 to 13.3 s of the pickup
  walk, where the rear plate showed edge-on before SAM 3.1's plate lane found it (`scripts/make_sample.py`). Those two
  files no longer match the SHA-256 in `videos/*.json` and the sealed manifest. Every other sample file is the run's own
  bytes, except `run.json` (footage credit and replay flag).

## Limitations

- No live run has yet shown "already there" end to end, a second look inside a run, or a printed size. On the sample
  pair every panel was not comparable (above).
- Twin panels can swap: on the return walk SAM 3.1's hood lane held the trunk lid and the other way round, and nothing
  yet checks the two lids against the headlights or tail lights.
- Gate G2 blocks a panel when Spark flags dirt or glare on either crop, so a weathered car compares little.
- Stays Put is not calibrated (on this car it could not tell paint from glare), so it changes no label.
- Muse Voice merges sentences said less than about 3 s apart into one long turn and mishears some words ("like scrat"
  for "light scratch"); transcripts carry speaker markers.
- No eval numbers yet, and no friend's or renter's walk: every car here comes from a dataset.
- Decoding SAM 3.1 masks takes about 0.4 s each on the Mac it was built on, so a 20 s walk already hits the panel
  stage's time budget and keeps 2 keyframes per panel.
- The sample cannot be re-verified from the repo: `walkaround verify` needs the run's stored model answers
  (`spark.json`, `voice.json`), which the app never serves. Verify works on your own runs.
- Built and tested on macOS with Python 3.13; not tried on Windows.

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
