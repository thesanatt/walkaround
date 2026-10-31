# Walkaround web

The browser half of Walkaround:

- **Upload page.** The phone hashes each walk with hash-wasm in a Web Worker while it uploads in 8 MB chunks, then shows the seal card with the SHA-256 prefix.
- **Run page.** One pure reducer (`src/state/reducer.ts`) is fed by either driver: live server-sent events, or a stored run replayed with delays. Each SAM 3.1 lane is parsed by `@meta-sam/parser` in a worker. The outlines are drawn by frame index over the blurred walk, which plays through `@meta-sam/react` and `@meta-sam/video`. Below the walks: how the run was checked (each model and what it did, counted from the run's files), the coverage car with the top-up form, the crops, the match with its second looks and the two dates, a compare-in-place toggle (the match cut), the Stays Put clips, the narration chips and the latest claim placed against the run.
- **Claim page.** A claim photo and the bill's words go to `POST /api/claims`; the answer is a `contracts.Claim`, shown with its exit in plain words and its rule id, the served pixelated copy of the claim photo beside the dated pickup frame, and a one-tap panel chooser after E2 or E2b (one button when only one panel fits; the tap is recorded as the renter's input).
- **Kiosk route.** A 1920 x 1080 stage for screen capture, one scene per video beat.

Stays Put is shown as the run froze `thresholds.staysput.cap_enabled`. While it is false (the lead's call after D1, docs/D1_MEMO.md note 2) the clips and their drift numbers show, but no caption states a verdict ("paint", "reflection"), the control is called the control clip, and the kiosk moment's upgraded form falls back to the boxed pickup frame.

## Commands

Node 22.18 or newer.

| command | what it does |
|---|---|
| `npm install` | installs the pinned packages |
| `npm run dev` | Vite on port 5173 with the mock backend (no FastAPI needed) |
| `WALKAROUND_API=http://127.0.0.1:8000 npm run dev` | the same, with `/api` and `/healthz` proxied to the FastAPI app |
| `npm run build` | type-checks and builds `dist/`, which app.py serves (`/assets` plus the SPA routes) |
| `npm test` | vitest: the reducer and both drivers, replay pacing, the new derived files, the kiosk scenes, lane parsing, hash and upload, the mock's allowlist and routes |
| `npm run mock` | `dist/` plus the mock backend on port 5174, without Vite |
| `npm run mock:fixture` | rebuilds `mock/fixture/` from the repo's fixtures through `walkaround/contracts.py` (deterministic) |

## Routes the browser consumes

All of them are in `src/api/client.ts`:

- `GET /api/status`, `GET /api/sample`, `GET /healthz`
- `POST /api/runs`, `GET /api/runs/{id}`
- `PUT /api/uploads/{upload_id}/chunks/{n}?run_id=`, `POST /api/uploads/{upload_id}/finalize`
- `GET /api/runs/{id}/events` (SSE): the driver closes the stream after `run.completed` or `run.failed`. A stream that closes before any event (a run that failed before the seal, whose events never reach the chain) is followed by `GET /api/runs/{id}`, and the page shows run.json's status and errors.
- `GET /api/runs/{id}/replay?mode=fixed|recorded|instant`
- `GET /api/runs/{id}/files/{path}`, with `?view=kiosk` on the kiosk. The files read: `videos/{role}.json`, `derived/*.json` (with `derived/looks.json` and `derived/claim.json`), `open_weights.ndjson` at the run root, `deletions.json`, `sam/{role}/{lane}.ndjson.gz`, the blurred proxy, crops and masks, clips, chips, `claim/claim_<n>.json` and the claim photo's pixelated copy `claim/<claim_id>.blurred.jpg`. The unblurred claim upload and `claim/<claim_id>.answers.json` are never requested. `open_weights.ndjson` and `claim/claim_<n>.json` are in the binding allowlist of docs/CONTRACTS.md ("Phase 2 file shapes"); until privacy.py carries them the real server answers 404 there and the page shows less (no open-weights link, only the claim in `derived/claim.json`).
- `GET /api/runs/{id}/sheet`, `sheet.pdf`, `sheet.json` and `bundle.zip`; the kiosk adds `?view=kiosk` to the sheet and never shows a keep_plate run's sheet
- `POST /api/runs/{id}/topup` (multipart `video` plus `sha256_browser`, hashed on the device); a 429 prints the server's `reason`
- `POST /api/claims` (multipart `image`, `text`, `run_id`, optional `tire_code`, optional `panel_id` for the renter's tap)

## When the derived files are read

Each event that names a file is followed by a `<kind>.loaded` event with that file, so live and replay feed the reducer the same sequence: `match.derived` reads matches, verdicts, `derived/looks.json` and `open_weights.ndjson` (the second looks run inside the match step); `claim.derived`, `run.completed` and `run.failed` read the run's claims (`claims.loaded`). claim.py writes `claim/claim_<n>.json` for every claim and mirrors the latest into `derived/claim.json` only while the run is open, so a claim placed after a run closed exists only as its numbered file: `loadClaims` (`src/state/loader.ts`) reads `claim/claim_1.json` up until two numbers in a row are missing, then `derived/claim.json`, one claim per id, oldest first. `src/lib/looks.ts` reads the look rows (the binding `fallback_from`, `valid` and `request_id`; a row with verdict null did not run and is skipped) and the open-weights lines, and skips any row it cannot place.

## Kiosk

`/kiosk/{run_id}?scene=<name>&autostart=1`

Scenes, keyed to docs/VIDEO_PLAN.md beats: seal and masks (B5), fragments (B6), coverage (B7), crops (B8), moment (B9), tape (B10), hood (B11), claim (B12), sheet (B13), topup (B14), blur (B15), checks and guardrails (B16). Space or the right arrow moves to the next scene.

Each scene in `src/kiosk/scenes.ts` carries its delay table: `steps` (what appears when, in ms from the scene start; the scene clock moves only at these) and, for the scenes that replay the run's own events, `delays` (stage budgets laid over `kiosk/delays.json`). The moment runs dim 0, trail 3000, cut 7000, return 7600 (the 8-frame dissolve), join 8200, card 10000, in a 12 s hold. A scene that shows pixels from the walks prints the run's `footage_credit` bottom left.

| parameter | effect |
|---|---|
| `t=<ms>` | freezes the scene that many ms after its start, for a frame-by-frame capture (the Stays Put clips pause on the matching frame) |
| `form=upgraded` | the moment's trail step shows the mark's Stays Put clip beside the reflection control, only on a run with `staysput.cap_enabled` true and a fixed track; otherwise, and without it, the boxed pickup frame |
| `panel=<panel_id>` | overrides the scene's panel |
| `theme=dark` | the dark theme; every take is light otherwise, whatever the capture machine prefers |
| `fit=1` | scales the stage down to the window for a preview |

The dates print in the browser's time zone, named ("EDT"), so start the capture browser with the zone pinned (for example `TZ=America/New_York`) to make takes agree. A walk whose camera wrote no `creation_time` gets no wall clock anywhere (the receipt time never stands in): the uCO3D pair C walks and the narrated pickup carry none, so a B9 card recorded on them says "camera time not present" on both sides.

The replay badge is on every scene. Against the mock, a "mock data" tag is on the stage too.

## The mock

`mock/fixture/mock-sample-pair/` is one pair in the exact shapes of `walkaround/contracts.py`. Every record is built through the pydantic models, and the event log is hash-chained and verified.

- **Real:** the SAM lanes are the saved 10/26 streams in `tests/fixtures/sam/`.
- **Hand-made dev data:** the walk served as the blurred proxy (ffmpeg's test pattern at the lanes' size, rate and frame count, so no face, plate or unlicensed footage is served or committed), the panels and tracks, the crops (drawn, labelled "not a photo"), the marks (one boxed because the narration named it, so its confidence is low and K12 holds it at needs your eye), the verdicts, the second looks in `derived/looks.json` and the lines of `open_weights.ndjson` (both in the "Phase 2 file shapes" form), three narration turns (one linked to a box-pass mark, one seeding a mark, one with no mark), five claims (placed with a size, E5, E4, E2, both sides shown) with `derived/claim.json` holding the latest, coverage after a top-up clip, and the sheet. run.json carries today's `config/thresholds.json`, so the mock shows Stays Put uncalibrated as a run frozen now would.

The mock serves a run's files through `mock/allowlist.ts`, app.servable's floor plus privacy.served_file_allowed with the binding allowlist, so `npm run dev` answers 404 where app.py does. It simulates a live run after an upload, using the same pacing as a fixed-delay replay. Its claim route answers by the words typed: "bumper" gives E4, "close" or "tight" gives E5, "dent" shows both sides, other words with no side give E2, anything else places it. A tap (`panel_id`) is answered the way claim.decide answers one: the picked panel stands and "Panel picked by you" leads the paragraph; the mock compares marks only on the right rear door, a panel never filmed at pickup comes back E3, a glimpsed one E4.

Kiosk captures made against the mock are dev material: the walk is a test pattern and the records are hand-made, so keep them out of anything published.
