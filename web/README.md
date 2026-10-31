# Walkaround web

The browser half of Walkaround:

- **Upload page.** The phone hashes each walk with hash-wasm in a Web Worker while it uploads in 8 MB chunks, then shows the seal card with the SHA-256 prefix.
- **Run page.** One pure reducer (`src/state/reducer.ts`) is fed by either driver: live server-sent events, or a stored run replayed with delays. Each SAM 3.1 lane is parsed by `@meta-sam/parser` in a worker. The outlines are drawn by frame index over the blurred walk, which plays through `@meta-sam/react` and `@meta-sam/video`.
- **Kiosk route.** A 1920 x 1080 stage for screen capture.

## Commands

Node 22.18 or newer.

| command | what it does |
|---|---|
| `npm install` | installs the pinned packages |
| `npm run dev` | Vite on port 5173 with the mock backend (no FastAPI needed) |
| `WALKAROUND_API=http://127.0.0.1:8000 npm run dev` | the same, with `/api` and `/healthz` proxied to the FastAPI app |
| `npm run build` | type-checks and builds `dist/`, which app.py serves (`/assets` plus the SPA routes) |
| `npm test` | vitest: the reducer, both drivers, replay pacing, lane parsing, hash and upload |
| `npm run mock` | `dist/` plus the mock backend on port 5174, without Vite |
| `npm run mock:fixture` | rebuilds `mock/fixture/` from the repo's fixtures through `walkaround/contracts.py` |

## Routes the browser consumes

All of them are in `src/api/client.ts`:

- `GET /api/status`, `GET /api/sample`, `GET /healthz`
- `POST /api/runs`, `GET /api/runs/{id}`
- `PUT /api/uploads/{upload_id}/chunks/{n}?run_id=`, `POST /api/uploads/{upload_id}/finalize`
- `GET /api/runs/{id}/events` (SSE): the driver closes the stream after `run.completed` or `run.failed`. A stream that closes before any event (a run that failed before the seal, whose events never reach the chain) is followed by `GET /api/runs/{id}`, and the page shows run.json's status and errors.
- `GET /api/runs/{id}/replay?mode=fixed|recorded|instant`
- `GET /api/runs/{id}/files/{path}`, with `?view=kiosk` on the kiosk. The files read are `videos/{role}.json`, `derived/*.json`, `deletions.json`, `sam/{role}/{lane}.ndjson.gz`, the blurred proxy, crops and masks, clips, chips and `claim/claim_1.jpg`.
- `GET /api/runs/{id}/sheet`, `sheet.pdf`, `sheet.json` and `bundle.zip`; the kiosk adds `?view=kiosk` to the sheet and never shows a keep_plate run's sheet
- `POST /api/runs/{id}/topup` (multipart `video` plus `sha256_browser`, hashed on the device) and `POST /api/claims` (multipart)

## Kiosk

`/kiosk/{run_id}?scene=<name>&autostart=1`

The scenes, keyed to docs/VIDEO_PLAN.md beats: seal, masks (B5); fragments (B6); coverage (B7); crops (B8); moment (B9); tape (B10); hood (B11); claim (B12); sheet (B13); topup (B14); blur (B15); guardrails (B16). Space or the right arrow moves to the next scene.

| parameter | effect |
|---|---|
| `t=<ms>` | freezes the scene that many ms after its start, for a frame-by-frame capture |
| `form=upgraded` | adds the Stays Put clips to the moment |
| `panel=<panel_id>` | overrides the scene's panel |
| `fit=1` | scales the stage down to the window for a preview |

The replay badge is on every scene. Against the mock, a "mock data" tag is on the stage too.

## The mock

`mock/fixture/mock-sample-pair/` is one pair in the exact shapes of `walkaround/contracts.py`. Every record is built through the pydantic models, and the event log is hash-chained and verified.

- **Real:** the walk is the cookbook ping pong clip in `data/test/`, and the SAM lanes are the saved 10/26 streams in `tests/fixtures/sam/`.
- **Hand-made dev data:** the panels, the crops (drawn, labelled "not a photo"), the marks, verdicts and narration, and the sheet.

The mock simulates a live run after an upload, using the same pacing as a fixed-delay replay.
