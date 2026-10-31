# Walkaround module contracts

Written by the contracts lane on 2026-10-26 from docs/ARCHITECTURE.md (sections 3 to 9, 11 to 14) and
docs/API_NOTES.md (the "Measured on 10/26" section wins where the two disagree). Lanes implement these signatures
verbatim. Underscore helpers are free. If a lane needs a change, it says so in its report and the integrator edits
this file; nobody edits another lane's module.

Shared ground, read first:

- `walkaround/contracts.py`: every record of the data model as a pydantic v2 model, every enumeration, and
  `canonical_json` / `write_json` / `read_json` / `read_json_list` / `sha256_json` / `chain_hash`. Derived JSON is
  written only through `contracts.write_json` (sorted keys, floats rounded to 6 places, compact, ASCII, trailing
  newline) so a rebuild is byte-identical and `model.digest()` equals the file hash.
- `walkaround/paths.py`: the runs/<run_id>/ layout as functions. A directory function creates the directory, a
  file function creates its parent. Records store paths relative to the run dir (`paths.rel(run_id, path)`).
- `config/thresholds.json`: every threshold with a `source`. Code reads it once through
  `contracts.load_thresholds()` (plain nested values, `_meta` stripped) and passes the dict down as `th`;
  `contracts.config_hash(th)` is the config hash on run.json, every trace and the sheet. Pure rule functions take
  `th` as an argument; nothing reads the file inside a rule.
- `walkaround/rules.py`: the rule id table (G1 to G10 gates, K1 to K13 caps, L1 to L6 labels, E1 to E8 claim exits).
  `rules.rule(id)` raises KeyError on an unknown id, so no sheet prints an id outside the table.
- Ledger: every model call goes through `walkaround.ledger.record(kind, model, *, note, run_id, **usage)` exactly
  once. Kinds: `sam_video` (frames=...), `sam_image` (images=1), `spark` (input_tokens, output_tokens,
  cached_tokens, reasoning_tokens), `voice` (audio_seconds). Extra keys are stored on the row: lanes pass
  `engine`, `request_id`, `unit`, `quantity`, `latency_ms={upload, first_delta, done}`, `parse_cpu_ms`, `status`,
  `retries` so the row matches `contracts.LedgerRow` (`usd` is the row's dollars column).
- Live calls: only inside functions this file marks LIVE, and tests reach them only behind `LIVE=1`. Every
  `/v1/responses` request carries `store=False`.
- Verdicts: deterministic code owns every label; a model answer can only lower one (K4, K7, K12, K13, G2, G10).
- Types used below: `Box4 = list[float]` of x1, y1, x2, y2 with half-open right and bottom; `Point2 = list[float]`;
  `media_size = tuple[int, int]` (width, height after rotation); `th = dict[str, Any]` from `load_thresholds()`;
  `role: str` matches `contracts.Role` (pickup, return, topup_<n>, claim_<n>).
- Module layout: flat modules under `walkaround/` with the lane names used here (video, sam_wire, sam_client,
  spark_client, voice_client, panels, marks, geometry, match, staysput, privacy, seal, eventlog, orchestrator, app,
  sheet). The architecture's package directories (walkaround/media/ and so on) are not used; the flat names match
  the lanes and the existing stubs.

## Fixtures

- `tests/fixtures/sam/pingpong_text.events.ndjson`: the real SAM 3.1 video stream of 10/26, text prompt
  "ping pong ball", metadata include_confidence true and mask_encoding one_bit, on data/test/pingpong.mp4
  (1280x720, 24 fps, 141 frames). 141 events: response.created, response.in_progress, output_item.added,
  content_part.added, 134 output_text.delta (one per frame with a record), output_text.done, content_part.done,
  output_item.done, response.completed. Frames 0 to 136 with 134 frames carrying a record, one object id "0", a box
  and a one_bit mask on every record, mask raster size equal to the box size, c = 0.7299578189849854 on every
  record (constant on a text lane), `response.completed.response.usage.video_frames_processed` = 141, `store`
  echo true (the probe did not send store false). Both finalizers are present and the done event carries its own
  `sequence_number` (138), so the API itself emitted it that night: the reader must accept either finalizer, both,
  or neither.
- `tests/fixtures/sam/pingpong_box.events.ndjson`: the same clip with a box prompt built by `build_object_prompt`
  at frame 0 and metadata propagation_direction both: object id "1" on 134 frames, c varying per frame (0.890625,
  0.8668561577796936, 0.9177275896072388, ...).
- `tests/sam_conformance/cases/*.json` (42 parser cases) and `object-prompts/*.json` (22 builder cases): the wire
  reader must agree with meta_sam_parser on every case; the builder cases drive `sam_client.prompt_box`.
- A zero-match fixture (response.completed with no delta) does not exist yet; the sam lane writes one by hand from
  the pingpong stream (drop the deltas, empty the part text) and E2 replaces it with a real "face" lane.

## video (M1), walkaround/video.py

Keeps the stub's `VideoInfo` dataclass and these signatures (one default changed):

| Function | Behaviour | In -> out | Ledger |
|---|---|---|---|
| `probe(path: Path) -> VideoInfo` | ffprobe; width and height are the displayed size after rotation | file -> VideoInfo | none |
| `make_proxy(src: Path, dst: Path, *, fps: float = 15.0, max_edge: int = 960, preset: str = "medium", crf: int = 20) -> VideoInfo` | H.264 proxy, rotation baked in, no audio, yuv420p, even dims, `-fps_mode cfr`, lanczos, faststart; idempotent when dst is newer than src. CHANGED: default fps 10 -> 15 (section 16 item 2); `preset` and `crf` added for E1 | file -> file | none |
| `read_frame_at(path: Path, t_seconds: float) -> np.ndarray` | one RGB uint8 frame from the ORIGINAL at pts t, exact seek, rotation applied | | none |
| `iter_frames(path: Path, *, fps: float | None = None, max_edge: int | None = None) -> Iterator[tuple[int, float, np.ndarray]]` | (index, t_seconds, rgb) through ffmpeg | | none |
| `sharpness(rgb_or_gray: np.ndarray) -> float` | Laplacian variance | | none |
| `sha256_file(path: Path) -> str` | streamed hex digest | | none |
| `extract_wav(src: Path, dst: Path, *, rate: int = 24000) -> Path` | `-vn -ac 1 -ar rate -c:a pcm_s16le -map_metadata -1`; raises if no audio stream | | none |

Added:

| Function | Behaviour | In -> out |
|---|---|---|
| `ffprobe_info(info: VideoInfo, *, container: str, device: str | None = None, hdr: bool = False, vfr: bool = False) -> contracts.FfprobeInfo` | display_width/height after rotation; creation_time or "not present" | VideoInfo -> FfprobeInfo |
| `pts_map(proxy: Path) -> list[float]` | `ffprobe -show_frames` pts_time per proxy frame, index = frame | file -> list |
| `proxy_record(run_id: str, role: str, proxy: Path) -> contracts.ProxyRecord` | probes the proxy, writes `paths.pts_map_path`, hashes, fills the record with run-relative paths | file -> ProxyRecord |
| `pts_to_proxy_frame(pts: list[float], t: float) -> int` and `proxy_frame_to_pts(pts: list[float], f: int) -> float` | nearest frame / its pts; the one place the 33 ms mismatch of 5.9 is computed | |
| `cut_clip(src: Path, dst: Path, *, pts_center: float, frames: int, windows_src: list[Box4], size: int = 1008, crf: int = 18, blur_boxes: list[list[Box4]] | None = None, block_px: int = 20) -> contracts.ClipRecord` | the panel-locked Stays Put clip: `frames` source frames centred on pts_center, one square window per frame (len(windows_src) == frames), each cropped and resized to size x size, optional per-frame pixelation through `privacy.pixelate`, libx264; asserts the encoded file is size x size via probe | 4K + windows -> ClipRecord |
| `render_blurred_proxy(proxy: Path, dst: Path, boxes_by_frame: dict[int, list[Box4]], *, block_px: int, dilate_frac: float, dilate_px: int) -> VideoInfo` | decode, `privacy.pixelate` per frame, re-encode (the display proxy) | file -> file |
| `write_jpeg(rgb: np.ndarray, dst: Path, *, quality: int = 90) -> str` and `write_png_mask(mask: np.ndarray, dst: Path) -> str` | write, return sha256 | |
| `cut_crop(rgb: np.ndarray, crop: contracts.Crop) -> np.ndarray` | cut the source box `[dx, dy, dx + w/scale, dy + h/scale]` and resize to (w, h) | frame + Crop -> crop pixels |

Must not: call any API; write an unblurred frame outside `paths.quarantine_dir`. Test: proxy frame i maps to source
pts within one frame on a synthetic VFR clip; `cut_clip` round-trips a known box; a portrait source reports 540x960.

## sam_wire (M2), walkaround/sam_wire.py, pure, no network, no decode at parse time

The thin reader of the wire grammar (meta-sam/protocol/sam3.md): one line per frame, `<Nf>` (optional `;fields`
ignored with a warning diagnostic) then comma-separated records `id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..[;c=..]|>
<|mask;x=..;y=..[;c=..];data=H,W,<marker><payload>|>`. Split token bodies on `;`, never search for `key=`; the
payload runs from the marker (`~` lossless, `!` one_bit) to the closing `|>`; a missing `c` is None, never 0.

| Function | Behaviour | In -> out |
|---|---|---|
| `read_events(path: Path) -> Iterator[dict[str, Any]]` | one JSON event per line from `.ndjson` or `.ndjson.gz` | file -> events |
| `lane_text(events: Iterable[dict[str, Any]]) -> str` | the deltas concatenated in order; when there were no deltas, the finalizer text (`response.output_text.done.text` or `content_part.done.part.text`); "" for a zero-match lane | events -> text |
| `parse_lane_text(text: str, *, media: Literal["image", "video"]) -> tuple[list[contracts.IndexFrame], list[contracts.Diagnostic]]` | the thin reader; images normalise `<0f>` to frame 0; `MaskRef.payload_offset` is the character offset of the marker in `text`, `payload_len` to the `|>`; malformed records become diagnostics, never exceptions | text -> index |
| `@dataclass LaneOutcome(status: LaneStatus, response_id: str | None, store_echo: bool | None, video_frames_processed: int | None, usage: dict[str, Any], incomplete_reason: str | None, error: str | None, diagnostics: list[Diagnostic], text_chars: int)` | | |
| `index_from_events(events: Iterable[dict[str, Any]], *, media: Literal["image", "video"]) -> tuple[list[contracts.IndexFrame], LaneOutcome]` | status rules: `response.completed` with no delta and an empty or absent output_text part -> ZERO_MATCHES with an empty index; `response.completed` with records -> COMPLETED; no `response.completed`, or `response.incomplete` / `response.failed` / `error`, or any diagnostic of severity error -> INCOMPLETE. usage and store echo read from `response.completed.response` through plain dict access (the typed SDK usage lacks `video_frames_processed`) | events -> index + outcome |
| `mask_payload(text: str, ref: contracts.MaskRef) -> str` | `text[ref.payload_offset : ref.payload_offset + ref.payload_len]` | |
| `decode_mask(ref: contracts.MaskRef, payload: str) -> np.ndarray` | bool array (ref.h, ref.w) via `meta_sam_parser.decode_mask_to_raster(SegmentationMask(encoding=ref.encoding.value, payload=payload, width=ref.w, height=ref.h))`; raises `InvalidSegmentationMaskError` through | |
| `paste_mask(raster: np.ndarray, box: list[int], ref: contracts.MaskRef, media_size: tuple[int, int]) -> np.ndarray` | bool (H, W) frame with the box-local raster at (box x1 + ref.x, box y1 + ref.y) | |
| `class DecodePool` with `__init__(self, workers: int)`, `decode(self, lane_path: Path, refs: list[contracts.MaskRef]) -> list[np.ndarray]`, `stats(self) -> tuple[int, float]` (masks decoded, CPU seconds), `close(self) -> None`, context manager | ProcessPoolExecutor over `decode_mask`; reads the lane text once per call; the only place a mask is decoded in the request path | lane + refs -> rasters |
| `oracle_index(events: Iterable[dict[str, Any]], *, media: Literal["image", "video"]) -> list[contracts.IndexFrame]` | tests only: the same index through `meta_sam_parser.parse_responses_stream`; `parse_lane_text` must equal it on every conformance case and both fixtures (boxes, ids, c, frame indices, mask dims) and `decode_mask` must equal `decode_mask_to_raster` | |

Ledger: none (the client records the lane). Must not: decode a mask nobody asked for; raise on a bad record.

## sam_client (M2), walkaround/sam_client.py, async, LIVE

Keeps `Box`, `Detection`, `SegmentationRun`, `replay`, `mask_array`, `full_mask` from the stub as the
parser-backed structures used for images, replay and tests. Changed: `upload_file`, `segment_video`,
`segment_video_many`, `segment_image` become `async` (the cookbook pattern and 16 concurrent lanes need it; D1
scripts call them through `asyncio.run`); `segment_video` returns the thin index, not a parsed run; new keyword
arguments below. The docstring line about synthesising output_text.done stays true to the shim: the shim adds the
done event only when the API did not send one.

Streaming pattern, copied from the cookbook's sam_api.py and required verbatim: `stream_events(stream)` converts
SDK events with `model_dump()` and synthesises `response.output_text.done` from `response.content_part.done` once
per lane; the request body is consumed inside `async with stream, aclosing(stream_events(stream)) as events:`.
Each raw event dict is appended to `paths.lane_events_path` as it arrives and handed to `on_event`.

| Function | Behaviour | In -> out | Ledger |
|---|---|---|---|
| `client() -> AsyncOpenAI` | `AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), timeout=240, max_retries=0)`; the client does its own 429 backoff (500 ms base, jitter, Retry-After, 6 tries); tests monkeypatch | | |
| `stream_events(stream: AsyncIterable[Any]) -> AsyncIterator[dict[str, Any]]` | the cookbook shim, verbatim | | |
| `sam_metadata(*, include_confidence: bool = True, mask_encoding: str | None = "one_bit", propagation_direction: str | None = None) -> dict[str, str]` | string values ("true"); propagation_direction only for object prompts | | |
| `async upload_file(path: Path, *, run_id: str | None = None, expires_after_s: int = 3600) -> contracts.FilesApiRecord` | POST /v1/files purpose user_data, `expires_after={"anchor": "created_at", "seconds": expires_after_s}`; cached by sha256 in `paths.uploads_cache_path()`; remux to MP4 faststart and retry once on a 4xx | file -> FilesApiRecord | none |
| `async delete_file(file_id: str) -> dict[str, Any]` | DELETE /v1/files/{id}; the body verbatim (`deleted: true` on 10/26) | | none |
| `async delete_response(response_id: str) -> dict[str, Any]` | DELETE /v1/responses/{id}; used only when store false was refused | | none |
| `async segment_video(file_id: str, prompt_text: str, *, run_id: str, role: str, lane: str, frames_sent: int, media_size: tuple[int, int], metadata: dict[str, str], kind: Literal["text", "object"] = "text", noun: contracts.Noun | None = None, request_index: int = 0, on_event: Callable[[dict[str, Any]], None] | None = None, max_tries: int = 6) -> tuple[contracts.LaneRecord, list[contracts.IndexFrame]]` | one streamed POST /v1/responses, model sam-3.1, `store=False`, input `[{type: message, role: user, content: [{type: input_text, text: prompt_text}, {type: input_video, file_id}]}]`; raw events to `paths.lane_events_path(run_id, role, lane)`, gzipped to `lane_events_gz_path` at the end, index to `lane_index_path`, summary to `lane_summary_path`; status from `sam_wire.index_from_events`; a stream that ends early or a 429 past max_tries gives status FAILED/INCOMPLETE in the record (never an exception) so coverage can print not_run | file_id + prompt -> LaneRecord + index | `sam_video`, frames = `video_frames_processed` when present else frames_sent, note = lane, plus request_id, latency_ms {upload, first_delta, done}, parse_cpu_ms, status, retries |
| `async segment_video_many(file_id: str, prompts: list[tuple[str, str]], *, run_id: str, role: str, frames_sent: int, media_size: tuple[int, int], metadata: dict[str, str], concurrency: int = 16, on_event: Callable[[str, dict[str, Any]], None] | None = None) -> dict[str, tuple[contracts.LaneRecord, list[contracts.IndexFrame]]]` | `prompts` are (lane, prompt_text) in the fixed order of section 5 step 5; a semaphore of `concurrency`; `on_event(lane, event)`; never raises for one lane | | one row per lane |
| `async segment_clip(clip: Path, prompt_text: str, *, run_id: str, role: str, lane: str, media_size: tuple[int, int], metadata: dict[str, str], frames_sent: int, on_event=None) -> tuple[contracts.LaneRecord, list[contracts.IndexFrame]]` | the same request with `{type: input_video, video_url: data:video/mp4;base64,...}` for Stays Put and anchor clips (asserts the file is under 50 MB and its probed size equals media_size) | clip -> LaneRecord + index | `sam_video` |
| `async segment_image(image: Path | np.ndarray, concept: str, *, run_id: str | None, role: str | None = None, tag: str = "", include_confidence: bool = True) -> SegmentationRun` | data URL input_image, `store=False`, parsed with meta_sam_parser (images are small); raw events saved to `paths.image_requests_dir(run_id, role)/<tag>.<response_id>.ndjson` when run_id and role are given | image -> SegmentationRun | `sam_image`, images=1 |
| `replay(events_path: Path, *, media: str, media_size: tuple[int, int], concept: str) -> SegmentationRun` | parser-backed rebuild from saved events, no network | | none |
| `mask_array(det: Detection) -> np.ndarray`, `full_mask(det: Detection, media_size: tuple[int, int]) -> np.ndarray` | as the stub | | |
| `prompt_box(box: Box4, media_size: tuple[int, int], *, pad_px: int = 2) -> meta_sam_parser.PromptBox` | the box rule of step 6: integer edges, padded to at least pad_px per side, clamped to [0, w] x [0, h] with left < right <= w and top < bottom <= h; a collapsed 1 px box pads to 2 px | | |
| `object_prompt(objects: list[meta_sam_parser.PromptObject], media_size: tuple[int, int]) -> str` | `build_object_prompt(size=media_size, objects=objects)`; `ObjectPromptError` propagates with its `.code` so the caller logs it (section 14 item 35) | | |

Tests: the wire reader against every conformance case and both fixtures; a recorded stream replays to the same
index; a zero-match stream yields ZERO_MATCHES with zero records; a 429 fixture backs off 6 times then records
FAILED; the ledger row carries `video_frames_processed`.

## spark_client (M2), walkaround/spark_client.py, sync, LIVE

Keeps `prepare_image`, `grid_to_pixels`, `_client()` monkeypatching and the request shape (Responses API,
`text.format` json_schema strict, explicit `reasoning.effort`, `max_output_tokens`). Changed: `store=False` on
every call; the mark schema becomes marks_v1 of section 5.7 (type enum of 8 incl. dirt, reflection, unsure; point
plus bbox; `size_class` S/M/L; `confidence` as the enum low/medium/high; `note`; `conditions {wet, dirty, glare,
dark}`; `landmarks`; `neighbour_car_visible`), so the stub's `Mark`, `MarkKind`, `SizeClass`, `MARK_KINDS`,
`SIZE_CLASSES`, `COMPARE_SCHEMA` and `compare_marks` are replaced by the contracts enums and the functions below;
status incomplete is handled as truncation, never as "no marks". `MAX_OUTPUT_TOKENS` = 4000, `RETRY_MAX_OUTPUT_TOKENS`
= 8000. Streaming (`stream=True`) is used when `th["spark"]["stream_when_expected_over_s"]` applies (the timeline
and note_prefill purposes).

| Function | Behaviour | In -> out | Ledger |
|---|---|---|---|
| `SCHEMAS: dict[str, dict[str, Any]]` | marks_v1, count_v1, look_v1, seed_v1, claim_name_v1, claim_box_v1, timeline_v1, note_prefill_v1; every object `additionalProperties: false`, all properties required | | |
| `prepare_image(rgb: np.ndarray, max_edge: int = 1280) -> tuple[str, int, int, float]` | JPEG data URL, (url, sent_w, sent_h, scale) | | |
| `grid_to_pixels(x: int, y: int, width: int, height: int) -> tuple[int, int]` and `grid_box_to_pixels(box: contracts.GridBox, width: int, height: int) -> tuple[int, int, int, int]` | `round(v * dim / 1000)` clamped to [0, dim - 1] | | |
| `clamp_grid(value: Any) -> int | None` | 1001 clamps to 1000; farther than `grid_slack` off the grid is None (the mark is dropped) | | |
| `ask(parts: list[dict[str, Any]], *, purpose: contracts.SparkPurpose, schema_name: str, effort: str, max_output_tokens: int, run_id: str | None, input_refs: list[contracts.InputRef], stream: bool = False) -> contracts.SparkRequestRecord` | one call: system prompt fixed and minimal, `store=False`, `reasoning={"effort": effort}`, strict schema; the record stores `raw_text` before parsing, `status`, `incomplete_reason` (`incomplete_details.reason`), `usage` through `model_dump()`, `ms`; `parsed`/`valid` from `json.loads` plus a schema check; no interpretation | parts -> SparkRequestRecord | `spark` with input_tokens, output_tokens, cached_tokens, reasoning_tokens, request_id, status, note=purpose |
| `ask_with_retry(parts, *, purpose, schema_name, effort, run_id, input_refs) -> contracts.SparkRequestRecord` | outcomes of step 8 in order: completed and valid -> done; completed and schema miss -> one retry (retry=1), then valid False; status incomplete with reason max_output_tokens -> one retry at 8000, then `truncated=True` | | two rows at most |
| `@dataclass MarksResult(answer: contracts.SparkMarksAnswer | None, truncated: bool, record: contracts.SparkRequestRecord, sent_width: int, sent_height: int, scale: float, orig_width: int, orig_height: int)` | `answer` None means truncated or invalid after the retry | | |
| `find_marks(crop_rgb: np.ndarray, panel_id: contracts.PanelId, *, landmarks: list[str], effort: str = "low", run_id: str | None = None, input_ref: contracts.InputRef | None = None) -> MarksResult` | the marks_v1 grounding call on one blurred crop; prompt per section 5 step 8 | crop -> MarksResult | via ask |
| `@dataclass CountResult(points: list[contracts.GridPoint], count: int, truncated: bool, record)` and `count_marks(crop_rgb, panel_id, *, effort="low", run_id=None, input_ref=None) -> CountResult` | count_v1 {points[], count}: point at every instance before counting | | via ask |
| `@dataclass LookResult(verdict: contracts.LookVerdict, reason: str, record)` and `second_look(crop_a: np.ndarray, crop_b: np.ndarray, *, panel_id: contracts.PanelId, run_id: str | None = None, input_refs: list[contracts.InputRef] = []) -> LookResult` | look_v1 on two crops with the candidate marks already drawn as outlines by the caller; invalid or truncated -> UNSURE (a downgrade). Replaces the stub's `compare_marks`: the model never upgrades | crops -> LookResult | via ask |
| `@dataclass SeedResult(found: bool, bbox: contracts.GridBox | None, type: contracts.MarkKind | None, record)` and `seed_mark(crop_rgb, panel_id, phrase: str, *, run_id=None, input_ref=None) -> SeedResult` | seed_v1 on the panel's sharpest blurred crop for a narration turn | | via ask |
| `@dataclass ClaimNameResult(panel: contracts.PanelId | None, side: contracts.Side, position: contracts.Position, context: str, front_direction: str | None, landmarks: list[contracts.Landmark], record)` and `claim_name(image_rgb, *, run_id=None, input_ref=None) -> ClaimNameResult` | claim_name_v1 on the full claim image | | via ask |
| `@dataclass ClaimBoxResult(bbox: contracts.GridBox | None, record)` and `claim_box(image_rgb, *, run_id=None, input_ref=None) -> ClaimBoxResult` | the drawn-box call when the grayscale test fires | | via ask |
| `timeline(file_id: str, *, run_id=None) -> contracts.SparkRequestRecord` and `note_prefill(claim_text: str, agreement_pdf: Path, *, run_id=None) -> contracts.SparkRequestRecord` | S2 and S2b probes only (E17, E18), `stream=True` | | via ask |
| `point_to_crop_pixels(p: contracts.GridPoint, result: MarksResult) -> tuple[int, int]` and `box_to_crop_pixels(b: contracts.GridBox, result: MarksResult) -> tuple[int, int, int, int]` | grid -> sent pixels -> original crop pixels (undo the resize); replaces `mark_to_crop_pixels` | | |

Tests: schema validation rejects 1001 before the clamp and accepts after; the retry path; the truncated path
records `truncated=True` and never an empty valid answer; `second_look` returns UNSURE on bad JSON.

## voice_client (M2), walkaround/voice_client.py, sync, LIVE

Keeps `Turn`, `Transcript`, `TranscribeError`, `check_wav`, `MODES`, `TRANSPORT` monkeypatching and
`transcribe_wav(wav_path: Path, *, mode: str = "ENDPOINTING", keywords: list[str] | None = None, run_id: str | None = None) -> Transcript`.
Changed: `Transcript` gains `mode_applied: bool` and `request: dict[str, Any]` (the request JSON part verbatim);
`_parse_turns` no longer invents a whole-clip turn when `turns` is empty (16b: the transcript shows without pins,
no chips, never a fake turn); `DEFAULT_KEYWORDS` becomes the 21 of `th["nouns"]["voice_keywords"]`; the ledger
row carries `request_id=sessionId`. The 10/26 probe saw ENDPOINTING return one 15 s turn for a two-speaker clip,
so E10 decides pins; this module only reports what came back.

| Function | Behaviour | In -> out | Ledger |
|---|---|---|---|
| `check_wav(wav_path: Path) -> float` | seconds; raises ValueError naming every problem and the ffmpeg fix | | |
| `validate_mode(mode: str) -> str` | exact membership in PUSH_TO_TALK, ENDPOINTING, DIARIZATION, else ValueError before any wire call (the server ignores a bad mode) | | |
| `transcribe_wav(wav_path, *, mode="ENDPOINTING", keywords=None, run_id=None) -> Transcript` | one multipart POST; one 429 retry honouring Retry-After; `mode_applied = not (mode != "PUSH_TO_TALK" and not turns and transcript.strip())` | WAV -> Transcript | `voice`, audio_seconds = audioDurationMs / 1000, note=mode, status "mode not applied" when false |
| `voice_record(transcript: Transcript, *, role: str, wav_sha256: str) -> contracts.VoiceRecord` | request and response verbatim, mode_applied, dollars | Transcript -> VoiceRecord | none |

## panels (M5), walkaround/panels.py, pure functions plus one derive entry

Everything per frame runs on boxes and c; masks are decoded only for keyframe candidates, through the
`sam_wire.DecodePool` the caller passes in. Nothing here touches marks or calls Spark.

| Function | Behaviour | In -> out |
|---|---|---|
| `car_boxes_by_frame(car_frames: list[contracts.IndexFrame]) -> dict[int, list[int]]` | the largest "car" box per frame | |
| `border_frac(box: Box4, media_size: tuple[int, int]) -> float` | share of the box perimeter on the frame edge | |
| `filter_objects(frames: list[contracts.IndexFrame], noun: contracts.Noun, car_boxes: dict[int, list[int]], media_size: tuple[int, int], th: dict) -> list[contracts.IndexFrame]` | keep an object when >= `car_box_containment_min` of its box area lies inside the car box and its area clears the per-noun floor (`area_floor_small` for `small_nouns`, else `area_floor_default`); no c floor | |
| `build_fragments(frames: list[contracts.IndexFrame], *, fps: float, th: dict) -> list[contracts.TrackFragment]` | per object id over consecutive frames; split at gap > max(gap_split_s, gap_split_frames / fps); keep >= max(fragment_min_frames, round(fps x fragment_min_s)) frames | |
| `merge_fragments(fragments: list[contracts.TrackFragment], frames: list[contracts.IndexFrame], *, fps: float, th: dict) -> list[tuple[list[contracts.TrackFragment], list[contracts.MergeRecord]]]` | greedy in time order per 5.1 item 4 (gap <= 1.0 s, predicted centroid within 0.5 box widths, area ratio in [0.6, 1.6], aspect within 20%); one group per merged track with its MergeRecords | |
| `build_tracks(role: str, noun: contracts.Noun, groups, frames, pts: list[float], car_boxes, media_size, *, th: dict) -> list[contracts.PanelTrack]` | per_frame rows (frame, pts, box, box_area_frac, in_car_box, c, border_frac), c_present_share; panel_id None until the vote | IndexFrames -> PanelTracks |
| `needs_anchor_pass(tracks: list[contracts.PanelTrack], frames: list[contracts.IndexFrame], th: dict) -> bool` | longest track under `anchor_pass_coverage_below` of the frames where any fragment of the noun is visible | |
| `anchor_prompts(track: contracts.PanelTrack, kf_frames: list[int], window_masks: dict[int, np.ndarray] | None, media_size, th: dict) -> list[meta_sam_parser.PromptObject]` | one PromptObject per panel with a FramePrompt per chosen frame (at most 3): the box through `sam_client.prompt_box` plus negative points at the glass centroid and 0.1 box widths past the seam (negative points are backlog B5; return none when `window_masks` is None) | |
| `absorb_anchor(tracks: list[contracts.PanelTrack], anchor_frames: list[contracts.IndexFrame], *, frame_offset: int, th: dict) -> list[contracts.PanelTrack]` | box IoU >= `anchor_absorb_iou_min` on shared frames; trimmed frame 0 mapped back by frame_offset; records MergeRecord rule anchor_absorb and AnchorReprompt | |
| `walk_direction(tracks: list[contracts.PanelTrack], th: dict) -> int | None` | sign of the median per-frame x displacement of panel box centroids; None under `direction_min_px_per_frame` | |
| `vote_sides(tracks: list[contracts.PanelTrack], direction: int | None, th: dict) -> tuple[list[contracts.Panel], list[contracts.PanelTrack]]` | R1 to R4 with `side_vote_weights`, margin >= `side_vote_margin_min` else side_uncertain; bumper front/rear by shared frames with headlight/tail light; front_sign; every canonical PanelId appears (instance_from canonical when no track); tracks come back with panel_id, side, position filled | tracks -> Panels + tracks |
| `keyframe_candidates(track: contracts.PanelTrack, th: dict) -> list[int]` | the box pass: top `keyframe_box_pass_candidates` frames by box area x (1 - border_frac) x aspect over the track max; border_frac > `border_frac_keyframe_max` excluded | |
| `score_keyframe(track: contracts.PanelTrack, f_proxy: int, *, mask: np.ndarray, car_mask: np.ndarray | None, window_mask: np.ndarray | None, frame_rgb: np.ndarray, exclude_masks: list[np.ndarray], th: dict) -> tuple[contracts.KeyframeScore, np.ndarray, bool]` | the mask pass: completeness, sharpness inside the mask (plate and face excluded), glare_frac (V > 250, S < 40), frontality; car-mask containment check; returns (score, the panel mask after window subtraction, window_subtracted). Score factors are normalised to the candidates' max by the caller | |
| `select_keyframes(scored: list[tuple[int, contracts.KeyframeScore]], *, fps: float, th: dict) -> list[int]` | top `keyframe_top_k` at least `keyframe_min_gap_s` apart | |
| `coverage_for_role(panels: list[contracts.Panel], tracks: list[contracts.PanelTrack], keyframes: list[contracts.Keyframe], *, role: str, fps: float, lane_status: dict[contracts.Noun, contracts.LaneStatus], th: dict) -> dict[contracts.PanelId, contracts.PanelCoverage]` | the states of 5.4: filmed, glimpsed (median c over present frames at or above the run's 25th percentile for that noun; area_only basis when c_present_share < `c_present_floor`), not_filmed (lane completed or zero_matches, no track), not_run (lane failed/incomplete), not_seen (canonical slot, no track in either walk) | |
| `plan_topup(coverage: contracts.Coverage, th: dict) -> contracts.Coverage` | `missing` wheels first, then lower bumpers, then the rest; `topup_text`; `nouns_to_rerun` = missing nouns plus car, license plate, face; odometer asked once | |
| `async derive(run_id: str, role: str, *, lanes: dict[str, tuple[contracts.LaneRecord, list[contracts.IndexFrame]]], pts: list[float], media_size: tuple[int, int], fps: float, decode: sam_wire.DecodePool, blur_keyframe: Callable[..., Awaitable[tuple[np.ndarray, contracts.BlurRecord]]], th: dict) -> tuple[list[contracts.PanelTrack], list[contracts.Panel], list[contracts.Keyframe], dict[contracts.PanelId, contracts.PanelCoverage]]` | the glue: filter, fragments, merge, votes, keyframes (decoding only the candidates), crops written through `privacy.blur_keyframe` then `video.write_jpeg`, moments via `geometry.moments_from_mask`; writes tracks.json, panels.json, keyframes.json, coverage.json through `contracts.write_json` | lanes -> derived files |

Ledger: none directly (anchor requests go through sam_client). Tests: synthetic tracks merge or not by each rule;
a side-vote tie prints side_uncertain; a not_run noun never prints not_filmed; a zero-match lane prints
not_filmed; a track with c absent on every frame falls to the area_only basis.

## marks (M6), walkaround/marks.py

| Function | Behaviour | In -> out | Ledger |
|---|---|---|---|
| `landmarks_for(panel_id: contracts.PanelId, th: dict) -> list[str]` | the closed list for the panel's family (door, bumper, wheel, hood), at most `landmarks_max` | | |
| `draw_outline(crop_rgb: np.ndarray, bbox_crop: Box4, *, width_px: int = 2) -> np.ndarray` | thin outline for the second look | | |
| `map_spark_mark(sm: contracts.SparkMark, result: spark_client.MarksResult, crop: contracts.Crop) -> tuple[Box4, Point2, Box4, Point2]` | (bbox_crop, point_crop, bbox_src, point_src): grid -> sent px (`spark_client.grid_to_pixels`) -> crop px (`geometry.sent_to_crop`) -> source px (`geometry.crop_to_src`) | | |
| `marks_from_answer(answer: contracts.SparkMarksAnswer, result, kf: contracts.Keyframe, crop: contracts.Crop, *, th: dict) -> list[contracts.Mark]` | source spark_box; `size_px_src`; caps set here: K3 (under `small_mark_px`), K4 (type in CAPPED_MARK_KINDS), K12 (confidence low); ids `<kf_id>_m<n>` | | |
| `count_union(box_marks: list[contracts.Mark], count: spark_client.CountResult, result, kf, crop, whitened: dict[str, contracts.WhitenedMark], *, th: dict) -> list[contracts.Mark]` | a count point farther than `count_mode_new_point_min_units` from every box mark becomes a spark_count mark with K8 | | |
| `dedup_within_walk(marks: list[contracts.Mark], whitened: dict[str, contracts.WhitenedMark], keyframes: list[contracts.Keyframe], *, th: dict) -> tuple[list[contracts.Mark], float | None]` | 5.6: cluster within `cluster_radius_units` across the panel's keyframes, keep the medoid, `merged_from`, sigma_view returned; single_frame (K2) when 1 of 3 while the other two were complete and sharp; collinear merge; groups (cluster_id) and K9 for droplet-like or road-spray-like panels | | |
| `async derive_marks(run_id: str, role: str, keyframes: list[contracts.Keyframe], *, th: dict, count_mode: bool = False) -> tuple[list[contracts.Mark], list[contracts.Keyframe], list[contracts.SparkRequestRecord]]` | per keyframe: load the blurred crop, `spark_client.find_marks` (and `count_marks` when enabled), set `kf.conditions` and `kf.marks_status` (complete, truncated), map and cap marks, whiten through `geometry.whiten` with the keyframe's own moments, dedup; writes marks.json and appends to spark.json | keyframes -> marks.json, spark.json | via spark_client |

Must not: decide a verdict; see both walks except through `spark_client.second_look`, which `match.derive_match`
calls; call Spark without an effort and max_output_tokens.

## geometry (M7), walkaround/geometry.py, pure numpy

| Function | Behaviour |
|---|---|
| `moments_from_mask(mask: np.ndarray) -> contracts.Moments` | centroid, central moments, theta = 0.5 atan2(2 mu11, mu20 - mu02), Sigma^(-1/2) as a 2x2 list |
| `whiten(point: tuple[float, float], moments: contracts.Moments, front_sign: int) -> tuple[float, float]` | w = Sigma^(-1/2) (p - c) rotated so axis 1 runs along the principal axis toward the car's front and axis 2 image-down; raises ValueError when front_sign is not 1 or -1 (the panel is then not comparable, G6) |
| `whiten_many(points: np.ndarray, moments, front_sign: int) -> np.ndarray` | (N, 2) in, (N, 2) out |
| `radius(w: tuple[float, float]) -> float` | norm |
| `carry_moments(moments: contracts.Moments, box_from: Box4, box_to: Box4) -> contracts.Moments` | the box-to-box affine (translation and per-axis scale) applied to the keyframe moments; used per clip frame so a clip costs 3 decodes |
| `crop_geometry(box_src: Box4, media_size: tuple[int, int], *, pad_frac: float, long_edge: int) -> contracts.Crop` | padded source box, long edge `long_edge`, `path=""` for the caller to fill; w, h, scale, dx, dy |
| `crop_to_src(p: Point2, crop: contracts.Crop) -> Point2`, `src_to_crop(p: Point2, crop) -> Point2`, `box_crop_to_src(b: Box4, crop) -> Box4`, `box_src_to_crop(b, crop) -> Box4` | the Crop mapping |
| `proxy_to_src(box: Box4, proxy_size: tuple[int, int], src_size: tuple[int, int]) -> Box4` and `src_to_proxy(...)` | the proxy ratio |
| `sent_to_crop(p_sent: tuple[int, int], sent_size: tuple[int, int], crop_size: tuple[int, int]) -> Point2` | undo `prepare_image`'s resize |
| `box_iou(a: Box4, b: Box4) -> float` and `containment(inner: Box4, outer: Box4) -> float` | the per-frame box rules of 5.1 |
| `whitened_mask_iou(mask_a: np.ndarray, mom_a: contracts.Moments, sign_a: int, mask_b: np.ndarray, mom_b: contracts.Moments, sign_b: int, *, grid: int = 256) -> float` | both masks whitened onto one grid; G4 and G8 |
| `wheel_coords(point_crop: Point2, mask: np.ndarray) -> tuple[float, float]` | (radius over the inscribed radius via the distance transform, angle) |
| `bumper_arc(point_crop: Point2, anchors: tuple[Point2, Point2] | None, mask: np.ndarray) -> tuple[float, float] | None` | arc coordinate between the two anchors and height over the mask height at that x; None when an anchor is missing (caller falls back to the moment frame with tau x 1.5 and records frame_used) |
| `hood_coords(point_crop: Point2, moments, front_sign: int, mask: np.ndarray) -> tuple[float, int]` | lateral whitened coordinate and the depth third from the windshield edge |
| `boundary_distance_units(point_crop: Point2, mask: np.ndarray, moments, front_sign: int) -> float` | whitened distance to the mask boundary, for the curvature cap |

Tests: a random affine leaves whitened coordinates unchanged within 1e-6; the carried moments against a full
decode on a synthetic clip under 0.02 units; the 20-degree synthetic residual bound (0.08 centre, 0.25 corners).

## match (M9), walkaround/match.py, pure, no API

| Function | Behaviour | In -> out |
|---|---|---|
| `calibrate(*, sigma_frame: float | None, sigma_spark: float | None, sigma_view: float | None, pseudo_pair_d: list[tuple[float, float]] | None, th: dict) -> contracts.Calibration` | 5.10: sigma_within, tau_pre0 = max(floor, 2.5 sqrt(2) sigma_within), tau_eye = 2 tau_pre, tau_fix = max(0.08, 3 sigma_frame); pseudo pair rows are (d, r_mean) tuples of mutual-NN pairs under tau_eye, each d brought back to r = 0 by the band factor (a bare float is read as r = 0): if fewer than 90% fall under tau_pre, tau_pre rises to the 90th percentile capped at 0.4 and `calibrated=False` above the cap | numbers -> Calibration |
| `tau_r(tau0: float, r: float, th: dict) -> float` | tau0 x (1 + band_radius_factor r^2) | |
| `pair_keyframes(kfs_pickup: list[contracts.Keyframe], kfs_return: list[contracts.Keyframe]) -> tuple[contracts.Keyframe, contracts.Keyframe, float]` | smallest abs(ln(ar_pickup / ar_return)) among the 9 candidates | |
| `handle_residual(kf_pickup: contracts.Keyframe, kf_return: contracts.Keyframe) -> Point2 | None` | whitened handle centroid difference; None when either handle is missing (self-test skipped and printed as skipped) | |
| `panel_gates(panel: contracts.Panel, kf_pickup, kf_return, *, mask_iou: float | None, residual: Point2 | None, lane_status: dict[str, dict[contracts.Noun, contracts.LaneStatus]], marks_truncated_all: dict[str, bool], calibration: contracts.Calibration, th: dict) -> list[str]` | the failing G ids in order G1..G10; empty means comparable | |
| `cost(a: contracts.Mark, b: contracts.Mark, wa: contracts.WhitenedMark, wb: contracts.WhitenedMark, calibration, th: dict) -> contracts.MatchCandidate` | d = abs(w_a - w_b) + type cost (0.1 compatible, 0.4 incompatible) + 0.1 abs(ln(s1 / s2)); r_mean; tau_pre_r and tau_eye_r by `tau_r` | |
| `mutual_nn(cands: list[contracts.MatchCandidate], th: dict) -> list[contracts.MatchCandidate]` | mutual nearest neighbours; ratio >= 1.5 and margin >= 0.5 tau_pre only when a side has two or more candidates; one candidate passes by rule; `mutual`, `ratio`, `margin` filled | |
| `verdict(d: float | None, tau_pre_r: float, tau_eye_r: float, flags: contracts.VerdictFlags) -> tuple[contracts.VerdictLabel, list[str]]` | THE single entry point, numbers and booleans only: d None -> L1 not_seen_at_return (pickup) or L2 new_or_unseen (return) by `flags.return_mark`; d < tau_pre_r -> L3 already_there; tau_pre_r <= d < tau_eye_r -> L4 needs_your_eye (K1); d >= tau_eye_r -> L5 and the unmatched label by `flags.return_mark`; then any set K flag lowers already_there, new_or_unseen or not_seen_at_return to needs_your_eye. Returns the label and the rule ids applied (label rule first, then the K ids from `flags.set_caps()`). No path moves a label up | numbers -> label |
| `match_panel(panel: contracts.Panel, marks_pickup: list[contracts.Mark], marks_return: list[contracts.Mark], whitened: dict[str, contracts.WhitenedMark], kf_pair: tuple[contracts.Keyframe, contracts.Keyframe], gates: list[str], calibration, th: dict) -> contracts.PanelMatch` | state, reasons = gates, aspect_diff, mask_iou, handle residual subtracted from the return marks, frame_used, pairs by cost and mutual_nn, unmatched lists, groups ("4 chips here at pickup, 5 at return, 1 new or unseen") | |
| `wheel_match(panel, marks_pickup, marks_return, whitened, *, th: dict) -> contracts.PanelMatch` | radius plus the best circular shift when both walks have >= 2 marks; one mark each side matches only on compatible type, radius within 0.1, size within a factor 2, else K10 | |
| `flags_for(pair: contracts.MatchCandidate | None, mark: contracts.Mark, other: contracts.Mark | None, mark_tracks: dict[str, contracts.MarkTrack], look: contracts.LookVerdict | None, kf_status: dict[str, contracts.MarksStatus], *, return_mark: bool) -> contracts.VerdictFlags` | assembles K1..K13 from `mark.caps`, the pair, the Stays Put verdict (K6), the second look (K7), the other keyframe's marks_status (K13) | |
| `verdicts_for(pm: contracts.PanelMatch, marks: dict[str, contracts.Mark], mark_tracks: dict[str, contracts.MarkTrack], looks: dict[tuple[str, str], contracts.LookVerdict], keyframes: dict[str, contracts.Keyframe], calibration, trace_base: contracts.Trace, th: dict) -> list[contracts.Verdict]` | one Verdict per pair and per unmatched mark through `verdict()`; `distance_cm_est` = d_pos x `cm_per_unit_for_print`; trace with rule ids, thresholds used, request ids, config_hash | |
| `derive_match(run_id: str, *, th: dict, second_look: Callable[[contracts.MatchCandidate], contracts.LookVerdict] | None = None) -> tuple[list[contracts.PanelMatch], list[contracts.Verdict], contracts.Calibration]` | reads panels, keyframes, marks, mark_tracks, calibration; runs gates, matches, the optional second look on already_there pairs (at most `second_looks_per_pair_max`), verdicts; writes matches.json and verdicts.json; under 1 s without the second look | derived -> matches.json, verdicts.json |

Tests: identical mark sets -> all already_there; a 0.3-unit shift -> needs_your_eye; one candidate per side
passes the ratio test by rule; a table test over every G and K id with its id in the trace; nothing raises a label.

## staysput (M8), walkaround/staysput.py (backlog B2; contract given so M7 and M9 can code against it)

| Function | Behaviour |
|---|---|
| `windows_for(marks: list[contracts.Mark], kf: contracts.Keyframe, track: contracts.PanelTrack, pts: list[float], *, th: dict) -> list[tuple[list[str], list[Box4]]]` | square windows of 0.3 x panel width (min 400 px) per clip frame, panel box interpolated between proxy frames, overlapping marks share one window, up to 16 per clip |
| `prompt_for(marks: list[contracts.Mark], window_frame45: Box4, clip_size: int, th: dict) -> str` | one PromptObject per mark, FramePrompt at frame 45, boxes through `sam_client.prompt_box` in clip pixels |
| `analyse(track_frames: list[contracts.IndexFrame], mark_id: str, panel_boxes_clip: list[Box4], kf_moments: contracts.Moments, front_sign: int, direction: int | None, *, th: dict) -> contracts.MarkTrack` | b1, b2, b_pass, panned, drift_rms against carried moments, sign test, curvature and highlight caps, verdict fixed/sliding/inconclusive/no_track |
| `async derive_staysput(run_id: str, role: str, *, th: dict) -> list[contracts.MarkTrack]` | clips via `video.cut_clip` (blurred, image checks on frame 45 first through `privacy.image_checks`), requests via `sam_client.segment_clip`, analysis, the reflection control; writes mark_tracks.json |

## privacy (M12), walkaround/privacy.py

| Function | Behaviour | In -> out | Ledger |
|---|---|---|---|
| `pixelate(rgb: np.ndarray, regions: list[Box4], *, block_px: int, dilate_frac: float = 0.0, dilate_px: int = 0) -> np.ndarray` | pure: each region dilated then replaced by block means; the same function for proxies, frames and clips | | |
| `plate_extent(mask: np.ndarray) -> tuple[float, float, float] | None` | (long side of the min-area rectangle in px, apparent h / w, tilt degrees) from the unblurred plate mask; None for an empty mask | | |
| `async image_checks(rgb_unblurred: np.ndarray, *, run_id: str, role: str, tag: str, th: dict) -> tuple[list[sam_client.Detection], list[sam_client.Detection], contracts.BlurRecord]` | LIVE: "license plate" then "face" through `sam_client.segment_image` on the in-memory unblurred frame; BlurRecord with plate_boxes, face_boxes, request_ids, plate_ruler_px / plate_ruler_c / plate_tilt_hw from the best plate (c >= `plate_c_min`, not touching the border) | frame -> detections + BlurRecord | `sam_image` x 2 via sam_client |
| `track_boxes_at(f_proxy: int, tracks: list[contracts.PanelTrack], *, fps: float, media_size_proxy: tuple[int, int], media_size_src: tuple[int, int], th: dict) -> list[Box4]` | plate and face track boxes at the frame, gaps under `track_gap_bridge_s` bridged by carrying the last box, scaled to source pixels | | |
| `blur_frame(rgb_unblurred: np.ndarray, *, track_boxes_src: list[Box4], detections: list[sam_client.Detection], keep_plate_region: bool, th: dict) -> np.ndarray` | pixelates the union of track boxes and image masks (faces always; the plate unless keep_plate_region) with `track_box_dilation_frac` + `track_box_dilation_px`, `pixel_block_px_at_4k` | | |
| `async blur_keyframe(run_id: str, role: str, frame_rgb: np.ndarray, *, f_proxy: int, tracks: list[contracts.PanelTrack], fps: float, sizes: tuple[tuple[int, int], tuple[int, int]], keep_plate: bool, th: dict) -> tuple[np.ndarray, contracts.BlurRecord]` | the step-7 order: image checks FIRST on the unblurred frame, plate extent recorded, THEN pixelation; returns the blurred frame (and, when keep_plate, the owner-only variant is produced by a second call with keep_plate_region True by the sheet writer); writes nothing | | via image_checks |
| `async render_blurred_proxy(run_id: str, role: str, tracks: list[contracts.PanelTrack], *, fps: float, th: dict) -> tuple[contracts.BlurredProxyRecord, contracts.DeletionRecord]` | `video.render_blurred_proxy` from the plate and face tracks; deletes the unblurred proxy and returns its DeletionRecord | | |
| `quarantine_put(run_id: str, role: str, src: Path) -> Path` | move the original to `paths.quarantine_original` | | |
| `delete_quarantine(run_id: str, *, now: str) -> list[contracts.DeletionRecord]` | every file under quarantine, hashed before removal | | |
| `async delete_remote(run_id: str, videos: list[contracts.VideoRecord], *, response_ids: list[str] | None, now: str) -> list[contracts.DeletionRecord]` | LIVE: `sam_client.delete_file` per upload (3 retries, failure recorded with reason), `delete_response` per id only when response_ids is given (store false refused) | | none |
| `close_deletions(run_id: str, rows: list[contracts.DeletionRecord], *, keep_plate: bool, chosen_by: str | None) -> contracts.Deletions` | writes deletions.json | | |
| `sweep(*, now: str, retention_hours: int, temp_minutes: int, keep_run_ids: set[str]) -> list[contracts.DeletionRecord]` | deletes runs past retention except keep runs and any quarantine temp older than temp_minutes | | |
| `served_file_allowed(run: contracts.RunManifest, rel_path: str, *, run_owner: bool) -> bool` | the serving rule of section 8 item 5: owner-only sheet frames with keep_plate; nothing from quarantine ever | | |
| `privacy_line(run: contracts.RunManifest, plate_found: dict[str, bool]) -> str` | the sentence of section 8 item 9 word for word, with the store false or deleted-per-request variant and the "no plate detected" clause | | |

Tests: deletions.json lists every unblurred path and none exists afterward; with keep_plate false no served file
carries plate pixels; `plate_extent` on an unblurred fixture vs the same frame pixelated differs by over 20%.

## seal (M4), walkaround/seal.py

| Function | Behaviour | In -> out |
|---|---|---|
| `build_manifest(run_id: str, files: list[Path]) -> contracts.SealManifest` | ManifestEntry per file (run-relative path, sha256, bytes); writes bundle/manifest.json | files -> SealManifest |
| `stamp_ots(path: Path) -> Path` | `ots stamp <path>` (never `--wait`); returns the .ots path; raises on a missing binary with the install line | |
| `ots_status(ots_path: Path) -> str` | `pending`, `confirmed:<block height>` or `failed:<reason>` from `ots info` | |
| `upgrade_ots(ots_path: Path) -> str` | `ots upgrade`; the hourly task; returns `ots_status` | |
| `stamp_tsa(path: Path, *, tsa_url: str = "https://freetsa.org/tsr", timeout_s: float = 20.0) -> tuple[Path, Path, str, str]` | `openssl ts -query` to a .tsq, POST with Content-Type application/timestamp-query, .tsr saved; returns (tsq, tsr, tsr_time ISO, serial); DigiCert fallback when freetsa fails, issuer in Stamps.tsa_issuer | |
| `verify_tsr(path: Path, tsr: Path, *, cacert: Path, tsa_crt: Path) -> bool` | `openssl ts -verify` | |
| `minutes_after_camera(sealed_at: str, creation_time: str) -> int | Literal["camera time not present"]` | integer minutes; the literal when creation_time is "not present" | |
| `seal_run(run_id: str, originals: list[Path], *, now: str, camera_time: str) -> tuple[contracts.SealManifest, contracts.ThreeClocks]` | manifest, ots on manifest and originals, freetsa; clocks.sealed_at = now, sealed_minutes_after_camera; first_model_call_at stays None (the orchestrator fills it at the first Files upload) | |
| `write_manifest_run(run_id: str, *, deletions: list[contracts.DeletionRecord], events_head_hash: str, config_hash: str, app_commit: str, pdf_sha256: str | None) -> contracts.SealManifest` | manifest_run.json with every kept artifact, stamped with `stamp_ots` | |
| `verify_run(run_id: str) -> list[str]` | re-hash every ManifestEntry, check the event chain through `eventlog.verify_chain`, compare derived files with a rebuild when `--out` exists; empty list means "rebuild identical" | run dir -> problems |

Ledger: none. Must not: run in the request path with `--wait`. Tests: `ots info` output parses pending and
confirmed on a fixture; `openssl ts -verify` passes on a fixture tsr; the minutes line prints "camera time not
present" when creation_time is absent.

## sheet (M13), walkaround/sheet.py

| Function | Behaviour | In -> out |
|---|---|---|
| `build_sheet(run_id: str, *, th: dict) -> contracts.SheetModel` | reads run.json, videos/*.json, bundle/manifest*.json, derived/*.json, deletions.json, the ledger rows of the run; fills page 1 (clocks, originals, plate line, privacy line, the five note lines), panel blocks (wheels and bumpers `lead_with_frames`), the not-comparable list with rule ids, the method page (thresholds, config hash, eval lines, deletions, storage line, rebuild status, "what the seal does not prove"), appendices and the letter; never recomputes a verdict | run dir -> SheetModel |
| `note_lines(sheet: contracts.SheetModel) -> list[str]` | the five lines in the order of section 7 item 6 (dated frames and hashes; request the full report; the alternative estimate and two-quote rule; executive and state escalation; the card dispute with its clocks and the collections warning) | |
| `letter_md(sheet: contracts.SheetModel) -> str`, `verify_md(sheet: contracts.SheetModel) -> str`, `certification_md(sheet) -> str`, `glossary_md() -> str` | appendix texts | |
| `size_line(size: contracts.SizeEstimate | None) -> str` | "about 0.9 in, between 0.6 and 1.3 in (ruler: the license plate, 12 in)" or "size not stated: <reason>" | |
| `ordinary_wear_line(mark_type: contracts.MarkKind, panel_id: contracts.PanelId) -> str | None` | the Hertz REV 7.2025 sentence for a body scuff, else None | |
| `staysput_line(mt: contracts.MarkTrack | None) -> str | None` | "drift 0.03, b1 0.20, b2 0.01, not panned" or "inconclusive (camera did not move enough)" | |
| `render_html(sheet: contracts.SheetModel, *, template_dir: Path) -> str` | Jinja2, IBM Plex, footer with run id and the 12-hex prefixes on every page, the three clocks on page 1 | SheetModel -> HTML |
| `render_pdf(html_path: Path, pdf_path: Path, *, timeout_s: int = 60) -> Path` | Playwright Chromium; raises `TimeoutError` (the caller serves the HTML and shows "PDF retry") | |
| `write_sheet(run_id: str, *, th: dict) -> tuple[contracts.SheetModel, dict[str, str]]` | sheet.json (through `contracts.write_json`), sheet.html, evidence.pdf (missing on PDF failure); returns the model and {file: sha256} | run dir -> sheet/ |

Ledger: none. Must not: re-run any model. Tests: renders every label, every exit, every refusal reason, the
Ordinary Wear line and the three clocks from a fixture SheetModel; footer hash prefix on every page.

## eventlog (M3), walkaround/eventlog.py (the orchestrator and app depend on it)

| Function | Behaviour |
|---|---|
| `append(run_id: str, kind: contracts.EventKind, payload: dict[str, Any], *, ts: str | None = None) -> contracts.Event` | next seq, prev = last hash (or the sha256 of manifest.json for seq 0, read from `paths.manifest_json`), hash = `contracts.chain_hash`; one line appended; subscribers notified |
| `read(run_id: str, *, since_seq: int = -1) -> Iterator[contracts.Event]` | lines after since_seq |
| `head_hash(run_id: str) -> str | None` | |
| `verify_chain(run_id: str, *, manifest_sha256: str) -> bool` | every hash recomputes; one edited line fails |
| `async subscribe(run_id: str, *, since_seq: int = -1) -> AsyncIterator[contracts.Event]` | the SSE tail: history from since_seq then live |

Payload shapes per kind are those of section 4 (sam.request {request_id, role, noun or prompt_id, params, store};
sam.first_delta {ms}; sam.frame {role, noun, frame, objects [{id, box, c}]} with no raster; sam.lane_done
{lane_path, sha256, frames, ids, status, frames_sent, video_frames_processed, parse_cpu_s, dollars};
spark.request {max_output_tokens, effort, store}; spark.response {sha256, status, incomplete_reason};
voice.done {mode_applied}; match.derived {sha256 of matches, verdicts}; ledger.row {dollars_so_far};
stage {name, pct}; run.partial {missing[]}; files.deleted {file_id, response}; responses.deleted
{request_id, response}). Mask text never appears in an event.

## orchestrator (M14), walkaround/orchestrator.py

| Function | Behaviour |
|---|---|
| `class Orchestrator.__init__(self, *, th: dict, engine: contracts.SamEngine, max_concurrent_runs: int, media_workers: int, decode_workers: int, now: Callable[[], str] = utc_now)` | semaphores, the queue, the decode pool, the health flag |
| `async create_run(self, *, mode: contracts.RunMode, kind: contracts.RunKind, keep_plate: bool, live_or_replay: contracts.LiveOrReplay = LIVE) -> contracts.RunManifest` | ULID run_id, `paths.ensure_run_layout`, thresholds frozen with config_hash, run.json written, run.created emitted |
| `async finalize_upload(self, run_id: str, *, upload_id: str, role: str, sha256_browser: str, bytes: int, filename: str) -> contracts.VideoRecord` | reassemble chunks into quarantine, `hashlib.file_digest`, raise `HashMismatch(sha256_browser, sha256_server)` (the app turns it into a 409 with both values); upload.done emitted; the caller starts the run (the app's finalize route after admit, the CLI through pipeline.run_pair) |
| `async run_pipeline(self, run_id: str) -> contracts.RunManifest` | the stage order as built (orchestrator.STAGES): seal -> media (proxy, pts, WAV) -> Files upload (first_model_call_at) -> 16 lanes per video -> panels, keyframes, coverage (the car filter) -> blur (proxy_blurred from plate and face tracks rebuilt from their lanes, unblurred proxy deleted) -> marks -> Stays Put -> voice and narration -> match and verdicts -> release (remote and local deletions, deletions.json, a first manifest_run.json) -> sheet -> close (final manifest_run.json) -> run.completed; status set at each stage; spend ceiling and the 12 min deadline enforced; a stage failure past retries emits run.partial with the missing list and still writes the sheet |
| `async run_topup(self, run_id: str, role: str) -> contracts.Coverage` | only `coverage.nouns_to_rerun` on the new clip; rerun_seconds measured and printed |
| `async resume_all(self) -> list[str]` | on boot: runs without run.completed become interrupted and resume lanes without lane_done, re-uploading the proxy when the file_id expired |
| `spend_cents(self, run_id: str) -> int` | from ledger rows of the run |
| `caps_remaining(self, *, et_day: str) -> dict[str, int]` | {pickup, pair} from ledger dollars and run kinds per ET day against `pickup_runs_per_day`, `pair_runs_per_day`, the daily card ceiling |
| `async health_probe(self) -> bool` | GET /v1/models; three failures in a row set `self.paused` |
| `async sweep(self) -> list[contracts.DeletionRecord]` | `privacy.sweep` hourly |
| `emit(self, run_id: str, kind: contracts.EventKind, payload: dict[str, Any]) -> contracts.Event` | `eventlog.append` |
| `set_stage(self, run_id: str, status: contracts.RunStatus, pct: int) -> None` | run.json status plus a stage event |

Must not: render; compute a verdict. Tests: resume after a simulated restart re-runs only lanes without
lane_done; the spend ceiling aborts remaining nouns as not_run; the caps count from a synthetic ledger per ET day.

## app (M15), walkaround/app.py, FastAPI

| Route | Request -> response |
|---|---|
| `PUT /api/uploads/{upload_id}/chunks/{n}` | raw bytes <= 8 MB -> 204; idempotent by index; 413 over `chunk_route_max_body_bytes` |
| `POST /api/uploads/{upload_id}/finalize` | JSON {run_id, role, sha256_browser, bytes, filename, keep_plate} -> `contracts.VideoRecord`; 409 {sha256_browser, sha256_server} on a mismatch; 413 over 1 GiB |
| `POST /api/runs` | {mode, kind, keep_plate} -> `contracts.RunManifest`; 429 {reason, pickup_left, pairs_left} when the caps or the pause apply |
| `GET /api/runs/{id}` | `contracts.RunManifest` |
| `GET /api/runs/{id}/events` | SSE: `id: <seq>`, `event: <kind>`, `data: <Event JSON>`; `Last-Event-ID` resumes from seq + 1; `: ping` every `sse_ping_s` |
| `GET /api/runs/{id}/replay?mode=fixed|recorded|instant` | `contracts.ReplayPayload` (delays_ms from web/kiosk/delays.json for fixed; recorded delays capped at 2000 ms) |
| `GET /api/runs/{id}/sheet` / `sheet.json` / `sheet.pdf` / `bundle.zip` | HTML / `contracts.SheetModel` / PDF / zip of bundle/ |
| `GET /api/runs/{id}/files/{path}` | a stored blurred artifact (gz lanes, indexes, crops, clips, chips, proxy_blurred) after `privacy.served_file_allowed`; 404 for anything else |
| `POST /api/runs/{id}/topup` | multipart video -> `contracts.Coverage` |
| `POST /api/claims` | multipart {image, text, run_id, tire_code?} -> `contracts.Claim` |
| `GET /api/status` | {paused: bool, pickup_left: int, pairs_left: int, queue_position: int | null, engine} |
| `GET /healthz` | {ok: true, engine, probe_ok, free_bytes} |
| `GET /`, `/run/{id}`, `/run/{id}/sheet`, `/claim`, `/kiosk/{id}`, `/verify`, `/eval/label.html` | the Vite build from `paths.web_dist_dir()`, same origin |

`create_app(orchestrator: Orchestrator) -> FastAPI` is the one factory; tests build it with a fake orchestrator.
Tests: chunk assembly hashes equal on a 50 MB random file; 409 on mismatch; SSE resumes from Last-Event-ID.

## web reducer (M16), web/src/state/reducer.ts, event format

- Input: `contracts.Event` JSON exactly as the SSE `data:` or `ReplayPayload.events[]` carries it:
  `{seq, ts, run_id, kind, payload, prev, hash}`; `kind` is one of `contracts.EventKind`.
- `reduce(state: RunState, event: Event): RunState` is pure and never branches on live vs replay.
- `RunState` = `{run: RunManifest | null, videos: Record<Role, VideoRecord>, lanes: Record<Role, Record<string,
  {status, frames, ids, first_delta_ms, dollars}>>, frames: Record<Role, Record<string, {frame, objects}>> (the last
  sam.frame per noun, boxes only), panels: Panel[], coverage: Coverage | null, keyframes: Keyframe[], marks: Mark[],
  matches: PanelMatch[], verdicts: Verdict[], narration: Narration | null, markTracks: MarkTrack[], sheet: {written:
  boolean, sha256s}, deletions: DeletionRecord[], ledger: {dollarsSoFar, inFlight, p50FirstDeltaMs, parseCpuMs},
  stage: {name, pct}, errors: string[], partial: string[] | null, replay: {mode, badgeDate} | null}`.
- Derived JSON referenced by an event (panels.derived, marks.derived, match.derived, staysput.derived,
  narration.derived, coverage.updated, sheet.written) is fetched by the driver from
  `/api/runs/{id}/files/derived/<name>.json` and delivered as a synthetic `{kind: "<kind>.loaded", payload}` event
  the reducer also handles, so live and replay feed identical sequences.
- Drivers: live = `EventSource('/api/runs/{id}/events')` with Last-Event-ID; replay = `ReplayPayload` paced by
  `delays_ms` per stage (fixed), by capped `ts` deltas (recorded) or not at all (instant).
- Worker protocol (web/src/workers/lane.worker.ts; as built it posts an SVG path per object, ImageBitmaps only on request, see the integration notes): request `{type: "parse", url, role, noun}`; the worker fetches the gz
  lane, parses with `@meta-sam/parser`, posts `{type: "snapshot", role, noun, revision, frame, objects: [{id, box,
  c, mask: ImageBitmap}]}` and `{type: "done", role, noun, frames}`. No lane parsing on the main thread.
- Test: a recorded event log reaches the same final state live and replayed (vitest); fixed delays within 20 ms.

## D1 scripts (M19), scripts/gate_*.py

Every script: `main(argv: list[str] | None = None) -> int`, argparse, `--out runs/gate_<row>_<name>` (a fixture
directory with `gate_summary.md` and the JSON it measured), `--fixture <path>` to replay saved events with no call,
`LIVE=1` required for any call (otherwise the script prints what it would send and exits 2); every call writes a
ledger row with `note="D1 E<n> <what>"`; every number printed also lands in the JSON. Scripts import pipeline
modules and are never imported by them.

| Script | Rows | Signature of the top-level helpers |
|---|---|---|
| `scripts/gate_setup.py` | E0 | `check_tools() -> dict[str, str]`; `hash_originals(paths: list[Path]) -> dict[str, str]`; `proxies(src: Path, out: Path, fps_list: list[float]) -> list[contracts.ProxyRecord]` |
| `scripts/gate_sam.py` | E2, E3, E4, E9, E15 | `async sweep(file_id: str, nouns: list[contracts.Noun], *, run_id, role, frames_sent, media_size, out: Path) -> list[contracts.LaneRecord]` (16 at once, store false, one text lane with propagation_direction both); `c_semantics(records: list[contracts.LaneRecord], indexes) -> dict`; `fragment_report(tracks: list[contracts.PanelTrack]) -> dict`; `anchor_bridge(...)` (E4); `blur_coverage(...)` (E9); `lane_sizes(out: Path) -> dict` (E15) |
| `scripts/gate_spark.py` | E6, E7, E11, E18 | keeps the existing CLI (`IMAGE_OR_VIDEO --panel --crop --t --effort --n --out`); adds `--truncate 600` (the on-purpose truncation call), `--count` (E7), `--second-look A B` and `--claim` (E11), `--note-prefill PDF` (E18); `stability(runs, crop_shape) -> dict` stays |
| `scripts/gate_voice.py` | E10 | keeps the existing CLI; adds `--pins panels.json` to print turn spans against panel windows and the mode_applied flag |
| `scripts/gate_stays_put.py` | E5 | `async run_clip(src: Path, *, pts_center, window_src: Box4, mark_box_src: Box4, out: Path) -> contracts.MarkTrack`; `--panned` and `--backtrack` flags label the clip |
| `scripts/gate_handle.py` | E8 | `sigma_frame(...) -> float` from one "door handle" video request over 10 consecutive frames; `cross_take_residual(kf_a, kf_b) -> Point2` |
| `scripts/gate_seal.py` | E12 | ots and freetsa smoke on a text file; `store_false_probe() -> dict` (sam-3.1 and muse-spark-1.3 echo); `delete_probe(file_id) -> dict` |
| `scripts/gate_wallclock.py` | E13 | one pickup-only check and one pair through the orchestrator with timings and dollars from the ledger |
| `scripts/gate_s1.py`, `scripts/gate_s2.py` | E16, E17 | Baseten second look x 5; the timeline_v1 call |
| `scripts/gate_fal.py` | E19 | the fal.ai adapter on a recorded fixture emitting `contracts.IndexFrame` rows |

## Changes to the existing stubs, in one list

1. `video.make_proxy`: default `fps` 10.0 -> 15.0; `preset` and `crf` keyword arguments added. Everything else kept.
2. `sam_client`: `upload_file`, `segment_video`, `segment_video_many`, `segment_image` become `async`; `upload_file`
   returns `FilesApiRecord` (file_id plus times) instead of `str` and takes `expires_after_s`; `segment_video`
   takes `prompt_text, *, run_id, role, lane, frames_sent, media_size, metadata, kind, noun, request_index,
   on_event, max_tries` and returns `(LaneRecord, list[IndexFrame])` instead of a parsed `SegmentationRun`;
   `segment_video_many` takes `(lane, prompt_text)` pairs; new `segment_clip`, `delete_file`, `delete_response`,
   `sam_metadata`, `prompt_box`, `object_prompt`, `stream_events`, `client`. `Box`, `Detection`,
   `SegmentationRun`, `replay`, `mask_array`, `full_mask` kept. The docstring keeps the shim but says it adds the
   done event only when the API did not.
3. `spark_client`: `Mark`, `MarkKind`, `SizeClass`, `MARK_KINDS`, `SIZE_CLASSES`, `MARKS_SCHEMA`, `COMPARE_SCHEMA`,
   `compare_marks`, `mark_to_crop_pixels`, `marks_prompt(panel_name)` replaced by the contracts enums, `SCHEMAS`,
   `second_look`, `point_to_crop_pixels` / `box_to_crop_pixels`, and prompts keyed by `PanelId`; `find_marks`
   takes `panel_id` and `landmarks`; `store=False` added; truncation is its own state; `MarksResult.answer` is a
   `SparkMarksAnswer`. `prepare_image`, `grid_to_pixels`, `_client`, `_ask` shape kept.
4. `voice_client`: `Transcript.mode_applied` and `Transcript.request` added; no synthetic whole-clip turn;
   `DEFAULT_KEYWORDS` = the 21 words; `validate_mode` and `voice_record` added. `check_wav`, `transcribe_wav`
   signature, `Turn`, `TranscribeError`, `TRANSPORT` kept. tests/test_voice_client.py must drop its assertion on
   the synthetic turn if it has one.
5. `ledger.record`: unchanged; lanes pass the extra row keys listed under "Shared ground".
6. `config.CAR_PANELS`: becomes `contracts.LANE_NOUNS` (car, car window, fender added; "front bumper" and "rear
   bumper" removed as prompts). config.py keeps the list for now; nothing new imports it.

## Doubts and places the contracts chose where the architecture was silent or disagreed

1. output_text.done: section 5 step 5 says "no synthesis of output_text.done anywhere", while docs/API_NOTES.md,
   the cookbook reference client and the lane brief say to copy the shim. The shim is harmless (meta_sam_parser
   0.1.0 finalises on either event and accepts both when the text agrees) and the 10/26 fixture shows the API
   itself emitting both. Contract: keep the shim, and the wire reader accepts either finalizer, both or neither.
2. Index payload offsets: section 4 says they point "into the gz lane". A byte offset into a gzip stream cannot be
   seeked, so `MaskRef.payload_offset` counts characters into the lane's concatenated output text (ASCII, so
   chars equal bytes); a decode gunzips the lane once and slices.
3. Within-walk dedup is listed under M6 in section 3 and under M9 in 5.6. It lives in `marks.dedup_within_walk`
   (M6) using `geometry.whiten`; match only reads the result.
4. `verdict(d, tau_pre_r, tau_eye_r, flags)` needs to know whether an unmatched mark is a pickup or return mark.
   `VerdictFlags.return_mark` carries it as a boolean so the signature stays numbers and booleans only.
5. Canonical panel ids: section 4 lists `headlights`, `tail lights`, `mirrors` as group slots and per-side ids for
   doors, fenders and wheels. The enum follows that (22 slots: 13 body, 4 wheels, 5 light and glass groups);
   per-side light instances are tracks under the group slot, not separate slots.
6. The ledger row of section 4 names `dollars`; ledger.py already writes `usd` and LEDGER.md copies it. The
   contract keeps `usd` as the column and documents the alias.
7. Module layout is flat (`walkaround/<lane>.py`) rather than the architecture's package directories, because the
   stubs and the lane assignments are flat.
8. `LaneStatus.FAILED` is added to section 4's completed | zero_matches | incomplete so a 429 past backoff and a
   builder error are distinguishable from a stream that broke mid-way (both still print not_run).
9. Thresholds that section 5 gives only as words (blur dilation 10% plus 24 px, block 20 px, b2 floor 0.05,
   two-pass score exponents) are in thresholds.json marked "estimate"; D1 replaces them and the config hash
   changes with them.
10. The zero-match fixture does not exist yet (the 10/26 probe had no zero-match lane); the sam lane makes one by
    hand and E2 replaces it with a real "face" lane.

## Integration notes (2026-10-31)

The integrator reconciled the lane reports against this file. Where a lane deviated and the deviation was sound, the
deviation stands and is listed here; the tables above keep their original wording except for the five lines fixed in
place (calibrate's pseudo-pair rows, finalize_upload, run_pipeline's stage order, the worker file, gate_stays_put.py).

New since the lanes:

- `walkaround/pipeline.py`: `run_pair(pickup, return_, run_id=None, nouns=None, **kw) -> RunManifest` and
  `async run_pair_async(..., container=None, keep_plate=False, orch=None, on_event=None)` run the floor on local
  files through the browser's upload path and every orchestrator stage; an unfinished run of the same id resumes (the
  walks must hash to the stored originals). `rebuild(run_id, *, out=None, thresholds=None, orch=None) -> (dict, out)`
  (a closed run goes to runs/_rebuild/<run_id>) and `verify(run_id, *, out=None, orch=None) -> list[str]`.
  `stand_in_prompts(nouns, *, container=None) -> (noun_list, prompts)` maps up to four stand-in prompts onto the
  single-slot panels hood, trunk lid, windshield, rear window. The CLI (`walkaround`, now a console script) calls these:
  `walkaround run P [R] [--run-id ID] [--nouns N...] [--container N]`, `walkaround rebuild ID`, `walkaround verify ID`.
- `contracts.RunManifest.prompts: dict[str, str]` (test clips only: the text a lane is sent with instead of its noun;
  `{"car": ""}` means no car request, the whole frame is the car box) and `contracts.SheetModel.stand_in_line: str`
  (page 1 says a run used stand-in prompts). web/src/contracts/types.ts carries `prompts?`.
- `Orchestrator.create_run(..., run_id=None, noun_list=None, prompts=None)`: a chosen run id (letters, digits, _ and -,
  not taken), a narrowed lane list (car, license plate and face always stay), stand-in prompts. `prompt_text(run,
  noun)`, `whole_frame_container(run)`.
- Identical requests are asked once per run. A walk whose original has the same sha256 as an earlier walk of the run
  (`reuse_source`) gets that walk's finished lanes copied (diagnostic `reused_identical_request`, no request, no ledger
  row, dollars 0); `shared_checks(run_id)` asks the plate and face image checks once per distinct unblurred frame
  (passed to `privacy.blur_keyframe` as the new optional `checks=`); `marks.derive_marks` reads a valid marks_v1 answer
  back for the same (crop sha256, panel). Measured 10/31 with scripts/probe_sam_determinism.py (run id
  probe_det_1031 in the ledger, $0.04): SAM 3.1 answers two identical video requests with 18 of 178 boxes one pixel
  apart and two identical image requests with one face mask 2 px apart, so without this a self-pair could never match
  itself at distance 0.
- `marks.rebuild_marks(run_id, role, *, th) -> list[Mark]`: marks.json rows, sigma_view and calibration.json
  recomputed from spark.json and keyframes.json with no call (box mode). Orchestrator.rebuild now regenerates
  coverage.json, marks.json, calibration.json, matches.json, verdicts.json and sheet.json; mark_tracks, narration and
  claim stay "not rebuilt" until their lanes exist.
- `spark_client.sent_size(w, h, max_edge)` and `spark_client.marks_result_from_record(record, *, orig_width,
  orig_height, landmarks, max_edge)`.
- `privacy.privacy_line(run, plate_found, deletions=None)`: a Files API delete that failed prints "Walkaround's delete
  request failed, so that expiry applies" instead of claiming the delete (section 14 item 25).
- `ledger.rows(path=None)` and `ledger.summary(path=None)` read `LEDGER_PATH` at call time.
- The orchestrator records `WavRecord.duration_s` from the WAV itself, not the container.

Accepted lane deviations (all additive unless noted):

- video: `frame_times`, `clip_span`, `iter_frames_at`, `iso_z`, `VideoError`, `NoAudioError`, `ClipOutOfRange`,
  `VIDEO_STREAM`, `sharpness(..., mask=None)`, `render_blurred_proxy(..., preset, crf)`, VideoInfo fields vfr, hdr,
  device, start_time, color_space, color_transfer, color_primaries. pts_map values are the proxy's own pts on the
  source timeline (up to one source frame from the shown frame's own pts); pts_to_proxy_frame is nearest-frame.
  render_blurred_proxy accepts boxes past the last frame.
- sam_client: `LaneRecord.lane_id` is the noun slug and `lane_path` is relative to the run directory (resolve with
  paths.run_dir); a lane that never got an answer writes no ledger row; `segment_image` raises SamRequestError when a
  check does not complete (callers catch it); `upload_s` is never set; `store` is the effective store value sent.
- spark_client: `ask(..., retry, schema, check_schema)`, `ask_with_retry(..., stream, schema, check_schema)`,
  `find_marks/count_marks(..., max_edge)`, result dataclasses carry `attempts`, ClaimNameResult carries noun and bbox,
  `second_look(..., input_refs=None)`. SparkRequestRecord has no store echo field (gate_spark records it itself).
- voice_client: VoiceTurn.speaker is stored as a string.
- panels: `vote_sides(..., direction_by_role=, frozen_roles=, expected_by_role=)`, `walk_direction(..., backup=)`,
  `fragmented_panels(tracks, th)`, `privacy_tracks(role, noun, frames, pts, car_boxes, media_size, *, fps, th)` (the
  orchestrator's blur rebuilds plate and face tracks with it); needs_anchor_pass is per noun before the vote and per
  panel after it; derive types `decode` as any object with `.decode(lane_path, refs)`.
- marks, geometry, match: additive helpers `geometry.moments_from_points, whitening_matrix, unwhiten, whitened_mask,
  inscribed_radius`, `marks.whiten_marks, whitened_from_marks, collinear_merge, crop_from_proxy_box`,
  `match.measured_sigmas(th)` (reads optional thresholds leaves matching.sigma_frame and matching.sigma_spark,
  absent until D1); `wheel_match(..., calibration=None)`; G4 and G8 fire only when both keyframes exist; derived/
  sigma_view.json is a per-role helper file (not in DERIVED_FILES). geometry.perspective_residual_* hold only after
  the handle residual is subtracted (the bare moment frame gives about 0.24 at 20 degrees).
- privacy, seal, sheet: `blur_frame(..., plate_track_boxes_src=, plate_detections=)`, `track_boxes_at(..., nouns=)`,
  `delete_remote(..., retries=None)`, `seal_run(..., stamp=True)`, `write_manifest_run(..., stamps=None)`,
  `verify_run(..., rebuild_dir=None, originals=None)`, `seal.verify`, `privacy.read_original_frame`; blur_keyframe
  always pixelates the plate (keep_plate stays in the signature); manifest.json holds only {path, sha256, bytes} per
  original (capture times live in videos/<role>.json, stamps in the seal.ots and seal.tsr events and in
  manifest_run.json); manifest_run.json excludes run.json and the event log and pins the log with events_head_hash;
  the method page's rebuild_status cannot say "rebuild identical" at sheet time; stamps ride in
  sheet.method.thresholds["stamps"] until SheetModel has a field.
- orchestrator and app: there is no walkaround/eventlog.py; `orchestrator.FileEventLog` implements the M3 contract.
  Pre-seal events wait in events.pending.ndjson and are chained at the seal (seal.manifest is not seq 1); a run that
  fails before the seal shows its waiting events unchained (`visible_events`). New public methods admit,
  discard_uploads, check_can_topup, card_ceiling_reached, visible_events, is_closed, reuse_source, shared_checks;
  exceptions VideoUnreadable, RebuildRefused, UploadRefused; `caps_remaining(..., exclude=)`;
  `finalize_upload(sha256_browser: str | None)`. rebuild without --out is refused for a closed run (the CLI rebuilds
  into runs/_rebuild/). voice.json holds a list of VoiceRecord. App extras: GET /api/runs (public runs only), GET
  /api/sample, /api/status queue_eta_minutes, 204 from /events for an ended run with nothing new, error bodies 429
  {reason, pickup_left, pairs_left} on finalize and start, 403 on public runs for top-up and claims.
- web: sam.* events name lanes by slug (car_door); RunState also carries media, seal (with files_uploaded), tracks,
  topup, spark, voice, partialText, completed, lastSeq, lastTs and the synthetic kinds run.loaded, video.loaded,
  deletions.loaded, run.ended.loaded; WalkPlayer uses @meta-sam/react's useMediaPlayer with worker-traced paths;
  MatchPane shows the keyframe crops with the SAM mask tinted under them.
- D1: E8 lives in gate_sam.py (--handle, --take2), E17 in gate_sam.py (--timeline), E0 and the E12 seal smoke and E13
  in run_d1.py; E16 and E19 are recorded as not run. Ledger notes carry the lane slug, the Spark purpose or the Voice
  mode, and the E row rides in every run_id (<prefix>_E6_...). Take 2 runs under role pickup with its own run id.
  run_d1.py takes any readable pickup, return, take 2 and narration path and names a missing or unreadable one.

Still open: narration (M10), claim (M11), Stays Put (M8) and the second look are not wired; their stages record gaps
and the sheet prints them.

## Phase 2 file shapes (lead, 2026-10-31, binding for the second_look, claim, narration and web lanes)

Reviewers: check the lanes against this section. Integrator: reconcile any lane that differs to this section.

- `derived/looks.json` (`paths.looks_json`): a JSON array, canonical like every derived file, sorted by
  (pickup_mark, return_mark), one row per second look: `{"pickup_mark": str, "return_mark": str, "panel_id": PanelId,
  "verdict": "same"|"different"|"unsure", "reason": str (one plain sentence, at most 200 chars), "engine":
  "baseten"|"meta-model-api", "model": str (the response's own model field), "request_id": str|null, "latency_ms":
  int|null, "fallback_from": "baseten"|null (Baseten failed and hosted Muse Spark 1.3 answered), "valid": bool (false
  when the answer failed the schema; verdict is then "unsure")}`. Only "different" and "unsure" lower a verdict (K7).
  Rebuild replays it with no network call.
- `open_weights.ndjson` at the run root (`paths.open_weights_log`), not in derived/: append-only provenance, never
  rebuilt, one line per Muse Spark Open call on Baseten, failed calls included: `{"ts": ISO 8601 UTC, "run_id": str,
  "engine": "baseten", "model": str (as returned; the requested id when the call failed), "request_id": str|null,
  "latency_ms": int, "pickup_mark": str, "return_mark": str, "panel_id": str, "verdict": "same"|"different"|"unsure"|null,
  "reason": str, "valid": bool, "error": str|null (redacted)}`. No image bytes, no keys.
- Claims: `derived/claim.json` (`paths.claim_json`) is the latest `contracts.Claim`; every claim is also
  `claim/<claim_id>.json` (`paths.claim_record_path`) with `claim_id = "claim_<n>"` from 1, now a field on
  `contracts.Claim`. The upload `claim/<claim_id>.jpg` is never served; the app serves only the pixelated copy
  `claim/<claim_id>.blurred.jpg` (`paths.claim_blurred_path`). POST /api/claims returns the Claim it wrote.
- `RunManifest.footage_credit: str | None` (new, optional): a sample run's credit line, shown under the walk player.
- Privacy allowlist (integrator): looks.json, open_weights.ndjson, claim/<claim_id>.json, claim/<claim_id>.blurred.jpg,
  the narration chips and the blurred Stays Put clips; never claim/<claim_id>.jpg.
