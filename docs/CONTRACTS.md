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
- `walkaround/rules.py`: the rule id table (G1 to G10 gates, G2 retired 10/31; K1 to K15 caps; L1 to L6 labels; E1 to E8
  claim exits).
  `rules.rule(id)` raises KeyError on an unknown id, so no sheet prints an id outside the table.
- Ledger: every model call goes through `walkaround.ledger.record(kind, model, *, note, run_id, **usage)` exactly
  once. Kinds: `sam_video` (frames=...), `sam_image` (images=1), `spark` (input_tokens, output_tokens,
  cached_tokens, reasoning_tokens), `voice` (audio_seconds). Extra keys are stored on the row: lanes pass
  `engine`, `request_id`, `unit`, `quantity`, `latency_ms={upload, first_delta, done}`, `parse_cpu_ms`, `status`,
  `retries` so the row matches `contracts.LedgerRow` (`usd` is the row's dollars column).
- Live calls: only inside functions this file marks LIVE, and tests reach them only behind `LIVE=1`. Every
  `/v1/responses` request carries `store=False`.
- Verdicts: deterministic code owns every label; a model answer can only lower one (K4, K7, K12, K13, K15, G10).
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

## staysput (M8), walkaround/staysput.py (built 10/31; records and shows, caps nothing while cap_enabled is false)

| Function | Behaviour |
|---|---|
| `windows_for(marks, kf, track, pts, *, th, src_size, proxy_size) -> list[tuple[list[str], list[Box4]]]` | square windows of 0.3 x panel width (min 400 px) per clip frame over a 3 s clip of 2 x round(1.5 x fps) + 1 source frames (at most 181); pts are the original's frame times; the panel box is interpolated between proxy frames; overlapping marks share one window (16 per clip); marks under min_bbox_px_at_4k are left out (section 14 item 17) |
| `prompt_for(marks, window_frame45, clip_size, th, *, prompt_frame=None, src_size=None) -> str` | one PromptObject per mark at the prompt frame, boxes through `sam_client.prompt_box` in clip pixels |
| `analyse(track_frames, mark_id, panel_boxes_src, kf_moments, front_sign, direction, *, th, object_id, windows_src, clip_size, kf_box_src, crop, src_size, proxy_size, role, fps=None, prompt_frame=None) -> contracts.MarkTrack` | panel boxes in SOURCE pixels (None where unknown); b1, b2, b_pass, panned, drift_rms against carried moments, the sign test on non-overlapping pairs scaled to fps, box share over the propagated frames; verdict fixed/sliding/inconclusive/no_track; the caps are set by the caller |
| `async derive_staysput(run_id, role, *, th, stop=None) -> list[contracts.MarkTrack]` | LIVE. Plans a walk's clips (none when its plate or face lane did not complete; a prompt-frame check box the walk's tracks do not cover refuses the clip), cuts them with `video.cut_clip` (blurred, plate and face checks on the prompt frame first), asks `sam_client.segment_clip`, adds the reflection control when a hood glare blob passes the 5.9 item 6 test, writes clips/<role>/staysput.json (plans, no pixels) and derived/mark_tracks.json (keyframe caps on every row; a not_run row for every mark with no clip). A resumed walk asks nothing twice; a byte-identical walk reuses the other walk's check and lane. stop (the orchestrator's past_deadline) holds back clips not started |
| `role_done(run_id, role) -> bool`, `planned_roles(run_id)`, `request_ids(run_id) -> list[str]` | the stage's done check per walk; every response id Stays Put asked (clip checks live only in staysput.json) |
| `rebuild_staysput(run_id, *, th) -> list[contracts.MarkTrack]` | mark_tracks.json from the stored plans, clip lanes and keyframe crops; no network, no original; byte-identical; a run that never planned a clip and has no mark_tracks.json is left as it is |

`tau_fix` reads staysput.sigma_frame (absent until E5 is recalibrated, so the 0.08 floor holds), never matching.sigma_frame.
match ignores Stays Put for K6 while thresholds staysput.cap_enabled is false (`match.staysput_caps`); the sheet then
prints the measured numbers with "recorded only: Stays Put is not calibrated yet and changes no label" and the web
shows numbers only. Clip lanes are sam/<role>/sp_<clip_id>.* (served by the sam rule); clips/<role>/<clip_id>.mp4 is
served; staysput.json and temp files are not.

## narration (M10), walkaround/narration.py (built 10/31)

| Function | Behaviour |
|---|---|
| `async derive_narration(run_id, *, th) -> contracts.Narration` | LIVE (seed_v1 only; Voice runs in the orchestrator). Cuts each Voice turn where the words name a new panel, pins each piece to the panel SAM shows most during it (side and position words from the text, else the side vote), "about" pins when SAM never shows the named panel; mode_applied false when a speech mode returned a transcript with no turns (the transcript is kept on NarrationRole.transcript). Every turn gets a chip, chips/<role>/<turn_id>.m4a holding the whole Voice turn (NarrationTurn.start_ms and end_ms are the Voice turn's span; the character-offset estimate only chooses the pin and is stored as est_ms). A pinned piece that names a mark word asks `spark_client.seed_mark` once on that walk's sharpest blurred crop of the panel; a found box on the panel becomes a narration_seed Mark (cap K14, confidence low) or links to the box-pass mark inside the match's tau_eye band |
| `rebuild_narration(run_id, *, th) -> contracts.Narration` | narration.json, derived/narration_pieces.json, the chips and the seed rows of marks.json from voice.json, seeds.json and spark.json; no call; byte-identical |
| `merge_seed_marks(run_id, *, th) -> list[contracts.Mark]` | the seed rows back into marks.json with no call (a helper; not wired) |
| `seed_records(run_id) -> list[tuple[str, SparkRequestRecord]]` | (role, record) per seed answer, so the orchestrator chains spark.request and spark.response for them |
| `split_turn(..., passenger=Side.RIGHT)`, `pin_piece(..., fallback="turn_start_panel")`, `side_words`, `seed_cap`, `role_states`, `seeds_json(run_id)`, `pieces_json(run_id)` | helpers |

Files: seeds.json at the run root (every seed question, never rewritten, never served); derived/narration_pieces.json
(per piece: estimated, est_ms, chip_ms, the seed outcome with status, reason, request id, mark id and box; type_from and
size_from). Turn ids are t<n> for a whole Voice turn and t<n>_<k> for the k-th piece cut from it; seed mark ids are
<kf_id>_seed_<turn_id>. match pairs box marks first and seeds only with what is left (`match.seeded_ids`,
`match.seeds_last_nn`), so a seed can lower a label (K14) but never unsettle a box pair; wheels leave seeds unmatched.

## claim (M11), walkaround/claim.py (built 10/31)

| Function | Behaviour |
|---|---|
| `async derive_claim(run_id, *, image: Path, text: str, tire_code: str | None, th, panel_id: str | None = None, claim_id: str | None = None) -> contracts.Claim` | LIVE: 16 sam_image calls plus 1 to 3 Spark calls, billed under ledger run id '<run_id>.<claim_id>' (out of the run's spend and its rebuild ledger). Plate and face checks on the unblurred working copy (long edge 2048), then claim/<claim_id>.blurred.jpg, which every later call reads; the upload is deleted once the derive ends, success or refusal (when it sits inside the run). Exits E1 to E8 and E2b (both sides shown); no label when the pickup marks prove nothing (_pickup_unread). Writes claim/<claim_id>.json and claim/<claim_id>.answers.json, and derived/claim.json only while the run is open. The same photo bytes reuse every earlier answer |
| `rebuild_claim(run_id, claim_id, *, th, write=True) -> Claim`, `rebuild_claims(run_id, *, th) -> dict[str, str]` | from the recorded answers and the pixelated photo, no call; derived/claim.json is rewritten from the claim it already holds |
| `cut_windows(windows, mask, masks, th)` | the door's window subtraction, as panels does it |
| `ClaimRefused` (ValueError), `ClaimRebuildError` | the app maps ClaimRefused to 400, 409 (no coverage yet) or 503 (the photo's checks did not complete) |

Claim.roi.box is in working-copy pixels (long edge claim.WORK_EDGE = 2048, the frame of claim/<id>.blurred.jpg), not
Claim.image.w and h; web/src/lib/claim.ts mirrors 2048. ClaimNaming.spark is the slot Spark itself named. The claim's
second look still runs on hosted Muse Spark 1.3, not Spark Open. claim/<id>.answers.json is never served.

## ruler (M11), walkaround/ruler.py (built 10/31)

| Function | Behaviour |
|---|---|
| `plate_ruler(blur, *, th, source_kf=None) -> Ruler | str` | pixels per inch = long side / (12 in x cos tilt) from the BlurRecord; refuses over 30 degrees of tilt, under 60 px, c under 0.5 |
| `wheel_ruler(...)`, `wheel_ruler_from_box(...)`, `wheel_scale(tire_code) -> (diameter, code) | str` | the wheel's major axis as the tire diameter from the typed code; no code gives "tire size not given", an unreadable one "the tire code typed could not be read" |
| `extent_from_mask(...)`, `extent_from_track(mt)`, `size_estimate(...)`, `best_size(extent, rulers, *, mark_type, panel_id, frame_w, th, crop_flag=False, drawn_box_overlap=False) -> SizeEstimate` | callers pass only the rulers adjacent to the mark's panel; a mark with no Stays Put track is refused, never sized from a Spark box |
| `size_marks(run_id, *, th, tire_code=None)`, `attach_sizes(...)`, `size_verdicts(run_id, *, th, tire_code=None) -> list[Verdict]` | Verdict.size on every mark verdict, written back to verdicts.json; deterministic, so the match stage and rebuild both call it right after match.derive_match |
| `compare_hertz`, `threshold_type_for` (WHEEL_MARK for scratch and scuff only), `tire_diameter_in`, `normalize_tire_code` | |

Refusal constants: NO_PLATE, TIRE_NOT_GIVEN, TIRE_UNREADABLE, NO_MARK_MASK, CROP_FLAG, DRAWN_BOX, PROXY_FRAME,
NO_CAMERA_DISTANCE. There is no run-level tire code yet, so walk wheel marks print "size not stated: tire size not given".

## second_look (B11), walkaround/second_look.py, and open_weights, walkaround/open_weights.py (built 10/31)

| Function | Behaviour |
|---|---|
| `make_second_look(run_id, *, th, engine=None) -> Callable[[MatchCandidate], LookVerdict | None] | None` | None when thresholds spark.second_look_engine is "off" (or missing); "spark" is hosted Muse Spark 1.3 only; "spark_open" is Muse Spark Open on Baseten with hosted Spark answering when Baseten gives none (two Baseten failures in a row send the rest of the run to hosted Spark). Each call outlines the pair on both walks' blurred crops (crops/<role>/<pickup>__<return>.look.jpg, the JPEG as sent), asks, writes the row to derived/looks.json and returns same, different or unsure, or None when the look did not run (the deterministic label stands). Only different and unsure set K7 |
| `replay_second_look(run_id, *, th=None)` | answers from looks.json, no call, no write; None when the run has no looks or th has the look off |
| `stored_looks(run_id) -> list[LookRecord]` | looks.json rows (LookRecord: the binding Phase 2 fields plus fallback, outlines, input_refs, question_sha256, records) |
| `open_weights.ask / ask_attempts / second_look / count_marks` | LIVE: OpenAI-compatible Chat Completions at https://inference.baseten.co/v1 with BASETEN_API_KEY; model id from GET /v1/models (meta-models/muse-spark-open answered every call); response_format json_schema stepping down to json_object then none on a format refusal; one ledger row of kind spark_open per HTTP request (Baseten list price, run on credits) and one open_weights.ndjson line per call with a run id. Inside pytest it refuses to build a real client unless LIVE=1. count_marks is ready and not wired |

A looks.json row whose look did not run has verdict null, engine, model, request_id and latency_ms null, valid false,
reason "not run: ...". Failed Baseten calls log reason "" and verdict null. Extra fields are allowed: looks.json
fallback, outlines, input_refs, question_sha256, records; ndjson purpose, status, model_requested, http_status,
baseten_request_id.

## eval (M17), scripts/eval_run.py and scripts/pseudo_pair.py (built 10/31)

- `eval_run.evaluate(label_paths, *, from_snapshots=False, overrides=None, th=None) -> (results, refused)`; results.json
  schema walkaround_eval_v1 (pairs, method_lines, not_computed, inputs with the sha256 of each input after redaction). A
  refused labels file means exit 1 and nothing written. `load_run`, `snapshot_run` (JSON and panel masks only,
  redacted), `same_slot`, `method_lines`, `markdown(results, refused=None)`.
- Labels: eval/labels/<pair_id>.json (walkaround_labels_v1, eval/README.md); pair_id equals the file stem; footage is
  repo-relative and never under data/eval_internal/; reference marks sit at 15 fps proxy slots, matched by kf.pts.
- `pseudo_pair`: list, cut, scenes, calibrate (refuses before_after pairs); `noise_floor`, `floor_line`, `pair_for_run`.
- GET /eval/label.html serves dist/eval/label.html (web/public/eval/label.html) as a file; it reads only owner-served run
  files and saves nothing. SheetMethod.eval_lines stays empty: results.json is not a frozen run file, so wiring it into
  the sheet would break byte-identical rebuilds.

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
| `POST /api/claims` | multipart {image, text, run_id, tire_code?, panel_id?} -> `contracts.Claim`; the photo is held in memory (50 MB cap) and written to quarantine/tmp/claim_<n>.jpg, never a served folder; 409 while the run is still working or has no coverage, 400 for a photo or panel id the claim refuses, 503 when the photo's checks did not complete, 429 on the pause or the card ceiling, 403 on a public run; one claim at a time per run, numbered after the highest claim_<n> on disk |
| `GET /api/status` | {paused: bool, pickup_left: int, pairs_left: int, queue_position: int | null, engine} |
| `GET /healthz` | {ok: true, engine, probe_ok, free_bytes} |
| `GET /`, `/run/{id}`, `/run/{id}/sheet`, `/claim`, `/kiosk/{id}`, `/verify` | the Vite build from `paths.web_dist_dir()`, same origin |
| `GET /eval/label.html` | dist/eval/label.html served as a file (the eval labeling page); 404 when the build lacks it |

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

## Integration notes, phase 2 (2026-10-31 evening)

The integrator wired the six phase 2 lanes (Stays Put, narration, claim and ruler, second look and open weights, eval,
web). Accepted deviations are in each module's section above; what the integration itself changed:

- Stage order: seal, media, files, lanes, panels, blur, marks, narration (pct 74), staysput (78), match, release,
  sheet, close. Narration runs before Stays Put so a narration seed mark gets a clip. stage_done("staysput") needs
  mark_tracks.json and staysput.role_done for every walk; the stage passes stop=past_deadline and records a gap when a
  walk is left incomplete. The narration stage chains spark.request and spark.response for each new seed answer.
- The match stage builds second_look.make_second_look(run_id, th=...) unless the run is past its spend ceiling or
  deadline (the stage stays model=False so the free match always runs), passes it to match.derive_match, chains a
  spark.request and spark.response (role "pair") for every model answer behind looks.json, then calls
  ruler.size_verdicts. thresholds spark.second_look_engine is "spark_open" (source line in config/thresholds.json).
- match.staysput_caps(mark_tracks, th): K6 reads the Stays Put rows only while staysput.cap_enabled is on (default on
  for a run frozen before the key existed, so old rebuilds stay identical); traces still name the Stays Put request ids.
  claim.judge uses the same gate. K14 "named by the renter" is a new cap (rules.py, VerdictFlags.K14, the web rules).
- Rebuild order (no call): coverage, marks, narration (the seed rows come back into marks.json), Stays Put, match with
  second_look.replay_second_look, ruler.size_verdicts, claim.rebuild_claims, the sheet. NOT_REBUILT is no longer
  printed for mark_tracks.json, narration.json or claim.json. derived/narration_pieces.json joined paths.DERIVED_FILES,
  so verify compares it; derived/looks.json stays out (a frozen model-era fact like spark.json).
- _response_ids adds staysput.request_ids and the hosted Muse Spark ids behind looks.json; Baseten's chatcmpl- ids are
  never sent to Meta's DELETE.
- privacy: served at the run root deletions.json and open_weights.ndjson; under claim/ only claim_<n>.blurred.jpg and
  claim_<n>.json (never claim_<n>.jpg or claim_<n>.answers.json); derived/looks.json, derived/narration_pieces.json,
  crops/<role>/*.look.jpg, chips/<role>/*.m4a and clips/<role>/*.mp4 were already served; seeds.json, staysput.json,
  spark.json and voice.json are not. privacy_line(..., baseten=True) adds BASETEN_CLAUSE (the same words as
  web/src/lib/privacy.ts) when a run sent a look to Baseten.
- sheet: staysput_line(mt, *, calibrated=True) prints the numbers plus STAYSPUT_UNCALIBRATED while cap_enabled is off;
  the calibrated inconclusive line says which test it stopped at (the web's calibrated branch mirrors it); the claim
  appendix prints the size line only when the claim has no paragraph (the paragraph already carries it).
- Orchestrator.create_run(..., keep=False, footage_credit=None), pipeline.run_pair_async(..., keep=False,
  footage_credit=None), and the CLI flags `walkaround run --keep --footage-credit TEXT`. A kept run is public: listed,
  served without owner rights, no claim and no top-up.
- contracts: LedgerKind.SPARK_OPEN, VerdictFlags.K14, NarrationRole.transcript; match.derive_match's second_look hint
  is Callable[[MatchCandidate], LookVerdict | None].
- thresholds (config hash changed): staysput min_drift_frames, max_clips_per_walk, window_mark_margin,
  control_blob_min_px, control_blob_max_frac, check_box_covered_min, clip_frames_max; voice chip_min_ms,
  seeds_per_walk_max and the pin_fallback source; spark.second_look_engine; orchestrator.baseten_rpm 30; claim
  dark_v_max, on_panel_frac; ruler camera_hfov_deg; matching.sigma_spark 0.117 from D1 E6 (tau_pre0 0.4 at the cap,
  tau_eye 0.8, calibrated false). matching.sigma_frame stays absent (it would print a Stays Put tau_fix of 0.507).
  spark.max_output_tokens stays 4000: spark_client's marks path reads its own constants (4000, one retry at 8000), so
  a cut answer already gets the 8000 budget lead note 3 asks for.

Still open: the claim's second look runs on hosted Spark; SheetMethod.eval_lines is empty; there is no run-level tire
code; Claim has no work_w/work_h and ClaimNaming no Spark noun, side and position (the web mirrors 2048 and offers the
bill's slots only); count mode on Spark Open is not wired.

## Match lane, round 3 (2026-10-31 night): K15 replaces gate G2

Lead decision, Oct 31: Muse Spark's view conditions (wet, dirty, glare) are model judgments, and a model may only
lower a verdict, never block a panel. What is binding from here:

- G2 is retired. `match.panel_gates` never returns it; the order is G1, G3..G10, each unchanged. The G2 row stays in
  rules.py as "crop condition (retired)" so older files that name it still print.
- The keyframe pair prefers a clean view. `match.pair_keyframes(kfs_pickup, kfs_return, *, th=None)` first narrows
  each walk with `match.prefer_clean(kfs, th)`: when one of the walk's keyframes carries a view flag and the walk has
  a clean usable keyframe (marks_status complete, no wet, dirty or glare, crop, moments and front_sign present,
  sharpness at least matching.sharpness_rel_min), only those compete; otherwise all of them, so a walk with no flag
  pairs exactly as before. Then the smallest |ln aspect ratio| as before. The choice is made in match: panels picks
  keyframes before Muse Spark has seen them.
- K15 "view flagged". A walk's view is its paired keyframe plus the keyframe each involved mark was placed in. A mark
  found in one walk only (L1, L2, and both marks of an L5 pair) gets K15 when a view of either walk is flagged, and
  verdict() lowers it to needs your eye like any K flag. `match.flags_for(..., view_flagged=False)` sets it (a
  keyword with a default, so claim.py's call is unchanged). `match.view_flags(kf)` gives wet, dirty, glare in that
  order; dark is not a view flag (it never gated).
- A pair inside tau_eye_r never gets K15. It keeps what the distance and its own caps give (L3 already there, L4 with
  K1 in the band) and carries the note: a mark in the same place in both walks was there at pickup, damage or dirt.
  It still gets the second look when it is a provisional already there.
- `Verdict.view_note: str | None` (contracts, additive) is the reason in words. One walk: "return view flagged dirty:
  could be dirt" for the view the mark was found in (wet: water, glare: a reflection), "pickup view flagged glare:
  could have hidden the mark" for the view it is missing from, joined by "; ". A pair: "return view flagged dirty:
  the mark is in the same place in both walks, so it was there at pickup, whatever it is" (L3), "...: a mark shows
  near this place in both walks" (in the band). `Verdict.reasons` stays rules.describe of the K ids.
- Sheet: `SheetVerdictLine.view_line` copies view_note and templates/sheet.html prints it in the detail column. The
  glossary says "K1 to K15", "G1, G3 to G10" and has a "view flagged" line.
- File bytes: view_note and view_line are left out of the JSON when None (a wrap serializer, as
  MarkTrack.neighbour_test does), so a run with no view flag keeps its matches.json and verdicts.json bytes; its
  sheet.json changes only through the glossary.
- Older runs: `match.why_rebuild_differs(run_id, name)` reads the stored file. derived/matches.json or
  derived/verdicts.json naming G2, or a sheet.json naming G2 in its not-comparable list or with a glossary that does
  not name K15, gives `match.RULE_CHANGE_1031`. `walkaround verify` appends it to the "rebuild differs" line, once,
  then "the same rule change". A tampered file keeps the bare line.
- Superseded wording above: "a model answer can only lower one (K4, K7, K12, K13, G2, G10)" now reads K15 for G2, and
  rules.py holds K1 to K15.
- Web, for the integrator: web/src/lib/rules.ts needs the K15 row and the new G2 text (the mirror), types.ts Verdict
  `view_note?: string | null`, and the verdict card and list should print view_note.

Replayed on pairC_core1 with the rebuild path (`walkaround rebuild pairC_core1 --out DIR`), no model call, keyframes
unchanged: 0 of 21 panels comparable before, 1 of 21 after (the windshield; gates G1 15, G3 3, G4 11, G6 4, G8 1,
G9 2, no G2). The windshield gives 10 verdicts, all needs your eye: one reflection pair in the band (0.46 units) with
the note, 6 pickup marks not seen at return and 3 return marks new or unseen, each with K15 (8 of the 9 are also K4).
The trunk lid still fails G4 (mask IoU 0.50, the twin swap). Rebuilding the rebuilt copy again gives the same bytes
for every derived file and sheet.json.

## Polish lane (2026-10-31 night): Baseten key slots, Voice speaker markers, the ruler's best view and the tire code

Second look on the Baseten hackathon key (walkaround/open_weights.py, second_look.py, config.py, .env.example):
- `open_weights.KEY_SLOTS`, tried in order: `KeySlot("hackathon", "BASETEN_HACKATHON_API_KEY",
  "meta-models/muse-spark-open-1026")`, then `KeySlot("account", "BASETEN_API_KEY", "meta-models/muse-spark-open")`,
  then the hosted Muse Spark fallback in second_look as before. The id is fixed per key; GET /v1/models is no longer
  asked (with the hackathon key it lists both ids and the plain one answers 403, LEDGER row 22). A slot with no key is
  skipped. A 401, 402, 403 or 404 rests the slot for FAILURE_TTL_S and the next slot is asked in the same call; a
  timeout, a dropped connection, a 400 not about the format and 408/409/429/5xx past three tries still raise
  OpenWeightsError at once. No key at all raises `open_weights.NO_KEY`; every slot resting raises "no Baseten key
  slot answered in the last 10 minutes (hackathon key: 403; account key: 401)". (Superseded by "Review fixes, round
  3" below: a timeout, a dropped connection or a rate limit or 5xx past its tries now moves to the next slot, and the
  message reads "no Baseten key slot can be asked now, each rests after a failure (...)".)
- Additive field `key_slot` ("hackathon" or "account", the slot name, never the key) on every open_weights.ndjson
  line, failed calls included, on every spark_open ledger row, and on derived/looks.json rows
  (`LookRecord.key_slot`, null when hosted Spark answered or the look did not run). `open_weights.key_slot_of(record)`
  gives the slot of a Spark Open record that came back in this process.
- Gone: `open_weights.list_models`, `model_candidates`, `api_key` (no caller outside the module). New:
  `HACKATHON_MODEL`, `ACCOUNT_MODEL`, `KeySlot`, `SLOT_FAILURES`, `key_slots()`, `slot_order()`, `slot_key(slot)`.
  `set_client(client, *, slots=None)` answers for every slot, or only the slots named.
- config: `_SECRET_ENV` holds BASETEN_HACKATHON_API_KEY; `redact` also replaces Baseten-shaped tokens (b10_, 8
  characters, a dot, 32 or more); an empty value in walkaround/.env or ~/meta2026/.keys sets nothing, so the empty
  `BASETEN_HACKATHON_API_KEY=` line .env.example now carries never hides a value in the second file.
- Superseded above: "model id from GET /v1/models (meta-models/muse-spark-open answered every call)" in the
  second_look row.

Voice speaker markers (walkaround/narration.py):
- `narration.SPEAKER_MARKER` matches `<|...|>`. Turns are still cut on the raw words, so pieces, est_ms, pins and the
  seed questions (seeds.json phrases, the text Spark reads) are unchanged and stored answers still match. The words
  shown leave the marker out, also when a cut falls inside one: `Piece.shown`, `NarrationTurn.text`,
  `NarrationRole.transcript` (no-turns case, `narration.shown_text`) and a found seed mark's `description`.
  SeedOutcome.phrase in narration_pieces.json keeps the raw phrase (what Spark read).
- pairC_core1: the rebuilt derived/narration.json differs from the stored one in two texts (t1_3 "front door like
  scratch near the handle", t1_4 "trunk lid two small chips by the edge"; the stored ones end in "<|speaker_" and
  start with "A|>"), and sheet.json's chips with it. narration_pieces.json, marks.json and both claims rebuild
  byte-identical.

The ruler (walkaround/ruler.py, contracts, config/thresholds.json, sheet.py, __main__.py):
- A walk mark is sized on its best view: every keyframe of the walk whose frame lies inside the mark's Stays Put clip
  (frame times from clips/<role>/staysput.json, `ClipPlan.pts_src`) is a candidate; the plate or wheel is the one
  measured on that keyframe and the mark's extent comes from its own track within 3 clip frames of it (median, at
  least 3 boxes), so ruler and mark stay on one frame. The most frontal ruler that sizes the mark wins (ties: the
  frame nearest the mark's keyframe), the plate before the wheel. With no plan (a sample copy, an older run) only the
  mark's own keyframe counts, with the track's median extent over the clip, as before. The crop flag of the mark's
  own keyframe still refuses first.
- `ruler.plate_hw_min` 0.35 (thresholds, estimate; the module default for runs frozen without it): a plate flatter
  than that is refused as "the license plate is flatter than a 12 by 6 in plate, so its size is not known"
  (`ruler.NOT_A_US_PLATE`). Without it a European or Indian strip plate (pair C's front plate reads h / w 0.06 to
  0.25) passed as a frontal 12 in plate and would have sized marks about 40% small.
- `SizeEstimate.ruler_notes: list[str]` (additive, left out of the JSON when empty): why each other ruler next to the
  mark did not size it ("the wheel: tire size not given", "the license plate is turned more than 30 degrees from the
  camera", `ruler.NO_WHEEL`, `ruler.NO_PLATE`). `sheet.size_line` prints them inside the ruler parentheses after the
  ruler used, or after the reason a size was not stated. best_size (claims) is unchanged and gives no notes.
- `RunManifest.tire_code: str | None` (additive, left out of run.json when None), normalized. `walkaround run
  --tire-code 155/80R13` reads it with ruler.normalize_tire_code before anything is spent (an unreadable code exits
  2) and stores it on a new run's run.json (integrator 10/31: `pipeline.run_pair_async(..., tire_code=)` passes it to
  `Orchestrator.create_run(..., tire_code=)`; the `__main__.stamp_tire_code` wrapper is gone; a resumed run keeps its
  own code). `ruler.run_tire_code(run_id)` reads it; size_marks and
  size_verdicts use it when no tire_code is passed (tire_code="" means none), so the match stage and rebuild need no
  change.
- sheet: a not comparable verdict (L6) gets no size line; it printed "size not stated: no ruler within 0.6 m of this
  mark", which was not the reason.
- Superseded above: "There is no run-level tire code yet" in the ruler section and the integration notes.

## Privacy lane (2026-10-31 night): blocks that grow with the box, tracks carried past their sightings, the display video re-derived

The release agent found on pairC_core1 that 5 px blocks on the 960 px display videos left a close plate readable (return
walk, 4.4 s; the privacy lane found 5.4 to 5.8 s too, where uCO3D's own grey patch is also missing), and that the pickup
walk's rear plate shows edge-on from frame 185 (12.33 s), 30 frames before its plate track starts at frame 215. What is
binding from here (walkaround/privacy.py, one keyword in video.py, config/thresholds.json):

- Block size. `privacy.block_for_box(box, *, block_px, blocks_across)` is at least block_px and big enough that the box's
  long side spans at most `privacy.box_blocks_across` (4) blocks. `pixelate(..., blocks_across=4.0)` (new keyword; the
  default comes from PRIVACY_DEFAULTS, None keeps the old fixed block) applies it per box, and boxes that touch share the
  coarsest block of their connected region. `blur_frame` does the same for boxes and masks (a mask's block from its
  detection box). Crops, keyframes, sheet frames and claim photos (blur_frame) and Stays Put clips (video._clip_frames
  calls pixelate with the default) all get it with no change in those modules.
- Margin. A box is dilated by track_box_dilation_frac of its long side (it was each side's own length) plus
  track_box_dilation_px, the rule `_dilate_mask` already used for masks, so a plate seen edge-on still gets a margin
  across its thin side.
- Track boxes. `track_boxes_at(..., carried=None)` returns every sighting within `track_box_union_s` (0.1 s) of the frame,
  so a one-frame box glitch (pairC_core1 return frame 70, a 14 x 18 px box) no longer uncovers the plate; the last box
  held for track_gap_bridge_s after a sighting, as before, so every box the old rule gave is still there; and, for up to
  `track_carry_s` (2.0 s) before each run of sightings and after it, a carried box. By default it is extrapolated from
  the run's edge (size no smaller than the edge box or the median over `track_carry_fit_s`, speed the median step
  capped at `track_carry_speed_max_frac_per_s` of the frame width per second) and grows by `track_carry_grow_per_s`
  (0.5) of its size per second carried. A gap between two runs is carried from both sides. Keyframes, Stays Put clips,
  owner frames and make_sample get this with no change.
- Display video. `privacy.render_display(proxy, dst, tracks, *, fps, frames, th)` pixelates `display_boxes`:
  track_boxes_at with `carried` from `flow_carry`, which moves each carried box frame by frame with the median
  Lucas-Kanade flow of a 10 x 10 grid around it on the unblurred proxy (gray, 480 px), checked forward and backward,
  never smaller than the edge box, growing by `track_flow_grow_per_s` (0.2); where the flow loses it, it goes on at its
  last step and grows at track_carry_grow_per_s. render_blurred_proxy (the blur stage) calls it and keeps its signature.
  `video.render_blurred_proxy(..., blocks_across=None)` is the one additive keyword in video.py, passed to pixelate.
- Thresholds (privacy, new leaves): box_blocks_across 4, track_box_union_s 0.1, track_carry_s 2.0, track_carry_fit_s 0.4,
  track_carry_grow_per_s 0.5, track_carry_speed_max_frac_per_s 1.0, track_flow_grow_per_s 0.2. `privacy.PRIVACY_DEFAULTS`
  holds the same values (`privacy_params(th)`), so a run frozen before them blurs with today's rule, never a weaker one.
- Derived file. `privacy.rederive_display(run_id, role, original, out_dir, *, th=None)` checks the original against the
  sealed sha256_server, cuts the proxy again with the run's media settings and compares it with the sealed proxy hash,
  rebuilds the walk's plate and face tracks from the stored lanes (`privacy.walk_privacy_tracks`, the same
  panels.privacy_tracks call as orchestrator._privacy_tracks), renders with render_display, deletes its unblurred proxy
  and returns the hashes with `matches_sealed`. It never writes inside the run. For a run made with this code it matches
  byte for byte on the same ffmpeg and OpenCV builds (tests/test_privacy.py builds one; the proxy cut and the x264
  encode depend on those builds too). `scripts/rederive_display.py RUN_ID ORIGINAL --role R --out DIR` wraps it.
- Rebuild and verify call no blur, so their outputs do not change. A sealed run's display videos are never rewritten
  in place.

Measured on pairC_core1 with no model call: both proxies cut again from the uCO3D originals match their sealed hashes;
the old blur code gives back both sealed display videos byte for byte (051ff573, 8c38d3b4); the new code gives pickup
f9a90802 and return 38c012b6, the same bytes on two separate runs, about 4 s a walk. The frames where a plate shows in
the source with no grey patch over it (return frame 66 at 4.4 s and frames 81 to 87 at 5.4 to 5.8 s, pickup frames 185
to 190 at 12.33 to 12.67 s) now sit under blocks of 33, 23 to 25 and 37 to 39 px (5 px before), with no character
readable on side-by-side sheets checked by eye; so does every frame in the 2 s carried around each of the 8 plate
tracks. On pairC_core1's 61 keyframes the union and the carry would add blur to parts of 2 of 97 marks (a
reflection on the front bumper, 36% of its box, and dirt on the rear bumper, 61%), none on the trunk lid.

For the release lane: make_sample.py's --plate-block and --cover hand fix can give way to copying the two files
rederive_display writes. They then match the hashes rederive_display prints, not the ones the run sealed, and the
README's sample paragraph should say that.

## Stays Put neighbours lane (2026-10-31 night): the neighbour test of docs/STAYS_PUT_ANCHORS.md, built in

Owned: walkaround/staysput.py, tests/test_staysput.py, tests/fixtures/staysput/, web/src/lib/neighbours.ts,
web/src/components/ClipTrail.tsx, StaysPutPair.tsx, StaysPutList.tsx, web/test/staysput_neighbours.test.ts. Small
additive edits in shared files: contracts.py, config/thresholds.json, sheet.py, web/src/contracts/types.ts,
web/src/lib/lines.ts (staysPutLine), web/src/styles.css.

- Planning. Each clip still starts from its mark-centred window (5.9 item 1). `neighbour_candidates(walk_marks,
  tracks, kf, *, th, src_size, proxy_size)` lists what it may also track: other marks on the same crop whose type sits
  on the paint (scratch, scuff, chip, dent, crack, dirt; never reflection or unsure), the lane parts on the keyframe's
  own proxy frame (side mirror, wheel, headlight, tail light, license plate, car window, windshield, rear window; never
  a face, the car or a panel; not when the frame edge cuts the box) and the keyframe's door handle (a box from
  `handle.point_src`). `pick_neighbours(...)` takes them nearest first: all the base window holds, then the window
  grows until the clip holds `neighbours_target` (6, the clip's own paint marks count), within
  `neighbour_window_max_panel_frac` x the panel's long side and never shrinking a mark under
  `neighbour_mark_min_clip_px`; at most `neighbours_max` (12) and 16 objects per clip. A window that grew but still
  holds fewer than `neighbours_min` (4) goes back to its base side. A group whose marks all ride as neighbours in an
  earlier clip of the same keyframe gets no clip: its marks are tested there, so each mark is tested in exactly one
  clip (fewer requests than before on a keyframe with spread-out marks).
- Plan file: `ClipPlan.neighbours: list[PlanNeighbour]` (object ids after the marks: kind, ref, box_src, box_clip) and
  `ClipPlan.neighbour_rule` (1; None on plans made before tonight). Both are in the resume key. Same SAM request, no
  extra cost.
- Analysis, pure: `neighbour_test(track_frames, object_id, neighbours, *, n_frames, prompt_frame, fps, th) ->
  contracts.NeighbourTest`. A neighbour is in the fit when SAM returned it on the prompt frame and on at least 80% of
  the clip. Per window (whole clip; +/- round(0.5 s x fps) frames): the mark's median residual against the
  least-squares affine of the neighbours' centres (prompt frame to frame f), over the median of each neighbour's own
  leave-one-out median (floored at 0.5 clip px); fixed when both ratios are under 1.5, sliding when both are over 2.5,
  else inconclusive; fewer than 4 neighbours in the fit is inconclusive; the mark itself on under
  `no_track_box_share_below` of the clip is no_track; frames whose anchors lie on one line are skipped. A mark's
  neighbours are the plan's neighbours plus the clip's other paint marks. On the two stored lanes it gives the lead's
  numbers exactly (trunk paint 0.59 and 0.84x; S550 reflection 4.77 and 3.83x; crease 0.68 and 0.35x).
- Rows: `MarkTrack.neighbour_test: NeighbourTest | None` (verdict, ratio_full, ratio_near, mark and neighbour
  medians in clip px, near_s, near_frames, used, reason, and per neighbour: object id, kind, ref, frames, used, its
  leave-one-out medians and its centres per frame for the page's markers). None on plans made before the test, and a
  model_serializer drops the key then, so older runs (pairC_core1 checked) rebuild byte-identical. not_run with the
  planned neighbour list when the clip lane gave no answer. `MarkTrack.verdict` is still the drift test's; K6 is
  untouched and staysput.cap_enabled stays false.
- Sheet: `sheet.neighbour_line(mt)` ("neighbour test: moves with its 6 neighbours, 0.6x their wobble over the clip and
  0.8x within 0.5 s; fixed"); `staysput_line` puts it before the "recorded only ... changes no label" note. The web
  mirrors it (lib/neighbours.ts neighbourLine, lines.ts staysPutLine), shows a plain caption under each clip ("Slides
  against its 6 neighbours, 4.8x their wobble (3.8x within 0.5 s). Recorded only: it changes no verdict yet.") and a
  small ring on each tracked neighbour (dashed when it was not in the fit).
- Thresholds (staysput, each with its source): neighbours_enabled, neighbours_max, neighbours_target, neighbours_min,
  neighbour_track_share_min, neighbour_fixed_below, neighbour_sliding_above, neighbour_near_half_window_s,
  neighbour_window_margin, neighbour_window_max_panel_frac, neighbour_mark_min_clip_px,
  neighbour_part_max_window_frac, neighbour_residual_floor_px, neighbour_dedupe_iou, neighbour_handle_half_w_frac,
  neighbour_handle_half_h_frac. `neighbour_window_max_growth` (a cap on the window as a multiple of its base side) is
  only in staysput.DEFAULTS, off; set it to keep faint marks sharp (see the pair C result).
- Measured live on 10/31 ($0.078; ledger run ids sp_neigh_pairC_1031 and sp_neigh_s550_1031): on the S550 the
  reflection slides in both clips (3.85 and 3.70x on the lead's clip, 5.14 and 4.01x on a square product clip cut by
  video.cut_clip at 24 fps) while the crease and the S550 badge read fixed (0.88 and 0.65x; 0.94 and 1.07x; badge 0.83
  and 0.73x, inconclusive once at 1.29 and 1.69x); two thin chrome edges were never followed by SAM. On pair C's
  pickup trunk lid the windows grew to 969 and 1181 px and SAM dropped the faint marks on 13 to 84 of 89 frames (in
  pairC_core1's 400 px windows the chip, the scratch and one dirt mark held all 89), so under the 80% rule no trunk mark
  got a ratio; with the coverage rules loosened the paint reads 0.4 to 1.0x and the chip 2.2 and 1.1x, none sliding.

## Panels lane (2026-11-01): the twin check, fast keyframe masks, the claim's look

Offline work on runs/pairC_core1's stored lanes; no model call, $0.

- Twin check (panels.twin_check, apply_twin_check, twin_cues, twin_decision, lid_groups). The lane noun of a hood or
  trunk-lid track is only the prior: on pair C the "hood" lane held the trunk lid whenever the camera was at the rear
  and the "trunk lid" lane the hood at the front. Before keyframes are picked, the lid tracks of both lanes that show
  one lid (median box IoU >= panels.anchor_absorb_iou_min on shared frames) are judged together from cues the walk
  already tracked, each +1 front, -1 rear or 0: plate (a plate box on the lid box marks the rear), mirror (near the lid
  front, far rear, in lid sizes = the box's longer side), lights and glass (headlight or tail light, windshield or rear
  window, beside the lid on clearly more frames for one lane than the other). Weights plate 2, mirror 2, lights 1,
  glass 1; a sum reaching margin_min decides. Under it the lane noun stands when the lid's tracks all come from one
  lane and no cue points the other way; otherwise twin_check.decided is false, the tracks keep their lane slot as a
  placeholder, candidate_slots lists hood and trunk lid, the panel is side_uncertain (G6) and coverage prints "front or
  rear uncertain". Then each light and glass track takes the end of the placed lid it sits beside (on >= min_frames and
  >= neighbour_share_min of its frames, neighbour_majority to 1), with no record and its lane noun when there is no such
  evidence. Only the placed lids, not lights or glass, also anchor the side vote (_RoleContext ends): with lights and
  glass as anchors too, pair C's wheels and fenders moved both ways.
- Contracts: `PanelTrack.twin_check: TwinCheck | None` (dropped from the JSON when None, so earlier tracks.json files
  round-trip byte for byte; pairC_core1 checked) and `TwinCheck {lid, votes, numbers, score, decided}`. thresholds
  panels.twin_check (one dict leaf); absent, as in every run frozen before 11/1, the check is off and those runs vote,
  derive and rebuild as before. A relabelled track keeps its lane noun, so the keyframe pass decodes its masks from its
  own lane (t.noun), never the panel's noun.
- Proof on pairC_core1 (scripts in the lane report; tests/fixtures/twin/pairC_core1_lane_boxes.json.gz holds the lane
  boxes, no pixel or mask, for tests/test_panels.py): 15 tracks move. Lids: pickup_hood_01 (frames 155 to 295) to the
  trunk lid, score -3 (plate on it in 94% of 96 plate frames, tail lights beside it); return_hood_00 (0 to 103) to the
  trunk lid, -4 (plate 91%, mirror 2.57 lid sizes away); return_trunk_lid_01 (184 to 303) to the hood, +3 (mirror 0.37,
  windshield beside it); the other four lids stay. Lights and glass: pickup_headlight_03, return_headlight_00 and _01 to
  tail lights, return_tail_light_03 and _04 to headlights, pickup_windshield_02, return_windshield_00 and _01 to the rear
  window. Side vote: pickup_wheel_02 and pickup_fender_01 leave their side-uncertain placeholders for the right rear
  slots, return_car_door_00 (frames 0 to 15, the rear and the left side) goes from right rear to left rear door; the
  side_uncertain panels go from 2 to 0. Replayed keyframes put every hood and trunk-lid crop on the right lid; G4 on the
  paired keyframes: trunk lid 0.50 to 0.92 (passes 0.7), hood 0.68 to 0.73 (passes G4, still under G8's 0.8). The old
  windshield pair compared the pickup's rear window with the return's windshield at IoU 0.81 and would have passed G4;
  the pickup windshield now has no keyframe (its two real tracks touch the frame edge), so that panel is not compared.
- Still wrong on pair C, not from this lane: the "tail light" lane's id 2 on the return walk is the orange mud flap by
  the front wheel; it stays a tail light (beside the hood on only 37% of its frames) and still gives the return
  tail-light keyframes (G4 0.71 against a real tail light). pickup_car_door_04 and _05 (frames 262 to 295, the left
  side at the end of the walk) still vote right, because the confused glass and light lanes stay noun anchors.
- Claim (claim.photo_twin_check): a vote for the hood or the trunk lid goes through the same cues on the photo (the
  plate boxes of the privacy check, the vote's own mirror, light and glass detections, one frame of evidence), and
  the decided lid is what must agree with Spark; split evidence names no lid and the paragraph says the photo does not
  show which end (E2). The record rides on the winning vote row as sam_votes[i]["twin_check"]. claim_1 of pairC_core1
  keeps E2 under the check: its photo has no plate the check found (the plate was already pixelated), no mirror, and
  both light lanes and both glass lanes on the same lamp and glass. Both claims rebuild byte-identical with the run's
  frozen thresholds.
- Fast keyframe masks (walkaround/fastmask.py, new): a port of meta_sam_parser's one_bit range decoder with the context
  of the two rows above built per row in numpy, the coder inlined, and the library's canonical re-encode replaced by a
  check that the payload is used up exactly; rows= stops early. Full sweep of pairC_core1's lanes: 10,547 masks, 455
  million pixels, 0 differences (lossless Stays Put lanes go to the library). One_bit decode about 7x faster
  (0.22 against 1.6 us per pixel). tests/test_fastmask.py checks the fixtures, the conformance corpus, encoded rasters,
  rows=, refusals and a sample of the stored lanes (skipped when runs/ is absent).
- Keyframe pass (panels._Masks, _CarTest, _KeyframeRun): one_bit masks decode in process through fastmask, the
  DecodePool only takes what it refuses (thresholds panels.fast_mask_decode, true when absent). The car masks, 80% of
  the old decode (101 and 106 per walk, 61 and 64 s CPU), are decoded only for the candidates a panel is about to keep
  and only down to the panel mask's last row; the containment test reads no pixel below it, so the answer is the whole
  decode's. Against HEAD's eager pass, with the budget out of the way and the twin check off, both keep the same 71
  keyframes (ids, frames, scores and crop sha256 equal). Up to panels.keyframe_panels_in_flight (4; 1 when absent)
  panels are scored and blurred at once; a frame two panels keep is blurred once and both keyframes carry that answer.
  tracks.json per_frame area_frac is now the mask's share before the car test except for candidates that failed it (0).
- Timings (offline replay of panels.derive on pairC_core1 with a no-call blur; M4 Max, 2 decode workers before):
  decode CPU 78.1 and 79.9 s per walk before, 4.7 and 4.8 s after; derive wall 48.9 and 44.4 s before, 9.0 and 6.4 s
  after (16.2 and 10.3 s with one panel at a time); keyframes 27 and 34 before (over the 60 s budget, top 2), 28 and 42
  after (top 3). 90 s projection: decoding scales with candidates (9 per panel, at most 21 panels) and mask sizes, not
  with frames; lane text reads are 0.03 s for 16 lanes at 304 frames (about 0.15 s at 1350). A 90 s walk at 15 fps
  should stay near 5 s of decode CPU, about 12 s in the worst case (every car mask a full 960x540 frame), under the
  60 s budget, so it keeps the top 3; the old pass would have needed about 80 to 130 s and fallen to top 2. The live
  blur (two image requests per kept frame) was most of the 3 min 13 s panels stage on 10/31; four panels in flight
  should cut it to about a quarter (not measured live).
- The claim's second look (claim._look_record): asked through second_look.look_images when second_look has it (Muse
  Spark Open on Baseten first, hosted Muse Spark 1.3 when Baseten gives no answer; proposed signature
  look_images(image_a, image_b, *, panel_id, run_id, ledger_run_id, input_refs, th, pair) -> (standing record or None,
  records, notes)), keeping hosted Muse Spark otherwise and whenever thresholds spark.second_look_engine is "off". The
  record that stands is the one kept in claim_<n>.answers.json, so rebuild replays it with no call.

## Integration notes, round 3 (2026-10-31 night)

What the integrator changed on top of the five lanes (panels, match, privacy, polish, Stays Put neighbours):

- `second_look.look_images(image_a, image_b, *, panel_id, run_id, ledger_run_id=None, input_refs, th, pair=None,
  provenance=None) -> (standing record or None, records, notes)`: the lead's signature plus one keyword. Same engine
  order and breaker as SecondLook._ask (one breaker per run id, BASETEN_FAILURES_MAX failures in a row send the rest
  to hosted Spark); "off" asks nothing. Ledger rows carry ledger_run_id (default run_id); hosted Spark's rows too.
- `open_weights.ask`, `ask_attempts` and `second_look` take `ledger_run_id` (the run id on the ledger rows) and
  `log_path` (where the provenance line goes; default runs/<run_id>/open_weights.ndjson). A line whose ledger id
  differs from run_id carries it as `ledger_run_id`; a walk's lines keep their shape byte for byte.
- A claim's look writes its provenance to `claim/<claim_id>.open_weights.ndjson` (`paths.claim_open_weights_log`), not
  to the run's open_weights.ndjson as the lead's note of 10/31 said: a claim always comes after the run is sealed, and
  the run's open_weights.ndjson is hashed into bundle/manifest_run.json whenever a look ran in the walk, so an appended
  line would make `walkaround verify` print "run manifest: hash differs: open_weights.ndjson". No folder is named after
  the claim's ledger id ("<run>.<claim>"). The claim file is not served (claim/ serves only claim_<n>.blurred.jpg and
  claim_<n>.json); the standing record (engine, model) is in claim_<n>.answers.json, so rebuild replays it with no
  call. `claim._look_record` calls look_images directly. The run's privacy line is not touched by a claim (the sheet
  is sealed first).
- Claim and the G2 decision: with thresholds `claim.view_flag_caps` true, a wet, dirty or glare pickup frame no longer
  stops a claim comparison. A claim mark the pickup walk shows nothing near gets K15 when none of the panel's complete,
  sharp pickup keyframes is clean (`claim._pickup_view_flags`); a near pair gets no K15 and the paragraph says "The
  pickup frame was flagged <flags>, but a mark shows at this spot on it, so something was there at pickup." Absent (every
  run frozen before 10/31) the old refusal stands, so those claims rebuild from the answers they recorded (a compare
  would need a marks call they never made).
- Tire code: `pipeline.run_pair_async(..., tire_code=)` and `Orchestrator.create_run(..., tire_code=)` carry it to
  run.json; `__main__.stamp_tire_code` is gone. The app's POST /api/runs still takes none.
- verify: `__main__.why_differs` also asks `narration.why_rebuild_differs`, so pairC_core1's narration.json line gets
  its reason.
- spark_client and voice_client redact through `config.redact` (their copies missed BASETEN_HACKATHON_API_KEY and the
  b10_ shape).
- thresholds: `staysput.neighbour_window_max_growth` 2.0 (the lane left it in DEFAULTS at 1000, off). Measured on the
  pair C trunk lid: windows grown to 969 and 1181 px (2.4x and 3x the 400 px base) lost the faint marks on 13 to 84 of
  89 frames, while the 400 px windows of pairC_core1 held three of them on all 89. 2x is an estimate between the two;
  the S550 case may then find fewer than 4 neighbours. `claim.view_flag_caps` true (above).
- Web: rules.ts carries K15 and the retired G2 (a test now checks its id set against rules.py); the verdict card and
  the verdict list print `view_note`; sizeLine prints `ruler_notes` as sheet.size_line does; looks.json and
  open_weights.ndjson rows read `key_slot`; the second-look line names the Baseten model id after the engine; types
  for `PanelTrack.twin_check`, `RunManifest.tire_code`, `SizeEstimate.ruler_notes`, `Verdict.view_note`.
- Allowlist: unchanged. No lane added a served file; the new claim provenance file and the unblurred proxy that
  `privacy.rederive_display` cuts outside the run are never served.

## Review fixes, round 3 (2026-11-01, early morning; review-fix lane)

Ten findings from two reviewers of round 3, each checked before it was fixed. All offline; no model call, $0.

- Twin check (panels): thresholds `panels.twin_check.margin_min` 3 (was 2). One strong cue alone (plate or mirror,
  weight 2) or the light and glass lanes alone (1 + 1, and those lanes swap themselves) no longer file a walk's lid
  against its lane noun; the lid splits (twin_check.decided false, side uncertain, not compared). Every pair C lid
  decision scores 3 or 4 and is unchanged on pairC_core1's lanes and pairC_core2's tracks. New leaf
  `photo_margin_min` 2 inside the same value: `claim.photo_twin_check` keeps margin 2 for a claim photo, where Spark's
  own panel name must still agree. Runs frozen before keep their stored tracks (verify on pairC_core2: rebuild
  identical).
- Claim photo flags (claim): new thresholds `claim.photo_flag_caps_one_walk_only` true. A wet, dirty or glare claim
  photo caps the claim mark with K11 (exit E8) only when no pickup mark lies within tau_eye_r of it. A nearer pickup
  mark keeps its label, the second look is asked as for any provisional already there, `claim.exit` stays None and
  the paragraph adds "The claim photo was flagged <flags>, but a pickup mark sits at this spot, so something was there
  at pickup." (L3) or "...; a pickup mark shows near this spot." (band). Absent (runs frozen before 11/1): K11 on
  every E8 claim mark, as before. `_Compared` gains `near` and `photo_view_flags`; `_compare(..., photo_flags=)`.
- Key slots (open_weights): a timeout, a dropped connection, or a 408, 409, 429 or 5xx past its tries (or a
  Retry-After over 30 s) on a slot rests that slot for `TRANSIENT_TTL_S` (60 s) and asks the next slot inside the same
  call when one can still be asked; on the last slot it raises as before and the slot does not rest. A 400 not about
  the format stays final. `_State.resting` holds (time it may be asked again, why). The all-resting message is now
  "no Baseten key slot can be asked now, each rests after a failure (hackathon key: 403; account key: 401)".
- Claim breaker (second_look): `_Breaker(limit, ttl_s=None)`; look_images' per-run breaker gets
  `ttl_s=open_weights.FAILURE_TTL_S`, so ten minutes after its last failure a later claim asks Spark Open again (the
  walk's breaker lives for one match stage and keeps no TTL). `second_look._clock` is the clock tests move.
- Allowlist (privacy): `claim/claim_<n>.open_weights.ndjson` is served to owner and public, like the run's own
  open_weights.ndjson (no pixels, no keys). claim_<n>.jpg and claim_<n>.answers.json stay refused.
- Claim Baseten clause (claim, privacy): `privacy.CLAIM_BASETEN_CLAUSE` ("For the second look, two blurred crops, one
  from the pickup walk and one from the claim photo, were sent to Muse Spark Open on Baseten, a company separate from
  Meta.") goes into a compared claim's paragraph before the provenance tail when its look reached Baseten
  (`claim._sent_to_baseten`: a look answer from Spark Open, or a line in claim/<claim_id>.open_weights.ndjson, for the
  claim or the earlier claim of the same photo it reuses). Older claims had hosted looks only, so their paragraphs
  rebuild unchanged. The claim page shows the paragraph, so no web change.
- Display blur (privacy): `privacy.track_carry_s` 3.0 (was 2.0, and PRIVACY_DEFAULTS with it). New leaves
  `privacy.track_flow_cover_path` true and `privacy.track_flow_cover_max_frac` 0.1 (both also in PRIVACY_DEFAULTS):
  on the display video each flow-carried box also covers every box before it back to the sighting it left, out to 0.1
  of the frame's long edge past the box. Measured on the return walk of pair C (the same original in both runs): the
  front plate track starts at frame 216, the flow carried its 21 x 14 px box 80 px right while the edge-on plate
  stayed, so frames 185 to 189 showed the plate strip outside the carried block even with a 3 s carry; with the path
  cover those frames sit under blocks (checked by eye on pairC_core1 and on pairC_core2 re-derived to a scratch
  folder). Uncapped, pairC_core1's pickup had up to 26% of a frame under carried boxes; capped, at most 14%. Crops and
  clips keep the extrapolated box plus their own image checks. Older snapshots lack the new keys and get the defaults,
  so a re-derived display video is never blurred more weakly than a new one; pairC_core2's own snapshot keeps 2.0 s.
- pairC_core1's two display videos were made again from the sealed originals with this blur and put in place by
  `scripts/rederive_display.py ... --install --reason TEXT`, which calls `privacy.install_rederived_display` and writes
  runs/pairC_core1/rederived_display.json (sealed sha256, new sha256, when, why, the privacy values; at the run root,
  never served). The sealed records (videos/<role>.json, bundle/manifest_run.json) stay as sealed; `walkaround verify`
  now prints each media line with `privacy.rederived_display_note` ("display video re-derived on 2026-11-01 from the
  sealed original (sealed sha256 ...): <reason> (rederived_display.json)", then "the same re-derivation").
- Footage credit (sheet, contracts): new section `sheet` in thresholds with `sheet.footage_credit` true.
  `SheetModel.footage_credit` and `footage_license_url` (left out of sheet.json when None, so every sheet frozen before
  rebuilds byte for byte): run.footage_credit on page 1 with the license link (`sheet.FOOTAGE_LICENSES`: CC BY 4.0,
  Pexels License), in every page footer (so the PDF) and as a line in the letter. The sample's sheet was sealed before
  the rule and keeps no credit; a run made from now on prints it.
- Evidence (README numbers): docs/evidence/events_pairC_core1.ndjson.gz (pairC_core1's event log, the same bytes as
  the old sample's at bfa0448) for its 6 min 49 s and 3 min 13 s; docs/evidence/voice_turns_pairC.json (the Voice turns
  of the D1 gate, pairC_core1 and pairC_core2 on the same narration) for the 11.2 s and 16.1 s turns.

## Stays Put ranking and ruler notes lane (2026-11-01)

Owned: walkaround/staysput.py, walkaround/ruler.py, tests/test_staysput.py, tests/test_ruler.py. Shared: one leaf in
config/thresholds.json (ruler.view_flag_sizes) and this section. No model call, $0.

The clip cap's order (lead decision 1):
- Past 16 clips a walk keeps clips in this order, each clip standing where its best item does
  (`staysput._priority`): the glare control (one clip, 5.9 item 4), then a clip holding a mark paired across the walks,
  then one holding a mark a narration seed named (a paired mark a seed also names comes first of all), then the rest;
  inside each tier by Spark's confidence, ties by clip id. With nothing paired or seeded the order is the old one.
- Stage order: marks, narration, Stays Put, match. Stays Put plans the pickup walk after the marks and narration of
  both walks, so marks.json (with the seed rows), keyframes.json, panels.json and the lane summaries of both walks are
  there; matches.json is not (the match runs later) and calibration.json is first written by the match. So the match's
  own mark ids are not available when the cap is applied.
- `staysput.cross_walk_pairs(run_id, *, th) -> list[tuple[str, str]]` (pickup mark, return mark): the closest
  cross-walk candidates in the match's own metric. On each panel with keyframes and placed marks in both walks and no
  panel-level gate failing (`staysput.PANEL_LEVEL_GATES` G1, G6, G7, G10, from match.panel_gates on the pair
  pair_keyframes picks): match.cost and match.seeds_last_nn on the whitened marks, kept when d < tau_eye_r, with the
  stored calibration or the one match.calibrate gives when there is none (nothing written). Left out on purpose: the
  view gates of a keyframe pair (G3, G4, G5, G8, G9), since the match may compare the panel on another pair
  (matching.next_keyframe_pair), the door handle's residual and the group pairs. Public match functions only.
- `staysput.seeded_marks(run_id) -> frozenset[str]`: the narration_seed rows of marks.json plus every narration.json
  turn's seed_mark_id (found or on_existing_mark).
- `RolePlans.paired`, `RolePlans.seeded` (additive, left out of clips/<role>/staysput.json when empty): the walk's
  marks that ranked first. A file that cannot be read leaves the order to confidence with a walk note in
  `RolePlans.skipped`; no clip is lost over it. Neither is in the plan key.

The neighbour reason (lead decision 3):
- `staysput.NEIGHBOUR_RULE` 2: the reason reads "1 neighbour tracked through the clip, 4 needed" ("0 neighbours", "3
  neighbours" as before). `neighbour_test(..., rule=NEIGHBOUR_RULE)` takes the plan's neighbour_rule; a plan made under
  rule 1 keeps "1 neighbours", so pairC_core2's mark_tracks.json and sheet rebuild byte for byte (`walkaround verify
  pairC_core2`: rebuild identical). neighbour_rule is in the plan key, so a walk planned under rule 1 and resumed
  under rule 2 asks its clips again.

Sizes on a flagged view (lead decision 2):
- thresholds `ruler.view_flag_sizes` true. A walk mark whose keyframe Muse Spark flagged wet, dirty or glare is sized
  like any other, and the size carries a note in K15's words, last among its `ruler_notes`: "return view flagged dirty:
  the size may include dirt", "pickup view flagged dirty and glare: the size may include dirt or a reflection"
  (`ruler.view_note(role, flags)`), for the flags of the mark's own keyframe and of the keyframe the size was measured
  on. sheet.size_line and the web's sizeLine already print ruler_notes inside the ruler parentheses, so neither changed.
- Still refused: a dark keyframe (`ruler.DARK_FRAME`, "darkness on that frame"; dark is no view flag, as in the match)
  and every geometric reason (plate turned past 30 degrees, a strip plate, no wheel measured, no tire code, no ruler
  within 0.6 m, no track). A refused size carries no view note.
- Absent leaf (every run frozen before 11/1): any flag refuses with `ruler.CROP_FLAG` as before. best_size (claims) is
  unchanged.
- Superseded above for runs with the leaf: "The crop flag of the mark's own keyframe still refuses first" (polish
  lane); ruler_notes is no longer only about the other ruler.

Replay of pairC_core2 offline (scratch copy, source frame times from the two data/ originals, whose sha256 match the
sealed ones; $0). The 10/31 order reproduces both walks' stored plans in every key field but the rule number.
- Ranked first: pickup paired pickup_trunk_lid_kf1_m1 and pickup_hood_kf1_m1, kf2_m8, kf2_m9 (the hood fails only view
  gates on its first pair); seeded pickup_trunk_lid_kf1_m1 and pickup_right_front_door_kf1_seed_t2. Return paired
  return_trunk_lid_kf1_m2 and return_hood_kf1_m2, m8, m11.
- Clips gained: pickup_trunk_lid_kf1_sp01 (the chip, alone, 400 px, no neighbour), pickup_right_front_door_kf1_sp02
  (the door seed); return_trunk_lid_kf1_sp01 (return_trunk_lid_kf1_m1 and the chip m2, 400 px, no neighbour).
- Clips lost: pickup_trunk_lid_kf1_sp02 (m2 scratch, m3 dirt) and pickup_trunk_lid_kf2_sp01 (kf2_m2 dirt), three
  needs-your-eye verdicts on the compared trunk lid (not seen at return, L1, lowered by K15), while the hood (not
  compared, G4 and G8) keeps 9 of the 16 pickup clips; return_left_front_wheel_kf1_sp01 (an unsure mark).
- Sizes: none would print. The flag no longer refuses, but every mark then meets a geometric refusal: 40 no ruler
  within 0.6 m, 37 the front strip plate (flatter than 12 by 6 in), 12 the rear plate turned past 30 degrees, 2 a
  wheel at the frame edge, 1 a wheel turned past 30 degrees, 1 no tire code. Every keyframe inside the chips' new clips
  reads the rear plate box at h / w 0.60 to 0.74 (33 to 48 degrees), so the chips would get no size either; that box
  is uCO3D's grey patch, not a 12 by 6 in plate.

## Match lane, round 4 (2026-11-01): tight pairs go to the second look; the next keyframe pair on view gates

Lead decisions, Nov 1, with the lead's reasons. Both sit behind thresholds leaves under `matching` that every run frozen
before 11/1 lacks, so those runs match, rebuild and verify as they did (pairC_core2 checked: its frozen rebuild gives back
matches.json, verdicts.json and sheet.json byte for byte). Offline, no model call, $0.

1. Tight pairs go to the second look (`matching.tight_pair_second_look` true; `match.tight_pair_look(th)`). For a pair
   inside tau_pre_r (L3), K4 (type dirt, reflection or unsure) and K12 (Spark confidence low on either mark) no longer cap
   the label by themselves. They become notes on the pair, and the pair is sent to the second look as any L3 pair is
   (engine order unchanged: Spark Open on the hackathon key with meta-models/muse-spark-open-1026, then the account key,
   then hosted Muse Spark 1.3), which may lower it to needs your eye (K7). If the second look gives no usable answer (not
   asked because of the spend ceiling, the deadline, the look budget or the look turned off, or every engine failed), K4
   and K12 cap as before. Marks seen in one walk only (L1, L2, L5) and pairs in the band (K1) keep K4 and K12 as caps.
   Reason: an already-there call rests on geometry, the same spot in both walks, not on what the mark is; the second look
   compares the two crops directly and is the veto; without it, stay conservative. No model answer raises a label above
   what the distance gives: L3 comes from d < tau_pre_r, and with a usable look only K7 or another cap can lower it.
   - `match.derive_match`: `_provisional_already_there(..., deferred=match.TYPE_CAPS)` leaves K4 and K12 out when it
     decides which pairs are asked, so they alone never keep a pair from the look. Any other cap (K2, K3, K5, K13, K14,
     and K6 once Stays Put caps) still does, as before.
   - `match.verdicts_for`: `looks` maps (pickup mark, return mark) to the answer. A key whose value is None is a look put
     to a model that gave no usable answer; a missing key is a look never asked. `match.tight_pair_flags(pair, a, b,
     flags, look, *, asked)` returns the flags verdict() takes, the note ids and the note.
   - Records: `Trace.note_ids` (the K ids that applied as notes and lowered nothing, ["K4", "K12"] in that order; left
     out of the JSON when empty) and `Verdict.type_note` (left out when None), on every tight pair carrying K4 or K12:
     what Muse Spark said with the ids ("Muse Spark typed both marks dirt (K4), with low confidence on both marks (K12)"),
     then one of "a note here, since the mark sits in the same place in both walks and the second look compared the two
     crops and said same"; "a note here; the second look compared the two crops and said different (K7)" (or "was unsure
     (K7)"); "the second look gave no usable answer, so K4 and K12 cap the pair as before"; "the second look was not
     asked, so K4 and K12 cap the pair as before"; "another cap already lowers this pair, so no second look was asked and
     K4 and K12 cap it too".
   - Rebuild: replay_second_look answers from looks.json, and a row whose look did not run replays as None with the key
     present, so a live run and its rebuild print the same note (tested).
   - Claims are not changed: claim._compare keeps K4 and K12 as caps on a near pair, so a claim's look still runs only on
     a pair already there after them. The claim owner may follow.

2. Next keyframe pair on view gates (`matching.next_keyframe_pair` true; `match.next_keyframe_pair(th)`). When a panel's
   chosen keyframe pair (pair_keyframes, unchanged) fails a view gate (`match.VIEW_GATES`: G3 sharpness, G4 mask IoU, G5
   handle residual, G8 hood IoU, G9 window subtraction) and no panel gate, match tries the panel's other keyframe pairs in
   `match.keyframe_pair_order(kfs_pickup, kfs_return, *, th)` and compares on the first that passes every gate. The order:
   the chosen pair; then clean views first (how many of the two keyframes prefer_clean would not offer: 0, 1 or 2); then
   the product of the two keyframe scores, highest first (a pair needs both views good; a sum would rank one sharp view
   with one poor view above two fair ones); then the keyframe numbers (kf<n>, pickup first), then the ids, so the order
   never depends on how keyframes.json lists them. G1, G6, G7 and G10 (`match.PANEL_GATES`) stay panel-level: when the
   chosen pair fails one, no other pair is tried; a candidate that fails one of its own (a keyframe with no front_sign,
   G6) is skipped. When no pair passes, the panel keeps the chosen pair, its gates and its IoU, as before.
   - Records: `PanelMatch.kf_pairs_tried: list[KeyframePairTry]` (new model: kf_pair, gates, mask_iou, frames_agree),
     every pair tried in order, the chosen pair first, on every panel with keyframes in both walks (one entry when the
     chosen pair passes); left out of the JSON when empty, so a run matched without the rule keeps its bytes.
     `PanelMatch.kf_pair` and `mask_iou` are the pair compared on. `frames_agree` false says the IoU was not taken because
     a keyframe whose marks are compared turned its panel axis too far from the pair's pickup keyframe
     (match._frames_agree, still reported as G4).
   - Sheet: `match.how_checked(pm)` gives the panel's line, `SheetPanelBlock.how_checked` carries it (left out when None)
     and templates/sheet.html prints "how it was checked: ..." under the frames. Examples: "compared on keyframes
     pickup_trunk_lid_kf2 with return_trunk_lid_kf1, the first pair tried"; "compared on keyframes A with C; tried first:
     A with B (G3 blurry keyframe)"; "not compared: none of the 6 keyframe pairs passed every gate; tried: ... (G4 mask
     IoU 0.69, G8 hood IoU 0.69); ... (G4 mask IoU: panel axes disagree, ...)"; "not compared: keyframes A with B (G1 not
     filmed); G1 is a panel gate, so no other pair was tried".
   - `match.why_rebuild_differs`: `_paired_before_1031` compares today's chosen pair with the first try, not with the
     pair compared on.

- Sheet: `SheetVerdictLine.type_line` copies Verdict.type_note and the detail column prints it under the view line.
  `sheet.glossary_md(th=None)` adds two lines (`sheet.GLOSSARY_TIGHT_PAIR`, `sheet.GLOSSARY_KEYFRAME_PAIR`) only when the
  run's thresholds turn the rules on; build_sheet passes the run's thresholds, so a sheet frozen before 11/1 rebuilds
  byte for byte.
- rules.py and web/src/lib/rules.ts are unchanged: no new rule id, and the rule texts are stored in verdicts.json reasons,
  so editing the K4 or K12 text would change every frozen run's bytes.
- Superseded wording above: derive_match's row ("the optional second look on already_there pairs") now reads "on
  provisional already_there pairs, with K4 and K12 left out under rule 1"; pair_keyframes' pair is the first pair tried,
  and the pair compared on is PanelMatch.kf_pair.
- Web, for the integrator: types.ts `Verdict.type_note?: string | null`, `Trace.note_ids?: string[]`,
  `PanelMatch.kf_pairs_tried?: {kf_pair: [string, string]; gates: string[]; mask_iou: number | null; frames_agree:
  boolean}[]`; VerdictList and MatchPane print type_note where they print view_note; the run page may print the
  how-checked line per panel (match.how_checked's words from kf_pairs_tried).

Replayed on runs/pairC_core2 with the rebuild path (orchestrator.rebuild into a scratch --out, the run's thresholds plus
the two leaves; no model call, outbound sockets refused):
- Panels compared: 3 of 21 before and 3 of 21 after rule 2 (trunk lid, rear bumper, right rear wheel). Three panels
  failed only view gates on their chosen pair and tried every other pair, none passing: the hood (6 pairs, G4 and G8 on
  each; IoU 0.69 and 0.66 on the two pairs with pickup kf2, the panel axes disagreeing on the other four: pickup kf3 sits
  at 26 degrees and kf1 at -12, with marks placed on both), the front bumper (2 pairs, IoU 0.48 and 0.54) and the right
  front door (2 pairs: G4 with the axes disagreeing, return kf1 holds marks 59 degrees from the pickup keyframe, and G9 on
  both, the return keyframes had the window cut out and the pickup keyframe did not). The other 15 fail a panel gate (G1
  on all 15, G6 as well on the roof and the mirrors), so no other pair was tried (7 of them have keyframes in one walk
  only, so no pair at all).
- The trunk chips (pickup_trunk_lid_kf1_m1 with return_trunk_lid_kf1_m2, d 0.078 inside tau_pre_r 0.695): L3 by
  distance; K4 and K12 wait for the look and no other cap applies (the flagged views give the pair a note, not K15), so
  they are the one pair sent to the second look. Labels before the look: already there, provisional. Same would give L3
  with note_ids K4 and K12; different or unsure L4 with K7; no usable answer L4, L3, K12, K4, as stored. Offline no look
  is asked, so the rebuild prints L4, L3, K12, K4 with "...: the second look was not asked, so K4 and K12 cap the pair as
  before".

## Integration notes, round 4 (2026-11-01)

What the integrator changed on top of the match lane and the Stays Put ranking and ruler notes lane:

- Claims follow rule 1 (`claim._compare`), under the same leaf `matching.tight_pair_second_look` read from the run's
  frozen thresholds. A claim mark within tau_pre_r of a pickup mark (near, not ambiguous) no longer has its look kept
  away by K4 or K12 alone: the provisional label leaves them out, the claim's look is asked (Spark Open first, as
  before), same keeps the claim already there with K4 and K12 as notes, different or unsure lowers it (K7); with no look
  (the pickup crop is gone) they cap as before, and any other cap (K1, K3, K5, K10, ...) still keeps the look away. The
  claim paragraph adds the note in the walk's words with the claim named ("Muse Spark typed both marks dirt (K4), with
  low confidence on both marks (K12): a note here, since the claim mark sits where the pickup mark is and the second
  look compared the two crops and said same."). `_Compared.type_note` carries it; no Claim field changed. A claim's
  unusable look answer still reads as unsure (look_from_record), so it lowers. Runs frozen before 11/1 lack the leaf:
  their claims rebuild as stored (pairC_core1, claim_smoke_pairc). Superseded: "Claims are not changed" in the match
  lane's section above. Tests: tests/test_claim_1101.py.
- `match.tight_pair_flags(..., roles=("pickup", "return"))` and `match._spark_words(..., roles)`: a claim passes
  ("pickup", "claim"); the walk's words are unchanged.
- `staysput.PANEL_LEVEL_GATES` is now `match.PANEL_GATES` (one list of panel-level gates).
- Web: types.ts `KeyframePairTry`, `PanelMatch.kf_pairs_tried?`, `Trace.note_ids?`, `Verdict.type_note?`; the verdict
  card and the verdict list print `type_note` under `view_note`; the match pane prints "How it was checked: ..." per
  panel from `lib/lines.ts keyframePairLine`, match.how_checked word for word (not in the video's moment view).
  web/dist rebuilt. rules.ts unchanged (no new id).
- Kept as the lanes left them, for the lead: the Stays Put cap ranks by the closest cross-walk candidate, not the
  match's own pairs (the match runs after Stays Put); a dark walk crop still refuses a size; the view note on a size is
  the last of its ruler_notes, so contracts, sheet.size_line and the web needed no change; claims (best_size) still
  refuse a size on a flagged claim photo.
- A server process started before this change freezes new runs' thresholds from the config it loaded, so restart
  `walkaround serve` after pulling; the CLI loads the config fresh on every run.
- `claim.derive_claim` decides on the answers as stored: after each fetch it writes claim_<n>.answers.json and reads it
  back before deciding, so floats sit at 6 places as in rebuild_claim. Found live (pairC_core3_claim claim_4): a SAM
  confidence of 49/81 at full precision put a vote score 1e-6 off the rebuild. Claims stored before keep their bytes.
- The sample is samples/pairC_core3 (run pairC_core3, Nov 1, the round 4 code): the first live run whose trunk chips
  came out already there after a second look (Muse Spark Open, same). tests/test_staysput.py reads it.

## Lead decisions, second set (2026-11-01): K5 on tight pairs; the reshaped panel

Both sit behind new thresholds leaves under `matching`, so every run frozen before them (pairC_core3, eval_nhtsa15759,
eval_px4228872_front and all older runs) matches, rebuilds and verifies as it did. Offline, no model call.

1. K5 on tight pairs (`matching.tight_pair_k5` true, read only while `tight_pair_second_look` is on;
   `match.tight_pair_deferrable(th)` gives (), ("K4", "K12") or ("K4", "K5", "K12")). Like K4 and K12 since round 4,
   K5 (the two marks typed as incompatible kinds) no longer caps a pair inside tau_pre_r by itself: it waits for the
   second look (Spark Open on the hackathon key first, as before), is a note once the look answered (same keeps L3;
   different or unsure lowers through K7) and caps as before when the look gives no usable answer. Marks in one walk
   only and pairs in the band keep it as a cap. Note that K5 adds the incompatible type cost (0.4) to d, so in a walk
   a K5 pair is inside tau_pre_r only far from the panel centre; claims add the claim's radius to tau and reach it
   more often (pairC_core3_claim claim_4: d 0.684 inside 0.754).
   - `match.tight_pair_flags(..., deferrable=TYPE_CAPS)`: the caller passes `tight_pair_deferrable(th)`.
     `derive_match` (which pairs are asked), `verdicts_for` and `claim._compare` (judge and the note) all read it.
   - Words (`match._spark_words`): "Muse Spark typed the pickup mark dirt and the return mark chip (K4), two kinds
     that disagree (K5), with low confidence on the pickup mark (K12): a note here, ..."; K5 alone: "Muse Spark typed
     the pickup mark dent and the return mark scratch, two kinds that disagree (K5): ..."; no answer: "..., so K4, K5
     and K12 cap the pair as before". `Trace.note_ids` lists them in the order K4, K5, K12. A claim names the claim
     mark, as before.
   - Sheet: `sheet.GLOSSARY_TIGHT_PAIR_K5` replaces `GLOSSARY_TIGHT_PAIR` when K5 is deferrable. rules.py K5 text
     unchanged (stored in verdicts.json reasons).

2. Reshaped panel (`matching.reshaped_panel` true; `match.reshaped_panel(th)`, `match.reshape_check`,
   `geometry.mask_difference_blob_share`; numbers in `reshape_iou_min` 0.5, `reshape_aspect_tol` 0.02,
   `reshape_open_px` 5, `reshape_blob_share_min` 0.5). Precondition: the panel is not comparable, G4 is the only
   failing gate on every keyframe pair tried (`kf_pairs_tried`, or the one pair when the next-pair rule is off) and
   the best measured whitened mask IoU (the first pair on a tie; pairs whose axes disagree have none) is in [0.5, 0.7).
   Then on that pair: (a) the two keyframes' box aspect ratios (Keyframe.aspect_ratio) agree, |a - b| / max(a, b) <=
   0.02; (b) the symmetric difference of the two whitened masks on the IoU grid, after a 5 px square opening, has its
   largest 8-connected blob holding at least half of what is left (None when the opening leaves nothing: fails).
   - Records: `PanelMatch.reshape: ReshapeCheck | None` (kf_pair, mask_iou, aspect_ratios, aspect_off, blob_share,
     called), written only where the precondition held and left out of the JSON otherwise. The panel keeps state
     not_comparable and its gates (no mark was compared).
   - Verdict: when called, `verdicts_for` emits one panel verdict needs_your_eye, rule_id "L7" (new, rules.py and
     rules.ts: "outline changed in place"), trace.rule_ids ["L7", "G4"], reasons L7 and G4 described, mark_ids every
     mark of the panel, the trace thresholds holding the measured IoU, aspect_off, blob_share and the five leaves. It
     never says already there or new and names no pair. When not called, L6 as before.
   - `match.PANEL_VERDICT_RULES` ("L6", "L7"): no size, no ruler line (ruler.attach_sizes, sheet). The sheet's
     not-comparable list leaves a called panel out (it is listed with its verdict). `match.how_checked` adds
     `match.reshape_words` ("outline changed in place (L7): keyframes A with B, IoU 0.70, box aspect ratios 0.988 and
     0.984 (0.4% apart), largest blob 64% of the mask difference"; "not called reshaped: ..." when a test failed);
     web lib/lines.ts reshapeWords mirrors it word for word, MatchPane adds one sentence to the refusal.
     `sheet.GLOSSARY_RESHAPED` prints when the leaf is on. scripts/eval_run.py counts a called panel as a
     needs-your-eye panel call (row "call": "needs_your_eye", "rule": "L7").
   - Web types: `ReshapeCheck`, `PanelMatch.reshape?`.

Replayed offline (docs/evidence/replay_1101/replay.py: each run's frozen thresholds plus the six new leaves, rebuilt
into a scratch dir, outbound sockets refused; the rebuilt matches.json and verdicts.json are beside it):
- eval_nhtsa15759: the struck left rear door is needs your eye (L7) on pickup_left_rear_door_kf3 with
  return_left_rear_door_kf1 (IoU 0.699, aspect ratios 0.988 and 0.984, 0.4% apart, largest blob 64%), its two return
  marks named. Its other G4 panels: the front bumper reaches the test and fails both (aspect 3.384 vs 9.968, 66.1%
  apart; blob 42%); right front door, roof, rear bumper and headlights fail G1 too (no test). No other verdict changed.
- pairC_core3: no verdict changed. The front bumper reaches the test and fails both (aspect 5.692 vs 2.837, blob 48%);
  the hood fails G8 too, the right front door G9 too, the rest G1 too. The trunk chips are dirt with dirt (no K5).

Live after the change (round 5, Nov 1): eval_px4228872_cut and eval_px_two_takes were run with both leaves frozen in
(no tight K5 pair and no reshape precondition arose on either). The claim on pairC_core3_claim2 (a non-kept copy of
pairC_core3 whose frozen thresholds got the six leaves) is the first claim whose second look ran live: K4 and K5
waited, Muse Spark Open said different, K7 lowered it (docs/evidence/claim_pairC_core3_claim2/). A claim copy made
this way does not pass `walkaround verify` (its stored walk files are pairC_core3's: run_id and costs in the sheet,
and with the new leaves config_hash in every trace and a reshape record on the front bumper); its claim rebuilds byte
for byte, and keep is false, so the sweeper deletes it.

## Lead decision 3 (2026-11-01): the one-walk counter-look (K16)

Behind two new thresholds leaves under `matching`, so every run frozen before them (all runs up to eval_px_two_takes)
matches, rebuilds and verifies as it did. Found on eval_px_two_takes: a parking-sensor dot Muse Spark boxed in the
return walk only came out new_or_unseen (L5), the eval's first false alarm.

- Who is asked (`match.counter_asks`): on a comparable panel in the moment frame with a keyframe pair, every mark seen
  in one walk only (an unmatched mark, L1 or L2, or either mark of a pair beyond tau_eye_r, L5) whose verdict came out
  new_or_unseen or not_seen_at_return, which means no cap applied. A mark already capped is not asked. Wheels are not
  asked (their frame is radius and angle on a rim that turns), nor a mark whose spot cannot be placed (no w, the other
  keyframe without moments, front sign or crop, or the spot's centre outside the crop).
- The spot (`match.counter_spot`): the mark's whitened position carried into the other walk's keyframe of the pair
  compared on (`PanelMatch.kf_pair`): w - residual for a return mark, w + residual for a pickup mark (the match
  subtracts the handle residual from return marks), unwhitened with that keyframe's moments and front sign. The box
  covers the whitened circle of radius tau_eye_r (at the mark's r) around it plus the mark's own box corners carried
  the same way, clipped to the crop. tau_eye_r because a mark found anywhere in it would have matched this one or put
  it in the band, needs your eye, which is what K16 does.
- `match.CounterAsk` (frozen dataclass): mark_id, role, panel_id, other_kf, spot (x1, y1, x2, y2 in the other crop's
  pixels), w (the spot in the other frame), radius_units; `.other_role`; `.key` is (pickup field, return field) of
  its looks.json row: the mark in its own walk's field, the other walk's keyframe id in the other.
- Order and cap (`match.rank_counter_asks`, `matching.counter_looks_max` 8): Muse Spark's confidence on the mark (high,
  medium, low), then panel order, then mark id; the first 8 per run are asked. derive_match computes every panel's
  verdicts first, asks, then computes them again with the answers.
- The look (`second_look.SecondLook.counter`): image 1 the mark outlined on its own keyframe crop
  (crops/<role>/<mark_id>__counter.look.jpg), image 2 the other keyframe crop with the spot outlined
  (crops/<other role>/<mark_id>__spot.look.jpg), asked with `spark_client.counter_prompt(panel_id, mark_role)` in the
  same engine order and breaker as the pair look (Spark Open on the hackathon key with meta-models/muse-spark-open-1026,
  the account key, then hosted Muse Spark 1.3). `open_weights.second_look` and `spark_client.second_look` take
  `prompt` (and Spark Open `kind`); question_sha256 hashes the counter prompt, so a resumed run reuses an answered
  counter question.
- Records: a looks.json row with `kind` "counter" and `mark_id`, pickup_mark and return_mark as in CounterAsk.key,
  outlines and input_refs in the order shown (the mark's walk first). Both fields are left out of a pair row, so pair
  rows keep their bytes. open_weights.ndjson lines carry `kind` "counter" (absent on a pair look).
- Verdict (`match.counter_flags`, `verdicts_for(..., counters, counter_skipped)`): same or unsure sets K16, so the label
  is needs_your_eye with rule ids L4, the label rule, K16 (for example ["L4", "L5", "K16"]); different keeps the label;
  no usable answer (verdict None: every engine failed or the images could not be built) keeps it; a mark the cap left
  out keeps it. `Verdict.counter_note` (left out when None) says which: "the second look compared this mark with the
  same spot in the pickup walk and said same: the pickup walk may show it (K16)"; "... and was unsure: ..."; "... and
  said different, so the label stands"; "the second look at the same spot in the pickup walk gave no usable answer, so
  the label stands"; "... was not asked: the run's 8 counter-looks went to surer marks first, so the label stands".
  With no look at all (engine off, spend ceiling, deadline) nothing is asked and no note is written.
- Rule K16 "the other walk may show it" in rules.py and web rules.ts; `VerdictFlags.K16`; ARCHITECTURE section 6.
- Rebuild: `replay_second_look` returns an object that answers pairs as before and has `.counter(ask)` answering from
  the counter rows by mark id, no call; derive_match uses `second_look.counter` when no `counter_look` is passed, so the
  orchestrator needed no change. Live and rebuild ask the same marks in the same order, so the bytes match (tested).
- Sheet: `SheetVerdictLine.counter_line` (left out when None) printed in the detail column on the same template line as
  the type line, so a sheet without it renders byte for byte; `sheet.GLOSSARY_COUNTER_LOOK` prints when the leaf is on.
- Web: types.ts `Verdict.counter_note?`, `LookRow.kind?`, `LookRow.mark_id?`; `lookFor` ties a counter row to the
  one-mark verdict, `loweredByLook(v, look)` checks K16 for a counter row; SecondLookLine says "second look at the same
  spot in the pickup walk" and links "return crop" and "pickup crop with the spot"; VerdictList and MatchPane print
  counter_note.
- Eval: `scripts/eval_run.py choose_primary` takes the labeled run, else the newest of eval_<pair> and
  eval_<pair>_<n>, else the newest run of the pair, so a re-run under a new id is the table's run.
- Tests: tests/test_counter_look.py (27), web test/counter_look.test.ts (2).

Live after the change (Nov 1, about 2:15 to 2:20 AM EST, code f24ae02): eval_px_two_takes_r6 and eval_nhtsa15759_r6
were run with both leaves frozen in. On the NHTSA re-run the crushed left front door's return mark went to the
counter-look and Muse Spark Open said different (a clean pickup door), so it stays new_or_unseen; the struck left rear
door is needs your eye (L7) live. The two-takes re-run compared no panel (the return rear bumper counted as glimpsed),
so the false alarm was asked on a non-kept copy of eval_px_two_takes (docs/evidence/counter_look_1101/): Spark Open said
different (intact paint at the pickup spot), so the sensor dot stays new_or_unseen. Both new runs rebuild identical;
pairC_core3, eval_px_two_takes and eval_nhtsa15759 (frozen without the leaves) still do.

## Lead decision 4 (2026-11-01): factory fixtures are not damage

Behind one thresholds leaf, `matching.fixtures_not_damage` (with `matching.fixture_overlap_min` 0.5 and
`matching.fixture_half_grid`), so every run frozen before it sends the old prompt, drops nothing, and rebuilds and
verifies as it did (checked on eval_px_two_takes, eval_px_two_takes_r6, eval_nhtsa15759_r6 and pairC_core3). Found on
eval_px_two_takes: the eval's only false alarm is a parking-sensor dot Muse Spark typed chip in the return walk only.

- The prompt (`spark_client.marks_prompt(panel_id, landmarks, fixtures=False)`, `spark_client.FIXTURES_SENTENCE`): with
  the leaf on, marks_v1's instruction gets one sentence after "Do not miss any mark, including small ones.": "Do not box
  factory fixtures: parking sensors, badges and emblems, door locks and handles, washer nozzles, fuel door edges, panel
  gaps and trim seams are part of the car, not damage." With it off the text is byte for byte the old one (pinned by
  sha256 in tests). `find_marks(..., fixtures=False)` passes it; `marks.derive_marks` and the claim's marks call pass
  `match.fixtures_not_damage(th)`. The schema sent is unchanged. The stored-answer replay is untouched: the reuse key
  is (crop sha256, panel) and the rebuild reads spark.json, so no call and no prompt is needed to rebuild.
- Fixture zones (`match.fixture_zones(kf, records, th)`, `match.FixtureZone`): on a keyframe crop, every fixture
  landmark of a valid marks answer (purpose marks, one input, that crop's path), read with
  `spark_client.answer_landmarks(record, marks.landmarks_for(panel, th))` exactly as find_marks reads it, so only names
  on the panel's own list; then the SAM door-handle point (`Keyframe.handle.point_src`) carried into the crop and onto
  the grid. A fixture is a name in `fixture_half_grid` (sensor, badge, handle, lock, washer_nozzle, valve, hub,
  plate_corner_tl, _tr, _bl, _br); its zone is its point plus that name's half width and half height on Spark's 0 to 1000
  grid of the crop (the SAM point uses handle). mirror_base, window corners, wheel_arch, lower_trim, fog_lamp,
  grille_corner, rim_lip and the hood's front corners are not fixtures.
- Who is dropped (`match.fixture_drops(marks, keyframes, records, th)`): a mark with a grid box whose box has at least
  `fixture_overlap_min` of its area inside a zone of its own crop (`match.fixture_overlap`; a box with no area counts as
  its centre). The zone covering the most of the box names it, the first on a tie. A narration seed (the renter named
  it) is never dropped. Empty with the leaf off.
- In the match (`derive_match`): dropped marks are taken out of each panel's marks before the gates and the match, so
  they are in no pair, no unmatched list, no group, no counter-look and no verdict. `PanelMatch.fixtures`
  (list of `FixtureDrop`: mark_id, fixture, source "spark" or "sam_handle", overlap rounded to 4 places) records them,
  left out of matches.json when empty so old files keep their bytes.
- Never silent: `match.how_checked` adds `match.fixture_words` after the keyframe-pair and reshape words, one clause per
  mark, for example "ignored: return_rear_bumper_kf1_m1 sits on a parking sensor (Muse Spark's landmark, 87% of its
  box), so it was not matched" (the percent is floor(overlap x 100 + 0.5), the same in Python and the web). A panel with
  no tries now gets a how-checked line when it has a fixture. The sheet frame still draws the mark, in the grey of a mark
  with no label. `sheet.GLOSSARY_FIXTURES` prints when the leaf is on.
- Web: types.ts `FixtureDrop` and `PanelMatch.fixtures?`; lines.ts `fixtureWords` and `keyframePairLine` add it word for
  word; web/dist rebuilt.
- No rule id: a dropped mark has no label, so nothing prints beside one. ARCHITECTURE section 6 has the paragraph.
- Limits found while building it: `spark.landmarks_max` 4 cuts the bumper's list to the four plate corners, so Spark is
  never asked for "sensor" and no run has a sensor landmark; no run records a SAM door-handle point yet; over every
  kept eval run and pairC_core3 the zone test would drop nothing (the closest real marks, NHTSA dents beside a handle
  landmark, sit 3% to 14% on it). Part (a) is a guard for when those landmarks are there; part (b) is what can reach
  the sensor dot.
- Tests: tests/test_fixtures.py (12: the leaf, the prompt's bytes on and off, the request sent, the overlap, who is
  dropped and who stays, the panel's list, the SAM handle, derive_match end to end, a frozen run, the rebuild with no
  call, how_checked with no tries, the glossary); web test/fixtures.test.ts (2). test_contracts has a FixtureDrop case;
  test_match's derive_marks fake records the flag; test_match_1101's frozen leaves include the new one.

Live after the change (Nov 1, about 3:05 to 3:08 AM EST): eval_px_two_takes_r7 ran with the leaf frozen in. The rear
bumper was compared; Spark boxed three return marks there (a scuff, a reflection, a speck), none on the first run's dot,
and flagged glare, so all three are needs_your_eye (K15) and nothing is new. No fixture landmark came back on a bumper
crop, so nothing was dropped. On the first run's crop, five marks_v1 answers with the old prompt and five with the
sentence all boxed a chip or scuff within 75 source pixels of the dot (docs/evidence/fixtures_1101/). The false alarm is
not fixed by this decision; the re-run did not reach it. eval_px_two_takes_r7 rebuilds identical.

## Lead 11/1, 2 m laps: fenders from wheels, the door's glass edge, view-matched return keyframes, the seams sentence

Four thresholds leaves, each off when a run's frozen thresholds lack it, so every run frozen before them derives,
rebuilds and verifies as it did. Found on synth_pair1, synth_pair2 and pairC_core3, measured offline by re-deriving
synth_pair2 and pairC_core3 keyframes from their own source files into a scratch copy (no call; the blur was a stub,
nothing kept). Sanat's shoot (docs/SHOOT_LIST.md) is two laps about 2 m away with the car filling the screen, which is
the case all three panel changes are for.

- `panels.door_glass_edge` (true). At a 2 m stop SAM's door mask includes the window glass, and the glass is cut by the
  top of the frame, so every frontal door frame failed the box pass (`border_frac_keyframe_max` 0.15; synth_pair2's
  stop at the left front door had box border shares 0.29 to 0.59) and only far-oblique corner views became keyframes.
  With the leaf, `panels.pass_border_frac` leaves the top edge out of a door box's border share in the box pass
  (`keyframe_candidates`, `box_pass_score`); the mask pass then measures completeness after the car windows are cut out,
  as before, and coverage's filmed test is unchanged. A door cut at a side or the bottom, and every other noun, are
  as before.
- `panels.fender_from_wheels` (box_side_d 0.35, box_above_d 0.6, reach_d 1.6, above_d 0.75, below_side_d 0.5,
  door_gap_d 0.6, min_frames 5, open_px 5; sizes in wheel diameters d, the longer side of the wheel box; estimates).
  SAM 3.1 has no noun for the panel over a wheel: on stored frames "fender", "car fender", "front fender", "quarter
  panel" and "rear quarter panel" all came back empty on the synthetic sedan (91 image requests, ledger note
  fender_noun_1101), "side panel" found the front fender in one frame but as a video lane its 3 tracks wandered over
  doors, bumper and quarters and missed the front fender stop (one lane, fender_lane_1101), and on pair C "fender"
  boxed only the arch lip (0.3% to 0.9% of the frame, under filmed_area_frac_min 0.03). With the leaf, after the side
  vote `panels.apply_wheel_fenders` drops the role's fender-lane tracks and `panels.fender_tracks` builds one fender
  track per wheel track: id `<role>_fender_from_wheel_<n>` (`is_wheel_fender`, `wheel_id_of`), no SAM ids or fragments,
  slot from the wheel's candidate slots when they agree on side and position, else the side from the decided doors
  beside the wheel (`_door_side`: one vote per shared frame where a door box is within door_gap_d x d and reaches past
  the wheel's top; at least min_frames, no tie), boxes from `fender_box` (the wheel widened by box_side_d x d, from
  box_above_d x d over it to its centre, inside the car box). The four fender panels are rebuilt from every role's
  fender tracks (votes zero). Built tracks never enter the side vote (`derive` keeps them out of `_vote_with_prior`).
  The box pass does not shut a built fender out by its border share (its box is an estimate); at each candidate
  `_built_fender_mask` decodes the car and every part of `FENDER_CUT_NOUNS` at that frame and `wheel_fender_mask`
  keeps the car mask in a band around the wheel (reach_d x d either side of its centre from above_d x d over its top
  down to its centre; below the centre only below_side_d x d beyond its sides, so the sill under the doors stays out),
  less every named part, opened by open_px, then the connected piece with the most pixels just over the wheel's top.
  The row's box, area and border then follow that mask, so the crop is cut around the fender. Everything after
  (completeness, the filmed test, crops, moments, marks, the gates) is the code every panel uses.
- `panels.view_match_return` (true). In the return walk, `panels.view_matched_first` measures each scored candidate
  against the panel's pickup keyframes the way G4 does (`geometry.whitened_mask_iou`, the candidate's moments, the
  panel's front_sign), only with the same window cut (G9) and sharpness at or over matching.sharpness_rel_min (G3);
  when the best of all candidates beats the best of the first keyframe_top_k, it goes first. Keyframe selection then
  runs as before (gap, car test, blur). Whitening removes any affine difference, so this ranks perspective and mask
  definition, not size or aspect. The gate thresholds are unchanged.
- `matching.seams_not_marks` (false). `spark_client.SEAMS_SENTENCE` after the fixtures sentence: "Panel seams, body
  lines, the edges where paint meets glass or trim, and shading or reflections on smooth or dark glossy paint are not
  marks: box a spot only where the surface itself is scraped, broken or bent, and keep the box tight on that spot."
  `marks_prompt(..., seams)`, `find_marks(..., seams)`, marks.py and claim.py pass `match.seams_not_marks(th)`. Tested
  before adopting (docs/evidence/seams_1101/: prompt_ab.py, prompt_ab.json; 200 Spark calls, $0.92): every stored
  synth_pair2 crop, 2 answers per prompt, scored against the render's ground truth. False boxes 83 to 74, boxes on a
  ground-truth mark 5 to 3. Not adopted; the leaf stays false.

Offline (keyframes re-derived from the stored lanes and the source files; gates G3, G4, G6, G9 on the best keyframe
pair; filmed slots per walk):

| | synth_pair2 before | after | pairC_core3 before | after |
|---|---|---|---|---|
| filmed slots, pickup / return | 8 / 8 | 12 / 12 | 8 / 11 | 12 / 14 |
| fender slots filmed in both walks | 0 of 4 | 4 of 4 | 0 of 4 | 1 of 4 |
| panels passing those gates on some pair | 0 | 5 | 3 | 6 |
| left / right front door best whitened IoU | 0.63 (G4, G9) / 0.74 (G3) | 0.96 / 0.99 | - / 0.66 (G4, G9) | - / 0.81 |

Tests: tests/test_walk_2m.py (9). test_match's strict find_marks fake takes seams.

Live after the change (Nov 1, about 4:58 to 5:10 AM EST, LEDGER rows 58 and 59), both runs rebuild identical:

| | synth_pair2 | synth_pair2_r8 | pairC_core3 | pairC_r8 |
|---|---|---|---|---|
| filmed slots, pickup / return | 8 / 8 | 12 / 12 | 8 / 11 | 12 / 14 |
| fender slots filmed in both walks | 0 of 4 | 4 of 4 | 0 of 4 | 2 of 4 |
| panels compared | 3 | 9 | 3 | 5 |
| marks | 39 | 49 | 94 | 121 |
| cost | $4.93 | $5.23 | $3.31 | $3.35 |

synth_pair2_r8 scored with synthetic/score.py (synthetic/SCORE_pair2_r8.md; geometric only, the boxes were not
visually reviewed this round): 6 of 9 ground-truth panels compared (was 0 of 9), and all four fenders and quarters;
nearly every planted mark is now shown on its own panel's keyframe at a usable size (the 20 cm scratch at 467 px, the
dent at 323 px), but Muse Spark boxed only the 5 cm fender scratch, labelled already_there as it should be: own-panel
recall 1 of 14 eligible object-walks (was 1 of 6), 1 of 17 of all present (unchanged), new damage 0 of 3. False marks
46 (was 36). Two return marks on the left front door are new_or_unseen; both hit no ground-truth object (one is about
137 source px from the real new 10 cm scratch, the other near the sticker). More panels compared means more chances
for a false new; Spark's localization is now the limit, not coverage.
