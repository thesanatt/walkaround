// TypeScript mirror of walkaround/contracts.py for the records the browser reads.
// Field names are the Python names; enums are string unions. canonical_json writes every field, defaults included,
// so fields are required here unless the Python type allows None. Nothing is validated at runtime.

export type Role = string; // pickup | return | topup_<n> | claim_<n>
export type WalkRole = 'pickup' | 'return';

export type Box4 = [number, number, number, number];
export type Point2 = [number, number];

export type RunStatus =
  | 'created'
  | 'sealing'
  | 'sealed'
  | 'converting'
  | 'tracking'
  | 'blurring'
  | 'marking'
  | 'staysput'
  | 'matching'
  | 'sheet'
  | 'done'
  | 'interrupted'
  | 'failed'
  | 'partial';

export type RunKind = 'pair' | 'pickup_check' | 'topup' | 'claim';
export type LaneStatus = 'completed' | 'zero_matches' | 'incomplete' | 'failed';
export type CoverageState = 'filmed' | 'glimpsed' | 'not_filmed' | 'not_run' | 'not_seen';
export type VerdictLabel = 'already_there' | 'needs_your_eye' | 'new_or_unseen' | 'not_seen_at_return' | 'not_comparable';
export type MarkKind = 'scratch' | 'scuff' | 'chip' | 'dent' | 'crack' | 'dirt' | 'reflection' | 'unsure';
export type Side = 'left' | 'right' | 'none';
export type Position = 'front' | 'rear' | 'none';
export type ReplayMode = 'fixed' | 'recorded' | 'instant';
export type ClaimExit = 'E1' | 'E2' | 'E3' | 'E4' | 'E5' | 'E6' | 'E7' | 'E8';

export type PanelId =
  | 'left_front_door'
  | 'left_rear_door'
  | 'right_front_door'
  | 'right_rear_door'
  | 'left_front_fender'
  | 'left_rear_fender'
  | 'right_front_fender'
  | 'right_rear_fender'
  | 'hood'
  | 'roof'
  | 'trunk_lid'
  | 'front_bumper'
  | 'rear_bumper'
  | 'left_front_wheel'
  | 'left_rear_wheel'
  | 'right_front_wheel'
  | 'right_rear_wheel'
  | 'headlights'
  | 'tail_lights'
  | 'mirrors'
  | 'windshield'
  | 'rear_window';

// contracts.EventKind, in declaration order. The reducer switches on these exact strings.
export const EVENT_KINDS = [
  'run.created',
  'upload.started',
  'upload.done',
  'seal.manifest',
  'seal.ots',
  'seal.tsr',
  'media.probed',
  'media.proxy_done',
  'files.uploaded',
  'sam.request',
  'sam.first_delta',
  'sam.frame',
  'sam.lane_done',
  'sam.failed',
  'blur.done',
  'panels.derived',
  'coverage.updated',
  'topup.requested',
  'spark.request',
  'spark.response',
  'marks.derived',
  'voice.done',
  'narration.derived',
  'staysput.derived',
  'match.derived',
  'claim.derived',
  'sheet.written',
  'files.deleted',
  'responses.deleted',
  'quarantine.deleted',
  'ledger.row',
  'stage',
  'run.partial',
  'run.completed',
  'run.failed',
] as const;

export type EventKind = (typeof EVENT_KINDS)[number];

// The SSE stream ends after either of these (orchestrator.TERMINAL_EVENTS).
export const TERMINAL_KINDS: ReadonlySet<string> = new Set(['run.completed', 'run.failed']);

export interface Event {
  seq: number;
  ts: string;
  run_id: string;
  kind: EventKind;
  payload: Record<string, unknown>;
  prev: string;
  hash: string;
}

export interface ThreeClocks {
  sealed_at: string | null;
  first_model_call_at: string | null;
  sealed_minutes_after_camera: number | 'camera time not present' | null;
  camera_time: string | null;
}

export interface RunManifest {
  run_id: string;
  created_at: string;
  mode: 'car' | 'apartment';
  kind: RunKind;
  status: RunStatus;
  live_or_replay: 'live' | 'replay';
  keep: boolean;
  keep_plate: boolean;
  footage_credit?: string | null; // a sample run's footage credit, shown under the walk player
  tire_code?: string | null; // walkaround run --tire-code, normalized; the wheel ruler reads it (absent when none)
  app_commit: string | null;
  config_hash: string | null;
  thresholds: Record<string, unknown>;
  fps: number | null;
  noun_list: string[];
  prompts?: Record<string, string>; // test clips only: the text a lane was sent with instead of its noun
  retention_deadline: string | null;
  api_spend_cents: number;
  spend_ceiling_cents: number;
  clocks: ThreeClocks;
  responses_store: 'false_accepted' | 'deleted_per_request' | 'unknown';
  sam_engine: string;
  events_head_hash: string | null;
  counts: { panels: number; marks: number; matches: number };
  errors: { stage: string; text: string }[];
}

export interface FfprobeInfo {
  codec: string;
  container: string;
  width: number;
  height: number;
  rotation: number;
  display_width: number;
  display_height: number;
  fps_avg: number;
  vfr: boolean;
  duration_s: number;
  creation_time: string; // ISO timestamp or "not present"
  device: string | null;
  hdr: boolean;
  has_audio: boolean;
}

export interface ProxyRecord {
  path: string;
  fps: number;
  width: number;
  height: number;
  frames: number;
  pts_map_path: string;
  sha256: string | null;
}

export interface FilesApiRecord {
  file_id: string;
  uploaded_at: string;
  expires_at: string | null;
  deleted_at: string | null;
  delete_response: Record<string, unknown> | null;
}

export interface VideoRecord {
  role: Role;
  original_filename: string;
  bytes: number;
  sha256_browser: string | null;
  sha256_server: string | null;
  hash_match: boolean | null;
  received_at: string | null;
  chunks_expected: number;
  chunks_received: number;
  ffprobe: FfprobeInfo | null;
  proxy: ProxyRecord | null;
  proxy_blurred: { path: string; sha256: string | null } | null;
  wav: { path: string; duration_s: number; sha256: string | null } | null;
  files_api: FilesApiRecord | null;
  quarantine_path: string | null;
  quarantine_deleted_at: string | null;
}

export interface TrackFragment {
  object_id: string;
  first_frame: number;
  last_frame: number;
}

export interface MergeRecord {
  rule: 'velocity_gap' | 'anchor_absorb' | 'moment_orientation';
  gap_s: number;
  pred_dist: number | null;
  area_ratio: number | null;
  angle: number | null;
}

export interface AnchorReprompt {
  request_id: string;
  frame: number;
  box: number[];
  negative_points: Point2[];
}

export interface FrameObs {
  frame: number;
  pts: number;
  box: Box4;
  box_area_frac: number;
  in_car_box: boolean;
  c: number | null;
  border_frac: number;
  mask_decoded: boolean;
  area_frac: number | null;
  complete: boolean | null;
}

export interface PanelTrack {
  track_id: string;
  role: Role;
  noun: string;
  sam_object_ids: string[];
  fragments: TrackFragment[];
  merges: MergeRecord[];
  anchor_reprompts: AnchorReprompt[];
  c_present_share: number;
  per_frame: FrameObs[];
  panel_id: PanelId | null;
  side: Side;
  position: Position;
  twin_check?: TwinCheck | null; // lid, light and glass tracks the twin check placed (absent on older runs)
}

// The twin check (walkaround/panels.py twin_check): which end of the car a hood or trunk-lid track's lid is on, from
// the plate, mirror, lights and glass cues (+1 front, -1 rear, 0 silent); decided false means front or rear uncertain.
export interface TwinCheck {
  lid: string[];
  votes: Record<string, number>;
  numbers: Record<string, number>;
  score: number;
  decided: boolean;
}

export interface PanelRole {
  track_ids: string[];
  windows: [number, number][];
  seconds_visible: number;
  state: CoverageState;
  state_basis: 'c' | 'area_only';
  reason: string;
  keyframe_ids: string[];
}

export interface Panel {
  panel_id: PanelId;
  noun: string;
  side: Side;
  position: Position;
  instance_from: 'tracks' | 'canonical';
  votes: { R1: number; R2: number; R3: number; R4: number };
  side_uncertain: boolean;
  front_sign: 1 | -1 | null;
  per_role: Record<Role, PanelRole>;
  comparable: boolean;
  not_comparable_reasons: string[];
}

export interface KeyframeScore {
  area: number;
  completeness: number;
  sharpness: number;
  glare: number;
  frontality: number;
  total: number;
}

// Crop pixel (u, v) maps to source (dx + u / scale, dy + v / scale).
export interface Crop {
  path: string;
  sha256: string | null;
  w: number;
  h: number;
  scale: number;
  dx: number;
  dy: number;
}

export interface Moments {
  cx: number;
  cy: number;
  mu20: number;
  mu02: number;
  mu11: number;
  theta: number;
  sigma_inv_half: [[number, number], [number, number]];
}

export interface BlurRecord {
  plate_boxes: Box4[];
  face_boxes: Box4[];
  request_ids: string[];
  plate_ruler_px: number | null;
  plate_ruler_c: number | null;
  plate_tilt_hw: number | null;
}

export interface Conditions {
  wet: boolean;
  dirty: boolean;
  glare: boolean;
  dark: boolean;
}

export interface Keyframe {
  kf_id: string;
  role: Role;
  panel_id: PanelId;
  track_id: string;
  f_proxy: number;
  pts: number;
  score: KeyframeScore;
  aspect_ratio: number;
  box_src: Box4;
  crop: Crop | null;
  mask_png: string | null;
  moments: Moments | null;
  front_sign: 1 | -1 | null;
  window_subtracted: boolean;
  blur: BlurRecord;
  handle: { point_src: Point2 | null; c: number | null; request_id: string | null };
  conditions: Conditions;
  marks_status: 'complete' | 'truncated' | 'not_run';
}

export interface GridPoint {
  x: number;
  y: number;
}

export interface GridBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface Mark {
  mark_id: string;
  role: Role;
  panel_id: PanelId;
  kf_id: string;
  source: 'spark_box' | 'spark_count' | 'narration_seed';
  type: MarkKind;
  confidence: 'low' | 'medium' | 'high';
  size_class: 'S' | 'M' | 'L';
  description: string;
  grid_point: GridPoint | null;
  grid_bbox: GridBox | null;
  bbox_crop: Box4 | null;
  point_crop: Point2 | null;
  bbox_src: Box4 | null;
  point_src: Point2 | null;
  w: Point2 | null;
  r: number | null;
  size_units: number | null;
  size_px_src: number | null;
  single_frame: boolean;
  cluster_id: string | null;
  merged_from: string[];
  spark_request_id: string | null;
  caps: string[];
  // Box fix 11/1 (thresholds dedup.sam_box_refine): which box the mark uses; bbox_crop above is that box. Absent on
  // runs without the leaf.
  box_refine?: BoxRefine;
}

export interface BoxRefine {
  used: 'sam' | 'spark';
  spark_bbox_crop: Box4;
  spark_point_crop: Point2 | null;
  sam_bbox_crop: Box4 | null;
  sam_c: number | null;
  mask_px: number | null;
  inside: number | null;
  area_ratio: number | null;
  sam_request_id: string | null;
  note: string;
}

export interface MatchCandidate {
  pickup_mark: string;
  return_mark: string;
  d: number;
  d_pos: number;
  type_cost: number;
  size_cost: number;
  r_mean: number;
  tau_pre_r: number;
  tau_eye_r: number;
  ratio: number | null;
  margin: number | null;
  mutual: boolean;
}

export interface MarkGroup {
  type: MarkKind;
  n_pickup: number;
  n_return: number;
  text: string;
}

// One keyframe pair the match tried on a panel (thresholds matching.next_keyframe_pair, 11/1): the gates it failed
// (empty when it passed every gate) and its whitened mask IoU; frames_agree false means the IoU was not taken because
// a keyframe whose marks are compared turned its panel axis too far.
export interface KeyframePairTry {
  kf_pair: [string, string];
  gates: string[];
  mask_iou: number | null;
  frames_agree: boolean;
}

// The reshaped-panel test (thresholds matching.reshaped_panel, lead 11/1, second): recorded where G4 alone failed on every
// keyframe pair tried with the best IoU in [0.5, 0.7); called means the panel is needs your eye (L7).
export interface ReshapeCheck {
  kf_pair: [string, string];
  mask_iou: number;
  aspect_ratios: [number, number];
  aspect_off: number;
  blob_share: number | null;
  called: boolean;
}

// A mark left out of matching as a factory fixture (thresholds matching.fixtures_not_damage, lead decision 4 of 11/1):
// its box sits on a fixture landmark Muse Spark pointed at in the same crop, or on the SAM door-handle point.
export interface FixtureDrop {
  mark_id: string;
  fixture: string;
  source: 'spark' | 'sam_handle';
  overlap: number; // share of the mark's box inside the fixture's box
}

export interface PanelMatch {
  panel_id: PanelId;
  state: 'comparable' | 'not_comparable';
  reasons: string[];
  kf_pair: [string, string] | null;
  aspect_diff: number | null;
  mask_iou: number | null;
  handle_residual: Point2 | null;
  frame_used: 'moment' | 'bumper_arc' | 'hood_axis' | 'wheel';
  pairs: MatchCandidate[];
  unmatched_pickup: string[];
  unmatched_return: string[];
  groups: MarkGroup[];
  kf_pairs_tried?: KeyframePairTry[]; // every pair tried in order, the chosen pair first; absent in runs matched before 11/1
  reshape?: ReshapeCheck | null; // absent unless the reshaped-panel precondition held
  fixtures?: FixtureDrop[]; // absent unless a mark was left out as a factory fixture
}

export interface Trace {
  rule_ids: string[];
  thresholds: Record<string, unknown>;
  sam_request_ids: string[];
  spark_request_ids: string[];
  event_seqs: number[];
  config_hash: string | null;
  note_ids?: string[]; // K ids that applied as notes and lowered nothing (K4 and K12 on a tight pair the look answered, 11/1)
}

export interface SizeEstimate {
  inches: number | null;
  bound_lo: number | null;
  bound_hi: number | null;
  ruler: 'plate' | 'wheel' | null;
  ruler_source_kf: string | null;
  tire_code: string | null;
  threshold_type: 'body_scratch' | 'dent' | 'wheel_mark' | 'body_scuff' | null;
  refused_reason: string | null;
  ruler_notes?: string[]; // why each other ruler next to the mark did not size it (left out of the file when empty)
}

export interface Verdict {
  verdict_id: string;
  panel_id: PanelId;
  label: VerdictLabel;
  rule_id: string;
  mark_ids: string[];
  distance_units: number | null;
  distance_cm_est: number | null;
  reasons: string[];
  trace: Trace;
  size: SizeEstimate | null;
  view_note?: string | null; // a flagged view in words (K15, or the note on a mark seen in both walks); absent when none
  type_note?: string | null; // K4 and K12 on a pair inside tau_pre in words and what the second look made of them (11/1)
  counter_note?: string | null; // a mark seen in one walk only: what the counter-look at the other walk's same spot said (K16, 11/1; K17 on the two-question look)
}

export interface PanelCoverage {
  state: CoverageState;
  state_basis: 'c' | 'area_only';
  seconds: number;
  best_kf: string | null;
  reason: string;
}

export interface Coverage {
  per_role: Record<Role, Partial<Record<PanelId, PanelCoverage>>>;
  missing: PanelId[];
  topup_text: string;
  nouns_to_rerun: string[];
  rerun_seconds: number | null;
  odometer: { asked: boolean; filmed: boolean; frame_path: string | null };
}

export interface NarrationTurn {
  turn_id: string;
  start_ms: number;
  end_ms: number;
  text: string;
  keywords_hit: string[];
  pinned_panel: PanelId | null;
  pin_seconds: number | null;
  pin_kind: 'overlap' | 'about' | null;
  seed_mark_id: string | null;
  chip_path: string | null;
}

export interface Narration {
  per_role: Record<Role, { turns: NarrationTurn[]; transcript?: string | null }>;
  mode_applied: boolean;
}

export interface TrackPoint {
  f: number;
  box_clip: Box4;
  c: number | null;
  inside_panel: boolean;
  w: Point2 | null;
  u_rel: number | null;
}

export interface ClipRecord {
  path: string;
  sha256: string | null;
  f0_src: number;
  frames: number;
  window_px: number;
  windows_src: Box4[];
  w: number;
  h: number;
}

// One rigid neighbour SAM 3.1 tracked in the mark's clip (contracts.StaysPutNeighbour): a lane part or another mark.
// centres are [f, x, y] in clip pixels.
export interface StaysPutNeighbour {
  object_id: string;
  kind: string;
  ref: string;
  frames: number;
  used: boolean;
  residual_full_px: number | null;
  residual_near_px: number | null;
  centres: [number, number, number][];
}

// Stays Put by neighbours (contracts.NeighbourTest, docs/STAYS_PUT_ANCHORS.md): the mark's residual against an affine
// fit to its neighbours over theirs, over the whole clip and within near_s of the prompt frame.
export interface NeighbourTest {
  verdict: 'fixed' | 'sliding' | 'inconclusive' | 'no_track' | 'not_run';
  ratio_full: number | null;
  ratio_near: number | null;
  mark_full_px: number | null;
  mark_near_px: number | null;
  neighbours_full_px: number | null;
  neighbours_near_px: number | null;
  near_s: number;
  near_frames: number;
  used: number;
  reason: string | null;
  neighbours: StaysPutNeighbour[];
}

export interface MarkTrack {
  mark_id: string;
  role: Role;
  clip: ClipRecord | null;
  request_id: string | null;
  prompt_text: string;
  track: TrackPoint[];
  b1: number | null;
  b2: number | null;
  b_pass: boolean;
  panned: boolean;
  drift_rms: number | null;
  sign_agreement: number | null;
  pairs_n: number;
  verdict: 'fixed' | 'sliding' | 'inconclusive' | 'no_track' | 'not_run';
  curvature_cap: boolean;
  highlight_cap: boolean;
  is_control: boolean;
  neighbour_test?: NeighbourTest | null; // absent on clips planned before the neighbour test existed
}

export type LookVerdict = 'same' | 'different' | 'unsure';

// One derived/looks.json row as the page reads it (walkaround/second_look.py LookRecord, CONTRACTS.md "Phase 2 file
// shapes", fields trimmed to what is printed). engine "baseten" is Muse Spark Open; "meta-model-api" is hosted Muse
// Spark 1.3, which answered after Baseten gave none when fallback_from is "baseten" (the free-text fallback says why).
// A row with verdict null is a look that did not run; the reader drops it, since it changes no label.
export interface LookRow {
  pickup_mark: string;
  return_mark: string;
  panel_id: PanelId | null;
  verdict: LookVerdict;
  reason: string;
  engine: string | null;
  model: string | null;
  key_slot?: 'hackathon' | 'account' | null; // which Baseten key answered (never the key); null for hosted Spark
  fallback_from: 'baseten' | null;
  valid: boolean | null; // false when the answer failed look_v1 (the verdict is then unsure)
  fallback: string | null; // why hosted Spark answered, shown only as a tooltip
  request_id: string | null; // the row's own, else the last of its records
  inputs: string[]; // the two outlined crops as sent, run-relative paths
  kind?: 'counter' | 'counter2' | null; // the one-walk counter-look (K16): the mark's own walk first, then the other walk with the spot outlined; counter2 asks two questions (lead decision 5 of 11/1)
  mark_id?: string | null; // on a counter-look, the mark seen in one walk only
  mark_seen?: 'yes' | 'no' | 'unsure' | null; // counter2 only: is there a damage mark in the box on the mark's own crop (part 1); verdict is part 2
}

// One open_weights.ndjson line (walkaround/open_weights.py): one per Muse Spark Open request on Baseten, failed ones
// included, kept as provenance. Only the fields the page counts are read.
export interface OpenWeightsRow {
  request_id: string | null;
  model: string | null;
  key_slot?: string | null; // "hackathon" (meta-models/muse-spark-open-1026) or "account"
  purpose: string | null;
  verdict: LookVerdict | null;
  status: string | null; // completed, schema_miss, truncated, incomplete or failed
  valid: boolean | null;
  error: string | null; // redacted by open_weights.py
  ts: string | null;
}

export interface DeletionRecord {
  target: string;
  kind: 'file' | 'files_api' | 'response' | 'quarantine';
  sha256: string | null;
  bytes: number | null;
  deleted_at: string;
  reason: string;
  api_response: Record<string, unknown> | null;
}

export interface ReplayPayload {
  run_id: string;
  mode: ReplayMode;
  recorded_on: string;
  delays_ms: Record<string, number>;
  events: Event[];
}

// GET /api/status (app.py).
export interface StatusResponse {
  paused: boolean;
  pickup_left: number;
  pairs_left: number;
  queue_position: number | null;
  queue_eta_minutes?: number | null; // app.py adds it beside the contract's fields
  engine: string;
}

// GET /api/samples (app.py): every finished check this server replays, in samples/index.json order; the first is the
// one GET /api/sample returns.
export interface SampleRow {
  run_id: string;
  title: string | null;
  kind: string;
  recorded_on: string;
  claims: number;
  first: boolean;
}

export interface SamplesResponse {
  samples: SampleRow[];
}

// GET /api/sample (app.py): the precomputed demo pair, replayed, never live.
export interface SampleResponse {
  run_id: string;
  precomputed: boolean;
  recorded_on: string;
  badge: string;
  replay: string;
}

export interface Claim {
  claim_id?: string | null; // "claim_<n>": its photo is served only as claim/<claim_id>.blurred.jpg
  image: { sha256: string; received_at: string; w: number; h: number; low_res: boolean };
  text: string;
  scanner_style: boolean;
  roi: { box: Box4; source: 'drawn' | 'whole' } | null;
  naming: { spark: PanelId | null; sam_votes: Record<string, unknown>[]; agreed: boolean };
  panel_id: PanelId | null;
  side_source: 'text' | 'spark' | 'both_shown' | null;
  coverage_state: CoverageState | null;
  placement: { method: 'full_panel' | 'landmark_fit' | 'none'; w_claim: Point2 | null; r_claim: number | null };
  match: PanelMatch | null;
  size: SizeEstimate | null;
  exit: ClaimExit | null;
  paragraph: string;
  note_prefill: { fields: Record<string, string>; verified_against_typed: boolean } | null;
}
