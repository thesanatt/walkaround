import type {
  Claim,
  Coverage,
  DeletionRecord,
  Event,
  FfprobeInfo,
  Keyframe,
  LookRow,
  Mark,
  MarkTrack,
  Narration,
  OpenWeightsRow,
  Panel,
  PanelMatch,
  PanelTrack,
  ProxyRecord,
  ReplayMode,
  Role,
  RunManifest,
  Verdict,
  VideoRecord,
} from '../contracts/types.ts';

// A lane's life in the UI: requested -> streaming -> one of the four contract statuses.
export type LaneUiStatus = 'requested' | 'streaming' | 'completed' | 'zero_matches' | 'incomplete' | 'failed';

// Keyed by lane id (contracts.noun_slug: 'car_door'), the spelling sam.* events carry.
export interface LaneState {
  status: LaneUiStatus;
  frames: number;
  ids: string[];
  first_delta_ms: number | null;
  dollars: number;
  lane_path: string | null;
  sha256: string | null;
  request_id: string | null;
  frames_sent: number | null;
  video_frames_processed: number | null;
  error: string | null;
}

export interface FrameObject {
  id: string;
  box: number[];
  c: number | null;
}

export interface LastFrame {
  frame: number;
  objects: FrameObject[];
}

export interface SealState {
  manifest_sha256: string | null;
  ots: string | null; // pending | confirmed:<block> | failed:<reason>
  tsr_time: string | null;
  tsa_issuer: string | null;
  files_uploaded: boolean; // a files.uploaded event arrived: run.clocks.first_model_call_at may be shown
}

export interface UploadDone {
  bytes: number;
  sha256: string;
  filename: string;
  warning: string | null;
}

// What the events say about one walk's media, so the page works before (or without) videos/<role>.json.
export interface RoleMedia {
  upload: UploadDone | null; // upload.done
  ffprobe: FfprobeInfo | null; // media.probed
  proxy: ProxyRecord | null; // media.proxy_done: fps, size and frames only; the unblurred file is never fetched
  blurredPath: string | null; // blur.done: the only video the browser plays
  fileId: string | null; // files.uploaded
}

export interface VoiceState {
  mode_applied: boolean | null;
  turns: number | null;
  text: string | null;
}

export interface Ledger {
  dollarsSoFar: number;
  inFlight: number;
  p50FirstDeltaMs: number | null;
  parseCpuMs: number;
  firstDeltas: number[];
}

export interface RunState {
  run: RunManifest | null;
  videos: Record<Role, VideoRecord>;
  media: Record<Role, RoleMedia>;
  lanes: Record<Role, Record<string, LaneState>>;
  frames: Record<Role, Record<string, LastFrame>>; // the last sam.frame per lane, boxes only
  seal: SealState;
  tracks: PanelTrack[];
  offCar: string[]; // GET /api/runs/{id}/off_car: panel tracks mostly off the car's mask, hidden by the walk overlay
  panels: Panel[];
  coverage: Coverage | null;
  topup: { missing: string[]; topup_text: string; nouns_to_rerun: string[] } | null;
  keyframes: Keyframe[];
  marks: Mark[];
  matches: PanelMatch[];
  verdicts: Verdict[];
  narration: Narration | null;
  markTracks: MarkTrack[];
  looks: LookRow[]; // derived/looks.json: the second looks, each of which can only lower a label
  openWeights: OpenWeightsRow[]; // open_weights.ndjson: the Muse Spark Open calls on Baseten behind those looks
  claims: Claim[]; // every claim placed against this run (claim/<id>.json, and derived/claim.json while it is open), oldest first
  claim: Claim | null; // the latest of them
  spark: { requested: number; answered: number; truncated: number };
  voice: Record<Role, VoiceState>;
  sheet: { written: boolean; sha256s: Record<string, string> };
  deletions: DeletionRecord[];
  ledger: Ledger;
  stage: { name: string; pct: number };
  errors: string[];
  partial: string[] | null;
  partialText: string | null;
  completed: boolean;
  replay: { mode: ReplayMode; badgeDate: string } | null;
  lastSeq: number;
  lastTs: string | null;
}

// Kinds a driver adds around the contract events. "<kind>.loaded" carries the derived JSON an event references.
export type SyntheticKind =
  | 'reset'
  | 'run.loaded'
  | 'video.loaded'
  | 'replay.started'
  | 'panels.derived.loaded'
  | 'coverage.updated.loaded'
  | 'marks.derived.loaded'
  | 'narration.derived.loaded'
  | 'staysput.derived.loaded'
  | 'match.derived.loaded'
  | 'looks.loaded' // derived/looks.json and open_weights.ndjson, read with the match
  | 'claim.derived.loaded'
  | 'claims.loaded' // claim/claim_<n>.json and derived/claim.json, read when the run ends or claim.derived arrives
  | 'deletions.loaded'
  | 'run.ended.loaded'; // run.json of a run that ended with no chained event (failed before the seal)

export interface SyntheticEvent {
  kind: SyntheticKind;
  payload: Record<string, unknown>;
}

export type AnyEvent = Event | SyntheticEvent;
