import type {
  Coverage,
  DeletionRecord,
  Event,
  Keyframe,
  Mark,
  MarkTrack,
  Narration,
  Panel,
  PanelMatch,
  PanelTrack,
  ReplayMode,
  Role,
  RunManifest,
  Verdict,
  VideoRecord,
} from '../contracts/types.ts';

// A lane's life in the UI: requested -> streaming -> one of the four contract statuses.
export type LaneUiStatus = 'requested' | 'streaming' | 'completed' | 'zero_matches' | 'incomplete' | 'failed';

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
}

export interface SparkCounts {
  requested: number;
  answered: number;
  truncated: number;
}

// What upload.done said, before videos/<role>.json is fetched.
export interface UploadDone {
  bytes: number;
  sha256: string;
  filename: string;
  warning: string | null;
}

export interface RunState {
  run: RunManifest | null;
  videos: Record<Role, VideoRecord>;
  uploads: Record<Role, UploadDone>;
  lanes: Record<Role, Record<string, LaneState>>;
  frames: Record<Role, Record<string, LastFrame>>;
  seal: SealState;
  tracks: PanelTrack[];
  panels: Panel[];
  coverage: Coverage | null;
  keyframes: Keyframe[];
  marks: Mark[];
  matches: PanelMatch[];
  verdicts: Verdict[];
  narration: Narration | null;
  markTracks: MarkTrack[];
  spark: SparkCounts;
  voice: { done: boolean; mode_applied: boolean | null };
  sheet: { written: boolean; sha256s: Record<string, string> };
  deletions: DeletionRecord[];
  ledger: { dollarsSoFar: number; inFlight: number; p50FirstDeltaMs: number | null; parseCpuMs: number; firstDeltas: number[] };
  stage: { name: string; pct: number };
  errors: string[];
  partial: string[] | null;
  completed: boolean;
  replay: { mode: ReplayMode; badgeDate: string } | null;
  lastSeq: number;
  lastTs: string | null;
}

// The synthetic kinds a driver adds around the contract events. The reducer treats them like any other kind.
export const SYNTHETIC_KINDS = {
  runLoaded: 'run.loaded',
  videoLoaded: 'video.loaded',
  replayStarted: 'replay.started',
  derivedSuffix: '.loaded',
} as const;

export type AnyEvent = Event | { kind: string; payload: Record<string, unknown>; seq?: number; ts?: string };
