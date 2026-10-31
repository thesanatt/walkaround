import type {
  Claim,
  Coverage,
  DeletionRecord,
  FfprobeInfo,
  Keyframe,
  Mark,
  MarkTrack,
  Narration,
  Panel,
  PanelMatch,
  PanelTrack,
  ProxyRecord,
  ReplayMode,
  RunManifest,
  Verdict,
  VideoRecord,
} from '../contracts/types.ts';
import { mergeClaims } from '../lib/claim.ts';
import { readLooks, readOpenWeights } from '../lib/looks.ts';
import { laneId } from '../lib/panels.ts';
import type { AnyEvent, LaneState, LaneUiStatus, RoleMedia, RunState } from './types.ts';

export function initialState(): RunState {
  return {
    run: null,
    videos: {},
    media: {},
    lanes: {},
    frames: {},
    seal: { manifest_sha256: null, ots: null, tsr_time: null, tsa_issuer: null, files_uploaded: false },
    tracks: [],
    panels: [],
    coverage: null,
    topup: null,
    keyframes: [],
    marks: [],
    matches: [],
    verdicts: [],
    narration: null,
    markTracks: [],
    looks: [],
    openWeights: [],
    claims: [],
    claim: null,
    spark: { requested: 0, answered: 0, truncated: 0 },
    voice: {},
    sheet: { written: false, sha256s: {} },
    deletions: [],
    ledger: { dollarsSoFar: 0, inFlight: 0, p50FirstDeltaMs: null, parseCpuMs: 0, firstDeltas: [] },
    stage: { name: 'created', pct: 0 },
    errors: [],
    partial: null,
    partialText: null,
    completed: false,
    replay: null,
    lastSeq: -1,
    lastTs: null,
  };
}

const LANE_STATUSES: ReadonlySet<string> = new Set(['completed', 'zero_matches', 'incomplete', 'failed']);

function str(v: unknown, fallback = ''): string {
  return typeof v === 'string' ? v : fallback;
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function list<T>(v: unknown): T[] {
  return Array.isArray(v) ? (v as T[]) : [];
}

function stringMap(v: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (v && typeof v === 'object') for (const [k, val] of Object.entries(v)) if (typeof val === 'string') out[k] = val;
  return out;
}

function emptyLane(): LaneState {
  return {
    status: 'requested',
    frames: 0,
    ids: [],
    first_delta_ms: null,
    dollars: 0,
    lane_path: null,
    sha256: null,
    request_id: null,
    frames_sent: null,
    video_frames_processed: null,
    error: null,
  };
}

function emptyMedia(): RoleMedia {
  return { upload: null, ffprobe: null, proxy: null, blurredPath: null, fileId: null };
}

function laneKey(payload: Record<string, unknown>): { role: string; lane: string } | null {
  const role = str(payload['role']);
  const lane = str(payload['noun']) || str(payload['prompt_id']) || str(payload['lane']);
  return role && lane ? { role, lane: laneId(lane) } : null;
}

function withLane(state: RunState, role: string, lane: string, fn: (l: LaneState) => LaneState): RunState {
  const roleLanes = state.lanes[role] ?? {};
  return { ...state, lanes: { ...state.lanes, [role]: { ...roleLanes, [lane]: fn(roleLanes[lane] ?? emptyLane()) } } };
}

function withMedia(state: RunState, role: string, patch: Partial<RoleMedia>): RunState {
  return { ...state, media: { ...state.media, [role]: { ...(state.media[role] ?? emptyMedia()), ...patch } } };
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

function inFlight(state: RunState): number {
  let n = 0;
  for (const roleLanes of Object.values(state.lanes)) {
    for (const lane of Object.values(roleLanes)) if (lane.status === 'requested' || lane.status === 'streaming') n += 1;
  }
  return n + Math.max(0, state.spark.requested - state.spark.answered);
}

function recount(state: RunState): RunState {
  const n = inFlight(state);
  return n === state.ledger.inFlight ? state : { ...state, ledger: { ...state.ledger, inFlight: n } };
}

// files.deleted {file_id, response, reason}, responses.deleted {request_id, response, reason}, quarantine.deleted {files}.
function deletionRows(kind: string, payload: Record<string, unknown>, ts: string): DeletionRecord[] {
  const reason = str(payload['reason']);
  const response = (payload['response'] as Record<string, unknown> | null | undefined) ?? null;
  const row = (target: string, k: DeletionRecord['kind'], api: Record<string, unknown> | null): DeletionRecord => ({
    target,
    kind: k,
    sha256: null,
    bytes: null,
    deleted_at: ts,
    reason,
    api_response: api,
  });
  if (kind === 'files.deleted' && str(payload['file_id'])) return [row(str(payload['file_id']), 'files_api', response)];
  if (kind === 'responses.deleted' && str(payload['request_id'])) return [row(str(payload['request_id']), 'response', response)];
  if (kind === 'quarantine.deleted') return list<unknown>(payload['files']).map((f) => row(String(f), 'quarantine', null));
  return [];
}

// The one reducer. Pure: no I/O, no clock, no branch on live versus replay. A driver feeds the same kinds either way.
// Muse Voice sometimes writes speaker markers such as "<|speaker_A|>" into a transcript, and the narration lane can cut
// a piece in the middle of one ("... the handle <|speaker_" then "A|> trunk lid ..."). The words shown drop them; the
// stored narration.json keeps what Voice said.
const SPEAKER_TOKEN = /<\|[^|<>]*\|>|<\|[^|<>]*$|^[^|<>\s]*\|>/g;

export function cleanTurnText(text: string): string {
  return text.replace(SPEAKER_TOKEN, ' ').replace(/\s+/g, ' ').trim();
}

function cleanNarration(narration: Narration): Narration {
  const per_role = Object.fromEntries(
    Object.entries(narration.per_role).map(([role, nr]) => [role, { ...nr, turns: nr.turns.map((t) => ({ ...t, text: cleanTurnText(t.text) })) }]),
  ) as Narration['per_role'];
  return { ...narration, per_role };
}

export function reduce(state: RunState, event: AnyEvent): RunState {
  const payload = (event.payload ?? {}) as Record<string, unknown>;
  if ('seq' in event && typeof event.seq === 'number') {
    state = { ...state, lastSeq: event.seq, lastTs: typeof event.ts === 'string' ? event.ts : state.lastTs };
  }
  const ts = state.lastTs ?? '';

  switch (event.kind) {
    case 'reset':
      return initialState();

    case 'run.loaded': {
      // The stage comes only from stage events (the SSE history starts at seq 0), never from run.json's status.
      const run = payload as unknown as RunManifest;
      return run.run_id ? { ...state, run } : state;
    }

    case 'video.loaded': {
      const record = payload['record'] as VideoRecord | undefined;
      const role = str(payload['role']) || record?.role;
      if (!record || !role) return state;
      const media = state.media[role] ?? emptyMedia();
      return withMedia({ ...state, videos: { ...state.videos, [role]: record } }, role, {
        ffprobe: media.ffprobe ?? record.ffprobe,
        proxy: media.proxy ?? record.proxy,
        blurredPath: media.blurredPath ?? record.proxy_blurred?.path ?? null,
        fileId: media.fileId ?? record.files_api?.file_id ?? null,
      });
    }

    case 'replay.started':
      return { ...state, replay: { mode: str(payload['mode'], 'fixed') as ReplayMode, badgeDate: str(payload['recorded_on']) } };

    case 'run.created':
    case 'upload.started':
      return state;

    case 'upload.done': {
      const role = str(payload['role']);
      if (!role) return state;
      return withMedia(state, role, {
        upload: {
          bytes: num(payload['bytes']) ?? 0,
          sha256: str(payload['sha256']),
          filename: str(payload['filename']),
          warning: str(payload['warning']) || null,
        },
      });
    }

    case 'seal.manifest':
      return { ...state, seal: { ...state.seal, manifest_sha256: str(payload['sha256']) || state.seal.manifest_sha256 } };

    case 'seal.ots':
      return { ...state, seal: { ...state.seal, ots: str(payload['status']) || state.seal.ots || 'pending' } };

    case 'seal.tsr':
      return {
        ...state,
        seal: { ...state.seal, tsr_time: str(payload['tsr_time']) || state.seal.tsr_time, tsa_issuer: str(payload['issuer']) || state.seal.tsa_issuer },
      };

    case 'media.probed': {
      const role = str(payload['role']);
      const ffprobe = (payload['ffprobe'] as FfprobeInfo | undefined) ?? null;
      return role && ffprobe ? withMedia(state, role, { ffprobe }) : state;
    }

    case 'media.proxy_done': {
      const role = str(payload['role']);
      const proxy = (payload['proxy'] as ProxyRecord | undefined) ?? null;
      return role && proxy ? withMedia(state, role, { proxy }) : state;
    }

    case 'files.uploaded': {
      const role = str(payload['role']);
      const next = state.seal.files_uploaded ? state : { ...state, seal: { ...state.seal, files_uploaded: true } };
      return role ? withMedia(next, role, { fileId: str(payload['file_id']) || null }) : next;
    }

    case 'blur.done': {
      const role = str(payload['role']);
      const path = str(payload['path']);
      return role && path ? withMedia(state, role, { blurredPath: path }) : state;
    }

    case 'sam.request': {
      const key = laneKey(payload);
      if (!key) return state;
      return recount(withLane(state, key.role, key.lane, (l) => ({ ...l, status: 'requested', request_id: str(payload['request_id']) || l.request_id })));
    }

    case 'sam.first_delta': {
      const key = laneKey(payload);
      const ms = num(payload['ms']);
      let next = state;
      if (key) {
        next = withLane(next, key.role, key.lane, (l) => ({
          ...l,
          status: l.status === 'requested' ? 'streaming' : l.status,
          first_delta_ms: ms ?? l.first_delta_ms,
        }));
      }
      if (ms !== null) {
        const firstDeltas = [...next.ledger.firstDeltas, ms];
        next = { ...next, ledger: { ...next.ledger, firstDeltas, p50FirstDeltaMs: median(firstDeltas) } };
      }
      return recount(next);
    }

    case 'sam.frame': {
      const key = laneKey(payload);
      const frame = num(payload['frame']);
      if (!key || frame === null) return state;
      const objects = list<{ id: unknown; box: unknown; c: unknown }>(payload['objects']).map((o) => ({ id: String(o.id), box: list<number>(o.box), c: num(o.c) }));
      const roleFrames = state.frames[key.role] ?? {};
      const next: RunState = { ...state, frames: { ...state.frames, [key.role]: { ...roleFrames, [key.lane]: { frame, objects } } } };
      return recount(
        withLane(next, key.role, key.lane, (l) => ({
          ...l,
          status: l.status === 'requested' ? 'streaming' : l.status,
          frames: Math.max(l.frames, frame + 1),
        })),
      );
    }

    case 'sam.lane_done': {
      const key = laneKey(payload);
      if (!key) return state;
      const raw = str(payload['status'], 'completed');
      const status = (LANE_STATUSES.has(raw) ? raw : 'completed') as LaneUiStatus;
      const next = withLane(state, key.role, key.lane, (l) => ({
        ...l,
        status,
        frames: num(payload['frames']) ?? l.frames,
        ids: list<unknown>(payload['ids']).map(String),
        dollars: num(payload['dollars']) ?? l.dollars,
        lane_path: str(payload['lane_path']) || l.lane_path,
        sha256: str(payload['sha256']) || l.sha256,
        frames_sent: num(payload['frames_sent']) ?? l.frames_sent,
        video_frames_processed: num(payload['video_frames_processed']) ?? l.video_frames_processed,
      }));
      const parseMs = Math.round((num(payload['parse_cpu_s']) ?? 0) * 1000);
      return recount({ ...next, ledger: { ...next.ledger, parseCpuMs: next.ledger.parseCpuMs + parseMs } });
    }

    case 'sam.failed': {
      // An incomplete lane emits sam.failed and then sam.lane_done; a failed lane emits sam.failed only.
      const key = laneKey(payload);
      if (!key) return state;
      const status: LaneUiStatus = str(payload['status']) === 'incomplete' ? 'incomplete' : 'failed';
      return recount(withLane(state, key.role, key.lane, (l) => ({ ...l, status, error: str(payload['error']) || status })));
    }

    case 'panels.derived':
    case 'coverage.updated':
    case 'marks.derived':
    case 'narration.derived':
    case 'staysput.derived':
    case 'match.derived':
    case 'claim.derived':
    case 'spark.request':
      // The derived files arrive as "<kind>.loaded"; spark.request counts one call in flight until its response.
      return event.kind === 'spark.request' ? recount({ ...state, spark: { ...state.spark, requested: state.spark.requested + 1 } }) : state;

    case 'topup.requested':
      return {
        ...state,
        topup: {
          missing: list<unknown>(payload['missing']).map(String),
          topup_text: str(payload['topup_text']),
          nouns_to_rerun: list<unknown>(payload['nouns_to_rerun']).map(String),
        },
      };

    case 'panels.derived.loaded': {
      const tracks = payload['tracks'] as PanelTrack[] | undefined;
      const panels = payload['panels'] as Panel[] | undefined;
      const keyframes = payload['keyframes'] as Keyframe[] | undefined;
      return { ...state, tracks: tracks ?? state.tracks, panels: panels ?? state.panels, keyframes: keyframes ?? state.keyframes };
    }

    case 'coverage.updated.loaded':
      return { ...state, coverage: (payload['coverage'] as Coverage | undefined) ?? state.coverage };

    case 'marks.derived.loaded': {
      const keyframes = payload['keyframes'] as Keyframe[] | undefined;
      return { ...state, marks: list<Mark>(payload['marks']), keyframes: keyframes ?? state.keyframes };
    }

    case 'narration.derived.loaded': {
      const narration = payload['narration'] as Narration | undefined;
      return { ...state, narration: narration ? cleanNarration(narration) : state.narration };
    }

    case 'staysput.derived.loaded':
      return { ...state, markTracks: list<MarkTrack>(payload['mark_tracks']) };

    case 'match.derived.loaded':
      return { ...state, matches: list<PanelMatch>(payload['matches']), verdicts: list<Verdict>(payload['verdicts']) };

    case 'looks.loaded': {
      const raw = payload['open_weights'];
      return { ...state, looks: readLooks(payload['looks']), openWeights: readOpenWeights(typeof raw === 'string' ? raw : null) };
    }

    case 'claims.loaded': {
      const claims = mergeClaims(list<unknown>(payload['claims']));
      return claims.length ? { ...state, claims, claim: claims[claims.length - 1]! } : state;
    }

    case 'claim.derived.loaded': {
      const claim = payload['claim'] as Claim | undefined;
      if (!claim?.image) return state;
      const claims = mergeClaims(state.claims, [claim]);
      return { ...state, claims, claim: claims[claims.length - 1]! };
    }

    case 'spark.response': {
      // orchestrator._emit_spark emits spark.request then spark.response for each stored call; max() keeps the count
      // right for a log that has responses only.
      const truncated = payload['truncated'] === true || str(payload['status']) === 'incomplete' ? 1 : 0;
      const answered = state.spark.answered + 1;
      return recount({ ...state, spark: { requested: Math.max(state.spark.requested, answered), answered, truncated: state.spark.truncated + truncated } });
    }

    case 'voice.done': {
      const role = str(payload['role'], 'pickup');
      const applied = payload['mode_applied'];
      return {
        ...state,
        voice: {
          ...state.voice,
          [role]: { mode_applied: typeof applied === 'boolean' ? applied : null, turns: num(payload['turns']), text: str(payload['text']) || null },
        },
      };
    }

    case 'sheet.written':
      return { ...state, sheet: { written: true, sha256s: stringMap(payload['sha256']) } };

    case 'files.deleted':
    case 'responses.deleted':
    case 'quarantine.deleted': {
      const rows = deletionRows(event.kind, payload, ts);
      return rows.length ? { ...state, deletions: [...state.deletions, ...rows] } : state;
    }

    case 'deletions.loaded': {
      // deletions.json is the full record (hashes, bytes, the unblurred proxy row that has no event of its own).
      const rows = list<DeletionRecord>(payload['rows']);
      return rows.length ? { ...state, deletions: rows } : state;
    }

    case 'ledger.row': {
      const dollars = num(payload['dollars_so_far']);
      return dollars === null ? state : { ...state, ledger: { ...state.ledger, dollarsSoFar: dollars } };
    }

    case 'stage':
      return { ...state, stage: { name: str(payload['name'], state.stage.name), pct: num(payload['pct']) ?? state.stage.pct } };

    case 'run.partial':
      return { ...state, partial: list<unknown>(payload['missing']).map(String), partialText: str(payload['text']) || null };

    case 'run.completed': {
      const status = str(payload['status'], 'done');
      return { ...state, completed: true, stage: { name: status, pct: 100 } };
    }

    case 'run.failed':
      return {
        ...state,
        completed: true,
        errors: [...state.errors, `${str(payload['stage'], 'run')}: ${str(payload['text'], 'failed')}`],
        stage: { name: 'failed', pct: state.stage.pct },
      };

    case 'run.ended.loaded': {
      // No event will ever come for this run, so run.json is the only word on how it ended.
      const run = payload as unknown as RunManifest;
      if (!run.run_id) return state;
      if (state.completed) return { ...state, run };
      const errors = list<{ stage?: unknown; text?: unknown }>(payload['errors']).map((e) => `${str(e.stage, 'run')}: ${str(e.text, 'failed')}`);
      return { ...state, run, completed: true, errors: [...state.errors, ...errors], stage: { name: str(payload['status'], 'failed'), pct: state.stage.pct } };
    }

    default:
      return state;
  }
}

// Every role this run expects, from the run kind; falls back to what the events named.
export function rolesOf(state: RunState): string[] {
  const kind = state.run?.kind;
  if (kind === 'pair') return ['pickup', 'return'];
  if (kind) return ['pickup'];
  const seen = new Set([...Object.keys(state.media), ...Object.keys(state.lanes)].filter((r) => r === 'pickup' || r === 'return'));
  return seen.has('return') ? ['pickup', 'return'] : ['pickup'];
}

// The proxy fps for a walk: media.proxy_done, else run.json, else the 15 fps default of thresholds.json.
export function fpsOf(state: RunState, role: string): number {
  return state.media[role]?.proxy?.fps ?? state.run?.fps ?? 15;
}
