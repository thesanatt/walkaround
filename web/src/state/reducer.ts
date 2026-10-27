import type {
  Coverage,
  DeletionRecord,
  Keyframe,
  Mark,
  MarkTrack,
  Narration,
  Panel,
  PanelMatch,
  PanelTrack,
  ReplayMode,
  RunManifest,
  Verdict,
  VideoRecord,
} from '../contracts/types.ts';
import type { AnyEvent, LaneState, RunState } from './types.ts';

export function initialState(): RunState {
  return {
    run: null,
    videos: {},
    uploads: {},
    lanes: {},
    frames: {},
    seal: { manifest_sha256: null, ots: null, tsr_time: null, tsa_issuer: null },
    tracks: [],
    panels: [],
    coverage: null,
    keyframes: [],
    marks: [],
    matches: [],
    verdicts: [],
    narration: null,
    markTracks: [],
    spark: { requested: 0, answered: 0, truncated: 0 },
    voice: { done: false, mode_applied: null },
    sheet: { written: false, sha256s: {} },
    deletions: [],
    ledger: { dollarsSoFar: 0, inFlight: 0, p50FirstDeltaMs: null, parseCpuMs: 0, firstDeltas: [] },
    stage: { name: 'created', pct: 0 },
    errors: [],
    partial: null,
    completed: false,
    replay: null,
    lastSeq: -1,
    lastTs: null,
  };
}

function str(v: unknown, fallback = ''): string {
  return typeof v === 'string' ? v : fallback;
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
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

function laneKey(payload: Record<string, unknown>): { role: string; noun: string } | null {
  const role = str(payload['role']);
  const noun = str(payload['noun']) || str(payload['prompt_id']) || str(payload['lane']);
  if (!role || !noun) return null;
  return { role, noun };
}

function withLane(state: RunState, role: string, noun: string, fn: (lane: LaneState) => LaneState): RunState {
  const roleLanes = state.lanes[role] ?? {};
  const lane = roleLanes[noun] ?? emptyLane();
  return { ...state, lanes: { ...state.lanes, [role]: { ...roleLanes, [noun]: fn(lane) } } };
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const lo = sorted[mid - 1];
  const hi = sorted[mid];
  if (hi === undefined) return null;
  return sorted.length % 2 === 0 && lo !== undefined ? (lo + hi) / 2 : hi;
}

function countInFlight(state: RunState): number {
  let n = 0;
  for (const roleLanes of Object.values(state.lanes)) {
    for (const lane of Object.values(roleLanes)) {
      if (lane.status === 'requested' || lane.status === 'streaming') n += 1;
    }
  }
  return n + Math.max(0, state.spark.requested - state.spark.answered);
}

function recount(state: RunState): RunState {
  return { ...state, ledger: { ...state.ledger, inFlight: countInFlight(state) } };
}

function asArray<T>(v: unknown): T[] {
  return Array.isArray(v) ? (v as T[]) : [];
}

// files.deleted {file_id, response, reason}, responses.deleted {request_id, response, reason} and
// quarantine.deleted {files[]} as orchestrator.py emits them; a driver may also hand over whole rows.
function deletionRows(kind: string, payload: Record<string, unknown>, ts: string | null): DeletionRecord[] {
  const row = payload['row'] as DeletionRecord | undefined;
  const rows = asArray<DeletionRecord>(payload['rows']);
  if (row || rows.length) return row ? [row] : rows;
  const deletedAt = ts ?? '';
  const reason = str(payload['reason']);
  const response = (payload['response'] as Record<string, unknown> | null | undefined) ?? null;
  if (kind === 'files.deleted' && str(payload['file_id'])) {
    return [{ target: str(payload['file_id']), kind: 'files_api', sha256: null, bytes: null, deleted_at: deletedAt, reason, api_response: response }];
  }
  if (kind === 'responses.deleted' && str(payload['request_id'])) {
    return [{ target: str(payload['request_id']), kind: 'response', sha256: null, bytes: null, deleted_at: deletedAt, reason, api_response: response }];
  }
  if (kind === 'quarantine.deleted') {
    return asArray<unknown>(payload['files']).map((f) => ({
      target: String(f),
      kind: 'quarantine' as const,
      sha256: null,
      bytes: null,
      deleted_at: deletedAt,
      reason,
      api_response: null,
    }));
  }
  return [];
}

// The one reducer. Pure. Never branches on live versus replay; a driver feeds the same kinds either way.
export function reduce(state: RunState, event: AnyEvent): RunState {
  const payload = (event.payload ?? {}) as Record<string, unknown>;
  const seq = typeof event.seq === 'number' ? event.seq : state.lastSeq;
  const ts = typeof event.ts === 'string' ? event.ts : state.lastTs;
  state = seq !== state.lastSeq || ts !== state.lastTs ? { ...state, lastSeq: seq, lastTs: ts } : state;

  switch (event.kind) {
    case 'reset':
      return initialState();

    case 'run.loaded':
      return { ...state, run: payload as unknown as RunManifest, stage: stageFromRun(state, payload) };

    case 'video.loaded': {
      const record = payload['record'] as VideoRecord | undefined;
      const role = str(payload['role']) || record?.role;
      if (!record || !role) return state;
      return { ...state, videos: { ...state.videos, [role]: record } };
    }

    case 'replay.started':
      return { ...state, replay: { mode: str(payload['mode'], 'fixed') as ReplayMode, badgeDate: str(payload['recorded_on']) } };

    case 'run.created':
      return { ...state, stage: { name: 'created', pct: 0 } };

    case 'upload.done': {
      const role = str(payload['role']);
      if (!role) return state;
      const done = {
        bytes: num(payload['bytes']) ?? 0,
        sha256: str(payload['sha256']),
        filename: str(payload['filename']),
        warning: str(payload['warning']) || null,
      };
      return { ...state, uploads: { ...state.uploads, [role]: done } };
    }

    case 'upload.started':
    case 'media.probed':
    case 'media.proxy_done':
    case 'files.uploaded':
      return state;

    case 'seal.manifest':
      return { ...state, seal: { ...state.seal, manifest_sha256: str(payload['sha256']) || state.seal.manifest_sha256 } };

    case 'seal.ots':
      return { ...state, seal: { ...state.seal, ots: str(payload['status']) || str(payload['manifest_ots']) || 'pending' } };

    case 'seal.tsr':
      return {
        ...state,
        seal: {
          ...state.seal,
          tsr_time: str(payload['tsr_time']) || state.seal.tsr_time,
          tsa_issuer: str(payload['tsa_issuer']) || str(payload['issuer']) || state.seal.tsa_issuer,
        },
      };

    case 'sam.request': {
      const key = laneKey(payload);
      if (!key) return state;
      return recount(
        withLane(state, key.role, key.noun, (lane) => ({
          ...lane,
          status: 'requested',
          request_id: str(payload['request_id']) || lane.request_id,
        })),
      );
    }

    case 'sam.first_delta': {
      const key = laneKey(payload);
      const ms = num(payload['ms']);
      let next = state;
      if (key) {
        next = withLane(next, key.role, key.noun, (lane) => ({
          ...lane,
          status: lane.status === 'requested' ? 'streaming' : lane.status,
          first_delta_ms: ms ?? lane.first_delta_ms,
        }));
      }
      if (ms !== null) {
        const firstDeltas = [...next.ledger.firstDeltas, ms];
        next = { ...next, ledger: { ...next.ledger, firstDeltas, p50FirstDeltaMs: median(firstDeltas) } };
      }
      return next;
    }

    case 'sam.frame': {
      const key = laneKey(payload);
      const frame = num(payload['frame']);
      if (!key || frame === null) return state;
      const objects = asArray<{ id: unknown; box: unknown; c: unknown }>(payload['objects']).map((o) => ({
        id: String(o.id),
        box: asArray<number>(o.box),
        c: num(o.c),
      }));
      const roleFrames = state.frames[key.role] ?? {};
      let next: RunState = {
        ...state,
        frames: { ...state.frames, [key.role]: { ...roleFrames, [key.noun]: { frame, objects } } },
      };
      next = withLane(next, key.role, key.noun, (lane) => ({
        ...lane,
        status: lane.status === 'requested' ? 'streaming' : lane.status,
        frames: Math.max(lane.frames, frame + 1),
      }));
      return next;
    }

    case 'sam.lane_done': {
      const key = laneKey(payload);
      if (!key) return state;
      const status = str(payload['status'], 'completed') as LaneState['status'];
      const parseCpuS = num(payload['parse_cpu_s']) ?? 0;
      const next = withLane(state, key.role, key.noun, (lane) => ({
        ...lane,
        status,
        frames: num(payload['frames']) ?? lane.frames,
        ids: asArray<unknown>(payload['ids']).map(String),
        dollars: num(payload['dollars']) ?? lane.dollars,
        lane_path: str(payload['lane_path']) || lane.lane_path,
        sha256: str(payload['sha256']) || lane.sha256,
        frames_sent: num(payload['frames_sent']) ?? lane.frames_sent,
        video_frames_processed: num(payload['video_frames_processed']) ?? lane.video_frames_processed,
      }));
      return recount({ ...next, ledger: { ...next.ledger, parseCpuMs: next.ledger.parseCpuMs + Math.round(parseCpuS * 1000) } });
    }

    case 'sam.failed': {
      const key = laneKey(payload);
      if (!key) return state;
      const status = str(payload['status']) === 'incomplete' ? 'incomplete' : 'failed';
      return recount(
        withLane(state, key.role, key.noun, (lane) => ({
          ...lane,
          status,
          error: str(payload['error']) || str(payload['text']) || status,
        })),
      );
    }

    case 'blur.done':
      return state;

    case 'panels.derived':
    case 'coverage.updated':
    case 'topup.requested':
    case 'marks.derived':
    case 'narration.derived':
    case 'staysput.derived':
    case 'match.derived':
    case 'claim.derived':
      return state;

    case 'panels.derived.loaded':
      return {
        ...state,
        tracks: asArray<PanelTrack>(payload['tracks']).length ? asArray<PanelTrack>(payload['tracks']) : state.tracks,
        panels: asArray<Panel>(payload['panels']).length ? asArray<Panel>(payload['panels']) : state.panels,
        keyframes: asArray<Keyframe>(payload['keyframes']).length ? asArray<Keyframe>(payload['keyframes']) : state.keyframes,
      };

    case 'coverage.updated.loaded':
    case 'topup.requested.loaded':
      return { ...state, coverage: (payload['coverage'] as Coverage | undefined) ?? state.coverage };

    case 'marks.derived.loaded':
      return { ...state, marks: asArray<Mark>(payload['marks']) };

    case 'narration.derived.loaded':
      return { ...state, narration: (payload['narration'] as Narration | undefined) ?? state.narration };

    case 'staysput.derived.loaded':
      return { ...state, markTracks: asArray<MarkTrack>(payload['mark_tracks']) };

    case 'match.derived.loaded':
      return {
        ...state,
        matches: asArray<PanelMatch>(payload['matches']),
        verdicts: asArray<Verdict>(payload['verdicts']),
      };

    case 'spark.request':
      return recount({ ...state, spark: { ...state.spark, requested: state.spark.requested + 1 } });

    case 'spark.response': {
      // orchestrator.py emits only the response, so the request count follows the answers when it lags.
      const truncated = str(payload['status']) === 'incomplete' || payload['truncated'] === true ? 1 : 0;
      const answered = state.spark.answered + 1;
      return recount({
        ...state,
        spark: { requested: Math.max(state.spark.requested, answered), answered, truncated: state.spark.truncated + truncated },
      });
    }

    case 'voice.done': {
      const applied = payload['mode_applied'];
      return { ...state, voice: { done: true, mode_applied: typeof applied === 'boolean' ? applied : null } };
    }

    case 'sheet.written': {
      const raw = (payload['sha256s'] ?? payload['sha256']) as Record<string, unknown> | undefined;
      const sha: Record<string, string> = {};
      if (raw && typeof raw === 'object') {
        for (const [k, v] of Object.entries(raw)) if (typeof v === 'string') sha[k] = v;
      }
      return { ...state, sheet: { written: true, sha256s: sha } };
    }

    case 'files.deleted':
    case 'responses.deleted':
    case 'quarantine.deleted': {
      const added = deletionRows(event.kind, payload, ts);
      return added.length ? { ...state, deletions: [...state.deletions, ...added] } : state;
    }

    case 'ledger.row': {
      const dollars = num(payload['dollars_so_far']);
      return dollars === null ? state : { ...state, ledger: { ...state.ledger, dollarsSoFar: dollars } };
    }

    case 'stage':
      return { ...state, stage: { name: str(payload['name'], state.stage.name), pct: num(payload['pct']) ?? state.stage.pct } };

    case 'run.partial':
      return { ...state, partial: asArray<unknown>(payload['missing']).map(String), stage: { name: 'partial', pct: 100 } };

    case 'run.completed':
      return { ...state, completed: true, stage: { name: str(payload['status']) === 'partial' ? 'partial' : 'done', pct: 100 } };

    case 'run.failed':
      return {
        ...state,
        errors: [...state.errors, str(payload['text']) || str(payload['error']) || 'run failed'],
        stage: { name: 'failed', pct: state.stage.pct },
      };

    default:
      return state;
  }
}

function stageFromRun(state: RunState, payload: Record<string, unknown>): RunState['stage'] {
  const status = str(payload['status']);
  if (!status || state.stage.pct > 0) return state.stage;
  return { name: status, pct: status === 'done' ? 100 : 0 };
}
