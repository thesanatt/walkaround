import type { Event, ReplayMode } from '../contracts/types.ts';

export interface ScheduledEvent {
  event: Event;
  atMs: number;
}

export interface ScheduleOptions {
  // Stage budgets for fixed mode, keyed by the stage event's name (web/kiosk/delays.json or ReplayPayload.delays_ms).
  delaysMs: Record<string, number>;
  // Recorded mode caps every gap at this many ms (thresholds.json orchestrator.replay_recorded_delay_cap_ms).
  capMs?: number;
  // A stage with no budget entry gets this many ms in fixed mode.
  defaultStageMs?: number;
}

const DEFAULT_CAP_MS = 2000;
const DEFAULT_STAGE_MS = 1000;

function parseTs(ts: string): number {
  const t = Date.parse(ts);
  return Number.isFinite(t) ? t : NaN;
}

// Pure: turns a replay payload's events into (event, time from start) pairs for one of the three modes.
// fixed: the events between two stage markers share that stage's budget evenly, so every take is identical.
// recorded: the stored ts deltas, each capped. instant: everything at 0.
export function scheduleReplay(events: Event[], mode: ReplayMode, options: ScheduleOptions): ScheduledEvent[] {
  if (mode === 'instant') return events.map((event) => ({ event, atMs: 0 }));

  if (mode === 'recorded') {
    const cap = options.capMs ?? DEFAULT_CAP_MS;
    let at = 0;
    let prev: number | null = null;
    return events.map((event) => {
      const t = parseTs(event.ts);
      if (prev !== null && Number.isFinite(t) && Number.isFinite(prev)) {
        at += Math.min(cap, Math.max(0, t - prev));
      }
      if (Number.isFinite(t)) prev = t;
      return { event, atMs: at };
    });
  }

  // fixed
  const defaultMs = options.defaultStageMs ?? DEFAULT_STAGE_MS;
  const segments: { name: string; events: Event[] }[] = [];
  let current: { name: string; events: Event[] } = { name: 'created', events: [] };
  for (const event of events) {
    if (event.kind === 'stage') {
      const name = typeof event.payload['name'] === 'string' ? (event.payload['name'] as string) : current.name;
      if (current.events.length) segments.push(current);
      current = { name, events: [event] };
    } else {
      current.events.push(event);
    }
  }
  if (current.events.length) segments.push(current);

  const out: ScheduledEvent[] = [];
  let start = 0;
  for (const segment of segments) {
    const budget = options.delaysMs[segment.name] ?? defaultMs;
    const n = segment.events.length;
    segment.events.forEach((event, i) => {
      // The stage marker itself lands at the segment start; the rest spread to the end of the budget.
      const atMs = i === 0 && event.kind === 'stage' ? start : start + Math.round((budget * (i + 1)) / n);
      out.push({ event, atMs });
    });
    start += budget;
  }
  return out;
}

export function totalDurationMs(schedule: ScheduledEvent[]): number {
  return schedule.length ? Math.max(...schedule.map((s) => s.atMs)) : 0;
}
