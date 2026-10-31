import type { Event, ReplayMode } from '../contracts/types.ts';

export interface ScheduledEvent {
  event: Event;
  atMs: number;
}

// thresholds.json orchestrator.replay_recorded_delay_cap_ms
export const RECORDED_CAP_MS = 2000;

// The stage each event belongs to: the latest stage event's name at or before it, "created" before the first
// (orchestrator._stage_of_each).
export function stageOfEach(events: readonly Event[]): string[] {
  let current = 'created';
  return events.map((e) => {
    if (e.kind === 'stage' && typeof e.payload['name'] === 'string') current = e.payload['name'];
    return current;
  });
}

// Milliseconds to wait before each event, the same numbers orchestrator.replay_delays computes:
// fixed spreads a stage's budget evenly over that stage's events, recorded uses the ts gaps capped at capMs,
// instant waits for nothing.
export function replayWaits(events: readonly Event[], mode: ReplayMode, delaysMs: Record<string, number>, capMs = RECORDED_CAP_MS): number[] {
  if (mode === 'instant' || !events.length) return events.map(() => 0);
  if (mode === 'recorded') {
    return events.map((e, i) => {
      if (i === 0) return 0;
      const gap = Date.parse(e.ts) - Date.parse(events[i - 1]!.ts);
      return Number.isFinite(gap) ? Math.min(capMs, Math.max(0, gap)) : 0;
    });
  }
  const stages = stageOfEach(events);
  const counts = new Map<string, number>();
  for (const s of stages) counts.set(s, (counts.get(s) ?? 0) + 1);
  return stages.map((s) => (delaysMs[s] ?? delaysMs['default'] ?? 0) / counts.get(s)!);
}

// Pure: (event, ms from start) pairs; each event lands after the sum of the waits up to and including its own.
export function scheduleReplay(events: readonly Event[], mode: ReplayMode, delaysMs: Record<string, number>, capMs = RECORDED_CAP_MS): ScheduledEvent[] {
  const waits = replayWaits(events, mode, delaysMs, capMs);
  let at = 0;
  return events.map((event, i) => {
    at += waits[i]!;
    return { event, atMs: at };
  });
}

export function totalMs(schedule: readonly ScheduledEvent[]): number {
  return schedule.length ? schedule[schedule.length - 1]!.atMs : 0;
}
