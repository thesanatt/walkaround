import { ENDED_STATUSES } from '../state/live.ts';
import type { RunState } from '../state/types.ts';

// sheet.written carries sheet.write_sheet's {run-relative path: sha256}, e.g. "sheet/evidence.pdf"; evidence.pdf is
// absent when the PDF render failed. A bare file name is accepted too.
export function sheetFileHash(sha256s: Record<string, string>, name: string): string | null {
  for (const [path, sha] of Object.entries(sha256s)) if (path === name || path.endsWith(`/${name}`)) return sha;
  return null;
}

// The replay badge follows the run, not the URL: a replay driver, a stored replay run, or a finished keep run (the
// sample and eval pairs) streamed through the live driver are all replays of stored events.
export function replayBadgeDate(state: RunState): string | null {
  if (state.replay) return state.replay.badgeDate;
  const run = state.run;
  if (run && (run.live_or_replay === 'replay' || (run.keep && ENDED_STATUSES.has(run.status)))) return run.created_at;
  return null;
}
