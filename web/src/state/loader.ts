import { api } from '../api/client.ts';
import type { Event } from '../contracts/types.ts';
import type { AnyEvent } from './types.ts';

export type Dispatch = (event: AnyEvent) => void;

async function settle<T>(p: Promise<T>): Promise<T | null> {
  try {
    return await p;
  } catch {
    return null;
  }
}

// Derived JSON referenced by an event is fetched here and handed to the reducer as "<kind>.loaded",
// so the live and replay drivers feed identical sequences. Missing files are silently skipped
// (a partial run has fewer derived files).
export async function loadForEvent(runId: string, event: Event, dispatch: Dispatch): Promise<void> {
  const payload = event.payload ?? {};
  switch (event.kind) {
    case 'upload.done':
    case 'media.probed':
    case 'media.proxy_done':
    case 'blur.done': {
      const role = typeof payload['role'] === 'string' ? payload['role'] : null;
      const roles = role ? [role] : ['pickup', 'return'];
      await Promise.all(
        roles.map(async (r) => {
          const record = await settle(api.video(runId, r));
          if (record) dispatch({ kind: 'video.loaded', payload: { role: r, record } });
        }),
      );
      return;
    }
    case 'panels.derived': {
      const [tracks, panels, keyframes] = await Promise.all([
        settle(api.derived<unknown[]>(runId, 'tracks.json')),
        settle(api.derived<unknown[]>(runId, 'panels.json')),
        settle(api.derived<unknown[]>(runId, 'keyframes.json')),
      ]);
      dispatch({ kind: 'panels.derived.loaded', payload: { tracks: tracks ?? [], panels: panels ?? [], keyframes: keyframes ?? [] } });
      return;
    }
    case 'coverage.updated':
    case 'topup.requested': {
      const coverage = await settle(api.derived<unknown>(runId, 'coverage.json'));
      if (coverage) dispatch({ kind: `${event.kind}.loaded`, payload: { coverage } });
      return;
    }
    case 'marks.derived': {
      // keyframes.json gains conditions and marks_status at this step, so it is read again with the marks.
      const [marks, keyframes] = await Promise.all([
        settle(api.derived<unknown[]>(runId, 'marks.json')),
        settle(api.derived<unknown[]>(runId, 'keyframes.json')),
      ]);
      if (keyframes) dispatch({ kind: 'panels.derived.loaded', payload: { tracks: [], panels: [], keyframes } });
      if (marks) dispatch({ kind: 'marks.derived.loaded', payload: { marks } });
      return;
    }
    case 'narration.derived': {
      const narration = await settle(api.derived<unknown>(runId, 'narration.json'));
      if (narration) dispatch({ kind: 'narration.derived.loaded', payload: { narration } });
      return;
    }
    case 'staysput.derived': {
      const tracks = await settle(api.derived<unknown[]>(runId, 'mark_tracks.json'));
      if (tracks) dispatch({ kind: 'staysput.derived.loaded', payload: { mark_tracks: tracks } });
      return;
    }
    case 'match.derived': {
      const [matches, verdicts] = await Promise.all([
        settle(api.derived<unknown[]>(runId, 'matches.json')),
        settle(api.derived<unknown[]>(runId, 'verdicts.json')),
      ]);
      dispatch({ kind: 'match.derived.loaded', payload: { matches: matches ?? [], verdicts: verdicts ?? [] } });
      return;
    }
    default:
      return;
  }
}

export const LOADING_KINDS = new Set([
  'upload.done',
  'media.probed',
  'media.proxy_done',
  'blur.done',
  'panels.derived',
  'coverage.updated',
  'topup.requested',
  'marks.derived',
  'narration.derived',
  'staysput.derived',
  'match.derived',
]);
