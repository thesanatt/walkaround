import { useEffect, useMemo, useState } from 'react';

import { routes } from '../api/client.ts';
import { colorForNoun, nounSlug, PRIVACY_NOUNS } from '../lib/panels.ts';
import type { RunState } from '../state/types.ts';
import { LaneStore } from './laneStore.ts';

// Hands every finished lane of the run to the worker as soon as sam.lane_done names its gz file.
// Privacy lanes (license plate, face) are never drawn; their boxes exist only to blur.
export function useLanes(runId: string | null, state: RunState): { store: LaneStore; version: number } {
  const store = useMemo(() => new LaneStore(), [runId]);
  const [version, setVersion] = useState(0);

  useEffect(() => {
    const unsubscribe = store.subscribe(() => setVersion(store.version));
    return () => {
      unsubscribe();
      store.dispose();
    };
  }, [store]);

  useEffect(() => {
    if (!runId) return;
    for (const [role, lanes] of Object.entries(state.lanes)) {
      for (const [noun, lane] of Object.entries(lanes)) {
        if (PRIVACY_NOUNS.has(noun)) continue;
        if (lane.status !== 'completed' && lane.status !== 'zero_matches' && lane.status !== 'incomplete') continue;
        if (lane.status === 'zero_matches') continue;
        if (store.has(role, noun)) continue;
        const rel = lane.lane_path ?? `sam/${role}/${nounSlug(noun)}.ndjson.gz`;
        store.request(role, noun, routes.file(runId, rel), colorForNoun(noun));
      }
    }
  }, [runId, state.lanes, store]);

  return { store, version };
}
