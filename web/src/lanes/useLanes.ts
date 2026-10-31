import { useEffect, useMemo, useState } from 'react';

import { routes, type FileView } from '../api/client.ts';
import { DRAWN_LANES } from '../lib/panels.ts';
import type { RunState } from '../state/types.ts';
import { LaneStore } from './laneStore.ts';

// Hands each finished panel lane of the given walks to the worker as soon as sam.lane_done names its gz file.
// Zero-match and failed lanes have nothing to draw; plate and face lanes are never drawn.
export function useLanes(runId: string | null, state: RunState, roles: readonly string[], view: FileView = 'owner'): { store: LaneStore; version: number } {
  const store = useMemo(() => new LaneStore(), [runId]);
  const [version, setVersion] = useState(0);
  const roleKey = roles.join(',');

  useEffect(() => {
    const unsubscribe = store.subscribe(() => setVersion(store.version));
    return () => {
      unsubscribe();
      store.dispose();
    };
  }, [store]);

  useEffect(() => {
    if (!runId) return;
    for (const role of roleKey.split(',')) {
      const lanes = state.lanes[role] ?? {};
      for (const lane of DRAWN_LANES) {
        const l = lanes[lane];
        if (!l || (l.status !== 'completed' && l.status !== 'incomplete') || store.has(role, lane)) continue;
        const rel = l.lane_path && l.lane_path.endsWith('.ndjson.gz') ? l.lane_path : `sam/${role}/${lane}.ndjson.gz`;
        store.request(role, lane, routes.file(runId, rel, view));
      }
    }
  }, [runId, state.lanes, store, roleKey, view]);

  return { store, version };
}
