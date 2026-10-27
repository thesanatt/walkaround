import { formatDollars } from '../lib/format.ts';
import type { RunState } from '../state/types.ts';

export function LedgerStrip({ ledger }: { ledger: RunState['ledger'] }) {
  return (
    <div className="ledger-strip mono small" aria-label="ledger">
      <span>{formatDollars(ledger.dollarsSoFar)} of model calls so far</span>
      <span>{ledger.inFlight} in flight</span>
      <span>{ledger.p50FirstDeltaMs === null ? 'first delta: none yet' : `p50 first delta ${(ledger.p50FirstDeltaMs / 1000).toFixed(1)} s`}</span>
      <span>parse CPU {(ledger.parseCpuMs / 1000).toFixed(1)} s</span>
    </div>
  );
}
