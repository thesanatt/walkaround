import { formatDollars } from '../lib/format.ts';
import type { Ledger } from '../state/types.ts';

// Dollars so far, requests in flight, p50 first delta, parse CPU: the ledger as it moves.
export function LedgerStrip({ ledger }: { ledger: Ledger }) {
  return (
    <div className="ledger-strip mono small" aria-label="ledger">
      <span>{formatDollars(ledger.dollarsSoFar)} of model calls so far</span>
      <span>{ledger.inFlight} in flight</span>
      <span>{ledger.p50FirstDeltaMs === null ? 'no first delta yet' : `p50 first delta ${(ledger.p50FirstDeltaMs / 1000).toFixed(1)} s`}</span>
      <span>parse CPU {(ledger.parseCpuMs / 1000).toFixed(1)} s</span>
    </div>
  );
}
