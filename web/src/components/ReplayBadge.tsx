import { formatDate } from '../lib/format.ts';

// Fixed bottom right on every replayed screen. Never hidden.
export function ReplayBadge({ date }: { date: string }) {
  return <div className="replay-badge mono">replayed from a run on {formatDate(date)}</div>;
}
