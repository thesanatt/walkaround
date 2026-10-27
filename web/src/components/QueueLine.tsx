import type { StatusResponse } from '../contracts/types.ts';

// "position N in line": /api/status reports the position; the minutes estimate is not in the contract, so none is printed.
export function QueueLine({ status }: { status: StatusResponse | null }) {
  if (!status || status.queue_position === null || status.queue_position <= 0) return null;
  return <p className="queue-line">position {status.queue_position} in line; this page updates when the run starts</p>;
}
