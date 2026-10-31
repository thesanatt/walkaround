import type { StatusResponse } from '../contracts/types.ts';

// "position N in line, about M minutes": app.py estimates M from the last three runs.
export function QueueLine({ status }: { status: StatusResponse | null }) {
  if (!status || status.queue_position === null || status.queue_position <= 0) return null;
  const eta = status.queue_eta_minutes;
  return (
    <p className="queue-line">
      Position {status.queue_position} in line{typeof eta === 'number' ? `, about ${Math.max(1, Math.round(eta))} ${Math.round(eta) === 1 ? 'minute' : 'minutes'}` : ''}. This page updates
      when the run starts.
    </p>
  );
}
