import type { MarkTrack } from '../contracts/types.ts';
import { staysPutLine } from '../lib/lines.ts';

export function StaysPutList({ markTracks }: { markTracks: MarkTrack[] }) {
  if (!markTracks.length) return <div className="empty">Stays Put rows come with the Stays Put step: each mark tracked through its own short clip.</div>;
  return (
    <ul className="staysput-list">
      {markTracks.map((mt) => (
        <li key={`${mt.role}-${mt.mark_id}`}>
          <span className="mono small">
            {mt.role} {mt.mark_id}
          </span>
          {mt.is_control ? ' (reflection control)' : ''}: {staysPutLine(mt) ?? 'not run'}
        </li>
      ))}
    </ul>
  );
}
