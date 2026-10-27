import type { MarkTrack } from '../contracts/types.ts';

// The sheet's Stays Put line per mark: "drift 0.03, b1 0.20, b2 0.01, not panned" or the inconclusive sentence.
export function staysPutLine(mt: MarkTrack): string {
  if (mt.verdict === 'not_run') return 'not run';
  if (mt.verdict === 'no_track') return 'no track (the mark did not track through its clip)';
  if (mt.verdict === 'inconclusive') return mt.panned ? 'inconclusive (panned)' : 'inconclusive (camera did not move enough)';
  const parts = [
    mt.drift_rms !== null ? `drift ${mt.drift_rms.toFixed(2)}` : null,
    mt.b1 !== null ? `b1 ${mt.b1.toFixed(2)}` : null,
    mt.b2 !== null ? `b2 ${mt.b2.toFixed(2)}` : null,
    mt.panned ? 'panned' : 'not panned',
  ].filter((p): p is string => p !== null);
  return `${mt.verdict}: ${parts.join(', ')}`;
}

export function StaysPutList({ markTracks }: { markTracks: MarkTrack[] }) {
  if (!markTracks.length) return <div className="empty">Stays Put rows come with staysput.derived: each mark tracked through its own clip.</div>;
  return (
    <ul className="staysput-list">
      {markTracks.map((mt) => (
        <li key={`${mt.role}-${mt.mark_id}`}>
          <span className="mono small">
            {mt.role} {mt.mark_id}
          </span>
          {mt.is_control ? ' (reflection control)' : ''}: {staysPutLine(mt)}
          {mt.curvature_cap ? '; curvature cap' : ''}
          {mt.highlight_cap ? '; highlight cap' : ''}
        </li>
      ))}
    </ul>
  );
}
