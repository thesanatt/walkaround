import type { Mark, MarkTrack } from '../contracts/types.ts';
import { controlName, staysPutCaption, staysPutLine, staysPutTone } from '../lib/lines.ts';
import { neighbourCaption, neighbourTone } from '../lib/neighbours.ts';
import { panelName } from '../lib/panels.ts';

export interface StaysPutListProps {
  markTracks: MarkTrack[];
  marks?: Mark[];
  calibrated: boolean; // thresholds staysput.cap_enabled
  missing?: boolean;
}

// Every tracked mark of the run: what the track decided in plain words, then the sheet's numbers. Uncalibrated, the
// caption says it was tracked and the numbers carry no verdict.
export function StaysPutList({ markTracks, marks = [], calibrated, missing = false }: StaysPutListProps) {
  if (!markTracks.length && missing) return <div className="empty">Stays Put did not run in this run, so no mark has its own clip. The match above stands without it.</div>;
  if (!markTracks.length) return <div className="empty">Stays Put rows come with the Stays Put step: each mark tracked through its own short clip.</div>;
  const byId = new Map(marks.map((m) => [m.mark_id, m]));
  return (
    <ul className="staysput-list">
      {markTracks.map((mt) => {
        const m = byId.get(mt.mark_id);
        return (
          <li key={`${mt.role}-${mt.mark_id}`}>
            <span className="sp-who">
              {mt.role} {m ? `${m.type} on the ${panelName(m.panel_id)}` : mt.mark_id}
              {mt.is_control ? `, the ${controlName(calibrated)}` : ''}
            </span>
            <span className="sp-says">
              <span className={`sp-caption ${staysPutTone(mt, calibrated)}`}>{staysPutCaption(mt, calibrated)}</span>
              {neighbourCaption(mt) ? <span className={`sp-neighbours ${neighbourTone(mt, calibrated)}`}>{neighbourCaption(mt)}</span> : null}
            </span>
            <span className="mono small muted">{staysPutLine(mt, calibrated) ?? 'not run'}</span>
          </li>
        );
      })}
    </ul>
  );
}
