import type { PanelTrack } from '../contracts/types.ts';
import { formatClock, formatClockTenths } from '../lib/format.ts';
import { familyNoun, panelName } from '../lib/panels.ts';

export interface FragmentsRowProps {
  role: string;
  panelId: string | null;
  tracks: PanelTrack[];
  fps: number;
  durationS: number | null;
  color: string;
  phase?: 'fragments' | 'joined' | 'anchor'; // the kiosk shows the pieces, then the joined track, then the anchor pass
}

// One panel's timeline: each SAM id's fragment as a span, the merged track named underneath, and the anchor
// re-prompts (the box and the negative points that keep the glass out) when the anchor pass ran.
export function FragmentsRow({ role, panelId, tracks, fps, durationS, color, phase = 'anchor' }: FragmentsRowProps) {
  if (!panelId) return <div className="empty">Pick a panel in the list to see its track.</div>;
  const own = tracks.filter((t) => t.role === role && t.panel_id === panelId);
  if (!own.length) return <div className="empty">No track for the {panelName(panelId)} in the {role} walk yet. Tracks come with the panel step.</div>;
  const fragments = own.flatMap((t) => t.fragments).sort((a, b) => a.first_frame - b.first_frame);
  const lastFrame = Math.max(...fragments.map((f) => f.last_frame + 1));
  const total = durationS && durationS > 0 ? durationS : lastFrame / fps;
  const windows = fragments.map((f) => {
    const [a, b] = [f.first_frame / fps, (f.last_frame + 1) / fps];
    return formatClock(a) === formatClock(b) || b - a < 2 ? `${formatClockTenths(a)} to ${formatClockTenths(b)}` : `${formatClock(a)} to ${formatClock(b)}`;
  });
  const anchors = own.flatMap((t) => t.anchor_reprompts);
  const merged = fragments.length > own.length;
  const joined = phase !== 'fragments';
  const pct = (frame: number) => `${(frame / fps / total) * 100}%`;

  return (
    <div className="fragments">
      <p className="fragments-line">
        <strong>{panelName(panelId)}</strong>: {windows.join(' and ')}
        {merged && joined ? <span className="fragments-joined">; same {familyNoun(panelId)}, fragments {fragments.length} to {own.length}</span> : ''}
      </p>
      <div className="timeline" role="img" aria-label={`${panelName(panelId)} in the ${role} walk: ${windows.join(', ')}`}>
        {fragments.map((f, i) => (
          <span
            key={`${f.object_id}-${i}`}
            className="fragment"
            style={{
              left: `${(f.first_frame / fps / total) * 100}%`,
              width: `${Math.max(0.6, ((f.last_frame + 1 - f.first_frame) / fps / total) * 100)}%`,
              background: color,
            }}
            title={`SAM id ${f.object_id}: frames ${f.first_frame} to ${f.last_frame}`}
          />
        ))}
        <span className="timeline-axis">
          <span>0:00</span>
          <span>{formatClock(total)}</span>
        </span>
      </div>
      {joined ? (
        <div className="timeline track-row" aria-hidden="true">
          {own.map((t) => (
            <span
              key={t.track_id}
              className="track-span"
              style={{
                left: pct(t.fragments[0]?.first_frame ?? 0),
                width: `${Math.max(0.6, (((t.fragments[t.fragments.length - 1]?.last_frame ?? 0) + 1 - (t.fragments[0]?.first_frame ?? 0)) / fps / total) * 100)}%`,
                borderColor: color,
              }}
            >
              <span className="track-label">one track</span>
            </span>
          ))}
        </div>
      ) : null}
      <ul className={`fragments-merged mono small ${joined ? '' : 'hidden'}`}>
        {own.map((t) => (
          <li key={t.track_id}>
            {t.track_id}: SAM ids {t.sam_object_ids.join(', ')}
            {t.merges.length ? `; merged by ${t.merges.map((m) => `${m.rule.replace('_', ' ')} (gap ${m.gap_s.toFixed(1)} s)`).join(', ')}` : ''}
          </li>
        ))}
      </ul>
      <p className={`small muted anchor-line ${phase === 'anchor' ? '' : 'hidden'}`}>
        {anchors.length
          ? anchors.map((a) => `anchor pass: re-prompted at frame ${a.frame} with a box${a.negative_points.length ? ` and ${a.negative_points.length} negative ${a.negative_points.length === 1 ? 'point' : 'points'} on the glass` : ''}`).join('; ')
          : 'no anchor pass on this track'}
      </p>
    </div>
  );
}
