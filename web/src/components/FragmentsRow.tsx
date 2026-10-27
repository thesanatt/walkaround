import { useState } from 'react';

import type { PanelTrack } from '../contracts/types.ts';
import { formatClock } from '../lib/format.ts';
import { panelName } from '../lib/panels.ts';

export interface FragmentsRowProps {
  role: string;
  panelId: string | null;
  tracks: PanelTrack[];
  fps: number | null;
  durationS: number | null;
  color: string;
}

// The timeline row for one panel: its fragments as spans, the merged track as one line underneath,
// and the pass-1 / anchor toggle (disabled until a track carries an anchor re-prompt).
export function FragmentsRow({ role, panelId, tracks, fps, durationS, color }: FragmentsRowProps) {
  const [anchor, setAnchor] = useState(false);
  if (!panelId) return <div className="empty">Pick a panel in the list to see its track.</div>;
  const own = tracks.filter((t) => t.role === role && t.panel_id === panelId);
  if (!own.length) return <div className="empty">No track for {panelName(panelId)} in the {role} walk yet.</div>;
  const rate = fps && fps > 0 ? fps : 15;
  const total = durationS && durationS > 0 ? durationS : Math.max(...own.flatMap((t) => t.fragments.map((f) => (f.last_frame + 1) / rate)), 1);
  const hasAnchor = own.some((t) => t.anchor_reprompts.length > 0);
  const fragments = own.flatMap((t) => t.fragments);
  const windows = fragments.map((f) => `${formatClock(f.first_frame / rate)} to ${formatClock((f.last_frame + 1) / rate)}`);
  const noun = own[0]?.noun.replace('car ', '') ?? 'panel';

  return (
    <div className="fragments">
      <p className="fragments-line">
        <strong>{panelName(panelId)}</strong>: {windows.join(' and ')}
        {fragments.length > own.length ? `, same ${noun}, fragments ${fragments.length} to ${own.length}` : fragments.length === 1 ? ', one track' : ''}
      </p>
      <div className="timeline" role="img" aria-label={`fragments of ${panelName(panelId)}`}>
        {fragments.map((f, i) => (
          <span
            key={`${f.object_id}-${i}`}
            className="fragment"
            style={{
              left: `${(f.first_frame / rate / total) * 100}%`,
              width: `${Math.max(0.5, (((f.last_frame + 1 - f.first_frame) / rate) / total) * 100)}%`,
              background: color,
            }}
            title={`id ${f.object_id}: frames ${f.first_frame} to ${f.last_frame}`}
          />
        ))}
        <span className="timeline-axis">
          <span>0:00</span>
          <span>{formatClock(total)}</span>
        </span>
      </div>
      <div className="fragments-merged">
        {own.map((t) => (
          <span key={t.track_id} className="mono small">
            {t.track_id}: ids {t.sam_object_ids.join(', ')}
            {t.merges.length ? `, ${t.merges.length} ${t.merges.length === 1 ? 'merge' : 'merges'} (${t.merges.map((m) => m.rule).join(', ')})` : ''}
          </span>
        ))}
      </div>
      <label className={`toggle ${hasAnchor ? '' : 'disabled'}`}>
        <input type="checkbox" checked={anchor} disabled={!hasAnchor} onChange={(e) => setAnchor(e.target.checked)} />
        {hasAnchor ? (anchor ? 'anchor mask with the negative glass point' : 'pass-1 text-prompt mask') : 'no anchor pass on this track'}
      </label>
    </div>
  );
}
