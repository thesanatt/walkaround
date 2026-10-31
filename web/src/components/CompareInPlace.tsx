import { useState } from 'react';

import type { Keyframe, Mark, PanelMatch, Verdict } from '../contracts/types.ts';
import { paneKeyframes, panelFocus } from '../lib/focus.ts';
import { markLabel } from '../lib/lines.ts';
import { colorForPanel } from '../lib/panels.ts';
import { MatchCut } from './MatchCut.tsx';

// The match cut on the run page: flip between the pickup frame and the return frame placed where the pickup panel
// was, so you can see for yourself whether the joined marks sit in the same spot.
export function CompareInPlace({ panelId, match, verdicts, marks, keyframes }: { panelId: string | null; match: PanelMatch | undefined; verdicts: Verdict[]; marks: Mark[]; keyframes: Keyframe[] }) {
  const [side, setSide] = useState<'pickup' | 'return'>('pickup');
  if (!panelId) return null;
  const byId = new Map(marks.map((m) => [m.mark_id, m]));
  const focus = panelFocus(panelId, match, verdicts);
  if (!focus.pair) return null;
  const { a, b } = paneKeyframes(panelId, match, focus, byId, keyframes);
  if (!a?.crop || !b?.crop) return null;
  const markA = byId.get(focus.pair.pickup_mark);
  return (
    <div className="compare-in-place">
      <div className="role-tabs" role="group" aria-label="compare in place">
        <button type="button" className={side === 'pickup' ? 'on' : ''} aria-pressed={side === 'pickup'} onClick={() => setSide('pickup')}>
          pickup
        </button>
        <button type="button" className={side === 'return' ? 'on' : ''} aria-pressed={side === 'return'} onClick={() => setSide('return')}>
          return, placed where the pickup panel was
        </button>
      </div>
      <MatchCut kfA={a} kfB={b} markA={markA} markB={byId.get(focus.pair.return_mark)} color={colorForPanel(panelId)} phase={side === 'pickup' ? 'pickup' : 'join'} labelA={markA ? markLabel(markA) : null} />
      <p className="small muted">The dashed box is where the pickup mark was. The return frame is only moved and scaled for this view; the match itself is computed in the panel's own frame.</p>
    </div>
  );
}
