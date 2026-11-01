import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { ClipTrail } from '../src/components/ClipTrail.tsx';
import { StaysPutList } from '../src/components/StaysPutList.tsx';
import { StaysPutPair } from '../src/components/StaysPutPair.tsx';
import type { MarkTrack, NeighbourTest, StaysPutNeighbour } from '../src/contracts/types.ts';
import { staysPutLine } from '../src/lib/lines.ts';
import { NEIGHBOUR_NOTE, neighbourCaption, neighbourLine, neighbourMarkers } from '../src/lib/neighbours.ts';
import { finalFixtureState } from './helpers.ts';

const html = (el: Parameters<typeof renderToStaticMarkup>[0]) => renderToStaticMarkup(el);

function neighbour(id: string, kind: string, used = true): StaysPutNeighbour {
  return { object_id: id, kind, ref: kind, frames: used ? 90 : 40, used, residual_full_px: used ? 4.4 : null, residual_near_px: used ? 1.7 : null, centres: [[44, 100, 120], [45, 102, 121]] };
}

// The S550 door reflection and the pair C trunk paint as walkaround/staysput.neighbour_test gives them (docs/STAYS_PUT_ANCHORS.md).
const SLIDING: NeighbourTest = {
  verdict: 'sliding', ratio_full: 4.768, ratio_near: 3.829, mark_full_px: 27.4, mark_near_px: 7.9, neighbours_full_px: 5.75, neighbours_near_px: 2.06,
  near_s: 0.5, near_frames: 15, used: 6, reason: null,
  neighbours: [neighbour('3', 'door handle'), neighbour('4', 'door handle'), neighbour('5', 'side mirror'), neighbour('6', 'wheel'), neighbour('7', 'tail light'), neighbour('8', 'wheel'), neighbour('9', 'license plate', false)],
};
const FIXED: NeighbourTest = { ...SLIDING, verdict: 'fixed', ratio_full: 0.586, ratio_near: 0.839, used: 5 };

async function tracks(): Promise<{ mark: MarkTrack; control: MarkTrack }> {
  const state = await finalFixtureState();
  const [mark, control] = state.markTracks as [MarkTrack, MarkTrack];
  return { mark: { ...mark, verdict: 'sliding', drift_rms: 0.21, b1: 0.3, b2: 0.01, neighbour_test: FIXED }, control: { ...control, neighbour_test: SLIDING } };
}

describe('Stays Put by neighbours, as the page says it', () => {
  it('words the test as the sheet does', async () => {
    const { mark, control } = await tracks();
    expect(neighbourLine(mark)).toBe('neighbour test: moves with its 5 neighbours, 0.6x their wobble over the clip and 0.8x within 0.5 s; fixed');
    expect(neighbourLine(control)).toBe('neighbour test: slides against its 6 neighbours, 4.8x their wobble over the clip and 3.8x within 0.5 s; sliding');
    const between = { ...mark, neighbour_test: { ...FIXED, verdict: 'inconclusive' as const, ratio_full: 1.8, ratio_near: 1.2, used: 6 } };
    expect(neighbourLine(between)).toBe("neighbour test: 1.8x its 6 neighbours' wobble over the clip and 1.2x within 0.5 s; inconclusive");
    const few = { ...mark, neighbour_test: { ...FIXED, verdict: 'inconclusive' as const, ratio_full: null, ratio_near: null, used: 3, reason: '3 neighbours tracked through the clip, 4 needed' } };
    expect(neighbourLine(few)).toBe('neighbour test: 3 neighbours tracked through the clip, 4 needed; inconclusive');
    expect(neighbourCaption(few)).toBe('Not compared with its neighbours: 3 neighbours tracked through the clip, 4 needed');
    expect(neighbourLine({ ...mark, neighbour_test: { ...FIXED, verdict: 'no_track' } })).toBe('neighbour test: SAM did not follow the mark through enough of the clip');
    expect(neighbourLine({ ...mark, neighbour_test: { ...FIXED, verdict: 'not_run' } })).toBeNull();
    expect(neighbourLine({ ...mark, neighbour_test: undefined })).toBeNull(); // a clip planned before the test existed
    expect(staysPutLine(mark, false)).toBe('drift 0.21, b1 0.30, b2 0.01, not panned; neighbour test: moves with its 5 neighbours, 0.6x their wobble over the clip and 0.8x within 0.5 s; fixed');
    expect(staysPutLine({ ...mark, neighbour_test: undefined }, false)).toBe('drift 0.21, b1 0.30, b2 0.01, not panned');
  });

  it('shows it next to the clip, says it changes no verdict yet, and rings the neighbours', async () => {
    const { mark, control } = await tracks();
    expect(neighbourCaption(mark)).toBe('Moves with its 5 neighbours, 0.6x their wobble (0.8x within 0.5 s)');
    expect(neighbourCaption(control)).toBe('Slides against its 6 neighbours, 4.8x their wobble (3.8x within 0.5 s)');
    const pair = html(createElement(StaysPutPair, { track: mark, control, mark: undefined, color: '#5fc2ad', calibrated: false, numbers: true }));
    expect(pair).toContain('Moves with its 5 neighbours, 0.6x their wobble (0.8x within 0.5 s). ' + NEIGHBOUR_NOTE);
    expect(pair).toContain('Slides against its 6 neighbours, 4.8x their wobble (3.8x within 0.5 s). ' + NEIGHBOUR_NOTE);
    expect(pair).toContain('sp-neighbours uncalibrated');
    expect(html(createElement(StaysPutPair, { track: mark, control, mark: undefined, color: '#5fc2ad', calibrated: true }))).toContain('sp-neighbours fixed');
    expect(neighbourMarkers(control, 45).map((r) => [r.kind, r.x, r.y, r.used])).toEqual([...SLIDING.neighbours.map((n) => [n.kind, 102, 121, n.used])]);
    expect(neighbourMarkers(control, 12)).toEqual([]);
    const clip = html(createElement(ClipTrail, { mt: control, color: '#d9a84a', caption: 'the control clip', atFrame: 45 }));
    expect(clip.split('class="nb-ring"').length - 1).toBe(7);
    expect(clip.split('stroke-dasharray').length - 1).toBe(1); // the plate was not tracked through enough of the clip
    const list = html(createElement(StaysPutList, { markTracks: [mark, control], calibrated: false }));
    expect(list).toContain('Moves with its 5 neighbours');
    expect(list).toContain('neighbour test: slides against its 6 neighbours');
  });
});
