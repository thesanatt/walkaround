import { describe, expect, it } from 'vitest';

import type { LookRow, PanelTrack, Verdict } from '../src/contracts/types.ts';
import { byStrength, panelCards, panelFocus } from '../src/lib/focus.ts';
import { majorityLooks, readLooks } from '../src/lib/looks.ts';
import { trackIndex } from '../src/lib/trackIndex.ts';
import { sceneHeading } from '../src/pages/KioskPage.tsx';
import { sceneByName } from '../src/kiosk/scenes.ts';
import { initialState } from '../src/state/reducer.ts';

// hero_focus4's pickup car_door lane: SAM id 1 is the front door to frame 246, then two rear door tracks reuse it.
const track = (track_id: string, panel_id: string, frags: [number, number, string][]): PanelTrack =>
  ({ track_id, role: 'pickup', noun: 'car door', panel_id, sam_object_ids: [...new Set(frags.map((f) => f[2]))], fragments: frags.map(([first_frame, last_frame, object_id]) => ({ first_frame, last_frame, object_id })) }) as unknown as PanelTrack;
const TRACKS = [
  track('pickup_car_door_01', 'left_front_door', [[0, 246, '1']]),
  track('pickup_car_door_02', 'left_rear_door', [[71, 99, '2']]),
  track('pickup_car_door_05', 'left_rear_door', [[507, 546, '1']]),
  track('pickup_car_door_06', 'left_rear_door', [[555, 591, '1']]),
];

describe('the walk overlay reads the track that holds the frame', () => {
  it('names a reused SAM id by the track of that frame, as the door check decided', () => {
    const index = trackIndex(TRACKS);
    expect(index.at('pickup', 'car_door', '1', 90)?.panel_id).toBe('left_front_door');
    expect(index.at('pickup', 'car_door', '1', 520)?.panel_id).toBe('left_rear_door');
    expect(index.at('pickup', 'car_door', '1', 300)).toBeUndefined();
  });

  it('hides a track the run puts off the car, and objects in no track once the walk has tracks', () => {
    const index = trackIndex(TRACKS, ['pickup_car_door_02']);
    expect(index.shows('pickup', 'car_door', '1', 90)).toBe(true);
    expect(index.shows('pickup', 'car_door', '2', 90)).toBe(false); // the SUV's door behind the Focus
    expect(index.shows('pickup', 'car_door', '1', 300)).toBe(false); // no track holds id 1 here
    expect(index.shows('return', 'car_door', '3', 10)).toBe(true); // no return track yet: the lanes still stream
  });
});

const look = (over: Partial<LookRow>): LookRow =>
  ({ pickup_mark: 'p1', return_mark: 'r1', panel_id: 'hood', verdict: 'same', reason: '', engine: 'baseten', model: null, fallback_from: null, valid: true, fallback: null, request_id: null, inputs: [], kind: null, mark_id: null, mark_seen: null, ...over }) as LookRow;

describe('the second look counts looks, not repeat answers', () => {
  it('folds each look asked again into one row with the majority answer', () => {
    const rows = majorityLooks([
      look({ request_id: 'a' }),
      look({ request_id: 'b' }),
      look({ pickup_mark: 'p2', return_mark: 'r2', verdict: 'same' }),
      look({ pickup_mark: 'p2', return_mark: 'r2', verdict: 'different', request_id: 'c' }),
      look({ pickup_mark: 'p2', return_mark: 'r2', verdict: 'different', request_id: 'd' }),
      look({ pickup_mark: 'p3', return_mark: 'r3', verdict: 'same' }),
      look({ pickup_mark: 'p3', return_mark: 'r3', verdict: 'unsure' }),
      look({ kind: 'counter2', mark_id: 'r9', return_mark: 'r9', verdict: 'different', mark_seen: 'yes' }),
      look({ kind: 'counter2', mark_id: 'r9', return_mark: 'r9', verdict: 'different', mark_seen: 'yes' }),
    ]);
    expect(rows.map((r) => [r.pickup_mark, r.verdict, r.answers])).toEqual([
      ['p1', 'same', 2],
      ['p2', 'different', 3],
      ['p3', 'unsure', 2], // a split is unsure: same must outnumber the rest to keep the pair
      ['p1', 'different', 2],
    ]);
    expect(rows[1]!.request_id).toBe('c'); // the first answer on the majority's side
    expect(rows[3]!.mark_seen).toBe('yes');
  });

  it('reads looks.json rows (ask 1 is the repeat) as one look each', () => {
    const stored = [
      { pickup_mark: 'p1', return_mark: 'r1', verdict: 'same', engine: 'baseten' },
      { pickup_mark: 'p1', return_mark: 'r1', verdict: 'same', engine: 'baseten', ask: 1 },
    ];
    expect(readLooks(stored).length).toBe(1);
  });
});

const verdict = (verdict_id: string, label: Verdict['label'], mark_ids: string[] = [verdict_id]): Verdict =>
  ({ verdict_id, panel_id: 'right_rear_door', label, rule_id: 'L2', mark_ids, distance_units: null, distance_cm_est: null, reasons: [], trace: { rule_ids: [] }, size: null }) as unknown as Verdict;

describe('a panel leads with its strongest verdict', () => {
  const speck = verdict('v1', 'needs_your_eye');
  const strip = verdict('v2', 'new_or_unseen');
  const already = verdict('v3', 'already_there');
  const gone = verdict('v4', 'not_seen_at_return');

  it('orders new, not seen at return, already there, needs your eye', () => {
    expect(byStrength([speck, already, gone, strip]).map((v) => v.verdict_id)).toEqual(['v2', 'v4', 'v3', 'v1']);
    const focus = panelFocus('right_rear_door', undefined, [speck, strip]);
    expect(focus.verdict).toBe(strip);
    expect(panelCards('right_rear_door', focus, [speck, strip]).map((v) => v.verdict_id)).toEqual(['v2', 'v1']);
  });

  it('gives the verdict scene the heading of that verdict', () => {
    const state = { ...initialState(), verdicts: [speck, strip] };
    expect(sceneHeading(sceneByName('tape'), state, 'right_rear_door')).toBe('new or unseen');
  });
});
