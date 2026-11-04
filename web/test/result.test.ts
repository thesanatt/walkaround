import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { zoomWindow } from '../src/components/Frame.tsx';
import type { Mark, PanelMatch, Verdict } from '../src/contracts/types.ts';
import { RULES } from '../src/lib/rules.ts';
import { byEyeRows, comparedPanels, creditLine, headline, notComparedRows, subline, WHY, whyLine } from '../src/lib/result.ts';
import { RUN_DIR } from './helpers.ts';

const derived = <T>(name: string): T => JSON.parse(readFileSync(join(RUN_DIR, 'derived', name), 'utf8')) as T;

describe('the result page answer', () => {
  it('gives every needs-your-eye rule a line, and only for rules that exist', () => {
    for (const id of Object.keys(WHY)) expect(RULES[id], id).toBeDefined();
    const base = derived<Verdict[]>('verdicts.json')[0]!;
    const eye: Verdict = { ...base, label: 'needs_your_eye', rule_id: 'L4', trace: { ...base.trace, rule_ids: ['L4', 'K12', 'K4'] } };
    expect(whyLine(eye)).toBe(WHY['K12']);
    expect(whyLine({ ...eye, label: 'already_there', rule_id: 'L3', distance_cm_est: 3.2 })).toBe('In the same place at pickup and at return, about 3 cm apart.');
  });
});

describe('the finding frame', () => {
  it('zooms around the boxes inside the crop, and shows a frame with no box whole', () => {
    expect(zoomWindow(1000, 700, [], true)).toEqual([0, 0, 1000, 700]);
    const [x, y, w, h] = zoomWindow(1000, 700, [[980, 10, 990, 20]], true);
    expect(x + w).toBeLessThanOrEqual(1000);
    expect(y).toBeGreaterThanOrEqual(0);
    expect(w / h).toBeCloseTo(4 / 3);
    const [, , w3, h3] = zoomWindow(1000, 700, [[480, 340, 500, 352]], true, 280, 7, 2 / 3);
    expect(w3).toBe(280);
    expect(w3 / h3).toBeCloseTo(3 / 2);
  });
});

describe('the renter-first answer (second UI pass, Nov 1)', () => {
  const base = derived<Verdict[]>('verdicts.json')[0]!;
  const v = (label: Verdict['label'], panel: string, ids: string[], extra: Partial<Verdict> = {}): Verdict => ({
    ...base,
    verdict_id: `${label}-${panel}-${ids.join('')}`,
    label,
    panel_id: panel as Verdict['panel_id'],
    rule_id: ids[0] ?? 'L4',
    mark_ids: [],
    trace: { ...base.trace, rule_ids: ids },
    reasons: [],
    ...extra,
  });

  it('names the panel of a new mark, and never says nothing new while spots wait for the renter', () => {
    expect(headline([v('new_or_unseen', 'left_front_door', ['L2'])], 5)).toBe('1 new mark: left front door.');
    expect(headline([v('new_or_unseen', 'hood', ['L2']), v('new_or_unseen', 'left_front_door', ['L2'])], 5)).toBe('2 new marks: hood and left front door.');
    expect(headline([v('already_there', 'trunk_lid', ['L3'])], 3)).toBe('Nothing new on the 3 panels both walks show clearly.');
    expect(headline([v('already_there', 'trunk_lid', ['L3']), v('needs_your_eye', 'trunk_lid', ['L4', 'K4'])], 3)).toBe(
      'No new damage confirmed on the 3 panels both walks show clearly.',
    );
    expect(headline([], 0)).toBe('No panel was filmed clearly enough in both walks to compare.');
  });

  it('counts panels, not marks, in the second line', () => {
    const vs = [
      v('already_there', 'trunk_lid', ['L3']),
      v('needs_your_eye', 'trunk_lid', ['L4', 'K15']),
      v('not_comparable', 'roof', ['L6', 'G1']),
      v('not_comparable', 'hood', ['L6', 'G1', 'G4']),
      v('not_comparable', 'windshield', ['L6', 'G4']),
    ];
    expect(subline(vs, 3)).toBe("1 mark was there at pickup. 1 spot needs your eye. 3 panels weren't filmed clearly enough in both walks to compare.");
    expect(comparedPanels([], vs)).toEqual(['trunk_lid']);
    expect(notComparedRows(vs).map((r) => r.reason)).toEqual(['not filmed long enough in both walks', 'not filmed long enough in both walks', 'the two views did not line up']);
  });

  it('credits each Meta model only for what the run stored, and says code made the call', () => {
    const marks = new Map(derived<Mark[]>('marks.json').map((m) => [m.mark_id, m]));
    const mark = [...marks.values()][0]!;
    const one = v('new_or_unseen', mark.panel_id, ['L2'], { mark_ids: [mark.mark_id] });
    expect(creditLine(one, marks, [])).toMatch(/^SAM 3\.1 tracked this panel; Muse Spark (found the mark|found it while counting marks|heard)/);
    expect(creditLine(one, marks, [])).toMatch(/code made the call\.$/);
    expect(creditLine(one, marks, [])).not.toContain('Muse Spark Open');
    const counter = { pickup_mark: 'p', return_mark: mark.mark_id, mark_id: mark.mark_id, kind: 'counter' as const, verdict: 'different' as const, engine: 'baseten', panel_id: null, reason: '', model: null, fallback_from: null, valid: true, fallback: null, request_id: null, inputs: [] };
    expect(creditLine(one, marks, [counter])).toContain('Muse Spark Open saw no such mark at that spot in the pickup walk');
    expect(creditLine({ ...one, label: 'needs_your_eye' }, marks, [])).toMatch(/code left the call to you\.$/);
  });

});

describe('the R29 and R30 words (L8, K21, the gates G11 to G13, check by eye)', () => {
  const base = derived<Verdict[]>('verdicts.json')[0]!;
  const v = (label: Verdict['label'], panel: string, ids: string[]): Verdict => ({
    ...base,
    verdict_id: `${label}-${panel}`,
    label,
    panel_id: panel as Verdict['panel_id'],
    rule_id: ids[0] ?? 'L4',
    mark_ids: [],
    trace: { ...base.trace, rule_ids: ids },
    reasons: [],
    distance_cm_est: null,
  });
  const pm = (panel: string, state: PanelMatch['state'], kinds: ('glass' | 'wheels' | 'debris')[] = [], n = 0): PanelMatch => ({
    panel_id: panel as PanelMatch['panel_id'],
    state,
    reasons: state === 'comparable' ? [] : ['G1'],
    kf_pair: null,
    aspect_diff: null,
    mask_iou: null,
    handle_residual: null,
    frame_used: 'moment',
    pairs: [],
    unmatched_pickup: [],
    unmatched_return: [],
    groups: [],
    by_eye: kinds.map((kind) => ({ kind, text: kind, mark_ids: Array.from({ length: n }, (_, i) => `m${i}`) })),
  });

  it('gives K18 to K21 a plain line, and L8 its own already-there line', () => {
    for (const id of ['K18', 'K19', 'K20', 'K21']) expect(WHY[id], id).toBeTruthy();
    expect(whyLine(v('needs_your_eye', 'hood', ['L4', 'K21']))).toBe(WHY['K21']);
    expect(whyLine(v('already_there', 'mirrors', ['L8']))).toMatch(/^Boxed in one walk only, but lining up the two frames/);
    expect(RULES['L8']!.label).toBe('already there, found in both crops');
    expect(RULES['K21']!.label).toBe('the other walk did not see this spot');
  });

  it('words the gates G11, G12 and G13 on the not-compared rows', () => {
    const rows = notComparedRows([v('not_comparable', 'right_rear_fender', ['L6', 'G11']), v('not_comparable', 'windshield', ['L6', 'G12']), v('not_comparable', 'headlights', ['L6', 'G13'])]);
    expect(rows.map((r) => r.reason)).toEqual(['filmed at very different sizes in the two walks', 'mostly blurred for privacy in one walk', 'the left one in one walk and the right one in the other']);
  });

  it('lists glass, wheels and debris once each to check by eye, and never counts glass or wheels as compared', () => {
    const matches = [
      pm('left_front_door', 'comparable'),
      pm('hood', 'comparable', ['debris'], 3),
      pm('windshield', 'not_comparable', ['glass']),
      pm('rear_window', 'comparable', ['glass'], 1),
      pm('left_front_wheel', 'comparable', ['wheels'], 2),
      pm('right_front_wheel', 'not_comparable', ['wheels']),
    ];
    const rows = byEyeRows(matches);
    expect(rows.map((r) => r.key)).toEqual(['glass', 'wheels', 'debris-hood']);
    expect(rows[0]!.title).toBe('windshield and rear window');
    expect(rows[1]!.text).toBe('rims and tires are not checked mark by mark; the check covers the car body');
    expect(rows[2]!.text).toBe('dirt on this panel, 3 spots not listed one by one');
    expect(comparedPanels(matches, [])).toEqual(['left_front_door', 'hood']);
    expect(byEyeRows([pm('left_front_door', 'comparable')])).toEqual([]);
  });
});
