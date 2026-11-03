import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { MarkCard, zoomWindow } from '../src/components/MarkCard.tsx';
import { clipWindow } from '../src/components/PanelClip.tsx';
import { pinsFor, ResultMap } from '../src/components/ResultMap.tsx';
import type { Keyframe, Mark, PanelMatch, PanelTrack, Verdict } from '../src/contracts/types.ts';
import { RULES } from '../src/lib/rules.ts';
import { answerSentences, byEyeRows, comparedPanels, creditLine, EYE_KINDS, eyeGroups, eyeKind, headline, notComparableLine, notComparedRows, subline, tally, WHY, whyLine } from '../src/lib/result.ts';
import { RUN_DIR } from './helpers.ts';

const derived = <T>(name: string): T => JSON.parse(readFileSync(join(RUN_DIR, 'derived', name), 'utf8')) as T;

describe('the result page answer', () => {
  it('counts one verdict as one mark and says it in plain words', () => {
    const verdicts = derived<Verdict[]>('verdicts.json');
    const t = tally(verdicts);
    const n = (label: string) => verdicts.filter((v) => v.label === label).length;
    expect(t.already).toBe(n('already_there'));
    expect(t.eye).toBe(n('needs_your_eye'));
    const one = answerSentences({ already: 1, fresh: 0, eye: 1, gone: 0, notComparable: 0 });
    expect(one.join(' ')).toBe('1 mark was already there at pickup. 0 new. 1 needs your eye.');
    expect(answerSentences({ already: 2, fresh: 1, eye: 3, gone: 1, notComparable: 0 }).join(' ')).toBe(
      '2 marks were already there at pickup. 1 new. 3 need your eye. 1 not seen at return.',
    );
  });

  it('names the most common gate for the panels it could not compare', () => {
    const base = derived<Verdict[]>('verdicts.json')[0]!;
    const nc = (ids: string[]): Verdict => ({ ...base, label: 'not_comparable', rule_id: 'L6', trace: { ...base.trace, rule_ids: ['L6', ...ids] } });
    expect(notComparableLine([nc(['G1']), nc(['G1', 'G4']), nc(['G6'])])).toBe('3 panels could not be compared. Most were not filmed long enough in both walks.');
    expect(notComparableLine([nc(['G4'])])).toBe('1 panel could not be compared. It did not line up between the two walks.');
    expect(notComparableLine([])).toBeNull();
  });

  it('gives every needs-your-eye rule a line, and only for rules that exist', () => {
    for (const id of Object.keys(WHY)) expect(RULES[id], id).toBeDefined();
    const base = derived<Verdict[]>('verdicts.json')[0]!;
    const eye: Verdict = { ...base, label: 'needs_your_eye', rule_id: 'L4', trace: { ...base.trace, rule_ids: ['L4', 'K12', 'K4'] } };
    expect(whyLine(eye)).toBe(WHY['K12']);
    expect(whyLine({ ...eye, label: 'already_there', rule_id: 'L3', distance_cm_est: 3.2 })).toBe('In the same place at pickup and at return, about 3 cm apart.');
  });
});

describe('the mark card', () => {
  it('zooms around the boxes inside the crop, and shows a frame with no box whole', () => {
    expect(zoomWindow(1000, 700, [], true)).toEqual([0, 0, 1000, 700]);
    const [x, y, w, h] = zoomWindow(1000, 700, [[980, 10, 990, 20]], true);
    expect(x + w).toBeLessThanOrEqual(1000);
    expect(y).toBeGreaterThanOrEqual(0);
    expect(w / h).toBeCloseTo(4 / 3);
  });

  it('draws both walks of an "already there" pair, with its label and why', () => {
    const verdicts = derived<Verdict[]>('verdicts.json');
    const marks = new Map(derived<Mark[]>('marks.json').map((m) => [m.mark_id, m]));
    const keyframes = derived<Keyframe[]>('keyframes.json');
    const v = verdicts.find((x) => x.label === 'already_there')!;
    const page = renderToStaticMarkup(createElement(MarkCard, { v, marks, keyframes, roles: ['pickup', 'return'] }));
    expect(page).toContain('class="label already_there"');
    expect(page.split('<figure class="mark-frame">').length - 1).toBe(2);
    expect(page).toContain('In the same place at pickup and at return');
    expect(page).not.toContain('no mark here');
    expect(page).toContain('There at pickup');
    expect(page).toContain('code made the call.');
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

  it('groups needs your eye by reason, most specific first, each with something to do', () => {
    const vs = [
      v('needs_your_eye', 'trunk_lid', ['L4', 'K12', 'K4']),
      v('needs_your_eye', 'rear_bumper', ['L4', 'K12']),
      v('needs_your_eye', 'left_rear_door', ['L7', 'G4']),
      v('needs_your_eye', 'hood', ['L4', 'K7', 'K15']),
    ];
    const groups = eyeGroups(vs);
    expect(groups.map((g) => g.kind.key)).toEqual(['shape', 'pair', 'dirt', 'unsure']);
    expect(groups.every((g) => g.kind.todo.length > 0)).toBe(true);
    expect(groups[2]!.kind.title(1)).toBe('1 spot that may be dirt, water or glare');
    for (const k of EYE_KINDS) for (const id of k.ids) expect(RULES[id], id).toBeDefined();
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

  it('puts one pin per answer card and one per panel with spots to check', () => {
    const clear = [v('new_or_unseen', 'left_front_door', ['L2']), v('already_there', 'trunk_lid', ['L3'])];
    const eye = [v('needs_your_eye', 'trunk_lid', ['L4', 'K4']), v('needs_your_eye', 'trunk_lid', ['L4', 'K15'])];
    const pins = pinsFor(clear, eye, (id) => `eye-${id}`);
    expect(pins.map((p) => p.text)).toEqual(['1', '2', '2']);
    expect(pins[0]!.target).toBe(`card-${clear[0]!.verdict_id}`);
    expect(pins[2]!.words).toBe('Needs your eye: trunk lid, 2 spots');
    const svg = renderToStaticMarkup(createElement(ResultMap, { compared: ['trunk_lid'], notCompared: ['roof'], pins, onPin: () => undefined }));
    expect(svg).toContain('New: left front door');
    expect(svg.match(/class="rm-pin /g)?.length).toBe(3);
  });

  it('cuts the panel clip to the SAM track that holds the frame', () => {
    const track = { role: 'return', panel_id: 'left_front_door', fragments: [{ object_id: '1', first_frame: 180, last_frame: 240 }] } as unknown as PanelTrack;
    expect(clipWindow([track], 'return', 'left_front_door', 200, 15)).toEqual([180, 223]);
    expect(clipWindow([track], 'pickup', 'left_front_door', 200, 15)).toBeNull();
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

  it('gives K18 to K21 a plain line and a group, and L8 its own already-there line', () => {
    for (const id of ['K18', 'K19', 'K20', 'K21']) expect(WHY[id], id).toBeTruthy();
    expect(whyLine(v('needs_your_eye', 'hood', ['L4', 'K21']))).toBe(WHY['K21']);
    expect(eyeKind(v('needs_your_eye', 'hood', ['L4', 'K21', 'K15'])).key).toBe('unseen');
    expect(eyeKind(v('needs_your_eye', 'hood', ['L4', 'K18'])).key).toBe('pair');
    expect(eyeKind(v('needs_your_eye', 'hood', ['L4', 'K20'])).key).toBe('unsure');
    expect(whyLine(v('already_there', 'mirrors', ['L8']))).toMatch(/^Boxed in one walk only, but lining up the two frames/);
    expect(RULES['L8']!.label).toBe('already there, found in both crops');
    expect(RULES['K21']!.label).toBe('the other walk did not see this spot');
  });

  it('words the gates G11, G12 and G13 on the not-compared rows', () => {
    const rows = notComparedRows([v('not_comparable', 'right_rear_fender', ['L6', 'G11']), v('not_comparable', 'windshield', ['L6', 'G12']), v('not_comparable', 'headlights', ['L6', 'G13'])]);
    expect(rows.map((r) => r.reason)).toEqual(['filmed at very different sizes in the two walks', 'mostly blurred for privacy in one walk', 'the left one in one walk and the right one in the other']);
    expect(notComparableLine([v('not_comparable', 'right_rear_fender', ['L6', 'G11'])])).toBe('1 panel could not be compared. It was filmed at very different sizes in the two walks.');
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
