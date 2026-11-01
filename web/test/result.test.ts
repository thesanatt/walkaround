import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { MarkCard, zoomWindow } from '../src/components/MarkCard.tsx';
import type { Keyframe, Mark, Verdict } from '../src/contracts/types.ts';
import { RULES } from '../src/lib/rules.ts';
import { answerSentences, notComparableLine, tally, WHY, whyLine } from '../src/lib/result.ts';
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
    expect(page).not.toContain('no matching mark found');
  });
});
