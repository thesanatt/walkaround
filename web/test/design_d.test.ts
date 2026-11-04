import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import type { LookRow, Mark, PanelMatch, Verdict } from '../src/contracts/types.ts';
import { answerLines, answersFor, findingsOf, groupsOf, sideOf, spotBox } from '../src/lib/findings.ts';
import { readLooks } from '../src/lib/looks.ts';
import { mathFor } from '../src/lib/math.ts';
import { SRC } from '../src/lib/mathlines.ts';
import { tokenize } from '../src/lib/pycode.ts';
import { comparedPanels } from '../src/lib/result.ts';
import { HERE } from './helpers.ts';

const REPO = resolve(HERE, '..', '..');
const HERO = join(REPO, 'samples', 'hero_focus5e');
const json = <T>(rel: string): T => JSON.parse(readFileSync(join(HERO, rel), 'utf8')) as T;
const verdicts = json<Verdict[]>('derived/verdicts.json');
const marks = new Map(json<Mark[]>('derived/marks.json').map((m) => [m.mark_id, m]));
const matches = json<PanelMatch[]>('derived/matches.json');
const looks: LookRow[] = readLooks(json<unknown>('derived/looks.json'));
const byId = (id: string) => verdicts.find((v) => v.verdict_id === id)!;

describe('the math panel quotes real lines', () => {
  it('finds every quoted line, word for word, at its number in the repo', () => {
    for (const line of Object.values(SRC)) {
      const file = readFileSync(join(REPO, line.file), 'utf8').split('\n');
      expect(file[line.n - 1], `${line.file}:${line.n}`).toBe(line.raw);
    }
  });

  it('colours a line without losing a character', () => {
    for (const line of Object.values(SRC)) expect(tokenize(line.raw).map(([t]) => t).join('')).toBe(line.raw);
    expect(tokenize('    d_pos = _d_pos(wa, wb)').filter(([, k]) => k === 'fn').map(([t]) => t)).toEqual(['_d_pos']);
    expect(tokenize('label = VerdictLabel.NEEDS_YOUR_EYE').map(([, k]) => k)).toEqual(['var', 'op', 'type', 'op', 'const']);
  });

  it("gives the hero's new mark the run's own numbers: 0.915 past 0.849, L5, the look 2 of 3", () => {
    const v = byId('right_rear_door_v2');
    const m = mathFor(v, marks, matches.find((x) => x.panel_id === v.panel_id), looks);
    expect(m.decision).toMatchObject({ left: '0.915', op: '>', right: '0.849' });
    expect(m.steps.map((s) => s.title)).toEqual(['The nearest pickup mark on the right rear door', 'Second look: Muse Spark Open on Baseten', 'The rule that fired']);
    expect(m.steps[1]!.rows.map((r) => r.value)).toEqual(['2 of 3', '0 of 3', '"yes"', 'different']);
    expect(m.steps[2]!.rows.map((r) => r.value)).toContain('"L5"');
  });

  it('shows the cap that lowered a mark seen at pickup only', () => {
    const v = byId('left_front_door_v3');
    const m = mathFor(v, marks, matches.find((x) => x.panel_id === v.panel_id), looks);
    const rule = m.steps[m.steps.length - 1]!;
    expect(rule.rows.map((r) => r.value)).toEqual(['yes', '"L1"', 'K15', 'yes', 'needs your eye']);
  });
});

describe('the result screen and your checks', () => {
  it("says the hero's answer from its data", () => {
    expect(answerLines(verdicts, comparedPanels(matches, verdicts).length)).toEqual({ big: '1 new mark', where: 'Right rear door', tone: 'new' });
    expect(answerLines(verdicts.filter((v) => v.label !== 'new_or_unseen'), 3)).toEqual({ big: 'No new marks', where: '3 need your eye', tone: 'eye' });
  });

  it('offers the two answers that fit where the mark was seen', () => {
    expect(answersFor(byId('left_front_door_v3'), marks)).toEqual(['gone', 'still']);
    expect(answersFor(byId('left_front_door_v4'), marks)).toEqual(['new', 'already']);
    expect(answersFor(byId('left_front_door_v1'), marks)).toEqual(['already', 'new']);
  });

  it('counts a check under its answer and leaves the sealed label alone', () => {
    const list = findingsOf(verdicts, marks, { left_front_door_v3: 'gone', right_rear_door_v1: 'still', hood_v1: 'new' });
    expect(list.map((f) => f.v.verdict_id).slice(0, 4)).toEqual(['right_rear_door_v2', 'left_front_door_v3', 'left_front_door_v4', 'right_rear_door_v1']);
    const groups = Object.fromEntries(groupsOf(list).map((g) => [g.group, g.items.map((f) => f.v.verdict_id)]));
    expect(groups['eye']).toEqual(['left_front_door_v4']);
    expect(groups['gone']).toEqual(['left_front_door_v3']);
    expect(groups['already']).toContain('right_rear_door_v1');
    expect(groups['new']).toEqual(['right_rear_door_v2']); // a check only answers needs your eye
    expect(list.find((f) => f.v.verdict_id === 'left_front_door_v3')!.system).toBe('eye');
    expect(list.some((f) => f.v.label === 'not_comparable')).toBe(false);
  });

  it("outlines the same spot in the walk that has no mark, from the run's registration", () => {
    const v = byId('right_rear_door_v2');
    const match = matches.find((x) => x.panel_id === v.panel_id);
    const pickup = sideOf(v, 'pickup', marks, json('derived/keyframes.json'), match);
    expect(pickup.own).toEqual([]);
    expect(pickup.kf?.kf_id).toBe('pickup_right_rear_door_kf1');
    expect(spotBox(pickup.spot)).toEqual([492.92, 471.2, 538.8, 497.53]);
  });
});
