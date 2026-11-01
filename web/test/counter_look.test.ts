import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { SecondLookLine } from '../src/components/SecondLookLine.tsx';
import type { Verdict } from '../src/contracts/types.ts';
import { lookFor, loweredByLook, readLooks } from '../src/lib/looks.ts';
import { RULES } from '../src/lib/rules.ts';

// The one-walk counter-look (lead 11/1, third): looks.json rows with kind "counter" and verdicts capped by K16.
const rows = readLooks([
  { pickup_mark: 'p1', return_mark: 'r1', verdict: 'same', engine: 'baseten', model: 'meta-models/muse-spark-open-1026', valid: true, reason: 'same chip' },
  {
    pickup_mark: 'pickup_rear_bumper_kf2',
    return_mark: 'return_rear_bumper_kf1_m1',
    kind: 'counter',
    mark_id: 'return_rear_bumper_kf1_m1',
    verdict: 'same',
    engine: 'baseten',
    model: 'meta-models/muse-spark-open-1026',
    valid: true,
    reason: 'a round sensor dot shows in the box',
    input_refs: [{ path: 'crops/return/a.look.jpg' }, { path: 'crops/pickup/b.look.jpg' }],
  },
]);

function verdict(over: Partial<Verdict>): Verdict {
  return {
    verdict_id: 'rear_bumper_v2',
    panel_id: 'rear_bumper',
    label: 'needs_your_eye',
    rule_id: 'L4',
    mark_ids: ['return_rear_bumper_kf1_m1'],
    distance_units: null,
    distance_cm_est: null,
    reasons: [],
    trace: { rule_ids: ['L4', 'L5', 'K16'], thresholds: {}, sam_request_ids: [], spark_request_ids: [], event_seqs: [], config_hash: null },
    size: null,
    ...over,
  } as Verdict;
}

describe('the counter-look', () => {
  it('keeps kind and mark_id and ties the row to the one-mark verdict, never to a pair', () => {
    expect(rows.map((r) => r.kind)).toEqual([null, 'counter']);
    const v = verdict({});
    expect(lookFor(v, rows)?.mark_id).toBe('return_rear_bumper_kf1_m1');
    expect(lookFor(verdict({ mark_ids: ['p1', 'r1'] }), rows)?.kind).toBeNull();
    expect(lookFor(verdict({ mark_ids: ['other'] }), rows)).toBeUndefined();
  });

  it('says K16 lowered the label only when code put K16 on it', () => {
    const look = rows[1]!;
    expect(loweredByLook(verdict({}), look)).toBe(true);
    expect(loweredByLook(verdict({ label: 'new_or_unseen', rule_id: 'L5', trace: { ...verdict({}).trace, rule_ids: ['L5'] } }), look)).toBe(false);
    const html = renderToStaticMarkup(createElement(SecondLookLine, { look, lowered: true }));
    expect(html).toContain('second look at the same spot in the pickup walk');
    expect(html).toContain('K16: this lowered the label to needs your eye');
    expect(html).toContain('return crop</a>');
    expect(html).toContain('pickup crop with the spot</a>');
    expect(RULES['K16']?.label).toBe('the other walk may show it');
  });
});
