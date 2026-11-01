import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { SecondLookLine } from '../src/components/SecondLookLine.tsx';
import type { Verdict } from '../src/contracts/types.ts';
import { lookFor, loweredByLook, loweredIds, readLooks } from '../src/lib/looks.ts';
import { RULES } from '../src/lib/rules.ts';

// Lead decision 5 of 11/1: the two-question counter-look, looks.json rows with kind "counter2" and mark_seen.
const rows = readLooks([
  {
    pickup_mark: 'pickup_left_front_door_kf2',
    return_mark: 'return_left_front_door_kf1_m1',
    kind: 'counter2',
    mark_id: 'return_left_front_door_kf1_m1',
    mark_seen: 'no',
    verdict: 'different',
    engine: 'baseten',
    model: 'meta-models/muse-spark-open-1026',
    valid: true,
    reason: 'the box shows clean paint and a reflection',
    input_refs: [{ path: 'crops/return/a.look.jpg' }, { path: 'crops/pickup/b.look.jpg' }],
  },
]);

function verdict(ids: string[]): Verdict {
  return {
    verdict_id: 'left_front_door_v2',
    panel_id: 'left_front_door',
    label: ids.length > 1 ? 'needs_your_eye' : 'new_or_unseen',
    rule_id: ids[0]!,
    mark_ids: ['return_left_front_door_kf1_m1'],
    distance_units: null,
    distance_cm_est: null,
    reasons: [],
    trace: { rule_ids: ids, thresholds: {}, sam_request_ids: [], spark_request_ids: [], event_seqs: [], config_hash: null },
    size: null,
  } as Verdict;
}

describe('the two-question counter-look', () => {
  it('keeps kind counter2 and part 1, and ties the row to the one-mark verdict', () => {
    const look = rows[0]!;
    expect(look.kind).toBe('counter2');
    expect(look.mark_seen).toBe('no');
    expect(lookFor(verdict(['L4', 'L2', 'K17']), rows)).toBe(look);
  });

  it('names K17, K16 or both only when code put them on the verdict', () => {
    const look = rows[0]!;
    expect(loweredIds(verdict(['L4', 'L2', 'K17']), look)).toEqual(['K17']);
    expect(loweredIds(verdict(['L4', 'L2', 'K16', 'K17']), look)).toEqual(['K16', 'K17']);
    expect(loweredByLook(verdict(['L2']), look)).toBe(false);
    const v = verdict(['L4', 'L2', 'K17']);
    const html = renderToStaticMarkup(createElement(SecondLookLine, { look, lowered: loweredByLook(v, look), ids: loweredIds(v, look) }));
    expect(html).toContain('second look at the mark and the same spot in the pickup walk');
    expect(html).toContain('damage in the box: <strong>no</strong>');
    expect(html).toContain('same mark at the pickup spot: <strong>no</strong>');
    expect(html).toContain('K17: this lowered the label to needs your eye');
    expect(RULES['K17']?.label).toBe('the second look saw no damage in the box');
  });
});
