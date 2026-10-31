import { describe, expect, it } from 'vitest';

import type { Event } from '../src/contracts/types.ts';
import { lookCounts, lookEngine, lookFor, readLooks, readOpenWeights } from '../src/lib/looks.ts';
import { loadClaims, loadFor, type RunFiles } from '../src/state/loader.ts';
import { initialState, reduce } from '../src/state/reducer.ts';
import type { AnyEvent } from '../src/state/types.ts';
import { deepFreeze, finalFixtureState, fixtureFiles, liveSequence, recordedEvents } from './helpers.ts';

const P1 = 'kf_pickup_right_rear_door_m1';
const R1 = 'kf_return_right_rear_door_m1';
const LP = 'kf_pickup_left_front_door_m1';
const LR = 'kf_return_left_front_door_m2';

function event(kind: Event['kind'], payload: Record<string, unknown> = {}): Event {
  return { seq: 0, ts: '2026-10-27T15:00:00Z', run_id: 'r', kind, payload, prev: '', hash: '' };
}

describe('the second looks, the open-weights log and the claim reach the state', () => {
  it('reads looks.json and open_weights.ndjson right after the match files', async () => {
    const seen = await liveSequence();
    const kinds = seen.map((e) => e.kind);
    const at = kinds.indexOf('match.derived.loaded');
    expect(at).toBeGreaterThan(0);
    expect(kinds[at + 1]).toBe('looks.loaded');
    const state = seen.reduce(reduce, initialState());
    expect(state.looks.map((l) => [l.pickup_mark, l.return_mark, l.verdict])).toEqual([
      [LP, LR, 'different'],
      [P1, R1, 'same'],
    ]);
    expect(state.looks.every((l) => l.engine === 'baseten' && l.fallback === null)).toBe(true);
    expect(state.looks.map((l) => l.request_id)).toEqual(['bt_mock_look_1', 'bt_mock_look_2']); // from each row's records
    expect(state.looks[1]!.inputs).toEqual([
      'crops/pickup/kf_pickup_right_rear_door_m1__kf_return_right_rear_door_m1.look.jpg',
      'crops/return/kf_pickup_right_rear_door_m1__kf_return_right_rear_door_m1.look.jpg',
    ]);
    expect(state.looks.map((l) => l.panel_id)).toEqual(['left_front_door', 'right_rear_door']);
    expect(state.openWeights.every((r) => r.purpose === 'second_look')).toBe(true);
    expect(state.openWeights.map((r) => [r.request_id, r.verdict, r.status])).toEqual([
      ['bt_mock_look_1', 'different', 'completed'],
      ['bt_mock_look_2', 'same', 'completed'],
    ]);
  });

  it('ties each look to the verdict about its pair, and only lowers', async () => {
    const state = await finalFixtureState();
    const byId = new Map(state.verdicts.map((v) => [v.verdict_id, v]));
    const [v1, v2, v3, v5] = [byId.get('v1')!, byId.get('v2')!, byId.get('v3')!, byId.get('v5')!];
    expect(lookFor(v1, state.looks)?.verdict).toBe('same');
    expect(v1.label).toBe('already_there');
    expect(lookFor(v5, state.looks)?.verdict).toBe('different');
    expect(v5.label).toBe('needs_your_eye');
    expect(v5.trace.rule_ids).toContain('K7');
    // The seeded chip has confidence low (K12), so it was never a provisional "already there" and got no look.
    expect(lookFor(v2, state.looks)).toBeUndefined();
    expect(v2.trace.rule_ids).toEqual(['L4', 'L3', 'K12']);
    expect(lookFor(v3, state.looks)).toBeUndefined(); // one mark, no pair, no look
    expect(lookEngine(state.looks[0]!)).toBe('Muse Spark Open (Baseten)');
    expect(lookCounts(state.looks)).toEqual({ same: 1, different: 1, unsure: 0 });
  });

  it('reads every claim at the end of a stored run, and on claim.derived', async () => {
    const state = await finalFixtureState();
    expect(state.claims.map((c) => c.claim_id)).toEqual(['claim_1', 'claim_2', 'claim_3', 'claim_4', 'claim_5']);
    expect(state.claim?.claim_id).toBe('claim_5'); // the latest, the one derived/claim.json mirrors while the run is open
    const loaded = await loadFor(event('claim.derived'), fixtureFiles());
    expect(loaded.map((e) => e.kind)).toEqual(['claims.loaded']);
    const none: RunFiles = { json: async () => null, run: async () => null };
    expect(await loadFor(event('claim.derived'), none)).toEqual([]);
  });

  it('finds a claim placed after the run closed, when claim.py writes only claim/<claim_id>.json', async () => {
    const files = fixtureFiles();
    const closed: RunFiles = { ...files, json: <T>(rel: string) => (rel === 'derived/claim.json' ? Promise.resolve(null) : files.json<T>(rel)) };
    const claims = await loadClaims(closed);
    expect(claims.map((c) => c.claim_id)).toEqual(['claim_1', 'claim_2', 'claim_3', 'claim_4', 'claim_5']);
    const ended = await loadFor(event('run.completed'), closed);
    expect(ended.map((e) => e.kind)).toEqual(['run.loaded', 'deletions.loaded', 'claims.loaded']);
    // The numbered files are read until two in a row are missing, so one gap does not hide a later claim.
    const asked: string[] = [];
    const gappy: RunFiles = {
      json: async <T>(rel: string) => {
        asked.push(rel);
        return (rel === 'claim/claim_1.json' || rel === 'claim/claim_3.json' ? ({ claim_id: rel.slice(6, 13), image: { sha256: 'a'.repeat(64), received_at: '2026-11-01T00:00:00Z', w: 1, h: 1, low_res: false } } as T) : null);
      },
      run: async () => null,
    };
    expect((await loadClaims(gappy)).map((c) => c.claim_id)).toEqual(['claim_1', 'claim_3']);
    expect(asked).toEqual(['claim/claim_1.json', 'claim/claim_2.json', 'claim/claim_3.json', 'claim/claim_4.json', 'claim/claim_5.json', 'derived/claim.json']);
  });

  it('loads nothing for the looks when a run has neither file, and survives a server without text reads', async () => {
    const bare: RunFiles = { json: async <T>(rel: string) => (rel === 'derived/verdicts.json' ? ([] as T) : null), run: async () => null };
    expect((await loadFor(event('match.derived'), bare)).map((e) => e.kind)).toEqual(['match.derived.loaded']);
    const onlyLog: RunFiles = { ...bare, text: async (rel) => (rel === 'open_weights.ndjson' ? '{"request_id":"b1","verdict":"same"}\n' : null) };
    const out = await loadFor(event('match.derived'), onlyLog);
    expect(out.map((e) => e.kind)).toEqual(['match.derived.loaded', 'looks.loaded']);
    const state = out.reduce(reduce, initialState());
    expect(state.looks).toEqual([]);
    expect(state.openWeights.map((r) => r.request_id)).toEqual(['b1']);
  });

  it('keeps the reducer pure over the new events', () => {
    let state = deepFreeze(initialState());
    const image = (received_at: string) => ({ sha256: 'a'.repeat(64), received_at, w: 1, h: 1, low_res: false });
    const events: AnyEvent[] = [
      { kind: 'looks.loaded', payload: { looks: [{ pickup_mark: 'a', return_mark: 'b', verdict: 'different', reason: 'r', engine: 'baseten' }], open_weights: '' } },
      { kind: 'claim.derived.loaded', payload: { claim: { claim_id: 'claim_1', image: image('2026-11-01T00:00:00Z') } } },
      { kind: 'claims.loaded', payload: { claims: [{ claim_id: 'claim_2', image: image('2026-11-01T00:05:00Z') }, { claim_id: 'claim_1', image: image('2026-11-01T00:00:00Z') }] } },
    ];
    for (const e of events) state = deepFreeze(reduce(state, deepFreeze(structuredClone(e))));
    expect(state.looks[0]?.verdict).toBe('different');
    expect(state.claims.map((c) => c.claim_id)).toEqual(['claim_1', 'claim_2']);
    expect(state.claim?.claim_id).toBe('claim_2');
  });

  it('ignores a claim file without an image record', () => {
    const state = reduce(initialState(), { kind: 'claim.derived.loaded', payload: { claim: { text: 'no image' } } });
    expect(state.claim).toBeNull();
    expect(reduce(initialState(), { kind: 'claims.loaded', payload: { claims: [{ text: 'no image' }, 'junk', null] } }).claims).toEqual([]);
  });

  it('the recorded log still ends at run.completed with no claim event of its own', () => {
    const kinds = recordedEvents().map((e) => e.kind);
    expect(kinds.at(-1)).toBe('run.completed');
    expect(kinds).not.toContain('claim.derived');
  });
});

describe('the readers take what the lanes write and skip what they cannot place', () => {
  it('reads look rows flat or with the answer inside a parsed reply, and pairs given as a list', () => {
    const rows = readLooks([
      { pickup_mark: 'a', return_mark: 'b', verdict: 'same', reason: 'x', engine: 'baseten', model: 'muse-spark-open', request_id: 'q1' },
      { pair: ['c', 'd'], parsed: { verdict: 'UNSURE', reason: 'too small' }, engine: 'meta-model-api' },
      { mark_ids: ['e', 'f'], answer: { verdict: 'different' } },
      { pickup_mark: 'g', verdict: 'same' }, // no return mark: skipped
      { pickup_mark: 'h', return_mark: 'i', verdict: 'maybe' }, // not a look verdict: skipped
      'junk',
      null,
    ]);
    expect(rows.map((r) => [r.pickup_mark, r.return_mark, r.verdict, r.reason])).toEqual([
      ['a', 'b', 'same', 'x'],
      ['c', 'd', 'unsure', 'too small'],
      ['e', 'f', 'different', ''],
    ]);
    expect(lookEngine(rows[0]!)).toBe('Muse Spark Open (Baseten)');
    expect(lookEngine(rows[1]!)).toBe('Muse Spark 1.3');
    expect(lookEngine(rows[2]!)).toBe('not asked'); // no engine on the row
    expect(readLooks({ looks: [{ pickup_mark: 'a', return_mark: 'b', verdict: 'same' }] })).toHaveLength(1);
    expect(readLooks(null)).toEqual([]);
  });

  it('names who answered: Spark Open, hosted Spark after a fallback, or nobody', () => {
    const [open, fallback, notAsked] = readLooks([
      { pickup_mark: 'a', return_mark: 'b', verdict: 'same', reason: 'x', engine: 'baseten', model: 'meta-models/muse-spark-open', fallback: null, records: [{ request_id: 'r1' }, { request_id: 'r2' }] },
      { pickup_mark: 'c', return_mark: 'd', verdict: 'different', reason: 'y', engine: 'meta-model-api', model: 'muse-spark-1.3', fallback_from: 'baseten', valid: true, fallback: 'Spark Open gave no answer: 503 after 3 tries' },
      { pickup_mark: 'e', return_mark: 'f', verdict: 'unsure', reason: 'not asked: no crop for e', engine: null, model: null },
    ]);
    expect(open!.request_id).toBe('r2');
    expect(lookEngine(open!)).toBe('Muse Spark Open (Baseten)');
    expect(lookEngine(fallback!)).toBe('Muse Spark 1.3, because Spark Open on Baseten gave no answer');
    expect([fallback!.fallback_from, fallback!.valid]).toEqual(['baseten', true]);
    expect(fallback!.fallback).toContain('503'); // kept for the tooltip only
    expect(lookEngine(notAsked!)).toBe('not asked');
    expect(lookEngine({ ...fallback!, fallback_from: null })).toBe('Muse Spark 1.3'); // the free text alone is no fallback
  });

  it('reads the binding fields of the rows the lanes write now', async () => {
    const state = await finalFixtureState();
    expect(state.looks.map((l) => [l.request_id, l.fallback_from, l.valid])).toEqual([
      ['bt_mock_look_1', null, true],
      ['bt_mock_look_2', null, true],
    ]);
    expect(state.openWeights.map((r) => [r.request_id, r.valid, r.error])).toEqual([
      ['bt_mock_look_1', true, null],
      ['bt_mock_look_2', true, null],
    ]);
    // A look that did not run (verdict null) changes no label, so it is not a look that stands.
    expect(readLooks([{ pickup_mark: 'a', return_mark: 'b', verdict: null, reason: 'not run: no crop for a', engine: null }])).toEqual([]);
  });

  it('reads one row per line of open_weights.ndjson and skips a broken line', () => {
    const text = ['{"request_id":"b1","model":"muse-spark-open","pickup_mark":"a","return_mark":"b","parsed":{"verdict":"same"},"status":"completed","ts":"2026-10-27T15:10:24Z"}', '{not json', '', '[1,2]', '{"request_id":"b2","status":"incomplete"}'].join('\n');
    const rows = readOpenWeights(text);
    expect(rows.map((r) => [r.request_id, r.verdict, r.status])).toEqual([
      ['b1', 'same', 'completed'],
      ['b2', null, 'incomplete'],
    ]);
    expect(readOpenWeights(null)).toEqual([]);
  });
});
