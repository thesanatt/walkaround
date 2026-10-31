import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { ApiError } from '../src/api/client.ts';
import { BlurBeat } from '../src/components/BlurBeat.tsx';
import { CoverageCar, revealOrder, slotStates } from '../src/components/CoverageCar.tsx';
import { HowChecked } from '../src/components/HowChecked.tsx';
import { MatchCut } from '../src/components/MatchCut.tsx';
import { conditionsText, DatesCard, gapText, MatchPane } from '../src/components/MatchPane.tsx';
import { NarrationChips } from '../src/components/NarrationChips.tsx';
import { SecondLookLine } from '../src/components/SecondLookLine.tsx';
import { StaysPutList } from '../src/components/StaysPutList.tsx';
import { controlTrack, StaysPutPair } from '../src/components/StaysPutPair.tsx';
import { topupErrorText, topupGains } from '../src/components/TopUpBanner.tsx';
import type { Claim, ClaimExit, Keyframe, MarkTrack } from '../src/contracts/types.ts';
import { howChecked } from '../src/lib/checks.ts';
import { claimCandidates, claimPhotoPath, claimWorkSize, effectiveExit, EXIT_WORDS, placementLine, samVoteLine, sideLine, sparkNamedLine } from '../src/lib/claim.ts';
import { applyCut, matchCutTransform, panelAnchor } from '../src/lib/geometry.ts';
import { markLabel, staysPutCalibrated, staysPutCaption } from '../src/lib/lines.ts';
import { lookEngine, openWeightsFailed, readLooks, readOpenWeights } from '../src/lib/looks.ts';
import { ClaimFrame, claimFrame, ClaimPhoto, ClaimResult, ClaimSteps } from '../src/pages/ClaimPage.tsx';
import { initialState, reduce } from '../src/state/reducer.ts';
import type { RunState } from '../src/state/types.ts';
import { finalFixtureState, RUN_DIR } from './helpers.ts';

const html = (el: Parameters<typeof renderToStaticMarkup>[0]) => renderToStaticMarkup(el);
const count = (text: string, needle: string) => text.split(needle).length - 1;
const claimFile = (id: string) => JSON.parse(readFileSync(join(RUN_DIR, 'claim', `${id}.json`), 'utf8')) as Claim;

describe('how it was checked, counted from the run', () => {
  it('names each model with what it did in the fixture run', async () => {
    const lines = howChecked(await finalFixtureState());
    expect(lines.map((l) => [l.model, l.did, l.state])).toEqual([
      // The fixture froze today's thresholds (staysput.cap_enabled false), so the control is not called a reflection.
      ['SAM 3.1', 'tracked 3 panels in 2 walks and 1 mark through its own clip, plus 1 control clip', 'done'],
      ['Muse Spark 1.3', 'found 8 marks in 6 blurred crops, 1 of them from your words', 'done'],
      ['Muse Voice Transcribe', 'pinned 3 of 3 spoken notes to a panel', 'done'],
      ['Muse Spark Open on Baseten', 'took a second look at 2 matches: 1 same and 1 different', 'done'],
      ['Code, no model', 'matched 3 pairs and wrote 5 labels', 'done'],
    ]);
    // The original sits in quarantine during the run, so the row says what holds for every frame kept or shown.
    expect(lines[0]!.detail).toBe('31 of 32 lanes answered, 1 not run; the plates and faces it found are blurred in every frame kept or shown');
    const calibrated = await finalFixtureState();
    calibrated.run = { ...calibrated.run!, thresholds: { staysput: { cap_enabled: true } } };
    expect(howChecked(calibrated)[0]!.did).toBe('tracked 3 panels in 2 walks and 1 mark through its own clip, plus 1 reflection control');
    expect(lines[2]!.detail).toBe('1 note gave Muse Spark a mark to box; 1 note named a mark it had already boxed');
    expect(lines[3]!.link).toEqual({ rel: 'open_weights.ndjson', text: 'open_weights.ndjson' });
  });

  it('waits before the run has data and says not run when a model never ran', async () => {
    expect(howChecked(initialState()).map((l) => l.state)).toEqual(['waiting', 'waiting', 'waiting', 'waiting', 'waiting']);
    const done = await finalFixtureState();
    const noLooks: RunState = { ...done, looks: [], openWeights: [] };
    const open = howChecked(noLooks)[3]!;
    expect([open.did, open.state, open.link]).toEqual(['not run in this run', 'not_run', null]);
    // Hosted Spark only (engine "spark"): no Baseten request, so no log line either.
    const metaLooks: RunState = { ...done, looks: done.looks.map((l) => ({ ...l, engine: 'meta-model-api', model: 'muse-spark-1.3' })), openWeights: [] };
    expect(howChecked(metaLooks)[1]!.detail).toContain('it took a second look at 2 pairs');
    expect(howChecked(metaLooks)[3]!.did).toBe('not run in this run');
    // The binding field decides the fallback; the free-text fallback is only the tooltip.
    const fellBack: RunState = { ...done, looks: done.looks.map((l) => ({ ...l, engine: 'meta-model-api', model: 'muse-spark-1.3', fallback_from: 'baseten' as const, fallback: null })) };
    expect(howChecked(fellBack)[3]!.did).toBe('gave no answer, so hosted Muse Spark 1.3 took 2 second looks');
    const textOnly: RunState = { ...done, looks: done.looks.map((l) => ({ ...l, engine: 'meta-model-api', model: 'muse-spark-1.3', fallback: 'Spark Open gave no answer: 503' })), openWeights: [] };
    expect(howChecked(textOnly)[3]!.did).toBe('not run in this run');
    const failedLine = { request_id: null, model: null, purpose: 'second_look', verdict: null, status: 'failed', valid: false, error: '503 Service Unavailable', ts: null };
    const someFailed: RunState = { ...done, openWeights: [...done.openWeights, failedLine] };
    expect(howChecked(someFailed)[3]!.detail).toContain('1 request got no answer');
    const noTurns: RunState = { ...done, narration: { per_role: {}, mode_applied: false } };
    expect(howChecked(noTurns)[2]!.did).toBe('returned your words without turns, so nothing was pinned');
    const silent = { mode_applied: null, turns: null, text: 'no narration found: the clip has no audio track' };
    const noAudio: RunState = { ...done, narration: null, voice: { pickup: silent, return: silent } };
    expect(howChecked(noAudio)[2]!.did).toBe('no audio track in these walks');
    const pickupCheck: RunState = { ...done, run: { ...done.run!, kind: 'pickup_check' }, verdicts: [], matches: [] };
    expect(howChecked(pickupCheck)[4]!.did).toBe('kept the dated frames; a pickup check has no return walk to match');
  });

  it('says "no audio track" only for the walks that have none (pair C: narrated pickup, silent return)', () => {
    const ev = (seq: number, kind: string, payload: Record<string, unknown>) => ({ seq, ts: '2026-11-01T00:00:00Z', run_id: 'p', kind, payload, prev: '', hash: '' }) as Parameters<typeof reduce>[1];
    let s = reduce(initialState(), { kind: 'run.loaded', payload: { run_id: 'p', kind: 'pair', noun_list: [], thresholds: {}, clocks: {} } });
    s = reduce(s, ev(0, 'stage', { name: 'staysput', pct: 82 }));
    s = reduce(s, ev(1, 'voice.done', { role: 'pickup', mode_applied: true, turns: 3, sha256: 'x' }));
    s = reduce(s, ev(2, 'voice.done', { role: 'return', mode_applied: null, text: 'no narration found: the clip has no audio track' }));
    const waiting = howChecked(s)[2]!;
    expect([waiting.did, waiting.state]).toEqual(['waiting for the narration step', 'waiting']);
    expect(waiting.detail).toBe('pins what you say to the panel on screen; the return walk has no audio track');
    s = reduce(s, ev(3, 'run.partial', { missing: ['narration'], text: '' }));
    const failed = howChecked({ ...s, completed: true })[2]!;
    expect([failed.did, failed.state]).toEqual(['not run in this run', 'not_run']);
    expect(failed.detail).toContain('the return walk has no audio track');
  });

  it('says Spark Open was tried when every request failed and no look stands', async () => {
    const done = await finalFixtureState();
    // second_look.py writes verdict null for a look nothing answered; the reader drops it, so no look stands.
    const looks = readLooks([{ pickup_mark: 'a', return_mark: 'b', panel_id: 'hood', verdict: null, reason: 'not run: Muse Spark 1.3 failed', engine: null, fallback: 'Spark Open gave no answer: 503; Muse Spark 1.3 failed' }]);
    expect(looks).toEqual([]);
    const line = '{"engine":"baseten","error":"503 Service Unavailable","model":"meta-models/muse-spark-open","request_id":null,"status":"failed","valid":false,"verdict":null}\n';
    const open = howChecked({ ...done, looks, openWeights: readOpenWeights(line.repeat(3)) })[3]!;
    expect([open.did, open.state]).toEqual(['was tried and gave no answer', 'not_run']);
    expect(open.detail).toBe('3 requests got no usable answer, so no label changed; each one is in the provenance log');
    expect(open.link).toEqual({ rel: 'open_weights.ndjson', text: 'open_weights.ndjson' });
    const schemaMiss = readOpenWeights('{"status":"schema_miss","valid":false,"error":"verdict: not one of same, different, unsure"}\n');
    expect(schemaMiss.map(openWeightsFailed)).toEqual([true]);
    expect(readOpenWeights('{"status":"completed","valid":true,"error":null}\n').map(openWeightsFailed)).toEqual([false]);
  });

  it('reads a fallback row in the binding shape (fallback_from, no fallback text)', async () => {
    const done = await finalFixtureState();
    const looks = readLooks([
      { pickup_mark: 'a', return_mark: 'b', panel_id: 'hood', verdict: 'same', reason: 'same chip', engine: 'meta-model-api', model: 'muse-spark-1.3', request_id: 'r1', latency_ms: 900, fallback_from: 'baseten', valid: true },
    ]);
    expect(lookEngine(looks[0]!)).toBe('Muse Spark 1.3, because Spark Open on Baseten gave no answer');
    expect(howChecked({ ...done, looks })[3]!.did).toBe('gave no answer, so hosted Muse Spark 1.3 took 1 second look');
  });

  it('renders five rows with the open-weights log linked', async () => {
    const page = html(createElement(HowChecked, { state: await finalFixtureState() }));
    expect(count(page, '<dt>')).toBe(5);
    expect(page).toContain('href="open_weights.ndjson"');
    expect(html(createElement(HowChecked, { state: await finalFixtureState(), rows: 2 }))).toContain('Muse Spark 1.3');
  });
});

describe('the second look on each verdict it touched', () => {
  it('prints the engine, the answer and its reason, and links the log', async () => {
    const state = await finalFixtureState();
    const pane = (panelId: string) =>
      html(
        createElement(MatchPane, {
          panelId,
          match: state.matches.find((m) => m.panel_id === panelId),
          panel: state.panels.find((p) => p.panel_id === panelId),
          keyframes: state.keyframes,
          marks: state.marks,
          verdicts: state.verdicts,
          videos: state.videos,
          media: state.media,
          run: state.run,
          looks: state.looks,
        }),
      );
    const door = pane('right_rear_door');
    expect(count(door, 'class="second-look')).toBe(1); // the seeded chip got no look
    expect(door).toContain('second look: Muse Spark Open (Baseten), <strong>same</strong>: the same light scuff below the handle line');
    expect(door).toContain('href="open_weights.ndjson"');
    expect(door).toContain('chip, S, from your words'); // the seeded chip's box
    expect(door).toContain('(seeded from your words)');
    const left = pane('left_front_door');
    expect(left).toContain('<strong>different</strong>: the return chip is about twice the size');
    expect(left).toContain('(K7: this lowered the label to needs your eye)');
  });
});

describe('a second look that fell back or could not be asked says so', () => {
  it('prints hosted Spark with the reason it answered, or the reason nothing was asked', async () => {
    const state = await finalFixtureState();
    const look = state.looks.find((l) => l.verdict === 'same')!;
    const fallback = html(createElement(SecondLookLine, { look: { ...look, engine: 'meta-model-api', model: 'muse-spark-1.3', fallback_from: 'baseten', fallback: 'Spark Open gave no answer: 503 after 3 tries' }, lowered: false }));
    expect(fallback).toContain('second look: Muse Spark 1.3, because Spark Open on Baseten gave no answer, <strong>same</strong>');
    expect(fallback).toContain('title="Spark Open gave no answer: 503 after 3 tries"');
    expect(fallback).toContain('href="open_weights.ndjson"'); // the failed Baseten tries are in the log
    // A look that did not run puts no K7 on the label (second_look.py, verdict null); only the verdict's rule ids say K7.
    const notAsked = html(createElement(SecondLookLine, { look: { ...look, engine: null, model: null, verdict: 'unsure', reason: 'not asked: no crop for kf_x_m1', inputs: [] }, lowered: false }));
    expect(notAsked).toContain('second look: not asked: no crop for kf_x_m1');
    expect(notAsked).not.toContain('K7');
    expect(notAsked).not.toContain('open_weights.ndjson');
    const seen = html(createElement(SecondLookLine, { look, lowered: false }));
    expect(seen).toContain('what it saw: <a href="crops/pickup/kf_pickup_right_rear_door_m1__kf_return_right_rear_door_m1.look.jpg">pickup crop</a>');
    expect(seen).not.toContain('K7');
    expect(html(createElement(SecondLookLine, { look: { ...look, verdict: 'unsure' }, lowered: true }))).toContain('(K7: this lowered the label to needs your eye)');
    expect(html(createElement(SecondLookLine, { look: { ...look, verdict: 'unsure' }, lowered: false }))).not.toContain('K7');
  });
});

describe('narration chips', () => {
  it('plays each chip and says which note seeded a mark', async () => {
    const state = await finalFixtureState();
    const page = html(createElement(NarrationChips, { narration: state.narration, roles: ['pickup', 'return'], onSeek: () => undefined, marks: state.marks }));
    expect(count(page, 'class="audio-chip"')).toBe(3);
    expect(page).toContain('your words name the scuff Muse Spark found on the right rear door');
    expect(page).toContain('seeded from your words: Muse Spark boxed a chip on the right rear door where you said');
    expect(count(page, 'seed-badge')).toBe(2);
    expect(page).toContain('said while the right rear door was on screen');
    expect(page).toContain('about the hood');
    expect(page).toContain('no narration found in this walk');
  });
});

describe('the coverage car', () => {
  it('shades every slot by its walk, a top-up counting for pickup', async () => {
    const state = await finalFixtureState();
    const before = slotStates(state.coverage, state.panels, 'pickup');
    const after = slotStates(state.coverage, state.panels, 'pickup', true);
    expect(before['rear_bumper']).toBe('glimpsed');
    expect(after['rear_bumper']).toBe('filmed');
    expect(after['roof']).toBe('not_run');
    expect(slotStates(state.coverage, state.panels, 'return', false)['rear_bumper']).toBe('not_run');
    expect(topupGains(state.coverage!)).toEqual([{ panel: 'rear_bumper', state: 'filmed', seconds: 9.6 }]);
    const order = revealOrder(state.panels, 'pickup');
    expect(order).toHaveLength(22);
    expect(order[0]).toBe('right_rear_door'); // first seen at 0:00
    expect(order.indexOf('roof')).toBeGreaterThan(order.indexOf('front_bumper')); // never filmed, so last
  });

  it('draws 22 slots, the callouts and the legend', async () => {
    const state = await finalFixtureState();
    const page = html(createElement(CoverageCar, { coverage: state.coverage, panels: state.panels, role: 'pickup', withTopups: true, callouts: ['roof'] }));
    expect(count(page, 'class="cc-panel ')).toBe(22);
    expect(page).toContain('roof: not run');
    expect(page).toContain('aria-label="pickup walk coverage: ');
    for (const s of ['filmed', 'glimpsed', 'not filmed', 'not run', 'not seen']) expect(page).toContain(`</svg>${s}</span>`);
    const unknown = html(createElement(CoverageCar, { coverage: null, panels: [], role: 'pickup' }));
    expect(unknown).toContain('coverage: not computed yet');
    expect(count(unknown, 'cc-panel pending')).toBe(22);
  });
});

describe('the match cut', () => {
  const kf = (patch: Partial<Keyframe>): Keyframe =>
    ({
      kf_id: 'k',
      crop: { path: 'c.jpg', sha256: null, w: 1280, h: 860, scale: 0.5, dx: 0, dy: 0 },
      box_src: [0, 0, 2560, 1720],
      moments: { cx: 640, cy: 430, mu20: 90000, mu02: 40000, mu11: 0, theta: 0, sigma_inv_half: [[1, 0], [0, 1]] },
      ...patch,
    }) as Keyframe;

  it('lands the return panel centre on the pickup panel centre at the same spread', () => {
    const a = kf({});
    expect(matchCutTransform(a, a)).toEqual({ s: 1, tx: 0, ty: 0 });
    const shifted = kf({ moments: { ...a.moments!, cx: 600, cy: 450 } });
    const t = matchCutTransform(a, shifted);
    expect(applyCut([600, 450], t)).toEqual([640, 430]);
    const bigger = kf({ moments: { ...a.moments!, mu20: 360000, mu02: 160000 } });
    expect(matchCutTransform(a, bigger).s).toBeCloseTo(2 / 3, 6); // half the spread would be 0.5; held at 2/3
    const noMoments = kf({ moments: null });
    expect(panelAnchor(noMoments)?.c).toEqual([640, 430]); // the panel box, mapped into crop pixels
    expect(matchCutTransform(a, kf({ crop: null, moments: null }))).toEqual({ s: 1, tx: 0, ty: 0 });
  });

  it('shows the pickup box, then the return frame with a ghost, then the join', async () => {
    const state = await finalFixtureState();
    const kfA = state.keyframes.find((k) => k.kf_id === 'kf_pickup_right_rear_door')!;
    const kfB = state.keyframes.find((k) => k.kf_id === 'kf_return_right_rear_door')!;
    const markA = state.marks.find((m) => m.mark_id === 'kf_pickup_right_rear_door_m1');
    const markB = state.marks.find((m) => m.mark_id === 'kf_return_right_rear_door_m1');
    const props = { kfA, kfB, markA, markB, color: '#5fc2ad', labelA: '"small scuff, right rear door"' };
    const pickup = html(createElement(MatchCut, { ...props, phase: 'pickup' }));
    expect(pickup).toMatch(/class="cut-pickup"[^>]*opacity="1"/);
    expect(pickup).toContain('&quot;small scuff, right rear door&quot;');
    const ret = html(createElement(MatchCut, { ...props, phase: 'return' }));
    expect(ret).toMatch(/class="cut-pickup"[^>]*opacity="0"/);
    expect(ret).toContain('class="cut-ghost"');
    expect(ret).not.toContain('small scuff');
    expect(count(html(createElement(MatchCut, { ...props, phase: 'join', still: true })), 'class="join-line"')).toBe(1);
    expect(html(createElement(MatchCut, { ...props, kfB: { ...kfB, crop: null }, phase: 'join' }))).toContain('needs both keyframe crops');
  });
});

describe('the two dates', () => {
  it('sets each date beside its time, with the seal and the gap', async () => {
    const state = await finalFixtureState();
    const kfA = state.keyframes.find((k) => k.kf_id === 'kf_pickup_right_rear_door')!;
    const kfB = state.keyframes.find((k) => k.kf_id === 'kf_return_right_rear_door')!;
    const card = html(createElement(DatesCard, { kfA, kfB, videos: state.videos, media: state.media, run: state.run }));
    expect(count(card, 'class="date-day mono"')).toBe(2);
    expect(card).toContain('sealed 7 minutes after the camera time');
    expect(card).toContain('same day, 45 minutes later');
    expect(html(createElement(DatesCard, { kfA, kfB, videos: state.videos, media: state.media, run: state.run, stacked: true }))).toContain('dates-card stacked');
    expect(gapText('2026-10-27T14:00:00Z', '2026-10-31T20:02:00Z')).toBe('4 days later');
    expect(gapText(null, '2026-10-31T20:02:00Z')).toBeNull();
    // A return clock set before the pickup clock is printed as written, with the direction said.
    expect(gapText('2026-10-27T15:00:00Z', '2026-10-27T14:15:00Z')).toBe('same day, 45 minutes earlier');
    expect(gapText('2026-10-31T15:00:00Z', '2026-10-27T15:00:00Z')).toBe('4 days earlier');
    expect(gapText('2026-10-27T15:00:00Z', '2026-10-27T15:01:00Z')).toBe('same day, 1 minute later');
  });

  it('names the flags behind G2', async () => {
    const state = await finalFixtureState();
    const a = state.keyframes.find((k) => k.kf_id === 'kf_pickup_hood');
    const b = state.keyframes.find((k) => k.kf_id === 'kf_return_hood');
    expect(conditionsText(a, b)).toBe(' (glare on the return frame)');
    expect(conditionsText(undefined, undefined)).toBe('');
  });
});

describe('Stays Put captions', () => {
  it('reads the calibration from the thresholds the run froze', async () => {
    const state = await finalFixtureState();
    expect(staysPutCalibrated(state.run)).toBe(false); // config/thresholds.json today: cap_enabled false
    expect(staysPutCalibrated({ thresholds: { staysput: { cap_enabled: true } } })).toBe(true);
    expect(staysPutCalibrated({ thresholds: {} })).toBe(false); // a run frozen before the key existed
    expect(staysPutCalibrated(null)).toBe(false);
  });

  it('says what each track decided once calibrated, and nothing more', async () => {
    const state = await finalFixtureState();
    const [mark, control] = state.markTracks as [MarkTrack, MarkTrack];
    expect(staysPutCaption(mark, true)).toBe('Stays put: paint');
    expect(staysPutCaption(control, true)).toBe('Slides with the camera: reflection');
    expect(staysPutCaption({ ...mark, verdict: 'sliding' }, true)).toBe('Slides with the camera: likely a reflection');
    expect(staysPutCaption({ ...mark, verdict: 'inconclusive', panned: true }, true)).toBe('Between paint and reflection: not enough to call');
    expect(staysPutCaption({ ...mark, verdict: 'inconclusive', b_pass: false }, true)).toBe('Too little camera motion to tell');
    expect(staysPutCaption({ ...mark, verdict: 'inconclusive', drift_rms: null }, true)).toBe('Too few frames with the whole panel in view to tell');
    expect(staysPutCaption({ ...mark, verdict: 'no_track' }, true)).toBe('SAM 3.1 lost it in this clip');
    expect(controlTrack(state.markTracks)?.mark_id).toBe('kf_pickup_hood_m1');
    const pair = html(createElement(StaysPutPair, { track: mark, control, mark: state.marks.find((m) => m.mark_id === mark.mark_id), color: '#5fc2ad', calibrated: true, numbers: true }));
    expect(pair).toContain('Stays put: paint');
    expect(pair).toContain('Slides with the camera: reflection');
    expect(pair).toContain('a reflection, tracked the same way');
    expect(pair).toContain('drift 0.03, b1 0.21, b2 0.02, not panned; fixed');
    expect(html(createElement(StaysPutPair, { track: undefined, control: undefined, mark: undefined, color: '#000', calibrated: true }))).toContain('No Stays Put clip for this panel');
    expect(markLabel(state.marks.find((m) => m.mark_id === mark.mark_id)!)).toBe('scuff, S');
    expect(markLabel(state.marks.find((m) => m.mark_id === 'kf_pickup_right_rear_door_m2')!)).toBe('chip, S, from your words');
  });

  it('states no verdict while uncalibrated: real paint labelled sliding is not called a reflection', async () => {
    const state = await finalFixtureState();
    const [mark, control] = state.markTracks as [MarkTrack, MarkTrack];
    expect(staysPutCaption(mark, false)).toBe('Tracked through its own clip. Stays Put is not calibrated yet and changes no label.');
    expect(staysPutCaption(control, false)).toBe('Tracked the same way. Stays Put is not calibrated yet.');
    expect(staysPutCaption({ ...mark, verdict: 'no_track' }, false)).toBe('SAM 3.1 lost it in this clip'); // a fact, not a verdict
    const pair = html(createElement(StaysPutPair, { track: mark, control, mark: state.marks.find((m) => m.mark_id === mark.mark_id), color: '#5fc2ad', calibrated: false, numbers: true }));
    expect(pair).toContain('the control clip, tracked the same way');
    expect(pair).toContain('drift 0.03, b1 0.21, b2 0.02, not panned<');
    expect(pair).toContain('sp-caption uncalibrated');
    for (const claim of ['paint', 'reflection', '; fixed', '; sliding']) expect(pair).not.toContain(claim);
    // pair C's trunk chips: real paint, drift 0.21, labelled sliding by the uncalibrated rule
    const chips: MarkTrack = { ...mark, mark_id: 'chip', verdict: 'sliding', drift_rms: 0.21, b1: 0.3, b2: 0.01 };
    const chipMark = { ...state.marks[0]!, mark_id: 'chip', type: 'chip' as const, panel_id: 'trunk_lid' as const };
    const list = html(createElement(StaysPutList, { markTracks: [chips, control], marks: [chipMark], calibrated: false }));
    expect(list).toContain('pickup chip on the trunk lid');
    expect(list).toContain('drift 0.21, b1 0.30, b2 0.01, not panned');
    expect(list).toContain(', the control clip');
    expect(list).not.toContain('reflection');
    expect(html(createElement(StaysPutList, { markTracks: [chips], marks: [chipMark], calibrated: true }))).toContain('Slides with the camera: likely a reflection');
  });
});

describe('claims, exits in plain words', () => {
  it('has words for every exit the contract allows', () => {
    const exits: (ClaimExit | 'E2b')[] = ['E1', 'E2', 'E2b', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'];
    for (const e of exits) expect(EXIT_WORDS[e]?.length ?? 0).toBeGreaterThan(20);
  });

  it('reads the placed claim', () => {
    const c = claimFile('claim_1');
    expect(samVoteLine(c)).toBe('14 image requests, the strongest names right rear door; they agree with Muse Spark');
    expect(sideLine(c)).toBe('from your words');
    expect(placementLine(c)).toBe('on the whole panel mask, give or take 0.10 units');
    expect(claimPhotoPath(c)).toBe('claim/claim_1.blurred.jpg');
    expect(claimCandidates(c)).toEqual([]);
  });

  it('offers the two panels after E2 and both sides when no side was named', () => {
    const e2 = claimFile('claim_4');
    expect(e2.exit).toBe('E2');
    expect(claimCandidates(e2)).toEqual(['right_rear_door', 'right_rear_fender']); // Spark's, then the vote winner's
    const both = claimFile('claim_5');
    expect(both.exit).toBeNull(); // claim.py leaves exit empty when it shows both sides
    expect(effectiveExit(both)).toBe('E2b');
    expect(claimCandidates(both)).toEqual(['left_rear_door', 'right_rear_door']); // car doors that fit "rear"
    const noWinnerPanel: Claim = { ...e2, naming: { ...e2.naming, sam_votes: [{ noun: 'fender', status: 'completed', matches: 1, score: 0.6, winner: true, panel_id: null }] }, text: 'scratch, front' };
    expect(claimCandidates(noWinnerPanel)).toEqual(['right_rear_door', 'left_front_fender', 'right_front_fender']);
    expect(effectiveExit(claimFile('claim_1'))).toBeNull();
    expect(claimPhotoPath({ ...e2, claim_id: null })).toBeNull();
  });

  it('renders an exit with its rule id, and the chooser after E2', () => {
    const e5 = html(createElement(ClaimResult, { claim: claimFile('claim_2'), ctx: null }));
    expect(e5).toContain('E5 cannot place');
    expect(e5).toContain('The claim mark cannot be placed precisely on the panel');
    expect(e5).toContain('title="E5 cannot place: ');
    const e2 = html(createElement(ClaimResult, { claim: claimFile('claim_4'), ctx: null, onPick: () => undefined }));
    expect(count(e2, 'class="btn small"')).toBe(2);
    expect(e2).toContain('>right rear fender</button>');
    expect(e2).toContain('which panel does the bill mean?');
    const both = html(createElement(ClaimResult, { claim: claimFile('claim_5'), ctx: null, onPick: () => undefined }));
    expect(both).toContain('E2b side unknown');
    expect(both).toContain('>left rear door</button>');
    expect(both).not.toContain('Placed against the pickup walk');
    const notRun: Claim = { ...claimFile('claim_1'), placement: { method: 'none', w_claim: null, r_claim: null }, match: null, size: null, coverage_state: 'not_run', paragraph: 'Right rear door. The pickup run did not finish checking this panel, so nothing was compared.' };
    expect(html(createElement(ClaimResult, { claim: notRun, ctx: null }))).toContain('Not compared.');
    const placed = html(createElement(ClaimResult, { claim: claimFile('claim_1'), ctx: null }));
    expect(placed).toContain('Placed against the pickup walk.');
    expect(placed).toContain('about 1.1 in, between 0.9 and 1.3 in (ruler: the license plate, 12 in)');
  });

  it('words E2 as claim.py does, and gives one candidate its button (one tap confirms)', () => {
    expect(EXIT_WORDS['E2']).toContain('your pick is recorded as your input');
    expect(EXIT_WORDS['E2']).not.toContain('sheet'); // a claim never rewrites a closed run's sheet
    expect(EXIT_WORDS['E2']).not.toContain('named different panels'); // an unfinished vote is E2 too
    // claim.py fires E2 when the vote did not finish: Spark and the partial vote can name the same panel.
    const e2 = claimFile('claim_4');
    const one: Claim = { ...e2, naming: { ...e2.naming, sam_votes: e2.naming.sam_votes.map((v) => ({ ...v, winner: v['noun'] === 'car door', panel_id: v['noun'] === 'car door' ? 'right_rear_door' : null })) } };
    expect(claimCandidates(one)).toEqual(['right_rear_door']);
    const page = html(createElement(ClaimResult, { claim: one, ctx: null, onPick: () => undefined }));
    expect(count(page, 'class="btn small"')).toBe(1);
    expect(page).toContain('is this the panel the bill means?');
    // No panel at all to offer: say what to do instead of asking for a pick with no button.
    const none: Claim = { ...e2, naming: { spark: null, sam_votes: [], agreed: false } };
    expect(html(createElement(ClaimResult, { claim: none, ctx: null, onPick: () => undefined }))).toContain('No panel to offer from this answer.');
  });

  it('says what Muse Spark named without contradicting the paragraph', () => {
    // claim_5: Spark and the vote agree on "car door" with no side, so naming.spark is empty (claim.py keeps one slot only).
    const both = claimFile('claim_5');
    expect(sparkNamedLine(both)).toBe('a car door; which one is not named');
    expect(html(createElement(ClaimSteps, { claim: both }))).not.toContain('no car panel');
    expect(sparkNamedLine(claimFile('claim_1'))).toBe('right rear door');
    expect(sparkNamedLine({ ...claimFile('claim_4'), naming: { ...claimFile('claim_4').naming, spark: null } })).toBe('no single panel');
    expect(sparkNamedLine({ ...claimFile('claim_4'), exit: 'E1', naming: { spark: null, sam_votes: [], agreed: false } })).toBe('no car panel');
  });

  it('draws the claim box in the working copy claim.py measured it on (long edge at most 2048 px)', () => {
    expect(claimWorkSize({ w: 4032, h: 3024 })).toEqual([2048, 1536]);
    expect(claimWorkSize({ w: 3024, h: 4032 })).toEqual([1536, 2048]);
    expect(claimWorkSize({ w: 900, h: 675 })).toEqual([900, 675]); // never upscaled
    expect(claimWorkSize({ w: 4097, h: 3 })).toEqual([2048, 1]); // Python's round(1.4996) is 1, and never 0
    const c1 = claimFile('claim_1');
    const big: Claim = { ...c1, image: { ...c1.image, w: 4032, h: 3024 }, roi: { box: [1000, 700, 1100, 800], source: 'drawn' } };
    const photo = html(createElement(ClaimPhoto, { claim: big, src: 'x.jpg' }));
    expect(photo).toContain('viewBox="0 0 2048 1536"'); // the box sits 48.8% across, where claim.py found it
    expect(photo).toContain('x="1000"');
  });

  it('shows the served, pixelated claim photo once the claim comes back, never the local file', () => {
    const placed = html(createElement(ClaimResult, { claim: claimFile('claim_1'), ctx: null }));
    expect(placed).toContain('src="claim/claim_1.blurred.jpg"');
    expect(placed).not.toContain('blob:');
    expect(html(createElement(ClaimResult, { claim: { ...claimFile('claim_1'), claim_id: null }, ctx: null }))).toContain('no photo to show');
  });
});

describe('the pickup frame a claim lands on', () => {
  const ctxOf = (state: RunState) => ({ keyframes: state.keyframes, panels: state.panels, marks: state.marks, coverage: state.coverage, videos: state.videos });

  it('takes the keyframe claim.py used and boxes the pickup mark only when code joined it', async () => {
    const state = await finalFixtureState();
    const c1 = claimFile('claim_1');
    const hit = claimFrame(c1, ctxOf(state));
    expect([hit?.kf.kf_id, hit?.mark?.mark_id]).toEqual(['kf_pickup_right_rear_door', 'kf_pickup_right_rear_door_m1']);
    // Past tau_eye_r claim.py lists the claim mark as unmatched (L5): the frame stays, the box goes.
    const far: Claim = { ...c1, match: { ...c1.match!, pairs: [{ ...c1.match!.pairs[0]!, d: 0.9, tau_eye_r: 0.4 }], unmatched_return: ['claim_1_m1'] } };
    const farHit = claimFrame(far, ctxOf(state));
    expect([farHit?.kf.kf_id, farHit?.mark]).toEqual(['kf_pickup_right_rear_door', undefined]);
  });

  it('finds a panel filmed only in a top-up clip, and dates it from that clip', async () => {
    const state = await finalFixtureState();
    const c1 = claimFile('claim_1');
    const base = state.keyframes.find((k) => k.kf_id === 'kf_pickup_right_rear_door')!;
    const topupKf: Keyframe = { ...base, kf_id: 'kf_topup_1_rrd', role: 'topup_1', pts: 3.2 };
    const viaPair: Claim = { ...c1, match: { ...c1.match!, pairs: [], kf_pair: ['kf_topup_1_rrd', 'claim_1'] } };
    expect(claimFrame(viaPair, { keyframes: [topupKf], marks: [], coverage: null })?.kf.kf_id).toBe('kf_topup_1_rrd');
    // No match (E4 and the like): the coverage's best keyframe over the pickup side, as claim._pickup_keyframe.
    const noMatch: Claim = { ...c1, match: null };
    const coverage = { ...state.coverage!, per_role: { ...state.coverage!.per_role, topup_1: { right_rear_door: { state: 'filmed' as const, state_basis: 'c' as const, seconds: 9.9, best_kf: 'kf_topup_1_rrd', reason: '' } } } };
    expect(claimFrame(noMatch, { keyframes: [...state.keyframes, topupKf], marks: [], coverage })?.kf.kf_id).toBe('kf_topup_1_rrd');
    expect(claimFrame(noMatch, { keyframes: [topupKf], marks: [], coverage: null })?.kf.kf_id).toBe('kf_topup_1_rrd'); // the best-scored pickup-side frame
    const video = { ...state.videos['pickup']!, role: 'topup_1', ffprobe: { ...state.videos['pickup']!.ffprobe!, creation_time: '2026-10-27T15:30:00Z' } };
    const frame = html(createElement(ClaimFrame, { claim: viaPair, ctx: { keyframes: [topupKf], panels: [], marks: [], coverage: null, videos: { topup_1: video } } }));
    expect(frame).toContain('this frame from top-up clip 1 at 0:03');
    expect(frame).toContain('Oct 27, 2026');
    expect(frame).toContain('in top-up clip 1 at 0:03');
    // Without that clip's record no camera time is claimed either way.
    const unknown = html(createElement(ClaimFrame, { claim: viaPair, ctx: { keyframes: [topupKf], panels: [], marks: [], coverage: null, videos: {} } }));
    expect(unknown).not.toContain('camera time not present');
  });
});

describe('the smaller copy fixes', () => {
  it('prints the reason a top-up was refused', () => {
    expect(topupErrorText(new ApiError(429, { reason: 'this run reached its spend ceiling of 500 cents', pickup_left: 3, pairs_left: 1 }))).toBe('The top-up did not run: this run reached its spend ceiling of 500 cents.');
    expect(topupErrorText(new ApiError(429, null))).toBe('Live runs are paused for today, so the top-up did not run.');
  });

  it('counts only what the frame checks found on the blur beat, and claims nothing about storage it cannot back', async () => {
    const state = await finalFixtureState();
    const kf = state.keyframes.find((k) => k.kf_id === 'kf_pickup_right_rear_door')!;
    const beat = html(createElement(BlurBeat, { kf }));
    expect(beat).toContain('This crop was blurred before it was saved or sent to Muse Spark.');
    expect(beat).toContain('what the plate and face checks found on this frame');
    expect(beat).not.toContain('No plate was detected');
    expect(beat).not.toContain('before storage');
  });
});
