import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { Event, ReplayPayload, Verdict } from '../src/contracts/types.ts';
import { CHECK_ROWS, CROP_MARKS, revealCount, SCENES, sceneByName, stepAt, ticks, type Scene } from '../src/kiosk/scenes.ts';
import { LaneStore } from '../src/lanes/laneStore.ts';
import { KIOSK_CARDS, SceneBody, sceneHeading, sourceFor } from '../src/pages/KioskPage.tsx';
import { BUNDLED_DELAYS, startReplay, type Pacer } from '../src/state/replay.ts';
import { scheduleReplay } from '../src/state/schedule.ts';
import type { AnyEvent, RunState } from '../src/state/types.ts';
import { finalFixtureState, fixtureFiles, recordedEvents } from './helpers.ts';

const count = (text: string, needle: string) => text.split(needle).length - 1;

class VirtualPacer implements Pacer {
  private t = 0;
  private queue: { at: number; fn: () => void; id: number }[] = [];
  private nextId = 1;
  now(): number {
    return this.t;
  }
  setTimeout(fn: () => void, ms: number): unknown {
    const id = this.nextId++;
    this.queue.push({ at: this.t + ms, fn, id });
    return id;
  }
  clearTimeout(handle: unknown): void {
    this.queue = this.queue.filter((q) => q.id !== handle);
  }
  async runAll(): Promise<void> {
    for (let guard = 0; this.queue.length && guard < 1_000_000; guard += 1) {
      this.queue.sort((a, b) => a.at - b.at || a.id - b.id);
      const next = this.queue.shift()!;
      this.t = next.at;
      next.fn();
      await Promise.resolve();
    }
  }
}

function body(state: RunState, name: string, elapsed: number, opts: { upgraded?: boolean; panel?: string } = {}): string {
  const scene = sceneByName(name);
  return renderToStaticMarkup(
    createElement(SceneBody, {
      scene,
      state,
      panelId: opts.panel ?? scene.panel ?? null,
      elapsed,
      t: elapsed,
      upgraded: opts.upgraded ?? false,
      runId: 'mock-sample-pair',
      lanes: new LaneStore(),
      query: new URLSearchParams(),
    }),
  );
}

describe('every scene carries its own delay table', () => {
  it('starts each table at 0, in order, inside the hold', () => {
    const names = new Set<string>();
    for (const s of SCENES) {
      expect(names.has(s.name)).toBe(false);
      names.add(s.name);
      expect(s.steps.length).toBeGreaterThan(0);
      expect(s.steps[0]!.atMs).toBe(0);
      for (let i = 1; i < s.steps.length; i += 1) expect(s.steps[i]!.atMs).toBeGreaterThan(s.steps[i - 1]!.atMs);
      expect(s.steps.at(-1)!.atMs).toBeLessThan(s.holdMs);
    }
    // The beats docs/VIDEO_PLAN.md names, all present.
    for (const n of ['seal', 'masks', 'fragments', 'coverage', 'crops', 'moment', 'tape', 'hood', 'claim', 'sheet', 'topup', 'blur', 'checks', 'guardrails']) expect(names.has(n)).toBe(true);
  });

  it('runs the moment in 12 s: dim, trail, the cut with its dissolve and join, the card', () => {
    const moment = sceneByName('moment');
    expect(moment.holdMs).toBe(12000);
    expect(moment.steps.map((s) => [s.atMs, s.name])).toEqual([
      [0, 'dim'],
      [3000, 'trail'],
      [7000, 'cut'],
      [7600, 'return'],
      [8200, 'join'],
      [10000, 'card'],
    ]);
    expect(stepAt(moment, 7599)?.name).toBe('cut');
    expect(stepAt(moment, 7600)?.name).toBe('return');
    expect(stepAt(moment, 11999)?.name).toBe('card');
  });

  it('reveals in even ticks', () => {
    expect(ticks(400, 1000, 200, 'fill').map((s) => s.atMs)).toEqual([400, 600, 800, 1000]);
    expect(revealCount(0, 300, 900)).toBe(0);
    expect(revealCount(300, 300, 900)).toBe(1);
    expect(revealCount(1199, 300, 900)).toBe(1);
    expect(revealCount(1200, 300, 900)).toBe(2);
  });

  it('feeds a paced scene the run with its own stage budgets laid over delays.json', () => {
    const seal = sceneByName('seal');
    expect(sourceFor(seal, null)).toEqual({ kind: 'replay', mode: 'fixed', untilMs: undefined, startAtStage: undefined, delays: seal.delays });
    expect(sourceFor(sceneByName('masks'), 2000)).toMatchObject({ mode: 'fixed', untilMs: 2000, startAtStage: 'tracking', delays: { tracking: 5000 } });
    expect(sourceFor(sceneByName('moment'), 5000)).toEqual({ kind: 'replay', mode: 'instant' });
  });

  it('lands the seal scene events where its table puts them, not where delays.json would', async () => {
    const events = recordedEvents();
    const seal = sceneByName('seal');
    const pacer = new VirtualPacer();
    const at = new Map<number, number>();
    const payload: ReplayPayload = { run_id: 'mock-sample-pair', mode: 'fixed', recorded_on: '2026-10-27', delays_ms: BUNDLED_DELAYS, events };
    const ended = new Promise<void>((resolve) => {
      startReplay('mock-sample-pair', 'fixed', (e: AnyEvent) => 'seq' in e && at.set(e.seq, pacer.now()), {
        files: fixtureFiles(),
        payload: async () => payload,
        pacer,
        delays: seal.delays,
        fetchFiles: false,
        onEnd: resolve,
      });
    });
    while (!at.size) {
      await pacer.runAll();
      await new Promise((r) => setTimeout(r, 1));
    }
    await pacer.runAll();
    await ended;
    const manifest = events.find((e: Event) => e.kind === 'seal.manifest')!;
    const own = scheduleReplay(events, 'fixed', { ...BUNDLED_DELAYS, ...seal.delays }).find((s) => s.event.seq === manifest.seq)!.atMs;
    const bundled = scheduleReplay(events, 'fixed', BUNDLED_DELAYS).find((s) => s.event.seq === manifest.seq)!.atMs;
    expect(at.get(manifest.seq)).toBeCloseTo(own, 6);
    expect(own).not.toBeCloseTo(bundled, 0);
    expect(own).toBeCloseTo(2800, 6); // upload.done by 1.6 s, then the manifest second of four sealing events
  });
});

describe('the scenes render each step from the stored run', () => {
  it('moment, guaranteed form: the boxed pickup frame labelled with the renter words, then the cut, the join and the card', async () => {
    const state = await finalFixtureState();
    const trail = body(state, 'moment', 5000);
    expect(trail).toContain('data-phase="pickup"');
    expect(trail).toContain('&quot;small scuff, right rear door&quot;');
    const ret = body(state, 'moment', 7800);
    expect(ret).toContain('data-phase="return"');
    expect(ret).toContain('class="cut-ghost"');
    expect(ret).not.toContain('class="join-line"');
    const join = body(state, 'moment', 9000);
    expect(count(join, 'class="join-line"')).toBe(1);
    expect(join).toContain('class="moment-side hidden"');
    const card = body(state, 'moment', 10500);
    expect(card).toContain('class="moment-side "');
    expect(card).toMatch(/<span class="label already_there">already there<\/span>/);
    expect(card).toContain('dates-card stacked');
    expect(card).toContain('same day, 45 minutes later');
    expect(card).toContain('sealed 7 minutes after the camera time');
  });

  it('moment, upgraded form: the mark stays put beside a reflection that slides, captions only, once calibrated', async () => {
    const fixture = await finalFixtureState();
    const state: RunState = { ...fixture, run: { ...fixture.run!, thresholds: { ...fixture.run!.thresholds, staysput: { cap_enabled: true } } } };
    const html = body(state, 'moment', 5000, { upgraded: true });
    expect(html).toContain('Stays put: paint');
    expect(html).toContain('Slides with the camera: reflection');
    expect(html).not.toContain('drift'); // the numbers live on the sheet and the run page, not in the moment
    // Without a fixed track the upgraded form falls back to the guaranteed one.
    const noFixed = { ...state, markTracks: state.markTracks.map((m) => (m.is_control ? m : { ...m, verdict: 'inconclusive' as const })) };
    expect(body(noFixed, 'moment', 5000, { upgraded: true })).toContain('data-phase="pickup"');
  });

  it('moment, upgraded form on an uncalibrated run: the boxed pickup frame, no Stays Put verdict on screen', async () => {
    const state = await finalFixtureState(); // today's thresholds: staysput.cap_enabled false
    const html = body(state, 'moment', 5000, { upgraded: true });
    expect(html).toContain('data-phase="pickup"');
    expect(html).not.toContain('Stays put: paint');
    expect(html).not.toContain('reflection');
  });

  it('coverage fills the car of each walk slot by slot, then names what each walk missed', async () => {
    const state = await finalFixtureState();
    expect(count(body(state, 'coverage', 0), 'cc-panel pending')).toBe(44); // a pair run: both walks, 22 slots each
    const mid = body(state, 'coverage', 1500);
    expect(count(mid, 'cc-panel pending')).toBeGreaterThan(0);
    expect(count(mid, 'cc-panel pending')).toBeLessThan(44);
    const end = body(state, 'coverage', 3400);
    expect(count(end, 'cc-panel pending')).toBe(0);
    expect(count(end, 'class="cov-short compact "')).toBe(2);
    expect(end).toContain('pickup walk');
    expect(end).toContain('return walk');
    // Filmed at pickup, glimpsed at return: it reads as missed in the return walk only (and the hood the other way).
    expect(end).toContain('<title>left front wheel: filmed</title>');
    expect(end).toContain('<title>left front wheel: glimpsed</title>');
    expect(end).toContain('glimpsed, 0.9 s');
    expect(end).toContain('<title>hood: glimpsed</title>');
    expect(end).toContain('<title>hood: filmed</title>');
    expect(end).toContain('rear bumper: glimpsed'); // the pickup callout, before the top-up
    expect(end).toContain('glimpsed, 0.8 s');
  });

  it('top-up: the ask as it was, the nouns it ran, then the bumper filmed and the odometer', async () => {
    const state = await finalFixtureState();
    const ask = body(state, 'topup', 0);
    expect(ask).toContain('rear bumper: 0.8 s, add 10 seconds?');
    expect(ask).toContain('odometer: add 3 seconds?');
    expect(ask).not.toContain('running:');
    expect(ask).toMatch(/cc-panel glimpsed[^"]*"[^>]*><title>rear bumper/);
    const running = body(state, 'topup', 5000);
    expect(running).toContain('running: bumper, roof, car, license plate, face (5 of 16 lanes)');
    const filmed = body(state, 'topup', 9000);
    expect(filmed).toContain('rear bumper: filmed, 9.6 s; the top-up lanes ran in 41 s');
    expect(filmed).toContain('odometer: filmed');
    expect(filmed).toMatch(/cc-panel filmed[^"]*"[^>]*><title>rear bumper/);
  });

  it('blur: the stored frame with what its checks found, under a heading that speaks for that frame', async () => {
    const state = await finalFixtureState();
    expect(sceneByName('blur').heading).toBe('blurred before the frame is saved');
    expect(body(state, 'blur', 0)).toContain('This crop was blurred before it was saved or sent to Muse Spark.');
  });

  it('checks: one row a beat, every model counted from the run', async () => {
    const state = await finalFixtureState();
    expect(count(body(state, 'checks', 0), '<dt>')).toBe(0);
    expect(count(body(state, 'checks', CHECK_ROWS.fromMs), '<dt>')).toBe(1);
    const all = body(state, 'checks', 6000);
    expect(count(all, '<dt>')).toBe(5);
    expect(all).toContain('Muse Spark Open on Baseten');
    expect(all).toContain('took 2 second looks: 1 same and 1 different');
  });

  it('crops: the boxes drop in one per tick', async () => {
    const state = await finalFixtureState();
    expect(count(body(state, 'crops', 0), 'class="mark-box"')).toBe(0);
    expect(count(body(state, 'crops', CROP_MARKS.fromMs), 'class="mark-box"')).toBe(1);
    expect(count(body(state, 'crops', CROP_MARKS.fromMs + CROP_MARKS.everyMs), 'class="mark-box"')).toBe(2);
  });

  it('hood: both frames, then the refusal naming the glare, then the line', async () => {
    const state = await finalFixtureState();
    expect(body(state, 'hood', 0)).not.toContain('class="refusal"');
    const refusal = body(state, 'hood', 2500);
    expect(refusal).toContain('Not compared: G2 crop condition (retired) (glare on the return frame)'); // a run stored before G2 retired (10/31)
    expect(refusal).not.toContain('Models find and compare');
    expect(body(state, 'hood', 5000)).toContain('Models find and compare; code decides.');
  });

  it('tape: the frames first, then the card with the pickup window, no dates', async () => {
    const state = await finalFixtureState();
    const frames = body(state, 'tape', 0);
    expect(frames).not.toContain('class="verdict-cards"');
    expect(frames).toContain('class="verdict-cards hidden"'); // held in place, so the frames do not move when it shows
    const card = body(state, 'tape', 1500);
    expect(card).toContain('new or unseen');
    expect(card).toContain('This panel was filmed at pickup');
    expect(card).not.toContain('dates-card');
  });

  it('tape: the heading is the label of the verdict the panel leads with, in the app own words', async () => {
    const state = await finalFixtureState();
    const tape = sceneByName('tape');
    const base = state.verdicts.find((v) => v.panel_id === 'left_front_door')!;
    const others = state.verdicts.filter((v) => v.panel_id !== 'left_front_door');
    const only = (...labels: Verdict['label'][]) => ({ ...state, verdicts: [...others, ...labels.map((label, i) => ({ ...base, verdict_id: `v${i}`, label }))] });
    expect(sceneHeading(tape, only('new_or_unseen'), 'left_front_door')).toBe('new or unseen');
    expect(sceneHeading(tape, only('needs_your_eye'), 'left_front_door')).toBe('needs your eye');
    expect(sceneHeading(tape, only('needs_your_eye', 'already_there'), 'left_front_door')).toBe('already there'); // the card's lead
    expect(sceneHeading(tape, only(), 'left_front_door')).toBe('');
    expect(sceneHeading(tape, state, 'left_front_door')).not.toBe('added between the walks');
    expect(sceneHeading(sceneByName('hood'), state, 'hood')).toBe('sky on the hood');
  });

  it('tape: at most the kiosk number of cards, the lead verdict first, the rest counted', async () => {
    const state = await finalFixtureState();
    const lead = state.verdicts.find((v) => v.panel_id === 'left_front_door')!;
    const eyes = [1, 2, 3, 4].map((i) => ({ ...lead, verdict_id: `eye_${i}`, label: 'needs_your_eye' as const }));
    const already = { ...lead, verdict_id: 'already_1', label: 'already_there' as const };
    const many = { ...state, verdicts: [...state.verdicts.filter((v) => v.panel_id !== 'left_front_door'), ...eyes, already] };
    const card = body(many, 'tape', 1500);
    expect(count(card, 'class="verdict-card ')).toBe(KIOSK_CARDS);
    expect(card.indexOf('verdict-card already_there')).toBeLessThan(card.indexOf('verdict-card needs_your_eye'));
    expect(card).toContain(`and ${5 - KIOSK_CARDS} more on the sheet`);
    expect(body(state, 'tape', 1500)).not.toContain('more on the sheet');
  });

  it('says what is missing instead of drawing nothing', async () => {
    const state = await finalFixtureState();
    const empty: RunState = { ...state, coverage: null, keyframes: [] };
    expect(body(empty, 'topup', 0)).toContain('Coverage is not in this run.');
    expect(body(empty, 'crops', 0)).toContain('No pickup keyframe for this panel.');
    const scene: Scene = sceneByName('claim');
    expect(body({ ...state, claims: [], claim: null, completed: false }, scene.name, 0)).toContain('loading the claim');
    expect(body({ ...state, claims: [], claim: null }, scene.name, 0)).toContain('This run has no claim photo.');
  });

  it('claim: the run own claims, the placed one first, then the one the flow refused', async () => {
    const state = await finalFixtureState();
    expect(state.claims.map((c) => c.claim_id)).toEqual(['claim_1', 'claim_2', 'claim_3', 'claim_4', 'claim_5']);
    const photo = body(state, 'claim', 0);
    expect(photo).toContain('src="claim/claim_1.blurred.jpg"');
    expect(photo).toContain('Claim photo: staged for this demo');
    const frame = body(state, 'claim', 4200);
    expect(frame).toContain('class="crop claim-frame"');
    expect(frame).toContain('scuff, S'); // the joined pickup mark, boxed
    const second = body(state, 'claim', 13500);
    expect(second).toContain('src="claim/claim_2.blurred.jpg"');
    expect(second).toContain('Staged by us, to show a photo the flow refuses to place.');
    expect(count(second, 'Claim photo: staged for this demo, standing in for the company')).toBe(1); // the main photo's caption only
    expect(second).toContain('E5 cannot place');
    // An E2b claim has no exit field but is not placed, so it is never the main photo.
    const onlyE2b = { ...state, claims: state.claims.filter((c) => c.claim_id === 'claim_5' || c.claim_id === 'claim_2') };
    expect(body(onlyE2b, 'claim', 0)).toContain('src="claim/claim_2.blurred.jpg"');
  });
});
