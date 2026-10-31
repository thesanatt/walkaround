import { describe, expect, it } from 'vitest';

import type { Event, ReplayPayload, RunManifest } from '../src/contracts/types.ts';
import { startLive } from '../src/state/live.ts';
import type { RunFiles } from '../src/state/loader.ts';
import { initialState, reduce, rolesOf } from '../src/state/reducer.ts';
import { BUNDLED_DELAYS, runSchedule, startReplay, type Pacer } from '../src/state/replay.ts';
import { replayWaits, scheduleReplay, stageOfEach } from '../src/state/schedule.ts';
import type { AnyEvent, RunState } from '../src/state/types.ts';
import { frameBatcher } from '../src/state/useRun.ts';
import { deepFreeze, FakeEventSource, fixtureFiles, fixtureRun, recordedEvents } from './helpers.ts';

// A virtual clock: setTimeout callbacks run in time order when advance() is called.
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

function payloadOf(events: Event[], mode: ReplayPayload['mode'] = 'fixed'): ReplayPayload {
  return { run_id: 'mock-sample-pair', mode, recorded_on: '2026-10-27', delays_ms: mode === 'fixed' ? BUNDLED_DELAYS : {}, events };
}

async function viaLive(events: Event[], jitterMs: number): Promise<AnyEvent[]> {
  const seen: AnyEvent[] = [];
  let source: FakeEventSource | null = null;
  await new Promise<void>((resolve, reject) => {
    startLive('mock-sample-pair', (e) => seen.push(e), {
      files: fixtureFiles(jitterMs),
      openEvents: (url) => (source = new FakeEventSource(url)),
      onEnd: resolve,
      onStatus: (s, d) => s === 'error' && reject(new Error(d)),
    });
    const wait = setInterval(() => {
      if (!source) return;
      clearInterval(wait);
      source.open();
      for (const e of events) source.emit(e);
    }, 1);
  });
  expect(source!.closed).toBe(true); // closed on run.completed, so the browser never reconnects to a finished run
  return seen;
}

async function viaReplay(events: Event[], jitterMs: number, mode: ReplayPayload['mode'] = 'fixed'): Promise<AnyEvent[]> {
  const seen: AnyEvent[] = [];
  const pacer = new VirtualPacer();
  const ended = new Promise<void>((resolve) => {
    startReplay('mock-sample-pair', mode, (e) => seen.push(e), { files: fixtureFiles(jitterMs), payload: async () => payloadOf(events, mode), pacer, onEnd: resolve });
  });
  // the driver dispatches replay.started right before it schedules, once the payload and the run are in
  while (!seen.length) await new Promise((r) => setTimeout(r, 1));
  await pacer.runAll();
  await ended;
  return seen;
}

function fold(events: AnyEvent[]): RunState {
  return events.reduce(reduce, initialState());
}

function withoutReplay(state: RunState): Omit<RunState, 'replay'> {
  const { replay: _replay, ...rest } = state;
  return rest;
}

describe('one reducer, two drivers', () => {
  const events = recordedEvents();

  it('reads a real hash-chained log with every stage of a pair', () => {
    expect(events.length).toBeGreaterThan(400);
    expect(events.map((e) => e.seq)).toEqual(events.map((_, i) => i));
    for (let i = 1; i < events.length; i += 1) expect(events[i]!.prev).toBe(events[i - 1]!.hash);
    expect(new Set(stageOfEach(events))).toEqual(new Set(['created', 'sealing', 'sealed', 'converting', 'tracking', 'blurring', 'marking', 'staysput', 'matching', 'sheet', 'done']));
  });

  it('reaches the same final state live and replayed, with the same dispatched sequence', async () => {
    const live = await viaLive(events, 0);
    const replayed = await viaReplay(events, 0);
    expect(replayed[0]!.kind).toBe('replay.started');
    expect(replayed.slice(1).map((e) => e.kind)).toEqual(live.map((e) => e.kind));
    const a = fold(live);
    const b = fold(replayed);
    expect(withoutReplay(b)).toEqual(withoutReplay(a));
    expect(b.replay).toEqual({ mode: 'fixed', badgeDate: '2026-10-27' });
    expect(a.replay).toBeNull();
  });

  it('keeps the sequence when the derived files arrive in a random order', async () => {
    const steady = await viaLive(events, 0);
    const jittered = await viaLive(events, 25);
    const replayed = await viaReplay(events, 25, 'instant');
    expect(jittered).toEqual(steady);
    expect(replayed.slice(1)).toEqual(steady);
  });

  it('ends with every derived file loaded and the run complete', async () => {
    const state = fold(await viaLive(events, 0));
    const run = fixtureRun();
    expect(state.completed).toBe(true);
    expect(state.stage).toEqual({ name: 'done', pct: 100 });
    expect(state.run?.run_id).toBe(run.run_id);
    expect(rolesOf(state)).toEqual(['pickup', 'return']);
    expect(state.panels).toHaveLength(22);
    expect(state.tracks.map((t) => t.panel_id)).toEqual(['right_rear_door', 'right_rear_door']);
    expect(state.keyframes.length).toBe(6);
    expect(state.verdicts.map((v) => v.label)).toEqual(['already_there', 'needs_your_eye', 'new_or_unseen', 'not_comparable']);
    expect(state.markTracks.map((m) => m.verdict)).toEqual(['fixed', 'sliding']);
    expect(state.narration?.per_role['pickup']?.turns[0]?.text).toBe('small scuff, right rear door');
    expect(state.coverage?.missing).toEqual(['rear_bumper', 'roof']);
    expect(state.sheet.written).toBe(true);
    expect(state.deletions.map((d) => d.kind).sort()).toEqual(['file', 'file', 'files_api', 'files_api', 'quarantine', 'quarantine']);
    expect(state.media['pickup']?.blurredPath).toBe('media/pickup.proxy_blurred.mp4');
    expect(state.seal.manifest_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(state.seal.ots).toBe('pending');
    expect(state.ledger.dollarsSoFar).toBeCloseTo(1.53);
    expect(state.ledger.inFlight).toBe(0);
  });

  it('keys lanes by lane id whatever spelling the event used', async () => {
    const state = fold(await viaLive(events, 0));
    expect(state.lanes['pickup']?.['car_door']).toMatchObject({ status: 'completed', frames: 134, ids: ['0'] });
    expect(state.lanes['return']?.['car_door']).toMatchObject({ status: 'completed', ids: ['1'] });
    expect(state.lanes['return']?.['bumper']?.status).toBe('incomplete');
    expect(state.lanes['pickup']?.['roof']?.status).toBe('failed');
    expect(state.lanes['pickup']?.['license_plate']?.status).toBe('zero_matches');
    const spaced = reduce(initialState(), { kind: 'sam.request', payload: { role: 'pickup', noun: 'car door', request_id: 'r1' } } as unknown as AnyEvent);
    expect(Object.keys(spaced.lanes['pickup'] ?? {})).toEqual(['car_door']);
  });

  it('is pure: it never mutates the state or the event it is given', async () => {
    const all = await viaLive(events, 0);
    let state = deepFreeze(initialState());
    for (const e of all) state = deepFreeze(reduce(state, deepFreeze(structuredClone(e))));
    expect(state.completed).toBe(true);
  });

  it('counts each Spark call from its request to its response, and notes the first files upload', async () => {
    const state = fold(await viaLive(events, 0));
    expect(events.filter((e) => e.kind === 'spark.request')).toHaveLength(6);
    expect(state.spark).toEqual({ requested: 6, answered: 6, truncated: 0 });
    expect(state.seal.files_uploaded).toBe(true);
    const sealedOnly = fold(events.slice(0, events.findIndex((e) => e.kind === 'files.uploaded')));
    expect(sealedOnly.seal.manifest_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(sealedOnly.seal.files_uploaded).toBe(false);
  });

  it('keys the sheet hashes by run-relative path, as sheet.write_sheet does', async () => {
    const state = fold(await viaLive(events, 0));
    expect(Object.keys(state.sheet.sha256s)).toEqual(['sheet/sheet.html']);
  });

  it('shows why a run failed before the seal instead of a load error', async () => {
    // _fail keeps run.failed pending (no chain before manifest.json); app.py answers 204, so no event ever arrives.
    const failed: RunManifest = { ...fixtureRun(), live_or_replay: 'live', status: 'failed', errors: [{ stage: 'seal', text: 'ffprobe could not read quarantine/pickup.orig' }] };
    const files: RunFiles = { json: async () => null, run: async () => failed };
    const seen: AnyEvent[] = [];
    const statuses: string[] = [];
    let source: FakeEventSource | null = null;
    const ended = new Promise<void>((resolve) => {
      startLive('r1', (e) => seen.push(e), { files, openEvents: (url) => (source = new FakeEventSource(url)), onStatus: (s) => statuses.push(s), onEnd: resolve });
    });
    while (!source) await new Promise((r) => setTimeout(r, 1));
    (source as FakeEventSource).refuse();
    await ended;
    const state = fold(seen);
    expect(statuses).toEqual(['connecting', 'ended']);
    expect(state.completed).toBe(true);
    expect(state.stage.name).toBe('failed');
    expect(state.errors).toEqual(['seal: ffprobe could not read quarantine/pickup.orig']);
  });

  it('replays a run that failed before the seal as failed, with its errors', async () => {
    const failed: RunManifest = { ...fixtureRun(), status: 'failed', errors: [{ stage: 'seal', text: 'ffprobe could not read quarantine/pickup.orig' }] };
    const seen: AnyEvent[] = [];
    await new Promise<void>((resolve) => {
      startReplay('r1', 'fixed', (e) => seen.push(e), { files: { json: async () => null, run: async () => failed }, payload: async () => payloadOf([]), pacer: new VirtualPacer(), onEnd: resolve });
    });
    const state = fold(seen);
    expect(state.stage.name).toBe('failed');
    expect(state.errors).toEqual(['seal: ffprobe could not read quarantine/pickup.orig']);
    expect(state.replay?.mode).toBe('fixed');
  });

  it('still reports a stream that closes empty on a run that has not ended', async () => {
    const running: RunManifest = { ...fixtureRun(), live_or_replay: 'live', status: 'tracking' };
    const statuses: [string, string | undefined][] = [];
    let source: FakeEventSource | null = null;
    startLive('r1', () => undefined, { files: { json: async () => null, run: async () => running }, openEvents: (url) => (source = new FakeEventSource(url)), onStatus: (s, d) => statuses.push([s, d]) });
    while (!source) await new Promise((r) => setTimeout(r, 1));
    (source as FakeEventSource).refuse();
    await new Promise((r) => setTimeout(r, 5));
    expect(statuses.at(-1)).toEqual(['error', 'the event stream closed before any event arrived']);
  });

  it('drops a resent event after a reconnect (Last-Event-ID overlap)', async () => {
    const seen: AnyEvent[] = [];
    let source: FakeEventSource | null = null;
    startLive('mock-sample-pair', (e) => seen.push(e), { files: fixtureFiles(), openEvents: (url) => (source = new FakeEventSource(url)) });
    while (!source) await new Promise((r) => setTimeout(r, 1));
    const s = source as FakeEventSource;
    s.open();
    for (const e of events.slice(0, 5)) s.emit(e);
    s.drop();
    for (const e of events.slice(3, 8)) s.emit(e);
    await new Promise((r) => setTimeout(r, 20));
    const seqs = seen.filter((e): e is Event => 'seq' in e).map((e) => e.seq);
    expect(seqs).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });
});

describe('replay pacing', () => {
  const events = recordedEvents();

  it('matches orchestrator.replay_delays: a stage budget spread evenly over its events', () => {
    // A hand-built log: 2 events before any stage, a sealing stage of 3 events, a tracking stage of 5.
    const mk = (seq: number, kind: Event['kind'], payload: Record<string, unknown> = {}): Event => ({ seq, ts: `2026-10-27T15:00:${String(seq).padStart(2, '0')}Z`, run_id: 'x', kind, payload, prev: '', hash: '' });
    const log = [mk(0, 'run.created'), mk(1, 'upload.done'), mk(2, 'stage', { name: 'sealing' }), mk(3, 'seal.manifest'), mk(4, 'seal.tsr'), mk(5, 'stage', { name: 'tracking' }), mk(6, 'sam.frame'), mk(7, 'sam.frame'), mk(8, 'sam.frame'), mk(9, 'sam.frame')];
    const waits = replayWaits(log, 'fixed', { created: 1000, sealing: 1500, tracking: 5000 });
    expect(waits).toEqual([500, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000]);
    expect(replayWaits(log, 'fixed', { default: 900 })[0]).toBe(450);
    expect(replayWaits(log, 'fixed', {})[0]).toBe(0);
    expect(replayWaits(log, 'recorded', {}, 2000)).toEqual([0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]);
    expect(replayWaits(log, 'instant', {})).toEqual(log.map(() => 0));
    expect(scheduleReplay(log, 'fixed', { created: 1000, sealing: 1500, tracking: 5000 }).map((s) => s.atMs)).toEqual([500, 1000, 1500, 2000, 2500, 3500, 4500, 5500, 6500, 7500]);
  });

  it('caps recorded gaps at 2 s', () => {
    const waits = replayWaits(events, 'recorded', {});
    expect(Math.max(...waits)).toBeLessThanOrEqual(2000);
    expect(waits.some((w) => w === 2000)).toBe(true);
  });

  it('spends the checked-in budget: about 34 s from first event to last', () => {
    const total = scheduleReplay(events, 'fixed', BUNDLED_DELAYS).at(-1)!.atMs;
    const stages = new Set(stageOfEach(events));
    const expected = [...stages].reduce((sum, s) => sum + (BUNDLED_DELAYS[s] ?? BUNDLED_DELAYS['default'] ?? 0), 0);
    expect(total).toBeCloseTo(expected, 6);
    expect(total).toBeGreaterThan(30_000);
    expect(total).toBeLessThan(40_000);
  });

  it('honours fixed delays within 20 ms on real timers', async () => {
    const short = Object.fromEntries(Object.keys(BUNDLED_DELAYS).map((k) => [k, Math.round(BUNDLED_DELAYS[k]! / 60)]));
    const schedule = scheduleReplay(events, 'fixed', short);
    const start = performance.now();
    const late: number[] = [];
    await new Promise<void>((resolve) => {
      runSchedule(schedule, (_e, atMs) => late.push(performance.now() - start - atMs), { onEnd: resolve });
    });
    expect(late).toHaveLength(events.length);
    expect(Math.min(...late)).toBeGreaterThanOrEqual(-2);
    expect(Math.max(...late)).toBeLessThan(20);
  });

  it('starts a scene at a stage and freezes at a virtual time', () => {
    const schedule = scheduleReplay(events, 'fixed', BUNDLED_DELAYS);
    const firstTracking = schedule.findIndex((s) => s.event.kind === 'stage' && s.event.payload['name'] === 'tracking');
    const emitted: number[] = [];
    runSchedule(schedule, (e) => emitted.push(e.seq), { untilMs: schedule[firstTracking]!.atMs - 1 });
    expect(emitted.length).toBe(firstTracking);
  });
});

describe('one render per frame', () => {
  const events = recordedEvents();

  it('applies queued events together, in order, once per frame, and at once on flush', () => {
    const batches: AnyEvent[][] = [];
    const frames: (() => void)[] = [];
    const batch = frameBatcher(
      (es) => batches.push(es),
      (fn) => {
        frames.push(fn);
        return () => frames.splice(frames.indexOf(fn), 1);
      },
    );
    for (const e of events.slice(0, 50)) batch.dispatch(e);
    expect(batches).toHaveLength(0);
    expect(frames).toHaveLength(1); // one frame scheduled however many events came in
    frames.shift()!();
    expect(batches).toHaveLength(1);
    expect(batches[0]!.map((e) => ('seq' in e ? e.seq : -1))).toEqual(events.slice(0, 50).map((e) => e.seq));
    for (const e of events.slice(50, 60)) batch.dispatch(e);
    batch.flush(); // a driver's end: nothing waits for the next frame
    expect(batches).toHaveLength(2);
    expect(frames).toHaveLength(0);
    batch.dispatch(events[60]!);
    batch.cancel();
    expect(frames).toHaveLength(0);
    batch.flush();
    expect(batches).toHaveLength(2);
    expect(fold(batches.flat()).lastSeq).toBe(59);
  });
});
