import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { Event, RunManifest } from '../src/contracts/types.ts';
import { startLive, type EventSourceLike } from '../src/state/live.ts';
import type { RunFiles } from '../src/state/loader.ts';
import { initialState, reduce } from '../src/state/reducer.ts';
import type { AnyEvent, RunState } from '../src/state/types.ts';

export const HERE = dirname(fileURLToPath(import.meta.url));
export const RUN_DIR = join(HERE, '..', 'mock', 'fixture', 'mock-sample-pair');
export const SAM_FIXTURES = resolve(HERE, '..', '..', 'tests', 'fixtures');

// The recorded event log of the fixture run (hash-chained by mock/build_fixture.py through contracts.chain_hash).
export function recordedEvents(): Event[] {
  return readFileSync(join(RUN_DIR, 'events.ndjson'), 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Event);
}

export function fixtureRun(): RunManifest {
  return JSON.parse(readFileSync(join(RUN_DIR, 'run.json'), 'utf8')) as RunManifest;
}

// RunFiles over the fixture directory. jitterMs > 0 delays each read by a random amount, so a test can show that the
// dispatched sequence does not depend on which fetch finishes first.
export function fixtureFiles(jitterMs = 0, seed = 7): RunFiles {
  let state = seed;
  const random = () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
  const later = <T>(value: T): Promise<T> => (jitterMs > 0 ? new Promise((r) => setTimeout(() => r(value), Math.floor(random() * jitterMs))) : Promise.resolve(value));
  return {
    json: <T>(rel: string) => {
      const path = join(RUN_DIR, ...rel.split('/'));
      return later(existsSync(path) ? (JSON.parse(readFileSync(path, 'utf8')) as T) : null);
    },
    text: (rel: string) => {
      const path = join(RUN_DIR, ...rel.split('/'));
      return later(existsSync(path) ? readFileSync(path, 'utf8') : null);
    },
    run: () => later(fixtureRun()),
  };
}

// Every event the fixture run's live driver dispatches, recorded events and "<kind>.loaded" files alike.
export async function liveSequence(files: RunFiles = fixtureFiles()): Promise<AnyEvent[]> {
  const seen: AnyEvent[] = [];
  let source: FakeEventSource | null = null;
  await new Promise<void>((resolve) => {
    startLive('mock-sample-pair', (e) => seen.push(e), { files, openEvents: (url) => (source = new FakeEventSource(url)), onEnd: resolve });
    const wait = setInterval(() => {
      if (!source) return;
      clearInterval(wait);
      source.open();
      for (const e of recordedEvents()) source.emit(e);
    }, 1);
  });
  return seen;
}

// The fixture run's final state through the live driver.
export async function finalFixtureState(): Promise<RunState> {
  return (await liveSequence()).reduce(reduce, initialState());
}

// An EventSource stand-in: emit() delivers to the listeners of the event's kind, as the browser does for `event: <kind>`.
export class FakeEventSource implements EventSourceLike {
  readyState = 0;
  onopen: ((ev: unknown) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  closed = false;
  readonly url: string;
  private readonly listeners = new Map<string, ((ev: MessageEvent<string>) => void)[]>();

  constructor(url: string) {
    this.url = url;
  }

  addEventListener(type: string, listener: (ev: MessageEvent<string>) => void): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.({});
  }

  emit(event: Event): void {
    if (this.closed) return;
    for (const l of this.listeners.get(event.kind) ?? []) l({ data: JSON.stringify(event) } as MessageEvent<string>);
  }

  drop(): void {
    this.readyState = 0;
    this.onerror?.({});
  }

  // The server answered 204 (or 404) to a reconnect: EventSource gives up and reports CLOSED.
  refuse(): void {
    this.readyState = 2;
    this.onerror?.({});
  }

  close(): void {
    this.closed = true;
    this.readyState = 2;
  }
}

export function deepFreeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const v of Object.values(value as Record<string, unknown>)) deepFreeze(v);
  }
  return value;
}
