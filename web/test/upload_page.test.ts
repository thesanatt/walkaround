import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import type { StatusResponse } from '../src/contracts/types.ts';
import { OWN_STEPS, UploadPage } from '../src/pages/UploadPage.tsx';
import { HERE } from './helpers.ts';

const ROOT = join(HERE, '..', '..');
const LIVE: StatusResponse = { paused: false, pickup_left: 6, pairs_left: 1, queue_position: null, queue_eta_minutes: null, engine: 'meta-model-api', replay_only: false };
const page = (status: StatusResponse | null) => renderToStaticMarkup(createElement(UploadPage, { status }));
const count = (text: string, needle: string) => text.split(needle).length - 1;
const esc = (text: string) => text.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll("'", '&#x27;');

// The top bar reads the address while it renders; node has no window.
const g = globalThis as { window?: unknown };
beforeAll(() => {
  g.window = { location: { pathname: '/upload', search: '' } };
});
afterAll(() => {
  delete g.window;
});

describe('the upload page on a copy that takes no new checks', () => {
  it('keeps both slots and the options on a live server, and while the status loads', () => {
    for (const html of [page(LIVE), page(null)]) {
      expect(count(html, 'type="file"')).toBe(2);
      expect(html).toContain('start-options');
      expect(html).not.toContain('start-steps');
    }
  });

  it('a replay-only copy says why and gives the three steps in place of the slots', () => {
    const html = page({ ...LIVE, paused: true, pickup_left: 0, pairs_left: 0, replay_only: true });
    expect(html).not.toContain('type="file"');
    expect(html).not.toContain('start-options');
    expect(html).toContain(esc("This public copy only replays the example, so strangers can't spend my model credits."));
    expect(count(html, '<pre>')).toBe(3);
    for (const step of OWN_STEPS) expect(html).toContain(esc(step.code));
    expect(html).toContain('href="https://github.com/thesanatt/walkaround#run-it-from-a-clean-clone">Setup on GitHub</a>');
    expect(html).toContain('Faces are always blurred');
  });

  it('a replay-only copy with a key that works still takes no upload', () => {
    expect(page({ ...LIVE, pickup_left: 0, pairs_left: 0, replay_only: true })).not.toContain('type="file"');
  });

  it('a paused clone (no working key) gets the key and start steps, not the public copy line', () => {
    const html = page({ ...LIVE, paused: true });
    expect(html).not.toContain('type="file"');
    expect(html).not.toContain('public copy');
    expect(html).toContain('no working Meta Model API key');
    expect(count(html, '<pre>')).toBe(2);
    expect(html).not.toContain('git clone');
  });

  it('a live server out of checks for today keeps the resting slots and its one line', () => {
    const html = page({ ...LIVE, pickup_left: 0, pairs_left: 0 });
    expect(count(html, 'type="file"')).toBe(2);
    expect(html).toContain('No new checks are left today.');
    expect(html).not.toContain('start-steps');
  });

  it('every command is the README word for word, and the key goes where .env.example puts it', () => {
    const readme = readFileSync(join(ROOT, 'README.md'), 'utf8');
    const section = readme.slice(readme.indexOf('## Run it from a clean clone'), readme.indexOf('## What happens in a run'));
    const lines = new Set(section.split('\n').map((l) => l.split('  #')[0]!.trim()));
    for (const step of OWN_STEPS) for (const line of step.code.split('\n')) expect(lines).toContain(line);
    expect(readFileSync(join(ROOT, '.env.example'), 'utf8')).toMatch(/^MODEL_API_KEY=/m);
  });
});
