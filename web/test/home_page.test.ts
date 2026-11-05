import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { StartPage } from '../src/pages/StartPage.tsx';
import { HERE } from './helpers.ts';

// The top bar reads the address while it renders; node has no window.
const g = globalThis as { window?: unknown };
beforeAll(() => {
  g.window = { location: { pathname: '/', search: '' } };
});
afterAll(() => {
  delete g.window;
});

describe('the home page', () => {
  it('links the crash-test sample under the Focus example, before the button', () => {
    const html = renderToStaticMarkup(createElement(StartPage));
    const crash = html.indexOf('href="/run/eval_nhtsa15759_r6?replay=instant">See it</a>');
    expect(crash).toBeGreaterThan(html.indexOf('href="/run/hero_focus5e?replay=instant"'));
    expect(crash).toBeLessThan(html.indexOf('href="/upload"'));
    expect(html).toContain('Another example: a crash-test car, before and after a side impact.');
    const index = JSON.parse(readFileSync(join(HERE, '..', '..', 'samples', 'index.json'), 'utf8')) as { samples: { run_id: string | null }[] };
    expect(index.samples.map((s) => s.run_id)).toContain('eval_nhtsa15759_r6');
  });
});
