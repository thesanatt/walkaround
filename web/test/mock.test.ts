import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { servedFileAllowed } from '../mock/allowlist.ts';
import { createMockApi, tapClaim } from '../mock/server.ts';
import { RUN_DIR } from './helpers.ts';

// Every path the web asks for, and what app.py must answer once privacy.py carries the binding allowlist of
// docs/CONTRACTS.md ("Phase 2 file shapes"); the mock answers the same today.
const TABLE: [string, boolean][] = [
  ['open_weights.ndjson', true], // how it was checked, the second-look line, the loader's text read
  ['deletions.json', true],
  ['derived/looks.json', true],
  ['derived/claim.json', true],
  ['claim/claim_1.json', true], // every claim, open run or closed
  ['claim/claim_12.json', true],
  ['claim/claim_1.blurred.jpg', true], // the pixelated copy, the only claim photo served
  ['claim/claim_1.jpg', false], // the upload, unblurred during the privacy window
  ['claim/claim_1.answers.json', false], // every model answer of the claim
  ['crops/pickup/kf_a_m1__kf_b_m2.look.jpg', true], // what the second look saw
  ['chips/pickup/t1.m4a', true],
  ['clips/pickup/kf_x_m1.mp4', true],
  ['derived/mark_tracks.json', true],
  ['media/pickup.proxy_blurred.mp4', true],
  ['media/pickup.proxy.mp4', false], // the unblurred proxy
  ['quarantine/pickup.orig', false],
  ['QUARANTINE/pickup.orig', false], // APFS matches names case-insensitively
  ['uploads/u1/0.part', false],
  ['spark.json', false],
  ['voice.json', false],
  ['events.pending.ndjson', false],
  ['derived/../run.json', false],
  ['.env', false],
  ['sheet/frames/kf_a.owner.jpg', true], // the owner may see an owner frame
];

describe('the mock serves exactly what app.py serves', () => {
  it('follows the binding allowlist for the owner', () => {
    for (const [path, allowed] of TABLE) expect([path, servedFileAllowed({ keep_plate: false }, path, true)]).toEqual([path, allowed]);
  });

  it('keeps owner frames and, under keep_plate, frames cut from 4K away from the kiosk', () => {
    expect(servedFileAllowed({ keep_plate: false }, 'sheet/frames/kf_a.owner.jpg', false)).toBe(false);
    expect(servedFileAllowed({ keep_plate: true }, 'claim/claim_1.blurred.jpg', false)).toBe(false);
    expect(servedFileAllowed({ keep_plate: true }, 'crops/pickup/kf_a.jpg', false)).toBe(false);
    expect(servedFileAllowed({ keep_plate: true }, 'derived/claim.json', false)).toBe(true);
    expect(servedFileAllowed({ keep_plate: true }, 'claim/claim_1.json', true)).toBe(true);
  });
});

describe('the mock routes', () => {
  let server: Server;
  let base = '';
  beforeAll(async () => {
    const api = createMockApi();
    server = createServer((req, res) =>
      api.handle(req, res, () => {
        res.statusCode = 404;
        res.end();
      }),
    );
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });
  afterAll(() => new Promise<void>((resolve) => server.close(() => resolve())));

  const status = async (rel: string, view = '') => (await fetch(`${base}/api/runs/mock-sample-pair/files/${rel}${view}`)).status;

  it('answers 404 where app.py does and serves the rest from the fixture', async () => {
    expect(await status('claim/claim_1.blurred.jpg')).toBe(200);
    expect(await status('claim/claim_1.json')).toBe(200);
    expect(await status('open_weights.ndjson')).toBe(200);
    expect(await status('claim/claim_1.answers.json')).toBe(404);
    expect(await status('quarantine/pickup.orig')).toBe(404);
    expect(await status('media/pickup.proxy.mp4')).toBe(404);
  });

  it('serves a synthetic walk as the blurred proxy, never a clip from data/', async () => {
    const res = await fetch(`${base}/api/runs/mock-sample-pair/files/media/pickup.proxy_blurred.mp4`, { headers: { range: 'bytes=0-15' } });
    expect(res.status).toBe(206);
    expect(Number(res.headers.get('content-range')?.split('/')[1])).toBeLessThan(200_000); // the test pattern, not a 3.8 MB walk
    expect(await status('media/return.proxy_blurred.mp4', '?view=kiosk')).toBe(200);
  });

  it('answers a tap the way claim.decide does: the picked panel stands, said first', () => {
    const placed = tapClaim(RUN_DIR, 'right_rear_door', 'scratch on the door');
    expect(placed.panel_id).toBe('right_rear_door');
    expect(placed.paragraph.startsWith('Panel picked by you: the right rear door. ')).toBe(true);
    expect(placed.match?.pairs.length).toBe(1);
    const fender = tapClaim(RUN_DIR, 'right_rear_fender', 'scratch on the door');
    expect([fender.panel_id, fender.exit, fender.match]).toEqual(['right_rear_fender', 'E3', null]);
    expect(fender.paragraph).toContain('Panel picked by you: the right rear fender.');
    // Glimpsed in the pickup walk but filmed in the top-up clip: the pickup side counts it filmed (claim.pickup_coverage).
    expect(tapClaim(RUN_DIR, 'rear_bumper', 'scuff')).toMatchObject({ exit: null, coverage_state: 'filmed' });
  });
});
