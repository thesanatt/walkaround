import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { parseVideoStream } from '@meta-sam/parser';
import { describe, expect, it } from 'vitest';

import { eventsFromNdjson, FrameCollector, iterate, maskPath, parseLaneEvents } from '../src/workers/laneParse.ts';
import { SAM_FIXTURES } from './helpers.ts';

const text = (name: string) => readFileSync(join(SAM_FIXTURES, name), 'utf8');

describe('lane parsing (the worker path, run in Node)', () => {
  it('parses the 10/26 text-prompt lane: 134 frames, one id, one c on every frame', async () => {
    const lane = await parseLaneEvents(eventsFromNdjson(text('sam/pingpong_text.events.ndjson')));
    expect(lane.outcome.status).toBe('completed');
    expect(lane.diagnostics).toBe(0);
    expect(lane.frames.size).toBe(134);
    const all = [...lane.frames.values()].flat();
    expect(new Set(all.map((o) => o.id))).toEqual(new Set(['0']));
    expect(new Set(all.map((o) => o.c))).toEqual(new Set([0.7299578189849854]));
    expect(lane.frames.get(0)?.[0]?.box).toEqual([593, 538, 621, 570]);
    for (const o of all) {
      expect(o.mask?.width).toBe(o.box[2] - o.box[0]);
      expect(o.mask?.height).toBe(o.box[3] - o.box[1]);
      expect(maskPath(o.mask)).toMatch(/^M/);
    }
  });

  it('keeps the per-frame confidence of a box-prompt lane', async () => {
    const lane = await parseLaneEvents(eventsFromNdjson(text('sam/pingpong_box.events.ndjson')));
    const cs = [...lane.frames.values()].flat().map((o) => o.c);
    expect(lane.frames.size).toBe(134);
    expect(cs.slice(0, 3)).toEqual([0.890625, 0.8668561577796936, 0.9177275896072388]);
  });

  it('reads a zero-match lane as completed with nothing to draw', async () => {
    const lane = await parseLaneEvents(eventsFromNdjson(text('sam_zero_match.events.ndjson')));
    expect(lane.outcome.status).toBe('completed');
    expect(lane.frames.size).toBe(0);
  });

  it('reads a stream that ended early as incomplete and keeps what arrived', async () => {
    const events = eventsFromNdjson(text('sam/pingpong_text.events.ndjson')).slice(0, 70);
    const lane = await parseLaneEvents(events);
    expect(lane.outcome.status).toBe('incomplete');
    expect(lane.frames.size).toBeGreaterThan(50);
  });

  it('hands out a frame only once its box and mask are both in, however the snapshots split', async () => {
    const parsed = parseVideoStream(iterate(eventsFromNdjson(text('sam/pingpong_text.events.ndjson'))));
    const collector = new FrameCollector();
    const frames: number[] = [];
    for await (const snapshot of parsed) for (const b of collector.take(snapshot.records)) frames.push(b.frame), expect(b.objects[0]?.mask).not.toBeNull();
    const result = await parsed.finalResult;
    for (const b of [...collector.take(result.records), ...collector.finish()]) frames.push(b.frame);
    expect(frames).toHaveLength(134);
    expect(new Set(frames).size).toBe(134);
  });

  it('skips a torn last line instead of throwing', () => {
    const lines = text('sam/pingpong_text.events.ndjson').split('\n').filter(Boolean);
    const torn = [...lines.slice(0, 10), lines[10]!.slice(0, 40)].join('\n');
    expect(eventsFromNdjson(torn)).toHaveLength(10);
  });
});
