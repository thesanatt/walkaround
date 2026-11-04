import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { CropFigure } from '../src/components/CropView.tsx';
import { Frame, zoomWindow } from '../src/components/Frame.tsx';
import type { Keyframe, Mark } from '../src/contracts/types.ts';

// Box fix 11/1: every overlay draws the mark box in the crop's own pixels, inside the same SVG user space as the crop
// image (image at 0,0 at the crop's stored size), so CSS scaling, letterboxing and devicePixelRatio move both together.

const kf = (w: number, h: number): Keyframe =>
  ({
    kf_id: 'pickup_hood_kf1', role: 'pickup', panel_id: 'hood', track_id: 't', f_proxy: 1, pts: 1, aspect_ratio: 1,
    box_src: [0, 0, w, h], crop: { path: 'crops/pickup/pickup_hood_kf1.jpg', sha256: null, w, h, scale: 1, dx: 0, dy: 0 },
    mask_png: null, moments: null, front_sign: 1, window_subtracted: false, marks_status: 'complete',
    conditions: { wet: false, dirty: false, glare: false, dark: false },
  }) as unknown as Keyframe;

const mark = (b: [number, number, number, number], extra: Partial<Mark> = {}): Mark =>
  ({
    mark_id: 'pickup_hood_kf1_m1', role: 'pickup', panel_id: 'hood', kf_id: 'pickup_hood_kf1', source: 'spark_box', type: 'chip',
    confidence: 'high', size_class: 'S', description: '', grid_point: null, grid_bbox: null, bbox_crop: b,
    point_crop: [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], bbox_src: b, point_src: null, w: null, r: null, size_units: null,
    size_px_src: null, single_frame: false, cluster_id: null, merged_from: [], spark_request_id: null, caps: [], ...extra,
  }) as Mark;

const nums = (s: string | undefined) => (s ?? '').split(/[ ,]+/).filter(Boolean).map(Number);
const attr = (html: string, tag: string, name: string): string[] => [...html.matchAll(new RegExp(`<${tag}[^>]*?\\s${name}="([^"]*)"`, 'g'))].map((m) => m[1]!);

describe('mark boxes sit on the crop pixels they name', () => {
  for (const [w, h, b] of [
    [1280, 441, [705, 111, 725, 119]],
    [938, 1280, [882, 610, 895, 624]],
    [1280, 1280, [10, 1200, 40, 1270]],
  ] as const) {
    it(`the finding frame, crop ${w}x${h}`, () => {
      const html = renderToStaticMarkup(createElement(Frame, { kf: kf(w, h), role: 'pickup', boxes: [{ box: [...b], tone: 'new' }] }));
      expect(attr(html, 'image', 'width')).toEqual([String(w)]);
      expect(attr(html, 'image', 'height')).toEqual([String(h)]);
      expect(html).not.toMatch(/<image[^>]*\s(x|y|preserveAspectRatio)=/);
      const [vx, vy, vw, vh] = nums(/viewBox="([^"]+)"/.exec(html)?.[1]);
      expect(vx! <= b[0] && vy! <= b[1] && vx! + vw! >= b[2] && vy! + vh! >= b[3]).toBe(true);
      expect([vx, vy, vw, vh]).toEqual(zoomWindow(w, h, [[...b]], true, 280, 7, 2 / 3));
      // the box is centred on the mark in screen space, at least 56 px a side
      const style = /class="frame-box new" style="([^"]+)"/.exec(html)?.[1] ?? '';
      const pct = (name: string) => Number(new RegExp(`${name}:([0-9.]+)%`).exec(style)?.[1]);
      expect(pct('left')).toBeCloseTo((100 * ((b[0] + b[2]) / 2 - vx!)) / vw!, 2);
      expect(pct('top')).toBeCloseTo((100 * ((b[1] + b[3]) / 2 - vy!)) / vh!, 2);
      expect(style).toContain('width:max(56px,');
    });

    it(`the crop figure, crop ${w}x${h}`, () => {
      const html = renderToStaticMarkup(createElement(CropFigure, { kf: kf(w, h), marks: [mark([...b])], title: 'pickup' }));
      expect(/viewBox="([^"]+)"/.exec(html)?.[1]).toBe(`0 0 ${w} ${h}`);
      expect(attr(html, 'img', 'width')).toEqual([String(w)]);
      expect(attr(html, 'rect', 'x').map(Number)).toEqual([b[0]]);
      expect(attr(html, 'rect', 'y').map(Number)).toEqual([b[1]]);
    });
  }

  it('draws the refined box, not the one Spark gave', () => {
    const refined = mark([110, 102, 114, 106], {
      box_refine: { used: 'sam', spark_bbox_crop: [100, 100, 140, 110], spark_point_crop: [120, 105], sam_bbox_crop: [110, 102, 114, 106], sam_c: 0.7, mask_px: 16, inside: 1, area_ratio: 0.04, sam_request_id: null, note: '' },
    });
    const html = renderToStaticMarkup(createElement(CropFigure, { kf: kf(400, 300), marks: [refined], title: 'pickup' }));
    expect(attr(html, 'rect', 'x').map(Number)).toEqual([110]);
  });
});

describe('how it was checked names the box', () => {
  it('says which box a mark uses', async () => {
    const { boxLine } = await import('../src/components/MatchPane.tsx');
    const base = mark([1, 1, 5, 5]);
    expect(boxLine(base)).toBe('');
    const r = { spark_bbox_crop: [1, 1, 5, 5] as [number, number, number, number], spark_point_crop: null, sam_bbox_crop: null, sam_c: null, mask_px: null, inside: null, area_ratio: null, sam_request_id: null };
    expect(boxLine({ ...base, box_refine: { ...r, used: 'sam', note: '' } })).toBe(' (box tightened with SAM 3.1)');
    expect(boxLine({ ...base, box_refine: { ...r, used: 'spark', note: 'SAM found nothing in the box' } })).toBe(" (Muse Spark's box: SAM found nothing in the box)");
  });
});
