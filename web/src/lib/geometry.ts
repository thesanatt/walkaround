import type { Box4, Crop, Keyframe, Point2 } from '../contracts/types.ts';

export interface Fit {
  scale: number;
  ox: number;
  oy: number;
  w: number;
  h: number;
}

// Contain fit of a (srcW x srcH) frame into a (dstW x dstH) canvas: one scale, centred.
export function containFit(srcW: number, srcH: number, dstW: number, dstH: number): Fit {
  const scale = Math.min(dstW / srcW, dstH / srcH);
  const w = srcW * scale;
  const h = srcH * scale;
  return { scale, ox: (dstW - w) / 2, oy: (dstH - h) / 2, w, h };
}

export function boxCenter(box: readonly number[]): Point2 {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box;
  return [(x1 + x2) / 2, (y1 + y2) / 2];
}

// Source pixels to crop pixels (contracts.Crop: crop (u, v) is source (dx + u / scale, dy + v / scale)).
export function boxSrcToCrop(b: Box4, crop: Crop): Box4 {
  return [(b[0] - crop.dx) * crop.scale, (b[1] - crop.dy) * crop.scale, (b[2] - crop.dx) * crop.scale, (b[3] - crop.dy) * crop.scale];
}

export function pointSrcToCrop(p: Point2, crop: Crop): Point2 {
  return [(p[0] - crop.dx) * crop.scale, (p[1] - crop.dy) * crop.scale];
}

export interface CutTransform {
  s: number;
  tx: number;
  ty: number;
}

// A keyframe's panel centre and size in its own crop pixels: the mask moments when present, else the panel box.
export function panelAnchor(kf: Keyframe): { c: Point2; r: number } | null {
  const m = kf.moments;
  if (m && Number.isFinite(m.cx) && Number.isFinite(m.cy) && m.mu20 + m.mu02 > 0) return { c: [m.cx, m.cy], r: Math.sqrt(m.mu20 + m.mu02) };
  if (!kf.crop) return null;
  const b = boxSrcToCrop(kf.box_src, kf.crop);
  return { c: boxCenter(b), r: Math.hypot(b[2] - b[0], b[3] - b[1]) / 2 };
}

// The match cut: the return crop scaled and moved so its panel centre lands on the pickup panel's centre and its
// panel spread matches. Return crop pixel p shows at s * p + (tx, ty) in pickup crop pixels. The scale is held to
// [2/3, 3/2] so a bad mask cannot blow the frame up; with no anchor on either side the frames are not moved.
export function matchCutTransform(kfA: Keyframe, kfB: Keyframe): CutTransform {
  const a = panelAnchor(kfA);
  const b = panelAnchor(kfB);
  if (!a || !b) return { s: 1, tx: 0, ty: 0 };
  const s = Math.min(1.5, Math.max(2 / 3, b.r > 0 ? a.r / b.r : 1));
  return { s, tx: a.c[0] - s * b.c[0], ty: a.c[1] - s * b.c[1] };
}

export function applyCut(p: Point2, t: CutTransform): Point2 {
  return [t.s * p[0] + t.tx, t.s * p[1] + t.ty];
}

export function applyCutBox(b: Box4, t: CutTransform): Box4 {
  return [t.s * b[0] + t.tx, t.s * b[1] + t.ty, t.s * b[2] + t.tx, t.s * b[3] + t.ty];
}
