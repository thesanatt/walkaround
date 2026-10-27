import type { Box4, Point2 } from '../contracts/types.ts';

export interface Fit {
  scale: number;
  ox: number;
  oy: number;
  w: number;
  h: number;
}

// Contain fit of a (srcW x srcH) frame into a (dstW x dstH) canvas: one scale, centered.
export function containFit(srcW: number, srcH: number, dstW: number, dstH: number): Fit {
  const scale = Math.min(dstW / srcW, dstH / srcH);
  const w = srcW * scale;
  const h = srcH * scale;
  return { scale, ox: (dstW - w) / 2, oy: (dstH - h) / 2, w, h };
}

export function boxCenter(box: Box4 | number[]): Point2 {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box;
  return [(x1 + x2) / 2, (y1 + y2) / 2];
}

// Source pixels (the original's display size) to proxy pixels, by the ratio of the two sizes.
export function srcToProxy(p: Point2, srcSize: [number, number], proxySize: [number, number]): Point2 {
  return [(p[0] * proxySize[0]) / srcSize[0], (p[1] * proxySize[1]) / srcSize[1]];
}

export function boxSrcToProxy(b: Box4, srcSize: [number, number], proxySize: [number, number]): Box4 {
  const [x1, y1] = srcToProxy([b[0], b[1]], srcSize, proxySize);
  const [x2, y2] = srcToProxy([b[2], b[3]], srcSize, proxySize);
  return [x1, y1, x2, y2];
}

export function applyFit(p: Point2, fit: Fit): Point2 {
  return [fit.ox + p[0] * fit.scale, fit.oy + p[1] * fit.scale];
}
