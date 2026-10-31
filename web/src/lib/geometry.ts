import type { Box4, Crop, Point2 } from '../contracts/types.ts';

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
