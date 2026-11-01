import type { MarkTrack, NeighbourTest } from '../contracts/types.ts';

// Stays Put by neighbours (docs/STAYS_PUT_ANCHORS.md): SAM 3.1 tracks rigid neighbours (door handles, mirror, wheels,
// lights, plate, windows, other marks) in the mark's own clip; paint moves with them, a reflection slides against them.
// The ratio is the mark's residual against an affine fit to the neighbours over the neighbours' own leave-one-out
// residual. It records and shows; it changes no verdict while thresholds staysput.cap_enabled is off.

export const NEIGHBOUR_NOTE = 'Recorded only: it changes no verdict yet.';

function ran(mt: MarkTrack | null | undefined): NeighbourTest | null {
  const nt = mt?.neighbour_test ?? null;
  return nt && nt.verdict !== 'not_run' ? nt : null;
}

function x(v: number): string {
  return `${v.toFixed(1)}x`;
}

// sheet.neighbour_line word for word: "neighbour test: moves with its 6 neighbours, 0.6x their wobble over the clip and
// 0.8x within 0.5 s; fixed". null when the clip has no neighbour test or its lane gave no answer.
export function neighbourLine(mt: MarkTrack | null | undefined): string | null {
  const nt = ran(mt);
  if (!nt) return null;
  if (nt.verdict === 'no_track') return 'neighbour test: SAM did not follow the mark through enough of the clip';
  if (nt.ratio_full === null || nt.ratio_near === null) return `neighbour test: ${nt.reason ?? 'too few frames to measure'}; inconclusive`;
  const ratios = (what: string) => `${x(nt.ratio_full!)} ${what} over the clip and ${x(nt.ratio_near!)} within ${String(nt.near_s)} s`;
  let words: string;
  if (nt.verdict === 'fixed') words = `moves with its ${nt.used} neighbours, ${ratios('their wobble')}`;
  else if (nt.verdict === 'sliding') words = `slides against its ${nt.used} neighbours, ${ratios('their wobble')}`;
  else words = ratios(`its ${nt.used} neighbours' wobble`);
  return `neighbour test: ${words}; ${nt.verdict}`;
}

// The short line next to the clip: "Moves with its 6 neighbours, 0.6x their wobble (0.8x within 0.5 s)".
export function neighbourCaption(mt: MarkTrack | null | undefined): string | null {
  const nt = ran(mt);
  if (!nt) return null;
  if (nt.verdict === 'no_track') return 'SAM 3.1 lost the mark, so it could not be compared with its neighbours';
  if (nt.ratio_full === null || nt.ratio_near === null) return `Not compared with its neighbours: ${nt.reason ?? 'too few frames to measure'}`;
  const near = `(${x(nt.ratio_near)} within ${String(nt.near_s)} s)`;
  if (nt.verdict === 'fixed') return `Moves with its ${nt.used} neighbours, ${x(nt.ratio_full)} their wobble ${near}`;
  if (nt.verdict === 'sliding') return `Slides against its ${nt.used} neighbours, ${x(nt.ratio_full)} their wobble ${near}`;
  return `Not clear against its ${nt.used} neighbours, ${x(nt.ratio_full)} their wobble ${near}`;
}

// The caption's class: the test's verdict colors it only once Stays Put is calibrated.
export function neighbourTone(mt: MarkTrack, calibrated: boolean): string {
  const nt = ran(mt);
  if (!nt) return 'not_run';
  return calibrated || nt.verdict === 'no_track' ? nt.verdict : 'uncalibrated';
}

export interface NeighbourMarker {
  id: string;
  kind: string;
  x: number;
  y: number;
  used: boolean;
}

// The neighbours SAM 3.1 placed on clip frame f, in clip pixels (the page draws a small ring on each).
export function neighbourMarkers(mt: MarkTrack, frame: number): NeighbourMarker[] {
  const nt = mt.neighbour_test;
  if (!nt) return [];
  const out: NeighbourMarker[] = [];
  for (const nb of nt.neighbours) {
    const at = nb.centres.find((c) => c[0] === frame);
    if (at) out.push({ id: nb.object_id, kind: nb.kind, x: at[1], y: at[2], used: nb.used });
  }
  return out;
}
