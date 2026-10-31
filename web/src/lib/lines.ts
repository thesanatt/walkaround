import type { FfprobeInfo, MarkTrack, SizeEstimate, VideoRecord } from '../contracts/types.ts';

// sheet._TIRE_CODE and sheet.tire_diameter_in: D = 2 x (SW x AR / 2540 + RD / 2) inches from a code like 225/60R16.
const TIRE_CODE = /^\s*(\d{3})\s*\/\s*(\d{2})\s*[A-Za-z]*\s*(\d{2}(?:\.\d)?)\s*$/;

export function tireDiameterIn(code: string | null | undefined): number | null {
  const m = TIRE_CODE.exec(code ?? '');
  if (!m) return null;
  return 2 * ((Number(m[1]) * Number(m[2])) / 2540 + Number(m[3]) / 2);
}

// sheet.size_line word for word for a stated estimate. No estimate at all prints nothing here (the sheet prints its
// no-ruler line).
export function sizeLine(size: SizeEstimate | null | undefined): string | null {
  if (!size) return null;
  if (size.refused_reason) return `size not stated: ${size.refused_reason}`;
  if (size.inches === null) return 'size not stated: no ruler qualified';
  let ruler = 'not recorded';
  if (size.ruler === 'plate') ruler = 'the license plate, 12 in';
  else if (size.ruler === 'wheel') {
    const diameter = tireDiameterIn(size.tire_code);
    ruler = diameter ? `the wheel, ${diameter.toFixed(1)} in from ${size.tire_code!.trim()}, typed by you` : 'the wheel';
  }
  let text = `about ${size.inches.toFixed(1)} in`;
  if (size.bound_lo !== null && size.bound_hi !== null) text += `, between ${size.bound_lo.toFixed(1)} and ${size.bound_hi.toFixed(1)} in`;
  return `${text} (ruler: ${ruler})`;
}

function num(v: number | null): string {
  return v === null ? 'n/a' : v.toFixed(2);
}

// sheet.staysput_line word for word: "drift 0.03, b1 0.20, b2 0.01, not panned; fixed", caps appended; null when not run.
export function staysPutLine(mt: MarkTrack | null | undefined): string | null {
  if (!mt || mt.verdict === 'not_run') return null;
  if (mt.verdict === 'no_track') return 'no track (SAM did not follow the mark through the clip)';
  if (mt.verdict === 'inconclusive') return mt.panned ? 'inconclusive (the camera panned: b1 under its floor, b2 over)' : 'inconclusive (camera did not move enough)';
  let line = `drift ${num(mt.drift_rms)}, b1 ${num(mt.b1)}, b2 ${num(mt.b2)}, ${mt.panned ? 'panned' : 'not panned'}; ${mt.verdict}`;
  if (mt.curvature_cap) line += '; curvature cap';
  if (mt.highlight_cap) line += '; highlight cap';
  return line;
}

// sheet._wall_clock: the camera's creation_time plus pts, as ISO UTC; null when the camera wrote no time. The receipt
// time never stands in: it is when the upload finished, not when the walk was filmed.
export function wallClock(ffprobe: FfprobeInfo | null | undefined, video: VideoRecord | null | undefined, pts: number): string | null {
  const camera = ffprobe?.creation_time ?? video?.ffprobe?.creation_time ?? null;
  if (!camera || camera === 'not present') return null;
  const t = Date.parse(camera);
  return Number.isFinite(t) ? new Date(t + pts * 1000).toISOString() : null;
}
