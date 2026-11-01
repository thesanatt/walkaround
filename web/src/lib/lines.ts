import type { FfprobeInfo, KeyframePairTry, Mark, MarkTrack, PanelMatch, RunManifest, SizeEstimate, VideoRecord } from '../contracts/types.ts';
import { neighbourLine } from './neighbours.ts';
import { ruleLabel } from './rules.ts';

// sheet._TIRE_CODE and sheet.tire_diameter_in: D = 2 x (SW x AR / 2540 + RD / 2) inches from a code like 225/60R16.
const TIRE_CODE = /^\s*(\d{3})\s*\/\s*(\d{2})\s*[A-Za-z]*\s*(\d{2}(?:\.\d)?)\s*$/;

export function tireDiameterIn(code: string | null | undefined): number | null {
  const m = TIRE_CODE.exec(code ?? '');
  if (!m) return null;
  return 2 * ((Number(m[1]) * Number(m[2])) / 2540 + Number(m[3]) / 2);
}

// sheet.size_line word for word for a stated estimate, each followed by why every other ruler next to the mark did not
// size it (ruler_notes). No estimate at all prints nothing here (the sheet prints its no-ruler line).
export function sizeLine(size: SizeEstimate | null | undefined): string | null {
  if (!size) return null;
  const notes = (size.ruler_notes ?? []).map((note) => `; ${note}`).join('');
  if (size.refused_reason) return `size not stated: ${size.refused_reason}${notes}`;
  if (size.inches === null) return 'size not stated: no ruler qualified';
  let ruler = 'not recorded';
  if (size.ruler === 'plate') ruler = 'the license plate, 12 in';
  else if (size.ruler === 'wheel') {
    const diameter = tireDiameterIn(size.tire_code);
    ruler = diameter ? `the wheel, ${diameter.toFixed(1)} in from ${size.tire_code!.trim()}, typed by you` : 'the wheel';
  }
  let text = `about ${size.inches.toFixed(1)} in`;
  if (size.bound_lo !== null && size.bound_hi !== null) text += `, between ${size.bound_lo.toFixed(1)} and ${size.bound_hi.toFixed(1)} in`;
  return `${text} (ruler: ${ruler}${notes})`;
}

function num(v: number | null): string {
  return v === null ? 'n/a' : v.toFixed(2);
}

// thresholds staysput.cap_enabled as the run froze it. The lead turned it off after D1 (docs/D1_MEMO.md note 2):
// real paint on pair C was labelled sliding, so until a recalibration turns it on Stays Put records and shows its
// tracks but caps no label, and the page states none of its verdicts as fact. A run without the key is uncalibrated.
export function staysPutCalibrated(run: Pick<RunManifest, 'thresholds'> | null | undefined): boolean {
  const sp = run?.thresholds?.['staysput'];
  return Boolean(sp && typeof sp === 'object' && (sp as Record<string, unknown>)['cap_enabled'] === true);
}

// What SAM 3.1 measured on the clip, with no verdict: "drift 0.21, b1 0.30, b2 0.01, not panned".
function staysPutNumbers(mt: MarkTrack): string {
  return `drift ${num(mt.drift_rms)}, b1 ${num(mt.b1)}, b2 ${num(mt.b2)}, ${mt.panned ? 'panned' : 'not panned'}`;
}

// sheet.staysput_line word for word when calibrated: "drift 0.03, b1 0.20, b2 0.01, not panned; fixed", caps
// appended; null when not run. Uncalibrated, only the measured numbers (the verdict and its caps are left out). A clip
// with the neighbour test adds its line (lib/neighbours.ts) as the sheet does.
export function staysPutLine(mt: MarkTrack | null | undefined, calibrated: boolean): string | null {
  const line = staysPutDriftLine(mt, calibrated);
  const near = line !== null && mt && mt.verdict !== 'no_track' ? neighbourLine(mt) : null;
  return near ? `${line}; ${near}` : line;
}

function staysPutDriftLine(mt: MarkTrack | null | undefined, calibrated: boolean): string | null {
  if (!mt || mt.verdict === 'not_run') return null;
  if (mt.verdict === 'no_track') return 'no track (SAM did not follow the mark through the clip)';
  if (!calibrated) return staysPutNumbers(mt);
  if (mt.verdict === 'inconclusive' && !mt.b_pass) return 'inconclusive (camera did not move enough)';
  if (mt.verdict === 'inconclusive' && mt.drift_rms === null) return 'inconclusive (the whole panel was in view on too few frames)';
  let line = `${staysPutNumbers(mt)}; ${mt.verdict}`;
  if (mt.verdict === 'inconclusive') line += ` (between fixed and sliding; sign test ${num(mt.sign_agreement)} on ${mt.pairs_n} pairs)`;
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

// The one-line caption under a Stays Put clip: what the track decided, in plain words. A mark on the paint moves
// with the panel; a reflection slides across it as the camera moves. Uncalibrated, it says only that the mark was
// tracked; a lost track or a mark Stays Put never ran on is still said.
export function staysPutCaption(mt: MarkTrack, calibrated: boolean): string {
  if (!calibrated && mt.verdict !== 'no_track' && mt.verdict !== 'not_run') {
    return mt.is_control ? 'Tracked the same way. Stays Put is not calibrated yet.' : 'Tracked through its own clip. Stays Put is not calibrated yet and changes no label.';
  }
  switch (mt.verdict) {
    case 'fixed':
      return 'Stays put: paint';
    case 'sliding':
      return mt.is_control ? 'Slides with the camera: reflection' : 'Slides with the camera: likely a reflection';
    case 'inconclusive':
      if (!mt.b_pass) return 'Too little camera motion to tell';
      return mt.drift_rms === null ? 'Too few frames with the whole panel in view to tell' : 'Between paint and reflection: not enough to call';
    case 'no_track':
      return 'SAM 3.1 lost it in this clip';
    case 'not_run':
      return 'Stays Put did not run on this mark';
  }
}

// The caption's class: the verdict colors it only when calibrated.
export function staysPutTone(mt: MarkTrack, calibrated: boolean): string {
  return calibrated || mt.verdict === 'no_track' || mt.verdict === 'not_run' ? mt.verdict : 'uncalibrated';
}

// What the control clip is called: a reflection only once Stays Put is calibrated.
export function controlName(calibrated: boolean): string {
  return calibrated ? 'reflection control' : 'control clip';
}

// "scuff, S", and "from your words" on a mark Muse Spark boxed because the renter named it.
export function markLabel(m: Mark): string {
  return `${m.type}, ${m.size_class}${m.source === 'narration_seed' ? ', from your words' : ''}`;
}

// match.how_checked word for word (the sheet's "how it was checked" line, thresholds matching.next_keyframe_pair,
// 11/1): the keyframe pair a panel was compared on and every pair tried before it with the gates each failed; for a
// panel no pair passed, every pair tried. null when the match recorded no tries (keyframes in one walk only, or a run
// matched before 11/1).
const PANEL_GATES = ['G1', 'G6', 'G7', 'G10'];

function joined(items: string[], last: string): string {
  return items.length === 1 ? items[0]! : `${items.slice(0, -1).join(', ')} ${last} ${items[items.length - 1]!}`;
}

function gateWords(gate: string, t: KeyframePairTry): string {
  const label = ruleLabel(gate);
  if (gate === 'G4' || gate === 'G8') {
    if (t.mask_iou !== null) return `${gate} ${label} ${t.mask_iou.toFixed(2)}`;
    return `${gate} ${label}: ${t.frames_agree ? 'not measured' : 'panel axes disagree'}`;
  }
  return `${gate} ${label}`;
}

export function keyframePairLine(match: PanelMatch | null | undefined): string | null {
  const tries = match?.kf_pairs_tried ?? [];
  if (!tries.length) return null;
  const pairWords = (t: KeyframePairTry) => `${t.kf_pair[0]} with ${t.kf_pair[1]}`;
  const tried = (t: KeyframePairTry) => `${pairWords(t)} (${t.gates.map((g) => gateWords(g, t)).join(', ')})`;
  const used = tries.find((t) => !t.gates.length);
  if (used) {
    const before = tries.filter((t) => t !== used);
    if (!before.length) return `compared on keyframes ${pairWords(used)}, the first pair tried`;
    return `compared on keyframes ${pairWords(used)}; tried first: ${before.map(tried).join('; ')}`;
  }
  const first = tries[0]!;
  const panelLevel = first.gates.filter((g) => PANEL_GATES.includes(g));
  if (panelLevel.length) {
    const which = panelLevel.length === 1 ? `${panelLevel[0]!} is a panel gate` : `${joined(panelLevel, 'and')} are panel gates`;
    return `not compared: keyframes ${tried(first)}; ${which}, so no other pair was tried`;
  }
  if (tries.length === 1) return `not compared: keyframes ${tried(first)}, and the panel has no other keyframe pair`;
  return `not compared: none of the ${tries.length} keyframe pairs passed every gate; tried: ${tries.map(tried).join('; ')}`;
}
