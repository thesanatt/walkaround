import type { Claim, Coverage, Keyframe, PanelCoverage, PanelId } from '../contracts/types.ts';
import { PANEL_BY_ID, PANELS, panelName } from './panels.ts';
import { ruleLabel } from './rules.ts';

// Each claim exit in plain words. The rule id and its short label print beside it (rules.py E1 to E8, E2b). A pick is
// the renter's input on the claim; a claim never rewrites a closed run's sheet, so these words promise nothing there.
export const EXIT_WORDS: Record<string, string> = {
  E1: 'There is no car panel in this photo, so nothing was placed.',
  E2: 'Muse Spark and the SAM 3.1 vote did not settle on one panel. Pick the panel the bill means; your pick is recorded as your input.',
  E2b: 'Neither the words on the bill nor the photo say which one, so each panel that fits is listed. Pick the one the bill means.',
  E3: 'This panel was not filmed at pickup, so there is no dated frame to compare.',
  E4: 'This panel was only glimpsed at pickup, so nothing is compared. The sharpest pickup frame and its seconds are shown.',
  E5: 'The claim mark cannot be placed precisely on the panel, so look yourself at the dated pickup frame.',
  E6: 'Muse Spark found no mark in the marked area of the photo.',
  E7: 'The marked area covers most of the panel, so it is treated as a region, not one mark.',
  E8: 'Glare, water or dirt on the claim photo, so the most this can say is needs your eye.',
};

export function exitTitle(exit: string): string {
  return `${exit} ${ruleLabel(exit)}`;
}

// The exit a claim stopped at. claim.py leaves exit empty when no side was named (both sides shown), which is E2b.
export function effectiveExit(claim: Claim): string | null {
  if (claim.exit) return claim.exit;
  return claim.side_source === 'both_shown' && !claim.panel_id ? 'E2b' : null;
}

interface Vote {
  noun: string | null;
  panel: PanelId | null;
  score: number;
  winner: boolean;
  asked: boolean;
}

// One SAM vote row as claim.sam_vote writes it: {noun, status, matches, score, winner, panel_id}.
function voteOf(row: Record<string, unknown>): Vote {
  const panel = typeof row['panel_id'] === 'string' && PANEL_BY_ID[row['panel_id']] ? (row['panel_id'] as PanelId) : null;
  const noun = typeof row['noun'] === 'string' ? row['noun'] : null;
  const num = (k: string) => (typeof row[k] === 'number' && Number.isFinite(row[k]) ? (row[k] as number) : null);
  const score = num('score') ?? (num('matches') ?? 0) * (num('c') ?? 1);
  return { noun, panel, score, winner: row['winner'] === true, asked: row['status'] !== 'not_asked' };
}

function topVote(claim: Claim): Vote | undefined {
  const rows = claim.naming.sam_votes.map(voteOf);
  return rows.find((r) => r.winner) ?? [...rows].sort((a, b) => b.score - a.score).find((r) => r.score > 0);
}

// "14 image requests, the strongest names car door" (the row is labelled SAM 3.1 vote).
export function samVoteLine(claim: Claim): string {
  const rows = claim.naming.sam_votes.map(voteOf).filter((r) => r.asked);
  if (!rows.length) return 'no image request was made';
  const top = topVote(claim);
  const name = top ? (top.panel ? panelName(top.panel) : top.noun) : null;
  const lead = `${rows.length} image ${rows.length === 1 ? 'request' : 'requests'}`;
  return `${lead}${name ? `, the strongest names ${name}` : ', none names a panel'}; ${claim.naming.agreed ? 'they agree with Muse Spark' : 'they do not agree with Muse Spark'}`;
}

// What Muse Spark named. claim.py keeps naming.spark only when Spark's noun pins one slot, so a sided noun with no side
// (every E2b, some E2) leaves it empty. Agreed means Spark's noun is the vote winner's, so that noun is printed then;
// "no car panel" is kept for E1. TODO(contract): ClaimNaming has no field for Spark's noun, side and position, so a
// disagreement with no single Spark panel can only say "no single panel".
export function sparkNamedLine(claim: Claim): string {
  if (claim.naming.spark) return panelName(claim.naming.spark);
  if (effectiveExit(claim) === 'E1') return 'no car panel';
  const noun = topVote(claim)?.noun;
  if (claim.naming.agreed && noun) return `a ${noun}; which one is not named`;
  return 'no single panel';
}

// claim.side_from_text and claim.position_from_text: one side or position only when the words name exactly one.
// "passenger" is the right side (thresholds claim.passenger_side_is, US); LF, RF, LR and RR are report shorthands.
// claim.side_from_text: an explicit side word (left, right, LT, RT, LH, RH, passenger, driver; US passenger side right)
// beats the LF/RF/LR/RR corners, so "LT RR DOOR" is the left rear door.
function sideOf(text: string): 'left' | 'right' | null {
  const explicit = new Set<string>();
  const corner = new Set<string>();
  for (const w of text.toLowerCase().match(/\b(left|right|lt|rt|lh|rh|passengers?|drivers?|lf|rf|lr|rr)\b/g) ?? []) {
    if (['lf', 'rf', 'lr', 'rr'].includes(w)) corner.add(w[0] === 'l' ? 'left' : 'right');
    else explicit.add(['left', 'lt', 'lh'].includes(w) || w.startsWith('driver') ? 'left' : 'right');
  }
  const found = explicit.size ? explicit : corner;
  return found.size === 1 ? ([...found][0] as 'left' | 'right') : null;
}

// claim.position_from_text: front or rear ('back' is rear, FRT front, a corner by its second letter).
function positionOf(text: string): 'front' | 'rear' | null {
  const found = new Set<string>();
  for (const w of text.toLowerCase().match(/\b(front|frt|rear|back|lf|rf|lr|rr)\b/g) ?? []) found.add(['front', 'frt', 'lf', 'rf'].includes(w) ? 'front' : 'rear');
  return found.size === 1 ? ([...found][0] as 'front' | 'rear') : null;
}

// The panels of a noun that fit the words on the bill (claim.panel_candidates, read from the text the same way).
function panelsOfNoun(noun: string, text: string): PanelId[] {
  const side = sideOf(text);
  const position = positionOf(text);
  return PANELS.filter((p) => p.noun === noun)
    .map((p) => p.id)
    .filter((id) => (!side || id.startsWith(side) || !/^(left|right)_/.test(id)) && (!position || id.includes(position) || !/(front|rear)/.test(id)));
}

// The panels the renter can pick from: Spark's and the vote's after E2, every fitting slot of the noun after E2b. One
// candidate still gets its button (one tap confirms). TODO(contract): with Spark's noun on ClaimNaming, an E2 whose
// Spark answer named no single slot could offer Spark's slots too, and E2b could use Spark's side and position.
export function claimCandidates(claim: Claim): PanelId[] {
  const out: PanelId[] = [];
  const add = (p: PanelId | null | undefined) => {
    if (p && !out.includes(p)) out.push(p);
  };
  const top = topVote(claim);
  const exit = effectiveExit(claim);
  if (exit === 'E2') {
    add(claim.naming.spark);
    if (top?.panel) add(top.panel);
    else if (top?.noun) for (const p of panelsOfNoun(top.noun, claim.text)) add(p);
  } else if (exit === 'E2b') {
    add(claim.naming.spark); // the slot Spark itself named, so a side clash with the bill offers both
    const noun = (claim.naming.spark && PANEL_BY_ID[claim.naming.spark]?.noun) || top?.noun;
    if (noun) for (const p of panelsOfNoun(noun, claim.text)) add(p);
  }
  return out.slice(0, 4);
}

export function sideLine(claim: Claim): string {
  switch (claim.side_source) {
    case 'text':
      return 'from your words';
    case 'spark':
      return 'from the photo (Muse Spark)';
    case 'both_shown':
      return 'not named, so both sides are shown';
    default:
      return 'not needed';
  }
}

export function placementLine(claim: Claim): string {
  const p = claim.placement;
  if (p.method === 'none') return 'not placed';
  const how = p.method === 'full_panel' ? 'on the whole panel mask' : 'by fitting the landmarks Muse Spark named';
  return p.r_claim !== null ? `${how}, give or take ${p.r_claim.toFixed(2)} units` : how;
}

// The served copy of a claim photo, plates and faces pixelated. The upload itself is never served.
export function claimPhotoPath(claim: Claim): string | null {
  return claim.claim_id ? `claim/${claim.claim_id}.blurred.jpg` : null;
}

// claim_<n> as a number, for ordering; 0 when the claim has no id.
export function claimNumber(claim: Claim): number {
  const m = /^claim_([0-9]+)$/.exec(claim.claim_id ?? '');
  return m ? Number(m[1]) : 0;
}

// The claims of a run in the order they were placed (received_at, then the number, as claim.rebuild_claims orders
// them), one per claim: claim/<id>.json and derived/claim.json carry the same claim twice while the run is open.
// The last one is the latest. A record with no image is skipped.
export function mergeClaims(...lists: unknown[][]): Claim[] {
  const byKey = new Map<string, Claim>();
  for (const list of lists) {
    for (const raw of list) {
      const c = raw as Claim | null;
      if (!c || typeof c !== 'object' || !c.image) continue;
      byKey.set(c.claim_id ?? `${c.image.sha256}@${c.image.received_at}`, c);
    }
  }
  return [...byKey.values()].sort((a, b) => (a.image.received_at ?? '').localeCompare(b.image.received_at ?? '') || claimNumber(a) - claimNumber(b));
}

// claim.WORK_EDGE and claim.working_copy: every check, model call and the ROI box run on a copy whose long edge is at
// most 2048 px (never upscaled), and claim/<id>.blurred.jpg stores that copy. image.w and image.h are the photo as
// uploaded, so the ROI is drawn in this size. TODO(contract): put the working size on Claim instead of mirroring it.
const CLAIM_WORK_EDGE = 2048;

// Python's round(): halves go to the even neighbour.
function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function claimWorkSize(image: { w: number; h: number }): [number, number] {
  const scale = Math.min(1, CLAIM_WORK_EDGE / Math.max(image.w, image.h, 1));
  if (scale >= 1) return [image.w, image.h];
  return [Math.max(1, roundHalfEven(image.w * scale)), Math.max(1, roundHalfEven(image.h * scale))];
}

// claim.PICKUP_SIDE: the pickup walk and its top-up clips are the pickup side of a claim.
export const PICKUP_SIDE = /^(pickup|topup_[0-9]+)$/;
const STATE_RANK: Record<string, number> = { filmed: 4, glimpsed: 3, not_filmed: 2, not_seen: 1, not_run: 0 };

// claim.pickup_coverage: the panel's best state over the pickup side, by state then seconds; null when no walk lists it.
export function pickupCoverage(coverage: Coverage | null | undefined, panelId: string): PanelCoverage | null {
  let best: PanelCoverage | null = null;
  for (const role of Object.keys(coverage?.per_role ?? {}).sort()) {
    if (!PICKUP_SIDE.test(role)) continue;
    const pc = coverage!.per_role[role]?.[panelId as PanelId];
    if (!pc) continue;
    const better = !best || (STATE_RANK[pc.state] ?? 0) > (STATE_RANK[best.state] ?? 0) || ((STATE_RANK[pc.state] ?? 0) === (STATE_RANK[best.state] ?? 0) && pc.seconds > best.seconds);
    if (better) best = pc;
  }
  return best;
}

// claim._pickup_keyframe: the coverage's best keyframe when it exists, else the panel's highest-scored pickup-side
// keyframe (ties to the larger id, as Python's max over (total, kf_id) does).
export function claimPickupKeyframe(keyframes: Keyframe[], panelId: string, bestKf: string | null | undefined): Keyframe | undefined {
  const named = bestKf ? keyframes.find((k) => k.kf_id === bestKf) : undefined;
  if (named) return named;
  let best: Keyframe | undefined;
  for (const k of keyframes) {
    if (k.panel_id !== panelId || !PICKUP_SIDE.test(k.role)) continue;
    if (!best || k.score.total > best.score.total || (k.score.total === best.score.total && k.kf_id > best.kf_id)) best = k;
  }
  return best;
}

// "the pickup walk" or "top-up clip 1".
export function pickupSideName(role: string): string {
  const m = /^topup_([0-9]+)$/.exec(role);
  return m ? `top-up clip ${m[1]}` : 'the pickup walk';
}
