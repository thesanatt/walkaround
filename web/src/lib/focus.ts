import type { Keyframe, Mark, MarkTrack, MatchCandidate, PanelMatch, PanelTrack, Verdict } from '../contracts/types.ts';
import { laneId } from './panels.ts';

export function bestKeyframe(keyframes: Keyframe[], role: string, panelId: string): Keyframe | undefined {
  let best: Keyframe | undefined;
  for (const k of keyframes) if (k.role === role && k.panel_id === panelId && (!best || k.score.total > best.score.total)) best = k;
  return best;
}

// A pair code joined: d under tau_eye_r. A pair at or past it stays in matches.json only so verdicts_for can print L5,
// one verdict per mark, so it never gets a join line.
export function joined(pair: MatchCandidate): boolean {
  return pair.d < pair.tau_eye_r;
}

export function joinedPairs(match: PanelMatch | undefined): MatchCandidate[] {
  return match?.state === 'comparable' ? match.pairs.filter(joined) : [];
}

export interface PanelFocus {
  verdict: Verdict | null;
  pair: MatchCandidate | null; // the joined pair the verdict is about, when it is about one
}

function pairOf(v: Verdict, pairs: MatchCandidate[]): MatchCandidate | null {
  if (v.mark_ids.length !== 2) return null;
  return pairs.find((p) => v.mark_ids.includes(p.pickup_mark) && v.mark_ids.includes(p.return_mark)) ?? null;
}

// The order a panel's verdicts are told in, strongest first: new, then not seen at return, then already there, then
// needs your eye (not comparable last). The sort is stable, so verdicts with one label keep the order code wrote.
const STRENGTH: Record<string, number> = { new_or_unseen: 0, not_seen_at_return: 1, already_there: 2, needs_your_eye: 3, not_comparable: 4 };

export function byStrength(verdicts: readonly Verdict[]): Verdict[] {
  const rank = (v: Verdict) => STRENGTH[v.label] ?? 5;
  return [...verdicts].sort((a, b) => rank(a) - rank(b));
}

// The one verdict a panel's views lead with: the strongest label on the panel, and among verdicts with that label the
// first about a joined pair, else the first. The heading, the first card, the match-cut alignment and the Stays Put
// trail all follow this one.
export function panelFocus(panelId: string | null, match: PanelMatch | undefined, verdicts: Verdict[]): PanelFocus {
  if (!panelId) return { verdict: null, pair: null };
  const pairs = joinedPairs(match);
  const own = byStrength(verdicts.filter((v) => v.panel_id === panelId));
  const top = own.filter((v) => v.label === own[0]?.label);
  for (const v of top) {
    const pair = pairOf(v, pairs);
    if (pair) return { verdict: v, pair };
  }
  const first = top[0] ?? null;
  return { verdict: first, pair: first ? pairOf(first, pairs) : null };
}

// A panel's cards: the focus verdict first, then the rest strongest first.
export function panelCards(panelId: string, focus: PanelFocus, verdicts: Verdict[]): Verdict[] {
  const own = byStrength(verdicts.filter((v) => v.panel_id === panelId));
  return focus.verdict && own.includes(focus.verdict) ? [focus.verdict, ...own.filter((v) => v !== focus.verdict)] : own;
}

// The keyframes a two-frame view shows. A mark's point_crop is in its own keyframe's crop, and dedup keeps the medoid
// from any of a walk's keyframes, so the focus marks' own keyframes come first (the pair's two, or a one-mark
// verdict's mark for its walk); then match.kf_pair; then the best-scored one.
export function paneKeyframes(
  panelId: string,
  match: PanelMatch | undefined,
  focus: PanelFocus,
  marks: Map<string, Mark>,
  keyframes: Keyframe[],
): { a: Keyframe | undefined; b: Keyframe | undefined } {
  const byId = (id: string | null | undefined) => (id ? keyframes.find((k) => k.kf_id === id) : undefined);
  const focusMark = (role: 'pickup' | 'return') =>
    focus.pair ? (role === 'pickup' ? focus.pair.pickup_mark : focus.pair.return_mark) : focus.verdict?.mark_ids.find((id) => marks.get(id)?.role === role);
  const pick = (role: 'pickup' | 'return', pairKf: string | undefined) => {
    const markId = focusMark(role);
    return byId(markId ? marks.get(markId)?.kf_id : undefined) ?? byId(pairKf) ?? bestKeyframe(keyframes, role, panelId);
  };
  return { a: pick('pickup', match?.kf_pair?.[0]), b: pick('return', match?.kf_pair?.[1]) };
}

// The joined pairs whose two marks both sit on the keyframes shown: only those get a line between the frames.
export function drawnPairs(match: PanelMatch | undefined, marks: Map<string, Mark>, kfA: Keyframe | undefined, kfB: Keyframe | undefined): MatchCandidate[] {
  if (!kfA || !kfB) return [];
  return joinedPairs(match).filter((p) => marks.get(p.pickup_mark)?.kf_id === kfA.kf_id && marks.get(p.return_mark)?.kf_id === kfB.kf_id);
}

// The panel's walk marks found on keyframes the view does not show (their own crops show them).
export function offPaneMarks(panelId: string, marks: Mark[], kfA: Keyframe | undefined, kfB: Keyframe | undefined): Mark[] {
  return marks.filter((m) => m.panel_id === panelId && (m.role === 'pickup' || m.role === 'return') && m.kf_id !== kfA?.kf_id && m.kf_id !== kfB?.kf_id);
}

// "<lane>|<SAM id>" for every object of a walk's tracks on this panel: a door merged from several fragments carries
// several ids, and the frames around a keyframe can carry any of them.
export function panelObjectKeys(tracks: PanelTrack[], role: string, panelId: string | null): Set<string> {
  const keys = new Set<string>();
  if (!panelId) return keys;
  for (const t of tracks) if (t.role === role && t.panel_id === panelId) for (const id of t.sam_object_ids) keys.add(`${laneId(t.noun)}|${id}`);
  return keys;
}

// The Stays Put track the moment shows: the focus verdict's own pickup mark. With no verdict on the panel, a pickup
// mark of this panel; never a mark of another panel.
export function focusTrack(focus: PanelFocus, panelId: string | null, markTracks: MarkTrack[], marks: Map<string, Mark>): MarkTrack | undefined {
  const usable = markTracks.filter((m) => m.role === 'pickup' && !m.is_control && m.clip);
  if (focus.verdict) {
    const ids = focus.pair ? [focus.pair.pickup_mark] : focus.verdict.mark_ids.filter((id) => marks.get(id)?.role !== 'return');
    return usable.find((m) => ids.includes(m.mark_id));
  }
  return usable.find((m) => marks.get(m.mark_id)?.panel_id === panelId);
}
