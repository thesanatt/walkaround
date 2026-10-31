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

// The one verdict a panel's views lead with: its first "already there", else the first verdict about a joined pair,
// else its first verdict. The card label, the match-cut alignment and the Stays Put trail all follow this one.
export function panelFocus(panelId: string | null, match: PanelMatch | undefined, verdicts: Verdict[]): PanelFocus {
  if (!panelId) return { verdict: null, pair: null };
  const pairs = joinedPairs(match);
  const own = verdicts.filter((v) => v.panel_id === panelId);
  const already = own.find((v) => v.label === 'already_there');
  if (already) return { verdict: already, pair: pairOf(already, pairs) };
  for (const v of own) {
    const pair = pairOf(v, pairs);
    if (pair) return { verdict: v, pair };
  }
  return { verdict: own[0] ?? null, pair: null };
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
