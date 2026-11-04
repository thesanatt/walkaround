import type { Box4, Keyframe, Mark, PanelMatch, RegistrationMark, Verdict, VerdictLabel } from '../contracts/types.ts';
import { bestKeyframe } from './focus.ts';
import { panelName } from './panels.ts';
import { joinWords } from './result.ts';

// The result screen's list and the renter's own checks. Code's labels (the sealed sheet) never change here: a check
// is the renter's answer, stored beside the run in this browser and always shown as theirs.

export type Group = 'new' | 'eye' | 'already' | 'gone';
export type Answer = 'gone' | 'still' | 'new' | 'already';

export const GROUP_ORDER: Group[] = ['new', 'eye', 'already', 'gone'];

export const GROUP_WORD: Record<Group, string> = {
  new: 'New',
  eye: 'Needs your eye',
  already: 'Already there',
  gone: 'Gone at return',
};

export const ANSWER_WORD: Record<Answer, string> = {
  gone: 'Gone at return',
  still: 'Still there',
  new: 'New',
  already: 'Already there',
};

export const ANSWER_GROUP: Record<Answer, Group> = { gone: 'gone', still: 'already', new: 'new', already: 'already' };

const LABEL_GROUP: Partial<Record<VerdictLabel, Group>> = {
  new_or_unseen: 'new',
  needs_your_eye: 'eye',
  already_there: 'already',
  not_seen_at_return: 'gone',
};

export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export interface Finding {
  v: Verdict;
  system: Group; // code's label, as sealed
  answer: Answer | null; // the renter's check, needs your eye only
  group: Group; // where the list counts it: the renter's answer when there is one
  words: string | null; // Muse Spark's words for the mark
}

// The walks that hold a mark of this verdict.
export function walksOf(v: Verdict, marks: Map<string, Mark>): Set<string> {
  return new Set(v.mark_ids.map((id) => marks.get(id)?.role).filter((r): r is NonNullable<typeof r> => r !== undefined));
}

// The two answers a renter can give: a mark seen at pickup only is gone or still there, one seen at return only is
// new or already there.
export function answersFor(v: Verdict, marks: Map<string, Mark>): [Answer, Answer] {
  const w = walksOf(v, marks);
  if (w.has('pickup') && !w.has('return')) return ['gone', 'still'];
  if (w.has('return') && !w.has('pickup')) return ['new', 'already'];
  return ['already', 'new'];
}

function wordsOf(v: Verdict, marks: Map<string, Mark>): string | null {
  for (const id of v.mark_ids) {
    const d = marks.get(id)?.description;
    if (d) return d;
  }
  return null;
}

// Every finding the result screen lists, in code's order: new, needs your eye, already there, gone at return. Panels
// not compared stay on the sealed sheet.
export function findingsOf(verdicts: Verdict[], marks: Map<string, Mark>, checks: Record<string, Answer>): Finding[] {
  const out: Finding[] = [];
  for (const group of GROUP_ORDER)
    for (const v of verdicts) {
      if (LABEL_GROUP[v.label] !== group) continue;
      const answer = v.label === 'needs_your_eye' ? (checks[v.verdict_id] ?? null) : null;
      out.push({ v, system: group, answer, group: answer ? ANSWER_GROUP[answer] : group, words: wordsOf(v, marks) });
    }
  return out;
}

export function groupsOf(list: Finding[]): { group: Group; items: Finding[] }[] {
  return GROUP_ORDER.map((group) => ({ group, items: list.filter((f) => f.group === group) })).filter((g) => g.items.length);
}

export interface AnswerLines {
  big: string;
  where: string | null;
  tone: 'new' | 'eye' | 'plain';
}

// The answer, very large: code's count of new marks and where, else "No new marks" and how many need your eye.
export function answerLines(verdicts: Verdict[], compared: number): AnswerLines {
  const fresh = verdicts.filter((v) => v.label === 'new_or_unseen');
  if (fresh.length) {
    const panels = [...new Set(fresh.map((v) => panelName(v.panel_id)))];
    return { big: `${fresh.length} new ${fresh.length === 1 ? 'mark' : 'marks'}`, where: cap(joinWords(panels)), tone: 'new' };
  }
  if (!compared) return { big: 'Nothing to compare', where: 'No panel was filmed clearly in both walks.', tone: 'plain' };
  const eye = verdicts.filter((v) => v.label === 'needs_your_eye').length;
  if (eye) return { big: 'No new marks', where: `${eye} ${eye === 1 ? 'needs' : 'need'} your eye`, tone: 'eye' };
  return { big: 'No new marks', where: null, tone: 'plain' };
}

export interface Side {
  role: string;
  kf: Keyframe | undefined;
  own: Mark[]; // the verdict's marks on this frame
  spot: RegistrationMark | null; // the mark of the other walk carried into this frame, when the walk has none
}

// One walk's frame for a finding: the frame its mark was found on, else the frame the other walk's mark was carried
// into, else that walk's best frame of the panel.
export function sideOf(v: Verdict, role: string, marks: Map<string, Mark>, keyframes: Keyframe[], match: PanelMatch | undefined): Side {
  const own = v.mark_ids.map((id) => marks.get(id)).filter((m): m is Mark => m !== undefined && m.role === role);
  const kfId = own[0]?.kf_id;
  let kf = kfId ? keyframes.find((k) => k.kf_id === kfId) : undefined;
  let spot: RegistrationMark | null = null;
  if (!own.length) {
    const reg = (match?.registration_marks ?? []).find((r) => v.mark_ids.includes(r.mark_id) && r.mapped_box && keyframes.some((k) => k.kf_id === r.other_kf && k.role === role));
    if (reg) {
      spot = reg;
      kf = keyframes.find((k) => k.kf_id === reg.other_kf);
    }
  }
  kf ??= bestKeyframe(keyframes, role, v.panel_id);
  return { role, kf, own: own.filter((m) => m.kf_id === kf?.kf_id), spot };
}

// Where the other walk's mark lands in this frame: the spot the presence test found when it found the patch, else
// where the mark maps to.
export function spotBox(reg: RegistrationMark | null): Box4 | null {
  if (!reg) return null;
  return reg.result === 'present' ? (reg.found_box ?? reg.mapped_box) : reg.mapped_box;
}

// The thumbnail's walk: return for a new mark and for one seen in both, pickup for a mark seen at pickup only.
export function thumbRole(v: Verdict, marks: Map<string, Mark>): string {
  const w = walksOf(v, marks);
  return w.has('return') ? 'return' : 'pickup';
}
