import type { Mark, Verdict, VerdictLabel } from '../contracts/types.ts';

// The answer at the top of the result page, counted from the verdicts code wrote. One verdict is one mark (a pair
// joined across the walks is one mark seen twice).
export interface Tally {
  already: number;
  fresh: number;
  eye: number;
  gone: number;
  notComparable: number;
}

export function tally(verdicts: Verdict[]): Tally {
  const n = (label: VerdictLabel) => verdicts.filter((v) => v.label === label).length;
  return { already: n('already_there'), fresh: n('new_or_unseen'), eye: n('needs_your_eye'), gone: n('not_seen_at_return'), notComparable: n('not_comparable') };
}

// "1 mark was already there at pickup. 0 new. 11 need your eye." The not-seen-at-return count joins only when it is
// not zero, since it never costs the renter anything.
export function answerSentences(t: Tally): string[] {
  const out = [`${t.already} ${t.already === 1 ? 'mark was' : 'marks were'} already there at pickup.`, `${t.fresh} new.`, `${t.eye} ${t.eye === 1 ? 'needs' : 'need'} your eye.`];
  if (t.gone) out.push(`${t.gone} not seen at return.`);
  return out;
}

const GATE_WORDS: Record<string, string> = {
  G1: 'were not filmed long enough in both walks',
  G3: 'had a blurry frame',
  G4: 'did not line up between the two walks',
  G5: 'failed the door handle check',
  G6: 'were seen from an unclear side',
  G7: 'were not fully tracked',
  G8: 'did not line up between the two walks',
  G9: 'had the window cut out in one walk only',
  G10: 'had their mark list cut short',
};

// "18 panels could not be compared. Most were not filmed long enough in both walks."
export function notComparableLine(verdicts: Verdict[]): string | null {
  const nc = verdicts.filter((v) => v.label === 'not_comparable');
  if (!nc.length) return null;
  const counts = new Map<string, number>();
  for (const v of nc) for (const id of new Set(v.trace.rule_ids.filter((r) => GATE_WORDS[r]))) counts.set(id, (counts.get(id) ?? 0) + 1);
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  const head = nc.length === 1 ? '1 panel could not be compared.' : `${nc.length} panels could not be compared.`;
  if (!top) return head;
  const words = GATE_WORDS[top[0]]!;
  if (nc.length === 1) return `${head} It ${words.replace(/^were /, 'was ').replace(/^had their /, 'had its ')}.`;
  return `${head} ${top[1] === nc.length ? 'All' : 'Most'} ${words}.`;
}

// One short plain line per needs-your-eye rule: why code left the call to the renter.
export const WHY: Record<string, string> = {
  K1: 'Close to a mark in the other walk, but not close enough to call it the same one.',
  K2: 'Seen in only one of three frames.',
  K3: 'Too small to judge from video.',
  K4: 'It may be dirt or a reflection.',
  K5: 'The two marks look like different kinds of damage.',
  K6: 'It may move with the camera, like a reflection.',
  K7: 'A second look was not sure it is the same mark.',
  K8: 'Found only when the model counted marks.',
  K9: 'The panel looks wet or sprayed.',
  K10: 'A single mark on a wheel, away from where wheel marks usually are.',
  K11: 'The claim box covers a whole region, not one mark.',
  K12: 'The model was not confident about this mark.',
  K13: "The other walk's look at this panel was cut short.",
  K14: 'Boxed only because you named it.',
  K15: 'Dirt, water or glare on this panel in one of the walks.',
  K16: 'The other walk may show it too.',
  K17: 'A second look did not see damage in the box.',
  L7: 'The panel outline changed. It may be bent.',
};

export function whyLine(v: Verdict): string {
  if (v.label === 'already_there')
    return v.distance_cm_est !== null
      ? `In the same place at pickup and at return, about ${v.distance_cm_est.toFixed(0)} cm apart.`
      : 'In the same place at pickup and at return.';
  if (v.label === 'new_or_unseen') return 'Not found on this panel at pickup.';
  if (v.label === 'not_seen_at_return') return 'Seen at pickup, not found at return.';
  if (v.label === 'not_comparable') return 'This panel could not be compared.';
  const id = [v.rule_id, ...v.trace.rule_ids].find((r) => WHY[r]);
  return id ? WHY[id]! : 'Code could not decide, so it leaves this one to you.';
}

// The order of the cards, which is the order of the answer sentence.
export const CARD_ORDER: VerdictLabel[] = ['already_there', 'new_or_unseen', 'needs_your_eye', 'not_seen_at_return'];

export const GROUP_TITLE: Record<string, string> = {
  already_there: 'Already there at pickup',
  new_or_unseen: 'New since pickup, or hidden at pickup',
  needs_your_eye: 'Needs your eye',
  not_seen_at_return: 'Not seen at return',
};

// "small dark speck", from the first mark that carries Muse Spark's words.
export function markWords(v: Verdict, marks: Map<string, Mark>): string | null {
  for (const id of v.mark_ids) {
    const d = marks.get(id)?.description;
    if (d) return d;
  }
  return null;
}
