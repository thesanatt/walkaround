import type { LookRow, Mark, PanelMatch, Verdict, VerdictLabel } from '../contracts/types.ts';
import { isCounterLook, lookFor } from './looks.ts';
import { panelName } from './panels.ts';

// One short plain line per needs-your-eye rule: why code left the call to the renter.
export const WHY: Record<string, string> = {
  K1: 'Close to a mark in the other walk, but not close enough to call it the same one.',
  K2: 'Seen in only one of three frames.',
  K3: 'Too small to judge from video.',
  K4: 'It may be dirt or a reflection.',
  K5: 'The two marks look like different kinds of damage.',
  K6: 'It may move with the camera, like a reflection.',
  K7: 'A second look was not sure it is the same mark.',
  K8: 'Found only on a second, rougher pass over the panel.',
  K9: 'The panel looks wet or sprayed.',
  K10: 'A single mark on a wheel, away from where wheel marks usually are.',
  K11: 'The claim box covers a whole region, not one mark.',
  K12: 'The model was not sure this is damage.',
  K13: "The other walk's look at this panel was cut short.",
  K14: 'Boxed only because you named it.',
  K15: 'Dirt, water or glare on this panel in one of the walks.',
  K16: 'The other walk may show it too.',
  K17: 'A second look did not see damage in the box.',
  K18: 'The other walk shows the same patch here, but nothing was boxed there.',
  K19: 'The other walk boxed a bigger area over this spot.',
  K20: 'It does not stand out from the paint around it.',
  K21: 'The other walk barely filmed this part of the panel.',
  L7: 'The panel outline changed. It may be bent.',
};

// L8: code called it already there from the pixels, so there is no distance to give.
const FOUND_IN_BOTH = 'Boxed in one walk only, but lining up the two frames shows the same patch in the other walk.';

export function whyLine(v: Verdict): string {
  if (v.label === 'already_there' && v.rule_id === 'L8' && v.distance_cm_est === null) return FOUND_IN_BOTH;
  if (v.label === 'already_there')
    return v.distance_cm_est !== null
      ? `In the same place at pickup and at return, about ${v.distance_cm_est.toFixed(0)} cm apart.`
      : 'In the same place at pickup and at return.';
  if (v.label === 'new_or_unseen') return 'Not found on this panel at pickup.';
  if (v.label === 'not_seen_at_return') return 'Seen at pickup, not found at return.';
  if (v.label === 'not_comparable') return 'This panel could not be compared.';
  const id = [v.rule_id, ...v.trace.rule_ids].find((r) => WHY[r]);
  return id ? WHY[id]! : 'The check could not decide, so it leaves this one to you.';
}

// The renter-first answer (critic, Nov 1): panels, not marks. Every count comes from the labels code wrote; this only
// words them.

// "a", "a and b", "a, b and c"
export function joinWords(words: string[]): string {
  if (words.length < 2) return words[0] ?? '';
  return `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`;
}

function unique<T>(xs: T[]): T[] {
  return [...new Set(xs)];
}

// The panels both walks show clearly enough to compare: the matcher's comparable panels, else every panel a
// verdict other than not comparable is on. Glass and the wheels (R30) get a check-by-eye line, not a comparison.
const notJudged = (m: PanelMatch) => (m.by_eye ?? []).some((b) => b.kind === 'glass' || b.kind === 'wheels');

export function comparedPanels(matches: PanelMatch[], verdicts: Verdict[]): string[] {
  const fromMatches = unique(matches.filter((m) => m.state === 'comparable' && !notJudged(m)).map((m) => m.panel_id));
  if (fromMatches.length) return fromMatches;
  return unique(verdicts.filter((v) => v.label !== 'not_comparable').map((v) => v.panel_id));
}

// "1 new mark: left front door." or "Nothing new on the 3 panels both walks show clearly." A spot code left to the
// renter keeps "nothing new" from being said outright.
export function headline(verdicts: Verdict[], compared: number): string {
  const fresh = verdicts.filter((v) => v.label === 'new_or_unseen');
  if (fresh.length) {
    const panels = unique(fresh.map((v) => panelName(v.panel_id)));
    return `${fresh.length} new ${fresh.length === 1 ? 'mark' : 'marks'}: ${joinWords(panels)}.`;
  }
  if (!compared) return 'No panel was filmed clearly enough in both walks to compare.';
  const where = compared === 1 ? 'the one panel both walks show clearly' : `the ${compared} panels both walks show clearly`;
  const eye = verdicts.some((v) => v.label === 'needs_your_eye');
  return eye ? `No new damage confirmed on ${where}.` : `Nothing new on ${where}.`;
}

// "1 mark was there at pickup. 11 spots need your eye. 18 panels weren't filmed clearly enough in both walks."
export function subline(verdicts: Verdict[], compared: number): string {
  const n = (label: VerdictLabel) => verdicts.filter((v) => v.label === label).length;
  const out: string[] = [];
  const fresh = n('new_or_unseen');
  if (fresh && compared) out.push(`Compared on ${compared} ${compared === 1 ? 'panel' : 'panels'}.`);
  const already = n('already_there');
  if (already) out.push(`${already} ${already === 1 ? 'mark was' : 'marks were'} there at pickup.`);
  const eye = n('needs_your_eye');
  if (eye) out.push(`${eye} ${eye === 1 ? 'spot needs' : 'spots need'} your eye.`);
  const gone = n('not_seen_at_return');
  if (gone) out.push(`${gone} ${gone === 1 ? 'mark' : 'marks'} from pickup not seen at return.`);
  const nc = verdicts.filter((v) => v.label === 'not_comparable');
  if (nc.length) {
    const filming = nc.filter((v) => v.trace.rule_ids.includes('G1')).length * 2 >= nc.length;
    const what = filming ? "weren't filmed clearly enough in both walks to compare" : "couldn't be lined up between the two walks";
    out.push(`${nc.length} ${nc.length === 1 ? 'panel' : 'panels'} ${what}.`);
  }
  return out.join(' ');
}

// The panels not compared, each with its reason in plain words.
const GATE_PLAIN: [string, string][] = [
  ['G1', 'not filmed long enough in both walks'],
  ['G7', 'not fully tracked'],
  ['G3', 'blurry in one walk'],
  ['G12', 'mostly blurred for privacy in one walk'],
  ['G6', 'unclear which side of the car it is'],
  ['G13', 'the left one in one walk and the right one in the other'],
  ['G9', 'the window was cut out in one walk only'],
  ['G11', 'filmed at very different sizes in the two walks'],
  ['G4', 'the two views did not line up'],
  ['G8', 'the two views did not line up'],
  ['G5', 'the door handle check failed'],
  ['G10', 'the mark list was cut short'],
];

export function notComparedRows(verdicts: Verdict[]): { panelId: string; reason: string; filming: boolean }[] {
  return verdicts
    .filter((v) => v.label === 'not_comparable')
    .map((v) => {
      const hit = GATE_PLAIN.find(([id]) => v.trace.rule_ids.includes(id));
      return { panelId: v.panel_id, reason: hit ? hit[1] : 'could not be compared', filming: v.trace.rule_ids.includes('G1') };
    });
}

// The check-by-eye lines code wrote in place of verdicts: one row for the glass, one for the wheels (car body only,
// R30), one per compared panel whose dirt was folded into a debris line.
export interface ByEyeRow {
  key: string;
  title: string;
  text: string;
}

export const BY_EYE_WORDS: Record<string, string> = {
  glass: 'reflections make marks on glass unreliable, so marks there are not compared',
  wheels: 'rims and tires are not checked mark by mark; the check covers the car body',
  debris: 'dirt on this panel',
};

export function byEyeRows(matches: PanelMatch[]): ByEyeRow[] {
  const of = (kind: string) => matches.filter((m) => (m.by_eye ?? []).some((b) => b.kind === kind));
  const rows: ByEyeRow[] = [];
  const glass = of('glass');
  if (glass.length) rows.push({ key: 'glass', title: joinWords(glass.map((m) => panelName(m.panel_id))), text: BY_EYE_WORDS['glass']! });
  if (of('wheels').length) rows.push({ key: 'wheels', title: 'wheels', text: BY_EYE_WORDS['wheels']! });
  for (const m of of('debris')) {
    const n = (m.by_eye ?? []).filter((b) => b.kind === 'debris').reduce((sum, b) => sum + b.mark_ids.length, 0);
    rows.push({ key: `debris-${m.panel_id}`, title: panelName(m.panel_id), text: `${BY_EYE_WORDS['debris']!}, ${n} ${n === 1 ? 'spot' : 'spots'} not listed one by one` });
  }
  return rows;
}

// One quiet line per card: which Meta model did what, and that code made the call. Built from the marks and the
// second look the run stored; a look the run never asked is not mentioned.
export function creditLine(v: Verdict, marks: Map<string, Mark>, looks: LookRow[]): string {
  const own = v.mark_ids.map((id) => marks.get(id)).filter((m): m is Mark => m !== undefined);
  const parts = ['SAM 3.1 tracked this panel'];
  const both = new Set(own.map((m) => m.role)).size > 1;
  if (own.some((m) => m.source === 'narration_seed')) parts.push('Muse Voice heard it named and Muse Spark boxed it');
  else if (own.length && own.every((m) => m.source === 'spark_count')) parts.push('Muse Spark found it while counting marks');
  else parts.push(both ? 'Muse Spark found it in both walks' : 'Muse Spark found the mark');
  if (own.some((m) => m.box_refine?.used === 'sam')) parts.push('SAM 3.1 tightened the box');
  const look = lookFor(v, looks);
  if (look?.engine) {
    const who = look.engine === 'baseten' ? 'Muse Spark Open' : 'Muse Spark';
    if (isCounterLook(look)) {
      const other = look.mark_id === look.pickup_mark ? 'return' : 'pickup';
      if (look.kind === 'counter2' && look.mark_seen === 'no') parts.push(`${who} saw no damage in the box`);
      else if (look.verdict === 'same') parts.push(`${who} said the ${other} walk may show it too`);
      else if (look.verdict === 'different') parts.push(`${who} saw no such mark at that spot in the ${other} walk`);
      else parts.push(`${who} could not tell what the ${other} walk shows there`);
    } else if (look.verdict === 'same') parts.push(`${who} agreed it is the same spot`);
    else if (look.verdict === 'different') parts.push(`${who} said it is not the same spot`);
    else parts.push(`${who} could not tell if it is the same spot`);
  }
  parts.push(v.label === 'needs_your_eye' ? 'code left the call to you' : 'code made the call');
  return `${parts.join('; ')}.`;
}
