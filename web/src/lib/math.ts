import type { LookRow, Mark, PanelMatch, Verdict } from '../contracts/types.ts';
import { lookFor } from './looks.ts';
import { SRC, type SourceLine } from './mathlines.ts';
import { panelName } from './panels.ts';
import { RULES } from './rules.ts';

// "Show the math" on a finding: the run's own numbers beside the lines of code that used them. Every value comes from
// the verdict's trace (the thresholds code stored with it), the match and the second look; nothing is recomputed.

export interface MathRow {
  line: SourceLine;
  value: string | null;
}

export interface MathStep {
  title: string;
  sub: string | null;
  rows: MathRow[];
}

export interface Decision {
  left: string;
  op: string;
  right: string;
  leftName: string;
  rightName: string;
}

export interface MathView {
  decision: Decision | null;
  steps: MathStep[];
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

// three decimals, as the sheet prints them; a short constant such as 0.4 stays as it is
export function fmt(v: number): string {
  const s = String(v);
  return s.includes('.') && s.split('.')[1]!.length > 3 ? v.toFixed(3) : s;
}

const row = (line: SourceLine, value: string | null = null): MathRow => ({ line, value });
const yesNo = (b: boolean) => (b ? 'yes' : 'no');

export function mathFor(v: Verdict, marks: Map<string, Mark>, match: PanelMatch | undefined, looks: LookRow[]): MathView {
  const t = v.trace.thresholds as Record<string, unknown>;
  const ids = v.trace.rule_ids;
  const d = num(t['d']);
  const pre = num(t['tau_pre_r']);
  const eye = num(t['tau_eye_r']);
  const own = v.mark_ids.map((id) => marks.get(id)).filter((m): m is Mark => m !== undefined);
  const roles = new Set(own.map((m) => m.role));
  const oneWalk = roles.size === 1;
  const other = roles.has('return') ? 'pickup' : 'return';
  const panel = panelName(v.panel_id);
  const steps: MathStep[] = [];
  let decision: Decision | null = null;

  // 1. the cost, or for L8 the presence test that stood in for it
  if (v.rule_id === 'L8' || ids.includes('L8')) {
    const ncc = num(t['ncc']);
    steps.push({
      title: `The same patch found in the ${other} walk`,
      sub: 'The two crops of the panel are lined up, then the spot is tested in the other walk.',
      rows: [row(SRC.present_if, ncc !== null ? `ncc ${fmt(ncc)}` : null), row(SRC.present, 'present')],
    });
  } else if (d !== null && pre !== null && eye !== null) {
    steps.push({
      title: oneWalk ? `The nearest ${other} mark on the ${panel}` : 'The cost between the two marks',
      sub: 'Distance on the panel, plus a cost when the type or size differs.',
      rows: [
        row(SRC.d_pos, num(t['d_pos']) !== null ? fmt(num(t['d_pos'])!) : null),
        row(SRC.type_cost, num(t['type_cost']) !== null ? fmt(num(t['type_cost'])!) : null),
        row(SRC.size_cost, num(t['size_cost']) !== null ? fmt(num(t['size_cost'])!) : null),
        row(SRC.r_mean, num(t['r_mean']) !== null ? fmt(num(t['r_mean'])!) : null),
        row(SRC.tau_pre_r, fmt(pre)),
        row(SRC.tau_eye_r, fmt(eye)),
        row(SRC.d, fmt(d)),
      ],
    });
    const what = oneWalk ? `cost to the nearest ${other} mark` : 'cost between the two marks';
    if (d < pre) decision = { left: fmt(d), op: '<', right: fmt(pre), leftName: `d, ${what}`, rightName: 'tau_pre_r, top of the already-there band' };
    else if (d < eye) decision = { left: fmt(d), op: '<', right: fmt(eye), leftName: `d, ${what}`, rightName: 'tau_eye_r, top of the needs-your-eye band' };
    else decision = { left: fmt(d), op: d > eye ? '>' : '>=', right: fmt(eye), leftName: `d, ${what}`, rightName: 'tau_eye_r, top of the needs-your-eye band' };
  }

  // 2. the second look, when one was asked
  const look = lookFor(v, looks);
  if (look?.engine) {
    const who = look.engine === 'baseten' ? 'Muse Spark Open on Baseten' : 'Muse Spark';
    const n = look.answers ?? 1;
    const votes = look.votes;
    const asked = n === 1 ? 'Asked once.' : `Asked ${n} times.`;
    if (look.kind === 'counter2') {
      steps.push({
        title: `Second look: ${who}`,
        sub: `${asked} The label stands only on damage yes and same mark no.`,
        rows: [
          row(SRC.look_yes, votes ? `${votes.yes} of ${n}` : null),
          row(SRC.look_same, votes ? `${votes.same} of ${n}` : null),
          row(SRC.look_mark, look.mark_seen ? `"${look.mark_seen}"` : null),
          row(SRC.look_spot, look.verdict),
        ],
      });
    } else if (!look.kind) {
      steps.push({
        title: `Second look: ${who}`,
        sub: `${asked} Same keeps the pair; anything else lowers it to needs your eye.`,
        rows: [row(SRC.pair_same, votes ? `${votes.same} of ${n}` : null), row(SRC.pair_diff, votes ? `${votes.different} of ${n}` : null), row(SRC.pair_out, look.verdict)],
      });
    }
  }

  // 3. the rule: the band decision, then the caps that may lower it
  const notes = new Set(v.trace.note_ids ?? []);
  const caps = ids.filter((id) => id.startsWith('K') && id !== 'K1' && !notes.has(id));
  const rules: MathRow[] = [];
  if (v.rule_id === 'L8' || ids.includes('L8')) rules.push(row(SRC.geo_verdict), row(SRC.geo_l8, 'L8'));
  else if (d === null) rules.push(row(SRC.if_none, 'yes'), row(SRC.unmatched_none, roles.has('return') ? '"L2"' : '"L1"'));
  else if (pre !== null && eye !== null) {
    rules.push(row(SRC.if_pre, `${fmt(d)} < ${fmt(pre)}: ${yesNo(d < pre)}`));
    if (d < pre) rules.push(row(SRC.already, '"L3"'));
    else {
      rules.push(row(SRC.if_eye, `${fmt(d)} < ${fmt(eye)}: ${yesNo(d < eye)}`));
      if (d < eye) rules.push(row(SRC.band, '"L4", "K1"'));
      else rules.push(row(SRC.else_), row(SRC.unmatched, '"L5"'));
    }
  }
  if (rules.length && v.rule_id !== 'L7') {
    rules.push(row(SRC.caps, caps.length ? caps.join(', ') : 'none'));
    if (caps.length && v.label === 'needs_your_eye') rules.push(row(SRC.if_caps, 'yes'), row(SRC.lowered, 'needs your eye'));
  }
  const named = caps.map((id) => RULES[id]?.label).filter((s): s is string => Boolean(s));
  const final = RULES[v.rule_id];
  steps.push({
    title: 'The rule that fired',
    sub: v.rule_id === 'L7' && final ? `L7: ${final.label}.` : named.length ? `Lowered by ${named.join(', ')}.` : final ? `${v.rule_id}: ${final.label}.` : null,
    rows: v.rule_id === 'L7' ? [] : rules,
  });

  // the registration result for a mark seen in one walk only, folded into the cost step's sub when there is one
  const reg = oneWalk ? (match?.registration_marks ?? []).find((r) => v.mark_ids.includes(r.mark_id)) : undefined;
  const seen = reg?.result === 'absent' ? 'no such mark' : reg?.result === 'present' ? 'the same patch' : 'nothing clear either way';
  if (reg && steps[0] && steps[0].title.startsWith('The nearest')) steps[0].sub = `${steps[0].sub} Lined up with the ${other} crop, that spot shows ${seen}.`;

  return { decision, steps };
}
