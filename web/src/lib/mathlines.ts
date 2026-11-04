// The source lines the math panel quotes, as they stand in this repo (test/mathlines.test.ts reads each file and
// fails when a line moved or changed, so the panel never shows code that is not there).
export interface SourceLine {
  file: string;
  n: number;
  raw: string;
}

const M = 'walkaround/match.py';
const R = 'walkaround/register.py';
const line = (file: string, n: number, raw: string): SourceLine => ({ file, n, raw });

export const SRC = {
  d_pos: line(M, 1994, '    d_pos = _d_pos(wa, wb)'),
  type_cost: line(M, 1995, '    type_cost = _type_cost(a.type, b.type, th)'),
  size_cost: line(M, 1996, '    size_cost = _size_cost(a, b, th)'),
  r_mean: line(M, 1997, '    r_mean = 0.5 * (wa.r + wb.r)'),
  tau_pre_r: line(M, 1998, '    tau_pre_r = tau_r(calibration.tau_pre, r_mean, th)'),
  tau_eye_r: line(M, 1999, '    tau_eye_r = tau_r(calibration.tau_eye, r_mean, th)'),
  d: line(M, 2007, '        d=d_pos + type_cost + size_cost,'),
  pair_same: line(M, 1334, '    same = sum(a == LookVerdict.SAME for a in usable)'),
  pair_diff: line(M, 1335, '    diff = sum(a == LookVerdict.DIFFERENT for a in usable)'),
  pair_out: line(M, 1337, '    out = _side(same, diff + unsure, LookVerdict.DIFFERENT, LookVerdict.UNSURE, diff, unsure, LookVerdict.SAME)'),
  look_yes: line(M, 1350, '    yes = sum(a.mark == "yes" for a in usable)'),
  look_same: line(M, 1354, '    same = sum(a.spot == LookVerdict.SAME for a in usable)'),
  look_mark: line(M, 1356, '    mark = _side(yes, no + m_unsure, "no", "unsure", no, m_unsure, "yes")'),
  look_spot: line(M, 1357, '    spot = _side(diff, same + s_unsure, LookVerdict.SAME, LookVerdict.UNSURE, same, s_unsure, LookVerdict.DIFFERENT)'),
  if_none: line(M, 2083, '    if d is None:'),
  unmatched_none: line(M, 2084, '        label, ids = unmatched, ["L2" if flags.return_mark else "L1"]'),
  if_pre: line(M, 2085, '    elif d < tau_pre_r:'),
  already: line(M, 2086, '        label, ids = VerdictLabel.ALREADY_THERE, ["L3"]'),
  if_eye: line(M, 2087, '    elif d < tau_eye_r:'),
  band: line(M, 2088, '        label, ids = VerdictLabel.NEEDS_YOUR_EYE, ["L4", "K1"]'),
  else_: line(M, 2089, '    else:'),
  unmatched: line(M, 2090, '        label, ids = unmatched, ["L5"]'),
  caps: line(M, 2091, '    caps = flags.set_caps()'),
  if_caps: line(M, 2092, '    if caps and label in LOWERABLE:'),
  lowered: line(M, 2093, '        label = VerdictLabel.NEEDS_YOUR_EYE'),
  geo_verdict: line(M, 2581, '        label, ids = verdict(0.0, 1.0, 2.0, flags)'),
  geo_l8: line(M, 2582, '        ids = ["L8" if i == "L3" else i for i in ids]'),
  present_if: line(R, 684, '    if sb_found >= float(cfg["saliency_floor"]) and (strong or kept):'),
  present: line(R, 685, '        out.result = PRESENT'),
  absent_if: line(R, 688, '    if best < float(cfg["ncc_absent"]) and gone < float(cfg["gone_frac"]) and sa >= float(cfg["saliency_absent_min"]):'),
  absent: line(R, 689, '        out.result = ABSENT'),
} as const;
