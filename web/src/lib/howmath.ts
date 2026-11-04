// The match.py lines the How it works page shows for the example's new mark, with the run's own values beside them.
// Values: hero_focus5e derived/verdicts.json, verdict right_rear_door_v2, trace.thresholds (d_pos 0.285402,
// type_cost 0.4, size_cost 0.230074, r_mean 0.35053, tau_pre_r 0.424574, tau_eye_r 0.849149, d 0.915476) and
// trace.rule_ids ["L5"], label new_or_unseen. test/howmath.test.ts checks every line against walkaround/match.py.
export interface HowLine {
  n: number;
  code: string; // the line as in the file, less `indent` spaces
  note?: string;
}

export interface HowExcerpt {
  indent: number;
  lines: HowLine[];
}

export const COST_LINES: HowExcerpt = {
  indent: 4,
  lines: [
    { n: 1994, code: 'd_pos = _d_pos(wa, wb)', note: '0.285' },
    { n: 1995, code: 'type_cost = _type_cost(a.type, b.type, th)', note: '0.4' },
    { n: 1996, code: 'size_cost = _size_cost(a, b, th)', note: '0.230' },
    { n: 1997, code: 'r_mean = 0.5 * (wa.r + wb.r)', note: '0.351' },
    { n: 1998, code: 'tau_pre_r = tau_r(calibration.tau_pre, r_mean, th)', note: '0.425' },
    { n: 1999, code: 'tau_eye_r = tau_r(calibration.tau_eye, r_mean, th)', note: '0.849' },
    { n: 2007, code: '    d=d_pos + type_cost + size_cost,', note: '0.915' },
  ],
};

export const RULE_LINES: HowExcerpt = {
  indent: 4,
  lines: [
    { n: 2081, code: 'unmatched = VerdictLabel.NEW_OR_UNSEEN if flags.return_mark else VerdictLabel.NOT_SEEN_AT_RETURN', note: 'New' },
    { n: 2082, code: 'ids: list[str]' },
    { n: 2083, code: 'if d is None:', note: 'd = 0.915' },
    { n: 2084, code: '    label, ids = unmatched, ["L2" if flags.return_mark else "L1"]' },
    { n: 2085, code: 'elif d < tau_pre_r:', note: '0.915 < 0.425, no' },
    { n: 2086, code: '    label, ids = VerdictLabel.ALREADY_THERE, ["L3"]' },
    { n: 2087, code: 'elif d < tau_eye_r:', note: '0.915 < 0.849, no' },
    { n: 2088, code: '    label, ids = VerdictLabel.NEEDS_YOUR_EYE, ["L4", "K1"]' },
    { n: 2089, code: 'else:' },
    { n: 2090, code: '    label, ids = unmatched, ["L5"]', note: 'L5, New' },
  ],
};
