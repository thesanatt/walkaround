import type { LookRow, LookVerdict, OpenWeightsRow, Verdict } from '../contracts/types.ts';

// The second look reads two crops with the candidate marks outlined and answers same, different or unsure. Code
// turns different and unsure into K7, which can only lower a label. These readers take the fields the page prints
// from looks.json and open_weights.ndjson and skip a row they cannot place.

const VERDICTS: ReadonlySet<string> = new Set(['same', 'different', 'unsure']);

function str(v: unknown): string | null {
  return typeof v === 'string' && v ? v : null;
}

function obj(v: unknown): Record<string, unknown> | null {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

function bool(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function verdictOf(v: unknown): LookVerdict | null {
  const s = str(v)?.toLowerCase() ?? null;
  return s && VERDICTS.has(s) ? (s as LookVerdict) : null;
}

// The pair a row is about: pickup_mark and return_mark, else a two-id pair or mark_ids list.
function pairOf(row: Record<string, unknown>): [string, string] | null {
  const a = str(row['pickup_mark']);
  const b = str(row['return_mark']);
  if (a && b) return [a, b];
  for (const key of ['pair', 'mark_ids']) {
    const ids = row[key];
    if (Array.isArray(ids) && ids.length === 2 && typeof ids[0] === 'string' && typeof ids[1] === 'string') return [ids[0], ids[1]];
  }
  return null;
}

// The answer may sit at the top level or inside a parsed model reply.
function answerOf(row: Record<string, unknown>): Record<string, unknown> {
  return obj(row['parsed']) ?? obj(row['answer']) ?? obj(row['look']) ?? row;
}

// The request id of the answer that stands: the row's own, else the last of its records.
function requestOf(row: Record<string, unknown>): string | null {
  const own = str(row['request_id']);
  if (own) return own;
  const records = Array.isArray(row['records']) ? row['records'] : [];
  for (let i = records.length - 1; i >= 0; i -= 1) {
    const id = str(obj(records[i])?.['request_id']);
    if (id) return id;
  }
  return null;
}

function inputsOf(row: Record<string, unknown>): string[] {
  const refs = Array.isArray(row['input_refs']) ? row['input_refs'] : [];
  return refs.map((r) => str(obj(r)?.['path'])).filter((p): p is string => p !== null);
}

// One row per look. Under matching.look_majority a look is asked up to 3 times and every answer is its own looks.json
// row, so the answers are grouped by look and the label is the majority, as match.py decides it.
export function readLooks(value: unknown): LookRow[] {
  return majorityLooks(readLookAnswers(value));
}

// Every usable answer as stored, repeats included.
export function readLookAnswers(value: unknown): LookRow[] {
  const rows = Array.isArray(value) ? value : (obj(value)?.['looks'] as unknown[] | undefined) ?? [];
  const out: LookRow[] = [];
  for (const raw of rows) {
    const row = obj(raw);
    const pair = row ? pairOf(row) : null;
    if (!row || !pair) continue;
    const answer = answerOf(row);
    const verdict = verdictOf(answer['verdict']) ?? verdictOf(row['verdict']);
    if (!verdict) continue; // verdict null: the look did not run, so it put no K7 on any label
    out.push({
      pickup_mark: pair[0],
      return_mark: pair[1],
      panel_id: (str(row['panel_id']) as LookRow['panel_id']) ?? null,
      verdict,
      reason: str(answer['reason']) ?? str(row['reason']) ?? '',
      engine: str(row['engine']),
      model: str(row['model']),
      key_slot: row['key_slot'] === 'hackathon' || row['key_slot'] === 'account' ? row['key_slot'] : null,
      fallback_from: row['fallback_from'] === 'baseten' ? 'baseten' : null,
      valid: bool(row['valid']),
      fallback: str(row['fallback']),
      request_id: requestOf(row),
      inputs: inputsOf(row),
      kind: row['kind'] === 'counter' || row['kind'] === 'counter2' ? row['kind'] : null,
      mark_id: str(row['mark_id']),
      mark_seen: row['mark_seen'] === 'yes' || row['mark_seen'] === 'no' || row['mark_seen'] === 'unsure' ? row['mark_seen'] : null,
    });
  }
  return out;
}

// match._side: the keeping answer wins only when it outnumbers the rest; else the lowering answer given most, and a
// tie (between keep and lower, or between the two lowering answers) is unsure.
function side<T extends string>(keep: number, lowered: number, other: number, kept: T, low: T, unsure: T): T {
  if (keep > lowered + other) return kept;
  if (lowered + other > keep && lowered > other) return low;
  return unsure;
}

function lookKey(l: LookRow): string {
  return [l.kind ?? 'pair', l.pickup_mark, l.return_mark, l.mark_id ?? ''].join('|');
}

// The answers of each look folded into one row, in the order the looks were first asked: match.majority_pair_look
// for a pair look (same keeps), match.majority_counter for a counter-look (different keeps; part 1 yes keeps). The
// row is the first answer that agrees with the majority, so its reason and request id are one that stood.
export function majorityLooks(answers: readonly LookRow[]): LookRow[] {
  const groups = new Map<string, LookRow[]>();
  for (const a of answers) {
    const key = lookKey(a);
    const list = groups.get(key) ?? [];
    list.push(a);
    groups.set(key, list);
  }
  const out: LookRow[] = [];
  for (const rows of groups.values()) {
    const count = (v: string) => rows.filter((r) => r.verdict === v).length;
    const same = count('same');
    const different = count('different');
    const unsure = rows.length - same - different;
    const verdict: LookVerdict = isCounterLook(rows[0]) ? side(different, same, unsure, 'different', 'same', 'unsure') : side(same, different, unsure, 'same', 'different', 'unsure');
    const standing = rows.find((r) => r.verdict === verdict) ?? rows[0]!;
    const row: LookRow = { ...standing, verdict, answers: rows.length };
    if (rows[0]!.kind === 'counter2') {
      const seen = rows.map((r) => r.mark_seen);
      const yes = seen.filter((m) => m === 'yes').length;
      const no = seen.filter((m) => m === 'no').length;
      row.mark_seen = side(yes, no, rows.length - yes - no, 'yes', 'no', 'unsure');
    }
    out.push(row);
  }
  return out;
}

// One JSON object per line; a line that does not parse is skipped, never fatal.
export function readOpenWeights(text: string | null | undefined): OpenWeightsRow[] {
  const out: OpenWeightsRow[] = [];
  for (const line of (text ?? '').split('\n')) {
    if (!line.trim()) continue;
    let row: Record<string, unknown> | null;
    try {
      row = obj(JSON.parse(line));
    } catch {
      row = null;
    }
    if (!row) continue;
    const answer = answerOf(row);
    out.push({
      request_id: str(row['request_id']),
      model: str(row['model']),
      key_slot: str(row['key_slot']),
      purpose: str(row['purpose']),
      verdict: verdictOf(answer['verdict']) ?? verdictOf(row['verdict']),
      status: str(row['status']),
      valid: bool(row['valid']),
      error: str(row['error']),
      ts: str(row['ts']),
    });
  }
  return out;
}

// Muse Spark Open answered on Baseten. A fallback row was answered by hosted Muse Spark 1.3 instead.
export function isOpenWeights(look: LookRow): boolean {
  return look.engine === 'baseten';
}

// Hosted Muse Spark 1.3 answered because Spark Open on Baseten failed or was skipped (the binding fallback_from).
export function fellBack(look: LookRow): boolean {
  return look.engine === 'meta-model-api' && look.fallback_from === 'baseten';
}

// A Baseten request that gave no usable answer: it failed outright, or came back invalid with an error.
export function openWeightsFailed(row: OpenWeightsRow): boolean {
  return row.status === 'failed' || (row.valid === false && row.error !== null);
}

export function lookEngine(look: LookRow): string {
  if (look.engine === 'baseten') return 'Muse Spark Open (Baseten)';
  if (look.engine === 'meta-model-api') return fellBack(look) ? 'Muse Spark 1.3, because Spark Open on Baseten gave no answer' : 'Muse Spark 1.3';
  return 'not asked';
}

export function isCounterLook(look?: LookRow): boolean {
  return look?.kind === 'counter' || look?.kind === 'counter2';
}

// The caps code put on the verdict for this look: K7 for a pair look, K16 for a counter-look, K16 and K17 for the
// two-question counter-look (K17: the second look saw no damage in the box). Only then did the look lower it; the
// look's own answer never says so by itself.
export function loweredIds(v: Verdict, look?: LookRow): string[] {
  const ids = look?.kind === 'counter2' ? ['K16', 'K17'] : look?.kind === 'counter' ? ['K16'] : ['K7'];
  return ids.filter((id) => v.rule_id === id || v.trace.rule_ids.includes(id) || v.reasons.includes(id));
}

export function loweredByLook(v: Verdict, look?: LookRow): boolean {
  return loweredIds(v, look).length > 0;
}

// The look behind a verdict: a pair look when both of its marks are the verdict's marks; a counter-look when the
// verdict is about the one mark it asked about.
export function lookFor(v: Verdict, looks: LookRow[]): LookRow | undefined {
  if (v.mark_ids.length === 1) return looks.find((l) => isCounterLook(l) && l.mark_id === v.mark_ids[0]);
  if (v.mark_ids.length !== 2) return undefined;
  return looks.find((l) => !isCounterLook(l) && v.mark_ids.includes(l.pickup_mark) && v.mark_ids.includes(l.return_mark));
}

export function lookCounts(looks: LookRow[]): Record<LookVerdict, number> {
  const out: Record<LookVerdict, number> = { same: 0, different: 0, unsure: 0 };
  for (const l of looks) out[l.verdict] += 1;
  return out;
}
