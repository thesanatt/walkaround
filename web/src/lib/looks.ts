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

export function readLooks(value: unknown): LookRow[] {
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
      kind: row['kind'] === 'counter' ? 'counter' : null,
      mark_id: str(row['mark_id']),
    });
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

// Whether code put the look's cap on the verdict: K7 for a pair look, K16 for a counter-look. Only then did the look
// lower it; the look's own answer never says so by itself.
export function loweredByLook(v: Verdict, look?: LookRow): boolean {
  const id = look?.kind === 'counter' ? 'K16' : 'K7';
  return v.rule_id === id || v.trace.rule_ids.includes(id) || v.reasons.includes(id);
}

// The look behind a verdict: a pair look when both of its marks are the verdict's marks; a counter-look when the
// verdict is about the one mark it asked about.
export function lookFor(v: Verdict, looks: LookRow[]): LookRow | undefined {
  if (v.mark_ids.length === 1) return looks.find((l) => l.kind === 'counter' && l.mark_id === v.mark_ids[0]);
  if (v.mark_ids.length !== 2) return undefined;
  return looks.find((l) => l.kind !== 'counter' && v.mark_ids.includes(l.pickup_mark) && v.mark_ids.includes(l.return_mark));
}

export function lookCounts(looks: LookRow[]): Record<LookVerdict, number> {
  const out: Record<LookVerdict, number> = { same: 0, different: 0, unsure: 0 };
  for (const l of looks) out[l.verdict] += 1;
  return out;
}
