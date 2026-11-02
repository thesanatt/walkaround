import { rolesOf } from '../state/reducer.ts';
import type { RunState } from '../state/types.ts';
import { joinedPairs } from './focus.ts';
import { controlName, staysPutCalibrated } from './lines.ts';
import { fellBack, isOpenWeights, lookCounts, openWeightsFailed } from './looks.ts';

// "How it was checked": what each model did in this run, counted from the run's own files (tracks, mark_tracks,
// marks, keyframes, narration, looks, open_weights, verdicts). Nothing here is a claim about other runs.

export type CheckKey = 'sam' | 'spark' | 'voice' | 'open' | 'code';
export type CheckState = 'done' | 'waiting' | 'not_run';

export interface CheckLine {
  key: CheckKey;
  model: string;
  did: string;
  detail: string;
  state: CheckState;
  link: { rel: string; text: string } | null;
}

function n(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`;
}

function andList(parts: string[]): string {
  if (parts.length <= 1) return parts.join('');
  return `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`;
}

const STAGE_ORDER = ['created', 'sealing', 'sealed', 'converting', 'tracking', 'blurring', 'marking', 'staysput', 'matching', 'sheet', 'done'];

// True once the run is past a stage: a later stage started, or the run ended.
function past(state: RunState, stage: string): boolean {
  if (state.completed) return true;
  return STAGE_ORDER.indexOf(state.stage.name) > STAGE_ORDER.indexOf(stage);
}

function samLine(state: RunState): CheckLine {
  const roles = rolesOf(state);
  let answered = 0;
  let notRun = 0;
  for (const role of roles) {
    for (const lane of Object.values(state.lanes[role] ?? {})) {
      if (lane.status === 'completed' || lane.status === 'zero_matches' || lane.status === 'incomplete') answered += 1;
      else if (lane.status === 'failed') notRun += 1;
    }
  }
  const total = (state.run?.noun_list.length || 16) * roles.length;
  const panels = new Set(state.tracks.map((t) => t.panel_id).filter((p): p is NonNullable<typeof p> => p !== null)).size;
  const marks = state.markTracks.filter((m) => !m.is_control && m.track.length > 0 && m.verdict !== 'no_track' && m.verdict !== 'not_run').length;
  const controls = state.markTracks.filter((m) => m.is_control && m.track.length > 0).length;
  const base = { key: 'sam' as const, model: 'SAM 3.1', link: null };
  if (!answered && !state.tracks.length) return { ...base, did: 'waiting for the walks', detail: 'outlines every named panel from plain nouns', state: 'waiting' };
  if (!state.tracks.length) return { ...base, did: `${answered} of ${total} lanes answered so far`, detail: 'panels come once the lanes finish', state: 'waiting' };
  const control = controlName(staysPutCalibrated(state.run));
  let did = `tracked ${n(panels, 'panel', 'panels')} in ${n(roles.length, 'walk', 'walks')}`;
  if (marks || controls) did += ` and ${marks === 1 ? '1 mark through its own clip' : `${marks} marks through their own clips`}${controls ? `, plus ${n(controls, control, `${control}s`)}` : ''}`;
  const blurred = roles.some((r) => state.media[r]?.blurredPath);
  const noStaysPut = state.partial?.includes('staysput') && !state.markTracks.length;
  // The original and the unblurred proxy sit in quarantine during the run, so "blurred before storage" would overstate.
  const detail =
    `${answered} of ${total} lanes answered${notRun ? `, ${notRun} not run` : ''}` +
    (blurred ? '; the plates and faces it found are blurred in every frame kept or shown' : '') +
    (noStaysPut ? '; Stays Put did not run, so no mark was tracked through its own clip' : '');
  return { ...base, did, detail, state: 'done' };
}

function sparkLine(state: RunState): CheckLine {
  const walkMarks = state.marks.filter((m) => m.role === 'pickup' || m.role === 'return');
  const crops = state.keyframes.filter((k) => k.marks_status !== 'not_run');
  const truncated = state.keyframes.filter((k) => k.marks_status === 'truncated').length;
  const seeded = walkMarks.filter((m) => m.source === 'narration_seed').length;
  const metaLooks = state.looks.filter((l) => l.engine === 'meta-model-api').length;
  const base = { key: 'spark' as const, model: 'Muse Spark 1.3', link: null };
  if (!crops.length) {
    const waiting = !past(state, 'marking');
    return { ...base, did: waiting ? 'waiting for the blurred crops' : 'saw no crop in this run', detail: 'boxes marks in one blurred crop per call', state: waiting ? 'waiting' : 'not_run' };
  }
  let did = `found ${n(walkMarks.length, 'mark', 'marks')} in ${n(crops.length, 'blurred crop', 'blurred crops')}`;
  if (seeded) did += `, ${seeded} of them from your words`;
  const parts = ['one crop per call, and it writes no label'];
  if (truncated) parts.push(`${n(truncated, 'answer was', 'answers were')} cut off after the retry`);
  if (metaLooks) parts.push(`it took a second look at ${n(metaLooks, 'pair', 'pairs')}`);
  return { ...base, did, detail: parts.join('; '), state: 'done' };
}

// The walks voice.done said have no audio track (orchestrator._stage_narration), in role order.
function silentWalks(state: RunState): { silent: string[]; expected: string[] } {
  const expected = [...new Set([...rolesOf(state), ...Object.keys(state.voice).filter((r) => r === 'pickup' || r === 'return')])];
  return { silent: expected.filter((r) => /no audio/i.test(state.voice[r]?.text ?? '')), expected };
}

function voiceLine(state: RunState): CheckLine {
  const base = { key: 'voice' as const, model: 'Muse Voice Transcribe', link: null };
  const narration = state.narration;
  if (!narration) {
    const { silent, expected } = silentWalks(state);
    const pins = 'pins what you say to the panel on screen';
    if (expected.length && silent.length === expected.length) return { ...base, did: `no audio track in ${expected.length === 1 ? 'this walk' : 'these walks'}`, detail: pins, state: 'not_run' };
    // One walk without audio (a narrated pickup, a silent return) says so for that walk only.
    const detail = silent.length ? `${pins}; the ${andList(silent)} ${silent.length === 1 ? 'walk has' : 'walks have'} no audio track` : pins;
    if (state.partial?.includes('narration')) return { ...base, did: 'not run in this run', detail, state: 'not_run' };
    if (past(state, 'staysput')) return { ...base, did: 'no narration was turned into notes', detail, state: 'not_run' };
    return { ...base, did: Object.keys(state.voice).length ? 'waiting for the narration step' : 'waiting for the voice step', detail, state: 'waiting' };
  }
  const turns = Object.values(narration.per_role).flatMap((r) => r.turns);
  const pinned = turns.filter((t) => t.pinned_panel).length;
  const sourceOf = (id: string | null) => (id ? state.marks.find((m) => m.mark_id === id)?.source : undefined);
  const seeds = turns.filter((t) => sourceOf(t.seed_mark_id) === 'narration_seed').length;
  const links = turns.filter((t) => t.seed_mark_id && sourceOf(t.seed_mark_id) !== 'narration_seed').length;
  if (!narration.mode_applied) return { ...base, did: 'returned your words without turns, so nothing was pinned', detail: 'the transcript is shown as it came back', state: 'done' };
  if (!turns.length) return { ...base, did: 'heard no narration in these walks', detail: 'pins what you say to the panel on screen', state: 'done' };
  const did = `pinned ${pinned} of ${n(turns.length, 'spoken note', 'spoken notes')} to a panel`;
  const parts = [seeds ? `${n(seeds, 'note', 'notes')} gave Muse Spark a mark to box` : '', links ? `${n(links, 'note', 'notes')} named a mark it had already boxed` : ''].filter(Boolean);
  const detail = parts.length ? parts.join('; ') : 'each note keeps its start and end time';
  return { ...base, did, detail, state: 'done' };
}

function openLine(state: RunState): CheckLine {
  const open = state.looks.filter(isOpenWeights);
  const hosted = state.looks.filter(fellBack).length;
  const failed = state.openWeights.filter(openWeightsFailed).length;
  const base = { key: 'open' as const, model: 'Muse Spark Open on Baseten' };
  const link = open.length || state.openWeights.length ? { rel: 'open_weights.ndjson', text: 'open_weights.ndjson' } : null;
  if (!open.length) {
    if (hosted) return { ...base, did: `gave no answer, so hosted Muse Spark 1.3 took ${n(hosted, 'second look', 'second looks')}`, detail: 'every request is in the provenance log', state: 'not_run', link };
    // Tried, but no look stands on it: every request failed, and no hosted answer stood in either.
    if (state.openWeights.length) {
      const detail = failed ? `${n(failed, 'request', 'requests')} got no usable answer, so no label changed; each one is in the provenance log` : 'no label changed; each request is in the provenance log';
      return { ...base, did: 'was tried and gave no answer', detail, state: 'not_run', link };
    }
    const waiting = !past(state, 'matching');
    return { ...base, did: waiting ? 'waiting for the match step' : 'not run in this run', detail: 'open weights, used only for the second look', state: waiting ? 'waiting' : 'not_run', link };
  }
  const c = lookCounts(open);
  const parts = [c.same ? `${c.same} same` : '', c.different ? `${c.different} different` : '', c.unsure ? `${c.unsure} unsure` : ''].filter(Boolean);
  const repeats = open.some((l) => (l.answers ?? 1) > 1);
  const extra = [
    repeats ? 'each look asked up to 3 times; the majority answer counts once' : '',
    hosted ? `${n(hosted, 'look', 'looks')} fell back to hosted Muse Spark 1.3` : '',
    failed ? `${n(failed, 'request', 'requests')} got no answer` : '',
  ].filter(Boolean);
  return {
    ...base,
    did: `took ${n(open.length, 'second look', 'second looks')}: ${andList(parts)}`,
    detail: ['different or unsure lowers a label to needs your eye; it never raises one', ...extra].join('; '),
    state: 'done',
    link,
  };
}

function codeLine(state: RunState): CheckLine {
  const base = { key: 'code' as const, model: 'Code, no model', link: null };
  if (state.run && state.run.kind !== 'pair') return { ...base, did: 'kept the dated frames; a pickup check has no return walk to match', detail: 'labels need both walks', state: 'done' };
  if (!state.verdicts.length) {
    const waiting = !past(state, 'matching');
    return { ...base, did: waiting ? 'waiting for the match step' : 'had no marks to compare', detail: 'matches marks and writes every label', state: waiting ? 'waiting' : 'done' };
  }
  const pairs = state.matches.reduce((sum, m) => sum + joinedPairs(m).length, 0);
  const hash = state.run?.config_hash ? `; thresholds ${state.run.config_hash.slice(0, 12)}` : '';
  return { ...base, did: `matched ${n(pairs, 'pair', 'pairs')} and wrote ${n(state.verdicts.length, 'label', 'labels')}`, detail: `a model answer can only lower a label${hash}`, state: 'done' };
}

export function howChecked(state: RunState): CheckLine[] {
  return [samLine(state), sparkLine(state), voiceLine(state), openLine(state), codeLine(state)];
}
