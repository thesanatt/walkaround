import type { PanelId } from '../contracts/types.ts';

// Kiosk scenes keyed to docs/VIDEO_PLAN.md beats. /kiosk/<run>?scene=<name>&autostart=1 starts one by URL;
// ?t=<ms> freezes it at that many ms after its start for a frame-by-frame capture.
export type SceneLayout =
  | 'seal'
  | 'walk'
  | 'fragments'
  | 'coverage'
  | 'crops'
  | 'moment'
  | 'verdict'
  | 'hood'
  | 'claim'
  | 'sheet'
  | 'topup'
  | 'blur'
  | 'checks'
  | 'guardrails';

// How the replay feeds a scene: from the first event, from the tracking stage (everything earlier applied at once),
// or the finished run at once so the scene's own steps do the timing.
export type SceneReplay = 'from_start' | 'from_tracking' | 'final';

// One row of a scene's delay table: from atMs on, the scene shows `name`.
export interface SceneStep {
  atMs: number;
  name: string;
}

export interface Scene {
  name: string;
  beat: string;
  layout: SceneLayout;
  heading: string; // a product heading, not a caption; the edit adds the voiceover captions
  panel: PanelId | null; // the default panel; ?panel= overrides it, and a run without it falls back to its first match
  replay: SceneReplay;
  holdMs: number; // how long the beat runs; the capture script reads it, the page never advances by itself
  delays?: Record<string, number>; // stage budgets for this scene's fixed replay of the run (kiosk/delays.json fills the rest)
  steps: SceneStep[]; // the scene's delay table, ms from its start; the scene clock moves only at these
  footage: boolean; // shows pixels from the walks, so the run's footage credit goes on screen
}

// Evenly spaced steps of one name, for a reveal that fills in over time (the coverage car, the crop's boxes).
export function ticks(fromMs: number, toMs: number, everyMs: number, name: string): SceneStep[] {
  const out: SceneStep[] = [];
  for (let at = fromMs; at <= toMs; at += everyMs) out.push({ atMs: at, name });
  return out;
}

export const COVERAGE_FILL = { fromMs: 400, toMs: 3000, everyMs: 200 };
export const CROP_MARKS = { fromMs: 800, everyMs: 350 };
export const CHECK_ROWS = { fromMs: 300, everyMs: 900 };

export const SCENES: Scene[] = [
  {
    name: 'seal',
    beat: 'B5',
    layout: 'seal',
    heading: 'two walks, hashed on the phone, sealed on arrival',
    panel: null,
    replay: 'from_start',
    holdMs: 7000,
    // upload.done at 1.0 s and 1.6 s, the manifest at 2.8 s, the freetsa token at 3.4 s, OpenTimestamps at 4.0 s
    delays: { created: 1600, sealing: 2400, sealed: 800 },
    steps: [{ atMs: 0, name: 'cards' }],
    footage: false,
  },
  {
    name: 'masks',
    beat: 'B5',
    layout: 'walk',
    heading: 'pickup walk',
    panel: null,
    replay: 'from_tracking',
    holdMs: 8000,
    // The tracking stage paced over 5 s instead of 14, so the pickup panel lanes finish (and their outlines bloom)
    // inside the first half of the hold. TODO(D1): check on the real sample pair's fixed replay.
    delays: { tracking: 5000 },
    steps: [{ atMs: 0, name: 'walk' }],
    footage: true,
  },
  {
    name: 'fragments',
    beat: 'B6',
    layout: 'fragments',
    heading: 'one panel, two visits, one track',
    panel: 'right_rear_door',
    replay: 'final',
    holdMs: 6000,
    steps: [
      { atMs: 0, name: 'fragments' },
      { atMs: 2200, name: 'joined' },
      { atMs: 4000, name: 'anchor' },
    ],
    footage: true,
  },
  {
    name: 'coverage',
    beat: 'B7',
    layout: 'coverage',
    heading: 'coverage',
    panel: null,
    replay: 'final',
    holdMs: 5000,
    steps: [{ atMs: 0, name: 'outline' }, ...ticks(COVERAGE_FILL.fromMs, COVERAGE_FILL.toMs, COVERAGE_FILL.everyMs, 'fill'), { atMs: 3400, name: 'missing' }],
    footage: false,
  },
  {
    name: 'crops',
    beat: 'B8',
    layout: 'crops',
    heading: 'one crop, one panel, one Spark call',
    panel: 'right_rear_door',
    replay: 'final',
    holdMs: 5000,
    steps: [{ atMs: 0, name: 'crop' }, ...ticks(CROP_MARKS.fromMs, CROP_MARKS.fromMs + 5 * CROP_MARKS.everyMs, CROP_MARKS.everyMs, 'marks')],
    footage: true,
  },
  {
    name: 'moment',
    beat: 'B9',
    layout: 'moment',
    heading: '',
    panel: 'right_rear_door',
    replay: 'final',
    holdMs: 12000,
    // dim: the walk dims to the panel with the renter's words; trail: the mark's own track beside the reflection
    // (upgraded form) or the boxed pickup frame (guaranteed form); cut, return, join: the match cut with its 8-frame
    // dissolve and the join line; card: "already there" with the two dates.
    steps: [
      { atMs: 0, name: 'dim' },
      { atMs: 3000, name: 'trail' },
      { atMs: 7000, name: 'cut' },
      { atMs: 7600, name: 'return' },
      { atMs: 8200, name: 'join' },
      { atMs: 10000, name: 'card' },
    ],
    footage: true,
  },
  {
    name: 'tape',
    beat: 'B10',
    layout: 'verdict',
    heading: 'added between the walks',
    panel: 'left_front_door',
    replay: 'final',
    holdMs: 8000,
    steps: [
      { atMs: 0, name: 'frames' },
      { atMs: 1500, name: 'card' },
    ],
    footage: true,
  },
  {
    name: 'hood',
    beat: 'B11',
    layout: 'hood',
    heading: 'sky on the hood',
    panel: 'hood',
    replay: 'final',
    holdMs: 8000,
    steps: [
      { atMs: 0, name: 'frames' },
      { atMs: 2500, name: 'refusal' },
      { atMs: 5000, name: 'line' },
    ],
    footage: true,
  },
  {
    name: 'claim',
    beat: 'B12',
    layout: 'claim',
    heading: 'a claim photo against the pickup walk',
    panel: null,
    replay: 'final',
    holdMs: 16000,
    steps: [
      { atMs: 0, name: 'photo' },
      { atMs: 1800, name: 'spark' },
      { atMs: 2800, name: 'sam' },
      { atMs: 4200, name: 'frame' },
      { atMs: 6500, name: 'size' },
      { atMs: 13500, name: 'second' },
    ],
    footage: true,
  },
  { name: 'sheet', beat: 'B13', layout: 'sheet', heading: 'the sheet', panel: null, replay: 'final', holdMs: 15000, steps: [{ atMs: 0, name: 'sheet' }], footage: true },
  {
    name: 'topup',
    beat: 'B14',
    layout: 'topup',
    heading: 'what is missing, before you drive off',
    panel: null,
    replay: 'final',
    holdMs: 14000,
    steps: [
      { atMs: 0, name: 'ask' },
      { atMs: 4000, name: 'running' },
      { atMs: 9000, name: 'filmed' },
    ],
    footage: false,
  },
  // The heading speaks for the frame on screen: the original sits in quarantine until the run ends, so "blurred before
  // storage" said of the whole run would overstate (TODO(Sanat): sign off on this wording and on the B15 VO line).
  { name: 'blur', beat: 'B15', layout: 'blur', heading: 'blurred before the frame is saved', panel: 'right_rear_door', replay: 'final', holdMs: 4000, steps: [{ atMs: 0, name: 'blur' }], footage: true },
  {
    name: 'checks',
    beat: 'B16',
    layout: 'checks',
    heading: 'how this run was checked',
    panel: null,
    replay: 'final',
    holdMs: 9000,
    steps: [{ atMs: 0, name: 'rows' }, ...ticks(CHECK_ROWS.fromMs, CHECK_ROWS.fromMs + 4 * CHECK_ROWS.everyMs, CHECK_ROWS.everyMs, 'rows')],
    footage: false,
  },
  { name: 'guardrails', beat: 'B16', layout: 'guardrails', heading: 'what it cost and what was deleted', panel: null, replay: 'final', holdMs: 8000, steps: [{ atMs: 0, name: 'ledger' }], footage: false },
];

export function sceneByName(name: string | null): Scene {
  return SCENES.find((s) => s.name === name) ?? SCENES.find((s) => s.name === 'masks')!;
}

export function neighbourScene(name: string, direction: 1 | -1): Scene {
  const index = SCENES.findIndex((s) => s.name === name);
  return SCENES[Math.max(0, Math.min(SCENES.length - 1, (index < 0 ? 0 : index) + direction))]!;
}

export function stepAt(scene: Scene, elapsedMs: number): SceneStep | null {
  let current: SceneStep | null = null;
  for (const step of scene.steps) if (step.atMs <= elapsedMs) current = step;
  return current;
}

// How many items of a reveal are on screen at elapsedMs: the first at fromMs, one more every everyMs.
export function revealCount(elapsedMs: number, fromMs: number, everyMs: number): number {
  return elapsedMs < fromMs ? 0 : Math.floor((elapsedMs - fromMs) / everyMs) + 1;
}
