import { rolesOf } from '../state/reducer.ts';
import type { RunState } from '../state/types.ts';

export function laneCounts(state: RunState): { done: number; total: number; notRun: number } {
  const nouns = state.run?.noun_list.length || 16;
  const roles = rolesOf(state);
  let done = 0;
  let notRun = 0;
  for (const role of roles) {
    for (const lane of Object.values(state.lanes[role] ?? {})) {
      if (lane.status === 'completed' || lane.status === 'zero_matches' || lane.status === 'incomplete') done += 1;
      else if (lane.status === 'failed') notRun += 1;
    }
  }
  return { done, total: nouns * roles.length, notRun };
}

// The proxy fps the run froze (thresholds.media.proxy_fps), else 15.
function proxyFps(state: RunState): number {
  const fps = (state.run?.thresholds?.['media'] as Record<string, unknown> | undefined)?.['proxy_fps'];
  return typeof fps === 'number' && fps > 0 ? fps : 15;
}

// The step line names the model and the count ("SAM 3.1: 11 of 32 lanes done").
export function stageText(state: RunState): string {
  const { done, total, notRun } = laneCounts(state);
  switch (state.stage.name) {
    case 'created':
      return Object.keys(state.media).length || Object.keys(state.videos).length ? 'received; sealing next' : 'waiting for the upload';
    case 'sealing':
      return 'sealing: manifest, OpenTimestamps, freetsa token';
    case 'sealed':
      return 'sealed before any model call';
    case 'converting':
      return `converting: the ${proxyFps(state)} fps proxy, the frame time map, the audio`;
    case 'tracking':
      return `SAM 3.1: ${done} of ${total} lanes done${notRun ? `, ${notRun} not run` : ''}`;
    case 'blurring':
      return 'blurring plates and faces on the proxy before anything is stored';
    case 'marking':
      return `Muse Spark: ${state.spark.answered} crops answered${state.spark.truncated ? `, ${state.spark.truncated} truncated` : ''}`;
    case 'staysput':
      return 'Stays Put: SAM 3.1 tracks each mark through its own clip';
    case 'matching':
      return 'matching marks across the two walks (code only, no model)';
    case 'sheet':
      return 'writing the sheet';
    case 'done':
      return 'done';
    case 'partial':
      return state.partialText ?? `incomplete: ${state.partial?.join(', ') ?? 'some stages'} missing; the sheet shows what was tracked`;
    case 'failed':
      return state.errors[state.errors.length - 1] ?? 'the run failed';
    case 'interrupted':
      return 'interrupted by a server restart; resuming the lanes that had not finished';
    default:
      return state.stage.name;
  }
}

export function StageLine({ state }: { state: RunState }) {
  return (
    <div className="stage-line" role="status">
      <div className="stage-bar" aria-hidden="true">
        <span style={{ width: `${Math.max(0, Math.min(100, state.stage.pct))}%` }} />
      </div>
      <span>{stageText(state)}</span>
    </div>
  );
}
