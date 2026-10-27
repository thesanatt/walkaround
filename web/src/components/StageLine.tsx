import type { RunState } from '../state/types.ts';

function laneCounts(state: RunState): { done: number; total: number; failed: number } {
  const nouns = state.run?.noun_list.length ?? 16;
  const roles = Object.keys(state.lanes).length || 1;
  let done = 0;
  let failed = 0;
  for (const lanes of Object.values(state.lanes)) {
    for (const lane of Object.values(lanes)) {
      if (lane.status === 'completed' || lane.status === 'zero_matches' || lane.status === 'incomplete') done += 1;
      if (lane.status === 'failed') failed += 1;
    }
  }
  return { done, total: nouns * roles, failed };
}

// The step line names the model and the count, never a spinner alone.
export function stageText(state: RunState): string {
  const { done, total, failed } = laneCounts(state);
  switch (state.stage.name) {
    case 'created':
      return 'run created, waiting for the upload';
    case 'sealing':
      return 'sealing: manifest, OpenTimestamps, freetsa';
    case 'sealed':
      return 'sealed; converting next';
    case 'converting':
      return 'converting: proxy, pts map, WAV';
    case 'tracking':
      return `SAM 3.1: ${done} of ${total} lanes done${failed ? `, ${failed} not run` : ''}`;
    case 'blurring':
      return 'blurring plates and faces on the proxy';
    case 'marking':
      return `Spark: ${state.spark.answered} of ${state.spark.requested} crops${state.spark.truncated ? `, ${state.spark.truncated} truncated` : ''}`;
    case 'staysput':
      return 'Stays Put: tracking each mark through its clip';
    case 'matching':
      return 'matching marks across the two walks (code only)';
    case 'sheet':
      return 'writing the sheet';
    case 'done':
      return 'done';
    case 'partial':
      return `incomplete: ${state.partial?.length ?? 0} ${state.partial?.length === 1 ? 'item' : 'items'} missing`;
    case 'failed':
      return state.errors[state.errors.length - 1] ?? 'failed';
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
