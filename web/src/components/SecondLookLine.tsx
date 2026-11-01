import type { LookRow } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { fellBack, isCounterLook, lookEngine } from '../lib/looks.ts';

// "second look: Muse Spark Open (Baseten), same: <reason>", with the two outlined crops it was shown and the
// provenance log. lowered is whether the verdict carries K7, or K16 for a counter-look (code's rule id, not the look's
// answer): only then does the line say the look lowered the label. ids names the caps when the caller knows them (the
// two-question counter-look can put K17, K16 or both).
const SPOT_WORD: Record<string, string> = { same: 'yes', different: 'no', unsure: 'unsure' };

export function SecondLookLine({ look, lowered, ids }: { look: LookRow; lowered: boolean; ids?: string[] }) {
  const url = useFileUrl();
  const asked = look.engine !== null;
  const logged = look.engine === 'baseten' || fellBack(look);
  const counter = isCounterLook(look);
  const two = look.kind === 'counter2';
  const own = counter && look.mark_id === look.pickup_mark ? 'pickup' : 'return';
  const other = own === 'pickup' ? 'return' : 'pickup';
  const name = two ? `second look at the mark and the same spot in the ${other} walk` : counter ? `second look at the same spot in the ${other} walk` : 'second look';
  const capIds = ids && ids.length > 0 ? ids.join(' and ') : counter ? (two ? 'K16 or K17' : 'K16') : 'K7';
  return (
    <p className={`second-look ${look.verdict}`} title={look.fallback ?? undefined}>
      {asked ? (
        two ? (
          <>
            {name}: {lookEngine(look)}, damage in the box: <strong>{look.mark_seen ?? 'unsure'}</strong>, same mark at the {other} spot:{' '}
            <strong>{SPOT_WORD[look.verdict] ?? look.verdict}</strong>
            {look.reason ? `: ${look.reason}` : ''}
            {look.engine === 'baseten' && look.model ? ` (${look.model})` : ''}
          </>
        ) : (
          <>
            {name}: {lookEngine(look)}, <strong>{look.verdict}</strong>
            {look.reason ? `: ${look.reason}` : ''}
            {look.engine === 'baseten' && look.model ? ` (${look.model})` : ''}
          </>
        )
      ) : (
        <>
          {name}: {look.reason || 'not asked'}
        </>
      )}
      {lowered ? ` (${capIds}: this lowered the label to needs your eye)` : ''}
      {look.inputs.length === 2 ? (
        <>
          {'; what it saw: '}
          <a href={url(look.inputs[0]!)}>{counter ? own : 'pickup'} crop</a>, <a href={url(look.inputs[1]!)}>{counter ? `${other} crop with the spot` : 'return crop'}</a>
        </>
      ) : null}
      {logged ? (
        <>
          {'; log: '}
          <a href={url('open_weights.ndjson')}>open_weights.ndjson</a>
        </>
      ) : null}
    </p>
  );
}
