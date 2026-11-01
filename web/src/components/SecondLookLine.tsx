import type { LookRow } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { fellBack, lookEngine } from '../lib/looks.ts';

// "second look: Muse Spark Open (Baseten), same: <reason>", with the two outlined crops it was shown and the
// provenance log. lowered is whether the verdict carries K7, or K16 for a counter-look (code's rule id, not the look's
// answer): only then does the line say the look lowered the label.
export function SecondLookLine({ look, lowered }: { look: LookRow; lowered: boolean }) {
  const url = useFileUrl();
  const asked = look.engine !== null;
  const logged = look.engine === 'baseten' || fellBack(look);
  const counter = look.kind === 'counter';
  const own = counter && look.mark_id === look.pickup_mark ? 'pickup' : 'return';
  const other = own === 'pickup' ? 'return' : 'pickup';
  const name = counter ? `second look at the same spot in the ${other} walk` : 'second look';
  return (
    <p className={`second-look ${look.verdict}`} title={look.fallback ?? undefined}>
      {asked ? (
        <>
          {name}: {lookEngine(look)}, <strong>{look.verdict}</strong>
          {look.reason ? `: ${look.reason}` : ''}
          {look.engine === 'baseten' && look.model ? ` (${look.model})` : ''}
        </>
      ) : (
        <>
          {name}: {look.reason || 'not asked'}
        </>
      )}
      {lowered ? ` (${counter ? 'K16' : 'K7'}: this lowered the label to needs your eye)` : ''}
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
