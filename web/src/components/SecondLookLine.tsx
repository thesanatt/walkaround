import type { LookRow } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { fellBack, lookEngine } from '../lib/looks.ts';

// "second look: Muse Spark Open (Baseten), same: <reason>", with the two outlined crops it was shown and the
// provenance log. lowered is whether the verdict carries K7 (code's rule id, not the look's answer): only then does
// the line say the look lowered the label.
export function SecondLookLine({ look, lowered }: { look: LookRow; lowered: boolean }) {
  const url = useFileUrl();
  const asked = look.engine !== null;
  const logged = look.engine === 'baseten' || fellBack(look);
  return (
    <p className={`second-look ${look.verdict}`} title={look.fallback ?? undefined}>
      {asked ? (
        <>
          second look: {lookEngine(look)}, <strong>{look.verdict}</strong>
          {look.reason ? `: ${look.reason}` : ''}
        </>
      ) : (
        <>second look: {look.reason || 'not asked'}</>
      )}
      {lowered ? ' (K7: this lowered the label to needs your eye)' : ''}
      {look.inputs.length === 2 ? (
        <>
          {'; what it saw: '}
          <a href={url(look.inputs[0]!)}>pickup crop</a>, <a href={url(look.inputs[1]!)}>return crop</a>
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
