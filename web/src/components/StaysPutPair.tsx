import type { Mark, MarkTrack } from '../contracts/types.ts';
import { staysPutCaption, staysPutLine, staysPutTone } from '../lib/lines.ts';
import { panelName } from '../lib/panels.ts';
import { ClipTrail } from './ClipTrail.tsx';

export interface StaysPutPairProps {
  track: MarkTrack | undefined; // the mark's own clip
  control: MarkTrack | undefined; // the reflection control, on purpose another panel
  mark: Mark | undefined;
  color: string;
  calibrated: boolean; // thresholds staysput.cap_enabled: only then do the captions state a verdict
  numbers?: boolean; // the sheet's drift line under each caption (run page); the kiosk shows the caption alone
  atFrame?: number; // a frozen kiosk capture: both clips paused on this frame
}

const CONTROL_COLOR = '#d9a84a';

// The Stays Put test as two clips side by side: the mark tracked through its own clip, and the control tracked the
// same way. Each caption says what the track decided, nothing more, and decides nothing until Stays Put is calibrated.
export function StaysPutPair({ track, control, mark, color, calibrated, numbers = false, atFrame }: StaysPutPairProps) {
  if (!track?.clip && !control?.clip) return <div className="empty">No Stays Put clip for this panel. The match below still stands on its own.</div>;
  return (
    <div className="staysput-pair">
      {track?.clip ? (
        <figure className="staysput-clip main">
          <ClipTrail mt={track} color={color} caption={`the ${mark?.type ?? 'mark'} on the ${panelName(mark?.panel_id)}, tracked by SAM 3.1 through its own clip`} bare atFrame={atFrame} />
          <figcaption>
            <span className={`sp-caption ${staysPutTone(track, calibrated)}`}>{staysPutCaption(track, calibrated)}</span>
            {numbers && staysPutLine(track, calibrated) ? <span className="sp-numbers mono">{staysPutLine(track, calibrated)}</span> : null}
          </figcaption>
        </figure>
      ) : (
        <div className="empty">This mark has no Stays Put clip.</div>
      )}
      {control?.clip ? (
        <figure className="staysput-clip inset">
          <ClipTrail mt={control} color={CONTROL_COLOR} caption={calibrated ? 'a reflection, tracked the same way' : 'the control clip, tracked the same way'} bare atFrame={atFrame} />
          <figcaption>
            <span className={`sp-caption ${staysPutTone(control, calibrated)}`}>{staysPutCaption(control, calibrated)}</span>
            {numbers && staysPutLine(control, calibrated) ? <span className="sp-numbers mono">{staysPutLine(control, calibrated)}</span> : null}
          </figcaption>
        </figure>
      ) : null}
    </div>
  );
}

// The control the pair shows: a tracked control with a clip, a sliding one first.
export function controlTrack(markTracks: MarkTrack[]): MarkTrack | undefined {
  const usable = markTracks.filter((m) => m.is_control && m.clip);
  return usable.find((m) => m.verdict === 'sliding') ?? usable[0];
}
