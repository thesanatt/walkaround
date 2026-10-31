import type { Keyframe, Mark } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { bestKeyframe } from '../lib/focus.ts';
import { formatClock } from '../lib/format.ts';
import { markLabel } from '../lib/lines.ts';
import { colorForPanel, panelName } from '../lib/panels.ts';

export { bestKeyframe };

export function conditionFlags(kf: Keyframe): string[] {
  if (kf.marks_status === 'not_run') return [];
  const c = kf.conditions;
  const out = [c.wet ? 'wet' : null, c.dirty ? 'dirty' : null, c.glare ? 'glare' : null, c.dark ? 'dark' : null].filter((x): x is string => x !== null);
  return out.length ? out : ['dry, no glare'];
}

export function marksStatusText(kf: Keyframe, count: number): string {
  if (kf.marks_status === 'truncated') return 'marks incomplete: the Spark answer was cut off after the retry';
  if (kf.marks_status === 'not_run') return 'Muse Spark has not seen this crop yet';
  return count ? `${count} ${count === 1 ? 'mark' : 'marks'} from Muse Spark` : 'Muse Spark found no marks on this crop';
}

// One blurred crop with the Spark boxes dropping in one by one (type and size class on each), flags in a corner.
// shown: how many boxes are on screen so far (the kiosk drops them in by its step table); all of them when absent.
export function CropFigure({ kf, marks, title, shown }: { kf: Keyframe; marks: Mark[]; title: string; shown?: number }) {
  const url = useFileUrl();
  const own = marks.filter((m) => m.kf_id === kf.kf_id);
  const visible = shown === undefined ? own : own.slice(0, shown);
  if (!kf.crop) return <div className="empty">{title}: keyframe at {formatClock(kf.pts)} has no crop file.</div>;
  const { w, h } = kf.crop;
  const color = colorForPanel(kf.panel_id);
  // Sizes in crop pixels, scaled to the crop so a label reads the same at any display width.
  const label = { fontSize: Math.round(w / 34), strokeWidth: Math.max(3, w / 220) };
  const stroke = Math.max(3, w / 400);
  return (
    <figure className="crop">
      <div className="crop-frame">
        <img src={url(kf.crop.path)} width={w} height={h} alt={`${panelName(kf.panel_id)} from the ${title} walk at ${formatClock(kf.pts)}, plates and faces pixelated`} />
        <svg viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
          {visible.map((m, i) =>
            m.bbox_crop ? (
              <g key={m.mark_id} className="mark-box" style={shown === undefined ? { animationDelay: `${i * 250}ms` } : undefined}>
                <rect x={m.bbox_crop[0]} y={m.bbox_crop[1]} width={m.bbox_crop[2] - m.bbox_crop[0]} height={m.bbox_crop[3] - m.bbox_crop[1]} stroke={color} strokeWidth={stroke} />
                <text x={m.bbox_crop[0]} y={Math.max(label.fontSize, m.bbox_crop[1] - label.fontSize / 3)} style={label}>
                  {markLabel(m)}
                </text>
              </g>
            ) : null,
          )}
        </svg>
        {kf.marks_status !== 'not_run' ? (
          <div className="crop-flags">
            {conditionFlags(kf).map((f) => (
              <span key={f}>{f}</span>
            ))}
          </div>
        ) : null}
      </div>
      <figcaption>
        {title}, frame {kf.f_proxy} at {formatClock(kf.pts)}: {marksStatusText(kf, own.length)}
        {kf.window_subtracted ? '; window subtracted from the door mask' : ''}
      </figcaption>
    </figure>
  );
}

// The walk's best crop of the panel, then every other keyframe of it that holds a mark: a deduplicated mark keeps the
// keyframe it was found on, so each mark is drawn on its own crop.
export function CropView({ role, panelId, keyframes, marks }: { role: string; panelId: string | null; keyframes: Keyframe[]; marks: Mark[] }) {
  if (!panelId) return <div className="empty">Pick a panel in the list to see its crop.</div>;
  const kf = bestKeyframe(keyframes, role, panelId);
  if (!kf) return <div className="empty">No {role} keyframe for the {panelName(panelId)} yet. Keyframes come with the panel step.</div>;
  const others = keyframes.filter((k) => k.role === role && k.panel_id === panelId && k.kf_id !== kf.kf_id && marks.some((m) => m.kf_id === k.kf_id));
  return (
    <div className="crop-stack">
      <CropFigure kf={kf} marks={marks} title={role} />
      {others.map((k) => (
        <CropFigure key={k.kf_id} kf={k} marks={marks} title={`${role}, another frame`} />
      ))}
    </div>
  );
}
