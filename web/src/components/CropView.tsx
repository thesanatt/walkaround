import { routes } from '../api/client.ts';
import type { Keyframe, Mark } from '../contracts/types.ts';
import { formatClock } from '../lib/format.ts';
import { colorForPanel, panelName } from '../lib/panels.ts';

export function conditionFlags(kf: Keyframe): string[] {
  if (kf.marks_status === 'not_run') return [];
  const c = kf.conditions;
  const out: string[] = [];
  if (c.wet) out.push('wet');
  if (c.dirty) out.push('dirty');
  if (c.glare) out.push('glare');
  if (c.dark) out.push('dark');
  return out.length ? out : ['dry, no glare'];
}

export function bestKeyframe(keyframes: Keyframe[], role: string, panelId: string): Keyframe | undefined {
  let best: Keyframe | undefined;
  for (const k of keyframes) {
    if (k.role !== role || k.panel_id !== panelId) continue;
    if (!best || k.score.total > best.score.total) best = k;
  }
  return best;
}

export function marksStatusText(kf: Keyframe, count: number): string {
  if (kf.marks_status === 'truncated') return 'marks incomplete (the Spark answer was truncated after the retry)';
  if (kf.marks_status === 'not_run') return 'Spark has not seen this crop yet';
  return `${count} ${count === 1 ? 'mark' : 'marks'} from Muse Spark`;
}

// One blurred crop with the Spark boxes dropping in one by one, type and size class on each, flags in the corner.
export function CropFigure({ runId, kf, marks, title }: { runId: string; kf: Keyframe; marks: Mark[]; title: string }) {
  const own = marks.filter((m) => m.kf_id === kf.kf_id);
  if (!kf.crop) {
    return (
      <div className="empty">
        {title}: keyframe {kf.kf_id} at {formatClock(kf.pts)} has no crop file yet.
      </div>
    );
  }
  const { w, h } = kf.crop;
  const color = colorForPanel(kf.panel_id);
  const flags = conditionFlags(kf);
  return (
    <figure className="crop">
      <div className="crop-frame">
        <img
          src={routes.file(runId, kf.crop.path)}
          width={w}
          height={h}
          alt={`${panelName(kf.panel_id)} crop from the ${title} walk at ${formatClock(kf.pts)}; plates and faces are blurred`}
        />
        <svg viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
          {own.map((m, i) =>
            m.bbox_crop ? (
              <g key={m.mark_id} className="mark-box" style={{ animationDelay: `${i * 250}ms` }}>
                <rect x={m.bbox_crop[0]} y={m.bbox_crop[1]} width={m.bbox_crop[2] - m.bbox_crop[0]} height={m.bbox_crop[3] - m.bbox_crop[1]} stroke={color} />
                <text x={m.bbox_crop[0]} y={Math.max(26, m.bbox_crop[1] - 8)}>{`${m.type}, ${m.size_class}`}</text>
              </g>
            ) : null,
          )}
        </svg>
        {flags.length ? (
          <div className="crop-flags">
            {flags.map((f) => (
              <span key={f}>{f}</span>
            ))}
          </div>
        ) : null}
      </div>
      <figcaption>
        {title}: frame {kf.f_proxy} at {formatClock(kf.pts)}; {marksStatusText(kf, own.length)}
        {kf.window_subtracted ? '; window subtracted' : ''}
      </figcaption>
    </figure>
  );
}

export function CropView({ runId, role, panelId, keyframes, marks }: { runId: string; role: string; panelId: string | null; keyframes: Keyframe[]; marks: Mark[] }) {
  if (!panelId) return <div className="empty">Pick a panel in the list to see its crop.</div>;
  const kf = bestKeyframe(keyframes, role, panelId);
  if (!kf) {
    return (
      <div className="empty">
        {role}: no keyframe for {panelName(panelId)} yet. Keyframes come with panels.derived.
      </div>
    );
  }
  return <CropFigure runId={runId} kf={kf} marks={marks} title={role} />;
}
