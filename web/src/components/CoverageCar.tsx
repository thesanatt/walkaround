import { useId } from 'react';

import type { Coverage, CoverageState, Panel, PanelId } from '../contracts/types.ts';
import { COVERAGE_LABEL, panelName } from '../lib/panels.ts';

// A top-down car, front to the right, so its left side is the top edge. Every one of the 22 slots is a shape named by
// its panel id; wheels and mirrors sit outside the body as tabs. The fill is the coverage state of one walk.
export interface Shape {
  id: PanelId;
  d: string[];
  at: [number, number]; // where a callout points
}

export const SHAPES: Shape[] = [
  { id: 'rear_bumper', d: ['M84 74 L74 74 Q60 78 60 104 L60 196 Q60 222 74 226 L84 226 Z'], at: [72, 150] },
  { id: 'tail_lights', d: ['M84 74 L98 72 L98 100 L84 100 Z', 'M84 200 L98 200 L98 228 L84 226 Z'], at: [91, 86] },
  { id: 'trunk_lid', d: ['M84 100 L164 100 L164 200 L84 200 Z'], at: [124, 150] },
  { id: 'rear_window', d: ['M164 100 L206 105 L206 195 L164 200 Z'], at: [185, 150] },
  { id: 'roof', d: ['M206 105 L334 105 L334 195 L206 195 Z'], at: [270, 150] },
  { id: 'windshield', d: ['M334 105 L386 100 L386 200 L334 195 Z'], at: [360, 150] },
  { id: 'hood', d: ['M386 100 L478 100 L478 200 L386 200 Z'], at: [432, 150] },
  { id: 'front_bumper', d: ['M478 74 L486 74 Q500 78 500 104 L500 196 Q500 222 486 226 L478 226 Z'], at: [489, 150] },
  { id: 'headlights', d: ['M462 72 L478 74 L478 100 L462 100 Z', 'M462 200 L478 200 L478 226 L462 228 Z'], at: [470, 86] },
  { id: 'left_rear_fender', d: ['M98 72 L176 70 L176 100 L98 100 Z'], at: [137, 85] },
  { id: 'left_rear_door', d: ['M176 70 L270 70 L270 100 L176 100 Z'], at: [223, 85] },
  { id: 'left_front_door', d: ['M270 70 L368 70 L368 100 L270 100 Z'], at: [319, 85] },
  { id: 'left_front_fender', d: ['M368 70 L462 72 L462 100 L368 100 Z'], at: [415, 85] },
  { id: 'right_rear_fender', d: ['M98 200 L176 200 L176 230 L98 228 Z'], at: [137, 215] },
  { id: 'right_rear_door', d: ['M176 200 L270 200 L270 230 L176 230 Z'], at: [223, 215] },
  { id: 'right_front_door', d: ['M270 200 L368 200 L368 230 L270 230 Z'], at: [319, 215] },
  { id: 'right_front_fender', d: ['M368 200 L462 200 L462 228 L368 230 Z'], at: [415, 215] },
  { id: 'left_rear_wheel', d: ['M114 52 L162 52 Q166 52 166 56 L166 70 L110 70 L110 56 Q110 52 114 52 Z'], at: [138, 61] },
  { id: 'left_front_wheel', d: ['M396 52 L444 52 Q448 52 448 56 L448 70 L392 70 L392 56 Q392 52 396 52 Z'], at: [420, 61] },
  { id: 'right_rear_wheel', d: ['M110 230 L166 230 L166 244 Q166 248 162 248 L114 248 Q110 248 110 244 Z'], at: [138, 239] },
  { id: 'right_front_wheel', d: ['M392 230 L448 230 L448 244 Q448 248 444 248 L396 248 Q392 248 392 244 Z'], at: [420, 239] },
  { id: 'mirrors', d: ['M346 70 L354 56 L368 56 L368 70 Z', 'M346 230 L354 244 L368 244 L368 230 Z'], at: [357, 61] },
];

export const BODY = 'M84 70 L462 70 Q500 72 500 110 L500 190 Q500 228 462 230 L84 230 Q60 228 60 196 L60 104 Q60 72 84 70 Z';

const RANK: Record<CoverageState, number> = { filmed: 4, glimpsed: 3, not_filmed: 2, not_run: 1, not_seen: 0 };

// One walk's state per slot from coverage.json, else from panels.json. With topups, a slot filmed in a top-up clip
// counts for the walk it topped up (the best state wins).
export function slotStates(coverage: Coverage | null, panels: Panel[], role: string, withTopups = false): Partial<Record<PanelId, CoverageState>> {
  const out: Partial<Record<PanelId, CoverageState>> = {};
  for (const p of panels) {
    const s = p.per_role[role]?.state;
    if (s) out[p.panel_id] = s;
  }
  const roles = [role, ...(withTopups && coverage ? Object.keys(coverage.per_role).filter((r) => r.startsWith('topup_')) : [])];
  for (const r of roles) {
    for (const [id, cov] of Object.entries(coverage?.per_role[r] ?? {})) {
      if (!cov) continue;
      const prev = out[id as PanelId];
      if (r === role || !prev || RANK[cov.state] > RANK[prev]) out[id as PanelId] = cov.state;
    }
  }
  return out;
}

export interface WalkGap {
  id: PanelId;
  state: CoverageState;
  seconds: number;
}

// One walk's slots short of filmed, wheels first, then bumpers, then the rest in car order (coverage's own order).
export function walkGaps(coverage: Coverage | null, panels: Panel[], role: string): WalkGap[] {
  const states = slotStates(coverage, panels, role);
  const group = (id: string) => (id.endsWith('_wheel') ? 0 : id.endsWith('_bumper') ? 1 : 2);
  return SHAPES.map((s) => s.id)
    .filter((id) => states[id] !== undefined && states[id] !== 'filmed')
    .sort((a, b) => group(a) - group(b))
    .map((id) => ({ id, state: states[id]!, seconds: coverage?.per_role[role]?.[id]?.seconds ?? 0 }));
}

// The order the slots light up in: by when each first appears in the walk, the slots never filmed last.
export function revealOrder(panels: Panel[], role: string): PanelId[] {
  const start = (p: Panel) => p.per_role[role]?.windows[0]?.[0] ?? Number.POSITIVE_INFINITY;
  return [...panels].sort((a, b) => start(a) - start(b) || SHAPES.findIndex((s) => s.id === a.panel_id) - SHAPES.findIndex((s) => s.id === b.panel_id)).map((p) => p.panel_id);
}

export interface CoverageCarProps {
  coverage: Coverage | null;
  panels: Panel[];
  role: string;
  withTopups?: boolean;
  revealed?: ReadonlySet<string> | null; // slots not in it draw as not yet known (the kiosk's fill)
  callouts?: PanelId[]; // slots that get their name and state written beside the car
  selected?: string | null;
  onSelect?: (panelId: string) => void;
  title?: string;
}

const LEGEND: CoverageState[] = ['filmed', 'glimpsed', 'not_filmed', 'not_run', 'not_seen'];

// Outlined slots draw last, so a neighbour's seam never covers a not-filmed or selected outline.
function drawOrder(shapes: Shape[], stateOf: (id: PanelId) => CoverageState | 'pending', selected: string | null): Shape[] {
  const rank = (s: Shape) => (s.id === selected ? 2 : stateOf(s.id) === 'not_filmed' ? 1 : 0);
  return [...shapes].sort((a, b) => rank(a) - rank(b));
}

export function CoverageCar({ coverage, panels, role, withTopups = false, revealed = null, callouts = [], selected = null, onSelect, title }: CoverageCarProps) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const states = slotStates(coverage, panels, role, withTopups);
  const known = Object.keys(states).length > 0;
  const stateOf = (id: PanelId): CoverageState | 'pending' => (revealed && !revealed.has(id) ? 'pending' : (states[id] ?? 'pending'));
  const counts = LEGEND.map((s) => [s, SHAPES.filter((x) => stateOf(x.id) === s).length] as const).filter(([, n]) => n > 0);
  const summary = counts.map(([s, n]) => `${n} ${COVERAGE_LABEL[s]}`).join(', ');
  return (
    <figure className="coverage-car">
      <svg viewBox="0 0 560 300" role="img" aria-label={`${title ?? `${role} walk`} coverage: ${known ? summary : 'not computed yet'}`}>
        <defs>
          <pattern id={`hatch-${uid}`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" className="cc-hatch-bg" />
            <line x1="0" y1="0" x2="0" y2="6" className="cc-hatch-line" strokeWidth="2.4" />
          </pattern>
          <pattern id={`cross-${uid}`} width="7" height="7" patternUnits="userSpaceOnUse">
            <rect width="7" height="7" className="cc-cross-bg" />
            <path d="M0 0 L7 7 M7 0 L0 7" className="cc-cross-line" strokeWidth="0.9" />
          </pattern>
        </defs>
        <path d={BODY} className="cc-body" />
        {drawOrder(SHAPES, stateOf, selected).map((s) => {
          const st = stateOf(s.id);
          const fill = st === 'glimpsed' ? `url(#hatch-${uid})` : st === 'not_run' ? `url(#cross-${uid})` : undefined;
          return (
            <g
              key={s.id}
              className={`cc-panel ${st} ${selected === s.id ? 'selected' : ''} ${onSelect ? 'clickable' : ''}`}
              onClick={onSelect ? () => onSelect(s.id) : undefined}
            >
              <title>{`${panelName(s.id)}: ${st === 'pending' ? 'not known yet' : COVERAGE_LABEL[st]}`}</title>
              {s.d.map((d, i) => (
                <path key={i} d={d} style={fill ? { fill } : undefined} />
              ))}
            </g>
          );
        })}
        <path d={BODY} className="cc-outline" />
        <g className="cc-labels" aria-hidden="true">
          <text x={280} y={34} textAnchor="middle">
            left side
          </text>
          <text x={280} y={278} textAnchor="middle">
            right side
          </text>
          <text x={512} y={154} className="cc-front">
            front
          </text>
          <path d="M512 160 L540 160 M534 155 L540 160 L534 165" className="cc-arrow" />
        </g>
        {callouts
          .map((id) => SHAPES.find((s) => s.id === id))
          .filter((s): s is Shape => Boolean(s))
          .map((s) => {
            const st = stateOf(s.id);
            if (st === 'pending') return null;
            // Near an end of the car the text runs inward, so it never leaves the drawing.
            const [x, y] = s.at;
            const anchor = x < 140 ? 'start' : x > 420 ? 'end' : 'middle';
            const tx = anchor === 'start' ? x - 4 : anchor === 'end' ? x + 4 : x;
            return (
              <g key={`co-${s.id}`} className="cc-callout">
                <circle cx={x} cy={y} r={3} />
                <text x={tx} y={y + 14} textAnchor={anchor}>
                  {`${panelName(s.id)}: ${COVERAGE_LABEL[st]}`}
                </text>
              </g>
            );
          })}
      </svg>
      <figcaption>
        <span className="cc-title">{title ?? `${role} walk`}</span>
        <span className="cc-legend">
          {LEGEND.map((s) => (
            <span key={s} className={`cc-key ${s}`}>
              <svg viewBox="0 0 14 14" width="14" height="14" aria-hidden="true">
                <rect x="1" y="1" width="12" height="12" rx="2" className={`cc-swatch ${s}`} style={s === 'glimpsed' ? { fill: `url(#hatch-${uid})` } : s === 'not_run' ? { fill: `url(#cross-${uid})` } : undefined} />
              </svg>
              {COVERAGE_LABEL[s]}
            </span>
          ))}
        </span>
      </figcaption>
    </figure>
  );
}
