import type { PanelId, Verdict } from '../contracts/types.ts';
import { panelName } from '../lib/panels.ts';
import { BODY, SHAPES } from './CoverageCar.tsx';
import { CARD_WORD, cap } from './MarkCard.tsx';

// The result as a top-down car, front to the right: compared panels outlined, panels not compared hatched, and a pin
// per answer card (numbered) and per panel with spots left to the renter. The list under it says each pin in words,
// and every pin and row jumps to its card.
export interface Pin {
  panelId: PanelId;
  label: Verdict['label'];
  text: string; // the number on a card pin, the count on a needs-your-eye pin
  target: string; // the element id to jump to
  words: string;
}

export function pinsFor(clear: Verdict[], eye: Verdict[], eyeTarget: (panelId: string) => string): Pin[] {
  const pins: Pin[] = clear.map((v, i) => ({
    panelId: v.panel_id,
    label: v.label,
    text: String(i + 1),
    target: `card-${v.verdict_id}`,
    words: `${CARD_WORD[v.label] ?? v.label}: ${panelName(v.panel_id)}`,
  }));
  const perPanel = new Map<string, number>();
  for (const v of eye) perPanel.set(v.panel_id, (perPanel.get(v.panel_id) ?? 0) + 1);
  for (const [panelId, n] of perPanel)
    pins.push({ panelId: panelId as PanelId, label: 'needs_your_eye', text: String(n), target: eyeTarget(panelId), words: `Needs your eye: ${panelName(panelId)}${n > 1 ? `, ${n} spots` : ''}` });
  return pins;
}

export function ResultMap({ compared, notCompared, pins, onPin }: { compared: string[]; notCompared: string[]; pins: Pin[]; onPin: (pin: Pin) => void }) {
  const placed = new Map<string, number>();
  const dot = (p: Pin): [number, number] => {
    const shape = SHAPES.find((s) => s.id === p.panelId);
    const [x, y] = shape?.at ?? [280, 150];
    const k = placed.get(p.panelId) ?? 0;
    placed.set(p.panelId, k + 1);
    return [x + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 22, y];
  };
  const stateOf = (id: string) => (compared.includes(id) ? 'compared' : notCompared.includes(id) ? 'not-compared' : 'none');
  return (
    <figure className="result-map">
      <svg viewBox="40 24 500 252" role="img" aria-label={`Car seen from above, front to the right. ${compared.length} panels compared, ${notCompared.length} not compared.`}>
        <defs>
          <pattern id="rm-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" className="rm-hatch-line" strokeWidth="1.6" />
          </pattern>
        </defs>
        <path d={BODY} className="rm-body" />
        {SHAPES.map((s) => {
          const st = stateOf(s.id);
          return (
            <g key={s.id} className={`rm-panel ${st}`}>
              <title>{`${cap(panelName(s.id))}: ${st === 'compared' ? 'compared' : st === 'not-compared' ? 'not compared' : 'no result'}`}</title>
              {s.d.map((d, i) => (
                <path key={i} d={d} style={st === 'not-compared' ? { fill: 'url(#rm-hatch)' } : undefined} />
              ))}
            </g>
          );
        })}
        {SHAPES.filter((s) => stateOf(s.id) === 'compared').map((s) => (
          <g key={`o-${s.id}`} className="rm-outline">
            {s.d.map((d, i) => (
              <path key={i} d={d} />
            ))}
          </g>
        ))}
        <text x={512} y={154} className="rm-front">
          front
        </text>
        {pins.map((p) => {
          const [x, y] = dot(p);
          return (
            <g key={`${p.target}-${p.text}`} className={`rm-pin ${p.label}`} role="link" tabIndex={0} aria-label={p.words} onClick={() => onPin(p)} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onPin(p)}>
              <circle cx={x} cy={y} r={10} />
              <text x={x} y={y + 4} textAnchor="middle">
                {p.text}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption>
        <ul className="rm-legend">
          {pins.map((p) => (
            <li key={`l-${p.target}-${p.text}`}>
              <button type="button" className="link-button" onClick={() => onPin(p)}>
                <span className={`pin-badge ${p.label}`} aria-hidden="true">
                  {p.text}
                </span>
                {cap(p.words)}
              </button>
            </li>
          ))}
        </ul>
        <p className="rm-note">
          SAM 3.1 tracked these panels through both walks.{' '}
          <span className="rm-key">
            <svg viewBox="0 0 14 14" width="14" height="14" aria-hidden="true">
              <rect x="1.5" y="1.5" width="11" height="11" className="rm-swatch compared" />
            </svg>
            compared
          </span>{' '}
          <span className="rm-key">
            <svg viewBox="0 0 14 14" width="14" height="14" aria-hidden="true">
              <rect x="1" y="1" width="12" height="12" className="rm-swatch" style={{ fill: 'url(#rm-hatch)' }} />
            </svg>
            not compared
          </span>
        </p>
      </figcaption>
    </figure>
  );
}
