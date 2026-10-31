import { useId } from 'react';

import type { Keyframe, Mark } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { applyCut, applyCutBox, matchCutTransform } from '../lib/geometry.ts';
import { hexToUnit, panelName } from '../lib/panels.ts';

// pickup: the pickup crop with its mark boxed. return: the cut, an 8-frame dissolve to the return crop placed so its
// panel sits where the pickup panel sat, the pickup box left as a dashed ghost. join: one line from the ghost to the
// return mark.
export type CutPhase = 'pickup' | 'return' | 'join';

export interface MatchCutProps {
  kfA: Keyframe;
  kfB: Keyframe;
  markA: Mark | undefined;
  markB: Mark | undefined;
  color: string;
  phase: CutPhase;
  labelA?: string | null; // written over the pickup box before the cut
  still?: boolean; // a frozen capture (?t=): no transitions
}

export function MatchCut({ kfA, kfB, markA, markB, color, phase, labelA = null, still = false }: MatchCutProps) {
  const url = useFileUrl();
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  if (!kfA.crop || !kfB.crop) return <div className="empty">The match cut needs both keyframe crops of the {panelName(kfA.panel_id)}.</div>;
  const wA = kfA.crop.w;
  const hA = kfA.crop.h;
  const t = matchCutTransform(kfA, kfB);
  const z = { stroke: wA / 320, dot: wA / 260, join: wA / 330, title: Math.round(wA / 48), label: Math.round(wA / 40) };
  const cut = phase !== 'pickup';
  const bA = markA?.bbox_crop ?? null;
  const pA = markA?.point_crop ?? null;
  const bB = markB?.bbox_crop ? applyCutBox(markB.bbox_crop, t) : null;
  const pB = markB?.point_crop ? applyCut(markB.point_crop, t) : null;
  const [r, g, b] = hexToUnit(color);
  const label = cut ? 'return' : 'pickup';

  return (
    <div className={`match-cut ${still ? 'still' : ''}`} data-phase={phase}>
      <svg viewBox={`0 0 ${wA} ${hA}`} role="img" aria-label={`${panelName(kfA.panel_id)}: ${cut ? 'the return frame, placed where the pickup frame was' : 'the pickup frame'}`}>
        <defs>
          <clipPath id={`cut-clip-${uid}`}>
            <rect width={wA} height={hA} />
          </clipPath>
          <filter id={`cut-tint-${uid}`} colorInterpolationFilters="sRGB">
            <feColorMatrix type="matrix" values={`0 0 0 0 ${r} 0 0 0 0 ${g} 0 0 0 0 ${b} 0.333 0.333 0.333 0 0`} />
          </filter>
        </defs>
        <g clipPath={`url(#cut-clip-${uid})`}>
          <rect width={wA} height={hA} className="pane-empty" />
          <g transform={`translate(${t.tx} ${t.ty}) scale(${t.s})`}>
            <image href={url(kfB.crop.path)} width={kfB.crop.w} height={kfB.crop.h} />
            {kfB.mask_png ? <image className="cut-mask" href={url(kfB.mask_png)} width={kfB.crop.w} height={kfB.crop.h} filter={`url(#cut-tint-${uid})`} opacity={cut ? 0.22 : 0} /> : null}
          </g>
          <image className="cut-pickup" href={url(kfA.crop.path)} width={wA} height={hA} opacity={cut ? 0 : 1} />
          {bA ? (
            <rect
              className={cut ? 'cut-ghost' : 'cut-box'}
              x={bA[0]}
              y={bA[1]}
              width={bA[2] - bA[0]}
              height={bA[3] - bA[1]}
              fill="none"
              stroke={cut ? '#f4f1ea' : color}
              strokeWidth={z.stroke}
              strokeDasharray={cut ? `${z.stroke * 3} ${z.stroke * 2}` : undefined}
            />
          ) : null}
          {!cut && bA && labelA ? (
            <text className="pane-label" x={bA[0]} y={Math.max(z.label, bA[1] - z.label / 2.5)} style={{ fontSize: z.label, strokeWidth: z.label / 5 }}>
              {labelA}
            </text>
          ) : null}
          {cut && bB ? <rect className="cut-box" x={bB[0]} y={bB[1]} width={bB[2] - bB[0]} height={bB[3] - bB[1]} fill="none" stroke={color} strokeWidth={z.stroke} /> : null}
          {phase === 'join' && pA && pB ? (
            <>
              <line className="join-line" x1={pA[0]} y1={pA[1]} x2={pB[0]} y2={pB[1]} stroke={color} strokeWidth={z.join} pathLength={1} />
              <circle cx={pA[0]} cy={pA[1]} r={z.dot} fill="#f4f1ea" stroke="#0b0d0c" strokeWidth={z.stroke / 2} />
              <circle cx={pB[0]} cy={pB[1]} r={z.dot} fill={color} stroke="#0b0d0c" strokeWidth={z.stroke / 2} />
            </>
          ) : null}
        </g>
        <text className="pane-title" x={z.title * 0.7} y={z.title * 1.5} style={{ fontSize: z.title, strokeWidth: z.title / 5 }}>
          {label}
        </text>
      </svg>
    </div>
  );
}
