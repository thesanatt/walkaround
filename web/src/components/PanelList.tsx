import type { Coverage, Panel, PanelTrack } from '../contracts/types.ts';
import { COVERAGE_LABEL, PANELS, type PanelInfo } from '../lib/panels.ts';
import type { LaneState } from '../state/types.ts';

export interface PanelListProps {
  role: string;
  panels: Panel[];
  coverage: Coverage | null;
  tracks: PanelTrack[];
  lanes: Record<string, LaneState>;
  selected: string | null;
  onSelect: (panelId: string) => void;
}

function laneText(lane: LaneState | undefined): string {
  if (!lane) return 'waiting';
  switch (lane.status) {
    case 'requested':
      return 'SAM 3.1 requested';
    case 'streaming':
      return `SAM 3.1 streaming, frame ${lane.frames}`;
    case 'completed':
      return `lane done, ${lane.frames} frames, ${lane.ids.length} ${lane.ids.length === 1 ? 'id' : 'ids'}`;
    case 'zero_matches':
      return 'not found';
    case 'incomplete':
      return 'track incomplete';
    case 'failed':
      return 'not run';
  }
}

export function fragmentsLine(info: PanelInfo, tracks: PanelTrack[]): string | null {
  const own = tracks.filter((t) => t.panel_id === info.id);
  if (!own.length) return null;
  const fragments = own.reduce((n, t) => n + t.fragments.length, 0);
  if (fragments <= own.length) return null;
  const noun = info.noun.replace('car ', '');
  return `same ${noun}, fragments ${fragments} to ${own.length}`;
}

// Fills as lanes finish: first the lane status by noun, then the coverage state once the panels derive.
export function PanelList({ role, panels, coverage, tracks, lanes, selected, onSelect }: PanelListProps) {
  const byId = new Map(panels.map((p) => [p.panel_id, p]));
  const roleTracks = tracks.filter((t) => t.role === role);
  const families: [string, PanelInfo['family']][] = [
    ['body panels', 'body'],
    ['wheels', 'wheel'],
    ['lights and glass', 'light_glass'],
  ];
  return (
    <div className="panel-list">
      {families.map(([title, family]) => (
        <section key={family}>
          <h4>{title}</h4>
          <ul>
            {PANELS.filter((p) => p.family === family).map((info) => {
              const panel = byId.get(info.id);
              const cov = coverage?.per_role[role]?.[info.id];
              const state = cov?.state ?? panel?.per_role[role]?.state ?? null;
              const seconds = cov?.seconds ?? panel?.per_role[role]?.seconds_visible ?? null;
              const frag = fragmentsLine(info, roleTracks);
              const isSelected = selected === info.id;
              return (
                <li key={info.id} className={`panel-row ${state ?? 'pending'} ${isSelected ? 'selected' : ''}`}>
                  <button type="button" onClick={() => onSelect(info.id)}>
                    <span className="swatch" style={{ background: info.color }} aria-hidden="true" />
                    <span className="panel-name">{info.name}</span>
                    <span className="panel-state">
                      {state ? (
                        <>
                          {COVERAGE_LABEL[state] ?? state}
                          {seconds !== null && state !== 'not_seen' && state !== 'not_run' ? ` ${seconds.toFixed(1)} s` : ''}
                          {cov?.state_basis === 'area_only' ? ' (area basis)' : ''}
                          {panel?.side_uncertain ? ' side uncertain' : ''}
                        </>
                      ) : (
                        <span className="muted">{laneText(lanes[info.noun])}</span>
                      )}
                    </span>
                    {frag ? <span className="panel-frag mono small">{frag}</span> : null}
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
}
