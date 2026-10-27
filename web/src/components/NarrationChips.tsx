import { routes } from '../api/client.ts';
import type { Narration } from '../contracts/types.ts';
import { panelName } from '../lib/panels.ts';

function ms(value: number): string {
  const s = value / 1000;
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')}`;
}

// Muse Voice turns with startMs and endMs; a click seeks the walk to the turn. No turns means no chips.
export function NarrationChips({ runId, narration, roles, onSeek }: { runId: string; narration: Narration | null; roles: string[]; onSeek: (role: string, seconds: number) => void }) {
  if (!narration) return <div className="empty">Narration comes with narration.derived. Without a transcript there are no chips.</div>;
  return (
    <div className="narration">
      {!narration.mode_applied ? <p className="small muted">The transcript came back as one turn, so it is shown without pins.</p> : null}
      {roles.map((role) => {
        const turns = narration.per_role[role]?.turns ?? [];
        return (
          <div key={role} className="narration-role">
            <h4>{role}</h4>
            {turns.length ? (
              <div className="chips">
                {turns.map((t) => (
                  <span key={t.turn_id} className="row">
                    <button type="button" className="chip" onClick={() => onSeek(role, t.start_ms / 1000)}>
                      <span className="mono">
                        {ms(t.start_ms)} to {ms(t.end_ms)}
                      </span>
                      : {t.text}
                      {t.pinned_panel ? ` (${panelName(t.pinned_panel)})` : ''}
                    </button>
                    {t.chip_path ? <audio controls preload="none" src={routes.file(runId, t.chip_path)} aria-label={`narration chip ${t.turn_id}`} /> : null}
                  </span>
                ))}
              </div>
            ) : (
              <p className="small muted">no narration found in this walk</p>
            )}
          </div>
        );
      })}
    </div>
  );
}
