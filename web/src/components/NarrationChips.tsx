import type { Narration } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { formatClockTenths } from '../lib/format.ts';
import { panelName } from '../lib/panels.ts';

// Muse Voice turns with their start and end; a click seeks the walk to the turn, the chip plays the renter's words.
export function NarrationChips({ narration, roles, onSeek }: { narration: Narration | null; roles: string[]; onSeek: (role: string, seconds: number) => void }) {
  const url = useFileUrl();
  if (!narration) return <div className="empty">Narration comes after the voice step. A walk filmed without talking has no chips.</div>;
  return (
    <div className="narration">
      {!narration.mode_applied ? <p className="small muted">Muse Voice returned the transcript without turns, so it is not pinned to panels.</p> : null}
      {roles.map((role) => {
        const turns = narration.per_role[role]?.turns ?? [];
        return (
          <div key={role} className="narration-role">
            <h4>{role} walk</h4>
            {turns.length ? (
              <ul className="turns">
                {turns.map((t) => (
                  <li key={t.turn_id}>
                    <button type="button" className="chip" onClick={() => onSeek(role, t.start_ms / 1000)}>
                      <span className="mono">
                        {formatClockTenths(t.start_ms / 1000)} to {formatClockTenths(t.end_ms / 1000)}
                      </span>{' '}
                      "{t.text}"{t.pinned_panel ? `, ${panelName(t.pinned_panel)}` : ''}
                    </button>
                    {t.chip_path ? <audio controls preload="none" src={url(t.chip_path)} aria-label={`the renter saying "${t.text}"`} /> : null}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="small muted">no narration found in this walk</p>
            )}
          </div>
        );
      })}
    </div>
  );
}
