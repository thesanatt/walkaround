import type { Mark, Narration, NarrationTurn } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { formatClockTenths } from '../lib/format.ts';
import { panelName } from '../lib/panels.ts';
import { AudioChip } from './AudioChip.tsx';

function pinText(t: NarrationTurn): string | null {
  if (!t.pinned_panel) return null;
  return t.pin_kind === 'about' ? `about the ${panelName(t.pinned_panel)}` : `said while the ${panelName(t.pinned_panel)} was on screen`;
}

// The mark a turn points to: one Muse Spark boxed because the renter named it (a narration_seed mark), or one the box
// pass had already found, which the words then link to.
export function seededMark(t: NarrationTurn, marks: Mark[]): Mark | undefined {
  if (!t.seed_mark_id) return undefined;
  return marks.find((m) => m.mark_id === t.seed_mark_id);
}

// "unsure" is no kind of mark on its own, so it reads "a mark of unsure type" (as claim.py words it).
function kindOf(m: Mark): string {
  return m.type === 'unsure' ? 'mark of unsure type' : m.type;
}

function seedText(m: Mark): string {
  return m.source === 'narration_seed'
    ? `seeded from your words: Muse Spark boxed a ${kindOf(m)} on the ${panelName(m.panel_id)} where you said`
    : `your words name the ${kindOf(m)} Muse Spark found on the ${panelName(m.panel_id)}`;
}

// Muse Voice turns with their start and end. A click on the words seeks the walk to the turn; the chip plays the
// renter's own words; a turn that seeded a mark says so.
export function NarrationChips({ narration, roles, onSeek, marks = [], missing = false }: { narration: Narration | null; roles: string[]; onSeek: (role: string, seconds: number) => void; marks?: Mark[]; missing?: boolean }) {
  const url = useFileUrl();
  if (!narration && missing) return <div className="empty">Narration was not turned into notes in this run. The walks are compared without it.</div>;
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
                {turns.map((t) => {
                  const seed = seededMark(t, marks);
                  const pin = pinText(t);
                  return (
                    <li key={t.turn_id} className="turn">
                      <button type="button" className="turn-words" onClick={() => onSeek(role, t.start_ms / 1000)} title="show this moment in the walk">
                        <span className="turn-time mono">
                          {formatClockTenths(t.start_ms / 1000)} to {formatClockTenths(t.end_ms / 1000)}
                        </span>
                        <span className="turn-text">"{t.text}"</span>
                      </button>
                      {t.chip_path ? <AudioChip src={url(t.chip_path)} text={t.text} seconds={(t.end_ms - t.start_ms) / 1000} /> : null}
                      <span className="turn-meta small">
                        {pin ?? 'not pinned to a panel'}
                        {seed ? <span className={`seed-badge ${seed.source === 'narration_seed' ? 'seeded' : 'linked'}`}>{seedText(seed)}</span> : null}
                      </span>
                    </li>
                  );
                })}
              </ul>
            ) : narration.per_role[role]?.transcript ? (
              <p className="turn-text">"{narration.per_role[role]?.transcript}"</p>
            ) : (
              <p className="small muted">no narration found in this walk</p>
            )}
          </div>
        );
      })}
    </div>
  );
}
