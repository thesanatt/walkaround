import type { Coverage } from '../contracts/types.ts';
import { COVERAGE_LABEL, COVERAGE_ORDER, panelName } from '../lib/panels.ts';

// The list form of the coverage car: every slot in one of five states per walk, then what is missing.
export function CoverageList({ coverage, roles, showMissing = true }: { coverage: Coverage | null; roles: string[]; showMissing?: boolean }) {
  if (!coverage) return <div className="empty">Coverage is computed once the panel lanes finish.</div>;
  return (
    <div className="coverage">
      {roles.map((role) => {
        const per = coverage.per_role[role];
        if (!per) return null;
        const groups = new Map<string, string[]>();
        for (const [panelId, cov] of Object.entries(per)) {
          if (!cov) continue;
          const items = groups.get(cov.state) ?? [];
          items.push(`${panelName(panelId)}${cov.seconds > 0 && (cov.state === 'filmed' || cov.state === 'glimpsed') ? ` ${cov.seconds.toFixed(1)} s` : ''}`);
          groups.set(cov.state, items);
        }
        return (
          <div key={role} className="coverage-role">
            <h4>{role} walk</h4>
            {COVERAGE_ORDER.map((state) => {
              const items = groups.get(state);
              if (!items?.length) return null;
              return (
                <p key={state} className={`coverage-group ${state}`}>
                  <span className="coverage-state">{COVERAGE_LABEL[state]}</span>
                  <span>{items.join(', ')}</span>
                </p>
              );
            })}
          </div>
        );
      })}
      {!showMissing ? null : coverage.missing.length ? (
        <div className="topup">
          <p>
            <strong>missing:</strong> {coverage.missing.map(panelName).join(', ')}
          </p>
          {coverage.topup_text ? <p>{coverage.topup_text}</p> : null}
          {coverage.odometer.asked ? <p>odometer: {coverage.odometer.filmed ? 'filmed' : 'not filmed, add 3 seconds?'}</p> : null}
          {coverage.rerun_seconds !== null ? <p className="mono small">the top-up lanes ran in {coverage.rerun_seconds.toFixed(0)} s</p> : null}
        </div>
      ) : (
        <p className="muted small">Nothing missing: every slot was filmed or glimpsed in at least one walk.</p>
      )}
    </div>
  );
}
