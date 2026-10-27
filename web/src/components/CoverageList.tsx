import type { Coverage } from '../contracts/types.ts';
import { COVERAGE_LABEL, panelName } from '../lib/panels.ts';

// The list form of the coverage car (the SVG car is a backlog item; the list keeps the same states).
export function CoverageList({ coverage, roles }: { coverage: Coverage | null; roles: string[] }) {
  if (!coverage) return <div className="empty">Coverage is computed once the panel lanes finish.</div>;
  return (
    <div className="coverage">
      {roles.map((role) => {
        const per = coverage.per_role[role];
        if (!per) return null;
        const groups = new Map<string, string[]>();
        for (const [panelId, cov] of Object.entries(per)) {
          if (!cov) continue;
          const list = groups.get(cov.state) ?? [];
          list.push(`${panelName(panelId)}${cov.seconds > 0 ? ` ${cov.seconds.toFixed(1)} s` : ''}`);
          groups.set(cov.state, list);
        }
        return (
          <div key={role} className="coverage-role">
            <h4>{role}</h4>
            {['filmed', 'glimpsed', 'not_filmed', 'not_run', 'not_seen'].map((state) => {
              const items = groups.get(state);
              if (!items?.length) return null;
              return (
                <p key={state} className={`coverage-group ${state}`}>
                  <span className="coverage-state">{COVERAGE_LABEL[state]}</span> {items.join(', ')}
                </p>
              );
            })}
          </div>
        );
      })}
      {coverage.missing.length ? (
        <div className="topup">
          <p>
            <strong>missing:</strong> {coverage.missing.map(panelName).join(', ')}
          </p>
          {coverage.topup_text ? <p>{coverage.topup_text}</p> : null}
          {coverage.odometer.asked && !coverage.odometer.filmed ? <p>odometer: not filmed, add 3 seconds?</p> : null}
          {coverage.rerun_seconds !== null ? <p className="mono small">top-up lanes re-ran in {coverage.rerun_seconds.toFixed(0)} s</p> : null}
        </div>
      ) : (
        <p className="muted">nothing missing in the list</p>
      )}
    </div>
  );
}
