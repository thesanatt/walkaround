import type { Mark, Verdict } from '../contracts/types.ts';
import { panelName, VERDICT_LABEL } from '../lib/panels.ts';
import { ruleTitle } from '../lib/rules.ts';

// Every verdict with its label rule first, then every G and K rule that touched it, and the events it came from.
export function VerdictList({ verdicts, marks }: { verdicts: Verdict[]; marks: Mark[] }) {
  if (!verdicts.length) return <div className="empty">Verdicts come with the match step. Code writes every label; a model answer can only lower one.</div>;
  const markById = new Map(marks.map((m) => [m.mark_id, m]));
  return (
    <div className="table-wrap">
      <table className="verdicts">
        <thead>
          <tr>
            <th>panel</th>
            <th>label</th>
            <th>marks</th>
            <th>distance</th>
            <th>rules</th>
            <th>events</th>
          </tr>
        </thead>
        <tbody>
          {verdicts.map((v) => (
            <tr key={v.verdict_id}>
              <td>{panelName(v.panel_id)}</td>
              <td>
                <span className={`label ${v.label}`}>{VERDICT_LABEL[v.label] ?? v.label}</span>
              </td>
              <td>
                {v.mark_ids
                  .map((id) => {
                    const m = markById.get(id);
                    return m ? `${m.role} ${m.type} ${m.size_class}` : id;
                  })
                  .join('; ')}
              </td>
              <td className="mono small">
                {v.distance_units === null ? '' : `${v.distance_units.toFixed(2)} units`}
                {v.distance_cm_est === null ? '' : `, about ${v.distance_cm_est.toFixed(0)} cm`}
              </td>
              <td className="mono small">
                {[v.rule_id, ...v.trace.rule_ids.filter((r) => r !== v.rule_id)].map((id) => (
                  <abbr key={id} title={ruleTitle(id)}>
                    {id}{' '}
                  </abbr>
                ))}
              </td>
              <td className="mono small">{v.trace.event_seqs.length ? v.trace.event_seqs.join(', ') : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
