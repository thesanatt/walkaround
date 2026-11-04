import type { MathRow, MathView } from '../lib/math.ts';
import { tokenize } from '../lib/pycode.ts';

// The steps of one finding's math: a short title, one line of why, and the real lines of code with the run's values
// on the right, in VS Code's Dark+ colours (lib/pycode.ts).
function Code({ rows }: { rows: MathRow[] }) {
  if (!rows.length) return null;
  const file = rows[0]!.line.file;
  const first = rows[0]!.line.n;
  const last = rows[rows.length - 1]!.line.n;
  const indent = Math.min(...rows.map((r) => r.line.raw.length - r.line.raw.trimStart().length));
  const slash = file.lastIndexOf('/');
  return (
    <div className="code">
      <div className="code-head">
        <span className="code-dir">{file.slice(0, slash + 1)}</span>
        {file.slice(slash + 1)}
        <span className="code-range">{first === last ? first : `${first}-${last}`}</span>
      </div>
      {rows.map((r, i) => {
        const gap = i > 0 && r.line.n - rows[i - 1]!.line.n > 1;
        return (
          <div key={`${r.line.file}:${r.line.n}`} className={`code-row${gap ? ' gap' : ''}`}>
            <span className="code-n">{r.line.n}</span>
            <code className="code-line">
              {tokenize(r.line.raw.slice(indent)).map(([text, k], j) => (
                <span key={j} className={`t-${k}`}>
                  {text}
                </span>
              ))}
            </code>
            {r.value !== null ? (
              <span className="code-val">
                {tokenize(r.value).map(([text, k], j) => (
                  <span key={j} className={`v-${k}`}>
                    {text}
                  </span>
                ))}
              </span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

export function MathPanel({ math, id }: { math: MathView; id: string }) {
  const d = math.decision;
  return (
    <section className="math" id={id} aria-label="the math">
      {d ? (
        <div className="math-decision">
          <p className="math-big">
            <span>{d.left}</span> <span className="math-op">{d.op}</span> <span>{d.right}</span>
          </p>
          <p className="math-names">
            <span>{d.leftName}</span>
            <span>{d.rightName}</span>
          </p>
        </div>
      ) : null}
      <ol className="math-steps">
        {math.steps.map((s) => (
          <li key={s.title}>
            <h3>{s.title}</h3>
            {s.sub ? <p className="math-sub">{s.sub}</p> : null}
            <Code rows={s.rows} />
          </li>
        ))}
      </ol>
    </section>
  );
}
