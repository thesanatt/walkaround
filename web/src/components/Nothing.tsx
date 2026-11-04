import { exampleUrl, useExample } from '../lib/example.ts';
import { Link } from '../lib/router.tsx';
import { Bar } from './Bar.tsx';

// Nothing at this address: an unknown page, or a run the server would not give. line: what went wrong, plainly.
export function Nothing({ line }: { line: string }) {
  const example = useExample();
  return (
    <div className="nothing">
      <Bar />
      <h1>No check at this address</h1>
      <p className="nothing-line">{line}</p>
      <p className="nothing-links">
        <Link to="/">Start over</Link>
        {example ? <Link to={exampleUrl(example)}>See an example</Link> : null}
      </p>
    </div>
  );
}
