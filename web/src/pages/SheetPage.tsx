import { useEffect, useRef, useState } from 'react';

import { api, routes } from '../api/client.ts';
import { Link } from '../lib/router.tsx';

const SCROLL_PX_PER_S = 60;

// The HTML sheet in a frame, with the PDF and bundle links. The kiosk scrolls it at about 60 px a second.
export function SheetPage({ runId, query, kiosk = false }: { runId: string; query: URLSearchParams; kiosk?: boolean }) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [html, setHtml] = useState<'checking' | 'ready' | 'missing'>('checking');
  const [pdf, setPdf] = useState<'checking' | 'ready' | 'retry'>('checking');
  const scroll = kiosk || query.get('scroll') === '1';

  useEffect(() => {
    let live = true;
    void api.exists(routes.sheetHtml(runId, kiosk ? 'kiosk' : 'owner')).then((ok) => live && setHtml(ok ? 'ready' : 'missing'));
    if (!kiosk) void api.exists(routes.sheetPdf(runId)).then((ok) => live && setPdf(ok ? 'ready' : 'retry'));
    return () => {
      live = false;
    };
  }, [runId, kiosk]);

  useEffect(() => {
    const el = frame.current;
    if (!scroll || !el || html !== 'ready') return;
    let raf = 0;
    let start = 0;
    const step = (t: number) => {
      start ||= t;
      try {
        el.contentWindow?.scrollTo(0, ((t - start) / 1000) * SCROLL_PX_PER_S);
      } catch {
        // the sheet is same-origin in production; nothing to scroll otherwise
      }
      raf = requestAnimationFrame(step);
    };
    const begin = () => {
      cancelAnimationFrame(raf);
      start = 0;
      raf = requestAnimationFrame(step);
    };
    el.addEventListener('load', begin);
    return () => {
      el.removeEventListener('load', begin);
      cancelAnimationFrame(raf);
    };
  }, [scroll, html, runId]);

  return (
    <div className="sheet-page">
      {!kiosk ? (
        <div className="run-head">
          <h1>
            sheet for run <span className="mono">{runId}</span>
          </h1>
          <div className="controls">
            <Link to={`/run/${encodeURIComponent(runId)}`}>back to the run</Link>
            {pdf === 'ready' ? <a href={routes.sheetPdf(runId)}>evidence.pdf</a> : <span className="muted">{pdf === 'checking' ? 'checking the PDF' : 'PDF retry'}</span>}
            <a href={routes.bundleZip(runId)}>bundle.zip</a>
            <a href={routes.sheetJson(runId)}>sheet.json</a>
          </div>
        </div>
      ) : null}
      {html === 'missing' ? (
        <div className="empty">The sheet is not written yet. It comes at the end of the run; a partial run still gets one.</div>
      ) : html === 'ready' ? (
        <iframe ref={frame} className="sheet-frame" src={routes.sheetHtml(runId, kiosk ? 'kiosk' : 'owner')} title={`evidence sheet for run ${runId}`} />
      ) : (
        <p className="muted">checking for the sheet</p>
      )}
    </div>
  );
}
