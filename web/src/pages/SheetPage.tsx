import { useEffect, useRef, useState } from 'react';

import { api, routes } from '../api/client.ts';
import { Link } from '../lib/router.tsx';

// The HTML sheet in an iframe; PDF and bundle downloads; ?scroll=1 scrolls at about 60 px per second for the kiosk take.
export function SheetPage({ runId, query, kiosk = false }: { runId: string; query: URLSearchParams; kiosk?: boolean }) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [pdf, setPdf] = useState<'checking' | 'ready' | 'retry'>('checking');
  const [missing, setMissing] = useState(false);
  const scroll = kiosk || query.get('scroll') === '1';

  useEffect(() => {
    let stopped = false;
    api.pdfReady(runId).then((ok) => {
      if (!stopped) setPdf(ok ? 'ready' : 'retry');
    });
    return () => {
      stopped = true;
    };
  }, [runId]);

  useEffect(() => {
    fetch(routes.sheetHtml(runId), { method: 'HEAD' })
      .then((res) => setMissing(!res.ok))
      .catch(() => setMissing(true));
  }, [runId]);

  useEffect(() => {
    if (!scroll) return;
    const el = frame.current;
    if (!el) return;
    let handle: ReturnType<typeof setInterval> | null = null;
    const start = () => {
      handle = setInterval(() => {
        try {
          el.contentWindow?.scrollBy(0, 1);
        } catch {
          // a cross-origin frame cannot be scrolled; same-origin in production
        }
      }, 1000 / 60);
    };
    el.addEventListener('load', start);
    return () => {
      el.removeEventListener('load', start);
      if (handle) clearInterval(handle);
    };
  }, [scroll, runId]);

  return (
    <div className="sheet-page">
      {!kiosk ? (
        <div className="run-head">
          <h1>
            sheet for run <span className="mono">{runId}</span>
          </h1>
          <div className="controls">
            <Link to={`/run/${encodeURIComponent(runId)}`}>back to the run</Link>
            <a href={routes.sheetJson(runId)}>sheet.json</a>
            {pdf === 'ready' ? <a href={routes.sheetPdf(runId)}>evidence.pdf</a> : <span className="muted">{pdf === 'checking' ? 'checking the PDF' : 'PDF retry'}</span>}
            <a href={routes.bundleZip(runId)}>bundle.zip</a>
          </div>
        </div>
      ) : null}
      {missing ? (
        <div className="empty">The sheet is not written yet. It comes at the end of the run; a partial run still gets one.</div>
      ) : (
        <iframe ref={frame} className="sheet-frame" src={routes.sheetHtml(runId)} title={`evidence sheet for run ${runId}`} />
      )}
    </div>
  );
}
