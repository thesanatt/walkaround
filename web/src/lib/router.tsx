import { useEffect, useState, type MouseEvent, type ReactNode } from 'react';

// A history-API router in forty lines: the app has six routes and no need for a library.
export interface Location {
  path: string;
  query: URLSearchParams;
}

function current(): Location {
  return { path: window.location.pathname, query: new URLSearchParams(window.location.search) };
}

export function navigate(to: string, options: { replace?: boolean } = {}): void {
  if (options.replace) window.history.replaceState(null, '', to);
  else window.history.pushState(null, '', to);
  window.dispatchEvent(new PopStateEvent('popstate'));
}

export function useLocation(): Location {
  const [loc, setLoc] = useState<Location>(current);
  useEffect(() => {
    const onChange = () => setLoc(current());
    window.addEventListener('popstate', onChange);
    return () => window.removeEventListener('popstate', onChange);
  }, []);
  return loc;
}

// '/run/:id/sheet' against '/run/abc/sheet' gives { id: 'abc' }; null when the shapes differ.
export function matchPath(pattern: string, path: string): Record<string, string> | null {
  const p = pattern.split('/').filter(Boolean);
  const s = path.split('/').filter(Boolean);
  if (p.length !== s.length) return null;
  const params: Record<string, string> = {};
  for (let i = 0; i < p.length; i += 1) {
    const a = p[i] ?? '';
    const b = s[i] ?? '';
    if (a.startsWith(':')) params[a.slice(1)] = decodeURIComponent(b);
    else if (a !== b) return null;
  }
  return params;
}

export function Link({ to, children, className, replace, current }: { to: string; children: ReactNode; className?: string; replace?: boolean; current?: boolean }) {
  const onClick = (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    navigate(to, { replace: replace ?? false });
  };
  return (
    <a href={to} className={className} onClick={onClick} aria-current={current ? 'page' : undefined}>
      {children}
    </a>
  );
}
