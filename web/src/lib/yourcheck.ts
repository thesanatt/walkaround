import { useCallback, useEffect, useState } from 'react';

import type { Answer } from './findings.ts';

// The renter's checks of needs-your-eye findings, per run, in this browser only. They are never sent anywhere and
// never part of the sealed sheet. localStorage can be missing or throw (private windows, blocked storage), so every
// read and write is wrapped and the page works without it.
const KEY = (runId: string) => `walkaround.checks.${runId}`;
const ANSWERS: ReadonlySet<string> = new Set(['gone', 'still', 'new', 'already']);

export interface StoredCheck {
  answer: Answer;
  at: string; // ISO time of the tap
}

function read(runId: string): Record<string, StoredCheck> {
  try {
    const raw = window.localStorage.getItem(KEY(runId));
    const parsed: unknown = raw ? JSON.parse(raw) : {};
    const out: Record<string, StoredCheck> = {};
    if (parsed && typeof parsed === 'object')
      for (const [id, c] of Object.entries(parsed as Record<string, unknown>)) {
        const row = c as Partial<StoredCheck> | null;
        if (row && typeof row.answer === 'string' && ANSWERS.has(row.answer)) out[id] = { answer: row.answer as Answer, at: typeof row.at === 'string' ? row.at : '' };
      }
    return out;
  } catch {
    return {};
  }
}

function write(runId: string, checks: Record<string, StoredCheck>): void {
  try {
    if (Object.keys(checks).length) window.localStorage.setItem(KEY(runId), JSON.stringify(checks));
    else window.localStorage.removeItem(KEY(runId));
  } catch {
    // storage blocked: the checks last as long as the page
  }
}

export function useYourChecks(runId: string) {
  const [checks, setChecks] = useState<Record<string, StoredCheck>>(() => read(runId));
  useEffect(() => setChecks(read(runId)), [runId]);
  // another tab of the same run
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY(runId)) setChecks(read(runId));
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [runId]);
  const set = useCallback(
    (verdictId: string, answer: Answer | null) =>
      setChecks((prev) => {
        const next = { ...prev };
        if (answer) next[verdictId] = { answer, at: new Date().toISOString() };
        else delete next[verdictId];
        write(runId, next);
        return next;
      }),
    [runId],
  );
  const answers: Record<string, Answer> = Object.fromEntries(Object.entries(checks).map(([id, c]) => [id, c.answer]));
  return { checks, answers, set };
}
