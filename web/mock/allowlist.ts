// What the mock serves from a run directory: app.servable's floor, then privacy.served_file_allowed, so `npm run dev`
// answers 404 exactly where app.py does. The claim rule and open_weights.ndjson follow the binding allowlist of
// docs/CONTRACTS.md ("Phase 2 file shapes"): claim/<claim_id>.json and claim/<claim_id>.blurred.jpg are served, the
// upload claim/<claim_id>.jpg and its answers never. TODO(integrator): privacy.py still lacks those two entries
// (_SERVED_RUN_FILES has only deletions.json; _SERVED["claim"] allows any .jpg and no .json); until it has them, the
// real server refuses open_weights.ndjson and claim/<claim_id>.json while this mock serves them.

const NEVER_SERVED_TOPS = new Set(['quarantine', 'uploads']);
const NEVER_SERVED_SUFFIXES = ['.orig', '.part', '.tmp', '.proxy.mp4'];
const PENDING_EVENTS_NAME = 'events.pending.ndjson';

const SERVED: Record<string, { depths: number[]; endings: string[] }> = {
  videos: { depths: [2], endings: ['.json'] },
  media: { depths: [2], endings: ['.proxy_blurred.mp4', '.pts.json'] },
  sam: { depths: [3], endings: ['.ndjson.gz', '.index.json', '.summary.json'] },
  derived: { depths: [2], endings: ['.json'] },
  crops: { depths: [3], endings: ['.jpg', '.png'] },
  keyframes: { depths: [3], endings: ['.jpg'] },
  clips: { depths: [3], endings: ['.mp4'] },
  chips: { depths: [3], endings: ['.m4a'] },
  claim: { depths: [2], endings: ['.blurred.jpg', '.json'] },
  sheet: { depths: [2, 3], endings: ['.html', '.json', '.pdf', '.jpg', '.zip'] },
  bundle: { depths: [2], endings: ['.json', '.ots', '.tsq', '.tsr', '.md', '.pem', '.crt'] },
};
const SERVED_RUN_FILES = new Set(['deletions.json', 'open_weights.ndjson']);
const NEVER_IN_NAME = ['.part', '.tmp', '.orig', '.bak'];
const OWNER_ONLY_UNDER_KEEP_PLATE = new Set(['sheet', 'crops', 'keyframes', 'clips', 'claim']);

// app.servable: the floor under every served file.
function servable(parts: string[]): boolean {
  if (!parts.length || parts.some((p) => p === '' || p === '.' || p === '..' || p.startsWith('.'))) return false;
  const name = parts[parts.length - 1]!;
  if (NEVER_SERVED_TOPS.has(parts[0]!) || name === PENDING_EVENTS_NAME) return false;
  return !NEVER_SERVED_SUFFIXES.some((s) => name.endsWith(s));
}

// owner is false for ?view=kiosk and for a public run, as app.may_serve decides.
export function servedFileAllowed(run: { keep_plate: boolean }, relPath: string, owner: boolean): boolean {
  if (!relPath || relPath.startsWith('/') || relPath.includes('\\')) return false;
  const parts = relPath.split('/').map((p) => p.toLowerCase()); // APFS matches names case-insensitively
  if (!servable(parts)) return false;
  const name = parts[parts.length - 1]!;
  if (NEVER_IN_NAME.some((bad) => name.includes(bad))) return false;
  if (parts.length === 1) return SERVED_RUN_FILES.has(name);
  const rule = SERVED[parts[0]!];
  if (!rule || !rule.depths.includes(parts.length) || !rule.endings.some((e) => name.endsWith(e))) return false;
  if (parts[0] === 'claim' && name.endsWith('.answers.json')) return false; // every model answer of a claim, never served
  if (parts[0] === 'sheet' && parts.length === 3 && parts[1] !== 'frames') return false;
  if (name.includes('.owner.') && !owner) return false;
  if (run.keep_plate && !owner && OWNER_ONLY_UNDER_KEEP_PLATE.has(parts[0]!)) return false;
  return true;
}
