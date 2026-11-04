export function hashPrefix(hex: string | null | undefined, n = 12): string {
  return hex ? hex.slice(0, n) : '';
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

// Video time as m:ss.
export function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

// Video time with tenths, for narration turns: 0:11.4
export function formatClockTenths(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00.0';
  const m = Math.floor(seconds / 60);
  return `${m}:${(seconds - m * 60).toFixed(1).padStart(4, '0')}`;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DATE_ONLY = /^(\d{4})-(\d{2})-(\d{2})$/;

// The viewer's zone as a short name ("EDT", "UTC", "GMT+1"), so a printed time is never ambiguous.
function zoneOf(d: Date): string {
  try {
    return new Intl.DateTimeFormat('en-US', { timeZoneName: 'short' }).formatToParts(d).find((p) => p.type === 'timeZoneName')?.value ?? '';
  } catch {
    return '';
  }
}

// "Oct 27, 2026, 10:14 EDT" in the viewer's zone, named. The literals "not present" and "camera time not present"
// pass through.
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return 'not present';
  if (iso === 'not present' || iso === 'camera time not present') return iso;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const zone = zoneOf(d);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}, ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}${zone ? ` ${zone}` : ''}`;
}

// The two halves of formatDateTime, for a card that sets the day and the time on separate lines.
export function formatDay(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}

export function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const zone = zoneOf(d);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}${zone ? ` ${zone}` : ''}`;
}

// "Oct 27, 2026". A bare YYYY-MM-DD (ReplayPayload.recorded_on) is a calendar date, not UTC midnight.
export function formatDate(value: string | null | undefined): string {
  if (!value) return 'an unknown date';
  const m = DATE_ONLY.exec(value);
  if (m) return `${MONTHS[Number(m[2]) - 1]} ${Number(m[3])}, ${m[1]}`;
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}

function diffMs(a: string | null | undefined, b: string | null | undefined): number | null {
  if (!a || !b) return null;
  const d = Date.parse(b) - Date.parse(a);
  return Number.isFinite(d) ? d : null;
}

// Calendar days between two timestamps in the viewer's zone (Oct 27 23:50 to Oct 28 00:10 is one day).
export function daysBetween(a: string | null | undefined, b: string | null | undefined): number | null {
  if (diffMs(a, b) === null) return null;
  const day = (s: string) => {
    const d = new Date(s);
    return Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) / 86_400_000;
  };
  return day(b!) - day(a!);
}

export function minutesBetween(a: string | null | undefined, b: string | null | undefined): number | null {
  const d = diffMs(a, b);
  return d === null ? null : Math.round(d / 60_000);
}

export function formatDollars(usd: number): string {
  return `$${usd.toFixed(2)}`;
}

export function plural(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`;
}

// "Nov 2, 6:42 PM EST" in the viewer's zone, named; withYear gives "Nov 2, 2026, 6:42 PM EST".
export function formatStamp(iso: string | null | undefined, withYear = false): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const zone = zoneOf(d);
  const h = d.getHours() % 12 || 12;
  const year = withYear ? `, ${d.getFullYear()}` : '';
  return `${MONTHS[d.getMonth()]} ${d.getDate()}${year}, ${h}:${String(d.getMinutes()).padStart(2, '0')} ${d.getHours() < 12 ? 'AM' : 'PM'}${zone ? ` ${zone}` : ''}`;
}
