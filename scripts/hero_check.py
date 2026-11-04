"""Helpers for scripts/hero.sh, the one command for the hero pair. No model call, no network.

    hero_check.py estimate PICKUP RETURN [--seconds A B] [--ref RUN_ID ...]
        cost and time of a live pair of these walks, scaled from the reference runs' ledger rows and event logs, against
        the per-run spend ceiling, the run deadline, today's card ceiling and the Meta credit left. Exit 3 when the
        padded estimate could cross one of them.
    hero_check.py summary RUN_ID      a short verdict summary of a finished run, plus the Plan A checks for the video
    hero_check.py intake JSON         the lines worth reading from one tools/data_hunt/intake_check.py result
    hero_check.py watch LOG PID       print each new stage of a live run's event log while process PID lives
    hero_check.py clips RUN_ID [--minimal]
        the --clip arguments for make_sample.py: the Stays Put clips the run page opens with (the default panel's
        focus mark in each walk, the panel's other marks, the control), as web/src/pages/RunPage.tsx defaultPanel,
        web/src/lib/focus.ts and StaysPutPair.controlTrack pick them

How the estimate scales: the SAM lanes are paid per proxy frame (15 fps whatever the source size, 16 lanes a walk), so
the lane rows scale with the walk seconds and everything else (Stays Put clips, keyframe images, Muse Spark crops, the
second look, voice) is taken as it was. The upper bound scales the whole run by the walk seconds. Time: the stages up
to the blur (seal, proxy, lanes, panels) scale with the seconds, the rest is taken as it was; the upper bound scales
the whole wall time. A cap is called at risk when the upper bound times SAFETY passes it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
REFS = ("friend_pair2", "synth_pair2_r9")
SAFETY = 1.25
META_GRANT_USD = 150.0  # the Meta Model API credit granted for the hackathon (LEDGER.md, checked 10/31)
ET = ZoneInfo("America/New_York")


def runs_root() -> Path:
    return Path(os.environ.get("WALKAROUND_RUNS") or REPO / "runs")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ledger_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "ledger.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ts(text: str) -> dt.datetime:
    text = text.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")


def probe_seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def thresholds() -> dict[str, Any]:
    return read_json(REPO / "config" / "thresholds.json")


def limits() -> dict[str, float]:
    o = thresholds()["orchestrator"]
    ceiling = int(os.environ.get("RUN_SPEND_CEILING_CENTS") or o["run_spend_ceiling_cents"]["value"])
    return {"ceiling_usd": ceiling / 100, "deadline_s": float(o["run_deadline_s"]["value"]), "day_usd": o["daily_card_ceiling_cents"]["value"] / 100}


def ref_stats(root: Path, rows: list[dict[str, Any]], run_id: str) -> dict[str, float]:
    """Walk seconds, dollars (lanes and the rest) and wall seconds (to the blur and in all) of one finished run."""
    run_dir = root / run_id
    secs = sum(read_json(run_dir / "videos" / f"{role}.json")["ffprobe"]["duration_s"] for role in ("pickup", "return"))
    mine = [r for r in rows if r.get("run_id") == run_id]
    total = sum(r.get("usd") or 0 for r in mine)
    lanes = sum(r.get("usd") or 0 for r in mine if r.get("kind") == "sam_video" and not str(r.get("note", "")).startswith("sp_"))
    t0 = t_blur = t_end = None
    for line in (run_dir / "events.ndjson").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        t = ts(e["ts"])
        t0 = t0 or t
        if e["kind"] == "blur.done":
            t_blur = t
        if e["kind"] == "run.completed":
            t_end = t
            break
    if t0 is None or t_end is None:
        raise ValueError(f"{run_id}: no run.completed event")
    t_blur = t_blur or t_end
    return {"secs": secs, "usd": total, "lanes_usd": lanes, "wall_s": (t_end - t0).total_seconds(), "scaled_s": (t_blur - t0).total_seconds()}


def estimate(secs: float, refs: dict[str, dict[str, float]]) -> dict[str, float]:
    """Central and upper cost and time for walks of `secs` seconds in all, the largest over the reference runs."""
    out = {"usd": 0.0, "usd_upper": 0.0, "wall_s": 0.0, "wall_upper_s": 0.0}
    for r in refs.values():
        k = secs / r["secs"]
        out["usd"] = max(out["usd"], r["lanes_usd"] * k + (r["usd"] - r["lanes_usd"]))
        out["usd_upper"] = max(out["usd_upper"], r["usd"] * k)
        out["wall_s"] = max(out["wall_s"], r["scaled_s"] * k + (r["wall_s"] - r["scaled_s"]))
        out["wall_upper_s"] = max(out["wall_upper_s"], r["wall_s"] * k)
    out["usd_upper"] = max(out["usd_upper"], out["usd"])
    out["wall_upper_s"] = max(out["wall_upper_s"], out["wall_s"])
    return out


def meta_spend(rows: list[dict[str, Any]], *, today: dt.date | None = None) -> tuple[float, float]:
    """(all Meta Model API dollars in the ledger, those of today's ET day). Baseten rows bill another credit."""
    today = today or dt.datetime.now(ET).date()
    total = day = 0.0
    for r in rows:
        if r.get("engine") == "baseten" or r.get("kind") == "spark_open":
            continue
        usd = r.get("usd") or 0
        total += usd
        if ts(str(r["ts"])).astimezone(ET).date() == today:
            day += usd
    return total, day


def cmd_estimate(args: argparse.Namespace) -> int:
    root = runs_root()
    rows = ledger_rows(root)
    if args.seconds:
        walks = list(args.seconds)
    else:
        walks = [probe_seconds(Path(p)) for p in (args.pickup, args.return_)]
    secs = sum(walks)
    refs = {rid: ref_stats(root, rows, rid) for rid in (args.ref or REFS)}
    e = estimate(secs, refs)
    lim = limits()
    used, today = meta_spend(rows)
    left = META_GRANT_USD - used
    for rid, r in refs.items():
        print(f"ref {rid}: {r['secs']:.1f} s of walk, ${r['usd']:.2f} (lanes ${r['lanes_usd']:.2f}), {r['wall_s'] / 60:.1f} min")
    print(f"this pair: {' + '.join(f'{w:.1f}' for w in walks)} s of walk")
    print(f"estimate: ${e['usd']:.2f} (upper ${e['usd_upper']:.2f}), {e['wall_s'] / 60:.1f} min (upper {e['wall_upper_s'] / 60:.1f} min)")
    print(f"caps: per run ${lim['ceiling_usd']:.2f}, deadline {lim['deadline_s'] / 60:.0f} min, today ${today:.2f} of ${lim['day_usd']:.0f} spent")
    print(f"Meta credit: ${used:.2f} in the local ledger, about ${left:.2f} of ${META_GRANT_USD:.0f} left (calls made on Railway are not in it)")
    risks = []
    if e["usd_upper"] * SAFETY > lim["ceiling_usd"]:
        risks.append(f"spend: {SAFETY:g} x upper ${e['usd_upper'] * SAFETY:.2f} passes the per-run ceiling ${lim['ceiling_usd']:.2f}")
    if e["wall_upper_s"] * SAFETY > lim["deadline_s"]:
        risks.append(f"time: {SAFETY:g} x upper {e['wall_upper_s'] * SAFETY / 60:.1f} min passes the deadline {lim['deadline_s'] / 60:.0f} min")
    if today + e["usd_upper"] > lim["day_usd"]:
        risks.append(f"day: ${today:.2f} spent today + upper ${e['usd_upper']:.2f} passes the daily ${lim['day_usd']:.0f}")
    if e["usd_upper"] > left:
        risks.append(f"credit: upper ${e['usd_upper']:.2f} is more than the ${left:.2f} left")
    for line in risks:
        print("AT RISK " + line)
    if not risks:
        print(f"fits: {SAFETY:g} x the upper bound stays under every cap")
    return 3 if risks else 0


def default_panel(verdicts: list[dict[str, Any]], matches: list[dict[str, Any]], coverage: dict[str, Any], tracks: list[dict[str, Any]]) -> str | None:
    """RunPage.defaultPanel: an "already there" verdict, else a comparable panel, else a filmed one, else a track."""
    for v in verdicts:
        if v.get("label") == "already_there":
            return v["panel_id"]
    for m in matches:
        if m.get("state") == "comparable":
            return m["panel_id"]
    for panel, cell in ((coverage.get("per_role") or {}).get("pickup") or {}).items():
        if (cell or {}).get("state") == "filmed":
            return panel
    return next((t["panel_id"] for t in tracks if t.get("panel_id")), None)


def joined_pairs(match: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not match or match.get("state") != "comparable":
        return []
    return [p for p in match.get("pairs") or [] if p["d"] < p["tau_eye_r"]]


def panel_focus(panel: str | None, match: dict[str, Any] | None, verdicts: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """focus.ts panelFocus: the panel's first "already there", else its first verdict about a joined pair, else its first."""
    if not panel:
        return None, None
    pairs = joined_pairs(match)
    own = [v for v in verdicts if v["panel_id"] == panel]

    def pair_of(v: dict[str, Any]) -> dict[str, Any] | None:
        ids = v.get("mark_ids") or []
        if len(ids) != 2:
            return None
        return next((p for p in pairs if p["pickup_mark"] in ids and p["return_mark"] in ids), None)

    already = next((v for v in own if v.get("label") == "already_there"), None)
    if already:
        return already, pair_of(already)
    for v in own:
        p = pair_of(v)
        if p:
            return v, p
    return (own[0] if own else None), None


class Run:
    def __init__(self, run_id: str) -> None:
        self.dir = runs_root() / run_id
        if not (self.dir / "run.json").is_file():
            raise SystemExit(f"no run {run_id} in {runs_root()}")
        d = self.dir / "derived"
        self.run = read_json(self.dir / "run.json")
        self.verdicts = self._list(d / "verdicts.json")
        self.matches = self._list(d / "matches.json")
        self.marks = {m["mark_id"]: m for m in self._list(d / "marks.json")}
        self.mark_tracks = self._list(d / "mark_tracks.json")
        self.looks = self._list(d / "looks.json")
        self.tracks = self._list(d / "tracks.json")
        self.coverage = read_json(d / "coverage.json") if (d / "coverage.json").is_file() else {}
        self.narration = read_json(d / "narration.json") if (d / "narration.json").is_file() else {}
        self.panel = default_panel(self.verdicts, self.matches, self.coverage, self.tracks)
        match = next((m for m in self.matches if m["panel_id"] == self.panel), None)
        self.focus, self.pair = panel_focus(self.panel, match, self.verdicts)

    @staticmethod
    def _list(path: Path) -> list[dict[str, Any]]:
        return read_json(path) if path.is_file() else []


def pick_clips(r: Run, *, minimal: bool = False) -> list[str]:
    usable = [m for m in r.mark_tracks if (m.get("clip") or {}).get("path")]
    picked: list[str] = []

    def add(m: dict[str, Any] | None) -> None:
        if m and m["clip"]["path"] not in picked:
            picked.append(m["clip"]["path"])

    marks = r.marks
    pickup = [m for m in usable if m["role"] == "pickup" and not m.get("is_control")]
    if r.focus:
        ids = [r.pair["pickup_mark"]] if r.pair else [i for i in r.focus.get("mark_ids") or [] if (marks.get(i) or {}).get("role") != "return"]
        add(next((m for m in pickup if m["mark_id"] in ids), None))
    else:
        add(next((m for m in pickup if (marks.get(m["mark_id"]) or {}).get("panel_id") == r.panel), None))
    if r.pair:
        add(next((m for m in usable if m["mark_id"] == r.pair["return_mark"]), None))
    controls = [m for m in usable if m.get("is_control")]
    add(next((m for m in controls if m.get("verdict") == "sliding"), controls[0] if controls else None))
    if not minimal:
        for m in usable:
            if not m.get("is_control") and (marks.get(m["mark_id"]) or {}).get("panel_id") == r.panel:
                add(m)
    return picked


def cmd_clips(args: argparse.Namespace) -> int:
    print(" ".join(f"--clip {p}" for p in pick_clips(Run(args.run_id), minimal=args.minimal)))
    return 0


def wall_seconds(run_dir: Path) -> float | None:
    first = last = None
    for line in (run_dir / "events.ndjson").read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            first = first or ts(e["ts"])
            if e["kind"] == "run.completed":
                last = ts(e["ts"])
                break
    return (last - first).total_seconds() if first and last else None


def cmd_summary(args: argparse.Namespace) -> int:
    r = Run(args.run_id)
    run = r.run
    wall = wall_seconds(r.dir)
    print(f"run {args.run_id}: {run['status']}, ${run.get('api_spend_cents', 0) / 100:.2f}" + (f", {wall / 60:.1f} min" if wall else ""))
    for err in run.get("errors") or []:
        print(f"  error: {str(err)[:160]}")
    labels = Counter(v["label"] for v in r.verdicts)
    print("  verdicts: " + ", ".join(f"{k.replace('_', ' ')} {n}" for k, n in labels.most_common()))
    for v in r.verdicts:
        if v["label"] == "not_comparable":
            continue
        cm = f", about {v['distance_cm_est']:.0f} cm" if v.get("distance_cm_est") is not None else ""
        why = (v.get("reasons") or [""])[0]
        print(f"  {v['panel_id']}: {v['label'].replace('_', ' ')} ({v['rule_id']}{cm}) {why[:100]}")
    missing = r.coverage.get("missing") or []
    print(f"  not filmed in some walk: {', '.join(missing) if missing else 'none'}")
    if r.looks:
        print("  second look: " + ", ".join(f"{x['panel_id']} {x.get('verdict')}" for x in r.looks))
    pins = [t.get("pinned_panel") for role in (r.narration.get("per_role") or {}).values() for t in role.get("turns") or []]
    print(f"  narration turns: {len(pins)}, pinned to {', '.join(p for p in pins if p) or 'no panel'}")
    label = r.focus["label"] if r.focus else None
    look = next((x.get("verdict") for x in r.looks if x["panel_id"] == r.panel), None)
    print(f"  run page opens on: {r.panel} ({label.replace('_', ' ') if label else 'no verdict'})")
    plan_a = label == "already_there" and look == "same"
    print(f"  Plan A (P4): already there {'yes' if label == 'already_there' else 'no'}, second look said same {'yes' if look == 'same' else 'no'}"
          f" -> {'check the mark at 100 percent and verify' if plan_a else 'Plan B (pairC_core3)'}")
    return 0


def cmd_intake(args: argparse.Namespace) -> int:
    d = read_json(Path(args.json))
    if "error" in d:
        print(f"{d['file']}: {d['error']}")
        return 0
    w, h = d["display_wh"]
    audio = "yes" if d["audio"] else "no"
    print(f"{d['file']}: {w}x{h} {d['codec']} {d['fps']:g} fps {d['duration_s']:g} s, audio {audio}, hard checks {'pass' if d['hard_pass'] else 'FAIL'}")
    for k, ok in d["checks"].items():
        if not ok:
            print(f"  failed: {k}")
    for wn in d["warnings"]:
        if not wn.endswith(" failed"):
            print(f"  warning: {wn}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    path, pos, seen = Path(args.log), 0, set()
    while True:
        alive = True
        try:
            os.kill(args.pid, 0)
        except OSError:
            alive = False
        if path.is_file():
            with path.open(encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            for line in chunk.splitlines():
                if '"kind":"stage"' not in line:
                    continue
                try:
                    stage = json.loads(line)["payload"]
                except (ValueError, KeyError):
                    continue
                if stage.get("name") not in seen:
                    seen.add(stage.get("name"))
                    print(f"  {dt.datetime.now():%H:%M:%S} {stage.get('name')} {stage.get('pct', '')}%", flush=True)
        if not alive:
            return 0
        time.sleep(2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("pickup", nargs="?")
    e.add_argument("return_", nargs="?", metavar="return")
    e.add_argument("--seconds", type=float, nargs=2, metavar=("PICKUP_S", "RETURN_S"))
    e.add_argument("--ref", action="append")
    s = sub.add_parser("summary")
    s.add_argument("run_id")
    c = sub.add_parser("clips")
    c.add_argument("run_id")
    c.add_argument("--minimal", action="store_true")
    i = sub.add_parser("intake")
    i.add_argument("json")
    w = sub.add_parser("watch")
    w.add_argument("log")
    w.add_argument("pid", type=int)
    args = ap.parse_args(argv)
    if args.cmd == "estimate" and not args.seconds and not (args.pickup and args.return_):
        ap.error("estimate needs PICKUP RETURN or --seconds A B")
    return {"estimate": cmd_estimate, "summary": cmd_summary, "clips": cmd_clips, "intake": cmd_intake, "watch": cmd_watch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
