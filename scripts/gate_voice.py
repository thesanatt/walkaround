"""D1 gate row E10, Muse Voice Transcribe: ENDPOINTING (and DIARIZATION as the comparison) on the narration with
the 21 keywords, the mode string validated, turn spans against the panel windows SAM found in gate_sam.

    LIVE=1 python scripts/gate_voice.py VIDEO_OR_WAV [--modes ENDPOINTING DIARIZATION] [--keywords TERM ...] \
        [--pins gate_sam.json | panels.json] [--role pickup] --out DIR
    python scripts/gate_voice.py --fixture tests/fixtures/d1/voice_fixture.json [--pins gate_sam.json] --out DIR

A video is turned into a 24 kHz mono WAV first (walkaround.video.extract_wav). Prints one line per turn, writes
<out>/transcript_<mode>.json, <out>/gate_voice.json and <out>/gate_summary.md. Every call goes through
walkaround.voice_client.transcribe_wav, which writes the ledger row (note = the mode, status "mode not applied"
when a speech mode returned no turns). --pins takes gate_sam.json (the lane windows of the first sweep) or a
derived/panels.json. The fixture path drives transcribe_wav through an httpx mock transport over a synthetic WAV,
with the ledger pointed at <out>/ledger.dryrun.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (str(REPO), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import gate_sam  # noqa: E402
from walkaround import contracts, ledger, voice_client  # noqa: E402
from walkaround.config import VOICE_MODEL  # noqa: E402
from walkaround.contracts import PANEL_NOUN, Noun, PanelId  # noqa: E402
from walkaround.voice_client import MODES, Transcript, TranscribeError, Turn, transcribe_wav  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "d1" / "voice_fixture.json"
PANEL_WORDS = sorted(voice_client.SPLIT_NOUNS)  # the 13 panel nouns a turn can name (car itself is not a pin)
Window = tuple[int, int]


# ------------------------------------------------------------------------------------------------------------------
# Audio and turns
# ------------------------------------------------------------------------------------------------------------------


def audio_for(source: Path, out_dir: Path) -> Path:
    """The WAV itself, or one extracted from the video."""
    if source.suffix.lower() == ".wav":
        return source
    video = gate_sam.optional_module("video")
    if video is None or not gate_sam.implemented(video, "extract_wav"):
        raise RuntimeError("walkaround.video.extract_wav is not implemented yet")
    return video.extract_wav(source, out_dir / "audio.wav", rate=24000)


def synthetic_wav(dst: Path, *, seconds: float = 1.0, rate: int = 24000) -> Path:
    """A 24 kHz mono 16-bit tone, enough for check_wav; the fixture transport ignores the audio."""
    n = int(seconds * rate)
    frames = b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 220 * i / rate))) for i in range(n))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return dst


def turn_line(turn: Turn) -> str:
    who = f"{turn.speaker}: " if turn.speaker else ""
    return f"[{turn.start_ms / 1000:7.2f}s - {turn.end_ms / 1000:7.2f}s] {who}{turn.text}"


def gate_verdict(transcript: Transcript, mode: str) -> str:
    n = len(transcript.turns)
    if not transcript.mode_applied:
        return "FAIL: mode not applied (a transcript came back with empty turns); inspect the request JSON first"
    if n == 0:
        return "CHECK: no turns and no transcript. Was there speech in the clip?"
    if mode == "PUSH_TO_TALK":
        return f"PASS: transcript came back ({n} turn, no timestamps in PUSH_TO_TALK)"
    timed = sum(1 for t in transcript.turns if t.end_ms > t.start_ms)
    if timed == n:
        return f"PASS: {n} turns, every one with start and end timestamps"
    return f"CHECK: {n - timed} of {n} turns have no usable timestamps"


# ------------------------------------------------------------------------------------------------------------------
# Panel windows (from gate_sam.json or panels.json) and the pins
# ------------------------------------------------------------------------------------------------------------------


def windows_from_gate_sam(doc: dict[str, Any], role: str) -> dict[str, list[Window]]:
    """Per noun, the visible windows in ms of the first sweep's lanes (the plain 'D1 E2 <noun>' lanes only)."""
    sweeps = [s for s in doc.get("sweeps", []) if "roles" in s]
    if not sweeps:
        return {}
    roles = sweeps[0]["roles"]
    role_out = roles.get(role) or next(iter(roles.values()))
    fps = float(sweeps[0].get("fps") or role_out.get("fps") or 15.0)
    out: dict[str, list[Window]] = {}
    for lane_id, m in role_out.get("lanes", {}).items():
        noun = m.get("noun")
        if not noun or lane_id != f"D1 E2 {noun}":
            continue
        windows = m.get("windows_s")
        if windows is None and m.get("frame_range"):
            windows = [[m["frame_range"][0] / fps, (m["frame_range"][1] + 1) / fps]]
        out[noun] = [(int(round(a * 1000)), int(round(b * 1000))) for a, b in (windows or [])]
    return out


def windows_from_panels(rows: list[dict[str, Any]], role: str) -> dict[str, list[Window]]:
    out: dict[str, list[Window]] = {}
    for row in rows:
        try:
            panel = PanelId(row["panel_id"])
        except (KeyError, ValueError):
            continue
        noun = PANEL_NOUN[panel].value
        per_role = (row.get("per_role") or {}).get(role) or {}
        for a, b in per_role.get("windows") or []:
            out.setdefault(noun, []).append((int(round(float(a) * 1000)), int(round(float(b) * 1000))))
    return out


def load_windows(path: Path, role: str) -> dict[str, list[Window]]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict) and "sweeps" in doc:
        return windows_from_gate_sam(doc, role)
    if isinstance(doc, list):
        return windows_from_panels(doc, role)
    raise SystemExit(f"{path}: neither a gate_sam.json nor a panels.json list")


def overlap_ms(span: Window, window: Window) -> int:
    return max(0, min(span[1], window[1]) - max(span[0], window[0]))


def pin_turns(turns: list[Turn], windows: dict[str, list[Window]], keywords: list[str] | None = None) -> list[dict[str, Any]]:
    """Per turn: the keywords heard, the panel nouns named, and where SAM showed them (overlap) or what was on
    screen at the turn start (about, the fallback of the E10 fail branch)."""
    rows = []
    for i, turn in enumerate(turns):
        hits = voice_client.keyword_hits(turn.text, keywords)
        nouns = [k for k in hits if k in voice_client.SPLIT_NOUNS]
        span = (turn.start_ms, turn.end_ms)
        placed = {}
        for noun in nouns:
            ov = max((overlap_ms(span, w) for w in windows.get(noun, [])), default=0)
            placed[noun] = ov
        overlap_nouns = [n for n, ov in placed.items() if ov > 0]
        at_start = sorted(n for n, ws in windows.items() if any(w[0] <= turn.start_ms < w[1] for w in ws))
        pin = overlap_nouns[0] if overlap_nouns else (at_start[0] if at_start else None)
        rows.append(
            {
                "turn": i,
                "start_ms": turn.start_ms,
                "end_ms": turn.end_ms,
                "speaker": turn.speaker,
                "text": turn.text,
                "keywords": hits,
                "nouns": nouns,
                "overlap_ms": placed,
                "landed": bool(overlap_nouns) if nouns else None,
                "on_screen_at_start": at_start,
                "pin": pin,
                "pin_kind": "overlap" if overlap_nouns else ("about" if pin else None),
            }
        )
    return rows


def pin_share(rows: list[dict[str, Any]]) -> dict[str, Any]:
    naming = [r for r in rows if r["nouns"]]
    landed = sum(1 for r in naming if r["landed"])
    return {"turns": len(rows), "noun_naming_turns": len(naming), "landed": landed, "share": round(landed / len(naming), 4) if naming else None, "about_pins": sum(1 for r in rows if r["pin_kind"] == "about")}


# ------------------------------------------------------------------------------------------------------------------
# Runs
# ------------------------------------------------------------------------------------------------------------------


def transcript_record(transcript: Transcript, *, source: Path, wav: Path, mode: str, latency_s: float) -> dict[str, Any]:
    return {
        "source": str(source),
        "wav": str(wav),
        "wav_bytes": wav.stat().st_size,
        "mode": mode,
        "model": VOICE_MODEL,
        "session_id": transcript.session_id,
        "duration_ms": transcript.duration_ms,
        "usd": transcript.usd,
        "latency_s": round(latency_s, 3),
        "mode_applied": transcript.mode_applied,
        "request": transcript.request,
        "text": transcript.text,
        "turns": [asdict(t) for t in transcript.turns],
        "raw": transcript.raw,
    }


def run_modes(wav: Path, *, source: Path, modes: list[str], keywords: list[str] | None, run_id: str, windows: dict[str, list[Window]] | None, out: Path, th: dict) -> dict[str, Any]:
    results: dict[str, Any] = {"modes": {}, "keywords": keywords if keywords is not None else voice_client.DEFAULT_KEYWORDS}
    for mode in modes:
        voice_client.validate_mode(mode)
        started = time.monotonic()
        try:
            transcript = transcribe_wav(wav, mode=mode, keywords=keywords, run_id=f"{run_id}_E10_{mode}")
        except (ValueError, TranscribeError, RuntimeError, OSError, httpx.HTTPError) as exc:
            results["modes"][mode] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "turns": 0, "mode_applied": None}
            continue
        latency_s = time.monotonic() - started
        record = transcript_record(transcript, source=source, wav=wav, mode=mode, latency_s=latency_s)
        (out / f"transcript_{mode}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        pins = pin_turns(transcript.turns, windows or {}, keywords) if windows is not None else pin_turns(transcript.turns, {}, keywords)
        results["modes"][mode] = {
            "turns": len(transcript.turns),
            "timed_turns": sum(1 for t in transcript.turns if t.end_ms > t.start_ms),
            "speakers": sorted({t.speaker for t in transcript.turns if t.speaker}),
            "mode_applied": transcript.mode_applied,
            "duration_ms": transcript.duration_ms,
            "usd": transcript.usd,
            "latency_s": round(latency_s, 3),
            "session_id": transcript.session_id,
            "verdict": gate_verdict(transcript, mode),
            "pins": pins,
            "pin_share": pin_share(pins) if windows is not None else None,
            "keywords_heard": sorted({k for r in pins for k in r["keywords"]}),
        }
    results["windows_ms"] = {k: [list(w) for w in v] for k, v in (windows or {}).items()}
    results["evaluation"] = evaluate(results, th)
    return results


def evaluate(results: dict[str, Any], th: dict) -> dict[str, Any]:
    main_mode = th["voice"]["mode"]
    m = results["modes"].get(main_mode) or {}
    share = (m.get("pin_share") or {}).get("share")
    return {
        "mode": main_mode,
        "turns_non_empty": (m.get("turns", 0) > 0) if m else None,
        "mode_applied": m.get("mode_applied"),
        "pin_share": share,
        "pins_pass": (share >= 0.8) if share is not None else None,
        "pin_fallback": th["voice"]["pin_fallback"] if share is not None and share < 0.8 else None,
        "diarization_turns": (results["modes"].get("DIARIZATION") or {}).get("turns"),
        "diarization_speakers": (results["modes"].get("DIARIZATION") or {}).get("speakers"),
        "errors": {k: v["error"] for k, v in results["modes"].items() if "error" in v},
    }


def run_live(args: argparse.Namespace, th: dict) -> dict[str, Any]:
    out = Path(args.out)
    prefix = args.run_prefix or f"d1_{gate_sam.utc_ts()}"
    source = Path(args.source)
    wav = audio_for(source, out)
    windows = load_windows(Path(args.pins), args.role) if args.pins else None
    results = run_modes(wav, source=source, modes=list(args.modes), keywords=args.keywords, run_id=prefix, windows=windows, out=out, th=th)
    results.update({"mode": "live", "run_prefix": prefix, "source": str(source), "wav": str(wav), "spend_usd": gate_sam.spend_for_prefix(prefix)})
    return results


def install_fixture(fixture: dict[str, Any], out: Path) -> None:
    """transcribe_wav runs for real over a mock transport that answers with the fixture's response per mode."""
    ledger.LEDGER_PATH = out / "ledger.dryrun.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        mode = "ENDPOINTING"
        for candidate in MODES:
            if f'"mode": "{candidate}"'.encode() in body:
                mode = candidate
        answer = fixture["modes"].get(mode) or fixture["modes"]["ENDPOINTING"]
        return httpx.Response(200, json=answer)

    voice_client.TRANSPORT = httpx.MockTransport(handler)


def run_fixture(args: argparse.Namespace, th: dict) -> dict[str, Any]:
    out = Path(args.out)
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    install_fixture(fixture, out)
    wav = synthetic_wav(out / "fixture_tone.wav")
    windows = load_windows(Path(args.pins), args.role) if args.pins else {"bumper": [(0, 2000), (4500, 6000)], "car door": [(1400, 3800)], "wheel": [(3500, 4800)]}
    modes = list(args.modes) if args.modes else list(fixture["modes"])
    results = run_modes(wav, source=Path(args.fixture), modes=modes, keywords=args.keywords, run_id="d1_fixture", windows=windows, out=out, th=th)
    results.update({"mode": "fixture", "run_prefix": None, "source": str(args.fixture), "wav": str(wav), "spend_usd": 0.0, "fixture": {"path": str(args.fixture), "sha256": gate_sam.sha256_file(Path(args.fixture)), "note": fixture.get("note")}, "windows_source": str(args.pins) if args.pins else "synthetic windows (fixture)", "dryrun_ledger": str(ledger.LEDGER_PATH)})
    return results


# ------------------------------------------------------------------------------------------------------------------
# Summary and CLI
# ------------------------------------------------------------------------------------------------------------------


def _fmt(value: Any, places: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def summary_markdown(results: dict[str, Any]) -> str:
    ev = results.get("evaluation", {})
    lines = [f"# Voice gate ({results['mode']})", "", f"- source {results.get('source')}; model {VOICE_MODEL}; keywords {len(results.get('keywords', []))}", f"- {ev.get('mode')}: turns non-empty {_fmt(ev.get('turns_non_empty'))}; mode applied {_fmt(ev.get('mode_applied'))}; noun-naming turns landing where SAM shows the panel {_fmt(ev.get('pin_share'), 2)} (pass {_fmt(ev.get('pins_pass'))}); fallback {ev.get('pin_fallback') or '-'}", f"- DIARIZATION: {ev.get('diarization_turns')} turns, speakers {ev.get('diarization_speakers')}"]
    for mode, m in results.get("modes", {}).items():
        lines += ["", f"## {mode}", ""]
        if "error" in m:
            lines.append(f"error: {m['error']}")
            continue
        lines.append(f"{m['turns']} turns ({m['timed_turns']} timed), mode applied {_fmt(m['mode_applied'])}, {m['duration_ms'] / 1000:.1f} s of audio, ${m['usd']:.4f}, {m['latency_s']:.1f} s latency, session {m['session_id']}; {m['verdict']}")
        if m.get("pin_share"):
            ps = m["pin_share"]
            lines.append(f"pins: {ps['landed']} of {ps['noun_naming_turns']} noun-naming turns overlap the panel's SAM window ({_fmt(ps['share'], 2)}); {ps['about_pins']} 'about' pins")
        lines += ["", "| turn | span | nouns | landed | pin | kind | text |", "|---|---|---|---|---|---|---|"]
        for r in m.get("pins", []):
            lines.append(f"| {r['turn']} | {r['start_ms'] / 1000:.2f}-{r['end_ms'] / 1000:.2f} s | {', '.join(r['nouns']) or '-'} | {_fmt(r['landed'])} | {r['pin'] or '-'} | {r['pin_kind'] or '-'} | {r['text'][:80]} |")
    lines += ["", f"Spend on this gate (ledger rows with the run prefix): ${results.get('spend_usd', 0.0):.4f}", ""]
    return "\n".join(lines)


def print_plan(args: argparse.Namespace, th: dict) -> None:
    print("gate_voice would send (set LIVE=1 to run):")
    print(f"  {args.source}: one WAV per mode {list(args.modes)} to {VOICE_MODEL} at $0.18 per hour of audio (about $0.005 per minute)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=str, help="mono 16-bit WAV at 16 or 24 kHz, or a video with an audio track")
    parser.add_argument("--mode", default=None, choices=MODES, help="one mode (kept from the old CLI; --modes runs several)")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=None, help="modes to run (default ENDPOINTING DIARIZATION)")
    parser.add_argument("--keywords", action="append", metavar="TERM", help="bias term, repeatable (default: the 21 words of thresholds nouns.voice_keywords)")
    parser.add_argument("--pins", help="gate_sam.json (lane windows) or derived/panels.json to place turns against")
    parser.add_argument("--role", default="pickup", help="role of the windows in --pins (default pickup)")
    parser.add_argument("--run-prefix", default=None, help="ledger run_id prefix (default d1_<timestamp>)")
    parser.add_argument("--fixture", help="saved responses per mode for a dry run through transcribe_wav (no call)")
    parser.add_argument("--out", type=str, required=True, metavar="DIR", help="run directory, e.g. runs/gate_voice")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    th = contracts.load_thresholds()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.modes is None:
        args.modes = [args.mode] if args.mode else ["ENDPOINTING", "DIARIZATION"]
    results = run_fixture(args, th) if args.fixture else run_live(args, th)
    results["thresholds_config_hash"] = contracts.config_hash(th)
    results["written_at"] = gate_sam.utc_ts()
    gate_sam.write_json(out / "gate_voice.json", results)
    (out / "gate_summary.md").write_text(summary_markdown(results), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    th = contracts.load_thresholds()
    if args.modes is None:
        args.modes = [args.mode] if args.mode else ["ENDPOINTING", "DIARIZATION"]
    if not args.fixture:
        if not args.source:
            print("gate_voice: a SOURCE (or --fixture) is required", file=sys.stderr)
            return 2
        if not gate_sam.live_allowed():
            print_plan(args, th)
            return 2
    results = run(args)
    for mode, m in results["modes"].items():
        for r in m.get("pins", []):
            print(f"{mode} {turn_line(Turn(r['start_ms'], r['end_ms'], r['text'], r['speaker']))}")
    print((Path(args.out) / "gate_summary.md").read_text(encoding="utf-8"))
    print(f"wrote {Path(args.out) / 'gate_voice.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
