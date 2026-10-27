"""Gate run for Muse Voice Transcribe: one narration in, turns with timestamps out.

    python scripts/gate_voice.py VIDEO_OR_WAV [--mode ENDPOINTING] --out runs/<name>

A video is turned into a 24 kHz mono WAV first (walkaround.video.extract_wav). Prints one line per turn, writes
<out>/transcript.json and <out>/gate_summary.md (duration, cost, latency) and appends a row to runs/ledger.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from walkaround.config import VOICE_MODEL  # noqa: E402
from walkaround.voice_client import MODES, Transcript, TranscribeError, Turn, transcribe_wav  # noqa: E402


def audio_for(source: Path, out_dir: Path) -> Path:
    """The WAV itself, or one extracted from the video."""
    if source.suffix.lower() == ".wav":
        return source
    from walkaround.video import extract_wav  # imported here so a WAV run never touches ffmpeg

    return extract_wav(source, out_dir / "audio.wav", rate=24000)


def turn_line(turn: Turn) -> str:
    who = f"{turn.speaker}: " if turn.speaker else ""
    return f"[{turn.start_ms / 1000:7.2f}s - {turn.end_ms / 1000:7.2f}s] {who}{turn.text}"


def gate_verdict(transcript: Transcript, mode: str) -> str:
    n = len(transcript.turns)
    if n == 0:
        return "CHECK: no turns came back. Was there speech in the clip?"
    if mode == "PUSH_TO_TALK":
        return f"PASS: transcript came back ({n} turn, no timestamps in PUSH_TO_TALK)"
    timed = sum(1 for t in transcript.turns if t.end_ms > t.start_ms)
    if timed == n:
        return f"PASS: {n} turns, every one with start and end timestamps"
    return f"CHECK: {n - timed} of {n} turns have no usable timestamps"


def write_outputs(out_dir: Path, source: Path, wav: Path, transcript: Transcript, mode: str, latency_s: float) -> None:
    duration_s = transcript.duration_ms / 1000
    record = {
        "source": str(source),
        "wav": str(wav),
        "wav_bytes": wav.stat().st_size,
        "mode": mode,
        "model": VOICE_MODEL,
        "session_id": transcript.session_id,
        "duration_ms": transcript.duration_ms,
        "usd": transcript.usd,
        "latency_s": round(latency_s, 3),
        "text": transcript.text,
        "turns": [asdict(t) for t in transcript.turns],
        "raw": transcript.raw,
    }
    (out_dir / "transcript.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Voice gate: {source.name}",
        "",
        f"Run {out_dir.name}, {time.strftime('%Y-%m-%d %H:%M:%S %Z')}.",
        "",
        "| What | Value |",
        "|---|---|",
        f"| Source | {source} |",
        f"| WAV sent | {wav} ({wav.stat().st_size / 1024 / 1024:.2f} MB) |",
        f"| Mode | {mode} |",
        f"| Model | {VOICE_MODEL} |",
        f"| Audio duration | {duration_s:.1f} s (audioDurationMs {transcript.duration_ms}) |",
        f"| Turns | {len(transcript.turns)} |",
        f"| Cost | ${transcript.usd:.4f} ({int(duration_s)} whole seconds at $0.18 per hour) |",
        f"| Latency | {latency_s:.2f} s (wall clock around the request) |",
        f"| Session | {transcript.session_id} |",
        "",
        f"Gate: {gate_verdict(transcript, mode)}",
        "",
        "## Turns",
        "",
        "```",
        *[turn_line(t) for t in transcript.turns],
        "```",
        "",
        "## Transcript",
        "",
        transcript.text,
        "",
    ]
    (out_dir / "gate_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="mono 16-bit WAV at 16 or 24 kHz, or a video with an audio track")
    parser.add_argument("--mode", default="ENDPOINTING", choices=MODES, help="ENDPOINTING gives turn timestamps")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR", help="run directory, e.g. runs/voice_gate")
    parser.add_argument("--keywords", action="append", metavar="TERM", help="bias term, repeatable (default: panel and mark words)")
    args = parser.parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        wav = audio_for(args.source, out_dir)
        started = time.monotonic()
        transcript = transcribe_wav(wav, mode=args.mode, keywords=args.keywords, run_id=out_dir.name)
        latency_s = time.monotonic() - started
    except (ValueError, TranscribeError, RuntimeError, OSError, httpx.HTTPError) as exc:
        print(f"gate_voice: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for turn in transcript.turns:
        print(turn_line(turn))
    write_outputs(out_dir, args.source, wav, transcript, args.mode, latency_s)
    print(
        f"{transcript.duration_ms / 1000:.1f}s of audio, {len(transcript.turns)} turns, "
        f"${transcript.usd:.4f}, {latency_s:.1f}s latency, mode {args.mode}"
    )
    print(f"Gate: {gate_verdict(transcript, args.mode)}")
    print(f"wrote {out_dir / 'transcript.json'} and {out_dir / 'gate_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
