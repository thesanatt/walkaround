"""Muse Voice Transcribe, file endpoint, for the renter's narration while filming.

POST https://api.meta.ai/v1/asr/transcribe, multipart parts: request (JSON, application/json:
{"mode":"ENDPOINTING","model":"muse-voice-transcribe-1.0","audioEncoding":"WAV","keywords":[...]}) and audio (WAV,
mono 16-bit PCM 16 or 24 kHz, <= 10 min, body <= 32 MB). ENDPOINTING returns turns with startMs/endMs; PUSH_TO_TALK
returns no turns. Response: sessionId, transcript, audioDurationMs, turns[{turnId,startMs,endMs,transcript,speaker}].

The request mirrors the cookbook's transcribe_recording (06_muse_voice/01_voice_api_fundamentals/utils.py).
"""

from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from . import ledger
from .config import VOICE_MODEL, VOICE_TRANSCRIBE_URL, api_key

DEFAULT_KEYWORDS = [
    "scratch", "scuff", "dent", "chip", "crack", "bumper", "fender", "hood", "trunk", "door", "wheel", "rim",
    "mirror", "headlight", "tail light", "windshield", "roof", "driver side", "passenger side", "front", "rear",
]

# The server ignores an unknown mode instead of rejecting it, so the client checks the name first.
MODES = ("PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION")
SAMPLE_RATES = (16000, 24000)
MAX_AUDIO_SECONDS = 10 * 60
MAX_BYTES = 32 * 1024 * 1024
TIMEOUT_S = 120.0
MAX_RETRY_WAIT_S = 60.0
# Optional language hint sent as languageBias, e.g. ["english"]. None lets the model detect the language.
LANGUAGE_BIAS: list[str] | None = None

# Tests set this to an httpx.MockTransport. None means the real network.
TRANSPORT: httpx.BaseTransport | None = None
_sleep = time.sleep

STATUS_HINTS = {
    400: "audio longer than the 10-minute cap, or a malformed request",
    401: "the credential was not accepted",
    404: "endpoint not found",
    413: "body over 32 MB",
    429: "rate limited; back off and retry",
    500: "the server exceeded its processing budget",
}


class TranscribeError(RuntimeError):
    """The server rejected the request or failed to transcribe it."""


@dataclass(frozen=True)
class Turn:
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None


@dataclass
class Transcript:
    text: str
    turns: list[Turn]
    duration_ms: int
    session_id: str | None = None
    usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


def check_wav(wav_path: Path) -> float:
    """Seconds of audio in a WAV the endpoint accepts (mono, 16-bit PCM, 16 or 24 kHz, <= 10 min, <= 32 MB).
    Raises ValueError naming every problem and the ffmpeg line that fixes it."""
    wav_path = Path(wav_path)
    size = wav_path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"{wav_path}: {size / 1024 / 1024:.1f} MB; the endpoint takes at most 32 MB")
    try:
        with wave.open(str(wav_path), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            frames = reader.getnframes()
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{wav_path}: not a PCM WAV file ({exc})") from exc
    seconds = frames / rate if rate else 0.0
    problems = []
    if channels != 1:
        problems.append(f"{channels} channels (need mono)")
    if width != 2:
        problems.append(f"{width * 8}-bit samples (need 16-bit)")
    if rate not in SAMPLE_RATES:
        problems.append(f"{rate} Hz (need 16000 or 24000 Hz)")
    if frames == 0:
        problems.append("no audio samples")
    if seconds > MAX_AUDIO_SECONDS:
        problems.append(f"{seconds:.1f} s of audio (the limit is 10 minutes)")
    if problems:
        raise ValueError(
            f"{wav_path}: {', '.join(problems)}. Convert it with: "
            f"ffmpeg -i {wav_path} -vn -ac 1 -ar 24000 -c:a pcm_s16le -map_metadata -1 fixed.wav"
        )
    return seconds


def _request_json(mode: str, keywords: list[str] | None) -> dict[str, Any]:
    body: dict[str, Any] = {"mode": mode, "model": VOICE_MODEL, "audioEncoding": "WAV"}
    if keywords:
        body["keywords"] = list(keywords)
    if LANGUAGE_BIAS:
        body["languageBias"] = list(LANGUAGE_BIAS)
    return body


def _retry_after_s(response: httpx.Response) -> float:
    """Seconds to wait from Retry-After (a number or an HTTP date), clamped to [0.5, MAX_RETRY_WAIT_S]."""
    value = response.headers.get("retry-after", "").strip()
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            delay = 1.0
    return min(max(delay, 0.5), MAX_RETRY_WAIT_S)


def _post(wav_path: Path, wav_bytes: bytes, request_json: dict[str, Any]) -> httpx.Response:
    """One multipart POST, repeated once after a 429."""
    headers = {"Authorization": f"Bearer {api_key()}"}
    files = {
        "request": (None, json.dumps(request_json), "application/json"),
        "audio": (wav_path.name, wav_bytes, "audio/wav"),
    }
    with httpx.Client(timeout=httpx.Timeout(TIMEOUT_S), transport=TRANSPORT) as client:
        response = client.post(VOICE_TRANSCRIBE_URL, headers=headers, files=files)
        if response.status_code == 429:
            _sleep(_retry_after_s(response))
            response = client.post(VOICE_TRANSCRIBE_URL, headers=headers, files=files)
    return response


def _check_status(response: httpx.Response) -> None:
    if response.status_code == 200:
        return
    hint = STATUS_HINTS.get(response.status_code, "unexpected response")
    raise TranscribeError(f"[{response.status_code}] {hint}: {response.text[:300]}")


def _parse_turns(raw: dict[str, Any], duration_ms: int) -> list[Turn]:
    turns = [
        Turn(int(t.get("startMs") or 0), int(t.get("endMs") or 0), t.get("transcript") or "", t.get("speaker"))
        for t in raw.get("turns") or []
    ]
    text = raw.get("transcript") or ""
    if not turns and text.strip():
        # PUSH_TO_TALK returns no turns: one turn spans the whole clip.
        turns = [Turn(0, duration_ms, text)]
    return turns


def transcribe_wav(
    wav_path: Path, *, mode: str = "ENDPOINTING", keywords: list[str] | None = None, run_id: str | None = None
) -> Transcript:
    """One HTTP request. Records a ledger row (kind voice, audio_seconds from audioDurationMs)."""
    wav_path = Path(wav_path)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    header_seconds = check_wav(wav_path)
    body = _request_json(mode, DEFAULT_KEYWORDS if keywords is None else keywords)
    started = time.monotonic()
    response = _post(wav_path, wav_path.read_bytes(), body)
    latency_s = time.monotonic() - started
    _check_status(response)
    try:
        raw = response.json()
    except ValueError as exc:
        raise TranscribeError(f"[200] response was not JSON: {response.text[:300]}") from exc
    if not isinstance(raw, dict):
        raise TranscribeError(f"[200] response was not a JSON object: {response.text[:300]}")
    # The server meters whole seconds; the WAV header stands in if it did not report a duration.
    duration_ms = int(raw.get("audioDurationMs") or round(header_seconds * 1000))
    row = ledger.record(
        "voice", VOICE_MODEL, note=mode, run_id=run_id, audio_seconds=duration_ms / 1000, seconds=round(latency_s, 3)
    )
    return Transcript(
        text=raw.get("transcript") or "",
        turns=_parse_turns(raw, duration_ms),
        duration_ms=duration_ms,
        session_id=raw.get("sessionId"),
        usd=row["usd"],
        raw=raw,
    )
