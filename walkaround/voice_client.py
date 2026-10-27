"""Muse Voice Transcribe, file endpoint, for the renter's narration while filming (M2, F5 part).

POST https://api.meta.ai/v1/asr/transcribe, multipart exactly as the cookbook's transcribe_recording
(06_muse_voice/01_voice_api_fundamentals/utils.py): a "request" part (application/json: {"mode", "model",
"audioEncoding": "WAV", "keywords": [...], "languageBias": [...] only when set}) and an "audio" part (the WAV bytes).
The WAV must be RIFF mono 16-bit PCM at 16 or 24 kHz, at most 10 minutes (400) and 32 MB (413). Response JSON:
sessionId, transcript, audioDurationMs, turns[{turnId, startMs, endMs, transcript, speaker}]; speaker only in
DIARIZATION; turns empty in PUSH_TO_TALK. The server ignores an unknown mode and transcribes in the default mode, so
the mode string is validated here before any wire call.

What this module does not do: invent timing. Muse Voice gives turn-level timestamps only (no word timing), and the
10/26 probe saw ENDPOINTING return ONE turn from 0.5 to 15.3 s for a 16.6 s two-speaker clip. transcribe_wav
reports what came back; mode_applied is false when a speech mode returned a transcript with no turns (section 14
item 16b: ledger flag "mode not applied", the transcript shows without pins, never a fake whole-clip turn).
For the one-long-turn case, split_turn is the documented helper: when a turn spans more than one panel window it
is cut into sentence groups where a new panel noun from the keyword list begins, the turn's time is spread over the
pieces in proportion to their character positions, and every piece is flagged estimated. The narration lane (M10)
decides when to use it.
"""

from __future__ import annotations

import json
import re
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
from .contracts import PANEL_NOUNS, Noun, VoiceRecord, VoiceRequest, VoiceResponse

# config/thresholds.json nouns.voice_keywords: the 14 panel nouns plus 7 words (21, under the cookbook's 50 cap).
# tests/test_voice_client.py asserts this list equals the thresholds file.
DEFAULT_KEYWORDS = [
    "car",
    "car door",
    "bumper",
    "wheel",
    "hood",
    "trunk lid",
    "fender",
    "roof",
    "headlight",
    "tail light",
    "side mirror",
    "windshield",
    "rear window",
    "car window",
    "scuff",
    "scratch",
    "chip",
    "dent",
    "crack",
    "tape",
    "odometer",
]

# The server ignores an unknown mode instead of rejecting it, so the client checks the name first.
MODES = ("PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION")
DEFAULT_MODE = "ENDPOINTING"  # thresholds voice.mode
SAMPLE_RATES = (16000, 24000)
MAX_AUDIO_SECONDS = 10 * 60
MAX_BYTES = 32 * 1024 * 1024
TIMEOUT_S = 300.0  # the cookbook's value: a 10 minute recording needs a generous wait
MAX_RETRY_WAIT_S = 60.0
ENGINE = "meta-model-api"
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
    estimated: bool = False  # True only for a piece split_turn cut from a longer turn (times spread by text position)


@dataclass
class Transcript:
    text: str
    turns: list[Turn]
    duration_ms: int
    session_id: str | None = None
    usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)  # the response verbatim
    mode_applied: bool = True
    request: dict[str, Any] = field(default_factory=dict)  # the request JSON part verbatim


# ----------------------------------------------------------------------------------------------------------------
# Validation before the wire
# ----------------------------------------------------------------------------------------------------------------


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


def validate_mode(mode: str) -> str:
    """Exact membership in PUSH_TO_TALK, ENDPOINTING, DIARIZATION; a misspelt mode never reaches the wire."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    return mode


# ----------------------------------------------------------------------------------------------------------------
# The request
# ----------------------------------------------------------------------------------------------------------------


def _request_json(mode: str, keywords: list[str] | None) -> dict[str, Any]:
    """The JSON part, in the cookbook's key order; optional fields only when set so a typo never hides."""
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


def _post(wav_path: Path, wav_bytes: bytes, request_json: dict[str, Any]) -> tuple[httpx.Response, int]:
    """One multipart POST, repeated once after a 429. Returns the response and how many retries it took."""
    headers = {"Authorization": f"Bearer {api_key()}"}
    files = {
        # The params travel as a JSON part, not as form fields (cookbook utils.py).
        "request": (None, json.dumps(request_json), "application/json"),
        "audio": (wav_path.name, wav_bytes, "audio/wav"),
    }
    retries = 0
    with httpx.Client(timeout=httpx.Timeout(TIMEOUT_S), transport=TRANSPORT) as client:
        response = client.post(VOICE_TRANSCRIBE_URL, headers=headers, files=files)
        if response.status_code == 429:
            _sleep(_retry_after_s(response))
            retries = 1
            response = client.post(VOICE_TRANSCRIBE_URL, headers=headers, files=files)
    return response, retries


def _check_status(response: httpx.Response) -> None:
    if response.status_code == 200:
        return
    hint = STATUS_HINTS.get(response.status_code, "unexpected response")
    raise TranscribeError(f"[{response.status_code}] {hint}: {response.text[:300]}")


def _parse_turns(raw: dict[str, Any]) -> list[Turn]:
    """The turns as they came. No turn is ever invented: an empty list stays empty (16b)."""
    return [
        Turn(int(t.get("startMs") or 0), int(t.get("endMs") or 0), t.get("transcript") or "", t.get("speaker"))
        for t in raw.get("turns") or []
        if isinstance(t, dict)
    ]


def transcribe_wav(
    wav_path: Path, *, mode: str = DEFAULT_MODE, keywords: list[str] | None = None, run_id: str | None = None
) -> Transcript:
    """One multipart POST; one 429 retry honouring Retry-After. Records a ledger row (kind voice, audio_seconds from
    audioDurationMs, request_id the sessionId, status 'mode not applied' when a speech mode returned no turns)."""
    wav_path = Path(wav_path)
    mode = validate_mode(mode)
    header_seconds = check_wav(wav_path)
    request = _request_json(mode, DEFAULT_KEYWORDS if keywords is None else keywords)
    started = time.monotonic()
    response, retries = _post(wav_path, wav_path.read_bytes(), request)
    ms = int(round((time.monotonic() - started) * 1000))
    _check_status(response)
    try:
        raw = response.json()
    except ValueError as exc:
        raise TranscribeError(f"[200] response was not JSON: {response.text[:300]}") from exc
    if not isinstance(raw, dict):
        raise TranscribeError(f"[200] response was not a JSON object: {response.text[:300]}")
    # The server meters whole seconds; the WAV header stands in if it did not report a duration.
    duration_ms = int(raw.get("audioDurationMs") or round(header_seconds * 1000))
    text = raw.get("transcript") or ""
    turns = _parse_turns(raw)
    mode_applied = not (mode != "PUSH_TO_TALK" and not turns and bool(text.strip()))
    session_id = raw.get("sessionId")
    row = ledger.record(
        "voice",
        VOICE_MODEL,
        note=mode,
        run_id=run_id,
        audio_seconds=duration_ms / 1000,
        engine=ENGINE,
        request_id=str(session_id) if session_id is not None else None,
        unit="seconds",
        quantity=duration_ms / 1000,
        latency_ms={"upload": None, "first_delta": None, "done": ms},
        status="completed" if mode_applied else "mode not applied",
        retries=retries,
    )
    return Transcript(
        text=text,
        turns=turns,
        duration_ms=duration_ms,
        session_id=str(session_id) if session_id is not None else None,
        usd=float(row["usd"]),
        raw=raw,
        mode_applied=mode_applied,
        request=request,
    )


def voice_record(transcript: Transcript, *, role: str, wav_sha256: str) -> VoiceRecord:
    """voice.json: the request part and the response verbatim, mode_applied, dollars."""
    raw = dict(transcript.raw)
    if raw.get("turns") is None:
        raw["turns"] = []
    if raw.get("audioDurationMs") is None:
        raw["audioDurationMs"] = transcript.duration_ms
    if raw.get("transcript") is None:
        raw["transcript"] = transcript.text
    return VoiceRecord(
        role=role,
        request=VoiceRequest.model_validate(transcript.request),
        response=VoiceResponse.model_validate(raw),
        wav_sha256=wav_sha256,
        mode_applied=transcript.mode_applied,
        dollars=transcript.usd,
    )


# ----------------------------------------------------------------------------------------------------------------
# Keywords in a turn and the one-long-turn split
# ----------------------------------------------------------------------------------------------------------------

# Spoken forms of the keyword list. Longer phrases win over shorter ones at the same spot ("rear window" before the
# bare "window" of "car window").
_ALIASES: dict[str, tuple[str, ...]] = {
    "car door": ("car door", "car doors", "door", "doors"),
    "bumper": ("bumper", "bumpers"),
    "wheel": ("wheel", "wheels", "rim", "rims", "tire", "tires", "tyre", "tyres", "hubcap", "hubcaps"),
    "hood": ("hood", "bonnet"),
    "trunk lid": ("trunk lid", "trunk", "boot lid", "boot"),
    "fender": ("fender", "fenders", "quarter panel", "quarter panels"),
    "roof": ("roof",),
    "headlight": ("headlight", "headlights", "head light", "head lights", "headlamp", "headlamps"),
    "tail light": ("tail light", "tail lights", "taillight", "taillights", "tail lamp", "tail lamps", "brake light", "brake lights"),
    "side mirror": ("side mirror", "side mirrors", "wing mirror", "wing mirrors", "mirror", "mirrors"),
    "windshield": ("windshield", "windscreen"),
    "rear window": ("rear window", "back window", "rear windscreen", "back glass"),
    "car window": ("car window", "car windows", "window", "windows"),
    "scuff": ("scuff", "scuffs", "scuffed", "scuffing"),
    "scratch": ("scratch", "scratches", "scratched"),
    "chip": ("chip", "chips", "chipped"),
    "dent": ("dent", "dents", "dented", "ding", "dings"),
    "crack": ("crack", "cracks", "cracked"),
    "tape": ("tape", "taped"),
    "odometer": ("odometer", "mileage", "odo"),
}
# Panel nouns that start a new piece in split_turn: the SAM lane nouns minus car (the whole car) and the privacy nouns.
SPLIT_NOUNS: frozenset[str] = frozenset(n.value for n in PANEL_NOUNS if n is not Noun.CAR)


def _patterns(keyword: str) -> re.Pattern[str]:
    forms = _ALIASES.get(keyword, (keyword, keyword + "s"))
    alternatives = "|".join(re.escape(f).replace(r"\ ", r"\s+") for f in sorted(forms, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


def keyword_hits(text: str, keywords: list[str] | None = None) -> list[str]:
    """Keywords heard in text, in order of first occurrence, spelled as the list spells them. A longer phrase claims
    its span first, so 'rear window' is not also a 'car window'."""
    matches: list[tuple[int, int, str]] = []
    for keyword in DEFAULT_KEYWORDS if keywords is None else keywords:
        for m in _patterns(keyword).finditer(text):
            matches.append((m.start(), m.end(), keyword))
    matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    taken: list[tuple[int, int]] = []
    hits: list[str] = []
    for start, end, keyword in matches:
        if any(start < t_end and end > t_start for t_start, t_end in taken):
            continue
        taken.append((start, end))
        if keyword not in hits:
            hits.append(keyword)
    return hits


def _overlap_ms(turn: Turn, window: tuple[int, int]) -> int:
    return max(0, min(turn.end_ms, window[1]) - max(turn.start_ms, window[0]))


def split_turn(turn: Turn, *, windows: list[tuple[int, int]] | None = None, keywords: list[str] | None = None) -> list[Turn]:
    """Cut one long turn where a sentence names a panel the current piece does not, when the turn spans more than
    one panel window (windows in ms; None means split on the words alone).

    There are no word timestamps, so each piece's start and end are the turn's span spread by character position and
    the piece is flagged estimated. A turn inside one window, or naming at most one panel, comes back unchanged.
    """
    if windows is not None and sum(1 for w in windows if _overlap_ms(turn, w) > 0) < 2:
        return [turn]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", turn.text.strip()) if s]
    pieces: list[list[str]] = []
    current: set[str] = set()
    for sentence in sentences:
        panels = {k for k in keyword_hits(sentence, keywords) if k in SPLIT_NOUNS}
        if pieces and panels and current and not (panels & current):
            pieces.append([sentence])
            current = panels
            continue
        if not pieces:
            pieces.append([])
        pieces[-1].append(sentence)
        current |= panels
    if len(pieces) < 2:
        return [turn]
    texts = [" ".join(piece) for piece in pieces]
    total = sum(len(t) for t in texts) + len(texts) - 1
    span = turn.end_ms - turn.start_ms
    out: list[Turn] = []
    cursor = 0
    for i, text in enumerate(texts):
        start = turn.start_ms + round(span * cursor / total)
        cursor += len(text)
        end = turn.end_ms if i == len(texts) - 1 else turn.start_ms + round(span * cursor / total)
        cursor += 1  # the space between pieces
        out.append(Turn(start, end, text, turn.speaker, estimated=True))
    return out


def split_long_turns(turns: list[Turn], *, windows: list[tuple[int, int]] | None = None, keywords: list[str] | None = None) -> list[Turn]:
    """split_turn over a list, in order."""
    out: list[Turn] = []
    for turn in turns:
        out.extend(split_turn(turn, windows=windows, keywords=keywords))
    return out


__all__ = [
    "DEFAULT_KEYWORDS",
    "DEFAULT_MODE",
    "MODES",
    "SPLIT_NOUNS",
    "TRANSPORT",
    "Transcript",
    "TranscribeError",
    "Turn",
    "check_wav",
    "keyword_hits",
    "split_long_turns",
    "split_turn",
    "transcribe_wav",
    "validate_mode",
    "voice_record",
]
