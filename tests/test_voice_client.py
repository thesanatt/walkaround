"""Offline tests for walkaround.voice_client: fake transport, generated WAVs, the ledger on a temp path.

The one live test runs only with LIVE=1 (the WAV of the cookbook speech sample, about $0.001)."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import wave
from datetime import datetime, timedelta, timezone
from email.parser import BytesParser
from email.policy import HTTP
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from walkaround import ledger, voice_client
from walkaround.config import VOICE_MODEL, VOICE_TRANSCRIBE_URL
from walkaround.contracts import LedgerRow, VoiceMode, VoiceRecord, canonical_json, load_thresholds
from walkaround.voice_client import TranscribeError, Turn, check_wav, split_turn, transcribe_wav, validate_mode, voice_record

FAKE_KEY = "LLM_test_team_test_secret"
SPEECH_CLIP = Path(
    "/private/tmp/claude-501/-Users-sanat-meta2026/086cb85f-3e5e-4a16-a188-2d31e59c8048/scratchpad/ext/"
    "meta-model-cookbook/07_sam/03_headlocked_speech_bubble_captions/assets/sample-input.mp4"
)

TURNS_RESPONSE = {
    "sessionId": "ABC123",
    "transcript": "Small scuff on the right rear door. Chip on the hood.",
    "audioDurationMs": 80000,
    "turns": [
        {"turnId": 0, "startMs": 1200, "endMs": 3400, "transcript": "Small scuff on the right rear door."},
        {"turnId": 1, "startMs": 12400, "endMs": 14100, "transcript": "Chip on the hood."},
    ],
}

# The 10/26 probe: ENDPOINTING on a 16.6 s two-speaker clip came back as one turn from 0.54 to 15.34 s.
ONE_LONG_TURN = {
    "sessionId": "LONG1",
    "transcript": "Small scuff on the right rear door. Chip on the hood. Okay. The rear window is fine.",
    "audioDurationMs": 16560,
    "turns": [
        {
            "turnId": 0,
            "startMs": 540,
            "endMs": 15340,
            "transcript": "Small scuff on the right rear door. Chip on the hood. Okay. The rear window is fine.",
        }
    ],
}


def write_wav(path: Path, *, rate: int = 24000, channels: int = 1, width: int = 2, seconds: float = 1.0) -> Path:
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(bytes(frames * channels * width))
    return path


def write_wav_header_only(path: Path, *, rate: int, seconds: int) -> Path:
    """A RIFF header whose data chunk claims `seconds` of mono 16-bit audio with no samples behind it."""
    data_bytes = rate * 2 * seconds
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, rate, rate * 2, 2, 16)
    data = struct.pack("<4sI", b"data", data_bytes)
    riff = struct.pack("<4sI4s", b"RIFF", 4 + len(fmt) + len(data) + data_bytes, b"WAVE")
    path.write_bytes(riff + fmt + data)
    return path


class FakeServer:
    """httpx.MockTransport handler that records requests and answers from a queue (an exception in it is raised)."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def multipart_parts(request: httpx.Request) -> dict[str, tuple[str | None, str, bytes]]:
    """name -> (filename, content type, payload) for each part of a multipart body."""
    raw = b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.content
    message = BytesParser(policy=HTTP).parsebytes(raw)
    return {
        part.get_param("name", header="content-disposition"): (
            part.get_filename(),
            part.get_content_type(),
            part.get_payload(decode=True),
        )
        for part in message.iter_parts()
    }


def request_json(request: httpx.Request) -> dict:
    return json.loads(multipart_parts(request)["request"][2])


def serve(monkeypatch, *responses: httpx.Response | Exception) -> FakeServer:
    server = FakeServer(*responses)
    monkeypatch.setattr(voice_client, "TRANSPORT", httpx.MockTransport(server))
    return server


@pytest.fixture
def ledger_path(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(ledger, "LEDGER_PATH", path)
    monkeypatch.setenv("MODEL_API_KEY", FAKE_KEY)
    return path


@pytest.fixture
def good_wav(tmp_path):
    return write_wav(tmp_path / "narration.wav", seconds=1.5)


# WAV validation ----------------------------------------------------------------------------------------------------


def test_check_wav_accepts_24k_mono(tmp_path):
    assert check_wav(write_wav(tmp_path / "ok.wav", rate=24000, seconds=2.0)) == pytest.approx(2.0)


def test_check_wav_accepts_16k_mono(tmp_path):
    assert check_wav(write_wav(tmp_path / "ok16.wav", rate=16000, seconds=0.5)) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"channels": 2}, "2 channels"),
        ({"width": 1}, "8-bit"),
        ({"rate": 44100}, "44100 Hz"),
    ],
)
def test_check_wav_rejects_bad_format(tmp_path, kwargs, message):
    bad = write_wav(tmp_path / "bad.wav", **kwargs)
    with pytest.raises(ValueError, match=message) as info:
        check_wav(bad)
    assert "ffmpeg -i" in str(info.value) and "-ar 24000 -c:a pcm_s16le" in str(info.value)


def test_check_wav_rejects_non_wav(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"ID3\x03\x00 not a wav at all")
    with pytest.raises(ValueError, match="not a PCM WAV"):
        check_wav(junk)


def test_check_wav_rejects_empty_audio(tmp_path):
    with pytest.raises(ValueError, match="no audio samples"):
        check_wav(write_wav(tmp_path / "empty.wav", seconds=0))


def test_check_wav_rejects_long_audio(tmp_path):
    long = write_wav_header_only(tmp_path / "long.wav", rate=16000, seconds=601)
    with pytest.raises(ValueError, match="10 minutes"):
        check_wav(long)
    assert check_wav(write_wav_header_only(tmp_path / "max.wav", rate=16000, seconds=600)) == 600.0


def test_check_wav_rejects_big_file(tmp_path):
    big = write_wav(tmp_path / "big.wav")
    os.truncate(big, voice_client.MAX_BYTES + 1)
    with pytest.raises(ValueError, match="32 MB"):
        check_wav(big)


# Mode and keywords -------------------------------------------------------------------------------------------------


def test_validate_mode_is_exact():
    for mode in voice_client.MODES:
        assert validate_mode(mode) == mode
    assert set(voice_client.MODES) == {m.value for m in VoiceMode} == set(load_thresholds()["voice"]["modes_allowed"])
    for bad in ("endpointing", "ENDPOINTING ", "Endpointing", "DIARISATION", ""):
        with pytest.raises(ValueError, match="mode must be one of"):
            validate_mode(bad)


def test_bad_mode_sends_nothing(monkeypatch, ledger_path, good_wav):
    server = serve(monkeypatch)
    with pytest.raises(ValueError, match="mode must be one of"):
        transcribe_wav(good_wav, mode="endpointing")
    assert server.requests == [] and ledger.rows(ledger_path) == []


def test_default_keywords_are_the_21_of_thresholds():
    th = load_thresholds()
    assert voice_client.DEFAULT_KEYWORDS == th["nouns"]["voice_keywords"]
    assert len(voice_client.DEFAULT_KEYWORDS) == 21
    assert voice_client.DEFAULT_MODE == th["voice"]["mode"] == "ENDPOINTING"


def test_constants_mirror_thresholds():
    th = load_thresholds()
    assert voice_client.TIMEOUT_S == th["orchestrator"]["request_timeout_s"]
    assert voice_client.MAX_AUDIO_SECONDS == th["voice"]["max_audio_s"]
    assert voice_client.MAX_BYTES == th["voice"]["max_body_bytes"]
    assert VOICE_MODEL == th["voice"]["model"]
    assert th["media"]["wav_rate"] in voice_client.SAMPLE_RATES


# transcribe_wav ----------------------------------------------------------------------------------------------------


def test_transcribe_turns_record_and_ledger_row(monkeypatch, ledger_path, good_wav):
    serve(monkeypatch, httpx.Response(200, json=TURNS_RESPONSE))
    result = transcribe_wav(good_wav, run_id="gate1")
    assert result.turns == [
        Turn(1200, 3400, "Small scuff on the right rear door."),
        Turn(12400, 14100, "Chip on the hood."),
    ]
    assert result.text == TURNS_RESPONSE["transcript"]
    assert result.duration_ms == 80000
    assert result.session_id == "ABC123"
    assert result.raw == TURNS_RESPONSE
    assert result.mode_applied is True
    assert result.request == {"mode": "ENDPOINTING", "model": VOICE_MODEL, "audioEncoding": "WAV", "keywords": voice_client.DEFAULT_KEYWORDS}
    rows = ledger.rows(ledger_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "voice" and row["model"] == VOICE_MODEL and row["note"] == "ENDPOINTING" and row["run_id"] == "gate1"
    assert row["audio_seconds"] == 80.0 and row["quantity"] == 80.0 and row["unit"] == "seconds"
    assert row["usd"] == pytest.approx(80 * 0.18 / 3600, abs=1e-6) and result.usd == row["usd"]
    assert row["request_id"] == "ABC123" and row["engine"] == "meta-model-api"
    assert row["status"] == "completed" and row["retries"] == 0 and row["latency_ms"]["done"] >= 0
    LedgerRow.model_validate(row)


def test_whole_seconds_are_billed(monkeypatch, ledger_path, good_wav):
    # ledger.cost_for floors to whole seconds; the price page only says "metered in whole seconds". TODO(D1): check one
    # voice row against the billing page; if Meta rounds up, ledger.py moves to math.ceil and this test with it.
    serve(monkeypatch, httpx.Response(200, json={**TURNS_RESPONSE, "audioDurationMs": 80900}))
    result = transcribe_wav(good_wav)
    assert result.usd == pytest.approx(80 * 0.18 / 3600, abs=1e-6)
    assert ledger.rows(ledger_path)[0]["audio_seconds"] == 80.9


def test_push_to_talk_has_no_turns_and_no_invented_turn(monkeypatch, ledger_path, good_wav):
    body = {"sessionId": "S", "transcript": "Scuff on the door.", "audioDurationMs": 4000, "turns": []}
    serve(monkeypatch, httpx.Response(200, json=body))
    result = transcribe_wav(good_wav, mode="PUSH_TO_TALK")
    assert result.turns == [] and result.text == "Scuff on the door."
    assert result.mode_applied is True  # PUSH_TO_TALK never carries turns, so nothing failed
    row = ledger.rows(ledger_path)[0]
    assert row["note"] == "PUSH_TO_TALK" and row["status"] == "completed"


def test_endpointing_with_empty_turns_is_mode_not_applied(monkeypatch, ledger_path, good_wav):
    body = {"sessionId": "S2", "transcript": "Dent on the hood.", "audioDurationMs": 2500}  # no turns key at all
    serve(monkeypatch, httpx.Response(200, json=body))
    result = transcribe_wav(good_wav)
    assert result.turns == [] and result.text == "Dent on the hood."
    assert result.mode_applied is False
    assert ledger.rows(ledger_path)[0]["status"] == "mode not applied"


def test_diarization_speaker_label(monkeypatch, ledger_path, good_wav):
    body = {
        "sessionId": "D",
        "transcript": "Hi. Hello.",
        "audioDurationMs": 2000,
        "turns": [
            {"turnId": 0, "startMs": 0, "endMs": 900, "transcript": "Hi.", "speaker": "A"},
            {"turnId": 1, "startMs": 1000, "endMs": 1900, "transcript": "Hello.", "speaker": "B"},
        ],
    }
    server = serve(monkeypatch, httpx.Response(200, json=body))
    result = transcribe_wav(good_wav, mode="DIARIZATION")
    assert result.turns == [Turn(0, 900, "Hi.", "A"), Turn(1000, 1900, "Hello.", "B")]
    assert result.mode_applied is True
    assert request_json(server.requests[0])["mode"] == "DIARIZATION"


def test_silence_gives_no_turns_and_counts_as_applied(monkeypatch, ledger_path, good_wav):
    serve(monkeypatch, httpx.Response(200, json={"sessionId": "S", "transcript": "", "audioDurationMs": 1500, "turns": []}))
    result = transcribe_wav(good_wav)
    assert result.turns == [] and result.text == "" and result.mode_applied is True


def test_duration_falls_back_to_wav_header(monkeypatch, ledger_path, good_wav):
    serve(monkeypatch, httpx.Response(200, json={"transcript": "x", "turns": []}))
    result = transcribe_wav(good_wav)
    assert result.duration_ms == 1500 and result.session_id is None
    row = ledger.rows(ledger_path)[0]
    assert row["audio_seconds"] == 1.5 and row["request_id"] is None


def test_413_raises_clear_error(monkeypatch, ledger_path, good_wav):
    serve(monkeypatch, httpx.Response(413, text="Payload Too Large"))
    with pytest.raises(TranscribeError, match=r"\[413\] body over 32 MB: Payload Too Large"):
        transcribe_wav(good_wav)
    assert ledger.rows(ledger_path) == []


def test_other_errors_carry_status_and_body(monkeypatch, ledger_path, good_wav):
    serve(monkeypatch, httpx.Response(401, text="bad token"))
    with pytest.raises(TranscribeError, match=r"\[401\] the credential was not accepted: bad token"):
        transcribe_wav(good_wav)


def test_error_bodies_never_carry_the_key(monkeypatch, ledger_path, good_wav):
    """A 401 that echoes the bearer key, or any key-shaped token, never reaches the exception text."""
    body = f"invalid credential {FAKE_KEY}; header was Bearer {FAKE_KEY}; also LLM_123_SECRETSECRET"
    serve(monkeypatch, httpx.Response(401, text=body), httpx.Response(200, text=f"<html>{FAKE_KEY}</html>"))
    for expected in (r"\[401\] the credential was not accepted: invalid credential \[redacted\]", r"\[200\] response was not JSON: <html>\[redacted\]"):
        with pytest.raises(TranscribeError, match=expected) as info:
            transcribe_wav(good_wav)
        assert FAKE_KEY not in str(info.value) and "secret" not in str(info.value).lower()


@pytest.mark.parametrize("error", [httpx.ConnectError("connection refused"), httpx.ReadTimeout("timed out")])
def test_transport_error_retries_once_then_succeeds(monkeypatch, ledger_path, good_wav, error):
    waits = []
    monkeypatch.setattr(voice_client, "_sleep", waits.append)
    server = serve(monkeypatch, error, httpx.Response(200, json=TURNS_RESPONSE))
    result = transcribe_wav(good_wav)
    assert len(result.turns) == 2 and len(server.requests) == 2 and waits == [voice_client.TRANSPORT_RETRY_WAIT_S]
    assert ledger.rows(ledger_path)[0]["retries"] == 1


def test_transport_error_twice_is_a_transcribe_error(monkeypatch, ledger_path, good_wav):
    monkeypatch.setattr(voice_client, "_sleep", lambda seconds: None)
    server = serve(monkeypatch, httpx.ConnectError("connection refused"), httpx.ReadTimeout("timed out"))
    with pytest.raises(TranscribeError, match="no response after one retry: ReadTimeout: timed out") as info:
        transcribe_wav(good_wav)
    assert isinstance(info.value, RuntimeError) and isinstance(info.value.__cause__, httpx.ReadTimeout)
    assert len(server.requests) == 2 and ledger.rows(ledger_path) == []


def test_missing_key_is_a_transcribe_error(monkeypatch, ledger_path, good_wav):
    def no_key() -> str:
        raise RuntimeError("MODEL_API_KEY is not set.")

    monkeypatch.setattr(voice_client, "api_key", no_key)
    server = serve(monkeypatch)
    with pytest.raises(TranscribeError, match="MODEL_API_KEY is not set"):
        transcribe_wav(good_wav)
    assert server.requests == [] and ledger.rows(ledger_path) == []


def test_non_json_200_raises(monkeypatch, ledger_path, good_wav):
    serve(monkeypatch, httpx.Response(200, text="<html>"))
    with pytest.raises(TranscribeError, match="not JSON"):
        transcribe_wav(good_wav)
    assert ledger.rows(ledger_path) == []


def test_multipart_request_shape_matches_the_cookbook(monkeypatch, ledger_path, good_wav):
    server = serve(monkeypatch, httpx.Response(200, json=TURNS_RESPONSE))
    transcribe_wav(good_wav, keywords=["scuff", "hood"])
    (request,) = server.requests
    assert request.method == "POST"
    assert str(request.url) == VOICE_TRANSCRIBE_URL
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    parts = multipart_parts(request)
    assert list(parts) == ["request", "audio"]
    filename, ctype, payload = parts["request"]
    assert (filename, ctype) == (None, "application/json")
    assert json.loads(payload) == {
        "mode": "ENDPOINTING",
        "model": VOICE_MODEL,
        "audioEncoding": "WAV",
        "keywords": ["scuff", "hood"],
    }
    assert list(json.loads(payload)) == ["mode", "model", "audioEncoding", "keywords"]
    filename, ctype, payload = parts["audio"]
    assert (filename, ctype) == ("narration.wav", "audio/wav")
    assert payload == good_wav.read_bytes()
    assert payload[:4] == b"RIFF"


def test_default_keywords_and_empty_keywords(monkeypatch, ledger_path, good_wav):
    server = serve(monkeypatch, httpx.Response(200, json=TURNS_RESPONSE), httpx.Response(200, json=TURNS_RESPONSE))
    transcribe_wav(good_wav)
    second = transcribe_wav(good_wav, keywords=[])
    first_json, second_json = (request_json(r) for r in server.requests)
    assert first_json["keywords"] == voice_client.DEFAULT_KEYWORDS
    assert "keywords" not in second_json and "languageBias" not in first_json
    assert second.request == second_json


def test_language_bias_when_set(monkeypatch, ledger_path, good_wav):
    monkeypatch.setattr(voice_client, "LANGUAGE_BIAS", ["english"])
    server = serve(monkeypatch, httpx.Response(200, json=TURNS_RESPONSE))
    result = transcribe_wav(good_wav)
    assert request_json(server.requests[0])["languageBias"] == ["english"]
    assert result.request["languageBias"] == ["english"]


def test_429_retries_once_honoring_retry_after(monkeypatch, ledger_path, good_wav):
    waits = []
    monkeypatch.setattr(voice_client, "_sleep", waits.append)
    server = serve(monkeypatch, httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(200, json=TURNS_RESPONSE))
    result = transcribe_wav(good_wav)
    assert len(result.turns) == 2 and len(server.requests) == 2 and waits == [3.0]
    rows = ledger.rows(ledger_path)
    assert len(rows) == 1 and rows[0]["retries"] == 1


def test_429_twice_raises(monkeypatch, ledger_path, good_wav):
    monkeypatch.setattr(voice_client, "_sleep", lambda seconds: None)
    server = serve(monkeypatch, httpx.Response(429), httpx.Response(429, text="slow down"))
    with pytest.raises(TranscribeError, match=r"\[429\] rate limited; back off and retry: slow down"):
        transcribe_wav(good_wav)
    assert len(server.requests) == 2 and ledger.rows(ledger_path) == []


def test_retry_after_parsing():
    soon = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=5), usegmt=True)
    assert 3.0 <= voice_client._retry_after_s(httpx.Response(429, headers={"Retry-After": soon})) <= 5.0
    assert voice_client._retry_after_s(httpx.Response(429, headers={"Retry-After": "9999"})) == voice_client.MAX_RETRY_WAIT_S
    assert voice_client._retry_after_s(httpx.Response(429, headers={"Retry-After": "0"})) == 0.5
    assert voice_client._retry_after_s(httpx.Response(429)) == 1.0
    assert voice_client._retry_after_s(httpx.Response(429, headers={"Retry-After": "soon"})) == 1.0


def test_bad_wav_sends_nothing(monkeypatch, ledger_path, tmp_path):
    server = serve(monkeypatch)
    with pytest.raises(ValueError, match="2 channels"):
        transcribe_wav(write_wav(tmp_path / "stereo.wav", channels=2))
    assert server.requests == [] and ledger.rows(ledger_path) == []


# voice_record --------------------------------------------------------------------------------------------------------


def test_voice_record_is_verbatim_and_canonical(monkeypatch, ledger_path, good_wav):
    serve(monkeypatch, httpx.Response(200, json={**TURNS_RESPONSE, "extraServerField": {"k": 1}}))
    transcript = transcribe_wav(good_wav, run_id="r1")
    record = voice_record(transcript, role="pickup", wav_sha256="b" * 64)
    assert isinstance(record, VoiceRecord)
    assert record.role == "pickup" and record.mode_applied is True and record.dollars == transcript.usd
    assert record.request.mode is VoiceMode.ENDPOINTING and record.request.keywords == voice_client.DEFAULT_KEYWORDS
    assert record.request.languageBias is None and record.request.audioEncoding == "WAV"
    assert record.response.sessionId == "ABC123" and record.response.audioDurationMs == 80000
    assert [t.transcript for t in record.response.turns] == ["Small scuff on the right rear door.", "Chip on the hood."]
    assert record.response.turns[0].turnId == 0 and record.response.turns[0].speaker is None
    assert record.response.model_dump()["extraServerField"] == {"k": 1}  # verbatim keeps what the server added
    text = canonical_json(record)
    assert text.isascii() and VoiceRecord.from_json(text).digest() == record.digest()


def test_numeric_speaker_is_kept_as_its_string(monkeypatch, ledger_path, good_wav):
    """contracts.VoiceTurn holds speaker as a string; the real DIARIZATION type is unmeasured until E10."""
    body = {
        "sessionId": "D2",
        "transcript": "Hi. Hello.",
        "audioDurationMs": 2000,
        "turns": [
            {"turnId": 0, "startMs": 0, "endMs": 900, "transcript": "Hi.", "speaker": 0},
            {"turnId": 1, "startMs": 1000, "endMs": 1900, "transcript": None, "speaker": 1},
        ],
    }
    serve(monkeypatch, httpx.Response(200, json=body))
    result = transcribe_wav(good_wav, mode="DIARIZATION")
    assert [t.speaker for t in result.turns] == ["0", "1"] and result.turns[1].text == ""
    record = voice_record(result, role="pickup", wav_sha256="f" * 64)
    assert [t.speaker for t in record.response.turns] == ["0", "1"] and record.response.turns[1].transcript == ""
    assert result.raw["turns"][0]["speaker"] == 0  # the response in memory stays as it came


def test_voice_record_mode_not_applied(monkeypatch, ledger_path, good_wav):
    serve(monkeypatch, httpx.Response(200, json={"sessionId": "S2", "transcript": "Dent on the hood.", "audioDurationMs": 2500}))
    transcript = transcribe_wav(good_wav)
    record = voice_record(transcript, role="return", wav_sha256="c" * 64)
    assert record.mode_applied is False and record.response.turns == [] and record.response.transcript == "Dent on the hood."


# keyword hits and the one-long-turn split --------------------------------------------------------------------------------


def test_keyword_hits_spoken_forms_and_longest_phrase_wins():
    text = "Small scuff on the right rear door. Chip on the hood. The rear window is fine, taillights too, and a tiny ding on the trunk."
    assert voice_client.keyword_hits(text) == ["scuff", "car door", "chip", "hood", "rear window", "tail light", "dent", "trunk lid"]
    assert voice_client.keyword_hits("the window has a crack") == ["car window", "crack"]
    assert voice_client.keyword_hits("nothing to see") == []
    assert voice_client.keyword_hits("bumpers and wheels scratched", ["bumper", "wheel"]) == ["bumper", "wheel"]
    assert voice_client.keyword_hits("the odometer reads 40 thousand") == ["odometer"]


def test_split_turn_cuts_one_long_turn_at_new_panel_nouns():
    turn = Turn(**{"start_ms": 540, "end_ms": 15340, "text": ONE_LONG_TURN["turns"][0]["transcript"]})
    windows = [(0, 6000), (6000, 12000), (12000, 16000)]
    pieces = split_turn(turn, windows=windows)
    assert [p.text for p in pieces] == ["Small scuff on the right rear door.", "Chip on the hood. Okay.", "The rear window is fine."]
    assert all(p.estimated for p in pieces)
    assert pieces[0].start_ms == 540 and pieces[-1].end_ms == 15340
    for earlier, later in zip(pieces, pieces[1:]):
        assert earlier.end_ms <= later.start_ms
    assert voice_client.keyword_hits(pieces[0].text) == ["scuff", "car door"]
    assert voice_client.keyword_hits(pieces[1].text) == ["chip", "hood"]
    # Time is spread by character position: the first piece covers about a third of the span.
    span = 15340 - 540
    assert abs((pieces[0].end_ms - pieces[0].start_ms) - span * len(pieces[0].text) / (len(turn.text))) < 400


def test_split_turn_leaves_a_turn_alone_inside_one_window_or_naming_one_panel():
    turn = Turn(540, 15340, ONE_LONG_TURN["turns"][0]["transcript"])
    assert split_turn(turn, windows=[(0, 16000)]) == [turn]
    assert split_turn(turn, windows=[(0, 400), (15500, 16000)]) == [turn]
    one_panel = Turn(0, 5000, "There is a scuff here on the door. A scratch too. Both on the door.")
    assert split_turn(one_panel) == [one_panel]
    no_panel = Turn(0, 5000, "Okay. Starting the walk now. Everything looks fine.")
    assert split_turn(no_panel) == [no_panel]
    assert split_turn(Turn(0, 1000, "")) == [Turn(0, 1000, "")]


def test_split_turn_without_windows_splits_on_words_alone_and_keeps_speaker():
    turn = Turn(0, 10000, "Scratch on the bumper. Now the left wheel, a scuff on the rim.", speaker="A")
    pieces = split_turn(turn)
    assert [p.text for p in pieces] == ["Scratch on the bumper.", "Now the left wheel, a scuff on the rim."]
    assert all(p.speaker == "A" and p.estimated for p in pieces)
    assert voice_client.split_long_turns([turn, Turn(10000, 11000, "Done.")]) == pieces + [Turn(10000, 11000, "Done.")]


def test_split_turn_on_the_probe_shape_end_to_end(monkeypatch, ledger_path, good_wav):
    serve(monkeypatch, httpx.Response(200, json=ONE_LONG_TURN))
    transcript = transcribe_wav(good_wav)
    assert len(transcript.turns) == 1 and transcript.mode_applied is True  # one long turn is still a turn
    pieces = voice_client.split_long_turns(transcript.turns, windows=[(0, 6000), (6000, 12000), (12000, 16000)])
    assert len(pieces) == 3


# The real response of 10/27 as a fixture --------------------------------------------------------------------------------------


def test_real_endpointing_one_turn_fixture(monkeypatch, ledger_path, good_wav):
    """The ENDPOINTING answer of 10/27 on the cookbook's 16.6 s two-speaker clip
    (tests/fixtures/voice/endpointing_one_turn_live_2026-10-27.json): one turn, so pins need split_turn or E10."""
    data = json.loads((Path(__file__).parent / "fixtures" / "voice" / "endpointing_one_turn_live_2026-10-27.json").read_text())
    serve(monkeypatch, httpx.Response(200, json=data))
    result = transcribe_wav(good_wav, run_id="fixture")
    assert result.duration_ms == 16560 and result.session_id == data["sessionId"]
    assert len(result.turns) == 1 and result.turns[0] == Turn(460, 15660, data["turns"][0]["transcript"])
    assert result.mode_applied is True and result.usd == pytest.approx(16 * 0.18 / 3600, abs=1e-6)
    record = voice_record(result, role="pickup", wav_sha256="e" * 64)
    assert record.response.turns[0].turnId == 0 and record.response.transcript == data["transcript"]
    # No panel noun is spoken, so the split leaves the turn alone even across three windows.
    assert voice_client.keyword_hits(result.text) == []
    assert voice_client.split_long_turns(result.turns, windows=[(0, 6000), (6000, 12000), (12000, 16560)]) == result.turns


# Live smoke --------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="live Voice call; set LIVE=1 (about $0.001)")
def test_live_transcribe_the_speech_sample(tmp_path):
    assert SPEECH_CLIP.is_file(), f"speech sample missing: {SPEECH_CLIP}"
    wav = tmp_path / "sample.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(SPEECH_CLIP), "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-map_metadata", "-1", str(wav)],
        check=True,
    )
    seconds = check_wav(wav)
    result = transcribe_wav(wav, run_id="smoke_m2")
    print("live voice:", result.duration_ms, "ms,", len(result.turns), "turns, mode_applied", result.mode_applied, "usd", result.usd)
    print("live voice turns:", [(t.start_ms, t.end_ms, t.text[:60]) for t in result.turns])
    assert result.text.strip(), result.raw
    assert result.duration_ms > 0 and abs(result.duration_ms / 1000 - seconds) < 2.0
    assert result.request["mode"] == "ENDPOINTING" and result.request["keywords"] == voice_client.DEFAULT_KEYWORDS
    record = voice_record(result, role="pickup", wav_sha256="d" * 64)
    assert record.response.transcript == result.text
