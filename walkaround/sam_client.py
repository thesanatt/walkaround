"""SAM 3.1 through the Meta Model API: Files upload, one concept per streamed request, raw lanes, the thin index.

Request shape (docs/API_NOTES.md; cookbook 07_sam/01_api_basics/python/sam_api.py):
  POST /v1/responses, model "sam-3.1", stream=True, store=False,
  input=[{"type":"message","role":"user","content":[{"type":"input_text","text":prompt}, media]}]
  media = {"type":"input_video","file_id":...} | {"type":"input_video","video_url":data_url}
        | {"type":"input_image","image_url":data_url}
  metadata={"include_confidence":"true","mask_encoding":"one_bit"} on a text lane;
  {"include_confidence":"true","propagation_direction":"both"} on a box or point prompt (sam_metadata builds both).
The live API emits response.content_part.done and may or may not emit response.output_text.done; stream_events (the
cookbook shim, verbatim) synthesizes the latter only when the API did not send one. Every raw event is appended to the
lane's events.ndjson as it arrives; sam_wire reads the index from those events and never decodes a mask. A lane is
never an exception: 429 past the backoff, a broken stream or a refused request end in the LaneRecord's status and
error, so coverage can print not_run. Every answered request writes one ledger row (sam_video bills from
usage.video_frames_processed when the API reports it, else frames_sent, with the API's 50-frame minimum).
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import io
import json
import math
import mimetypes
import numbers
import random
import re
import subprocess
import time
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterable, AsyncIterator, Callable, Literal

import httpx
import numpy as np
from meta_sam_parser import (
    FramePrompt,
    ObjectPromptError,
    PromptBox,
    PromptObject,
    PromptPoint,
    ResponsesStreamError,
    SegmentationMask,
    build_object_prompt,
    decode_mask_to_raster,
)
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from . import contracts, ledger, paths, sam_wire
from .config import BASE_URL, SAM_MODEL, api_key
from .contracts import Box4, Diagnostic, FilesApiRecord, IndexFrame, LaneRecord, LaneStatus, Noun, noun_slug

__all__ = [
    "Box",
    "Detection",
    "SegmentationRun",
    "SamRequestError",
    "client",
    "stream_events",
    "sam_metadata",
    "response_input",
    "upload_file",
    "delete_file",
    "delete_response",
    "segment_video",
    "segment_video_many",
    "segment_clip",
    "segment_image",
    "replay",
    "mask_array",
    "full_mask",
    "prompt_box",
    "object_prompt",
    "FramePrompt",
    "PromptBox",
    "PromptObject",
    "PromptPoint",
    "ObjectPromptError",
]

ENGINE = "meta-model-api"
TIMEOUT_S = 240.0  # thresholds orchestrator.request_timeout_s
RATE_LIMIT_TRIES = 6  # thresholds orchestrator.rate_limit_retries
BACKOFF_BASE_S = 0.5  # thresholds orchestrator.rate_limit_backoff_base_s
BACKOFF_CAP_S = 30.0
STREAM_RETRIES = 1  # thresholds orchestrator.request_retries: one more try for a stream that broke or 5xx
FILES_DELETE_RETRIES = 3  # thresholds privacy.files_delete_retries
INLINE_MAX_BYTES = 50 * 1024 * 1024  # thresholds media.inline_data_url_max_bytes
FILES_EXPIRES_AFTER_S = 3600  # thresholds media.files_expires_after_s, the API minimum
JPEG_QUALITY = 90

_sleep = asyncio.sleep  # tests replace it


class SamRequestError(RuntimeError):
    """An image request got no answer from the API after the retries."""


# ----------------------------------------------------------------------------------------------------------------
# Parser-backed structures (images, replay, tests)
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float  # half-open
    bottom: float  # half-open

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass(frozen=True)
class Detection:
    """One object in one frame. mask is box-local (mask.width x mask.height == box size) at (left, top)."""

    object_id: str
    frame_index: int | None  # None for images
    box: Box
    mask: SegmentationMask | None
    confidence: float | None


@dataclass
class SegmentationRun:
    concept: str
    media: str  # "video" | "image"
    media_width: int
    media_height: int
    frames: dict[int, list[Detection]] = field(default_factory=dict)  # frame_index -> detections (images use 0)
    frames_billed: int = 0  # frames the API reported processing (video), >= 50 billed
    usd: float = 0.0
    events_path: Path | None = None
    diagnostics: list[str] = field(default_factory=list)
    outcome: str = "completed"  # completed | incomplete | failed (the parser's outcome or a stream failure)
    response_id: str | None = None

    @property
    def object_ids(self) -> list[str]:
        ids = {d.object_id for dets in self.frames.values() for d in dets}
        return sorted(ids, key=lambda s: (len(s), s))

    def track(self, object_id: str) -> list[Detection]:
        """Detections of one object ordered by frame."""
        return [d for i in sorted(self.frames) for d in self.frames[i] if d.object_id == object_id]


# ----------------------------------------------------------------------------------------------------------------
# Client, shim, request shapes
# ----------------------------------------------------------------------------------------------------------------


def client() -> AsyncOpenAI:
    """One client per call; it does no retries of its own (the lane code owns the 429 backoff)."""
    return AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), timeout=TIMEOUT_S, max_retries=0)


async def stream_events(stream: AsyncIterable[Any]) -> AsyncIterator[dict[str, Any]]:
    """Convert the SDK's typed events to the plain Responses-event dicts the parser
    consumes, and finalize each output-text lane. The API streams
    `response.content_part.done` but not `response.output_text.done`, which the
    parser uses to close a lane, so synthesize it from the content part."""
    completed_lanes: set[tuple[object, object, object]] = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (
            data.get("item_id"),
            data.get("output_index"),
            data.get("content_index"),
        )
        if data.get("type") == "response.output_text.done":
            if lane not in completed_lanes:
                completed_lanes.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in completed_lanes
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            completed_lanes.add(lane)
            yield {
                "type": "response.output_text.done",
                "item_id": data.get("item_id"),
                "output_index": data.get("output_index"),
                "content_index": data.get("content_index"),
                "text": part["text"],
            }


def sam_metadata(
    *,
    include_confidence: bool = True,
    mask_encoding: str | None = "one_bit",
    propagation_direction: str | None = None,
) -> dict[str, str]:
    """The metadata dict of one request; string values. propagation_direction only for object prompts."""
    out: dict[str, str] = {}
    if include_confidence:
        out["include_confidence"] = "true"
    if mask_encoding is not None:
        out["mask_encoding"] = mask_encoding
    if propagation_direction is not None:
        out["propagation_direction"] = propagation_direction
    return out


def response_input(prompt: str, media: dict[str, str]) -> list[dict[str, Any]]:
    phrase = prompt.strip()
    if not phrase:
        raise ValueError("The concept must be a short noun phrase.")
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": phrase}, media],
        }
    ]


def _request(prompt_text: str, media: dict[str, str], metadata: dict[str, str]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": SAM_MODEL,
        "stream": True,
        "store": False,
        "input": response_input(prompt_text, media),
    }
    if metadata:
        body["metadata"] = dict(metadata)
    return body


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso(unix: Any) -> str | None:
    if isinstance(unix, bool) or not isinstance(unix, (int, float)):
        return None
    return datetime.fromtimestamp(float(unix), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _plus_seconds(iso: str, seconds: int) -> str:
    base = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (base + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backoff_delay(attempt: int, retry_after: str | None) -> float:
    """Seconds to wait before try attempt + 1: 500 ms doubling with jitter, never under Retry-After."""
    base = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** (attempt - 1)))
    delay = base * (0.5 + random.random())
    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except ValueError:
            try:
                parsed = datetime.strptime(retry_after, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)
                delay = max(delay, (parsed - datetime.now(timezone.utc)).total_seconds())
            except ValueError:
                pass
    return max(0.0, delay)


def _retry_after(error: APIStatusError) -> str | None:
    try:
        return error.response.headers.get("retry-after")
    except Exception:  # a hand-made error in tests may carry no response
        return None


def _mentions_store(error: APIStatusError) -> bool:
    """True when a 400 names the store parameter (param store, or the word store; never 'restore' or 'stored')."""
    body = error.body if isinstance(error.body, dict) else {}
    inner = body.get("error") if isinstance(body.get("error"), dict) else body
    param = inner.get("param") if isinstance(inner, dict) else None
    return param == "store" or re.search(r"\bstore\b", str(error).lower()) is not None


# ----------------------------------------------------------------------------------------------------------------
# The streaming core shared by video lanes, clips and images
# ----------------------------------------------------------------------------------------------------------------


class _Sink:
    """Receives each raw event as it arrives; reset() forgets a failed attempt."""

    def __init__(self, on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.on_event = on_event

    def __call__(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


class _LaneSink(_Sink):
    """Appends each event to the lane's events.ndjson as it arrives, flushing per event."""

    def __init__(self, path: Path, on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        super().__init__(on_event)
        self.path = Path(path)
        self._fh = self.path.open("w", encoding="utf-8")

    def __call__(self, event: dict[str, Any]) -> None:
        self._fh.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")
        self._fh.flush()
        super().__call__(event)

    def reset(self) -> None:
        self._fh.close()
        self._fh = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()


@dataclass
class _StreamResult:
    events: list[dict[str, Any]]
    first_delta_s: float | None
    elapsed_s: float
    retries: int
    store_sent: bool  # False when the API rejected store false and the field was dropped
    error: str | None


def _has_terminal(events: list[dict[str, Any]]) -> bool:
    return any(e.get("type") in ("response.completed", "response.incomplete", "response.failed", "error") for e in events)


async def _run_stream(request: dict[str, Any], sink: _Sink, *, max_tries: int) -> _StreamResult:
    """POST /v1/responses with stream=True: 429 backoff, the store-false fallback, one retry for a broken stream."""
    rate_tries = 0
    stream_tries = 0
    retries = 0
    store_sent = "store" in request
    while True:
        events: list[dict[str, Any]] = []
        first: float | None = None
        t0 = time.monotonic()
        body = dict(request)
        if not store_sent:
            body.pop("store", None)
        try:
            async with client() as api:
                stream = await api.responses.create(**body)
                # The parser returns on the terminal event, so stream_events is still suspended over the response
                # body. Close it here, before the response, or the loop's asyncgen finalizer aclose()s a generator
                # that is running (cookbook sam_api.py).
                async with stream, aclosing(stream_events(stream)) as evs:
                    async for event in evs:
                        if first is None and event.get("type") == "response.output_text.delta":
                            first = time.monotonic() - t0
                        events.append(event)
                        sink(event)
        except RateLimitError as error:
            rate_tries += 1
            retries += 1
            if rate_tries >= max_tries:
                return _StreamResult(events, first, time.monotonic() - t0, retries, store_sent, f"429 after {rate_tries} tries: {error}")
            sink.reset()
            await _sleep(_backoff_delay(rate_tries, _retry_after(error)))
            continue
        except APIStatusError as error:
            if error.status_code == 400 and store_sent and _mentions_store(error):
                store_sent = False
                retries += 1
                sink.reset()
                continue
            if error.status_code >= 500 and stream_tries < STREAM_RETRIES:
                stream_tries += 1
                retries += 1
                sink.reset()
                continue
            return _StreamResult(events, first, time.monotonic() - t0, retries, store_sent, f"{error.status_code}: {error}")
        except (APIError, httpx.HTTPError, OSError) as error:
            if stream_tries < STREAM_RETRIES and not _has_terminal(events):
                stream_tries += 1
                retries += 1
                sink.reset()
                continue
            return _StreamResult(events, first, time.monotonic() - t0, retries, store_sent, f"{type(error).__name__}: {error}")
        if not _has_terminal(events) and stream_tries < STREAM_RETRIES:
            stream_tries += 1
            retries += 1
            sink.reset()
            continue
        return _StreamResult(events, first, time.monotonic() - t0, retries, store_sent, None)


def _ms(seconds: float | None) -> int | None:
    return None if seconds is None else int(round(seconds * 1000))


# ----------------------------------------------------------------------------------------------------------------
# Files API
# ----------------------------------------------------------------------------------------------------------------


def _load_cache() -> dict[str, Any]:
    path = paths.uploads_cache_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(cache: dict[str, Any]) -> None:
    paths.uploads_cache_path().write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _cache_fresh(entry: dict[str, Any]) -> bool:
    expires = entry.get("expires_at")
    if not isinstance(expires, str):
        return False
    try:
        deadline = datetime.strptime(expires, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return deadline - timedelta(seconds=60) > datetime.now(timezone.utc)


def _remux_faststart(src: Path) -> Path:
    """ffmpeg -c copy into an MP4 with faststart next to the source (section 14 item 7)."""
    dst = src.with_name(src.stem + ".remux.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)],
        check=True,
        capture_output=True,
    )
    return dst


async def _files_create(path: Path, expires_after_s: int | None) -> Any:
    kwargs: dict[str, Any] = {"file": (path.name, path.read_bytes(), "video/mp4"), "purpose": "user_data"}
    if expires_after_s is not None:
        kwargs["expires_after"] = {"anchor": "created_at", "seconds": int(expires_after_s)}
    attempt = 0
    while True:
        try:
            async with client() as api:
                return await api.files.create(**kwargs)
        except RateLimitError as error:
            attempt += 1
            if attempt >= RATE_LIMIT_TRIES:
                raise
            await _sleep(_backoff_delay(attempt, _retry_after(error)))


async def upload_file(path: Path, *, run_id: str | None = None, expires_after_s: int = FILES_EXPIRES_AFTER_S) -> FilesApiRecord:
    """POST /v1/files purpose user_data with expires_after; cached by sha256 in runs/uploads.json."""
    path = Path(path)
    digest = _sha256_file(path)
    cache = _load_cache()
    hit = cache.get(digest)
    if isinstance(hit, dict) and _cache_fresh(hit) and isinstance(hit.get("file_id"), str):
        return FilesApiRecord(file_id=hit["file_id"], uploaded_at=hit["uploaded_at"], expires_at=hit.get("expires_at"))
    t0 = time.monotonic()
    try:
        uploaded = await _files_create(path, expires_after_s)
    except APIStatusError as error:
        if error.status_code == 429 or error.status_code >= 500:
            raise
        if error.status_code == 400 and "expires_after" in str(error):
            uploaded = await _files_create(path, None)
        else:
            uploaded = await _files_create(_remux_faststart(path), expires_after_s)
    file_id = getattr(uploaded, "id", None)
    if not isinstance(file_id, str) or not file_id.startswith("file-"):
        raise ValueError("The Files API returned an invalid file ID.")
    uploaded_at = _now()
    expires_at = _iso(getattr(uploaded, "expires_at", None)) or _plus_seconds(uploaded_at, expires_after_s)
    record = FilesApiRecord(file_id=file_id, uploaded_at=uploaded_at, expires_at=expires_at)
    cache[digest] = {
        "file_id": file_id,
        "uploaded_at": uploaded_at,
        "expires_at": expires_at,
        "seconds": round(time.monotonic() - t0, 3),
        "bytes": path.stat().st_size,
        "name": path.name,
        "run_id": run_id,
    }
    _save_cache(cache)
    return record


def _forget_file(file_id: str) -> None:
    cache = _load_cache()
    kept = {k: v for k, v in cache.items() if not (isinstance(v, dict) and v.get("file_id") == file_id)}
    if len(kept) != len(cache):
        _save_cache(kept)


async def delete_file(file_id: str) -> dict[str, Any]:
    """DELETE /v1/files/{id}; the body verbatim (deleted true on 10/26). 404 returns a body with deleted false."""
    attempt = 0
    while True:
        try:
            async with client() as api:
                deleted = await api.files.delete(file_id)
            body = deleted.model_dump() if hasattr(deleted, "model_dump") else dict(deleted)
            _forget_file(file_id)
            return body
        except APIStatusError as error:
            if error.status_code == 404:
                _forget_file(file_id)
                return {"id": file_id, "deleted": False, "status_code": 404, "error": str(error)}
            if error.status_code != 429 and error.status_code < 500:
                raise
            attempt += 1
            if attempt >= FILES_DELETE_RETRIES:
                raise
            await _sleep(_backoff_delay(attempt, _retry_after(error)))
        except (APIConnectionError, APITimeoutError):
            attempt += 1
            if attempt >= FILES_DELETE_RETRIES:
                raise
            await _sleep(_backoff_delay(attempt, None))


async def delete_response(response_id: str) -> dict[str, Any]:
    """DELETE /v1/responses/{id}; used only when store false was refused. Returns status and body."""
    attempt = 0
    while True:
        try:
            async with client() as api:
                raw = await api.responses.with_raw_response.delete(response_id)
            http = raw.http_response
            text = http.text
            try:
                body: Any = json.loads(text) if text else None
            except json.JSONDecodeError:
                body = text
            return {"id": response_id, "status_code": http.status_code, "deleted": 200 <= http.status_code < 300, "body": body}
        except APIStatusError as error:
            if error.status_code != 429 and error.status_code < 500:
                return {"id": response_id, "status_code": error.status_code, "deleted": False, "error": str(error)}
            attempt += 1
            if attempt >= FILES_DELETE_RETRIES:
                raise
            await _sleep(_backoff_delay(attempt, _retry_after(error)))
        except (APIConnectionError, APITimeoutError):
            attempt += 1
            if attempt >= FILES_DELETE_RETRIES:
                raise
            await _sleep(_backoff_delay(attempt, None))


# ----------------------------------------------------------------------------------------------------------------
# Video lanes
# ----------------------------------------------------------------------------------------------------------------


def _gzip_lane(src: Path, dst: Path) -> str:
    """Gzip the lane with a fixed mtime (deterministic bytes), remove the plain file, return the gz sha256."""
    data = src.read_bytes()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        gz.write(data)
    payload = buffer.getvalue()
    dst.write_bytes(payload)
    src.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _sorted_ids(frames: list[IndexFrame]) -> list[str]:
    ids = {o.id for f in frames for o in f.objects}
    return sorted(ids, key=lambda s: (len(s), s))


def _lane_status(result: _StreamResult, outcome: sam_wire.LaneOutcome) -> LaneStatus:
    if result.error is not None and not result.events:
        return LaneStatus.FAILED
    return outcome.status


async def _segment_lane(
    media: dict[str, str],
    prompt_text: str,
    *,
    run_id: str,
    role: str,
    lane: str,
    frames_sent: int,
    media_size: tuple[int, int],
    metadata: dict[str, str],
    kind: Literal["text", "object"],
    noun: Noun | None,
    request_index: int,
    on_event: Callable[[dict[str, Any]], None] | None,
    max_tries: int,
) -> tuple[LaneRecord, list[IndexFrame]]:
    lane_id = paths.lane_id(lane)
    request = _request(prompt_text, media, metadata)
    events_path = paths.lane_events_path(run_id, role, lane_id)
    sink = _LaneSink(events_path, on_event)
    try:
        result = await _run_stream(request, sink, max_tries=max_tries)
    finally:
        sink.close()
    cpu0 = time.process_time()
    frames, outcome = sam_wire.index_from_events(result.events, media="video")
    parse_cpu_s = time.process_time() - cpu0
    status = _lane_status(result, outcome)
    diagnostics = list(outcome.diagnostics)
    if not result.store_sent:
        diagnostics.append(
            Diagnostic(severity="warning", code="store_false_rejected", message="The API rejected store false; the request was retried without the field.")
        )
    record = LaneRecord(
        lane_id=lane_id,
        role=role,
        kind=kind,
        media="video",
        noun=noun,
        prompt_text=prompt_text,
        media_size=[int(media_size[0]), int(media_size[1])],
        params=dict(metadata),
        store=not result.store_sent,
        store_echo=outcome.store_echo,
        request_id=outcome.response_id,
        request_index=request_index,
        status=status,
        first_delta_s=None if result.first_delta_s is None else round(result.first_delta_s, 3),
        completed_s=round(result.elapsed_s, 3),
        frames_sent=frames_sent,
        frames_returned=len({f.frame for f in frames}),
        video_frames_processed=outcome.video_frames_processed,
        ids=_sorted_ids(frames),
        zero_matches=status == LaneStatus.ZERO_MATCHES,
        parse_cpu_s=round(parse_cpu_s, 4),
        retries=result.retries,
        diagnostics=diagnostics,
        error=result.error or outcome.error,
    )
    if result.events:
        gz_path = paths.lane_events_gz_path(run_id, role, lane_id)
        record.lane_sha256 = _gzip_lane(events_path, gz_path)
        record.lane_path = paths.rel(run_id, gz_path)
        index_path = paths.lane_index_path(run_id, role, lane_id)
        contracts.write_json(index_path, frames)
        # Hand back exactly what the file holds (c rounded to 6 places), so a resume reads the same numbers.
        frames = contracts.read_json_list(index_path, IndexFrame)
        record.index_path = paths.rel(run_id, index_path)
        billed = outcome.video_frames_processed if outcome.video_frames_processed is not None else frames_sent
        row = ledger.record(
            "sam_video",
            SAM_MODEL,
            note=lane_id,
            run_id=run_id,
            frames=billed,
            engine=ENGINE,
            request_id=outcome.response_id,
            unit="frames",
            quantity=billed,
            latency_ms={"upload": None, "first_delta": _ms(result.first_delta_s), "done": _ms(result.elapsed_s)},
            parse_cpu_ms=int(round(parse_cpu_s * 1000)),
            status=status.value,
            retries=result.retries,
            video_frames_processed=outcome.video_frames_processed,
            frames_sent=frames_sent,
            role=role,
        )
        record.dollars = float(row["usd"])
    else:
        events_path.unlink(missing_ok=True)
    contracts.write_json(paths.lane_summary_path(run_id, role, lane_id), record)
    return record, frames


async def segment_video(
    file_id: str,
    prompt_text: str,
    *,
    run_id: str,
    role: str,
    lane: str,
    frames_sent: int,
    media_size: tuple[int, int],
    metadata: dict[str, str],
    kind: Literal["text", "object"] = "text",
    noun: Noun | None = None,
    request_index: int = 0,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    max_tries: int = RATE_LIMIT_TRIES,
) -> tuple[LaneRecord, list[IndexFrame]]:
    """One streamed SAM request on an uploaded video: raw lane, gz, index and summary on disk, one ledger row."""
    if not file_id.startswith("file-"):
        raise ValueError("Expected a Files API ID beginning with 'file-'.")
    return await _segment_lane(
        {"type": "input_video", "file_id": file_id},
        prompt_text,
        run_id=run_id,
        role=role,
        lane=lane,
        frames_sent=frames_sent,
        media_size=media_size,
        metadata=metadata,
        kind=kind,
        noun=noun,
        request_index=request_index,
        on_event=on_event,
        max_tries=max_tries,
    )


_NOUN_BY_SLUG: dict[str, Noun] = {noun_slug(n): n for n in Noun}


def _prompt_kind(prompt_text: str) -> Literal["text", "object"]:
    text = prompt_text.lstrip()
    return "object" if text.startswith("<") and "<|" in text else "text"


def _noun_for(lane: str, prompt_text: str) -> Noun | None:
    if _prompt_kind(prompt_text) == "object":
        return None
    return _NOUN_BY_SLUG.get(paths.lane_id(lane)) or _NOUN_BY_SLUG.get(noun_slug(prompt_text))


def _failed_record(lane: str, role: str, prompt_text: str, media_size: tuple[int, int], metadata: dict[str, str], index: int, error: str) -> LaneRecord:
    return LaneRecord(
        lane_id=paths.lane_id(lane),
        role=role,
        kind=_prompt_kind(prompt_text),
        noun=_noun_for(lane, prompt_text),
        prompt_text=prompt_text,
        media_size=[int(media_size[0]), int(media_size[1])],
        params=dict(metadata),
        request_index=index,
        status=LaneStatus.FAILED,
        error=error,
    )


async def segment_video_many(
    file_id: str,
    prompts: list[tuple[str, str]],
    *,
    run_id: str,
    role: str,
    frames_sent: int,
    media_size: tuple[int, int],
    metadata: dict[str, str],
    concurrency: int = 16,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, tuple[LaneRecord, list[IndexFrame]]]:
    """segment_video for each (lane, prompt_text) in order, under a semaphore; one lane never raises for another."""
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def one(index: int, lane: str, prompt_text: str) -> tuple[str, tuple[LaneRecord, list[IndexFrame]]]:
        callback = (lambda event, lane=lane: on_event(lane, event)) if on_event is not None else None
        async with semaphore:
            try:
                return lane, await segment_video(
                    file_id,
                    prompt_text,
                    run_id=run_id,
                    role=role,
                    lane=lane,
                    frames_sent=frames_sent,
                    media_size=media_size,
                    metadata=metadata,
                    kind=_prompt_kind(prompt_text),
                    noun=_noun_for(lane, prompt_text),
                    request_index=index,
                    on_event=callback,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:  # a bug in one lane must not take the other fifteen down
                record = _failed_record(lane, role, prompt_text, media_size, metadata, index, f"{type(error).__name__}: {error}")
                return lane, (record, [])

    results = await asyncio.gather(*(one(i, lane, text) for i, (lane, text) in enumerate(prompts)))
    return dict(results)


def _probe_size(path: Path) -> tuple[int, int]:
    """Displayed (width, height) of a video after its rotation metadata, through ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        rotation = int(round(float(tag)))
    if abs(rotation) % 180 == 90:
        width, height = height, width
    return width, height


async def segment_clip(
    clip: Path,
    prompt_text: str,
    *,
    run_id: str,
    role: str,
    lane: str,
    media_size: tuple[int, int],
    metadata: dict[str, str],
    frames_sent: int,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[LaneRecord, list[IndexFrame]]:
    """The same request with an inline data URL video (Stays Put and anchor clips, under 50 MB)."""
    clip = Path(clip)
    size = clip.stat().st_size
    if size > INLINE_MAX_BYTES:
        raise ValueError(f"{clip.name} is {size} bytes; inline video is capped at {INLINE_MAX_BYTES}.")
    probed = _probe_size(clip)
    if probed != (int(media_size[0]), int(media_size[1])):
        raise ValueError(f"{clip.name} is {probed[0]}x{probed[1]} after rotation, not {media_size[0]}x{media_size[1]}.")
    data_url = "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode("ascii")
    return await _segment_lane(
        {"type": "input_video", "video_url": data_url},
        prompt_text,
        run_id=run_id,
        role=role,
        lane=lane,
        frames_sent=frames_sent,
        media_size=media_size,
        metadata=metadata,
        kind=_prompt_kind(prompt_text),
        noun=_noun_for(lane, prompt_text),
        request_index=0,
        on_event=on_event,
        max_tries=RATE_LIMIT_TRIES,
    )


# ----------------------------------------------------------------------------------------------------------------
# Images, replay (parser-backed)
# ----------------------------------------------------------------------------------------------------------------


def _image_data_url(image: Path | np.ndarray) -> tuple[str, tuple[int, int]]:
    """A data URL plus (width, height). Arrays are RGB or gray uint8 and go out as JPEG."""
    from PIL import Image

    if isinstance(image, np.ndarray):
        pil = Image.fromarray(np.ascontiguousarray(image))
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), pil.size
    path = Path(image)
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        raise ValueError(f"Could not infer a media type for {path}.")
    with Image.open(path) as pil:
        size = pil.size
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii"), size


def _run_from_result(result: Any, *, media: str, media_size: tuple[int, int], concept: str) -> SegmentationRun:
    run = SegmentationRun(concept=concept, media=media, media_width=int(media_size[0]), media_height=int(media_size[1]))
    last: Detection | None = None
    for record in result.records:
        if record.kind == "text":
            run.diagnostics.append(f"text line: {record.text}")
            continue
        frame = None if record.frame is None else record.frame.frame_index
        key = 0 if frame is None else frame
        if record.kind == "box":
            last = Detection(
                object_id=record.object_id,
                frame_index=frame,
                box=Box(record.left, record.top, record.right, record.bottom),
                mask=None,
                confidence=record.confidence,
            )
            run.frames.setdefault(key, []).append(last)
        elif record.kind == "mask" and last is not None and last.object_id == record.object_id and last.frame_index == frame:
            dets = run.frames[key]
            dets[dets.index(last)] = last = replace(last, mask=record.mask)
    for item in result.diagnostics:
        run.diagnostics.append(f"{item.severity} {item.code} line {item.line}: {item.message}")
    run.outcome = result.outcome.status
    if result.outcome.status == "incomplete":
        detail = getattr(result.outcome, "detail", None)
        run.diagnostics.append(f"incomplete: {getattr(result.outcome, 'reason', 'unknown')}" + (f" ({detail})" if detail else ""))
    return run


def _usage_frames(events: list[dict[str, Any]]) -> tuple[str | None, int | None]:
    collected = sam_wire.index_from_events(events, media="video")[1]
    return collected.response_id, collected.video_frames_processed


async def _run_from_events(events: list[dict[str, Any]], *, media: str, media_size: tuple[int, int], concept: str) -> SegmentationRun:
    try:
        result = await sam_wire.parse_with_oracle(events, media=media)  # type: ignore[arg-type]
    except ResponsesStreamError as error:
        run = SegmentationRun(concept=concept, media=media, media_width=int(media_size[0]), media_height=int(media_size[1]))
        run.outcome = "failed"
        run.diagnostics.append(f"parser {error.code}: {error}")
    else:
        run = _run_from_result(result, media=media, media_size=media_size, concept=concept)
    run.response_id, processed = _usage_frames(events)
    run.frames_billed = processed or 0
    return run


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def segment_image(
    image: Path | np.ndarray,
    concept: str,
    *,
    run_id: str | None,
    role: str | None = None,
    tag: str = "",
    include_confidence: bool = True,
) -> SegmentationRun:
    """One SAM image request (data URL input, store false), parsed with meta_sam_parser. Ledger: sam_image."""
    data_url, media_size = _image_data_url(image)
    metadata = sam_metadata(include_confidence=include_confidence, mask_encoding=None)
    request = _request(concept, {"type": "input_image", "image_url": data_url}, metadata)
    events: list[dict[str, Any]] = []
    result = await _run_stream(request, _Sink(events.append), max_tries=RATE_LIMIT_TRIES)
    if result.error is not None and not events:
        raise SamRequestError(result.error)
    cpu0 = time.process_time()
    run = await _run_from_events(events, media="image", media_size=media_size, concept=concept)
    parse_cpu_s = time.process_time() - cpu0
    if result.error is not None:
        run.outcome = "incomplete" if run.outcome == "completed" else run.outcome
        run.diagnostics.append(result.error)
    if not result.store_sent:
        run.diagnostics.append("store_false_rejected: the request was retried without the field")
    if run_id is not None and role is not None:
        name = f"{tag or 'image'}.{run.response_id or 'no_id'}.ndjson"
        run.events_path = paths.image_requests_dir(run_id, role) / name
        run.events_path.write_text("".join(json.dumps(e, ensure_ascii=True, default=str) + "\n" for e in events), encoding="utf-8")
    row = ledger.record(
        "sam_image",
        SAM_MODEL,
        note=f"{concept} {tag}".strip(),
        run_id=run_id,
        images=1,
        engine=ENGINE,
        request_id=run.response_id,
        unit="images",
        quantity=1,
        latency_ms={"upload": None, "first_delta": _ms(result.first_delta_s), "done": _ms(result.elapsed_s)},
        parse_cpu_ms=int(round(parse_cpu_s * 1000)),
        status=run.outcome,
        retries=result.retries,
        role=role,
    )
    run.usd = float(row["usd"])
    return run


def replay(events_path: Path, *, media: str, media_size: tuple[int, int], concept: str) -> SegmentationRun:
    """Rebuild a SegmentationRun from saved events without any API call (parser-backed)."""
    events = list(sam_wire.read_events(Path(events_path)))
    run = _run_sync(_run_from_events(events, media=media, media_size=media_size, concept=concept))
    run.events_path = Path(events_path)
    return run


def mask_array(det: Detection) -> np.ndarray:
    """Box-local boolean array (mask.height, mask.width) decoded from the lossless or one_bit payload."""
    if det.mask is None:
        raise ValueError(f"object {det.object_id} has no mask on this frame")
    raster = decode_mask_to_raster(det.mask)
    return np.frombuffer(raster, dtype=np.uint8).reshape(det.mask.height, det.mask.width).astype(bool)


def full_mask(det: Detection, media_size: tuple[int, int]) -> np.ndarray:
    """Boolean array (H, W) of the whole frame with the box-local mask pasted at (left, top)."""
    raster = mask_array(det)
    ref = contracts.MaskRef(h=raster.shape[0], w=raster.shape[1], encoding=contracts.MaskEncoding(det.mask.encoding), payload_offset=0, payload_len=0)  # type: ignore[union-attr]
    return sam_wire.paste_mask(raster, [int(det.box.left), int(det.box.top), int(det.box.right), int(det.box.bottom)], ref, media_size)


# ----------------------------------------------------------------------------------------------------------------
# Object prompts
# ----------------------------------------------------------------------------------------------------------------


def prompt_box(box: Box4, media_size: tuple[int, int], *, pad_px: int = 2) -> PromptBox:
    """Section 5 step 6: integer edges padded by pad_px per side, clamped so 0 <= left < right <= w and top < bottom <= h."""
    width, height = int(media_size[0]), int(media_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"media_size must be positive, got {media_size!r}")
    x1, y1, x2, y2 = (float(v) for v in box)
    pad = max(0, int(pad_px))
    left = math.floor(min(x1, x2)) - pad
    top = math.floor(min(y1, y2)) - pad
    right = math.ceil(max(x1, x2)) + pad
    bottom = math.ceil(max(y1, y2)) + pad
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return PromptBox(left=left, top=top, right=right, bottom=bottom)


def _whole(value: Any) -> Any:
    """numpy and other integral types become int; anything else reaches the builder as is, so it can reject it."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    return value


def object_prompt(objects: list[PromptObject], media_size: tuple[int, int]) -> str:
    """build_object_prompt(size=media_size, objects=objects); ObjectPromptError propagates with its .code."""
    return build_object_prompt(size=(_whole(media_size[0]), _whole(media_size[1])), objects=list(objects))
