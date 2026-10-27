"""The thin SAM 3.1 wire reader: boxes, ids, c and raw mask payloads from a lane, no decode at parse time.

Grammar (meta-sam/protocol/sam3.md): one line per frame, `<Nf>` (optional `;fields` ignored with a warning) then
comma-separated records `id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..[;c=..]|><|mask;x=0;y=0[;c=..];data=H,W,<payload>|>`.
Token bodies are split on `;`, never searched for `key=`. The payload runs from its marker (`~` lossless, `!` one_bit)
to the closing `|>` and stays in the lane text; MaskRef only records where it sits. meta_sam_parser decodes every mask
while parsing (about 0.1 s per door-sized mask), which is why this reader exists (ARCHITECTURE section 5 step 5 and
16.26); the parser stays the conformance oracle (oracle_index) and the one decoder (decode_mask, DecodePool).

Index shape: one IndexFrame per run of accepted lines with the same frame index, in output order. Frames may skip
and may come back later (forward then backward propagation); a later mask for the same id and frame replaces the
earlier one when rendered. c is the box token's c; None means absent, never zero.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import numpy as np
from meta_sam_parser import (
    InvalidSegmentationMaskError,
    SegmentationMask,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)

from .contracts import Diagnostic, IndexFrame, IndexObject, LaneStatus, MaskEncoding, MaskRef

Media = Literal["image", "video"]

# The parser's grammar, ported pattern for pattern so both readers accept and reject the same lines.
_MAX_SAFE = (1 << 53) - 1
_JS_WS = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_LINE_CHAR = r"[^\n\r\u2028\u2029]"
_HEADER = re.compile(rf"^<([0-9]+)f(?:;([^>]*))?>({_LINE_CHAR}*)$")
_RECORD_START = re.compile(r",?([0-9]+)(?=<\|)")
_TOKEN = re.compile(r"<\|([A-Za-z][A-Za-z0-9_.-]*)(?:;([^|]*))?\|>")
_BOUNDARY = frozenset(",0123456789")
_SIGNED_INT = re.compile(r"-?[0-9]+")
_UNSIGNED_INT = re.compile(r"[0-9]+")
_MASK_DATA = re.compile(r"([0-9]+),([0-9]+),([!~][^|]+)")
_CONFIDENCE = re.compile(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_BOX_KEYS = frozenset({"x1", "y1", "x2", "y2", "w", "h", "c"})
_MASK_KEYS = frozenset({"x", "y", "data", "c"})
_HEADER_KEYS: frozenset[str] = frozenset()

PARSER_DIAGNOSTIC_CODES = frozenset(
    {
        "malformed_record",
        "invalid_frame",
        "unexpected_frame",
        "invalid_box",
        "invalid_mask_size",
        "ignored_field",
        "ignored_token",
        "ignored_confidence",
    }
)
# Codes only this reader emits. invalid_mask_payload is the one parser error it cannot emit: payloads are checked
# at decode time, and decode_mask raises InvalidSegmentationMaskError then.
READER_DIAGNOSTIC_CODES = frozenset(
    {"text_line", "response_lane", "output_after_done", "missing_text", "finalizer_conflict", "repeated_finalizer"}
)


# ----------------------------------------------------------------------------------------------------------------
# Events on disk
# ----------------------------------------------------------------------------------------------------------------


def read_events(path: Path) -> Iterator[dict[str, Any]]:
    """One JSON event per line from a .ndjson or .ndjson.gz lane file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ----------------------------------------------------------------------------------------------------------------
# Lane text and outcome from the Responses events
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class LaneOutcome:
    status: LaneStatus
    response_id: str | None
    store_echo: bool | None
    video_frames_processed: int | None
    usage: dict[str, Any]
    incomplete_reason: str | None
    error: str | None
    diagnostics: list[Diagnostic]
    text_chars: int


@dataclass
class _Collected:
    text: str = ""
    saw_delta: bool = False
    completed: bool = False
    response_id: str | None = None
    store_echo: bool | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    incomplete_reason: str | None = None
    error: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)


def _lane_of(event: dict[str, Any]) -> tuple[str, int, int] | None:
    item_id = event.get("item_id")
    output_index = event.get("output_index")
    content_index = event.get("content_index")
    ok = (
        isinstance(item_id, str)
        and item_id
        and isinstance(output_index, int)
        and not isinstance(output_index, bool)
        and isinstance(content_index, int)
        and not isinstance(content_index, bool)
        and 0 <= output_index <= _MAX_SAFE
        and 0 <= content_index <= _MAX_SAFE
    )
    return (item_id, output_index, content_index) if ok else None


def _nested_str(value: Any, *keys: str) -> str | None:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, str) else None


class _Collector:
    """Walks the Responses events once: lane identity, deltas, finalizers, terminal event, usage."""

    def __init__(self) -> None:
        self.out = _Collected()
        self._parts: list[str] = []
        self._lane: tuple[str, int, int] | None = None
        self._done = False
        self._done_text: str | None = None
        self._finalizers: set[str] = set()
        self._flagged: set[str] = set()

    def _flag(self, severity: Literal["warning", "error"], code: str, message: str) -> None:
        if code in self._flagged:
            return
        self._flagged.add(code)
        self.out.diagnostics.append(Diagnostic(severity=severity, code=code, message=message))

    def _accept_lane(self, event: dict[str, Any]) -> bool:
        lane = _lane_of(event)
        if lane is None:
            self._flag("error", "response_lane", "The output text event has an invalid lane identity.")
            return False
        if self._lane is None:
            self._lane = lane
        elif self._lane != lane:
            self._flag("error", "response_lane", "The response interleaved multiple output text lanes.")
            return False
        return True

    def _delta(self, event: dict[str, Any]) -> None:
        if not self._accept_lane(event):
            return
        if self._done:
            self._flag("error", "output_after_done", "The response emitted output text after finalizing its lane.")
            return
        delta = event.get("delta")
        if not isinstance(delta, str):
            self._flag("error", "missing_text", "The output text delta is missing its text.")
            return
        self.out.saw_delta = True
        self._parts.append(delta)

    def _finalize(self, event: dict[str, Any], text: Any) -> None:
        if not self._accept_lane(event):
            return
        if not isinstance(text, str):
            self._flag("error", "missing_text", "The completed output text is missing its text.")
            return
        kind = str(event.get("type"))
        current = "".join(self._parts) if self.out.saw_delta else (self._done_text or "")
        if self._done:
            if text != current:
                self._flag("error", "finalizer_conflict", "The response finalized its output text lane twice with conflicting text.")
            elif kind in self._finalizers:
                self._flag("warning", "repeated_finalizer", f"The response repeated its {kind} finalizer with identical text.")
            self._finalizers.add(kind)
            return
        if self.out.saw_delta and text != current:
            self._flag("error", "finalizer_conflict", "The finalized output text conflicts with its accumulated deltas.")
        elif not self.out.saw_delta:
            self._done_text = text
        self._done = True
        self._finalizers.add(kind)

    def _terminal(self, kind: str, event: dict[str, Any]) -> None:
        response = event.get("response")
        if kind == "response.completed":
            self.out.completed = True
            usage = response.get("usage") if isinstance(response, dict) else None
            if isinstance(usage, dict):
                self.out.usage = usage
        elif kind == "response.incomplete":
            self.out.incomplete_reason = _nested_str(response, "incomplete_details", "reason") or "response"
        elif kind == "response.failed":
            self.out.error = _nested_str(response, "error", "message") or "The response failed."
        elif kind == "error":
            message = _nested_str(event, "error", "message")
            if message is None:
                message = _nested_str(event, "message")
            self.out.error = message if message else "The Responses stream reported an error."

    def push(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if not isinstance(kind, str):
            return
        response = event.get("response")
        if isinstance(response, dict):
            if isinstance(response.get("id"), str):
                self.out.response_id = response["id"]
            if isinstance(response.get("store"), bool):
                self.out.store_echo = response["store"]
        if kind == "response.output_text.delta":
            self._delta(event)
        elif kind == "response.output_text.done":
            self._finalize(event, event.get("text"))
        elif kind == "response.content_part.done":
            part = event.get("part")
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type == "output_text":
                self._finalize(event, part.get("text"))
            elif part_type == "refusal":
                refusal = part.get("refusal")
                self.out.error = f"refused: {refusal if isinstance(refusal, str) else 'The response was refused.'}"
        elif kind in ("response.refusal.delta", "response.refusal.done"):
            text = event.get("refusal") if isinstance(event.get("refusal"), str) else event.get("delta")
            self.out.error = f"refused: {text if isinstance(text, str) else 'The response was refused.'}"
        elif kind in ("response.completed", "response.incomplete", "response.failed", "error"):
            self._terminal(kind, event)

    def finish(self) -> _Collected:
        self.out.text = "".join(self._parts) if self.out.saw_delta else (self._done_text or "")
        return self.out


def _collect(events: Iterable[dict[str, Any]]) -> _Collected:
    collector = _Collector()
    for event in events:
        collector.push(event)
    return collector.finish()


def lane_text(events: Iterable[dict[str, Any]]) -> str:
    """The deltas concatenated in order; with no deltas, the finalizer text; "" for a zero-match lane."""
    return _collect(events).text


# ----------------------------------------------------------------------------------------------------------------
# The line grammar
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pending:
    code: str
    subject: str
    message: str


@dataclass(frozen=True)
class _Fields:
    values: dict[str, str | None]
    offsets: dict[str, int]  # offset of each value's first character inside the token body
    repeated_confidence: bool


@dataclass(frozen=True)
class _Token:
    name: str
    body: str | None
    body_offset: int  # offset of the body inside the line body


@dataclass(frozen=True)
class _Record:
    object_id: str
    box: tuple[str, str, str, str, str, str]
    mask_h: int
    mask_w: int
    payload: str
    payload_offset: int  # inside the line body
    box_c: float | None


def _trim(value: str) -> str:
    return value.strip(_JS_WS)


def _leading(value: str) -> int:
    return len(value) - len(value.lstrip(_JS_WS))


def _safe_int(text: str) -> int | None:
    value = int(text)
    return value if abs(value) <= _MAX_SAFE else None


def _read_fields(body: str | None, owner: str, known: frozenset[str], warnings: list[_Pending]) -> _Fields | None:
    """Split one `;` field list by name. None means a repeated key other than c made the record malformed."""
    values: dict[str, str | None] = {}
    offsets: dict[str, int] = {}
    repeated_confidence = False
    position = 0
    for part in [] if body is None else body.split(";"):
        start = position
        position += len(part) + 1
        text = _trim(part)
        if not text:
            continue
        key_text, separator, value_text = text.partition("=")
        key = _trim(key_text)
        value = _trim(value_text) if separator else None
        if key not in known:
            name = key if key else text
            warnings.append(_Pending("ignored_field", f"{owner}:{name}", f'Ignored unknown {owner} field "{name}".'))
            continue
        if key in values:
            if key != "c":
                return None
            repeated_confidence = True
            continue
        values[key] = value
        offsets[key] = start + _leading(part) + len(key_text) + 1 + _leading(value_text)
    return _Fields(values, offsets, repeated_confidence)


def _read_confidence(fields: _Fields, warnings: list[_Pending]) -> float | None:
    if "c" not in fields.values:
        return None
    text = fields.values["c"]
    value = (
        float(text)
        if not fields.repeated_confidence and text is not None and _CONFIDENCE.fullmatch(text) is not None
        else math.nan
    )
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        warnings.append(_Pending("ignored_confidence", "c", "Ignored a confidence value that is not a number from 0 through 1."))
        return None
    return value + 0.0


def _read_record(object_id: str, tokens: list[_Token], warnings: list[_Pending]) -> _Record | None:
    box_tokens = [t for t in tokens if t.name == "box"]
    mask_tokens = [t for t in tokens if t.name == "mask"]
    if len(box_tokens) != 1 or len(mask_tokens) != 1:
        return None
    for token in tokens:
        if token.name not in ("box", "mask"):
            warnings.append(_Pending("ignored_token", token.name, f'Ignored unknown token "{token.name}".'))
    box = _read_fields(box_tokens[0].body, "box", _BOX_KEYS, warnings)
    mask = _read_fields(mask_tokens[0].body, "mask", _MASK_KEYS, warnings)
    if box is None or mask is None:
        return None
    coords = tuple(box.values.get(k) for k in ("x1", "y1", "x2", "y2"))
    dims = tuple(box.values.get(k) for k in ("w", "h"))
    if (
        any(v is None or _SIGNED_INT.fullmatch(v) is None for v in coords)
        or any(v is None or _UNSIGNED_INT.fullmatch(v) is None for v in dims)
        or mask.values.get("x") != "0"
        or mask.values.get("y") != "0"
    ):
        return None
    data = _MASK_DATA.fullmatch(mask.values.get("data") or "")
    if data is None:
        return None
    box_c = _read_confidence(box, warnings)
    _read_confidence(mask, warnings)  # the mask token's c is validated for its warning but not indexed
    return _Record(
        object_id=object_id,
        box=coords + dims,  # type: ignore[arg-type]
        mask_h=int(data.group(1)),
        mask_w=int(data.group(2)),
        payload=data.group(3),
        payload_offset=mask_tokens[0].body_offset + mask.offsets["data"] + data.start(3),
        box_c=box_c,
    )


class _Reader:
    """parse_lane_text's state: the frames so far, diagnostics, the once-per-stream warning set."""

    def __init__(self, media: Media) -> None:
        self.media = media
        self.frames: list[IndexFrame] = []
        self.diagnostics: list[Diagnostic] = []
        self._warned: set[tuple[str, str]] = set()
        self._line = 0

    def accept(self, raw: str, offset: int) -> None:
        """One line. offset is where raw starts in the lane text."""
        self._line += 1
        line = _trim(raw)
        if not line:
            return
        if line.startswith("<"):
            header = _HEADER.fullmatch(line)
            if header is not None:
                self._accept_api_line(header, offset + _leading(raw))
                return
        self._diagnose("warning", "text_line", "Kept a non-record line as text.")

    def _diagnose(self, severity: Literal["warning", "error"], code: str, message: str) -> None:
        self.diagnostics.append(Diagnostic(severity=severity, code=code, message=message, line=self._line))

    def _warn(self, pending: list[_Pending]) -> None:
        for item in pending:
            key = (item.code, item.subject)
            if key in self._warned:
                continue
            self._warned.add(key)
            self._diagnose("warning", item.code, item.message)

    def _accept_api_line(self, header: re.Match[str], line_offset: int) -> None:
        frame_index = _safe_int(header.group(1))
        if frame_index is None or frame_index < 0:
            self._diagnose("error", "invalid_frame", "Frame references must be non-negative safe integers.")
            return
        if self.media == "image" and frame_index != 0:
            self._diagnose("error", "unexpected_frame", "Image segmentation records require frame zero.")
            return
        header_warnings: list[_Pending] = []
        _read_fields(header.group(2), "frame header", _HEADER_KEYS, header_warnings)
        body = header.group(3)
        body_offset = line_offset + header.start(3)
        if not body:
            self._diagnose("error", "malformed_record", "Malformed SAM API object record.")
            return
        position = 0
        while position < len(body):
            start = _RECORD_START.match(body, position)
            tokens: list[_Token] = []
            end = position if start is None else start.end()
            token = None if start is None else _TOKEN.match(body, end)
            while token is not None:
                tokens.append(_Token(token.group(1), token.group(2), token.start(2) if token.group(2) is not None else token.end()))
                end = token.end()
                token = _TOKEN.match(body, end)
            record_warnings: list[_Pending] = []
            record = (
                None
                if start is None or (end < len(body) and body[end] not in _BOUNDARY)
                else _read_record(start.group(1), tokens, record_warnings)
            )
            if record is None:
                self._diagnose("error", "malformed_record", "Malformed SAM API object record.")
                return
            values = tuple(_safe_int(v) for v in record.box)
            if any(v is None for v in values):
                self._diagnose("error", "invalid_box", "SAM API box coordinates are invalid.")
                return
            left, top, right_in, bottom_in, width, height = values  # type: ignore[misc]
            if (
                width <= 0
                or height <= 0
                or left < 0
                or top < 0
                or right_in < left
                or bottom_in < top
                or right_in >= width
                or bottom_in >= height
            ):
                self._diagnose("error", "invalid_box", "SAM API box coordinates are invalid.")
                return
            self._warn(header_warnings)
            header_warnings = []
            self._warn(record_warnings)
            mask: MaskRef | None = None
            area = record.mask_w * record.mask_h
            if record.mask_w <= 0 or record.mask_h <= 0 or area > _MAX_SAFE:
                self._diagnose("error", "invalid_mask_size", "Mask dimensions must be positive.")
            else:
                mask = MaskRef(
                    h=record.mask_h,
                    w=record.mask_w,
                    encoding=MaskEncoding.LOSSLESS if record.payload.startswith("~") else MaskEncoding.ONE_BIT,
                    payload_offset=body_offset + record.payload_offset,
                    payload_len=len(record.payload),
                )
            obj = IndexObject(id=record.object_id, box=[left, top, right_in + 1, bottom_in + 1], c=record.box_c, mask=mask)
            frame = frame_index if self.media == "video" else 0
            if self.frames and self.frames[-1].frame == frame:
                self.frames[-1].objects.append(obj)
            else:
                self.frames.append(IndexFrame(frame=frame, objects=[obj]))
            position = end


def parse_lane_text(text: str, *, media: Media) -> tuple[list[IndexFrame], list[Diagnostic]]:
    """The thin reader. Malformed lines become diagnostics, never exceptions; nothing is decoded."""
    reader = _Reader(media)
    offset = 0
    segments = text.split("\n")
    for i, segment in enumerate(segments):
        last = i == len(segments) - 1
        if last and not segment:
            break  # an empty tail after the final newline is not a line
        raw = segment[:-1] if segment.endswith("\r") else segment
        reader.accept(raw, offset)
        offset += len(segment) + 1
    return reader.frames, reader.diagnostics


def index_from_events(events: Iterable[dict[str, Any]], *, media: Media) -> tuple[list[IndexFrame], LaneOutcome]:
    """Index plus outcome from the raw Responses events of one lane (live or replayed)."""
    collected = _collect(events)
    frames, diagnostics = parse_lane_text(collected.text, media=media)
    diagnostics = collected.diagnostics + diagnostics
    processed = collected.usage.get("video_frames_processed")
    if isinstance(processed, bool) or not isinstance(processed, int):
        processed = None
    has_error = any(d.severity == "error" for d in diagnostics)
    incomplete_reason = collected.incomplete_reason
    if collected.error is not None or incomplete_reason is not None:
        status = LaneStatus.INCOMPLETE
    elif not collected.completed:
        status = LaneStatus.INCOMPLETE
        incomplete_reason = "eof"
    elif has_error:
        status = LaneStatus.INCOMPLETE
    elif not frames:
        status = LaneStatus.ZERO_MATCHES
    else:
        status = LaneStatus.COMPLETED
    outcome = LaneOutcome(
        status=status,
        response_id=collected.response_id,
        store_echo=collected.store_echo,
        video_frames_processed=processed,
        usage=collected.usage,
        incomplete_reason=incomplete_reason,
        error=collected.error,
        diagnostics=diagnostics,
        text_chars=len(collected.text),
    )
    return frames, outcome


# ----------------------------------------------------------------------------------------------------------------
# Masks: slice, decode, paste
# ----------------------------------------------------------------------------------------------------------------


def mask_payload(text: str, ref: MaskRef) -> str:
    """The payload (marker included) of one mask, sliced from the lane text."""
    return text[ref.payload_offset : ref.payload_offset + ref.payload_len]


def _raster_to_array(raster: bytes, h: int, w: int) -> np.ndarray:
    return np.frombuffer(raster, dtype=np.uint8).reshape(h, w).astype(bool)


def decode_mask(ref: MaskRef, payload: str) -> np.ndarray:
    """Bool array (ref.h, ref.w) through meta_sam_parser.decode_mask_to_raster; InvalidSegmentationMaskError raises through."""
    mask = SegmentationMask(encoding=ref.encoding.value, payload=payload, width=ref.w, height=ref.h)
    return _raster_to_array(decode_mask_to_raster(mask), ref.h, ref.w)


def paste_mask(raster: np.ndarray, box: list[int], ref: MaskRef, media_size: tuple[int, int]) -> np.ndarray:
    """Bool (H, W) frame with the box-local raster at (box x1 + ref.x, box y1 + ref.y), clipped to the frame."""
    width, height = media_size
    out = np.zeros((height, width), dtype=bool)
    x0 = int(box[0]) + ref.x
    y0 = int(box[1]) + ref.y
    h, w = raster.shape
    x1, y1 = min(x0 + w, width), min(y0 + h, height)
    sx, sy = max(0, -x0), max(0, -y0)
    x0, y0 = max(x0, 0), max(y0, 0)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = raster[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
    return out


def _decode_job(job: tuple[int, int, str, str]) -> tuple[bytes | None, float, str | None]:
    """Runs in a worker process: (raster bytes, CPU seconds, error message)."""
    h, w, encoding, payload = job
    start = time.process_time()
    try:
        raster = decode_mask_to_raster(SegmentationMask(encoding=encoding, payload=payload, width=w, height=h))  # type: ignore[arg-type]
    except InvalidSegmentationMaskError as error:
        return None, time.process_time() - start, str(error)
    return raster, time.process_time() - start, None


class DecodePool:
    """ProcessPoolExecutor over decode_mask: the only place a mask is decoded in the request path."""

    def __init__(self, workers: int) -> None:
        self._pool = ProcessPoolExecutor(max_workers=max(1, int(workers)))
        self._decoded = 0
        self._cpu_s = 0.0

    def decode(self, lane_path: Path, refs: list[MaskRef]) -> list[np.ndarray]:
        """Rasters for refs, reading the lane text once. A bad payload raises InvalidSegmentationMaskError."""
        if not refs:
            return []
        text = lane_text(read_events(lane_path))
        jobs = [(ref.h, ref.w, ref.encoding.value, mask_payload(text, ref)) for ref in refs]
        out: list[np.ndarray] = []
        for ref, (raster, cpu_s, error) in zip(refs, self._pool.map(_decode_job, jobs)):
            self._cpu_s += cpu_s
            if raster is None:
                raise InvalidSegmentationMaskError(error or "Mask payload could not be decoded.")
            self._decoded += 1
            out.append(_raster_to_array(raster, ref.h, ref.w))
        return out

    def stats(self) -> tuple[int, float]:
        """(masks decoded, CPU seconds spent decoding) since construction."""
        return self._decoded, self._cpu_s

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> DecodePool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ----------------------------------------------------------------------------------------------------------------
# The oracle (tests only)
# ----------------------------------------------------------------------------------------------------------------


async def parse_with_oracle(events: Iterable[dict[str, Any]], *, media: Media) -> Any:
    """meta_sam_parser's final result for the events; its errors raise through."""

    async def source() -> Any:
        for event in events:
            yield event

    fmt = image_segmentation_format() if media == "image" else video_segmentation_format()
    parsed = parse_responses_stream(source(), fmt)
    async with parsed:
        return await parsed.final_result()


def oracle_index(events: Iterable[dict[str, Any]], *, media: Media) -> list[IndexFrame]:
    """Tests only: the same index through meta_sam_parser. MaskRef offsets are 0 and payload_len the payload length."""
    import asyncio

    result = asyncio.run(parse_with_oracle(list(events), media=media))
    frames: list[IndexFrame] = []
    for record in result.records:
        if record.kind == "text":
            continue  # prose lines carry no frame; the reader reports them as a text_line warning
        frame = 0 if record.frame is None else record.frame.frame_index
        if record.kind == "box":
            obj = IndexObject(
                id=record.object_id,
                box=[int(record.left), int(record.top), int(record.right), int(record.bottom)],
                c=record.confidence,
            )
            if frames and frames[-1].frame == frame:
                frames[-1].objects.append(obj)
            else:
                frames.append(IndexFrame(frame=frame, objects=[obj]))
        elif record.kind == "mask":
            target = next((o for o in reversed(frames[-1].objects) if o.id == record.object_id), None) if frames else None
            if target is not None and frames[-1].frame == frame:
                target.mask = MaskRef(
                    h=record.mask.height,
                    w=record.mask.width,
                    encoding=MaskEncoding(record.mask.encoding),
                    payload_offset=0,
                    payload_len=len(record.mask.payload),
                )
    return frames
