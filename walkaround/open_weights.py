"""Muse Spark Open, Meta's open-weight model, served by Baseten's OpenAI-compatible Model APIs (S1, lane B11).

Spark Open runs the second look: the call that can only lower a match to needs your eye. Each call is one Chat
Completions request to https://inference.baseten.co/v1 with the hosted look_v1 question (spark_client.look_prompt and
SYSTEM_PROMPT), the images as JPEG data URLs, reasoning_effort set, and max_tokens from the caller (thresholds
spark.max_output_tokens, and retry_max_output_tokens on the retry after a cut). Baseten accepts response_format
json_schema for this model but does not enforce it (https://docs.baseten.co/inference/structured-outputs ), so the
prompt also carries the schema and every answer is checked here with spark_client.schema_problems. Only a top-level
JSON object can be the answer, and two that disagree make the answer unusable. A 400 or 422 whose text names the
response format steps the format down (json_schema, then json_object, then none); any other 400 is final.

Model id: GET /v1/models once per process, then meta-models/muse-spark-open before meta-models/muse-spark-open-1026
(ids the listing shows come first); a 404 moves on to the next id, because other teams saw 404 on one and 200 on the
other. An id that returned 404 and a stepped-down format are tried again after FAILURE_TTL_S, so one bad minute does
not switch Spark Open off for the life of the server. The response's own model field goes on the ledger row and the
provenance line.

Failures: 401, 403, a timeout, a dropped connection, a 400 not about the format, or a 408, 409, 429 or 5xx past three
tries raise OpenWeightsError, and so do two attempts that both came back with no answer (an error object, no
choices); the caller then falls back to hosted Spark 1.3. Every chat request writes one ledger row (kind spark_open)
and, with a run id, one line of runs/<run_id>/open_weights.ndjson in the CONTRACTS.md "Phase 2 file shapes" form:
{ts, run_id, engine, model, request_id, latency_ms, pickup_mark, return_mark, panel_id, verdict, reason, valid,
error}, plus purpose, status, model_requested, http_status and baseten_request_id (the x-baseten-request-id header,
the id Baseten's own logs use). A failed call names the requested model; verdict is set only on a valid answer.
Neither file ever holds image bytes or a key.

The client is injectable for the offline tests: set_client() takes a fake with .models.list() and
.chat.completions.create(**kwargs); reset() forgets the probe, the dead ids, the format and the throttle clock. Inside
a pytest run no real client is built unless LIVE=1, so an offline test that forgot its fake cannot spend credits with
the key in walkaround/.env.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import numpy as np
from openai import APIStatusError, OpenAIError

from . import ledger, paths, spark_client
from .config import load_env, redact
from .contracts import (
    GridPoint,
    InputRef,
    LookVerdict,
    PanelId,
    ReasoningEffort,
    SparkEngine,
    SparkPurpose,
    SparkRequestRecord,
    SparkStatus,
    SparkUsage,
)

BASE_URL = "https://inference.baseten.co/v1"
MODEL_CANDIDATES = ("meta-models/muse-spark-open", "meta-models/muse-spark-open-1026")
LEDGER_KIND = "spark_open"
ENGINE = SparkEngine.BASETEN
DEFAULT_EFFORT = spark_client.DEFAULT_EFFORT
MAX_TOKENS = spark_client.MAX_OUTPUT_TOKENS  # reasoning shares the budget on Baseten too ("max_tokens caps the whole completion")
RETRY_MAX_TOKENS = spark_client.RETRY_MAX_OUTPUT_TOKENS
TIMEOUT_S = 60.0  # 11 looks took 2.4 to 4.5 s (10/31); a hung call costs this twice before the caller's breaker opens
CONNECT_TIMEOUT_S = 10.0
RATE_LIMIT_TRIES = 3  # tries in all on 408, 409, 429, 5xx (529 is Baseten's "overloaded"); hosted Spark is the real retry
BACKOFF_BASE_S = 0.5
MAX_BACKOFF_S = 8.0
MAX_RETRY_AFTER_S = 30.0
FAILURE_TTL_S = 600.0  # an id that returned 404, or a format a 400 refused, is tried again after this
DEFAULT_RPM = 12.0  # thresholds orchestrator.baseten_rpm; the account allows 120 (x-ratelimit-limit-requests, 10/31)
FORMATS = ("json_schema", "json_object", "none")
FORMAT_ERROR = re.compile(r"response_format|json_schema|json_object|schema|guided|structured", re.IGNORECASE)
CUT_FINISH_REASONS = ("length", "content_filter")  # any other finish_reason with content is an answer to check
LOG_TEXT_MAX = 300  # characters of an error kept on a ledger row or a provenance line
REASON_MAX = 200  # a reason on looks.json and open_weights.ndjson: one plain sentence (CONTRACTS.md, Phase 2 file shapes)
REQUEST_ID_HEADER = "x-baseten-request-id"


class OpenWeightsError(RuntimeError):
    """Baseten gave no answer to use. attempts holds the records of the responses that did come back before it failed."""

    def __init__(self, message: str, attempts: list[SparkRequestRecord] | None = None) -> None:
        super().__init__(message)
        self.attempts = list(attempts or [])


@dataclass
class _State:
    probed: bool = False
    listed: list[str] | None = None  # ids from GET /v1/models; None when the listing failed
    answered: str | None = None  # the id that answered last; it leads the next call
    dead: dict[str, float] = field(default_factory=dict)  # id -> clock time of its 404
    fmt: str = FORMATS[0]
    fmt_at: float | None = None  # clock time of the last step down
    last_start: float | None = None


_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()
_STATE = _State()
_CLIENT: Any = None
_sleep = time.sleep  # tests replace both
_clock = time.monotonic


def api_key() -> str:
    """BASETEN_API_KEY from the environment, walkaround/.env or ~/meta2026/.keys."""
    load_env()
    key = os.environ.get("BASETEN_API_KEY")
    if not key:
        raise OpenWeightsError("BASETEN_API_KEY is not set; put it in walkaround/.env")
    return key


def _client() -> Any:
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("LIVE") != "1":
                # an offline test with no fake must never reach Baseten with the real key that load_env finds
                raise OpenWeightsError("no Baseten client in an offline test: inject one with set_client, or set LIVE=1")
            from openai import OpenAI

            # max_retries 0: this module owns the backoff, so a timeout is never repeated by the SDK as well
            timeout = httpx.Timeout(TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
            _CLIENT = OpenAI(base_url=BASE_URL, api_key=api_key(), timeout=timeout, max_retries=0)
        return _CLIENT


def set_client(client: Any) -> None:
    """Inject a client; None goes back to the real one on the next call."""
    global _CLIENT
    _CLIENT = client


def reset() -> None:
    """Forget the model probe, the dead ids, the response format and the throttle clock."""
    global _STATE
    with _LOCK:
        _STATE = _State()


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(vars(obj)) if hasattr(obj, "__dict__") else {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short(text: str) -> str:
    return redact(text)[:LOG_TEXT_MAX]


# Plain text for the two file shapes ---------------------------------------------------------------------------------

_PUNCTUATION = str.maketrans({chr(0x2018): "'", chr(0x2019): "'", chr(0x201C): '"', chr(0x201D): '"', chr(0x2013): " - ", chr(0x2014): " - ", chr(0x2026): "...", chr(0x00A0): " "})  # quotes, dashes, ellipsis, no-break space
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _flat(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text.translate(_PUNCTUATION)).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.split())


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 3]
    return (head.rsplit(" ", 1)[0] if " " in head else head).rstrip(" ,;:") + "..."


def plain_text(text: str, limit: int = REASON_MAX) -> str:
    """text on one line in plain ASCII, cut at a word to at most limit characters."""
    return _cap(_flat(text), limit)


def plain_reason(text: str, limit: int = REASON_MAX) -> str:
    """The first sentence of text in plain ASCII, at most limit characters: the reason field of both file shapes."""
    return _cap(_SENTENCE_END.split(_flat(text), maxsplit=1)[0], limit)


# Model id and response format ----------------------------------------------------------------------------------------


def list_models() -> list[str] | None:
    """Ids from GET /v1/models, or None when the listing fails (the candidates are then tried in their fixed order)."""
    client = _client()
    try:
        page = client.models.list()
    except (OpenAIError, httpx.HTTPError, OSError):
        return None
    data = getattr(page, "data", None)
    if data is None:
        data = _as_dict(page).get("data")
    ids = [item.get("id") if isinstance(item, dict) else getattr(item, "id", None) for item in data or []]
    return [i for i in ids if isinstance(i, str)]


def model_candidates() -> list[str]:
    """The ids to try, in order: the one that answered last, then listed ids, then the rest, each in MODEL_CANDIDATES
    order; an id that returned 404 in the last FAILURE_TTL_S is left out. The listing runs once per process."""
    with _LOCK:
        probed = _STATE.probed
    if not probed:
        listed = list_models()
        with _LOCK:
            if not _STATE.probed:
                _STATE.listed, _STATE.probed = listed, True
    with _LOCK:
        if _STATE.dead:
            now = _clock()
            _STATE.dead = {m: t for m, t in _STATE.dead.items() if now - t < FAILURE_TTL_S}
        listed = _STATE.listed or []
        order = [m for m in MODEL_CANDIDATES if m in listed] + [m for m in MODEL_CANDIDATES if m not in listed]
        if _STATE.answered in order:
            order.remove(_STATE.answered)
            order.insert(0, _STATE.answered)
        return [m for m in order if m not in _STATE.dead]


def _response_format() -> str:
    """json_schema, or the step down a format refusal chose in the last FAILURE_TTL_S."""
    with _LOCK:
        if _STATE.fmt_at is not None and _clock() - _STATE.fmt_at >= FAILURE_TTL_S:
            _STATE.fmt, _STATE.fmt_at = FORMATS[0], None
        return _STATE.fmt


def _format_refused(exc: APIStatusError) -> bool:
    """A 400 or 422 whose text names the response format."""
    return exc.status_code in (400, 422) and bool(FORMAT_ERROR.search(f"{exc} {getattr(exc, 'body', None) or ''}"))


# One request -------------------------------------------------------------------------------------------------------


def json_instruction(schema: dict[str, Any]) -> str:
    """The format line added under the question, since Baseten does not enforce the schema for this model."""
    return "Reply with only one JSON object, no code fence and no other text, that fits this JSON Schema: " + json.dumps(schema, separators=(",", ":"))


def request_body(prompt: str, image_urls: list[str], *, model: str, schema_name: str, schema: dict[str, Any], effort: str, max_tokens: int, fmt: str) -> dict[str, Any]:
    """The Chat Completions body: the hosted system prompt, then one user message with the text and the images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": f"{prompt} {json_instruction(schema)}"}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": spark_client.SYSTEM_PROMPT}, {"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "reasoning_effort": effort,
    }
    if fmt == "json_schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
    elif fmt == "json_object":
        body["response_format"] = {"type": "json_object"}
    return body


def _throttle(rpm: float | None) -> None:
    """Request starts at least 60 / rpm seconds apart, across threads."""
    if not rpm or rpm <= 0:
        return
    with _LOCK:
        now = _clock()
        wait = 0.0 if _STATE.last_start is None else max(0.0, _STATE.last_start + 60.0 / rpm - now)
        _STATE.last_start = now + wait
    if wait > 0:
        _sleep(wait)


def _retry_delay(exc: APIStatusError, tries: int) -> float | None:
    """Seconds before the next try, or None when the error is final: another status, the tries used up, or a
    Retry-After longer than MAX_RETRY_AFTER_S. 500 ms doubling with jitter, never under Retry-After."""
    status = exc.status_code
    if tries >= RATE_LIMIT_TRIES or not (status in (408, 409, 429) or status >= 500):
        return None
    delay = min(MAX_BACKOFF_S, BACKOFF_BASE_S * 2 ** (tries - 1)) * (1 - 0.25 * random.random())
    retry_after = (exc.response.headers.get("retry-after") or "").strip()
    if retry_after:
        try:
            wait = float(retry_after)
        except ValueError:
            try:
                wait = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError):
                wait = 0.0
        if wait > MAX_RETRY_AFTER_S:
            return None
        delay = max(delay, wait)
    return delay


def _create(client: Any, body: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """One chat request: (the response as a dict, Baseten's request id header). The raw response is read when the
    client offers it, because the header is the id Baseten's logs know the call by."""
    completions = client.chat.completions
    raw_api = getattr(completions, "with_raw_response", None)
    if raw_api is None:
        return _as_dict(completions.create(**body)), None
    raw = raw_api.create(**body)
    return _as_dict(raw.parse()), raw.headers.get(REQUEST_ID_HEADER)


def _header_id(exc: APIStatusError) -> str | None:
    response = getattr(exc, "response", None)
    return response.headers.get(REQUEST_ID_HEADER) if response is not None else None


def _message(data: dict[str, Any]) -> tuple[str, str | None]:
    """(content text, finish_reason) of the first choice."""
    choices = data.get("choices") or []
    if not choices:
        return "", None
    choice = _as_dict(choices[0])
    content = _as_dict(choice.get("message") or {}).get("content")
    if isinstance(content, list):
        content = "".join(str(_as_dict(part).get("text") or "") for part in content)
    return (content if isinstance(content, str) else ""), choice.get("finish_reason")


def _usage(data: dict[str, Any]) -> SparkUsage:
    """Chat Completions usage; Baseten counts reasoning inside completion_tokens."""
    usage = _as_dict(data.get("usage") or {})
    details_in = _as_dict(usage.get("prompt_tokens_details") or {})
    details_out = _as_dict(usage.get("completion_tokens_details") or {})
    return SparkUsage(
        input=int(usage.get("prompt_tokens") or 0),
        output=int(usage.get("completion_tokens") or 0),
        reasoning=int(details_out.get("reasoning_tokens") or 0),
        cached=int(details_in.get("cached_tokens") or 0),
    )


def _top_level_objects(text: str) -> list[str]:
    """The outermost {...} spans of text, in order. An object nested in another is never a span of its own, and
    braces inside JSON strings do not count."""
    spans: list[str] = []
    depth, start, in_string, escaped = 0, 0, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"' and depth:
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start : i + 1])
    return spans


def extract_json(text: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """The answer object: the top-level JSON object in the text that fits the schema (a code fence or a sentence
    around it is fine). An object nested inside another is never the answer, and two fitting objects that disagree
    on anything but the reason are a problem, so an ambiguous answer counts as unusable. (object or None, problems);
    no problems means it fits."""
    candidates: list[dict[str, Any]] = []
    for span in _top_level_objects(text):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
    if not candidates:
        return None, ["no JSON object in the answer"]
    fitting = [obj for obj in candidates if not spark_client.schema_problems(schema, obj)]
    if not fitting:
        return candidates[-1], spark_client.schema_problems(schema, candidates[-1])
    claims = {json.dumps({k: v for k, v in obj.items() if k != "reason"}, sort_keys=True) for obj in fitting}
    if len(claims) > 1:
        return fitting[-1], [f"{len(fitting)} answer objects that disagree"]
    return fitting[-1], []


def came_back(record: SparkRequestRecord) -> bool:
    """The model answered: a completed response, or one cut at the token budget. An error object, no choices, another
    stop or a failed call is no answer."""
    return record.status is SparkStatus.COMPLETED or record.incomplete_reason == "max_output_tokens"


def look_answer(record: SparkRequestRecord) -> tuple[LookVerdict, str]:
    """(verdict, reason) of a look_v1 record that came back: its own, or unsure when it failed the schema or was cut."""
    if record.valid and record.parsed is not None:
        return LookVerdict(record.parsed["verdict"]), str(record.parsed.get("reason") or "")
    return LookVerdict.UNSURE, f"second look unusable: {record.incomplete_reason or 'the answer did not fit look_v1'}"


def _log(run_id: str | None, context: dict[str, str | None] | None, **fields: Any) -> None:
    """Append one provenance line to runs/<run_id>/open_weights.ndjson."""
    if not run_id:
        return
    where = context or {}
    line = {
        "ts": _now_iso(),
        "run_id": run_id,
        "engine": ENGINE.value,
        "pickup_mark": where.get("pickup_mark"),
        "return_mark": where.get("return_mark"),
        "panel_id": where.get("panel_id"),
        **fields,
    }
    text = json.dumps(line, sort_keys=True, ensure_ascii=True) + "\n"
    with _LOG_LOCK:
        with paths.open_weights_log(run_id).open("a", encoding="utf-8") as fh:
            fh.write(text)


def _failed(
    model: str,
    *,
    purpose: SparkPurpose,
    run_id: str | None,
    ms: int,
    retry: int,
    http_status: int | None,
    error: str,
    baseten_id: str | None = None,
    context: dict[str, str | None] | None = None,
) -> None:
    """The ledger row and the provenance line of a request that got no answer."""
    ledger.record(
        LEDGER_KIND,
        model,
        note=purpose.value,
        run_id=run_id,
        input_tokens=0,
        output_tokens=0,
        cached_tokens=0,
        reasoning_tokens=0,
        engine=ENGINE.value,
        request_id=None,
        unit="tokens",
        quantity=0,
        tokens={"in": 0, "out": 0, "reasoning": 0, "cached": 0},
        latency_ms={"upload": None, "first_delta": None, "done": ms},
        status="failed",
        retries=retry,
        model_requested=model,
        baseten_request_id=baseten_id,
        http_status=http_status,
        error=error,
    )
    _log(
        run_id,
        context,
        purpose=purpose.value,
        model=model,
        model_requested=model,
        request_id=None,
        baseten_request_id=baseten_id,
        http_status=http_status,
        latency_ms=ms,
        status="failed",
        verdict=None,
        reason="",
        valid=False,
        error=error,
    )


def _answered(
    data: dict[str, Any],
    model: str,
    *,
    purpose: SparkPurpose,
    schema_name: str,
    schema: dict[str, Any],
    effort: str,
    max_tokens: int,
    run_id: str | None,
    input_refs: list[InputRef],
    retry: int,
    ms: int,
    baseten_id: str | None,
    context: dict[str, str | None] | None = None,
) -> SparkRequestRecord:
    """Ledger row, record and provenance line of a request that came back. finish_reason length is a cut
    (incomplete_reason max_output_tokens, as on the hosted path), never an answer."""
    text, finish = _message(data)
    error = _as_dict(data.get("error") or {})
    status = SparkStatus.INCOMPLETE if finish in CUT_FINISH_REASONS or error or not data.get("choices") else SparkStatus.COMPLETED
    incomplete_reason: str | None = None
    if status is SparkStatus.INCOMPLETE:
        if finish == "length":
            incomplete_reason = "max_output_tokens"
        elif error:
            incomplete_reason = f"error: {error.get('code')}: {_short(str(error.get('message')))}"
        else:
            incomplete_reason = str(finish or "no choices in the response")
    parsed, problems = extract_json(text, schema)
    valid = status is SparkStatus.COMPLETED and parsed is not None and not problems
    truncated = incomplete_reason == "max_output_tokens" and retry >= 1
    usage = _usage(data)
    returned = data.get("model") if isinstance(data.get("model"), str) else None
    request_id = data.get("id") if isinstance(data.get("id"), str) and data.get("id") else None
    if valid:
        row_status = "completed"
    elif status is SparkStatus.COMPLETED:
        row_status = "schema_miss"
    elif truncated:
        row_status = "truncated"
    else:
        row_status = "incomplete"
    row = ledger.record(
        LEDGER_KIND,
        returned or model,
        note=purpose.value,
        run_id=run_id,
        input_tokens=usage.input,
        output_tokens=usage.output,
        cached_tokens=usage.cached,
        reasoning_tokens=usage.reasoning,
        engine=ENGINE.value,
        request_id=request_id,
        unit="tokens",
        quantity=usage.input + usage.output,
        tokens={"in": usage.input, "out": usage.output, "reasoning": usage.reasoning, "cached": usage.cached},
        latency_ms={"upload": None, "first_delta": None, "done": ms},
        status=row_status,
        retries=retry,
        model_requested=model,
        baseten_request_id=baseten_id,
    )
    record = SparkRequestRecord(
        request_id=request_id or f"local-{uuid.uuid4().hex[:12]}",
        purpose=purpose,
        model=returned or model,
        engine=ENGINE,
        reasoning_effort=ReasoningEffort(effort),
        max_output_tokens=max_tokens,
        store=False,
        input_refs=list(input_refs),
        schema_name=schema_name,
        raw_text=text,
        parsed=parsed,
        valid=valid,
        status=status,
        incomplete_reason=incomplete_reason,
        truncated=truncated,
        retry=1 if retry else 0,
        usage=usage,
        dollars=float(row["usd"]),
        ms=ms,
    )
    answer = parsed if isinstance(parsed, dict) else {}
    _log(
        run_id,
        context,
        purpose=purpose.value,
        model=returned or model,
        model_requested=model,
        request_id=request_id,
        baseten_request_id=baseten_id,
        http_status=200,
        latency_ms=ms,
        status=row_status,
        verdict=answer.get("verdict") if valid else None,
        reason=plain_reason(answer["reason"]) if isinstance(answer.get("reason"), str) else "",
        valid=valid,
        error=None if valid else (incomplete_reason or "; ".join(problems))[:LOG_TEXT_MAX],
    )
    return record


def ask(
    prompt: str,
    image_urls: list[str],
    *,
    purpose: SparkPurpose,
    schema_name: str,
    run_id: str | None,
    input_refs: list[InputRef],
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = MAX_TOKENS,
    retry: int = 0,
    schema: dict[str, Any] | None = None,
    rpm: float | None = DEFAULT_RPM,
    context: dict[str, str | None] | None = None,
) -> SparkRequestRecord:
    """One answer from Spark Open: the next model id on a 404, the next response format on a 400 or 422 about the
    format, the backoff on 408, 409, 429 and 5xx. A ledger row and a provenance line per request (context gives the
    line its pickup_mark, return_mark and panel_id). Raises OpenWeightsError when no response came."""
    effort = spark_client.validate_effort(effort)
    schema = schema if schema is not None else spark_client.SCHEMAS[schema_name]
    tries = 0
    misses: list[str] = []
    while True:
        candidates = model_candidates()
        if not candidates:
            gone = "; ".join(misses) if misses else f"every id returned 404 in the last {FAILURE_TTL_S / 60:.0f} minutes"
            raise OpenWeightsError(f"no Muse Spark Open id answered on Baseten ({gone})")
        model = candidates[0]
        fmt = _response_format()
        body = request_body(prompt, image_urls, model=model, schema_name=schema_name, schema=schema, effort=effort, max_tokens=max_tokens, fmt=fmt)
        client = _client()
        _throttle(rpm)
        started = _clock()
        try:
            data, baseten_id = _create(client, body)
        except APIStatusError as exc:
            ms = int(round((_clock() - started) * 1000))
            error = f"{exc.status_code}: {_short(str(exc))}"
            _failed(model, purpose=purpose, run_id=run_id, ms=ms, retry=retry, http_status=exc.status_code, error=error, baseten_id=_header_id(exc), context=context)
            if exc.status_code == 404:
                with _LOCK:
                    _STATE.dead[model] = _clock()
                misses.append(f"{model}: 404")
                continue
            if fmt != FORMATS[-1] and _format_refused(exc):
                with _LOCK:
                    if _STATE.fmt == fmt:
                        _STATE.fmt, _STATE.fmt_at = FORMATS[FORMATS.index(fmt) + 1], _clock()
                continue
            tries += 1
            delay = _retry_delay(exc, tries)
            if delay is None:
                raise OpenWeightsError(f"Baseten answered {error}") from exc
            _sleep(delay)
            continue
        except (OpenAIError, httpx.HTTPError, OSError) as exc:
            ms = int(round((_clock() - started) * 1000))
            error = f"{type(exc).__name__}: {_short(str(exc))}"
            _failed(model, purpose=purpose, run_id=run_id, ms=ms, retry=retry, http_status=None, error=error, context=context)
            raise OpenWeightsError(f"Baseten call failed: {error}") from exc
        ms = int(round((_clock() - started) * 1000))
        with _LOCK:
            _STATE.answered = model
        return _answered(
            data,
            model,
            purpose=purpose,
            schema_name=schema_name,
            schema=schema,
            effort=effort,
            max_tokens=max_tokens,
            run_id=run_id,
            input_refs=input_refs,
            retry=retry,
            ms=ms,
            baseten_id=baseten_id,
            context=context,
        )


def ask_attempts(
    prompt: str,
    image_urls: list[str],
    *,
    purpose: SparkPurpose,
    schema_name: str,
    run_id: str | None,
    input_refs: list[InputRef],
    effort: str = DEFAULT_EFFORT,
    schema: dict[str, Any] | None = None,
    rpm: float | None = DEFAULT_RPM,
    max_tokens: int = MAX_TOKENS,
    retry_max_tokens: int = RETRY_MAX_TOKENS,
    context: dict[str, str | None] | None = None,
) -> list[SparkRequestRecord]:
    """The hosted path's outcomes in order: valid -> done; a cut -> one retry at retry_max_tokens; any other miss ->
    one retry at max_tokens. Two answers at most. OpenWeightsError when the retry gets no response (it carries the
    first answer) or when neither attempt came back with an answer."""
    common: dict[str, Any] = dict(purpose=purpose, schema_name=schema_name, run_id=run_id, input_refs=input_refs, effort=effort, schema=schema, rpm=rpm, context=context)
    first = ask(prompt, image_urls, max_tokens=max_tokens, retry=0, **common)
    if first.valid:
        return [first]
    budget = retry_max_tokens if first.incomplete_reason == "max_output_tokens" else max_tokens
    try:
        second = ask(prompt, image_urls, max_tokens=budget, retry=1, **common)
    except OpenWeightsError as exc:
        raise OpenWeightsError(str(exc), attempts=[first]) from exc
    attempts = [first, second]
    if not any(came_back(a) for a in attempts):
        raise OpenWeightsError(f"Baseten gave no answer: {second.incomplete_reason}", attempts=attempts)
    return attempts


def _context(panel_id: PanelId, pair: tuple[str, str] | None) -> dict[str, str | None]:
    return {"pickup_mark": pair[0] if pair else None, "return_mark": pair[1] if pair else None, "panel_id": str(getattr(panel_id, "value", panel_id))}


# The second look and count mode on Spark Open ------------------------------------------------------------------------


def second_look(
    crop_a: np.ndarray,
    crop_b: np.ndarray,
    *,
    panel_id: PanelId,
    run_id: str | None = None,
    input_refs: list[InputRef] | None = None,
    effort: str = DEFAULT_EFFORT,
    rpm: float | None = DEFAULT_RPM,
    pair: tuple[str, str] | None = None,
    max_tokens: int = MAX_TOKENS,
    retry_max_tokens: int = RETRY_MAX_TOKENS,
) -> spark_client.LookResult:
    """spark_client.second_look on Spark Open: the same question on the same two outlined crops (pickup, return).
    pair (pickup_mark, return_mark) goes on every provenance line. The last attempt that came back stands; a bad or
    cut answer is unsure, a downgrade; no answer at all raises OpenWeightsError."""
    effort = spark_client.validate_effort(effort)
    urls = [spark_client.prepare_image(crop)[0] for crop in (crop_a, crop_b)]
    attempts = ask_attempts(
        spark_client.look_prompt(panel_id),
        urls,
        purpose=SparkPurpose.SECOND_LOOK,
        schema_name="look_v1",
        run_id=run_id,
        input_refs=list(input_refs or []),
        effort=effort,
        rpm=rpm,
        max_tokens=max_tokens,
        retry_max_tokens=retry_max_tokens,
        context=_context(panel_id, pair),
    )
    record = next(a for a in reversed(attempts) if came_back(a))
    verdict, reason = look_answer(record)
    return spark_client.LookResult(verdict=verdict, reason=reason, record=record, attempts=attempts)


def count_marks(
    crop_rgb: np.ndarray,
    panel_id: PanelId,
    *,
    effort: str = DEFAULT_EFFORT,
    run_id: str | None = None,
    input_ref: InputRef | None = None,
    max_edge: int = spark_client.IMAGE_LONG_EDGE,
    rpm: float | None = DEFAULT_RPM,
) -> spark_client.CountResult:
    """Count mode (count_v1, K8) on Spark Open with spark_client.count_marks's signature; hosted Spark answers instead
    when Baseten gives no answer, and the record's engine says which one did."""
    effort = spark_client.validate_effort(effort)
    url = spark_client.prepare_image(crop_rgb, max_edge)[0]
    refs = [input_ref] if input_ref is not None else []
    try:
        attempts = ask_attempts(
            spark_client.count_prompt(panel_id), [url], purpose=SparkPurpose.COUNT, schema_name="count_v1", run_id=run_id, input_refs=refs, effort=effort, rpm=rpm, context=_context(panel_id, None)
        )
    except OpenWeightsError:
        return spark_client.count_marks(crop_rgb, panel_id, effort=effort, run_id=run_id, input_ref=input_ref, max_edge=max_edge)
    record = attempts[-1]
    points: list[GridPoint] = []
    count = 0
    if record.valid and record.parsed is not None:
        for raw in record.parsed.get("points") or []:
            x, y = spark_client.clamp_grid(raw.get("x")), spark_client.clamp_grid(raw.get("y"))
            if x is not None and y is not None:
                points.append(GridPoint(x=x, y=y))
        count = max(int(record.parsed.get("count") or 0), 0)
    return spark_client.CountResult(points=points, count=count, truncated=record.truncated, record=record, attempts=attempts)


__all__ = [
    "BASE_URL",
    "DEFAULT_RPM",
    "ENGINE",
    "FAILURE_TTL_S",
    "LEDGER_KIND",
    "MODEL_CANDIDATES",
    "OpenWeightsError",
    "REASON_MAX",
    "api_key",
    "ask",
    "ask_attempts",
    "came_back",
    "count_marks",
    "extract_json",
    "json_instruction",
    "list_models",
    "look_answer",
    "model_candidates",
    "plain_reason",
    "plain_text",
    "request_body",
    "reset",
    "second_look",
    "set_client",
]
