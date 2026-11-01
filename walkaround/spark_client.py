"""Muse Spark 1.3 as a grounded mark finder and as a downgrade-only second look (M2, F2).

Rules: deterministic code owns every verdict; an answer from here can only lower one (K4, K7, K12, K13, G2, G10).
Every call is one POST /v1/responses through the OpenAI SDK with store false, an explicit reasoning effort, a strict
json_schema text format and max_output_tokens set. A status of incomplete is truncation, its own state, never
"no marks". Every call writes one ledger row before its answer is parsed: the call cost money even if the JSON is bad.

Request shape (docs/API_NOTES.md, measured 10/26): input_image as a JPEG data URL (long edge 1280 by default, 1008
for Stays Put clip frames), the grounding instruction in the user message under a one-line system prompt,
text.format {type: json_schema, name, schema, strict: true} (flat, no nested json_schema key; strict needs
additionalProperties false and every key required on each object; anyOf only below the root). Coordinates come back
on a 0 to 1000 grid over the image the model received: pixel = round(v * dim / 1000), clamped; 1001 occurs and
clamps to 1000, anything farther off the grid is not a point on the image and that mark is dropped.

Retries (section 14): 408, 409, 429 and 5xx back off inside _create, 6 tries in all; a timeout or a dropped
connection is final for that attempt and _ask_attempts makes the one retry of item 10. The SDK's own retry loop is off
because it would repeat a 240 s timeout five more times. Error text is redacted before it is stored, so a server that
echoes the bearer key never puts it in spark.json.

The client is injectable: tests call set_client() (or monkeypatch _CLIENT) with a fake whose .responses.create
returns canned response dicts, so the offline suite never touches the network.
"""

from __future__ import annotations

import base64
import io
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from openai import APIStatusError, OpenAIError
from PIL import Image
from pydantic import ValidationError

from . import ledger
from .config import BASE_URL, SPARK_MODEL, api_key, redact
from .contracts import (
    PANEL_NOUNS,
    Confidence,
    GridBox,
    GridPoint,
    InputRef,
    Landmark,
    LookVerdict,
    MarkKind,
    Noun,
    PanelId,
    Position,
    ReasoningEffort,
    Side,
    SizeClass,
    SparkEngine,
    SparkMarksAnswer,
    SparkPurpose,
    SparkRequestRecord,
    SparkStatus,
    SparkUsage,
)

GRID = 1000
# The docs say the model can emit 1001. Values this close to the grid edge are clamped; anything farther out is not a
# point on the image we sent, so that mark is dropped (thresholds spark.grid_slack).
GRID_SLACK = 20
IMAGE_LONG_EDGE = 1280  # thresholds spark.image_long_edge_px
CLIP_LONG_EDGE = 1008  # Stays Put clip frames (thresholds media.clip_size_px)
JPEG_QUALITY = 90  # thresholds spark.jpeg_quality
# Reasoning tokens share the output budget (https://dev.meta.ai/docs/reasoning), so the floor is 4000 and the one
# retry after a truncation doubles it (thresholds spark.max_output_tokens, spark.retry_max_output_tokens).
MAX_OUTPUT_TOKENS = 4000
RETRY_MAX_OUTPUT_TOKENS = 8000
DEFAULT_EFFORT = "low"  # thresholds spark.effort; callers pass th["spark"]["effort"]
EFFORTS = tuple(e.value for e in ReasoningEffort)
# How many landmarks one answer keeps (thresholds spark.landmarks_max, section 5 step 8 "up to 4 landmarks from the
# panel's closed list"). It caps the count, never the vocabulary: the whole closed list goes into the enum and prompt.
LANDMARKS_MAX = 4
TIMEOUT_S = 240.0  # thresholds orchestrator.request_timeout_s; then one retry (14 item 10)
RATE_LIMIT_TRIES = 6  # thresholds orchestrator.rate_limit_retries: tries in all on 408, 409, 429, 5xx (14 item 1)
BACKOFF_BASE_S = 0.5  # thresholds orchestrator.rate_limit_backoff_base_s, doubled per try, jitter, capped
MAX_BACKOFF_S = 8.0
MAX_RETRY_AFTER_S = 60.0  # a longer Retry-After fails the call instead of holding a worker thread
ENGINE = SparkEngine.META_MODEL_API

# Fixed and minimal, so the cached prefix is the same on every call (API notes: $0.15 per M cached input tokens).
SYSTEM_PROMPT = "You inspect photos of rental cars for the renter's records and answer only with JSON that fits the schema."

MARK_TYPES = [k.value for k in MarkKind]
SIZE_CLASSES = [s.value for s in SizeClass]
CONFIDENCE_LEVELS = [c.value for c in Confidence]
PANEL_NOUN_NAMES = [n.value for n in PANEL_NOUNS]  # the 14 nouns of section 16 item 3, car included


# ----------------------------------------------------------------------------------------------------------------
# Schemas (strict: additionalProperties false and every property required on every object)
# ----------------------------------------------------------------------------------------------------------------


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_GRID_INT = {"type": "integer"}
_POINT = _obj({"x": _GRID_INT, "y": _GRID_INT})
_BOX = _obj({"x_min": _GRID_INT, "y_min": _GRID_INT, "x_max": _GRID_INT, "y_max": _GRID_INT})
_MARK = _obj(
    {
        "type": {"type": "string", "enum": MARK_TYPES},
        "point": _POINT,
        "bbox": _BOX,
        "size_class": {"type": "string", "enum": SIZE_CLASSES},
        "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
        "note": {"type": "string"},
    }
)
_CONDITIONS = _obj({k: {"type": "boolean"} for k in ("wet", "dirty", "glare", "dark")})
_SIDE = {"type": "string", "enum": [s.value for s in Side]}
_POSITION = {"type": "string", "enum": [p.value for p in Position]}


def marks_schema(landmark_names: list[str] | None = None) -> dict[str, Any]:
    """marks_v1 of section 5.7; the landmark name enum is the panel's closed list when one is given."""
    name = {"type": "string", "enum": list(landmark_names)} if landmark_names else {"type": "string"}
    return _obj(
        {
            "marks": {"type": "array", "items": _MARK},
            "conditions": _CONDITIONS,
            "landmarks": {"type": "array", "items": _obj({"name": name, "point": _POINT})},
            "neighbour_car_visible": {"type": "boolean"},
        }
    )


# Only marks_v1 has been sent live (10/27). TODO(D1): confirm each of the others is accepted under strict mode:
# count_v1 (E7), look_v1 (E6 second look), claim_name_v1 and claim_box_v1 (E11; anyOf with null below the root is
# "ok" per the structured-output page, never measured), seed_v1 (no gate row: tests/test_spark_client.py
# test_live_seed_v1_anyof_null, about $0.01), timeline_v1 (E17) and note_prefill_v1 (E18). Copy the accepted request
# shapes into docs/API_NOTES.md.
SCHEMAS: dict[str, dict[str, Any]] = {
    "marks_v1": marks_schema(),
    "count_v1": _obj({"points": {"type": "array", "items": _POINT}, "count": {"type": "integer"}}),
    "look_v1": _obj({"verdict": {"type": "string", "enum": [v.value for v in LookVerdict]}, "reason": {"type": "string"}}),
    "seed_v1": _obj(
        {
            "found": {"type": "boolean"},
            "bbox": _nullable(_BOX),
            "type": _nullable({"type": "string", "enum": MARK_TYPES}),
        }
    ),
    "claim_name_v1": _obj(
        {
            "panel": {"type": "string", "enum": PANEL_NOUN_NAMES + ["none"]},
            "side": _SIDE,
            "position": _POSITION,
            "front_direction": {"type": "string", "enum": ["left", "right", "unknown"]},
            "panel_bbox": _nullable(_BOX),
            "context": {"type": "string"},
            "landmarks": {"type": "array", "items": _obj({"name": {"type": "string"}, "point": _POINT})},
        }
    ),
    "claim_box_v1": _obj({"found": {"type": "boolean"}, "bbox": _nullable(_BOX)}),
    "timeline_v1": _obj(
        {
            "panels": {
                "type": "array",
                "items": _obj(
                    {
                        "panel": {"type": "string", "enum": PANEL_NOUN_NAMES},
                        "side": _SIDE,
                        "start_s": {"type": "number"},
                        "end_s": {"type": "number"},
                    }
                ),
            }
        }
    ),
    "note_prefill_v1": _obj(
        {k: {"type": "string"} for k in ("panel", "pickup_date", "return_date", "amount", "agreement_number")}
    ),
}


def schema_problems(schema: dict[str, Any], value: Any, path: str = "$") -> list[str]:
    """Check value against the subset of JSON Schema these strict schemas use. Empty list means it fits."""
    if "anyOf" in schema:
        if any(not schema_problems(option, value, path) for option in schema["anyOf"]):
            return []
        return [f"{path}: fits none of the anyOf options"]
    problems: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} is not in the enum")
    kind = schema.get("type")
    if kind is not None and not _is_type(value, kind):
        return problems + [f"{path}: expected {kind}, got {type(value).__name__}"]
    if kind == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path}.{key}: missing")
        if schema.get("additionalProperties") is False:
            problems.extend(f"{path}.{key}: not in the schema" for key in value if key not in properties)
        for key, sub in properties.items():
            if key in value:
                problems.extend(schema_problems(sub, value[key], f"{path}.{key}"))
    elif kind == "array":
        items = schema.get("items")
        if items is not None:
            for i, item in enumerate(value):
                problems.extend(schema_problems(items, item, f"{path}[{i}]"))
    elif kind in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: {value} under the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: {value} over the maximum {schema['maximum']}")
    return problems


def _is_type(value: Any, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    if isinstance(value, bool):
        return False
    if kind == "integer":
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if kind == "number":
        return isinstance(value, (int, float))
    return True


# ----------------------------------------------------------------------------------------------------------------
# Images and the grid
# ----------------------------------------------------------------------------------------------------------------


def _to_pil(rgb: np.ndarray) -> Image.Image:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGB")
    return Image.fromarray(np.ascontiguousarray(arr[:, :, :3]), "RGB")


def sent_size(width: int, height: int, max_edge: int = IMAGE_LONG_EDGE) -> tuple[int, int]:
    """The size prepare_image sends an image of width x height at: long edge <= max_edge, never upscaled."""
    scale = min(1.0, max_edge / max(width, height))
    if scale >= 1.0:
        return int(width), int(height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def prepare_image(rgb: np.ndarray, max_edge: int = IMAGE_LONG_EDGE) -> tuple[str, int, int, float]:
    """JPEG data URL of the image resized so the long edge <= max_edge (never upscaled). (url, w, h, scale)."""
    img = _to_pil(rgb)
    width, height = img.size
    size = sent_size(width, height, max_edge)
    if size != (width, height):
        img = img.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return data_url, img.width, img.height, img.width / width


def grid_to_pixels(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """round(v * dim / 1000), clamped into [0, dim - 1], on the image the model received."""
    px = min(max(round(x * width / GRID), 0), width - 1)
    py = min(max(round(y * height / GRID), 0), height - 1)
    return int(px), int(py)


def grid_box_to_pixels(box: GridBox, width: int, height: int) -> tuple[int, int, int, int]:
    """The same rule per edge; the right and bottom edges are half-open, so they may equal dim and always exceed the left and top."""
    x1, y1 = grid_to_pixels(box.x_min, box.y_min, width, height)
    x2 = min(max(round(box.x_max * width / GRID), x1 + 1), width)
    y2 = min(max(round(box.y_max * height / GRID), y1 + 1), height)
    return x1, y1, int(x2), int(y2)


def clamp_grid(value: Any) -> int | None:
    """A grid coordinate: 1001 clamps to 1000; farther than GRID_SLACK off the grid is None (the mark is dropped)."""
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v < -GRID_SLACK or v > GRID + GRID_SLACK:
        return None
    return int(min(max(round(v), 0), GRID))


def _grid_point(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    x, y = clamp_grid(raw.get("x")), clamp_grid(raw.get("y"))
    if x is None or y is None:
        return None
    return {"x": x, "y": y}


def _grid_box(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    values = [clamp_grid(raw.get(k)) for k in ("x_min", "y_min", "x_max", "y_max")]
    if any(v is None for v in values):
        return None
    x_min, y_min, x_max, y_max = values  # type: ignore[misc]
    return {"x_min": min(x_min, x_max), "y_min": min(y_min, y_max), "x_max": max(x_min, x_max), "y_max": max(y_min, y_max)}


def validate_effort(effort: str) -> str:
    """Exact membership in the reasoning effort enum, checked before any wire call."""
    if effort not in EFFORTS:
        raise ValueError(f"reasoning effort must be one of {', '.join(EFFORTS)}, not {effort!r}")
    return effort


# ----------------------------------------------------------------------------------------------------------------
# Prompts (keyed by PanelId; the grounding instruction lives in the user message)
# ----------------------------------------------------------------------------------------------------------------

_DISPLAY_NAMES = {
    PanelId.HEADLIGHTS: "headlights",
    PanelId.TAIL_LIGHTS: "tail lights",
    PanelId.MIRRORS: "side mirrors",
    PanelId.WINDSHIELD: "windshield",
    PanelId.REAR_WINDOW: "rear window",
}

_GRID_SENTENCE = "Coordinates are on a 0 to 1000 grid over the whole image, (0, 0) at the top left, x to the right, y down."


def panel_display_name(panel_id: PanelId) -> str:
    """'left_front_door' -> 'left front door'."""
    return _DISPLAY_NAMES.get(panel_id, panel_id.value.replace("_", " "))


# Lead decision 4 of 11/1 (thresholds matching.fixtures_not_damage): one plain sentence after "do not miss", sent only
# when the run's thresholds have the leaf, so a run frozen before it asks (and replays) the prompt it always did.
FIXTURES_SENTENCE = (
    "Do not box factory fixtures: parking sensors, badges and emblems, door locks and handles, washer nozzles, fuel "
    "door edges, panel gaps and trim seams are part of the car, not damage. "
)


def marks_prompt(panel_id: PanelId, landmarks: list[str], *, fixtures: bool = False) -> str:
    """The marks_v1 grounding instruction of section 5 step 8: expert framing, grid, point plus bbox, do not miss; with
    fixtures, the sentence that factory fixtures are not damage."""
    panel = panel_display_name(panel_id)
    names = ", ".join(landmarks)  # the whole closed list; LANDMARKS_MAX caps how many come back, not the names
    landmark_line = (
        f"landmarks: point at up to {LANDMARKS_MAX} of these if they are visible, naming each exactly as written here: {names}. "
        if names
        else "landmarks: return an empty list. "
    )
    return (
        f"You are an object grounding expert. This image is a close crop of the {panel} of a car, cut from a rental "
        f"walkaround video. Find every scratch, scuff, chip, dent and crack on the painted, plastic or glass surface "
        f"of the {panel}. Do not miss any mark, including small ones. {FIXTURES_SENTENCE if fixtures else ''}For each "
        f"mark give one point at its centre and one bbox that covers it. {_GRID_SENTENCE} type is scratch, scuff, chip, dent or crack for damage; dirt or "
        "reflection when a spot is dirt, a water drop, a shadow or a reflection that could pass for damage; unsure "
        "when you cannot tell. size_class: S is smaller than a coin, M up to a hand, L larger than a hand. confidence "
        "is low, medium or high. note is a few words on what you see. conditions describe the whole crop: wet, dirty, "
        f"glare, dark. {landmark_line}neighbour_car_visible is true when another vehicle shows in the crop. If the "
        "surface is clean, return an empty marks list."
    )


def count_prompt(panel_id: PanelId) -> str:
    panel = panel_display_name(panel_id)
    return (
        f"You are an object grounding expert. This image is a close crop of the {panel} of a car. Point at every "
        f"scratch, scuff, chip, dent and crack on the {panel}, one point per mark, before you count them. "
        f"{_GRID_SENTENCE} Do not miss any mark, including small ones; skip reflections, water drops and dirt. count "
        "is the number of points. If the surface is clean, return no points and count 0."
    )


def look_prompt(panel_id: PanelId) -> str:
    panel = panel_display_name(panel_id)
    return (
        f"The first image shows the {panel} of a car at pickup and the second image shows the same {panel} at return. "
        "On each image one candidate mark is outlined with a thin box. Deterministic matching paired these two outlined "
        "spots as the same mark. Look at both spots. Answer same when you see no reason to doubt the pairing. Answer "
        "different when the two spots look like different marks or a mark is missing from one image. Answer unsure "
        "when glare, water, dirt or blur make it impossible to tell. Give a one sentence reason."
    )


def counter_prompt(panel_id: PanelId, mark_role: str) -> str:
    """The counter-look (lead 11/1, third): a mark found in one walk only, and the other walk's view of the same spot."""
    panel = panel_display_name(panel_id)
    other = "pickup" if mark_role == "return" else "return"
    return (
        f"The first image shows the {panel} of a car at {mark_role}, with one mark outlined with a thin box. The second "
        f"image shows the same {panel} at {other}, with a box outlined where that mark would sit; the box is larger than "
        f"the mark to allow for the change of view. Deterministic matching found no mark there at {other}. Look inside "
        f"the box on the second image. Answer same when the same mark shows there too. Answer different when the box "
        "shows clean paint or a clearly different mark. Answer unsure when glare, water, dirt, blur or the view make "
        "it impossible to tell. Give a one sentence reason."
    )


def seed_prompt(panel_id: PanelId, phrase: str) -> str:
    panel = panel_display_name(panel_id)
    return (
        f"You are an object grounding expert. This image is a close crop of the {panel} of a car. While filming, the "
        f'renter said: "{phrase}". Find the mark those words describe on the {panel}. {_GRID_SENTENCE} If you find '
        "it, set found true and give its bbox and its type (scratch, scuff, chip, dent, crack, dirt, reflection or "
        "unsure). If nothing matches the words, set found false with bbox null and type null."
    )


def claim_name_prompt() -> str:
    nouns = ", ".join(PANEL_NOUN_NAMES)
    return (
        "You are an object grounding expert. This photo was sent with a damage claim for a rental car. Name the car "
        f"part the photo centres on: panel is one of {nouns}, or none when no car is visible; car means the whole "
        "car with no single panel in focus. side is left or right as seen from a driver sitting in the car (in the US "
        "the driver sits on the left), or none when you cannot tell or the panel has no side. position is front or "
        "rear, or none. front_direction says whether the front of the car points to the left or the right of the "
        f"image, or unknown. panel_bbox is the bbox of the named panel; {_GRID_SENTENCE} Use null when panel is car "
        f"or none. context is one sentence on what is visible around the panel. landmarks: point at up to {LANDMARKS_MAX} visible "
        "features such as a door handle, mirror, wheel arch, badge, plate corner, fog lamp or fuel door, each with a "
        "short name and a point."
    )


def claim_box_prompt() -> str:
    return (
        "Someone may have drawn a rectangle, circle or arrow on this photo with a coloured marker or an editing app "
        "to point at damage. If there is such a drawn shape, set found true and give the bbox of the area it points "
        f"at. {_GRID_SENTENCE} If nothing was drawn on the photo, set found false and bbox null."
    )


def timeline_prompt() -> str:
    nouns = ", ".join(n for n in PANEL_NOUN_NAMES if n != "car")
    return (
        "This is a walkaround video of a rental car. List the car panels the camera passes over, in order, with the "
        f"time in seconds when each one enters and leaves the frame. panel is one of {nouns}. side is left or right "
        "as seen from a driver sitting in the car, or none."
    )


def note_prefill_prompt(claim_text: str) -> str:
    return (
        "The attached PDF is a car rental agreement and the text below is a damage claim about that rental. Fill the "
        "fields from the two documents, copying words and numbers as they are written: panel (the body panel the "
        "claim names), pickup_date, return_date, amount (the claimed charge with its currency sign), agreement_number. "
        "Use an empty string for a field neither document gives.\n\nClaim text:\n" + claim_text
    )


# ----------------------------------------------------------------------------------------------------------------
# The client and one call
# ----------------------------------------------------------------------------------------------------------------

_CLIENT: Any = None
_CLIENT_LOCK = threading.Lock()  # marks.derive_marks calls this module from up to 12 threads at once
_sleep = time.sleep  # tests replace it


def _client() -> Any:
    """OpenAI SDK client for the Meta Model API, made on first use. Tests inject a fake through set_client."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            from openai import OpenAI

            # max_retries 0: _create owns the 429 backoff, so a timeout is not repeated by the SDK as well.
            _CLIENT = OpenAI(base_url=BASE_URL, api_key=api_key(), timeout=TIMEOUT_S, max_retries=0)
        return _CLIENT


def set_client(client: Any) -> None:
    """Inject a client with .responses.create(**kwargs); None goes back to the real one on the next call."""
    global _CLIENT
    _CLIENT = client


def _redact(text: str) -> str:
    """Error text with every configured key value and key-shaped token replaced, before it is stored anywhere (config.redact: every key variable, Baseten's two included, and every key shape)."""
    return redact(text)


def _retry_delay(exc: APIStatusError, tries: int) -> float | None:
    """Seconds before the next try after `tries` tries, or None when this error is final: a status other than 408, 409,
    429 or 5xx, the 6 tries used, or a Retry-After over a minute. 500 ms doubling, jitter, never under Retry-After."""
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


def _create(request: dict[str, Any]) -> Any:
    """responses.create with the 429 and 5xx backoff of section 14 item 1. Timeouts and connection errors are raised
    at once: ask records them and _ask_attempts makes the single retry of item 10."""
    tries = 0
    while True:
        try:
            return _client().responses.create(**request)
        except APIStatusError as exc:
            tries += 1
            delay = _retry_delay(exc, tries)
            if delay is None:
                raise
            _sleep(delay)


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(vars(obj)) if hasattr(obj, "__dict__") else {}


def _output_text(response: Any, data: dict[str, Any]) -> str:
    """The SDK's output_text property when there is one, else the output_text parts of the message items."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    parts: list[str] = []
    for item in data.get("output") or []:
        item = _as_dict(item)
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            part = _as_dict(part)
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "".join(parts)


def _usage(data: dict[str, Any]) -> SparkUsage:
    """usage through model_dump(): input, output, reasoning (inside output) and cached (inside input) tokens."""
    usage = _as_dict(data.get("usage") or {})
    details_in = _as_dict(usage.get("input_tokens_details") or {})
    details_out = _as_dict(usage.get("output_tokens_details") or {})
    return SparkUsage(
        input=int(usage.get("input_tokens") or 0),
        output=int(usage.get("output_tokens") or 0),
        reasoning=int(details_out.get("reasoning_tokens") or 0),
        cached=int(details_in.get("cached_tokens") or 0),
    )


def _request(
    parts: list[dict[str, Any]], *, schema_name: str, schema: dict[str, Any], effort: str, max_output_tokens: int, stream: bool
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": SPARK_MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"type": "message", "role": "user", "content": parts},
        ],
        "reasoning": {"effort": effort},
        "text": {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if stream:
        body["stream"] = True
    return body


_FINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


def _send(request: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """One create call (with its 429 backoff). Returns the final response as a dict plus its output text, streamed or not."""
    response = _create(request)
    if not request.get("stream"):
        data = _as_dict(response)
        return data, _output_text(response, data)
    deltas: list[str] = []
    final: dict[str, Any] = {}
    try:
        for event in response:
            ev = _as_dict(event)
            kind = ev.get("type")
            if kind == "response.output_text.delta" and isinstance(ev.get("delta"), str):
                deltas.append(ev["delta"])
            elif kind in _FINAL_EVENTS:
                final = _as_dict(ev.get("response") or {})
            elif kind == "error":
                final = {"status": "failed", "error": {"code": ev.get("code"), "message": ev.get("message")}}
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if not final:
        final = {"status": "incomplete", "incomplete_details": {"reason": "stream ended before response.completed"}}
    return final, _output_text(None, final) or "".join(deltas)


def _parse(text: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """json.loads plus the schema check. (parsed object or None, problems)."""
    try:
        parsed = json.loads(text) if text else None
    except json.JSONDecodeError as exc:
        return None, [f"not JSON: {exc.msg} at char {exc.pos}"]
    if not isinstance(parsed, dict):
        return None, ["the answer is not a JSON object"]
    return parsed, schema_problems(schema, parsed)


def ask(
    parts: list[dict[str, Any]],
    *,
    purpose: SparkPurpose,
    schema_name: str,
    effort: str,
    max_output_tokens: int,
    run_id: str | None,
    input_refs: list[InputRef],
    stream: bool = False,
    retry: int = 0,
    schema: dict[str, Any] | None = None,
    check_schema: dict[str, Any] | None = None,
) -> SparkRequestRecord:
    """One call, one ledger row, one record; no interpretation. `schema` overrides SCHEMAS[schema_name] (the
    per-panel landmark enum); `check_schema`, when given, is what `valid` is checked against instead of the schema
    sent; `retry` 1 marks the second attempt, after which an incomplete answer is truncated."""
    effort = validate_effort(effort)
    schema = schema if schema is not None else SCHEMAS[schema_name]
    request = _request(parts, schema_name=schema_name, schema=schema, effort=effort, max_output_tokens=max_output_tokens, stream=stream)
    started = time.monotonic()
    error: str | None = None
    try:
        data, text = _send(request)
    except (OpenAIError, httpx.HTTPError, OSError) as exc:
        # _create already backed off on 429 and 5xx; what reaches here is final for this attempt and must not crash a run.
        data, text, error = {}, "", f"error: {type(exc).__name__}: {_redact(str(exc))[:200]}"
    ms = int(round((time.monotonic() - started) * 1000))

    api_error = _as_dict(data.get("error") or {})
    if error is None and api_error:
        error = f"error: {api_error.get('code')}: {_redact(str(api_error.get('message')))[:200]}"
    status_text = str(data.get("status") or ("incomplete" if error else "completed"))
    status = SparkStatus.COMPLETED if status_text == "completed" and error is None else SparkStatus.INCOMPLETE
    incomplete_reason: str | None = None
    if status is SparkStatus.INCOMPLETE:
        details = _as_dict(data.get("incomplete_details") or {})
        reason = details.get("reason")
        incomplete_reason = (_redact(str(reason)) if reason else None) or error or status_text
    parsed, problems = _parse(text, check_schema if check_schema is not None else schema)
    valid = status is SparkStatus.COMPLETED and parsed is not None and not problems
    truncated = incomplete_reason == "max_output_tokens" and retry >= 1
    usage = _usage(data)
    request_id = str(data.get("id") or f"local-{uuid.uuid4().hex[:12]}")
    if valid:
        row_status = "completed"
    elif status is SparkStatus.COMPLETED:
        row_status = "schema_miss"
    elif truncated:
        row_status = "truncated"
    elif error:
        row_status = "failed"
    else:
        row_status = "incomplete"
    row = ledger.record(
        "spark",
        SPARK_MODEL,
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
    )
    return SparkRequestRecord(
        request_id=request_id,
        purpose=purpose,
        model=SPARK_MODEL,
        engine=ENGINE,
        reasoning_effort=ReasoningEffort(effort),
        max_output_tokens=max_output_tokens,
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


def _ask_attempts(
    parts: list[dict[str, Any]],
    *,
    purpose: SparkPurpose,
    schema_name: str,
    effort: str,
    run_id: str | None,
    input_refs: list[InputRef],
    stream: bool = False,
    schema: dict[str, Any] | None = None,
    check_schema: dict[str, Any] | None = None,
) -> list[SparkRequestRecord]:
    """Step 8 outcomes in order: completed and valid -> done; completed and a schema miss -> one retry; incomplete
    with reason max_output_tokens -> one retry at 8000, then truncated; an error -> one retry at the same budget.

    Two rows at most, so only a cut on the first attempt earns the 8000 budget. A schema miss or an error followed by
    a cut at 4000 ends there, truncated: G10 "marks incomplete" is the safer of the two readings ("no marks found" is
    the other), and the record's max_output_tokens of 4000 shows the larger budget was never tried."""
    common = dict(
        purpose=purpose, schema_name=schema_name, effort=effort, run_id=run_id, input_refs=input_refs, stream=stream, schema=schema, check_schema=check_schema
    )
    first = ask(parts, max_output_tokens=MAX_OUTPUT_TOKENS, retry=0, **common)
    if first.valid:
        return [first]
    budget = RETRY_MAX_OUTPUT_TOKENS if first.incomplete_reason == "max_output_tokens" else MAX_OUTPUT_TOKENS
    second = ask(parts, max_output_tokens=budget, retry=1, **common)
    return [first, second]


def ask_with_retry(
    parts: list[dict[str, Any]],
    *,
    purpose: SparkPurpose,
    schema_name: str,
    effort: str,
    run_id: str | None,
    input_refs: list[InputRef],
    stream: bool = False,
    schema: dict[str, Any] | None = None,
    check_schema: dict[str, Any] | None = None,
) -> SparkRequestRecord:
    """The record of the attempt that stands (retry 0 or 1); two ledger rows at most."""
    return _ask_attempts(
        parts, purpose=purpose, schema_name=schema_name, effort=effort, run_id=run_id, input_refs=input_refs, stream=stream, schema=schema, check_schema=check_schema
    )[-1]


# ----------------------------------------------------------------------------------------------------------------
# Marks (marks_v1), count mode (count_v1)
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class MarksResult:
    """answer None means truncated (truncated True, G10 and K13 upstream) or invalid after the retry (a ledger flag)."""

    answer: SparkMarksAnswer | None
    truncated: bool
    record: SparkRequestRecord
    sent_width: int
    sent_height: int
    scale: float  # sent / original (<= 1)
    orig_width: int
    orig_height: int
    attempts: list[SparkRequestRecord] = field(default_factory=list)  # every record of this call, first to last


def _marks_answer(parsed: dict[str, Any] | None, *, landmark_names: list[str] | None = None) -> SparkMarksAnswer | None:
    """The 1001 clamp and the slack drop, then the contract model (which rejects anything still off the grid).
    With landmark_names given, a landmark outside that closed list is dropped (None keeps every name)."""
    if not isinstance(parsed, dict):
        return None
    marks: list[dict[str, Any]] = []
    for item in parsed.get("marks") or []:
        if not isinstance(item, dict):
            continue
        point, bbox = _grid_point(item.get("point")), _grid_box(item.get("bbox"))
        if point is None or bbox is None:
            continue  # farther than the slack off the image we sent: not a point on it
        marks.append(
            {
                "type": item.get("type"),
                "point": point,
                "bbox": bbox,
                "size_class": item.get("size_class"),
                "confidence": item.get("confidence"),
                "note": str(item.get("note") or ""),
            }
        )
    landmarks: list[dict[str, Any]] = []
    for item in parsed.get("landmarks") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        if landmark_names is not None and item["name"] not in landmark_names:
            continue
        point = _grid_point(item.get("point"))
        if point is not None and len(landmarks) < LANDMARKS_MAX:
            landmarks.append({"name": str(item["name"]), "point": point})
    conditions = parsed.get("conditions") if isinstance(parsed.get("conditions"), dict) else {}
    try:
        return SparkMarksAnswer.model_validate(
            {
                "marks": marks,
                "conditions": {k: bool(conditions.get(k, False)) for k in ("wet", "dirty", "glare", "dark")},
                "landmarks": landmarks,
                "neighbour_car_visible": bool(parsed.get("neighbour_car_visible", False)),
            }
        )
    except ValidationError:
        return None


def _refs(input_ref: InputRef | None) -> list[InputRef]:
    return [input_ref] if input_ref is not None else []


def find_marks(
    crop_rgb: np.ndarray,
    panel_id: PanelId,
    *,
    landmarks: list[str],
    effort: str = DEFAULT_EFFORT,
    run_id: str | None = None,
    input_ref: InputRef | None = None,
    max_edge: int = IMAGE_LONG_EDGE,
    fixtures: bool = False,
) -> MarksResult:
    """The marks_v1 grounding call on one blurred crop. Points and boxes stay in the grid of the sent image. fixtures
    adds the factory-fixtures sentence to the prompt (thresholds matching.fixtures_not_damage); the schema is the same.

    The schema sent carries the panel's whole closed landmark list as the enum (strict, so the server enforces it)
    and the prompt asks for up to LANDMARKS_MAX of them; `valid` is checked against marks_v1 with the landmark name
    free, a name outside the list is dropped, and at most LANDMARKS_MAX landmarks are kept, so a landmark label can
    never throw away the marks of a crop."""
    effort = validate_effort(effort)
    orig_height, orig_width = np.asarray(crop_rgb).shape[:2]
    data_url, sent_width, sent_height, scale = prepare_image(crop_rgb, max_edge)
    names = [str(n) for n in dict.fromkeys(landmarks) if n]  # order kept, duplicates dropped (an enum lists each once)
    parts = [
        {"type": "input_text", "text": marks_prompt(panel_id, names, fixtures=fixtures)},
        {"type": "input_image", "image_url": data_url},
    ]
    attempts = _ask_attempts(
        parts,
        purpose=SparkPurpose.MARKS,
        schema_name="marks_v1",
        schema=marks_schema(names),
        check_schema=SCHEMAS["marks_v1"],
        effort=effort,
        run_id=run_id,
        input_refs=_refs(input_ref),
    )
    record = attempts[-1]
    return MarksResult(
        answer=_marks_answer(record.parsed, landmark_names=names) if record.valid else None,
        truncated=record.truncated,
        record=record,
        sent_width=sent_width,
        sent_height=sent_height,
        scale=scale,
        orig_width=int(orig_width),
        orig_height=int(orig_height),
        attempts=attempts,
    )


def answer_landmarks(record: SparkRequestRecord, landmarks: list[str]) -> list[dict[str, Any]]:
    """The landmarks of a stored marks_v1 answer, read exactly as find_marks reads them (names outside the panel's list
    dropped, at most LANDMARKS_MAX, points clamped to the grid); empty for an answer that is not valid."""
    names = [str(n) for n in dict.fromkeys(landmarks) if n]
    answer = _marks_answer(record.parsed, landmark_names=names) if record.valid else None
    return [lm.model_dump() for lm in answer.landmarks] if answer is not None else []


def marks_result_from_record(
    record: SparkRequestRecord,
    *,
    orig_width: int,
    orig_height: int,
    landmarks: list[str],
    max_edge: int = IMAGE_LONG_EDGE,
) -> MarksResult:
    """A stored marks_v1 record read back exactly as find_marks reads a fresh one; no call and no ledger row. Used by
    the rebuild and when a run asks the same question about the same crop bytes twice."""
    names = [str(n) for n in dict.fromkeys(landmarks) if n]
    sent_width, sent_height = sent_size(int(orig_width), int(orig_height), max_edge)
    return MarksResult(
        answer=_marks_answer(record.parsed, landmark_names=names) if record.valid else None,
        truncated=record.truncated,
        record=record,
        sent_width=sent_width,
        sent_height=sent_height,
        scale=sent_width / int(orig_width),
        orig_width=int(orig_width),
        orig_height=int(orig_height),
        attempts=[record],
    )


@dataclass
class CountResult:
    """count_v1: a point at every instance before counting. A point farther than 0.1 units from every box-mode mark
    becomes a spark_count mark with K8 (marks.count_union); the points here are raw grid points, clamped."""

    points: list[GridPoint]
    count: int
    truncated: bool
    record: SparkRequestRecord
    attempts: list[SparkRequestRecord] = field(default_factory=list)


def count_marks(
    crop_rgb: np.ndarray,
    panel_id: PanelId,
    *,
    effort: str = DEFAULT_EFFORT,
    run_id: str | None = None,
    input_ref: InputRef | None = None,
    max_edge: int = IMAGE_LONG_EDGE,
) -> CountResult:
    """The second pass of section 5.7 on the same crop."""
    effort = validate_effort(effort)
    data_url, *_ = prepare_image(crop_rgb, max_edge)
    parts = [{"type": "input_text", "text": count_prompt(panel_id)}, {"type": "input_image", "image_url": data_url}]
    attempts = _ask_attempts(parts, purpose=SparkPurpose.COUNT, schema_name="count_v1", effort=effort, run_id=run_id, input_refs=_refs(input_ref))
    record = attempts[-1]
    points: list[GridPoint] = []
    count = 0
    if record.valid and record.parsed is not None:
        for raw in record.parsed.get("points") or []:
            point = _grid_point(raw)
            if point is not None:
                points.append(GridPoint(**point))
        count = max(int(record.parsed.get("count") or 0), 0)
    return CountResult(points=points, count=count, truncated=record.truncated, record=record, attempts=attempts)


# ----------------------------------------------------------------------------------------------------------------
# Second look (look_v1): keep or downgrade, never upgrade
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class LookResult:
    """verdict same keeps the deterministic label; different or unsure become K7. Nothing here creates a match."""

    verdict: LookVerdict
    reason: str
    record: SparkRequestRecord
    attempts: list[SparkRequestRecord] = field(default_factory=list)

    @property
    def downgrade(self) -> bool:
        return self.verdict is not LookVerdict.SAME


def second_look(
    crop_a: np.ndarray,
    crop_b: np.ndarray,
    *,
    panel_id: PanelId,
    run_id: str | None = None,
    input_refs: list[InputRef] | None = None,
    effort: str = DEFAULT_EFFORT,
    prompt: str | None = None,
) -> LookResult:
    """Two crops (pickup, return) with the candidate marks already outlined by the caller. Invalid or truncated
    answers are UNSURE, a downgrade; the model can never upgrade. prompt replaces look_prompt (the counter-look
    passes counter_prompt; its first crop is the mark's own walk)."""
    effort = validate_effort(effort)
    url_a = prepare_image(crop_a)[0]
    url_b = prepare_image(crop_b)[0]
    parts = [
        {"type": "input_text", "text": prompt if prompt is not None else look_prompt(panel_id)},
        {"type": "input_image", "image_url": url_a},
        {"type": "input_image", "image_url": url_b},
    ]
    attempts = _ask_attempts(parts, purpose=SparkPurpose.SECOND_LOOK, schema_name="look_v1", effort=effort, run_id=run_id, input_refs=list(input_refs or []))
    record = attempts[-1]
    if not record.valid or record.parsed is None:
        why = record.incomplete_reason or "the answer did not fit look_v1"
        return LookResult(verdict=LookVerdict.UNSURE, reason=f"second look unusable: {why}", record=record, attempts=attempts)
    return LookResult(
        verdict=LookVerdict(record.parsed["verdict"]),
        reason=str(record.parsed.get("reason") or ""),
        record=record,
        attempts=attempts,
    )


# ----------------------------------------------------------------------------------------------------------------
# Seed (seed_v1), claim naming (claim_name_v1), drawn box (claim_box_v1)
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class SeedResult:
    found: bool
    bbox: GridBox | None
    type: MarkKind | None
    record: SparkRequestRecord
    attempts: list[SparkRequestRecord] = field(default_factory=list)


def seed_mark(
    crop_rgb: np.ndarray,
    panel_id: PanelId,
    phrase: str,
    *,
    run_id: str | None = None,
    input_ref: InputRef | None = None,
    effort: str = DEFAULT_EFFORT,
) -> SeedResult:
    """seed_v1 on the panel's sharpest blurred crop for one narration turn."""
    effort = validate_effort(effort)
    data_url, *_ = prepare_image(crop_rgb)
    parts = [{"type": "input_text", "text": seed_prompt(panel_id, phrase)}, {"type": "input_image", "image_url": data_url}]
    attempts = _ask_attempts(parts, purpose=SparkPurpose.SEED, schema_name="seed_v1", effort=effort, run_id=run_id, input_refs=_refs(input_ref))
    record = attempts[-1]
    if not record.valid or record.parsed is None:
        return SeedResult(found=False, bbox=None, type=None, record=record, attempts=attempts)
    bbox = _grid_box(record.parsed.get("bbox"))
    kind = record.parsed.get("type")
    found = bool(record.parsed.get("found")) and bbox is not None
    return SeedResult(
        found=found,
        bbox=GridBox(**bbox) if found and bbox is not None else None,
        type=MarkKind(kind) if found and kind in MARK_TYPES else None,
        record=record,
        attempts=attempts,
    )


@dataclass
class ClaimNameResult:
    """Layer 1 of section 5.11: the panel from the enumeration (None when the noun, side or position do not pin one
    slot), the raw noun, the grounded box of the named panel, and the context line."""

    panel: PanelId | None
    side: Side
    position: Position
    context: str
    front_direction: str | None
    landmarks: list[Landmark]
    record: SparkRequestRecord
    noun: Noun | None = None
    bbox: GridBox | None = None
    attempts: list[SparkRequestRecord] = field(default_factory=list)


_FIXED_PANELS = {
    Noun.HOOD: PanelId.HOOD,
    Noun.ROOF: PanelId.ROOF,
    Noun.TRUNK_LID: PanelId.TRUNK_LID,
    Noun.HEADLIGHT: PanelId.HEADLIGHTS,
    Noun.TAIL_LIGHT: PanelId.TAIL_LIGHTS,
    Noun.SIDE_MIRROR: PanelId.MIRRORS,
    Noun.WINDSHIELD: PanelId.WINDSHIELD,
    Noun.REAR_WINDOW: PanelId.REAR_WINDOW,
}
_SIDED_PANELS = {Noun.CAR_DOOR: "door", Noun.FENDER: "fender", Noun.WHEEL: "wheel"}


def panel_from_naming(noun: Noun | None, side: Side, position: Position) -> PanelId | None:
    """Noun plus side and position -> the canonical slot, or None when they do not pin one (car, car window, none)."""
    if noun is None:
        return None
    if noun in _FIXED_PANELS:
        return _FIXED_PANELS[noun]
    if noun in _SIDED_PANELS:
        if side is Side.NONE or position is Position.NONE:
            return None
        return PanelId(f"{side.value}_{position.value}_{_SIDED_PANELS[noun]}")
    if noun is Noun.BUMPER:
        return {Position.FRONT: PanelId.FRONT_BUMPER, Position.REAR: PanelId.REAR_BUMPER}.get(position)
    return None


def claim_name(
    image_rgb: np.ndarray,
    *,
    run_id: str | None = None,
    input_ref: InputRef | None = None,
    effort: str = DEFAULT_EFFORT,
) -> ClaimNameResult:
    """claim_name_v1 on the full claim image: the panel noun, side, position, the panel's box and landmarks."""
    effort = validate_effort(effort)
    data_url, *_ = prepare_image(image_rgb)
    parts = [{"type": "input_text", "text": claim_name_prompt()}, {"type": "input_image", "image_url": data_url}]
    attempts = _ask_attempts(parts, purpose=SparkPurpose.CLAIM_NAME, schema_name="claim_name_v1", effort=effort, run_id=run_id, input_refs=_refs(input_ref))
    record = attempts[-1]
    if not record.valid or record.parsed is None:
        return ClaimNameResult(None, Side.NONE, Position.NONE, "", None, [], record, attempts=attempts)
    parsed = record.parsed
    noun = Noun(parsed["panel"]) if parsed.get("panel") in PANEL_NOUN_NAMES else None
    side = Side(parsed.get("side") or "none")
    position = Position(parsed.get("position") or "none")
    front = parsed.get("front_direction")
    landmarks: list[Landmark] = []
    for item in parsed.get("landmarks") or []:
        point = _grid_point(item.get("point")) if isinstance(item, dict) else None
        if point is not None and item.get("name") and len(landmarks) < LANDMARKS_MAX:
            landmarks.append(Landmark(name=str(item["name"]), point=GridPoint(**point)))
    bbox = _grid_box(parsed.get("panel_bbox"))
    return ClaimNameResult(
        panel=panel_from_naming(noun, side, position),
        side=side,
        position=position,
        context=str(parsed.get("context") or ""),
        front_direction=front if front in ("left", "right") else None,
        landmarks=landmarks,
        record=record,
        noun=noun,
        bbox=GridBox(**bbox) if bbox is not None else None,
        attempts=attempts,
    )


@dataclass
class ClaimBoxResult:
    bbox: GridBox | None
    record: SparkRequestRecord
    attempts: list[SparkRequestRecord] = field(default_factory=list)


def claim_box(
    image_rgb: np.ndarray,
    *,
    run_id: str | None = None,
    input_ref: InputRef | None = None,
    effort: str = DEFAULT_EFFORT,
) -> ClaimBoxResult:
    """The drawn-box call when the grayscale test fires; code verifies the box against the pixels afterwards."""
    effort = validate_effort(effort)
    data_url, *_ = prepare_image(image_rgb)
    parts = [{"type": "input_text", "text": claim_box_prompt()}, {"type": "input_image", "image_url": data_url}]
    attempts = _ask_attempts(parts, purpose=SparkPurpose.CLAIM_BOX, schema_name="claim_box_v1", effort=effort, run_id=run_id, input_refs=_refs(input_ref))
    record = attempts[-1]
    bbox = None
    if record.valid and record.parsed is not None and record.parsed.get("found"):
        raw = _grid_box(record.parsed.get("bbox"))
        bbox = GridBox(**raw) if raw is not None else None
    return ClaimBoxResult(bbox=bbox, record=record, attempts=attempts)


# ----------------------------------------------------------------------------------------------------------------
# Probes S2 and S2b (E17, E18): streamed, because they can run past the 30 s non-streaming limit
# ----------------------------------------------------------------------------------------------------------------


def timeline(file_id: str, *, run_id: str | None = None, effort: str = DEFAULT_EFFORT) -> SparkRequestRecord:
    """timeline_v1 over the whole blurred proxy through the Files API id."""
    # input_file with a file_id was measured on 10/26 (non-streamed, 6 s clip). TODO(E17): confirm it streamed with
    # timeline_v1, and whether input_video with the same file_id differs.
    effort = validate_effort(effort)
    parts = [{"type": "input_text", "text": timeline_prompt()}, {"type": "input_file", "file_id": file_id}]
    return ask_with_retry(parts, purpose=SparkPurpose.TIMELINE, schema_name="timeline_v1", effort=effort, run_id=run_id, input_refs=[], stream=True)


def note_prefill(claim_text: str, agreement_pdf: Path, *, run_id: str | None = None, effort: str = DEFAULT_EFFORT) -> SparkRequestRecord:
    """note_prefill_v1 from the claim text and a rental agreement PDF sent inline."""
    # TODO(E18): the inline part {type: input_file, filename, file_data: data:application/pdf;base64,...} is the
    # OpenAI shape and has never been sent to the Meta API; if E18 gets a 400, upload the PDF and send its file_id.
    effort = validate_effort(effort)
    pdf = Path(agreement_pdf)
    file_data = "data:application/pdf;base64," + base64.b64encode(pdf.read_bytes()).decode("ascii")
    parts = [
        {"type": "input_text", "text": note_prefill_prompt(claim_text)},
        {"type": "input_file", "filename": pdf.name, "file_data": file_data},
    ]
    return ask_with_retry(parts, purpose=SparkPurpose.NOTE_PREFILL, schema_name="note_prefill_v1", effort=effort, run_id=run_id, input_refs=[], stream=True)


# ----------------------------------------------------------------------------------------------------------------
# Grid -> sent pixels -> original crop pixels
# ----------------------------------------------------------------------------------------------------------------


def _undo_resize(v: int, scale: float, dim: int) -> int:
    return int(min(max(round(v / (scale or 1.0)), 0), dim - 1))


def point_to_crop_pixels(p: GridPoint, result: MarksResult) -> tuple[int, int]:
    """Grid -> pixels of the sent image -> pixels of the crop before the resize."""
    px, py = grid_to_pixels(p.x, p.y, result.sent_width, result.sent_height)
    return _undo_resize(px, result.scale, result.orig_width), _undo_resize(py, result.scale, result.orig_height)


def box_to_crop_pixels(b: GridBox, result: MarksResult) -> tuple[int, int, int, int]:
    """The same for a box; right and bottom stay half-open and at least one pixel past the left and top."""
    sx1, sy1, sx2, sy2 = grid_box_to_pixels(b, result.sent_width, result.sent_height)
    scale = result.scale or 1.0
    x1, y1 = _undo_resize(sx1, scale, result.orig_width), _undo_resize(sy1, scale, result.orig_height)
    x2 = int(min(max(round(sx2 / scale), x1 + 1), result.orig_width))
    y2 = int(min(max(round(sy2 / scale), y1 + 1), result.orig_height))
    return x1, y1, x2, y2


__all__ = [
    "SCHEMAS",
    "MAX_OUTPUT_TOKENS",
    "RETRY_MAX_OUTPUT_TOKENS",
    "IMAGE_LONG_EDGE",
    "CLIP_LONG_EDGE",
    "EFFORTS",
    "SYSTEM_PROMPT",
    "MarksResult",
    "CountResult",
    "LookResult",
    "SeedResult",
    "ClaimNameResult",
    "ClaimBoxResult",
    "ask",
    "ask_with_retry",
    "box_to_crop_pixels",
    "claim_box",
    "claim_name",
    "clamp_grid",
    "count_marks",
    "find_marks",
    "grid_box_to_pixels",
    "grid_to_pixels",
    "marks_schema",
    "note_prefill",
    "panel_display_name",
    "panel_from_naming",
    "point_to_crop_pixels",
    "prepare_image",
    "schema_problems",
    "second_look",
    "seed_mark",
    "set_client",
    "timeline",
    "validate_effort",
]
