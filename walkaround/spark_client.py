"""Muse Spark 1.3 as a grounded mark finder and as a downgrade-only second look.

Rules (CLAUDE.md): the model never writes the verdict. find_marks returns marks with coordinates; deterministic code
in match.py decides pre-existing vs new; compare_marks may only say "not sure" (a downgrade), never "same".

Request shape (docs/API_NOTES.md): Responses API, input_image data URL (long edge <= 1280), grounding in the
0 to 1000 grid in the user message, strict JSON schema via text.format, reasoning effort low. Map grid -> crop pixels
with round(v * dim / 1000) on the image the model received, clamp to bounds.

The text.format shape was checked against https://dev.meta.ai/docs/api-reference/responses/schemas on 2026-10-26:
the json_schema variant is flat ({"type": "json_schema", "name", "schema", "strict"}), no nested json_schema object,
and strict needs additionalProperties false plus every key in required on each object, no allOf or oneOf.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image

from . import ledger
from .config import BASE_URL, SPARK_MODEL, api_key

MarkKind = Literal["scratch", "scuff", "dent", "chip", "crack", "stain", "other"]
SizeClass = Literal["small", "medium", "large"]

GRID = 1000
MAX_EDGE = 1280

MARK_KINDS = ["scratch", "scuff", "dent", "chip", "crack", "stain", "other"]
SIZE_CLASSES = ["small", "medium", "large"]
EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")

# The docs say the model can emit 1001. Values this close to the grid edge are clamped; anything further out is not
# a point on the image we sent, so that mark is dropped.
GRID_SLACK = 20
JPEG_QUALITY = 90
# Reasoning tokens count against max_output_tokens. Effort low keeps them small; the JSON is a few hundred tokens.
MAX_OUTPUT_TOKENS = 4096
COMPARE_MAX_OUTPUT_TOKENS = 2048

SYSTEM_PROMPT = "You inspect close-up photos of surfaces for damage and answer only with JSON that fits the schema."

_MARK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": MARK_KINDS},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "size": {"type": "string", "enum": SIZE_CLASSES},
        "confidence": {"type": "number"},
        "description": {"type": "string"},
    },
    "required": ["kind", "x", "y", "size", "confidence", "description"],
    "additionalProperties": False,
}

MARKS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "marks": {"type": "array", "items": _MARK_SCHEMA},
        "panel_condition": {
            "type": "object",
            "properties": {
                "glare": {"type": "boolean"},
                "wet": {"type": "boolean"},
                "dirty": {"type": "boolean"},
                "notes": {"type": "string"},
            },
            "required": ["glare", "wet", "dirty", "notes"],
            "additionalProperties": False,
        },
    },
    "required": ["marks", "panel_condition"],
    "additionalProperties": False,
}

COMPARE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"keep": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["keep", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Mark:
    kind: str  # MarkKind
    x: int  # grid 0..1000 in the sent image
    y: int
    size: str  # SizeClass
    confidence: float  # 0..1 as reported by the model
    description: str
    bbox: tuple[int, int, int, int] | None = None  # grid x_min, y_min, x_max, y_max if given


@dataclass
class MarksResult:
    marks: list[Mark]
    sent_width: int  # size of the image the model received
    sent_height: int
    scale: float  # sent / original (<= 1)
    usage: dict[str, Any] = field(default_factory=dict)
    usd: float = 0.0
    raw: str = ""
    condition: dict[str, Any] = field(default_factory=dict)  # glare, wet, dirty, notes from panel_condition
    orig_width: int = 0  # size of the crop before the resize, for mark_to_crop_pixels
    orig_height: int = 0
    seconds: float = 0.0  # wall time of the call


def marks_prompt(panel_name: str) -> str:
    """Grounding prompt for find_marks, per the image-understanding doc: expert framing, grid, do not miss."""
    return (
        f"You are an object grounding expert. This image is a close crop of the {panel_name}. Find every scratch, "
        f"scuff, dent, chip, crack or stain on the painted or glass surface of the {panel_name}. Do not miss any "
        "mark, including small ones. Put one point at the center of each distinct mark. Coordinates are on a 0 to "
        "1000 grid over the whole image, (0, 0) at the top left, x to the right, y down. For each mark give its "
        "kind, a size class (small: coin sized or less, medium: up to a hand, large: bigger than a hand), your "
        "confidence from 0 to 1 and a short description. Reflections, water drops, dirt, shadows, badges and panel "
        "gaps are not marks: skip them, and flag glare, wet or dirty conditions under panel_condition with a note. "
        "If the surface is clean, return an empty marks list."
    )


def compare_prompt(panel_name: str, mark_a: Mark, mark_b: Mark) -> str:
    """Second-look prompt for compare_marks. The answer can only be keep true or false with a reason."""
    return (
        f"The first image and the second image show the {panel_name} filmed at two different times. Coordinates "
        "are on a 0 to 1000 grid over each image, (0, 0) at the top left. Deterministic matching paired the "
        f"{mark_a.kind} at ({mark_a.x}, {mark_a.y}) in the first image with the {mark_b.kind} at "
        f"({mark_b.x}, {mark_b.y}) in the second image as the same mark. Look at both spots. Answer keep true if you "
        "see no reason to doubt the pairing. Answer keep false if the two spots look like different marks, if a mark "
        "is missing from one image, or if glare, water, dirt or blur make it impossible to tell. Give a one "
        "sentence reason."
    )


_CLIENT: Any = None


def _client() -> Any:
    """OpenAI SDK client for the Meta Model API, made on first use. Tests monkeypatch this."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        _CLIENT = OpenAI(base_url=BASE_URL, api_key=api_key(), timeout=120.0, max_retries=3)
    return _CLIENT


def _to_pil(rgb: np.ndarray) -> Image.Image:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGB")
    return Image.fromarray(np.ascontiguousarray(arr[:, :, :3]), "RGB")


def prepare_image(rgb: np.ndarray, max_edge: int = MAX_EDGE) -> tuple[str, int, int, float]:
    """JPEG data URL of the image resized so the long edge <= max_edge. Returns (data_url, w, h, scale)."""
    img = _to_pil(rgb)
    width, height = img.size
    scale = min(1.0, max_edge / max(width, height))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        img = img.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return data_url, img.width, img.height, img.width / width


def grid_to_pixels(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """round(v * dim / 1000), clamped into [0, dim-1]."""
    px = min(max(round(x * width / GRID), 0), width - 1)
    py = min(max(round(y * height / GRID), 0), height - 1)
    return int(px), int(py)


def _ask(parts: list[dict[str, Any]], *, name: str, schema: dict[str, Any], effort: str, max_output_tokens: int) -> tuple[Any, float]:
    """One Responses call with a strict JSON schema. Returns (response, seconds)."""
    started = time.monotonic()
    response = _client().responses.create(
        model=SPARK_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"type": "message", "role": "user", "content": parts},
        ],
        reasoning={"effort": effort},
        text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        max_output_tokens=max_output_tokens,
    )
    return response, time.monotonic() - started


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    details_in = getattr(usage, "input_tokens_details", None)
    details_out = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cached_tokens": int(getattr(details_in, "cached_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(details_out, "reasoning_tokens", 0) or 0),
    }


def _record(response: Any, *, note: str, run_id: str | None, seconds: float) -> tuple[dict[str, int], float]:
    """Ledger row for one call. Written before parsing: the call cost money even if the JSON is bad."""
    usage = _usage_dict(response)
    row = ledger.record("spark", SPARK_MODEL, note=note, run_id=run_id, seconds=round(seconds, 3), **usage)
    return usage, row["usd"]


def _parse_json(response: Any) -> dict[str, Any]:
    text = getattr(response, "output_text", "") or ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        status = getattr(response, "status", None)
        raise ValueError(f"Spark returned non-JSON output (status {status!r}): {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Spark returned JSON that is not an object: {text[:200]!r}")
    return data


def _grid_value(value: Any) -> int | None:
    """Clamped grid coordinate, or None when it is too far off the image to be a point on it."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < -GRID_SLACK or v > GRID + GRID_SLACK:
        return None
    return int(min(max(round(v), 0), GRID))


def _unit(value: Any) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_marks(items: Any) -> list[Mark]:
    # bbox stays None: MARKS_SCHEMA asks for one point per mark.
    marks: list[Mark] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        x, y = _grid_value(item.get("x")), _grid_value(item.get("y"))
        if x is None or y is None:
            continue
        kind = item.get("kind") if item.get("kind") in MARK_KINDS else "other"
        size = item.get("size") if item.get("size") in SIZE_CLASSES else "medium"
        marks.append(
            Mark(
                kind=kind,
                x=x,
                y=y,
                size=size,
                confidence=_unit(item.get("confidence")),
                description=str(item.get("description", "")),
            )
        )
    return marks


def _parse_condition(data: Any) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    return {
        "glare": bool(data.get("glare", False)),
        "wet": bool(data.get("wet", False)),
        "dirty": bool(data.get("dirty", False)),
        "notes": str(data.get("notes", "")),
    }


def find_marks(crop_rgb: np.ndarray, panel_name: str, *, effort: str = "low", run_id: str | None = None) -> MarksResult:
    """Ask Spark for every visible mark on this panel crop as points in the 0..1000 grid, with kind, size class,
    confidence and a short description. Strict JSON. Records a ledger row (kind spark)."""
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
    orig_height, orig_width = np.asarray(crop_rgb).shape[:2]
    data_url, sent_width, sent_height, scale = prepare_image(crop_rgb)
    parts = [
        {"type": "input_text", "text": marks_prompt(panel_name)},
        {"type": "input_image", "image_url": data_url},
    ]
    response, seconds = _ask(parts, name="marks", schema=MARKS_SCHEMA, effort=effort, max_output_tokens=MAX_OUTPUT_TOKENS)
    usage, usd = _record(response, note=panel_name, run_id=run_id, seconds=seconds)
    data = _parse_json(response)
    return MarksResult(
        marks=_parse_marks(data.get("marks")),
        sent_width=sent_width,
        sent_height=sent_height,
        scale=scale,
        usage=usage,
        usd=usd,
        raw=getattr(response, "output_text", "") or "",
        condition=_parse_condition(data.get("panel_condition")),
        orig_width=int(orig_width),
        orig_height=int(orig_height),
        seconds=seconds,
    )


def compare_marks(
    crop_a: np.ndarray, crop_b: np.ndarray, mark_a: Mark, mark_b: Mark, *, panel_name: str, run_id: str | None = None
) -> tuple[bool, str]:
    """Second look at a deterministic match. Returns (keep, reason). keep=False downgrades the match to
    'needs your eye'. The model can never upgrade or create a match. Records a ledger row."""
    # This call can only downgrade. match.py pairs marks by geometry before it gets here and the only effect of the
    # answer is keep=False, which sends that pair to "needs your eye". keep=True changes nothing, and nothing here
    # returns a mark, a pair or a verdict, so there is no path from this function to a new or upgraded match.
    url_a, *_ = prepare_image(crop_a)
    url_b, *_ = prepare_image(crop_b)
    parts = [
        {"type": "input_text", "text": compare_prompt(panel_name, mark_a, mark_b)},
        {"type": "input_image", "image_url": url_a},
        {"type": "input_image", "image_url": url_b},
    ]
    response, seconds = _ask(
        parts, name="compare", schema=COMPARE_SCHEMA, effort="low", max_output_tokens=COMPARE_MAX_OUTPUT_TOKENS
    )
    _record(response, note=f"compare {panel_name}", run_id=run_id, seconds=seconds)
    data = _parse_json(response)
    # A missing answer counts as doubt.
    return bool(data.get("keep", False)), str(data.get("reason", ""))


def mark_to_crop_pixels(mark: Mark, result: MarksResult) -> tuple[int, int]:
    """Mark position in ORIGINAL crop pixels (undo the resize)."""
    px, py = grid_to_pixels(mark.x, mark.y, result.sent_width, result.sent_height)
    scale = result.scale or 1.0
    width = result.orig_width or max(1, round(result.sent_width / scale))
    height = result.orig_height or max(1, round(result.sent_height / scale))
    x = min(max(round(px / scale), 0), width - 1)
    y = min(max(round(py / scale), 0), height - 1)
    return int(x), int(y)
