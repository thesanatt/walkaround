"""Offline tests for walkaround.spark_client: a fake client with canned Responses, the ledger on a temp path.

The one live test runs only with LIVE=1 (one marks_v1 call on a generated 1280 px image, about $0.01)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest
from openai import APITimeoutError, AuthenticationError, BadRequestError, InternalServerError, RateLimitError
from PIL import Image, ImageDraw
from pydantic import ValidationError

from walkaround import ledger, spark_client
from walkaround.config import PRICES, SPARK_MODEL
from walkaround.contracts import (
    Confidence,
    GridBox,
    GridPoint,
    InputRef,
    LedgerRow,
    LookVerdict,
    MarkKind,
    Noun,
    PanelId,
    Position,
    ReasoningEffort,
    Side,
    SizeClass,
    SparkMarksAnswer,
    SparkPurpose,
    SparkRequestRecord,
    SparkStatus,
    canonical_json,
    load_thresholds,
)

SHA = "a" * 64

GOOD_MARKS = {
    "marks": [
        {
            "type": "scratch",
            "point": {"x": 500, "y": 400},
            "bbox": {"x_min": 450, "y_min": 390, "x_max": 560, "y_max": 410},
            "size_class": "M",
            "confidence": "high",
            "note": "thin dark line",
        },
        {
            "type": "dirt",
            "point": {"x": 1001, "y": 0},
            "bbox": {"x_min": 990, "y_min": 0, "x_max": 1001, "y_max": 12},
            "size_class": "S",
            "confidence": "low",
            "note": "speck at the corner",
        },
        {
            "type": "chip",
            "point": {"x": 1500, "y": 20},
            "bbox": {"x_min": 1490, "y_min": 10, "x_max": 1510, "y_max": 30},
            "size_class": "S",
            "confidence": "medium",
            "note": "off the image, dropped",
        },
    ],
    "conditions": {"wet": False, "dirty": False, "glare": True, "dark": False},
    "landmarks": [{"name": "handle", "point": {"x": 700, "y": 500}}],
    "neighbour_car_visible": False,
}


def completed(payload, *, rid: str = "resp_1", inp: int = 1400, out: int = 600, reasoning: int = 450, cached: int = 0) -> dict:
    """A Responses object as model_dump() gives it, status completed."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "id": rid,
        "object": "response",
        "model": SPARK_MODEL,
        "status": "completed",
        "incomplete_details": None,
        "error": None,
        "store": False,
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]},
        ],
        "usage": {
            "input_tokens": inp,
            "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
            "output_tokens": out,
            "output_tokens_details": {"reasoning_tokens": reasoning},
            "total_tokens": inp + out,
        },
    }


def incomplete(text: str = "", *, reason: str = "max_output_tokens", rid: str = "resp_cut", out: int = 4000) -> dict:
    data = completed(text, rid=rid, out=out, reasoning=out)
    data["status"] = "incomplete"
    data["incomplete_details"] = {"reason": reason}
    return data


class FakeResponses:
    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        assert self.replies, "the fake ran out of canned replies"
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if kwargs.get("stream"):
            return iter(reply)
        return reply


class FakeClient:
    def __init__(self, *replies) -> None:
        self.responses = FakeResponses(list(replies))


@pytest.fixture
def ledger_path(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(ledger, "LEDGER_PATH", path)
    return path


@pytest.fixture
def fake(monkeypatch, ledger_path):
    """Install a fake client; tests load replies with fake.responses.replies.extend(...)."""
    client = FakeClient()
    monkeypatch.setattr(spark_client, "_CLIENT", client)
    return client


def crop(width: int = 600, height: int = 400) -> np.ndarray:
    rgb = np.full((height, width, 3), 180, np.uint8)
    rgb[200:204, 100:500] = 40  # a dark line
    return rgb


def requests_of(fake: FakeClient) -> list[dict]:
    return fake.responses.requests


# Schemas, grid, images -----------------------------------------------------------------------------------------


def every_object_is_strict(schema: dict) -> bool:
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False or set(schema.get("required", [])) != set(props):
            return False
        return all(every_object_is_strict(v) for v in props.values())
    if schema.get("type") == "array":
        return every_object_is_strict(schema["items"])
    if "anyOf" in schema:
        return all(every_object_is_strict(v) for v in schema["anyOf"])
    return True


def test_schemas_present_and_strict():
    th = load_thresholds()
    assert set(spark_client.SCHEMAS) == set(th["spark"]["schema_names"]) | {"claim_box_v1"}
    for name, schema in spark_client.SCHEMAS.items():
        assert every_object_is_strict(schema), name
        assert "anyOf" not in schema, f"{name}: anyOf is not allowed at the root"
    mark = spark_client.SCHEMAS["marks_v1"]["properties"]["marks"]["items"]["properties"]
    assert mark["type"]["enum"] == [k.value for k in MarkKind] == th["spark"]["mark_types"]
    assert mark["size_class"]["enum"] == th["spark"]["size_classes"]
    assert mark["confidence"]["enum"] == th["spark"]["confidence_levels"]
    assert spark_client.MAX_OUTPUT_TOKENS == th["spark"]["max_output_tokens"] == 4000
    assert spark_client.RETRY_MAX_OUTPUT_TOKENS == th["spark"]["retry_max_output_tokens"] == 8000
    assert spark_client.IMAGE_LONG_EDGE == th["spark"]["image_long_edge_px"]
    assert spark_client.GRID_SLACK == th["spark"]["grid_slack"]
    assert spark_client.DEFAULT_EFFORT == th["spark"]["effort"]
    # Every other value the module mirrors from config/thresholds.json, so a config change cannot leave it behind.
    assert spark_client.GRID == th["spark"]["grid"]
    assert spark_client.JPEG_QUALITY == th["spark"]["jpeg_quality"]
    assert spark_client.LANDMARKS_MAX == th["spark"]["landmarks_max"]
    assert spark_client.CLIP_LONG_EDGE == th["media"]["clip_size_px"]
    assert spark_client.TIMEOUT_S == th["orchestrator"]["request_timeout_s"]
    assert spark_client.RATE_LIMIT_TRIES == th["orchestrator"]["rate_limit_retries"]
    assert spark_client.BACKOFF_BASE_S == th["orchestrator"]["rate_limit_backoff_base_s"]
    assert SPARK_MODEL == th["spark"]["model"]


def test_marks_schema_landmark_enum():
    schema = spark_client.marks_schema(["handle", "lock"])
    assert schema["properties"]["landmarks"]["items"]["properties"]["name"]["enum"] == ["handle", "lock"]
    assert "enum" not in spark_client.SCHEMAS["marks_v1"]["properties"]["landmarks"]["items"]["properties"]["name"]


def test_schema_problems_subset():
    look = spark_client.SCHEMAS["look_v1"]
    assert spark_client.schema_problems(look, {"verdict": "same", "reason": "ok"}) == []
    assert any("not in the enum" in p for p in spark_client.schema_problems(look, {"verdict": "maybe", "reason": "x"}))
    assert any("missing" in p for p in spark_client.schema_problems(look, {"verdict": "same"}))
    assert any("not in the schema" in p for p in spark_client.schema_problems(look, {"verdict": "same", "reason": "x", "extra": 1}))
    seed = spark_client.SCHEMAS["seed_v1"]
    assert spark_client.schema_problems(seed, {"found": False, "bbox": None, "type": None}) == []
    assert spark_client.schema_problems(seed, {"found": True, "bbox": {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}, "type": "dent"}) == []
    assert spark_client.schema_problems(seed, {"found": True, "bbox": "nope", "type": "dent"})
    assert spark_client.schema_problems(spark_client.SCHEMAS["count_v1"], {"points": [], "count": 2.5})
    assert spark_client.schema_problems(spark_client.SCHEMAS["count_v1"], {"points": [{"x": 1, "y": 2}], "count": 1}) == []
    assert spark_client.schema_problems({"type": "integer"}, True)


def test_prepare_image_resizes_long_edge_and_never_upscales():
    url, w, h, scale = spark_client.prepare_image(np.zeros((1600, 2560, 3), np.uint8))
    assert (w, h, scale) == (1280, 800, 0.5)
    assert url.startswith("data:image/jpeg;base64,")
    url, w, h, scale = spark_client.prepare_image(np.zeros((300, 200), np.uint8))
    assert (w, h, scale) == (200, 300, 1.0)
    url, w, h, scale = spark_client.prepare_image(np.zeros((2000, 2000, 3), np.uint8), max_edge=1008)
    assert (w, h) == (1008, 1008) and scale == pytest.approx(0.504)


def test_grid_to_pixels_rounds_and_clamps():
    assert spark_client.grid_to_pixels(500, 500, 1280, 800) == (640, 400)
    assert spark_client.grid_to_pixels(1000, 1000, 1280, 800) == (1279, 799)
    assert spark_client.grid_to_pixels(0, 0, 1280, 800) == (0, 0)
    assert spark_client.grid_to_pixels(1, 1, 1280, 800) == (1, 1)
    box = GridBox(x_min=0, y_min=250, x_max=1000, y_max=250)
    assert spark_client.grid_box_to_pixels(box, 1280, 800) == (0, 200, 1280, 201)


def test_clamp_grid():
    assert spark_client.clamp_grid(1001) == 1000
    assert spark_client.clamp_grid(1020) == 1000
    assert spark_client.clamp_grid(1021) is None
    assert spark_client.clamp_grid(-20) == 0
    assert spark_client.clamp_grid(-21) is None
    assert spark_client.clamp_grid(499.6) == 500
    assert spark_client.clamp_grid("12") == 12
    assert spark_client.clamp_grid("abc") is None
    assert spark_client.clamp_grid(None) is None
    assert spark_client.clamp_grid(True) is None


def test_contract_rejects_1001_before_the_clamp_and_accepts_after():
    with pytest.raises(ValidationError):
        GridPoint(x=1001, y=0)
    assert GridPoint(x=spark_client.clamp_grid(1001), y=0).x == 1000
    answer = spark_client._marks_answer(GOOD_MARKS)
    assert isinstance(answer, SparkMarksAnswer)
    assert [m.type for m in answer.marks] == [MarkKind.SCRATCH, MarkKind.DIRT]  # the 1500 mark is dropped
    assert answer.marks[1].point.x == 1000 and answer.marks[1].bbox.x_max == 1000
    assert answer.conditions.glare and not answer.conditions.wet
    assert answer.landmarks[0].name == "handle"
    assert spark_client._marks_answer({"marks": [{"type": "lava"}]}) is not None  # bad items are dropped, not fatal
    assert spark_client._marks_answer(None) is None


def test_effort_validation_happens_before_any_call(fake):
    with pytest.raises(ValueError, match="reasoning effort"):
        spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[], effort="none")
    assert requests_of(fake) == []
    assert spark_client.EFFORTS == tuple(e.value for e in ReasoningEffort)


# find_marks ------------------------------------------------------------------------------------------------------


def test_find_marks_request_shape_record_and_ledger(fake, ledger_path):
    fake.responses.replies.append(completed(GOOD_MARKS, inp=1400, out=600, reasoning=450, cached=100))
    ref = InputRef(path="crops/pickup/kf1.jpg", sha256=SHA)
    result = spark_client.find_marks(crop(2560, 1600), PanelId.LEFT_FRONT_DOOR, landmarks=["handle", "lock", "mirror_base", "wheel_arch", "badge"], run_id="run1", input_ref=ref)

    (request,) = requests_of(fake)
    assert request["model"] == SPARK_MODEL
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "low"}
    assert request["max_output_tokens"] == 4000
    assert "stream" not in request
    fmt = request["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["name"] == "marks_v1" and fmt["strict"] is True
    assert fmt["schema"]["properties"]["landmarks"]["items"]["properties"]["name"]["enum"] == ["handle", "lock", "mirror_base", "wheel_arch", "badge"]
    assert fmt["schema"]["additionalProperties"] is False
    system, user = request["input"]
    assert system == {"role": "system", "content": spark_client.SYSTEM_PROMPT}
    assert user["type"] == "message" and user["role"] == "user"
    text_part, image_part = user["content"]
    assert text_part["type"] == "input_text"
    assert "left front door" in text_part["text"] and "0 to 1000 grid" in text_part["text"] and "Do not miss" in text_part["text"]
    assert "handle, lock, mirror_base, wheel_arch, badge" in text_part["text"] and "up to 4 of these" in text_part["text"]
    assert image_part["type"] == "input_image" and image_part["image_url"].startswith("data:image/jpeg;base64,")

    assert result.answer is not None and len(result.answer.marks) == 2
    assert result.truncated is False
    assert (result.sent_width, result.sent_height, result.scale, result.orig_width, result.orig_height) == (1280, 800, 0.5, 2560, 1600)
    record = result.record
    assert isinstance(record, SparkRequestRecord)
    assert record.request_id == "resp_1" and record.purpose is SparkPurpose.MARKS
    assert record.schema_name == "marks_v1" and record.valid and record.status is SparkStatus.COMPLETED
    assert record.store is False and record.max_output_tokens == 4000 and record.reasoning_effort is ReasoningEffort.LOW
    assert record.retry == 0 and record.truncated is False and record.incomplete_reason is None
    assert record.input_refs == [ref]
    assert json.loads(record.raw_text) == GOOD_MARKS and record.parsed == GOOD_MARKS
    assert (record.usage.input, record.usage.output, record.usage.reasoning, record.usage.cached) == (1400, 600, 450, 100)
    expected = 1300 * PRICES["spark_input_token"] + 100 * PRICES["spark_cached_token"] + 600 * PRICES["spark_output_token"]
    assert record.dollars == pytest.approx(expected, abs=1e-6)
    assert record.ms >= 0
    canonical_json(record)  # the record is writable as canonical JSON
    assert result.attempts == [record]

    rows = ledger.rows(ledger_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "spark" and row["model"] == SPARK_MODEL and row["note"] == "marks" and row["run_id"] == "run1"
    assert row["usd"] == pytest.approx(expected, abs=1e-6)
    assert (row["input_tokens"], row["output_tokens"], row["cached_tokens"], row["reasoning_tokens"]) == (1400, 600, 100, 450)
    assert row["tokens"] == {"in": 1400, "out": 600, "reasoning": 450, "cached": 100}
    assert row["request_id"] == "resp_1" and row["status"] == "completed" and row["retries"] == 0
    assert row["engine"] == "meta-model-api" and row["unit"] == "tokens" and row["quantity"] == 2000
    assert row["latency_ms"]["done"] >= 0
    LedgerRow.model_validate(row)


def test_landmark_outside_the_closed_list_is_dropped_not_a_schema_miss(fake, ledger_path):
    """The strict enum goes to the server; locally an unknown landmark name never throws away the marks of a crop."""
    fake.responses.replies.append(completed(GOOD_MARKS, rid="lm"))
    result = spark_client.find_marks(crop(), PanelId.FRONT_BUMPER, landmarks=["fog_lamp", "sensor"])
    assert len(requests_of(fake)) == 1
    assert requests_of(fake)[0]["text"]["format"]["schema"]["properties"]["landmarks"]["items"]["properties"]["name"]["enum"] == ["fog_lamp", "sensor"]
    assert result.record.valid and result.answer is not None
    assert [m.type for m in result.answer.marks] == [MarkKind.SCRATCH, MarkKind.DIRT] and result.answer.landmarks == []
    assert ledger.rows(ledger_path)[0]["status"] == "completed"
    kept = spark_client._marks_answer(GOOD_MARKS, landmark_names=["handle"])
    assert kept is not None and [lm.name for lm in kept.landmarks] == ["handle"]
    assert spark_client._marks_answer(GOOD_MARKS, landmark_names=[]).landmarks == []
    many = {**GOOD_MARKS, "landmarks": [{"name": f"n{i}", "point": {"x": i, "y": i}} for i in range(6)]}
    assert len(spark_client._marks_answer(many).landmarks) == spark_client.LANDMARKS_MAX


def test_find_marks_sends_the_whole_closed_list_and_keeps_at_most_four(fake):
    """5 step 8: up to 4 landmarks FROM the panel's closed list. The list itself (8 names on a door) is never cut."""
    th = load_thresholds()
    door = th["spark"]["landmarks"]["door"]
    assert len(door) == 8 > spark_client.LANDMARKS_MAX == th["spark"]["landmarks_max"]
    answered = [{"name": n, "point": {"x": 100 + 50 * i, "y": 500}} for i, n in enumerate(reversed(door))][:6]
    fake.responses.replies.append(completed({**GOOD_MARKS, "landmarks": answered}, rid="lm8"))
    result = spark_client.find_marks(crop(), PanelId.RIGHT_REAR_DOOR, landmarks=door + ["handle"])
    sent = requests_of(fake)[0]
    assert sent["text"]["format"]["schema"]["properties"]["landmarks"]["items"]["properties"]["name"]["enum"] == door  # the repeat is dropped
    text = sent["input"][1]["content"][0]["text"]
    assert ", ".join(door) in text and f"up to {spark_client.LANDMARKS_MAX} of these" in text
    assert result.answer is not None
    assert [lm.name for lm in result.answer.landmarks] == ["badge", "lower_trim", "wheel_arch", "window_rear_corner"]


def test_stays_put_crop_is_sent_at_1008(fake):
    import base64
    import io

    fake.responses.replies.append(completed(GOOD_MARKS))
    result = spark_client.find_marks(crop(2016, 2016), PanelId.HOOD, landmarks=[], max_edge=spark_client.CLIP_LONG_EDGE)
    url = requests_of(fake)[0]["input"][1]["content"][1]["image_url"]
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert sent.size == (1008, 1008) and (result.sent_width, result.sent_height, result.scale) == (1008, 1008, 0.5)


def test_find_marks_without_landmarks_asks_for_an_empty_list(fake):
    fake.responses.replies.append(completed({**GOOD_MARKS, "landmarks": []}))
    result = spark_client.find_marks(crop(), PanelId.ROOF, landmarks=[])
    text = requests_of(fake)[0]["input"][1]["content"][0]["text"]
    assert "landmarks: return an empty list" in text
    assert result.answer is not None and result.answer.landmarks == []


def test_schema_miss_retries_once_then_valid(fake, ledger_path):
    fake.responses.replies.extend([completed("not json at all", rid="r1"), completed(GOOD_MARKS, rid="r2")])
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=["badge"])
    assert [r["max_output_tokens"] for r in requests_of(fake)] == [4000, 4000]
    assert result.answer is not None and result.record.request_id == "r2" and result.record.retry == 1
    assert len(result.answer.marks) == 2 and result.answer.landmarks == []  # "handle" is outside this panel's ["badge"] list
    assert [a.request_id for a in result.attempts] == ["r1", "r2"]
    assert result.attempts[0].valid is False and result.attempts[0].status is SparkStatus.COMPLETED
    assert result.attempts[0].parsed is None and result.attempts[0].raw_text == "not json at all"
    assert [row["status"] for row in ledger.rows(ledger_path)] == ["schema_miss", "completed"]


def test_schema_miss_twice_gives_no_answer_and_is_not_truncated(fake, ledger_path):
    wrong = {"marks": [{"kind": "scratch"}], "conditions": {}, "landmarks": [], "neighbour_car_visible": False}
    fake.responses.replies.extend([completed(wrong, rid="r1"), completed(wrong, rid="r2")])
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[])
    assert result.answer is None and result.truncated is False
    assert result.record.valid is False and result.record.status is SparkStatus.COMPLETED and result.record.retry == 1
    assert result.record.parsed == wrong  # stored even though it misses the schema
    assert len(requests_of(fake)) == 2
    assert [row["status"] for row in ledger.rows(ledger_path)] == ["schema_miss", "schema_miss"]


def test_truncation_retries_at_8000_then_records_truncated(fake, ledger_path):
    fake.responses.replies.extend([incomplete('{"marks": [{"type": "scr', rid="c1"), incomplete('{"marks": [', rid="c2", out=8000)])
    result = spark_client.find_marks(crop(), PanelId.FRONT_BUMPER, landmarks=["fog_lamp"], run_id="run1")
    assert [r["max_output_tokens"] for r in requests_of(fake)] == [4000, 8000]
    assert result.answer is None  # never an empty valid answer
    assert result.truncated is True
    record = result.record
    assert record.truncated is True and record.valid is False and record.status is SparkStatus.INCOMPLETE
    assert record.incomplete_reason == "max_output_tokens" and record.retry == 1 and record.max_output_tokens == 8000
    assert record.raw_text == '{"marks": ['
    first = result.attempts[0]
    assert first.status is SparkStatus.INCOMPLETE and first.truncated is False and first.retry == 0
    rows = ledger.rows(ledger_path)
    assert [row["status"] for row in rows] == ["incomplete", "truncated"]
    assert rows[1]["retries"] == 1 and rows[1]["output_tokens"] == 8000
    assert rows[1]["usd"] > 0  # the truncated call still cost money


def test_schema_miss_then_cut_at_4000_is_truncated_without_a_third_call(fake, ledger_path):
    """Two rows at most: the 8000 budget follows only a cut on the first attempt; a later cut still reads truncated."""
    fake.responses.replies.extend([completed("garbage", rid="m1"), incomplete('{"marks": [', rid="m2")])
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[])
    assert [r["max_output_tokens"] for r in requests_of(fake)] == [4000, 4000]
    assert result.answer is None and result.truncated is True
    assert result.record.max_output_tokens == 4000 and result.record.incomplete_reason == "max_output_tokens"
    assert [row["status"] for row in ledger.rows(ledger_path)] == ["schema_miss", "truncated"]


def test_truncation_then_success_is_not_truncated(fake):
    fake.responses.replies.extend([incomplete(rid="c1"), completed(GOOD_MARKS, rid="ok", out=2500)])
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[])
    assert result.answer is not None and result.truncated is False and result.record.retry == 1
    assert requests_of(fake)[1]["max_output_tokens"] == 8000


def test_api_error_is_recorded_not_raised(fake, ledger_path):
    request = httpx.Request("POST", "https://api.meta.ai/v1/responses")
    fake.responses.replies.extend([APITimeoutError(request=request), APITimeoutError(request=request)])
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[], run_id="run1")
    assert len(requests_of(fake)) == 2  # a timeout gets the one retry of section 14 item 10, never the 429 budget
    assert result.answer is None and result.truncated is False
    assert result.record.status is SparkStatus.INCOMPLETE
    assert result.record.incomplete_reason.startswith("error: APITimeoutError")
    assert result.record.request_id.startswith("local-") and result.record.dollars == 0.0
    rows = ledger.rows(ledger_path)
    assert [row["status"] for row in rows] == ["failed", "failed"] and all(row["usd"] == 0 for row in rows)


def test_response_failed_status_is_incomplete(fake):
    data = completed("", rid="f1")
    data["status"] = "failed"
    data["error"] = {"code": "server_error", "message": "boom"}
    fake.responses.replies.extend([data, completed(GOOD_MARKS, rid="f2")])
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[])
    assert result.attempts[0].status is SparkStatus.INCOMPLETE
    assert result.attempts[0].incomplete_reason == "error: server_error: boom"
    assert result.answer is not None


def status_error(status: int, retry_after: str | None = None, message: str = ""):
    request = httpx.Request("POST", "https://api.meta.ai/v1/responses")
    headers = {"retry-after": retry_after} if retry_after else {}
    cls = {400: BadRequestError, 401: AuthenticationError, 429: RateLimitError, 503: InternalServerError}[status]
    return cls(message or f"Error code: {status}", response=httpx.Response(status, headers=headers, request=request), body=None)


def test_429_and_5xx_back_off_inside_one_attempt(fake, ledger_path, monkeypatch):
    waits = []
    monkeypatch.setattr(spark_client, "_sleep", waits.append)
    fake.responses.replies.extend([status_error(429, "3"), status_error(503), completed(GOOD_MARKS, rid="ok")])
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[])
    assert result.answer is not None and result.record.retry == 0 and len(requests_of(fake)) == 3
    assert waits[0] == 3.0 and 0.75 <= waits[1] <= 1.0  # Retry-After honoured, then 500 ms doubled with jitter
    assert [row["status"] for row in ledger.rows(ledger_path)] == ["completed"]  # the backoff stays inside one attempt


def test_rate_limit_gives_up_after_six_tries_per_attempt(fake, ledger_path, monkeypatch):
    monkeypatch.setattr(spark_client, "_sleep", lambda seconds: None)
    fake.responses.replies.extend([status_error(429) for _ in range(2 * spark_client.RATE_LIMIT_TRIES)])
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[])
    assert len(requests_of(fake)) == 2 * spark_client.RATE_LIMIT_TRIES  # 6 tries, then the retry of _ask_attempts
    assert result.answer is None and result.record.incomplete_reason.startswith("error: RateLimitError")
    assert [row["status"] for row in ledger.rows(ledger_path)] == ["failed", "failed"]


def test_client_errors_and_a_long_retry_after_are_not_retried(fake, monkeypatch):
    monkeypatch.setattr(spark_client, "_sleep", lambda seconds: pytest.fail(f"unexpected wait of {seconds} s"))
    fake.responses.replies.extend([status_error(400), status_error(429, "120")])
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[])
    assert len(requests_of(fake)) == 2  # one try in each attempt
    assert result.attempts[0].incomplete_reason.startswith("error: BadRequestError")
    assert result.attempts[1].incomplete_reason.startswith("error: RateLimitError")


def test_real_client_leaves_retries_to_this_module(monkeypatch):
    monkeypatch.setattr(spark_client, "_CLIENT", None)
    monkeypatch.setattr(spark_client, "api_key", lambda: "LLM_0_offline_placeholder")
    client = spark_client._client()
    assert client.max_retries == 0 and client.timeout == spark_client.TIMEOUT_S == 240.0


def test_error_text_never_stores_a_key(fake, monkeypatch):
    """A server that echoes the bearer key in an error body must not put it into spark.json."""
    monkeypatch.setenv("MODEL_API_KEY", "odd-format-key-0123456789")
    denied = status_error(401, message="Incorrect API key provided: LLM_123_SECRETSECRET (Bearer odd-format-key-0123456789)")
    failed = completed("", rid="f1")
    failed["status"] = "failed"
    failed["error"] = {"code": "invalid_api_key", "message": "key odd-format-key-0123456789 was refused"}
    fake.responses.replies.extend([denied, failed])
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[])
    first, second = result.attempts
    assert first.incomplete_reason.startswith("error: AuthenticationError: Incorrect API key provided: [redacted]")
    assert second.incomplete_reason == "error: invalid_api_key: key [redacted] was refused"
    for record in result.attempts:
        text = canonical_json(record)
        assert "SECRET" not in text and "odd-format-key" not in text


def test_sdk_response_object_works_through_output_text(fake):
    from openai.types.responses import Response

    data = completed(GOOD_MARKS, rid="sdk1")
    data.update({"created_at": 1.0, "parallel_tool_calls": False, "tool_choice": "auto", "tools": []})
    fake.responses.replies.append(Response.model_validate(data))
    result = spark_client.find_marks(crop(), PanelId.HOOD, landmarks=[])
    assert result.answer is not None and result.record.request_id == "sdk1"
    assert result.record.usage.reasoning == 450


# Grid -> crop pixels ---------------------------------------------------------------------------------------------


def test_point_and_box_to_crop_pixels_undo_the_resize(fake):
    fake.responses.replies.append(completed(GOOD_MARKS))
    result = spark_client.find_marks(crop(2560, 1600), PanelId.HOOD, landmarks=[])
    assert spark_client.point_to_crop_pixels(GridPoint(x=500, y=500), result) == (1280, 800)
    assert spark_client.point_to_crop_pixels(GridPoint(x=1000, y=1000), result) == (2558, 1598)
    assert spark_client.box_to_crop_pixels(GridBox(x_min=0, y_min=0, x_max=1000, y_max=1000), result) == (0, 0, 2560, 1600)
    x1, y1, x2, y2 = spark_client.box_to_crop_pixels(GridBox(x_min=500, y_min=500, x_max=500, y_max=500), result)
    assert (x1, y1) == (1280, 800) and x2 > x1 and y2 > y1


# count mode, second look, seed, claim naming, drawn box -------------------------------------------------------------


def test_count_marks_points_are_clamped(fake):
    fake.responses.replies.append(completed({"points": [{"x": 10, "y": 20}, {"x": 1001, "y": 5}, {"x": 2000, "y": 5}], "count": 3}, rid="cnt"))
    result = spark_client.count_marks(crop(), PanelId.LEFT_REAR_DOOR)
    assert result.points == [GridPoint(x=10, y=20), GridPoint(x=1000, y=5)]
    assert result.count == 3 and result.truncated is False
    assert result.record.purpose is SparkPurpose.COUNT and result.record.schema_name == "count_v1"
    text = requests_of(fake)[0]["input"][1]["content"][0]["text"]
    assert "before you count" in text and "left rear door" in text


def test_count_marks_truncated(fake):
    fake.responses.replies.extend([incomplete(rid="t1"), incomplete(rid="t2")])
    result = spark_client.count_marks(crop(), PanelId.HOOD)
    assert result.points == [] and result.count == 0 and result.truncated is True


@pytest.mark.parametrize("verdict", ["same", "different", "unsure"])
def test_second_look_passes_the_verdict_through(fake, verdict):
    fake.responses.replies.append(completed({"verdict": verdict, "reason": "looked at both"}, rid="look"))
    result = spark_client.second_look(crop(), crop(), panel_id=PanelId.RIGHT_FRONT_DOOR, input_refs=[InputRef(path="a.jpg", sha256=SHA)])
    assert result.verdict is LookVerdict(verdict) and result.reason == "looked at both"
    assert result.downgrade is (verdict != "same")
    (request,) = requests_of(fake)
    parts = request["input"][1]["content"]
    assert [p["type"] for p in parts] == ["input_text", "input_image", "input_image"]
    assert "right front door" in parts[0]["text"] and request["text"]["format"]["name"] == "look_v1"
    assert result.record.purpose is SparkPurpose.SECOND_LOOK and result.record.input_refs[0].path == "a.jpg"


def test_second_look_bad_json_is_unsure(fake):
    fake.responses.replies.extend([completed("garbage", rid="g1"), completed('{"verdict": "keep"}', rid="g2")])
    result = spark_client.second_look(crop(), crop(), panel_id=PanelId.HOOD)
    assert result.verdict is LookVerdict.UNSURE and result.downgrade
    assert result.reason.startswith("second look unusable")
    assert len(result.attempts) == 2


def test_second_look_truncated_is_unsure(fake):
    fake.responses.replies.extend([incomplete(rid="l1"), incomplete(rid="l2")])
    result = spark_client.second_look(crop(), crop(), panel_id=PanelId.HOOD)
    assert result.verdict is LookVerdict.UNSURE and "max_output_tokens" in result.reason


def test_seed_mark_found_and_not_found(fake):
    fake.responses.replies.append(completed({"found": True, "bbox": {"x_min": 100, "y_min": 100, "x_max": 1001, "y_max": 200}, "type": "chip"}, rid="s1"))
    result = spark_client.seed_mark(crop(), PanelId.HOOD, "small chip on the hood", run_id="r")
    assert result.found and result.bbox == GridBox(x_min=100, y_min=100, x_max=1000, y_max=200) and result.type is MarkKind.CHIP
    assert "small chip on the hood" in requests_of(fake)[0]["input"][1]["content"][0]["text"]
    fake.responses.replies.append(completed({"found": False, "bbox": None, "type": None}, rid="s2"))
    result = spark_client.seed_mark(crop(), PanelId.HOOD, "nothing here")
    assert not result.found and result.bbox is None and result.type is None
    assert result.record.purpose is SparkPurpose.SEED


def test_claim_name_maps_to_the_panel_enum(fake):
    fake.responses.replies.append(
        completed(
            {
                "panel": "car door",
                "side": "left",
                "position": "front",
                "front_direction": "right",
                "panel_bbox": {"x_min": 100, "y_min": 200, "x_max": 900, "y_max": 800},
                "context": "a dark sedan in a parking lot",
                "landmarks": [{"name": "door handle", "point": {"x": 600, "y": 500}}, {"name": "mirror", "point": {"x": 1200, "y": 5}}],
            },
            rid="cn",
        )
    )
    result = spark_client.claim_name(crop(), input_ref=InputRef(path="claim/claim_1.jpg", sha256=SHA))
    assert result.panel is PanelId.LEFT_FRONT_DOOR and result.noun is Noun.CAR_DOOR
    assert result.side is Side.LEFT and result.position is Position.FRONT and result.front_direction == "right"
    assert result.bbox == GridBox(x_min=100, y_min=200, x_max=900, y_max=800)
    assert [lm.name for lm in result.landmarks] == ["door handle"]  # the off-grid landmark is dropped
    assert result.context == "a dark sedan in a parking lot"
    assert result.record.purpose is SparkPurpose.CLAIM_NAME
    schema = requests_of(fake)[0]["text"]["format"]["schema"]
    assert "none" in schema["properties"]["panel"]["enum"] and "car door" in schema["properties"]["panel"]["enum"]


@pytest.mark.parametrize(
    "noun, side, position, panel",
    [
        (Noun.CAR_DOOR, Side.RIGHT, Position.REAR, PanelId.RIGHT_REAR_DOOR),
        (Noun.FENDER, Side.LEFT, Position.REAR, PanelId.LEFT_REAR_FENDER),
        (Noun.WHEEL, Side.RIGHT, Position.FRONT, PanelId.RIGHT_FRONT_WHEEL),
        (Noun.CAR_DOOR, Side.NONE, Position.FRONT, None),
        (Noun.BUMPER, Side.NONE, Position.FRONT, PanelId.FRONT_BUMPER),
        (Noun.BUMPER, Side.LEFT, Position.NONE, None),
        (Noun.HOOD, Side.NONE, Position.NONE, PanelId.HOOD),
        (Noun.HEADLIGHT, Side.LEFT, Position.FRONT, PanelId.HEADLIGHTS),
        (Noun.SIDE_MIRROR, Side.LEFT, Position.NONE, PanelId.MIRRORS),
        (Noun.CAR, Side.NONE, Position.NONE, None),
        (Noun.CAR_WINDOW, Side.LEFT, Position.FRONT, None),
        (None, Side.NONE, Position.NONE, None),
    ],
)
def test_panel_from_naming(noun, side, position, panel):
    assert spark_client.panel_from_naming(noun, side, position) is panel


def test_claim_name_keeps_four_landmarks_after_dropping_off_grid_ones(fake):
    names = ["off", "handle", "mirror", "badge", "wheel arch", "fuel door"]
    points = [{"name": n, "point": {"x": 1500 if n == "off" else 100 * (i + 1), "y": 500}} for i, n in enumerate(names)]
    reply = {"panel": "car door", "side": "left", "position": "rear", "front_direction": "left", "panel_bbox": None, "context": "", "landmarks": points}
    fake.responses.replies.append(completed(reply, rid="cn4"))
    result = spark_client.claim_name(crop())
    assert result.panel is PanelId.LEFT_REAR_DOOR
    assert [lm.name for lm in result.landmarks] == ["handle", "mirror", "badge", "wheel arch"]


def test_claim_name_none_and_invalid(fake):
    fake.responses.replies.append(completed({"panel": "none", "side": "none", "position": "none", "front_direction": "unknown", "panel_bbox": None, "context": "a desk", "landmarks": []}, rid="n"))
    result = spark_client.claim_name(crop())
    assert result.panel is None and result.noun is None and result.bbox is None and result.front_direction is None
    fake.responses.replies.extend([completed("x", rid="b1"), completed("y", rid="b2")])
    result = spark_client.claim_name(crop())
    assert result.panel is None and result.context == "" and result.landmarks == [] and len(result.attempts) == 2


def test_claim_box(fake):
    fake.responses.replies.append(completed({"found": True, "bbox": {"x_min": 300, "y_min": 300, "x_max": 500, "y_max": 450}}, rid="cb"))
    result = spark_client.claim_box(crop())
    assert result.bbox == GridBox(x_min=300, y_min=300, x_max=500, y_max=450) and result.record.purpose is SparkPurpose.CLAIM_BOX
    fake.responses.replies.append(completed({"found": False, "bbox": None}, rid="cb2"))
    assert spark_client.claim_box(crop()).bbox is None


# Streamed probes ---------------------------------------------------------------------------------------------------


def stream_events(final: dict, text: str) -> list[dict]:
    half = len(text) // 2
    return [
        {"type": "response.created", "response": {"id": final["id"], "status": "in_progress"}},
        {"type": "response.output_text.delta", "delta": text[:half]},
        {"type": "response.output_text.delta", "delta": text[half:]},
        {"type": "response.completed", "response": final},
    ]


def test_timeline_streams_with_the_file_id(fake, ledger_path):
    payload = {"panels": [{"panel": "car door", "side": "left", "start_s": 1.0, "end_s": 4.5}]}
    fake.responses.replies.append(stream_events(completed(payload, rid="tl"), json.dumps(payload)))
    record = spark_client.timeline("file-123", run_id="probe")
    (request,) = requests_of(fake)
    assert request["stream"] is True and request["store"] is False
    assert request["input"][1]["content"][1] == {"type": "input_file", "file_id": "file-123"}
    assert record.valid and record.parsed == payload and record.purpose is SparkPurpose.TIMELINE
    assert record.request_id == "tl" and ledger.rows(ledger_path)[0]["note"] == "timeline"


def test_stream_without_a_final_event_is_incomplete(fake):
    events = [{"type": "response.output_text.delta", "delta": '{"panels": ['}]
    fake.responses.replies.extend([events, list(events)])
    record = spark_client.timeline("file-1")
    assert record.status is SparkStatus.INCOMPLETE and record.valid is False
    assert record.incomplete_reason == "stream ended before response.completed"
    assert record.raw_text == '{"panels": ['


def test_stream_error_event_is_recorded_as_failed(fake, ledger_path):
    events = [{"type": "response.created", "response": {"id": "e1", "status": "in_progress"}}, {"type": "error", "code": "server_error", "message": "boom"}]
    fake.responses.replies.extend([events, list(events)])
    record = spark_client.timeline("file-1")
    assert record.status is SparkStatus.INCOMPLETE and record.valid is False
    assert record.incomplete_reason == "error: server_error: boom"
    assert [row["status"] for row in ledger.rows(ledger_path)] == ["failed", "failed"]


def test_stream_final_without_output_uses_the_deltas(fake):
    payload = {"panels": []}
    final = completed(payload, rid="d")
    final["output"] = []
    fake.responses.replies.append(stream_events(final, json.dumps(payload)))
    record = spark_client.timeline("file-1")
    assert record.valid and record.parsed == payload


def test_note_prefill_sends_the_pdf_inline(fake, tmp_path):
    pdf = tmp_path / "agreement.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    payload = {"panel": "left front door", "pickup_date": "2026-10-27", "return_date": "2026-10-29", "amount": "$440", "agreement_number": "A1"}
    fake.responses.replies.append(stream_events(completed(payload, rid="np"), json.dumps(payload)))
    record = spark_client.note_prefill("They charged $440 for the left front door", pdf, run_id="probe")
    (request,) = requests_of(fake)
    file_part = request["input"][1]["content"][1]
    assert file_part["type"] == "input_file" and file_part["filename"] == "agreement.pdf"
    assert file_part["file_data"].startswith("data:application/pdf;base64,")
    assert "They charged $440" in request["input"][1]["content"][0]["text"]
    assert record.valid and record.parsed == payload and record.purpose is SparkPurpose.NOTE_PREFILL


# ask directly --------------------------------------------------------------------------------------------------------


def test_ask_schema_override_and_input_refs(fake):
    fake.responses.replies.append(completed({"verdict": "same", "reason": "r"}, rid="a1"))
    refs = [InputRef(path="x.jpg", sha256=SHA)]
    record = spark_client.ask(
        [{"type": "input_text", "text": "hi"}],
        purpose=SparkPurpose.SECOND_LOOK,
        schema_name="look_v1",
        effort="medium",
        max_output_tokens=1234,
        run_id=None,
        input_refs=refs,
    )
    assert record.valid and record.reasoning_effort is ReasoningEffort.MEDIUM and record.max_output_tokens == 1234
    assert record.input_refs == refs
    assert requests_of(fake)[0]["reasoning"] == {"effort": "medium"}
    with pytest.raises(ValueError):
        spark_client.ask([], purpose=SparkPurpose.MARKS, schema_name="marks_v1", effort="turbo", max_output_tokens=1, run_id=None, input_refs=[])


# The real response of 10/27 as a fixture --------------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "spark"


def test_real_marks_v1_response_fixture(fake, ledger_path):
    """The muse-spark-1.3 answer of 10/27 (tests/fixtures/spark/marks_v1_live_2026-10-27.json, the SDK's model_dump()
    with the encrypted reasoning blob nulled) on the generated 1280 x 800 panel: the drawn scratch and the drawn chip."""
    data = json.loads((FIXTURES / "marks_v1_live_2026-10-27.json").read_text())
    fake.responses.replies.append(data)
    result = spark_client.find_marks(panel_with_scratch(), PanelId.LEFT_FRONT_DOOR, landmarks=["handle", "lock", "mirror_base", "wheel_arch"], run_id="fixture")
    record = result.record
    assert record.valid and record.status is SparkStatus.COMPLETED and not record.truncated
    assert record.request_id == "resp_6ae0e9901bb3190e731d4a51" and data["store"] is False
    assert (record.usage.input, record.usage.output, record.usage.reasoning, record.usage.cached) == (1938, 829, 634, 0)
    assert record.dollars == pytest.approx(1938 * PRICES["spark_input_token"] + 829 * PRICES["spark_output_token"], abs=1e-6)
    assert data["usage"]["input_tokens_details"]["cache_write_tokens"] is None  # the API sends null here, not 0
    answer = result.answer
    assert answer is not None and [m.type for m in answer.marks] == [MarkKind.SCRATCH, MarkKind.CHIP]
    assert not answer.conditions.any_flag() and answer.landmarks == [] and not answer.neighbour_car_visible
    scratch, chip = answer.marks
    sx, sy = spark_client.point_to_crop_pixels(scratch.point, result)
    assert abs(sx - 590) < 60 and abs(sy - 326) < 60  # the line runs (420, 300) to (760, 352)
    cx, cy = spark_client.point_to_crop_pixels(chip.point, result)
    assert abs(cx - 908) < 80 and abs(cy - 567) < 80  # the dot sits at (900..916, 560..574)
    x1, y1, x2, y2 = spark_client.box_to_crop_pixels(scratch.bbox, result)
    assert x1 < 590 < x2 and y1 < 326 < y2
    row = ledger.rows(ledger_path)[0]
    assert row["reasoning_tokens"] == 634 and row["status"] == "completed" and row["usd"] == record.dollars


# Live smoke ------------------------------------------------------------------------------------------------------------


def panel_with_scratch(width: int = 1280, height: int = 800) -> np.ndarray:
    """A light gray painted panel with a soft gradient, one dark diagonal scratch and a small chip."""
    img = Image.new("RGB", (width, height))
    px = np.zeros((height, width, 3), np.uint8)
    ramp = np.linspace(165, 205, width, dtype=np.float32)
    px[:, :, 0] = ramp[None, :]
    px[:, :, 1] = ramp[None, :] + 2
    px[:, :, 2] = ramp[None, :] + 6
    rng = np.random.default_rng(7)
    px = np.clip(px.astype(np.int16) + rng.integers(-3, 4, px.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(px)
    draw = ImageDraw.Draw(img)
    draw.line([(420, 300), (760, 352)], fill=(45, 40, 38), width=4)
    draw.line([(424, 305), (757, 356)], fill=(90, 85, 80), width=1)
    draw.ellipse([(900, 560), (916, 574)], fill=(35, 32, 30))
    return np.asarray(img)


@pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="live Spark call; set LIVE=1 (about $0.01)")
def test_live_find_marks_on_a_generated_panel():
    result = spark_client.find_marks(
        panel_with_scratch(),
        PanelId.LEFT_FRONT_DOOR,
        landmarks=["handle", "lock", "mirror_base", "wheel_arch"],
        run_id="smoke_m2",
    )
    record = result.record
    print("live spark:", record.status.value, "valid", record.valid, "truncated", record.truncated, "usage", record.usage.model_dump(), "usd", record.dollars, "ms", record.ms)
    assert record.status is SparkStatus.COMPLETED, record.incomplete_reason
    assert record.valid, record.raw_text[:300]
    assert result.answer is not None
    parsed = json.loads(record.raw_text)
    assert set(parsed) == {"marks", "conditions", "landmarks", "neighbour_car_visible"}
    for mark in result.answer.marks:
        assert mark.type in MarkKind and mark.size_class in SizeClass and mark.confidence in Confidence
        assert 0 <= mark.point.x <= 1000 and 0 <= mark.bbox.x_max <= 1000
    print("live spark marks:", [(m.type.value, m.point.x, m.point.y, m.size_class.value, m.confidence.value) for m in result.answer.marks])
    assert record.usage.input > 0 and record.dollars > 0


@pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="live Spark call; set LIVE=1 (about $0.01)")
def test_live_seed_v1_anyof_null():
    """seed_v1 puts anyOf [object, null] below the root, as claim_name_v1 and claim_box_v1 do; no D1 row sends it."""
    result = spark_client.seed_mark(panel_with_scratch(), PanelId.LEFT_FRONT_DOOR, "a long scratch across the door", run_id="smoke_m2_seed")
    record = result.record
    print("live seed:", record.status.value, "valid", record.valid, "found", result.found, "bbox", result.bbox, "usd", record.dollars)
    assert record.status is SparkStatus.COMPLETED, record.incomplete_reason
    assert record.valid, record.raw_text[:300]
