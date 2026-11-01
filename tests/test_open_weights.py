"""Offline tests for walkaround.open_weights (Muse Spark Open on Baseten): a fake OpenAI-compatible client, the ledger
and the runs directory on temp paths. The key slots (the hackathon key with -1026 first, then the account key with the
plain id; a refusal rests a slot and moves on; the listing is never consulted), the request shape,
the response format step-down (only on a refusal that names the format) and its expiry, schema checks on answers
Baseten does not constrain (top-level objects only; two that disagree are unusable), the cut and retry rules, the
bounded backoff, no-answer responses, the ledger row and price, the provenance lines in the binding shape (no image
bytes, no key) and the guard that keeps a real client out of an offline test.

The one live test runs only with LIVE=1: one look on two generated crops through Baseten (credits, no Meta spend)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import APIStatusError, APITimeoutError, AuthenticationError, BadRequestError, InternalServerError, NotFoundError, RateLimitError

from walkaround import config, ledger, open_weights, paths, spark_client
from walkaround.contracts import InputRef, LedgerRow, LookVerdict, PanelId, SparkEngine, SparkPurpose, SparkStatus

HACKATHON, ACCOUNT = open_weights.HACKATHON_MODEL, open_weights.ACCOUNT_MODEL
SUFFIX, NO_SUFFIX = HACKATHON, ACCOUNT  # by spelling: meta-models/muse-spark-open-1026 is the hackathon key's id
SAME = {"verdict": "same", "reason": "Both outlines sit on one chip by the lock."}
DIFFERENT = {"verdict": "different", "reason": "The return outline is on a leaf."}
FAKE_KEY = "bt-test-key-0123456789abcdef"
FORMAT_REFUSAL = "Error code: 400 - response_format json_schema is not supported for this model"
# CONTRACTS.md, Phase 2 file shapes: every open_weights.ndjson line carries these
BINDING_LINE = {"ts", "run_id", "engine", "model", "request_id", "latency_ms", "pickup_mark", "return_mark", "panel_id", "verdict", "reason", "valid", "error"}
LOG_KEYS = BINDING_LINE | {"purpose", "model_requested", "baseten_request_id", "http_status", "status", "key_slot"}


def chat(payload, *, rid: str = "chatcmpl-1", model: str = NO_SUFFIX, finish: str | None = "stop", prompt: int = 1800, completion: int = 420, reasoning: int = 300, cached: int = 0) -> dict:
    """A Chat Completions object as model_dump() gives it."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "id": rid,
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text, "reasoning_content": "thinking"}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


def error_reply(rid: str = "chatcmpl-err") -> dict:
    """A 200 that carries an error object and no choices: a response with no answer in it."""
    return {"id": rid, "object": "chat.completion", "model": NO_SUFFIX, "choices": [], "error": {"code": "server_error", "message": "upstream reset"}}


def status_error(status: int, retry_after: str | None = None, message: str = "", request_id: str | None = None) -> APIStatusError:
    request = httpx.Request("POST", "https://inference.baseten.co/v1/chat/completions")
    headers = {"retry-after": retry_after} if retry_after else {}
    if request_id:
        headers["x-baseten-request-id"] = request_id
    cls = {400: BadRequestError, 401: AuthenticationError, 404: NotFoundError, 429: RateLimitError, 503: InternalServerError}.get(status, APIStatusError)
    return cls(message or f"Error code: {status}", response=httpx.Response(status, headers=headers, request=request), body=None)


def timeout() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "https://inference.baseten.co/v1/chat/completions"))


class FakeChat:
    def __init__(self) -> None:
        self.replies: list = []
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        assert self.replies, "the fake ran out of canned replies"
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class RawReply:
    def __init__(self, data: dict, headers: dict[str, str]) -> None:
        self.data = data
        self.headers = httpx.Headers(headers)

    def parse(self) -> dict:
        return self.data


class FakeRawChat(FakeChat):
    """A chat fake that also offers .with_raw_response, as the OpenAI SDK does, with Baseten's request id header."""

    def __init__(self) -> None:
        super().__init__()
        self.with_raw_response = SimpleNamespace(create=self.raw)

    def raw(self, **kwargs) -> RawReply:
        return RawReply(self.create(**kwargs), {"x-baseten-request-id": f"bt-{len(self.requests)}"})


class FakeModels:
    def __init__(self, listed) -> None:
        self.listed = listed
        self.calls = 0

    def list(self):
        self.calls += 1
        if isinstance(self.listed, BaseException):
            raise self.listed
        return {"object": "list", "data": [{"id": i, "object": "model"} for i in self.listed]}


class FakeBaseten:
    def __init__(self, listed=(NO_SUFFIX,)) -> None:
        self.models = FakeModels(listed)
        self.chat = SimpleNamespace(completions=FakeChat())

    @property
    def requests(self) -> list[dict]:
        return self.chat.completions.requests

    def reply(self, *replies) -> None:
        self.chat.completions.replies.extend(replies)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    return root


@pytest.fixture
def waits(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(open_weights, "_sleep", slept.append)
    return slept


@pytest.fixture
def fake(runs_dir, waits, monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", FAKE_KEY)
    client = FakeBaseten()
    open_weights.reset()
    open_weights.set_client(client)
    yield client
    open_weights.set_client(None)
    open_weights.reset()


@pytest.fixture
def clock(monkeypatch):
    """A clock the test moves by hand: now[0] is what open_weights reads."""
    now = [1000.0]
    monkeypatch.setattr(open_weights, "_clock", lambda: now[0])
    return now


def crop(shade: int = 170) -> np.ndarray:
    rgb = np.full((300, 400, 3), shade, np.uint8)
    rgb[140:160, 190:210] = 60  # a dark chip
    return rgb


def look(**kw) -> spark_client.LookResult:
    kw.setdefault("rpm", None)
    return open_weights.second_look(crop(), crop(160), panel_id=PanelId.TRUNK_LID, **kw)


def provenance(run_id: str) -> list[dict]:
    path = paths.open_weights_log(run_id)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


# Key slots: the hackathon key and its -1026 id first, then the account key, a refusal rests a slot -----------------


class ByModel(FakeChat):
    """Baseten as the hackathon key saw it on 10/31 (LEDGER row 22): -1026 answers, the plain id is forbidden."""

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs["model"] != HACKATHON:
            raise status_error(403, message="Error code: 403 - this key may not call that model")
        return chat(SAME, model=HACKATHON, rid=f"chatcmpl-{len(self.requests)}")


def test_the_hackathon_key_goes_first_with_its_own_id(fake):
    fake.reply(chat(SAME, model=HACKATHON, rid="chatcmpl-h"))
    result = look(run_id="run1", pair=("p0", "r0"))
    assert result.verdict is LookVerdict.SAME and result.record.model == HACKATHON
    assert [r["model"] for r in fake.requests] == [HACKATHON]
    (row,) = ledger.rows()
    assert (row["key_slot"], row["model_requested"], row["model"], row["status"]) == ("hackathon", HACKATHON, HACKATHON, "completed")
    (line,) = provenance("run1")
    assert (line["key_slot"], line["model"], line["model_requested"]) == ("hackathon", HACKATHON, HACKATHON)
    assert open_weights.key_slot_of(result.record) == "hackathon"
    assert fake.models.calls == 0  # GET /v1/models is never asked: the ids are fixed per key


def test_the_listing_bug_is_gone_the_hackathon_key_never_asks_the_plain_id(fake):
    """The listing shows both ids to the hackathon key, and the old order picked the plain id first and got 403."""
    fake.models.listed = (ACCOUNT, HACKATHON)
    fake.chat = SimpleNamespace(completions=ByModel())
    open_weights.set_client(fake, slots=("hackathon",))
    result = look(run_id="run1")
    assert result.verdict is LookVerdict.SAME and [r["model"] for r in fake.requests] == [HACKATHON]
    assert [line["status"] for line in provenance("run1")] == ["completed"] and fake.models.calls == 0


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_a_refused_hackathon_key_rests_and_the_account_key_answers(fake, clock, status):
    fake.reply(status_error(status), chat(SAME, model=ACCOUNT, rid="chatcmpl-a"))
    result = look(run_id="run1")
    assert result.verdict is LookVerdict.SAME and result.record.model == ACCOUNT
    assert [r["model"] for r in fake.requests] == [HACKATHON, ACCOUNT]
    rows = ledger.rows()
    assert [(r["key_slot"], r["status"], r["model_requested"], r.get("http_status")) for r in rows] == [("hackathon", "failed", HACKATHON, status), ("account", "completed", ACCOUNT, None)]
    assert [(line["key_slot"], line["status"], line["model"]) for line in provenance("run1")] == [("hackathon", "failed", HACKATHON), ("account", "completed", ACCOUNT)]
    assert open_weights.key_slot_of(result.record) == "account"
    fake.reply(chat(DIFFERENT, model=ACCOUNT))
    assert look().verdict is LookVerdict.DIFFERENT and fake.requests[-1]["model"] == ACCOUNT  # the hackathon slot rests
    clock[0] += open_weights.FAILURE_TTL_S  # ten minutes later it leads again
    fake.reply(chat(SAME, model=HACKATHON))
    look()
    assert fake.requests[-1]["model"] == HACKATHON


def test_every_slot_refused_raises_and_later_calls_ask_nothing_until_the_ttl(fake, clock):
    fake.reply(status_error(404), status_error(403))
    with pytest.raises(open_weights.OpenWeightsError, match=r"no Baseten key slot can be asked now, each rests after a failure \(hackathon key: 404; account key: 403\)"):
        look()
    clock[0] += open_weights.FAILURE_TTL_S - 1
    with pytest.raises(open_weights.OpenWeightsError, match=r"hackathon key: 404; account key: 403"):
        look()
    assert len(fake.requests) == 2
    clock[0] += 1  # both refusals are ten minutes old: the slots are tried again, in order
    fake.reply(chat(SAME, model=HACKATHON))
    assert look().verdict is LookVerdict.SAME
    assert [r["model"] for r in fake.requests] == [HACKATHON, ACCOUNT, HACKATHON]


def test_the_error_names_every_resting_slot(fake, clock):
    fake.reply(status_error(403), chat(SAME, model=ACCOUNT))
    look()  # the hackathon key rests after its 403 and the account key answers
    fake.reply(status_error(401))
    with pytest.raises(open_weights.OpenWeightsError, match=r"\(hackathon key: 403; account key: 401\)"):
        look()


def test_only_the_slots_with_a_key_are_asked(runs_dir, monkeypatch):
    monkeypatch.setattr(open_weights, "load_env", lambda: None)  # no .env and no ~/meta2026/.keys, like a fresh clone
    open_weights.reset()
    open_weights.set_client(None)
    cases = [
        ({"BASETEN_HACKATHON_API_KEY": "b10_hk", "BASETEN_API_KEY": "acct"}, ["hackathon", "account"]),
        ({"BASETEN_API_KEY": "acct"}, ["account"]),
        ({"BASETEN_HACKATHON_API_KEY": "b10_hk"}, ["hackathon"]),
        ({}, []),
    ]
    for env, names in cases:
        for name in ("BASETEN_HACKATHON_API_KEY", "BASETEN_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        assert [s.name for s in open_weights.key_slots()] == names
    assert [(s.name, s.env, s.model) for s in open_weights.KEY_SLOTS] == [("hackathon", "BASETEN_HACKATHON_API_KEY", HACKATHON), ("account", "BASETEN_API_KEY", ACCOUNT)]


def test_a_client_for_one_slot_answers_only_for_it(fake):
    open_weights.set_client(fake, slots=("account",))
    fake.reply(chat(SAME, model=ACCOUNT))
    look(run_id="run1")
    assert [r["model"] for r in fake.requests] == [ACCOUNT] and provenance("run1")[0]["key_slot"] == "account"
    with pytest.raises(ValueError, match="no key slot named spare"):
        open_weights.set_client(fake, slots=("spare",))


def test_key_slot_of_knows_only_spark_open_records_seen_here(fake):
    fake.reply(chat(SAME, model=HACKATHON, rid="chatcmpl-seen"))
    record = look().record
    assert open_weights.key_slot_of(record) == "hackathon"
    assert open_weights.key_slot_of(record.model_copy(update={"engine": SparkEngine.META_MODEL_API})) is None
    open_weights.reset()
    assert open_weights.key_slot_of(record) is None


# The request ------------------------------------------------------------------------------------------------------


def test_request_shape_is_the_hosted_look_question_with_images_as_data_urls(fake):
    fake.reply(chat(SAME))
    look(effort="low")
    body = fake.requests[0]
    assert set(body) == {"model", "messages", "max_tokens", "reasoning_effort", "response_format"}
    assert body["max_tokens"] == 4000 and body["reasoning_effort"] == "low"
    system, user = body["messages"]
    assert system == {"role": "system", "content": spark_client.SYSTEM_PROMPT}
    text, image_a, image_b = user["content"]
    schema = spark_client.SCHEMAS["look_v1"]
    assert text == {"type": "text", "text": spark_client.look_prompt(PanelId.TRUNK_LID) + " " + open_weights.json_instruction(schema)}
    for part, rgb in ((image_a, crop()), (image_b, crop(160))):
        assert part["type"] == "image_url" and part["image_url"]["url"] == spark_client.prepare_image(rgb)[0]
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "look_v1", "schema": schema, "strict": True}}


def test_the_caller_sets_the_token_budgets(fake):
    fake.reply(chat("", finish="length"), chat(SAME))
    look(max_tokens=8000, retry_max_tokens=12000)
    assert [r["max_tokens"] for r in fake.requests] == [8000, 12000]


def test_a_format_refusal_steps_the_format_down_and_the_step_expires(fake, clock):
    fake.reply(status_error(400, message=FORMAT_REFUSAL), status_error(422, message="json_object is not supported"), chat(SAME))
    assert look().verdict is LookVerdict.SAME
    assert [r.get("response_format", {}).get("type") for r in fake.requests] == ["json_schema", "json_object", None]
    fake.reply(chat(SAME))
    look()
    assert "response_format" not in fake.requests[-1]  # the step holds
    clock[0] += open_weights.FAILURE_TTL_S
    fake.reply(chat(SAME))
    look()
    assert fake.requests[-1]["response_format"]["type"] == "json_schema"  # ten minutes on, json_schema is tried again


def test_a_format_refusal_with_no_format_left_raises(fake):
    fake.reply(*[status_error(400, message=FORMAT_REFUSAL) for _ in range(3)])
    with pytest.raises(open_weights.OpenWeightsError, match="400"):
        look()
    assert len(fake.requests) == 3


def test_a_400_that_is_not_about_the_format_is_final_and_keeps_the_format(fake):
    fake.reply(status_error(400, message="Error code: 400 - image too large"))
    with pytest.raises(open_weights.OpenWeightsError, match="400"):
        look()
    fake.reply(chat(SAME))
    look()
    assert [r["response_format"]["type"] for r in fake.requests] == ["json_schema", "json_schema"]


# Answers: Baseten does not enforce the schema ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(SAME),
        "```json\n" + json.dumps(SAME) + "\n```",
        "Looking at both outlines, " + json.dumps(SAME) + " is my answer.",
        json.dumps({"verdict": "maybe", "reason": "draft"}) + "\n" + json.dumps(SAME),
        json.dumps(SAME) + "\n" + json.dumps({"verdict": "same", "reason": "Said twice."}),
        '{"verdict": "same", "reason": "The {left} spot and the \\"same\\" chip}."}',
    ],
)
def test_the_answer_object_is_found_and_checked(fake, text):
    fake.reply(chat(text))
    result = look()
    assert result.verdict is LookVerdict.SAME and result.record.valid and result.record.raw_text == text
    assert len(fake.requests) == 1


NESTED = {"verdict": "different", "reason": "The return spot is a leaf.", "first_image": {"verdict": "same", "reason": "chip"}}


@pytest.mark.parametrize(
    "text",
    [
        "```json\n" + json.dumps(NESTED) + "\n```",  # a nested object that fits is never the answer
        json.dumps(NESTED),
        'Draft: {"verdict": "different", "reason": "a"} Final: {"verdict": "same", "reason": "b"}',  # two answers that disagree
        '{"verdict": "different", "reason": "x", "extra": {"verdict": "same", "reason": "y"},}',  # a broken outer object
    ],
)
def test_a_nested_or_ambiguous_answer_is_unsure_never_same(fake, text):
    fake.reply(chat(text), chat(text))
    result = look()
    assert result.verdict is LookVerdict.UNSURE and [a.valid for a in result.attempts] == [False, False]


def test_a_bad_answer_gets_one_retry_then_is_unsure(fake):
    fake.reply(chat({"verdict": "maybe", "reason": "x"}, rid="a1"), chat("I cannot tell from these crops.", rid="a2"))
    result = look(run_id="run1")
    assert result.verdict is LookVerdict.UNSURE and result.reason.startswith("second look unusable")
    assert [a.valid for a in result.attempts] == [False, False] and [a.retry for a in result.attempts] == [0, 1]
    assert [r["max_tokens"] for r in fake.requests] == [4000, 4000]
    assert [r["status"] for r in ledger.rows()] == ["schema_miss", "schema_miss"]
    lines = provenance("run1")
    assert [(line["status"], line["verdict"], line["valid"]) for line in lines] == [("schema_miss", None, False), ("schema_miss", None, False)]
    assert "not in the enum" in lines[0]["error"] and lines[1]["error"] == "no JSON object in the answer"


def test_a_cut_answer_retries_at_8000_then_is_truncated_and_unsure(fake):
    fake.reply(chat("", finish="length", completion=4000), chat('{"verdict": "sa', finish="length", completion=8000))
    result = look()
    assert [r["max_tokens"] for r in fake.requests] == [4000, 8000]
    assert result.verdict is LookVerdict.UNSURE and result.record.truncated and result.record.status is SparkStatus.INCOMPLETE
    assert result.record.incomplete_reason == "max_output_tokens" and "max_output_tokens" in result.reason
    assert [r["status"] for r in ledger.rows()] == ["incomplete", "truncated"]


def test_a_cut_then_an_answer_stands(fake):
    fake.reply(chat("", finish="length"), chat(DIFFERENT, rid="ok"))
    result = look()
    assert result.verdict is LookVerdict.DIFFERENT and result.record.retry == 1 and not result.record.truncated


def test_an_answer_then_an_error_object_keeps_the_answer_that_came_back(fake):
    fake.reply(chat("no json", rid="first"), error_reply())
    result = look()
    assert result.verdict is LookVerdict.UNSURE and result.record.request_id == "first" and len(result.attempts) == 2


def test_two_responses_with_no_answer_raise_so_the_caller_falls_back(fake):
    fake.reply(error_reply("e1"), error_reply("e2"))
    with pytest.raises(open_weights.OpenWeightsError, match="Baseten gave no answer: error: server_error") as info:
        look()
    assert [a.request_id for a in info.value.attempts] == ["e1", "e2"]
    assert [r["status"] for r in ledger.rows()] == ["incomplete", "incomplete"]


def test_extract_json_reports_the_problems_of_the_last_object():
    schema = spark_client.SCHEMAS["look_v1"]
    assert open_weights.extract_json("", schema) == (None, ["no JSON object in the answer"])
    obj, problems = open_weights.extract_json('{"verdict": "same"}', schema)
    assert obj == {"verdict": "same"} and any("missing" in p for p in problems)
    assert open_weights.extract_json('["same"]', schema)[0] is None
    _, problems = open_weights.extract_json('{"verdict": "different", "reason": "a"} {"verdict": "same", "reason": "b"}', schema)
    assert problems == ["2 answer objects that disagree"]


# Failures the caller falls back on --------------------------------------------------------------------------------


def test_429_and_5xx_back_off_inside_one_call(fake, waits):
    fake.reply(status_error(429, "2"), status_error(503), chat(SAME))
    assert look().verdict is LookVerdict.SAME
    assert waits[0] == 2.0 and 0.75 <= waits[1] <= 1.0  # Retry-After honoured, then 500 ms doubled with jitter
    assert [r["status"] for r in ledger.rows()] == ["failed", "failed", "completed"]


def test_rate_limits_give_up_after_three_tries(fake):
    open_weights.set_client(fake, slots=("hackathon",))  # the only key: nothing else to ask
    fake.reply(status_error(429), status_error(529), status_error(429))
    with pytest.raises(open_weights.OpenWeightsError, match="429"):
        look()
    assert len(fake.requests) == 3


def test_a_long_retry_after_fails_at_once(fake):
    open_weights.set_client(fake, slots=("hackathon",))
    fake.reply(status_error(429, "120"))
    with pytest.raises(open_weights.OpenWeightsError):
        look()
    assert len(fake.requests) == 1


def test_a_timeout_on_the_last_slot_raises_at_once(fake):
    open_weights.set_client(fake, slots=("hackathon",))
    fake.reply(timeout())
    with pytest.raises(open_weights.OpenWeightsError, match="Baseten call failed: APITimeoutError"):
        look(run_id="run1")
    assert len(fake.requests) == 1
    assert [(line["status"], line["key_slot"]) for line in provenance("run1")] == [("failed", "hackathon")]
    fake.reply(chat(SAME, model=HACKATHON))
    assert look().verdict is LookVerdict.SAME  # the last slot does not rest: the next look asks it again
    assert [r["model"] for r in fake.requests] == [HACKATHON, HACKATHON]


# Review 11/1: a per-key rate limit, an outage of one id or a dropped call moves on to the next key slot -------------


def test_a_timeout_on_the_hackathon_key_moves_to_the_account_key_and_rests_it_briefly(fake, clock):
    fake.reply(timeout(), chat(SAME, model=ACCOUNT, rid="chatcmpl-a"))
    result = look(run_id="run1")
    assert result.verdict is LookVerdict.SAME and open_weights.key_slot_of(result.record) == "account"
    assert [r["model"] for r in fake.requests] == [HACKATHON, ACCOUNT]
    assert [(line["key_slot"], line["status"]) for line in provenance("run1")] == [("hackathon", "failed"), ("account", "completed")]
    assert [(r["key_slot"], r["status"]) for r in ledger.rows()] == [("hackathon", "failed"), ("account", "completed")]
    fake.reply(chat(DIFFERENT, model=ACCOUNT))
    look()
    assert fake.requests[-1]["model"] == ACCOUNT  # the hackathon slot rests
    clock[0] += open_weights.TRANSIENT_TTL_S  # a minute on it leads again
    fake.reply(chat(SAME, model=HACKATHON))
    look()
    assert fake.requests[-1]["model"] == HACKATHON


def test_rate_limits_past_three_tries_move_to_the_account_key(fake, waits):
    fake.reply(status_error(429), status_error(429), status_error(503), chat(SAME, model=ACCOUNT))
    assert look().verdict is LookVerdict.SAME
    assert [r["model"] for r in fake.requests] == [HACKATHON, HACKATHON, HACKATHON, ACCOUNT]
    assert len(waits) == 2  # the backoff between the hackathon key's three tries, none before the account key


def test_a_long_retry_after_on_the_hackathon_key_moves_to_the_account_key(fake):
    fake.reply(status_error(429, "120"), chat(SAME, model=ACCOUNT))
    assert look().verdict is LookVerdict.SAME and [r["model"] for r in fake.requests] == [HACKATHON, ACCOUNT]


def test_a_dropped_call_on_both_keys_raises_with_one_request_each(fake):
    fake.reply(timeout(), timeout())
    with pytest.raises(open_weights.OpenWeightsError, match="Baseten call failed: APITimeoutError"):
        look(run_id="run1")
    assert [r["model"] for r in fake.requests] == [HACKATHON, ACCOUNT]
    assert [(line["key_slot"], line["status"]) for line in provenance("run1")] == [("hackathon", "failed"), ("account", "failed")]


def test_a_400_not_about_the_format_never_moves_to_the_next_key(fake):
    fake.reply(status_error(400, message="Error code: 400 - image too large"))
    with pytest.raises(open_weights.OpenWeightsError, match="400"):
        look()
    assert [r["model"] for r in fake.requests] == [HACKATHON]  # the same request would fail on any key


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_on_both_keys_raise_after_one_request_each(fake, status):
    fake.reply(status_error(status), status_error(status))
    with pytest.raises(open_weights.OpenWeightsError, match=f"hackathon key: {status}; account key: {status}"):
        look(run_id="run1")
    assert [r["model"] for r in fake.requests] == [HACKATHON, ACCOUNT]
    assert [(line["status"], line["key_slot"]) for line in provenance("run1")] == [("failed", "hackathon"), ("failed", "account")]


def test_a_failed_retry_raises_with_the_first_answer(fake):
    open_weights.set_client(fake, slots=("hackathon",))
    fake.reply(chat("no json", rid="first"), timeout())
    with pytest.raises(open_weights.OpenWeightsError) as info:
        look()
    assert [a.request_id for a in info.value.attempts] == ["first"]


def test_no_key_raises_before_any_request(runs_dir, monkeypatch):
    monkeypatch.setenv("LIVE", "1")  # past the offline guard, to reach the key check
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("BASETEN_HACKATHON_API_KEY", raising=False)
    monkeypatch.setattr(open_weights, "load_env", lambda: None)
    open_weights.reset()
    open_weights.set_client(None)
    try:
        with pytest.raises(open_weights.OpenWeightsError, match="no Baseten key is set: put BASETEN_HACKATHON_API_KEY or BASETEN_API_KEY"):
            look(run_id="run1")
    finally:
        open_weights.set_client(None)
        open_weights.reset()
    assert ledger.rows() == [] and provenance("run1") == []


def test_no_real_client_is_built_in_an_offline_test(runs_dir, monkeypatch):
    monkeypatch.delenv("LIVE", raising=False)
    monkeypatch.setenv("BASETEN_API_KEY", FAKE_KEY)  # a key is there, as walkaround/.env would give one
    open_weights.reset()
    open_weights.set_client(None)
    try:
        with pytest.raises(open_weights.OpenWeightsError, match="no Baseten client in an offline test"):
            look(run_id="run1")
        assert open_weights._CLIENTS == {}
    finally:
        open_weights.set_client(None)
        open_weights.reset()
    assert ledger.rows() == [] and provenance("run1") == []


# Ledger, price, provenance -------------------------------------------------------------------------------------------


def test_the_ledger_row_carries_the_response_model_tokens_and_list_price(fake):
    fake.reply(chat(SAME, model="meta-models/muse-spark-open-816b", rid="chatcmpl-abc", prompt=1800, completion=420, reasoning=300, cached=200))
    ref = InputRef(path="crops/pickup/p__r.look.jpg", sha256="b" * 64)
    result = look(run_id="run1", input_refs=[ref])
    (row,) = ledger.rows()
    LedgerRow.model_validate(row)
    assert row["kind"] == "spark_open" and row["engine"] == "baseten" and row["note"] == "second_look" and row["run_id"] == "run1"
    assert row["model"] == "meta-models/muse-spark-open-816b" and row["model_requested"] == HACKATHON and row["key_slot"] == "hackathon"
    assert row["request_id"] == "chatcmpl-abc" and row["status"] == "completed" and row["retries"] == 0
    assert row["tokens"] == {"in": 1800, "out": 420, "reasoning": 300, "cached": 200}
    assert row["latency_ms"]["done"] >= 0 and row["unit"] == "tokens" and row["quantity"] == 2220
    expected = (1600 * 1.25 + 200 * 0.15 + 420 * 4.25) / 1e6
    assert row["usd"] == pytest.approx(round(expected, 6)) and result.record.dollars == row["usd"]
    assert result.record.input_refs == [ref] and result.record.purpose is SparkPurpose.SECOND_LOOK


def test_cost_for_spark_open_uses_baseten_list_prices():
    assert ledger.cost_for("spark_open", input_tokens=1_000_000) == pytest.approx(1.25)
    assert ledger.cost_for("spark_open", output_tokens=1_000_000) == pytest.approx(4.25)
    assert ledger.cost_for("spark_open", input_tokens=1_000_000, cached_tokens=1_000_000) == pytest.approx(0.15)
    assert ledger.cost_for("spark_open") == 0.0


def test_provenance_lines_follow_the_binding_shape_and_hold_no_image_bytes_or_key(fake):
    fake.reply(status_error(404, message=f"model not found for key {FAKE_KEY}"), chat(SAME, model=SUFFIX, rid="chatcmpl-9"))
    look(run_id="run1", pair=("p0", "r0"))
    text = paths.open_weights_log("run1").read_text()
    lines = provenance("run1")
    assert [set(line) for line in lines] == [LOG_KEYS, LOG_KEYS]
    failed, answered = lines
    assert failed["status"] == "failed" and failed["http_status"] == 404 and failed["model"] == HACKATHON and failed["model_requested"] == HACKATHON
    assert failed["verdict"] is None and failed["reason"] == "" and failed["valid"] is False and failed["request_id"] is None
    assert (failed["key_slot"], answered["key_slot"]) == ("hackathon", "account")
    assert answered["engine"] == "baseten" and answered["model"] == SUFFIX and answered["request_id"] == "chatcmpl-9"
    assert answered["verdict"] == "same" and answered["reason"] == SAME["reason"] and answered["valid"] is True and answered["error"] is None
    assert answered["ts"].endswith("Z") and isinstance(answered["latency_ms"], int) and answered["run_id"] == "run1"
    for line in lines:
        assert (line["pickup_mark"], line["return_mark"], line["panel_id"]) == ("p0", "r0", "trunk_lid")
    assert FAKE_KEY not in text and "data:image" not in text and "base64" not in text
    assert FAKE_KEY not in ledger.LEDGER_PATH.read_text()


def test_the_reason_on_a_line_is_one_plain_sentence(fake):
    apostrophe, dash, left, right = chr(0x2019), chr(0x2014), chr(0x201C), chr(0x201D)
    long_reason = f"The spot{apostrophe}s edge {dash} a chip {left}by the lock{right} matches. " + "It also sits near the badge. " * 12
    fake.reply(chat({"verdict": "same", "reason": long_reason}))
    result = look(run_id="run1")
    (line,) = provenance("run1")
    assert line["reason"] == "The spot's edge - a chip \"by the lock\" matches."
    assert result.record.parsed["reason"] == long_reason  # the record keeps the answer as it came


def test_plain_reason_is_ascii_one_sentence_and_capped():
    assert open_weights.plain_reason("Approx. 2 cm, e.g. a stone hit. Then more.") == "Approx. 2 cm, e.g. a stone hit."
    capped = open_weights.plain_reason(" ".join(["word"] * 80))
    assert len(capped) <= open_weights.REASON_MAX and capped.endswith("...") and capped.isascii()
    assert open_weights.plain_reason(f"caf{chr(0xE9)}  glare{chr(0x2026)}") == "cafe glare..."


def test_the_baseten_request_id_header_is_kept_on_the_row_and_the_line(fake):
    fake.chat = SimpleNamespace(completions=FakeRawChat())
    fake.reply(status_error(404, request_id="bt-404"), chat(SAME, model=SUFFIX, rid="chatcmpl-7"))
    result = look(run_id="run1")
    assert result.verdict is LookVerdict.SAME and result.record.request_id == "chatcmpl-7"
    assert [line["baseten_request_id"] for line in provenance("run1")] == ["bt-404", "bt-2"]
    assert [row["baseten_request_id"] for row in ledger.rows()] == ["bt-404", "bt-2"]


def test_no_run_id_writes_no_provenance(fake, runs_dir):
    fake.reply(chat(SAME))
    look()
    assert not list(runs_dir.glob("*/open_weights.ndjson"))


def test_request_starts_are_spaced_by_the_rpm(fake, waits, monkeypatch):
    clock = iter([100.0, 100.0, 100.5, 101.0, 101.0, 101.5])  # per call: the throttle, the start, the end
    monkeypatch.setattr(open_weights, "_clock", lambda: next(clock))
    fake.reply(chat(SAME), chat(SAME))
    look(rpm=12)
    look(rpm=12)
    assert waits == [pytest.approx(4.0)]  # 60 / 12 = 5 s between starts; one second had passed


# Count mode on Spark Open (K8) -----------------------------------------------------------------------------------------


def test_count_marks_on_spark_open_clamps_points(fake):
    fake.reply(chat({"points": [{"x": 100, "y": 1001}, {"x": 2000, "y": 5}], "count": 2}))
    result = open_weights.count_marks(crop(), PanelId.TRUNK_LID, rpm=None, run_id="run1")
    assert result.record.engine is SparkEngine.BASETEN and result.count == 2
    assert [(p.x, p.y) for p in result.points] == [(100, 1000)]
    assert fake.requests[0]["messages"][1]["content"][0]["text"].startswith(spark_client.count_prompt(PanelId.TRUNK_LID))
    (line,) = provenance("run1")
    assert (line["purpose"], line["pickup_mark"], line["return_mark"], line["panel_id"], line["verdict"]) == ("count", None, None, "trunk_lid", None)


def test_count_marks_falls_back_to_hosted_spark(fake, monkeypatch):
    fake.reply(status_error(401), status_error(401))
    hosted = []

    def fake_hosted(crop_rgb, panel_id, **kw):
        hosted.append(kw)
        return "hosted answer"

    monkeypatch.setattr(spark_client, "count_marks", fake_hosted)
    assert open_weights.count_marks(crop(), PanelId.TRUNK_LID, rpm=None, run_id="run1") == "hosted answer"
    assert hosted and hosted[0]["run_id"] == "run1"


# Live -------------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="live Baseten call; set LIVE=1 (Baseten credits, no Meta spend)")
def test_live_second_look_on_spark_open():
    open_weights.reset()
    open_weights.set_client(None)
    a, b = crop(), crop(165)
    b[140:160, 190:210] = 170  # no chip at the outline on the return crop
    for rgb in (a, b):
        rgb[136:164, 186:188] = rgb[136:164, 212:214] = (255, 0, 255)
        rgb[136:138, 186:214] = rgb[162:164, 186:214] = (255, 0, 255)
    result = open_weights.second_look(a, b, panel_id=PanelId.TRUNK_LID, run_id="smoke_b11_unit", pair=("p_unit", "r_unit"))
    print("live spark open:", result.verdict.value, result.reason, result.record.model, open_weights.key_slot_of(result.record), result.record.ms, "ms", result.record.usage.model_dump())
    assert result.record.engine is SparkEngine.BASETEN and result.record.model
    assert result.verdict in LookVerdict


# A claim's look: ledger rows under the claim's id, provenance where the caller says (integrator 10/31) ---------------


def test_ledger_run_id_and_log_path_keep_a_claim_out_of_the_runs_sealed_file(fake, runs_dir):
    fake.reply(status_error(403), chat(SAME, model=ACCOUNT, rid="chatcmpl-claim"))
    target = paths.claim_open_weights_log("run1", "claim_1")
    result = look(run_id="run1", ledger_run_id="run1.claim_1", log_path=target, pair=("p0", "claim_1"))
    assert result.verdict is LookVerdict.SAME and result.record.model == ACCOUNT
    assert [(r["run_id"], r["status"], r["key_slot"]) for r in ledger.rows()] == [("run1.claim_1", "failed", "hackathon"), ("run1.claim_1", "completed", "account")]
    lines = [json.loads(text) for text in target.read_text().splitlines()]
    assert [(x["run_id"], x["ledger_run_id"], x["status"], x["return_mark"]) for x in lines] == [("run1", "run1.claim_1", "failed", "claim_1"), ("run1", "run1.claim_1", "completed", "claim_1")]
    assert provenance("run1") == [] and not (runs_dir / "run1.claim_1").exists()


def test_a_walk_look_keeps_its_line_shape(fake):
    fake.reply(chat(SAME, model=HACKATHON, rid="chatcmpl-walk"))
    look(run_id="run1", pair=("p0", "r0"))
    (line,) = provenance("run1")
    assert "ledger_run_id" not in line and set(line) == LOG_KEYS
    assert [r["run_id"] for r in ledger.rows()] == ["run1"]
