"""Offline tests for walkaround.second_look: the callable match.derive_match takes, on a temporary run with fake
Baseten and fake hosted Spark clients. The engine switch is off unless thresholds turn it on; Spark Open answers and
the row follows the binding shape; a bad answer is unsure with no fallback; a Baseten failure falls back to hosted
Spark and says so; two failures in a row skip Baseten; a look that cannot be asked, or that no engine answers, does
not run and leaves the label alone; a resumed run reuses only looks that ran and asks again when the effort changed;
each round keeps only the pairs it asked; the replay answers from looks.json with no call and no write, so
derive_match rebuilds byte-identical; the look only ever lowers a label."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest
from openai import APIConnectionError

from tests.test_open_weights import ACCOUNT, DIFFERENT, HACKATHON, SAME, SUFFIX, FakeBaseten, chat, status_error, timeout
from walkaround import config, geometry, ledger, match, open_weights, paths, second_look, spark_client
from walkaround.contracts import (
    Confidence,
    CoverageState,
    Crop,
    Keyframe,
    KeyframeScore,
    LookVerdict,
    Mark,
    MarkKind,
    MarkSource,
    MarksStatus,
    MatchCandidate,
    Noun,
    Panel,
    PanelId,
    PanelRole,
    SizeClass,
    SparkEngine,
    Verdict,
    VerdictLabel,
    load_thresholds,
    read_json_list,
    write_json,
)

TRUNK = PanelId.TRUNK_LID
ROLES = ("pickup", "return")
RUN = "01TESTRUNLOOK0000000000000"
W, H = 400, 300
RECT_MASK = np.zeros((H, W), dtype=bool)
RECT_MASK[20:280, 20:380] = True
MOMENTS = geometry.moments_from_mask(RECT_MASK)
# whitened positions and crop boxes of the marks, the same in both walks
SPOTS = {"0": ((0.0, 0.0), [190.0, 140.0, 210.0, 160.0]), "1": ((1.0, 0.5), [290.0, 180.0, 310.0, 200.0]), "2": ((-1.2, -0.8), [80.0, 60.0, 100.0, 80.0])}
# CONTRACTS.md, Phase 2 file shapes: every derived/looks.json row carries these
BINDING_ROW = {"pickup_mark", "return_mark", "panel_id", "verdict", "reason", "engine", "model", "request_id", "latency_ms", "fallback_from", "valid"}


@pytest.fixture
def th() -> dict[str, Any]:
    values = load_thresholds()
    values["spark"]["second_look_engine"] = "spark_open"
    return values


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    return root


class FakeHostedResponses:
    def __init__(self) -> None:
        self.replies: list = []
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        assert self.replies, "the hosted fake ran out of canned replies"
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def hosted_answer(payload: dict, rid: str = "resp_hosted") -> dict:
    return {
        "id": rid,
        "model": spark_client.SPARK_MODEL,
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": json.dumps(payload)}]}],
        "usage": {"input_tokens": 3000, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 500, "output_tokens_details": {"reasoning_tokens": 400}},
    }


def connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "https://api.meta.ai/v1/responses"))


@pytest.fixture
def baseten(runs_dir, monkeypatch) -> FakeBaseten:
    client = FakeBaseten()
    monkeypatch.setattr(open_weights, "_sleep", lambda seconds: None)
    open_weights.reset()
    open_weights.set_client(client)
    yield client
    open_weights.set_client(None)
    open_weights.reset()


@pytest.fixture
def hosted(runs_dir, monkeypatch) -> FakeHostedResponses:
    responses = FakeHostedResponses()
    monkeypatch.setattr(spark_client, "_CLIENT", type("FakeHosted", (), {"responses": responses})())
    monkeypatch.setattr(spark_client, "_sleep", lambda seconds: None)
    return responses


# A temporary run: one trunk lid filmed in both walks, three marks per walk at the same spots ---------------------------


def keyframe(role: str, mask_png: str) -> Keyframe:
    return Keyframe(
        kf_id=f"{role}_kf1",
        role=role,
        panel_id=TRUNK,
        track_id=f"{role}_trunk_t1",
        f_proxy=10,
        pts=0.5,
        score=KeyframeScore(area=0.1, completeness=1.0, sharpness=1.0, glare=0.0, frontality=1.0, total=0.1),
        aspect_ratio=1.2,
        box_src=[0.0, 0.0, float(W), float(H)],
        crop=Crop(path=f"crops/{role}/{role}_kf1.jpg", w=W, h=H, scale=1.0, dx=0.0, dy=0.0),
        mask_png=mask_png,
        moments=MOMENTS,
        front_sign=1,
        marks_status=MarksStatus.COMPLETE,
    )


def mark(key: str, role: str, *, box: bool = True, point: bool = True) -> Mark:
    (u, v), bbox = SPOTS[key]
    return Mark(
        mark_id=f"{role[0]}{key}",
        role=role,
        panel_id=TRUNK,
        kf_id=f"{role}_kf1",
        source=MarkSource.SPARK_BOX,
        type=MarkKind.CHIP,
        confidence=Confidence.HIGH,
        size_class=SizeClass.S,
        w=[u, v],
        r=math.hypot(u, v),
        size_units=0.1,
        bbox_crop=bbox if box else None,
        point_crop=[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] if point else None,
        spark_request_id=f"req_{role}",
    )


def seed(run_id: str = RUN, *, marks: list[Mark] | None = None) -> None:
    paths.ensure_run_layout(run_id)
    kfs = []
    for role in ROLES:
        image = np.full((H, W, 3), 200, np.uint8)
        for _, (x1, y1, x2, y2) in SPOTS.values():
            image[int(y1) + 5 : int(y2) - 5, int(x1) + 5 : int(x2) - 5] = 70  # a dark chip inside each box
        cv2.imwrite(str(paths.run_dir(run_id) / f"crops/{role}/{role}_kf1.jpg"), image)
        mask = paths.crop_mask_png_path(run_id, role, f"{role}_kf1")
        cv2.imwrite(str(mask), RECT_MASK.astype(np.uint8) * 255)
        kfs.append(keyframe(role, paths.rel(run_id, mask)))
    panel = Panel(panel_id=TRUNK, noun=Noun.TRUNK_LID, front_sign=1, per_role={role: PanelRole(state=CoverageState.FILMED, keyframe_ids=[f"{role}_kf1"]) for role in ROLES}, comparable=True)
    write_json(paths.panels_json(run_id), [panel])
    write_json(paths.keyframes_json(run_id), kfs)
    write_json(paths.marks_json(run_id), marks if marks is not None else [mark(k, role) for role in ROLES for k in ("0", "1", "2")][:5])


def all_marks() -> list[Mark]:
    return [mark(k, role) for role in ROLES for k in ("0", "1", "2")]


def pair(key: str = "0", other: str | None = None) -> MatchCandidate:
    return MatchCandidate(pickup_mark=f"p{key}", return_mark=f"r{other or key}", d=0.0, d_pos=0.0, type_cost=0.0, size_cost=0.0, r_mean=0.0, tau_pre_r=0.25, tau_eye_r=0.5)


def rows(run_id: str = RUN) -> list[second_look.LookRecord]:
    return second_look.stored_looks(run_id)


def raw_rows(run_id: str = RUN) -> list[dict]:
    return json.loads(paths.looks_json(run_id).read_text())


def provenance(run_id: str = RUN) -> list[dict]:
    path = paths.open_weights_log(run_id)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def edge_tint(rgb: np.ndarray, box: list[float]) -> float:
    """Mean of (R + B) / 2 - G on the box border: about 0 on grey paint, high on the magenta outline."""
    x1, y1, x2, y2 = (int(v) for v in box)
    band = np.concatenate([rgb[y1:y2, x1], rgb[y1:y2, x2 - 1], rgb[y1, x1:x2], rgb[y2 - 1, x1:x2]]).astype(float)
    return float(((band[:, 0] + band[:, 2]) / 2 - band[:, 1]).mean())


def sent_images(body: dict) -> list[bytes]:
    return [base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) for part in body["messages"][1]["content"][1:]]


# The switch ----------------------------------------------------------------------------------------------------------


def test_the_look_is_off_unless_thresholds_turn_it_on(baseten, hosted, th):
    seed()
    plain = load_thresholds()
    plain["spark"].pop("second_look_engine", None)
    assert second_look.make_second_look(RUN, th=plain) is None  # no key: nothing is asked
    off = {**th, "spark": {**th["spark"], "second_look_engine": "off"}}
    assert second_look.make_second_look(RUN, th=off) is None and second_look.make_second_look(RUN, th=th, engine="off") is None
    baseten.reply(chat(SAME))
    second_look.make_second_look(RUN, th=th)(pair("0"))
    assert second_look.replay_second_look(RUN, th=th) is not None and second_look.replay_second_look(RUN, th=off) is None


def test_an_unknown_engine_is_refused(th):
    with pytest.raises(ValueError, match="engine must be one of"):
        second_look.make_second_look(RUN, th=th, engine="glimmer")
    th["spark"]["second_look_engine"] = "glimmer"
    with pytest.raises(ValueError, match="engine must be one of"):
        second_look.make_second_look(RUN, th=th)


# Spark Open answers -------------------------------------------------------------------------------------------------


def test_spark_open_answers_and_the_row_follows_the_binding_shape(baseten, hosted, th):
    seed()
    baseten.reply(chat(SAME, model=SUFFIX, rid="chatcmpl-look1"))
    look = second_look.make_second_look(RUN, th=th)
    assert look(pair("0")) is LookVerdict.SAME
    (raw,) = raw_rows()
    assert BINDING_ROW <= set(raw)
    assert {k: raw[k] for k in BINDING_ROW - {"latency_ms"}} == {
        "pickup_mark": "p0",
        "return_mark": "r0",
        "panel_id": "trunk_lid",
        "verdict": "same",
        "reason": SAME["reason"],
        "engine": "baseten",
        "model": SUFFIX,
        "request_id": "chatcmpl-look1",
        "fallback_from": None,
        "valid": True,
    }
    assert raw["key_slot"] == "hackathon"  # the hackathon key answered with meta-models/muse-spark-open-1026
    (row,) = rows()
    assert isinstance(row.latency_ms, int) and row.latency_ms == row.records[0].ms and row.fallback is None
    assert [r.request_id for r in row.records] == ["chatcmpl-look1"] and row.records[0].input_refs == row.input_refs
    assert row.outlines == [SPOTS["0"][1], SPOTS["0"][1]]
    # the JPEGs kept are the bytes Baseten received, one per walk, with the magenta outline on the box edge
    images = sent_images(baseten.requests[0])
    assert [ref.path for ref in row.input_refs] == ["crops/pickup/p0__r0.look.jpg", "crops/return/p0__r0.look.jpg"]
    for ref, sent in zip(row.input_refs, images):
        kept = (paths.run_dir(RUN) / ref.path).read_bytes()
        assert kept == sent and ref.sha256 == hashlib.sha256(sent).hexdigest()
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(kept, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert edge_tint(rgb, SPOTS["0"][1]) > 60 and abs(edge_tint(np.full_like(rgb, 200), SPOTS["0"][1])) < 1
    assert baseten.requests[0]["reasoning_effort"] == th["spark"]["effort"]
    (line,) = provenance()
    assert (line["pickup_mark"], line["return_mark"], line["panel_id"], line["valid"]) == ("p0", "r0", "trunk_lid", True)
    assert line["model"] == SUFFIX and line["verdict"] == "same" and line["request_id"] == "chatcmpl-look1" and line["run_id"] == RUN
    assert [r["kind"] for r in ledger.rows()] == ["spark_open"] and hosted.requests == []


def test_a_bad_spark_open_answer_is_unsure_with_no_fallback(baseten, hosted, th):
    seed()
    baseten.reply(chat({"verdict": "probably", "reason": "x"}, rid="bt1"), chat("they look alike", rid="bt2"))
    look = second_look.make_second_look(RUN, th=th)
    assert look(pair("0")) is LookVerdict.UNSURE
    (row,) = rows()
    assert row.engine is SparkEngine.BASETEN and row.fallback is None and row.fallback_from is None and row.reason.startswith("second look unusable")
    assert row.valid is False and row.request_id == "bt2" and len(row.records) == 2 and hosted.requests == []


def test_the_reason_on_a_row_is_one_plain_sentence(baseten, hosted, th):
    seed()
    dash, apostrophe = chr(0x2014), chr(0x2019)
    long_reason = f"Both spots match {dash} the chip{apostrophe}s edge lines up. " + "It also sits near the lock. " * 10
    baseten.reply(chat({"verdict": "same", "reason": long_reason}))
    second_look.make_second_look(RUN, th=th)(pair("0"))
    (row,) = rows()
    assert row.reason == "Both spots match - the chip's edge lines up." and row.records[0].parsed["reason"] == long_reason


def test_thresholds_set_the_token_budgets(baseten, hosted, th):
    seed()
    th["spark"]["max_output_tokens"], th["spark"]["retry_max_output_tokens"] = 8000, 12000
    baseten.reply(chat("", finish="length"), chat(SAME))
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is LookVerdict.SAME
    assert [r["max_tokens"] for r in baseten.requests] == [8000, 12000]


# The hosted fallback and the breaker ------------------------------------------------------------------------------------


def test_a_baseten_failure_falls_back_to_hosted_spark_and_says_why(baseten, hosted, th):
    seed()
    baseten.reply(timeout())
    hosted.replies.append(hosted_answer(DIFFERENT, rid="resp_fb"))
    look = second_look.make_second_look(RUN, th=th)
    assert look(pair("0")) is LookVerdict.DIFFERENT
    (row,) = rows()
    assert row.engine is SparkEngine.META_MODEL_API and row.model == spark_client.SPARK_MODEL and row.fallback_from == "baseten"
    assert row.fallback.startswith("Spark Open gave no answer: Baseten call failed: APITimeoutError")
    assert row.request_id == "resp_fb" and row.valid is True
    assert [r.request_id for r in row.records] == ["resp_fb"] and row.records[0].input_refs == row.input_refs
    assert len(hosted.requests) == 1 and len(hosted.requests[0]["input"][1]["content"]) == 3  # the question and the two crops
    assert [(line["status"], line["pickup_mark"]) for line in provenance()] == [("failed", "p0")]
    assert [(r["kind"], r["status"]) for r in ledger.rows()] == [("spark_open", "failed"), ("spark", "completed")]


def test_the_account_key_answers_when_the_hackathon_key_is_refused_and_the_row_says_so(baseten, hosted, th):
    seed()
    baseten.reply(status_error(403), chat(DIFFERENT, model=ACCOUNT, rid="chatcmpl-acct"))
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is LookVerdict.DIFFERENT
    (row,) = rows()
    assert (row.engine, row.model, row.key_slot, row.request_id) == (SparkEngine.BASETEN, ACCOUNT, "account", "chatcmpl-acct")
    assert row.fallback is None and row.fallback_from is None and hosted.requests == []
    assert [r["model"] for r in baseten.requests] == [HACKATHON, ACCOUNT]
    assert [(line["key_slot"], line["status"]) for line in provenance()] == [("hackathon", "failed"), ("account", "completed")]


def test_a_hosted_answer_has_no_key_slot(baseten, hosted, th):
    seed()
    baseten.reply(timeout())
    hosted.replies.append(hosted_answer(SAME, rid="resp_h"))
    second_look.make_second_look(RUN, th=th)(pair("0"))
    (raw,) = raw_rows()
    assert raw["engine"] == "meta-model-api" and raw["key_slot"] is None


def test_hosted_gets_the_same_jpeg_bytes_as_baseten_would(baseten, hosted, th):
    seed()
    baseten.reply(status_error(401), status_error(401))
    hosted.replies.append(hosted_answer(SAME))
    second_look.make_second_look(RUN, th=th)(pair("0"))
    (row,) = rows()
    parts = hosted.requests[0]["input"][1]["content"][1:]
    sent = [base64.b64decode(p["image_url"].split(",", 1)[1]) for p in parts]
    assert [hashlib.sha256(s).hexdigest() for s in sent] == [ref.sha256 for ref in row.input_refs]


def test_a_failed_retry_keeps_the_first_spark_open_answer_on_the_row(baseten, hosted, th):
    seed()
    baseten.reply(chat("no json", rid="bt1"), timeout())
    hosted.replies.append(hosted_answer(SAME, rid="resp_2"))
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is LookVerdict.SAME
    (row,) = rows()
    assert [r.request_id for r in row.records] == ["bt1", "resp_2"] and [r.engine for r in row.records] == [SparkEngine.BASETEN, SparkEngine.META_MODEL_API]
    assert row.request_id == "resp_2" and row.fallback_from == "baseten"


def test_two_failures_in_a_row_send_the_rest_of_the_run_to_hosted_spark(baseten, hosted, th):
    seed(marks=all_marks())
    baseten.reply(timeout(), status_error(503), status_error(503), status_error(503))
    hosted.replies.extend([hosted_answer(SAME, rid=f"resp_{i}") for i in range(3)])
    look = second_look.make_second_look(RUN, th=th)
    assert [look(pair(k)) for k in ("0", "1", "2")] == [LookVerdict.SAME] * 3
    assert len(baseten.requests) == 4  # one timeout, then three 503s on the second look; nothing on the third
    by_pair = {r.pickup_mark: r for r in rows()}
    assert by_pair["p2"].fallback.startswith("Spark Open skipped after 2 failed looks in a row")
    assert all(r.engine is SparkEngine.META_MODEL_API and r.fallback_from == "baseten" for r in by_pair.values())


def test_a_success_resets_the_breaker(baseten, hosted, th):
    seed(marks=all_marks())
    baseten.reply(timeout(), chat(SAME), timeout(), chat(DIFFERENT))
    hosted.replies.extend([hosted_answer(SAME), hosted_answer(SAME)])
    look = second_look.make_second_look(RUN, th=th)
    assert [look(pair(k)) for k in ("0", "1", "2")] == [LookVerdict.SAME, LookVerdict.SAME, LookVerdict.SAME]
    assert len(baseten.requests) == 3


def test_a_hosted_crash_after_a_bad_spark_open_answer_keeps_that_answer(baseten, hosted, th, monkeypatch):
    seed()
    baseten.reply(chat("no json here", rid="bt_first"), timeout())

    def no_key(*args, **kwargs):
        raise RuntimeError("MODEL_API_KEY is not set. Put it in walkaround/.env (see .env.example).")

    monkeypatch.setattr(spark_client, "second_look", no_key)
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is LookVerdict.UNSURE  # Spark Open did answer, unusably
    (row,) = rows()
    assert row.engine is SparkEngine.BASETEN and row.valid is False and row.request_id == "bt_first" and row.panel_id is TRUNK
    assert [r.request_id for r in row.records] == ["bt_first"] and row.fallback_from is None
    assert "Spark Open gave no answer" in row.fallback and "Muse Spark 1.3 failed: RuntimeError: MODEL_API_KEY is not set" in row.fallback
    assert [(r["kind"], r["status"]) for r in ledger.rows()] == [("spark_open", "schema_miss"), ("spark_open", "failed")]


# Looks that do not run leave the label alone ---------------------------------------------------------------------------


def test_a_hosted_crash_with_no_spark_open_answer_does_not_run(baseten, hosted, th, monkeypatch):
    seed()
    baseten.reply(status_error(401), status_error(401))

    def boom(*args, **kwargs):
        raise RuntimeError("no key")

    monkeypatch.setattr(spark_client, "second_look", boom)
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is None
    (row,) = rows()
    assert row.verdict is None and row.engine is None and row.valid is False and row.records == [] and row.panel_id is TRUNK
    assert row.reason == "not run: Muse Spark 1.3 failed: RuntimeError: no key"
    assert row.fallback.startswith("Spark Open gave no answer: no Baseten key slot answered in the last 10 minutes (hackathon key: 401; account key: 401)")


def test_both_engines_down_does_not_run_and_a_resume_asks_again(baseten, hosted, th):
    seed()
    baseten.reply(timeout())
    hosted.replies.extend([connection_error(), connection_error()])
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is None
    (row,) = rows()
    assert row.verdict is None and row.question_sha256 is not None and len(row.records) == 2  # the two hosted errors, kept
    assert row.reason == "not run: Muse Spark 1.3 gave no answer: error: APIConnectionError: Connection error."
    assert "Spark Open gave no answer" in row.fallback
    baseten.reply(chat(SAME, rid="bt_up"))  # the network is back
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is LookVerdict.SAME
    assert len(baseten.requests) == 2 and rows()[0].request_id == "bt_up"


@pytest.mark.parametrize(
    "marks, why",
    [
        ([mark("0", "pickup", box=False, point=False), mark("0", "return")], "no box or point for p0"),
        ([mark("0", "pickup")], "a mark of the pair is not in marks.json"),
    ],
)
def test_a_look_that_cannot_be_asked_does_not_run_and_costs_nothing(baseten, hosted, th, marks, why):
    seed(marks=marks)
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is None
    (raw,) = raw_rows()
    assert BINDING_ROW <= set(raw)
    (row,) = rows()
    assert row.reason == f"not run: {why}" and row.verdict is None and row.engine is None and row.valid is False and row.panel_id is TRUNK
    assert row.records == [] and baseten.requests == [] and hosted.requests == [] and ledger.rows() == []


def test_a_missing_crop_file_does_not_run(baseten, hosted, th):
    seed()
    (paths.run_dir(RUN) / "crops/return/return_kf1.jpg").unlink()
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is None
    assert rows()[0].reason == "not run: crop file missing for r0" and baseten.requests == []


def test_a_look_that_crashes_does_not_run_and_is_recorded(baseten, hosted, th, monkeypatch):
    seed()

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(second_look, "_outlined", boom)
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is None
    (row,) = rows()
    assert row.reason == "not run: the look failed: RuntimeError: disk on fire" and row.panel_id is TRUNK and row.verdict is None


# Engines, reuse, pruning ---------------------------------------------------------------------------------------------


def test_engine_spark_asks_hosted_spark_only(baseten, hosted, th):
    seed()
    hosted.replies.append(hosted_answer(SAME))
    assert second_look.make_second_look(RUN, th=th, engine="spark")(pair("0")) is LookVerdict.SAME
    (row,) = rows()
    assert row.engine is SparkEngine.META_MODEL_API and row.fallback is None and row.fallback_from is None and baseten.requests == []


def test_the_same_question_is_asked_once_per_run(baseten, hosted, th):
    seed()
    baseten.reply(chat(DIFFERENT))
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is LookVerdict.DIFFERENT
    before = paths.looks_json(RUN).read_bytes()
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is LookVerdict.DIFFERENT  # a resumed run
    assert len(baseten.requests) == 1 and paths.looks_json(RUN).read_bytes() == before


def test_a_changed_effort_asks_again(baseten, hosted, th):
    seed()
    baseten.reply(chat(DIFFERENT), chat(SAME))
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is LookVerdict.DIFFERENT
    th["spark"]["effort"] = "medium"
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is LookVerdict.SAME
    assert [r["reasoning_effort"] for r in baseten.requests] == ["low", "medium"]


def test_looks_json_is_sorted_by_pair(baseten, hosted, th):
    seed(marks=all_marks())
    baseten.reply(chat(SAME), chat(SAME), chat(SAME))
    look = second_look.make_second_look(RUN, th=th)
    for key in ("2", "0", "1"):
        look(pair(key))
    assert [(r.pickup_mark, r.return_mark) for r in rows()] == [("p0", "r0"), ("p1", "r1"), ("p2", "r2")]


def test_each_round_keeps_only_the_pairs_it_asked(baseten, hosted, th):
    seed(marks=all_marks())
    baseten.reply(chat(SAME), chat(DIFFERENT))
    first = second_look.make_second_look(RUN, th=th)
    first(pair("0"))
    first(pair("1"))
    assert second_look.make_second_look(RUN, th=th)(pair("1")) is LookVerdict.DIFFERENT  # reused, not asked again
    assert [(r.pickup_mark, r.verdict) for r in rows()] == [("p1", LookVerdict.DIFFERENT)] and len(baseten.requests) == 2


# Replay and the match ------------------------------------------------------------------------------------------------


def test_replay_answers_from_looks_json_with_no_call_and_no_write(baseten, hosted, th):
    seed(marks=all_marks())
    baseten.reply(chat(SAME), chat(DIFFERENT))
    hosted.replies.append(hosted_answer({"verdict": "unsure", "reason": "glare"}))
    look = second_look.make_second_look(RUN, th=th)
    first_two = [look(pair("0")), look(pair("1"))]
    baseten.reply(status_error(401), status_error(401))
    live = [*first_two, look(pair("2"))]
    assert live == [LookVerdict.SAME, LookVerdict.DIFFERENT, LookVerdict.UNSURE]
    files = {p: p.read_bytes() for p in (paths.looks_json(RUN), paths.open_weights_log(RUN), ledger.LEDGER_PATH)}
    asked = (len(baseten.requests), len(hosted.requests))
    replay = second_look.replay_second_look(RUN, th=th)
    assert [replay(pair(k)) for k in ("0", "1", "2")] == live
    assert replay(pair("0", "1")) is None  # no row, no K7
    assert (len(baseten.requests), len(hosted.requests)) == asked
    assert {p: p.read_bytes() for p in files} == files


def test_replay_of_a_look_that_did_not_run_is_none(baseten, hosted, th):
    seed()
    (paths.run_dir(RUN) / "crops/pickup/pickup_kf1.jpg").unlink()
    assert second_look.make_second_look(RUN, th=th)(pair("0")) is None
    assert second_look.replay_second_look(RUN)(pair("0")) is None


def test_replay_of_a_run_with_no_looks_is_none(runs_dir):
    paths.ensure_run_layout(RUN)
    assert second_look.replay_second_look(RUN) is None


def labels(verdicts: list[Verdict]) -> dict[tuple[str, ...], tuple[str, list[str]]]:
    return {tuple(v.mark_ids): (v.label.value, v.trace.rule_ids) for v in verdicts}


def test_derive_match_with_the_look_lowers_only_and_the_replay_rebuilds_byte_identical(baseten, hosted, th):
    seed()  # p0, p1, p2 at pickup; r0, r1 at return: two pairs at d = 0 and one pickup mark unmatched
    _, plain, _ = match.derive_match(RUN, th=th)
    assert labels(plain)[("p0", "r0")][0] == labels(plain)[("p1", "r1")][0] == "already_there"
    baseten.reply(chat(SAME, rid="look_a"), chat(DIFFERENT, rid="look_b"))
    _, verdicts, _ = match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))
    looked = {(r.pickup_mark, r.return_mark): r.verdict for r in rows()}
    assert sorted(looked.values(), key=lambda v: v.value) == [LookVerdict.DIFFERENT, LookVerdict.SAME]
    for (p, r), verdict in looked.items():
        label, rule_ids = labels(verdicts)[(p, r)]
        if verdict is LookVerdict.SAME:
            assert (label, rule_ids) == ("already_there", ["L3"])
        else:
            assert (label, rule_ids) == ("needs_your_eye", ["L4", "L3", "K7"])
    assert labels(verdicts)[("p2",)] == labels(plain)[("p2",)]  # an unmatched mark is never looked at
    live = (paths.matches_json(RUN).read_bytes(), paths.verdicts_json(RUN).read_bytes())
    kept = (paths.looks_json(RUN).read_bytes(), paths.open_weights_log(RUN).read_bytes())
    asked = len(baseten.requests)
    match.derive_match(RUN, th=th, second_look=second_look.replay_second_look(RUN, th=th))
    assert (paths.matches_json(RUN).read_bytes(), paths.verdicts_json(RUN).read_bytes()) == live
    assert (paths.looks_json(RUN).read_bytes(), paths.open_weights_log(RUN).read_bytes()) == kept and len(baseten.requests) == asked
    stored = read_json_list(paths.verdicts_json(RUN), Verdict)
    assert sum(v.label == VerdictLabel.NEEDS_YOUR_EYE for v in stored) == 1


def test_a_look_that_did_not_run_leaves_the_deterministic_label(baseten, hosted, th):
    seed()
    (paths.run_dir(RUN) / "crops/return/return_kf1.jpg").unlink()  # no return crop: neither look can be asked
    match.derive_match(RUN, th=th)
    plain = (paths.matches_json(RUN).read_bytes(), paths.verdicts_json(RUN).read_bytes())
    _, verdicts, _ = match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))
    assert (paths.matches_json(RUN).read_bytes(), paths.verdicts_json(RUN).read_bytes()) == plain
    assert labels(verdicts)[("p0", "r0")] == labels(verdicts)[("p1", "r1")] == ("already_there", ["L3"])
    assert [r.verdict for r in rows()] == [None, None] and baseten.requests == [] and hosted.requests == []
    match.derive_match(RUN, th=th, second_look=second_look.replay_second_look(RUN, th=th))
    assert (paths.matches_json(RUN).read_bytes(), paths.verdicts_json(RUN).read_bytes()) == plain


# look_images: two outlined images the caller already holds (a claim's look), lead's signature of 10/31 ------------------

CLAIM_BILL = f"{RUN}.claim_1"


def _two_images() -> tuple[np.ndarray, np.ndarray]:
    a = np.full((H, W, 3), 200, np.uint8)
    a[140:160, 190:210] = 70
    return a, a.copy()


def test_look_images_asks_spark_open_first_bills_the_claim_and_logs_where_told(baseten, hosted, th, monkeypatch):
    monkeypatch.setattr(second_look, "_IMAGE_BREAKERS", {})
    paths.ensure_run_layout(RUN)
    baseten.reply(chat(SAME, model=HACKATHON, rid="chatcmpl-claimlook"))
    target = paths.claim_open_weights_log(RUN, "claim_1")
    a, b = _two_images()
    standing, records, notes = second_look.look_images(
        a, b, panel_id=TRUNK, run_id=RUN, ledger_run_id=CLAIM_BILL, input_refs=[], th=th, pair=("p0", "claim_1"), provenance=target
    )
    assert standing is not None and standing.engine is SparkEngine.BASETEN and standing.model == HACKATHON and notes == []
    assert [r.request_id for r in records] == ["chatcmpl-claimlook"] and open_weights.key_slot_of(standing) == "hackathon"
    assert [(row["kind"], row["run_id"]) for row in ledger.rows()] == [("spark_open", CLAIM_BILL)]
    (line,) = [json.loads(text) for text in target.read_text().splitlines()]
    assert (line["run_id"], line["ledger_run_id"], line["pickup_mark"], line["return_mark"], line["verdict"]) == (RUN, CLAIM_BILL, "p0", "claim_1", "same")
    assert not paths.open_weights_log(RUN).is_file()  # the run's own file, sealed at run end, is never appended to
    assert not (paths.runs_root() / CLAIM_BILL).exists()  # and no folder is named after the ledger id
    assert hosted.requests == []


def test_look_images_falls_back_to_hosted_spark_under_the_claims_ledger_id(baseten, hosted, th, monkeypatch):
    monkeypatch.setattr(second_look, "_IMAGE_BREAKERS", {})
    paths.ensure_run_layout(RUN)
    baseten.reply(timeout())
    hosted.replies.append(hosted_answer(DIFFERENT, rid="resp_claim_fb"))
    target = paths.claim_open_weights_log(RUN, "claim_1")
    a, b = _two_images()
    standing, records, notes = second_look.look_images(
        a, b, panel_id=TRUNK, run_id=RUN, ledger_run_id=CLAIM_BILL, input_refs=[], th=th, pair=("p0", "claim_1"), provenance=target
    )
    assert standing is not None and standing.engine is SparkEngine.META_MODEL_API and standing.request_id == "resp_claim_fb"
    assert notes and notes[0].startswith("Spark Open gave no answer: Baseten call failed: APITimeoutError")
    assert [(row["kind"], row["status"], row["run_id"]) for row in ledger.rows()] == [("spark_open", "failed", CLAIM_BILL), ("spark", "completed", CLAIM_BILL)]
    assert [json.loads(text)["status"] for text in target.read_text().splitlines()] == ["failed"]
    assert [r.request_id for r in records] == ["resp_claim_fb"]


def test_look_images_keeps_one_breaker_per_run_and_asks_nothing_when_off(baseten, hosted, th, monkeypatch):
    monkeypatch.setattr(second_look, "_IMAGE_BREAKERS", {})
    paths.ensure_run_layout(RUN)
    a, b = _two_images()
    baseten.reply(timeout(), timeout())
    hosted.replies.extend([hosted_answer(SAME, rid="h1"), hosted_answer(SAME, rid="h2"), hosted_answer(SAME, rid="h3")])
    for _ in range(3):
        second_look.look_images(a, b, panel_id=TRUNK, run_id=RUN, input_refs=[], th=th)
    assert len(baseten.requests) == 2 and len(hosted.requests) == 3  # two failures in a row: the third goes to hosted only
    off = dict(th, spark={**th["spark"], "second_look_engine": "off"})
    assert second_look.look_images(a, b, panel_id=TRUNK, run_id=RUN, input_refs=[], th=off) == (None, [], ["the second look is off (thresholds spark.second_look_engine)"])
    assert len(hosted.requests) == 3
