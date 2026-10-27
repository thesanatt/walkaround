"""Offline tests for walkaround.sam_wire and walkaround.sam_client: the conformance corpus, the 10/26 fixtures, a fake
client. One live smoke runs only with LIVE=1 (about $0.06)."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from meta_sam_parser import InvalidSegmentationMaskError, decode_mask_to_raster
from openai import APIConnectionError, BadRequestError, RateLimitError

from walkaround import contracts, ledger, paths, sam_client, sam_wire
from walkaround.config import SAM_MODEL
from walkaround.contracts import IndexFrame, LaneStatus, MaskEncoding, MaskRef, Noun

ROOT = Path(__file__).resolve().parent
TEXT_FIXTURE = ROOT / "fixtures" / "sam" / "pingpong_text.events.ndjson"
BOX_FIXTURE = ROOT / "fixtures" / "sam" / "pingpong_box.events.ndjson"
ZERO_FIXTURE = ROOT / "fixtures" / "sam_zero_match.events.ndjson"
CASES = sorted((ROOT / "sam_conformance" / "cases").glob("*.json"))
PROMPT_CASES = sorted((ROOT / "sam_conformance" / "object-prompts").glob("*.json"))
PINGPONG = ROOT.parent / "data" / "test" / "pingpong.mp4"
SIZE = (1280, 720)
FAKE_KEY = "LLM|test-team|test-secret"


def events_of(path: Path) -> list[dict[str, Any]]:
    return list(sam_wire.read_events(path))


# ----------------------------------------------------------------------------------------------------------------
# Conformance corpus -> Responses events
# ----------------------------------------------------------------------------------------------------------------


def case_events(case: dict[str, Any]) -> list[dict[str, Any]]:
    """The case's normalized source events as the Responses-event dicts the readers consume."""
    chunks = case["chunks"]
    out: list[dict[str, Any]] = []
    for e in case["events"]:
        lane = e.get("lane") or {}
        base = {k: lane[k] for k in ("item_id", "output_index", "content_index") if k in lane}
        kind = e["type"]
        if kind == "output_text_delta":
            out.append({"type": "response.output_text.delta", **base, "delta": chunks[e["chunk"]]})
        elif kind == "output_text_done":
            out.append({"type": "response.output_text.done", **base, "text": e["text"]})
        elif kind == "content_part_done":
            part: dict[str, Any] = {"type": e["part_type"]}
            part["refusal" if e["part_type"] == "refusal" else "text"] = e["text"]
            out.append({"type": "response.content_part.done", **base, "part": part})
        elif kind == "response_completed":
            out.append({"type": "response.completed", "response": {"id": "resp_case", "status": "completed"}})
        elif kind == "response_incomplete":
            response: dict[str, Any] = {"status": "incomplete"}
            if "detail" in e:
                response["incomplete_details"] = {"reason": e["detail"]}
            out.append({"type": "response.incomplete", "response": response})
        elif kind == "response_failed":
            out.append({"type": "response.failed", "response": {"status": "failed", "error": {"message": e["message"]}}})
        elif kind == "error":
            out.append({"type": "error", "error": {"message": e["message"]}})
        elif kind == "refusal_delta":
            out.append({"type": "response.refusal.delta", **base, "delta": e["text"]})
        elif kind == "refusal_done":
            out.append({"type": "response.refusal.done", **base, "refusal": e["text"]})
        else:
            raise ValueError(kind)
    source = case.get("source")
    if source and source["operation"] == "iterator":
        return []
    if source and source["operation"] == "next":
        return out[: source["after_events"]]
    return out


def flatten(frames: list[IndexFrame], text: str, media: str) -> list[tuple[Any, ...]]:
    """Box and mask tuples in output order, the normalized form the cases use."""
    out: list[tuple[Any, ...]] = []
    for frame in frames:
        index = None if media == "image" else frame.frame
        for obj in frame.objects:
            out.append(("box", obj.id, index, obj.box[0], obj.box[1], obj.box[2], obj.box[3], obj.c))
            if obj.mask is not None:
                out.append(("mask", obj.id, index, obj.mask.encoding.value, sam_wire.mask_payload(text, obj.mask), obj.mask.w, obj.mask.h))
    return out


def expected_records(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for r in records:
        if r["kind"] == "box":
            out.append(("box", r["object_id"], r["frame_index"], r["left"], r["top"], r["right"], r["bottom"], r.get("confidence")))
        elif r["kind"] == "mask":
            m = r["mask"]
            out.append(("mask", r["object_id"], r["frame_index"], m["encoding"], m["payload"], m["width"], m["height"]))
    return out


def ref_for(record: tuple[Any, ...]) -> MaskRef:
    return MaskRef(h=record[6], w=record[5], encoding=MaskEncoding(record[3]), payload_offset=0, payload_len=len(record[4]))


@pytest.mark.parametrize("case_path", CASES, ids=[p.stem for p in CASES])
def test_reader_matches_conformance_case(case_path: Path) -> None:
    case = json.loads(case_path.read_text())
    media = case["media"]
    events = case_events(case)
    frames, outcome = sam_wire.index_from_events(events, media=media)
    text = sam_wire.lane_text(events)
    expected = case["expected"]
    if "result" in expected:
        exp_records, exp_diags = expected["result"]["records"], expected["result"]["diagnostics"]
        if expected["result"]["outcome"]["status"] == "completed":
            if any(d["severity"] == "error" for d in exp_diags):
                assert outcome.status is LaneStatus.INCOMPLETE
            else:
                assert outcome.status in (LaneStatus.COMPLETED, LaneStatus.ZERO_MATCHES)
                assert (outcome.status is LaneStatus.ZERO_MATCHES) == (not frames)
        else:
            assert outcome.status is LaneStatus.INCOMPLETE
    else:
        snapshots = expected.get("snapshots") or []
        exp_records = snapshots[-1]["records"] if snapshots else []
        exp_diags = snapshots[-1]["diagnostics"] if snapshots else []
        if expected["error"]["code"] in ("response_refusal", "response_failed", "response_error"):
            assert outcome.status is LaneStatus.INCOMPLETE
            assert outcome.error
    theirs = expected_records(exp_records)
    mine: list[tuple[Any, ...]] = []
    for record in flatten(frames, text, media):
        if record[0] == "mask" and record not in theirs:
            # The thin reader keeps a payload the parser rejected; it must fail at decode time instead.
            with pytest.raises(InvalidSegmentationMaskError):
                sam_wire.decode_mask(ref_for(record), record[4])
            continue
        mine.append(record)
    assert mine == theirs
    my_diags = sorted((d.severity, d.code, d.message, d.line) for d in outcome.diagnostics if d.code in sam_wire.PARSER_DIAGNOSTIC_CODES)
    their_diags = sorted((d["severity"], d["code"], d["message"], d["line"]) for d in exp_diags if d["code"] != "invalid_mask_payload")
    assert my_diags == their_diags
    for record in mine:
        if record[0] == "mask":
            raster = sam_wire.decode_mask(ref_for(record), record[4])
            assert raster.shape == (record[6], record[5]) and raster.dtype == bool
    if "result" in expected:
        # The contract's oracle check: the same index through meta_sam_parser (offsets aside).
        oracle = sam_wire.oracle_index(events, media=media)
        assert strip_offsets(frames, text, drop_invalid=True) == strip_offsets(oracle, None)


def strip_offsets(frames: list[IndexFrame], text: str | None, *, drop_invalid: bool = False) -> list[tuple[Any, ...]]:
    out = []
    for frame in frames:
        objects = []
        for obj in frame.objects:
            mask = None
            if obj.mask is not None:
                keep = True
                if drop_invalid and text is not None:
                    try:
                        sam_wire.decode_mask(obj.mask, sam_wire.mask_payload(text, obj.mask))
                    except InvalidSegmentationMaskError:
                        keep = False
                if keep:
                    mask = (obj.mask.h, obj.mask.w, obj.mask.encoding.value, obj.mask.payload_len)
            objects.append((obj.id, tuple(obj.box), obj.c, mask))
        out.append((frame.frame, tuple(objects)))
    return out


@pytest.mark.parametrize("case_path", PROMPT_CASES, ids=[p.stem for p in PROMPT_CASES])
def test_object_prompt_builder_cases(case_path: Path) -> None:
    case = json.loads(case_path.read_text())
    size = case["input"]["size"]
    objects = []
    for obj in case["input"]["objects"]:
        prompts = []
        for prompt in obj["prompts"]:
            box = prompt.get("box")
            prompts.append(
                sam_client.FramePrompt(
                    frame=prompt.get("frame", 0),
                    box=None if box is None else sam_client.PromptBox(box["left"], box["top"], box["right"], box["bottom"]),
                    points=tuple(
                        sam_client.PromptPoint(p["x"], p["y"], p["label"]) if "label" in p else sam_client.PromptPoint(p["x"], p["y"])
                        for p in prompt.get("points", [])
                    ),
                )
            )
        objects.append(sam_client.PromptObject(id=obj["id"], prompts=prompts))
    expected = case["expected"]
    if "text" in expected:
        assert sam_client.object_prompt(objects, (size["width"], size["height"])) == expected["text"]
    else:
        with pytest.raises(sam_client.ObjectPromptError) as info:
            sam_client.object_prompt(objects, (size["width"], size["height"]))
        assert info.value.code == expected["error"]


# ----------------------------------------------------------------------------------------------------------------
# The 10/26 fixtures and the zero-match lane
# ----------------------------------------------------------------------------------------------------------------


def test_text_fixture_index() -> None:
    events = events_of(TEXT_FIXTURE)
    frames, outcome = sam_wire.index_from_events(events, media="video")
    assert outcome.status is LaneStatus.COMPLETED
    assert outcome.response_id == "resp_6adfef8768a71da59fd247f8"
    assert outcome.store_echo is True  # the probe did not send store false
    assert outcome.video_frames_processed == 141
    assert outcome.usage["video_frames_processed"] == 141
    assert outcome.diagnostics == []
    assert len(frames) == 134 and frames[0].frame == 0 and frames[-1].frame == 136
    assert [f.frame for f in frames] == sorted(f.frame for f in frames)
    objects = [o for f in frames for o in f.objects]
    assert {o.id for o in objects} == {"0"}
    assert {o.c for o in objects} == {0.7299578189849854}  # constant on a text lane
    for obj in objects:
        assert obj.mask is not None and obj.mask.encoding is MaskEncoding.ONE_BIT
        assert (obj.mask.w, obj.mask.h) == (obj.box[2] - obj.box[0], obj.box[3] - obj.box[1])
    text = sam_wire.lane_text(events)
    assert outcome.text_chars == len(text) == 22163
    first = frames[0].objects[0]
    assert first.box == [593, 538, 621, 570]
    payload = sam_wire.mask_payload(text, first.mask)
    assert payload.startswith("!!!!!=F^%Y0") and len(payload) == first.mask.payload_len
    raster = sam_wire.decode_mask(first.mask, payload)
    assert raster.shape == (32, 28) and raster.dtype == bool and 0 < raster.sum() < raster.size
    assert strip_offsets(frames, None) == strip_offsets(sam_wire.oracle_index(events, media="video"), None)


def test_box_fixture_has_per_frame_confidence() -> None:
    events = events_of(BOX_FIXTURE)
    frames, outcome = sam_wire.index_from_events(events, media="video")
    assert outcome.status is LaneStatus.COMPLETED
    assert outcome.video_frames_processed == 141 and len(frames) == 134
    objects = [o for f in frames for o in f.objects]
    assert {o.id for o in objects} == {"1"}
    cs = [o.c for o in objects]
    assert cs[:3] == [0.890625, 0.8668561577796936, 0.9177275896072388]
    assert len(set(cs)) > 100
    assert strip_offsets(frames, None) == strip_offsets(sam_wire.oracle_index(events, media="video"), None)


def test_zero_match_fixture_is_zero_matches_not_incomplete() -> None:
    frames, outcome = sam_wire.index_from_events(events_of(ZERO_FIXTURE), media="video")
    assert frames == []
    assert outcome.status is LaneStatus.ZERO_MATCHES
    assert outcome.video_frames_processed == 141
    assert outcome.store_echo is False
    assert outcome.text_chars == 0 and outcome.error is None and outcome.incomplete_reason is None


def test_truncated_stream_is_incomplete_with_eof() -> None:
    events = events_of(TEXT_FIXTURE)
    cut = [e for e in events if e["type"] != "response.completed"]
    frames, outcome = sam_wire.index_from_events(cut, media="video")
    assert len(frames) == 134 and outcome.status is LaneStatus.INCOMPLETE and outcome.incomplete_reason == "eof"
    assert outcome.video_frames_processed is None
    half = [e for e in events if e["type"] != "response.output_text.delta" or e["sequence_number"] < 60]
    half = [e for e in half if e["type"] not in ("response.output_text.done", "response.content_part.done", "response.output_item.done", "response.completed")]
    frames, outcome = sam_wire.index_from_events(half, media="video")
    assert 0 < len(frames) < 134 and outcome.status is LaneStatus.INCOMPLETE


def test_finalizer_variants_read_the_same() -> None:
    events = events_of(TEXT_FIXTURE)
    base, _ = sam_wire.index_from_events(events, media="video")
    only_part = [e for e in events if e["type"] != "response.output_text.done"]
    only_done = [e for e in events if e["type"] != "response.content_part.done"]
    neither = [e for e in events if e["type"] not in ("response.output_text.done", "response.content_part.done")]
    for variant in (only_part, only_done, neither):
        frames, outcome = sam_wire.index_from_events(variant, media="video")
        assert outcome.status is LaneStatus.COMPLETED and frames == base
    # No deltas at all: the finalizer text feeds the reader.
    no_deltas = [e for e in events if e["type"] != "response.output_text.delta"]
    frames, outcome = sam_wire.index_from_events(no_deltas, media="video")
    assert outcome.status is LaneStatus.COMPLETED and frames == base
    conflicting = [dict(e, text=e["text"] + "x") if e["type"] == "response.output_text.done" else e for e in events]
    frames, outcome = sam_wire.index_from_events(conflicting, media="video")
    assert outcome.status is LaneStatus.INCOMPLETE and frames == base
    assert [d.code for d in outcome.diagnostics if d.severity == "error"] == ["finalizer_conflict"]


def test_read_events_gz_round_trip(tmp_path: Path) -> None:
    events = events_of(TEXT_FIXTURE)
    gz = tmp_path / "lane.ndjson.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    assert list(sam_wire.read_events(gz)) == events


def test_paste_mask_clips_at_the_frame_edge() -> None:
    raster = np.ones((3, 4), dtype=bool)
    ref = MaskRef(h=3, w=4, encoding=MaskEncoding.ONE_BIT, payload_offset=0, payload_len=0)
    full = sam_wire.paste_mask(raster, [8, 5, 12, 8], ref, (10, 6))
    assert full.shape == (6, 10) and full.sum() == 2 and full[5, 8] and full[5, 9]


def test_decode_pool_matches_decode_mask_to_raster(tmp_path: Path) -> None:
    events = events_of(TEXT_FIXTURE)
    frames, _ = sam_wire.index_from_events(events, media="video")
    text = sam_wire.lane_text(events)
    gz = tmp_path / "lane.ndjson.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    refs = [frames[i].objects[0].mask for i in (0, 50, 133)]
    with sam_wire.DecodePool(2) as pool:
        rasters = pool.decode(gz, refs)
        assert pool.stats()[0] == 3 and pool.stats()[1] >= 0.0
        assert pool.decode(gz, []) == []
    for ref, raster in zip(refs, rasters):
        expected = decode_mask_to_raster(sam_client.SegmentationMask(encoding=ref.encoding.value, payload=sam_wire.mask_payload(text, ref), width=ref.w, height=ref.h))
        assert raster.shape == (ref.h, ref.w)
        assert np.array_equal(raster, np.frombuffer(expected, dtype=np.uint8).reshape(ref.h, ref.w).astype(bool))
    bad = refs[0].model_copy(update={"payload_len": refs[0].payload_len - 3})
    with sam_wire.DecodePool(1) as pool, pytest.raises(InvalidSegmentationMaskError):
        pool.decode(gz, [bad])


# ----------------------------------------------------------------------------------------------------------------
# A fake AsyncOpenAI for the client
# ----------------------------------------------------------------------------------------------------------------


class FakeStream:
    def __init__(self, events: list[dict[str, Any]], *, stop_after: int | None = None) -> None:
        self._events = events if stop_after is None else events[:stop_after]
        self.closed = False

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __aiter__(self) -> Any:
        return self._gen()

    async def _gen(self) -> Any:
        for event in self._events:
            yield event

    async def aclose(self) -> None:
        self.closed = True


class FakeResponses:
    def __init__(self, server: FakeServer) -> None:
        self.server = server
        self.with_raw_response = self

    async def create(self, **kwargs: Any) -> FakeStream:
        self.server.requests.append(kwargs)
        step = self.server.script.pop(0) if self.server.script else self.server.default
        if isinstance(step, BaseException):
            raise step
        stream = FakeStream(step) if isinstance(step, list) else step
        self.server.streams.append(stream)
        return stream

    async def delete(self, response_id: str) -> Any:
        self.server.deleted_responses.append(response_id)
        return FakeRaw(httpx.Response(200, json={"id": response_id, "object": "response", "deleted": True}))


class FakeRaw:
    def __init__(self, response: httpx.Response) -> None:
        self.http_response = response


class FakeFiles:
    def __init__(self, server: FakeServer) -> None:
        self.server = server

    async def create(self, **kwargs: Any) -> Any:
        self.server.uploads.append(kwargs)
        if self.server.upload_errors:
            raise self.server.upload_errors.pop(0)
        return FakeFile(f"file-{len(self.server.uploads):04d}", expires_at=1793064343)

    async def delete(self, file_id: str) -> Any:
        self.server.deleted_files.append(file_id)
        return FakeFile(file_id, deleted=True)


class FakeFile:
    def __init__(self, file_id: str, **fields: Any) -> None:
        self.id = file_id
        self.bytes = 123
        self.expires_at = fields.get("expires_at")
        self.deleted = fields.get("deleted")

    def model_dump(self) -> dict[str, Any]:
        return {"id": self.id, "object": "file" if self.deleted is None else "file.deleted", "deleted": self.deleted, "expires_at": self.expires_at}


class FakeServer:
    """What the fake client answers: a script of streams or exceptions, then the default stream."""

    def __init__(self, default: list[dict[str, Any]] | FakeStream | None = None) -> None:
        self.default = default if default is not None else events_of(TEXT_FIXTURE)
        self.script: list[Any] = []
        self.requests: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.uploads: list[dict[str, Any]] = []
        self.upload_errors: list[BaseException] = []
        self.deleted_files: list[str] = []
        self.deleted_responses: list[str] = []


class FakeClient:
    def __init__(self, server: FakeServer) -> None:
        self.responses = FakeResponses(server)
        self.files = FakeFiles(server)

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def http_error(cls: type, status: int, message: str, *, headers: dict[str, str] | None = None, body: Any = None) -> Any:
    request = httpx.Request("POST", "https://api.meta.ai/v1/responses")
    return cls(message, response=httpx.Response(status, headers=headers or {}, request=request), body=body)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Runs under tmp_path, a fake key, a tmp ledger, no real sleeping."""
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "runs" / "ledger.jsonl")
    monkeypatch.setenv("MODEL_API_KEY", FAKE_KEY)
    return tmp_path


@pytest.fixture
def server(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    fake = FakeServer()
    monkeypatch.setattr(sam_client, "client", lambda: FakeClient(fake))
    return fake


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(sam_client, "_sleep", fake_sleep)
    return waited


async def test_segment_video_writes_lane_index_summary_and_ledger(server: FakeServer, sandbox: Path) -> None:
    seen: list[dict[str, Any]] = []
    record, frames = await sam_client.segment_video(
        "file-abc",
        "ping pong ball",
        run_id="r1",
        role="pickup",
        lane="ping pong ball",
        frames_sent=141,
        media_size=SIZE,
        metadata=sam_client.sam_metadata(),
        noun=None,
        on_event=seen.append,
    )
    request = server.requests[0]
    assert request["model"] == SAM_MODEL and request["stream"] is True and request["store"] is False
    assert request["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit"}
    assert request["input"] == [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ping pong ball"}, {"type": "input_video", "file_id": "file-abc"}]}]
    assert server.streams[0].closed
    assert len(seen) == 142 and seen[-1]["type"] == "response.completed"
    assert record.lane_id == "ping_pong_ball" and record.status is LaneStatus.COMPLETED and record.kind == "text"
    assert record.frames_sent == 141 and record.frames_returned == 134 and record.video_frames_processed == 141
    assert record.ids == ["0"] and record.zero_matches is False and record.retries == 0 and record.error is None
    assert record.store is False and record.store_echo is True and record.request_id == "resp_6adfef8768a71da59fd247f8"
    assert record.first_delta_s is not None and record.completed_s is not None and record.parse_cpu_s >= 0
    assert record.dollars == pytest.approx(141 * 0.0002)
    assert record.params == {"include_confidence": "true", "mask_encoding": "one_bit"}
    run_dir = paths.run_dir("r1")
    gz = run_dir / record.lane_path
    assert record.lane_path == "sam/pickup/ping_pong_ball.ndjson.gz" and gz.is_file()
    assert not (run_dir / "sam" / "pickup" / "ping_pong_ball.ndjson").exists()
    assert record.lane_sha256 == contracts.hashlib.sha256(gz.read_bytes()).hexdigest()
    replayed, outcome = sam_wire.index_from_events(sam_wire.read_events(gz), media="video")
    assert outcome.status is LaneStatus.COMPLETED
    # The returned frames are what the index file holds (c rounded to 6 places); the raw lane replays to the same.
    assert contracts.canonical_json(replayed) == contracts.canonical_json(frames)
    index = contracts.read_json_list(run_dir / record.index_path, IndexFrame)
    assert index == frames and len(frames) == 134
    assert frames[0].objects[0].c == 0.729958 and replayed[0].objects[0].c == 0.7299578189849854
    summary = contracts.read_json(paths.lane_summary_path("r1", "pickup", "ping_pong_ball"), contracts.LaneRecord)
    assert summary == record
    rows = ledger.rows(ledger.LEDGER_PATH)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "sam_video" and row["model"] == SAM_MODEL and row["note"] == "ping_pong_ball" and row["run_id"] == "r1"
    assert row["frames"] == 141 and row["video_frames_processed"] == 141 and row["frames_sent"] == 141
    assert row["usd"] == pytest.approx(141 * 0.0002) and row["unit"] == "frames" and row["quantity"] == 141
    assert row["status"] == "completed" and row["retries"] == 0 and row["request_id"] == record.request_id
    assert row["latency_ms"]["done"] >= 0 and row["latency_ms"]["first_delta"] >= 0 and row["latency_ms"]["upload"] is None
    assert contracts.LedgerRow.model_validate(row).engine == "meta-model-api"
    # The replayed lane rebuilds the same parser-backed run too.
    run = sam_client.replay(gz, media="video", media_size=SIZE, concept="ping pong ball")
    assert run.object_ids == ["0"] and len(run.frames) == 134 and run.frames_billed == 141 and run.outcome == "completed"
    det = run.frames[0][0]
    assert (det.box.left, det.box.top, det.box.right, det.box.bottom) == (593, 538, 621, 570)
    assert sam_client.mask_array(det).shape == (32, 28)
    full = sam_client.full_mask(det, SIZE)
    assert full.shape == (720, 1280) and full.sum() == sam_client.mask_array(det).sum() and full[538:570, 593:621].sum() == full.sum()


async def test_segment_video_bills_frames_sent_when_usage_is_missing(server: FakeServer) -> None:
    events = events_of(TEXT_FIXTURE)
    for e in events:
        if e["type"] == "response.completed":
            e["response"]["usage"] = None
    server.default = events
    record, _ = await sam_client.segment_video("file-abc", "car door", run_id="r2", role="pickup", lane="car door", frames_sent=30, media_size=SIZE, metadata={}, noun=Noun.CAR_DOOR)
    assert record.video_frames_processed is None and record.noun is Noun.CAR_DOOR
    row = ledger.rows(ledger.LEDGER_PATH)[0]
    assert row["frames"] == 30 and row["usd"] == pytest.approx(50 * 0.0002)  # the API minimum of 50 frames
    assert "metadata" not in server.requests[0]


async def test_zero_match_lane_records_zero_matches(server: FakeServer) -> None:
    server.default = events_of(ZERO_FIXTURE)
    record, frames = await sam_client.segment_video("file-abc", "face", run_id="r3", role="pickup", lane="face", frames_sent=141, media_size=SIZE, metadata=sam_client.sam_metadata(), noun=Noun.FACE)
    assert frames == [] and record.status is LaneStatus.ZERO_MATCHES and record.zero_matches is True
    assert record.frames_returned == 0 and record.ids == [] and record.video_frames_processed == 141 and record.store_echo is False
    assert record.lane_path and record.index_path and record.dollars == pytest.approx(141 * 0.0002)
    assert contracts.read_json_list(paths.run_dir("r3") / record.index_path, IndexFrame) == []
    saved = list(sam_wire.read_events(paths.run_dir("r3") / record.lane_path))
    assert [e["type"] for e in saved].count("response.output_text.done") == 1  # the shim synthesized it once


async def test_rate_limit_backs_off_six_times_then_fails(server: FakeServer, sleeps: list[float]) -> None:
    server.script = [http_error(RateLimitError, 429, "slow down", headers={"retry-after": "1"}) for _ in range(6)]
    record, frames = await sam_client.segment_video("file-abc", "wheel", run_id="r4", role="return", lane="wheel", frames_sent=600, media_size=SIZE, metadata=sam_client.sam_metadata(), noun=Noun.WHEEL)
    assert frames == [] and record.status is LaneStatus.FAILED and record.retries == 6
    assert record.error and record.error.startswith("429 after 6 tries")
    assert len(server.requests) == 6 and len(sleeps) == 5 and all(s >= 1.0 for s in sleeps)
    assert sleeps == sorted(sleeps) or max(sleeps) >= 1.0
    assert ledger.rows(ledger.LEDGER_PATH) == []  # nothing was processed, nothing is billed
    assert record.lane_path is None and not (paths.sam_dir("r4", "return") / "wheel.ndjson").exists()
    summary = contracts.read_json(paths.lane_summary_path("r4", "return", "wheel"), contracts.LaneRecord)
    assert summary.status is LaneStatus.FAILED


async def test_rate_limit_then_success_counts_retries(server: FakeServer, sleeps: list[float]) -> None:
    server.script = [http_error(RateLimitError, 429, "slow down"), http_error(RateLimitError, 429, "slow down")]
    record, frames = await sam_client.segment_video("file-abc", "hood", run_id="r5", role="pickup", lane="hood", frames_sent=141, media_size=SIZE, metadata=sam_client.sam_metadata(), noun=Noun.HOOD)
    assert record.status is LaneStatus.COMPLETED and record.retries == 2 and len(frames) == 134
    assert len(sleeps) == 2 and 0.25 <= sleeps[0] <= 0.75 and 0.5 <= sleeps[1] <= 1.5
    assert ledger.rows(ledger.LEDGER_PATH)[0]["retries"] == 2


async def test_store_false_rejected_is_retried_without_the_field(server: FakeServer) -> None:
    server.script = [http_error(BadRequestError, 400, "Unknown parameter: 'store'", body={"error": {"message": "Unknown parameter: 'store'", "param": "store"}})]
    record, frames = await sam_client.segment_video("file-abc", "roof", run_id="r6", role="pickup", lane="roof", frames_sent=141, media_size=SIZE, metadata=sam_client.sam_metadata(), noun=Noun.ROOF)
    assert record.status is LaneStatus.COMPLETED and len(frames) == 134 and record.retries == 1
    assert server.requests[0]["store"] is False and "store" not in server.requests[1]
    assert record.store is True and record.store_echo is True
    assert [d.code for d in record.diagnostics] == ["store_false_rejected"]


async def test_other_400_fails_without_retry(server: FakeServer) -> None:
    server.script = [http_error(BadRequestError, 400, "Invalid prompt", body={"error": {"message": "Invalid prompt", "param": "input"}})]
    record, frames = await sam_client.segment_video("file-abc", "roof", run_id="r6b", role="pickup", lane="roof", frames_sent=141, media_size=SIZE, metadata={}, noun=Noun.ROOF)
    assert record.status is LaneStatus.FAILED and frames == [] and len(server.requests) == 1 and record.error.startswith("400")


async def test_broken_stream_retries_once_then_incomplete(server: FakeServer) -> None:
    events = events_of(TEXT_FIXTURE)
    server.script = [FakeStream(events, stop_after=40), FakeStream(events, stop_after=60)]
    record, frames = await sam_client.segment_video("file-abc", "fender", run_id="r7", role="pickup", lane="fender", frames_sent=141, media_size=SIZE, metadata=sam_client.sam_metadata(), noun=Noun.FENDER)
    assert record.status is LaneStatus.INCOMPLETE and record.retries == 1 and len(server.requests) == 2
    assert len(frames) == 56 and record.frames_returned == 56 and record.video_frames_processed is None
    assert record.error is None and record.lane_path is not None
    saved = list(sam_wire.read_events(paths.run_dir("r7") / record.lane_path))
    assert len(saved) == 60  # the lane file holds only the attempt that was kept
    row = ledger.rows(ledger.LEDGER_PATH)[0]
    assert row["status"] == "incomplete" and row["frames"] == 141  # billed from frames_sent when usage never came


async def test_connection_error_mid_stream_retries_once(server: FakeServer) -> None:
    class BrokenStream(FakeStream):
        async def _gen(self) -> Any:
            for i, event in enumerate(self._events):
                if i == 20:
                    raise APIConnectionError(request=httpx.Request("POST", "https://api.meta.ai/v1/responses"))
                yield event

    server.script = [BrokenStream(events_of(TEXT_FIXTURE))]
    record, frames = await sam_client.segment_video("file-abc", "car", run_id="r8", role="pickup", lane="car", frames_sent=141, media_size=SIZE, metadata=sam_client.sam_metadata(), noun=Noun.CAR)
    assert record.status is LaneStatus.COMPLETED and record.retries == 1 and len(frames) == 134 and len(server.requests) == 2


async def test_segment_video_many_runs_every_lane_and_never_raises(server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    box_events = events_of(BOX_FIXTURE)
    server.script = [events_of(TEXT_FIXTURE), http_error(BadRequestError, 400, "bad", body=None), events_of(ZERO_FIXTURE), box_events]
    seen: list[tuple[str, str]] = []
    prompt = sam_client.object_prompt([sam_client.PromptObject(id=1, prompts=[sam_client.FramePrompt(frame=0, box=sam_client.prompt_box([593, 538, 621, 570], SIZE))])], SIZE)
    results = await sam_client.segment_video_many(
        "file-abc",
        [("car", "car"), ("license plate", "license plate"), ("face", "face"), ("mark_1", prompt)],
        run_id="many",
        role="pickup",
        frames_sent=141,
        media_size=SIZE,
        metadata=sam_client.sam_metadata(),
        concurrency=2,
        on_event=lambda lane, event: seen.append((lane, event["type"])),
    )
    assert list(results) == ["car", "license plate", "face", "mark_1"]
    car, plate, face, mark = (results[k][0] for k in results)
    assert car.status is LaneStatus.COMPLETED and car.noun is Noun.CAR and car.request_index == 0
    assert plate.status is LaneStatus.FAILED and plate.noun is Noun.LICENSE_PLATE and plate.request_index == 1
    assert face.status is LaneStatus.ZERO_MATCHES and face.noun is Noun.FACE
    assert mark.status is LaneStatus.COMPLETED and mark.kind == "object" and mark.noun is None and mark.request_index == 3
    assert {lane for lane, _ in seen} == {"car", "face", "mark_1"}
    assert len(ledger.rows(ledger.LEDGER_PATH)) == 3


async def test_segment_video_many_turns_a_lane_bug_into_failed(server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bug in one lane")

    monkeypatch.setattr(sam_client, "segment_video", boom)
    results = await sam_client.segment_video_many("file-abc", [("car", "car")], run_id="bug", role="pickup", frames_sent=10, media_size=SIZE, metadata={})
    record, frames = results["car"]
    assert record.status is LaneStatus.FAILED and "bug in one lane" in record.error and frames == []


async def test_segment_clip_sends_a_data_url_and_checks_the_size(server: FakeServer, tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=0.5", "-pix_fmt", "yuv420p", str(clip)], check=True)
    server.default = events_of(BOX_FIXTURE)
    prompt = "<0f>1<|box;x1=10;y1=10;x2=20;y2=20;w=64;h=48|>"
    record, frames = await sam_client.segment_clip(clip, prompt, run_id="clip", role="pickup", lane="mark_7", media_size=(64, 48), metadata=sam_client.sam_metadata(mask_encoding=None, propagation_direction="both"), frames_sent=5)
    media = server.requests[0]["input"][0]["content"][1]
    assert media["type"] == "input_video" and media["video_url"].startswith("data:video/mp4;base64,")
    assert server.requests[0]["metadata"] == {"include_confidence": "true", "propagation_direction": "both"}
    assert record.kind == "object" and record.noun is None and record.status is LaneStatus.COMPLETED and len(frames) == 134
    with pytest.raises(ValueError, match="after rotation"):
        await sam_client.segment_clip(clip, prompt, run_id="clip", role="pickup", lane="mark_8", media_size=(48, 64), metadata={}, frames_sent=5)
    big = tmp_path / "big.mp4"
    big.write_bytes(b"\0" * (sam_client.INLINE_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="capped"):
        await sam_client.segment_clip(big, prompt, run_id="clip", role="pickup", lane="mark_9", media_size=(64, 48), metadata={}, frames_sent=5)


def image_case_events() -> list[dict[str, Any]]:
    case = json.loads((ROOT / "sam_conformance" / "cases" / "captured-api-image-lossless.json").read_text())
    return case_events(case)


async def test_segment_image_parses_and_saves_events(server: FakeServer) -> None:
    server.default = image_case_events()
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    run = await sam_client.segment_image(image, "license plate", run_id="img", role="pickup", tag="kf1")
    request = server.requests[0]
    assert request["store"] is False and request["metadata"] == {"include_confidence": "true"}
    assert request["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
    assert run.media == "image" and (run.media_width, run.media_height) == (60, 40) and run.outcome == "completed"
    assert run.frames_billed == 0 and run.usd == pytest.approx(0.0025) and run.response_id == "resp_case"
    dets = run.frames[0]
    assert len(dets) == 1 and dets[0].frame_index is None and dets[0].mask is not None and dets[0].mask.encoding == "lossless"
    raster = sam_client.mask_array(dets[0])
    assert raster.shape == (dets[0].mask.height, dets[0].mask.width)
    assert run.events_path is not None and run.events_path.name == "kf1.resp_case.ndjson" and run.events_path.is_file()
    assert run.events_path.parent == paths.image_requests_dir("img", "pickup")
    row = ledger.rows(ledger.LEDGER_PATH)[0]
    assert row["kind"] == "sam_image" and row["images"] == 1 and row["usd"] == pytest.approx(0.0025) and row["note"] == "license plate kf1"
    assert row["unit"] == "images" and row["quantity"] == 1 and row["status"] == "completed"


async def test_segment_image_without_run_keeps_no_file_and_raises_on_no_answer(server: FakeServer, sleeps: list[float], tmp_path: Path) -> None:
    from PIL import Image

    png = tmp_path / "frame.png"
    Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8)).save(png)
    server.default = image_case_events()
    run = await sam_client.segment_image(png, "face", run_id=None)
    assert run.events_path is None and (run.media_width, run.media_height) == (20, 10)
    assert server.requests[-1]["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    server.script = [http_error(RateLimitError, 429, "slow down") for _ in range(6)]
    with pytest.raises(sam_client.SamRequestError):
        await sam_client.segment_image(png, "face", run_id=None)
    assert len(sleeps) == 5


async def test_upload_file_caches_by_sha256_and_sets_expiry(server: FakeServer, tmp_path: Path) -> None:
    video = tmp_path / "proxy.mp4"
    video.write_bytes(b"not really mp4 but bytes")
    first = await sam_client.upload_file(video, run_id="r1")
    assert first.file_id == "file-0001" and first.uploaded_at.endswith("Z")
    assert first.expires_at == "2026-10-27T01:25:43Z"  # the API's expires_at (unix 1793064343)
    upload = server.uploads[0]
    assert upload["purpose"] == "user_data" and upload["expires_after"] == {"anchor": "created_at", "seconds": 3600}
    assert upload["file"] == ("proxy.mp4", video.read_bytes(), "video/mp4")
    cache = json.loads(paths.uploads_cache_path().read_text())
    digest = contracts.hashlib.sha256(video.read_bytes()).hexdigest()
    assert cache[digest]["file_id"] == "file-0001" and cache[digest]["run_id"] == "r1" and cache[digest]["seconds"] >= 0
    # Fresh entries are reused; stale ones are not.
    cache[digest]["expires_at"] = "2099-01-01T00:00:00Z"
    paths.uploads_cache_path().write_text(json.dumps(cache))
    again = await sam_client.upload_file(video)
    assert again.file_id == "file-0001" and len(server.uploads) == 1
    cache[digest]["expires_at"] = "2020-01-01T00:00:00Z"
    paths.uploads_cache_path().write_text(json.dumps(cache))
    third = await sam_client.upload_file(video)
    assert third.file_id == "file-0002" and len(server.uploads) == 2
    body = await sam_client.delete_file("file-0002")
    assert body["deleted"] is True and body["id"] == "file-0002" and server.deleted_files == ["file-0002"]
    assert digest not in json.loads(paths.uploads_cache_path().read_text())


async def test_upload_file_remuxes_on_a_4xx(server: FakeServer, tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=0.3", "-pix_fmt", "yuv420p", str(clip)], check=True)
    server.upload_errors = [http_error(BadRequestError, 400, "unsupported container", body=None)]
    record = await sam_client.upload_file(clip)
    assert record.file_id == "file-0002" and len(server.uploads) == 2
    assert server.uploads[1]["file"][0] == "clip.remux.mp4" and (tmp_path / "clip.remux.mp4").is_file()


async def test_delete_response_returns_status_and_body(server: FakeServer) -> None:
    body = await sam_client.delete_response("resp_1")
    assert body == {"id": "resp_1", "status_code": 200, "deleted": True, "body": {"id": "resp_1", "object": "response", "deleted": True}}
    assert server.deleted_responses == ["resp_1"]


def test_sam_metadata_shapes() -> None:
    assert sam_client.sam_metadata() == {"include_confidence": "true", "mask_encoding": "one_bit"}
    assert sam_client.sam_metadata(mask_encoding=None, propagation_direction="both") == {"include_confidence": "true", "propagation_direction": "both"}
    assert sam_client.sam_metadata(include_confidence=False, mask_encoding="lossless") == {"mask_encoding": "lossless"}


def test_prompt_box_pads_clamps_and_builds() -> None:
    box = sam_client.prompt_box([100.4, 50.6, 120.2, 80.9], SIZE)
    assert (box.left, box.top, box.right, box.bottom) == (98, 48, 123, 83)
    collapsed = sam_client.prompt_box([10.0, 10.0, 10.0, 10.0], SIZE, pad_px=0)
    assert collapsed.right - collapsed.left >= 1 and collapsed.bottom - collapsed.top >= 1
    one_px = sam_client.prompt_box([10.0, 10.0, 11.0, 11.0], SIZE)
    assert (one_px.left, one_px.top, one_px.right, one_px.bottom) == (8, 8, 13, 13)
    edge = sam_client.prompt_box([1278.0, 718.0, 1280.0, 720.0], SIZE)
    assert (edge.left, edge.top, edge.right, edge.bottom) == (1276, 716, 1280, 720)
    corner = sam_client.prompt_box([0.0, 0.0, 1.0, 1.0], SIZE)
    assert (corner.left, corner.top) == (0, 0) and corner.right == 3 and corner.bottom == 3
    portrait = sam_client.prompt_box([500.0, 900.0, 540.0, 960.0], (540, 960))
    text = sam_client.object_prompt([sam_client.PromptObject(id=3, prompts=[sam_client.FramePrompt(frame=45, box=portrait)])], (540, 960))
    assert text == "<45f>3<|box;x1=498;y1=898;x2=539;y2=959;w=540;h=960|>"
    reprompt = sam_client.object_prompt([sam_client.PromptObject(id=1, prompts=[sam_client.FramePrompt(frame=0, box=one_px), sam_client.FramePrompt(frame=30, box=one_px, points=[sam_client.PromptPoint(5, 5, "negative")])])], SIZE)
    assert reprompt.startswith("<0f>1<|box;") and "<30f>1<|box;" in reprompt and "-<|point;x=5;y=5;w=1280;h=720|>" in reprompt
    with pytest.raises(sam_client.ObjectPromptError) as info:
        sam_client.object_prompt([sam_client.PromptObject(id=1, prompts=[sam_client.FramePrompt(box=one_px)]), sam_client.PromptObject(id=1, prompts=[sam_client.FramePrompt(box=one_px)])], SIZE)
    assert info.value.code == "duplicate_object_id"


def test_object_prompt_accepts_numpy_sizes_and_rejects_fractions() -> None:
    box = sam_client.prompt_box([10.0, 10.0, 20.0, 20.0], SIZE)
    objects = [sam_client.PromptObject(id=1, prompts=[sam_client.FramePrompt(box=box)])]
    text = sam_client.object_prompt(objects, (np.int64(1280), np.int32(720)))
    assert text == "<0f>1<|box;x1=8;y1=8;x2=21;y2=21;w=1280;h=720|>"
    with pytest.raises(sam_client.ObjectPromptError) as info:
        sam_client.object_prompt(objects, (1280.5, 720))
    assert info.value.code == "invalid_size"


def test_store_rejection_is_detected_by_param_or_whole_word() -> None:
    assert sam_client._mentions_store(http_error(BadRequestError, 400, "bad", body={"error": {"message": "bad", "param": "store"}}))
    assert sam_client._mentions_store(http_error(BadRequestError, 400, "Unknown parameter: 'store'", body=None))
    assert not sam_client._mentions_store(http_error(BadRequestError, 400, "could not restore the stored file", body=None))


def test_zero_match_without_any_output_item_is_still_zero_matches() -> None:
    events = [e for e in events_of(ZERO_FIXTURE) if not e["type"].startswith(("response.output_item", "response.content_part"))]
    frames, outcome = sam_wire.index_from_events(events, media="video")
    assert frames == [] and outcome.status is LaneStatus.ZERO_MATCHES and outcome.diagnostics == []


def test_backoff_honors_retry_after() -> None:
    assert sam_client._backoff_delay(1, "3") >= 3.0
    assert 0.25 <= sam_client._backoff_delay(1, None) <= 0.75
    assert sam_client._backoff_delay(6, None) <= sam_client.BACKOFF_CAP_S * 1.5


def test_stub_structures_still_work() -> None:
    box = sam_client.Box(1, 2, 4, 6)
    assert (box.width, box.height, box.area) == (3, 4, 12)
    run = sam_client.SegmentationRun(concept="x", media="video", media_width=10, media_height=10)
    det = sam_client.Detection("10", 0, box, None, None)
    run.frames[0] = [det, sam_client.Detection("2", 0, box, None, 0.5)]
    assert run.object_ids == ["2", "10"] and run.track("2")[0].confidence == 0.5
    with pytest.raises(ValueError):
        sam_client.mask_array(det)


# ----------------------------------------------------------------------------------------------------------------
# Live smoke (LIVE=1 only; about $0.06: two 141-frame video requests plus the upload and delete)
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="set LIVE=1 to spend about $0.06 on the real API")
async def test_live_smoke_text_and_box_lanes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Lane files go to tmp; the ledger rows go to the real runs/ledger.jsonl so the spend is on record.
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    uploaded = await sam_client.upload_file(PINGPONG, run_id="smoke")
    assert uploaded.file_id.startswith("file-") and uploaded.expires_at is not None
    try:
        text_record, text_frames = await sam_client.segment_video(uploaded.file_id, "ping pong ball", run_id="smoke", role="pickup", lane="ping pong ball", frames_sent=141, media_size=SIZE, metadata=sam_client.sam_metadata())
        assert text_record.status is LaneStatus.COMPLETED, text_record
        assert text_record.frames_returned >= 100 and text_record.video_frames_processed == 141 and text_record.ids
        first = text_frames[0].objects[0]
        prompt = sam_client.object_prompt([sam_client.PromptObject(id=1, prompts=[sam_client.FramePrompt(frame=text_frames[0].frame, box=sam_client.prompt_box(first.box, SIZE))])], SIZE)
        box_record, box_frames = await sam_client.segment_video(uploaded.file_id, prompt, run_id="smoke", role="pickup", lane="box_1", frames_sent=141, media_size=SIZE, metadata=sam_client.sam_metadata(mask_encoding=None, propagation_direction="both"), kind="object")
        assert box_record.status is LaneStatus.COMPLETED, box_record
        assert box_record.frames_returned >= 100
        cs = {o.c for f in box_frames for o in f.objects}
        assert None not in cs and len(cs) > 1, "box prompts must return per-frame confidence"
        print(
            f"\nLIVE smoke: text first_delta {text_record.first_delta_s}s done {text_record.completed_s}s frames {text_record.frames_returned}"
            f" store_echo {text_record.store_echo} store_sent_false {not text_record.store} parse_cpu {text_record.parse_cpu_s}s;"
            f" box first_delta {box_record.first_delta_s}s done {box_record.completed_s}s frames {box_record.frames_returned} distinct c {len(cs)};"
            f" dollars {text_record.dollars + box_record.dollars:.4f}"
        )
    finally:
        body = await sam_client.delete_file(uploaded.file_id)
        print(f"LIVE smoke: delete body {body}")
        assert body.get("deleted") is True
