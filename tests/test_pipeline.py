"""walkaround.pipeline end to end on the test clip with only the network replaced: the real video, sam_client,
sam_wire, panels, privacy, marks, geometry, match, seal, sheet and orchestrator code. SAM answers with the saved 10/26
ping pong stream (license plate and face with the zero-match stream), the Files API with a fake id, Spark with the
saved marks_v1 answer. No key, no socket, no subprocess beyond ffmpeg."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from walkaround import config, contracts as c, ledger, marks, paths, pipeline, sam_client, seal, spark_client
from walkaround.orchestrator import REUSED_CODE, Orchestrator

ROOT = Path(__file__).resolve().parent
CLIP = ROOT.parent / "data" / "test" / "pingpong.mp4"
TEXT = [json.loads(line) for line in (ROOT / "fixtures/sam/pingpong_text.events.ndjson").read_text().splitlines() if line.strip()]
ZERO = [json.loads(line) for line in (ROOT / "fixtures/sam_zero_match.events.ndjson").read_text().splitlines() if line.strip()]
SPARK = json.loads((ROOT / "fixtures/spark/marks_v1_live_2026-10-27.json").read_text())
PRIVACY_PROMPTS = ("license plate", "face")


class FakeStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self) -> Any:
        return self._gen()

    async def _gen(self) -> Any:
        for event in self.events:
            yield SimpleNamespace(model_dump=lambda event=event: copy.deepcopy(event))


def stored_false(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = copy.deepcopy(events)
    for event in out:
        if isinstance(event.get("response"), dict):
            event["response"]["store"] = False  # what an accepted store false echoes
    return out


class FakeApi:
    def __init__(self, calls: dict[str, list[str]]) -> None:
        self.calls = calls
        self.responses = SimpleNamespace(create=self.create)
        self.files = SimpleNamespace(create=self.files_create, delete=self.files_delete)

    async def __aenter__(self) -> FakeApi:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def create(self, **body: Any) -> FakeStream:
        assert body["store"] is False and body["stream"] is True
        text, media = body["input"][0]["content"][0]["text"], body["input"][0]["content"][1]["type"]
        if media == "input_video":
            self.calls["video"].append(text)
            return FakeStream(stored_false(ZERO if text in PRIVACY_PROMPTS else TEXT))
        self.calls["image"].append(text)
        return FakeStream(stored_false(ZERO))

    async def files_create(self, **kwargs: Any) -> Any:
        self.calls["files"].append("create")
        now = int(time.time())
        return SimpleNamespace(id=f"file-pipeline{len(self.calls['files'])}", created_at=now, expires_at=now + 3600)

    async def files_delete(self, file_id: str) -> Any:
        self.calls["files"].append("delete")
        return SimpleNamespace(model_dump=lambda: {"id": file_id, "object": "file", "deleted": True})


class FakeSpark:
    def __init__(self, calls: dict[str, list[str]]) -> None:
        self.calls = calls
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **request: Any) -> Any:
        self.calls["spark"].append(request["model"])
        answer = copy.deepcopy(SPARK)
        answer["id"] = f"resp_spark_{len(self.calls['spark'])}"
        text = next(p["text"] for item in answer["output"] if item.get("type") == "message" for p in item["content"] if p.get("type") == "output_text")
        return SimpleNamespace(model_dump=lambda: copy.deepcopy(answer), output_text=text)


@pytest.fixture
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    monkeypatch.setenv("MODEL_API_KEY", "LLM|test-team|test-secret")
    monkeypatch.setenv("APP_COMMIT", "test")
    calls: dict[str, list[str]] = {"video": [], "image": [], "files": [], "spark": []}
    monkeypatch.setattr(sam_client, "client", lambda: FakeApi(calls))
    spark_client.set_client(FakeSpark(calls))

    def offline_stamp(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no timestamping in an offline test")

    monkeypatch.setattr(seal, "stamp_ots", offline_stamp)
    monkeypatch.setattr(seal, "stamp_tsa", offline_stamp)
    th = c.load_thresholds()
    th["media"]["proxy_max_edge"] = 1280  # the fixture stream's geometry: 1280x720 at 24 fps
    th["media"]["proxy_fps"] = 24
    th["panels"]["area_floor_default"] = 0.0001  # a ping pong ball is small for a car panel
    th["panels"]["area_floor_small"] = 0.0001
    th["coverage"]["filmed_area_frac_min"] = 0.0005  # and filmed at that size
    th["matching"]["sharpness_rel_min"] = 0.0  # G3 is tested in test_match; here any frame of the clip may pair
    th["panels"]["keyframe_top_k"] = 1  # one keyframe per walk: a near-round ball turns its axis between frames (G4)
    th["spark"]["second_look_engine"] = "off"  # FakeSpark answers marks_v1 only; the look's wiring is tested in test_orchestrator
    orch = Orchestrator(
        th=th, engine=c.SamEngine.META_MODEL_API, max_concurrent_runs=1, media_workers=1, decode_workers=1,
        overrides={"seal.seal_run": functools.partial(seal.seal_run, stamp=False)},
    )
    yield {"orch": orch, "calls": calls, "root": root, "th": th}
    orch.close()
    spark_client.set_client(None)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stand_in_prompts_map_nouns_onto_single_slot_panels() -> None:
    lanes, prompts = pipeline.stand_in_prompts(["ping pong ball", "person"])
    assert prompts == {"hood": "ping pong ball", "trunk lid": "person", "car": ""}  # no car request: the whole frame
    assert lanes == [c.Noun.CAR, c.Noun.LICENSE_PLATE, c.Noun.FACE, c.Noun.HOOD, c.Noun.TRUNK_LID]
    assert pipeline.stand_in_prompts(["dog"], container="room")[1] == {"hood": "dog", "car": "room"}
    with pytest.raises(ValueError):
        pipeline.stand_in_prompts([])
    with pytest.raises(ValueError):
        pipeline.stand_in_prompts(["a", "b", "c", "d", "e"])


async def test_run_pair_self_pair_end_to_end_then_rebuild_and_verify(offline: dict[str, Any]) -> None:
    orch, calls, th = offline["orch"], offline["calls"], offline["th"]
    seen: list[c.Event] = []
    final = await pipeline.run_pair_async(CLIP, CLIP, "selfpair1", ["ping pong ball"], orch=orch, on_event=seen.append)
    run_id = final.run_id
    assert run_id == "selfpair1" and final.kind == c.RunKind.PAIR
    assert final.prompts == {"car": "", "hood": "ping pong ball"}
    assert final.noun_list == [c.Noun.CAR, c.Noun.LICENSE_PLATE, c.Noun.FACE, c.Noun.HOOD]
    assert final.status == c.RunStatus.DONE and final.errors == [], final.errors
    assert seen[0].kind == c.EventKind.RUN_CREATED and seen[-1].kind == c.EventKind.RUN_COMPLETED

    # only the pickup asked anything: 3 lanes (no car request), two image checks per distinct keyframe frame, one Spark
    # call per distinct crop, one Stays Put clip with its two prompt-frame checks (the return walk reuses all of it)
    clips = [t for t in calls["video"] if t.startswith("<")]
    assert sorted(t for t in calls["video"] if not t.startswith("<")) == sorted(["ping pong ball", "license plate", "face"])
    assert len(clips) == 1
    car = c.read_json(paths.lane_summary_path(run_id, "pickup", c.Noun.CAR), c.LaneRecord)
    assert car.request_id is None and car.prompt_text == "" and car.diagnostics[0].code == "whole_frame_container"
    kfs = c.read_json_list(paths.keyframes_json(run_id), c.Keyframe)
    pickup_kfs = [k for k in kfs if k.role == "pickup"]
    return_kfs = [k for k in kfs if k.role == "return"]
    assert pickup_kfs and len(return_kfs) == len(pickup_kfs)
    boxes = [t for t in calls["image"] if t.startswith("<0f>")]  # dedup.sam_box_refine: one request per pickup crop with box marks
    pickup_marks = [m for m in c.read_json_list(paths.marks_json(run_id), c.Mark) if m.role == "pickup" and m.source == c.MarkSource.SPARK_BOX]
    assert len(boxes) == len({m.kf_id for m in pickup_marks}) and all(m.box_refine is not None for m in pickup_marks)
    assert len(calls["image"]) - len(boxes) == 2 * len({k.f_proxy for k in pickup_kfs}) + 2 * len(clips)
    assert len(calls["spark"]) == len({k.crop.sha256 for k in pickup_kfs if k.crop})
    for kf in kfs:
        assert kf.crop is not None and (paths.run_dir(run_id) / kf.crop.path).is_file()
        assert kf.marks_status == c.MarksStatus.COMPLETE
    assert [k.crop.sha256 for k in pickup_kfs] == [k.crop.sha256 for k in return_kfs]  # same bytes, same crops
    for noun in final.noun_list:
        lane = c.read_json(paths.lane_summary_path(run_id, "return", noun), c.LaneRecord)
        if noun is c.Noun.CAR:
            assert lane.diagnostics[0].code == "whole_frame_container"
            continue
        assert lane.role == "return" and any(d.code == REUSED_CODE for d in lane.diagnostics) and lane.dollars == 0.0
        assert (paths.run_dir(run_id) / lane.lane_path).is_file()
    assert {r["role"] for r in ledger.rows() if r["kind"] == "sam_video"} == {"pickup"}
    assert calls["files"] == ["create", "delete"] and orch.load_video(run_id, "return").files_api is None  # one upload

    # the match: every mark of the self-pair matches itself at distance 0
    matches = c.read_json_list(paths.matches_json(run_id), c.PanelMatch)
    stored_marks = c.read_json_list(paths.marks_json(run_id), c.Mark)
    compared = [pm for pm in matches if pm.state == c.MatchState.COMPARABLE]
    assert compared, [(pm.panel_id, pm.reasons) for pm in matches]
    for pm in compared:
        on_panel = [m for m in stored_marks if m.panel_id == pm.panel_id and m.role == "pickup"]
        assert on_panel and len(pm.pairs) == len(on_panel)
        assert all(p.d == 0 and p.d_pos == 0 for p in pm.pairs)
        assert pm.unmatched_pickup == [] and pm.unmatched_return == []
    verdicts = c.read_json_list(paths.verdicts_json(run_id), c.Verdict)
    assert verdicts and all(v.distance_units in (None, 0.0) for v in verdicts)

    # what a finished run leaves behind
    for path in (paths.events_path(run_id), paths.panels_json(run_id), paths.calibration_json(run_id), paths.sheet_html(run_id), paths.deletions_json(run_id), paths.manifest_run_json(run_id)):
        assert path.is_file(), path
    assert not list(paths.run_dir(run_id).rglob("*.orig")) and not list(paths.run_dir(run_id).rglob("*.proxy.mp4"))
    sheet = c.read_json(paths.sheet_json(run_id), c.SheetModel)
    assert "stand-in prompts" in sheet.stand_in_line and "hood sent as 'ping pong ball'" in sheet.stand_in_line
    assert "car: no request, the whole frame stands in" in sheet.stand_in_line
    assert "stand-in prompts" in paths.sheet_html(run_id).read_text()

    # rebuild reproduces every derived file it regenerates, byte for byte, with no call
    before = {name: sha(paths.derived_dir(run_id) / name) for name in ("marks.json", "calibration.json", "matches.json", "verdicts.json", "coverage.json", "mark_tracks.json")}
    counts = {k: len(v) for k, v in calls.items()}
    result, out = pipeline.rebuild(run_id, orch=orch)
    assert out == paths.runs_root() / "_rebuild"  # a closed run is rebuilt beside itself, never in place
    for name, digest in before.items():
        assert result[name] == digest, name
    assert result["sheet.json"] == sha(paths.sheet_json(run_id))
    assert pipeline.verify(run_id, orch=orch) == []
    assert {k: len(v) for k, v in calls.items()} == counts  # rebuild and verify asked nothing

    # a tampered marks file is caught
    assert pipeline.verify(run_id, orch=orch) == []  # narration and the seed rows rebuild to the same bytes
    marks_path = paths.marks_json(run_id)
    rows = json.loads(marks_path.read_text())
    rows[0]["w"] = [9.0, 9.0]
    c.write_json(marks_path, rows)
    assert "rebuild differs: derived/marks.json" in pipeline.verify(run_id, orch=orch)


async def test_run_pair_refuses_a_finished_run_id_and_resumes_an_unfinished_one(offline: dict[str, Any]) -> None:
    orch, calls = offline["orch"], offline["calls"]
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False, run_id="resume1", noun_list=[c.Noun.HOOD], prompts={"hood": "ping pong ball", "car": "ping pong ball"})
    assert run.noun_list == [c.Noun.CAR, c.Noun.LICENSE_PLATE, c.Noun.FACE, c.Noun.HOOD]
    final = await pipeline.run_pair_async(CLIP, None, "resume1", orch=orch)  # nothing uploaded yet: sends the file
    assert final.status == c.RunStatus.DONE and final.kind == c.RunKind.PICKUP_CHECK
    assert sorted(t for t in calls["video"] if not t.startswith("<")) == sorted(["ping pong ball", "license plate", "face", "ping pong ball"])
    assert len([t for t in calls["video"] if t.startswith("<")]) == 1  # the Stays Put clip
    with pytest.raises(FileExistsError):
        await pipeline.run_pair_async(CLIP, None, "resume1", orch=orch)
    with pytest.raises(ValueError):
        await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False, run_id="_hidden")
    with pytest.raises(ValueError):
        await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False, prompts={"roof": "dog"}, noun_list=[c.Noun.HOOD])


def test_rebuild_marks_reads_the_answer_derive_marks_used(offline: dict[str, Any], tmp_path: Path) -> None:
    """A keyframe whose crop bytes match another's reads that answer back; one with its own record reads its own."""
    run_id = "rb1"
    paths.ensure_run_layout(run_id)
    import numpy as np

    from walkaround import video

    crop_rgb = np.full((40, 60, 3), 128, dtype=np.uint8)
    rows = []
    for role in ("pickup", "return"):
        path = paths.crop_path(run_id, role, f"{role}_hood_kf1")
        digest = video.write_jpeg(crop_rgb, path, quality=90)
        rows.append(
            c.Keyframe(
                kf_id=f"{role}_hood_kf1", role=role, panel_id=c.PanelId.HOOD, track_id=f"{role}_hood_00", f_proxy=3, pts=0.2,
                score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=1.5,
                box_src=[0, 0, 60, 40], crop=c.Crop(path=paths.rel(run_id, path), sha256=digest, w=60, h=40, scale=1.0, dx=0, dy=0),
                moments=None, front_sign=1, marks_status=c.MarksStatus.COMPLETE,
            )
        )
    c.write_json(paths.keyframes_json(run_id), rows)
    text = json.dumps({"marks": [{"type": "scratch", "point": {"x": 500, "y": 500}, "bbox": {"x_min": 400, "y_min": 400, "x_max": 600, "y_max": 600}, "size_class": "M", "confidence": "high", "note": "line"}], "conditions": {"wet": False, "dirty": False, "glare": True, "dark": False}, "landmarks": [], "neighbour_car_visible": False})
    record = c.SparkRequestRecord(
        request_id="resp_1", purpose=c.SparkPurpose.MARKS, model="muse-spark-1.3", schema_name="marks_v1", raw_text=text,
        parsed=json.loads(text), valid=True, input_refs=[c.InputRef(path=rows[0].crop.path, sha256=rows[0].crop.sha256)],
    )
    c.write_json(paths.spark_json(run_id), [record])
    th = offline["th"]
    pickup = marks.rebuild_marks(run_id, "pickup", th=th)
    back = marks.rebuild_marks(run_id, "return", th=th)
    assert [m.spark_request_id for m in pickup] == ["resp_1"] and [m.spark_request_id for m in back] == ["resp_1"]
    assert pickup[0].bbox_crop == back[0].bbox_crop and pickup[0].role == "pickup" and back[0].role == "return"
    stored = c.read_json_list(paths.marks_json(run_id), c.Mark)
    assert sorted(m.role for m in stored) == ["pickup", "return"]
    assert c.read_json_list(paths.spark_json(run_id), c.SparkRequestRecord) == [record]  # frozen facts untouched
