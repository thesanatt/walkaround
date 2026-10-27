"""FastAPI routes (M15) through Starlette's TestClient with the orchestrator on fakes: chunked uploads with the
browser hash checked on receipt, run creation against the caps, SSE progress live and resumed from Last-Event-ID,
the replay payload with its badge date, the sample pair flagged precomputed, served files, health."""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.test_orchestrator import STAMPS, Fakes, make_orch, runs_dir, seed_replay_run, th, write_run  # noqa: F401
from walkaround import contracts as c, paths
from walkaround.app import create_app
from walkaround.orchestrator import Orchestrator

TODAY = "2026-10-27T18:00:00.000Z"
CHUNK = 8 * 1024 * 1024
DELAYS = {"created": 100, "sealing": 200, "sealed": 50, "converting": 300, "tracking": 1000, "blurring": 100, "marking": 400, "staysput": 200, "matching": 100, "sheet": 100, "done": 50}


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


@pytest.fixture
def orch(runs_dir: Path, th: dict[str, Any], fakes: Fakes, tmp_path: Path) -> Orchestrator:
    delays = tmp_path / "delays.json"
    delays.write_text(json.dumps(DELAYS), encoding="utf-8")
    return make_orch(th, fakes, delays_path=delays, now=lambda: TODAY)


@pytest.fixture
def client(orch: Orchestrator) -> Any:
    with TestClient(create_app(orch, background=False)) as client:
        yield client


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create(client: TestClient, kind: str = "pickup_check", **extra: Any) -> str:
    response = client.post("/api/runs", json={"kind": kind, **extra})
    assert response.status_code == 201, response.text
    return response.json()["run_id"]


def put_chunks(client: TestClient, run_id: str, upload_id: str, data: bytes) -> int:
    count = 0
    for index, start in enumerate(range(0, max(len(data), 1), CHUNK)):
        response = client.put(f"/api/uploads/{upload_id}/chunks/{index}", params={"run_id": run_id}, content=data[start : start + CHUNK])
        assert response.status_code == 204, response.text
        count += 1
    return count


def finalize(client: TestClient, run_id: str, upload_id: str, role: str, data: bytes, *, digest: str | None = None, **extra: Any) -> httpx.Response:
    body = {"run_id": run_id, "role": role, "sha256_browser": digest or sha(data), "bytes": len(data), "filename": f"{role}.mp4", **extra}
    return client.post(f"/api/uploads/{upload_id}/finalize", json=body)


def upload(client: TestClient, run_id: str, role: str, data: bytes) -> httpx.Response:
    put_chunks(client, run_id, f"up-{role}", data)
    return finalize(client, run_id, f"up-{role}", role, data)


def wait_done(client: TestClient, run_id: str, timeout_s: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in ("done", "partial", "failed"):
            return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish: {run['status']}")


def sse_events(response: httpx.Response) -> list[dict[str, Any]]:
    """Parse text/event-stream lines into dicts of id, event and the decoded data."""
    out: list[dict[str, Any]] = []
    current: dict[str, str] = {}
    for line in response.iter_lines():
        if line == "":
            if current:
                out.append({"id": int(current["id"]), "event": current["event"], "data": json.loads(current["data"])})
                current = {}
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(":")
        current[key] = value.lstrip()
    if current:
        out.append({"id": int(current["id"]), "event": current["event"], "data": json.loads(current["data"])})
    return out


# Uploads ---------------------------------------------------------------------------------------------------------------


def test_chunked_upload_of_20mb_is_hashed_on_receipt_and_idempotent_by_index(client: TestClient, orch: Orchestrator) -> None:
    run_id = create(client, "pair")  # a pair waits for its second walk, so nothing starts here
    data = os.urandom(20 * 1024 * 1024)
    assert put_chunks(client, run_id, "u1", data) == 3
    retry = client.put("/api/uploads/u1/chunks/1", params={"run_id": run_id}, content=data[CHUNK : 2 * CHUNK])
    assert retry.status_code == 204
    response = finalize(client, run_id, "u1", "pickup", data, digest=sha(data).upper(), keep_plate=True)
    assert response.status_code == 200, response.text
    record = c.VideoRecord.model_validate_json(response.text)
    assert record.hash_match is True and record.sha256_server == sha(data) and record.chunks_received == 3 and record.bytes == len(data)
    stored = paths.quarantine_original(run_id, "pickup")
    assert stored.stat().st_size == len(data) and sha(stored.read_bytes()) == sha(data)
    assert not (paths.run_dir(run_id) / "uploads" / "u1").exists()
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "created" and run["keep_plate"] is True
    assert not client.get(f"/api/runs/{run_id}/files/quarantine/pickup.orig").is_success


def test_finalize_answers_409_with_both_hashes_and_400_for_missing_chunks(client: TestClient) -> None:
    run_id = create(client, "pair")
    data = os.urandom(3000)
    put_chunks(client, run_id, "u2", data)
    response = finalize(client, run_id, "u2", "pickup", data, digest="b" * 64)
    assert response.status_code == 409
    assert response.json() == {"sha256_browser": "b" * 64, "sha256_server": sha(data), "detail": "hash mismatch; upload again"}
    assert not paths.quarantine_original(run_id, "pickup").exists()
    client.put("/api/uploads/u3/chunks/0", params={"run_id": run_id}, content=b"only the first chunk")
    response = client.post("/api/uploads/u3/finalize", json={"run_id": run_id, "role": "return", "sha256_browser": "a" * 64, "bytes": 3 * CHUNK, "filename": "b.mp4"})
    assert response.status_code == 400 and response.json()["missing"] == [1, 2]
    assert client.post("/api/uploads/u3/finalize", json={"run_id": "NOPE", "role": "return", "sha256_browser": "a" * 64, "bytes": 1, "filename": "b.mp4"}).status_code == 404
    assert client.post("/api/uploads/u3/finalize", json={"run_id": run_id, "role": "driver", "sha256_browser": "a" * 64, "bytes": 1, "filename": "b.mp4"}).status_code == 400


def test_body_limits_413_on_the_chunk_route_and_elsewhere(client: TestClient, th: dict[str, Any]) -> None:
    run_id = create(client, "pair")
    too_big = client.put("/api/uploads/u4/chunks/0", params={"run_id": run_id}, content=b"x" * (int(th["media"]["chunk_route_max_body_bytes"]) + 1))
    assert too_big.status_code == 413
    assert client.put("/api/uploads/u4/chunks/5000", params={"run_id": run_id}, content=b"x").status_code == 400
    over_1gib = client.post("/api/uploads/u4/finalize", json={"run_id": run_id, "role": "pickup", "sha256_browser": "a" * 64, "bytes": 2 * 1024 ** 3, "filename": "huge.mov"})
    assert over_1gib.status_code == 413
    fat_json = client.post("/api/runs", content=b"{" + b" " * (2 * 1024 * 1024) + b"}", headers={"content-type": "application/json"})
    assert fat_json.status_code == 413


# Runs ------------------------------------------------------------------------------------------------------------------


def test_create_run_index_and_lookup(client: TestClient, th: dict[str, Any]) -> None:
    run_id = create(client, "pickup_check", keep_plate=True)
    assert len(run_id) == 26
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "created" and run["kind"] == "pickup_check" and run["keep_plate"] is True
    assert run["config_hash"] == c.config_hash(th) and run["thresholds"]["orchestrator"]["pickup_runs_per_day"] == th["orchestrator"]["pickup_runs_per_day"]
    index = client.get("/api/runs").json()
    assert [r["run_id"] for r in index] == [run_id] and index[0]["thresholds"] == {}
    assert client.get("/api/runs/NOPE").status_code == 404
    assert client.get("/api/runs/NOPE/events").status_code == 404
    assert client.get("/api/runs/NOPE/replay").status_code == 404


def test_create_run_429_when_the_caps_or_the_pause_apply(client: TestClient, orch: Orchestrator, th: dict[str, Any]) -> None:
    write_run("RUNPAIRX", created_at="2026-10-27T16:00:00Z", kind=c.RunKind.PAIR)
    response = client.post("/api/runs", json={"kind": "pair"})
    assert response.status_code == 429
    assert response.json() == {"reason": "no full pairs left today", "pickup_left": th["orchestrator"]["pickup_runs_per_day"], "pairs_left": 0}
    assert client.post("/api/runs", json={"kind": "pickup_check"}).status_code == 201
    status = client.get("/api/status").json()
    assert status == {"paused": False, "pickup_left": th["orchestrator"]["pickup_runs_per_day"], "pairs_left": 0, "queue_position": None, "engine": "meta-model-api"}
    orch.paused = True
    response = client.post("/api/runs", json={"kind": "pickup_check"})
    assert response.status_code == 429 and "paused" in response.json()["reason"]
    assert client.get("/api/status").json()["paused"] is True


def test_start_route_needs_every_upload_then_runs_the_pipeline(client: TestClient, orch: Orchestrator) -> None:
    run_id = create(client, "pair")
    assert upload(client, run_id, "pickup", b"pickup walk " * 400).status_code == 200
    blocked = client.post(f"/api/runs/{run_id}/start")
    assert blocked.status_code == 409 and "return" in blocked.json()["detail"]
    assert upload(client, run_id, "return", b"return walk " * 400).status_code == 200  # the last upload starts it
    assert client.post(f"/api/runs/{run_id}/start").status_code == 202  # idempotent while it runs
    run = wait_done(client, run_id)
    assert run["status"] == "done" and run["counts"]["panels"] == 0 and run["events_head_hash"]
    assert client.post(f"/api/runs/{run_id}/start").status_code == 200
    assert client.get(f"/api/runs/{run_id}/files/derived/matches.json").json() == []


# SSE ----------------------------------------------------------------------------------------------------------------------


def test_sse_streams_progress_live_and_resumes_from_last_event_id(client: TestClient, orch: Orchestrator, fakes: Fakes) -> None:
    import asyncio

    async def slow_marks(run_id: str, role: str, keyframes: list, *, th: dict, count_mode: bool = False) -> tuple[list, list, list]:
        await asyncio.sleep(0.4)  # keep the run open so the subscriber joins mid-run
        return await fakes.derive_marks(run_id, role, keyframes, th=th)

    orch.overrides["marks.derive_marks"] = slow_marks
    run_id = create(client, "pickup_check")
    assert upload(client, run_id, "pickup", b"pickup walk " * 400).status_code == 200  # the only upload: the run starts
    assert client.get(f"/api/runs/{run_id}").json()["status"] not in ("done", "partial", "failed")
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        events = sse_events(response)
    assert [e["id"] for e in events] == list(range(len(events)))
    assert events[0]["event"] == "run.created" and events[0]["data"]["seq"] == 0 and events[0]["data"]["kind"] == "run.created"
    assert events[-1]["event"] == "run.completed" and events[-1]["data"]["payload"]["status"] == "done"
    stages = [e["data"]["payload"]["name"] for e in events if e["event"] == "stage"]
    assert stages[:3] == ["sealing", "sealed", "converting"] and stages[-1] == "done" and "marking" in stages
    assert any(e["event"] == "marks.derived" for e in events)
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "done"

    resume_from = events[3]["id"]
    with client.stream("GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(resume_from)}) as response:
        resumed = sse_events(response)
    assert resumed[0]["id"] == resume_from + 1 and resumed[-1]["event"] == "run.completed"
    assert [e["data"] for e in resumed] == [e["data"] for e in events[resume_from + 1 :]]
    with client.stream("GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(events[-1]["id"])}) as response:
        assert sse_events(response) == []  # nothing after the end, and the stream closes instead of waiting


# Replay, sample, health ------------------------------------------------------------------------------------------------------


def test_replay_route_carries_the_badge_date_and_the_fixed_delays(client: TestClient, orch: Orchestrator) -> None:
    events = seed_replay_run(orch, "RUNREPLAY1", STAMPS)
    payload = c.ReplayPayload.model_validate_json(client.get("/api/runs/RUNREPLAY1/replay").text)
    assert payload.mode == c.ReplayMode.FIXED and payload.recorded_on == "2026-10-27" and payload.delays_ms == DELAYS
    assert payload.events == events
    instant = client.get("/api/runs/RUNREPLAY1/replay", params={"mode": "instant"}).json()
    assert instant["delays_ms"] == {} and instant["mode"] == "instant" and len(instant["events"]) == 9
    assert client.get("/api/runs/RUNREPLAY1/replay", params={"mode": "recorded"}).json()["delays_ms"] == {}
    assert client.get("/api/runs/RUNREPLAY1/replay", params={"mode": "fast"}).status_code == 400


def test_sample_endpoint_is_flagged_precomputed(client: TestClient, orch: Orchestrator, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.get("/api/sample").status_code == 404
    seed_replay_run(orch, "RUNSAMPLE", STAMPS)
    monkeypatch.setenv("WALKAROUND_SAMPLE_RUN", "RUNSAMPLE")
    sample = client.get("/api/sample").json()
    assert sample == {"run_id": "RUNSAMPLE", "precomputed": True, "recorded_on": "2026-10-27", "badge": "replayed from a run on 2026-10-27", "replay": "/api/runs/RUNSAMPLE/replay?mode=fixed"}
    assert client.get(sample["replay"]).json()["recorded_on"] == "2026-10-27"


def test_healthz_and_spa_fallback(client: TestClient) -> None:
    health = client.get("/healthz").json()
    assert health["ok"] is True and health["engine"] == "meta-model-api" and health["probe_ok"] is None and health["free_bytes"] > 0
    for route in ("/", "/run/abc", "/run/abc/sheet", "/claim", "/kiosk/abc", "/verify"):
        response = client.get(route)
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/html") and "Walkaround" in response.text


# Files, sheet, top-up ----------------------------------------------------------------------------------------------------------


def test_files_sheet_and_bundle_routes_after_a_run(client: TestClient, orch: Orchestrator) -> None:
    run_id = create(client, "pickup_check")
    assert client.get(f"/api/runs/{run_id}/sheet").status_code == 404
    assert upload(client, run_id, "pickup", b"pickup walk " * 400).status_code == 200
    wait_done(client, run_id)
    sheet = client.get(f"/api/runs/{run_id}/sheet")
    assert sheet.status_code == 200 and sheet.headers["content-type"].startswith("text/html") and "sheet" in sheet.text
    assert c.SheetModel.model_validate_json(client.get(f"/api/runs/{run_id}/sheet.json").text).run_id == run_id
    assert client.get(f"/api/runs/{run_id}/sheet.pdf").status_code == 404  # no PDF from the fake: "PDF retry"
    bundle = client.get(f"/api/runs/{run_id}/bundle.zip")
    assert bundle.status_code == 200 and bundle.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(paths.bundle_zip(run_id))
    assert {"manifest.json", "manifest_run.json"} <= set(archive.namelist())
    coverage = client.get(f"/api/runs/{run_id}/files/derived/coverage.json")
    assert coverage.status_code == 200 and "pickup" in coverage.json()["per_role"]
    assert client.get(f"/api/runs/{run_id}/files/sam/pickup/car.summary.json").json()["status"] == "completed"
    for denied in ("quarantine/pickup.orig", "media/pickup.proxy.mp4", "../ledger.jsonl", "derived/nope.json", "run.json/../quarantine/pickup.orig"):
        assert client.get(f"/api/runs/{run_id}/files/{denied}").status_code == 404, denied
    deletions = client.get(f"/api/runs/{run_id}/files/deletions.json").json()
    assert {row["kind"] for row in deletions["rows"]} == {"file", "files_api", "quarantine"}


def test_topup_route_reruns_the_missing_nouns_on_the_new_clip(client: TestClient, orch: Orchestrator, fakes: Fakes) -> None:
    fakes.missing_panels = [c.PanelId.LEFT_REAR_WHEEL]
    run_id = create(client, "pickup_check")
    assert client.post(f"/api/runs/{run_id}/topup", files={"video": ("clip.mp4", b"x", "video/mp4")}).status_code == 409
    assert upload(client, run_id, "pickup", b"pickup walk " * 400).status_code == 200
    wait_done(client, run_id)
    response = client.post(f"/api/runs/{run_id}/topup", files={"video": ("clip.mp4", b"topup clip bytes " * 300, "video/mp4")})
    assert response.status_code == 200, response.text
    coverage = c.Coverage.model_validate_json(response.text)
    assert set(coverage.per_role) == {"pickup", "topup_1"} and coverage.rerun_seconds is not None
    assert fakes.lane_calls[-1] == ("topup_1", ["car", "license_plate", "face"])
    video = orch.load_video(run_id, "topup_1")
    assert video is not None and video.hash_match is True and video.original_filename == "clip.mp4"
