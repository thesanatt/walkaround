"""FastAPI routes (M15) through Starlette's TestClient with the orchestrator on fakes: chunked uploads with the
browser hash checked on receipt, run creation against the caps, SSE progress live and resumed from Last-Event-ID,
the replay payload with its badge date, the sample pair flagged precomputed, who may fetch which file, the top-up
streamed into the run directory, health."""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.test_orchestrator import STAMPS, Fakes, make_orch, runs_dir, seed_replay_run, th, write_ledger_row, write_run  # noqa: F401
from walkaround import app as app_module
from walkaround import contracts as c, paths
from walkaround.app import create_app, servable
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


def multipart(fields: dict[str, str], file_bytes: bytes, *, name: str = "video", filename: str = "clip.mp4") -> tuple[bytes, str]:
    boundary = "walkaroundboundary"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: video/mp4\r\n\r\n'.encode() + file_bytes + b"\r\n")
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


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
    repeated = finalize(client, run_id, "lost", "pickup", data)  # the response was lost; the browser asks again
    assert repeated.status_code == 200 and c.VideoRecord.model_validate_json(repeated.text) == record


def test_finalize_answers_409_with_both_hashes_and_400_for_missing_chunks_or_a_wrong_walk(client: TestClient) -> None:
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
    check = create(client, "pickup_check")
    put_chunks(client, check, "u4", data)
    refused = finalize(client, check, "u4", "return", data)
    assert refused.status_code == 400 and "not a walk" in refused.json()["detail"] and "missing" not in refused.json()


def test_body_limits_413_on_the_chunk_route_and_elsewhere(client: TestClient, th: dict[str, Any]) -> None:
    run_id = create(client, "pair")
    too_big = client.put("/api/uploads/u4/chunks/0", params={"run_id": run_id}, content=b"x" * (int(th["media"]["chunk_route_max_body_bytes"]) + 1))
    assert too_big.status_code == 413
    assert client.put("/api/uploads/u4/chunks/128", params={"run_id": run_id}, content=b"x").status_code == 400  # 128 x 8 MiB is 1 GiB
    assert client.put("/api/uploads/u4/chunks/127", params={"run_id": run_id}, content=b"x").status_code == 204
    assert client.put("/api/uploads/u4/chunks/0", params={"run_id": "NOPE"}, content=b"x").status_code == 404
    over_1gib = client.post("/api/uploads/u4/finalize", json={"run_id": run_id, "role": "pickup", "sha256_browser": "a" * 64, "bytes": 2 * 1024 ** 3, "filename": "huge.mov"})
    assert over_1gib.status_code == 413
    fat_json = client.post("/api/runs", content=b"{" + b" " * (2 * 1024 * 1024) + b"}", headers={"content-type": "application/json"})
    assert fat_json.status_code == 413
    chunked = client.post("/api/runs", content=(b" " * 65536 for _ in range(20)), headers={"content-type": "application/json"})  # no Content-Length
    assert chunked.status_code == 413
    small_chunked = client.post("/api/runs", content=iter([b'{"kind": ', b'"pickup_check"}']), headers={"content-type": "application/json"})
    assert small_chunked.status_code == 201


# Runs ------------------------------------------------------------------------------------------------------------------


def test_create_run_lookup_and_an_index_of_public_runs_only(client: TestClient, th: dict[str, Any]) -> None:
    run_id = create(client, "pickup_check", keep_plate=True)
    assert len(run_id) == 26
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "created" and run["kind"] == "pickup_check" and run["keep_plate"] is True
    assert run["config_hash"] == c.config_hash(th) and run["thresholds"]["orchestrator"]["pickup_runs_per_day"] == th["orchestrator"]["pickup_runs_per_day"]
    assert client.get("/api/runs").json() == []  # a renter's run id is a receipt: never listed
    write_run("RUNEVAL", created_at="2026-10-27T10:00:00Z", kind=c.RunKind.PAIR, status=c.RunStatus.DONE, keep=True)
    index = client.get("/api/runs").json()
    assert [r["run_id"] for r in index] == ["RUNEVAL"] and index[0]["thresholds"] == {}
    assert client.get("/api/runs/NOPE").status_code == 404
    assert client.get("/api/runs/NOPE/events").status_code == 404
    assert client.get("/api/runs/NOPE/replay").status_code == 404
    assert client.post("/api/runs", json={"kind": "topup"}).status_code == 400
    assert client.post("/api/runs", json={"kind": "pair", "mode": "apartment"}).status_code == 400


def test_create_run_429_when_the_caps_or_the_pause_apply(client: TestClient, orch: Orchestrator, th: dict[str, Any]) -> None:
    write_run("RUNPAIRX", created_at="2026-10-27T16:00:00Z", kind=c.RunKind.PAIR)
    response = client.post("/api/runs", json={"kind": "pair"})
    assert response.status_code == 429
    assert response.json() == {"reason": "no full pairs left today", "pickup_left": th["orchestrator"]["pickup_runs_per_day"], "pairs_left": 0}
    assert client.post("/api/runs", json={"kind": "pickup_check"}).status_code == 201
    status = client.get("/api/status").json()
    assert status == {"paused": False, "pickup_left": th["orchestrator"]["pickup_runs_per_day"] - 1, "pairs_left": 0, "queue_position": None, "queue_eta_minutes": None, "engine": "meta-model-api"}
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
    ended = client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(events[-1]["id"])})
    assert ended.status_code == 204  # nothing after the end: 204 stops the browser reconnecting


def test_sse_sends_the_waiting_events_when_the_run_fails_before_the_chain_starts(client: TestClient, orch: Orchestrator, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "SSE_POLL_S", 0.1)
    release = threading.Event()

    def stuck_seal(run_id: str, originals: list, **kwargs: Any) -> Any:
        release.wait(5)
        raise OSError("disk full")

    orch.overrides["seal.seal_run"] = stuck_seal
    run_id = create(client, "pickup_check")
    assert upload(client, run_id, "pickup", b"pickup walk " * 400).status_code == 200
    threading.Timer(0.3, release.set).start()
    t0 = time.monotonic()
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        assert response.status_code == 200
        events = sse_events(response)
    assert time.monotonic() - t0 < 4
    # no manifest.json, so no chain: the waiting events, numbered as the chain would have, unchained, run.failed last
    assert [e["id"] for e in events] == list(range(len(events))) and events[0]["event"] == "run.created"
    assert events[-1]["event"] == "run.failed" and events[-1]["data"]["payload"] == {"stage": "seal", "text": "OSError: disk full"}
    assert events[-1]["data"]["hash"] == "" and "quarantine.deleted" in [e["event"] for e in events]
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "failed"
    assert client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(events[-1]["id"])}).status_code == 204
    with client.stream("GET", f"/api/runs/{run_id}/events") as again:  # a fresh tab gets the story again
        assert [e["event"] for e in sse_events(again)] == [e["event"] for e in events]
    replay = client.get(f"/api/runs/{run_id}/replay", params={"mode": "instant"}).json()
    assert [e["kind"] for e in replay["events"]] == [e["event"] for e in events]
    assert not list(paths.quarantine_dir(run_id).rglob("*.orig"))


def test_an_unreadable_upload_is_sealed_and_its_failure_reaches_the_stream(client: TestClient, orch: Orchestrator) -> None:
    def bad_probe(path: Path) -> Any:
        raise RuntimeError("ffprobe: Invalid data found when processing input")

    orch.overrides["video.probe"] = bad_probe
    run_id = create(client, "pickup_check")
    assert upload(client, run_id, "pickup", b"not a video " * 100).status_code == 200
    assert wait_done(client, run_id)["status"] == "failed"
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        assert response.status_code == 200
        events = sse_events(response)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run.created" and "seal.manifest" in kinds and kinds[-1] == "run.failed"
    text = events[-1]["data"]["payload"]["text"]
    assert text.startswith("could not read this video; on iPhone set Most Compatible and re-record") and "ffprobe: Invalid data" in text
    assert orch.events.verify_chain(run_id, manifest_sha256=hashlib.sha256(paths.manifest_json(run_id).read_bytes()).hexdigest())


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
    assert [r["run_id"] for r in client.get("/api/runs").json()] == ["RUNSAMPLE"]


def test_healthz_and_spa_fallback(client: TestClient) -> None:
    health = client.get("/healthz").json()
    assert health["ok"] is True and health["engine"] == "meta-model-api" and health["probe_ok"] is None and health["free_bytes"] > 0
    for route in ("/", "/run/abc", "/run/abc/sheet", "/claim", "/kiosk/abc", "/verify"):
        response = client.get(route)
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/html") and "Walkaround" in response.text


# Files, sheet, top-up ----------------------------------------------------------------------------------------------------------


def test_servable_floor_ignores_case_like_apfs() -> None:
    for bad in ("media/PICKUP.PROXY.MP4", "QUARANTINE/pickup.mp4", "Uploads/x/0.part", "EVENTS.PENDING.NDJSON"):
        assert not servable(bad), bad


def test_servable_floor() -> None:
    for ok in ("derived/coverage.json", "sam/pickup/car.ndjson.gz", "media/pickup.proxy_blurred.mp4", "videos/pickup.json", "events.ndjson"):
        assert servable(ok), ok
    for bad in ("quarantine/pickup.orig", "uploads/u1/chunks/000000", "media/pickup.proxy.mp4", "../ledger.jsonl", "/etc/passwd", "events.pending.ndjson", ".run.json.1.2.tmp", "derived/../quarantine/x", "x.part"):
        assert not servable(bad), bad


def test_files_sheet_and_bundle_routes_after_a_run(client: TestClient, orch: Orchestrator) -> None:
    run_id = create(client, "pickup_check")
    assert client.get(f"/api/runs/{run_id}/sheet").status_code == 404
    assert upload(client, run_id, "pickup", b"pickup walk " * 400).status_code == 200
    wait_done(client, run_id)
    sheet = client.get(f"/api/runs/{run_id}/sheet")
    assert sheet.status_code == 200 and sheet.headers["content-type"].startswith("text/html") and "sheet" in sheet.text
    assert client.head(f"/api/runs/{run_id}/sheet").status_code == 200
    assert c.SheetModel.model_validate_json(client.get(f"/api/runs/{run_id}/sheet.json").text).run_id == run_id
    assert client.get(f"/api/runs/{run_id}/sheet.pdf").status_code == 404  # no PDF from the fake: "PDF retry"
    assert client.head(f"/api/runs/{run_id}/sheet.pdf").status_code == 404
    bundle = client.get(f"/api/runs/{run_id}/bundle.zip")
    assert bundle.status_code == 200 and bundle.headers["content-type"] == "application/zip"
    assert {"manifest.json", "manifest_run.json"} <= set(zipfile.ZipFile(io.BytesIO(bundle.content)).namelist())
    assert not paths.bundle_zip(run_id).exists()  # built in memory: the run directory does not change
    coverage = client.get(f"/api/runs/{run_id}/files/derived/coverage.json")
    assert coverage.status_code == 200 and "pickup" in coverage.json()["per_role"]
    assert client.get(f"/api/runs/{run_id}/files/sam/pickup/car.summary.json").json()["status"] == "completed"
    assert client.get(f"/api/runs/{run_id}/files/videos/pickup.json").json()["role"] == "pickup"
    for denied in ("quarantine/pickup.orig", "media/pickup.proxy.mp4", "../ledger.jsonl", "derived/nope.json", "run.json/../quarantine/pickup.orig", "uploads/x/chunks/000000"):
        assert client.get(f"/api/runs/{run_id}/files/{denied}").status_code == 404, denied
    deletions = client.get(f"/api/runs/{run_id}/files/deletions.json").json()
    assert {row["kind"] for row in deletions["rows"]} == {"file", "files_api", "quarantine"}


def test_the_sample_pair_keep_runs_and_the_kiosk_view_are_never_the_run_owner(client: TestClient, orch: Orchestrator, monkeypatch: pytest.MonkeyPatch) -> None:
    write_run("RUNOWNED", created_at="2026-10-27T10:00:00Z", kind=c.RunKind.PAIR, status=c.RunStatus.DONE, keep_plate=True)
    write_run("RUNSHOWN", created_at="2026-10-27T10:00:00Z", kind=c.RunKind.PAIR, status=c.RunStatus.DONE, keep_plate=True, keep=True)
    for run_id in ("RUNOWNED", "RUNSHOWN"):
        frame = paths.sheet_dir(run_id) / "frames" / "pickup_hood_kf1.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"plate kept")
    assert client.get("/api/runs/RUNOWNED/files/sheet/frames/pickup_hood_kf1.jpg").status_code == 200
    assert client.get("/api/runs/RUNOWNED/files/sheet/frames/pickup_hood_kf1.jpg", params={"view": "kiosk"}).status_code == 404
    assert client.get("/api/runs/RUNSHOWN/files/sheet/frames/pickup_hood_kf1.jpg").status_code == 404
    monkeypatch.setenv("WALKAROUND_SAMPLE_RUN", "RUNOWNED")
    assert client.get("/api/runs/RUNOWNED/files/sheet/frames/pickup_hood_kf1.jpg").status_code == 404


def test_sheet_routes_follow_the_owner_rule(client: TestClient, orch: Orchestrator) -> None:
    orch.overrides.pop("privacy.served_file_allowed")  # the real allowlist
    write_run("RUNMINE", created_at="2026-10-27T10:00:00Z", kind=c.RunKind.PAIR, status=c.RunStatus.DONE, keep_plate=True)
    write_run("RUNKEPT", created_at="2026-10-27T10:00:00Z", kind=c.RunKind.PAIR, status=c.RunStatus.DONE, keep_plate=True, keep=True)
    write_run("RUNBLUR", created_at="2026-10-27T10:00:00Z", kind=c.RunKind.PAIR, status=c.RunStatus.DONE)
    for run_id in ("RUNMINE", "RUNKEPT", "RUNBLUR"):
        paths.sheet_html(run_id).write_text('<img src="data:image/jpeg;base64,OWNERFRAME">', encoding="utf-8")  # sheet.py inlines the frames
        paths.sheet_json(run_id).write_text("{}\n", encoding="utf-8")
        paths.sheet_pdf(run_id).write_bytes(b"%PDF-1.7 owner frames")
    routes = ("sheet", "sheet.json", "sheet.pdf")
    assert [client.get(f"/api/runs/RUNMINE/{r}").status_code for r in routes] == [200, 200, 200]  # the renter's own URL
    assert [client.get(f"/api/runs/RUNMINE/{r}", params={"view": "kiosk"}).status_code for r in routes] == [404, 404, 404]
    assert [client.get(f"/api/runs/RUNKEPT/{r}").status_code for r in routes] == [404, 404, 404]  # listed, so never the owner
    assert [client.get(f"/api/runs/RUNBLUR/{r}", params={"view": "kiosk"}).status_code for r in routes] == [200, 200, 200]


def test_topup_route_refuses_public_runs_and_stops_on_the_pause_and_the_ceilings(client: TestClient, orch: Orchestrator, fakes: Fakes) -> None:
    run_id = create(client, "pickup_check")
    assert upload(client, run_id, "pickup", b"pickup walk " * 400).status_code == 200
    wait_done(client, run_id)
    body, ctype = multipart({}, b"a clip " * 1000)
    post = lambda: client.post(f"/api/runs/{run_id}/topup", content=body, headers={"content-type": ctype})  # noqa: E731
    orch.paused = True
    paused = post()
    assert paused.status_code == 429 and "paused" in paused.json()["reason"]
    orch.paused = False
    write_ledger_row("2026-10-27T10:00:00-0400", None, 20.0)  # the day's card ceiling, spent elsewhere
    ceiling = post()
    assert ceiling.status_code == 429 and ceiling.json()["reason"] == "live runs paused for today; the sample pair is precomputed"
    orch.update_run(run_id, lambda r: setattr(r, "keep", True))  # now listed at /api/runs
    assert post().status_code == 403
    assert fakes.calls["segment_video_many"] == 1 and orch.video_roles(run_id) == ["pickup"]
    assert [p for p in (paths.run_dir(run_id) / "uploads").rglob("*") if p.is_file()] == []


def test_finalize_refuses_a_topup_role(client: TestClient) -> None:
    run_id = create(client, "pickup_check")
    data = b"a stranger's clip " * 100
    put_chunks(client, run_id, "u9", data)
    response = finalize(client, run_id, "u9", "topup_1", data)
    assert response.status_code == 400 and "topup" in response.json()["detail"]
    assert not paths.quarantine_original(run_id, "topup_1").exists()


def test_finalize_admits_the_run_again_and_a_refused_run_keeps_no_upload(runs_dir: Path, th: dict[str, Any], fakes: Fakes) -> None:
    now = ["2026-10-27T13:00:00.000Z"]
    orch = make_orch(th, fakes, now=lambda: now[0])
    with TestClient(create_app(orch, background=False)) as client:
        early = create(client, "pair")
        now[0] = "2026-10-27T14:01:00.000Z"  # its reservation lapsed with nothing uploaded
        create(client, "pair")  # so the day's one pair went to another run
        assert upload(client, early, "pickup", b"pickup walk " * 400).status_code == 200  # one walk: nothing starts yet
        refused = upload(client, early, "return", b"return walk " * 400)
        assert refused.status_code == 429
        assert refused.json() == {"reason": "no full pairs left today", "pickup_left": th["orchestrator"]["pickup_runs_per_day"], "pairs_left": 0}
        assert client.get(f"/api/runs/{early}").json()["status"] == "created" and fakes.calls["seal_run"] == 0
    assert not list(paths.quarantine_dir(early).rglob("*.orig"))
    rows = c.read_json(paths.deletions_json(early), c.Deletions).rows
    assert sorted(r.target for r in rows) == ["quarantine/pickup.orig", "quarantine/return.orig"]
    assert all(r.reason == "not started: no full pairs left today" for r in rows)


def test_topup_route_streams_the_clip_into_the_run_and_reruns_the_missing_nouns(client: TestClient, orch: Orchestrator, fakes: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    import starlette.formparsers

    def no_spool(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the top-up must not spool the clip to a temp file outside the run")

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", no_spool)
    fakes.missing_panels = [c.PanelId.LEFT_REAR_WHEEL]
    run_id = create(client, "pickup_check")
    body, ctype = multipart({}, b"x")
    assert client.post(f"/api/runs/{run_id}/topup", content=body, headers={"content-type": ctype}).status_code == 409
    assert upload(client, run_id, "pickup", b"pickup walk " * 400).status_code == 200
    wait_done(client, run_id)
    clip = os.urandom(9 * 1024 * 1024)  # two chunks
    body, ctype = multipart({"sha256_browser": sha(clip)}, clip, filename="wheel.mov")
    response = client.post(f"/api/runs/{run_id}/topup", content=body, headers={"content-type": ctype})
    assert response.status_code == 200, response.text
    coverage = c.Coverage.model_validate_json(response.text)
    assert set(coverage.per_role) == {"pickup", "topup_1"} and coverage.rerun_seconds is not None
    assert fakes.lane_calls[-1] == ("topup_1", ["car", "license_plate", "face"])
    video = orch.load_video(run_id, "topup_1")
    assert video is not None and video.hash_match is True and video.sha256_server == sha(clip) and video.original_filename == "wheel.mov" and video.chunks_received == 2
    assert [p for p in (paths.run_dir(run_id) / "uploads").rglob("*") if p.is_file()] == []
    body, ctype = multipart({"sha256_browser": "0" * 64}, b"another clip")
    mismatch = client.post(f"/api/runs/{run_id}/topup", content=body, headers={"content-type": ctype})
    assert mismatch.status_code == 409 and mismatch.json()["sha256_server"] == sha(b"another clip")
    assert orch.video_roles(run_id) == ["pickup", "topup_1"]  # the refused clip left no record and no bytes
    assert [p for p in (paths.run_dir(run_id) / "uploads").rglob("*") if p.is_file()] == [] and not list(paths.quarantine_dir(run_id).rglob("*.orig"))
    body, ctype = multipart({}, b"y", name="photo")
    assert client.post(f"/api/runs/{run_id}/topup", content=body, headers={"content-type": ctype}).status_code == 400
    assert client.post(f"/api/runs/{run_id}/topup", json={"video": "nope"}).status_code == 400
