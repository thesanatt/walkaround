"""The run folder's work lock (after hero_focus6, Nov 3): one process works on a live run at a time. A server's boot
resume leaves a run alone while its lock names a live process, takes over a stale lock, and a replay-only server
(both daily caps at 0) resumes and starts nothing. Fake pids throughout, liveness patched; no model call."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import walkaround.orchestrator as orchestrator_module
from tests.test_orchestrator import Crash, Fakes, make_orch, new_run, runs_dir, stage_names, th  # noqa: F401
from walkaround import contracts as c, paths, pipeline, seal
from walkaround.app import create_app
from walkaround.orchestrator import CapsExceeded, RunLocked, pid_alive

LIVE_PID = 4242  # a fake pid the patched liveness check calls alive
DEAD_PID = 4343  # and one it calls dead


@pytest.fixture
def alive(monkeypatch: pytest.MonkeyPatch) -> set[int]:
    pids = {LIVE_PID}
    monkeypatch.setattr(orchestrator_module, "pid_alive", lambda pid: pid in pids)
    return pids


def write_lock(run_id: str, pid: int, *, hostname: str | None = None) -> dict[str, Any]:
    owner = {"pid": pid, "hostname": hostname or socket.gethostname(), "started_at": "2026-11-03T19:31:02.000Z"}
    paths.work_lock_path(run_id).write_text(json.dumps(owner), encoding="utf-8")
    return owner


def read_lock(run_id: str) -> dict[str, Any] | None:
    path = paths.work_lock_path(run_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def lock_spy(orch: Any, fakes: Fakes, seen: list[dict[str, Any] | None]) -> None:
    """Read the lock while the marks stage runs, mid-pipeline."""

    async def derive_marks(run_id: str, role: str, keyframes: list, *, th: dict, count_mode: bool = False) -> Any:
        seen.append(read_lock(run_id))
        return await fakes.derive_marks(run_id, role, keyframes, th=th)

    orch.overrides["marks.derive_marks"] = derive_marks


async def crashed_run(orch: Any) -> c.RunManifest:
    """A pair that died in the marks stage, the state a boot resume finds."""
    run = await new_run(orch)

    async def crash(*args: Any, **kwargs: Any) -> Any:
        raise Crash()

    orch.overrides["marks.derive_marks"] = crash
    with pytest.raises(Crash):
        await orch.run_pipeline(run.run_id)
    return run


# 1. the lock while a run works -------------------------------------------------------------------------------------


def test_pid_alive_reads_os_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    assert pid_alive(os.getpid())

    def kill(err: type[Exception]) -> Any:
        def fake(pid: int, sig: int) -> None:
            raise err()

        return fake

    monkeypatch.setattr(os, "kill", kill(ProcessLookupError))
    assert not pid_alive(DEAD_PID)
    monkeypatch.setattr(os, "kill", kill(PermissionError))
    assert pid_alive(LIVE_PID)  # someone else's process is still a process


async def test_the_pipeline_holds_the_lock_and_lets_it_go_at_the_end(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await new_run(orch)
    seen: list[dict[str, Any] | None] = []
    lock_spy(orch, fakes, seen)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.DONE
    assert seen and all(o is not None and o["pid"] == os.getpid() and o["hostname"] == socket.gethostname() for o in seen)
    assert set(seen[0]) == {"pid", "hostname", "started_at"}
    assert not paths.work_lock_path(run.run_id).exists()
    assert not list(paths.run_dir(run.run_id).glob(".work.lock*"))  # no temp file left either
    assert orch.verify(run.run_id) == []  # the sealed manifest does not count the lock


def test_the_run_end_manifest_leaves_the_lock_out(runs_dir: Path) -> None:
    (paths.run_dir("sealed1") / "derived").mkdir(parents=True)
    (paths.run_dir("sealed1") / "derived" / "matches.json").write_text("[]", encoding="utf-8")
    write_lock("sealed1", os.getpid())  # the close stage writes manifest_run.json while the lock is held
    kept = [p.relative_to(paths.run_dir("sealed1")).as_posix() for p in seal._kept_artifacts("sealed1")]
    assert kept == ["derived/matches.json"]


async def test_the_lock_goes_on_an_error_and_on_cancel(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await crashed_run(orch)
    assert not paths.work_lock_path(run.run_id).exists()  # a crash (BaseException) unwinds through the lock

    orch2 = make_orch(th)
    started = asyncio.Event()

    async def hang(*args: Any, **kwargs: Any) -> Any:
        started.set()
        await asyncio.Event().wait()

    orch2.overrides["marks.derive_marks"] = hang
    task = asyncio.create_task(orch2.run_pipeline(run.run_id))
    await started.wait()
    assert read_lock(run.run_id)["pid"] == os.getpid()
    task.cancel()  # what asyncio.run does to the CLI's main task on Ctrl-C
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not paths.work_lock_path(run.run_id).exists()


async def test_a_second_process_cannot_take_a_held_lock(runs_dir: Path, th: dict, alive: set[int]) -> None:
    orch = make_orch(th)
    run = await new_run(orch)
    owner = write_lock(run.run_id, LIVE_PID)
    with pytest.raises(RunLocked) as caught:
        await orch.run_pipeline(run.run_id)
    assert str(LIVE_PID) in str(caught.value) and str(paths.work_lock_path(run.run_id)) in str(caught.value)
    assert read_lock(run.run_id) == owner  # left as it was
    assert orch.load_run(run.run_id).status == c.RunStatus.CREATED  # nothing written to the run


async def test_cli_holds_the_lock_from_staging_and_refuses_a_held_run(runs_dir: Path, th: dict, tmp_path: Path, alive: set[int]) -> None:
    clip = tmp_path / "walk.mp4"
    clip.write_bytes(b"pickup walk bytes " * 1000)
    fakes = Fakes()
    orch = make_orch(th, fakes)
    await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False, run_id="cli_held")
    write_lock("cli_held", LIVE_PID)
    with pytest.raises(RunLocked):
        await pipeline.run_pair_async(clip, None, "cli_held", orch=orch)
    assert not paths.quarantine_original("cli_held", "pickup").exists()  # refused before a byte was staged

    staged: list[dict[str, Any] | None] = []
    real_stage = pipeline.stage_local_file

    async def stage(orch_: Any, run_id: str, role: str, src: Path) -> Any:
        staged.append(read_lock(run_id))
        return await real_stage(orch_, run_id, role, src)

    seen: list[dict[str, Any] | None] = []
    lock_spy(orch, fakes, seen)
    pipeline.stage_local_file, saved = stage, pipeline.stage_local_file
    try:
        final = await pipeline.run_pair_async(clip, None, "cli_free", orch=orch)
    finally:
        pipeline.stage_local_file = saved
    assert final.status == c.RunStatus.DONE
    assert [o["pid"] for o in staged + seen] == [os.getpid()] * (len(staged) + len(seen)) and staged and seen
    assert not paths.work_lock_path("cli_free").exists()


# 2. the boot resume ------------------------------------------------------------------------------------------------


async def test_boot_resume_skips_a_run_a_live_process_holds(runs_dir: Path, th: dict, alive: set[int]) -> None:
    run = await crashed_run(make_orch(th))
    owner = write_lock(run.run_id, LIVE_PID)
    events_before = paths.events_path(run.run_id).read_bytes()
    orch2 = make_orch(th, Fakes())
    assert await orch2.resume_all() == []
    assert orch2.load_run(run.run_id).status == c.RunStatus.MARKING  # not turned into interrupted
    assert paths.events_path(run.run_id).read_bytes() == events_before  # nothing added to the chain
    assert read_lock(run.run_id) == owner and orch2._tasks == {}


async def test_boot_resume_skips_a_lock_from_another_host(runs_dir: Path, th: dict, alive: set[int]) -> None:
    run = await crashed_run(make_orch(th))
    write_lock(run.run_id, DEAD_PID, hostname="some-other-host")  # its pid cannot be checked from here
    assert await make_orch(th).resume_all() == []


async def test_boot_resume_takes_over_a_stale_lock(runs_dir: Path, th: dict, alive: set[int]) -> None:
    run = await crashed_run(make_orch(th))
    write_lock(run.run_id, DEAD_PID)
    fakes2 = Fakes()
    orch2 = make_orch(th, fakes2)
    assert await orch2.resume_all() == [run.run_id]
    assert read_lock(run.run_id)["pid"] == os.getpid()  # taken over before the task even runs
    await asyncio.gather(*orch2._tasks.values())
    assert orch2.load_run(run.run_id).status == c.RunStatus.DONE and "interrupted" in stage_names(orch2, run.run_id)
    assert fakes2.marks_roles == ["pickup", "return"]
    assert not paths.work_lock_path(run.run_id).exists()


async def test_boot_resume_without_a_lock_works_as_before(runs_dir: Path, th: dict, alive: set[int]) -> None:
    run = await crashed_run(make_orch(th))
    orch2 = make_orch(th)
    assert await orch2.resume_all() == [run.run_id]
    await asyncio.gather(*orch2._tasks.values())
    assert orch2.load_run(run.run_id).status == c.RunStatus.DONE and not paths.work_lock_path(run.run_id).exists()


# 3. a replay-only server --------------------------------------------------------------------------------------------


async def test_a_replay_only_server_resumes_and_starts_nothing(runs_dir: Path, th: dict) -> None:
    run = await crashed_run(make_orch(th))
    events_before = paths.events_path(run.run_id).read_bytes()
    replay = make_orch(th, limits={"pickup_runs_per_day": 0, "pair_runs_per_day": 0})
    assert replay.replay_only
    assert await replay.resume_all() == []
    with pytest.raises(CapsExceeded):
        replay.start(run.run_id)
    assert replay._tasks == {} and not paths.work_lock_path(run.run_id).exists()
    assert replay.load_run(run.run_id).status == c.RunStatus.MARKING
    assert paths.events_path(run.run_id).read_bytes() == events_before
    with pytest.raises(CapsExceeded):
        replay.check_can_topup(replay.load_run(run.run_id))
    assert not make_orch(th, limits={"pickup_runs_per_day": 3, "pair_runs_per_day": 0}).replay_only  # one cap open is live


def test_a_replay_only_app_boots_without_touching_live_runs(runs_dir: Path, th: dict) -> None:
    run = asyncio.run(crashed_run(make_orch(th)))
    events_before = paths.events_path(run.run_id).read_bytes()
    replay = make_orch(th, limits={"pickup_runs_per_day": 0, "pair_runs_per_day": 0})

    async def no_network(url: str, headers: dict[str, str]) -> int:
        return 200

    replay.http_get = no_network  # the boot also starts the health probe
    with TestClient(create_app(replay, background=True)) as client:  # the boot path: seed samples, resume_all
        response = client.post(f"/api/runs/{run.run_id}/start")
        assert response.status_code == 429
    assert replay._tasks == {} and paths.events_path(run.run_id).read_bytes() == events_before
    assert paths.quarantine_original(run.run_id, "pickup").exists()  # a run under way keeps its originals


def test_start_route_answers_409_while_another_process_holds_the_run(runs_dir: Path, th: dict, alive: set[int]) -> None:
    run = asyncio.run(crashed_run(make_orch(th)))
    write_lock(run.run_id, LIVE_PID)
    orch = make_orch(th)
    with TestClient(create_app(orch, background=False)) as client:
        response = client.post(f"/api/runs/{run.run_id}/start")
    assert response.status_code == 409 and "another process" in response.json()["detail"]
    assert orch._tasks == {} and orch.load_run(run.run_id).status == c.RunStatus.MARKING
