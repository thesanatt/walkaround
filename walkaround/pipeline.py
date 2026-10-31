"""The floor end to end from local files (docs/ARCHITECTURE.md section 5).

run_pair sends a pickup and a return walk through the path the browser uses (8 MB chunks, finalize with the hash of
the bytes sent) and then through every orchestrator stage in order: seal, proxy and WAV, the Files upload, the SAM
lanes, panels and keyframes with their blurred crops, the display proxy blur, Spark marks with whitening, Stays Put
and narration (recorded gaps until those lanes exist), match, run-end deletions, the sheet and the closing manifest.
rebuild and verify regenerate the derived files from the frozen model answers and compare them, with no network call.

A walk given twice (the same bytes as pickup and return) is a self-pair: the orchestrator reuses the pickup's lanes,
image checks and Spark answers for the return, so every mark must match itself at distance 0.

Stand-in nouns let a short test clip with no car stand in for a walk; see stand_in_prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from . import contracts as c
from . import paths
from .orchestrator import TERMINAL_STATUSES, WHOLE_FRAME, Orchestrator

STAND_IN_SLOTS: tuple[c.Noun, ...] = (c.Noun.HOOD, c.Noun.TRUNK_LID, c.Noun.WINDSHIELD, c.Noun.REAR_WINDOW)


def stand_in_prompts(nouns: list[str], *, container: str | None = None) -> tuple[list[c.Noun], dict[str, str]]:
    """Lanes and prompts for a clip with no car: each noun stands in for one single-slot panel (hood, trunk lid,
    windshield, rear window: no side vote, a fixed front), license plate and face go as themselves so the blur still
    runs. The car lane, the box every panel box must sit inside, is sent as container when one is named; without it
    no car request is made and the whole frame is that box (orchestrator.WHOLE_FRAME)."""
    phrases = [" ".join(str(n).split()) for n in nouns]
    if not phrases or len(phrases) > len(STAND_IN_SLOTS) or not all(phrases):
        raise ValueError(f"--nouns takes 1 to {len(STAND_IN_SLOTS)} noun phrases, got {nouns!r}")
    prompts = {slot.value: text for slot, text in zip(STAND_IN_SLOTS, phrases)}
    prompts[c.Noun.CAR.value] = " ".join(str(container).split()) if container else WHOLE_FRAME
    slots = [c.Noun(k) for k in prompts]
    return [n for n in c.LANE_NOUNS if n in slots or n in c.PRIVACY_NOUNS], prompts


def make_orchestrator(th: dict[str, Any] | None = None) -> Orchestrator:
    return Orchestrator.from_env(th)


async def stage_local_file(orch: Orchestrator, run_id: str, role: str, src: Path) -> c.VideoRecord:
    """A local file goes through the upload path: 8 MB chunks, then finalize with the hash of the bytes sent."""
    chunk_bytes = int(orch.th["media"]["chunk_bytes"])
    upload_id = f"cli-{role}"
    digest = hashlib.sha256()
    size = 0
    with open(src, "rb") as fh:
        for index, block in enumerate(iter(lambda: fh.read(chunk_bytes), b"")):
            digest.update(block)
            size += len(block)
            orch.chunk_path(upload_id, index, run_id).write_bytes(block)
    if size == 0:
        orch.chunk_path(upload_id, 0, run_id).write_bytes(b"")
    return await orch.finalize_upload(run_id, upload_id=upload_id, role=role, sha256_browser=digest.hexdigest(), bytes=size, filename=src.name)


def _sha256(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


async def _tail(orch: Orchestrator, run_id: str, on_event: Callable[[c.Event], None]) -> None:
    async for event in orch.events.subscribe(run_id, since_seq=-1):
        on_event(event)


async def _start_or_resume(
    orch: Orchestrator,
    walks: dict[str, Path],
    run_id: str | None,
    nouns: list[str] | None,
    container: str | None,
    keep_plate: bool,
) -> c.RunManifest:
    """A new run with its files sent, or an unfinished run of that id with whatever upload it still lacks."""
    if run_id is not None and orch.run_exists(run_id):
        run = orch.load_run(run_id)
        if run.status in TERMINAL_STATUSES:
            raise FileExistsError(f"run {run_id} already ended ({run.status.value}); verify or rebuild it, or pick another id")
        if set(orch.expected_roles(run)) != set(walks):
            raise ValueError(f"run {run_id} is a {run.kind.value} run; give it the same walks to resume it")
        for role, src in walks.items():
            video = orch.load_video(run_id, role)
            if video is not None and video.sha256_server and _sha256(src) != video.sha256_server:
                raise ValueError(f"{src} is not the {role} walk run {run_id} was started with")
    else:
        kind = c.RunKind.PAIR if "return" in walks else c.RunKind.PICKUP_CHECK
        lanes, prompts = stand_in_prompts(nouns, container=container) if nouns else (None, None)
        run = await orch.create_run(mode=c.RunMode.CAR, kind=kind, keep_plate=keep_plate, run_id=run_id, noun_list=lanes, prompts=prompts)
    for role, src in walks.items():
        if not paths.quarantine_original(run.run_id, role).is_file():
            await stage_local_file(orch, run.run_id, role, src)
    return orch.load_run(run.run_id)


async def run_pair_async(
    pickup: Path,
    return_: Path | None,
    run_id: str | None = None,
    nouns: list[str] | None = None,
    *,
    container: str | None = None,
    keep_plate: bool = False,
    orch: Orchestrator | None = None,
    on_event: Callable[[c.Event], None] | None = None,
) -> c.RunManifest:
    """The whole floor on local files; resumes an unfinished run of the same id. on_event sees every chained event."""
    orch = orch or make_orchestrator()
    walks = {"pickup": Path(pickup)} | ({"return": Path(return_)} if return_ is not None else {})
    for path in walks.values():
        if not path.is_file():
            raise FileNotFoundError(f"not a file: {path}")
    run = await _start_or_resume(orch, walks, run_id, nouns, container, keep_plate)
    printer = asyncio.create_task(_tail(orch, run.run_id, on_event)) if on_event is not None else None
    try:
        final = await orch.run_pipeline(run.run_id)
    finally:
        if printer is not None:
            try:
                await asyncio.wait_for(printer, timeout=5)  # the tail ends by itself at run.completed or run.failed
            except TimeoutError:
                printer.cancel()
                with suppress(asyncio.CancelledError):
                    await printer
    return final


def run_pair(pickup: Path, return_: Path | None, run_id: str | None = None, nouns: list[str] | None = None, **kwargs: Any) -> c.RunManifest:
    """run_pair_async from synchronous code (the CLI, the D1 scripts)."""
    return asyncio.run(run_pair_async(pickup, return_, run_id, nouns, **kwargs))


def rebuild(run_id: str, *, out: Path | None = None, thresholds: dict[str, Any] | None = None, orch: Orchestrator | None = None) -> tuple[dict[str, str], Path | None]:
    """Regenerate the derived files with no network call: {file: sha256 or reason} and where they were written. A closed
    run's sealed files stay as they are, so without out it is rebuilt into runs/_rebuild/<run_id>."""
    orch = orch or make_orchestrator()
    if out is None and orch.is_closed(run_id):
        out = paths.runs_root() / "_rebuild"
    return orch.rebuild(run_id, out=out, thresholds=thresholds), out


def verify(run_id: str, *, out: Path | None = None, orch: Orchestrator | None = None) -> list[str]:
    """Rebuild into a scratch copy and compare: [] means "rebuild identical", else each problem, the first differing
    file first (hashes of the originals still present, the event chain, the head the sheet printed)."""
    orch = orch or make_orchestrator()
    return orch.verify(run_id, out=out)


__all__ = ["STAND_IN_SLOTS", "make_orchestrator", "rebuild", "run_pair", "run_pair_async", "stage_local_file", "stand_in_prompts", "verify"]
