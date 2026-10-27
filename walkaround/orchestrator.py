"""The run state machine (M14, docs/ARCHITECTURE.md sections 5, 7, 9 and 14).

What lives here: the stage order with its status and progress per stage, resume from the files on disk, the
daily caps and the per-run spend ceiling read from the ledger, the 12 minute deadline, the replay driver that
re-emits stored events with the delays of web/kiosk/delays.json, rebuild and verify, and the health probe.
What does not: rendering, verdicts, model calls. The other lanes' modules are reached through `dep()`, resolved
lazily by dotted name ("video.make_proxy") so a module that is not written yet fails at call time, never at
import time, and a test can inject a fake for any one of them.

Stage completion is read from the files of the run directory, never from run.json, so a run interrupted anywhere
resumes at the first stage whose files are missing. The model-era facts (lanes, spark.json, voice.json,
tracks.json, keyframes.json) are frozen; `rebuild` regenerates the derived files from them without a network call.

Events and the chain genesis: section 4 makes the prev of seq 0 the sha256 of bundle/manifest.json, which does not
exist until the seal runs over the uploaded originals. Events emitted before that (run.created, upload.started,
upload.done, the sealing stage) wait in events.pending.ndjson and are chained in order the moment manifest.json is
written, so seq 0 is run.created with the manifest hash as prev and seal.manifest follows a few lines later.

Thresholds: run.json freezes the plain threshold dict in the form canonical JSON stores it (floats rounded to 6
places), and every stage reads that frozen dict, so a live run and a later rebuild compute from identical values.

Replay delays: web/kiosk/delays.json maps a stage name (the RunStatus value a `stage` event carries) to the total
milliseconds the stage takes in fixed mode; the driver spreads that total evenly over the stage's events.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import secrets
import shutil
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import contracts as c
from . import ledger, paths
from .config import BASE_URL, REPO_DIR

ET = ZoneInfo("America/New_York")
TERMINAL_EVENTS = frozenset({c.EventKind.RUN_COMPLETED, c.EventKind.RUN_FAILED})
TERMINAL_STATUSES = frozenset({c.RunStatus.DONE, c.RunStatus.FAILED, c.RunStatus.PARTIAL})
TRACKED_STATUSES = frozenset({c.LaneStatus.COMPLETED, c.LaneStatus.ZERO_MATCHES})
WALK_ROLES: dict[c.RunKind, tuple[str, ...]] = {
    c.RunKind.PAIR: ("pickup", "return"),
    c.RunKind.PICKUP_CHECK: ("pickup",),
    c.RunKind.TOPUP: ("pickup",),
    c.RunKind.CLAIM: ("pickup",),
}
STAGING_DIR_NAME = "_uploads"  # runs/_uploads/<upload_id>/chunks/<n>: chunks that arrive before finalize names a run
PENDING_EVENTS_NAME = "events.pending.ndjson"  # pre-seal events waiting for the chain genesis
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


# Small helpers ------------------------------------------------------------------------------------------------------


def utc_now() -> str:
    """ISO 8601 UTC with milliseconds, the ts of every event and clock this module writes."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def new_run_id(ms: int | None = None) -> str:
    """A ULID: 48 bits of Unix milliseconds then 80 random bits, Crockford base32, 26 chars, sortable by time."""
    stamp = int(time.time() * 1000) if ms is None else ms
    value = (stamp << 80) | secrets.randbits(80)
    chars = []
    for _ in range(26):
        chars.append(_B32[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def parse_ts(ts: str) -> datetime:
    """Parse the ledger's local-offset stamps and this module's UTC stamps into an aware datetime."""
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def et_day_of(ts: str) -> str:
    """The ET calendar day (YYYY-MM-DD) a timestamp falls in; the caps count per ET day (section 9)."""
    return parse_ts(ts).astimezone(ET).strftime("%Y-%m-%d")


def app_commit() -> str | None:
    """APP_COMMIT from the environment, else the checked-out commit read from .git (a file read, not a git call)."""
    if os.environ.get("APP_COMMIT"):
        return os.environ["APP_COMMIT"]
    head = Path(REPO_DIR) / ".git" / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if not text.startswith("ref:"):
        return text[:40] or None
    ref = Path(REPO_DIR) / ".git" / text[4:].strip()
    return ref.read_text(encoding="utf-8").strip()[:40] if ref.is_file() else None


def sha256_path(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _plain(value: Any) -> Any:
    """Event payloads and the frozen thresholds hold plain JSON values only, in their canonical (rounded) form."""
    return json.loads(c.canonical_json(value))


def role_order(role: str) -> tuple[int, str]:
    """pickup, return, then top-ups and claims by name."""
    return ({"pickup": 0, "return": 1}.get(role, 2), role)


class HashMismatch(Exception):
    """The browser's sha256 and the server's differ; the app answers 409 with both values (section 14 item 8)."""

    def __init__(self, sha256_browser: str, sha256_server: str) -> None:
        super().__init__(f"hash mismatch: browser {sha256_browser} server {sha256_server}")
        self.sha256_browser = sha256_browser
        self.sha256_server = sha256_server


class MissingChunks(Exception):
    def __init__(self, missing: list[int]) -> None:
        super().__init__(f"missing chunks {missing}")
        self.missing = missing


class CapsExceeded(Exception):
    """A live run may not start: the daily caps, the card ceiling or the probe pause (section 14 item 14)."""

    def __init__(self, reason: str, *, pickup_left: int, pairs_left: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.pickup_left = pickup_left
        self.pairs_left = pairs_left


class NoSpace(Exception):
    """The volume is under the free-space floor (section 14 item 27)."""


class StageGap(Exception):
    """A stage finished with a hole the sheet must show (ceiling, deadline, a lane not wired): recorded, not retried."""


# Stages --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    name: str
    status: c.RunStatus
    pct: int
    model: bool  # spends money: obeys the spend ceiling and the deadline
    fatal: bool  # a failure past retries ends the run instead of leaving a gap in the sheet
    pair_only: bool = False


STAGES: tuple[Stage, ...] = (
    Stage("seal", c.RunStatus.SEALING, 5, model=False, fatal=False),
    Stage("media", c.RunStatus.CONVERTING, 15, model=False, fatal=True),
    Stage("files", c.RunStatus.CONVERTING, 20, model=False, fatal=False),
    Stage("lanes", c.RunStatus.TRACKING, 45, model=True, fatal=False),
    Stage("panels", c.RunStatus.TRACKING, 55, model=True, fatal=False),
    Stage("blur", c.RunStatus.BLURRING, 60, model=False, fatal=False),
    Stage("marks", c.RunStatus.MARKING, 70, model=True, fatal=False),
    Stage("staysput", c.RunStatus.STAYSPUT, 78, model=True, fatal=False),
    Stage("narration", c.RunStatus.STAYSPUT, 82, model=True, fatal=False),
    Stage("match", c.RunStatus.MATCHING, 88, model=False, fatal=False, pair_only=True),
    Stage("sheet", c.RunStatus.SHEET, 95, model=False, fatal=False),
    Stage("close", c.RunStatus.SHEET, 98, model=False, fatal=False),
)
STAGE_BY_NAME: dict[str, Stage] = {s.name: s for s in STAGES}
SEALED_PCT = 10
NOT_REBUILT = "not rebuilt: its lane has no offline entry point yet; the recorded file stands"


class Orchestrator:
    """Sequences a run through STAGES and never computes a verdict or renders anything."""

    def __init__(
        self,
        *,
        th: dict[str, Any],
        engine: c.SamEngine,
        max_concurrent_runs: int,
        media_workers: int,
        decode_workers: int,
        now: Callable[[], str] = utc_now,
        events: Any | None = None,
        overrides: dict[str, Callable[..., Any]] | None = None,
        delays_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.th = th
        self.engine = engine
        self.now = now
        self.clock = clock
        self.sleep = sleep
        self.max_concurrent_runs = max_concurrent_runs
        self.media_workers = media_workers
        self.decode_workers = decode_workers
        self.delays_path = delays_path or paths.kiosk_delays_path()
        self.overrides: dict[str, Callable[..., Any]] = dict(overrides or {})
        self._events = events
        self._run_sem: asyncio.Semaphore | None = None
        self._media_sem: asyncio.Semaphore | None = None
        self._queue: list[str] = []
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._deadline: dict[str, float] = {}
        self._deletions: dict[str, list[c.DeletionRecord]] = {}
        self._decode_pool: Any | None = None
        self.paused = False
        self.probe_ok: bool | None = None
        self._probe_failures = 0
        self.http_get: Callable[[str, dict[str, str]], Awaitable[int]] = _http_get_status
        self.stage_fns: dict[str, Callable[[c.RunManifest], Awaitable[None]]] = {
            "seal": self._stage_seal,
            "media": self._stage_media,
            "files": self._stage_files,
            "lanes": self._stage_lanes,
            "panels": self._stage_panels,
            "blur": self._stage_blur,
            "marks": self._stage_marks,
            "staysput": self._stage_staysput,
            "narration": self._stage_narration,
            "match": self._stage_match,
            "sheet": self._stage_sheet,
            "close": self._stage_close,
        }

    # Dependencies ----------------------------------------------------------------------------------------------------

    def dep(self, name: str) -> Callable[..., Any]:
        """'module.function' from overrides first, else walkaround.<module> imported on first use."""
        if name in self.overrides:
            return self.overrides[name]
        module_name, _, attr = name.rpartition(".")
        module = importlib.import_module(f"walkaround.{module_name}")
        return getattr(module, attr)

    def has_dep(self, name: str) -> bool:
        try:
            self.dep(name)
        except (ImportError, AttributeError):
            return False
        return True

    @property
    def events(self) -> Any:
        """The eventlog module (append, read, head_hash, subscribe, verify_chain) or the stand-in until it lands."""
        if self._events is None:
            try:
                self._events = importlib.import_module("walkaround.eventlog")
            except ImportError:
                self._events = FileEventLog()
        return self._events

    def decode_pool(self) -> Any | None:
        if self._decode_pool is None and self.has_dep("sam_wire.DecodePool"):
            self._decode_pool = self.dep("sam_wire.DecodePool")(self.decode_workers)
        return self._decode_pool

    def close(self) -> None:
        """Release the decode pool at shutdown."""
        pool, self._decode_pool = self._decode_pool, None
        if pool is not None and hasattr(pool, "close"):
            pool.close()

    def run_sem(self) -> asyncio.Semaphore:
        if self._run_sem is None:
            self._run_sem = asyncio.Semaphore(self.max_concurrent_runs)
        return self._run_sem

    def media_sem(self) -> asyncio.Semaphore:
        if self._media_sem is None:
            self._media_sem = asyncio.Semaphore(self.media_workers)
        return self._media_sem

    # run.json ----------------------------------------------------------------------------------------------------------

    def load_run(self, run_id: str) -> c.RunManifest:
        return c.read_json(paths.run_json(run_id), c.RunManifest)

    def save_run(self, run: c.RunManifest) -> None:
        c.write_json(paths.run_json(run.run_id), run)

    def run_exists(self, run_id: str) -> bool:
        return (paths.runs_root() / run_id / "run.json").is_file()

    def run_th(self, run: c.RunManifest) -> dict[str, Any]:
        """The thresholds a run computes with: the dict frozen in its run.json, else this server's."""
        return run.thresholds or self.th

    def list_runs(self) -> list[c.RunManifest]:
        """Every run with a run.json, newest first."""
        runs = []
        for entry in paths.runs_root().iterdir():
            if entry.is_dir() and (entry / "run.json").is_file():
                try:
                    runs.append(self.load_run(entry.name))
                except Exception:
                    continue
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def load_video(self, run_id: str, role: str) -> c.VideoRecord | None:
        path = paths.video_json(run_id, role)
        return c.read_json(path, c.VideoRecord) if path.is_file() else None

    def save_video(self, run_id: str, video: c.VideoRecord) -> None:
        c.write_json(paths.video_json(run_id, video.role), video)

    @staticmethod
    def expected_roles(run: c.RunManifest) -> tuple[str, ...]:
        return WALK_ROLES[run.kind]

    def uploads_complete(self, run: c.RunManifest) -> bool:
        return all(paths.quarantine_original(run.run_id, role).is_file() for role in self.expected_roles(run))

    # Events ------------------------------------------------------------------------------------------------------------

    def chain_open(self, run_id: str) -> bool:
        """True once the chain genesis exists (manifest.json) or the chain has already started."""
        if paths.manifest_json(run_id).is_file():
            return True
        events_file = paths.run_dir(run_id) / "events.ndjson"
        return events_file.is_file() and events_file.stat().st_size > 0

    def emit(self, run_id: str, kind: c.EventKind, payload: dict[str, Any]) -> c.Event:
        """eventlog.append once the chain is open; before the seal the event waits in events.pending.ndjson."""
        plain = _plain(payload)
        if not self.chain_open(run_id):
            return self._emit_pending(run_id, kind, plain)
        self.flush_pending(run_id)
        return self.events.append(run_id, kind, plain, ts=self.now())

    def _emit_pending(self, run_id: str, kind: c.EventKind, payload: dict[str, Any]) -> c.Event:
        ts = self.now()
        with open(paths.run_dir(run_id) / PENDING_EVENTS_NAME, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": ts, "kind": kind.value, "payload": payload}, sort_keys=True) + "\n")
        return c.Event(seq=-1, ts=ts, run_id=run_id, kind=kind, payload=payload, prev="", hash="")

    def pending_events(self, run_id: str) -> list[c.Event]:
        """The events still waiting for the chain genesis, as provisional Events with seq -1."""
        path = paths.run_dir(run_id) / PENDING_EVENTS_NAME
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                out.append(c.Event(seq=-1, ts=raw["ts"], run_id=run_id, kind=c.EventKind(raw["kind"]), payload=raw["payload"], prev="", hash=""))
        return out

    def flush_pending(self, run_id: str) -> list[c.Event]:
        """Chain the waiting events in order with their original timestamps; the pending file goes away after."""
        pending = self.pending_events(run_id)
        if not pending:
            return []
        chained = [self.events.append(run_id, e.kind, e.payload, ts=e.ts) for e in pending]
        (paths.run_dir(run_id) / PENDING_EVENTS_NAME).unlink(missing_ok=True)
        return chained

    def set_stage(self, run_id: str, status: c.RunStatus, pct: int) -> None:
        run = self.load_run(run_id)
        run.status = status
        self.save_run(run)
        self.emit(run_id, c.EventKind.STAGE, {"name": status.value, "pct": pct})

    def read_events(self, run_id: str, *, since_seq: int = -1) -> list[c.Event]:
        return list(self.events.read(run_id, since_seq=since_seq))

    # Create and upload -------------------------------------------------------------------------------------------------

    async def create_run(
        self,
        *,
        mode: c.RunMode,
        kind: c.RunKind,
        keep_plate: bool,
        live_or_replay: c.LiveOrReplay = c.LiveOrReplay.LIVE,
    ) -> c.RunManifest:
        """ULID run_id, the layout on disk, thresholds frozen with their hash, run.json, run.created."""
        self.check_space()
        run_id = new_run_id()
        paths.ensure_run_layout(run_id)
        created = self.now()
        retention = parse_ts(created) + timedelta(hours=self.th["privacy"]["run_retention_hours"])
        run = c.RunManifest(
            run_id=run_id,
            created_at=created,
            mode=mode,
            kind=kind,
            keep_plate=keep_plate,
            live_or_replay=live_or_replay,
            app_commit=app_commit(),
            config_hash=c.config_hash(self.th),
            thresholds=_plain(self.th),
            retention_deadline=retention.strftime("%Y-%m-%dT%H:%M:%SZ"),
            spend_ceiling_cents=int(self.th["orchestrator"]["run_spend_ceiling_cents"]),
            sam_engine=self.engine,
        )
        self.save_run(run)
        self.emit(run_id, c.EventKind.RUN_CREATED, {"kind": kind.value, "mode": mode.value, "keep_plate": keep_plate, "config_hash": run.config_hash})
        return run

    def check_space(self) -> None:
        free = shutil.disk_usage(paths.runs_root()).free
        if free < int(self.th["orchestrator"]["volume_min_free_bytes"]):
            raise NoSpace(f"the volume has {free // 1_000_000} MB free; new runs need 500 MB")

    def chunk_path(self, upload_id: str, index: int, run_id: str | None = None) -> Path:
        """Where one chunk lands: under the run when the client names it, else the staging area."""
        if run_id:
            return paths.upload_chunk_path(run_id, upload_id, index)
        path = paths.runs_root() / STAGING_DIR_NAME / upload_id / "chunks" / f"{index:06d}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def chunks_dir(self, upload_id: str, run_id: str) -> Path:
        under_run = paths.uploads_dir(run_id, upload_id) / "chunks"
        if under_run.is_dir() and any(under_run.iterdir()):
            return under_run
        return paths.runs_root() / STAGING_DIR_NAME / upload_id / "chunks"

    async def finalize_upload(
        self,
        run_id: str,
        *,
        upload_id: str,
        role: str,
        sha256_browser: str,
        bytes: int,
        filename: str,
    ) -> c.VideoRecord:
        """Reassemble the chunks into quarantine, hash on receipt, 409 material on a mismatch, upload.done."""
        size = bytes
        chunk_bytes = int(self.th["media"]["chunk_bytes"])
        expected = max(1, -(-size // chunk_bytes))
        chunks_dir = self.chunks_dir(upload_id, run_id)
        present = sorted(chunks_dir.glob("[0-9]*")) if chunks_dir.is_dir() else []
        indexes = {int(p.name) for p in present if p.name.isdigit()}
        missing = [i for i in range(expected) if i not in indexes]
        if missing:
            raise MissingChunks(missing)
        self.emit(run_id, c.EventKind.UPLOAD_STARTED, {"role": role, "upload_id": upload_id, "chunks": expected})
        target = paths.quarantine_original(run_id, role)
        digest = await asyncio.to_thread(_assemble, [chunks_dir / f"{i:06d}" for i in range(expected)], target)
        shutil.rmtree(chunks_dir.parent, ignore_errors=True)
        record = c.VideoRecord(
            role=role,
            original_filename=filename,
            bytes=target.stat().st_size,
            sha256_browser=sha256_browser.lower(),
            sha256_server=digest,
            hash_match=digest == sha256_browser.lower(),
            received_at=self.now(),
            chunks_expected=expected,
            chunks_received=expected,
            quarantine_path=paths.rel(run_id, target),
        )
        self.save_video(run_id, record)
        if not record.hash_match:
            target.unlink(missing_ok=True)  # never keep, let alone process, bytes that do not match (section 14 item 8)
            raise HashMismatch(sha256_browser, digest)
        payload = {"role": role, "bytes": record.bytes, "sha256": digest, "filename": filename}
        if record.bytes > int(self.th["media"]["upload_warn_bytes"]):
            payload["warning"] = "over 500 MB; the proxy step will take longer"
        self.emit(run_id, c.EventKind.UPLOAD_DONE, payload)
        return record

    # Pipeline ----------------------------------------------------------------------------------------------------------

    def start(self, run_id: str) -> asyncio.Task[Any]:
        """Queue the run behind max_concurrent_runs and run it in the background."""
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            return task
        self._queue.append(run_id)
        task = asyncio.create_task(self._guarded(run_id), name=f"run-{run_id}")
        self._tasks[run_id] = task
        return task

    async def _guarded(self, run_id: str) -> None:
        async with self.run_sem():
            if run_id in self._queue:
                self._queue.remove(run_id)
            try:
                await self.run_pipeline(run_id)
            except Exception as exc:  # the pipeline reports its own failures; this is the last net
                self._fail(run_id, "pipeline", str(exc))

    def queue_position(self, run_id: str) -> int | None:
        return self._queue.index(run_id) + 1 if run_id in self._queue else None

    def stage_done(self, run: c.RunManifest, stage: Stage) -> bool:
        """Completion read from the files on disk (section 7: the files are the only state)."""
        run_id = run.run_id
        roles = self.expected_roles(run)
        if stage.name == "seal":
            return _exists(paths.bundle_dir(run_id) / "manifest.json")
        if stage.name == "media":
            return all((v := self.load_video(run_id, r)) is not None and v.proxy is not None for r in roles)
        if stage.name == "files":
            lanes_done = not any(self.missing_lanes(run, r) for r in roles)
            return lanes_done or all(self._file_id_valid(self.load_video(run_id, r)) for r in roles)
        if stage.name == "lanes":
            return not any(self.missing_lanes(run, r) for r in roles)
        if stage.name == "panels":
            return all(_exists(paths.derived_dir(run_id) / n) for n in ("tracks.json", "panels.json", "keyframes.json", "coverage.json"))
        if stage.name == "blur":
            return all((v := self.load_video(run_id, r)) is not None and v.proxy_blurred is not None for r in roles)
        if stage.name == "marks":
            return _exists(paths.derived_dir(run_id) / "marks.json")
        if stage.name == "staysput":
            return _exists(paths.derived_dir(run_id) / "mark_tracks.json")
        if stage.name == "narration":
            return _exists(paths.derived_dir(run_id) / "narration.json")
        if stage.name == "match":
            return _exists(paths.derived_dir(run_id) / "matches.json") and _exists(paths.derived_dir(run_id) / "verdicts.json")
        if stage.name == "sheet":
            return _exists(paths.sheet_dir(run_id) / "sheet.html")
        if stage.name == "close":
            return _exists(paths.run_dir(run_id) / "deletions.json") and _exists(paths.bundle_dir(run_id) / "manifest_run.json")
        raise KeyError(stage.name)

    def missing_lanes(self, run: c.RunManifest, role: str) -> list[c.Noun]:
        """Nouns of the run without a lane summary (lane_done) for the role: what a resume re-runs."""
        return [n for n in run.noun_list if not _exists(paths.sam_dir(run.run_id, role) / f"{paths.lane_id(n)}.summary.json")]

    def lane_status(self, run_id: str, role: str) -> dict[c.Noun, c.LaneStatus]:
        """Status per noun from the lane summaries on disk (what coverage prints as not_run or not_filmed)."""
        out: dict[c.Noun, c.LaneStatus] = {}
        for noun in c.LANE_NOUNS:
            summary = paths.lane_summary_path(run_id, role, noun)
            if summary.is_file():
                try:
                    out[noun] = c.read_json(summary, c.LaneRecord).status
                except Exception:
                    continue
        return out

    def _file_id_valid(self, video: c.VideoRecord | None) -> bool:
        if video is None or video.files_api is None or video.files_api.deleted_at is not None:
            return False
        if video.files_api.expires_at is None:
            return True
        return parse_ts(video.files_api.expires_at) > parse_ts(self.now()) + timedelta(minutes=2)

    def stages_for(self, run: c.RunManifest) -> list[Stage]:
        return [s for s in STAGES if not (s.pair_only and run.kind != c.RunKind.PAIR)]

    async def run_pipeline(self, run_id: str) -> c.RunManifest:
        """Stage by stage: skip what the files say is done, retry once, record gaps, always reach the sheet."""
        run = self.load_run(run_id)
        self._deadline[run_id] = self.clock() + float(self.th["orchestrator"]["run_deadline_s"])
        self._deletions.setdefault(run_id, self._load_deletions(run_id))
        if self.chain_open(run_id):
            self.flush_pending(run_id)
        missing: list[str] = []
        for stage in self.stages_for(run):
            run = self.load_run(run_id)
            if self.stage_done(run, stage):
                continue
            if stage.model and self._blocked(run, stage, missing):
                continue
            self.set_stage(run_id, stage.status, stage.pct)
            error = await self._run_stage(run, stage)
            if error is None:
                if stage.model:
                    self._note_spend(run_id)
                continue
            self._record_error(run_id, stage.name, error)
            if stage.fatal:
                return self._fail(run_id, stage.name, error)
            missing.append(stage.name)
        return self._finish(run_id, missing)

    def _note_spend(self, run_id: str) -> None:
        cents = self.spend_cents(run_id)
        run = self.load_run(run_id)
        run.api_spend_cents = cents
        self.save_run(run)
        self.emit(run_id, c.EventKind.LEDGER_ROW, {"dollars_so_far": cents / 100})

    def _blocked(self, run: c.RunManifest, stage: Stage, missing: list[str]) -> bool:
        """The ceiling or the deadline stops a model stage before it starts; the gap is recorded."""
        if stage.name == "lanes":
            return False  # the lanes stage writes not_run records per lane instead of skipping whole
        if self.over_ceiling(run):
            reason = f"spend ceiling {run.spend_ceiling_cents} cents reached"
        elif self.past_deadline(run.run_id):
            reason = f"run deadline {int(self.th['orchestrator']['run_deadline_s'])} s passed"
        else:
            return False
        self._record_error(run.run_id, stage.name, reason)
        missing.append(stage.name)
        return True

    async def _run_stage(self, run: c.RunManifest, stage: Stage) -> str | None:
        """One try plus request_retries; the error text when every try failed."""
        tries = 1 + int(self.th["orchestrator"]["request_retries"])
        error = None
        for attempt in range(tries):
            try:
                await self.stage_fns[stage.name](run)
                return None
            except StageGap as gap:
                return str(gap)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < tries:
                    await self.sleep(0)
        return error

    def _record_error(self, run_id: str, stage: str, text: str) -> None:
        run = self.load_run(run_id)
        run.errors.append(c.RunError(stage=stage, text=text[:500]))
        self.save_run(run)

    def _fail(self, run_id: str, stage: str, text: str) -> c.RunManifest:
        run = self.load_run(run_id)
        run.status = c.RunStatus.FAILED
        if not any(e.stage == stage and e.text == text[:500] for e in run.errors):
            run.errors.append(c.RunError(stage=stage, text=text[:500]))
        self.save_run(run)
        self.emit(run_id, c.EventKind.RUN_FAILED, {"stage": stage, "text": text})
        return self._with_head(run_id)

    def _finish(self, run_id: str, missing: list[str]) -> c.RunManifest:
        run = self.load_run(run_id)
        if missing:
            tracked = self.tracked_counts(run)
            self.emit(run_id, c.EventKind.RUN_PARTIAL, {"missing": missing, "text": self._partial_text(run, tracked), "tracked": tracked})
            run.status = c.RunStatus.PARTIAL
        else:
            run.status = c.RunStatus.DONE
        run.api_spend_cents = self.spend_cents(run_id)
        self.save_run(run)
        self.emit(run_id, c.EventKind.STAGE, {"name": run.status.value, "pct": 100})
        self.emit(run_id, c.EventKind.RUN_COMPLETED, {"status": run.status.value, "api_spend_cents": run.api_spend_cents})
        return self._with_head(run_id)

    def _with_head(self, run_id: str) -> c.RunManifest:
        run = self.load_run(run_id)
        run.events_head_hash = self.events.head_hash(run_id)
        self.save_run(run)
        return run

    def tracked_counts(self, run: c.RunManifest) -> dict[str, int]:
        """Per role, the nouns whose lane completed (with or without matches); a not_run lane does not count."""
        out = {}
        for role in self.expected_roles(run):
            status = self.lane_status(run.run_id, role)
            out[role] = sum(1 for n in run.noun_list if status.get(n) in TRACKED_STATUSES)
        return out

    def _partial_text(self, run: c.RunManifest, tracked: dict[str, int]) -> str:
        """Section 14 item 10: 'incomplete: X of 16 nouns tracked', X from the walk with the fewest."""
        return f"incomplete: {min(tracked.values(), default=0)} of {len(run.noun_list)} nouns tracked"

    # Stage bodies ------------------------------------------------------------------------------------------------------

    def _probe_video(self, run_id: str, role: str) -> c.VideoRecord:
        """ffprobe the original once; the record keeps the display size every SAM token carries."""
        video = self.load_video(run_id, role)
        if video is None:
            raise FileNotFoundError(f"videos/{role}.json missing; was the upload finalized?")
        if video.ffprobe is None:
            src = paths.quarantine_original(run_id, role)
            info = self.dep("video.probe")(src)
            container = Path(video.original_filename).suffix.lstrip(".").lower() or "mp4"
            video.ffprobe = self.dep("video.ffprobe_info")(info, container=container)
            self.save_video(run_id, video)
            self.emit(run_id, c.EventKind.MEDIA_PROBED, {"role": role, "ffprobe": video.ffprobe})
        return video

    async def _stage_seal(self, run: c.RunManifest) -> None:
        run_id = run.run_id
        roles = self.expected_roles(run)
        originals = [paths.quarantine_original(run_id, r) for r in roles]
        camera_time: str = c.CAMERA_TIME_NOT_PRESENT
        for role in roles:
            video = await asyncio.to_thread(self._probe_video, run_id, role)
            if role == "pickup" and video.ffprobe is not None and video.ffprobe.creation_time != "not present":
                camera_time = video.ffprobe.creation_time
        manifest, clocks = await asyncio.to_thread(self.dep("seal.seal_run"), run_id, originals, now=self.now(), camera_time=camera_time)
        fresh = self.load_run(run_id)
        fresh.clocks = c.ThreeClocks(
            sealed_at=clocks.sealed_at,
            first_model_call_at=fresh.clocks.first_model_call_at,
            sealed_minutes_after_camera=clocks.sealed_minutes_after_camera,
            camera_time=clocks.camera_time if clocks.camera_time else (camera_time if camera_time != c.CAMERA_TIME_NOT_PRESENT else None),
        )
        self.save_run(fresh)
        self.flush_pending(run_id)  # manifest.json exists now: run.created becomes seq 0 with its hash as prev
        self.emit(run_id, c.EventKind.SEAL_MANIFEST, {"sha256": sha256_path(paths.manifest_json(run_id)), "files": manifest.files})
        if manifest.stamps.tsr_time:
            self.emit(run_id, c.EventKind.SEAL_TSR, {"tsr_time": manifest.stamps.tsr_time, "serial": manifest.stamps.tsa_serial, "issuer": manifest.stamps.tsa_issuer})
        if manifest.stamps.manifest_ots:
            self.emit(run_id, c.EventKind.SEAL_OTS, {"status": manifest.stamps.manifest_ots})
        self.set_stage(run_id, c.RunStatus.SEALED, SEALED_PCT)

    async def _stage_media(self, run: c.RunManifest) -> None:
        for role in self.expected_roles(run):
            async with self.media_sem():
                await asyncio.to_thread(self._media_for_role, run, role)

    def _media_for_role(self, run: c.RunManifest, role: str) -> None:
        run_id = run.run_id
        video = self._probe_video(run_id, role)
        if video.proxy is not None:
            return
        th_media = self.run_th(run)["media"]
        src = paths.quarantine_original(run_id, role)
        proxy = paths.proxy_path(run_id, role)
        self.dep("video.make_proxy")(src, proxy, fps=float(th_media["proxy_fps"]), max_edge=int(th_media["proxy_max_edge"]), preset=th_media["proxy_preset"], crf=int(th_media["proxy_crf"]))
        video.proxy = self.dep("video.proxy_record")(run_id, role, proxy)
        if video.ffprobe is None or video.ffprobe.has_audio:
            try:
                wav = self.dep("video.extract_wav")(src, paths.wav_path(run_id, role), rate=int(th_media["wav_rate"]))
                video.wav = c.WavRecord(path=paths.rel(run_id, wav), duration_s=float(video.ffprobe.duration_s if video.ffprobe else 0.0), sha256=sha256_path(wav))
            except Exception as exc:  # no audio track is a normal case (section 14 item 16)
                self.emit(run_id, c.EventKind.VOICE_DONE, {"role": role, "mode_applied": False, "text": f"no audio: {exc}"})
        self.save_video(run_id, video)
        fresh = self.load_run(run_id)
        fresh.fps = video.proxy.fps
        self.save_run(fresh)
        self.emit(run_id, c.EventKind.MEDIA_PROXY_DONE, {"role": role, "proxy": video.proxy})

    async def _stage_files(self, run: c.RunManifest) -> None:
        for role in self.expected_roles(run):
            await self._ensure_file_id(run.run_id, role)

    async def _ensure_file_id(self, run_id: str, role: str) -> c.VideoRecord:
        """Upload the proxy to the Files API unless a live file_id exists; stamps first_model_call_at."""
        video = self.load_video(run_id, role)
        if video is None or video.proxy is None:
            raise FileNotFoundError(f"no proxy for {role}")
        if self._file_id_valid(video):
            return video
        proxy = paths.run_dir(run_id) / video.proxy.path
        record = await self.dep("sam_client.upload_file")(proxy, run_id=run_id, expires_after_s=int(self.th["media"]["files_expires_after_s"]))
        video.files_api = record
        self.save_video(run_id, video)
        run = self.load_run(run_id)
        if run.clocks.first_model_call_at is None:
            run.clocks.first_model_call_at = record.uploaded_at
            self.save_run(run)
        self.emit(run_id, c.EventKind.FILES_UPLOADED, {"role": role, "file_id": record.file_id, "expires_at": record.expires_at})
        return video

    async def _stage_lanes(self, run: c.RunManifest) -> None:
        gaps = []
        for role in self.expected_roles(run):
            gap = await self._lanes_for_role(run, role, self.missing_lanes(run, role))
            if gap:
                gaps.append(f"{role}: {gap}")
        if gaps:
            raise StageGap("; ".join(gaps))

    async def _lanes_for_role(self, run: c.RunManifest, role: str, nouns: list[c.Noun]) -> str | None:
        """Run the lanes the role still lacks; returns the reason when the budget stopped them (not_run)."""
        run_id = run.run_id
        if not nouns:
            return None
        if self.over_ceiling(self.load_run(run_id)):
            reason = f"spend ceiling {run.spend_ceiling_cents} cents reached before {len(nouns)} lanes"
            self._write_not_run(run_id, role, nouns, reason)
            return reason
        if self.past_deadline(run_id):
            reason = f"run deadline passed before {len(nouns)} lanes"
            self._write_not_run(run_id, role, nouns, reason)
            return reason
        video = await self._ensure_file_id(run_id, role)
        assert video.proxy is not None and video.files_api is not None
        prompts = [(paths.lane_id(n), n.value) for n in nouns]
        results = await self.dep("sam_client.segment_video_many")(
            video.files_api.file_id,
            prompts,
            run_id=run_id,
            role=role,
            frames_sent=video.proxy.frames,
            media_size=(video.proxy.width, video.proxy.height),
            metadata=self.dep("sam_client.sam_metadata")(),
            concurrency=int(self.th["orchestrator"]["sam_video_semaphore_per_run"]),
            on_event=self._lane_event_sink(run_id, role),
        )
        for lane, (record, frames) in results.items():
            self._emit_lane_done(run_id, role, lane, record, frames)
        return None

    def _lane_event_sink(self, run_id: str, role: str) -> Callable[[str, dict[str, Any]], None]:
        """Raw SSE events of one lane become sam.request, sam.first_delta and sam.frame events (boxes only)."""
        started: dict[str, float] = {}
        first_delta: set[str] = set()
        parse = self.dep("sam_wire.parse_lane_text") if self.has_dep("sam_wire.parse_lane_text") else None

        def sink(lane: str, event: dict[str, Any]) -> None:
            kind = event.get("type")
            try:
                if kind == "response.created":
                    started[lane] = self.clock()
                    response = event.get("response") or {}
                    self.emit(run_id, c.EventKind.SAM_REQUEST, {"request_id": response.get("id"), "role": role, "noun": lane, "params": response.get("metadata") or {}, "store": response.get("store")})
                elif kind == "response.output_text.delta":
                    if lane not in first_delta:
                        first_delta.add(lane)
                        ms = int((self.clock() - started.get(lane, self.clock())) * 1000)
                        self.emit(run_id, c.EventKind.SAM_FIRST_DELTA, {"role": role, "noun": lane, "ms": ms})
                    if parse is not None:
                        frames, _ = parse(event.get("delta") or "", media="video")
                        for frame in frames:
                            objects = [{"id": o.id, "box": o.box, "c": o.c} for o in frame.objects]
                            self.emit(run_id, c.EventKind.SAM_FRAME, {"role": role, "noun": lane, "frame": frame.frame, "objects": objects})
            except Exception:
                return  # a progress event never breaks a lane

        return sink

    def _emit_lane_done(self, run_id: str, role: str, lane: str, record: c.LaneRecord, frames: list[c.IndexFrame]) -> None:
        if record.status in (c.LaneStatus.FAILED, c.LaneStatus.INCOMPLETE):
            self.emit(run_id, c.EventKind.SAM_FAILED, {"role": role, "noun": lane, "status": record.status.value, "error": record.error, "request_id": record.request_id})
            if record.status == c.LaneStatus.FAILED:
                return
        payload = {
            "role": role,
            "noun": lane,
            "lane_path": record.lane_path,
            "sha256": record.lane_sha256,
            "frames": len(frames),
            "ids": record.ids,
            "status": record.status.value,
            "frames_sent": record.frames_sent,
            "video_frames_processed": record.video_frames_processed,
            "parse_cpu_s": record.parse_cpu_s,
            "dollars": record.dollars,
        }
        self.emit(run_id, c.EventKind.SAM_LANE_DONE, payload)

    def _write_not_run(self, run_id: str, role: str, nouns: list[c.Noun], reason: str) -> None:
        """A lane the ceiling or the deadline stopped: a FAILED summary so coverage prints not_run, never not_filmed."""
        video = self.load_video(run_id, role)
        size = [video.proxy.width, video.proxy.height] if video and video.proxy else [0, 0]
        for noun in nouns:
            record = c.LaneRecord(lane_id=paths.lane_id(noun), role=role, noun=noun, prompt_text=noun.value, media_size=size, status=c.LaneStatus.FAILED, error=reason)
            c.write_json(paths.lane_summary_path(run_id, role, noun), record)
            c.write_json(paths.lane_index_path(run_id, role, noun), [])
            self.emit(run_id, c.EventKind.SAM_FAILED, {"role": role, "noun": record.lane_id, "status": "failed", "error": reason, "request_id": None})

    def load_lanes(self, run: c.RunManifest, role: str) -> dict[str, tuple[c.LaneRecord, list[c.IndexFrame]]]:
        lanes = {}
        for noun in run.noun_list:
            summary = paths.lane_summary_path(run.run_id, role, noun)
            index = paths.lane_index_path(run.run_id, role, noun)
            if summary.is_file():
                frames = c.read_json_list(index, c.IndexFrame) if index.is_file() else []
                lanes[paths.lane_id(noun)] = (c.read_json(summary, c.LaneRecord), frames)
        return lanes

    def _pts(self, run_id: str, video: c.VideoRecord) -> list[float]:
        assert video.proxy is not None
        return [float(v) for v in json.loads((paths.run_dir(run_id) / video.proxy.pts_map_path).read_text(encoding="utf-8"))]

    async def _derive_role(self, run: c.RunManifest, role: str, lanes: dict[str, Any]) -> dict[c.PanelId, c.PanelCoverage]:
        """panels.derive for one role; returns that role's coverage."""
        run_id = run.run_id
        video = self.load_video(run_id, role)
        assert video is not None and video.proxy is not None
        _tracks, _panels, _keyframes, coverage = await self.dep("panels.derive")(
            run_id,
            role,
            lanes=lanes,
            pts=self._pts(run_id, video),
            media_size=(video.proxy.width, video.proxy.height),
            fps=video.proxy.fps,
            decode=self.decode_pool(),
            blur_keyframe=self.dep("privacy.blur_keyframe"),
            th=self.run_th(run),
        )
        return coverage

    def _coverage_after_derive(self, run: c.RunManifest, per_role: dict[str, dict[c.PanelId, c.PanelCoverage]]) -> c.Coverage:
        """panels.derive writes coverage.json itself (every role, odometer, rerun seconds); read it, or build it when
        a stand-in did not write one."""
        path = paths.coverage_json(run.run_id)
        if path.is_file():
            coverage = c.read_json(path, c.Coverage)
            for role, states in per_role.items():
                coverage.per_role.setdefault(role, states)
            return coverage
        coverage = self.dep("panels.plan_topup")(c.Coverage(per_role=per_role), self.run_th(run))
        c.write_json(path, coverage)
        return coverage

    async def _stage_panels(self, run: c.RunManifest) -> None:
        run_id = run.run_id
        per_role: dict[str, dict[c.PanelId, c.PanelCoverage]] = {}
        for role in self.expected_roles(run):
            per_role[role] = await self._derive_role(run, role, self.load_lanes(run, role))
        combined = self._coverage_after_derive(run, per_role)
        derived = paths.derived_dir(run_id)
        self.emit(run_id, c.EventKind.PANELS_DERIVED, {"sha256": {n: sha256_path(derived / f"{n}.json") for n in ("tracks", "panels", "keyframes")}})
        self.emit(run_id, c.EventKind.COVERAGE_UPDATED, {"sha256": sha256_path(derived / "coverage.json"), "missing": combined.missing})
        if combined.missing:
            self.emit(run_id, c.EventKind.TOPUP_REQUESTED, {"missing": combined.missing, "topup_text": combined.topup_text, "nouns_to_rerun": combined.nouns_to_rerun})
        fresh = self.load_run(run_id)
        fresh.counts.panels = len(c.read_json_list(paths.panels_json(run_id), c.Panel)) if paths.panels_json(run_id).is_file() else 0
        self.save_run(fresh)

    async def _stage_blur(self, run: c.RunManifest) -> None:
        run_id = run.run_id
        tracks = c.read_json_list(paths.tracks_json(run_id), c.PanelTrack) if paths.tracks_json(run_id).is_file() else []
        for role in self.expected_roles(run):
            video = self.load_video(run_id, role)
            assert video is not None and video.proxy is not None
            if video.proxy_blurred is not None:
                continue
            record, deletion = await self.dep("privacy.render_blurred_proxy")(run_id, role, [t for t in tracks if t.role == role], fps=video.proxy.fps, th=self.run_th(run))
            video.proxy_blurred = record
            self.save_video(run_id, video)
            self._add_deletions(run_id, [deletion])
            self.emit(run_id, c.EventKind.BLUR_DONE, {"role": role, "sha256": record.sha256, "path": record.path})

    async def _stage_marks(self, run: c.RunManifest) -> None:
        run_id = run.run_id
        keyframes = c.read_json_list(paths.keyframes_json(run_id), c.Keyframe)
        total = 0
        for role in self.expected_roles(run):
            marks, _kfs, records = await self.dep("marks.derive_marks")(run_id, role, [k for k in keyframes if k.role == role], th=self.run_th(run))
            total += len(marks)
            for record in records:
                self.emit(run_id, c.EventKind.SPARK_RESPONSE, {"role": role, "request_id": record.request_id, "status": record.status.value, "incomplete_reason": record.incomplete_reason, "truncated": record.truncated, "sha256": hashlib.sha256(record.raw_text.encode("utf-8")).hexdigest()})
        self.emit(run_id, c.EventKind.MARKS_DERIVED, {"sha256": sha256_path(paths.marks_json(run_id)), "marks": total})
        fresh = self.load_run(run_id)
        fresh.counts.marks = total
        self.save_run(fresh)

    async def _stage_staysput(self, run: c.RunManifest) -> None:
        run_id = run.run_id
        if not self.has_dep("staysput.derive_staysput"):
            raise StageGap("staysput module not wired (backlog B2)")
        count = 0
        for role in self.expected_roles(run):
            tracks = await self.dep("staysput.derive_staysput")(run_id, role, th=self.run_th(run))
            count += len(tracks)
        self.emit(run_id, c.EventKind.STAYSPUT_DERIVED, {"sha256": sha256_path(paths.mark_tracks_json(run_id)), "mark_tracks": count})

    async def _stage_narration(self, run: c.RunManifest) -> None:
        run_id = run.run_id
        th = self.run_th(run)
        records: list[c.VoiceRecord] = []
        if paths.voice_json(run_id).is_file():
            records = c.read_json_list(paths.voice_json(run_id), c.VoiceRecord)
        for role in self.expected_roles(run):
            if any(r.role == role for r in records):
                continue
            video = self.load_video(run_id, role)
            if video is None or video.wav is None:
                continue
            wav = paths.run_dir(run_id) / video.wav.path
            transcript = await asyncio.to_thread(self.dep("voice_client.transcribe_wav"), wav, mode=th["voice"]["mode"], keywords=list(th["nouns"]["voice_keywords"]), run_id=run_id)
            record = self.dep("voice_client.voice_record")(transcript, role=role, wav_sha256=video.wav.sha256 or sha256_path(wav))
            records.append(record)
            c.write_json(paths.voice_json(run_id), records)
            self.emit(run_id, c.EventKind.VOICE_DONE, {"role": role, "mode_applied": record.mode_applied, "turns": len(record.response.turns)})
        if not self.has_dep("narration.derive_narration"):
            raise StageGap("narration module not wired (backlog B6)")
        narration = await self.dep("narration.derive_narration")(run_id, th=th)
        self.emit(run_id, c.EventKind.NARRATION_DERIVED, {"sha256": sha256_path(paths.narration_json(run_id)), "mode_applied": narration.mode_applied})

    async def _stage_match(self, run: c.RunManifest) -> None:
        run_id = run.run_id
        th = self.run_th(run)
        if not paths.calibration_json(run_id).is_file():
            calibration = self.dep("match.calibrate")(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=th)
            c.write_json(paths.calibration_json(run_id), calibration)
        matches, verdicts, _calibration = await asyncio.to_thread(self.dep("match.derive_match"), run_id, th=th, second_look=None)
        self.emit(run_id, c.EventKind.MATCH_DERIVED, {"sha256": {"matches": sha256_path(paths.matches_json(run_id)), "verdicts": sha256_path(paths.verdicts_json(run_id))}, "matches": len(matches), "verdicts": len(verdicts)})
        fresh = self.load_run(run_id)
        fresh.counts.matches = len(matches)
        self.save_run(fresh)

    async def _stage_sheet(self, run: c.RunManifest) -> None:
        run_id = run.run_id
        _model, hashes = await asyncio.to_thread(self.dep("sheet.write_sheet"), run_id, th=self.run_th(run))
        self.emit(run_id, c.EventKind.SHEET_WRITTEN, {"sha256": hashes})

    async def _stage_close(self, run: c.RunManifest) -> None:
        """Run end: remote deletions with their receipts, the quarantine, deletions.json, manifest_run.json."""
        run_id = run.run_id
        now = self.now()
        videos = [v for r in self.expected_roles(run) if (v := self.load_video(run_id, r)) is not None]
        response_ids = self._response_ids(run_id) if run.responses_store == c.ResponsesStore.DELETED_PER_REQUEST else None
        rows = list(await self.dep("privacy.delete_remote")(run_id, videos, response_ids=response_ids, now=now))
        for row in rows:
            kind = c.EventKind.FILES_DELETED if row.kind == c.DeletionKind.FILES_API else c.EventKind.RESPONSES_DELETED
            key = "file_id" if row.kind == c.DeletionKind.FILES_API else "request_id"
            self.emit(run_id, kind, {key: row.target, "response": row.api_response, "reason": row.reason})
        for video in videos:
            if video.files_api is not None and any(r.target == video.files_api.file_id and r.api_response for r in rows):
                video.files_api.deleted_at = now
                video.files_api.delete_response = next(r.api_response for r in rows if r.target == video.files_api.file_id)
            video.quarantine_deleted_at = now
            self.save_video(run_id, video)
        rows += list(self.dep("privacy.delete_quarantine")(run_id, now=now))
        self.emit(run_id, c.EventKind.QUARANTINE_DELETED, {"files": [r.target for r in rows if r.kind == c.DeletionKind.QUARANTINE]})
        all_rows = self._add_deletions(run_id, rows)
        self.dep("privacy.close_deletions")(run_id, all_rows, keep_plate=run.keep_plate, chosen_by="renter at upload" if run.keep_plate else None)
        pdf = paths.sheet_dir(run_id) / "evidence.pdf"
        self.dep("seal.write_manifest_run")(
            run_id,
            deletions=all_rows,
            events_head_hash=self.events.head_hash(run_id) or "",
            config_hash=run.config_hash or "",
            app_commit=run.app_commit or "",
            pdf_sha256=sha256_path(pdf) if pdf.is_file() else None,
        )

    def _response_ids(self, run_id: str) -> list[str]:
        ids: list[str] = []
        for role in ("pickup", "return"):
            for summary in paths.sam_dir(run_id, role).glob("*.summary.json"):
                record = c.read_json(summary, c.LaneRecord)
                if record.request_id:
                    ids.append(record.request_id)
        if paths.spark_json(run_id).is_file():
            ids += [r.request_id for r in c.read_json_list(paths.spark_json(run_id), c.SparkRequestRecord)]
        return ids

    def _load_deletions(self, run_id: str) -> list[c.DeletionRecord]:
        path = paths.run_dir(run_id) / "deletions.json"
        return list(c.read_json(path, c.Deletions).rows) if path.is_file() else []

    def _add_deletions(self, run_id: str, rows: Iterable[c.DeletionRecord]) -> list[c.DeletionRecord]:
        """Deletion rows survive a crash: deletions.json is rewritten every time a row lands."""
        current = self._deletions.setdefault(run_id, self._load_deletions(run_id))
        for row in rows:
            if row not in current:
                current.append(row)
        run = self.load_run(run_id)
        c.write_json(paths.deletions_json(run_id), c.Deletions(run_id=run_id, rows=current, keep_plate=run.keep_plate, keep_plate_chosen_by="renter at upload" if run.keep_plate else None))
        return current

    # Top-up ------------------------------------------------------------------------------------------------------------

    async def run_topup(self, run_id: str, role: str) -> c.Coverage:
        """Only coverage.nouns_to_rerun on the new clip; the rerun time is measured and printed."""
        run = self.load_run(run_id)
        coverage = c.read_json(paths.coverage_json(run_id), c.Coverage)
        nouns = [c.Noun(n) for n in coverage.nouns_to_rerun]
        if self.load_video(run_id, role) is None:
            raise FileNotFoundError(f"videos/{role}.json missing; save the clip through finalize_upload first")
        t0 = self.clock()
        topup_run = run.model_copy(update={"noun_list": nouns})
        async with self.media_sem():
            await asyncio.to_thread(self._media_for_role, run, role)
        gap = await self._lanes_for_role(topup_run, role, self.missing_lanes(topup_run, role))
        if gap:
            self._record_error(run_id, "topup", gap)
        role_coverage = await self._derive_role(topup_run, role, self.load_lanes(topup_run, role))
        combined = self._coverage_after_derive(run, {role: role_coverage})
        combined.per_role[role] = role_coverage
        combined = self.dep("panels.plan_topup")(combined, self.run_th(run))
        combined.rerun_seconds = round(self.clock() - t0, 3)
        c.write_json(paths.coverage_json(run_id), combined)
        self.emit(run_id, c.EventKind.COVERAGE_UPDATED, {"sha256": sha256_path(paths.coverage_json(run_id)), "missing": combined.missing, "role": role, "rerun_seconds": combined.rerun_seconds})
        return combined

    # Resume ------------------------------------------------------------------------------------------------------------

    async def resume_all(self) -> list[str]:
        """On boot: live runs without a terminal status become interrupted and resume where the files stop."""
        resumed = []
        for run in self.list_runs():
            if run.live_or_replay != c.LiveOrReplay.LIVE or run.status in TERMINAL_STATUSES:
                continue
            if not self.uploads_complete(run) and not self.stage_done(run, STAGE_BY_NAME["media"]):
                continue
            run.status = c.RunStatus.INTERRUPTED
            self.save_run(run)
            self.emit(run.run_id, c.EventKind.STAGE, {"name": c.RunStatus.INTERRUPTED.value, "pct": 0})
            self.start(run.run_id)
            resumed.append(run.run_id)
        return resumed

    # Money -------------------------------------------------------------------------------------------------------------

    def ledger_rows(self) -> list[dict[str, Any]]:
        return ledger.rows(ledger.LEDGER_PATH)

    def spend_cents(self, run_id: str) -> int:
        return int(round(sum(float(r.get("usd", 0.0)) for r in self.ledger_rows() if r.get("run_id") == run_id) * 100))

    def over_ceiling(self, run: c.RunManifest) -> bool:
        return self.spend_cents(run.run_id) >= run.spend_ceiling_cents

    def past_deadline(self, run_id: str) -> bool:
        deadline = self._deadline.get(run_id)
        return deadline is not None and self.clock() > deadline

    def caps_remaining(self, *, et_day: str) -> dict[str, int]:
        """{pickup, pair} left today from ledger dollars and the kinds of the runs that spent them."""
        th = self.th["orchestrator"]
        dollars = 0.0
        run_ids: set[str] = set()
        for row in self.ledger_rows():
            if et_day_of(row["ts"]) != et_day:
                continue
            dollars += float(row.get("usd", 0.0))
            if row.get("run_id"):
                run_ids.add(str(row["run_id"]))
        used = {c.RunKind.PICKUP_CHECK: 0, c.RunKind.PAIR: 0}
        for run in self.list_runs():
            started = run.run_id in run_ids or (run.status != c.RunStatus.CREATED and et_day_of(run.created_at) == et_day)
            if run.live_or_replay == c.LiveOrReplay.LIVE and run.kind in used and started:
                used[run.kind] += 1
        if dollars * 100 >= float(th["daily_card_ceiling_cents"]):
            return {"pickup": 0, "pair": 0}
        return {
            "pickup": max(0, int(th["pickup_runs_per_day"]) - used[c.RunKind.PICKUP_CHECK]),
            "pair": max(0, int(th["pair_runs_per_day"]) - used[c.RunKind.PAIR]),
        }

    def check_can_start(self, kind: c.RunKind) -> dict[str, int]:
        """Raise CapsExceeded when the pause or today's caps stop a live run of this kind."""
        left = self.caps_remaining(et_day=et_day_of(self.now()))
        if self.paused:
            raise CapsExceeded("live runs paused; the sample pair is precomputed", pickup_left=left["pickup"], pairs_left=left["pair"])
        if kind == c.RunKind.PAIR and left["pair"] <= 0:
            raise CapsExceeded("no full pairs left today", pickup_left=left["pickup"], pairs_left=left["pair"])
        if kind == c.RunKind.PICKUP_CHECK and left["pickup"] <= 0:
            raise CapsExceeded("no pickup checks left today", pickup_left=left["pickup"], pairs_left=left["pair"])
        return left

    # Health and sweep --------------------------------------------------------------------------------------------------

    async def health_probe(self) -> bool:
        """GET /v1/models; three failures in a row pause live runs, one pass clears the pause."""
        try:
            from .config import api_key

            status = await self.http_get(f"{BASE_URL}/models", {"Authorization": f"Bearer {api_key()}"})
            ok = 200 <= status < 300
        except Exception:
            ok = False
        self.probe_ok = ok
        if ok:
            self._probe_failures = 0
            self.paused = False
        else:
            self._probe_failures += 1
            if self._probe_failures >= int(self.th["orchestrator"]["health_probe_failures_to_pause"]):
                self.paused = True
        return ok

    async def sweep(self) -> list[c.DeletionRecord]:
        keep = {r.run_id for r in self.list_runs() if r.keep}
        sample = self.sample_run_id()
        if sample:
            keep.add(sample)
        th = self.th["privacy"]
        return list(await asyncio.to_thread(self.dep("privacy.sweep"), now=self.now(), retention_hours=int(th["run_retention_hours"]), temp_minutes=int(th["unblurred_temp_max_minutes"]), keep_run_ids=keep))

    def sample_run_id(self) -> str | None:
        """The precomputed demo run: WALKAROUND_SAMPLE_RUN, else the first samples/<run_id>/ with a run.json."""
        env = os.environ.get("WALKAROUND_SAMPLE_RUN")
        if env:
            return env
        samples = Path(REPO_DIR) / "samples"
        if samples.is_dir():
            for entry in sorted(samples.iterdir()):
                if (entry / "run.json").is_file():
                    return entry.name
        return None

    # Replay ------------------------------------------------------------------------------------------------------------

    def load_delays(self) -> dict[str, int]:
        if not Path(self.delays_path).is_file():
            return {}
        raw = json.loads(Path(self.delays_path).read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in raw.items() if not str(k).startswith("_")}

    def replay_payload(self, run_id: str, *, mode: c.ReplayMode) -> c.ReplayPayload:
        run = self.load_run(run_id)
        events = self.read_events(run_id)
        delays = self.load_delays() if mode == c.ReplayMode.FIXED else {}
        return c.ReplayPayload(run_id=run_id, mode=mode, recorded_on=run.created_at[:10], delays_ms=delays, events=events)

    def replay_delays(self, events: list[c.Event], *, mode: c.ReplayMode) -> list[float]:
        """Seconds to wait before each event: the stage share (fixed), the capped ts delta (recorded), 0 (instant)."""
        if mode == c.ReplayMode.INSTANT or not events:
            return [0.0] * len(events)
        if mode == c.ReplayMode.RECORDED:
            cap = float(self.th["orchestrator"]["replay_recorded_delay_cap_ms"]) / 1000
            waits = [0.0]
            for prev, cur in zip(events, events[1:]):
                waits.append(min(cap, max(0.0, (parse_ts(cur.ts) - parse_ts(prev.ts)).total_seconds())))
            return waits
        delays = self.load_delays()
        stages = _stage_of_each(events)
        counts: dict[str, int] = {}
        for name in stages:
            counts[name] = counts.get(name, 0) + 1
        return [delays.get(name, delays.get("default", 0)) / 1000 / counts[name] for name in stages]

    async def replay_events(self, run_id: str, *, mode: c.ReplayMode = c.ReplayMode.FIXED) -> AsyncIterator[c.Event]:
        """The replay driver: stored events re-emitted with the pacing of the mode, no model call."""
        events = self.read_events(run_id)
        for wait, event in zip(self.replay_delays(events, mode=mode), events):
            if wait > 0:
                await self.sleep(wait)
            yield event

    # Rebuild and verify ------------------------------------------------------------------------------------------------

    def rebuild(self, run_id: str, *, out: Path | None = None, thresholds: dict[str, Any] | None = None) -> dict[str, str]:
        """Regenerate the derived files from the frozen facts with no network call; {file: sha256 | reason}."""
        run = self.load_run(run_id)
        th = thresholds or self.run_th(run)
        if out is None:
            return self._rebuild_in_place(run, th)
        out = Path(out)
        target = out / run_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(paths.run_dir(run_id), target, ignore=shutil.ignore_patterns("quarantine", "uploads", "*.orig", "*.part", PENDING_EVENTS_NAME))
        rows = [r for r in self.ledger_rows() if r.get("run_id") == run_id]  # the sheet prints the run's ledger totals
        (out / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        with _runs_root(out):
            return self._rebuild_in_place(run, th)

    def _rebuild_in_place(self, run: c.RunManifest, th: dict[str, Any]) -> dict[str, str]:
        run_id = run.run_id
        result: dict[str, str] = {name: NOT_REBUILT for name in ("marks.json", "mark_tracks.json", "narration.json", "claim.json")}
        result["coverage.json"] = self._rebuild_coverage(run, th)
        if run.kind == c.RunKind.PAIR and self.has_dep("match.derive_match"):
            try:
                self.dep("match.derive_match")(run_id, th=th, second_look=None)
                result["matches.json"] = sha256_path(paths.matches_json(run_id))
                result["verdicts.json"] = sha256_path(paths.verdicts_json(run_id))
            except Exception as exc:
                result["matches.json"] = result["verdicts.json"] = f"failed: {type(exc).__name__}: {exc}"
        if self.has_dep("sheet.write_sheet"):
            try:
                self.dep("sheet.write_sheet")(run_id, th=th)
                result["sheet.json"] = sha256_path(paths.sheet_json(run_id))
            except Exception as exc:
                result["sheet.json"] = f"failed: {type(exc).__name__}: {exc}"
        return result

    def _rebuild_coverage(self, run: c.RunManifest, th: dict[str, Any]) -> str:
        """coverage.json from panels, tracks, keyframes and the lane summaries, the way panels.derive's last call
        built it (odometer and the measured rerun seconds carried over from the file)."""
        run_id = run.run_id
        needed = (paths.panels_json(run_id), paths.tracks_json(run_id), paths.keyframes_json(run_id))
        if not all(p.is_file() for p in needed):
            return "not rebuilt: panels.json, tracks.json or keyframes.json missing"
        if not self.has_dep("panels.coverage_for_role") or not self.has_dep("panels.plan_topup"):
            return NOT_REBUILT
        try:
            panels = c.read_json_list(paths.panels_json(run_id), c.Panel)
            tracks = c.read_json_list(paths.tracks_json(run_id), c.PanelTrack)
            keyframes = c.read_json_list(paths.keyframes_json(run_id), c.Keyframe)
            prior = c.read_json(paths.coverage_json(run_id), c.Coverage) if paths.coverage_json(run_id).is_file() else c.Coverage()
            fps = run.fps or self._any_fps(run)
            roles = sorted({t.role for t in tracks} | set(prior.per_role), key=role_order)
            per_role = {
                role: self.dep("panels.coverage_for_role")(panels, tracks, keyframes, role=role, fps=fps, lane_status=self.lane_status(run_id, role), th=th)
                for role in roles
            }
            coverage = c.Coverage(per_role=per_role, odometer=prior.odometer, rerun_seconds=prior.rerun_seconds)
            coverage = self.dep("panels.plan_topup")(coverage, th)
            return c.write_json(paths.coverage_json(run_id), coverage)
        except Exception as exc:
            return f"failed: {type(exc).__name__}: {exc}"

    def _any_fps(self, run: c.RunManifest) -> float:
        for role in self.expected_roles(run):
            video = self.load_video(run.run_id, role)
            if video is not None and video.proxy is not None:
                return video.proxy.fps
        return float(self.run_th(run)["media"]["proxy_fps"])

    def verify(self, run_id: str, *, out: Path | None = None) -> list[str]:
        """Problems found: a rebuilt derived file that differs, a broken chain, a manifest entry that changed."""
        problems: list[str] = []
        scratch = Path(out) if out is not None else paths.runs_root() / "_verify"
        before = {f"derived/{name}": sha256_path(paths.derived_dir(run_id) / name) for name in derived_present(run_id)}
        if paths.sheet_json(run_id).is_file():
            before["sheet/sheet.json"] = sha256_path(paths.sheet_json(run_id))
        rebuilt = self.rebuild(run_id, out=scratch)
        for name, value in rebuilt.items():
            if len(value) != 64 or not all(ch in "0123456789abcdef" for ch in value):
                continue
            key = "sheet/sheet.json" if name == "sheet.json" else f"derived/{name}"
            if before.get(key, "") not in ("", value):
                problems.append(f"rebuild differs: {key}")
        manifest = paths.bundle_dir(run_id) / "manifest.json"
        if manifest.is_file() and hasattr(self.events, "verify_chain"):
            if not self.events.verify_chain(run_id, manifest_sha256=sha256_path(manifest)):
                problems.append("event chain broken")
        if self.has_dep("seal.verify_run"):
            try:
                for problem in self.dep("seal.verify_run")(run_id):
                    if problem.startswith("event chain broken") and any(p.startswith("event chain broken") for p in problems):
                        continue
                    if problem not in problems:
                        problems.append(problem)
            except Exception as exc:
                problems.append(f"seal.verify_run failed: {type(exc).__name__}: {exc}")
        return problems


# Module-level helpers ------------------------------------------------------------------------------------------------


def _exists(path: Path) -> bool:
    return Path(path).is_file()


def derived_present(run_id: str) -> list[str]:
    return [name for name in paths.DERIVED_FILES if (paths.derived_dir(run_id) / name).is_file()]


def _assemble(chunks: list[Path], target: Path) -> str:
    """Concatenate the chunks into target while hashing; returns the hex sha256 of what was written."""
    digest = hashlib.sha256()
    tmp = target.with_suffix(target.suffix + ".part")
    with open(tmp, "wb") as out:
        for chunk in chunks:
            with open(chunk, "rb") as fh:
                while block := fh.read(1 << 20):
                    digest.update(block)
                    out.write(block)
    os.replace(tmp, target)
    return digest.hexdigest()


def _stage_of_each(events: list[c.Event]) -> list[str]:
    """The stage name each event belongs to: the latest `stage` event's name, 'created' before the first."""
    names = []
    current = "created"
    for event in events:
        if event.kind == c.EventKind.STAGE:
            current = str(event.payload.get("name", current))
        names.append(current)
    return names


@contextmanager
def _runs_root(root: Path) -> Iterator[None]:
    """Point paths.RUNS_DIR at another root for the duration (how a rebuild writes into --out)."""
    saved = paths.RUNS_DIR
    paths.RUNS_DIR = Path(root)
    try:
        yield
    finally:
        paths.RUNS_DIR = saved


async def _http_get_status(url: str, headers: dict[str, str]) -> int:
    import httpx

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, headers=headers)
        return response.status_code


class FileEventLog:
    """The eventlog contract (CONTRACTS.md M3) as a stand-in until walkaround.eventlog lands: a hash-chained
    events.ndjson per run, prev of seq 0 = sha256 of bundle/manifest.json, in-process subscribers for the SSE tail.
    Appends may come from worker threads (the media stage), so the head is cached under a lock and subscribers are
    woken through their own loop."""

    def __init__(self) -> None:
        self._heads: dict[str, tuple[int, str]] = {}
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[c.Event]]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _path(run_id: str) -> Path:
        return paths.run_dir(run_id) / "events.ndjson"

    def read(self, run_id: str, *, since_seq: int = -1) -> Iterator[c.Event]:
        path = self._path(run_id)
        if not path.is_file():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = c.Event.model_validate_json(line)
                if event.seq > since_seq:
                    yield event

    def head_hash(self, run_id: str) -> str | None:
        head = self._head(run_id)
        return head[1] if head else None

    def _head(self, run_id: str) -> tuple[int, str] | None:
        if run_id not in self._heads:
            last = None
            for last in self.read(run_id):
                pass
            if last is None:
                return None
            self._heads[run_id] = (last.seq, last.hash)
        return self._heads[run_id]

    def append(self, run_id: str, kind: c.EventKind, payload: dict[str, Any], *, ts: str | None = None) -> c.Event:
        with self._lock:
            head = self._head(run_id)
            if head is None:
                manifest = paths.manifest_json(run_id)
                if not manifest.is_file():
                    raise FileNotFoundError("bundle/manifest.json must exist before seq 0: its sha256 is the chain genesis")
                seq, prev = 0, sha256_path(manifest)
            else:
                seq, prev = head[0] + 1, head[1]
            line = {"seq": seq, "ts": ts or utc_now(), "run_id": run_id, "kind": kind.value, "payload": payload, "prev": prev}
            event = c.Event(**line, hash=c.chain_hash(prev, line))
            with open(paths.events_path(run_id), "a", encoding="utf-8") as fh:
                fh.write(event.to_json() + "\n")
            self._heads[run_id] = (seq, event.hash)
            waiting = list(self._subscribers.get(run_id, []))
        for loop, queue in waiting:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                continue  # that subscriber's loop is gone
        return event

    async def subscribe(self, run_id: str, *, since_seq: int = -1) -> AsyncIterator[c.Event]:
        """History after since_seq, then live until a terminal event (or at once when the run already ended)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[c.Event] = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(run_id, []).append((loop, queue))
        try:
            last = since_seq
            terminal = False
            for event in self.read(run_id, since_seq=since_seq):
                last = event.seq
                terminal = event.kind in TERMINAL_EVENTS
                yield event
            if terminal or self._ended(run_id):
                return
            while True:
                event = await queue.get()
                if event.seq <= last:
                    continue
                last = event.seq
                yield event
                if event.kind in TERMINAL_EVENTS:
                    return
        finally:
            with self._lock:
                self._subscribers[run_id].remove((loop, queue))

    def _ended(self, run_id: str) -> bool:
        last = None
        for last in self.read(run_id):
            pass
        return last is not None and last.kind in TERMINAL_EVENTS

    def verify_chain(self, run_id: str, *, manifest_sha256: str) -> bool:
        prev = manifest_sha256
        for i, event in enumerate(self.read(run_id)):
            if event.seq != i or event.prev != prev or not event.check():
                return False
            prev = event.hash
        return True


__all__ = [
    "CapsExceeded",
    "FileEventLog",
    "HashMismatch",
    "MissingChunks",
    "NoSpace",
    "Orchestrator",
    "PENDING_EVENTS_NAME",
    "STAGES",
    "STAGE_BY_NAME",
    "Stage",
    "TERMINAL_EVENTS",
    "TERMINAL_STATUSES",
    "StageGap",
    "app_commit",
    "derived_present",
    "et_day_of",
    "new_run_id",
    "parse_ts",
    "role_order",
    "sha256_path",
    "utc_now",
]
