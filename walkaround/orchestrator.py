"""The run state machine (M14; docs/ARCHITECTURE.md sections 5, 7, 9 and 14; docs/CONTRACTS.md).

The orchestrator sequences a run through named stages; it never renders and never computes a verdict. Other lanes
are reached through dep("module.function"), imported on first use, so a lane that is not written yet fails at call
time as a recorded gap instead of at import, and a test can inject a fake for any one of them.

State is the run directory only. A stage counts as done when its files say so (manifest.json plus the seal clock,
proxy records, lane summaries, each role's entry in coverage.json, every keyframe's marks_status, and so on), so an
interrupted run resumes at the first stage whose files are missing; once a stage runs again, every later stage runs
again, since its inputs may have changed. Lanes resume one by one: a noun with no summary, or with a FAILED summary
while the unblurred proxy still exists, is sent again; a lane_done summary is never sent again.

Model-era facts are frozen (lanes and indexes, spark.json, voice.json, tracks.json, keyframes.json); rebuild
regenerates coverage, matches, verdicts and the sheet from them with no network call, under the thresholds frozen in
run.json, and once the run has closed only into another directory (manifest_run.json seals the files it would
rewrite). The sheet prints run.json's events_head_hash, which is set right before the sheet stage and left alone
after, so a rebuild reads the head the sheet printed; manifest_run.json carries the head at close.

Events and the chain genesis: section 4 makes the prev of seq 0 the sha256 of bundle/manifest.json, which exists only
once the seal has hashed the originals. Events before that (run.created, upload.started, upload.done, the sealing
stage) wait in events.pending.ndjson and are chained in order, with their own timestamps, the moment manifest.json is
written: seq 0 is run.created and seal.manifest follows the upload events. The seal never waits on a readable file (an
upload ffprobe cannot read is sealed with the camera time not present and fails at the media stage), so a failure
reaches the chain; a run that still ends before the chain opens shows its waiting events, unchained, numbered as the
chain would have numbered them (visible_events).

Money: the per-run ceiling and the daily caps are read from runs/ledger.jsonl. A created run holds its daily slot for
RESERVATION_S; when its uploads complete it is admitted again against the pause, the card ceiling and the caps, and a
run refused there keeps none of its unblurred bytes. Before a role's lanes start, the lanes that would push the run
past its ceiling (estimated from the proxy's frame count at the thresholds cost, keeping room for the keyframe image
checks) are written as not_run; the blur lanes come first in the fixed order, so they are the last to be cut.

Privacy at the edges: the display proxy is rendered only when the walk's license plate and face lanes completed (with
or without matches), from tracks rebuilt out of those lanes, so a deadline or ceiling that skips panels.derive cannot
produce an unblurred "blurred" proxy. Otherwise the walk has no display video and its unblurred proxy is deleted at run
end. A fatal failure still runs run-end deletion before run.failed.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import secrets
import shutil
import threading
import time
import wave
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import contracts as c
from . import ledger, paths
from .config import BASE_URL, REPO_DIR, redact

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
CREATABLE_KINDS = (c.RunKind.PAIR, c.RunKind.PICKUP_CHECK)  # top-ups and claims attach to an existing run
SAM_MODULES = {c.SamEngine.META_MODEL_API: "sam_client", c.SamEngine.FAL: "fal_sam", c.SamEngine.MODAL: "modal_sam3"}
STAGING_DIR_NAME = "_uploads"  # runs/_uploads/<upload_id>/chunks/<n>: chunks sent without a run id
PENDING_EVENTS_NAME = "events.pending.ndjson"  # pre-seal events waiting for the chain genesis
RESERVATION_S = 3600  # a created run holds its daily slot this long while its uploads come in
FRAME_EVENTS_PER_VIDEO_S = 5  # sam.frame events per lane per second of video; every frame is in the lane file
UNREADABLE_VIDEO = "could not read this video; on iPhone set Most Compatible and re-record"  # section 14 item 6
ENV_LIMITS = {
    "PICKUP_RUNS_PER_DAY": "pickup_runs_per_day",
    "PAIR_RUNS_PER_DAY": "pair_runs_per_day",
    "RUN_SPEND_CEILING_CENTS": "run_spend_ceiling_cents",
    "RUN_RETENTION_HOURS": "run_retention_hours",
}
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")  # a chosen run id; _ and . prefixes are the server's own folders
SAMPLE_INDEX_NAME = "index.json"  # samples/index.json: the order of the samples, with named slots (scripts/make_sample.py)
REQUIRED_NOUNS = (c.Noun.CAR, *c.PRIVACY_NOUNS)  # the car filter and the blur need these lanes on every run
REUSED_CODE = "reused_identical_request"
WHOLE_FRAME = ""  # a stand-in run's car prompt: no car request; the whole frame is the box every panel must sit inside
WHOLE_FRAME_CODE = "whole_frame_container"


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


def write_json_atomic(path: Path, value: Any) -> str:
    """contracts.write_json's exact bytes through a temp file and a rename, so a reader never sees half a file."""
    data = c.json_bytes(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def _plain(value: Any) -> Any:
    """Event payloads and the frozen thresholds hold plain JSON values only, in their canonical (rounded) form."""
    return json.loads(c.canonical_json(value))


def _scrubbed(value: Any) -> Any:
    """Every string in a payload with key values and key shapes replaced; events reach the browser as they are."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_scrubbed(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrubbed(v) for k, v in value.items()}
    return value


def _error_text(exc: BaseException) -> str:
    """The first real error, not the TaskGroup wrapper around it; a sentence written for the renter as it is."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    if isinstance(exc, VideoUnreadable):
        return redact(str(exc))
    return redact(f"{type(exc).__name__}: {exc}")


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


class UploadRefused(Exception):
    """A finalize the run cannot take: the role is not one of its walks, or that walk is already sealed."""


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


class VideoUnreadable(Exception):
    """ffprobe or ffmpeg could not read an upload; the text is the section 14 item 6 sentence plus the cause."""


class RebuildRefused(ValueError):
    """A rebuild that would overwrite the run itself, the global ledger or a closed run's sealed files."""


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
    Stage("seal", c.RunStatus.SEALING, 5, model=False, fatal=True),  # no manifest, no chain, no model call
    Stage("media", c.RunStatus.CONVERTING, 15, model=False, fatal=True),
    Stage("files", c.RunStatus.CONVERTING, 20, model=False, fatal=False),
    Stage("lanes", c.RunStatus.TRACKING, 45, model=True, fatal=False),
    Stage("panels", c.RunStatus.TRACKING, 55, model=True, fatal=False),  # needs tracks, so before the blur
    Stage("blur", c.RunStatus.BLURRING, 60, model=False, fatal=False),
    Stage("marks", c.RunStatus.MARKING, 70, model=True, fatal=False),
    Stage("narration", c.RunStatus.STAYSPUT, 74, model=True, fatal=False),  # before Stays Put, so seed marks get clips
    Stage("staysput", c.RunStatus.STAYSPUT, 78, model=True, fatal=False),
    Stage("match", c.RunStatus.MATCHING, 88, model=False, fatal=False, pair_only=True),
    Stage("release", c.RunStatus.SHEET, 92, model=False, fatal=False),  # deletions before the sheet prints their clocks
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
        limits: dict[str, int] | None = None,
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
        self.limits = self._limits(limits or {})
        self._events = events
        self._run_sem: asyncio.Semaphore | None = None
        self._media_sem: asyncio.Semaphore | None = None
        self._run_locks: dict[str, asyncio.Lock] = {}
        self._io_lock = threading.RLock()
        self._role_lock = threading.Lock()
        self._reserved_roles: set[tuple[str, str]] = set()
        self._queue: list[str] = []
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._deadline: dict[str, float] = {}
        self._decode_pool: Any | None = None
        self._checks_memo: dict[str, dict[str, Any]] = {}  # run_id -> frame digest -> image check answers
        self._off_car: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}  # run_id -> (tracks.json stamp, view)
        self._off_car_lock = threading.Lock()
        self.paused = False
        self.probe_ok: bool | None = None
        self._probe_failures = 0
        self.http_get: Callable[[str, dict[str, str]], Awaitable[int]] = _http_get_status
        self.stage_fns: dict[str, Callable[[c.RunManifest, bool], Awaitable[None]]] = {
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
            "release": self._stage_release,
            "sheet": self._stage_sheet,
            "close": self._stage_close,
        }

    @classmethod
    def from_env(cls, th: dict[str, Any] | None = None, **kwargs: Any) -> Orchestrator:
        """The server's orchestrator: thresholds from config/, engine, workers and the section 9 limits from the env."""
        th = th or c.load_thresholds()
        env, o = os.environ, th["orchestrator"]
        limits = {key: int(env[name]) for name, key in ENV_LIMITS.items() if env.get(name, "").strip()}
        return cls(
            th=th,
            engine=c.SamEngine(env.get("SAM_ENGINE", "meta-model-api")),
            max_concurrent_runs=int(env.get("MAX_CONCURRENT_RUNS", o["max_concurrent_runs"])),
            media_workers=int(env.get("MEDIA_WORKERS", o["media_workers"])),
            decode_workers=int(env.get("DECODE_WORKERS", o["decode_workers"])),
            limits=limits,
            **kwargs,
        )

    def _limits(self, overrides: dict[str, int]) -> dict[str, int]:
        """Operational limits; the env may change them without touching the frozen thresholds or the config hash."""
        o = self.th["orchestrator"]
        base = {
            "pickup_runs_per_day": int(o["pickup_runs_per_day"]),
            "pair_runs_per_day": int(o["pair_runs_per_day"]),
            "run_spend_ceiling_cents": int(o["run_spend_ceiling_cents"]),
            "daily_card_ceiling_cents": int(o["daily_card_ceiling_cents"]),
            "run_retention_hours": int(self.th["privacy"]["run_retention_hours"]),
        }
        base.update({k: int(v) for k, v in overrides.items()})
        return base

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

    def sam(self, name: str) -> Callable[..., Any]:
        """A SAM client function for this server's engine (SAM_ENGINE flips every video lane, section 14 item 33)."""
        return self.dep(f"{SAM_MODULES[self.engine]}.{name}")

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

    def run_lock(self, run_id: str) -> asyncio.Lock:
        """Serializes the writers of one run's derived files: the pipeline's stages and a top-up."""
        lock = self._run_locks.get(run_id)
        if lock is None:
            lock = self._run_locks[run_id] = asyncio.Lock()
        return lock

    # run.json and videos ---------------------------------------------------------------------------------------------

    def load_run(self, run_id: str) -> c.RunManifest:
        return c.read_json(paths.run_json(run_id), c.RunManifest)

    def save_run(self, run: c.RunManifest) -> None:
        with self._io_lock:
            write_json_atomic(paths.run_json(run.run_id), run)

    def update_run(self, run_id: str, change: Callable[[c.RunManifest], Any]) -> c.RunManifest:
        """Read, change and write run.json under one lock (media work runs in threads)."""
        with self._io_lock:
            run = self.load_run(run_id)
            change(run)
            write_json_atomic(paths.run_json(run_id), run)
            return run

    def run_exists(self, run_id: str) -> bool:
        return (paths.runs_root() / run_id / "run.json").is_file()

    def run_th(self, run: c.RunManifest) -> dict[str, Any]:
        """The thresholds a run computes with: the dict frozen in its run.json, else this server's."""
        return run.thresholds or self.th

    def list_runs(self) -> list[c.RunManifest]:
        """Every run with a run.json, newest first."""
        runs = []
        for entry in paths.runs_root().iterdir():
            if entry.is_dir() and not entry.name.startswith(("_", ".")) and (entry / "run.json").is_file():
                try:
                    runs.append(self.load_run(entry.name))
                except Exception:
                    continue
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def load_video(self, run_id: str, role: str) -> c.VideoRecord | None:
        path = paths.video_json(run_id, role)
        return c.read_json(path, c.VideoRecord) if path.is_file() else None

    def save_video(self, run_id: str, video: c.VideoRecord) -> None:
        with self._io_lock:
            write_json_atomic(paths.video_json(run_id, video.role), video)

    def update_video(self, run_id: str, role: str, change: Callable[[c.VideoRecord], Any]) -> c.VideoRecord:
        with self._io_lock:
            video = self.load_video(run_id, role)
            if video is None:
                raise FileNotFoundError(f"videos/{role}.json missing; was the upload finalized?")
            change(video)
            write_json_atomic(paths.video_json(run_id, role), video)
            return video

    def video_roles(self, run_id: str) -> list[str]:
        """Every role with a videos/<role>.json: the walks plus any top-up clip."""
        return sorted((p.stem for p in paths.videos_dir(run_id).glob("*.json")), key=role_order)

    @staticmethod
    def expected_roles(run: c.RunManifest) -> tuple[str, ...]:
        return WALK_ROLES[run.kind]

    def uploads_complete(self, run: c.RunManifest) -> bool:
        return all(paths.quarantine_original(run.run_id, role).is_file() for role in self.expected_roles(run))

    def reserve_role(self, run_id: str, prefix: str = "topup") -> str:
        """The next free <prefix>_<n>, held in memory until release_role so two top-ups never share a role."""
        with self._role_lock:
            taken = {p.stem for p in paths.videos_dir(run_id).glob(f"{prefix}_*.json")}
            taken |= {role for rid, role in self._reserved_roles if rid == run_id}
            n = 1
            while f"{prefix}_{n}" in taken:
                n += 1
            role = f"{prefix}_{n}"
            self._reserved_roles.add((run_id, role))
            return role

    def release_role(self, run_id: str, role: str) -> None:
        with self._role_lock:
            self._reserved_roles.discard((run_id, role))

    def role_reserved(self, run_id: str, role: str) -> bool:
        with self._role_lock:
            return (run_id, role) in self._reserved_roles

    # Events ------------------------------------------------------------------------------------------------------------

    def chain_open(self, run_id: str) -> bool:
        """True once the chain genesis exists (manifest.json) or the chain has already started."""
        if (paths.runs_root() / run_id / "bundle" / "manifest.json").is_file():
            return True
        events_file = paths.runs_root() / run_id / "events.ndjson"
        return events_file.is_file() and events_file.stat().st_size > 0

    def emit(self, run_id: str, kind: c.EventKind, payload: dict[str, Any]) -> c.Event:
        """eventlog.append once the chain is open; before the seal the event waits in events.pending.ndjson."""
        plain = _scrubbed(_plain(payload))
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
        self.update_run(run_id, lambda r: setattr(r, "status", status))
        self.emit(run_id, c.EventKind.STAGE, {"name": status.value, "pct": pct})

    def read_events(self, run_id: str, *, since_seq: int = -1) -> list[c.Event]:
        return list(self.events.read(run_id, since_seq=since_seq))

    def visible_events(self, run_id: str, *, since_seq: int = -1) -> list[c.Event]:
        """What a browser or a replay is shown: the chain once it opened; before that, the waiting events, unchained
        (prev and hash empty) and numbered from 0, the seqs the chain gives them when manifest.json lands. For a run
        that failed before the seal this is its whole story, run.failed included."""
        if self.chain_open(run_id):
            return self.read_events(run_id, since_seq=since_seq)
        waiting = [e.model_copy(update={"seq": i}) for i, e in enumerate(self.pending_events(run_id))]
        return [e for e in waiting if e.seq > since_seq]

    # Create and upload -------------------------------------------------------------------------------------------------

    async def create_run(
        self,
        *,
        mode: c.RunMode,
        kind: c.RunKind,
        keep_plate: bool,
        live_or_replay: c.LiveOrReplay = c.LiveOrReplay.LIVE,
        run_id: str | None = None,
        noun_list: list[c.Noun] | None = None,
        prompts: dict[str, str] | None = None,
        keep: bool = False,
        footage_credit: str | None = None,
        tire_code: str | None = None,
    ) -> c.RunManifest:
        """ULID run_id (or a chosen one that does not exist yet), the layout on disk, thresholds frozen with their hash,
        run.json, run.created. noun_list narrows the lanes (car, license plate and face always stay); prompts gives a
        lane stand-in text for a test clip that has no car (see prompt_text). keep (sample and eval runs) survives the
        retention sweep and makes the run public; footage_credit is the credit line shown under the walk player;
        tire_code (already normalized, ruler.normalize_tire_code) is the sidewall size the wheel ruler reads."""
        nouns, prompts = _check_nouns(noun_list, prompts)
        if run_id is not None:
            _check_new_run_id(run_id)
        await self.ensure_space()
        run_id = run_id or new_run_id()
        paths.ensure_run_layout(run_id, roles=WALK_ROLES[kind])
        created = self.now()
        retention = parse_ts(created) + timedelta(hours=self.limits["run_retention_hours"])
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
            spend_ceiling_cents=self.limits["run_spend_ceiling_cents"],
            sam_engine=self.engine,
            noun_list=nouns,
            prompts=prompts,
            keep=keep,
            footage_credit=" ".join(footage_credit.split()) if footage_credit else None,
            tire_code=tire_code or None,
        )
        self.save_run(run)
        payload: dict[str, Any] = {"kind": kind.value, "mode": mode.value, "keep_plate": keep_plate, "config_hash": run.config_hash}
        if prompts:
            payload["prompts"] = prompts
        self.emit(run_id, c.EventKind.RUN_CREATED, payload)
        return run

    @staticmethod
    def prompt_text(run: c.RunManifest, noun: c.Noun) -> str:
        """What a lane is sent with: the noun itself, or the run's stand-in text for it (a test clip without a car)."""
        return run.prompts.get(noun.value, noun.value)

    @staticmethod
    def whole_frame_container(run: c.RunManifest) -> bool:
        """A stand-in run without a container noun: no car request, the whole frame is the car box."""
        return run.prompts.get(c.Noun.CAR.value) == WHOLE_FRAME

    def free_bytes(self) -> int:
        return shutil.disk_usage(paths.runs_root()).free

    async def ensure_space(self) -> None:
        """Under the floor the sweeper runs at once; still under it, new runs are refused (section 14 item 27)."""
        floor = int(self.th["orchestrator"]["volume_min_free_bytes"])
        if self.free_bytes() >= floor:
            return
        await self.sweep()
        free = self.free_bytes()
        if free < floor:
            raise NoSpace(f"the server has {free // 1_000_000} MB free and a new run needs {floor // 1_000_000} MB; try again later")

    def chunk_path(self, upload_id: str, index: int, run_id: str | None = None) -> Path:
        """Where one chunk lands: under the run when the client names it, else the staging area."""
        if run_id:
            return paths.upload_chunk_path(run_id, upload_id, index)
        path = paths.runs_root() / STAGING_DIR_NAME / upload_id / "chunks" / f"{index:06d}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def chunks_dir(self, upload_id: str, run_id: str) -> Path:
        under_run = paths.runs_root() / run_id / "uploads" / upload_id / "chunks"
        if under_run.is_dir() and any(under_run.iterdir()):
            return under_run
        return paths.runs_root() / STAGING_DIR_NAME / upload_id / "chunks"

    def _check_role(self, run: c.RunManifest, role: str, sha256_browser: str | None) -> c.VideoRecord | None:
        """The record to return for a repeated finalize of the same bytes; UploadRefused when the run cannot take it."""
        existing = self.load_video(run.run_id, role)
        same = existing is not None and existing.hash_match and sha256_browser is not None and existing.sha256_server == sha256_browser.lower()
        if role in self.expected_roles(run):
            if same and paths.quarantine_original(run.run_id, role).is_file():
                return existing  # a finalize repeated after a lost response: idempotent
            if run.status != c.RunStatus.CREATED or self.chain_open(run.run_id):
                raise UploadRefused(f"the {role} walk of this run is already sealed; start a new run to send another file")
            return None
        if role.startswith("topup_"):
            # only the top-up route's own reservation: a bare finalize would leave an original nothing ever processes
            if not self.role_reserved(run.run_id, role):
                raise UploadRefused("a top-up clip is sent to POST /api/runs/<run_id>/topup, not through finalize")
            return existing if same else None
        raise UploadRefused(f"role {role} is not a walk of this {run.kind.value} run")

    async def finalize_upload(
        self,
        run_id: str,
        *,
        upload_id: str,
        role: str,
        sha256_browser: str | None,
        bytes: int,
        filename: str,
    ) -> c.VideoRecord:
        """Reassemble the chunks into quarantine, hash on receipt, 409 material on a mismatch, upload.done."""
        run = self.load_run(run_id)
        repeated = self._check_role(run, role, sha256_browser)
        if repeated is not None:
            return repeated
        size = bytes
        if size > int(self.th["media"]["upload_max_bytes"]):
            raise UploadRefused("the file is over 1 GiB; the server refuses it")
        expected = max(1, -(-size // int(self.th["media"]["chunk_bytes"])))
        chunks_dir = self.chunks_dir(upload_id, run_id)
        present = {int(p.name) for p in chunks_dir.glob("[0-9]*") if p.name.isdigit()} if chunks_dir.is_dir() else set()
        missing = [i for i in range(expected) if i not in present]
        if missing:
            raise MissingChunks(missing)
        self.emit(run_id, c.EventKind.UPLOAD_STARTED, {"role": role, "upload_id": upload_id, "chunks": expected})
        target = paths.quarantine_original(run_id, role)
        digest = await asyncio.to_thread(_assemble, [chunks_dir / f"{i:06d}" for i in range(expected)], target)
        shutil.rmtree(chunks_dir.parent, ignore_errors=True)
        browser = sha256_browser.lower() if sha256_browser else None
        record = c.VideoRecord(
            role=role,
            original_filename=filename,
            bytes=target.stat().st_size,
            sha256_browser=browser,
            sha256_server=digest,
            hash_match=None if browser is None else digest == browser,
            received_at=self.now(),
            chunks_expected=expected,
            chunks_received=expected,
            quarantine_path=paths.rel(run_id, target),
        )
        self.save_video(run_id, record)
        if record.hash_match is False:
            target.unlink(missing_ok=True)  # bytes that do not match are never kept, let alone processed (section 14 item 8)
            raise HashMismatch(browser or "", digest)
        payload: dict[str, Any] = {"role": role, "bytes": record.bytes, "sha256": digest, "filename": filename, "hash_match": record.hash_match}
        if record.bytes > int(self.th["media"]["upload_warn_bytes"]):
            payload["warning"] = "over 500 MB; the proxy step will take longer"
        self.emit(run_id, c.EventKind.UPLOAD_DONE, payload)
        return record

    # Pipeline ----------------------------------------------------------------------------------------------------------

    def start(self, run_id: str) -> asyncio.Task[Any]:
        """Queue the run behind max_concurrent_runs and run it in the background; a running run is not started twice."""
        self._tasks = {k: t for k, t in self._tasks.items() if not t.done()}
        task = self._tasks.get(run_id)
        if task is not None:
            return task
        self._queue.append(run_id)
        task = asyncio.create_task(self._guarded(run_id), name=f"run-{run_id}")
        self._tasks[run_id] = task
        return task

    def active(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    def admit(self, run: c.RunManifest) -> None:
        """Raise CapsExceeded when a run that has not started may not start now. Its reservation lapses after
        RESERVATION_S, so the pause, the card ceiling and today's caps (this run aside) are checked again here."""
        if run.live_or_replay != c.LiveOrReplay.LIVE or run.status != c.RunStatus.CREATED or self.active(run.run_id):
            return
        self.check_can_start(run.kind, exclude=run.run_id)

    async def discard_uploads(self, run_id: str, *, reason: str) -> list[c.DeletionRecord]:
        """A run refused at start keeps none of its unblurred bytes: originals and chunks go with deletions.json rows.
        The run stays created, so the same files can be sent again later."""
        now = self.now()
        removed = await asyncio.to_thread(self._delete_local, run_id, now, True)
        rows = [r.model_copy(update={"reason": f"not started: {reason}"}) for r in removed]
        gone = {r.target for r in rows}
        for role in self.video_roles(run_id):
            video = self.load_video(run_id, role)
            if video is not None and video.quarantine_path in gone:
                self.update_video(run_id, role, lambda v: setattr(v, "quarantine_deleted_at", now))
        if rows:
            self.emit(run_id, c.EventKind.QUARANTINE_DELETED, {"files": sorted(gone), "reason": f"not started: {reason}"})
        return self._add_deletions(run_id, rows)

    async def _guarded(self, run_id: str) -> None:
        try:
            async with self.run_sem():
                if run_id in self._queue:
                    self._queue.remove(run_id)
                await self.run_pipeline(run_id)
        except Exception as exc:  # the pipeline reports its own failures; this is the last net
            try:
                await self._fail(run_id, "pipeline", _error_text(exc))
            except Exception:
                with suppress(Exception):
                    self.update_run(run_id, lambda r: setattr(r, "status", c.RunStatus.FAILED))
        finally:
            if run_id in self._queue:
                self._queue.remove(run_id)

    def queue_position(self, run_id: str) -> int | None:
        return self._queue.index(run_id) + 1 if run_id in self._queue else None

    def stages_for(self, run: c.RunManifest) -> list[Stage]:
        return [s for s in STAGES if not (s.pair_only and run.kind != c.RunKind.PAIR)]

    def stage_done(self, run: c.RunManifest, stage: Stage) -> bool:
        """Completion read from the files on disk (section 7: the files are the only state)."""
        run_id = run.run_id
        roles = self.expected_roles(run)
        derived = paths.derived_dir(run_id)
        name = stage.name
        if name == "seal":
            return paths.manifest_json(run_id).is_file() and run.clocks.sealed_at is not None
        if name == "media":
            return all((v := self.load_video(run_id, r)) is not None and v.proxy is not None for r in roles)
        if name == "files":
            return all(not self.missing_lanes(run, r) or self._file_id_valid(self.load_video(run_id, r)) for r in roles)
        if name == "lanes":
            return not any(self.missing_lanes(run, r) for r in roles)
        if name == "panels":
            return all(self._role_derived(run_id, r) for r in roles)
        if name == "blur":
            # settled: the display proxy exists, or the walk can never have one (its refusal is recorded)
            return all(
                (v := self.load_video(run_id, r)) is not None and v.proxy is not None and (v.proxy_blurred is not None or self._blur_refusal(run, r) is not None)
                for r in roles
            )
        if name == "marks":
            return (derived / "marks.json").is_file() and all(self._marks_done(run_id, r) for r in roles)
        if name == "staysput":
            if not (derived / "mark_tracks.json").is_file():
                return False  # written after each walk, so it alone cannot say every walk finished
            return not self.has_dep("staysput.role_done") or all(self.dep("staysput.role_done")(run_id, r) for r in roles)
        if name == "narration":
            return (derived / "narration.json").is_file()
        if name == "match":
            return (derived / "matches.json").is_file() and (derived / "verdicts.json").is_file()
        if name == "release":
            remote_gone = all(v.files_api is None or v.files_api.deleted_at is not None for r in self.video_roles(run_id) if (v := self.load_video(run_id, r)))
            return paths.manifest_run_json(run_id).is_file() and remote_gone and (run.keep_plate or not self._local_unblurred(run_id))
        if name == "sheet":
            return paths.sheet_html(run_id).is_file()
        if name == "close":
            if not paths.manifest_run_json(run_id).is_file() or self._local_unblurred(run_id):
                return False
            events = self.read_events(run_id)
            written = [e.seq for e in events if e.kind == c.EventKind.SHEET_WRITTEN]
            head = c.read_json(paths.manifest_run_json(run_id), c.SealManifest).events_head_hash
            return not written or next((e.seq for e in events if e.hash == head), -1) > written[-1]  # written after the sheet
        raise KeyError(name)

    def _local_unblurred(self, run_id: str) -> list[Path]:
        """Originals, temp files, upload chunks and any unblurred proxy still on disk: what run end deletes."""
        return self._local_originals(run_id) + self._unblurred_proxies(run_id)

    @staticmethod
    def _local_originals(run_id: str) -> list[Path]:
        root = paths.runs_root() / run_id
        return [p for top in ("quarantine", "uploads") if (root / top).is_dir() for p in (root / top).rglob("*") if p.is_file()]

    @staticmethod
    def _unblurred_proxies(run_id: str) -> list[Path]:
        """media/<role>.proxy.mp4 files no blurred display proxy replaced (a refused blur, a failed run)."""
        media = paths.runs_root() / run_id / "media"
        return sorted(p for p in media.glob("*.proxy.mp4") if p.is_file()) if media.is_dir() else []

    def _proxy_available(self, run_id: str, role: str) -> bool:
        """The unblurred proxy still exists, so a lane can be sent (again) on this walk."""
        video = self.load_video(run_id, role)
        return video is not None and video.proxy is not None and video.proxy_blurred is None and (paths.run_dir(run_id) / video.proxy.path).is_file()

    def missing_lanes(self, run: c.RunManifest, role: str) -> list[c.Noun]:
        """Nouns a resume sends: no summary yet, or a FAILED one while the unblurred proxy can still be sent."""
        retry_failed = self._proxy_available(run.run_id, role)
        out = []
        for noun in run.noun_list:
            summary = paths.lane_summary_path(run.run_id, role, noun)
            if not summary.is_file():
                out.append(noun)
            elif retry_failed and self._lane_record(summary) is not None and self._lane_record(summary).status == c.LaneStatus.FAILED:
                out.append(noun)
        return out

    @staticmethod
    def _lane_record(summary: Path) -> c.LaneRecord | None:
        try:
            return c.read_json(summary, c.LaneRecord)
        except Exception:
            return None

    def lane_status(self, run_id: str, role: str) -> dict[c.Noun, c.LaneStatus]:
        """Status per noun from the lane summaries on disk (what coverage prints as not_run or not_filmed)."""
        out: dict[c.Noun, c.LaneStatus] = {}
        for noun in c.LANE_NOUNS:
            summary = paths.lane_summary_path(run_id, role, noun)
            record = self._lane_record(summary) if summary.is_file() else None
            if record is not None:
                out[noun] = record.status
        return out

    def _file_id_valid(self, video: c.VideoRecord | None) -> bool:
        if video is None or video.files_api is None or video.files_api.deleted_at is not None:
            return False
        if video.files_api.expires_at is None:
            return True
        return parse_ts(video.files_api.expires_at) > parse_ts(self.now()) + timedelta(minutes=2)

    def _role_derived(self, run_id: str, role: str) -> bool:
        derived = paths.derived_dir(run_id)
        if not all((derived / n).is_file() for n in ("tracks.json", "panels.json", "keyframes.json", "coverage.json")):
            return False
        try:
            return role in c.read_json(paths.coverage_json(run_id), c.Coverage).per_role
        except Exception:
            return False

    def _keyframes(self, run_id: str) -> list[c.Keyframe]:
        path = paths.keyframes_json(run_id)
        return c.read_json_list(path, c.Keyframe) if path.is_file() else []

    def _marks_done(self, run_id: str, role: str) -> bool:
        """Every keyframe of the role that has a crop carries a marks_status from Spark (complete or truncated)."""
        return all(k.marks_status != c.MarksStatus.NOT_RUN for k in self._keyframes(run_id) if k.role == role and k.crop is not None)

    async def run_pipeline(self, run_id: str) -> c.RunManifest:
        """Stage by stage: skip what the files say is done, rerun everything after the first stage that runs, retry
        once, record gaps, and always reach the sheet unless a fatal stage failed."""
        # TODO(D1 E12/E13): time the seal on the deploy image. The deadline starts before it, and `ots stamp` may take
        # its 120 s timeout per file when the calendars hang (three files on a pair), which would cut lanes as not_run.
        try:
            return await self._run_stages(run_id)
        finally:
            self._checks_memo.pop(run_id, None)

    async def _run_stages(self, run_id: str) -> c.RunManifest:
        self._deadline[run_id] = self.clock() + float(self.th["orchestrator"]["run_deadline_s"])
        if self.chain_open(run_id):
            self.flush_pending(run_id)
        missing: list[str] = []
        redo = False
        for stage in self.stages_for(self.load_run(run_id)):
            async with self.run_lock(run_id):
                run = self.load_run(run_id)
                if not redo and self.stage_done(run, stage):
                    continue
                first, redo = not redo, True
                if stage.model and self._blocked(run, stage, missing):
                    continue
                self.set_stage(run_id, stage.status, stage.pct)
                error = await self._run_stage(run, stage, keep_done=first)
                if stage.model:
                    self._note_spend(run_id)
            if error is None:
                continue
            self._record_error(run_id, stage.name, error)
            if stage.fatal:
                return await self._fail(run_id, stage.name, error)
            missing.append(stage.name)
        return self._finish(run_id, missing)

    def _note_spend(self, run_id: str) -> None:
        cents = self.spend_cents(run_id)
        self.update_run(run_id, lambda r: setattr(r, "api_spend_cents", cents))
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

    async def _run_stage(self, run: c.RunManifest, stage: Stage, *, keep_done: bool) -> str | None:
        """One try plus request_retries; the error text when every try failed. keep_done lets the stage skip the roles
        it already finished: true for the first stage a resume runs (nothing upstream changed) and for every retry."""
        tries = 1 + int(self.th["orchestrator"]["request_retries"])
        error = None
        for attempt in range(tries):
            try:
                await self.stage_fns[stage.name](self.load_run(run.run_id), keep_done or attempt > 0)
                return None
            except StageGap as gap:
                return str(gap)
            except Exception as exc:
                error = _error_text(exc)
                if attempt + 1 < tries:
                    await self.sleep(0)
        return error

    def _record_error(self, run_id: str, stage: str, text: str) -> None:
        text = redact(text)
        self.update_run(run_id, lambda r: r.errors.append(c.RunError(stage=stage, text=text[:500])))

    async def _fail(self, run_id: str, stage: str, text: str) -> c.RunManifest:
        """A fatal failure: the error recorded, run-end deletion done anyway (Files API uploads, originals, chunks, any
        unblurred proxy; deletions.json closed), then run.failed and the failed status back to back, so whoever sees
        the status finds the event too."""
        text = redact(text)

        def record(run: c.RunManifest) -> None:
            if not any(e.stage == stage and e.text == text[:500] for e in run.errors):
                run.errors.append(c.RunError(stage=stage, text=text[:500]))

        self.update_run(run_id, record)
        try:
            async with self.run_lock(run_id):
                run = self.load_run(run_id)
                rows = await self._release(run, local=True)
                await asyncio.to_thread(self.dep("privacy.close_deletions"), run_id, rows, keep_plate=run.keep_plate, chosen_by=_chosen_by(run))
        except Exception as exc:  # deletion is best effort here; the sweeper and the sheet-less run still say why
            self._record_error(run_id, "release", f"run-end deletion after the failure did not finish: {_error_text(exc)}")
        self.emit(run_id, c.EventKind.RUN_FAILED, {"stage": stage, "text": text})
        return self.update_run(run_id, lambda r: setattr(r, "status", c.RunStatus.FAILED))

    def _finish(self, run_id: str, missing: list[str]) -> c.RunManifest:
        run = self.load_run(run_id)
        status = c.RunStatus.PARTIAL if missing else c.RunStatus.DONE
        if missing:
            tracked = self.tracked_counts(run)
            self.emit(run_id, c.EventKind.RUN_PARTIAL, {"missing": missing, "text": self._partial_text(run, tracked), "tracked": tracked})
        cents = self.spend_cents(run_id)

        def change(r: c.RunManifest) -> None:
            r.status = status
            r.api_spend_cents = cents

        self.update_run(run_id, change)
        self.emit(run_id, c.EventKind.STAGE, {"name": status.value, "pct": 100})
        self.emit(run_id, c.EventKind.RUN_COMPLETED, {"status": status.value, "api_spend_cents": cents})
        return self.load_run(run_id)

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

    async def _probe_video(self, run_id: str, role: str) -> c.VideoRecord:
        """ffprobe the original once; the record keeps the display size every SAM token carries."""
        video = self.load_video(run_id, role)
        if video is None:
            raise FileNotFoundError(f"videos/{role}.json missing; was the upload finalized?")
        if video.ffprobe is not None:
            return video
        try:
            info = await asyncio.to_thread(self.dep("video.probe"), paths.quarantine_original(run_id, role))
        except Exception as exc:
            raise VideoUnreadable(f"{UNREADABLE_VIDEO} ({role}: {_error_text(exc)})") from exc
        container = Path(video.original_filename).suffix.lstrip(".").lower() or "mp4"
        ffprobe = self.dep("video.ffprobe_info")(info, container=container)
        video = self.update_video(run_id, role, lambda v: setattr(v, "ffprobe", ffprobe))
        self.emit(run_id, c.EventKind.MEDIA_PROBED, {"role": role, "ffprobe": ffprobe})
        return video

    async def _stage_seal(self, run: c.RunManifest, keep_done: bool) -> None:
        run_id = run.run_id
        roles = self.expected_roles(run)
        originals = [paths.quarantine_original(run_id, r) for r in roles]
        absent = [r for r, p in zip(roles, originals) if not p.is_file()]
        if absent:
            raise FileNotFoundError(f"quarantine/{absent[0]}.orig is missing; was the upload finalized?")
        camera_time: str = c.CAMERA_TIME_NOT_PRESENT
        for role in roles:
            try:
                video = await self._probe_video(run_id, role)
            except Exception:
                continue  # sealed anyway, camera time not present: the media stage fails it with section 14 item 6
            if role == "pickup" and video.ffprobe is not None and video.ffprobe.creation_time != "not present":
                camera_time = video.ffprobe.creation_time
        manifest, clocks = await asyncio.to_thread(self.dep("seal.seal_run"), run_id, originals, now=self.now(), camera_time=camera_time)

        def change(r: c.RunManifest) -> None:
            r.clocks = c.ThreeClocks(
                sealed_at=clocks.sealed_at,
                first_model_call_at=r.clocks.first_model_call_at,
                sealed_minutes_after_camera=clocks.sealed_minutes_after_camera,
                camera_time=clocks.camera_time,
            )

        self.update_run(run_id, change)
        self.flush_pending(run_id)  # manifest.json exists now: run.created becomes seq 0 with its hash as prev
        self.emit(run_id, c.EventKind.SEAL_MANIFEST, {"sha256": sha256_path(paths.manifest_json(run_id)), "files": manifest.files})
        stamps = manifest.stamps
        if stamps.tsr_time:
            self.emit(run_id, c.EventKind.SEAL_TSR, {"tsr_time": stamps.tsr_time, "serial": stamps.tsa_serial, "issuer": stamps.tsa_issuer})
        if stamps.manifest_ots:
            self.emit(run_id, c.EventKind.SEAL_OTS, {"status": stamps.manifest_ots, "pickup": stamps.pickup_ots, "return": stamps.return_ots})
        self.set_stage(run_id, c.RunStatus.SEALED, SEALED_PCT)

    async def _stage_media(self, run: c.RunManifest, keep_done: bool) -> None:
        # gather waits for every walk, so a retry never races an encode still running in a thread
        results = await asyncio.gather(*(self._media_role(run, role, wav=True) for role in self.expected_roles(run)), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _media_role(self, run: c.RunManifest, role: str, *, wav: bool) -> None:
        """Proxy, pts map and (for a walk) the WAV, under the media semaphore shared by every run."""
        run_id = run.run_id
        video = await self._probe_video(run_id, role)
        if video.proxy is not None:
            return
        th_media = self.run_th(run)["media"]
        src = paths.quarantine_original(run_id, role)
        proxy = paths.proxy_path(run_id, role)
        wav_record = None
        async with self.media_sem():
            try:
                await asyncio.to_thread(
                    self.dep("video.make_proxy"), src, proxy, fps=float(th_media["proxy_fps"]), max_edge=int(th_media["proxy_max_edge"]),
                    preset=th_media["proxy_preset"], crf=int(th_media["proxy_crf"]),
                )
                record = await asyncio.to_thread(self.dep("video.proxy_record"), run_id, role, proxy)
            except Exception as exc:
                raise VideoUnreadable(f"{UNREADABLE_VIDEO} ({role}: {_error_text(exc)})") from exc
            if wav and (video.ffprobe is None or video.ffprobe.has_audio):
                try:
                    wav_path = await asyncio.to_thread(self.dep("video.extract_wav"), src, paths.wav_path(run_id, role), rate=int(th_media["wav_rate"]))
                    wav_record = c.WavRecord(path=paths.rel(run_id, wav_path), duration_s=_wav_seconds(wav_path), sha256=sha256_path(wav_path))
                except Exception:  # no audio track is a normal case; the narration stage says so (section 14 item 16)
                    wav_record = None

        def change(v: c.VideoRecord) -> None:
            v.proxy = record
            if wav_record is not None:
                v.wav = wav_record

        self.update_video(run_id, role, change)
        self.update_run(run_id, lambda r: setattr(r, "fps", r.fps if r.fps is not None else record.fps))
        self.emit(run_id, c.EventKind.MEDIA_PROXY_DONE, {"role": role, "proxy": record})

    async def _stage_files(self, run: c.RunManifest, keep_done: bool) -> None:
        """Uploads before the lanes. A walk byte-identical to an earlier one waits: its lanes come from that walk's
        answers, and any it cannot reuse uploads in _lanes_for_role."""
        for role in self.expected_roles(run):
            if self.reuse_source(run, role) is None and self._to_request(run, role, self.missing_lanes(run, role)):
                await self._ensure_file_id(run.run_id, role)

    async def _ensure_file_id(self, run_id: str, role: str) -> c.VideoRecord:
        """Upload the proxy to the Files API unless a live file_id exists; stamps first_model_call_at."""
        video = self.load_video(run_id, role)
        if video is None or video.proxy is None:
            raise FileNotFoundError(f"no proxy for {role}")
        if self._file_id_valid(video):
            return video
        proxy = paths.run_dir(run_id) / video.proxy.path
        if not proxy.is_file():
            raise FileNotFoundError(f"the unblurred {role} proxy was deleted at blur.done; no lane can be sent on it")
        record = await self.sam("upload_file")(proxy, run_id=run_id, expires_after_s=int(self.th["media"]["files_expires_after_s"]))
        video = self.update_video(run_id, role, lambda v: setattr(v, "files_api", record))
        self.update_run(run_id, lambda r: setattr(r.clocks, "first_model_call_at", r.clocks.first_model_call_at or record.uploaded_at))
        self.emit(run_id, c.EventKind.FILES_UPLOADED, {"role": role, "file_id": record.file_id, "expires_at": record.expires_at})
        return video

    async def _stage_lanes(self, run: c.RunManifest, keep_done: bool) -> None:
        gaps = []
        for role in self.expected_roles(run):
            gap = await self._lanes_for_role(run, role, self.missing_lanes(run, role), deadline=self._deadline.get(run.run_id))
            if gap:
                gaps.append(f"{role}: {gap}")
        self._update_responses_store(run.run_id)
        if gaps:
            raise StageGap("; ".join(gaps))

    def lane_cost_cents(self, run: c.RunManifest, frames: int) -> float:
        cost = self.run_th(run)["cost"]
        return max(int(frames), int(cost["sam_video_min_frames"])) * float(cost["sam_video_usd_per_frame"]) * 100

    def _image_reserve_cents(self, run: c.RunManifest) -> float:
        """Room kept for the keyframe image checks (plate and face on the top keyframes of every slot) per walk."""
        th = self.run_th(run)
        per_role = len(c.PanelId) * int(th["panels"]["keyframe_top_k"]) * 2 * float(th["cost"]["sam_image_usd"]) * 100
        return per_role * sum(1 for r in self.expected_roles(run) if not self._role_derived(run.run_id, r))

    def _affordable(self, run: c.RunManifest, nouns: list[c.Noun], frames: int) -> tuple[list[c.Noun], list[c.Noun]]:
        """The leading nouns whose lanes fit under the ceiling and the rest; the fixed order puts car, plate, face first."""
        room = run.spend_ceiling_cents - self.spend_cents(run.run_id) - self._image_reserve_cents(run)
        per_lane = self.lane_cost_cents(run, frames)
        fit = len(nouns) if per_lane <= 0 else max(0, min(len(nouns), int(room // per_lane)))
        return nouns[:fit], nouns[fit:]

    async def _lanes_for_role(self, run: c.RunManifest, role: str, nouns: list[c.Noun], *, deadline: float | None) -> str | None:
        """Send the lanes the role still lacks. Lanes the ceiling or the deadline stops are written not_run, never
        not_filmed; the return value names them for the run.partial list."""
        run_id = run.run_id
        if not nouns:
            return None
        if deadline is not None and self.clock() > deadline:
            reason = f"run deadline {int(self.th['orchestrator']['run_deadline_s'])} s passed before {len(nouns)} lanes"
            self._write_not_run(run_id, role, nouns, reason)
            return reason
        video = self.load_video(run_id, role)
        if video is None or video.proxy is None:
            raise FileNotFoundError(f"no proxy for {role}")
        if c.Noun.CAR in nouns and self.whole_frame_container(run):
            self._write_whole_frame_container(run, role)
            nouns = [n for n in nouns if n is not c.Noun.CAR]
        reused = set(self._reuse_lanes(run, role, nouns))
        nouns = [n for n in nouns if n not in reused]
        if not nouns:
            return None
        notes = []
        nouns, cut = self._affordable(self.load_run(run_id), nouns, video.proxy.frames)
        if cut:
            reason = f"spend ceiling {run.spend_ceiling_cents} cents: {len(cut)} lanes would pass it"
            self._write_not_run(run_id, role, cut, reason)
            notes.append(reason)
        if not nouns:
            return "; ".join(notes)
        try:
            video = await self._ensure_file_id(run_id, role)
        except Exception as exc:
            self._write_not_run(run_id, role, nouns, f"Files API upload failed: {_error_text(exc)}")
            raise  # one stage retry sends these again (a FAILED summary is resent while the proxy exists)
        assert video.proxy is not None and video.files_api is not None
        results: dict[str, tuple[c.LaneRecord, list[c.IndexFrame]]] = {}
        remaining = (deadline - self.clock()) if deadline is not None else float(self.th["orchestrator"]["run_deadline_s"])
        try:
            async with asyncio.timeout(max(1.0, remaining)) as scope:
                results = await self.sam("segment_video_many")(
                    video.files_api.file_id,
                    [(paths.lane_id(n), self.prompt_text(run, n)) for n in nouns],
                    run_id=run_id,
                    role=role,
                    frames_sent=video.proxy.frames,
                    media_size=(video.proxy.width, video.proxy.height),
                    metadata=self.sam("sam_metadata")(),
                    concurrency=int(self.th["orchestrator"]["sam_video_semaphore_per_run"]),
                    on_event=self._lane_event_sink(run_id, role, video.proxy.fps),
                )
        except TimeoutError:
            if not scope.expired():
                raise
            notes.append(f"run deadline {int(self.th['orchestrator']['run_deadline_s'])} s passed while lanes were running")
        failed = []
        for noun in nouns:
            lane = paths.lane_id(noun)
            summary = paths.lane_summary_path(run_id, role, noun)
            if lane in results:
                record, frames = results[lane]
                if record.status == c.LaneStatus.FAILED:
                    c.write_json(summary, record)  # a lane that raised inside the client wrote nothing itself
                    if not paths.lane_index_path(run_id, role, noun).is_file():
                        c.write_json(paths.lane_index_path(run_id, role, noun), [])
                self._emit_lane_done(run_id, role, lane, record, frames)
            elif (record := self._lane_record(summary) if summary.is_file() else None) is not None and record.status != c.LaneStatus.FAILED:
                frames = c.read_json_list(paths.lane_index_path(run_id, role, noun), c.IndexFrame) if paths.lane_index_path(run_id, role, noun).is_file() else []
                self._emit_lane_done(run_id, role, lane, record, frames)  # finished before the deadline cut the rest
                continue
            else:
                self._write_not_run(run_id, role, [noun], "run deadline passed while the lane was running")
            final = self._lane_record(summary) if summary.is_file() else None
            if final is None or final.status == c.LaneStatus.FAILED:
                failed.append(lane)
        if failed:
            notes.append(f"{len(failed)} lanes not run ({', '.join(failed)})")
        return "; ".join(dict.fromkeys(notes)) or None

    def _lane_event_sink(self, run_id: str, role: str, fps: float) -> Callable[[str, dict[str, Any]], None]:
        """Raw SSE events of one lane become sam.request, sam.first_delta and sam.frame (boxes only, a few per
        second of video; the lane file keeps every frame)."""
        stride = max(1, round(float(fps or 0) / FRAME_EVENTS_PER_VIDEO_S))
        started: dict[str, float] = {}
        first_delta: set[str] = set()
        last_frame: dict[str, int] = {}
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
                    if parse is None:
                        return
                    frames, _ = parse(event.get("delta") or "", media="video")
                    for frame in frames:
                        prev = last_frame.get(lane)
                        if prev is not None and frame.frame - prev < stride:
                            continue
                        last_frame[lane] = frame.frame
                        objects = [{"id": o.id, "box": o.box, "c": o.c} for o in frame.objects]
                        self.emit(run_id, c.EventKind.SAM_FRAME, {"role": role, "noun": lane, "frame": frame.frame, "objects": objects})
            except Exception:
                return  # a progress event never breaks a lane

        return sink

    def _emit_lane_done(self, run_id: str, role: str, lane: str, record: c.LaneRecord, frames: list[c.IndexFrame], *, reused_from: str | None = None) -> None:
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
        if reused_from is not None:
            payload["reused_from"] = reused_from
        self.emit(run_id, c.EventKind.SAM_LANE_DONE, payload)

    def reuse_source(self, run: c.RunManifest, role: str) -> str | None:
        """An earlier walk of this run whose original has the same sha256 as this one (a self-pair, as in a "take 1
        against itself" check). The same bytes would be asked the same questions, and SAM 3.1 answers two identical
        requests a pixel apart (measured 10/31), so the earlier answers stand for both walks."""
        video = self.load_video(run.run_id, role)
        if video is None or not video.sha256_server:
            return None
        for other in self.expected_roles(run):
            if other == role:
                return None
            earlier = self.load_video(run.run_id, other)
            if earlier is not None and earlier.sha256_server == video.sha256_server:
                return other
        return None

    def _to_request(self, run: c.RunManifest, role: str, nouns: list[c.Noun]) -> list[c.Noun]:
        """The nouns that need a SAM request: not the whole-frame container, not a lane a byte-identical walk finished."""
        nouns = [n for n in nouns if not (n is c.Noun.CAR and self.whole_frame_container(run))]
        reusable = set(self._reusable_nouns(run, role, nouns))
        return [n for n in nouns if n not in reusable]

    def _write_whole_frame_container(self, run: c.RunManifest, role: str) -> c.LaneRecord:
        """A stand-in run's car lane: one whole-frame box per proxy frame, written as what it is (no request, no
        request id, no dollars, a whole_frame_container diagnostic), so the car filter keeps every box of the
        stand-in nouns and nothing else in the pipeline changes."""
        run_id = run.run_id
        video = self.load_video(run_id, role)
        assert video is not None and video.proxy is not None
        width, height, count = video.proxy.width, video.proxy.height, video.proxy.frames
        frames = [c.IndexFrame(frame=f, objects=[c.IndexObject(id="frame", box=[0, 0, width, height])]) for f in range(count)]
        index = paths.lane_index_path(run_id, role, c.Noun.CAR)
        c.write_json(index, frames)
        note = c.Diagnostic(severity="warning", code=WHOLE_FRAME_CODE, message="stand-in run: no car request was sent; the whole frame stands in for the car box")
        record = c.LaneRecord(
            lane_id=paths.lane_id(c.Noun.CAR), role=role, noun=c.Noun.CAR, prompt_text=WHOLE_FRAME, media_size=[width, height],
            status=c.LaneStatus.COMPLETED, index_path=paths.rel(run_id, index), frames_returned=count, ids=["frame"], diagnostics=[note],
        )
        c.write_json(paths.lane_summary_path(run_id, role, c.Noun.CAR), record)
        self._emit_lane_done(run_id, role, record.lane_id, record, frames)
        return record

    def _reusable_nouns(self, run: c.RunManifest, role: str, nouns: list[c.Noun]) -> list[c.Noun]:
        """The nouns whose lane the reuse source finished with this run's prompt text and parameters."""
        source = self.reuse_source(run, role)
        if source is None:
            return []
        metadata = self.sam("sam_metadata")()
        out = []
        for noun in nouns:
            summary = paths.lane_summary_path(run.run_id, source, noun)
            record = self._lane_record(summary) if summary.is_file() else None
            if record is not None and record.status in TRACKED_STATUSES and record.prompt_text == self.prompt_text(run, noun) and record.params == metadata:
                out.append(noun)
        return out

    def _reuse_lanes(self, run: c.RunManifest, role: str, nouns: list[c.Noun]) -> list[c.Noun]:
        """Copy the reuse source's finished lanes (gz, index, summary with a reused_identical_request diagnostic, no
        dollars) for this walk: no request is sent, no ledger row written. Returns the nouns copied."""
        run_id = run.run_id
        source = self.reuse_source(run, role)
        reused = self._reusable_nouns(run, role, nouns)
        for noun in reused:
            record = c.read_json(paths.lane_summary_path(run_id, source, noun), c.LaneRecord)
            note = c.Diagnostic(
                severity="warning",
                code=REUSED_CODE,
                message=f"the {role} original has the same sha256 as the {source} original, so the {source} lane's answer stands for it; no request was sent",
            )
            update: dict[str, Any] = {"role": role, "dollars": 0.0, "retries": 0, "upload_s": None, "first_delta_s": None, "completed_s": None, "diagnostics": [*record.diagnostics, note]}
            if record.lane_path:
                gz = paths.lane_events_gz_path(run_id, role, noun)
                shutil.copyfile(paths.run_dir(run_id) / record.lane_path, gz)
                update["lane_path"] = paths.rel(run_id, gz)
            frames: list[c.IndexFrame] = []
            if record.index_path:
                index = paths.lane_index_path(run_id, role, noun)
                shutil.copyfile(paths.run_dir(run_id) / record.index_path, index)
                update["index_path"] = paths.rel(run_id, index)
                frames = c.read_json_list(index, c.IndexFrame)
            copied = record.model_copy(update=update)
            c.write_json(paths.lane_summary_path(run_id, role, noun), copied)
            self._emit_lane_done(run_id, role, copied.lane_id, copied, frames, reused_from=source)
        return reused

    def _write_not_run(self, run_id: str, role: str, nouns: list[c.Noun], reason: str) -> None:
        """A lane the ceiling, the deadline or a failed upload stopped: a FAILED summary so coverage prints not_run."""
        video = self.load_video(run_id, role)
        size = [video.proxy.width, video.proxy.height] if video and video.proxy else [0, 0]
        run = self.load_run(run_id)
        for noun in nouns:
            record = c.LaneRecord(lane_id=paths.lane_id(noun), role=role, noun=noun, prompt_text=self.prompt_text(run, noun), media_size=size, status=c.LaneStatus.FAILED, error=reason)
            c.write_json(paths.lane_summary_path(run_id, role, noun), record)
            c.write_json(paths.lane_index_path(run_id, role, noun), [])
            self.emit(run_id, c.EventKind.SAM_FAILED, {"role": role, "noun": record.lane_id, "status": "failed", "error": reason, "request_id": None})

    def _update_responses_store(self, run_id: str) -> None:
        """store false accepted when every lane's response.created echoed store false; one echo of true means the
        API kept responses, so each request id is deleted at run end (section 14 item 25b)."""
        echoes = []
        for role in self.video_roles(run_id):
            for summary in paths.sam_dir(run_id, role).glob("*.summary.json"):
                record = self._lane_record(summary)
                if record is not None and record.store_echo is not None:
                    echoes.append(record.store_echo)
        if not echoes:
            return
        value = c.ResponsesStore.DELETED_PER_REQUEST if any(echoes) else c.ResponsesStore.FALSE_ACCEPTED
        if self.load_run(run_id).responses_store != value:
            self.update_run(run_id, lambda r: setattr(r, "responses_store", value))

    def load_lanes(self, run: c.RunManifest, role: str) -> dict[str, tuple[c.LaneRecord, list[c.IndexFrame]]]:
        lanes = {}
        for noun in run.noun_list:
            summary = paths.lane_summary_path(run.run_id, role, noun)
            index = paths.lane_index_path(run.run_id, role, noun)
            record = self._lane_record(summary) if summary.is_file() else None
            if record is not None:
                lanes[paths.lane_id(noun)] = (record, c.read_json_list(index, c.IndexFrame) if index.is_file() else [])
        return lanes

    def _pts(self, run_id: str, video: c.VideoRecord) -> list[float]:
        assert video.proxy is not None
        return [float(v) for v in json.loads((paths.run_dir(run_id) / video.proxy.pts_map_path).read_text(encoding="utf-8"))]

    def _stored_blur(self, run_id: str | None = None) -> Callable[..., Awaitable[Any]]:
        """privacy.blur_keyframe for every stored crop: the plate is always blurred there; only the sheet writer keeps
        it for the run owner under keep_plate (section 8 items 5 and 6). Within a run each distinct unblurred frame is
        image-checked once (shared_checks)."""
        blur = self.dep("privacy.blur_keyframe")
        checks = self.shared_checks(run_id) if run_id is not None and _accepts(blur, "checks") else None

        async def stored(*args: Any, **kwargs: Any) -> Any:
            kwargs["keep_plate"] = False
            if checks is not None:
                kwargs["checks"] = checks
            return await blur(*args, **kwargs)

        return stored

    def shared_checks(self, run_id: str) -> Callable[..., Awaitable[Any]]:
        """privacy.image_checks asked once per distinct unblurred frame of a run (keyed by the frame's sha256): a second
        keyframe on the same frame, or the same frame of a byte-identical walk, reads the first answer back. SAM 3.1
        gives two identical image requests masks a few pixels apart (measured 10/31), so asking twice would pixelate the
        same frame two ways. The memo lives for the pipeline call."""
        image_checks = self.dep("privacy.image_checks")
        memo = self._checks_memo.setdefault(run_id, {})

        async def checks(rgb_unblurred: Any, *, run_id: str, role: str, tag: str, th: dict[str, Any]) -> Any:
            key = _frame_digest(rgb_unblurred)
            if key not in memo:
                memo[key] = await image_checks(rgb_unblurred, run_id=run_id, role=role, tag=tag, th=th)
            plates, faces, record = memo[key]
            return list(plates), list(faces), record.model_copy(deep=True)

        return checks

    async def _derive_role(self, run: c.RunManifest, role: str) -> None:
        """panels.derive for one role; it writes tracks, panels, keyframes and coverage for every role on disk."""
        run_id = run.run_id
        video = self.load_video(run_id, role)
        if video is None or video.proxy is None:
            raise FileNotFoundError(f"no proxy record for {role}")
        lanes = await asyncio.to_thread(self.load_lanes, run, role)  # 16 index files parsed off the event loop
        _tracks, _panels, _keyframes, coverage = await self.dep("panels.derive")(
            run_id,
            role,
            lanes=lanes,
            pts=self._pts(run_id, video),
            media_size=(video.proxy.width, video.proxy.height),
            fps=video.proxy.fps,
            decode=self.decode_pool(),
            blur_keyframe=self._stored_blur(run_id),
            th=self.run_th(run),
        )
        prior = c.read_json(paths.coverage_json(run_id), c.Coverage) if paths.coverage_json(run_id).is_file() else c.Coverage()
        if role not in prior.per_role:  # a stand-in that wrote no coverage.json
            prior.per_role[role] = coverage
            c.write_json(paths.coverage_json(run_id), self.dep("panels.plan_topup")(prior, self.run_th(run)))

    def _emit_panels(self, run_id: str, role: str) -> c.Coverage:
        derived = paths.derived_dir(run_id)
        self.emit(run_id, c.EventKind.PANELS_DERIVED, {"role": role, "sha256": {n: sha256_path(derived / f"{n}.json") for n in ("tracks", "panels", "keyframes")}})
        coverage = c.read_json(paths.coverage_json(run_id), c.Coverage)
        self.emit(run_id, c.EventKind.COVERAGE_UPDATED, {"role": role, "sha256": sha256_path(paths.coverage_json(run_id)), "missing": coverage.missing})
        return coverage

    async def _stage_panels(self, run: c.RunManifest, keep_done: bool) -> None:
        run_id = run.run_id
        coverage = None
        for role in self.expected_roles(run):
            if keep_done and self._role_derived(run_id, role):
                continue
            await self._derive_role(run, role)
            coverage = self._emit_panels(run_id, role)
        if coverage is not None and coverage.missing:
            self.emit(run_id, c.EventKind.TOPUP_REQUESTED, {"missing": coverage.missing, "topup_text": coverage.topup_text, "nouns_to_rerun": coverage.nouns_to_rerun})
        panels = c.read_json_list(paths.panels_json(run_id), c.Panel) if paths.panels_json(run_id).is_file() else []
        seen = sum(1 for p in panels if p.instance_from == c.InstanceFrom.TRACKS)
        self.update_run(run_id, lambda r: setattr(r.counts, "panels", seen))

    def _blur_refusal(self, run: c.RunManifest, role: str) -> str | None:
        """Why this walk can have no display proxy, or None when it can be rendered (or already is). A plate or face
        lane that did not complete means the boxes to blur are unknown, so nothing is rendered and the unblurred
        proxy waits for run end."""
        run_id = run.run_id
        video = self.load_video(run_id, role)
        if video is None or video.proxy is None:
            return f"{role} walk: no display video, because it has no proxy"
        if video.proxy_blurred is not None:
            return None
        status = self.lane_status(run_id, role)
        # TODO(D1 E2/E9): count how often the hosted license plate and face lanes end incomplete on a real walk; each
        # one now costs that walk its display video (never its sheet), so a high rate needs a retry before giving up.
        for noun in c.PRIVACY_NOUNS:
            state = status.get(noun)
            if state not in TRACKED_STATUSES:
                said = state.value if state is not None else "missing"
                return f"{role} walk: no display video, because its {noun.value} lane is {said}; the unblurred proxy is deleted at run end"
        if not (paths.run_dir(run_id) / video.proxy.path).is_file():
            return f"{role} walk: no display video, because the unblurred proxy is already deleted"
        if not self.has_dep("panels.privacy_tracks") and not self._role_derived(run_id, role):
            return f"{role} walk: no display video, because its plate and face tracks were never built"
        return None

    def off_car_view(self, run: c.RunManifest) -> dict[str, Any]:
        """The walk overlay's display filter (R26): {"track_ids": the panel tracks mostly off the car's mask, "shares":
        every measured share, "min_inside", "in_core"}. A run whose thresholds carry panels.off_car_tracks dropped such
        tracks in panels.derive (in_core, nothing to hide); for a run frozen before it the same measure is read from
        its stored lanes (panels.stored_off_car). Nothing in the run changes. Cached per tracks.json stamp."""
        run_id = run.run_id
        th = self.run_th(run)
        path = paths.tracks_json(run_id)
        empty: dict[str, Any] = {"in_core": False, "min_inside": None, "shares": {}, "track_ids": []}
        if not path.is_file():
            return empty
        if self.dep("panels.off_car_settings")(th) is not None:
            return {**empty, "in_core": True}
        st = path.stat()
        stamp = (st.st_mtime_ns, st.st_size)
        with self._off_car_lock:
            hit = self._off_car.get(run_id)
            if hit is not None and hit[0] == stamp:
                return hit[1]
            tracks = c.read_json_list(path, c.PanelTrack)
            shares: dict[str, float] = {}
            for role in self.expected_roles(run):
                video = self.load_video(run_id, role)
                if video is None or video.proxy is None or not any(t.role == role for t in tracks):
                    continue
                lanes = self.load_lanes(run, role)
                size = (video.proxy.width, video.proxy.height)
                shares.update(self.dep("panels.stored_off_car")(run_id, role, tracks, lanes, size, th))
            cfg = self.dep("panels.off_car_settings")(th) or dict(self.dep("panels.OFF_CAR_DEFAULTS"))
            view = {**empty, "min_inside": float(cfg["min_inside"]), "shares": shares, "track_ids": self.dep("panels.off_car_ids")(shares, cfg)}
            self._off_car[run_id] = (stamp, view)
            return view

    def _lane_frames(self, run_id: str, role: str, noun: c.Noun) -> list[c.IndexFrame]:
        index = paths.lane_index_path(run_id, role, noun)
        return c.read_json_list(index, c.IndexFrame) if index.is_file() else []

    def _privacy_tracks(self, run: c.RunManifest, role: str) -> list[c.PanelTrack]:
        """The walk's plate and face tracks, rebuilt from its lanes by panels.privacy_tracks (pure, no model call), the
        function panels.derive writes them with; so the blur holds whether or not panels.derive ran for this walk."""
        run_id = run.run_id
        if not self.has_dep("panels.privacy_tracks"):
            tracks = c.read_json_list(paths.tracks_json(run_id), c.PanelTrack) if paths.tracks_json(run_id).is_file() else []
            return [t for t in tracks if t.role == role and t.noun in c.PRIVACY_NOUNS]
        video = self.load_video(run_id, role)
        assert video is not None and video.proxy is not None
        car_boxes = self.dep("panels.car_boxes_by_frame")(self._lane_frames(run_id, role, c.Noun.CAR))
        build = self.dep("panels.privacy_tracks")
        size, pts, th = (video.proxy.width, video.proxy.height), self._pts(run_id, video), self.run_th(run)
        return [t for noun in c.PRIVACY_NOUNS for t in build(role, noun, self._lane_frames(run_id, role, noun), pts, car_boxes, size, fps=video.proxy.fps, th=th)]

    async def _blur_role(self, run: c.RunManifest, role: str) -> str | None:
        """The display proxy from the plate and face tracks; the unblurred proxy goes with a deletions.json row. Returns
        the reason when the walk can have no display proxy."""
        run_id = run.run_id
        video = self.load_video(run_id, role)
        if video is None or video.proxy is None or video.proxy_blurred is not None:
            return None
        refusal = self._blur_refusal(run, role)
        if refusal is not None:
            return refusal
        tracks = await asyncio.to_thread(self._privacy_tracks, run, role)
        record, deletion = await self.dep("privacy.render_blurred_proxy")(run_id, role, tracks, fps=video.proxy.fps, th=self.run_th(run))
        self.update_video(run_id, role, lambda v: setattr(v, "proxy_blurred", record))
        self._add_deletions(run_id, [deletion])
        self.emit(run_id, c.EventKind.BLUR_DONE, {"role": role, "sha256": record.sha256, "path": record.path})
        return None

    async def _stage_blur(self, run: c.RunManifest, keep_done: bool) -> None:
        gaps = []
        for role in self.expected_roles(run):
            gap = await self._blur_role(run, role)
            if gap is not None:
                gaps.append(gap)
        if gaps:
            raise StageGap("; ".join(gaps))

    async def _stage_marks(self, run: c.RunManifest, keep_done: bool) -> None:
        run_id = run.run_id
        for role in self.expected_roles(run):
            if keep_done and self._marks_done(run_id, role) and paths.marks_json(run_id).is_file():
                continue
            keyframes = [k for k in self._keyframes(run_id) if k.role == role]
            marks, _kfs, records = await self.dep("marks.derive_marks")(run_id, role, keyframes, th=self.run_th(run))
            for record in records:
                self._emit_spark(run_id, role, record)
            self.emit(run_id, c.EventKind.MARKS_DERIVED, {"role": role, "sha256": sha256_path(paths.marks_json(run_id)), "marks": len(marks)})
        total = len(c.read_json_list(paths.marks_json(run_id), c.Mark)) if paths.marks_json(run_id).is_file() else 0
        self.update_run(run_id, lambda r: setattr(r.counts, "marks", total))

    def _emit_spark(self, run_id: str, role: str, record: c.SparkRequestRecord) -> None:
        """spark.request and spark.response for one stored call; the response hash covers the whole spark.json record."""
        self.emit(run_id, c.EventKind.SPARK_REQUEST, {"role": role, "request_id": record.request_id, "purpose": record.purpose.value, "max_output_tokens": record.max_output_tokens, "effort": record.reasoning_effort.value, "store": record.store})
        self.emit(
            run_id,
            c.EventKind.SPARK_RESPONSE,
            {
                "role": role,
                "request_id": record.request_id,
                "sha256": record.digest(),
                "raw_sha256": hashlib.sha256(record.raw_text.encode("utf-8")).hexdigest(),
                "status": record.status.value,
                "incomplete_reason": record.incomplete_reason,
                "truncated": record.truncated,
            },
        )

    async def _stage_staysput(self, run: c.RunManifest, keep_done: bool) -> None:
        run_id = run.run_id
        if not self.has_dep("staysput.derive_staysput"):
            raise StageGap("staysput module not wired (backlog B2)")
        count = 0
        roles = self.expected_roles(run)
        for role in roles:
            count += len(await self.dep("staysput.derive_staysput")(run_id, role, th=self.run_th(run), stop=lambda: self.past_deadline(run_id)))
        self.emit(run_id, c.EventKind.STAYSPUT_DERIVED, {"sha256": sha256_path(paths.mark_tracks_json(run_id)), "mark_tracks": count})
        if self.has_dep("staysput.role_done") and not all(self.dep("staysput.role_done")(run_id, r) for r in roles):
            raise StageGap(f"run deadline {int(self.th['orchestrator']['run_deadline_s'])} s passed during Stays Put; the clips not started wait for a resume")

    async def _stage_narration(self, run: c.RunManifest, keep_done: bool) -> None:
        """Voice per walk once (voice.json keeps every answer), then narration. A clip with no audio, over 10 minutes
        or over 32 MB is 'no narration found' for that walk, never a failed stage (section 14 item 16)."""
        run_id = run.run_id
        th = self.run_th(run)
        records = c.read_json_list(paths.voice_json(run_id), c.VoiceRecord) if paths.voice_json(run_id).is_file() else []
        for role in self.expected_roles(run):
            if any(r.role == role for r in records):
                continue
            video = self.load_video(run_id, role)
            if video is None or video.wav is None:
                self.emit(run_id, c.EventKind.VOICE_DONE, {"role": role, "mode_applied": None, "text": "no narration found: the clip has no audio track"})
                continue
            wav = paths.run_dir(run_id) / video.wav.path
            try:
                transcript = await asyncio.to_thread(self.dep("voice_client.transcribe_wav"), wav, mode=th["voice"]["mode"], keywords=list(th["nouns"]["voice_keywords"]), run_id=run_id)
            except Exception as exc:
                self.emit(run_id, c.EventKind.VOICE_DONE, {"role": role, "mode_applied": None, "text": f"no narration found: {_error_text(exc)}"[:300]})
                continue
            record = self.dep("voice_client.voice_record")(transcript, role=role, wav_sha256=video.wav.sha256 or sha256_path(wav))
            records.append(record)
            c.write_json(paths.voice_json(run_id), records)
            self.emit(run_id, c.EventKind.VOICE_DONE, {"role": role, "mode_applied": record.mode_applied, "turns": len(record.response.turns), "sha256": record.digest()})
        if not self.has_dep("narration.derive_narration"):
            raise StageGap("narration module not wired (backlog B6)")
        asked = {r.request_id for r in c.read_json_list(paths.spark_json(run_id), c.SparkRequestRecord)} if paths.spark_json(run_id).is_file() else set()
        narration = await self.dep("narration.derive_narration")(run_id, th=th)
        if self.has_dep("narration.seed_records"):
            for role, record in self.dep("narration.seed_records")(run_id):
                if record.request_id not in asked:  # seed_v1 answers join the chain like the marks pass's
                    asked.add(record.request_id)
                    self._emit_spark(run_id, role, record)
        self.emit(run_id, c.EventKind.NARRATION_DERIVED, {"sha256": sha256_path(paths.narration_json(run_id)), "mode_applied": narration.mode_applied})

    async def _stage_match(self, run: c.RunManifest, keep_done: bool) -> None:
        """Match and verdicts, the second look inside it (it can only lower a label), then the ruler's sizes. The stage
        stays model=False so the free match always runs; the look alone is skipped past the ceiling or the deadline."""
        run_id = run.run_id
        th = self.run_th(run)
        if self.has_dep("second_pass.derive_second_pass"):
            # the second marks pass (matching.second_marks_pass, 11/2): its marks join marks.json before the match; past
            # the ceiling or the deadline only the stored answers are replayed
            if not self.over_ceiling(run) and not self.past_deadline(run_id):
                asked = await self.dep("second_pass.derive_second_pass")(run_id, th=th)
                for role, record in asked:
                    self._emit_spark(run_id, role, record)
                if asked:
                    self._note_spend(run_id)
            else:
                await asyncio.to_thread(self.dep("second_pass.apply_second_pass"), run_id, th=th)
        look = None
        if self.has_dep("second_look.make_second_look") and not self.over_ceiling(run) and not self.past_deadline(run_id):
            look = self.dep("second_look.make_second_look")(run_id, th=th)  # None while spark.second_look_engine is off
        matches, verdicts, _calibration = await asyncio.to_thread(self.dep("match.derive_match"), run_id, th=th, second_look=look)
        if look is not None:
            self._emit_looks(run_id)
            self._note_spend(run_id)
        if self.has_dep("ruler.size_verdicts"):
            await asyncio.to_thread(self.dep("ruler.size_verdicts"), run_id, th=th)
        self.emit(run_id, c.EventKind.MATCH_DERIVED, {"sha256": {"matches": sha256_path(paths.matches_json(run_id)), "verdicts": sha256_path(paths.verdicts_json(run_id))}, "panels": len(matches), "verdicts": len(verdicts)})
        pairs = sum(len(pm.pairs) for pm in matches)
        self.update_run(run_id, lambda r: setattr(r.counts, "matches", pairs))

    def _emit_looks(self, run_id: str) -> None:
        """spark.request and spark.response for every model answer behind derived/looks.json (both engines), once each,
        so the chain covers the second looks like the marks pass. role is 'pair': a look shows both walks."""
        if not self.has_dep("second_look.stored_looks"):
            return
        seen = {e.payload.get("request_id") for e in self.read_events(run_id) if e.kind == c.EventKind.SPARK_RESPONSE}
        for look in self.dep("second_look.stored_looks")(run_id):
            for record in look.records:
                if record.request_id and record.request_id not in seen:
                    seen.add(record.request_id)
                    self._emit_spark(run_id, "pair", record)

    async def _write_sheet(self, run: c.RunManifest) -> None:
        """Freeze the head the sheet prints into run.json, then write the sheet (a rebuild reads the same head)."""
        run_id = run.run_id
        head = self.events.head_hash(run_id)
        self.update_run(run_id, lambda r: setattr(r, "events_head_hash", head))
        _model, hashes = await asyncio.to_thread(self.dep("sheet.write_sheet"), run_id, th=self.run_th(run))
        self.emit(run_id, c.EventKind.SHEET_WRITTEN, {"sha256": hashes})

    async def _stage_sheet(self, run: c.RunManifest, keep_done: bool) -> None:
        await self._write_sheet(run)

    async def _stage_release(self, run: c.RunManifest, keep_done: bool) -> None:
        """Run end, first half (section 5 step 14, section 8 item 7), before the sheet so it prints real deletion clocks:
        Files API uploads (and stored responses when store false was refused) deleted with their receipts, any
        unblurred proxy a refused blur left, the quarantine and upload chunks deleted unless keep_plate needs the
        original for the owner's sheet frames, deletions.json, and a first manifest_run.json, from which
        seal.verify_run at sheet time learns what was deleted."""
        rows = await self._release(run, local=not run.keep_plate)
        await self._write_manifest_run(run, rows)

    async def _stage_close(self, run: c.RunManifest, keep_done: bool) -> None:
        """Run end, second half: whatever is still on disk or remote goes (the keep_plate original, a late top-up),
        then the final manifest_run.json with the sheet, the PDF hash and the head at close."""
        rows = await self._release(run, local=True)
        await self._write_manifest_run(run, rows)

    async def _release(self, run: c.RunManifest, *, local: bool) -> list[c.DeletionRecord]:
        """Delete what is still remote, any unblurred proxy left on disk and, with local, the originals and chunks too;
        idempotent, so a resume repeats it. Hashing before removal runs in a worker thread, off the event loop."""
        run_id = run.run_id
        now = self.now()
        videos = [v for r in self.video_roles(run_id) if (v := self.load_video(run_id, r)) is not None]
        response_ids = self._pending_response_ids(run_id) if run.responses_store == c.ResponsesStore.DELETED_PER_REQUEST else None
        # one DELETE per file id: two walks with the same bytes share an upload through the client's sha256 cache
        pending = list({v.files_api.file_id: v for v in videos if v.files_api is not None and v.files_api.deleted_at is None}.values())
        rows = list(await self.dep("privacy.delete_remote")(run_id, pending, response_ids=response_ids, now=now)) if pending or response_ids else []
        for row in rows:
            if row.kind == c.DeletionKind.FILES_API:
                self.emit(run_id, c.EventKind.FILES_DELETED, {"file_id": row.target, "response": row.api_response, "reason": row.reason})
            elif row.kind == c.DeletionKind.RESPONSE:
                self.emit(run_id, c.EventKind.RESPONSES_DELETED, {"request_id": row.target, "response": row.api_response, "reason": row.reason})
        receipts = {r.target: r.api_response for r in rows if r.kind == c.DeletionKind.FILES_API and r.api_response is not None}
        removed = await asyncio.to_thread(self._delete_local, run_id, now, local)
        gone = {r.target for r in removed}
        for video in videos:

            def change(v: c.VideoRecord) -> None:
                if v.files_api is not None and v.files_api.file_id in receipts and v.files_api.deleted_at is None:
                    v.files_api.deleted_at = now
                    v.files_api.delete_response = receipts[v.files_api.file_id]
                if v.quarantine_path and v.quarantine_path in gone:
                    v.quarantine_deleted_at = now

            if (video.files_api is not None and video.files_api.file_id in receipts) or (video.quarantine_path in gone):
                self.update_video(run_id, video.role, change)
        if removed:
            self.emit(run_id, c.EventKind.QUARANTINE_DELETED, {"files": sorted(gone)})
        return self._add_deletions(run_id, rows + removed)

    def _delete_local(self, run_id: str, now: str, originals: bool) -> list[c.DeletionRecord]:
        """Unblurred bytes on disk, each hashed before removal (worker thread): any unblurred proxy no blurred one
        replaced and, with originals, the quarantine and the upload chunks."""
        rows = self._delete_unblurred_proxies(run_id, now=now)
        if originals and self._local_originals(run_id):
            rows += list(self.dep("privacy.delete_quarantine")(run_id, now=now)) + self._delete_upload_chunks(run_id, now=now, reason="run end")
        return rows

    def _delete_unblurred_proxies(self, run_id: str, *, now: str) -> list[c.DeletionRecord]:
        rows = []
        for path in self._unblurred_proxies(run_id):
            rows.append(c.DeletionRecord(target=paths.rel(run_id, path), kind=c.DeletionKind.FILE, sha256=sha256_path(path), bytes=path.stat().st_size, deleted_at=now, reason="unblurred proxy deleted at run end; no blurred display proxy replaced it"))
            path.unlink()
        return rows

    async def _write_manifest_run(self, run: c.RunManifest, rows: list[c.DeletionRecord]) -> None:
        """deletions.json closed, then manifest_run.json, in a worker thread: it hashes every kept artifact and runs
        `ots stamp` (up to 120 s). A stamp of an earlier manifest_run.json covers bytes now gone, so it goes first."""
        head = self.events.head_hash(run.run_id) or ""
        await asyncio.to_thread(self._write_manifest_run_sync, run, rows, head)

    def _write_manifest_run_sync(self, run: c.RunManifest, rows: list[c.DeletionRecord], head: str) -> None:
        run_id = run.run_id
        (paths.runs_root() / run_id / "bundle" / "manifest_run.json.ots").unlink(missing_ok=True)
        self.dep("privacy.close_deletions")(run_id, rows, keep_plate=run.keep_plate, chosen_by=_chosen_by(run))
        pdf = paths.sheet_pdf(run_id)
        self.dep("seal.write_manifest_run")(
            run_id,
            deletions=rows,
            events_head_hash=head,
            config_hash=run.config_hash or "",
            app_commit=run.app_commit or "",
            pdf_sha256=sha256_path(pdf) if pdf.is_file() else None,
            **({"rederived": run.rederived.text} if run.rederived is not None else {}),
        )

    def _delete_upload_chunks(self, run_id: str, *, now: str, reason: str) -> list[c.DeletionRecord]:
        """Chunks of an upload that never finalized are unblurred bytes too: hashed, removed, recorded."""
        root = paths.runs_root() / run_id / "uploads"
        rows = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []:
            rows.append(c.DeletionRecord(target=paths.rel(run_id, path), kind=c.DeletionKind.QUARANTINE, sha256=sha256_path(path), bytes=path.stat().st_size, deleted_at=now, reason=f"upload chunks never finalized ({reason})"))
            path.unlink()
        if root.is_dir():
            shutil.rmtree(root, ignore_errors=True)
        return rows

    def _response_ids(self, run_id: str) -> list[str]:
        """Every request id of the run: lanes, image checks, Spark, Stays Put (deleted only when store false was refused)."""
        ids: list[str] = []
        for role in self.video_roles(run_id):
            for summary in sorted(paths.sam_dir(run_id, role).glob("*.summary.json")):
                record = self._lane_record(summary)
                if record is not None and record.request_id:
                    ids.append(record.request_id)
        for kf in self._keyframes(run_id):
            ids += kf.blur.request_ids + ([kf.handle.request_id] if kf.handle.request_id else [])
        if paths.spark_json(run_id).is_file():
            ids += [r.request_id for r in c.read_json_list(paths.spark_json(run_id), c.SparkRequestRecord)]
        if paths.mark_tracks_json(run_id).is_file():
            ids += [t.request_id for t in c.read_json_list(paths.mark_tracks_json(run_id), c.MarkTrack) if t.request_id]
        if self.has_dep("staysput.request_ids"):
            ids += self.dep("staysput.request_ids")(run_id)  # the clip checks live only in clips/<role>/staysput.json
        if self.has_dep("second_look.stored_looks"):
            # hosted Muse Spark answers behind the second looks; Baseten's chatcmpl- ids never go to Meta's DELETE
            ids += [r.request_id for look in self.dep("second_look.stored_looks")(run_id) for r in look.records if r.engine == c.SparkEngine.META_MODEL_API and not r.request_id.startswith("local-")]
        return list(dict.fromkeys(i for i in ids if i))

    def _pending_response_ids(self, run_id: str) -> list[str]:
        """Request ids whose stored response has no deletion row yet."""
        done = {r.target for r in self._load_deletions(run_id) if r.kind == c.DeletionKind.RESPONSE and r.api_response is not None}
        return [i for i in self._response_ids(run_id) if i not in done]

    def _load_deletions(self, run_id: str) -> list[c.DeletionRecord]:
        path = paths.runs_root() / run_id / "deletions.json"
        return list(c.read_json(path, c.Deletions).rows) if path.is_file() else []

    def _add_deletions(self, run_id: str, rows: Iterable[c.DeletionRecord]) -> list[c.DeletionRecord]:
        """Deletion rows survive a crash: deletions.json is rewritten every time a row lands."""
        with self._io_lock:
            current = self._load_deletions(run_id)
            for row in rows:
                if row not in current:
                    current.append(row)
            run = self.load_run(run_id)
            write_json_atomic(paths.deletions_json(run_id), c.Deletions(run_id=run_id, rows=current, keep_plate=run.keep_plate, keep_plate_chosen_by=_chosen_by(run)))
            return current

    # Top-up ------------------------------------------------------------------------------------------------------------

    async def run_topup(self, run_id: str, role: str) -> c.Coverage:
        """Only coverage.nouns_to_rerun on the new clip; the rerun time is measured and printed (section 5.4). The clip
        gets its own blurred display proxy; when the run had already ended, its upload and original are deleted at
        once and the sheet and manifest_run.json are written again. A top-up that fails after run end still deletes
        them (nothing else would) and leaves the sheet as it was."""
        run = self.load_run(run_id)
        if not paths.coverage_json(run_id).is_file():
            raise FileNotFoundError("coverage.json missing: a top-up follows the first pass")
        if self.load_video(run_id, role) is None:
            raise FileNotFoundError(f"videos/{role}.json missing; save the clip through finalize_upload first")
        nouns = [c.Noun(n) for n in c.read_json(paths.coverage_json(run_id), c.Coverage).nouns_to_rerun]
        topup_run = run.model_copy(update={"noun_list": nouns})
        t0 = self.clock()
        try:
            return await self._topup(run, topup_run, role, t0)
        finally:
            self._checks_memo.pop(run_id, None)

    async def _topup(self, run: c.RunManifest, topup_run: c.RunManifest, role: str, t0: float) -> c.Coverage:
        run_id = run.run_id
        async with self.run_lock(run_id):
            try:
                await self._media_role(run, role, wav=False)
                deadline = self.clock() + float(self.th["orchestrator"]["run_deadline_s"])
                gap = await self._lanes_for_role(topup_run, role, self.missing_lanes(topup_run, role), deadline=deadline)
                if gap:
                    self._record_error(run_id, "topup", f"{role}: {gap}")
                await self._derive_role(topup_run, role)
                coverage = self.dep("panels.plan_topup")(c.read_json(paths.coverage_json(run_id), c.Coverage), self.run_th(run))
                coverage.rerun_seconds = round(self.clock() - t0, 3)
                c.write_json(paths.coverage_json(run_id), coverage)
                self.emit(run_id, c.EventKind.COVERAGE_UPDATED, {"role": role, "sha256": sha256_path(paths.coverage_json(run_id)), "missing": coverage.missing, "rerun_seconds": coverage.rerun_seconds})
                refusal = await self._blur_role(run, role)
                if refusal is not None:
                    self._record_error(run_id, "topup", refusal)
            except Exception:
                # after run end nothing else deletes this clip's original, upload or unblurred proxy, so they go now
                ended = self.load_run(run_id)
                if ended.status in TERMINAL_STATUSES:
                    with suppress(Exception):
                        await self._close_late_role(ended, rewrite_sheet=False)
                raise
            ended = self.load_run(run_id)
            if ended.status in TERMINAL_STATUSES:
                await self._close_late_role(ended)
        return coverage

    def check_can_topup(self, run: c.RunManifest) -> None:
        """Raise CapsExceeded when the pause, today's card ceiling or the run's own ceiling stops a top-up's lanes."""
        day = et_day_of(self.now())
        left = self.caps_remaining(et_day=day)
        if self.paused:
            reason = "live runs paused; the sample pair is precomputed"
        elif self.card_ceiling_reached(day):
            reason = "live runs paused for today; the sample pair is precomputed"
        elif self.over_ceiling(run):
            reason = f"this run reached its spend ceiling of {run.spend_ceiling_cents} cents"
        else:
            return
        raise CapsExceeded(reason, pickup_left=left["pickup"], pairs_left=left["pair"])

    async def _close_late_role(self, run: c.RunManifest, *, rewrite_sheet: bool = True) -> None:
        """A top-up after run end: its upload, original and any unblurred proxy go now, then the sheet and
        manifest_run.json again, in the pipeline's order (the owner's plate frames come from the original, so a
        keep_plate sheet is left as it was; a failed top-up changes no sheet)."""
        rows = await self._release(run, local=True)
        await self._write_manifest_run(run, rows)
        if rewrite_sheet and paths.sheet_html(run.run_id).is_file() and not run.keep_plate:
            await self._write_sheet(self.load_run(run.run_id))
            await self._write_manifest_run(run, rows)

    # Resume ------------------------------------------------------------------------------------------------------------

    async def resume_all(self) -> list[str]:
        """On boot: live runs without a terminal status become interrupted and resume where the files stop."""
        resumed = []
        for run in self.list_runs():
            if run.live_or_replay != c.LiveOrReplay.LIVE or run.status in TERMINAL_STATUSES or self.active(run.run_id):
                continue
            if not self.uploads_complete(run) and not self.stage_done(run, STAGE_BY_NAME["media"]):
                continue
            try:
                self.admit(run)  # a run that never started is admitted like a finalize would admit it
            except CapsExceeded as exc:
                await self.discard_uploads(run.run_id, reason=exc.reason)
                continue
            self.update_run(run.run_id, lambda r: setattr(r, "status", c.RunStatus.INTERRUPTED))
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

    def _day_ledger(self, et_day: str) -> tuple[float, set[str]]:
        """The ledger dollars of one ET day and the run ids that spent them."""
        dollars = 0.0
        spent_by: set[str] = set()
        for row in self.ledger_rows():
            try:
                if et_day_of(str(row["ts"])) != et_day:
                    continue
            except (KeyError, ValueError):
                continue
            dollars += float(row.get("usd", 0.0))
            if row.get("run_id"):
                spent_by.add(str(row["run_id"]))
        return dollars, spent_by

    def card_ceiling_reached(self, et_day: str) -> bool:
        return self._day_ledger(et_day)[0] * 100 >= float(self.limits["daily_card_ceiling_cents"])

    def caps_remaining(self, *, et_day: str, exclude: str | None = None) -> dict[str, int]:
        """{pickup, pair} left today: live runs that started that ET day (or hold a reservation from their creation
        within the last hour), and zero for both once the day's ledger dollars reach the card ceiling. exclude leaves
        one run out of the count (the run being admitted at start)."""
        dollars, spent_by = self._day_ledger(et_day)
        now = parse_ts(self.now())
        used = {c.RunKind.PICKUP_CHECK: 0, c.RunKind.PAIR: 0}
        for run in self.list_runs():
            if run.run_id == exclude or run.live_or_replay != c.LiveOrReplay.LIVE or run.kind not in used:
                continue
            if run.run_id not in spent_by:
                if et_day_of(run.created_at) != et_day:
                    continue
                started = run.status != c.RunStatus.CREATED or self.active(run.run_id)
                reserved = run.status == c.RunStatus.CREATED and (now - parse_ts(run.created_at)).total_seconds() < RESERVATION_S
                if not (started or reserved):
                    continue
            used[run.kind] += 1
        if dollars * 100 >= float(self.limits["daily_card_ceiling_cents"]):
            return {"pickup": 0, "pair": 0}
        return {
            "pickup": max(0, self.limits["pickup_runs_per_day"] - used[c.RunKind.PICKUP_CHECK]),
            "pair": max(0, self.limits["pair_runs_per_day"] - used[c.RunKind.PAIR]),
        }

    def check_can_start(self, kind: c.RunKind, *, exclude: str | None = None) -> dict[str, int]:
        """Raise CapsExceeded when the pause or today's caps stop a live run of this kind (exclude: see caps_remaining)."""
        left = self.caps_remaining(et_day=et_day_of(self.now()), exclude=exclude)
        if self.paused:
            raise CapsExceeded("live runs paused; the sample pair is precomputed", pickup_left=left["pickup"], pairs_left=left["pair"])
        if kind == c.RunKind.PAIR and left["pair"] <= 0:
            raise CapsExceeded("no full pairs left today", pickup_left=left["pickup"], pairs_left=left["pair"])
        if kind == c.RunKind.PICKUP_CHECK and left["pickup"] <= 0:
            raise CapsExceeded("no pickup checks left today", pickup_left=left["pickup"], pairs_left=left["pair"])
        return left

    def queue_eta_minutes(self, position: int | None) -> int | None:
        """'about M minutes' for the queue line: the mean wall time of the last finished live runs (section 14 item 13)."""
        if position is None:
            return None
        durations = []
        for run in self.list_runs():
            if run.live_or_replay != c.LiveOrReplay.LIVE or run.status not in TERMINAL_STATUSES:
                continue
            events = self.read_events(run.run_id)
            if len(events) >= 2:
                durations.append((parse_ts(events[-1].ts) - parse_ts(events[0].ts)).total_seconds())
            if len(durations) >= int(self.th["orchestrator"]["queue_eta_from_last_runs"]):
                break
        if not durations:
            return None
        return max(1, math.ceil(sum(durations) / len(durations) * position / max(1, self.max_concurrent_runs) / 60))

    # Health, sweep, stamps, samples ------------------------------------------------------------------------------------

    async def health_probe(self) -> bool:
        """GET /v1/models; three failures in a row pause live runs, one pass clears the pause. With no MODEL_API_KEY
        live runs pause at once (a clone with no key only replays), since a missing key never passes a later probe."""
        from .config import api_key

        try:
            key = api_key()
        except RuntimeError:
            self.probe_ok = False
            self.paused = True
            return False
        try:
            status = await self.http_get(f"{BASE_URL}/models", {"Authorization": f"Bearer {key}"})
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
        """privacy.sweep (runs past retention, old unblurred temp files) plus idle upload chunks, then the ots upgrade."""
        keep = {r.run_id for r in self.list_runs() if r.keep} | {rid for rid in self._tasks if self.active(rid)}
        keep.update(self.sample_run_ids())
        sample = self.sample_run_id()
        if sample:
            keep.add(sample)
        minutes = int(self.th["privacy"]["unblurred_temp_max_minutes"])
        rows = list(await asyncio.to_thread(self.dep("privacy.sweep"), now=self.now(), retention_hours=self.limits["run_retention_hours"], temp_minutes=minutes, keep_run_ids=keep))
        rows += await asyncio.to_thread(self._sweep_upload_chunks, minutes)
        await self.upgrade_stamps()
        return rows

    def _sweep_upload_chunks(self, minutes: int) -> list[c.DeletionRecord]:
        """Upload chunk directories idle longer than the unblurred temp limit (an abandoned upload)."""
        now_s, now = time.time(), self.now()
        root = paths.runs_root()
        staged = [(None, d) for d in (root / STAGING_DIR_NAME).iterdir()] if (root / STAGING_DIR_NAME).is_dir() else []
        under_runs = [(run.name, d) for run in root.iterdir() if run.is_dir() and (run / "uploads").is_dir() for d in (run / "uploads").iterdir()]
        rows: list[c.DeletionRecord] = []
        for run_id, upload in staged + under_runs:
            files = [p for p in upload.rglob("*") if p.is_file()]
            newest = max((p.stat().st_mtime for p in files), default=upload.stat().st_mtime)
            if (now_s - newest) / 60 <= minutes:
                continue
            found = []
            for path in sorted(files):
                target = paths.rel(run_id, path) if run_id else str(path.relative_to(root))
                found.append(c.DeletionRecord(target=target, kind=c.DeletionKind.QUARANTINE, sha256=sha256_path(path), bytes=path.stat().st_size, deleted_at=now, reason=f"upload chunks idle over {minutes} min"))
                path.unlink()
            shutil.rmtree(upload, ignore_errors=True)
            if run_id and found and (root / run_id / "run.json").is_file():
                self._add_deletions(run_id, found)
            rows += found
        return rows

    async def upgrade_stamps(self) -> list[str]:
        """The hourly ots upgrade (section 7 item 1): a pending manifest stamp that confirmed emits seal.ots."""
        if not self.has_dep("seal.upgrade_ots"):
            return []
        upgraded = []
        for run in self.list_runs():
            ots = paths.runs_root() / run.run_id / "bundle" / "manifest.json.ots"
            if not ots.is_file() or not self.chain_open(run.run_id):
                continue
            last = next((str(e.payload.get("status")) for e in reversed(self.read_events(run.run_id)) if e.kind == c.EventKind.SEAL_OTS), None)
            if last is not None and not last.startswith("pending"):
                continue
            try:
                status = await asyncio.to_thread(self.dep("seal.upgrade_ots"), ots)
            except NotImplementedError:
                return upgraded
            except Exception:
                continue
            if status != last:
                self.emit(run.run_id, c.EventKind.SEAL_OTS, {"status": status})
                upgraded.append(run.run_id)
        return upgraded

    def sample_entries(self) -> list[dict[str, Any]]:
        """samples/index.json's entries in order ({"slot"?, "run_id", "title"}); [] when the file is absent or unreadable."""
        index = Path(REPO_DIR) / "samples" / SAMPLE_INDEX_NAME
        try:
            raw = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        entries = raw.get("samples") if isinstance(raw, dict) else None
        return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []

    def sample_run_ids(self) -> list[str]:
        """The sample runs in the repo, in order: samples/index.json's run ids whose folder holds a run.json (an empty
        slot or a missing folder is skipped), then any other samples/<run_id>/ with a run.json, alphabetically."""
        samples = Path(REPO_DIR) / "samples"
        ids: list[str] = []
        for entry in self.sample_entries():
            run_id = entry.get("run_id")
            if isinstance(run_id, str) and RUN_ID_RE.match(run_id) and run_id not in ids and (samples / run_id / "run.json").is_file():
                ids.append(run_id)
        for folder in sorted(samples.iterdir()) if samples.is_dir() else []:
            if not folder.name.startswith(".") and folder.name not in ids and (folder / "run.json").is_file():
                ids.append(folder.name)
        return ids

    def sample_run_id(self) -> str | None:
        """The precomputed demo run: WALKAROUND_SAMPLE_RUN, else the first of sample_run_ids()."""
        env = os.environ.get("WALKAROUND_SAMPLE_RUN")
        if env:
            return env
        ids = self.sample_run_ids()
        return ids[0] if ids else None

    def seed_samples(self) -> list[str]:
        """Copy samples/<run_id>/ onto the runs volume when it is absent (section 4: the image seeds the demo at boot)."""
        samples = Path(REPO_DIR) / "samples"
        seeded = []
        for entry in sorted(samples.iterdir()) if samples.is_dir() else []:
            target = paths.runs_root() / entry.name
            if not entry.name.startswith(".") and (entry / "run.json").is_file() and not target.exists():
                shutil.copytree(entry, target)
                packed = target / "events.ndjson.gz"
                if packed.is_file() and not (target / "events.ndjson").exists():
                    (target / "events.ndjson").write_bytes(gzip.decompress(packed.read_bytes()))
                    packed.unlink()
                seeded.append(entry.name)
        return seeded

    def is_public(self, run: c.RunManifest) -> bool:
        """Every sample run and keep runs: listed in the index and never served as the run owner."""
        return run.keep or run.run_id == self.sample_run_id() or run.run_id in self.sample_run_ids()

    # Replay ------------------------------------------------------------------------------------------------------------

    def load_delays(self) -> dict[str, int]:
        if not Path(self.delays_path).is_file():
            return {}
        raw = json.loads(Path(self.delays_path).read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in raw.items() if not str(k).startswith("_")}

    def recorded_on(self, run: c.RunManifest) -> str:
        """The badge date of 'replayed from a run on <date>': the ET day the run was created."""
        return et_day_of(run.created_at)

    def replay_payload(self, run_id: str, *, mode: c.ReplayMode) -> c.ReplayPayload:
        run = self.load_run(run_id)
        delays = self.load_delays() if mode == c.ReplayMode.FIXED else {}
        return c.ReplayPayload(run_id=run_id, mode=mode, recorded_on=self.recorded_on(run), delays_ms=delays, events=self.visible_events(run_id))

    def replay_delays(self, events: list[c.Event], *, mode: c.ReplayMode) -> list[float]:
        """Seconds to wait before each event: the stage share (fixed), the capped ts delta (recorded), 0 (instant).
        web/src/state/schedule.ts computes the same numbers, so the CLI and the browser pace a replay alike."""
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
        """The replay driver: stored events re-emitted with the pacing of the mode, no model call. Each event is due at
        the sum of the waits before it, so sleep jitter never accumulates over a replay."""
        events = self.visible_events(run_id)
        start = self.clock()
        due = 0.0
        for wait, event in zip(self.replay_delays(events, mode=mode), events):
            due += wait
            ahead = due - (self.clock() - start)
            if wait > 0 and ahead > 0:
                await self.sleep(ahead)
            yield event

    # Rebuild and verify ------------------------------------------------------------------------------------------------

    def is_closed(self, run_id: str) -> bool:
        """manifest_run.json exists: the run's derived files, sheet and PDF are sealed as they are."""
        return (paths.runs_root() / run_id / "bundle" / "manifest_run.json").is_file()

    def _check_rebuild_out(self, run_id: str, out: Path) -> None:
        """Refuse an --out whose copy would replace the run itself (out = the runs directory, or a path inside the run)
        or whose ledger.jsonl is the global ledger."""
        run_dir = (paths.runs_root() / run_id).resolve()
        target = (out / run_id).resolve()
        if target == run_dir or run_dir in target.parents or target in run_dir.parents:
            raise RebuildRefused(f"--out {out} would overwrite run {run_id} itself; pick a directory outside {paths.runs_root()} or a subdirectory such as {paths.runs_root() / '_rebuild'}")
        if (out / "ledger.jsonl").resolve() == Path(ledger.LEDGER_PATH).resolve():
            raise RebuildRefused(f"--out {out} holds the global ledger.jsonl, which the rebuild would overwrite with one run's rows")

    def rebuild(self, run_id: str, *, out: Path | None = None, thresholds: dict[str, Any] | None = None) -> dict[str, str]:
        """Regenerate the derived files from the frozen facts with no network call; {file: sha256 | reason}. In place
        only before the run closes: a closed run's files are sealed in manifest_run.json and its PDF never renders to
        the same bytes twice, so it is rebuilt into out."""
        run = self.load_run(run_id)
        th = thresholds or self.run_th(run)
        if out is None:
            if self.is_closed(run_id):
                raise RebuildRefused(f"run {run_id} is closed (bundle/manifest_run.json seals its derived files, sheet and PDF); rebuild it into --out DIR")
            return self._rebuild_in_place(run, th)
        out = Path(out)
        self._check_rebuild_out(run_id, out)
        target = out / run_id
        if target.exists():
            shutil.rmtree(target)
        never = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", PENDING_EVENTS_NAME)  # no unblurred copy
        shutil.copytree(paths.run_dir(run_id), target, ignore=shutil.ignore_patterns(*never))
        rows = [r for r in self.ledger_rows() if r.get("run_id") == run_id]  # the sheet prints the run's ledger totals
        (out / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        with _runs_root(out):
            return self._rebuild_in_place(run, th)

    def _rebuild_in_place(self, run: c.RunManifest, th: dict[str, Any]) -> dict[str, str]:
        """In stage order, with no call: coverage, marks, narration (it puts the seed rows back into marks.json), Stays
        Put (match reads mark_tracks.json), match with the second looks replayed from looks.json, the ruler's sizes,
        the claims, the sheet."""
        run_id = run.run_id
        result: dict[str, str] = {}
        result["coverage.json"] = self._rebuild_coverage(run, th)
        result.update(self._rebuild_marks(run, th))
        result.update(self._rebuild_narration(run, th))
        result.update(self._rebuild_staysput(run, th))
        if run.kind == c.RunKind.PAIR and self.has_dep("match.derive_match"):
            try:
                if self.has_dep("second_pass.apply_second_pass") and self.dep("second_pass.apply_second_pass")(run_id, th=th) and paths.marks_json(run_id).is_file():
                    result["marks.json"] = sha256_path(paths.marks_json(run_id))
                replay = self.dep("second_look.replay_second_look")(run_id, th=th) if self.has_dep("second_look.replay_second_look") else None
                self.dep("match.derive_match")(run_id, th=th, second_look=replay)
                if self.has_dep("ruler.size_verdicts"):
                    self.dep("ruler.size_verdicts")(run_id, th=th)
                result["matches.json"] = sha256_path(paths.matches_json(run_id))
                result["verdicts.json"] = sha256_path(paths.verdicts_json(run_id))
            except Exception as exc:
                result["matches.json"] = result["verdicts.json"] = f"failed: {_error_text(exc)}"
        result.update(self._rebuild_claims(run, th))
        if self.has_dep("sheet.write_sheet"):
            try:
                self.dep("sheet.write_sheet")(run_id, th=th)
                result["sheet.json"] = sha256_path(paths.sheet_json(run_id))
            except Exception as exc:
                result["sheet.json"] = f"failed: {_error_text(exc)}"
        return result

    def _rebuild_staysput(self, run: c.RunManifest, th: dict[str, Any]) -> dict[str, str]:
        """mark_tracks.json from the stored clip plans, clip lanes and keyframe crops (staysput.rebuild_staysput, no call)."""
        run_id = run.run_id
        if not self.has_dep("staysput.rebuild_staysput"):
            return {"mark_tracks.json": NOT_REBUILT}
        try:
            self.dep("staysput.rebuild_staysput")(run_id, th=th)
        except Exception as exc:
            return {"mark_tracks.json": f"failed: {_error_text(exc)}"}
        path = paths.derived_dir(run_id) / "mark_tracks.json"
        return {"mark_tracks.json": sha256_path(path) if path.is_file() else "not rebuilt: the Stays Put stage never ran"}

    def _rebuild_marks(self, run: c.RunManifest, th: dict[str, Any]) -> dict[str, str]:
        """marks.json, and with it calibration.json's pooled sigma_view, from spark.json and keyframes.json for each walk
        the marks stage ran on (marks.rebuild_marks, no call)."""
        run_id = run.run_id
        if not paths.marks_json(run_id).is_file():
            return {"marks.json": "not rebuilt: marks.json missing (the marks stage never ran)"}
        if not paths.spark_json(run_id).is_file() or not self.has_dep("marks.rebuild_marks"):
            return {"marks.json": NOT_REBUILT}
        try:
            for role in self.expected_roles(run):
                self.dep("marks.rebuild_marks")(run_id, role, th=th)
        except Exception as exc:
            return {"marks.json": f"failed: {_error_text(exc)}"}
        out = {"marks.json": sha256_path(paths.marks_json(run_id))}
        if paths.calibration_json(run_id).is_file():
            out["calibration.json"] = sha256_path(paths.calibration_json(run_id))
        return out

    def _rebuild_narration(self, run: c.RunManifest, th: dict[str, Any]) -> dict[str, str]:
        """narration.json and the seed rows of marks.json (marks.rebuild_marks dropped them) from voice.json, seeds.json and
        spark.json (narration.rebuild_narration, no call); before the match, which reads the seeds."""
        run_id = run.run_id
        if not paths.narration_json(run_id).is_file():
            return {"narration.json": "not rebuilt: narration.json missing (the narration stage never ran)"}
        if not self.has_dep("narration.rebuild_narration"):
            return {"narration.json": NOT_REBUILT}
        try:
            self.dep("narration.rebuild_narration")(run_id, th=th)
        except Exception as exc:
            return {"narration.json": f"failed: {_error_text(exc)}"}
        out = {"narration.json": sha256_path(paths.narration_json(run_id))}
        pieces = paths.derived_dir(run_id) / "narration_pieces.json"
        if pieces.is_file():
            out["narration_pieces.json"] = sha256_path(pieces)
        if paths.marks_json(run_id).is_file():
            out["marks.json"] = sha256_path(paths.marks_json(run_id))
        return out

    def _rebuild_claims(self, run: c.RunManifest, th: dict[str, Any]) -> dict[str, str]:
        """Every claim/<claim_id>.json from its recorded answers and pixelated photo, and derived/claim.json from the
        claim it already holds (claim.rebuild_claims, no call); nothing when the run has no claim."""
        run_id = run.run_id
        folder = paths.runs_root() / run_id / "claim"
        if not (folder.is_dir() and any(folder.glob("claim_*.answers.json"))) and not paths.claim_json(run_id).is_file():
            return {}
        if not self.has_dep("claim.rebuild_claims"):
            return {"claim.json": NOT_REBUILT}
        try:
            return dict(self.dep("claim.rebuild_claims")(run_id, th=th))
        except Exception as exc:
            return {"claim.json": f"failed: {_error_text(exc)}"}

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
            coverage.rerun_seconds = prior.rerun_seconds
            return c.write_json(paths.coverage_json(run_id), coverage)
        except Exception as exc:
            return f"failed: {_error_text(exc)}"

    def _any_fps(self, run: c.RunManifest) -> float:
        for role in self.expected_roles(run):
            video = self.load_video(run.run_id, role)
            if video is not None and video.proxy is not None:
                return video.proxy.fps
        return float(self.run_th(run)["media"]["proxy_fps"])

    def verify(self, run_id: str, *, out: Path | None = None) -> list[str]:
        """Problems found: a rebuilt derived file that differs, a missing or broken chain, a sheet head the chain does
        not hold, a manifest entry that changed. The default scratch copy is removed afterwards (it holds copies of
        the run's artifacts, which the retention sweep would never reach)."""
        problems: list[str] = []
        scratch = Path(out) if out is not None else paths.runs_root() / "_verify" / run_id
        before = {f"derived/{name}": sha256_path(paths.derived_dir(run_id) / name) for name in derived_present(run_id)}
        if paths.sheet_json(run_id).is_file():
            before["sheet/sheet.json"] = sha256_path(paths.sheet_json(run_id))
        try:
            rebuilt = self.rebuild(run_id, out=scratch)
        finally:
            if out is None:
                shutil.rmtree(scratch, ignore_errors=True)
        for name, value in sorted(rebuilt.items()):
            if len(value) != 64 or not all(ch in "0123456789abcdef" for ch in value):
                continue
            key = "sheet/sheet.json" if name == "sheet.json" else f"derived/{name}"
            if before.get(key, "") not in ("", value):
                problems.append(f"rebuild differs: {key}")
        problems += self._log_problems(run_id)
        if self.has_dep("seal.verify_run"):
            try:
                for problem in self.dep("seal.verify_run")(run_id):
                    if problem.startswith("event chain broken") and any(p.startswith("event chain broken") for p in problems):
                        continue
                    if problem not in problems:
                        problems.append(problem)
            except Exception as exc:
                problems.append(f"seal.verify_run failed: {_error_text(exc)}")
        return problems

    # Rederive (R28) ----------------------------------------------------------------------------------------------------

    # the source's events from the first of these on are its own outputs and run end; the new run writes its own
    REDERIVE_CUT_KINDS = frozenset(
        {
            c.EventKind.MATCH_DERIVED,
            c.EventKind.SHEET_WRITTEN,
            c.EventKind.FILES_DELETED,
            c.EventKind.RESPONSES_DELETED,
            c.EventKind.QUARANTINE_DELETED,
            c.EventKind.RUN_PARTIAL,
            c.EventKind.RUN_COMPLETED,
            c.EventKind.RUN_FAILED,
        }
    )
    # copied from the source but written again by the new run (its sheet, its run-end seal, run.json, the log)
    REDERIVE_SKIP = ("sheet", "manifest_run.json", "manifest_run.json.ots", "letter.md", "note.md", "verify.md", "run.json", "events.ndjson")

    def rederive(self, run_id: str, new_run_id: str) -> c.RunManifest:
        """walkaround rederive RUN NEW_RUN (lead decision 5 of 11/2): a new run made from RUN's sealed originals (the same
        bundle/manifest.json and its stamps, so the seal over the originals still holds) and RUN's stored model answers
        (SAM tracks and keyframes, spark.json, voice.json, looks.json, the clip lanes), recomputed with this version's
        code and thresholds and no model call: coverage, marks, narration, Stays Put, the match and verdicts, the ruler,
        the claims and the sheet, the way rebuild recomputes them. Its event log is RUN's log up to RUN's first output
        event (match.derived, or the sheet on a pickup check), byte for byte, then its own events; its run-end
        manifest_run.json, chain head, config hash, app commit and PDF hash are its own. run.json, manifest_run.json and
        the sheet say what was recomputed, when, and from which stored answers. walkaround verify NEW_RUN rebuilds it."""
        if not self.run_exists(run_id):
            raise RebuildRefused(f"no run {run_id}")
        src = self.load_run(run_id)
        if not paths.spark_json(run_id).is_file() and src.live_or_replay == c.LiveOrReplay.REPLAY:
            raise RebuildRefused(f"run {run_id} is a replay copy without its stored model answers; rederive needs spark.json")
        if not paths.manifest_json(run_id).is_file():
            raise RebuildRefused(f"run {run_id} has no sealed originals (bundle/manifest.json)")
        if src.status not in (c.RunStatus.DONE, c.RunStatus.PARTIAL):
            raise RebuildRefused(f"run {run_id} is {src.status.value}; rederive works on a finished run")
        _check_new_run_id(new_run_id)
        source_dir, target = paths.run_dir(run_id), paths.runs_root() / new_run_id
        never = ("quarantine", "uploads", "*.orig", "*.part", "*.tmp", "*.proxy.mp4", PENDING_EVENTS_NAME, *self.REDERIVE_SKIP)
        shutil.copytree(source_dir, target, ignore=shutil.ignore_patterns(*never))
        try:
            return self._rederive_into(src, new_run_id)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise

    def _rederive_into(self, src: c.RunManifest, new_run_id: str) -> c.RunManifest:
        run_id = src.run_id
        lines = [line for line in (paths.run_dir(run_id) / "events.ndjson").read_text(encoding="utf-8").splitlines() if line.strip()]
        kept: list[str] = []
        for line in lines:
            if c.EventKind(json.loads(line)["kind"]) in self.REDERIVE_CUT_KINDS:
                break
            kept.append(line)
        (paths.runs_root() / new_run_id / "events.ndjson").write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        from_head = json.loads(kept[-1])["hash"] if kept else None
        now = self.now()
        stored_on = _et_day(src.created_at)
        recomputed_on = _et_day(now)
        commit = app_commit()
        text = (
            f"verdicts recomputed on {recomputed_on} with this version's rules (code {commit[:7] if commit else 'unknown'}) "
            f"from the model answers stored on {stored_on} by run {run_id}; no new model calls"
        )
        th = json.loads(c.canonical_json(self.th))  # the thresholds as run.json will hold them (6-place floats)
        deadline = None
        if src.retention_deadline is not None:
            deadline = (parse_ts(now) + (parse_ts(src.retention_deadline) - parse_ts(src.created_at))).strftime("%Y-%m-%dT%H:%M:%SZ")
        run = src.model_copy(
            update={
                "run_id": new_run_id,
                "created_at": now,
                "status": c.RunStatus.MATCHING if src.kind == c.RunKind.PAIR else c.RunStatus.SHEET,
                "app_commit": commit,
                "config_hash": c.config_hash(th),
                "thresholds": th,
                "api_spend_cents": 0,
                "events_head_hash": None,
                "errors": [],
                "retention_deadline": deadline,
                "rederived": c.Rederived(
                    from_run=run_id,
                    from_events_head_hash=from_head,
                    from_config_hash=src.config_hash,
                    from_app_commit=src.app_commit,
                    stored_on=stored_on,
                    recomputed_on=recomputed_on,
                    text=text,
                ),
            }
        )
        self.save_run(run)
        run = self.load_run(new_run_id)
        th = self.run_th(run)
        rows = self._load_deletions(new_run_id)  # what RUN deleted (its originals, proxies, uploads), as recorded
        self.emit(new_run_id, c.EventKind.STAGE, {"name": run.status.value, "pct": 88, "rederived_from": run_id})
        result: dict[str, str] = {"coverage.json": self._rebuild_coverage(run, th)}
        result.update(self._rebuild_marks(run, th))
        result.update(self._rebuild_narration(run, th))
        result.update(self._rebuild_staysput(run, th))
        if run.kind == c.RunKind.PAIR and self.has_dep("match.derive_match"):
            if self.has_dep("second_pass.apply_second_pass"):
                self.dep("second_pass.apply_second_pass")(new_run_id, th=th)
            replay = self.dep("second_look.replay_second_look")(new_run_id, th=th) if self.has_dep("second_look.replay_second_look") else None
            matches, verdicts, _cal = self.dep("match.derive_match")(new_run_id, th=th, second_look=replay)
            if self.has_dep("ruler.size_verdicts"):
                self.dep("ruler.size_verdicts")(new_run_id, th=th)
            self.emit(
                new_run_id,
                c.EventKind.MATCH_DERIVED,
                {
                    "sha256": {"matches": sha256_path(paths.matches_json(new_run_id)), "verdicts": sha256_path(paths.verdicts_json(new_run_id))},
                    "panels": len(matches),
                    "verdicts": len(verdicts),
                    "rederived": {"from_run": run_id, "from_events_head_hash": from_head, "text": text},
                },
            )
            pairs = sum(len(pm.pairs) for pm in matches)
            self.update_run(new_run_id, lambda r: setattr(r.counts, "matches", pairs))
        failed = {k: v for k, v in {**result, **self._rebuild_claims(run, th)}.items() if v.startswith("failed")}
        if failed:
            raise RuntimeError("rederive failed: " + "; ".join(f"{k} {v}" for k, v in sorted(failed.items())))
        self._write_manifest_run_sync(self.load_run(new_run_id), rows, self.events.head_hash(new_run_id) or "")  # release: the sheet's verify reads it
        self.set_stage(new_run_id, c.RunStatus.SHEET, 92)
        head = self.events.head_hash(new_run_id)
        self.update_run(new_run_id, lambda r: setattr(r, "events_head_hash", head))
        _model, hashes = self.dep("sheet.write_sheet")(new_run_id, th=th)
        self.emit(new_run_id, c.EventKind.SHEET_WRITTEN, {"sha256": hashes})
        self._write_manifest_run_sync(self.load_run(new_run_id), rows, self.events.head_hash(new_run_id) or "")  # close
        self.update_run(new_run_id, lambda r: setattr(r, "status", c.RunStatus.DONE))
        self.emit(new_run_id, c.EventKind.STAGE, {"name": c.RunStatus.DONE.value, "pct": 100})
        self.emit(new_run_id, c.EventKind.RUN_COMPLETED, {"status": c.RunStatus.DONE.value, "api_spend_cents": 0})
        return self.load_run(new_run_id)

    def _log_problems(self, run_id: str) -> list[str]:
        """The chain of a sealed run: present (seq 0 is chained the moment manifest.json exists, so an empty log means
        it was removed), intact, and holding the head the sheet printed."""
        root = paths.runs_root() / run_id
        manifest = root / "bundle" / "manifest.json"
        if not manifest.is_file():
            return []
        log = root / "events.ndjson"
        lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()] if log.is_file() else []
        if not lines:
            return ["event log missing: the run is sealed, so events.ndjson must start at seq 0"]
        problems = []
        if hasattr(self.events, "verify_chain"):
            try:
                intact = self.events.verify_chain(run_id, manifest_sha256=sha256_path(manifest))
            except Exception:  # a line that is not an event at all
                intact = False
            if not intact:
                problems.append("event chain broken")
        head = self.load_run(run_id).events_head_hash
        if head and head not in {_hash_of(line) for line in lines}:
            problems.append("events head hash printed on the sheet is not in the event log")
        return problems


# Module-level helpers ------------------------------------------------------------------------------------------------


def _check_new_run_id(run_id: str) -> None:
    """A chosen run id: letters, digits, _ and -, not starting with _ or . (the server's own folders), not taken."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"run id {run_id!r}: use 1 to 64 letters, digits, _ or -, starting with a letter or digit")
    if (paths.runs_root() / run_id).exists():
        raise FileExistsError(f"run {run_id} already exists; pick another id, or resume it")


def _check_nouns(noun_list: list[c.Noun] | None, prompts: dict[str, str] | None) -> tuple[list[c.Noun], dict[str, str]]:
    """The lanes a run sends, in the fixed order of section 5 step 5, and its stand-in prompts. Car, license plate and
    face are always sent: the car filter and the blur need them."""
    nouns = list(c.LANE_NOUNS) if noun_list is None else [c.Noun(n) for n in noun_list]
    unknown = [n.value for n in nouns if n not in c.LANE_NOUNS]
    if unknown:
        raise ValueError(f"not lane nouns: {', '.join(unknown)}")
    nouns = [n for n in c.LANE_NOUNS if n in set(nouns) | set(REQUIRED_NOUNS)]
    clean: dict[str, str] = {}
    for key, text in (prompts or {}).items():
        noun = c.Noun(key)
        if noun not in nouns:
            raise ValueError(f"a stand-in prompt for {noun.value}, which this run does not send")
        if noun is c.Noun.CAR and text == WHOLE_FRAME:
            clean[noun.value] = WHOLE_FRAME
            continue
        phrase = " ".join(str(text).split())
        if not phrase or len(phrase) > 64 or not phrase.isascii():
            raise ValueError(f"stand-in prompt for {noun.value}: a short ASCII noun phrase, got {text!r}")
        if phrase != noun.value:
            clean[noun.value] = phrase
    return nouns, clean


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _wav_seconds(path: Path) -> float:
    """The WAV's own length (frames over rate), not the container's duration."""
    with wave.open(str(path), "rb") as fh:
        return fh.getnframes() / float(fh.getframerate())


def _frame_digest(rgb: Any) -> str:
    import numpy as np

    arr = np.ascontiguousarray(rgb)
    return f"{hashlib.sha256(arr.tobytes()).hexdigest()}:{arr.shape}:{arr.dtype}"


def derived_present(run_id: str) -> list[str]:
    return [name for name in paths.DERIVED_FILES if (paths.derived_dir(run_id) / name).is_file()]


def _et_day(ts: str) -> str:
    """The ET calendar day of a timestamp, as the sheet prints dates: Nov 2, 2026."""
    d = parse_ts(ts).astimezone(ET)
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def _chosen_by(run: c.RunManifest) -> str | None:
    return "renter at upload" if run.keep_plate else None


def _hash_of(line: str) -> str | None:
    try:
        value = json.loads(line).get("hash")
    except (ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


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
    Appends may come from worker threads, so the head is cached under a lock and subscribers are woken through
    their own loop."""

    def __init__(self) -> None:
        self._heads: dict[str, tuple[int, str]] = {}
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[c.Event]]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _path(run_id: str) -> Path:
        return paths.runs_root() / run_id / "events.ndjson"

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
                manifest = paths.runs_root() / run_id / "bundle" / "manifest.json"
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
    "CREATABLE_KINDS",
    "CapsExceeded",
    "FileEventLog",
    "HashMismatch",
    "MissingChunks",
    "NoSpace",
    "Orchestrator",
    "PENDING_EVENTS_NAME",
    "RebuildRefused",
    "STAGES",
    "STAGE_BY_NAME",
    "Stage",
    "StageGap",
    "TERMINAL_EVENTS",
    "TERMINAL_STATUSES",
    "UNREADABLE_VIDEO",
    "UploadRefused",
    "VideoUnreadable",
    "app_commit",
    "derived_present",
    "et_day_of",
    "new_run_id",
    "parse_ts",
    "role_order",
    "sha256_path",
    "utc_now",
    "write_json_atomic",
]
