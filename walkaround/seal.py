"""The seal (docs/ARCHITECTURE.md sections 5 step 2 and 7; docs/CONTRACTS.md seal M4).

bundle/manifest.json holds the sha256 and byte count of every original. It is written once and never rewritten,
because its bytes are what `ots stamp` and the RFC 3161 token cover and what events.ndjson chains from (prev of seq 0).
The stamp results therefore live in the SealManifest returned by seal_run (for run.json and the seal.manifest event)
and are persisted in bundle/manifest_run.json at run end. OpenTimestamps and the freetsa token are typed hooks here:
the functions exist with their contract signatures and raise NotImplementedError until the stamping lane fills them;
seal_run leaves a stamp None while its hook is not built ("not stamped" on the sheet), records "failed:<reason>" for
a built hook that raised, and never fails the seal over a stamp.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from . import paths
from .contracts import (
    CAMERA_TIME_NOT_PRESENT,
    DeletionRecord,
    Event,
    ManifestEntry,
    RunManifest,
    SealManifest,
    Stamps,
    ThreeClocks,
    read_json,
    write_json,
)

NOT_PRESENT = "not present"
FREETSA_URL = "https://freetsa.org/tsr"
DIGICERT_TSA_URL = "http://timestamp.digicert.com"
OTS_INSTALL_LINE = "pip install opentimestamps-client==0.7.2 (the `ots` binary)"
_RUN_EXCLUDED_TOPS = ("quarantine", "uploads")
# Files that change after manifest_run.json is written (run.completed lands in the event log and run.json after the
# close stage) or that are the manifest itself. The chain head pins events.ndjson instead.
_RUN_EXCLUDED_FILES = (
    "run.json",
    "events.ndjson",
    "events.ndjson.gz",
    "bundle/manifest_run.json",
    "bundle/manifest_run.json.ots",
)


# Time helpers ------------------------------------------------------------------------------------------------------


def _parse_iso(text: str) -> datetime:
    dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_time(text: str | None) -> str:
    """ffprobe creation_time ('2026-10-27T15:04:05.000000Z') to the contract's second-precision UTC form."""
    if not text or text.strip().lower() in ("", NOT_PRESENT, CAMERA_TIME_NOT_PRESENT):
        return NOT_PRESENT
    try:
        return _iso_utc(_parse_iso(text))
    except ValueError:
        return NOT_PRESENT


def camera_time_of(path: Path) -> str:
    """The container's creation_time (format tag, then the first stream tag) via ffprobe, or 'not present'."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "format_tags=creation_time:stream_tags=creation_time",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False).stdout
        doc = json.loads(out or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return NOT_PRESENT
    candidates = [doc.get("format", {}).get("tags", {}).get("creation_time")]
    candidates += [s.get("tags", {}).get("creation_time") for s in doc.get("streams", [])]
    for value in candidates:
        if value:
            return normalize_time(value)
    return NOT_PRESENT


def minutes_after_camera(sealed_at: str, creation_time: str) -> int | Literal["camera time not present"]:
    """Whole minutes (rounded) from the camera's creation_time to sealed_at; the literal when the camera time is absent."""
    if normalize_time(creation_time) == NOT_PRESENT:
        return CAMERA_TIME_NOT_PRESENT
    delta = _parse_iso(sealed_at) - _parse_iso(creation_time)
    return int(round(delta.total_seconds() / 60))


# Hashing and the manifest -----------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _manifest_path_of(run_id: str, path: Path) -> str:
    """Run-relative path for a file inside the run dir; the bare name for a file outside it."""
    try:
        return paths.rel(run_id, path)
    except ValueError:
        return Path(path).name


def _entry(run_id: str, path: Path) -> ManifestEntry:
    path = Path(path)
    return ManifestEntry(path=_manifest_path_of(run_id, path), sha256=sha256_file(path), bytes=path.stat().st_size)


def build_manifest(run_id: str, files: list[Path]) -> SealManifest:
    """ManifestEntry per file (run-relative path, sha256, bytes), sorted by path; writes bundle/manifest.json."""
    entries = sorted((_entry(run_id, p) for p in files), key=lambda e: e.path)
    config_hash = app_commit = None
    run_file = paths.run_dir(run_id) / "run.json"
    if run_file.is_file():
        run = read_json(run_file, RunManifest)
        config_hash, app_commit = run.config_hash, run.app_commit
    manifest = SealManifest(run_id=run_id, files=entries, config_hash=config_hash, app_commit=app_commit)
    write_json(paths.manifest_json(run_id), manifest)
    return manifest


# Stamp hooks (not built in this lane; signatures per CONTRACTS.md) ------------------------------------------------


def stamp_ots(path: Path) -> Path:
    """`ots stamp <path>` (never --wait); returns <path>.ots. Raises FileNotFoundError with the install line when the
    binary is missing. Hook: not built in this lane."""
    raise NotImplementedError(f"OpenTimestamps stamping is not built in this lane; it needs {OTS_INSTALL_LINE}")


def ots_status(ots_path: Path) -> str:
    """'pending', 'confirmed:<block height>' or 'failed:<reason>' from `ots info <ots_path>`. Hook."""
    raise NotImplementedError("ots_status is not built in this lane (parse `ots info` for BitcoinBlockHeaderAttestation)")


def upgrade_ots(ots_path: Path) -> str:
    """`ots upgrade <ots_path>` (the hourly task) and then ots_status. Hook."""
    raise NotImplementedError("upgrade_ots is not built in this lane")


def stamp_tsa(path: Path, *, tsa_url: str = FREETSA_URL, timeout_s: float = 20.0) -> tuple[Path, Path, str, str]:
    """`openssl ts -query -data <path> -sha256 -cert -out <path>.tsq`, POST with Content-Type
    application/timestamp-query, <path>.tsr saved; returns (tsq, tsr, tsr_time ISO, serial). DigiCert
    (DIGICERT_TSA_URL) is the fallback when freetsa fails. Hook."""
    raise NotImplementedError("RFC 3161 stamping is not built in this lane")


def verify_tsr(path: Path, tsr: Path, *, cacert: Path, tsa_crt: Path) -> bool:
    """`openssl ts -verify -data <path> -in <tsr> -CAfile <cacert> -untrusted <tsa_crt>`. Hook."""
    raise NotImplementedError("verify_tsr is not built in this lane")


def _try_ots(path: Path) -> str | None:
    """Stamp one file; a hook that raises records the reason and never fails the seal. None while the hook is not
    built, so the sheet prints "not stamped" rather than a failure."""
    try:
        ots_path = stamp_ots(path)
    except NotImplementedError:
        return None
    except Exception as exc:  # noqa: BLE001 - calendars down, binary missing: section 14 item 24
        return f"failed:{type(exc).__name__}: {exc}"[:300]
    try:
        return ots_status(ots_path)
    except Exception:  # noqa: BLE001 - the stamp exists; its status is read by the hourly upgrade
        return "pending"


def _try_tsa(path: Path) -> tuple[str, str, str] | None:
    """(tsr_time, serial, issuer) or None when both TSAs failed or the hook is not built."""
    for url, issuer in ((FREETSA_URL, "freetsa"), (DIGICERT_TSA_URL, "DigiCert")):
        try:
            _, _, tsr_time, serial = stamp_tsa(path, tsa_url=url)
            return tsr_time, serial, issuer
        except NotImplementedError:
            return None
        except Exception:  # noqa: BLE001 - try the fallback issuer
            continue
    return None


def _role_of(path: Path) -> str:
    """quarantine/pickup.orig -> pickup; IMG_0001.MOV -> IMG_0001."""
    return Path(path).name.split(".", 1)[0]


# The seal, the run-end manifest, verify --------------------------------------------------------------------------


def seal_run(run_id: str, originals: list[Path], *, now: str, camera_time: str, stamp: bool = True) -> tuple[SealManifest, ThreeClocks]:
    """manifest.json over the originals, the stamp hooks on the manifest and each original, the three clocks.
    first_model_call_at stays None (the orchestrator fills it at the first Files upload). The returned manifest
    carries the stamps in memory; the file on disk keeps the bytes the stamps cover."""
    manifest = build_manifest(run_id, originals)
    stamps = Stamps()
    if stamp:
        manifest_path = paths.manifest_json(run_id)
        stamps.manifest_ots = _try_ots(manifest_path)
        for original in originals:
            status = _try_ots(Path(original))
            role = _role_of(original)
            if role == "pickup":
                stamps.pickup_ots = status
            elif role == "return":
                stamps.return_ots = status
        tsa = _try_tsa(manifest_path)
        if tsa is not None:
            stamps.tsr_time, stamps.tsa_serial, stamps.tsa_issuer = tsa
    manifest = manifest.model_copy(update={"stamps": stamps})
    camera = normalize_time(camera_time)
    clocks = ThreeClocks(
        sealed_at=now,
        first_model_call_at=None,
        sealed_minutes_after_camera=minutes_after_camera(now, camera),
        camera_time=None if camera == NOT_PRESENT else camera,
    )
    return manifest, clocks


def _kept_artifacts(run_id: str) -> list[Path]:
    root = paths.run_dir(run_id)
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.split("/", 1)[0] in _RUN_EXCLUDED_TOPS or rel in _RUN_EXCLUDED_FILES:
            continue
        out.append(p)
    return out


def _events(run_id: str) -> list[str]:
    events_path = paths.run_dir(run_id) / "events.ndjson"
    if not events_path.is_file():
        return []
    return [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stamps_from_events(run_id: str) -> Stamps:
    """The seal-time stamps live in the seal.ots and seal.tsr events (manifest.json itself is never rewritten); the
    last seal.ots wins, so the hourly upgrade's confirmed status replaces pending."""
    stamps = Stamps()
    for line in _events(run_id):
        try:
            event = Event.model_validate_json(line)
        except ValueError:
            continue
        if event.kind.value == "seal.ots" and isinstance(event.payload.get("status"), str):
            stamps.manifest_ots = event.payload["status"]
        elif event.kind.value == "seal.tsr":
            stamps.tsr_time = event.payload.get("tsr_time") or stamps.tsr_time
            stamps.tsa_serial = event.payload.get("serial") or stamps.tsa_serial
            stamps.tsa_issuer = event.payload.get("issuer") or stamps.tsa_issuer
    return stamps


def write_manifest_run(
    run_id: str,
    *,
    deletions: list[DeletionRecord],
    events_head_hash: str,
    config_hash: str,
    app_commit: str,
    pdf_sha256: str | None,
    stamps: Stamps | None = None,
) -> SealManifest:
    """manifest_run.json: every kept artifact (quarantine, upload chunks, run.json and the event log excluded; the
    chain head covers the log), the deletions, the chain head, the config hash, the app commit, the PDF hash and the
    seal-time stamps (from the seal events when not given); then the stamp hook on the file."""
    entries = [_entry(run_id, p) for p in _kept_artifacts(run_id)]
    manifest = SealManifest(
        run_id=run_id,
        files=entries,
        stamps=stamps if stamps is not None else _stamps_from_events(run_id),
        deletions=list(deletions),
        events_head_hash=events_head_hash,
        config_hash=config_hash,
        app_commit=app_commit,
        pdf_sha256=pdf_sha256,
    )
    path = paths.manifest_run_json(run_id)
    write_json(path, manifest)
    _try_ots(path)
    return manifest


def verify_manifest(manifest: SealManifest, run_dir: Path, *, expected_missing: set[str] = frozenset(), originals: list[Path] | None = None) -> list[str]:
    """Re-hash every entry. A missing entry is a problem unless it is in expected_missing (deleted at run end) or one
    of the renter's provided originals carries its sha256."""
    provided = {sha256_file(p): p for p in (originals or [])}
    problems: list[str] = []
    for entry in manifest.files:
        path = Path(run_dir) / entry.path
        if path.is_file():
            if path.stat().st_size != entry.bytes:
                problems.append(f"size differs: {entry.path}")
            if sha256_file(path) != entry.sha256:
                problems.append(f"hash differs: {entry.path}")
            continue
        if entry.sha256 in provided:
            continue
        if entry.path in expected_missing:
            if originals is not None:
                problems.append(f"no provided file matches {entry.path} (sha256 {entry.sha256[:12]})")
            continue
        problems.append(f"missing: {entry.path}")
    return problems


def _chain_problems(run_id: str, manifest_sha256: str, *, head_hash: str | None = None) -> list[str]:
    """The event chain through eventlog.verify_chain when that lane is present, else the same rule from contracts.
    head_hash, when given (manifest_run.json's events_head_hash), must be the hash of some event: the log at manifest
    time is then a prefix of the log now."""
    lines = _events(run_id)
    if not lines:
        return []
    try:
        from . import eventlog  # the eventlog lane; optional at verify time
    except ImportError:
        eventlog = None
    hashes: list[str] = []
    problems: list[str] = []
    if eventlog is not None and hasattr(eventlog, "verify_chain"):
        if not eventlog.verify_chain(run_id, manifest_sha256=manifest_sha256):
            problems.append("event chain broken")
        for line in lines:
            try:
                hashes.append(Event.model_validate_json(line).hash)
            except ValueError:
                continue
    else:
        prev = manifest_sha256
        for i, line in enumerate(lines):
            try:
                event = Event.model_validate_json(line)
            except ValueError:
                problems.append(f"event chain broken at line {i}: not an event")
                break
            if event.seq != i or event.prev != prev or not event.check():
                problems.append(f"event chain broken at seq {i}")
                break
            prev = event.hash
            hashes.append(event.hash)
    if head_hash and head_hash not in hashes:
        problems.append("run manifest: events head hash is not in the event chain")
    return problems


def _rebuild_problems(run_id: str, rebuild_dir: Path) -> list[str]:
    """Byte-compare the recomputed derived files (frozen model-era files excluded) and sheet.json with a rebuild."""
    root = paths.run_dir(run_id)
    names = [f"derived/{n}" for n in paths.DERIVED_FILES if n not in paths.FROZEN_DERIVED_FILES] + ["sheet/sheet.json"]
    for name in names:
        ours, theirs = root / name, Path(rebuild_dir) / name
        if ours.is_file() and theirs.is_file() and ours.read_bytes() != theirs.read_bytes():
            return [f"rebuild differs: {name}"]
    return []


def verify_run(run_id: str, *, rebuild_dir: Path | None = None, originals: list[Path] | None = None) -> list[str]:
    """Re-hash every manifest entry, check the event chain, compare derived files with a rebuild when one is given.
    An empty list means every check passed ('rebuild identical')."""
    root = paths.run_dir(run_id)
    manifest_path = paths.manifest_json(run_id)
    if not manifest_path.is_file():
        return ["bundle/manifest.json missing"]
    manifest = read_json(manifest_path, SealManifest)
    manifest_run_path = paths.manifest_run_json(run_id)
    manifest_run = read_json(manifest_run_path, SealManifest) if manifest_run_path.is_file() else None
    deleted = {d.target for d in (manifest_run.deletions if manifest_run else [])}
    problems = verify_manifest(manifest, root, expected_missing=deleted, originals=originals)
    if manifest_run is not None:
        problems += [f"run manifest: {p}" for p in verify_manifest(manifest_run, root)]
    head = manifest_run.events_head_hash if manifest_run is not None else None
    problems += _chain_problems(run_id, sha256_file(manifest_path), head_hash=head or None)
    if rebuild_dir is not None:
        problems += _rebuild_problems(run_id, rebuild_dir)
    return problems


def verify(run_id: str, *, rebuild_dir: Path | None = None, originals: list[Path] | None = None) -> list[str]:
    """verify_run under its short name."""
    return verify_run(run_id, rebuild_dir=rebuild_dir, originals=originals)


def status_text(problems: list[str]) -> str:
    """The method-page wording: 'rebuild identical' or the first problem."""
    return "rebuild identical" if not problems else problems[0]
