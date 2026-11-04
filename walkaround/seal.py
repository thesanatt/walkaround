"""The seal (docs/ARCHITECTURE.md section 5 step 2, section 7 items 1 to 3, section 14 item 24; CONTRACTS.md seal M4).

bundle/manifest.json holds the sha256 and byte count of every original. It is written once and never rewritten,
because its bytes are what `ots stamp` and the RFC 3161 token cover and what events.ndjson chains from (prev of seq 0).
So the stamp results travel in the SealManifest that seal_run returns: the orchestrator puts them in the seal.ots and
seal.tsr events, the sheet reads them back from the chain (up to the head it froze), and bundle/manifest_run.json
repeats them at run end. A stamp never fails the seal. OpenTimestamps needs the `ots` client (opentimestamps-client,
not in the venv); without it every OTS status reads "failed:<reason>". The RFC 3161 token comes from freetsa, with
DigiCert as the fallback and the issuer recorded.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx

from . import paths
from .contracts import (
    CAMERA_TIME_NOT_PRESENT,
    DeletionRecord,
    Event,
    EventKind,
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
TSA_ISSUERS = ((FREETSA_URL, "freetsa"), (DIGICERT_TSA_URL, "DigiCert"))
OTS_INSTALL_LINE = "pip install opentimestamps-client==0.7.2 (the `ots` binary)"
OTS_TIMEOUT_S = 120.0  # ots stamp and ots upgrade talk to the public calendars
OPENSSL_TIMEOUT_S = 30.0
_RUN_EXCLUDED_TOPS = ("quarantine", "uploads")
# Files that change after manifest_run.json is written (run.completed lands in the event log and run.json after the
# close stage) or that are the manifest itself. The chain head pins events.ndjson instead.
_RUN_EXCLUDED_FILES = (
    "run.json",
    "events.ndjson",
    "events.ndjson.gz",
    "bundle/manifest_run.json",
    paths.WORK_LOCK_NAME,  # held while the close stage writes manifest_run.json, gone right after
)
# An OpenTimestamps proof is rewritten in place by the hourly `ots upgrade` (and leaves a .bak while it runs); the
# proof carries its own commitment to the stamped file, so it is not hashed into manifest_run.json.
_RUN_EXCLUDED_SUFFIXES = (".ots", ".bak")
_MONTHS = {m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
_TSR_TIME = re.compile(r"([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?\s+(\d{4})\s+GMT")
_OTS_BLOCK = re.compile(r"BitcoinBlockHeaderAttestation\((\d+)\)")


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


# Subprocess helpers --------------------------------------------------------------------------------------------------


def _run(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)


def _last_line(proc: subprocess.CompletedProcess[str]) -> str:
    text = (proc.stderr or "").strip() or (proc.stdout or "").strip()
    return text.splitlines()[-1][:300] if text else ""


# OpenTimestamps ------------------------------------------------------------------------------------------------------


def _ots_binary() -> str:
    # The venv's own bin first, so .venv/bin/walkaround finds the client without an activated shell.
    found = shutil.which("ots", path=str(Path(sys.executable).parent)) or shutil.which("ots")
    if found is None:
        raise FileNotFoundError(f"the OpenTimestamps client is not installed; {OTS_INSTALL_LINE}")
    return found


def stamp_ots(path: Path) -> Path:
    """`ots stamp <path>` (never --wait); returns <path>.ots. FileNotFoundError with the install line when the client
    is missing; RuntimeError with ots's last line when the stamp fails (calendars unreachable)."""
    path = Path(path)
    binary = _ots_binary()
    ots_path = Path(f"{path}.ots")
    ots_path.unlink(missing_ok=True)  # a proof left by an interrupted seal; ots stamp refuses to overwrite one
    proc = _run([binary, "stamp", str(path)], timeout_s=OTS_TIMEOUT_S)
    if proc.returncode != 0 or not ots_path.is_file():
        raise RuntimeError(_last_line(proc) or f"ots stamp exited with {proc.returncode}")
    return ots_path


# TODO(D1 E12): opentimestamps-client is not installed here, so the parser and the .bak handling follow the client's
# documented output (verify PendingAttestation(...) / BitcoinBlockHeaderAttestation(<height>) lines; upgrade renames
# the old proof to .bak). Run `ots stamp`, `ots info` and, hours later, `ots upgrade` on a real proof to confirm.
def parse_ots_info(text: str) -> str:
    """'confirmed:<lowest Bitcoin block>', 'pending' or 'failed:<reason>' from what `ots info` prints."""
    heights = [int(h) for h in _OTS_BLOCK.findall(text)]
    if heights:
        return f"confirmed:{min(heights)}"
    if "PendingAttestation(" in text:
        return "pending"
    return "failed:no attestation in the proof"


def ots_status(ots_path: Path) -> str:
    """'pending', 'confirmed:<block height>' or 'failed:<reason>' from `ots info <ots_path>`."""
    ots_path = Path(ots_path)
    try:
        binary = _ots_binary()
    except FileNotFoundError as exc:
        return f"failed:{exc}"
    if not ots_path.is_file():
        return f"failed:{ots_path.name} not found"
    try:
        proc = _run([binary, "info", str(ots_path)], timeout_s=OPENSSL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return "failed:ots info timed out"
    if proc.returncode != 0:
        return "failed:" + (_last_line(proc) or f"ots info exited with {proc.returncode}")
    return parse_ots_info(proc.stdout)


def upgrade_ots(ots_path: Path) -> str:
    """`ots upgrade <ots_path>` (the hourly task), then ots_status. ots keeps the old proof as <name>.bak and refuses
    to start while one exists, so a stale .bak goes first and the new one once the upgraded proof is in place.
    FileNotFoundError without the client, so the hourly task emits nothing."""
    ots_path = Path(ots_path)
    binary = _ots_binary()
    bak = Path(f"{ots_path}.bak")
    _settle_bak(ots_path, bak)
    try:
        _run([binary, "upgrade", str(ots_path)], timeout_s=OTS_TIMEOUT_S)
    finally:
        _settle_bak(ots_path, bak)
    return ots_status(ots_path)


def _settle_bak(ots_path: Path, bak: Path) -> None:
    """Drop ots's backup when the proof is in place; put it back when an upgrade died between rename and write."""
    if not bak.is_file():
        return
    if ots_path.is_file():
        bak.unlink()
    else:
        bak.replace(ots_path)


# RFC 3161 ------------------------------------------------------------------------------------------------------------


def _openssl(args: list[str]) -> str:
    proc = _run(["openssl", *args], timeout_s=OPENSSL_TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(f"openssl {' '.join(args[:2])} failed: {_last_line(proc)}")
    return proc.stdout


def _post_tsq(url: str, body: bytes, *, timeout_s: float) -> bytes:
    """POST one RFC 3161 query and return the reply bytes (tests replace this with a local TSA)."""
    response = httpx.post(url, content=body, headers={"Content-Type": "application/timestamp-query"}, timeout=timeout_s, follow_redirects=True)
    response.raise_for_status()
    return response.content


def _field(text: str, label: str) -> str | None:
    """The value after `label` on the first line of openssl's -text output that starts with it."""
    for line in text.splitlines():
        if line.strip().startswith(label):
            return line.split(":", 1)[1].strip()
    return None


def _imprint(text: str) -> bytes:
    """The message imprint from the hex dump under 'Message data:' in `openssl ts -reply -text`."""
    out = bytearray()
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith("Message data:")), None)
    if start is None:
        return b""
    for line in lines[start + 1 :]:
        if not re.match(r"^\s*[0-9a-fA-F]{4} - ", line):
            break
        hex_part = line.split(" - ", 1)[1][:48].replace("-", " ")
        out += bytes(int(tok, 16) for tok in hex_part.split() if re.fullmatch(r"[0-9a-fA-F]{2}", tok))
    return bytes(out)


def _tsr_time(reply_text: str) -> str:
    match = _TSR_TIME.search(_field(reply_text, "Time stamp:") or "")
    if match is None:
        raise RuntimeError("the reply has no readable Time stamp line")
    month, day, hh, mm, ss, year = match.groups()
    return f"{int(year):04d}-{_MONTHS[month]:02d}-{int(day):02d}T{hh}:{mm}:{ss}Z"


def stamp_tsa(path: Path, *, tsa_url: str = FREETSA_URL, timeout_s: float = 20.0) -> tuple[Path, Path, str, str]:
    """`openssl ts -query -sha256 -cert` to <path>.tsq, a POST with Content-Type application/timestamp-query, the reply
    saved as <path>.tsr; returns (tsq, tsr, tsr_time ISO, serial). The reply must be granted, carry this file's
    sha256 and echo the query's nonce, else nothing is kept and RuntimeError says why. One TSA per call; seal_run
    tries freetsa, then DigiCert (section 14 item 24)."""
    path = Path(path)
    tsq, tsr = Path(f"{path}.tsq"), Path(f"{path}.tsr")
    try:
        _openssl(["ts", "-query", "-data", str(path), "-sha256", "-cert", "-out", str(tsq)])
        query = _openssl(["ts", "-query", "-in", str(tsq), "-text"])
        tsr.write_bytes(_post_tsq(tsa_url, tsq.read_bytes(), timeout_s=timeout_s))
        reply = _openssl(["ts", "-reply", "-in", str(tsr), "-text"])
        status = _field(reply, "Status:") or "no status"
        if not status.startswith("Granted"):
            raise RuntimeError(f"the TSA did not grant the stamp: {status}")
        if _imprint(reply) != bytes.fromhex(sha256_file(path)):
            raise RuntimeError("the token's message imprint is not this file's sha256")
        if _field(query, "Nonce:") != _field(reply, "Nonce:"):
            raise RuntimeError("the token does not echo the query's nonce")
        return tsq, tsr, _tsr_time(reply), _field(reply, "Serial number:") or ""
    except Exception:
        tsq.unlink(missing_ok=True)
        tsr.unlink(missing_ok=True)
        raise


def verify_tsr(path: Path, tsr: Path, *, cacert: Path, tsa_crt: Path) -> bool:
    """`openssl ts -verify -data <path> -in <tsr> -CAfile <cacert> -untrusted <tsa_crt>`; True on 'Verification: OK'."""
    cmd = ["openssl", "ts", "-verify", "-data", str(path), "-in", str(tsr), "-CAfile", str(cacert), "-untrusted", str(tsa_crt)]
    try:
        proc = _run(cmd, timeout_s=OPENSSL_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and "Verification: OK" in proc.stdout


def _try_ots(path: Path, *, keep_at: Path | None = None) -> str:
    """Stamp one file and return its status; a failure becomes 'failed:<reason>' and never fails the seal. keep_at
    moves the proof (an original's .ots would otherwise sit in quarantine and be deleted at run end)."""
    try:
        ots_path = Path(stamp_ots(path))
        if keep_at is not None and Path(keep_at) != ots_path:
            ots_path = ots_path.replace(keep_at)
    except Exception as exc:  # noqa: BLE001 - client missing, calendars down: section 14 item 24
        return f"failed:{type(exc).__name__}: {exc}"[:300]
    try:
        status = ots_status(ots_path)
    except Exception:  # noqa: BLE001 - the proof exists; the hourly upgrade reads it later
        return "pending"
    return "pending" if status.startswith("failed:") else status  # a stamp ots just wrote is pending by construction


def _try_tsa(path: Path) -> tuple[str, str, str] | None:
    """(tsr_time, serial, issuer) from the first TSA that grants a stamp, freetsa then DigiCert; None when both fail."""
    for url, issuer in TSA_ISSUERS:
        try:
            _, _, tsr_time, serial = stamp_tsa(path, tsa_url=url)
            return tsr_time, serial, issuer
        except Exception:  # noqa: BLE001 - try the next issuer
            continue
    return None


def _role_of(path: Path) -> str:
    """quarantine/pickup.orig -> pickup; IMG_0001.MOV -> IMG_0001."""
    return Path(path).name.split(".", 1)[0]


# The seal, the run-end manifest, verify --------------------------------------------------------------------------


def seal_run(run_id: str, originals: list[Path], *, now: str, camera_time: str, stamp: bool = True) -> tuple[SealManifest, ThreeClocks]:
    """manifest.json over the originals, OpenTimestamps on the manifest and on each original (proofs kept in bundle/
    as manifest.json.ots and <role>.ots), the freetsa token on the manifest, the three clocks. first_model_call_at
    stays None (the orchestrator fills it at the first Files upload). The returned manifest carries the stamps in
    memory; the file on disk keeps the bytes the stamps cover."""
    manifest = build_manifest(run_id, originals)
    stamps = Stamps()
    if stamp:
        manifest_path = paths.manifest_json(run_id)
        stamps.manifest_ots = _try_ots(manifest_path)
        for original in originals:
            role = _role_of(original)
            status = _try_ots(Path(original), keep_at=paths.ots_path(run_id, role))
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
        if rel.split("/", 1)[0] in _RUN_EXCLUDED_TOPS or rel in _RUN_EXCLUDED_FILES or rel.endswith(_RUN_EXCLUDED_SUFFIXES):
            continue
        out.append(p)
    return out


def _events(run_id: str) -> list[str]:
    events_path = paths.run_dir(run_id) / "events.ndjson"
    if not events_path.is_file():
        return []
    return [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def events_through(run_id: str, head_hash: str | None) -> list[Event]:
    """The run's events up to and including the one whose hash is head_hash: the log as it stood when that head was
    frozen. Every event when head_hash is None or is not in the log."""
    out: list[Event] = []
    for line in _events(run_id):
        try:
            event = Event.model_validate_json(line)
        except ValueError:
            continue
        out.append(event)
        if head_hash and event.hash == head_hash:
            break
    return out


def stamps_from_events(events: Iterable[Event]) -> Stamps:
    """The seal-time stamps from the seal.ots and seal.tsr events (manifest.json itself is never rewritten). The last
    seal.ots wins for the manifest, so the hourly upgrade's confirmed status replaces pending; the per-original
    statuses come from the seal stage's event, which carries pickup and return."""
    stamps = Stamps()
    for event in events:
        payload = event.payload
        if event.kind == EventKind.SEAL_OTS:
            for key, field in (("status", "manifest_ots"), ("pickup", "pickup_ots"), ("return", "return_ots")):
                if isinstance(payload.get(key), str):
                    setattr(stamps, field, payload[key])
        elif event.kind == EventKind.SEAL_TSR:
            stamps.tsr_time = payload.get("tsr_time") or stamps.tsr_time
            stamps.tsa_serial = payload.get("serial") or stamps.tsa_serial
            stamps.tsa_issuer = payload.get("issuer") or stamps.tsa_issuer
    return stamps


def _stamps_from_events(run_id: str) -> Stamps:
    return stamps_from_events(events_through(run_id, None))


def write_manifest_run(
    run_id: str,
    *,
    deletions: list[DeletionRecord],
    events_head_hash: str,
    config_hash: str,
    app_commit: str,
    pdf_sha256: str | None,
    stamps: Stamps | None = None,
    rederived: str | None = None,
) -> SealManifest:
    """manifest_run.json: every kept artifact (quarantine, upload chunks, run.json, the event log and the upgradeable
    .ots proofs excluded; the chain head covers the log), the deletions, the chain head, the config hash, the app
    commit, the PDF hash and the seal-time stamps (from the seal events when not given); then OpenTimestamps on the
    file itself (bundle/manifest_run.json.ots)."""
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
        rederived=rederived,
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


def chain_problems_through(run_id: str, manifest_sha256: str, *, head_hash: str | None) -> list[str]:
    """The chain rule (prev of seq 0 is manifest.json's sha256) checked from seq 0 up to head_hash only, so events
    appended after it (sheet.written, the deletions, run.completed, an hourly seal.ots) cannot change the answer.
    The whole log when head_hash is None; nothing to check without a log."""
    lines = _events(run_id)
    if not lines:
        return []
    prev = manifest_sha256
    for i, line in enumerate(lines):
        try:
            event = Event.model_validate_json(line)
        except ValueError:
            return [f"event chain broken at line {i}: not an event"]
        if event.seq != i or event.prev != prev or not event.check():
            return [f"event chain broken at seq {i}"]
        if head_hash and event.hash == head_hash:
            return []
        prev = event.hash
    return ["events head hash is not in the event chain"] if head_hash else []


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
