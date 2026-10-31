"""seal (M4): the manifest round trip, verify catching a flipped byte, the camera time from ffprobe, the three
clocks, OpenTimestamps through a fake `ots` client, `ots info` parsed on fixtures, the RFC 3161 token from a local
openssl TSA (stamp_tsa and verify_tsr, offline), manifest_run.json and the event chain checks. LIVE=1 adds one real
freetsa stamp verified against freetsa's published certificates (free, no model call)."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from walkaround import config
from walkaround import contracts as c
from walkaround import ledger, paths, seal

NOW = "2026-10-27T15:00:00Z"
CAMERA = "2026-10-27T14:37:00Z"
REAL_OTS_BINARY = seal._ots_binary
REAL_POST_TSQ = seal._post_tsq
needs_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not on PATH")

OTS_INFO_PENDING = """File sha256 hash: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
Timestamp:
append 1d42b1eb6e2c1d0a9ad5a3fc28fa0f1a
sha256
 -> append 6d62e0b4b9cfb2e9b5ffd6cb8a5b9f04
    sha256
    prepend 635f8a2c
    append 0f1a2c3d4e5f6a7b
    verify PendingAttestation('https://alice.btc.calendar.opentimestamps.org')
 -> append 9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d
    sha256
    prepend 635f8a2d
    append 1b2c3d4e5f6a7b8c
    verify PendingAttestation('https://bob.btc.calendar.opentimestamps.org')
"""
OTS_INFO_CONFIRMED = """File sha256 hash: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
Timestamp:
append 1d42b1eb6e2c1d0a9ad5a3fc28fa0f1a
sha256
 -> append 6d62e0b4b9cfb2e9b5ffd6cb8a5b9f04
    sha256
    prepend 635f8a2c
    sha256
    prepend 4b3e9e5c26cfd1bc4a6c0a1f2a3b4c5d
    sha256
    verify BitcoinBlockHeaderAttestation(921007)
 -> append 9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d
    sha256
    verify BitcoinBlockHeaderAttestation(921003)
 -> append 0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d
    verify PendingAttestation('https://finney.calendar.eternitywall.com')
"""
# A stand-in for the opentimestamps client: stamp writes <file>.ots (exit 1 for a file named *.fail), info prints the
# proof file itself, upgrade renames to .bak and writes a confirmed proof the way `ots upgrade` does.
FAKE_OTS = r"""
import os, sys
cmd, target = sys.argv[1], sys.argv[2]
if cmd == "stamp":
    if target.endswith(".fail"):
        print("Failed to create timestamp: all calendars unreachable", file=sys.stderr); sys.exit(1)
    with open(target + ".ots", "x") as fh:
        fh.write(PENDING)
elif cmd == "info":
    if not os.path.exists(target):
        print(f"Could not open {target}", file=sys.stderr); sys.exit(1)
    print(open(target).read())
elif cmd == "upgrade":
    bak = target + ".bak"
    if os.path.exists(bak):
        print(f"Could not backup timestamp: {bak!r} already exists", file=sys.stderr); sys.exit(1)
    os.rename(target, bak)
    with open(target, "x") as fh:
        fh.write(CONFIRMED)
"""


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    return root


@pytest.fixture(autouse=True)
def offline_stamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches a calendar or a TSA: no ots client unless a test installs the fake, every TSA unreachable
    unless a test installs the local one."""

    def no_ots() -> str:
        raise FileNotFoundError(f"the OpenTimestamps client is not installed; {seal.OTS_INSTALL_LINE}")

    def no_tsa(url: str, body: bytes, *, timeout_s: float) -> bytes:
        raise httpx.ConnectError(f"offline test: {url}")

    monkeypatch.setattr(seal, "_ots_binary", no_ots)
    monkeypatch.setattr(seal, "_post_tsq", no_tsa)


def _fake_ots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "bin" / "ots"
    script.parent.mkdir(parents=True, exist_ok=True)
    body = f"#!{sys.executable}\nPENDING = {OTS_INFO_PENDING!r}\nCONFIRMED = {OTS_INFO_CONFIRMED!r}\n{FAKE_OTS}"
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(seal, "_ots_binary", lambda: str(script))
    return script


class LocalTsa:
    """A throwaway CA and TSA made with openssl; reply() answers a query the way freetsa's endpoint does."""

    def __init__(self, root: Path, name: str = "Walkaround Test") -> None:
        self.dir = root
        root.mkdir(parents=True, exist_ok=True)
        self.cacert, self.crt, key = root / "cacert.pem", root / "tsa.crt", root / "tsa.key"
        ec = ["-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes"]
        self._ssl("req", "-x509", *ec, "-keyout", str(root / "ca.key"), "-out", str(self.cacert), "-days", "30", "-subj", f"/CN={name} Root",
                  "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign")
        self._ssl("req", *ec, "-keyout", str(key), "-out", str(root / "tsa.csr"), "-subj", f"/CN={name} TSA")
        (root / "ext.cnf").write_text("basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=critical,timeStamping\n", encoding="ascii")
        self._ssl("x509", "-req", "-in", str(root / "tsa.csr"), "-CA", str(self.cacert), "-CAkey", str(root / "ca.key"), "-CAcreateserial",
                  "-out", str(self.crt), "-days", "30", "-extfile", str(root / "ext.cnf"))
        (root / "tsaserial").write_text("1A\n", encoding="ascii")
        (root / "tsa.cnf").write_text(
            "[ tsa ]\ndefault_tsa = tsa1\n[ tsa1 ]\n"
            f"dir = {root}\nserial = $dir/tsaserial\ncrypto_device = builtin\nsigner_cert = $dir/tsa.crt\ncerts = $dir/cacert.pem\n"
            "signer_key = $dir/tsa.key\nsigner_digest = sha256\ndefault_policy = 1.2.3.4.1\nother_policies = 1.2.3.4.5.6\n"
            "digests = sha256, sha384, sha512\naccuracy = secs:1\nclock_precision_digits = 0\nordering = yes\ntsa_name = yes\n"
            "ess_cert_id_chain = no\ness_cert_id_alg = sha256\n",
            encoding="ascii",
        )
        self.urls: list[str] = []

    @staticmethod
    def _ssl(*args: str) -> None:
        subprocess.run(["openssl", *args], check=True, capture_output=True)

    def reply(self, url: str, body: bytes, *, timeout_s: float) -> bytes:
        self.urls.append(url)
        query, out = self.dir / "in.tsq", self.dir / "out.tsr"
        query.write_bytes(body)
        self._ssl("ts", "-reply", "-config", str(self.dir / "tsa.cnf"), "-queryfile", str(query), "-out", str(out))
        return out.read_bytes()


def _originals(run_id: str = "r1", *, run_json: bool = True) -> list[Path]:
    paths.ensure_run_layout(run_id)
    pickup = paths.quarantine_original(run_id, "pickup")
    pickup.write_bytes(bytes(range(256)) * 40)
    ret = paths.quarantine_original(run_id, "return")
    ret.write_bytes(b"return bytes " * 500)
    if run_json:
        c.write_json(paths.run_json(run_id), c.RunManifest(run_id=run_id, created_at=NOW, config_hash="ab" * 32, app_commit="abc1234"))
    return [pickup, ret]


def _event(seq: int, prev: str, kind: c.EventKind, payload: dict, run_id: str = "r1") -> c.Event:
    event = c.Event(seq=seq, ts="2026-10-27T15:00:01Z", run_id=run_id, kind=kind, payload=payload, prev=prev, hash="0" * 64)
    event.hash = c.chain_hash(prev, event.line_without_hash())
    return event


def _write_chain(run_id: str, events: list[c.Event]) -> None:
    paths.events_path(run_id).write_text("".join(e.to_json() + "\n" for e in events), encoding="utf-8")


def _make_clip(path: Path, *, creation_time: str | None) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=15", "-t", "0.4",
           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast"]
    if creation_time:
        cmd += ["-metadata", f"creation_time={creation_time}"]
    subprocess.run(cmd + [str(path)], check=True)


# Manifest and verify -----------------------------------------------------------------------------------------------------


def test_build_manifest_round_trip(runs: Path) -> None:
    originals = _originals()
    manifest = seal.build_manifest("r1", originals)
    path = paths.manifest_json("r1")
    assert path.read_bytes() == c.json_bytes(manifest)
    assert c.read_json(path, c.SealManifest) == manifest
    assert [e.path for e in manifest.files] == ["quarantine/pickup.orig", "quarantine/return.orig"]
    for entry, original in zip(manifest.files, originals):
        assert entry.sha256 == hashlib.sha256(original.read_bytes()).hexdigest()
        assert entry.bytes == original.stat().st_size
    assert manifest.config_hash == "ab" * 32 and manifest.app_commit == "abc1234"
    assert manifest.stamps == c.Stamps() and manifest.deletions == [] and manifest.pdf_sha256 is None
    assert seal.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_catches_a_flipped_byte(runs: Path) -> None:
    originals = _originals()
    seal.build_manifest("r1", originals)
    assert seal.verify_run("r1") == [] and seal.verify("r1") == []
    data = bytearray(originals[0].read_bytes())
    data[1000] ^= 0x01
    originals[0].write_bytes(bytes(data))
    assert seal.verify_run("r1") == ["hash differs: quarantine/pickup.orig"]
    originals[1].write_bytes(originals[1].read_bytes()[:-1])
    problems = seal.verify_run("r1")
    assert "size differs: quarantine/return.orig" in problems and "hash differs: quarantine/return.orig" in problems
    originals[1].unlink()
    assert "missing: quarantine/return.orig" in seal.verify_run("r1")
    paths.manifest_json("r1").unlink()
    assert seal.verify_run("r1") == ["bundle/manifest.json missing"]


def test_verify_accepts_a_provided_original_after_the_quarantine_is_gone(runs: Path, tmp_path: Path) -> None:
    originals = _originals()
    manifest = seal.build_manifest("r1", originals)
    copy = tmp_path / "IMG_0001.MOV"
    shutil.copy(originals[0], copy)
    deletions = [c.DeletionRecord(target=e.path, kind=c.DeletionKind.QUARANTINE, sha256=e.sha256, bytes=e.bytes, deleted_at=NOW, reason="run end") for e in manifest.files]
    seal.write_manifest_run("r1", deletions=deletions, events_head_hash="", config_hash="ab" * 32, app_commit="abc1234", pdf_sha256=None)
    for original in originals:
        original.unlink()
    assert seal.verify_run("r1") == []  # deleted at run end, as manifest_run.json says
    assert seal.verify_run("r1", originals=[copy]) == ["no provided file matches quarantine/return.orig (sha256 " + manifest.files[1].sha256[:12] + ")"]
    copy.write_bytes(copy.read_bytes() + b"x")
    assert "no provided file matches quarantine/pickup.orig" in seal.verify_run("r1", originals=[copy])[0]


# Times and clocks ------------------------------------------------------------------------------------------------------------


def test_minutes_after_camera() -> None:
    assert seal.minutes_after_camera(NOW, CAMERA) == 23
    assert seal.minutes_after_camera(NOW, "2026-10-27T14:37:00.000000Z") == 23
    assert seal.minutes_after_camera(NOW, "2026-10-27T16:37:00+02:00") == 23
    assert seal.minutes_after_camera(NOW, NOW) == 0
    assert seal.minutes_after_camera(NOW, "not present") == c.CAMERA_TIME_NOT_PRESENT
    assert seal.minutes_after_camera(NOW, c.CAMERA_TIME_NOT_PRESENT) == "camera time not present"
    assert seal.minutes_after_camera(NOW, "") == c.CAMERA_TIME_NOT_PRESENT


def test_normalize_time() -> None:
    assert seal.normalize_time("2026-10-27T14:37:00.000000Z") == CAMERA
    assert seal.normalize_time("2026-10-27T16:37:00+02:00") == CAMERA
    assert seal.normalize_time("2026-10-27 14:37:00") == CAMERA  # naive times count as UTC
    for absent in (None, "", "not present", c.CAMERA_TIME_NOT_PRESENT, "yesterday"):
        assert seal.normalize_time(absent) == seal.NOT_PRESENT


def test_camera_time_of_reads_ffprobe(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg not on PATH")
    stamped, bare = tmp_path / "stamped.mp4", tmp_path / "bare.mp4"
    _make_clip(stamped, creation_time=CAMERA)
    _make_clip(bare, creation_time=None)
    assert seal.camera_time_of(stamped) == CAMERA
    assert seal.camera_time_of(bare) == seal.NOT_PRESENT
    assert seal.camera_time_of(tmp_path / "missing.mp4") == seal.NOT_PRESENT


def test_seal_run_without_clients_records_failures_and_never_fails(runs: Path) -> None:
    originals = _originals()
    manifest, clocks = seal.seal_run("r1", originals, now=NOW, camera_time=CAMERA)
    assert clocks == c.ThreeClocks(sealed_at=NOW, first_model_call_at=None, sealed_minutes_after_camera=23, camera_time=CAMERA)
    for status in (manifest.stamps.manifest_ots, manifest.stamps.pickup_ots, manifest.stamps.return_ots):
        assert status.startswith("failed:FileNotFoundError: the OpenTimestamps client is not installed") and "opentimestamps-client" in status
    assert manifest.stamps.tsr_time is None and manifest.stamps.tsa_issuer is None  # freetsa and DigiCert both unreachable
    assert not list(paths.bundle_dir("r1").glob("*.ts[qr]"))  # a failed token leaves no file for verify.md to name
    assert paths.manifest_json("r1").read_bytes() == c.json_bytes(manifest.model_copy(update={"stamps": c.Stamps()}))
    _, clocks2 = seal.seal_run("r1", originals, now=NOW, camera_time=c.CAMERA_TIME_NOT_PRESENT)
    assert clocks2.sealed_minutes_after_camera == "camera time not present" and clocks2.camera_time is None
    unstamped, clocks3 = seal.seal_run("r1", originals, now=NOW, camera_time="2026-10-27T14:37:00.000000Z", stamp=False)
    assert clocks3.sealed_minutes_after_camera == 23 and clocks3.camera_time == CAMERA and unstamped.stamps == c.Stamps()


def test_ots_client_missing_names_the_install_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seal.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="opentimestamps-client"):
        REAL_OTS_BINARY()
    monkeypatch.setattr(seal, "_ots_binary", REAL_OTS_BINARY)
    with pytest.raises(FileNotFoundError, match="opentimestamps-client"):
        seal.stamp_ots(Path("x"))
    with pytest.raises(FileNotFoundError):
        seal.upgrade_ots(Path("x.ots"))
    assert seal.ots_status(Path("x.ots")).startswith("failed:the OpenTimestamps client is not installed")


def test_parse_ots_info_pending_confirmed_failed() -> None:
    assert seal.parse_ots_info(OTS_INFO_PENDING) == "pending"
    assert seal.parse_ots_info(OTS_INFO_CONFIRMED) == "confirmed:921003"  # the earliest block wins
    assert seal.parse_ots_info("File sha256 hash: 00\nTimestamp:\n").startswith("failed:")
    assert seal.parse_ots_info("") == "failed:no attestation in the proof"


def test_ots_commands_through_a_fake_client(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ots(tmp_path, monkeypatch)
    target = tmp_path / "manifest.json"
    target.write_bytes(b"{}\n")
    Path(f"{target}.ots").write_text("stale proof", encoding="ascii")  # left by an interrupted seal
    ots = seal.stamp_ots(target)
    assert ots == Path(f"{target}.ots") and "PendingAttestation" in ots.read_text(encoding="ascii")
    assert seal.ots_status(ots) == "pending"
    assert seal.ots_status(tmp_path / "missing.ots") == "failed:missing.ots not found"
    Path(f"{ots}.bak").write_text("an old backup", encoding="ascii")  # ots upgrade refuses to start while one exists
    assert seal.upgrade_ots(ots) == "confirmed:921003"
    assert not Path(f"{ots}.bak").exists() and "BitcoinBlockHeaderAttestation" in ots.read_text(encoding="ascii")
    failing = tmp_path / "x.fail"
    failing.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="calendars unreachable"):
        seal.stamp_ots(failing)


def test_seal_run_keeps_every_proof_in_the_bundle(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ots(tmp_path, monkeypatch)
    originals = _originals()
    manifest, _ = seal.seal_run("r1", originals, now=NOW, camera_time=CAMERA)
    assert manifest.stamps.manifest_ots == manifest.stamps.pickup_ots == manifest.stamps.return_ots == "pending"
    bundle = paths.bundle_dir("r1")
    assert {p.name for p in bundle.glob("*.ots")} == {"manifest.json.ots", "pickup.ots", "return.ots"}
    assert not list(paths.quarantine_dir("r1").glob("*.ots"))  # an original's proof outlives the quarantine

    def broken(path: Path) -> Path:
        raise RuntimeError("calendars unreachable")

    monkeypatch.setattr(seal, "stamp_ots", broken)
    manifest, _ = seal.seal_run("r1", originals, now=NOW, camera_time=CAMERA)
    assert manifest.stamps.manifest_ots.startswith("failed:RuntimeError: calendars unreachable")
    assert manifest.stamps.pickup_ots.startswith("failed:") and manifest.stamps.return_ots.startswith("failed:")


@needs_openssl
def test_stamp_tsa_and_verify_tsr_with_a_local_tsa(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tsa = LocalTsa(tmp_path / "tsa")
    monkeypatch.setattr(seal, "_post_tsq", tsa.reply)
    data = tmp_path / "manifest.json"
    data.write_bytes(b'{"files":[]}\n')
    tsq, tsr, tsr_time, serial = seal.stamp_tsa(data)
    assert (tsq, tsr) == (Path(f"{data}.tsq"), Path(f"{data}.tsr")) and tsq.is_file() and tsr.is_file()
    issued = (tsa.dir / "tsaserial").read_text(encoding="ascii").strip()  # openssl writes back the serial it used
    assert tsa.urls == [seal.FREETSA_URL] and serial.lower() == f"0x{issued}".lower()
    stamped = datetime.fromisoformat(tsr_time.replace("Z", "+00:00"))
    assert tsr_time.endswith("Z") and abs(stamped - datetime.now(timezone.utc)) < timedelta(minutes=5)
    assert seal.verify_tsr(data, tsr, cacert=tsa.cacert, tsa_crt=tsa.crt)
    other = LocalTsa(tmp_path / "other", name="Someone Else")
    assert not seal.verify_tsr(data, tsr, cacert=other.cacert, tsa_crt=other.crt)  # a different authority
    data.write_bytes(b'{"files":[1]}\n')
    assert not seal.verify_tsr(data, tsr, cacert=tsa.cacert, tsa_crt=tsa.crt)  # one changed byte
    assert not seal.verify_tsr(data, tmp_path / "missing.tsr", cacert=tsa.cacert, tsa_crt=tsa.crt)


@needs_openssl
def test_stamp_tsa_refuses_a_token_for_other_bytes_and_keeps_nothing(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tsa = LocalTsa(tmp_path / "tsa")
    decoy = tmp_path / "decoy.json"
    decoy.write_bytes(b"other bytes\n")
    subprocess.run(["openssl", "ts", "-query", "-data", str(decoy), "-sha256", "-cert", "-out", str(tmp_path / "decoy.tsq")], check=True, capture_output=True)
    wrong = tsa.reply(seal.FREETSA_URL, (tmp_path / "decoy.tsq").read_bytes(), timeout_s=1.0)
    monkeypatch.setattr(seal, "_post_tsq", lambda url, body, *, timeout_s: wrong)
    data = tmp_path / "manifest.json"
    data.write_bytes(b"{}\n")
    with pytest.raises(RuntimeError, match="imprint"):
        seal.stamp_tsa(data)
    assert not Path(f"{data}.tsq").exists() and not Path(f"{data}.tsr").exists()
    monkeypatch.setattr(seal, "_post_tsq", lambda url, body, *, timeout_s: b"<html>busy</html>")
    with pytest.raises(RuntimeError, match="openssl ts -reply"):
        seal.stamp_tsa(data)
    assert not Path(f"{data}.tsr").exists()


@needs_openssl
def test_seal_run_falls_back_to_digicert_and_names_it(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tsa = LocalTsa(tmp_path / "tsa")
    originals = _originals()
    monkeypatch.setattr(seal, "_post_tsq", tsa.reply)
    manifest, _ = seal.seal_run("r1", originals, now=NOW, camera_time=CAMERA)
    assert manifest.stamps.tsa_issuer == "freetsa" and manifest.stamps.tsr_time and manifest.stamps.tsa_serial
    assert paths.tsr_path("r1").is_file() and paths.tsq_path("r1").is_file()
    assert seal.verify_tsr(paths.manifest_json("r1"), paths.tsr_path("r1"), cacert=tsa.cacert, tsa_crt=tsa.crt)

    def freetsa_down(url: str, body: bytes, *, timeout_s: float) -> bytes:
        if url == seal.FREETSA_URL:
            raise httpx.ConnectError("freetsa down")
        return tsa.reply(url, body, timeout_s=timeout_s)

    monkeypatch.setattr(seal, "_post_tsq", freetsa_down)
    manifest, _ = seal.seal_run("r1", originals, now=NOW, camera_time=CAMERA)
    assert manifest.stamps.tsa_issuer == "DigiCert" and tsa.urls[-1] == seal.DIGICERT_TSA_URL


# manifest_run.json and the chain -------------------------------------------------------------------------------------------


def test_write_manifest_run_and_verify_the_chain(runs: Path, tmp_path: Path) -> None:
    run_id = "r1"
    originals = _originals(run_id)
    manifest = seal.build_manifest(run_id, originals)
    manifest_sha = seal.sha256_file(paths.manifest_json(run_id))
    events = [_event(0, manifest_sha, c.EventKind.RUN_CREATED, {"mode": "car"})]
    events.append(_event(1, events[-1].hash, c.EventKind.SEAL_MANIFEST, {"sha256": manifest_sha}))
    events.append(_event(2, events[-1].hash, c.EventKind.SEAL_OTS, {"status": "pending"}))
    events.append(_event(3, events[-1].hash, c.EventKind.SEAL_TSR, {"tsr_time": "2026-10-27T15:00:02Z", "serial": "0x1f", "issuer": "freetsa"}))
    events.append(_event(4, events[-1].hash, c.EventKind.SEAL_OTS, {"status": "confirmed:921000"}))
    _write_chain(run_id, events)
    marks = [c.Mark(mark_id="m1", role="pickup", panel_id=c.PanelId.HOOD, kf_id="k1", source=c.MarkSource.SPARK_BOX, type=c.MarkKind.CHIP, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S)]
    c.write_json(paths.marks_json(run_id), marks)
    c.write_json(paths.tracks_json(run_id), [])
    c.write_json(paths.sheet_json(run_id), c.SheetModel(run_id=run_id))
    paths.upload_chunk_path(run_id, "u1", 0).write_bytes(b"chunk")
    deletion = c.DeletionRecord(target="file_abc", kind=c.DeletionKind.FILES_API, deleted_at=NOW, reason="run end", api_response={"deleted": True})

    paths.ots_path(run_id, "manifest.json").write_text("pending proof", encoding="ascii")
    paths.ots_path(run_id, "pickup").write_text("pending proof", encoding="ascii")
    paths.tsr_path(run_id).write_bytes(b"token")
    manifest_run = seal.write_manifest_run(run_id, deletions=[deletion], events_head_hash=events[-1].hash, config_hash="ab" * 32, app_commit="abc1234", pdf_sha256=None)
    listed = [e.path for e in manifest_run.files]
    assert "bundle/manifest.json" in listed and "derived/marks.json" in listed and "sheet/sheet.json" in listed and "bundle/manifest.json.tsr" in listed
    for excluded in ("run.json", "events.ndjson", "bundle/manifest_run.json", "quarantine/pickup.orig", "uploads/u1/chunks/000000", "bundle/manifest.json.ots", "bundle/pickup.ots"):
        assert excluded not in listed, excluded
    assert manifest_run.deletions == [deletion] and manifest_run.events_head_hash == events[-1].hash
    assert manifest_run.stamps == c.Stamps(manifest_ots="confirmed:921000", tsr_time="2026-10-27T15:00:02Z", tsa_serial="0x1f", tsa_issuer="freetsa")
    assert c.read_json(paths.manifest_run_json(run_id), c.SealManifest) == manifest_run
    assert seal.verify_run(run_id) == []
    paths.ots_path(run_id, "manifest.json").write_text("upgraded proof with its Bitcoin block", encoding="ascii")
    assert seal.verify_run(run_id) == []  # the hourly ots upgrade rewrites a proof in place; that is not tampering

    # Events appended after the run manifest keep the head inside the chain: still fine.
    events.append(_event(5, events[-1].hash, c.EventKind.RUN_COMPLETED, {}))
    _write_chain(run_id, events)
    assert seal.verify_run(run_id) == []

    # A rebuild that matches passes; one derived file that differs is named; frozen files are not compared.
    rebuild = tmp_path / "rebuild"
    (rebuild / "derived").mkdir(parents=True)
    (rebuild / "sheet").mkdir()
    shutil.copy(paths.marks_json(run_id), rebuild / "derived" / "marks.json")
    shutil.copy(paths.sheet_json(run_id), rebuild / "sheet" / "sheet.json")
    (rebuild / "derived" / "tracks.json").write_bytes(b"[1]\n")
    assert seal.verify_run(run_id, rebuild_dir=rebuild) == []
    assert seal.status_text(seal.verify_run(run_id, rebuild_dir=rebuild)) == "rebuild identical"
    c.write_json(rebuild / "derived" / "marks.json", [])
    assert seal.verify_run(run_id, rebuild_dir=rebuild) == ["rebuild differs: derived/marks.json"]

    # One edited event breaks the chain at that seq.
    edited = events[2].model_copy(update={"payload": {"status": "confirmed:1"}})
    _write_chain(run_id, events[:2] + [edited] + events[3:])
    problems = seal.verify_run(run_id)
    assert problems[0] == "event chain broken at seq 2" and len(problems) == 2  # and the head is no longer reachable
    _write_chain(run_id, events)

    # A head hash that is not in the chain is a problem of the run manifest.
    bogus = manifest_run.model_copy(update={"events_head_hash": "cd" * 32})
    c.write_json(paths.manifest_run_json(run_id), bogus)
    assert seal.verify_run(run_id) == ["run manifest: events head hash is not in the event chain"]

    # A derived file changed after the run manifest was written is named too.
    c.write_json(paths.manifest_run_json(run_id), manifest_run)
    c.write_json(paths.marks_json(run_id), [])
    assert seal.verify_run(run_id) == ["run manifest: size differs: derived/marks.json", "run manifest: hash differs: derived/marks.json"]


def test_events_through_the_head_and_the_stamps_they_carry(runs: Path) -> None:
    run_id = "r1"
    seal.build_manifest(run_id, _originals(run_id))
    manifest_sha = seal.sha256_file(paths.manifest_json(run_id))
    events = [_event(0, manifest_sha, c.EventKind.RUN_CREATED, {})]
    events.append(_event(1, events[-1].hash, c.EventKind.SEAL_OTS, {"status": "pending", "pickup": "pending", "return": "failed:RuntimeError: calendars unreachable"}))
    events.append(_event(2, events[-1].hash, c.EventKind.SEAL_TSR, {"tsr_time": "2026-10-27T15:00:02Z", "serial": "0x1f", "issuer": "DigiCert"}))
    events.append(_event(3, events[-1].hash, c.EventKind.SHEET_WRITTEN, {}))
    events.append(_event(4, events[-1].hash, c.EventKind.SEAL_OTS, {"status": "confirmed:921003"}))  # the hourly upgrade
    _write_chain(run_id, events)
    upto = seal.events_through(run_id, events[2].hash)
    assert [e.seq for e in upto] == [0, 1, 2]
    assert seal.stamps_from_events(upto) == c.Stamps(manifest_ots="pending", pickup_ots="pending", return_ots="failed:RuntimeError: calendars unreachable", tsr_time="2026-10-27T15:00:02Z", tsa_serial="0x1f", tsa_issuer="DigiCert")
    assert seal.stamps_from_events(seal.events_through(run_id, None)).manifest_ots == "confirmed:921003"
    assert seal.stamps_from_events(seal.events_through(run_id, None)).pickup_ots == "pending"  # the upgrade event carries the manifest only
    assert len(seal.events_through(run_id, "ee" * 32)) == 5  # a head not in the log: the whole log
    assert seal.chain_problems_through(run_id, manifest_sha, head_hash=events[2].hash) == []
    with open(paths.events_path(run_id), "a", encoding="utf-8") as fh:
        fh.write('{"seq": 5, "half a line\n')  # a crash after the head cannot change what the sheet checked
    assert seal.chain_problems_through(run_id, manifest_sha, head_hash=events[2].hash) == []
    assert seal.chain_problems_through(run_id, manifest_sha, head_hash=None) == ["event chain broken at line 5: not an event"]
    assert seal.chain_problems_through(run_id, manifest_sha, head_hash="ee" * 32)[0].startswith("event chain broken at line 5")
    _write_chain(run_id, events)
    assert seal.chain_problems_through(run_id, manifest_sha, head_hash="ee" * 32) == ["events head hash is not in the event chain"]
    edited = events[1].model_copy(update={"payload": {"status": "confirmed:1"}})
    _write_chain(run_id, [events[0], edited, *events[2:]])
    assert seal.chain_problems_through(run_id, manifest_sha, head_hash=events[2].hash) == ["event chain broken at seq 1"]


def test_chain_prev_of_seq_zero_is_the_manifest_hash(runs: Path) -> None:
    originals = _originals()
    seal.build_manifest("r1", originals)
    _write_chain("r1", [_event(0, "ff" * 32, c.EventKind.RUN_CREATED, {})])
    assert seal.verify_run("r1") == ["event chain broken at seq 0"]
    paths.events_path("r1").write_text("not json\n", encoding="utf-8")
    assert seal.verify_run("r1") == ["event chain broken at line 0: not an event"]


# LIVE ---------------------------------------------------------------------------------------------------------------------


@needs_openssl
@pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="LIVE=1 only: one free freetsa stamp plus its two certificates")
def test_live_freetsa_stamp_verifies_against_freetsa_certificates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seal, "_post_tsq", REAL_POST_TSQ)
    data = tmp_path / "walkaround_seal_smoke.txt"
    data.write_text(f"walkaround seal lane smoke {datetime.now(timezone.utc).isoformat()}\n", encoding="ascii")
    tsq, tsr, tsr_time, serial = seal.stamp_tsa(data)
    certs = {}
    for name in ("cacert.pem", "tsa.crt"):
        response = httpx.get(f"https://freetsa.org/files/{name}", timeout=20.0)
        response.raise_for_status()
        certs[name] = tmp_path / name
        certs[name].write_bytes(response.content)
    assert seal.verify_tsr(data, tsr, cacert=certs["cacert.pem"], tsa_crt=certs["tsa.crt"])
    print(f"LIVE freetsa: time {tsr_time}, serial {serial}, token {tsr.stat().st_size} bytes")
