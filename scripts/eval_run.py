"""Eval harness (docs/ARCHITECTURE.md section 10; eval/README.md): numbers with N from runs and labels.

Per labeled pair: mark precision and recall per panel, verdict accuracy, the false "new" rate where nothing changed,
coverage-state accuracy, Stays Put agreement where mark_tracks exist, run-to-run stability, and cost and wall clock
per walk from the ledger, plus the calibration, the handle residual, the chance-match probability, c_present_share per
noun and, on a pseudo pair, the noise floor. Every printed number is also in eval/results.json, written as canonical
JSON with no clock time, so the same runs and labels give the same bytes; a refused labels file writes nothing.
--snapshot copies each run's inputs (JSON and panel masks, never a frame; no model text, spoken words, file name,
camera time or device) to eval/snapshots/ so the numbers rebuild from the repo alone (--from-snapshots). A missing
run or label prints the command that makes it. Pairs are reported one by one and never pooled (docs/DATA_HUNT.md 4.1).
No model is called.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for _entry in (str(REPO), str(SCRIPTS)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import pseudo_pair  # noqa: E402
from walkaround import contracts, geometry, ledger, paths  # noqa: E402
from walkaround.contracts import (  # noqa: E402
    WHEEL_PANELS,
    Calibration,
    Coverage,
    CoverageState,
    Keyframe,
    LaneRecord,
    Mark,
    MarkKind,
    MarkSource,
    MarkTrack,
    MatchState,
    PanelId,
    PanelMatch,
    PanelTrack,
    RunManifest,
    SizeClass,
    SparkRequestRecord,
    StaysPutVerdict,
    Verdict,
    VerdictLabel,
    VideoRecord,
)

LABELS_DIR = REPO / "eval" / "labels"
RESULTS = REPO / "eval" / "results.json"
SNAPSHOTS = REPO / "eval" / "snapshots"
CONTRACTS_PY = REPO / "walkaround" / "contracts.py"
LABEL_SCHEMA = "walkaround_labels_v1"
RESULT_SCHEMA = "walkaround_eval_v1"
SNAPSHOT_SCHEMA = "walkaround_snapshot_v1"
DEVELOPER_STATUS = "labeled by the developer from the frames, not yet reviewed"
ROLES = ("pickup", "return")
KINDS = ("cross_walk", "pseudo", "before_after")
SPLITS = ("fit", "report")
TABLES = ("A", "B", "C", "D", "E")
COVERAGE_LABELS = ("filmed", "glimpsed", "not_filmed")
NOT_REAL_REASONS = ("reflection", "dirt", "other")
PERCENT_FROM = 20  # under N = 20 a count prints without a percentage (docs/DATA_HUNT.md 4.1)
SLOT_SLACK_US = 1  # a reference mark's t and a keyframe's pts name the same proxy slot when they agree to 1 us
PANEL_FRAME_AREA = 12.0  # units^2 a whitened panel spans (a uniform rectangle of unit variance; ARCHITECTURE 5.10)
SECOND_RUN_PAIRS = ("uco3d_pairC",)  # one optional second run for run-to-run stability; D1 E2 already measured SAM repeats
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PANEL_ORDER = {p.value: i for i, p in enumerate(PanelId)}
MARK_TYPES = {k.value for k in MarkKind}
SIZE_CLASSES = {s.value for s in SizeClass}
TOP_KEYS = {"schema", "pair_id", "status", "labeler", "labeled_on", "kind", "table", "split", "nothing_changed", "footage", "run_id", "panels", "marks", "reference_marks", "notes"}
FOOTAGE_KEYS = {"path", "sha256", "t0", "t1", "source", "credit"}
PANEL_KEYS = {"changed", "coverage", "marks_checked", "note"}
MARK_KEYS = {"mark_id", "source", "role", "panel_id", "kf_id", "real", "new", "not_real_reason", "point_crop", "type", "size_class", "note"}
REF_KEYS = {"ref_id", "panel_id", "role", "t", "point_src", "type", "size_class", "real", "new", "not_real_reason", "group", "note"}
TIMING_EVENTS = {"run.created", "upload.done", "seal.manifest", "stage", "sam.lane_done", "spark.response", "sheet.written", "run.partial", "run.completed", "run.failed"}
SNAPSHOT_DERIVED = ("panels.json", "keyframes.json", "marks.json", "matches.json", "verdicts.json", "coverage.json", "calibration.json", "mark_tracks.json", "tracks.json")
REDACTED = ("run.json", "spark.json", "videos/pickup.json", "videos/return.json", "derived/marks.json")
NOT_COMPUTED = [
    {"metric": "S1 agreement table", "reason": "E16 (the Baseten second look) has not run"},
    {"metric": "inconclusive share on a panned pass and on a backtracking take", "reason": "no licensed walk was filmed panned or backwards"},
    {"metric": "re-shoot match across days", "reason": "no licensed pair is known to be filmed days apart"},
    {"metric": "Stays Put rejection rate on glare frames with b_pass", "reason": "Stays Put is not calibrated and does not cap verdicts yet (D1 lead note 2), and no licensed walk has labeled glare frames"},
    {"metric": "SAM run-to-run variance as one noun three times", "reason": "D1 E2.variance measured it (runs/d1/uco3d_pairC); this harness compares lane object ids between two runs instead"},
]
NEW, ALREADY, NYE = VerdictLabel.NEW_OR_UNSEEN, VerdictLabel.ALREADY_THERE, VerdictLabel.NEEDS_YOUR_EYE


def count(n: int, m: int) -> dict[str, Any]:
    """{"n", "m", "text"}: "41 of 53 (77%)" from N = 20 up, "7 of 9" below it."""
    text = f"{n} of {m}" + (f" ({round(100 * n / m)}%)" if m >= PERCENT_FROM else "")
    return {"n": n, "m": m, "text": text}


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Labels ------------------------------------------------------------------------------------------------------------


class LabelError(ValueError):
    """A labels file that does not follow eval/README.md."""


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _point(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 2 and all(_num(x) for x in v)


def label_problems(d: Any) -> list[str]:
    """Every way d breaks the walkaround_labels_v1 schema of eval/README.md; empty when it follows it."""
    if not isinstance(d, dict):
        return ["the file does not hold a JSON object"]
    out = [f"unknown key {k!r}" for k in sorted(set(d) - TOP_KEYS)]
    if d.get("schema") != LABEL_SCHEMA:
        out.append(f"schema is {d.get('schema')!r}, not {LABEL_SCHEMA!r}")
    if not isinstance(d.get("pair_id"), str) or not SAFE_ID.match(d["pair_id"]):
        out.append("pair_id must be letters, digits, _ and -")
    if not isinstance(d.get("status"), str) or not d["status"].strip():
        out.append("status must say who labeled it and whether anyone reviewed it")
    for key, allowed in (("kind", KINDS), ("table", TABLES), ("split", SPLITS)):
        if d.get(key) not in allowed:
            out.append(f"{key} must be one of {', '.join(allowed)}")
    if d.get("nothing_changed") not in (True, False, None):
        out.append("nothing_changed must be true, false or null")
    out += _footage_problems(d.get("footage"))
    if d.get("run_id") is not None and (not isinstance(d["run_id"], str) or not SAFE_ID.match(d["run_id"])):
        out.append("run_id must be a run id or null")
    out += _panel_problems(d.get("panels", {}))
    out += _mark_problems(d.get("marks", []), d.get("run_id"))
    out += _reference_problems(d.get("reference_marks", []))
    return out


def _unlicensed(text: Any) -> bool:
    """A path or source under data/eval_internal/, which no eval may use (the organizers' ruling, Oct 31)."""
    return isinstance(text, str) and "eval_internal" in text.replace("\\", "/").casefold().split("/")


def _footage_problems(footage: Any) -> list[str]:
    if not isinstance(footage, dict) or set(footage) != set(ROLES):
        return ["footage needs exactly a pickup and a return entry"]
    out = []
    for role in ROLES:
        f = footage[role]
        if not isinstance(f, dict) or not isinstance(f.get("path"), str) or set(f) - FOOTAGE_KEYS:
            out.append(f"footage.{role} needs a path and only the keys {sorted(FOOTAGE_KEYS)}")
            continue
        parts = f["path"].replace("\\", "/").split("/")
        if not f["path"].strip() or f["path"].startswith(("/", "~")) or ".." in parts:
            out.append(f"footage.{role}.path must be a repo-relative walk file")
        if _unlicensed(f["path"]) or _unlicensed(f.get("source")):
            out.append(f"footage.{role} points under data/eval_internal/, which no eval may use")
        if f.get("sha256") is not None and not re.fullmatch(r"[0-9a-f]{64}", str(f["sha256"])):
            out.append(f"footage.{role}.sha256 must be 64 hex characters or null")
    return out


def _panel_problems(panels: Any) -> list[str]:
    if not isinstance(panels, dict):
        return ["panels must be an object keyed by panel id"]
    out = []
    for pid, p in panels.items():
        if pid not in PANEL_ORDER:
            out.append(f"unknown panel id {pid!r}")
            continue
        if not isinstance(p, dict) or set(p) - PANEL_KEYS:
            out.append(f"panels.{pid} must be an object with only {sorted(PANEL_KEYS)}")
            continue
        if p.get("changed") not in (True, False, None):
            out.append(f"panels.{pid}.changed must be true, false or null")
        cov = p.get("coverage", {})
        if not isinstance(cov, dict) or any(r not in ROLES or v not in (*COVERAGE_LABELS, None) for r, v in cov.items()):
            out.append(f"panels.{pid}.coverage maps pickup and return to filmed, glimpsed, not_filmed or null")
        checked = p.get("marks_checked", {})
        if not isinstance(checked, dict) or any(r not in ROLES or not isinstance(v, bool) for r, v in checked.items()):
            out.append(f"panels.{pid}.marks_checked maps pickup and return to true or false")
    return out


def _mark_problems(marks: Any, run_id: Any) -> list[str]:
    if not isinstance(marks, list):
        return ["marks must be a list"]
    if marks and run_id is None:
        return ["marks are tied to a run, so run_id must be set"]
    out: list[str] = []
    seen: set[str] = set()
    for i, m in enumerate(marks):
        where = f"marks[{i}]"
        if not isinstance(m, dict) or set(m) - MARK_KEYS:
            out.append(f"{where} must be an object with only {sorted(MARK_KEYS)}")
            continue
        if not isinstance(m.get("mark_id"), str) or m["mark_id"] in seen:
            out.append(f"{where}.mark_id must be a string used once")
        seen.add(str(m.get("mark_id")))
        out += _common_mark_problems(m, where)
        if m.get("source") not in ("system", "hand"):
            out.append(f"{where}.source must be system or hand")
        if not isinstance(m.get("kf_id"), str):
            out.append(f"{where}.kf_id must name the keyframe whose crop was labeled")
        if m.get("source") == "system" and (m.get("size_class") not in (*SIZE_CLASSES, None) or m.get("type") not in (*MARK_TYPES, None)):
            out.append(f"{where}: a size class the labeler set is S, M or L, and a type is from the marks_v1 list")
        if m.get("source") == "hand":
            if not _point(m.get("point_crop")):
                out.append(f"{where}.point_crop must be [x, y] in crop pixels")
            if m.get("type") not in MARK_TYPES or m.get("size_class") not in SIZE_CLASSES:
                out.append(f"{where} needs a mark type and a size class")
            if m.get("real") is not True:
                out.append(f"{where}: a hand mark is one the labeler saw, so real is true")
    return out


def _common_mark_problems(m: dict[str, Any], where: str) -> list[str]:
    out = []
    if m.get("role") not in ROLES:
        out.append(f"{where}.role must be pickup or return")
    if m.get("panel_id") not in PANEL_ORDER:
        out.append(f"{where}.panel_id {m.get('panel_id')!r} is not a panel id")
    if m.get("real") not in (True, False, None) or m.get("new") not in (True, False, None):
        out.append(f"{where}.real and .new must be true, false or null")
    if m.get("not_real_reason") not in (*NOT_REAL_REASONS, None):
        out.append(f"{where}.not_real_reason must be one of {', '.join(NOT_REAL_REASONS)} or null")
    return out


def _reference_problems(refs: Any) -> list[str]:
    if not isinstance(refs, list):
        return ["reference_marks must be a list"]
    out: list[str] = []
    seen: set[str] = set()
    for i, r in enumerate(refs):
        where = f"reference_marks[{i}]"
        if not isinstance(r, dict) or set(r) - REF_KEYS:
            out.append(f"{where} must be an object with only {sorted(REF_KEYS)}")
            continue
        if not isinstance(r.get("ref_id"), str) or r["ref_id"] in seen:
            out.append(f"{where}.ref_id must be a string used once")
        seen.add(str(r.get("ref_id")))
        out += _common_mark_problems(r, where)
        if not _num(r.get("t")) or r["t"] < 0 or not _point(r.get("point_src")):
            out.append(f"{where} needs t (seconds into the walk file) and point_src [x, y] in its pixels")
        if r.get("type") not in MARK_TYPES or r.get("size_class") not in SIZE_CLASSES or not isinstance(r.get("real"), bool):
            out.append(f"{where} needs a mark type, a size class and real true or false")
    return out


def load_labels(path: Path) -> dict[str, Any]:
    """A labels file checked against the schema and named <pair_id>.json; LabelError names every problem."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabelError(f"{_rel(path)}: not JSON ({exc})") from exc
    problems = label_problems(data)
    if isinstance(data, dict) and isinstance(data.get("pair_id"), str) and data["pair_id"] != Path(path).stem:
        problems.append(f"pair_id {data['pair_id']!r} differs from the file name; save it as eval/labels/{data['pair_id']}.json")
    if problems:
        raise LabelError(f"{_rel(path)}: " + "; ".join(problems))
    return data


def has_mark_labels(lab: dict[str, Any]) -> bool:
    return bool(lab.get("marks")) or any(any(p.get("marks_checked", {}).values()) for p in lab.get("panels", {}).values())


# Runs --------------------------------------------------------------------------------------------------------------


@dataclass
class RunData:
    run_id: str
    root: Path
    snapshot: bool
    run: RunManifest
    th: dict[str, Any]
    videos: dict[str, VideoRecord]
    keyframes: list[Keyframe]
    marks: list[Mark]
    matches: list[PanelMatch]
    verdicts: list[Verdict]
    coverage: Coverage | None
    calibration: Calibration | None
    mark_tracks: list[MarkTrack]
    tracks: list[PanelTrack]
    spark: list[SparkRequestRecord]
    lanes: list[LaneRecord]
    events: list[dict[str, Any]]
    ledger_rows: list[dict[str, Any]]
    inputs: dict[str, str]
    redacted: dict[str, bytes]


def redact(rel: str, data: Any) -> Any:
    """What a snapshot must not carry, removed the same way from runs/ and from a snapshot: Spark's text (the eval
    reads only which crop a call was on), the renter's spoken words on narration_seed marks, the original file name,
    the camera time and the device."""
    if rel == "spark.json" and isinstance(data, list):
        return [{**r, "raw_text": "", "parsed": None} if isinstance(r, dict) else r for r in data]
    if rel == "derived/marks.json" and isinstance(data, list):
        seed = MarkSource.NARRATION_SEED.value
        return [{**m, "description": ""} if isinstance(m, dict) and m.get("source") == seed else m for m in data]
    if rel.startswith("videos/") and isinstance(data, dict):
        out = {**data, "original_filename": Path(rel).stem + Path(str(data.get("original_filename", ""))).suffix.lower()}
        if isinstance(data.get("ffprobe"), dict):
            out["ffprobe"] = {**data["ffprobe"], "creation_time": "not present", "device": None}
        return out
    if rel == "run.json" and isinstance(data, dict) and isinstance(data.get("clocks"), dict):
        return {**data, "clocks": {**data["clocks"], "camera_time": None, "sealed_minutes_after_camera": None}}
    return data


class _Reader:
    """Reads the files of one run and records each one's sha256 under '<run_id>/<rel>'; a REDACTED file is read as
    the canonical bytes of its redacted JSON, which are also what a snapshot stores."""

    def __init__(self, run_id: str, root: Path) -> None:
        self.run_id, self.root, self.inputs, self.redacted = run_id, root, {}, {}

    def raw(self, rel: str) -> bytes | None:
        path = self.root / rel
        if not path.is_file():
            return None
        data = path.read_bytes()
        if rel in REDACTED and data.strip():
            data = contracts.json_bytes(redact(rel, json.loads(data)))
            self.redacted[rel] = data
        self.inputs[f"{self.run_id}/{rel}"] = _sha(data)
        return data

    def model(self, rel: str, model: type[Any]) -> Any:
        data = self.raw(rel)
        return None if not data or not data.strip() else model.model_validate_json(data)

    def models(self, rel: str, model: type[Any]) -> list[Any]:
        data = self.raw(rel)
        raw = json.loads(data) if data and data.strip() else []
        return [model.model_validate(x) for x in raw] if isinstance(raw, list) else []

    def canonical_lines(self, rel: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rows as canonical JSON lines: hashed under rel and read back, so runs/ and a snapshot give the same values."""
        text = "".join(contracts.canonical_json(row) + "\n" for row in rows)
        self.inputs[f"{self.run_id}/{rel}"] = _sha(text.encode("ascii"))
        return [json.loads(line) for line in text.splitlines()]


def _event_lines(root: Path) -> list[dict[str, Any]]:
    for name, opener in (("events.ndjson", Path.read_bytes), ("events.ndjson.gz", lambda p: gzip.decompress(p.read_bytes()))):
        path = root / name
        if path.is_file():
            text = opener(path).decode("utf-8")
            events = [json.loads(line) for line in text.splitlines() if line.strip()]
            return [{"seq": e["seq"], "ts": e["ts"], "kind": e["kind"], "payload": e.get("payload", {})} for e in events if e.get("kind") in TIMING_EVENTS]
    return []


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def load_run(run_id: str, root: Path, snapshot: bool) -> RunData:
    """Everything the metrics read from runs/<id>/ or eval/snapshots/<id>/; a missing optional file reads as empty."""
    r = _Reader(run_id, root)
    run = r.model("run.json", RunManifest)
    videos = {role: v for role in ROLES if (v := r.model(f"videos/{role}.json", VideoRecord)) is not None}
    lanes = [r.model(p.relative_to(root).as_posix(), LaneRecord) for p in sorted((root / "sam").glob("*/*.summary.json"))]
    if snapshot:
        events, rows = _jsonl(root / "events_timing.ndjson"), _jsonl(root / "ledger.jsonl")
    else:
        events, rows = _event_lines(root), [row for row in ledger.rows() if row.get("run_id") == run_id]
    return RunData(
        run_id=run_id,
        root=root,
        snapshot=snapshot,
        run=run,
        th=run.thresholds or contracts.load_thresholds(),
        videos=videos,
        keyframes=r.models("derived/keyframes.json", Keyframe),
        marks=r.models("derived/marks.json", Mark),
        matches=r.models("derived/matches.json", PanelMatch),
        verdicts=r.models("derived/verdicts.json", Verdict),
        coverage=r.model("derived/coverage.json", Coverage),
        calibration=r.model("derived/calibration.json", Calibration),
        mark_tracks=r.models("derived/mark_tracks.json", MarkTrack),
        tracks=r.models("derived/tracks.json", PanelTrack),
        spark=r.models("spark.json", SparkRequestRecord),
        lanes=lanes,
        events=r.canonical_lines("events_timing.ndjson", events),
        ledger_rows=r.canonical_lines("ledger.jsonl", rows),
        inputs=r.inputs,
        redacted=r.redacted,
    )


def snapshot_note(root: Path) -> str:
    """For a snapshot that no longer reads: which contracts.py made it, against the one here now."""
    path = root / "snapshot.json"
    if not path.is_file():
        return ""
    made = str(json.loads(path.read_text(encoding="utf-8")).get("contracts_sha256", ""))
    now = _sha(CONTRACTS_PY.read_bytes()) if CONTRACTS_PY.is_file() else ""
    return "" if made == now else f" (snapshot made with contracts.py {made[:12]}, now {now[:12]})"


def list_runs(roots: list[tuple[Path, bool]]) -> dict[str, tuple[Path, bool]]:
    """run_id -> (directory, from a snapshot); a run in runs/ wins over its snapshot."""
    found: dict[str, tuple[Path, bool]] = {}
    for root, snapshot in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and not entry.name.startswith(("_", ".")) and (entry / "run.json").is_file():
                found.setdefault(entry.name, (entry, snapshot))
    return found



def pair_runs(lab: dict[str, Any], shas: dict[str, dict[str, str | None]]) -> list[str]:
    """Runs of this pair: both originals hash to the labels' footage, or the id is the labeled run or eval_<pair>[_n]."""
    pair = lab["pair_id"]
    want = {role: lab["footage"][role].get("sha256") for role in ROLES}
    names = {lab.get("run_id"), f"eval_{pair}"}
    out = []
    for run_id in sorted(shas):
        by_sha = all(want.values()) and all(shas[run_id].get(role) == want[role] for role in ROLES)
        if by_sha or run_id in names or run_id.startswith(f"eval_{pair}_"):
            out.append(run_id)
    return out


def choose_primary(lab: dict[str, Any], loaded: dict[str, RunData], override: str | None) -> str | None:
    """The --run override, else the labeled run, else eval_<pair>, else the newest run of the pair."""
    if override is not None:
        return override if override in loaded else None
    for candidate in (lab.get("run_id"), f"eval_{lab['pair_id']}"):
        if candidate in loaded:
            return candidate
    return max(loaded, key=lambda rid: (loaded[rid].run.created_at, rid)) if loaded else None


# Mark truth --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Truth:
    key: str
    role: str
    panel_id: str
    source: str  # system, hand or reference
    real: bool
    new: bool | None
    size_class: str | None
    w: tuple[float, float] | None
    not_real_reason: str | None = None


@dataclass(frozen=True)
class Judged:
    state: str  # real, false or unjudged
    new: bool | None
    size_class: str
    reason: str | None = None


UNJUDGED = Judged("unjudged", None, "")


def new_flag(label_new: bool | None, panel_label: dict[str, Any]) -> bool | None:
    """A mark's own new flag, else False on a panel labeled unchanged (the mark was there at pickup), else unknown."""
    if label_new is not None:
        return label_new
    return False if panel_label.get("changed") is False else None


def _mask(run: RunData, kf: Keyframe) -> np.ndarray | None:
    if not kf.mask_png or not (run.root / kf.mask_png).is_file():
        return None
    data = (run.root / kf.mask_png).read_bytes()
    run.inputs[f"{run.run_id}/{kf.mask_png}"] = _sha(data)
    return np.asarray(Image.open(io.BytesIO(data)).convert("L")) > 0


def whiten_point(run: RunData, kf: Keyframe, point_crop: list[float]) -> tuple[float, float] | None:
    """A crop-pixel point in the panel frame through geometry, as marks.whiten_marks places a Spark mark: wheel
    coordinates on a wheel panel (needs the mask), the moment frame elsewhere; None without moments or front_sign."""
    if kf.moments is None or kf.front_sign is None:
        return None
    if kf.panel_id in WHEEL_PANELS:
        mask = _mask(run, kf)
        return None if mask is None or not mask.any() else geometry.wheel_coords(list(point_crop), mask)
    return geometry.whiten((float(point_crop[0]), float(point_crop[1])), kf.moments, kf.front_sign)


def same_slot(a: float, b: float) -> bool:
    """Two stored times name the same proxy slot (both rounded to whole microseconds, as pts maps store them)."""
    return abs(round(a * 1_000_000) - round(b * 1_000_000)) <= SLOT_SLACK_US


def place_reference(ref: dict[str, Any], run: RunData | None) -> tuple[float, float] | None:
    """A reference mark lands on a keyframe of its panel and walk cut at the same proxy slot (kf.pts equals t), so
    the crop came from the source frame its point was read on; the point goes through the crop map and the
    whitening. None when no keyframe sits at that slot."""
    if run is None:
        return None
    for kf in run.keyframes:
        if kf.role == ref["role"] and kf.panel_id.value == ref["panel_id"] and kf.crop is not None and same_slot(kf.pts, float(ref["t"])):
            point = geometry.src_to_crop(ref["point_src"], kf.crop)
            if 0 <= point[0] < kf.crop.w and 0 <= point[1] < kf.crop.h:
                return whiten_point(run, kf, point)
    return None


def _labeled_truths(lab: dict[str, Any], labeled: RunData, radius: float, notes: Counter[str]) -> list[Truth]:
    """Judged system marks and hand marks of the labeled run. A hand mark within radius of a system mark called
    real is that mark found again (a blind first pass), so it counts once."""
    panels = lab.get("panels", {})
    by_id = {m.mark_id: m for m in labeled.marks}
    kfs = {kf.kf_id: kf for kf in labeled.keyframes}
    out: list[Truth] = []
    for lm in lab.get("marks", []):
        if lm.get("real") is None:
            notes["system marks left unjudged"] += 1
            continue
        if lm["source"] == "system":
            mark = by_id.get(lm["mark_id"])
            if mark is None:
                notes["labeled marks missing from the labeled run"] += 1
                continue
            w, size = (tuple(mark.w) if mark.w is not None else None), lm.get("size_class") or mark.size_class.value
        else:
            kf = kfs.get(lm["kf_id"])
            w, size = (whiten_point(labeled, kf, lm["point_crop"]) if kf is not None else None), lm["size_class"]
        if w is None:
            notes["marks with no panel-frame position"] += 1
        new = new_flag(lm.get("new"), panels.get(lm["panel_id"], {})) if lm["real"] else None
        out.append(Truth(lm["mark_id"], lm["role"], lm["panel_id"], lm["source"], bool(lm["real"]), new, size, w, lm.get("not_real_reason")))
    real_system = [t for t in out if t.source == "system" and t.real]
    kept = []
    for t in out:
        if t.source == "hand" and _nearest(t.role, t.panel_id, t.w, real_system, radius) is not None:
            notes["hand marks on a system mark called real"] += 1
            continue
        kept.append(t)
    return kept


def truths_for(lab: dict[str, Any], evaluated: RunData, labeled: RunData | None, radius: float) -> tuple[list[Truth], dict[str, int]]:
    """Labeled marks with panel-frame positions: judged system marks and hand marks of the labeled run, and the
    reference marks that land on a keyframe and repeat no labeled mark within radius (one per walk and group). notes
    counts what could not be placed and why."""
    panels = lab.get("panels", {})
    notes: Counter[str] = Counter()
    out = _labeled_truths(lab, labeled, radius, notes) if labeled is not None else []
    if labeled is None and lab.get("marks"):
        notes["mark labels made on a run that is not here"] += len(lab["marks"])
    labeled_truths = list(out)
    groups: set[tuple[str, str]] = set()
    for ref in lab.get("reference_marks", []):
        w = place_reference(ref, evaluated) or place_reference(ref, labeled)
        if w is None:
            notes["reference marks with no keyframe at their moment"] += 1
            continue
        if _nearest(ref["role"], ref["panel_id"], w, labeled_truths, radius) is not None:
            notes["reference marks already labeled on the run"] += 1
            continue
        group = (ref["role"], str(ref.get("group") or ref["ref_id"]))
        if group in groups:
            notes["reference marks repeating a placed group"] += 1
            continue
        groups.add(group)
        new = new_flag(ref.get("new"), panels.get(ref["panel_id"], {})) if ref["real"] else None
        out.append(Truth(ref["ref_id"], ref["role"], ref["panel_id"], "reference", bool(ref["real"]), new, ref["size_class"], w, ref.get("not_real_reason")))
    return out, dict(sorted(notes.items()))


def _dist(a: tuple[float, float], b: tuple[float, float], panel_id: str) -> float:
    if PanelId(panel_id) in WHEEL_PANELS:
        return geometry.wheel_distance(list(a), list(b))
    return math.dist(a, b)


def _nearest(role: str, panel_id: str, w: tuple[float, float] | None, truths: list[Truth], radius: float) -> Truth | None:
    if w is None:
        return None
    best: tuple[float, Truth] | None = None
    for t in truths:
        if t.role == role and t.panel_id == panel_id and t.w is not None:
            d = _dist(w, t.w, panel_id)
            if d <= radius and (best is None or d < best[0]):
                best = (d, t)
    return None if best is None else best[1]


def _checked(lab: dict[str, Any], pid: str, role: str) -> bool:
    return bool(lab.get("panels", {}).get(pid, {}).get("marks_checked", {}).get(role))


def judge_marks(evaluated: RunData, lab: dict[str, Any], truths: list[Truth], radii: dict[str, float]) -> dict[str, Judged]:
    """Each system mark of the evaluated run as real, false or unjudged: the labeler's own call on the labeled run,
    else the nearest labeled mark within the false-mark radius, else, on another run of the walks, false on a panel
    whose marks were all checked. On the labeled run a mark the labeler gave no call stays unjudged."""
    same = lab.get("run_id") == evaluated.run_id
    judged = {lm["mark_id"]: lm for lm in lab.get("marks", []) if lm["source"] == "system"} if same else {}
    panels = lab.get("panels", {})
    out: dict[str, Judged] = {}
    for m in evaluated.marks:
        pid, size = m.panel_id.value, m.size_class.value
        lm = judged.get(m.mark_id)
        if lm is not None and lm.get("real") is not None:
            size = lm.get("size_class") or size
            out[m.mark_id] = Judged("real", new_flag(lm.get("new"), panels.get(pid, {})), size) if lm["real"] else Judged("false", None, size, lm.get("not_real_reason"))
            continue
        near = _nearest(m.role, pid, tuple(m.w) if m.w is not None else None, truths, radii["false_mark"])
        if near is not None:
            out[m.mark_id] = Judged("real", near.new, size) if near.real else Judged("false", None, size, near.not_real_reason)
        elif m.w is not None and not same and _checked(lab, pid, m.role):
            out[m.mark_id] = Judged("false", None, size)
        else:
            out[m.mark_id] = Judged("unjudged", None, size)
    return out


def found_truths(evaluated: RunData, lab: dict[str, Any], truths: list[Truth], judged: dict[str, Judged], radius: float) -> dict[str, bool | None]:
    """Each real labeled mark: found when a system mark the labeler did not reject lies within the recall radius
    (on the labeled run a judged-real system mark finds itself); None when it cannot count (no position, or the run
    has no keyframe of that panel in that walk, so recall stays conditional on the panel being filmed)."""
    same = lab.get("run_id") == evaluated.run_id
    looked = {(kf.role, kf.panel_id.value) for kf in evaluated.keyframes}
    finders: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for m in evaluated.marks:
        if m.w is not None and judged.get(m.mark_id, UNJUDGED).state != "false":
            finders[(m.role, m.panel_id.value)].append(tuple(m.w))
    out: dict[str, bool | None] = {}
    for t in truths:
        if not t.real:
            continue
        if same and t.source == "system":
            out[t.key] = True
        elif t.w is None or (t.role, t.panel_id) not in looked:
            out[t.key] = None
        else:
            out[t.key] = any(_dist(t.w, w, t.panel_id) <= radius for w in finders[(t.role, t.panel_id)])
    return out


# Metrics -----------------------------------------------------------------------------------------------------------


def _cov_state(state: CoverageState) -> str:
    return "not_filmed" if state == CoverageState.NOT_SEEN else state.value


def _mirror(pid: str) -> str | None:
    for a, b in (("left_", "right_"), ("right_", "left_")):
        if pid.startswith(a):
            return b + pid[len(a):]
    return None


def coverage_metrics(run: RunData, lab: dict[str, Any]) -> dict[str, Any]:
    """Labeled coverage against coverage.json per panel and walk; not_seen reads as not_filmed, a lane that did not
    run is left out and counted. A side swap is a left and right pair whose states trade places."""
    if run.coverage is None:
        return {"status": "no coverage.json in this run"}
    system = {(role, pid.value): _cov_state(pc.state) for role, per in run.coverage.per_role.items() for pid, pc in per.items()}
    labeled = {(role, pid): state for pid, pl in lab.get("panels", {}).items() for role, state in (pl.get("coverage") or {}).items() if state is not None}
    right = total = not_run = 0
    confusion: Counter[str] = Counter()
    per_panel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    wrong = []
    for (role, pid), want in sorted(labeled.items(), key=lambda kv: (kv[0][0], PANEL_ORDER[kv[0][1]])):
        got = system.get((role, pid))
        if got is None or got == "not_run":
            not_run += 1
            continue
        total += 1
        right += got == want
        per_panel[pid][0] += got == want
        per_panel[pid][1] += 1
        confusion[f"{want} -> {got}"] += 1
        if got != want:
            wrong.append({"role": role, "panel_id": pid, "label": want, "system": got})
    swaps = set()
    for (role, pid), want in labeled.items():
        mirror = _mirror(pid)
        other = labeled.get((role, mirror)) if mirror else None
        if other is None or other == want:
            continue
        if system.get((role, pid)) == other and system.get((role, mirror)) == want:
            swaps.add((role, *sorted((pid, mirror))))
    panel_rows = {pid: count(*per_panel[pid]) for pid in sorted(per_panel, key=PANEL_ORDER.get)}
    return {"status": "ok", "right": count(right, total), "per_panel": panel_rows, "lane_not_run": not_run, "confusion": dict(sorted(confusion.items())), "wrong": wrong, "side_swaps": len(swaps)}


def labeler_overrides(lab: dict[str, Any]) -> dict[str, Any] | None:
    """The labeler-override rate (section 10 item 3): system marks called not real plus marks added by hand, over
    the system marks with a call plus the marks added. None without mark labels."""
    called = [lm for lm in lab.get("marks", []) if lm["source"] == "system" and lm.get("real") is not None]
    hand = [lm for lm in lab.get("marks", []) if lm["source"] == "hand"]
    if not called and not hand:
        return None
    return count(sum(lm["real"] is False for lm in called) + len(hand), len(called) + len(hand))


def mark_metrics(run: RunData, lab: dict[str, Any], truths: list[Truth], judged: dict[str, Judged], radii: dict[str, float], notes: dict[str, int]) -> dict[str, Any]:
    """Precision over the system marks with a call, recall over the real labeled marks that can count, both per
    panel and in all, recall per size class, and what was left unjudged or unplaced."""
    states = [j for j in judged.values() if j.state != "unjudged"]
    if not states and not truths:
        return {"status": "no mark labels for this run", "notes": notes}
    unjudged_checked = sum(1 for m in run.marks if judged.get(m.mark_id, UNJUDGED).state == "unjudged" and _checked(lab, m.panel_id.value, m.role))
    found = found_truths(run, lab, truths, judged, radii["recall"])
    per_panel: dict[str, dict[str, Any]] = {}
    panel_ids = sorted({m.panel_id.value for m in run.marks} | {t.panel_id for t in truths}, key=PANEL_ORDER.get)
    for pid in panel_ids:
        js = [judged[m.mark_id] for m in run.marks if m.panel_id.value == pid and judged.get(m.mark_id, UNJUDGED).state != "unjudged"]
        fs = [found[t.key] for t in truths if t.panel_id == pid and found.get(t.key) is not None]
        if js or fs:
            per_panel[pid] = {"precision": count(sum(j.state == "real" for j in js), len(js)), "recall": count(sum(fs), len(fs))}
    counted = [t for t in truths if t.real and found.get(t.key) is not None]
    by_size = {size: count(sum(bool(found[t.key]) for t in counted if t.size_class == size), sum(t.size_class == size for t in counted)) for size in sorted(SIZE_CLASSES)}
    n_real = sum(j.state == "real" for j in states)
    return {
        "status": "ok",
        "precision": count(n_real, len(states)),
        "false_mark_rate": count(len(states) - n_real, len(states)),
        "recall": count(sum(bool(found[t.key]) for t in counted), len(counted)),
        "recall_by_size": by_size,
        "unjudged_system_marks": sum(j.state == "unjudged" for j in judged.values()),
        "unjudged_on_checked_panels": unjudged_checked,
        "real_marks_not_counted": sum(1 for t in truths if t.real and found.get(t.key) is None),
        "labeler_overrides": labeler_overrides(lab) if lab.get("run_id") == run.run_id else None,
        "per_panel": per_panel,
        "notes": notes,
    }


def verdict_metrics(run: RunData, lab: dict[str, Any], judged: dict[str, Judged]) -> dict[str, Any]:
    """Panel calls against the changed labels, the false "new" rate where nothing changed, the needs-your-eye share,
    re-shoot match, and with mark labels the false-new, false-pre-existing and mark-level verdict rates."""
    roles = {m.mark_id: m.role for m in run.marks}
    by_panel = {pm.panel_id.value: pm for pm in run.matches}
    comparable = {pid for pid, pm in by_panel.items() if pm.state == MatchState.COMPARABLE}
    vs = [v for v in run.verdicts if v.panel_id.value in comparable and v.label != VerdictLabel.NOT_COMPARABLE]
    panels = lab.get("panels", {})

    def call(pid: str) -> str:
        here = {v.label for v in vs if v.panel_id.value == pid}
        return "changed" if NEW in here else ("needs_your_eye" if NYE in here else "unchanged")

    rows, gates = [], Counter()
    right = decided = nye_panels = not_comparable = not_compared = changed_total = changed_found = 0
    for pid in sorted(panels, key=PANEL_ORDER.get):
        changed = panels[pid].get("changed")
        if changed is None:
            continue
        truth = "changed" if changed else "unchanged"
        pm = by_panel.get(pid)
        if pm is None:
            not_compared += 1
            rows.append({"panel_id": pid, "label": truth, "call": "not compared (no keyframe in either walk)"})
            continue
        if pm.state != MatchState.COMPARABLE:
            not_comparable += 1
            gates.update(r for r in pm.reasons if r.startswith("G"))
            rows.append({"panel_id": pid, "label": truth, "call": "not comparable", "gates": [r for r in pm.reasons if r.startswith("G")]})
            continue
        got = call(pid)
        if changed:
            changed_total += 1
            changed_found += got == "changed"
        if got == "needs_your_eye":
            nye_panels += 1
        else:
            decided += 1
            right += got == truth
        rows.append({"panel_id": pid, "label": truth, "call": got})
    out: dict[str, Any] = {
        "panel_calls_right": count(right, decided),
        "changed_panels_called_changed": count(changed_found, changed_total),
        "panels_needs_your_eye": nye_panels,
        "panels_not_comparable": not_comparable,
        "not_comparable_gates": dict(sorted(gates.items())),
        "panels_not_compared": not_compared,
        "panel_rows": rows,
        "needs_your_eye_share": count(sum(v.label == NYE for v in vs), len(vs)),
        "label_counts": dict(sorted(Counter(v.label.value for v in run.verdicts).items())),
    }

    def mark_of(v: Verdict, role: str) -> str | None:
        return next((i for i in v.mark_ids if roles.get(i) == role), None)

    return_vs = [v for v in vs if mark_of(v, "return") is not None]
    if lab.get("nothing_changed") is True:
        out["false_new_nothing_changed"] = count(sum(v.label == NEW for v in return_vs), len(return_vs))
        out["panels_with_a_new_call"] = count(len({v.panel_id.value for v in return_vs if v.label == NEW}), len(comparable))
    already_pickup = {i for v in vs if v.label == ALREADY for i in v.mark_ids if roles.get(i) == "pickup"}
    unchanged = {pid for pid in comparable if panels.get(pid, {}).get("changed") is False}
    pickup_marks = [m for m in run.marks if m.role == "pickup" and m.panel_id.value in unchanged]
    out["re_shoot_match_unchanged_panels"] = count(sum(m.mark_id in already_pickup for m in pickup_marks), len(pickup_marks))

    def ret_state(v: Verdict) -> Judged:
        return judged.get(mark_of(v, "return") or "", UNJUDGED)

    pre = [v for v in return_vs if ret_state(v).state == "real" and ret_state(v).new is False]
    new = [v for v in return_vs if ret_state(v).state == "real" and ret_state(v).new is True]
    if pre or new:
        out["false_new"] = count(sum(v.label == NEW for v in pre), len(pre))
        out["false_pre_existing"] = count(sum(v.label == ALREADY for v in new), len(new))
        decided_vs = [v for v in pre + new if v.label in (ALREADY, NEW)]
        out["mark_verdicts_right"] = count(sum((v.label == NEW) == bool(ret_state(v).new) for v in decided_vs), len(decided_vs))
        nye_vs = [v for v in return_vs if v.label == NYE]  # a pickup-only one can never resolve as already there
        out["needs_your_eye_resolved_already_there"] = count(sum(ret_state(v).state == "real" and ret_state(v).new is False for v in nye_vs), len(nye_vs))
        out["needs_your_eye_pickup_only"] = sum(v.label == NYE and mark_of(v, "return") is None for v in vs)
    real_pickup = [m for m in run.marks if m.role == "pickup" and m.panel_id.value in comparable and judged.get(m.mark_id, UNJUDGED).state == "real"]
    if real_pickup:
        out["re_shoot_match_real_marks"] = count(sum(m.mark_id in already_pickup for m in real_pickup), len(real_pickup))
    return out


def staysput_metrics(run: RunData, judged: dict[str, Judged]) -> dict[str, Any]:
    """Stays Put verdicts against the labels: a real mark should stay fixed, a mark labeled a reflection should slide,
    dirt on the paint stays fixed; inconclusive and no_track tracks are counted apart, controls apart again."""
    if not run.mark_tracks:
        return {"status": "not run", "reason": "mark_tracks.json is missing or empty in this run"}
    agree = total = 0
    controls: list[bool] = []
    tracks = [mt for mt in run.mark_tracks if not mt.is_control and mt.verdict != StaysPutVerdict.NOT_RUN]  # a mark with no clip is no track
    for mt in run.mark_tracks:
        if mt.is_control:
            controls.append(mt.verdict == StaysPutVerdict.SLIDING)
            continue
        j = judged.get(mt.mark_id, UNJUDGED)
        if mt.verdict not in (StaysPutVerdict.FIXED, StaysPutVerdict.SLIDING) or j.state == "unjudged":
            continue
        expected = {"real": StaysPutVerdict.FIXED}.get(j.state) or {"reflection": StaysPutVerdict.SLIDING, "dirt": StaysPutVerdict.FIXED}.get(j.reason or "")
        if expected is None:
            continue
        total += 1
        agree += mt.verdict == expected
    inconclusive = sum(mt.verdict in (StaysPutVerdict.INCONCLUSIVE, StaysPutVerdict.NO_TRACK) for mt in tracks)
    return {
        "status": "ok",
        "agreement": count(agree, total),
        "inconclusive_or_no_track": count(inconclusive, len(tracks)),
        "controls_sliding": count(sum(controls), len(controls)),
        "verdicts": dict(sorted(Counter(mt.verdict.value for mt in run.mark_tracks).items())),
    }


def stability_metrics(base: RunData, other: RunData, radius: float) -> dict[str, Any]:
    """Two runs of the same walks: SAM object ids per lane, and, when both froze the same thresholds (a threshold
    change is not model variance), coverage states, best keyframe moments, marks found again within the recall radius
    and verdict label counts per panel."""
    ids_a = {(lr.role, lr.lane_id): len(lr.ids) for lr in base.lanes}
    ids_b = {(lr.role, lr.lane_id): len(lr.ids) for lr in other.lanes}
    common = sorted(set(ids_a) & set(ids_b))
    out: dict[str, Any] = {
        "other_run": other.run_id,
        "config_hashes": [(base.run.config_hash or "")[:12], (other.run.config_hash or "")[:12]],
        "same_config": base.run.config_hash == other.run.config_hash,
        "lane_object_ids_same": count(sum(ids_a[k] == ids_b[k] for k in common), len(common)),
    }
    if out["same_config"]:
        out.update(_derived_stability(base, other, radius))
    return out


def _derived_stability(base: RunData, other: RunData, radius: float) -> dict[str, Any]:
    cov_n = cov_m = kf_n = kf_m = 0
    if base.coverage and other.coverage:
        kfs_a, kfs_b = {k.kf_id: k for k in base.keyframes}, {k.kf_id: k for k in other.keyframes}
        frame_s = 1.0 / float(base.run.fps or 15.0)
        for role in ROLES:
            per_a, per_b = base.coverage.per_role.get(role, {}), other.coverage.per_role.get(role, {})
            for pid in PanelId:
                a, b = per_a.get(pid), per_b.get(pid)
                if a is None or b is None or CoverageState.NOT_RUN in (a.state, b.state):
                    continue
                cov_m += 1
                cov_n += _cov_state(a.state) == _cov_state(b.state)
                ka, kb = kfs_a.get(a.best_kf or ""), kfs_b.get(b.best_kf or "")
                if ka is not None and kb is not None:
                    kf_m += 1
                    kf_n += abs(ka.pts - kb.pts) <= frame_s + 1e-9
    found = 0
    placed = [m for m in base.marks if m.w is not None]
    for m in placed:
        ws = [tuple(o.w) for o in other.marks if o.w is not None and o.role == m.role and o.panel_id == m.panel_id]
        found += any(_dist(tuple(m.w), w, m.panel_id.value) <= radius for w in ws)

    def label_counts(run: RunData) -> dict[str, dict[str, int]]:
        out: dict[str, Counter[str]] = defaultdict(Counter)
        for v in run.verdicts:
            out[v.panel_id.value][v.label.value] += 1
        return {k: dict(c) for k, c in out.items()}

    la, lb = label_counts(base), label_counts(other)
    shared = sorted(set(la) | set(lb), key=PANEL_ORDER.get)
    return {
        "coverage_same": count(cov_n, cov_m),
        "best_keyframe_same_moment": count(kf_n, kf_m),
        "marks_found_again": count(found, len(placed)),
        "marks": [len(base.marks), len(other.marks)],
        "panel_verdicts_same": count(sum(la.get(p) == lb.get(p) for p in shared), len(shared)),
    }


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def wall_clock(events: list[dict[str, Any]]) -> dict[str, Any]:
    """From the run's own event clock: first event to the end, last upload to the end, and seconds per stage."""
    if not events:
        return {}
    end = next((e for e in reversed(events) if e["kind"] in ("run.completed", "run.failed")), events[-1])
    uploads = [e for e in events if e["kind"] == "upload.done"]
    stages: dict[str, float] = {}
    marks = [e for e in events if e["kind"] == "stage"] + [end]
    for a, b in zip(marks, marks[1:]):
        name = a["payload"].get("name", "?")
        stages[name] = stages.get(name, 0.0) + (_ts(b["ts"]) - _ts(a["ts"])).total_seconds()
    return {
        "first_event_to_end_s": round((_ts(end["ts"]) - _ts(events[0]["ts"])).total_seconds(), 3),
        "upload_to_end_s": round((_ts(end["ts"]) - _ts(uploads[-1]["ts"])).total_seconds(), 3) if uploads else None,
        "stage_s": [{"name": k, "s": round(v, 3)} for k, v in stages.items()],
        "ended_with": end["kind"],
    }


def _spark_role(rec: SparkRequestRecord) -> str | None:
    roles = {ref.path.split("/")[1] for ref in rec.input_refs if ref.path.count("/") >= 2 and ref.path.split("/")[1] in ROLES}
    return roles.pop() if len(roles) == 1 else None


def cost_metrics(run: RunData) -> dict[str, Any]:
    """Dollars per walk from the run's ledger rows (a SAM row carries its walk; a Spark row's walk comes from its
    spark.response event or its crop in spark.json; a Voice row stays unattributed, because voice.json holds the
    transcript and is never snapshotted), dollars per second of video, the wall clock, p50 first delta per lane, parse
    CPU per lane and Spark tokens per call."""
    spark_role = {rec.request_id: _spark_role(rec) for rec in run.spark}
    spark_role.update({e["payload"]["request_id"]: e["payload"].get("role") for e in run.events if e["kind"] == "spark.response" and e["payload"].get("request_id")})
    by_role: dict[str, float] = defaultdict(float)
    by_kind: dict[str, float] = defaultdict(float)
    calls: Counter[str] = Counter()
    first_delta: dict[str, list[float]] = defaultdict(list)
    parse_ms: dict[str, int] = defaultdict(int)
    tokens: dict[str, list[int]] = defaultdict(list)
    for row in run.ledger_rows:
        usd, kind = float(row.get("usd") or 0.0), str(row.get("kind"))
        role = row.get("role") if row.get("role") in ROLES else spark_role.get(row.get("request_id")) if kind == "spark" else None
        by_role[role or "unattributed"] += usd
        by_kind[kind] += usd
        calls[kind] += 1
        if kind == "sam_video":
            lane = str(row.get("note") or "?")
            delta = (row.get("latency_ms") or {}).get("first_delta")
            if delta is not None:
                first_delta[lane].append(float(delta))
            parse_ms[lane] += int(row.get("parse_cpu_ms") or 0)
        if kind == "spark":
            for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
                tokens[key].append(int(row.get(key) or 0))
    seconds = {role: v.ffprobe.duration_s for role, v in run.videos.items() if v.ffprobe is not None}
    walks = {}
    for role in ROLES:
        usd, video_s = by_role.get(role, 0.0), seconds.get(role)
        walks[role] = {"usd": round(usd, 6), "video_s": video_s, "usd_per_video_s": round(usd / video_s, 6) if video_s else None}
    return {
        "rows": len(run.ledger_rows),
        "usd_total": round(sum(by_role.values()), 6),
        "walks": walks,
        "usd_unattributed": round(by_role.get("unattributed", 0.0), 6),
        "usd_by_kind": {k: round(v, 6) for k, v in sorted(by_kind.items())},
        "calls_by_kind": dict(sorted(calls.items())),
        "wall_clock": wall_clock(run.events),
        "first_delta_p50_s": {lane: round(statistics.median(v) / 1000.0, 3) for lane, v in sorted(first_delta.items())},
        "parse_cpu_ms_by_lane": dict(sorted(parse_ms.items())),
        "spark_tokens_p50": {k: statistics.median(v) for k, v in sorted(tokens.items())},
    }


def c_present_metrics(run: RunData) -> dict[str, Any]:
    """c_present_share per noun over the run's tracks (median, with the track count)."""
    shares: dict[str, list[float]] = defaultdict(list)
    for t in run.tracks:
        shares[t.noun.value].append(t.c_present_share)
    return {noun: {"median": round(statistics.median(v), 6), "tracks": len(v)} for noun, v in sorted(shares.items())}


def billable_metrics(run: RunData, buckets_mm: list[float]) -> dict[str, Any] | None:
    """new_or_unseen verdicts (the ones that could be billed) whose ruler size reaches each bucket, and the same for
    needs_your_eye apart; already_there and not_seen_at_return are never billable. None when none carries a size."""

    def buckets(label: VerdictLabel) -> dict[str, Any]:
        sizes = [float(v.size.inches) * 25.4 for v in run.verdicts if v.label == label and v.size is not None and v.size.inches is not None]
        return {f"{b:g} mm": count(sum(s >= b for s in sizes), len(sizes)) for b in buckets_mm}

    out = {"new_or_unseen": buckets(NEW), "needs_your_eye": buckets(NYE)}
    return out if any(row["m"] for per in out.values() for row in per.values()) else None


def handle_residuals(run: RunData) -> dict[str, Any]:
    """Length of the whitened door-handle residual (matches.json handle_residual) over comparable panels."""
    lengths = sorted(math.hypot(*pm.handle_residual) for pm in run.matches if pm.state == MatchState.COMPARABLE and pm.handle_residual is not None)
    if not lengths:
        return {"n": 0}
    return {"n": len(lengths), "median": round(statistics.median(lengths), 6), "max": round(lengths[-1], 6)}


def chance_match(run: RunData) -> dict[str, Any] | None:
    """ARCHITECTURE 5.10: the chance a false return mark lands within tau_pre of a pickup mark, pi x tau_pre^2 / 12
    per pickup mark, times the pickup marks per comparable panel. None without calibration.json."""
    if run.calibration is None:
        return None
    comparable = {pm.panel_id for pm in run.matches if pm.state == MatchState.COMPARABLE}
    per_mark = math.pi * run.calibration.tau_pre**2 / PANEL_FRAME_AREA
    pickup = sum(1 for m in run.marks if m.role == "pickup" and m.panel_id in comparable)
    density = pickup / len(comparable) if comparable else 0.0
    return {
        "tau_pre": run.calibration.tau_pre,
        "per_pickup_mark": round(per_mark, 6),
        "pickup_marks": pickup,
        "comparable_panels": len(comparable),
        "per_false_mark": round(min(1.0, per_mark * density), 6),
    }


# Commands for what is missing ----------------------------------------------------------------------------------------


WALK_FOLDERS = ("data/candidates", "data/raw", "data/derived")  # never data/eval_internal/ (the organizers' ruling)


def resolve_walk(path: str) -> str:
    """The walk's path as the labels give it, or the same file name where docs/DATA_HUNT.md 3.1 moved it."""
    if (REPO / path).is_file():
        return path
    name = Path(path).name
    for folder in WALK_FOLDERS:
        hits = sorted((REPO / folder).rglob(name)) if (REPO / folder).is_dir() else []
        if hits:
            return _rel(hits[0])
    return path


def run_commands(lab: dict[str, Any], *, run_id: str | None = None) -> list[str]:
    pair = lab["pair_id"]
    walks = [resolve_walk(lab["footage"][role]["path"]) for role in ROLES]
    cmds = []
    if pair in pseudo_pair.PAIRS and any(not (REPO / w).is_file() for w in walks):
        cmds.append(f".venv/bin/python scripts/pseudo_pair.py cut {pair}")
    cmds.append(f"LIVE=1 .venv/bin/walkaround run {walks[0]} {walks[1]} --run-id {run_id or f'eval_{pair}'}")
    return cmds


def label_commands(run_id: str, pair: str) -> list[str]:
    """Serve with a 30-day retention, so the sweeper keeps the run's crops past a day (runs are made with keep
    false); open the page; move the newest download, since a second one is saved as '<pair> (1).json'."""
    return [
        "RUN_RETENTION_HOURS=720 .venv/bin/walkaround serve",
        f"open 'http://127.0.0.1:8000/eval/label.html?run={run_id}&pair={pair}'",
        f'mv "$(ls -t ~/Downloads/{pair}*.json | head -1)" eval/labels/{pair}.json',
    ]


def _missing(what: str, commands: list[str], note: str = "") -> dict[str, Any]:
    return {"what": what, "commands": commands, "note": note}


# One pair ------------------------------------------------------------------------------------------------------------


def evaluate_pair(lab: dict[str, Any], label_path: Path, runs: dict[str, tuple[Path, bool]], shas: dict[str, dict[str, str | None]], cache: dict[str, RunData], th: dict[str, Any], override: str | None) -> dict[str, Any]:
    pair = lab["pair_id"]
    radii = {"recall": float(th["eval"]["recall_radius_units"]), "false_mark": float(th["eval"]["false_mark_radius_units"])}
    entry: dict[str, Any] = {
        "labels": {"path": _rel(label_path), "status": lab["status"], "kind": lab["kind"], "table": lab["table"], "split": lab["split"], "nothing_changed": lab["nothing_changed"]},
        "missing": [],
    }

    def get(run_id: str) -> RunData:
        if run_id not in cache:
            cache[run_id] = load_run(run_id, *runs[run_id])
        return cache[run_id]

    loaded: dict[str, RunData] = {}
    for rid in sorted(set(pair_runs(lab, shas)) | ({override} if override in runs else set())):
        try:
            loaded[rid] = get(rid)
        except (ValueError, OSError) as exc:  # a pydantic ValidationError is a ValueError
            extra = snapshot_note(runs[rid][0]) if runs[rid][1] else ""
            entry["missing"].append(_missing("readable run", [], f"{rid} does not read as this repo's contracts: {str(exc).splitlines()[0]}{extra}"))
    primary = choose_primary(lab, loaded, override)
    entry["runs"], entry["primary_run"] = sorted(loaded), primary
    if primary is None:
        note = f"--run named {override}, which is not a run of this pair" if override else "no run of this pair in runs/ or eval/snapshots/"
        entry["missing"].append(_missing("run", run_commands(lab), note))
        return entry
    run = loaded[primary]
    entry["run_status"] = run.run.status.value
    labeled = None
    if lab.get("run_id"):
        labeled = loaded.get(lab["run_id"])  # pair_runs always lists the labeled run when it is on disk
        if lab["run_id"] not in runs:
            entry["missing"].append(_missing("labeled run", [], f"{lab['run_id']} holds the mark labels' crops; restore runs/{lab['run_id']} or its snapshot"))
    truths, notes = truths_for(lab, run, labeled, radii["recall"])
    judged = judge_marks(run, lab, truths, radii)
    entry["coverage"] = coverage_metrics(run, lab)
    entry["marks"] = mark_metrics(run, lab, truths, judged, radii, notes)
    if not has_mark_labels(lab):
        note = "precision and recall need real or not calls on this run's marks; label it on the day of the run, since a snapshot keeps no crops"
        entry["missing"].append(_missing("mark labels", label_commands(primary, pair), note))
    entry["verdicts"] = verdict_metrics(run, lab, judged)
    entry["staysput"] = staysput_metrics(run, judged)
    entry["stability"] = [stability_metrics(run, loaded[rid], radii["recall"]) for rid in sorted(loaded) if rid != primary]
    if len(loaded) < 2 and pair in SECOND_RUN_PAIRS:
        note = "optional, about $4.33 (D1 E13); CLI runs skip the $20 daily card ceiling, so check the day's spend first"
        entry["missing"].append(_missing("second run (run-to-run stability)", run_commands(lab, run_id=f"eval_{pair}_2"), note))
    entry["cost"] = {rid: cost_metrics(loaded[rid]) for rid in sorted(loaded)}
    entry["calibration"] = run.calibration.model_dump(mode="json") if run.calibration else None
    entry["handle_residual_units"] = handle_residuals(run)
    entry["chance_match"] = chance_match(run)
    entry["c_present_share"] = c_present_metrics(run)
    entry["billable"] = billable_metrics(run, [float(b) for b in th["eval"]["billable_buckets_mm"]])
    if lab["kind"] == "pseudo":
        floor = pseudo_pair.noise_floor(run.matches, run.calibration, run.th)
        kind = pseudo_pair.PAIRS[pair].kind if pair in pseudo_pair.PAIRS else None
        entry["pseudo"] = {**floor, "line": pseudo_pair.floor_line(floor, kind)}
    return entry


def method_lines(pairs: dict[str, Any]) -> list[str]:
    """The method page's eval lines (ARCHITECTURE 7 item 6), each with N and its pair, never pooled: false new,
    re-shoot match, the needs-your-eye share and how many the labeler resolved, and the noise floor. A count of 0 of 0
    gives no line."""
    out = []
    for pair, entry in pairs.items():
        if not entry.get("primary_run"):
            continue
        v, lab = entry.get("verdicts", {}), entry.get("labels", {})
        tail = "; labels not yet reviewed" if "not yet reviewed" in str(lab.get("status", "")) else ""
        if "pseudo" in entry:
            if entry["pseudo"]["pickup_marks_matched"][1]:
                out.append(f"{pair}: {entry['pseudo']['line']}{tail}")
            continue
        if v.get("false_new_nothing_changed", {}).get("m"):
            out.append(f"{pair}: false new where nothing changed, {v['false_new_nothing_changed']['text']} return-mark verdicts{tail}")
        if v.get("false_new", {}).get("m"):
            out.append(f"{pair}: false new on marks labeled pre-existing, {v['false_new']['text']}{tail}")
        shoot = v.get("re_shoot_match_real_marks") or v.get("re_shoot_match_unchanged_panels")
        if shoot and shoot["m"]:
            out.append(f"{pair}: re-shoot match, {shoot['text']} pickup marks found again as already there{tail}")
        if v.get("needs_your_eye_share", {}).get("m"):
            resolved = v.get("needs_your_eye_resolved_already_there")
            more = f"; the labeler resolved {resolved['text']} as already there" if resolved and resolved["m"] else ""
            out.append(f"{pair}: needs your eye, {v['needs_your_eye_share']['text']} verdicts{more}{tail}")
    return out


def evaluate(label_paths: list[Path], *, from_snapshots: bool = False, overrides: dict[str, str] | None = None, th: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    """The whole results document and the list of labels files that were refused (schema or file name)."""
    th = th or contracts.load_thresholds()
    roots = ([] if from_snapshots else [(paths.runs_root(), False)]) + [(SNAPSHOTS, True)]
    runs = list_runs(roots)
    shas = {rid: pseudo_pair.run_shas(root) for rid, (root, _) in runs.items()}
    errors: list[str] = []
    refused: set[str] = set()
    labels: dict[str, tuple[Path, dict[str, Any]]] = {}
    inputs: dict[str, str] = {}
    for path in sorted(label_paths):
        try:
            lab = load_labels(path)
        except LabelError as exc:
            errors.append(str(exc))
            refused.add(Path(path).stem)
            continue
        labels[lab["pair_id"]] = (path, lab)  # one file per pair: load_labels holds pair_id to the file name
        inputs[_rel(path)] = _sha(Path(path).read_bytes())
    cache: dict[str, RunData] = {}
    pairs: dict[str, Any] = {}
    for pair in sorted(set(labels) | set(pseudo_pair.PAIRS)):
        if pair not in labels:
            if pair in refused:
                pairs[pair] = {"missing": [_missing("labels", [], f"eval/labels/{pair}.json was refused; the message names every problem")]}
                continue
            pp = pseudo_pair.PAIRS[pair]
            cut = [] if all(p.is_file() for p in pseudo_pair.walk_paths(pp)) else [f".venv/bin/python scripts/pseudo_pair.py cut {pair}"]
            cmds = [*cut, pseudo_pair.run_command(pp), *label_commands(pseudo_pair.run_id_for(pair), pair)]
            pairs[pair] = {"missing": [_missing("labels", cmds, f"no eval/labels/{pair}.json")]}
            continue
        path, lab = labels[pair]
        pairs[pair] = evaluate_pair(lab, path, runs, shas, cache, th, (overrides or {}).get(pair))
    for run in cache.values():
        inputs.update(run.inputs)
    results = {
        "schema": RESULT_SCHEMA,
        "config_hash": contracts.config_hash(th),
        "radii_units": {"recall": float(th["eval"]["recall_radius_units"]), "false_mark": float(th["eval"]["false_mark_radius_units"])},
        "pairs": pairs,
        "method_lines": method_lines(pairs),
        "not_computed": NOT_COMPUTED,
        "inputs": dict(sorted(inputs.items())),
    }
    return results, errors


# Snapshots ---------------------------------------------------------------------------------------------------------


def snapshot_run(run: RunData, out_root: Path | None = None) -> Path:
    """Copy what the metrics read (JSON and panel masks, never a frame, no voice.json) to eval/snapshots/<run_id>/:
    the REDACTED files as the redacted bytes load_run hashed, the timing events and ledger rows as its canonical
    lines, and snapshot.json naming the contracts.py the snapshot was read with."""
    if run.snapshot:
        return run.root
    dst = (out_root or SNAPSHOTS) / run.run_id
    if dst.exists():
        shutil.rmtree(dst)
    rels = ["run.json", "spark.json", *(f"videos/{role}.json" for role in ROLES), *(f"derived/{n}" for n in SNAPSHOT_DERIVED)]
    rels += [p.relative_to(run.root).as_posix() for p in sorted((run.root / "sam").glob("*/*.summary.json"))]
    rels += [p.relative_to(run.root).as_posix() for p in sorted((run.root / "crops").glob("*/*.mask.png"))]
    for rel in rels:
        src = run.root / rel
        if not src.is_file():
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel in run.redacted:
            (dst / rel).write_bytes(run.redacted[rel])
        elif rel in REDACTED and src.read_bytes().strip():
            (dst / rel).write_bytes(contracts.json_bytes(redact(rel, json.loads(src.read_bytes()))))
        else:
            shutil.copyfile(src, dst / rel)
    for name, rows in (("events_timing.ndjson", run.events), ("ledger.jsonl", run.ledger_rows)):
        (dst / name).parent.mkdir(parents=True, exist_ok=True)
        (dst / name).write_text("".join(contracts.canonical_json(row) + "\n" for row in rows), encoding="ascii")
    note = {"schema": SNAPSHOT_SCHEMA, "run_id": run.run_id, "contracts_sha256": _sha(CONTRACTS_PY.read_bytes()), "redacted": sorted(REDACTED)}
    contracts.write_json(dst / "snapshot.json", note)
    return dst


# Markdown ----------------------------------------------------------------------------------------------------------


def _row(name: str, value: str) -> str:
    return f"| {name} | {value} |"


def _pair_rows(entry: dict[str, Any]) -> list[str]:
    rows = []
    v, cov, marks, sp = entry.get("verdicts", {}), entry.get("coverage", {}), entry.get("marks", {}), entry.get("staysput", {})
    if "false_new_nothing_changed" in v:
        rows.append(_row('False "new" where nothing changed', f"{v['false_new_nothing_changed']['text']} return-mark verdicts; panels with a new call {v['panels_with_a_new_call']['text']}"))
    if v:
        extra = f"needs your eye {v['panels_needs_your_eye']}, not comparable {v['panels_not_comparable']}, not compared {v['panels_not_compared']}"
        rows.append(_row("Panel calls right", f"{v['panel_calls_right']['text']} ({extra})"))
        if v["changed_panels_called_changed"]["m"]:
            rows.append(_row("Changed panels called new", v["changed_panels_called_changed"]["text"]))
        rows.append(_row("Needs your eye", f"{v['needs_your_eye_share']['text']} verdicts on comparable panels"))
        rows.append(_row("Re-shoot match, unchanged panels", f"{v['re_shoot_match_unchanged_panels']['text']} pickup marks already there at return"))
        for key, name in (("false_new", "False new (labeled pre-existing)"), ("false_pre_existing", "False pre-existing (labeled new)"), ("mark_verdicts_right", "Mark verdicts right"), ("re_shoot_match_real_marks", "Re-shoot match, real marks")):
            if key in v:
                rows.append(_row(name, v[key]["text"]))
        if "needs_your_eye_resolved_already_there" in v:
            rows.append(_row("Needs your eye resolved as already there", f"{v['needs_your_eye_resolved_already_there']['text']} with a return mark (pickup only, never resolvable: {v['needs_your_eye_pickup_only']})"))
    if cov.get("status") == "ok":
        rows.append(_row("Coverage state right", f"{cov['right']['text']} panel views (side swaps {cov['side_swaps']}, lane not run {cov['lane_not_run']})"))
    if marks.get("status") == "ok":
        on_checked = f"; {marks['unjudged_on_checked_panels']} on checked panels" if marks["unjudged_on_checked_panels"] else ""
        rows.append(_row("Mark precision", f"{marks['precision']['text']} (unjudged {marks['unjudged_system_marks']}{on_checked})"))
        rows.append(_row("Mark recall", f"{marks['recall']['text']} (S {marks['recall_by_size']['S']['text']}, M {marks['recall_by_size']['M']['text']}, L {marks['recall_by_size']['L']['text']})"))
        if marks.get("labeler_overrides"):
            rows.append(_row("Labeler overrides", f"{marks['labeler_overrides']['text']} (marks called not real or added by hand)"))
    elif marks:
        rows.append(_row("Mark precision and recall", marks["status"]))
    if sp:
        rows.append(_row("Stays Put agreement", sp["agreement"]["text"] + f" (inconclusive or no track {sp['inconclusive_or_no_track']['text']})" if sp.get("status") == "ok" else sp.get("status", "")))
    for s in entry.get("stability", []):
        if s["same_config"]:
            text = f"coverage {s['coverage_same']['text']}; marks found again {s['marks_found_again']['text']}; panel verdicts {s['panel_verdicts_same']['text']}; SAM lane ids {s['lane_object_ids_same']['text']}"
        else:
            text = f"SAM lane ids {s['lane_object_ids_same']['text']}; thresholds differ ({s['config_hashes'][0]} vs {s['config_hashes'][1]}), so nothing derived is compared"
        rows.append(_row(f"Run to run vs {s['other_run']}", text))
    cal = entry.get("calibration")
    if cal:
        sigmas = ", ".join(f"{k} {cal[k]:.3f}" for k in ("sigma_frame", "sigma_spark", "sigma_view") if cal.get(k) is not None)
        rows.append(_row("Calibration", f"tau_pre {cal['tau_pre']:.3f}, tau_eye {cal['tau_eye']:.3f}, calibrated {str(cal['calibrated']).lower()}" + (f"; {sigmas}" if sigmas else "")))
    hr, cm = entry.get("handle_residual_units") or {}, entry.get("chance_match")
    if hr.get("n"):
        rows.append(_row("Handle residual", f"median {hr['median']:.3f} units, max {hr['max']:.3f}, on {hr['n']} comparable panels"))
    if cm and cm["comparable_panels"]:
        rows.append(_row("Chance match", f"{100 * cm['per_false_mark']:.1f}% per false return mark (computed: {100 * cm['per_pickup_mark']:.1f}% per pickup mark at tau_pre {cm['tau_pre']:.3f}, {cm['pickup_marks']} pickup marks on {cm['comparable_panels']} comparable panels)"))
    bill = entry.get("billable")
    if bill:
        for label, name in (("new_or_unseen", "Would-be-billable, new or unseen"), ("needs_your_eye", "Would-be-billable, needs your eye")):
            if any(row["m"] for row in bill[label].values()):
                rows.append(_row(name, "; ".join(f"at {k} {row['text']}" for k, row in bill[label].items())))
    if "pseudo" in entry:
        p = entry["pseudo"]
        rows.append(_row("Noise floor", f"{p['line']}; tau_pre implied {p['tau_pre_implied']:.3f}"))
    return rows


def markdown(results: dict[str, Any], refused: list[str] | None = None) -> str:
    out = [
        "# Eval",
        "",
        f"Config hash {results['config_hash'][:12]}; recall radius {results['radii_units']['recall']} units, false-mark radius "
        f"{results['radii_units']['false_mark']} units. Counts under 20 print without a percentage. Each pair stands alone; nothing is pooled.",
    ]
    if refused:
        lines = [r.encode("ascii", "backslashreplace").decode("ascii") for r in refused]
        out += ["", "Refused labels files (results.json was not written):"] + [f"- {r}" for r in lines]
    costs = []
    for pair, entry in results["pairs"].items():
        out += ["", f"## {pair}"]
        lab = entry.get("labels")
        if lab:
            out.append(f"Labels: {lab['path']} ({lab['status']}); {lab['kind']}, table {lab['table']}, {lab['split']} pair.")
        if entry.get("primary_run"):
            others = [r for r in entry["runs"] if r != entry["primary_run"]]
            out.append(f"Run: {entry['primary_run']} ({entry.get('run_status', '?')})" + (f"; also {', '.join(others)}" if others else "") + ".")
            rows = _pair_rows(entry)
            if rows:
                out += ["", "| Metric | Value |", "|---|---|", *rows]
        for miss in entry.get("missing", []):
            note = miss["note"][:1].upper() + miss["note"][1:]
            out += ["", f"Missing: {miss['what']}." + (f" {note}." if note else "")]
            if miss["commands"]:
                out += ["```", *miss["commands"], "```"]
        for rid, c in sorted(entry.get("cost", {}).items()):
            costs.append((pair, rid, c))
    if costs:
        out += ["", "## Cost and wall clock per walk (runs/ledger.jsonl)", "", "| Pair | Run | Walk | Video s | Dollars | Dollars per video s |", "|---|---|---|---|---|---|"]
        for pair, rid, c in costs:
            for role, w in c["walks"].items():
                per = "" if w["usd_per_video_s"] is None else f"{w['usd_per_video_s']:.4f}"
                secs = "" if w["video_s"] is None else f"{w['video_s']:.1f}"
                out.append(f"| {pair} | {rid} | {role} | {secs} | {w['usd']:.4f} | {per} |")
            wc = c["wall_clock"]
            span = f"{wc['upload_to_end_s']:.1f} s from the last upload to {wc['ended_with']}" if wc.get("upload_to_end_s") is not None else "no event clock"
            out.append(f"| {pair} | {rid} | all | | {c['usd_total']:.4f} | {span}; unattributed ${c['usd_unattributed']:.4f} |")
    if results.get("method_lines"):
        out += ["", "Method page lines (results.json method_lines):"] + [f"- {line}" for line in results["method_lines"]]
    out += ["", "Not computed:"] + [f"- {n['metric']}: {n['reason']}." for n in results["not_computed"]]
    return "\n".join(out) + "\n"


# Command line ------------------------------------------------------------------------------------------------------


def _overrides(values: list[str]) -> dict[str, str]:
    out = {}
    for value in values:
        pair, sep, run_id = value.partition("=")
        if not sep or not SAFE_ID.match(pair) or not SAFE_ID.match(run_id):
            raise SystemExit(f"--run takes PAIR=RUN_ID, got {value!r}")
        out[pair] = run_id
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_run.py", description="eval numbers with N from runs and labels; no model call")
    parser.add_argument("--labels", type=Path, default=LABELS_DIR, help="folder of labels files (default eval/labels)")
    parser.add_argument("--out", type=Path, default=RESULTS, help="results file (default eval/results.json)")
    parser.add_argument("--markdown", type=Path, help="also write the printed tables here")
    parser.add_argument("--run", action="append", default=[], metavar="PAIR=RUN_ID", help="evaluate this run for the pair")
    parser.add_argument("--from-snapshots", action="store_true", help="read eval/snapshots/ only, as a fresh clone would")
    parser.add_argument("--snapshot", action="store_true", help="copy each evaluated run's inputs to eval/snapshots/ first")
    parser.add_argument("--no-write", action="store_true", help="print only; leave the results file alone")
    args = parser.parse_args(argv)
    overrides = _overrides(args.run)
    label_paths = sorted(Path(args.labels).glob("*.json"))
    results, errors = evaluate(label_paths, from_snapshots=args.from_snapshots, overrides=overrides)
    if args.snapshot and not args.from_snapshots:
        runs = list_runs([(paths.runs_root(), False)])
        for run_id in sorted({rid for e in results["pairs"].values() for rid in e.get("runs", [])} & set(runs)):
            dst = snapshot_run(load_run(run_id, *runs[run_id]))
            print(f"snapshot: {_rel(dst)}", file=sys.stderr)
        results, errors = evaluate(label_paths, from_snapshots=True, overrides=overrides)
    text = markdown(results, errors)
    sys.stdout.write(text)
    for error in errors:
        print(f"labels file refused: {error}", file=sys.stderr)
    if args.markdown:
        args.markdown.write_text(text, encoding="ascii")
    if errors:
        print(f"{_rel(args.out)} was not written: fix the refused labels files first", file=sys.stderr)
    elif not args.no_write:
        contracts.write_json(args.out, results)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
