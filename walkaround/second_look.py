"""The second look (docs/ARCHITECTURE.md section 5 step 10, rule K7), run on Muse Spark Open through Baseten (S1, B11).

make_second_look returns the callable match.derive_match takes, or None when the engine is off. The engine comes from
thresholds spark.second_look_engine: "off" (also when the key is missing, so nothing is asked until the lead turns it
on), "spark" (hosted Muse Spark 1.3 only) or "spark_open" (Spark Open on Baseten through walkaround.open_weights,
with hosted Spark answering when Baseten gives no answer). For each provisional already_there pair the two candidate
marks are outlined on their blurred crops (marks.draw_outline) and the JPEGs are kept as sent
(crops/<role>/<pickup_mark>__<return_mark>.look.jpg), so the provenance can show what the model saw. Each look is a
derived/looks.json row in the CONTRACTS.md "Phase 2 file shapes" form, plus fallback (why hosted Spark answered, or
why nothing did), outlines, input_refs, question_sha256 and every record.

Nothing here raises a label. same changes nothing; different or unsure become K7 (needs your eye); an answer that came
back but failed look_v1 or was cut is unsure (valid false). A look that did not run (a mark or a crop missing, or no
engine answered) returns None: its row has verdict null and says why, and the deterministic label stands (section 14
items 31 and 32).

Each derive_match call writes only the rows it asked, so looks.json never keeps a pair the final match did not look
at (a call that asks nothing leaves the file as it was). A resumed run reuses a row with a verdict whose question
(panel, prompt, both images, engine, effort, token budgets) is unchanged; a look that did not run is asked again.
replay_second_look answers from looks.json with no network call and no write, so rebuild and verify stay
byte-identical.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np
from pydantic import Field

from . import open_weights, paths, spark_client
from .config import redact
from .contracts import (
    Box4,
    Contract,
    InputRef,
    Keyframe,
    LookVerdict,
    M,
    Mark,
    MatchCandidate,
    PanelId,
    Sha256,
    SparkEngine,
    SparkRequestRecord,
    canonical_json,
    read_json_list,
    write_json,
)
from .marks import draw_outline

ENGINES = ("off", "spark", "spark_open")  # no look; hosted Muse Spark 1.3 only; Spark Open on Baseten with the hosted fallback
ENGINE_DEFAULT = "off"  # a thresholds file without spark.second_look_engine asks nothing
# TODO(D1 E16, E6.second_look_control): spark_open (or spark) goes into thresholds only after a recorded pass: valid
# JSON on the first try for at least 90% of looks and different or unsure on the control pair (so far 9 of 9 valid and
# 6 of 6 controls different, on 3 hand-picked uCO3D pair C pairs: runs b11_smoke_1031, b11_smoke_1031b, b11_fix_smoke_1031).
BASETEN_FAILURES_MAX = 2  # failed Spark Open looks in a row before the rest of the run goes straight to hosted Spark
POINT_BOX_HALF_PX = 12  # a mark with a point and no box gets a 24 px outline around the point
OUTLINE_PX = 2


class LookRecord(Contract):
    """One derived/looks.json row. The binding fields (CONTRACTS.md, Phase 2 file shapes) describe the answer that
    stands; verdict None means the look did not run (no K7) and reason says why. fallback says why hosted Spark
    answered instead of Spark Open, or why no engine did; records holds every answer of both engines in order."""

    pickup_mark: str
    return_mark: str
    panel_id: PanelId | None = None  # None only when neither mark of the pair is in marks.json
    verdict: LookVerdict | None = None
    reason: str = ""  # one plain ASCII sentence, at most 200 characters
    engine: SparkEngine | None = None
    model: str | None = None
    request_id: str | None = None
    latency_ms: int | None = None
    fallback_from: Literal["baseten"] | None = None  # Baseten failed or was skipped and hosted Muse Spark 1.3 answered
    valid: bool = False
    fallback: str | None = None
    outlines: list[Box4] = Field(default_factory=list)  # [pickup, return] in crop pixels, as drawn
    input_refs: list[InputRef] = Field(default_factory=list)  # the two outlined JPEGs as sent
    question_sha256: Sha256 | None = None  # panel, prompt, both images, engine, effort and budgets; an equal question reuses the row
    records: list[SparkRequestRecord] = Field(default_factory=list)


class _NotRun(Exception):
    """The look cannot be put to a model; the message is the reason printed on the row."""


@dataclass
class _Breaker:
    limit: int
    failures: int = 0
    last: str = ""

    def fail(self, why: str) -> None:
        self.failures += 1
        self.last = why

    def ok(self) -> None:
        self.failures = 0

    @property
    def open(self) -> bool:
        return self.failures >= self.limit


@dataclass
class _Shown:
    """What either engine is shown for one pair: the outlined crops, the boxes drawn and the JPEGs as sent."""

    images: tuple[np.ndarray, np.ndarray]
    outlines: list[list[float]]
    refs: list[InputRef]


def engine_of(th: dict) -> str:
    """thresholds spark.second_look_engine, "off" when the key is missing."""
    engine = str(th.get("spark", {}).get("second_look_engine", ENGINE_DEFAULT))
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}, not {engine!r}")
    return engine


def stored_looks(run_id: str) -> list[LookRecord]:
    """looks.json rows; empty when the run has none."""
    path = paths.looks_json(run_id)
    return read_json_list(path, LookRecord) if path.is_file() and path.stat().st_size > 0 else []


def _prior(run_id: str) -> dict[tuple[str, str], LookRecord]:
    """The rows already there, for a resumed run; an unreadable looks.json counts as none, since this call rewrites it."""
    try:
        return {(r.pickup_mark, r.return_mark): r for r in stored_looks(run_id)}
    except (OSError, ValueError):
        return {}


def _read_list(path: Path, model: type[M]) -> list[M]:
    return read_json_list(path, model) if path.is_file() and path.stat().st_size > 0 else []


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def _outline_box(mark: Mark) -> list[float] | None:
    """The mark's box in crop pixels, or a small square around its point, or None."""
    if mark.bbox_crop is not None:
        return [float(v) for v in mark.bbox_crop]
    if mark.point_crop is not None:
        x, y = (float(v) for v in mark.point_crop)
        return [x - POINT_BOX_HALF_PX, y - POINT_BOX_HALF_PX, x + POINT_BOX_HALF_PX, y + POINT_BOX_HALF_PX]
    return None


def _outlined(run_id: str, pair: MatchCandidate, mark: Mark, keyframes: dict[str, Keyframe]) -> tuple[np.ndarray, list[float], InputRef]:
    """The mark's blurred crop with its outline, the box drawn, and the JPEG as sent (written next to the crops)."""
    kf = keyframes.get(mark.kf_id)
    box = _outline_box(mark)
    if kf is None or kf.crop is None:
        raise _NotRun(f"no crop for {mark.mark_id}")
    if box is None:
        raise _NotRun(f"no box or point for {mark.mark_id}")
    bgr = cv2.imread(str(paths.run_dir(run_id) / kf.crop.path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise _NotRun(f"crop file missing for {mark.mark_id}")
    image = draw_outline(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), box, width_px=OUTLINE_PX)
    jpeg = base64.b64decode(spark_client.prepare_image(image)[0].split(",", 1)[1])  # the bytes both engines send
    path = paths.crop_path(run_id, mark.role, _safe(f"{pair.pickup_mark}__{pair.return_mark}") + ".look")
    path.write_bytes(jpeg)
    return image, box, InputRef(path=paths.rel(run_id, path), sha256=hashlib.sha256(jpeg).hexdigest())


def _question(panel_id: PanelId, refs: list[InputRef], **asked: Any) -> str:
    """sha256 of what shapes the answer: panel, prompt, both images and how it is asked (engine, effort, budgets)."""
    doc = {"panel": panel_id, "prompt": spark_client.look_prompt(panel_id), "images": [r.sha256 for r in refs], **asked}
    return hashlib.sha256(canonical_json(doc).encode("ascii")).hexdigest()


def _not_run(pair: MatchCandidate, panel_id: PanelId | None, why: str) -> LookRecord:
    return LookRecord(pickup_mark=pair.pickup_mark, return_mark=pair.return_mark, panel_id=panel_id, reason=open_weights.plain_reason(f"not run: {why}"))


def _row(pair: MatchCandidate, panel_id: PanelId, shown: _Shown, question: str, records: list[SparkRequestRecord], *, notes: list[str], baseten_failed: bool) -> LookRecord:
    """The row of a look put to a model: the last record that came back stands; if none did, the look did not run."""
    standing = next((r for r in reversed(records) if open_weights.came_back(r)), None)
    common: dict[str, Any] = dict(
        pickup_mark=pair.pickup_mark,
        return_mark=pair.return_mark,
        panel_id=panel_id,
        fallback=open_weights.plain_text("; ".join(notes), open_weights.LOG_TEXT_MAX) if notes else None,
        outlines=shown.outlines,
        input_refs=shown.refs,
        question_sha256=question,
        records=records,
    )
    if standing is None:
        return LookRecord(**common, reason=open_weights.plain_reason(f"not run: {notes[-1] if notes else 'no engine answered'}"))
    verdict, reason = open_weights.look_answer(standing)
    return LookRecord(
        **common,
        verdict=verdict,
        reason=open_weights.plain_reason(reason),
        engine=standing.engine,
        model=standing.model,
        request_id=None if standing.request_id.startswith("local-") else standing.request_id,
        latency_ms=standing.ms,
        fallback_from="baseten" if baseten_failed and standing.engine is SparkEngine.META_MODEL_API else None,
        valid=standing.valid,
    )


class SecondLook:
    """The callable make_second_look returns: one look per pair, its row written to looks.json, its verdict returned."""

    def __init__(self, run_id: str, *, th: dict, engine: str) -> None:
        self.run_id = run_id
        self.engine = engine
        self.effort = str(th["spark"]["effort"])
        self.max_tokens = int(th["spark"]["max_output_tokens"])
        self.retry_max_tokens = int(th["spark"]["retry_max_output_tokens"])
        self.rpm = float(th["orchestrator"]["baseten_rpm"])
        self.breaker = _Breaker(BASETEN_FAILURES_MAX)
        self.asked: dict[tuple[str, str], LookRecord] = {}
        self._marks: dict[str, Mark] | None = None
        self._keyframes: dict[str, Keyframe] = {}
        self._prior: dict[tuple[str, str], LookRecord] = {}

    def __call__(self, pair: MatchCandidate) -> LookVerdict | None:
        try:
            row = self._look(pair)
        except Exception as exc:  # a look must never stop the match stage; one that did not run leaves the label alone
            row = _not_run(pair, self._panel(pair), f"the look failed: {type(exc).__name__}: {redact(str(exc))}")
        self.asked[(pair.pickup_mark, pair.return_mark)] = row
        write_json(paths.looks_json(self.run_id), [self.asked[key] for key in sorted(self.asked)])
        return row.verdict

    def _load(self) -> dict[str, Mark]:
        if self._marks is None:
            marks = {m.mark_id: m for m in _read_list(paths.marks_json(self.run_id), Mark)}
            self._keyframes = {kf.kf_id: kf for kf in _read_list(paths.keyframes_json(self.run_id), Keyframe)}
            self._prior = _prior(self.run_id)
            self._marks = marks
        return self._marks

    def _panel(self, pair: MatchCandidate) -> PanelId | None:
        marks = self._marks or {}
        mark = marks.get(pair.pickup_mark) or marks.get(pair.return_mark)
        return mark.panel_id if mark is not None else None

    def _look(self, pair: MatchCandidate) -> LookRecord:
        marks = self._load()
        a, b = marks.get(pair.pickup_mark), marks.get(pair.return_mark)
        if a is None or b is None:
            return _not_run(pair, self._panel(pair), "a mark of the pair is not in marks.json")
        try:
            image_a, box_a, ref_a = _outlined(self.run_id, pair, a, self._keyframes)
            image_b, box_b, ref_b = _outlined(self.run_id, pair, b, self._keyframes)
        except _NotRun as exc:
            return _not_run(pair, a.panel_id, str(exc))
        shown = _Shown(images=(image_a, image_b), outlines=[box_a, box_b], refs=[ref_a, ref_b])
        question = _question(a.panel_id, shown.refs, engine=self.engine, effort=self.effort, max_tokens=[self.max_tokens, self.retry_max_tokens])
        prior = self._prior.get((pair.pickup_mark, pair.return_mark))
        if prior is not None and prior.verdict is not None and prior.question_sha256 == question:
            return prior  # a resumed run: this question already has an answer
        records, notes, baseten_failed = self._ask(pair, a.panel_id, shown)
        return _row(pair, a.panel_id, shown, question, records, notes=notes, baseten_failed=baseten_failed)

    def _ask(self, pair: MatchCandidate, panel_id: PanelId, shown: _Shown) -> tuple[list[SparkRequestRecord], list[str], bool]:
        """Every record of the look in order, the notes on why hosted Spark answered (or nothing did), and whether
        Spark Open failed or was skipped before hosted Spark was asked."""
        image_a, image_b = shown.images
        records: list[SparkRequestRecord] = []
        notes: list[str] = []
        if self.engine == "spark_open":
            if self.breaker.open:
                notes.append(f"Spark Open skipped after {self.breaker.failures} failed looks in a row (last: {self.breaker.last})")
            else:
                try:
                    result = open_weights.second_look(
                        image_a,
                        image_b,
                        panel_id=panel_id,
                        run_id=self.run_id,
                        input_refs=shown.refs,
                        effort=self.effort,
                        rpm=self.rpm,
                        pair=(pair.pickup_mark, pair.return_mark),
                        max_tokens=self.max_tokens,
                        retry_max_tokens=self.retry_max_tokens,
                    )
                except open_weights.OpenWeightsError as exc:
                    self.breaker.fail(str(exc))
                    records.extend(exc.attempts)
                    notes.append(f"Spark Open gave no answer: {exc}")
                else:
                    self.breaker.ok()
                    return list(result.attempts or [result.record]), [], False
        try:
            hosted = spark_client.second_look(image_a, image_b, panel_id=panel_id, run_id=self.run_id, input_refs=shown.refs, effort=self.effort)
        except Exception as exc:  # a missing key or a bug; what Spark Open already answered stays on the row
            notes.append(f"Muse Spark 1.3 failed: {type(exc).__name__}: {redact(str(exc))}")
        else:
            attempts = list(hosted.attempts or [hosted.record])
            records.extend(attempts)
            if not any(open_weights.came_back(r) for r in attempts):
                notes.append(f"Muse Spark 1.3 gave no answer: {attempts[-1].incomplete_reason}")
        return records, notes, self.engine == "spark_open"


def make_second_look(run_id: str, *, th: dict, engine: str | None = None) -> Callable[[MatchCandidate], LookVerdict | None] | None:
    """The callable match.derive_match takes, or None when the engine is "off". engine None reads thresholds
    spark.second_look_engine ("off" when the key is missing). Each call outlines the pair on both walks' blurred
    crops, asks the engine, writes the row to looks.json and returns same, different or unsure, or None when the look
    did not run. Nothing raises out of a call except a failed write of looks.json."""
    engine = engine_of(th) if engine is None else engine
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}, not {engine!r}")
    return None if engine == "off" else SecondLook(run_id, th=th, engine=engine)


def replay_second_look(run_id: str, *, th: dict | None = None) -> Callable[[MatchCandidate], LookVerdict | None] | None:
    """The answers stored in looks.json, for rebuild and verify: no call, no write. None when the run has no looks or
    th has the look off (derive_match then asks nothing, as on the live run); a pair with no row, or whose look did
    not run, gets None, so no K7."""
    if th is not None and engine_of(th) == "off":
        return None
    path = paths.looks_json(run_id)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    verdicts = {(r.pickup_mark, r.return_mark): r.verdict for r in read_json_list(path, LookRecord)}

    def replay(pair: MatchCandidate) -> LookVerdict | None:
        return verdicts.get((pair.pickup_mark, pair.return_mark))

    return replay


__all__ = ["BASETEN_FAILURES_MAX", "ENGINES", "LookRecord", "SecondLook", "engine_of", "make_second_look", "replay_second_look", "stored_looks"]
