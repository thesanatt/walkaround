"""The second look (docs/ARCHITECTURE.md section 5 step 10, rule K7), run on Muse Spark Open through Baseten (S1, B11).

make_second_look returns the callable match.derive_match takes, or None when the engine is off. The engine comes from
thresholds spark.second_look_engine: "off" (also when the key is missing, so nothing is asked until the lead turns it
on), "spark" (hosted Muse Spark 1.3 only) or "spark_open" (Spark Open on Baseten through walkaround.open_weights,
with hosted Spark answering when Baseten gives no answer). For each provisional already_there pair the two candidate
marks are outlined on their blurred crops (marks.draw_outline) and the JPEGs are kept as sent
(crops/<role>/<pickup_mark>__<return_mark>.look.jpg), so the provenance can show what the model saw. Each look is a
derived/looks.json row in the CONTRACTS.md "Phase 2 file shapes" form, plus key_slot (which Baseten key answered:
"hackathon" for meta-models/muse-spark-open-1026, "account" for meta-models/muse-spark-open; null when hosted Spark
answered or nothing did), fallback (why hosted Spark answered, or why nothing did), outlines, input_refs,
question_sha256 and every record.

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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import cv2
import numpy as np
from pydantic import Field, model_serializer

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

if TYPE_CHECKING:
    from .match import CounterAsk

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
    key_slot: Literal["hackathon", "account"] | None = None  # the Baseten key that answered (open_weights.KEY_SLOTS), never the key
    request_id: str | None = None
    latency_ms: int | None = None
    fallback_from: Literal["baseten"] | None = None  # Baseten failed or was skipped and hosted Muse Spark 1.3 answered
    valid: bool = False
    fallback: str | None = None
    outlines: list[Box4] = Field(default_factory=list)  # [pickup, return] in crop pixels, as drawn
    input_refs: list[InputRef] = Field(default_factory=list)  # the two outlined JPEGs as sent
    question_sha256: Sha256 | None = None  # panel, prompt, both images, engine, effort and budgets; an equal question reuses the row
    records: list[SparkRequestRecord] = Field(default_factory=list)
    # The one-walk counter-look (lead 11/1, third): kind "counter", mark_id the mark seen in one walk only. Its own
    # walk's field holds the mark; the other walk's field holds that walk's keyframe id, the crop the spot is outlined
    # on. outlines and input_refs are [the mark's walk, the other walk], in the order shown. Both are left out of the
    # JSON on a pair look, so a pair row keeps its bytes.
    kind: Literal["counter"] | None = None
    mark_id: str | None = None

    @model_serializer(mode="wrap")
    def _omit_pair_fields(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict):
            for key in ("kind", "mark_id"):
                if data.get(key) is None:
                    data.pop(key, None)
        return data


class _NotRun(Exception):
    """The look cannot be put to a model; the message is the reason printed on the row."""


_clock = time.monotonic  # tests replace it


@dataclass
class _Breaker:
    """Failed Spark Open looks in a row. Open at limit failures; with ttl_s it closes again ttl_s after the last failure,
    so a breaker that outlives one match stage (look_images keeps one per run) asks Spark Open again later."""

    limit: int
    failures: int = 0
    last: str = ""
    ttl_s: float | None = None
    failed_at: float | None = None

    def fail(self, why: str) -> None:
        self.failures += 1
        self.last = why
        self.failed_at = _clock()

    def ok(self) -> None:
        self.failures = 0

    @property
    def open(self) -> bool:
        if self.failures >= self.limit and self.ttl_s is not None and self.failed_at is not None and _clock() - self.failed_at >= self.ttl_s:
            self.failures = 0  # rested long enough: the next look asks Spark Open again
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
    box = _outline_box(mark)
    if box is None:
        raise _NotRun(f"no box or point for {mark.mark_id}")
    return _drawn(run_id, keyframes.get(mark.kf_id), box, mark.role, f"{pair.pickup_mark}__{pair.return_mark}", mark.mark_id)


def _drawn(run_id: str, kf: Keyframe | None, box: list[float], role: str, name: str, what: str) -> tuple[np.ndarray, list[float], InputRef]:
    """A keyframe's blurred crop with box outlined, the box, and the JPEG as sent (crops/<role>/<name>.look.jpg)."""
    if kf is None or kf.crop is None:
        raise _NotRun(f"no crop for {what}")
    bgr = cv2.imread(str(paths.run_dir(run_id) / kf.crop.path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise _NotRun(f"crop file missing for {what}")
    image = draw_outline(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), box, width_px=OUTLINE_PX)
    jpeg = base64.b64decode(spark_client.prepare_image(image)[0].split(",", 1)[1])  # the bytes both engines send
    path = paths.crop_path(run_id, role, _safe(name) + ".look")
    path.write_bytes(jpeg)
    return image, box, InputRef(path=paths.rel(run_id, path), sha256=hashlib.sha256(jpeg).hexdigest())


def _question(panel_id: PanelId, refs: list[InputRef], *, prompt: str | None = None, **asked: Any) -> str:
    """sha256 of what shapes the answer: panel, prompt, both images and how it is asked (engine, effort, budgets)."""
    doc = {"panel": panel_id, "prompt": prompt if prompt is not None else spark_client.look_prompt(panel_id), "images": [r.sha256 for r in refs], **asked}
    return hashlib.sha256(canonical_json(doc).encode("ascii")).hexdigest()


def _key_fields(key: tuple[str, str], counter: str | None) -> dict[str, Any]:
    """pickup_mark and return_mark of a row, plus kind and mark_id on a counter-look (counter is its mark id)."""
    out: dict[str, Any] = {"pickup_mark": key[0], "return_mark": key[1]}
    if counter is not None:
        out.update(kind="counter", mark_id=counter)
    return out


def _not_run(pair: MatchCandidate | tuple[str, str], panel_id: PanelId | None, why: str, *, counter: str | None = None) -> LookRecord:
    key = pair if isinstance(pair, tuple) else (pair.pickup_mark, pair.return_mark)
    return LookRecord(**_key_fields(key, counter), panel_id=panel_id, reason=open_weights.plain_reason(f"not run: {why}"))


def _row(
    pair: MatchCandidate | tuple[str, str],
    panel_id: PanelId,
    shown: _Shown,
    question: str,
    records: list[SparkRequestRecord],
    *,
    notes: list[str],
    baseten_failed: bool,
    counter: str | None = None,
) -> LookRecord:
    """The row of a look put to a model: the last record that came back stands; if none did, the look did not run."""
    standing = next((r for r in reversed(records) if open_weights.came_back(r)), None)
    key = pair if isinstance(pair, tuple) else (pair.pickup_mark, pair.return_mark)
    common: dict[str, Any] = dict(
        **_key_fields(key, counter),
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
        key_slot=open_weights.key_slot_of(standing),
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
        records, notes, baseten_failed = self._ask((pair.pickup_mark, pair.return_mark), a.panel_id, shown)
        return _row(pair, a.panel_id, shown, question, records, notes=notes, baseten_failed=baseten_failed)

    def counter(self, ask: "CounterAsk") -> LookVerdict | None:
        """The one-walk counter-look (lead 11/1, third): the mark outlined on its own walk's crop, then the other walk's
        crop of the pair compared on with the spot outlined (ask.spot), asked with spark_client.counter_prompt in the
        same engine order as the pair look. The row (kind "counter") goes to looks.json; same, different or unsure is
        returned, or None when the look did not run (the label stands)."""
        try:
            row = self._counter(ask)
        except Exception as exc:  # a look must never stop the match stage
            row = _not_run(ask.key, ask.panel_id, f"the look failed: {type(exc).__name__}: {redact(str(exc))}", counter=ask.mark_id)
        self.asked[ask.key] = row
        write_json(paths.looks_json(self.run_id), [self.asked[key] for key in sorted(self.asked)])
        return row.verdict

    def _counter(self, ask: "CounterAsk") -> LookRecord:
        marks = self._load()
        mark = marks.get(ask.mark_id)
        if mark is None:
            return _not_run(ask.key, ask.panel_id, f"{ask.mark_id} is not in marks.json", counter=ask.mark_id)
        box = _outline_box(mark)
        try:
            if box is None:
                raise _NotRun(f"no box or point for {mark.mark_id}")
            image_a, box_a, ref_a = _drawn(self.run_id, self._keyframes.get(mark.kf_id), box, mark.role, f"{mark.mark_id}__counter", mark.mark_id)
            image_b, box_b, ref_b = _drawn(self.run_id, self._keyframes.get(ask.other_kf), list(ask.spot), ask.other_role, f"{mark.mark_id}__spot", ask.other_kf)
        except _NotRun as exc:
            return _not_run(ask.key, ask.panel_id, str(exc), counter=ask.mark_id)
        shown = _Shown(images=(image_a, image_b), outlines=[box_a, box_b], refs=[ref_a, ref_b])
        prompt = spark_client.counter_prompt(ask.panel_id, mark.role)
        question = _question(ask.panel_id, shown.refs, prompt=prompt, engine=self.engine, effort=self.effort, max_tokens=[self.max_tokens, self.retry_max_tokens])
        prior = self._prior.get(ask.key)
        if prior is not None and prior.verdict is not None and prior.question_sha256 == question and prior.kind == "counter":
            return prior  # a resumed run: this question already has an answer
        records, notes, baseten_failed = self._ask(ask.key, ask.panel_id, shown, prompt=prompt, kind="counter")
        return _row(ask.key, ask.panel_id, shown, question, records, notes=notes, baseten_failed=baseten_failed, counter=ask.mark_id)

    def _ask(self, key: tuple[str, str], panel_id: PanelId, shown: _Shown, *, prompt: str | None = None, kind: str | None = None) -> tuple[list[SparkRequestRecord], list[str], bool]:
        """Every record of the look in order, the notes on why hosted Spark answered (or nothing did), and whether
        Spark Open failed or was skipped before hosted Spark was asked. prompt and kind as the counter-look passes them
        (None on the pair look)."""
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
                        pair=key,
                        max_tokens=self.max_tokens,
                        retry_max_tokens=self.retry_max_tokens,
                        **({"prompt": prompt, "kind": kind} if kind else {}),
                    )
                except open_weights.OpenWeightsError as exc:
                    self.breaker.fail(str(exc))
                    records.extend(exc.attempts)
                    notes.append(f"Spark Open gave no answer: {exc}")
                else:
                    self.breaker.ok()
                    return list(result.attempts or [result.record]), [], False
        try:
            hosted = spark_client.second_look(image_a, image_b, panel_id=panel_id, run_id=self.run_id, input_refs=shown.refs, effort=self.effort, **({"prompt": prompt} if prompt else {}))
        except Exception as exc:  # a missing key or a bug; what Spark Open already answered stays on the row
            notes.append(f"Muse Spark 1.3 failed: {type(exc).__name__}: {redact(str(exc))}")
        else:
            attempts = list(hosted.attempts or [hosted.record])
            records.extend(attempts)
            if not any(open_weights.came_back(r) for r in attempts):
                notes.append(f"Muse Spark 1.3 gave no answer: {attempts[-1].incomplete_reason}")
        return records, notes, self.engine == "spark_open"


# look_images: one breaker per run, as SecondLook keeps one per match stage. It lives as long as the server, so it closes
# again open_weights.FAILURE_TTL_S after its last failure (review 11/1): later claims on the run ask Spark Open again.
_IMAGE_BREAKERS: dict[str, _Breaker] = {}
_IMAGE_BREAKERS_LOCK = threading.Lock()


def _image_breaker(run_id: str) -> _Breaker:
    with _IMAGE_BREAKERS_LOCK:
        return _IMAGE_BREAKERS.setdefault(run_id, _Breaker(BASETEN_FAILURES_MAX, ttl_s=open_weights.FAILURE_TTL_S))


def look_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    panel_id: PanelId,
    run_id: str,
    ledger_run_id: str | None = None,
    input_refs: list[InputRef],
    th: dict,
    pair: tuple[str, str] | None = None,
    provenance: Path | None = None,
) -> tuple[SparkRequestRecord | None, list[SparkRequestRecord], list[str]]:
    """The second look on two images already outlined (pickup first): a claim's mark is never in marks.json, so the
    claim hands its two crops here. Same engine order and breaker as SecondLook._ask: under "spark_open", Muse Spark
    Open on Baseten (the hackathon key, then the account key), hosted Muse Spark 1.3 when Baseten gives no answer or
    its breaker is open; under "spark", hosted only; under "off", nothing. Ledger rows carry ledger_run_id (a claim
    bills as "<run_id>.<claim_id>"; default run_id). Baseten's provenance lines go to provenance when given (a claim
    passes claim/<claim_id>.open_weights.ndjson, since the run's own open_weights.ndjson is sealed at run end), else
    to the run's open_weights.ndjson; no folder named after ledger_run_id is ever made. Returns (the record that
    stands: the last that came back, or None; every record in order; notes on why hosted Spark answered or nothing
    did)."""
    engine = engine_of(th)
    if engine == "off":
        return None, [], ["the second look is off (thresholds spark.second_look_engine)"]
    effort = str(th["spark"]["effort"])
    max_tokens = int(th["spark"]["max_output_tokens"])
    retry_max_tokens = int(th["spark"]["retry_max_output_tokens"])
    rpm = float(th["orchestrator"]["baseten_rpm"])
    bill = ledger_run_id or run_id
    records: list[SparkRequestRecord] = []
    notes: list[str] = []
    if engine == "spark_open":
        breaker = _image_breaker(run_id)
        if breaker.open:
            notes.append(f"Spark Open skipped after {breaker.failures} failed looks in a row (last: {breaker.last})")
        else:
            try:
                result = open_weights.second_look(
                    image_a,
                    image_b,
                    panel_id=panel_id,
                    run_id=run_id,
                    ledger_run_id=bill,
                    log_path=provenance,
                    input_refs=list(input_refs),
                    effort=effort,
                    rpm=rpm,
                    pair=pair,
                    max_tokens=max_tokens,
                    retry_max_tokens=retry_max_tokens,
                )
            except open_weights.OpenWeightsError as exc:
                breaker.fail(str(exc))
                records.extend(exc.attempts)
                notes.append(f"Spark Open gave no answer: {exc}")
            else:
                breaker.ok()
                attempts = list(result.attempts or [result.record])
                return next((r for r in reversed(attempts) if open_weights.came_back(r)), None), attempts, []
    try:
        hosted = spark_client.second_look(image_a, image_b, panel_id=panel_id, run_id=bill, input_refs=list(input_refs), effort=effort)
    except Exception as exc:  # a missing key or a bug; what Spark Open already answered stays in records
        notes.append(f"Muse Spark 1.3 failed: {type(exc).__name__}: {redact(str(exc))}")
    else:
        attempts = list(hosted.attempts or [hosted.record])
        records.extend(attempts)
        if not any(open_weights.came_back(r) for r in attempts):
            notes.append(f"Muse Spark 1.3 gave no answer: {attempts[-1].incomplete_reason}")
    return next((r for r in reversed(records) if open_weights.came_back(r)), None), records, notes


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
    rows = read_json_list(path, LookRecord)
    return _Replay({(r.pickup_mark, r.return_mark): r.verdict for r in rows if r.kind is None}, {r.mark_id: r.verdict for r in rows if r.kind == "counter" and r.mark_id})


class _Replay:
    """replay_second_look's answer: called with a pair, the stored pair look; counter(ask), the stored counter-look of
    the mark (lead 11/1, third). No call, no write; a pair or mark with no row, or whose look did not run, gets None."""

    def __init__(self, pairs: dict[tuple[str, str], LookVerdict | None], counters: dict[str, LookVerdict | None]) -> None:
        self.pairs = pairs
        self.counters = counters

    def __call__(self, pair: MatchCandidate) -> LookVerdict | None:
        return self.pairs.get((pair.pickup_mark, pair.return_mark))

    def counter(self, ask: "CounterAsk") -> LookVerdict | None:
        return self.counters.get(ask.mark_id)


__all__ = ["BASETEN_FAILURES_MAX", "ENGINES", "LookRecord", "SecondLook", "engine_of", "look_images", "make_second_look", "replay_second_look", "stored_looks"]
