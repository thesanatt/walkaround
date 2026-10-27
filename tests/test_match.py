"""Offline tests for walkaround.match (M9) and the within-walk side of walkaround.marks (M6): calibration arithmetic,
the cost, mutual nearest neighbours with a decoy, every row of the section 6 decision table, the gates, wheels,
the mapping chain, dedup with the single-frame and collinear rules, and the two derive entries on a temporary run
directory with a fake Spark. No model call anywhere; the ledger is redirected to the temporary directory."""

from __future__ import annotations

import asyncio
import itertools
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from walkaround import config, geometry, ledger, marks, match, paths, rules, spark_client
from walkaround.contracts import (
    Calibration,
    Conditions,
    Confidence,
    CoverageState,
    Crop,
    FrameUsed,
    GridBox,
    GridPoint,
    HandleRecord,
    Keyframe,
    KeyframeScore,
    LaneStatus,
    LookVerdict,
    Mark,
    MarkKind,
    MarkSource,
    MarksStatus,
    MarkTrack,
    MatchCandidate,
    MatchState,
    Noun,
    Panel,
    PanelId,
    PanelMatch,
    PanelRole,
    Side,
    SizeClass,
    SparkMark,
    SparkMarksAnswer,
    SparkPurpose,
    SparkRequestRecord,
    StaysPutVerdict,
    Trace,
    VerdictFlags,
    VerdictLabel,
    WhitenedMark,
    load_thresholds,
    read_json_list,
    write_json,
)

DOOR = PanelId.LEFT_FRONT_DOOR
ROLES = ("pickup", "return")


@pytest.fixture
def th() -> dict[str, Any]:
    return load_thresholds()


@pytest.fixture
def cal(th: dict) -> Calibration:
    return match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=th)


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    return root


# Builders ------------------------------------------------------------------------------------------------------------

CROP_W, CROP_H = 400, 300
RECT_MASK = np.zeros((CROP_H, CROP_W), dtype=bool)
RECT_MASK[20:280, 20:380] = True
RECT_MOMENTS = geometry.moments_from_mask(RECT_MASK)


def kf(
    kf_id: str,
    role: str,
    panel_id: PanelId = DOOR,
    *,
    sharpness: float = 1.0,
    completeness: float = 1.0,
    status: MarksStatus = MarksStatus.COMPLETE,
    aspect: float = 1.2,
    front_sign: int | None = 1,
    handle: list[float] | None = None,
    conditions: Conditions | None = None,
    window_subtracted: bool = False,
    mask_png: str | None = None,
) -> Keyframe:
    return Keyframe(
        kf_id=kf_id,
        role=role,
        panel_id=panel_id,
        track_id=f"{role}_{panel_id.value}_t1",
        f_proxy=10,
        pts=0.5,
        score=KeyframeScore(area=0.1, completeness=completeness, sharpness=sharpness, glare=0.0, frontality=1.0, total=0.1 * completeness * sharpness),
        aspect_ratio=aspect,
        box_src=[0.0, 0.0, float(CROP_W), float(CROP_H)],
        crop=Crop(path=f"crops/{role}/{kf_id}.jpg", w=CROP_W, h=CROP_H, scale=1.0, dx=0.0, dy=0.0),
        mask_png=mask_png,
        moments=RECT_MOMENTS,
        front_sign=front_sign,
        window_subtracted=window_subtracted,
        handle=HandleRecord(point_src=handle),
        conditions=conditions or Conditions(),
        marks_status=status,
    )


def mark(
    mark_id: str,
    role: str,
    *,
    u: float,
    v: float,
    kind: MarkKind = MarkKind.SCRATCH,
    size_units: float | None = 0.2,
    confidence: Confidence = Confidence.HIGH,
    caps: tuple[str, ...] = (),
    kf_id: str | None = None,
    panel_id: PanelId = DOOR,
    size_class: SizeClass = SizeClass.M,
    source: MarkSource = MarkSource.SPARK_BOX,
    cluster_id: str | None = None,
) -> Mark:
    return Mark(
        mark_id=mark_id,
        role=role,
        panel_id=panel_id,
        kf_id=kf_id or f"{role}_kf1",
        source=source,
        type=kind,
        confidence=confidence,
        size_class=size_class,
        w=[u, v],
        r=math.hypot(u, v),
        size_units=size_units,
        caps=list(caps),
        cluster_id=cluster_id,
        spark_request_id=f"req_{role}",
    )


def wm(m: Mark, frame: FrameUsed = FrameUsed.MOMENT) -> WhitenedMark:
    return WhitenedMark(mark_id=m.mark_id, role=m.role, panel_id=m.panel_id, kf_id=m.kf_id, u=m.w[0], v=m.w[1], r=m.r, frame_used=frame)


def whitened_of(*ms: Mark, frame: FrameUsed = FrameUsed.MOMENT) -> dict[str, WhitenedMark]:
    return {m.mark_id: wm(m, frame) for m in ms}


def panel(panel_id: PanelId = DOOR, *, states: tuple[CoverageState, CoverageState] = (CoverageState.FILMED, CoverageState.FILMED), side_uncertain: bool = False, front_sign: int | None = 1) -> Panel:
    noun = {DOOR: Noun.CAR_DOOR, PanelId.HOOD: Noun.HOOD}.get(panel_id, Noun.WHEEL if panel_id.value.endswith("_wheel") else Noun.CAR_DOOR)
    return Panel(
        panel_id=panel_id,
        noun=noun,
        side=Side.LEFT,
        side_uncertain=side_uncertain,
        front_sign=front_sign,
        per_role={role: PanelRole(state=state, keyframe_ids=[f"{role}_kf1"]) for role, state in zip(ROLES, states)},
        comparable=True,
    )


def gate_inputs(cal: Calibration, th: dict, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "mask_iou": 0.9,
        "residual": None,
        "lane_status": {role: {Noun.CAR_DOOR: LaneStatus.COMPLETED, Noun.HOOD: LaneStatus.COMPLETED} for role in ROLES},
        "marks_truncated_all": {role: False for role in ROLES},
        "calibration": cal,
        "th": th,
    }
    base.update(over)
    return base


# Calibration (5.10) --------------------------------------------------------------------------------------------------


def test_calibrate_with_no_sigmas_uses_the_floors(cal: Calibration) -> None:
    assert cal.sigma_within is None and cal.tau_pre == 0.25 and cal.tau_eye == 0.5 and cal.tau_fix == 0.08 and cal.calibrated


def test_calibrate_arithmetic_from_the_three_sigmas(th: dict) -> None:
    c = match.calibrate(sigma_frame=0.03, sigma_spark=0.06, sigma_view=0.08, pseudo_pair_d=None, th=th)
    sigma_within = math.sqrt(0.03**2 + 0.06**2 + 0.08**2)
    assert c.sigma_within == pytest.approx(sigma_within)
    assert c.tau_pre == pytest.approx(max(0.25, 2.5 * math.sqrt(2.0) * sigma_within))
    assert c.tau_eye == pytest.approx(2.0 * c.tau_pre) and c.tau_fix == pytest.approx(max(0.08, 3 * 0.03)) and c.calibrated
    big = match.calibrate(sigma_frame=0.05, sigma_spark=0.1, sigma_view=None, pseudo_pair_d=None, th=th)
    assert big.tau_pre == pytest.approx(2.5 * math.sqrt(2.0) * math.sqrt(0.05**2 + 0.1**2)) and big.tau_fix == pytest.approx(0.15)


def test_pseudo_pair_raises_tau_pre_to_the_90th_percentile(th: dict) -> None:
    rows = [(0.1, 0.0)] * 5 + [(0.3, 0.0)] * 5  # half the pairs over 0.25: the 90% rule fails
    c = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=rows, th=th)
    assert c.tau_pre == pytest.approx(0.3) and c.tau_eye == pytest.approx(0.6) and c.calibrated
    fine = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=[(0.1, 0.0)] * 19 + [(0.3, 0.0)], th=th)
    assert fine.tau_pre == 0.25  # 95% under tau_pre: unchanged
    far = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=[(0.9, 0.0)] * 10, th=th)
    assert far.tau_pre == 0.25  # nothing under tau_eye: no evidence, unchanged
    by_radius = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=[(0.3 * 1.5, 1.0)] * 10, th=th)
    assert by_radius.tau_pre == pytest.approx(0.3)  # d at r = 1 is brought back to r = 0 by the band factor
    bare = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=[0.3] * 10, th=th)
    assert bare.tau_pre == pytest.approx(0.3)


def test_pseudo_pair_cap_marks_the_matcher_uncalibrated(th: dict) -> None:
    c = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=[(0.45, 0.0)] * 10, th=th)
    assert c.tau_pre == 0.4 and c.tau_eye == pytest.approx(0.8) and not c.calibrated


def test_tau_r_and_measured_sigmas(th: dict) -> None:
    assert match.tau_r(0.25, 0.0, th) == 0.25 and match.tau_r(0.25, 2.0, th) == pytest.approx(0.75)
    assert match.measured_sigmas(th) == (None, None)
    th2 = {"matching": {**th["matching"], "sigma_frame": 0.04, "sigma_spark": 0.07}}
    assert match.measured_sigmas(th2) == (0.04, 0.07)


# Keyframe pairing and the handle ----------------------------------------------------------------------------------------


def test_pair_keyframes_picks_the_smallest_aspect_log_ratio() -> None:
    pickup = [kf("p1", "pickup", aspect=1.2), kf("p2", "pickup", aspect=1.05), kf("p3", "pickup", aspect=1.4)]
    ret = [kf("r1", "return", aspect=1.0), kf("r2", "return", aspect=1.35), kf("r3", "return", aspect=0.8)]
    a, b, diff = match.pair_keyframes(pickup, ret)
    assert (a.kf_id, b.kf_id) == ("p3", "r2") and diff == pytest.approx(abs(math.log(1.4 / 1.35)))
    with pytest.raises(ValueError):
        match.pair_keyframes([], ret)


def test_handle_residual_is_the_whitened_difference_and_none_when_missing() -> None:
    a = kf("p", "pickup", handle=[200.0, 150.0])
    b = kf("r", "return", handle=[220.0, 150.0])
    res = match.handle_residual(a, b)
    expect = geometry.whiten((220.0, 150.0), RECT_MOMENTS, 1)[0] - geometry.whiten((200.0, 150.0), RECT_MOMENTS, 1)[0]
    assert res == pytest.approx([expect, 0.0])
    assert match.handle_residual(a, kf("r2", "return")) is None
    assert match.handle_residual(a, kf("r3", "return", handle=[1.0, 1.0], front_sign=None)) is None


# Gates: every G id reached ------------------------------------------------------------------------------------------


def test_panel_gates_comparable_baseline(cal: Calibration, th: dict) -> None:
    assert match.panel_gates(panel(), kf("p", "pickup"), kf("r", "return"), **gate_inputs(cal, th)) == []


@pytest.mark.parametrize(
    "rule_id, build",
    [
        ("G1", lambda cal, th: (panel(states=(CoverageState.FILMED, CoverageState.GLIMPSED)), kf("p", "pickup"), kf("r", "return"), {})),
        ("G2", lambda cal, th: (panel(), kf("p", "pickup", conditions=Conditions(glare=True)), kf("r", "return"), {})),
        ("G3", lambda cal, th: (panel(), kf("p", "pickup"), kf("r", "return", sharpness=0.3), {})),
        ("G4", lambda cal, th: (panel(), kf("p", "pickup"), kf("r", "return"), {"mask_iou": 0.6})),
        ("G5", lambda cal, th: (panel(), kf("p", "pickup"), kf("r", "return"), {"residual": [0.3, 0.0]})),
        ("G6", lambda cal, th: (panel(side_uncertain=True), kf("p", "pickup"), kf("r", "return"), {})),
        ("G7", lambda cal, th: (panel(), kf("p", "pickup"), kf("r", "return"), {"lane_status": {"pickup": {Noun.CAR_DOOR: LaneStatus.COMPLETED}, "return": {Noun.CAR_DOOR: LaneStatus.FAILED}}})),
        ("G8", lambda cal, th: (panel(PanelId.HOOD), kf("p", "pickup", PanelId.HOOD), kf("r", "return", PanelId.HOOD), {"mask_iou": 0.75})),
        ("G9", lambda cal, th: (panel(), kf("p", "pickup", window_subtracted=True), kf("r", "return"), {})),
        ("G10", lambda cal, th: (panel(), kf("p", "pickup"), kf("r", "return"), {"marks_truncated_all": {"pickup": True, "return": False}})),
    ],
)
def test_each_gate_fires_alone(rule_id: str, build: Any, cal: Calibration, th: dict) -> None:
    p, a, b, over = build(cal, th)
    failing = match.panel_gates(p, a, b, **gate_inputs(cal, th, **over))
    assert failing == [rule_id]
    assert rules.rule(rule_id).id == rule_id


def test_gate_details(cal: Calibration, th: dict) -> None:
    # G1 for a panel filmed in one walk only; G4 and G8 need both keyframes, so they stay quiet here
    assert match.panel_gates(panel(states=(CoverageState.FILMED, CoverageState.NOT_FILMED)), kf("p", "pickup"), None, **gate_inputs(cal, th, mask_iou=None)) == ["G1"]
    # a zero-match lane is not_filmed (G1), never G7
    inputs = gate_inputs(cal, th, lane_status={"pickup": {Noun.CAR_DOOR: LaneStatus.COMPLETED}, "return": {Noun.CAR_DOOR: LaneStatus.ZERO_MATCHES}})
    assert match.panel_gates(panel(states=(CoverageState.FILMED, CoverageState.NOT_FILMED)), kf("p", "pickup"), None, **{**inputs, "mask_iou": None}) == ["G1"]
    # G5 only above tau_pre; a keyframe without a front_sign is G6; a missing IoU on a pair is G4
    assert match.panel_gates(panel(), kf("p", "pickup"), kf("r", "return"), **gate_inputs(cal, th, residual=[0.2, 0.1])) == []
    assert match.panel_gates(panel(), kf("p", "pickup"), kf("r", "return", front_sign=None), **gate_inputs(cal, th)) == ["G6"]
    assert match.panel_gates(panel(), kf("p", "pickup"), kf("r", "return"), **gate_inputs(cal, th, mask_iou=None)) == ["G4"]
    # the list keeps the G1..G10 order when several fail
    many = match.panel_gates(panel(side_uncertain=True), kf("p", "pickup", conditions=Conditions(wet=True)), kf("r", "return", window_subtracted=True), **gate_inputs(cal, th, mask_iou=0.1))
    assert many == ["G2", "G4", "G6", "G9"]


# Cost and mutual nearest neighbours ------------------------------------------------------------------------------------


def test_cost_terms(cal: Calibration, th: dict) -> None:
    a = mark("p1", "pickup", u=0.1, v=0.2)
    same = mark("r1", "return", u=0.1, v=0.2)
    c = match.cost(a, same, wm(a), wm(same), cal, th)
    assert c.d == 0.0 and c.d_pos == 0.0 and c.type_cost == 0.0 and c.size_cost == 0.0
    assert c.r_mean == pytest.approx(math.hypot(0.1, 0.2)) and c.tau_pre_r == pytest.approx(match.tau_r(0.25, c.r_mean, th)) and c.tau_eye_r == pytest.approx(2 * c.tau_pre_r)
    scuff = mark("r2", "return", u=0.4, v=0.6, kind=MarkKind.SCUFF)
    c = match.cost(a, scuff, wm(a), wm(scuff), cal, th)
    assert c.d_pos == pytest.approx(0.5) and c.type_cost == 0.1 and c.d == pytest.approx(0.6)
    dent = mark("r3", "return", u=0.1, v=0.2, kind=MarkKind.DENT, size_units=0.4)
    c = match.cost(a, dent, wm(a), wm(dent), cal, th)
    assert c.type_cost == 0.4 and c.size_cost == pytest.approx(0.1 * math.log(2.0)) and c.d == pytest.approx(0.4 + 0.1 * math.log(2.0))
    px_only = mark("r4", "return", u=0.1, v=0.2, size_units=None)
    assert match.cost(a, px_only, wm(a), wm(px_only), cal, th).size_cost == 0.0  # one side in units, one without: no cost
    bump_a = mark("pb", "pickup", u=0.0, v=0.0, panel_id=PanelId.FRONT_BUMPER)
    bump_b = mark("rb", "return", u=0.0, v=0.0, panel_id=PanelId.FRONT_BUMPER)
    c = match.cost(bump_a, bump_b, wm(bump_a), wm(bump_b), cal, th)
    assert c.tau_pre_r == pytest.approx(0.25 * 1.5) and c.tau_eye_r == pytest.approx(0.5 * 1.5)  # moment-frame fallback on a bumper


def test_cost_on_wheels_uses_the_polar_distance(cal: Calibration, th: dict) -> None:
    a = mark("pw", "pickup", u=0.8, v=0.0, panel_id=PanelId.LEFT_FRONT_WHEEL)
    b = mark("rw", "return", u=0.8, v=math.pi / 4, panel_id=PanelId.LEFT_FRONT_WHEEL)
    c = match.cost(a, b, wm(a, FrameUsed.WHEEL), wm(b, FrameUsed.WHEEL), cal, th)
    assert c.d_pos == pytest.approx(0.8 * math.pi / 4)
    wrap = mark("rw2", "return", u=0.8, v=math.pi - 0.1, panel_id=PanelId.LEFT_FRONT_WHEEL)
    a2 = mark("pw2", "pickup", u=0.8, v=-math.pi + 0.1, panel_id=PanelId.LEFT_FRONT_WHEEL)
    assert match.cost(a2, wrap, wm(a2, FrameUsed.WHEEL), wm(wrap, FrameUsed.WHEEL), cal, th).d_pos == pytest.approx(0.8 * 0.2)


def cands(cal: Calibration, th: dict, pickup: list[Mark], ret: list[Mark]) -> list[MatchCandidate]:
    return [match.cost(a, b, wm(a), wm(b), cal, th) for a in pickup for b in ret]


def test_mutual_nn_rejects_a_close_decoy_and_accepts_a_far_one(cal: Calibration, th: dict) -> None:
    a = mark("p1", "pickup", u=0.0, v=0.0)
    true = mark("r1", "return", u=0.05, v=0.0)
    decoy = mark("r2", "return", u=-0.06, v=0.0)
    out = match.mutual_nn(cands(cal, th, [a], [true, decoy]), th)
    assert out == []  # ratio 1.2 < 1.5: a tie among candidates cannot pass
    far_decoy = mark("r2", "return", u=-0.3, v=0.0)
    out = match.mutual_nn(cands(cal, th, [a], [true, far_decoy]), th)
    assert len(out) == 1 and out[0].return_mark == "r1" and out[0].mutual
    assert out[0].ratio == pytest.approx(6.0) and out[0].margin == pytest.approx(0.25)
    margin_decoy = mark("r2", "return", u=-0.09, v=0.0)  # ratio 1.8 passes, margin 0.04 < 0.125 fails
    assert match.mutual_nn(cands(cal, th, [a], [true, margin_decoy]), th) == []


def test_mutual_nn_single_candidate_passes_by_rule_and_non_mutual_pairs_drop(cal: Calibration, th: dict) -> None:
    a = mark("p1", "pickup", u=0.0, v=0.0)
    b = mark("r1", "return", u=0.9, v=0.0)  # far, but the only candidate: it is listed, the band decides later
    out = match.mutual_nn(cands(cal, th, [a], [b]), th)
    assert len(out) == 1 and out[0].ratio is None and out[0].margin is None and out[0].mutual
    # p1 and p2 both prefer r1; r1 prefers p1: p2 is not mutual and r2 is left for nobody
    p2 = mark("p2", "pickup", u=0.5, v=0.0)
    r2 = mark("r2", "return", u=2.0, v=0.0)
    r1 = mark("r1", "return", u=0.02, v=0.0)
    out = match.mutual_nn(cands(cal, th, [a, p2], [r1, r2]), th)
    assert [(c.pickup_mark, c.return_mark) for c in out] == [("p1", "r1")]
    assert match.mutual_nn([], th) == []


# verdict(): the decision table of section 6 ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, return_mark, label, ids",
    [
        (None, False, VerdictLabel.NOT_SEEN_AT_RETURN, ["L1"]),
        (None, True, VerdictLabel.NEW_OR_UNSEEN, ["L2"]),
        (0.1, False, VerdictLabel.ALREADY_THERE, ["L3"]),
        (0.2499, True, VerdictLabel.ALREADY_THERE, ["L3"]),
        (0.25, False, VerdictLabel.NEEDS_YOUR_EYE, ["L4", "K1"]),
        (0.4999, False, VerdictLabel.NEEDS_YOUR_EYE, ["L4", "K1"]),
        (0.5, False, VerdictLabel.NOT_SEEN_AT_RETURN, ["L5"]),
        (0.9, True, VerdictLabel.NEW_OR_UNSEEN, ["L5"]),
    ],
)
def test_verdict_label_rows(d: float | None, return_mark: bool, label: VerdictLabel, ids: list[str]) -> None:
    assert match.verdict(d, 0.25, 0.5, VerdictFlags(return_mark=return_mark)) == (label, ids)


@pytest.mark.parametrize("cap", [f"K{i}" for i in range(2, 14)])
def test_every_k_flag_lowers_each_clean_label(cap: str) -> None:
    for d, return_mark, base in ((None, False, "L1"), (None, True, "L2"), (0.1, False, "L3"), (0.9, True, "L5")):
        label, ids = match.verdict(d, 0.25, 0.5, VerdictFlags(return_mark=return_mark, **{cap: True}))
        assert label == VerdictLabel.NEEDS_YOUR_EYE
        assert ids[0] == "L4" and base in ids and ids[-1] == cap
    label, ids = match.verdict(0.3, 0.25, 0.5, VerdictFlags(**{cap: True}))
    assert label == VerdictLabel.NEEDS_YOUR_EYE and ids[:2] == ["L4", "K1"] and ids[-1] == cap


def test_no_combination_of_flags_moves_a_label_up() -> None:
    names = [n for n in VerdictFlags.model_fields if n.startswith("K")]
    for return_mark in (False, True):
        for d, _ in ((None, 0), (0.1, 0), (0.3, 0), (0.9, 0)):
            clean, _ids = match.verdict(d, 0.25, 0.5, VerdictFlags(return_mark=return_mark))
            for n in range(1, 4):
                for combo in itertools.combinations(names, n):
                    label, ids = match.verdict(d, 0.25, 0.5, VerdictFlags(return_mark=return_mark, **{c: True for c in combo}))
                    assert label == VerdictLabel.NEEDS_YOUR_EYE
                    assert set(combo) <= set(ids) and all(rules.rule(i) for i in ids)
            if clean != VerdictLabel.NEEDS_YOUR_EYE:
                assert clean in (VerdictLabel.ALREADY_THERE, VerdictLabel.NEW_OR_UNSEEN, VerdictLabel.NOT_SEEN_AT_RETURN)


def test_verdict_uses_the_band_at_the_pair_radius() -> None:
    assert match.verdict(0.3, 0.375, 0.75, VerdictFlags())[0] == VerdictLabel.ALREADY_THERE  # tau_pre_r at r = 1
    assert match.verdict(0.3, 0.25, 0.5, VerdictFlags())[0] == VerdictLabel.NEEDS_YOUR_EYE


# match_panel --------------------------------------------------------------------------------------------------------


def run_panel(cal: Calibration, th: dict, pickup: list[Mark], ret: list[Mark], *, gates: list[str] | None = None, handle_p: list[float] | None = None, handle_r: list[float] | None = None, p: Panel | None = None) -> tuple[PanelMatch, list]:
    kp, kr = kf("pickup_kf1", "pickup", handle=handle_p), kf("return_kf1", "return", handle=handle_r)
    pm = match.match_panel(p or panel(), pickup, ret, whitened_of(*pickup, *ret), (kp, kr), gates or [], cal, th)
    by_id = {m.mark_id: m for m in (*pickup, *ret)}
    verdicts = match.verdicts_for(pm, by_id, {}, {}, {kp.kf_id: kp, kr.kf_id: kr}, cal, Trace(config_hash="c" * 64), th)
    return pm, verdicts


def test_identical_mark_sets_are_all_already_there(cal: Calibration, th: dict) -> None:
    pickup = [mark(f"p{i}", "pickup", u=u, v=v) for i, (u, v) in enumerate([(0.0, 0.0), (1.0, 0.5), (-1.2, -0.8), (0.3, 1.1)])]
    ret = [mark(f"r{i}", "return", u=u, v=v) for i, (u, v) in enumerate([(0.0, 0.0), (1.0, 0.5), (-1.2, -0.8), (0.3, 1.1)])]
    pm, verdicts = run_panel(cal, th, pickup, ret)
    assert pm.state == MatchState.COMPARABLE and len(pm.pairs) == 4 and pm.unmatched_pickup == [] and pm.unmatched_return == []
    assert pm.kf_pair == ["pickup_kf1", "return_kf1"] and pm.aspect_diff == 0.0 and pm.frame_used == FrameUsed.MOMENT
    assert all(v.label == VerdictLabel.ALREADY_THERE and v.rule_id == "L3" for v in verdicts) and len(verdicts) == 4
    assert all(v.distance_units == 0.0 and v.distance_cm_est == 0.0 for v in verdicts)
    assert verdicts[0].trace.config_hash == "c" * 64 and verdicts[0].trace.spark_request_ids == ["req_pickup", "req_return"]
    assert verdicts[0].trace.thresholds["tau_pre"] == 0.25 and "tau_pre_r" in verdicts[0].trace.thresholds


def test_a_shift_of_0_3_units_is_needs_your_eye_and_prints_cm(cal: Calibration, th: dict) -> None:
    pickup = [mark("p0", "pickup", u=0.1, v=0.0)]
    ret = [mark("r0", "return", u=0.4, v=0.0)]
    pm, verdicts = run_panel(cal, th, pickup, ret)
    assert len(pm.pairs) == 1 and pm.pairs[0].d == pytest.approx(0.3)
    v = verdicts[0]
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.rule_id == "L4" and v.trace.rule_ids == ["L4", "K1"]
    assert v.distance_units == pytest.approx(0.3) and v.distance_cm_est == pytest.approx(0.3 * th["matching"]["cm_per_unit_for_print"])
    assert v.reasons == rules.describe(["K1"])


def test_beyond_the_eye_band_both_marks_keep_their_unmatched_labels(cal: Calibration, th: dict) -> None:
    pm, verdicts = run_panel(cal, th, [mark("p0", "pickup", u=0.0, v=0.0)], [mark("r0", "return", u=0.9, v=0.0)])
    assert len(pm.pairs) == 1 and pm.pairs[0].d >= pm.pairs[0].tau_eye_r
    labels = {v.mark_ids[0]: (v.label, v.rule_id) for v in verdicts}
    assert labels == {"p0": (VerdictLabel.NOT_SEEN_AT_RETURN, "L5"), "r0": (VerdictLabel.NEW_OR_UNSEEN, "L5")}


def test_one_sided_marks_are_unmatched_by_definition(cal: Calibration, th: dict) -> None:
    pm, verdicts = run_panel(cal, th, [mark("p0", "pickup", u=0.0, v=0.0), mark("p1", "pickup", u=1.0, v=1.0)], [])
    assert pm.pairs == [] and pm.unmatched_pickup == ["p0", "p1"]
    assert [(v.label, v.rule_id) for v in verdicts] == [(VerdictLabel.NOT_SEEN_AT_RETURN, "L1")] * 2
    pm, verdicts = run_panel(cal, th, [], [mark("r0", "return", u=0.0, v=0.0)])
    assert verdicts[0].label == VerdictLabel.NEW_OR_UNSEEN and verdicts[0].rule_id == "L2"


def test_handle_residual_is_subtracted_from_the_return_marks(cal: Calibration, th: dict) -> None:
    """The return keyframe's frame is shifted by 0.2 units along u (the handle shows it); the return mark carries
    the same shift, so after the subtraction the pair is already_there instead of in the band."""
    shift_px = 0.2 * math.sqrt(RECT_MOMENTS.mu20)
    pickup = [mark("p0", "pickup", u=0.1, v=0.0)]
    ret = [mark("r0", "return", u=0.3, v=0.0)]
    pm, verdicts = run_panel(cal, th, pickup, ret, handle_p=[200.0, 150.0], handle_r=[200.0 + shift_px, 150.0])
    assert pm.handle_residual == pytest.approx([0.2, 0.0], abs=1e-6)
    assert pm.pairs[0].d == pytest.approx(0.0, abs=1e-6) and verdicts[0].label == VerdictLabel.ALREADY_THERE
    pm, verdicts = run_panel(cal, th, pickup, ret)  # no handle: the shift stays and lands in the band... just under tau_pre
    assert pm.handle_residual is None and pm.pairs[0].d == pytest.approx(0.2)


def test_gated_panel_lists_every_mark_unmatched_and_gives_one_l6_verdict(cal: Calibration, th: dict) -> None:
    pm, verdicts = run_panel(cal, th, [mark("p0", "pickup", u=0.0, v=0.0)], [mark("r0", "return", u=0.0, v=0.0)], gates=["G2", "G4"])
    assert pm.state == MatchState.NOT_COMPARABLE and pm.reasons == ["G2", "G4"] and pm.pairs == []
    assert pm.unmatched_pickup == ["p0"] and pm.unmatched_return == ["r0"]
    assert len(verdicts) == 1 and verdicts[0].label == VerdictLabel.NOT_COMPARABLE and verdicts[0].rule_id == "L6"
    assert verdicts[0].trace.rule_ids == ["L6", "G2", "G4"] and verdicts[0].mark_ids == ["p0", "r0"] and verdicts[0].reasons == rules.describe(["G2", "G4"])


def test_group_line_for_a_cluster(cal: Calibration, th: dict) -> None:
    pickup = [mark(f"p{i}", "pickup", u=0.1 * i, v=0.0, kind=MarkKind.CHIP, cluster_id="g1") for i in range(3)]
    ret = [mark(f"r{i}", "return", u=0.1 * i, v=0.0, kind=MarkKind.CHIP, cluster_id="g1") for i in range(4)]
    pm, verdicts = run_panel(cal, th, pickup, ret)
    assert len(pm.groups) == 1
    g = pm.groups[0]
    assert (g.type, g.n_pickup, g.n_return) == (MarkKind.CHIP, 3, 4) and g.text == "3 chips here at pickup, 4 at return, 1 new or unseen"
    # inside the cluster every ratio is near 1, so the ratio test is waived and members pair greedily under tau_pre_r
    assert {(p.pickup_mark, p.return_mark) for p in pm.pairs} == {("p0", "r0"), ("p1", "r1"), ("p2", "r2")} and all(p.ratio is None for p in pm.pairs)
    assert sorted(v.label.value for v in verdicts) == ["already_there"] * 3 + ["new_or_unseen"]
    # two clusters far apart stay unmatched and the line says so
    far = [mark(f"r{i}", "return", u=2.0 + 0.1 * i, v=1.5, kind=MarkKind.CHIP, cluster_id="g2") for i in range(4)]
    pm, _ = run_panel(cal, th, pickup, far)
    assert pm.pairs == [] and pm.groups[0].text == "3 chips here at pickup, 4 at return, 4 new or unseen"
    # a lone mark of the same type outside any group does not count in the line
    pm, _ = run_panel(cal, th, pickup, [*ret, mark("r8", "return", u=-2.0, v=-1.0, kind=MarkKind.CHIP)])
    assert pm.groups[0].text == "3 chips here at pickup, 4 at return, 1 new or unseen" and pm.unmatched_return == ["r3", "r8"]
    assert match._plural(MarkKind.SCRATCH, 2) == "scratches" and match._plural(MarkKind.DIRT, 2) == "dirt marks" and match._plural(MarkKind.DENT, 1) == "dent"


# Wheels (5.8 item 8) -------------------------------------------------------------------------------------------------


def wheel_mark(mark_id: str, role: str, radius: float, angle: float, **kw: Any) -> Mark:
    return mark(mark_id, role, u=radius, v=angle, panel_id=PanelId.LEFT_FRONT_WHEEL, **kw)


def test_wheel_circular_shift_pairs_a_rotated_constellation(cal: Calibration, th: dict) -> None:
    pickup = [wheel_mark("p0", "pickup", 0.9, 0.0, kind=MarkKind.SCUFF), wheel_mark("p1", "pickup", 0.5, 1.0), wheel_mark("p2", "pickup", 0.95, 2.5, kind=MarkKind.CHIP)]
    turn = 1.9  # the wheel turned between walks
    ret = [wheel_mark("r0", "return", 0.9, 0.0 + turn, kind=MarkKind.SCUFF), wheel_mark("r1", "return", 0.5, 1.0 + turn), wheel_mark("r2", "return", 0.95, 2.5 + turn, kind=MarkKind.CHIP)]
    w = whitened_of(*pickup, *ret, frame=FrameUsed.WHEEL)
    pm = match.wheel_match(panel(PanelId.LEFT_FRONT_WHEEL), pickup, ret, w, th=th, calibration=cal)
    assert pm.frame_used == FrameUsed.WHEEL and {(p.pickup_mark, p.return_mark) for p in pm.pairs} == {("p0", "r0"), ("p1", "r1"), ("p2", "r2")}
    assert all(p.d < 1e-6 for p in pm.pairs) and pm.reasons == []


def test_one_mark_wheel_rule_and_k10(cal: Calibration, th: dict) -> None:
    p = [wheel_mark("p0", "pickup", 0.8, 0.3, size_units=0.1)]
    good = [wheel_mark("r0", "return", 0.85, 2.0, kind=MarkKind.SCUFF, size_units=0.15)]
    pm = match.wheel_match(panel(PanelId.LEFT_FRONT_WHEEL), p, good, whitened_of(*p, *good, frame=FrameUsed.WHEEL), th=th, calibration=cal)
    assert len(pm.pairs) == 1 and pm.reasons == [] and pm.pairs[0].d < pm.pairs[0].tau_pre_r
    verdicts = match.verdicts_for(pm, {m.mark_id: m for m in (*p, *good)}, {}, {}, {}, cal, Trace(), th)
    assert verdicts[0].label == VerdictLabel.ALREADY_THERE
    for bad in (
        wheel_mark("r0", "return", 0.95, 2.0, size_units=0.1),  # radius off by more than 0.1
        wheel_mark("r0", "return", 0.8, 2.0, kind=MarkKind.DENT, size_units=0.1),  # incompatible type
        wheel_mark("r0", "return", 0.8, 2.0, size_units=0.25),  # size off by more than a factor 2
    ):
        pm = match.wheel_match(panel(PanelId.LEFT_FRONT_WHEEL), p, [bad], whitened_of(*p, bad, frame=FrameUsed.WHEEL), th=th, calibration=cal)
        assert pm.pairs == [] and pm.reasons == ["K10"] and pm.unmatched_pickup == ["p0"] and pm.unmatched_return == ["r0"]
        verdicts = match.verdicts_for(pm, {"p0": p[0], "r0": bad}, {}, {}, {}, cal, Trace(), th)
        assert all(v.label == VerdictLabel.NEEDS_YOUR_EYE and "K10" in v.trace.rule_ids for v in verdicts)


def test_match_panel_routes_wheels_and_keeps_gates(cal: Calibration, th: dict) -> None:
    p = [wheel_mark("p0", "pickup", 0.8, 0.3)]
    r = [wheel_mark("r0", "return", 0.8, 1.3)]
    kp, kr = kf("pickup_kf1", "pickup", PanelId.LEFT_FRONT_WHEEL), kf("return_kf1", "return", PanelId.LEFT_FRONT_WHEEL)
    pm = match.match_panel(panel(PanelId.LEFT_FRONT_WHEEL), p, r, whitened_of(*p, *r, frame=FrameUsed.WHEEL), (kp, kr), [], cal, th)
    assert pm.frame_used == FrameUsed.WHEEL and pm.state == MatchState.COMPARABLE and len(pm.pairs) == 1 and pm.kf_pair == ["pickup_kf1", "return_kf1"]
    gated = match.match_panel(panel(PanelId.LEFT_FRONT_WHEEL), p, r, whitened_of(*p, *r, frame=FrameUsed.WHEEL), (kp, kr), ["G3"], cal, th)
    assert gated.state == MatchState.NOT_COMPARABLE and gated.reasons == ["G3"] and gated.pairs == [] and gated.unmatched_pickup == ["p0"]


# flags_for and the K table through verdicts_for --------------------------------------------------------------------


def test_flags_for_sources() -> None:
    a = mark("p0", "pickup", u=0.0, v=0.0, caps=("K3", "K12"))
    b = mark("r0", "return", u=0.3, v=0.0, kind=MarkKind.DENT, caps=("K2",))
    pair = MatchCandidate(pickup_mark="p0", return_mark="r0", d=0.3, d_pos=0.3, type_cost=0.0, size_cost=0.0, r_mean=0.15, tau_pre_r=0.25, tau_eye_r=0.5)
    tracks = {"r0": MarkTrack(mark_id="r0", role="return", verdict=StaysPutVerdict.SLIDING)}
    flags = match.flags_for(pair, a, b, tracks, LookVerdict.UNSURE, {"return_kf1": MarksStatus.TRUNCATED}, return_mark=False)
    assert flags.set_caps() == ["K1", "K12", "K13", "K2", "K3", "K5", "K6", "K7"]
    quiet = match.flags_for(None, mark("p1", "pickup", u=0.0, v=0.0), None, {}, LookVerdict.SAME, {"k": MarksStatus.COMPLETE}, return_mark=True)
    assert quiet.set_caps() == [] and quiet.return_mark
    fixed = match.flags_for(None, a, None, {"p0": MarkTrack(mark_id="p0", role="pickup", verdict=StaysPutVerdict.FIXED, curvature_cap=True)}, None, {}, return_mark=False)
    assert "K6" in fixed.set_caps()


K_CASES: dict[str, Any] = {
    "K2": {"a_caps": ("K2",)},
    "K3": {"a_caps": ("K3",)},
    "K4": {"a_kind": MarkKind.DIRT, "b_kind": MarkKind.DIRT, "a_caps": ("K4",)},
    "K5": {"b_kind": MarkKind.DENT},
    "K6": {"track": StaysPutVerdict.INCONCLUSIVE},
    "K7": {"look": LookVerdict.DIFFERENT},
    "K8": {"a_caps": ("K8",)},
    "K9": {"b_caps": ("K9",)},
    "K11": {"a_caps": ("K11",)},
    "K12": {"b_caps": ("K12",)},
    "K13": {"truncated": True},
}


@pytest.mark.parametrize("cap", sorted(K_CASES, key=lambda k: int(k[1:])))
def test_each_k_cap_lowers_an_already_there_pair_with_its_id_in_the_trace(cap: str, cal: Calibration, th: dict) -> None:
    case = K_CASES[cap]
    a = mark("p0", "pickup", u=0.0, v=0.0, kind=case.get("a_kind", MarkKind.SCRATCH), caps=case.get("a_caps", ()))
    b = mark("r0", "return", u=0.0, v=0.0, kind=case.get("b_kind", MarkKind.SCRATCH), caps=case.get("b_caps", ()))
    pair = match.cost(a, b, wm(a), wm(b), cal, th)
    pm = PanelMatch(panel_id=DOOR, state=MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], pairs=[pair])
    tracks = {"p0": MarkTrack(mark_id="p0", role="pickup", verdict=case["track"], request_id="resp_sp")} if "track" in case else {}
    looks = {("p0", "r0"): case["look"]} if "look" in case else {}
    kfs = {"pickup_kf1": kf("pickup_kf1", "pickup"), "return_kf1": kf("return_kf1", "return", status=MarksStatus.TRUNCATED if case.get("truncated") else MarksStatus.COMPLETE)}
    verdicts = match.verdicts_for(pm, {"p0": a, "r0": b}, tracks, looks, kfs, cal, Trace(), th)
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.rule_id == "L4" and cap in v.trace.rule_ids
    if cap == "K5":
        assert v.trace.rule_ids[:2] == ["L4", "K1"]  # the incompatible type cost of 0.4 puts the pair in the band as well
    else:
        assert "L3" in v.trace.rule_ids
    assert v.reasons == rules.describe([i for i in v.trace.rule_ids if i.startswith("K")])
    if "track" in case:
        assert v.trace.sam_request_ids == ["resp_sp"]
    ca, cb = mark("p0", "pickup", u=0.0, v=0.0), mark("r0", "return", u=0.0, v=0.0)
    clean_pm = PanelMatch(panel_id=DOOR, state=MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], pairs=[match.cost(ca, cb, wm(ca), wm(cb), cal, th)])
    clean = match.verdicts_for(clean_pm, {"p0": ca, "r0": cb}, {}, {}, {k: kf(k, r) for k, r in (("pickup_kf1", "pickup"), ("return_kf1", "return"))}, cal, Trace(), th)
    assert clean[0].label == VerdictLabel.ALREADY_THERE and clean[0].trace.rule_ids == ["L3"]


def test_second_look_can_only_lower(cal: Calibration, th: dict) -> None:
    a, b = mark("p0", "pickup", u=0.0, v=0.0), mark("r0", "return", u=0.0, v=0.0)
    pair = match.cost(a, b, wm(a), wm(b), cal, th)
    pm = PanelMatch(panel_id=DOOR, state=MatchState.COMPARABLE, pairs=[pair])
    for look, label in ((LookVerdict.SAME, VerdictLabel.ALREADY_THERE), (LookVerdict.UNSURE, VerdictLabel.NEEDS_YOUR_EYE), (LookVerdict.DIFFERENT, VerdictLabel.NEEDS_YOUR_EYE)):
        v = match.verdicts_for(pm, {"p0": a, "r0": b}, {}, {("p0", "r0"): look}, {}, cal, Trace(), th)[0]
        assert v.label == label
    far = mark("r1", "return", u=0.9, v=0.0)
    pm = PanelMatch(panel_id=DOOR, state=MatchState.COMPARABLE, pairs=[match.cost(a, far, wm(a), wm(far), cal, th)])
    v = match.verdicts_for(pm, {"p0": a, "r1": far}, {}, {("p0", "r1"): LookVerdict.SAME}, {}, cal, Trace(), th)
    assert {x.label for x in v} == {VerdictLabel.NOT_SEEN_AT_RETURN, VerdictLabel.NEW_OR_UNSEEN}  # "same" never creates a match


# Marks: mapping chain, caps, count mode, dedup ------------------------------------------------------------------------


def spark_record(request_id: str = "req1") -> SparkRequestRecord:
    return SparkRequestRecord(request_id=request_id, purpose=SparkPurpose.MARKS, model="muse-spark-1.3", schema_name="marks_v1")


def marks_result(answer: SparkMarksAnswer | None, *, sent: tuple[int, int] = (CROP_W, CROP_H), scale: float = 1.0, orig: tuple[int, int] = (CROP_W, CROP_H), truncated: bool = False, request_id: str = "req1") -> spark_client.MarksResult:
    return spark_client.MarksResult(answer=answer, truncated=truncated, record=spark_record(request_id), sent_width=sent[0], sent_height=sent[1], scale=scale, orig_width=orig[0], orig_height=orig[1])


def spark_mark(x: int, y: int, *, half: int = 20, kind: MarkKind = MarkKind.SCRATCH, confidence: Confidence = Confidence.HIGH, size_class: SizeClass = SizeClass.M) -> SparkMark:
    return SparkMark(type=kind, point=GridPoint(x=x, y=y), bbox=GridBox(x_min=max(0, x - half), y_min=max(0, y - 5), x_max=min(1000, x + half), y_max=min(1000, y + 5)), size_class=size_class, confidence=confidence, note="n")


def test_landmarks_for_each_family(th: dict) -> None:
    assert marks.landmarks_for(DOOR, th) == ["handle", "lock", "mirror_base", "window_front_corner"]
    assert marks.landmarks_for(PanelId.REAR_BUMPER, th) == ["plate_corner_tl", "plate_corner_tr", "plate_corner_bl", "plate_corner_br"]
    assert marks.landmarks_for(PanelId.RIGHT_REAR_WHEEL, th) == ["hub", "rim_lip", "valve"]
    assert marks.landmarks_for(PanelId.HOOD, th) == ["badge", "washer_nozzle", "front_corner_left", "front_corner_right"]
    assert marks.landmarks_for(PanelId.ROOF, th) == [] and marks.landmarks_for(PanelId.LEFT_FRONT_FENDER, th) == []


def test_map_spark_mark_walks_the_whole_chain() -> None:
    crop = Crop(path="", w=1280, h=960, scale=0.5, dx=100.0, dy=50.0)
    result = marks_result(None, sent=(640, 480), scale=0.5, orig=(1280, 960))
    sm = SparkMark(type=MarkKind.SCRATCH, point=GridPoint(x=500, y=500), bbox=GridBox(x_min=250, y_min=250, x_max=750, y_max=500), size_class=SizeClass.M, confidence=Confidence.HIGH)
    bbox_crop, point_crop, bbox_src, point_src = marks.map_spark_mark(sm, result, crop)
    assert point_crop == [640.0, 480.0] and point_src == [1380.0, 1010.0]  # grid 500 -> sent 320,240 -> crop 640,480 -> source
    assert bbox_crop == [320.0, 240.0, 962.0, 482.0]  # half-open: one sent pixel past x_max and y_max, then x2 in crop pixels
    assert bbox_src == [100.0 + 640.0, 50.0 + 480.0, 100.0 + 1924.0, 50.0 + 964.0]
    outline = marks.draw_outline(np.zeros((960, 1280, 3), dtype=np.uint8), bbox_crop)
    assert outline[240, 320].tolist() == [255, 0, 255] and outline[300, 500].tolist() == [0, 0, 0]


def test_marks_from_answer_sets_k3_k4_k12_and_ids(th: dict) -> None:
    answer = SparkMarksAnswer(marks=[spark_mark(500, 500), spark_mark(100, 100, half=4, kind=MarkKind.DIRT, confidence=Confidence.LOW)])
    k = kf("pickup_left_front_door_kf1", "pickup")
    out = marks.marks_from_answer(answer, marks_result(answer), k, k.crop, th=th)
    assert [m.mark_id for m in out] == ["pickup_left_front_door_kf1_m1", "pickup_left_front_door_kf1_m2"]
    assert out[0].caps == [] and out[0].source == MarkSource.SPARK_BOX and out[0].spark_request_id == "req1" and out[0].description == "n"
    assert out[1].caps == ["K3", "K4", "K12"] and out[1].size_px_src < th["dedup"]["small_mark_px"]
    assert out[0].point_crop == [200.0, 150.0] and out[0].point_src == [200.0, 150.0]


def test_whiten_marks_fills_w_r_and_size_units(th: dict) -> None:
    answer = SparkMarksAnswer(marks=[spark_mark(500, 500), spark_mark(950, 500)])
    k = kf("pickup_kf1", "pickup")
    out, whitened = marks.whiten_marks(marks.marks_from_answer(answer, marks_result(answer), k, k.crop, th=th), k, th=th)
    assert out[0].w == pytest.approx([0.0, 0.0], abs=0.01) and out[0].r == pytest.approx(0.0, abs=0.01)
    assert out[1].w[0] > 1.5 and whitened[out[1].mark_id].u == out[1].w[0] and whitened[out[1].mark_id].frame_used == FrameUsed.MOMENT
    assert out[0].size_units == pytest.approx(17.0 / math.sqrt(RECT_MOMENTS.mu20), rel=1e-6)  # a 40-grid box on a 400 px crop is 16 px plus the half-open edge
    no_frame, none = marks.whiten_marks(out, kf("pickup_kf1", "pickup", front_sign=None), th=th)
    assert none == {} and no_frame == out
    rebuilt = marks.whitened_from_marks(out)
    assert set(rebuilt) == {m.mark_id for m in out} and rebuilt[out[0].mark_id].u == out[0].w[0]


def test_count_union_adds_only_far_points_with_k8(th: dict) -> None:
    answer = SparkMarksAnswer(marks=[spark_mark(500, 500)])
    k = kf("pickup_kf1", "pickup")
    box_marks, whitened = marks.whiten_marks(marks.marks_from_answer(answer, marks_result(answer), k, k.crop, th=th), k, th=th)
    count = spark_client.CountResult(points=[GridPoint(x=505, y=502), GridPoint(x=900, y=500)], count=2, truncated=False, record=spark_record("req_count"))
    extra = marks.count_union(box_marks, count, marks_result(answer), k, k.crop, whitened, th=th)
    assert len(extra) == 1 and extra[0].source == MarkSource.SPARK_COUNT and extra[0].caps == ["K4", "K8", "K12"]
    assert extra[0].mark_id == "pickup_kf1_c2" and extra[0].spark_request_id == "req_count" and extra[0].w[0] > 1.0
    assert marks.count_union(box_marks, count, marks_result(answer), kf("pickup_kf1", "pickup", front_sign=None), k.crop, whitened, th=th) == []


def three_keyframes(role: str = "pickup", **kw: Any) -> list[Keyframe]:
    return [kf(f"{role}_kf{i}", role, **kw) for i in (1, 2, 3)]


def test_dedup_keeps_the_medoid_records_merged_from_and_sigma_view(th: dict) -> None:
    kfs = three_keyframes()
    a = mark("pickup_kf1_m1", "pickup", u=0.00, v=0.00, kf_id="pickup_kf1")
    b = mark("pickup_kf2_m1", "pickup", u=0.05, v=0.00, kf_id="pickup_kf2")
    c = mark("pickup_kf3_m1", "pickup", u=0.10, v=0.00, kf_id="pickup_kf3")
    far = mark("pickup_kf2_m2", "pickup", u=1.0, v=1.0, kf_id="pickup_kf2")
    out, sigma_view = marks.dedup_within_walk([a, b, c, far], whitened_of(a, b, c, far), kfs, th=th)
    by_id = {m.mark_id: m for m in out}
    assert set(by_id) == {"pickup_kf2_m1", "pickup_kf2_m2"}
    assert by_id["pickup_kf2_m1"].merged_from == ["pickup_kf1_m1", "pickup_kf3_m1"] and not by_id["pickup_kf2_m1"].single_frame
    assert sigma_view == pytest.approx(math.sqrt((0.05**2 + 0.05**2) / (2 * 2)))
    assert by_id["pickup_kf2_m2"].single_frame and "K2" in by_id["pickup_kf2_m2"].caps
    assert marks.dedup_within_walk([], {}, kfs, th=th) == ([], None)


def test_single_frame_rule_needs_the_other_two_complete_and_sharp(th: dict) -> None:
    lone = mark("pickup_kf1_m1", "pickup", u=0.0, v=0.0, kf_id="pickup_kf1")
    out, _ = marks.dedup_within_walk([lone], whitened_of(lone), three_keyframes(), th=th)
    assert out[0].single_frame and out[0].caps == ["K2"]
    blurry = [kf("pickup_kf1", "pickup"), kf("pickup_kf2", "pickup", sharpness=0.2), kf("pickup_kf3", "pickup")]
    out, _ = marks.dedup_within_walk([lone], whitened_of(lone), blurry, th=th)
    assert not out[0].single_frame and out[0].caps == []
    truncated = [kf("pickup_kf1", "pickup"), kf("pickup_kf2", "pickup"), kf("pickup_kf3", "pickup", status=MarksStatus.TRUNCATED)]
    out, _ = marks.dedup_within_walk([lone], whitened_of(lone), truncated, th=th)
    assert not out[0].single_frame
    two_only = three_keyframes()[:2]
    out, _ = marks.dedup_within_walk([lone], whitened_of(lone), two_only, th=th)
    assert not out[0].single_frame
    # two marks on the same keyframe never merge, however close
    twin = mark("pickup_kf1_m2", "pickup", u=0.01, v=0.0, kf_id="pickup_kf1")
    out, _ = marks.dedup_within_walk([lone, twin], whitened_of(lone, twin), three_keyframes(), th=th)
    assert len(out) == 2


def test_collinear_merge_joins_a_split_scratch(th: dict) -> None:
    k = kf("pickup_kf1", "pickup")
    answer = SparkMarksAnswer(marks=[SparkMark(type=MarkKind.SCRATCH, point=GridPoint(x=300, y=500), bbox=GridBox(x_min=200, y_min=490, x_max=400, y_max=510), size_class=SizeClass.M, confidence=Confidence.HIGH),
                                     SparkMark(type=MarkKind.SCRATCH, point=GridPoint(x=510, y=500), bbox=GridBox(x_min=410, y_min=490, x_max=610, y_max=510), size_class=SizeClass.M, confidence=Confidence.HIGH),
                                     SparkMark(type=MarkKind.SCRATCH, point=GridPoint(x=300, y=700), bbox=GridBox(x_min=290, y_min=600, x_max=310, y_max=800), size_class=SizeClass.M, confidence=Confidence.HIGH)])
    placed, whitened = marks.whiten_marks(marks.marks_from_answer(answer, marks_result(answer), k, k.crop, th=th), k, th=th)
    out, _ = marks.dedup_within_walk(placed, whitened, [k], th=th)
    assert len(out) == 2
    merged = next(m for m in out if m.merged_from)
    assert merged.merged_from == ["pickup_kf1_m2"] and merged.bbox_crop[0] == pytest.approx(80.0) and merged.bbox_crop[2] == pytest.approx(245.0)
    assert merged.grid_point is None and merged.w is not None and merged.size_units > placed[0].size_units
    assert next(m for m in out if not m.merged_from).mark_id == "pickup_kf1_m3"  # perpendicular: not merged


def test_groups_get_a_cluster_id_and_droplets_or_spray_get_k9(th: dict) -> None:
    kfs = [kf("pickup_kf1", "pickup")]
    chips = [mark(f"pickup_kf1_m{i}", "pickup", u=0.1 * i, v=0.0, kind=MarkKind.CHIP, kf_id="pickup_kf1") for i in range(4)]
    lone = mark("pickup_kf1_m9", "pickup", u=1.5, v=-1.5, kind=MarkKind.CHIP, kf_id="pickup_kf1")
    out, _ = marks.dedup_within_walk([*chips, lone], whitened_of(*chips, lone), kfs, th=th)
    ids = {m.mark_id: m.cluster_id for m in out}
    assert len({ids[c.mark_id] for c in chips}) == 1 and ids[chips[0].mark_id] == "pickup_left_front_door_chip_g1" and ids["pickup_kf1_m9"] is None
    assert all("K9" not in m.caps for m in out)
    droplets = [mark(f"pickup_kf1_d{i}", "pickup", u=0.3 * i - 1.0, v=0.3, size_units=0.01, kf_id="pickup_kf1") for i in range(8)]
    out, _ = marks.dedup_within_walk(droplets, whitened_of(*droplets), kfs, th=th)
    assert all("K9" in m.caps for m in out)
    spray = [mark(f"pickup_kf1_s{i}", "pickup", u=0.5 * i - 1.0, v=1.5, kind=MarkKind.SCUFF, kf_id="pickup_kf1") for i in range(3)]
    out, _ = marks.dedup_within_walk(spray, whitened_of(*spray), kfs, th=th)
    assert all("K9" in m.caps for m in out)
    roof = [mark(f"pickup_kf1_s{i}", "pickup", u=0.5 * i - 1.0, v=1.5, kind=MarkKind.SCUFF, kf_id="pickup_kf1", panel_id=PanelId.ROOF) for i in range(3)]
    out, _ = marks.dedup_within_walk(roof, whitened_of(*roof), [kf("pickup_kf1", "pickup", PanelId.ROOF)], th=th)
    assert all("K9" not in m.caps for m in out)


# The derive entries on a temporary run ----------------------------------------------------------------------------------


def write_crop(run_id: str, k: Keyframe, *, with_mask: bool = True) -> Keyframe:
    root = paths.run_dir(run_id)
    crop_file = root / k.crop.path
    crop_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_file), np.full((CROP_H, CROP_W, 3), 120, dtype=np.uint8))
    if not with_mask:
        return k
    mask_file = paths.crop_mask_png_path(run_id, k.role, k.kf_id)
    cv2.imwrite(str(mask_file), RECT_MASK.astype(np.uint8) * 255)
    return k.model_copy(update={"mask_png": paths.rel(run_id, mask_file)})


async def test_derive_marks_with_a_fake_spark_writes_marks_keyframes_and_sigma_view(runs_dir: Path, th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "01TESTRUNMARKS000000000000"
    paths.ensure_run_layout(run_id)
    kfs = [write_crop(run_id, k, with_mask=False) for k in three_keyframes()]
    write_json(paths.keyframes_json(run_id), kfs)
    calls: list[tuple[PanelId, list[str], str]] = []

    def fake_find_marks(crop_rgb: np.ndarray, panel_id: PanelId, *, landmarks: list[str], effort: str, run_id: str | None, input_ref: Any) -> spark_client.MarksResult:
        calls.append((panel_id, landmarks, effort))
        n = len(calls)
        if n == 3:
            return marks_result(None, truncated=True, request_id=f"req{n}")
        answer = SparkMarksAnswer(marks=[spark_mark(500 + 4 * n, 500), spark_mark(900, 400) if n == 1 else spark_mark(100, 100)], conditions=Conditions(glare=n == 2))
        return marks_result(answer, request_id=f"req{n}")

    monkeypatch.setattr(spark_client, "find_marks", fake_find_marks)
    out, updated, records = await marks.derive_marks(run_id, "pickup", kfs, th=th)
    assert len(calls) == 3 and calls[0] == (DOOR, ["handle", "lock", "mirror_base", "window_front_corner"], "low")
    assert [k.marks_status for k in updated] == [MarksStatus.COMPLETE, MarksStatus.COMPLETE, MarksStatus.TRUNCATED]
    assert updated[1].conditions.glare and not updated[0].conditions.glare
    assert [r.request_id for r in records] == ["req1", "req2", "req3"]
    stored = read_json_list(paths.marks_json(run_id), Mark)
    assert {m.mark_id for m in stored} == {m.mark_id for m in out} and len(out) == 3
    near = next(m for m in out if m.merged_from)
    assert near.merged_from and near.w is not None and near.r is not None and "K2" not in near.caps
    assert all(m.single_frame is False for m in out)  # the third keyframe was truncated, so nothing is single_frame
    refreshed = read_json_list(paths.keyframes_json(run_id), Keyframe)
    assert [k.marks_status for k in refreshed] == [MarksStatus.COMPLETE, MarksStatus.COMPLETE, MarksStatus.TRUNCATED]
    cal = read_json_list(paths.spark_json(run_id), SparkRequestRecord)
    assert [r.request_id for r in cal] == ["req1", "req2", "req3"]
    calibration = Calibration.model_validate_json(paths.calibration_json(run_id).read_text())
    assert calibration.sigma_view is not None and calibration.sigma_view > 0 and calibration.tau_pre == 0.25
    # a second role replaces only its own rows
    monkeypatch.setattr(spark_client, "find_marks", lambda *a, **k: marks_result(SparkMarksAnswer(marks=[spark_mark(500, 500)]), request_id="req_r"))
    ret_kfs = [write_crop(run_id, k, with_mask=False) for k in three_keyframes("return")]
    ret_out, _, _ = await marks.derive_marks(run_id, "return", ret_kfs, th=th)
    stored = read_json_list(paths.marks_json(run_id), Mark)
    assert {m.role for m in stored} == {"pickup", "return"} and len(stored) == len(out) + len(ret_out)
    assert not (runs_dir / "ledger.jsonl").exists()  # the fake made no model call


def seed_run(run_id: str, *, return_shift_u: float = 0.0, with_masks: bool = True) -> dict[str, Any]:
    paths.ensure_run_layout(run_id)
    kfs = [write_crop(run_id, kf(f"{role}_kf1", role, handle=[200.0, 150.0]), with_mask=with_masks) for role in ROLES]
    kfs.append(write_crop(run_id, kf("pickup_hood_kf1", "pickup", PanelId.HOOD), with_mask=with_masks))
    pickup = [mark(f"p{i}", "pickup", u=u, v=v, kf_id="pickup_kf1") for i, (u, v) in enumerate([(0.0, 0.0), (1.0, 0.5), (-1.2, -0.8)])]
    ret = [mark(f"r{i}", "return", u=u + return_shift_u, v=v, kf_id="return_kf1") for i, (u, v) in enumerate([(0.0, 0.0), (1.0, 0.5)])]
    ret.append(mark("r9", "return", u=1.5, v=-1.2, kf_id="return_kf1", kind=MarkKind.DENT))
    hood = [mark("h0", "pickup", u=0.0, v=0.0, kf_id="pickup_hood_kf1", panel_id=PanelId.HOOD)]
    panels = [panel(), panel(PanelId.HOOD, states=(CoverageState.FILMED, CoverageState.NOT_FILMED))]
    write_json(paths.panels_json(run_id), panels)
    write_json(paths.keyframes_json(run_id), kfs)
    write_json(paths.marks_json(run_id), [*pickup, *ret, *hood])
    return {"pickup": pickup, "return": ret}


def test_derive_match_end_to_end_is_deterministic_and_the_second_look_only_lowers(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNMATCH000000000000"
    seed_run(run_id)
    matches, verdicts, cal = match.derive_match(run_id, th=th)
    assert cal.tau_pre == 0.25 and paths.calibration_json(run_id).is_file()
    by_panel = {pm.panel_id: pm for pm in matches}
    door = by_panel[DOOR]
    assert door.state == MatchState.COMPARABLE and door.mask_iou == pytest.approx(1.0) and door.handle_residual == pytest.approx([0.0, 0.0])
    assert {(p.pickup_mark, p.return_mark) for p in door.pairs} == {("p0", "r0"), ("p1", "r1")}
    assert door.unmatched_pickup == ["p2"] and door.unmatched_return == ["r9"]
    hood = by_panel[PanelId.HOOD]
    assert hood.state == MatchState.NOT_COMPARABLE and hood.reasons == ["G1"] and hood.unmatched_pickup == ["h0"]
    labels = sorted((v.panel_id.value, v.label.value, v.rule_id) for v in verdicts)
    assert labels == sorted([("left_front_door", "already_there", "L3"), ("left_front_door", "already_there", "L3"), ("left_front_door", "not_seen_at_return", "L1"), ("left_front_door", "new_or_unseen", "L2"), ("hood", "not_comparable", "L6")])
    assert all(v.trace.config_hash == cal_hash for v, cal_hash in zip(verdicts, [verdicts[0].trace.config_hash] * len(verdicts))) and verdicts[0].trace.config_hash is not None
    first = (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes())
    match.derive_match(run_id, th=th)
    assert (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes()) == first
    asked: list[MatchCandidate] = []

    def look(pair: MatchCandidate) -> LookVerdict:
        asked.append(pair)
        return LookVerdict.DIFFERENT if pair.pickup_mark == "p1" else LookVerdict.SAME

    _, verdicts2, _ = match.derive_match(run_id, th=th, second_look=look)
    assert sorted(p.pickup_mark for p in asked) == ["p0", "p1"]
    by_marks = {tuple(v.mark_ids): v for v in verdicts2}
    assert by_marks[("p0", "r0")].label == VerdictLabel.ALREADY_THERE
    assert by_marks[("p1", "r1")].label == VerdictLabel.NEEDS_YOUR_EYE and by_marks[("p1", "r1")].trace.rule_ids == ["L4", "L3", "K7"]


def test_derive_match_without_masks_gates_on_g4_and_a_shift_lands_in_the_band(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNMATCH000000000001"
    seed_run(run_id, with_masks=False)
    matches, verdicts, _ = match.derive_match(run_id, th=th)
    door = next(pm for pm in matches if pm.panel_id == DOOR)
    assert door.state == MatchState.NOT_COMPARABLE and door.reasons == ["G4"] and door.mask_iou is None
    run_id = "01TESTRUNMATCH000000000002"
    seed_run(run_id, return_shift_u=0.3)
    matches, verdicts, _ = match.derive_match(run_id, th=th)
    door = next(pm for pm in matches if pm.panel_id == DOOR)
    assert door.state == MatchState.COMPARABLE and len(door.pairs) == 2
    paired = {tuple(v.mark_ids): v for v in verdicts if v.panel_id == DOOR and len(v.mark_ids) == 2}
    assert set(paired) == {("p0", "r0"), ("p1", "r1")} and all(v.distance_units == pytest.approx(0.3) for v in paired.values())
    # near the centre (r 0.15) the 0.3 shift sits in the band; at r 1.26 the band has widened to 0.45 and it is already_there
    assert paired[("p0", "r0")].label == VerdictLabel.NEEDS_YOUR_EYE and paired[("p0", "r0")].trace.rule_ids == ["L4", "K1"]
    assert paired[("p1", "r1")].label == VerdictLabel.ALREADY_THERE and paired[("p1", "r1")].trace.thresholds["tau_pre_r"] > 0.4
