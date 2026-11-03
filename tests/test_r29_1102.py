"""R29 (lead decisions of 11/2 night on hero_focus5b): a one-walk mark found in both crops on a strong fit is already
there by code (matching.geometric_already_there, L8); a panel one walk only glimpsed compares when both walks have a
keyframe, with K21 on a one-walk mark whose spot is off the other view (matching.glimpsed_compare); wheels re-filed
front or rear by their order once the part check settles their side (panels.wheel_order_after_side). No network."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from walkaround import contracts as c
from walkaround import match, panels, register, rules
from walkaround.contracts import CoverageState, LookVerdict, MarkKind, PanelId, VerdictLabel

from .test_match import DOOR, kf, mark, panel


def cal(th: dict) -> c.Calibration:
    return match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=th)


@pytest.fixture
def th() -> dict[str, Any]:
    return c.load_thresholds()


def _off(th: dict, *leaves: str) -> dict:
    out = copy.deepcopy(th)
    for leaf in leaves:
        out["matching"].pop(leaf, None)
    return out


class _Al:
    def __init__(self, *, ok: bool = True, model: str = "features", n_inliers: int = 30, spread: float = 0.4, reproj_px: float = 1.0, mask_iou: float = 0.95) -> None:
        self.ok, self.model, self.n_inliers, self.spread, self.reproj_px, self.mask_iou = ok, model, n_inliers, spread, reproj_px, mask_iou
        self.H = np.eye(3)
        self.H_alt = None
        self.kf_a, self.kf_b = "pickup_kf1", "return_kf1"

    def summary(self) -> dict[str, Any]:
        return {"model": self.model, "ok": self.ok, "why": None, "n_matches": self.n_inliers, "n_inliers": self.n_inliers, "spread": self.spread, "mask_iou": self.mask_iou, "reproj_px": self.reproj_px}


class _Pr:
    def __init__(self, result: str = register.PRESENT, u: float = 2.0) -> None:
        self.result, self.why, self.uncertainty_px = result, None, u
        self.mapped_box = (100.0, 100.0, 120.0, 110.0)
        self.found_box = (102.0, 101.0, 122.0, 111.0)
        self.ncc, self.contrast_a, self.contrast_b = 0.85, 6.0, 5.0


# strong_fit ----------------------------------------------------------------------------------------------------------


def test_strong_fit_features_and_outline(th: dict) -> None:
    cfg = match.geometric_already_there(th)
    assert cfg is not None
    assert match.strong_fit(_Al(), _Pr(), cfg)
    assert not match.strong_fit(_Al(n_inliers=10), _Pr(), cfg)
    assert not match.strong_fit(_Al(spread=0.2), _Pr(), cfg)
    assert not match.strong_fit(_Al(reproj_px=2.5), _Pr(), cfg)
    assert not match.strong_fit(_Al(mask_iou=0.86), _Pr(), cfg)
    assert match.strong_fit(_Al(model="outline", n_inliers=1, spread=0.0, reproj_px=3.4, mask_iou=0.966), _Pr(u=3.06), cfg)  # hero_focus5b mirrors
    assert not match.strong_fit(_Al(model="outline", mask_iou=0.93), _Pr(), cfg)
    assert not match.strong_fit(_Al(), _Pr(u=6.0), cfg)
    assert not match.strong_fit(_Al(ok=False), _Pr(), cfg)
    assert not match.strong_fit(None, _Pr(), cfg)


def test_geometric_leaf_off_or_absent(th: dict) -> None:
    assert match.geometric_already_there(_off(th, "geometric_already_there")) is None
    off = copy.deepcopy(th)
    off["matching"]["geometric_already_there"] = {"enabled": False}
    assert match.geometric_already_there(off) is None


# registration_step: already_there or capped ------------------------------------------------------------------------


class _Cache:
    def __init__(self, al: _Al) -> None:
        self.al = al
        self.cfg = dict(register.DEFAULTS)
        self.kf_by_id: dict[str, c.Keyframe] = {}

    def fit(self, a: str, b: str) -> _Al:
        return self.al

    def pixels(self, kf_id: str) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((200, 200, 3), np.uint8), np.ones((200, 200), bool)


def _step(th: dict, monkeypatch: pytest.MonkeyPatch, al: _Al, kind: MarkKind = MarkKind.SCUFF) -> c.PanelMatch:
    monkeypatch.setattr(register, "presence", lambda *a, **k: _Pr())
    monkeypatch.setattr(register, "inverse", lambda al: al)
    m = mark("return_kf1_m1", "return", u=0.1, v=0.1, kind=kind).model_copy(update={"bbox_crop": [100.0, 100.0, 120.0, 110.0]})
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], unmatched_return=[m.mark_id])
    return match.registration_step(pm, {"pickup": [], "return": [m]}, _Cache(al), cal(th), th)  # type: ignore[arg-type]


def test_present_on_strong_fit_is_already_there(th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    pm = _step(th, monkeypatch, _Al())
    assert [r.used for r in pm.registration_marks] == ["already_there"]
    assert "so already there (L8)" in (match.registration_words(pm) or "")


def test_present_on_weak_fit_or_leaf_off_stays_capped(th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    assert [r.used for r in _step(th, monkeypatch, _Al(n_inliers=8)).registration_marks] == ["capped"]
    assert [r.used for r in _step(_off(th, "geometric_already_there"), monkeypatch, _Al()).registration_marks] == ["capped"]


def test_dirt_seen_twice_stays_capped_for_the_debris_line(th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    assert [r.used for r in _step(th, monkeypatch, _Al(), kind=MarkKind.DIRT).registration_marks] == ["capped"]


# verdicts_for: L8, the ignored caps, the look ------------------------------------------------------------------------


def _geo_pm(mid: str, role: str = "return") -> c.PanelMatch:
    rec = c.RegistrationMark(mark_id=mid, other_kf="pickup_kf1" if role == "return" else "return_kf1", result="present", found_box=[50.0, 50.0, 56.0, 54.0], ncc=0.85, uncertainty_px=3.0, used="already_there")
    unmatched = {"unmatched_return": [mid]} if role == "return" else {"unmatched_pickup": [mid]}
    return c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], registration_marks=[rec], **unmatched)


def _verdicts(th: dict, pm: c.PanelMatch, ms: list[c.Mark], counters: dict | None = None, flagged: bool = False) -> list[c.Verdict]:
    cond = c.Conditions(dirty=flagged)
    kfs = {k.kf_id: k for k in (kf("pickup_kf1", "pickup", conditions=cond), kf("return_kf1", "return", conditions=cond))}
    return match.verdicts_for(pm, {m.mark_id: m for m in ms}, {}, {}, kfs, cal(th), c.Trace(), th, counters or {})


def test_geometric_mark_is_already_there_l8_with_box_doubts_ignored(th: dict) -> None:
    m = mark("return_kf1_m3", "return", u=0.2, v=0.1, kind=MarkKind.UNSURE, confidence=c.Confidence.LOW, caps=("K3", "K4", "K12"))
    (v,) = _verdicts(th, _geo_pm(m.mark_id), [m], flagged=True)
    assert v.label == VerdictLabel.ALREADY_THERE
    assert v.trace.rule_ids == ["L8"]
    assert v.trace.thresholds["registered_present"] is True


def test_geometric_mark_keeps_other_caps(th: dict) -> None:
    m = mark("return_kf1_m3", "return", u=0.2, v=0.1, caps=("K8",))
    (v,) = _verdicts(th, _geo_pm(m.mark_id), [m])
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE
    assert v.trace.rule_ids[:2] == ["L4", "L8"] and "K8" in v.trace.rule_ids


@pytest.mark.parametrize(
    ("answer", "label"),
    [
        (LookVerdict.SAME, VerdictLabel.ALREADY_THERE),
        (LookVerdict.DIFFERENT, VerdictLabel.NEEDS_YOUR_EYE),
        (LookVerdict.UNSURE, VerdictLabel.NEEDS_YOUR_EYE),
        (None, VerdictLabel.ALREADY_THERE),
        (match.CounterAnswer(mark="no", spot=LookVerdict.SAME), VerdictLabel.ALREADY_THERE),
        (match.CounterAnswer(mark="yes", spot=LookVerdict.DIFFERENT), VerdictLabel.NEEDS_YOUR_EYE),
    ],
)
def test_the_look_may_lower_a_geometric_mark(th: dict, answer: Any, label: VerdictLabel) -> None:
    m = mark("pickup_kf1_m1", "pickup", u=0.2, v=0.1)
    (v,) = _verdicts(th, _geo_pm(m.mark_id, "pickup"), [m], counters={m.mark_id: answer})
    assert v.label == label
    assert ("K7" in v.trace.rule_ids) == (label == VerdictLabel.NEEDS_YOUR_EYE)
    assert v.counter_note


def test_geometric_asks_outline_the_found_spot(th: dict) -> None:
    m = mark("return_kf1_m3", "return", u=0.2, v=0.1)
    kfs = {k.kf_id: k for k in (kf("pickup_kf1", "pickup"), kf("return_kf1", "return"))}
    (ask,) = match.geometric_asks(_geo_pm(m.mark_id), {m.mark_id: m}, kfs)
    assert ask.other_kf == "pickup_kf1" and not ask.confirm_mark
    x1, y1, x2, y2 = ask.spot
    assert x2 - x1 >= 24.0 - 1e-6 and y2 - y1 >= 24.0 - 1e-6
    assert x1 <= 50.0 and x2 >= 56.0


def test_rules_and_contract_round_trip() -> None:
    assert rules.rule("L8").label.startswith("already there")
    assert rules.rule("K21").label
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, spots=[c.SpotCheck(mark_id="m", other_kf="k", via="panel_frame", point=(1.0, 2.0), inside=False)])
    data = json.loads(pm.model_dump_json())
    assert data["spots"][0]["inside"] is False
    assert c.PanelMatch.model_validate(data) == pm
    assert "spots" not in json.loads(c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE).model_dump_json())
    rec = c.RegistrationMark(mark_id="m", other_kf="k", result="present", used="already_there")
    assert c.RegistrationMark.model_validate(json.loads(rec.model_dump_json())).used == "already_there"


# glimpsed panels ---------------------------------------------------------------------------------------------------


def _gates(th: dict, states: tuple[CoverageState, CoverageState], with_return: bool = True) -> list[str]:
    p = panel(states=states)
    return match.panel_gates(p, kf("pickup_kf1", "pickup"), kf("return_kf1", "return") if with_return else None, mask_iou=0.95, residual=None, lane_status={}, marks_truncated_all={}, calibration=cal(th), th=th)


def test_g1_needs_only_a_keyframe_in_each_walk(th: dict) -> None:
    glimpsed = (CoverageState.FILMED, CoverageState.GLIMPSED)
    assert "G1" not in _gates(th, glimpsed)
    assert "G1" in _gates(_off(th, "glimpsed_compare"), glimpsed)
    assert "G1" in _gates(th, glimpsed, with_return=False)
    assert "G1" not in _gates(_off(th, "glimpsed_compare"), (CoverageState.FILMED, CoverageState.FILMED))


def test_spot_checks_by_the_panel_frame(th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    kp, kr = kf("pickup_kf1", "pickup", mask_png="m.png"), kf("return_kf1", "return", mask_png="m.png")
    on = mark("pickup_kf1_m1", "pickup", u=0.0, v=0.0)
    off = mark("pickup_kf1_m2", "pickup", u=0.9, v=0.0)
    cx, _ = c_geometry_point(kr, 0.0, 0.0)
    mask = np.zeros((kr.crop.h, kr.crop.w), bool)  # type: ignore[union-attr]
    mask[:, : int(cx) + 5] = True  # the left half of the other view: the centre is on it, a spot far to the side is not
    monkeypatch.setattr(match, "_load_mask", lambda run_id, k: mask)
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], unmatched_pickup=[on.mark_id, off.mark_id])
    checks = match.spot_checks(pm, {"pickup": [on, off], "return": []}, {"pickup_kf1": kp, "return_kf1": kr}, run_id="x", cache=None, masks={})
    got = {s.mark_id: (s.via, s.inside) for s in checks}
    assert got[on.mark_id] == ("panel_frame", True)
    assert got[off.mark_id][1] is False


def c_geometry_point(k: c.Keyframe, u: float, v: float) -> tuple[float, float]:
    from walkaround import geometry

    return geometry.unwhiten((u, v), k.moments, k.front_sign)  # type: ignore[arg-type]


def test_k21_caps_a_one_walk_mark_off_the_other_view(th: dict) -> None:
    m = mark("return_kf1_m1", "return", u=0.1, v=0.1)
    base = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], unmatched_return=[m.mark_id])
    (v,) = _verdicts(th, base.model_copy(update={"spots": [c.SpotCheck(mark_id=m.mark_id, other_kf="pickup_kf1", via="panel_frame", inside=False)]}), [m])
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and "K21" in v.trace.rule_ids
    (v,) = _verdicts(th, base.model_copy(update={"spots": [c.SpotCheck(mark_id=m.mark_id, other_kf="pickup_kf1", via="panel_frame", inside=True)]}), [m])
    assert v.label == VerdictLabel.NEW_OR_UNSEEN
    assert "one walk only glimpsed" in (match.spot_words(base.model_copy(update={"spots": [c.SpotCheck(mark_id="a", other_kf="k", via="none")]})) or "")


# wheels re-filed by their order -------------------------------------------------------------------------------------


def _wheel(tid: str, side: c.Side, position: c.Position, xs: dict[int, float]) -> c.PanelTrack:
    rows = [c.FrameObs(frame=f, pts=f / 15.0, box=[x - 40.0, 300.0, x + 40.0, 380.0], box_area_frac=0.01, in_car_box=True) for f, x in sorted(xs.items())]
    pid = panels._lateral_panel_id(c.Noun.WHEEL, side, position)
    return c.PanelTrack(track_id=tid, role="pickup", noun=c.Noun.WHEEL, panel_id=pid, side=side, position=position, per_frame=rows, sam_object_ids=["1"], fragments=[c.TrackFragment(object_id="1", first_frame=rows[0].frame, last_frame=rows[-1].frame)])


def test_two_left_wheels_seen_together_are_front_and_rear(th: dict) -> None:
    front = _wheel("pickup_wheel_01", c.Side.LEFT, c.Position.FRONT, {f: 100.0 + f for f in range(20)})
    rear = _wheel("pickup_wheel_00", c.Side.LEFT, c.Position.FRONT, {f: 800.0 + f for f in range(20)})
    tracks = [front, rear]
    panels.apply_wheel_order(tracks, th)
    assert (front.panel_id, rear.panel_id) == (PanelId.LEFT_FRONT_WHEEL, PanelId.LEFT_REAR_WHEEL)
    assert rear.position is c.Position.REAR


def test_right_side_front_is_image_right_and_leaf_off_changes_nothing(th: dict) -> None:
    left_in_image = _wheel("w0", c.Side.RIGHT, c.Position.REAR, {f: 100.0 for f in range(20)})
    right_in_image = _wheel("w1", c.Side.RIGHT, c.Position.REAR, {f: 700.0 for f in range(20)})
    off = copy.deepcopy(th)
    off["panels"].pop("wheel_order_after_side", None)
    panels.apply_wheel_order([left_in_image, right_in_image], off)
    assert right_in_image.panel_id == PanelId.RIGHT_REAR_WHEEL
    panels.apply_wheel_order([left_in_image, right_in_image], th)
    assert (left_in_image.panel_id, right_in_image.panel_id) == (PanelId.RIGHT_REAR_WHEEL, PanelId.RIGHT_FRONT_WHEEL)


def test_wheels_sharing_too_few_frames_keep_their_slot(th: dict) -> None:
    a = _wheel("w0", c.Side.LEFT, c.Position.FRONT, {f: 100.0 for f in range(3)})
    b = _wheel("w1", c.Side.LEFT, c.Position.FRONT, {f: 800.0 for f in range(3)})
    panels.apply_wheel_order([a, b], th)
    assert a.panel_id == b.panel_id == PanelId.LEFT_FRONT_WHEEL


STORED = Path(__file__).resolve().parents[1] / "runs" / "hero_focus5" / "derived" / "tracks.json"


@pytest.mark.skipif(not STORED.is_file(), reason="needs the stored hero_focus5 run")
def test_hero_focus5_pickup_rear_wheel_is_found(th: dict) -> None:
    tracks = [c.PanelTrack.model_validate(x) for x in json.loads(STORED.read_text()) if x["role"] == "pickup"]
    panels.apply_wheel_order(tracks, th)
    got = {t.track_id: t.panel_id for t in tracks if t.noun is c.Noun.WHEEL}
    assert got["pickup_wheel_01"] == PanelId.LEFT_FRONT_WHEEL
    assert got["pickup_wheel_00"] == got["pickup_wheel_02"] == PanelId.LEFT_REAR_WHEEL
    assert got["pickup_wheel_03"] == PanelId.RIGHT_FRONT_WHEEL and got["pickup_wheel_04"] == PanelId.RIGHT_REAR_WHEEL


# wheels: an edge fit only ---------------------------------------------------------------------------------------------


def _wheel_step(th: dict, monkeypatch: pytest.MonkeyPatch, al: _Al, result: str = register.PRESENT) -> c.PanelMatch:
    monkeypatch.setattr(register, "presence", lambda *a, **k: _Pr(result))
    monkeypatch.setattr(register, "inverse", lambda al: al)
    wheel = PanelId.RIGHT_REAR_WHEEL
    m = mark("return_kf3_m2", "return", u=0.3, v=0.1, kind=MarkKind.SCUFF, panel_id=wheel, kf_id="return_kf3").model_copy(update={"bbox_crop": [261.0, 374.0, 325.0, 410.0]})
    pm = c.PanelMatch(panel_id=wheel, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], unmatched_return=[m.mark_id])
    cfg = match.geometric_already_there(th)
    assert cfg is not None and cfg["wheels"]
    return match.wheel_geometric_step(pm, {"pickup": [], "return": [m]}, _Cache(al), cfg)  # type: ignore[arg-type]


def test_a_wheel_mark_found_on_an_edge_fit_is_already_there(th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    pm = _wheel_step(th, monkeypatch, _Al(n_inliers=110, spread=0.48, reproj_px=0.89, mask_iou=0.98))
    assert [r.used for r in pm.registration_marks] == ["already_there"] and pm.registration[0].model == "features"


def test_a_wheel_outline_fit_or_no_patch_changes_nothing(th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    assert [r.used for r in _wheel_step(th, monkeypatch, _Al(model="outline", mask_iou=0.99)).registration_marks] == ["none"]
    assert [r.used for r in _wheel_step(th, monkeypatch, _Al(), result=register.ABSENT).registration_marks] == ["none"]
    assert [r.used for r in _wheel_step(th, monkeypatch, _Al(), result=register.CANNOT_TELL).registration_marks] == ["none"]


def test_glossary_names_l8_and_k21_only_with_their_leaves(th: dict) -> None:
    from walkaround import sheet

    on = sheet.glossary_md(th)
    assert sheet.GLOSSARY_GEOMETRIC in on and sheet.GLOSSARY_GLIMPSED in on
    off = sheet.glossary_md(_off(th, "geometric_already_there", "glimpsed_compare"))
    assert "(L8)" not in off and "(K21)" not in off
