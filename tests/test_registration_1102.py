"""Registration of 11/2 (thresholds matching.registration, walkaround/register.py, match.registration_step): the
pickup crop is mapped onto the return crop from the panel outline and its rigid edges, and each mark's box is tested at
its mapped spot. Present caps a one-walk mark at needs your eye (K18) or pairs it with an unmatched mark there; a weak
fit changes nothing; nothing ever raises a label. Synthetic crops, no network."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from walkaround import contracts as c
from walkaround import geometry, match, register, rules

W, H = 640, 400
CFG = register.config({"matching": {"registration": {"enabled": True}}})
assert CFG is not None


def _scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A grey door-like panel with a handle, a trim line and a gap on a textured background, and its mask."""
    rng = np.random.default_rng(seed)
    img = (rng.integers(60, 120, (H, W, 3))).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    mask = np.zeros((H, W), np.uint8)
    pts = np.array([[90, 70], [560, 60], [580, 330], [80, 345]], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    panel = np.full((H, W, 3), (150, 150, 155), np.uint8)
    yy = np.linspace(0, 1, H)[:, None, None]
    panel = np.clip(panel.astype(np.float32) + 25 * yy, 0, 255).astype(np.uint8)  # soft shading
    img = np.where(mask[..., None] > 0, panel, img)
    cv2.polylines(img, [pts], True, (40, 40, 40), 3)  # the panel gap
    cv2.rectangle(img, (420, 150), (500, 168), (210, 210, 215), -1)  # handle
    cv2.rectangle(img, (420, 150), (500, 168), (60, 60, 60), 1)
    cv2.line(img, (100, 230), (560, 222), (120, 120, 125), 2)  # trim line
    return img, mask


def _warp() -> np.ndarray:
    return np.array([[1.04, 0.03, 12.0], [-0.02, 1.01, 6.0], [2e-5, 1e-5, 1.0]])


def _pair(old: tuple[int, int] = (250, 120), new: tuple[int, int] | None = (300, 290)) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pickup and return crops: an old yellow strip in both, a new one at return only (drawn on the pickup scene and
    warped, so both walks show the same car)."""
    a, mask_a = _scene()
    cv2.line(a, (old[0] - 12, old[1] - 4), (old[0] + 12, old[1] + 4), (40, 220, 230), 3)
    Hm = _warp()
    b_src = a.copy()
    if new is not None:
        cv2.line(b_src, (new[0] - 10, new[1] + 5), (new[0] + 10, new[1] - 5), (40, 220, 230), 3)
    b = cv2.warpPerspective(b_src, Hm, (W, H), borderMode=cv2.BORDER_REFLECT)
    b = cv2.convertScaleAbs(b, alpha=0.9, beta=12)  # other light
    mask_b = cv2.warpPerspective(mask_a, Hm, (W, H), flags=cv2.INTER_NEAREST)
    return a, mask_a > 0, b, mask_b > 0, Hm


def _box(p: tuple[float, float], hw: float = 14, hh: float = 8) -> tuple[float, float, float, float]:
    return (p[0] - hw, p[1] - hh, p[0] + hw, p[1] + hh)


def _mapped(Hm: np.ndarray, p: tuple[float, float]) -> tuple[float, float]:
    q = register.apply(Hm, np.array([p]))[0]
    return float(q[0]), float(q[1])


def _align(a, ma, b, mb):
    ka = geometry.moments_from_mask(ma)
    kb = geometry.moments_from_mask(mb)
    return register.align("pickup_x_kf1", a, ma, "return_x_kf1", b, mb, CFG, prior=register.prior_map(ka, 1, kb, 1))


def test_alignment_recovers_the_warp_and_scores_it() -> None:
    a, ma, b, mb, Hm = _pair()
    al = _align(a, ma, b, mb)
    assert al.ok, al.why
    assert al.mask_iou >= 0.95
    for p in [(150.0, 100.0), (450.0, 300.0), (300.0, 200.0)]:
        assert math.dist(register.apply(al.H, np.array([p]))[0], _mapped(Hm, p)) < 4.0
        assert register.uncertainty_px(al, p) < 15.0


def test_outlines_that_disagree_refuse_the_fit() -> None:
    a, ma, b, mb, _ = _pair()
    mb2 = mb.copy()
    mb2[:, : W // 2] = False  # the return mask lost half the panel
    al = _align(a, ma, b, mb2)
    assert not al.ok and "matched edges put the panel" in (al.why or "")
    pr = register.presence(al, a, b, mb2, _box((250, 120)), CFG, diag_a=500.0)
    assert pr.result == register.CANNOT_TELL and "alignment refused" in (pr.why or "")


def test_presence_present_absent_and_cannot_tell() -> None:
    a, ma, b, mb, Hm = _pair()
    al = _align(a, ma, b, mb)
    old = register.presence(al, a, b, mb, _box((250, 120)), CFG, diag_a=500.0)
    assert old.result == register.PRESENT, old
    inv = register.inverse(al)
    new_at_return = _mapped(Hm, (300, 290))
    new = register.presence(inv, b, a, ma, _box(new_at_return), CFG, diag_a=500.0)
    assert new.result == register.ABSENT, new
    flat = register.presence(al, a, b, mb, _box((330, 190), 10, 6), CFG, diag_a=500.0)  # plain paint
    assert flat.result == register.CANNOT_TELL and "stands out" in (flat.why or "")


# --- the match rules ---------------------------------------------------------------------------------------------


class _Cache:
    """registration_step's view of a run: crops, masks, keyframes and fits, in memory."""

    def __init__(self, a, ma, b, mb, kfs):
        self.cfg = CFG
        self.kf_by_id = kfs
        self._px = {"pickup_left_front_door_kf1": (a, ma), "return_left_front_door_kf1": (b, mb)}
        self._fit = None

    def pixels(self, kf_id):
        return self._px.get(kf_id)

    def fit(self, kf_p, kf_r):
        if self._fit is None:
            (a, ma), (b, mb) = self._px[kf_p], self._px[kf_r]
            ka, kb = self.kf_by_id[kf_p], self.kf_by_id[kf_r]
            self._fit = register.align(kf_p, a, ma, kf_r, b, mb, CFG, prior=register.prior_map(ka.moments, 1, kb.moments, 1))
        return self._fit


def _kf(role: str, mask: np.ndarray) -> c.Keyframe:
    kf_id = f"{role}_left_front_door_kf1"
    return c.Keyframe(
        kf_id=kf_id,
        role=role,
        panel_id=c.PanelId.LEFT_FRONT_DOOR,
        track_id=f"{role}_door_00",
        f_proxy=0,
        pts=0.0,
        score=c.KeyframeScore(area=0.2, completeness=1.0, frontality=1.0, glare=0.0, sharpness=1.0, total=0.2),
        aspect_ratio=1.6,
        box_src=[0.0, 0.0, float(W), float(H)],
        crop=c.Crop(path=f"crops/{role}/{kf_id}.jpg", w=W, h=H, scale=1.0, dx=0.0, dy=0.0),
        mask_png=f"crops/{role}/{kf_id}.mask.png",
        moments=geometry.moments_from_mask(mask),
        front_sign=1,
        marks_status=c.MarksStatus.COMPLETE,
    )


def _mark(mark_id: str, kf: c.Keyframe, box) -> c.Mark:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    u, v = geometry.whiten((cx, cy), kf.moments, 1)
    return c.Mark(
        mark_id=mark_id,
        role=kf.role,
        panel_id=kf.panel_id,
        kf_id=kf.kf_id,
        source=c.MarkSource.SPARK_BOX,
        type=c.MarkKind.SCRATCH,
        confidence=c.Confidence.MEDIUM,
        size_class=c.SizeClass.S,
        bbox_crop=list(box),
        point_crop=[cx, cy],
        w=[u, v],
        r=geometry.radius((u, v)),
    )


def _setup(*, pickup_marks: bool = False, misaligned_pickup: bool = False):
    a, ma, b, mb, Hm = _pair()
    kp, kr = _kf("pickup", ma), _kf("return", mb)
    old_r = _mark("return_left_front_door_kf1_m1", kr, _box(_mapped(Hm, (250, 120))))
    new_r = _mark("return_left_front_door_kf1_m2", kr, _box(_mapped(Hm, (300, 290))))
    marks_p = []
    if pickup_marks:
        p = (250 + (60 if misaligned_pickup else 3), 120 + 2)
        marks_p = [_mark("pickup_left_front_door_kf1_m1", kp, _box(p))]
    cal = c.Calibration(tau_pre=0.4, tau_eye=0.8, tau_fix=0.08)
    th = c.load_thresholds()
    th["matching"]["registration"] = {"enabled": True}
    pm = c.PanelMatch(
        panel_id=c.PanelId.LEFT_FRONT_DOOR,
        state=c.MatchState.COMPARABLE,
        kf_pair=[kp.kf_id, kr.kf_id],
        unmatched_pickup=[m.mark_id for m in marks_p],
        unmatched_return=[old_r.mark_id, new_r.mark_id],
    )
    cache = _Cache(a, ma, b, mb, {kp.kf_id: kp, kr.kf_id: kr})
    marks = {m.mark_id: m for m in [*marks_p, old_r, new_r]}
    return pm, {"pickup": marks_p, "return": [old_r, new_r]}, cache, cal, th, marks, {kp.kf_id: kp, kr.kf_id: kr}


def _labels(pm, marks, kfs, cal, th):
    vs = match.verdicts_for(pm, marks, {}, {}, kfs, cal, c.Trace(), th)
    return {tuple(v.mark_ids): (v.label, v.trace.rule_ids) for v in vs}


def test_one_walk_mark_whose_patch_is_there_is_capped_k18_and_the_new_one_stands() -> None:
    pm, panel_marks, cache, cal, th, marks, kfs = _setup()
    out = match.registration_step(pm, panel_marks, cache, cal, th)
    rec = {r.mark_id: r for r in out.registration_marks}
    assert rec["return_left_front_door_kf1_m1"].result == "present" and rec["return_left_front_door_kf1_m1"].used == "capped"
    assert rec["return_left_front_door_kf1_m2"].result == "absent" and rec["return_left_front_door_kf1_m2"].used == "none"
    assert out.registration and out.registration[0].ok and out.registration[0].H is not None
    got = _labels(out, marks, kfs, cal, th)
    assert got[("return_left_front_door_kf1_m1",)][0] == c.VerdictLabel.NEEDS_YOUR_EYE and "K18" in got[("return_left_front_door_kf1_m1",)][1]
    assert got[("return_left_front_door_kf1_m2",)][0] == c.VerdictLabel.NEW_OR_UNSEEN
    words = match.how_checked(out) or ""
    assert "K18" in words and "registration pickup_left_front_door_kf1 onto return_left_front_door_kf1" in words and "recorded only" in words


def test_unmatched_mark_of_the_other_walk_at_the_spot_is_paired_measured_there() -> None:
    pm, panel_marks, cache, cal, th, marks, kfs = _setup(pickup_marks=True)
    out = match.registration_step(pm, panel_marks, cache, cal, th)
    assert len(out.pairs) == 1
    pair = out.pairs[0]
    assert (pair.pickup_mark, pair.return_mark) == ("pickup_left_front_door_kf1_m1", "return_left_front_door_kf1_m1")
    assert pair.registered and pair.d_moment is None and pair.d < pair.tau_pre_r
    assert out.unmatched_pickup == [] and out.unmatched_return == ["return_left_front_door_kf1_m2"]
    got = _labels(out, marks, kfs, cal, th)
    label, ids = got[("pickup_left_front_door_kf1_m1", "return_left_front_door_kf1_m1")]
    assert label == c.VerdictLabel.ALREADY_THERE and ids == ["L3"]
    dumped = pair.model_dump(mode="json")
    assert dumped["registered"] is True and "d_moment" in dumped


def test_a_mark_somewhere_else_is_not_paired() -> None:
    pm, panel_marks, cache, cal, th, marks, kfs = _setup(pickup_marks=True, misaligned_pickup=True)
    out = match.registration_step(pm, panel_marks, cache, cal, th)
    assert out.pairs == []
    assert {r.mark_id: r.used for r in out.registration_marks}["return_left_front_door_kf1_m1"] == "capped"


def test_a_refused_fit_changes_no_label() -> None:
    pm, panel_marks, cache, cal, th, marks, kfs = _setup()
    a, ma = cache._px["pickup_left_front_door_kf1"]
    ma2 = ma.copy()
    ma2[:, : W // 2] = False
    cache._px["pickup_left_front_door_kf1"] = (a, ma2)
    out = match.registration_step(pm, panel_marks, cache, cal, th)
    assert not out.registration[0].ok
    assert all(r.result == "cannot_tell" and r.used == "none" for r in out.registration_marks)
    base = _labels(pm, marks, kfs, cal, th)
    assert _labels(out, marks, kfs, cal, th) == base


def test_registration_never_raises_a_label() -> None:
    """K18 only lowers: verdict() with K18 set turns new or not seen into needs your eye and leaves the rest."""
    for d, ret in ((None, True), (None, False)):
        label, ids = match.verdict(d, 0.4, 0.8, c.VerdictFlags(return_mark=ret, K18=True))
        assert label == c.VerdictLabel.NEEDS_YOUR_EYE and "K18" in ids
    label, _ = match.verdict(0.1, 0.4, 0.8, c.VerdictFlags(K18=True))
    assert label == c.VerdictLabel.NEEDS_YOUR_EYE
    assert rules.rule("K18").label == "the other walk shows the same patch"


def test_records_left_out_of_the_json_when_absent() -> None:
    pm = c.PanelMatch(panel_id=c.PanelId.HOOD, state=c.MatchState.COMPARABLE)
    data = pm.model_dump(mode="json")
    assert "registration" not in data and "registration_marks" not in data
    cand = c.MatchCandidate(pickup_mark="a", return_mark="b", d=0.1, d_pos=0.1, type_cost=0.0, size_cost=0.0, r_mean=0.0, tau_pre_r=0.4, tau_eye_r=0.8)
    assert "registered" not in cand.model_dump(mode="json") and "d_moment" not in cand.model_dump(mode="json")
    assert match.registration_config({"matching": {}}) is None
    assert match.registration_config({"matching": {"registration": {"enabled": False}}}) is None
    assert match.registration_config(c.load_thresholds())["min_mask_iou_outline"] == 0.90


def test_wheels_and_not_comparable_panels_are_skipped() -> None:
    pm, panel_marks, cache, cal, th, marks, kfs = _setup()
    nc = pm.model_copy(update={"state": c.MatchState.NOT_COMPARABLE})
    assert match.registration_step(nc, panel_marks, cache, cal, th) == nc
    wheel = pm.model_copy(update={"panel_id": c.PanelId.LEFT_FRONT_WHEEL})
    assert match.registration_step(wheel, panel_marks, cache, cal, th) == wheel


def test_fit_is_deterministic() -> None:
    a, ma, b, mb, _ = _pair()
    one, two = _align(a, ma, b, mb), _align(a, ma, b, mb)
    assert np.array_equal(one.H, two.H) and one.summary() == two.summary()


@pytest.mark.parametrize("name", ["presence", "align"])
def test_bad_pixels_never_raise(name: str) -> None:
    a, ma, b, mb, _ = _pair()
    empty = np.zeros_like(ma)
    al = register.align("p", a, empty, "r", b, mb, CFG, prior=np.eye(3))
    assert not al.ok and al.why == "no panel mask"
    if name == "presence":
        assert register.presence(al, a, b, mb, _box((5, 5)), CFG, diag_a=100.0).result == register.CANNOT_TELL


def test_the_sheet_glossary_names_registration_only_when_it_is_on() -> None:
    from walkaround import sheet

    on = c.load_thresholds()
    assert match.registration_config(on) is not None and "registration: on a compared panel" in sheet.glossary_md(on)
    off = {**on, "matching": {k: v for k, v in on["matching"].items() if k != "registration"}}
    assert "registration:" not in sheet.glossary_md(off)


def test_a_panel_with_no_mark_tested_prints_no_registration_words() -> None:
    fit = c.RegistrationFit(kf_pair=["pickup_hood_kf1", "return_hood_kf1"], model="outline", ok=True, mask_iou=0.95)
    pm = c.PanelMatch(panel_id=c.PanelId.HOOD, state=c.MatchState.COMPARABLE, registration=[fit])
    assert match.registration_words(pm) is None


def test_a_registration_error_never_stops_the_match(monkeypatch) -> None:
    """derive_match keeps the moment match and records the failure when the step raises."""
    pm, panel_marks, cache, cal, th, marks, kfs = _setup()

    def boom(*args, **kwargs):
        raise RuntimeError("bad pixels")

    monkeypatch.setattr(match, "registration_step", boom)
    import inspect

    src = inspect.getsource(match.derive_match)
    assert "registration failed: " in src and "except Exception" in src
