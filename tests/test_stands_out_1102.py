"""K20 (thresholds matching.stands_out, 11/2): a mark seen in one walk only must stand out from its own surroundings
in plain pixels to be called new or not seen at return (register.own_contrast, match.stands_out_step); a box that does
not caps at needs your eye. A box over max_panel_share of its panel is an area and is not tested. No network."""

from __future__ import annotations

import copy
import json
from typing import Any

import cv2
import numpy as np
import pytest

from walkaround import contracts as c
from walkaround import match, paths, register, rules, sheet
from walkaround.contracts import MarkKind, VerdictLabel

from .test_match import DOOR, kf, mark, th  # noqa: F401  (fixtures)

H, W = 400, 600


def _paint(seed: int = 0) -> np.ndarray:
    """A light grey panel with a soft shading ramp and a little sensor noise."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, 30, W, dtype=np.float32)[None, :, None]
    img = 170.0 + ramp + rng.normal(0, 1.5, (H, W, 3)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _strip(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = img.copy()
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = (200, 120, 40)  # BGR blue tape
    return out


CFG = {**register.DEFAULTS}
TAPE = (200, 150, 240, 162)
CLEAN = (400, 250, 436, 272)


# register.own_contrast -------------------------------------------------------------------------------------------------


def test_tape_stands_out_and_clean_paint_does_not() -> None:
    img = _strip(_paint(), TAPE)
    tape = register.own_contrast(img, tuple(map(float, TAPE)), CFG)
    clean = register.own_contrast(img, tuple(map(float, CLEAN)), CFG)
    assert tape is not None and tape > 10.0
    assert clean is not None and clean < float(match.STANDS_OUT_DEFAULTS["min_contrast"])


def test_own_contrast_is_the_presence_tests_contrast_away_from_the_edge() -> None:
    img = _strip(_paint(), TAPE)
    box = tuple(map(float, TAPE))
    ring = float(CFG["ring_px"])
    wa = register._window(img.shape, box, ring + 1.0)
    assert wa is not None
    want = register.saliency(register._residual_at(img, wa, register.box_sigma(box, CFG)), register._shift(box, wa[0], wa[1]), ring)
    assert register.own_contrast(img, box, CFG) == want


def test_a_box_at_the_crop_edge_is_measured_on_the_clipped_ring_and_a_sliver_is_not() -> None:
    img = _strip(_paint(), (0, 100, 30, 112))
    edge = register.own_contrast(img, (0.0, 100.0, 30.0, 112.0), CFG)
    assert edge is not None and edge > 10.0
    assert register.own_contrast(img, (0.0, 380.0, 1.0, 400.0), CFG) is None


# match.stands_out_step ----------------------------------------------------------------------------------------------------


class _Pixels:
    def __init__(self, px: dict[str, tuple[np.ndarray, np.ndarray] | None]) -> None:
        self.px = px

    def pixels(self, kf_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        return self.px.get(kf_id)


def _boxed(mid: str, role: str, box: tuple[int, int, int, int], kf_id: str) -> c.Mark:
    m = mark(mid, role, u=0.0, v=0.0, kind=MarkKind.SCUFF)
    return m.model_copy(update={"kf_id": kf_id, "bbox_crop": [float(v) for v in box], "point_crop": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]})


def _world() -> tuple[c.PanelMatch, dict[str, c.Mark], dict[str, c.Keyframe], _Pixels]:
    kfs = {k.kf_id: k for k in (kf("pickup_kf1", "pickup"), kf("return_kf1", "return"), kf("return_kf2", "return"))}
    mask = np.ones((H, W), dtype=bool)
    pickup_img = _paint(1)
    return_img = _strip(_paint(2), TAPE)
    ms = {
        "p_stray": _boxed("p_stray", "pickup", CLEAN, "pickup_kf1"),  # Spark's box on clean paint
        "r_tape": _boxed("r_tape", "return", TAPE, "return_kf1"),  # a real new strip
        "r_crush": _boxed("r_crush", "return", (20, 20, 560, 380), "return_kf1"),  # most of the panel
        "r_blind": _boxed("r_blind", "return", CLEAN, "return_kf2"),  # its crop cannot be read
    }
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], unmatched_pickup=["p_stray"], unmatched_return=["r_tape", "r_crush", "r_blind"])
    px = _Pixels({"pickup_kf1": (pickup_img, mask), "return_kf1": (return_img, mask), "return_kf2": None})
    return pm, ms, kfs, px


def _on(th: dict) -> dict:
    out = copy.deepcopy(th)
    out["matching"]["stands_out"] = {"enabled": True}
    return out


def test_the_step_sorts_each_one_walk_mark(th: dict) -> None:
    pm, ms, _, px = _world()
    got = {r.mark_id: r for r in match.stands_out_step(pm, ms, px, match.stands_out_config(_on(th))).own_contrast}  # type: ignore[arg-type]
    assert got["p_stray"].result == "faint" and got["p_stray"].contrast is not None and got["p_stray"].contrast < 3.0
    assert got["r_tape"].result == "stands_out" and got["r_tape"].contrast is not None and got["r_tape"].contrast > 10.0
    assert got["r_crush"].result == "area" and got["r_crush"].contrast is None and got["r_crush"].panel_share is not None and got["r_crush"].panel_share > 0.1
    assert got["r_blind"].result == "not_measured"


def test_the_step_skips_a_panel_not_compared_and_pairs_inside_tau_eye(th: dict) -> None:
    pm, ms, _, px = _world()
    cfg = match.stands_out_config(_on(th))
    shut = pm.model_copy(update={"state": c.MatchState.NOT_COMPARABLE})
    assert match.stands_out_step(shut, ms, px, cfg).own_contrast == []  # type: ignore[arg-type]
    pair = c.MatchCandidate(pickup_mark="p_stray", return_mark="r_tape", d=0.05, d_pos=0.05, type_cost=0.0, size_cost=0.0, tau_pre_r=0.2, tau_eye_r=0.4, r_mean=0.0, mutual=True)
    paired = pm.model_copy(update={"pairs": [pair], "unmatched_pickup": [], "unmatched_return": []})
    assert match.stands_out_step(paired, ms, px, cfg).own_contrast == []  # type: ignore[arg-type]
    far = paired.model_copy(update={"pairs": [pair.model_copy(update={"d": 0.9})]})
    assert [r.mark_id for r in match.stands_out_step(far, ms, px, cfg).own_contrast] == ["p_stray", "r_tape"]  # type: ignore[arg-type]


def test_k20_caps_a_faint_box_and_keeps_the_rest_only_with_the_leaf(th: dict) -> None:
    pm, ms, kfs, px = _world()
    on = _on(th)
    pm = match.stands_out_step(pm, ms, px, match.stands_out_config(on))  # type: ignore[arg-type]
    cal = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=th)
    by = {tuple(v.mark_ids): v for v in match.verdicts_for(pm, ms, {}, {}, kfs, cal, c.Trace(), on)}
    stray = by[("p_stray",)]
    assert stray.label == VerdictLabel.NEEDS_YOUR_EYE and stray.trace.rule_ids == ["L4", "L1", "K20"]
    assert stray.trace.thresholds["own_contrast_result"] == "faint" and stray.trace.thresholds["min_contrast"] == 3.0
    assert any("does not stand out" in r for r in stray.reasons)
    assert by[("r_tape",)].label == VerdictLabel.NEW_OR_UNSEEN and by[("r_tape",)].trace.thresholds["own_contrast"] > 10.0
    assert by[("r_crush",)].label == VerdictLabel.NEW_OR_UNSEEN  # an area is not tested
    assert by[("r_blind",)].label == VerdictLabel.NEW_OR_UNSEEN  # not measured is not capped
    off = {tuple(v.mark_ids): v for v in match.verdicts_for(pm, ms, {}, {}, kfs, cal, c.Trace(), th)}
    assert off[("p_stray",)].label == VerdictLabel.NOT_SEEN_AT_RETURN and "own_contrast" not in off[("p_stray",)].trace.thresholds
    assert "K20" in (match.how_checked(pm) or "") and "p_stray" in (match.how_checked(pm) or "")


# Settings, rules, sheet, contracts ------------------------------------------------------------------------------------------------


def test_the_leaf_is_off_when_absent_and_on_in_config(th: dict) -> None:
    assert match.stands_out_config(th) is None
    assert match.stands_out_config(_on(th) | {}) is not None
    new = c.load_thresholds()
    cfg = match.stands_out_config(new)
    assert cfg is not None and cfg["min_contrast"] == 3.0 and cfg["max_panel_share"] == 0.1 and cfg["ring_px"] == register.DEFAULTS["ring_px"]
    off = copy.deepcopy(new)
    off["matching"]["stands_out"]["enabled"] = False
    assert match.stands_out_config(off) is None


def test_k20_is_in_the_rules_table_and_the_glossary_only_with_the_leaf(th: dict) -> None:
    assert "K20" in rules.cap_ids() and rules.rule("K20").label == "the box does not stand out"
    assert all(t.isascii() and "--" not in t for t in (rules.rule("K20").text, sheet.GLOSSARY_STANDS_OUT))
    assert sheet.GLOSSARY_STANDS_OUT in sheet.glossary_md(_on(th)) and sheet.GLOSSARY_STANDS_OUT not in sheet.glossary_md(th)


def test_own_contrast_is_left_out_of_matches_json_when_empty() -> None:
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE)
    assert "own_contrast" not in json.loads(c.json_bytes(pm))
    rec = c.OwnContrast(mark_id="m", kf_id="k", result="faint", contrast=0.5)
    assert json.loads(c.json_bytes(pm.model_copy(update={"own_contrast": [rec]})))["own_contrast"][0]["result"] == "faint"


# The hero_focus2 pin (skipped where the run is not on disk) ----------------------------------------------------------------


def _run_marks(run_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    root = paths.run_dir(run_id)
    if not (root / "derived" / "marks.json").is_file():
        return None
    kfs = {k["kf_id"]: k for k in json.loads((root / "derived" / "keyframes.json").read_text())}
    ms = {m["mark_id"]: m for m in json.loads((root / "derived" / "marks.json").read_text())}
    return kfs, ms


def test_hero_focus2_stray_bumper_box_is_faint_and_every_tape_strip_stands_out() -> None:
    got = _run_marks("hero_focus2")
    if got is None:
        pytest.skip("runs/hero_focus2 is not on disk")
    kfs, ms = got
    cfg = match.stands_out_config(c.load_thresholds())

    def contrast(mid: str) -> float | None:
        m = ms[mid]
        img = cv2.imread(str(paths.run_dir("hero_focus2") / kfs[m["kf_id"]]["crop"]["path"]))
        return register.own_contrast(img, tuple(m["bbox_crop"]), cfg)  # type: ignore[arg-type]

    stray = contrast("pickup_rear_bumper_kf2_m2")
    assert stray is not None and stray < 3.0
    for mid in ("return_right_rear_door_kf1_m1", "pickup_left_front_door_kf1_m1", "pickup_rear_bumper_kf2_m1"):
        v = contrast(mid)
        assert v is not None and v > 10.0, mid
