"""R28 (lead decisions of 11/2 on hero_focus5): same-walk copies merged into a decided pair (matching.same_walk_copies),
debris folded on any view (matching.debris_by_eye any_view), faint one-walk marks left off the list
(matching.faint_left_off), the mirrors' side from their own track (matching.side_from_track), and walkaround rederive.
No network."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from walkaround import __main__ as cli
from walkaround import contracts as c
from walkaround import match, paths
from walkaround.contracts import MarkKind, PanelId, VerdictLabel

from .test_match import DOOR, kf, mark
from .test_orchestrator import Fakes, make_orch, new_run, runs_dir  # noqa: F401  (fixtures)


@pytest.fixture
def th() -> dict[str, Any]:
    return c.load_thresholds()


def _off(th: dict, *leaves: str) -> dict:
    out = copy.deepcopy(th)
    for leaf in leaves:
        out["matching"].pop(leaf, None)
    return out


def _v(vid: str, label: VerdictLabel, ids: list[str], rule_ids: list[str] | None = None, panel: PanelId = DOOR) -> c.Verdict:
    return c.Verdict(verdict_id=vid, panel_id=panel, mark_ids=ids, label=label, rule_id=(rule_ids or ["L3"])[0], trace=c.Trace(rule_ids=rule_ids or ["L3"]))


def _boxed(mid: str, role: str, kf_id: str, box: tuple[float, float, float, float], *, u: float = 0.0, v: float = 0.0, kind: MarkKind = MarkKind.SCUFF) -> c.Mark:
    m = mark(mid, role, u=u, v=v, kind=kind, kf_id=kf_id)
    return m.model_copy(update={"bbox_crop": list(box), "point_crop": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]})


class _Fit:
    def __init__(self, H: np.ndarray, *, ok: bool, model: str = "outline", mask_iou: float = 0.87) -> None:
        self.H, self.ok, self.model, self.mask_iou = H, ok, model, mask_iou


class _Cache:
    """A stand-in RegistrationCache: fit(a, b) from a table, coincide numbers from register.DEFAULTS."""

    def __init__(self, fits: dict[tuple[str, str], _Fit]) -> None:
        from walkaround import register

        self.fits = fits
        self.cfg = dict(register.DEFAULTS)

    def fit(self, a: str, b: str) -> _Fit | None:
        return self.fits.get((a, b))


def _shift(dx: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _bumper() -> tuple[c.PanelMatch, list[c.Verdict], dict[str, c.Mark]]:
    """Two strips decided on pickup kf1 against return kf1, and their copies on pickup kf2 shifted 170 px left in that
    crop and 0.45 panel units away in the panel frame (the hero_focus5 rear bumper)."""
    ms = {
        "p1_a": _boxed("p1_a", "pickup", "pickup_kf1", (596, 277, 638, 297), u=0.05),
        "p1_b": _boxed("p1_b", "pickup", "pickup_kf1", (930, 261, 970, 287), u=-1.0),
        "r1_a": _boxed("r1_a", "return", "return_kf1", (606, 292, 648, 313), u=0.07),
        "r1_b": _boxed("r1_b", "return", "return_kf1", (935, 283, 976, 309), u=-0.93),
        "p2_a": _boxed("p2_a", "pickup", "pickup_kf2", (426, 277, 468, 297), u=0.5),
        "p2_b": _boxed("p2_b", "pickup", "pickup_kf2", (760, 261, 800, 287), u=-0.55),
        "p2_far": _boxed("p2_far", "pickup", "pickup_kf2", (100, 100, 120, 120), u=1.5),
    }
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf2", "return_kf1"])
    vs = [
        _v("left_front_door_v1", VerdictLabel.ALREADY_THERE, ["p1_a", "r1_a"]),
        _v("left_front_door_v2", VerdictLabel.ALREADY_THERE, ["p1_b", "r1_b"]),
        _v("left_front_door_v3", VerdictLabel.NEEDS_YOUR_EYE, ["p2_a"], ["L4", "L1", "K16"]),
        _v("left_front_door_v4", VerdictLabel.NEEDS_YOUR_EYE, ["p2_b"], ["L4", "L1", "K16"]),
        _v("left_front_door_v5", VerdictLabel.NOT_SEEN_AT_RETURN, ["p2_far"], ["L1"]),
    ]
    return pm, vs, ms


def _wh(ms: dict[str, c.Mark]) -> dict[str, c.WhitenedMark]:
    from walkaround.marks import whitened_from_marks

    return whitened_from_marks(list(ms.values()))


# 1. Same-walk copies --------------------------------------------------------------------------------------------------------


def test_a_refused_outline_fit_that_puts_two_copies_on_two_decided_marks_merges_both(th: dict) -> None:
    pm, vs, ms = _bumper()
    cache = _Cache({("pickup_kf2", "pickup_kf1"): _Fit(_shift(170.0), ok=False, mask_iou=0.87)})
    out, kept = match.copies_step(pm, vs, ms, _wh(ms), cache, th)  # type: ignore[arg-type]
    assert [v.mark_ids for v in kept] == [["p1_a", "r1_a"], ["p1_b", "r1_b"], ["p2_far"]]
    assert [v.verdict_id for v in kept] == ["left_front_door_v1", "left_front_door_v2", "left_front_door_v3"]
    got = {n.mark_id: n for n in out.copies}
    assert got["p2_a"].into_mark == "p1_a" and got["p2_a"].into_pair == ["p1_a", "r1_a"] and got["p2_a"].refused_fit and got["p2_a"].via == "registration"
    assert got["p2_b"].into_mark == "p1_b"
    words = match.how_checked(out) or ""
    assert "p2_a is p1_a seen again in pickup_kf2" in words and "no line of its own" in words and words.isascii()


def test_one_copy_on_a_refused_fit_is_not_enough_and_an_accepted_fit_is(th: dict) -> None:
    pm, vs, ms = _bumper()
    lone = [v for v in vs if v.mark_ids != ["p2_b"]]
    refused = _Cache({("pickup_kf2", "pickup_kf1"): _Fit(_shift(170.0), ok=False)})
    assert match.copies_step(pm, lone, ms, _wh(ms), refused, th)[1] == lone  # type: ignore[arg-type]
    low = _Cache({("pickup_kf2", "pickup_kf1"): _Fit(_shift(170.0), ok=False, mask_iou=0.80)})
    assert len(match.copies_step(pm, vs, ms, _wh(ms), low, th)[1]) == 5  # type: ignore[arg-type]
    accepted = _Cache({("pickup_kf2", "pickup_kf1"): _Fit(_shift(170.0), ok=True, model="features")})
    out, kept = match.copies_step(pm, lone, ms, _wh(ms), accepted, th)  # type: ignore[arg-type]
    assert [v.mark_ids for v in kept] == [["p1_a", "r1_a"], ["p1_b", "r1_b"], ["p2_far"]] and out.copies[0].fit == "features"


def test_a_copy_in_the_panel_frame_merges_and_a_mark_on_the_decided_keyframe_never_does(th: dict) -> None:
    pm, vs, ms = _bumper()
    ms["p2_a"] = ms["p2_a"].model_copy(update={"w": [0.1, 0.0], "r": 0.1})  # 0.05 units from p1_a
    ms["p1_c"] = _boxed("p1_c", "pickup", "pickup_kf1", (600, 280, 630, 295), u=0.06)  # beside p1_a on its own keyframe
    vs = [*vs, _v("left_front_door_v6", VerdictLabel.NEEDS_YOUR_EYE, ["p1_c"], ["L4", "L1"])]
    out, kept = match.copies_step(pm, vs, ms, _wh(ms), None, th)
    assert [n.mark_id for n in out.copies] == ["p2_a"] and out.copies[0].via == "panel_frame"
    assert ["p1_c"] in [v.mark_ids for v in kept] and ["p2_b"] in [v.mark_ids for v in kept]


def test_copies_need_the_leaf_a_compared_panel_and_an_already_there_pair(th: dict) -> None:
    pm, vs, ms = _bumper()
    cache = _Cache({("pickup_kf2", "pickup_kf1"): _Fit(_shift(170.0), ok=True, model="features")})
    assert match.copies_step(pm, vs, ms, _wh(ms), cache, _off(th, "same_walk_copies"))[1] == vs  # type: ignore[arg-type]
    shut = pm.model_copy(update={"state": c.MatchState.NOT_COMPARABLE})
    assert match.copies_step(shut, vs, ms, _wh(ms), cache, th)[1] == vs  # type: ignore[arg-type]
    eye = [v.model_copy(update={"label": VerdictLabel.NEEDS_YOUR_EYE}) if len(v.mark_ids) == 2 else v for v in vs]
    assert match.copies_step(pm, eye, ms, _wh(ms), cache, th)[1] == eye  # type: ignore[arg-type]


# 2. Debris on any view --------------------------------------------------------------------------------------------------


def _debris_world() -> tuple[c.PanelMatch, list[c.Verdict], dict[str, c.Mark], dict[str, c.Keyframe]]:
    kfs = {k.kf_id: k for k in (kf("pickup_kf1", "pickup"), kf("return_kf1", "return"))}  # clean views, no flag
    ms = {
        "leaf": mark("leaf", "pickup", u=0.1, v=0.1, kind=MarkKind.DIRT),
        "chip": mark("chip", "pickup", u=0.5, v=0.1, kind=MarkKind.CHIP),
        "fog": mark("fog", "return", u=0.9, v=0.1, kind=MarkKind.UNSURE),
        "pd": mark("pd", "pickup", u=-0.5, v=0.1, kind=MarkKind.DIRT),
        "rd": mark("rd", "return", u=-0.5, v=0.2, kind=MarkKind.DIRT),
    }
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"])
    vs = [
        _v("left_front_door_v1", VerdictLabel.NEEDS_YOUR_EYE, ["leaf"], ["L4", "L1", "K4"]),
        _v("left_front_door_v2", VerdictLabel.NEEDS_YOUR_EYE, ["chip"], ["L4", "L1", "K12"]),
        _v("left_front_door_v3", VerdictLabel.NEEDS_YOUR_EYE, ["fog"], ["L4", "L2", "K4"]),
        _v("left_front_door_v4", VerdictLabel.NEEDS_YOUR_EYE, ["pd", "rd"], ["L4", "K4"]),
    ]
    return pm, vs, ms, kfs


def test_one_walk_dirt_folds_on_a_clean_view_and_damage_unsure_and_pairs_stay(th: dict) -> None:
    pm, vs, ms, kfs = _debris_world()
    out, kept = match.debris_split(pm, vs, ms, kfs, th)
    assert [v.mark_ids for v in kept] == [["chip"], ["fog"], ["pd", "rd"]]
    assert out.by_eye[0].mark_ids == ["leaf"] and out.by_eye[0].any_view is True
    assert match.by_eye_words(out) == ["debris on this panel; check by eye (1 spot Muse Spark typed dirt, not listed one by one)"]
    before = copy.deepcopy(th)
    before["matching"]["debris_by_eye"].pop("any_view")
    assert match.debris_split(pm, vs, ms, kfs, before)[1] == vs  # the 11/2 rule: clean views fold nothing
    assert "any_view" not in json.loads(c.json_bytes(c.ByEyeLine(kind="debris", text="t")))


# 3. Faint marks left off the list -----------------------------------------------------------------------------------------


def test_a_faint_one_walk_line_leaves_the_list_and_how_checked_keeps_it(th: dict) -> None:
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, own_contrast=[c.OwnContrast(mark_id="r_box", kf_id="return_kf1", result="faint", contrast=0.01)])
    vs = [
        _v("left_front_door_v1", VerdictLabel.ALREADY_THERE, ["p", "r"]),
        _v("left_front_door_v2", VerdictLabel.NEEDS_YOUR_EYE, ["r_box"], ["L4", "L2", "K20"]),
        _v("left_front_door_v3", VerdictLabel.NEW_OR_UNSEEN, ["r_tape"], ["L2"]),
    ]
    out, kept = match.faint_step(pm, vs, th)
    assert [v.mark_ids for v in kept] == [["p", "r"], ["r_tape"]] and [v.verdict_id for v in kept] == ["left_front_door_v1", "left_front_door_v2"]
    assert out.left_off == ["r_box"]
    assert "r_box: its box does not stand out from the paint around it in its own crop (contrast 0.0), so it is left off the list (K20)" in (match.how_checked(out) or "")
    assert match.faint_step(pm, vs, _off(th, "faint_left_off"))[1] == vs
    assert "so needs your eye (K20)" in (match.how_checked(pm) or "")


# 4. The mirrors' side from their own track ----------------------------------------------------------------------------------


def _track(track_id: str, role: str, noun: c.Noun, frames: range, *, side: str = "none", panel: PanelId | None = None) -> c.PanelTrack:
    obs = [c.FrameObs(frame=f, pts=f / 15.0, box=[0.0, 0.0, 10.0, 10.0], c=0.9, box_area_frac=0.01, in_car_box=True) for f in frames]
    return c.PanelTrack(track_id=track_id, role=role, noun=noun, sam_object_ids=["1"], fragments=[], per_frame=obs, panel_id=panel, side=c.Side(side))


def test_a_mirror_track_takes_the_side_of_the_doors_it_shares_frames_with(th: dict) -> None:
    cfg = match.side_from_track(th)
    assert cfg is not None and cfg["panels"] == ["mirrors"]
    tracks = [
        _track("p_m0", "pickup", c.Noun.SIDE_MIRROR, range(0, 100), panel=PanelId.MIRRORS),
        _track("p_m1", "pickup", c.Noun.SIDE_MIRROR, range(300, 400), panel=PanelId.MIRRORS),
        _track("p_m2", "pickup", c.Noun.SIDE_MIRROR, range(190, 210), panel=PanelId.MIRRORS),
        _track("p_dl", "pickup", c.Noun.CAR_DOOR, range(0, 200), side="left", panel=PanelId.LEFT_FRONT_DOOR),
        _track("p_dr", "pickup", c.Noun.CAR_DOOR, range(200, 400), side="right", panel=PanelId.RIGHT_FRONT_DOOR),
        _track("p_du", "pickup", c.Noun.CAR_DOOR, range(300, 400), side="none", panel=PanelId.LEFT_REAR_DOOR),
    ]
    assert match.track_sides(tracks, cfg) == {"p_m0": "left", "p_m1": "right", "p_m2": "none"}  # p_m2: 10 and 10 frames


def _mirror_panel() -> c.Panel:
    filmed = c.PanelRole(state=c.CoverageState.FILMED)
    return c.Panel(panel_id=PanelId.MIRRORS, noun=c.Noun.SIDE_MIRROR, front_sign=None, per_role={"pickup": filmed, "return": filmed})


def test_g6_spares_a_mirror_whose_keyframes_got_a_side_and_g13_parts_two_sides(th: dict) -> None:
    cal = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=th)
    kp, kr = kf("pickup_kf1", "pickup", PanelId.MIRRORS, front_sign=1), kf("return_kf1", "return", PanelId.MIRRORS, front_sign=1)

    def gates(a: c.Keyframe, b: c.Keyframe, t: dict) -> list[str]:
        return match.panel_gates(_mirror_panel(), a, b, mask_iou=0.95, residual=None, lane_status={}, marks_truncated_all={}, calibration=cal, th=t)

    assert "G6" not in gates(kp, kr, th) and "G13" not in gates(kp, kr, th)
    assert "G6" in gates(kp, kr, _off(th, "side_from_track"))
    assert "G6" in gates(kp, kr.model_copy(update={"front_sign": None}), th)  # its track got no side
    assert "G13" in gates(kp, kr.model_copy(update={"front_sign": -1}), th)  # the left mirror against the right one


def test_assign_sides_signs_the_keyframe_and_whitens_its_marks(th: dict, runs_dir: Path) -> None:  # noqa: F811
    run_id = "r28_sides"
    paths.ensure_run_layout(run_id)
    tracks = [
        _track("pickup_side_mirror_00", "pickup", c.Noun.SIDE_MIRROR, range(300, 400), panel=PanelId.MIRRORS),
        _track("p_dr", "pickup", c.Noun.CAR_DOOR, range(200, 400), side="right", panel=PanelId.RIGHT_FRONT_DOOR),
    ]
    c.write_json(paths.tracks_json(run_id), tracks)
    k = kf("pickup_mirrors_kf1", "pickup", PanelId.MIRRORS, front_sign=None).model_copy(update={"track_id": "pickup_side_mirror_00"})
    m = _boxed("pm1", "pickup", "pickup_mirrors_kf1", (40, 40, 50, 50)).model_copy(update={"panel_id": PanelId.MIRRORS, "w": None, "r": None, "size_units": None})
    kfs, ms, sides = match.assign_sides(run_id, [k], [m], th)
    assert sides == {"pickup_mirrors_kf1": "right"} and kfs[0].front_sign == 1 and ms[0].w is not None
    assert match.assign_sides(run_id, [k], [m], _off(th, "side_from_track")) == ([k], [m], {})
    pm = c.PanelMatch(panel_id=PanelId.MIRRORS, state=c.MatchState.COMPARABLE, sides=sides)
    assert "pickup_mirrors_kf1 right" in (match.how_checked(pm) or "")


# Contracts: every new field is left out when empty -----------------------------------------------------------------------


def test_the_new_fields_are_left_out_when_empty() -> None:
    pm = json.loads(c.json_bytes(c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE)))
    assert not {"copies", "left_off", "sides"} & set(pm)
    run = json.loads(c.json_bytes(c.RunManifest(run_id="x", created_at="2026-11-02T00:00:00Z")))
    assert "rederived" not in run
    assert "rederived" not in json.loads(c.json_bytes(c.SealManifest(run_id="x", files=[])))
    assert "rederived_line" not in json.loads(c.json_bytes(c.SheetModel(run_id="x")))


def test_the_leaves_are_in_config_and_off_when_absent(th: dict) -> None:
    m = th["matching"]
    assert m["same_walk_copies"]["enabled"] and m["faint_left_off"]["enabled"] and m["side_from_track"]["enabled"] and m["debris_by_eye"]["any_view"]
    bare = _off(th, "same_walk_copies", "faint_left_off", "side_from_track")
    assert match.same_walk_copies(bare) is None and not match.faint_left_off(bare) and match.side_from_track(bare) is None
    assert match.orientation_free_panels(bare) == frozenset()


# 5. walkaround rederive ---------------------------------------------------------------------------------------------------


async def _finished(orch: Any) -> str:
    run = await new_run(orch)
    await orch.run_pipeline(run.run_id)
    return run.run_id


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


async def test_rederive_makes_its_own_sealed_run_that_verifies(runs_dir: Path, th: dict) -> None:  # noqa: F811
    orch = make_orch(th, Fakes())
    src = await _finished(orch)
    before = _snapshot(runs_dir / src)
    new = orch.rederive(src, "src_b")
    assert _snapshot(runs_dir / src) == before  # the source stays as it was
    assert new.status == c.RunStatus.DONE and new.api_spend_cents == 0 and new.rederived is not None
    assert new.rederived.from_run == src and new.rederived.text.endswith(f"by run {src}; no new model calls")
    assert new.rederived.text.startswith("verdicts recomputed on ") and "with this version's rules (code abc1234)" in new.rederived.text
    assert new.config_hash == c.config_hash(json.loads(c.canonical_json(th))) and new.thresholds == json.loads(c.canonical_json(th))
    # the same sealed originals: manifest.json and its proofs byte for byte
    assert paths.manifest_json("src_b").read_bytes() == paths.manifest_json(src).read_bytes()
    src_lines = (runs_dir / src / "events.ndjson").read_text().splitlines()
    new_lines = (runs_dir / "src_b" / "events.ndjson").read_text().splitlines()
    cut = next(i for i, line in enumerate(src_lines) if json.loads(line)["kind"] == "match.derived")
    assert new_lines[:cut] == src_lines[:cut]
    tail = [json.loads(line) for line in new_lines[cut:]]
    assert {e["run_id"] for e in tail} == {"src_b"} and tail[-1]["kind"] == "run.completed"
    assert [e["kind"] for e in tail].count("match.derived") == 1 and "sheet.written" in [e["kind"] for e in tail]
    sealed = c.read_json(paths.manifest_run_json("src_b"), c.SealManifest)
    assert sealed.run_id == "src_b" and sealed.rederived == new.rederived.text and sealed.events_head_hash != c.read_json(paths.manifest_run_json(src), c.SealManifest).events_head_hash
    assert not (runs_dir / "src_b" / "quarantine").exists() or not any((runs_dir / "src_b" / "quarantine").iterdir())
    assert orch.verify("src_b") == []
    assert orch.verify(src) == []


async def test_rederive_refuses_an_unknown_run_a_taken_id_and_an_unfinished_run(runs_dir: Path, th: dict) -> None:  # noqa: F811
    from walkaround.orchestrator import RebuildRefused

    orch = make_orch(th, Fakes())
    src = await _finished(orch)
    with pytest.raises(RebuildRefused):
        orch.rederive("nope", "x1")
    with pytest.raises(FileExistsError):
        orch.rederive(src, src)
    orch.update_run(src, lambda r: setattr(r, "status", c.RunStatus.MATCHING))
    with pytest.raises(RebuildRefused):
        orch.rederive(src, "x2")
    assert not (runs_dir / "x2").exists()


def test_cli_rederive_prints_the_sentence_and_verify_rebuilds_it(runs_dir: Path, th: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    orch = make_orch(th, Fakes())
    src = asyncio.run(_finished(orch))
    monkeypatch.setattr(cli, "make_orchestrator", lambda th=None: orch)
    assert cli.main(["rederive", src, "cli_b"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("cli_b: verdicts recomputed on ") and "no new model calls" in out
    assert cli.main(["verify", "cli_b"]) == 0 and capsys.readouterr().out.strip() == "rebuild identical"
    assert cli.main(["rederive", src, "cli_b"]) == 2 and "already exists" in capsys.readouterr().err


def test_the_glossary_names_each_rule_only_with_its_leaf(th: dict) -> None:
    from walkaround import sheet

    lines = (sheet.GLOSSARY_COPIES, sheet.GLOSSARY_DEBRIS_ANY, sheet.GLOSSARY_LEFT_OFF, sheet.GLOSSARY_SIDE_FROM_TRACK)
    on, off = sheet.glossary_md(th), sheet.glossary_md(_off(th, "same_walk_copies", "faint_left_off", "side_from_track", "debris_by_eye"))
    assert all(t in on and t not in off and t.isascii() and "--" not in t for t in lines)
