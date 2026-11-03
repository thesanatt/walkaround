"""R30 (lead decision of 11/3, car body only): the wheels (rims and tires) get no mark-level verdicts; like glass each
wheel panel gets one check-by-eye line (matching.wheels_by_eye). Wheel tracking, keyframes and crops stay. No network."""

from __future__ import annotations

import copy
from typing import Any

from walkaround import contracts as c
from walkaround import match, paths, sheet
from walkaround.contracts import PanelId, VerdictLabel, WHEEL_PANELS
from walkaround.contracts import write_json

from .test_match import DOOR, ROLES, kf, mark, panel, runs_dir, th, write_crop  # noqa: F401  (fixtures)

WHEEL = PanelId.LEFT_FRONT_WHEEL


def _on(th: dict, **value: Any) -> dict:
    out = copy.deepcopy(th)
    out["matching"]["wheels_by_eye"] = {"enabled": True, **value}
    return out


def seed_wheel_run(run_id: str) -> None:
    """A door and a left front wheel, each with a keyframe in both walks and marks in both walks."""
    paths.ensure_run_layout(run_id)
    kfs = [write_crop(run_id, kf(f"{role}_kf1", role, handle=[200.0, 150.0])) for role in ROLES]
    kfs += [write_crop(run_id, kf(f"{role}_wheel_kf1", role, WHEEL)) for role in ROLES]
    door = [mark("p0", "pickup", u=0.0, v=0.0), mark("r0", "return", u=0.0, v=0.0)]
    wheel = [
        mark("pw0", "pickup", u=0.5, v=0.2, kf_id="pickup_wheel_kf1", panel_id=WHEEL),
        mark("rw0", "return", u=0.5, v=0.2, kf_id="return_wheel_kf1", panel_id=WHEEL),
        mark("rw1", "return", u=-0.6, v=0.4, kf_id="return_wheel_kf1", panel_id=WHEEL),
    ]
    wheel_panel = panel(WHEEL).model_copy(update={"per_role": {role: c.PanelRole(state=c.CoverageState.FILMED, keyframe_ids=[f"{role}_wheel_kf1"]) for role in ROLES}})
    write_json(paths.panels_json(run_id), [panel(), wheel_panel])
    write_json(paths.keyframes_json(run_id), kfs)
    write_json(paths.marks_json(run_id), [*door, *wheel])


def test_the_leaf_is_off_when_absent_or_disabled_and_on_for_new_runs(th: dict) -> None:  # noqa: F811
    assert "wheels_by_eye" not in th["matching"]  # the shared fixture leaves it out, as frozen runs do
    assert match.wheels_by_eye(th) == frozenset()
    assert match.wheels_by_eye(_on(th, enabled=False)) == frozenset()
    assert match.wheels_by_eye(_on(th)) == set(WHEEL_PANELS)  # the default panel list is the four wheels
    new = c.load_thresholds()
    assert match.wheels_by_eye(new) == set(WHEEL_PANELS)
    assert match.glass_by_eye(new) == {PanelId.WINDSHIELD, PanelId.REAR_WINDOW}


def test_by_eye_panels_name_each_line_and_glass_wins_a_shared_panel(th: dict) -> None:  # noqa: F811
    on = _on(th)
    on["matching"]["glass_by_eye"] = {"enabled": True, "panels": ["windshield", "left_front_wheel"]}
    got = match.by_eye_panels(on)
    assert got[PanelId.WINDSHIELD] == ("glass", match.GLASS_LINE)
    assert got[WHEEL] == ("glass", match.GLASS_LINE)
    assert got[PanelId.RIGHT_REAR_WHEEL] == ("wheels", match.WHEELS_LINE)
    assert match.WHEELS_LINE == "wheels: check by eye; rims and tires are not checked mark by mark"


def test_a_wheel_gets_one_line_and_no_verdict_and_the_body_is_as_before(runs_dir: Any, th: dict) -> None:  # noqa: F811
    run_id = "01TESTRUNR30WHEELS00000000"
    seed_wheel_run(run_id)
    m0, before, _ = match.derive_match(run_id, th=th)
    assert any(v.panel_id == WHEEL for v in before)  # without the leaf the wheel marks get verdicts
    assert not any(pm.by_eye for pm in m0 if pm.panel_id == WHEEL)
    matches, verdicts, _ = match.derive_match(run_id, th=_on(th))
    wheel = next(pm for pm in matches if pm.panel_id == WHEEL)  # the panel stays: tracked and keyframed
    assert not [v for v in verdicts if v.panel_id == WHEEL]
    assert [(b.kind, b.text) for b in wheel.by_eye] == [("wheels", match.WHEELS_LINE)]
    assert wheel.by_eye[0].mark_ids == ["pw0", "rw0", "rw1"]
    assert not wheel.pairs and not wheel.unmatched_pickup and not wheel.unmatched_return
    assert wheel.kf_pair == ["pickup_wheel_kf1", "return_wheel_kf1"]
    body = lambda vs: sorted((v.panel_id.value, v.label.value, v.mark_ids) for v in vs if v.panel_id != WHEEL)  # noqa: E731
    assert body(verdicts) == body(before) and [v.label for v in verdicts] == [VerdictLabel.ALREADY_THERE]
    assert match.by_eye_words(wheel) == [f"{match.WHEELS_LINE} (3 spots Muse Spark boxed on it, not judged)"]


def test_a_wheel_with_no_marks_shows_the_bare_line_and_the_leaf_off_writes_no_by_eye(runs_dir: Any, th: dict) -> None:  # noqa: F811
    line = c.ByEyeLine(kind="wheels", text=match.WHEELS_LINE)
    pm = c.PanelMatch(panel_id=WHEEL, state=c.MatchState.NOT_COMPARABLE, reasons=["G1"], by_eye=[line])
    assert match.by_eye_words(pm) == [match.WHEELS_LINE]
    assert c.PanelMatch.model_validate_json(c.canonical_json(pm)) == pm
    run_id = "01TESTRUNR30WHEELS00000001"
    seed_wheel_run(run_id)
    match.derive_match(run_id, th=th)
    assert b'"by_eye"' not in paths.matches_json(run_id).read_bytes()


def test_the_glossary_names_the_wheels_only_with_the_leaf_on(th: dict) -> None:  # noqa: F811
    assert "wheels: each wheel gets one line" not in sheet.glossary_md(th)
    assert "wheels: each wheel gets one line" in sheet.glossary_md(_on(th))
    assert "wheels: each wheel gets one line" in sheet.glossary_md(c.load_thresholds())


def test_wheel_tracking_settings_are_untouched_by_the_leaf() -> None:
    new = c.load_thresholds()
    assert "wheel" in new["nouns"]["lanes"]  # SAM 3.1 still tracks the wheels
    assert new["panels"]["fender_from_wheels"]  # and the fenders are still built from them
