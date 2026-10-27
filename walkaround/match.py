"""Cross-walk matching and verdicts (docs/ARCHITECTURE.md 5.8, 5.10 and 6, docs/CONTRACTS.md M9). Pure, no API.

Deterministic code owns every label. verdict() is the single entry point and takes numbers and booleans only; a
model answer (the second look, K7) can only lower a label to needs_your_eye, never raise one. Every rule id printed
here exists in walkaround.rules. Calibration follows 5.10: tau_pre0 = max(floor, 2.5 sqrt(2) sigma_within), the
pseudo pair may raise tau_pre to its 90th percentile up to the cap, tau_eye = 2 tau_pre, tau_fix = max(0.08,
3 sigma_frame), and the band widens with the radius: tau(r) = tau0 (1 + 0.5 r^2).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from . import geometry, paths, rules
from .contracts import (
    LOWER_BUMPERS,
    WHEEL_PANELS,
    Calibration,
    CoverageState,
    FrameUsed,
    Keyframe,
    LaneRecord,
    LaneStatus,
    LookVerdict,
    Mark,
    MarkGroup,
    MarkKind,
    MarkTrack,
    MarksStatus,
    MatchCandidate,
    MatchState,
    Noun,
    Panel,
    PanelId,
    PanelMatch,
    StaysPutVerdict,
    Trace,
    Verdict,
    VerdictFlags,
    VerdictLabel,
    WhitenedMark,
    config_hash,
    read_json,
    read_json_list,
    write_json,
)

Point2 = list[float]
ROLES = ("pickup", "return")
CAP_IDS_FROM_MARK = ("K2", "K3", "K4", "K8", "K9", "K10", "K11", "K12")
LOWERABLE = (VerdictLabel.ALREADY_THERE, VerdictLabel.NEW_OR_UNSEEN, VerdictLabel.NOT_SEEN_AT_RETURN)


# Calibration (5.10) --------------------------------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def measured_sigmas(th: dict) -> tuple[float | None, float | None]:
    """(sigma_frame, sigma_spark) from optional thresholds leaves matching.sigma_frame and matching.sigma_spark,
    where the D1 rows E8 and E6 land once measured; None for each leaf that is absent."""
    m = th.get("matching", {})
    frame, spark = m.get("sigma_frame"), m.get("sigma_spark")
    return (None if frame is None else float(frame)), (None if spark is None else float(spark))


def _pseudo_rows(pseudo_pair_d: list[Any]) -> list[tuple[float, float]]:
    """(d, r_mean) per row; a bare number is a distance at r = 0."""
    rows: list[tuple[float, float]] = []
    for row in pseudo_pair_d:
        if isinstance(row, (int, float)):
            rows.append((float(row), 0.0))
        else:
            rows.append((float(row[0]), float(row[1]) if len(row) > 1 else 0.0))
    return rows


def calibrate(*, sigma_frame: float | None, sigma_spark: float | None, sigma_view: float | None, pseudo_pair_d: list[tuple[float, float]] | None, th: dict) -> Calibration:
    """5.10. pseudo_pair_d rows are (d, r_mean) of the pseudo pair's mutual-NN pairs; each d is brought back to
    r = 0 by the band factor before the 90th-percentile rule. calibrated is False when tau_pre hit the cap."""
    m = th["matching"]
    sigmas = [s for s in (sigma_frame, sigma_spark, sigma_view) if s is not None]
    sigma_within = math.sqrt(sum(s * s for s in sigmas)) if sigmas else None
    tau_pre = float(m["tau_pre0_floor"])
    if sigma_within is not None:
        tau_pre = max(tau_pre, float(m["tau_pre0_sigma_factor"]) * sigma_within)
    tau_eye0 = float(m["tau_eye_factor"]) * tau_pre
    if pseudo_pair_d:
        factor = float(m["band_radius_factor"])
        normalised = [d / (1.0 + factor * r * r) for d, r in _pseudo_rows(pseudo_pair_d)]
        under_eye = [d for d in normalised if d < tau_eye0]
        if under_eye:
            share = sum(1 for d in under_eye if d < tau_pre) / len(under_eye)
            if share < float(m["pseudo_pair_share_min"]):
                tau_pre = max(tau_pre, _percentile(under_eye, float(m["pseudo_pair_percentile"])))
    cap = float(m["tau_pre_cap"])
    calibrated = tau_pre <= cap
    tau_pre = min(tau_pre, cap)
    sp = th["staysput"]
    tau_fix = float(sp["tau_fix_floor"])
    if sigma_frame is not None:
        tau_fix = max(tau_fix, float(sp["tau_fix_sigma_factor"]) * sigma_frame)
    return Calibration(
        sigma_frame=sigma_frame,
        sigma_spark=sigma_spark,
        sigma_view=sigma_view,
        sigma_within=sigma_within,
        tau_pre=tau_pre,
        tau_eye=float(m["tau_eye_factor"]) * tau_pre,
        tau_fix=tau_fix,
        calibrated=calibrated,
    )


def tau_r(tau0: float, r: float, th: dict) -> float:
    """The band at radius r: tau0 x (1 + band_radius_factor r^2)."""
    return float(tau0) * (1.0 + float(th["matching"]["band_radius_factor"]) * float(r) * float(r))


# Keyframe pairing and the handle self-test ----------------------------------------------------------------------------


def _aspect_diff(a: Keyframe, b: Keyframe) -> float:
    if a.aspect_ratio <= 0 or b.aspect_ratio <= 0:
        return math.inf
    return abs(math.log(a.aspect_ratio / b.aspect_ratio))


def pair_keyframes(kfs_pickup: list[Keyframe], kfs_return: list[Keyframe]) -> tuple[Keyframe, Keyframe, float]:
    """The pair with the smallest |ln(ar_pickup / ar_return)| among the candidates (a yaw proxy)."""
    if not kfs_pickup or not kfs_return:
        raise ValueError("both walks need at least one keyframe")
    best: tuple[Keyframe, Keyframe, float] | None = None
    for a in kfs_pickup:
        for b in kfs_return:
            diff = _aspect_diff(a, b)
            if best is None or diff < best[2]:
                best = (a, b, diff)
    assert best is not None
    return best


def _handle_whitened(kf: Keyframe) -> tuple[float, float] | None:
    if kf.handle.point_src is None or kf.moments is None or kf.front_sign is None or kf.crop is None:
        return None
    p = geometry.src_to_crop(kf.handle.point_src, kf.crop)
    return geometry.whiten((p[0], p[1]), kf.moments, kf.front_sign)


def handle_residual(kf_pickup: Keyframe, kf_return: Keyframe) -> Point2 | None:
    """Whitened door-handle centroid of the return keyframe minus the pickup one; None when either is missing."""
    a, b = _handle_whitened(kf_pickup), _handle_whitened(kf_return)
    if a is None or b is None:
        return None
    return [b[0] - a[0], b[1] - a[1]]


# Panel gates (section 6, G1 to G10) ---------------------------------------------------------------------------------


def _conditions_flag(kf: Keyframe | None) -> bool:
    return kf is not None and (kf.conditions.wet or kf.conditions.dirty or kf.conditions.glare)


def panel_gates(
    panel: Panel,
    kf_pickup: Keyframe | None,
    kf_return: Keyframe | None,
    *,
    mask_iou: float | None,
    residual: Point2 | None,
    lane_status: dict[str, dict[Noun, LaneStatus]],
    marks_truncated_all: dict[str, bool],
    calibration: Calibration,
    th: dict,
) -> list[str]:
    """The failing gate ids in order G1..G10; an empty list means the panel is comparable."""
    m = th["matching"]
    failing: list[str] = []
    states = [panel.per_role.get(role).state if panel.per_role.get(role) is not None else CoverageState.NOT_SEEN for role in ROLES]
    kfs = (kf_pickup, kf_return)
    if any(s != CoverageState.FILMED for s in states) or any(kf is None for kf in kfs):
        failing.append("G1")
    if any(_conditions_flag(kf) for kf in kfs):
        failing.append("G2")
    if any(kf is not None and kf.score.sharpness < float(m["sharpness_rel_min"]) for kf in kfs):
        failing.append("G3")
    paired = kf_pickup is not None and kf_return is not None
    if paired and (mask_iou is None or mask_iou < float(m["mask_iou_min"])):
        failing.append("G4")
    if residual is not None and math.hypot(residual[0], residual[1]) > calibration.tau_pre:
        failing.append("G5")
    if panel.side_uncertain or panel.front_sign is None or any(kf is not None and kf.front_sign is None for kf in kfs):
        failing.append("G6")
    for role in ROLES:
        status = lane_status.get(role, {}).get(panel.noun)
        if status in (LaneStatus.FAILED, LaneStatus.INCOMPLETE):
            failing.append("G7")
            break
    if paired and panel.panel_id == PanelId.HOOD and (mask_iou is None or mask_iou < float(m["hood_mask_iou_min"])):
        failing.append("G8")
    if kf_pickup is not None and kf_return is not None and kf_pickup.window_subtracted != kf_return.window_subtracted:
        failing.append("G9")
    if any(marks_truncated_all.get(role, False) for role in ROLES):
        failing.append("G10")
    return failing


# Cost, mutual nearest neighbours ------------------------------------------------------------------------------------------


def _compatible(a: MarkKind, b: MarkKind, th: dict) -> bool:
    kinds = {MarkKind(k) for k in th["matching"]["compatible_types"]}
    return a == b or (a in kinds and b in kinds)


def _type_cost(a: MarkKind, b: MarkKind, th: dict) -> float:
    m = th["matching"]
    if a == b:
        return 0.0
    return float(m["type_cost_compatible"]) if _compatible(a, b, th) else float(m["type_cost_incompatible"])


def _size(mark: Mark) -> float | None:
    if mark.size_units is not None and mark.size_units > 0:
        return mark.size_units
    if mark.size_px_src is not None and mark.size_px_src > 0:
        return mark.size_px_src
    return None


def _size_cost(a: Mark, b: Mark, th: dict) -> float:
    sa, sb = _size(a), _size(b)
    if sa is None or sb is None:
        return 0.0
    if (a.size_units is None) != (b.size_units is None):
        return 0.0  # one in units, one in pixels: not comparable, no cost
    return float(th["matching"]["size_cost_factor"]) * abs(math.log(sa / sb))


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _d_pos(wa: WhitenedMark, wb: WhitenedMark) -> float:
    if wa.frame_used == FrameUsed.WHEEL and wb.frame_used == FrameUsed.WHEEL:
        r_mean = 0.5 * (wa.u + wb.u)
        return math.hypot(wa.u - wb.u, r_mean * _wrap_angle(wa.v - wb.v))
    return math.hypot(wa.u - wb.u, wa.v - wb.v)


def cost(a: Mark, b: Mark, wa: WhitenedMark, wb: WhitenedMark, calibration: Calibration, th: dict) -> MatchCandidate:
    """d = |w_a - w_b| + type cost + size_cost_factor |ln(s1 / s2)|; the bands at r_mean. On a lower bumper in the
    moment frame (no arc anchors) the bands carry the bumper_arc_fallback_tau_factor of 5.5."""
    d_pos = _d_pos(wa, wb)
    type_cost = _type_cost(a.type, b.type, th)
    size_cost = _size_cost(a, b, th)
    r_mean = 0.5 * (wa.r + wb.r)
    tau_pre_r = tau_r(calibration.tau_pre, r_mean, th)
    tau_eye_r = tau_r(calibration.tau_eye, r_mean, th)
    if a.panel_id in LOWER_BUMPERS and wa.frame_used == FrameUsed.MOMENT:
        factor = float(th["geometry"]["bumper_arc_fallback_tau_factor"])
        tau_pre_r *= factor
        tau_eye_r *= factor
    return MatchCandidate(
        pickup_mark=a.mark_id,
        return_mark=b.mark_id,
        d=d_pos + type_cost + size_cost,
        d_pos=d_pos,
        type_cost=type_cost,
        size_cost=size_cost,
        r_mean=r_mean,
        tau_pre_r=tau_pre_r,
        tau_eye_r=tau_eye_r,
    )


def mutual_nn(cands: list[MatchCandidate], th: dict) -> list[MatchCandidate]:
    """Mutual nearest neighbours by d. On a side with two or more candidates the Lowe ratio (second / nearest >=
    ratio_test_min) and the margin (second - nearest >= margin_min_tau_pre_factor x tau_pre_r) must hold; a side
    with one candidate passes by rule. Returned pairs carry mutual=True and the binding ratio and margin."""
    m = th["matching"]
    ratio_min = float(m["ratio_test_min"])
    margin_factor = float(m["margin_min_tau_pre_factor"])
    by_pickup: dict[str, list[MatchCandidate]] = {}
    by_return: dict[str, list[MatchCandidate]] = {}
    for c in cands:
        by_pickup.setdefault(c.pickup_mark, []).append(c)
        by_return.setdefault(c.return_mark, []).append(c)
    for lst in (*by_pickup.values(), *by_return.values()):
        lst.sort(key=lambda c: (c.d, c.pickup_mark, c.return_mark))
    out: list[MatchCandidate] = []
    for pickup_id in sorted(by_pickup):
        best = by_pickup[pickup_id][0]
        if by_return[best.return_mark][0] is not best:
            continue
        ratio: float | None = None
        margin: float | None = None
        passes = True
        for side in (by_pickup[pickup_id], by_return[best.return_mark]):
            if len(side) < 2:
                continue
            first, second = side[0].d, side[1].d
            side_ratio = second / max(first, 1e-9)
            side_margin = second - first
            passes = passes and side_ratio >= ratio_min and side_margin >= margin_factor * best.tau_pre_r
            ratio = side_ratio if ratio is None else min(ratio, side_ratio)
            margin = side_margin if margin is None else min(margin, side_margin)
        if passes:
            out.append(best.model_copy(update={"mutual": True, "ratio": ratio, "margin": margin}))
    return out


# The single entry point ------------------------------------------------------------------------------------------------------


def verdict(d: float | None, tau_pre_r: float, tau_eye_r: float, flags: VerdictFlags) -> tuple[VerdictLabel, list[str]]:
    """Section 6. d None: L1 not_seen_at_return (pickup) or L2 new_or_unseen (return); d < tau_pre_r: L3
    already_there; tau_pre_r <= d < tau_eye_r: L4 needs_your_eye with K1; d >= tau_eye_r: L5, the unmatched label.
    Any set K flag then lowers already_there, new_or_unseen or not_seen_at_return to needs_your_eye (L4). The ids
    come back label rule first, then the band rule when a cap lowered it, then the K ids. Nothing moves a label up."""
    unmatched = VerdictLabel.NEW_OR_UNSEEN if flags.return_mark else VerdictLabel.NOT_SEEN_AT_RETURN
    ids: list[str]
    if d is None:
        label, ids = unmatched, ["L2" if flags.return_mark else "L1"]
    elif d < tau_pre_r:
        label, ids = VerdictLabel.ALREADY_THERE, ["L3"]
    elif d < tau_eye_r:
        label, ids = VerdictLabel.NEEDS_YOUR_EYE, ["L4", "K1"]
    else:
        label, ids = unmatched, ["L5"]
    caps = flags.set_caps()
    if caps and label in LOWERABLE:
        label = VerdictLabel.NEEDS_YOUR_EYE
        ids = ["L4", *ids]
    for cap in caps:
        if cap not in ids:
            ids.append(cap)
    for rule_id in ids:
        rules.rule(rule_id)
    return label, ids


# Per-panel matching --------------------------------------------------------------------------------------------------


def _shift_return(whitened: dict[str, WhitenedMark], residual: Point2 | None) -> dict[str, WhitenedMark]:
    if residual is None:
        return dict(whitened)
    du, dv = float(residual[0]), float(residual[1])
    return {k: w.model_copy(update={"u": w.u - du, "v": w.v - dv, "residual_applied": [du, dv]}) for k, w in whitened.items()}


def _plural(kind: MarkKind, n: int) -> str:
    name = kind.value
    if kind in (MarkKind.DIRT, MarkKind.UNSURE):
        return f"{name} mark" if n == 1 else f"{name} marks"
    if n == 1:
        return name
    return name + "es" if name.endswith("ch") else name + "s"


def _groups(marks_pickup: list[Mark], marks_return: list[Mark], pairs: list[MatchCandidate]) -> list[MarkGroup]:
    """One line per type that has a group (5.6) on either walk, counting the group members only."""
    grouped_p = [m for m in marks_pickup if m.cluster_id is not None]
    grouped_r = [m for m in marks_return if m.cluster_id is not None]
    matched_return = {p.return_mark for p in pairs}
    out: list[MarkGroup] = []
    for kind in sorted({m.type for m in (*grouped_p, *grouped_r)}, key=lambda k: k.value):
        n_p = sum(1 for m in grouped_p if m.type == kind)
        n_r = sum(1 for m in grouped_r if m.type == kind)
        new = sum(1 for m in grouped_r if m.type == kind and m.mark_id not in matched_return)
        text = f"{n_p} {_plural(kind, n_p)} here at pickup, {n_r} at return, {new} new or unseen"
        out.append(MarkGroup(type=kind, n_pickup=n_p, n_return=n_r, text=text))
    return out


def _group_pairs(cands: list[MatchCandidate], pairs: list[MatchCandidate], marks_pickup: list[Mark], marks_return: list[Mark]) -> list[MatchCandidate]:
    """Inside a group the ratio test is waived (every ratio in a cluster is near 1, 5.8 item 9): still-unmatched
    group members of one type pair greedily by cost under tau_pre_r, so the per-mark labels add up to the group
    line. A wrong pairing inside one cluster is the same cluster; two clusters apart stay unmatched."""
    matched = {p.pickup_mark for p in pairs} | {p.return_mark for p in pairs}
    grouped = {m.mark_id: m for m in (*marks_pickup, *marks_return) if m.cluster_id is not None}
    extra: list[MatchCandidate] = []
    for c in sorted(cands, key=lambda c: (c.d, c.pickup_mark, c.return_mark)):
        if c.pickup_mark in matched or c.return_mark in matched:
            continue
        a, b = grouped.get(c.pickup_mark), grouped.get(c.return_mark)
        if a is None or b is None or a.type != b.type or c.d >= c.tau_pre_r:
            continue
        extra.append(c.model_copy(update={"mutual": True}))
        matched |= {c.pickup_mark, c.return_mark}
    return extra


def _unmatched(marks: list[Mark], matched: set[str]) -> list[str]:
    return [m.mark_id for m in marks if m.mark_id not in matched]


def match_panel(
    panel: Panel,
    marks_pickup: list[Mark],
    marks_return: list[Mark],
    whitened: dict[str, WhitenedMark],
    kf_pair: tuple[Keyframe, Keyframe],
    gates: list[str],
    calibration: Calibration,
    th: dict,
) -> PanelMatch:
    """One PanelMatch: state and reasons from the gates, the keyframe pair and aspect_diff, the handle residual
    subtracted from the return marks, pairs by cost and mutual_nn (pairs beyond tau_eye_r stay listed so verdicts_for
    can print L5), the unmatched lists and the group lines. mask_iou is filled by the caller that saw the masks."""
    kf_p, kf_r = kf_pair
    if panel.panel_id in WHEEL_PANELS:
        pm = wheel_match(panel, marks_pickup, marks_return, whitened, th=th, calibration=calibration)
        state = MatchState.NOT_COMPARABLE if gates else MatchState.COMPARABLE
        update: dict[str, Any] = {"state": state, "reasons": [*gates, *pm.reasons], "kf_pair": [kf_p.kf_id, kf_r.kf_id], "aspect_diff": _aspect_diff(kf_p, kf_r)}
        if gates:
            update.update({"pairs": [], "unmatched_pickup": [m.mark_id for m in marks_pickup], "unmatched_return": [m.mark_id for m in marks_return], "groups": []})
        return pm.model_copy(update=update)
    residual = handle_residual(kf_p, kf_r)
    pm = PanelMatch(
        panel_id=panel.panel_id,
        state=MatchState.NOT_COMPARABLE if gates else MatchState.COMPARABLE,
        reasons=list(gates),
        kf_pair=[kf_p.kf_id, kf_r.kf_id],
        aspect_diff=_aspect_diff(kf_p, kf_r),
        handle_residual=residual,
        frame_used=FrameUsed.MOMENT,
    )
    if gates:
        return pm.model_copy(update={"unmatched_pickup": [m.mark_id for m in marks_pickup], "unmatched_return": [m.mark_id for m in marks_return]})
    w_pickup = {m.mark_id: whitened[m.mark_id] for m in marks_pickup if m.mark_id in whitened}
    w_return = _shift_return({m.mark_id: whitened[m.mark_id] for m in marks_return if m.mark_id in whitened}, residual)
    cands = [cost(a, b, w_pickup[a.mark_id], w_return[b.mark_id], calibration, th) for a in marks_pickup if a.mark_id in w_pickup for b in marks_return if b.mark_id in w_return]
    pairs = mutual_nn(cands, th)
    pairs = [*pairs, *_group_pairs(cands, pairs, marks_pickup, marks_return)]
    matched = {p.pickup_mark for p in pairs} | {p.return_mark for p in pairs}
    return pm.model_copy(
        update={
            "pairs": pairs,
            "unmatched_pickup": _unmatched(marks_pickup, matched),
            "unmatched_return": _unmatched(marks_return, matched),
            "groups": _groups(marks_pickup, marks_return, pairs),
        }
    )


def _wheel_rule_ok(a: Mark, b: Mark, wa: WhitenedMark, wb: WhitenedMark, th: dict) -> bool:
    """5.8 item 8 for a one-mark wheel: compatible type, radius within tolerance, size within the factor."""
    m = th["matching"]
    if not _compatible(a.type, b.type, th):
        return False
    if abs(wa.u - wb.u) > float(m["wheel_radius_tolerance"]):
        return False
    sa, sb = _size(a), _size(b)
    if sa is not None and sb is not None and (a.size_units is None) == (b.size_units is None):
        ratio = max(sa, sb) / min(sa, sb)
        if ratio > float(m["wheel_size_factor_max"]):
            return False
    return True


def wheel_match(panel: Panel, marks_pickup: list[Mark], marks_return: list[Mark], whitened: dict[str, WhitenedMark], *, th: dict, calibration: Calibration | None = None) -> PanelMatch:
    """5.8 item 8. With >= 2 marks on both walks the return angles take the circular shift that pairs the most
    marks at the lowest cost, then cost and mutual_nn run as usual. With exactly one mark on a side, that mark
    pairs only with the single candidate passing the type, radius and size rule; otherwise the wheel's unmatched
    marks carry K10 (listed in reasons) and verdicts_for caps them. Angles alone never match."""
    cal = calibration or calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=th)
    pm = PanelMatch(panel_id=panel.panel_id, state=MatchState.COMPARABLE, frame_used=FrameUsed.WHEEL)
    w_p = {m.mark_id: whitened[m.mark_id] for m in marks_pickup if m.mark_id in whitened}
    w_r = {m.mark_id: whitened[m.mark_id] for m in marks_return if m.mark_id in whitened}
    placed_p = [m for m in marks_pickup if m.mark_id in w_p]
    placed_r = [m for m in marks_return if m.mark_id in w_r]
    pairs: list[MatchCandidate] = []
    reasons: list[str] = []
    min_for_shift = int(th["matching"]["wheel_min_marks_for_shift"])
    if len(placed_p) >= min_for_shift and len(placed_r) >= min_for_shift:
        best: tuple[int, float, list[MatchCandidate]] | None = None
        for a in placed_p:
            for b in placed_r:
                shift = w_p[a.mark_id].v - w_r[b.mark_id].v
                shifted = {k: w.model_copy(update={"v": _wrap_angle(w.v + shift)}) for k, w in w_r.items()}
                cands = [cost(x, y, w_p[x.mark_id], shifted[y.mark_id], cal, th) for x in placed_p for y in placed_r]
                found = mutual_nn(cands, th)
                in_band = [c for c in found if c.d < c.tau_eye_r]
                score = (len(in_band), -sum(c.d for c in in_band))
                if best is None or score > (best[0], best[1]):
                    best = (score[0], score[1], found)
        pairs = best[2] if best is not None else []
    elif placed_p and placed_r:
        singles = ([(a, placed_r) for a in placed_p] if len(placed_p) == 1 else []) + ([(b, placed_p) for b in placed_r] if len(placed_r) == 1 else [])
        for single, others in singles:
            is_pickup = single in placed_p
            fits = [o for o in others if _wheel_rule_ok(single, o, w_p[single.mark_id] if is_pickup else w_r[single.mark_id], w_r[o.mark_id] if is_pickup else w_p[o.mark_id], th)]
            if len(fits) == 1 and not pairs:
                a, b = (single, fits[0]) if is_pickup else (fits[0], single)
                wa = w_p[a.mark_id].model_copy(update={"v": 0.0})
                wb = w_r[b.mark_id].model_copy(update={"v": 0.0})
                pairs = [cost(a, b, wa, wb, cal, th).model_copy(update={"mutual": True})]
        if len(pairs) < max(len(placed_p), len(placed_r)):
            reasons.append("K10")
    matched = {p.pickup_mark for p in pairs} | {p.return_mark for p in pairs}
    return pm.model_copy(
        update={
            "reasons": reasons,
            "pairs": pairs,
            "unmatched_pickup": _unmatched(marks_pickup, matched),
            "unmatched_return": _unmatched(marks_return, matched),
            "groups": _groups(marks_pickup, marks_return, pairs),
        }
    )


# Flags, verdicts, traces ---------------------------------------------------------------------------------------------


def _staysput_cap(mt: MarkTrack | None) -> bool:
    return mt is not None and (mt.verdict in (StaysPutVerdict.SLIDING, StaysPutVerdict.INCONCLUSIVE) or mt.curvature_cap or mt.highlight_cap)


def flags_for(
    pair: MatchCandidate | None,
    mark: Mark,
    other: Mark | None,
    mark_tracks: dict[str, MarkTrack],
    look: LookVerdict | None,
    kf_status: dict[str, MarksStatus],
    *,
    return_mark: bool,
) -> VerdictFlags:
    """K1..K13 as booleans: K1 from the pair's band, K5 from the two types, K6 from Stays Put, K7 from the second
    look, K13 from any truncated keyframe in kf_status, the rest from the caps of the mark and its partner."""
    caps = set(mark.caps) | (set(other.caps) if other is not None else set())
    values: dict[str, bool] = {k: k in caps for k in CAP_IDS_FROM_MARK}
    values["K1"] = pair is not None and pair.tau_pre_r <= pair.d < pair.tau_eye_r
    compatible = {MarkKind.SCRATCH, MarkKind.SCUFF, MarkKind.CHIP}
    values["K5"] = pair is not None and other is not None and mark.type != other.type and not (mark.type in compatible and other.type in compatible)
    values["K6"] = any(_staysput_cap(mark_tracks.get(m.mark_id)) for m in (mark, other) if m is not None)
    values["K7"] = look in (LookVerdict.DIFFERENT, LookVerdict.UNSURE)
    values["K13"] = any(status == MarksStatus.TRUNCATED for status in kf_status.values())
    return VerdictFlags(return_mark=return_mark, **values)


def _trace(trace_base: Trace, rule_ids: list[str], marks: list[Mark], mark_tracks: dict[str, MarkTrack], thresholds: dict[str, Any]) -> Trace:
    spark_ids = sorted({m.spark_request_id for m in marks if m.spark_request_id})
    sam_ids = sorted({mark_tracks[m.mark_id].request_id for m in marks if m.mark_id in mark_tracks and mark_tracks[m.mark_id].request_id})
    return Trace(
        rule_ids=rule_ids,
        thresholds=thresholds,
        sam_request_ids=sorted({*trace_base.sam_request_ids, *sam_ids}),
        spark_request_ids=sorted({*trace_base.spark_request_ids, *spark_ids}),
        event_seqs=list(trace_base.event_seqs),
        config_hash=trace_base.config_hash,
    )


def _thresholds_used(calibration: Calibration, th: dict, *, tau_pre_r: float | None, tau_eye_r: float | None) -> dict[str, Any]:
    m = th["matching"]
    out: dict[str, Any] = {
        "tau_pre": calibration.tau_pre,
        "tau_eye": calibration.tau_eye,
        "band_radius_factor": m["band_radius_factor"],
        "type_cost_compatible": m["type_cost_compatible"],
        "type_cost_incompatible": m["type_cost_incompatible"],
        "size_cost_factor": m["size_cost_factor"],
        "ratio_test_min": m["ratio_test_min"],
        "margin_min_tau_pre_factor": m["margin_min_tau_pre_factor"],
        "cm_per_unit_for_print": m["cm_per_unit_for_print"],
    }
    if tau_pre_r is not None:
        out["tau_pre_r"] = tau_pre_r
    if tau_eye_r is not None:
        out["tau_eye_r"] = tau_eye_r
    return out


def _with_k10(mark: Mark, pm: PanelMatch, unmatched: set[str]) -> Mark:
    if "K10" in pm.reasons and mark.mark_id in unmatched and "K10" not in mark.caps:
        return mark.model_copy(update={"caps": [*mark.caps, "K10"]})
    return mark


def verdicts_for(
    pm: PanelMatch,
    marks: dict[str, Mark],
    mark_tracks: dict[str, MarkTrack],
    looks: dict[tuple[str, str], LookVerdict],
    keyframes: dict[str, Keyframe],
    calibration: Calibration,
    trace_base: Trace,
    th: dict,
) -> list[Verdict]:
    """One Verdict per pair (two when the pair lies beyond tau_eye_r, L5) and per unmatched mark, every label
    through verdict(); a not_comparable panel gives one L6 verdict naming its gates. distance_cm_est = d_pos x
    cm_per_unit_for_print."""
    cm_per_unit = float(th["matching"]["cm_per_unit_for_print"])
    panel_marks = [marks[i] for i in (*(p.pickup_mark for p in pm.pairs), *(p.return_mark for p in pm.pairs), *pm.unmatched_pickup, *pm.unmatched_return) if i in marks]
    kf_status = {kf.kf_id: kf.marks_status for kf in keyframes.values() if kf.panel_id == pm.panel_id}
    out: list[Verdict] = []

    def vid() -> str:
        return f"{pm.panel_id.value}_v{len(out) + 1}"

    if pm.state == MatchState.NOT_COMPARABLE:
        gate_ids = [r for r in pm.reasons if r.startswith("G")]
        rule_ids = ["L6", *gate_ids]
        out.append(
            Verdict(
                verdict_id=vid(),
                panel_id=pm.panel_id,
                label=VerdictLabel.NOT_COMPARABLE,
                rule_id="L6",
                mark_ids=sorted({m.mark_id for m in panel_marks}),
                reasons=rules.describe(gate_ids),
                trace=_trace(trace_base, rule_ids, panel_marks, mark_tracks, _thresholds_used(calibration, th, tau_pre_r=None, tau_eye_r=None)),
            )
        )
        return out

    def emit(label: VerdictLabel, ids: list[str], mark_ids: list[str], d_pos: float | None, pair: MatchCandidate | None) -> None:
        involved = [marks[i] for i in mark_ids if i in marks]
        thresholds = _thresholds_used(calibration, th, tau_pre_r=pair.tau_pre_r if pair else None, tau_eye_r=pair.tau_eye_r if pair else None)
        if pair is not None:
            thresholds.update({"d": pair.d, "d_pos": pair.d_pos, "type_cost": pair.type_cost, "size_cost": pair.size_cost, "r_mean": pair.r_mean})
        out.append(
            Verdict(
                verdict_id=vid(),
                panel_id=pm.panel_id,
                label=label,
                rule_id=ids[0],
                mark_ids=mark_ids,
                distance_units=d_pos,
                distance_cm_est=None if d_pos is None else d_pos * cm_per_unit,
                reasons=rules.describe([i for i in ids[1:] if i.startswith("K")]),
                trace=_trace(trace_base, ids, involved, mark_tracks, thresholds),
            )
        )

    for pair in pm.pairs:
        a, b = marks.get(pair.pickup_mark), marks.get(pair.return_mark)
        if a is None or b is None:
            continue
        look = looks.get((pair.pickup_mark, pair.return_mark))
        if pair.d < pair.tau_eye_r:
            flags = flags_for(pair, a, b, mark_tracks, look, kf_status, return_mark=False)
            label, ids = verdict(pair.d, pair.tau_pre_r, pair.tau_eye_r, flags)
            emit(label, ids, [a.mark_id, b.mark_id], pair.d_pos, pair)
        else:
            for mark, is_return in ((a, False), (b, True)):
                flags = flags_for(pair, mark, None, mark_tracks, None, kf_status, return_mark=is_return)
                label, ids = verdict(pair.d, pair.tau_pre_r, pair.tau_eye_r, flags)
                emit(label, ids, [mark.mark_id], None, pair)
    unmatched = set(pm.unmatched_pickup) | set(pm.unmatched_return)
    for mark_id, is_return in (*((i, False) for i in pm.unmatched_pickup), *((i, True) for i in pm.unmatched_return)):
        mark = marks.get(mark_id)
        if mark is None:
            continue
        mark = _with_k10(mark, pm, unmatched)
        r = mark.r if mark.r is not None else 0.0
        tau_pre_r = tau_r(calibration.tau_pre, r, th)
        tau_eye_r = tau_r(calibration.tau_eye, r, th)
        flags = flags_for(None, mark, None, mark_tracks, None, kf_status, return_mark=is_return)
        label, ids = verdict(None, tau_pre_r, tau_eye_r, flags)
        emit(label, ids, [mark.mark_id], None, None)
    return out


# The derive entry --------------------------------------------------------------------------------------------------------


def _read_list(path: Any, model: Any) -> list[Any]:
    return read_json_list(path, model) if path.is_file() and path.stat().st_size > 0 else []


def _load_mask(run_id: str, kf: Keyframe) -> np.ndarray | None:
    if not kf.mask_png:
        return None
    import cv2

    m = cv2.imread(str(paths.run_dir(run_id) / kf.mask_png), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


def _mask_iou(run_id: str, kf_p: Keyframe, kf_r: Keyframe) -> float | None:
    if kf_p.moments is None or kf_r.moments is None or kf_p.front_sign is None or kf_r.front_sign is None:
        return None
    a, b = _load_mask(run_id, kf_p), _load_mask(run_id, kf_r)
    if a is None or b is None:
        return None
    try:
        return geometry.whitened_mask_iou(a, kf_p.moments, kf_p.front_sign, b, kf_r.moments, kf_r.front_sign)
    except ValueError:
        return None


def _lane_status(run_id: str) -> dict[str, dict[Noun, LaneStatus]]:
    """Lane statuses from sam/<role>/<lane>.summary.json where present; a missing summary is not a failure."""
    out: dict[str, dict[Noun, LaneStatus]] = {}
    for role in ROLES:
        statuses: dict[Noun, LaneStatus] = {}
        for noun in Noun:
            path = paths.lane_summary_path(run_id, role, noun)
            if path.is_file() and path.stat().st_size > 0:
                statuses[noun] = read_json(path, LaneRecord).status
        out[role] = statuses
    return out


def _provisional_already_there(pair: MatchCandidate, a: Mark, b: Mark, mark_tracks: dict[str, MarkTrack], kf_status: dict[str, MarksStatus]) -> bool:
    flags = flags_for(pair, a, b, mark_tracks, None, kf_status, return_mark=False)
    label, _ = verdict(pair.d, pair.tau_pre_r, pair.tau_eye_r, flags)
    return label == VerdictLabel.ALREADY_THERE


def derive_match(run_id: str, *, th: dict, second_look: Callable[[MatchCandidate], LookVerdict] | None = None) -> tuple[list[PanelMatch], list[Verdict], Calibration]:
    """Read panels, keyframes, marks, mark_tracks and calibration; gate, pair and match every panel with a keyframe
    in either walk; ask the optional second look on provisional already_there pairs (at most
    second_looks_per_pair_max); write matches.json and verdicts.json. Under a second without the second look."""
    from .marks import whitened_from_marks

    panels = _read_list(paths.panels_json(run_id), Panel)
    keyframes = _read_list(paths.keyframes_json(run_id), Keyframe)
    marks = _read_list(paths.marks_json(run_id), Mark)
    mark_tracks = {mt.mark_id: mt for mt in _read_list(paths.mark_tracks_json(run_id), MarkTrack)}
    cal_path = paths.calibration_json(run_id)
    if cal_path.is_file() and cal_path.stat().st_size > 0:
        calibration = read_json(cal_path, Calibration)
    else:
        sigma_frame, sigma_spark = measured_sigmas(th)
        calibration = calibrate(sigma_frame=sigma_frame, sigma_spark=sigma_spark, sigma_view=None, pseudo_pair_d=None, th=th)
        write_json(cal_path, calibration)
    whitened = whitened_from_marks(marks)
    marks_by_id = {m.mark_id: m for m in marks}
    kf_by_id = {kf.kf_id: kf for kf in keyframes}
    lane_status = _lane_status(run_id)
    trace_base = Trace(config_hash=config_hash(th))
    looks_budget = int(th["spark"]["second_looks_per_pair_max"])
    looks_used = 0
    matches: list[PanelMatch] = []
    verdicts: list[Verdict] = []
    for panel in panels:
        kfs = {role: [kf for kf in keyframes if kf.panel_id == panel.panel_id and kf.role == role] for role in ROLES}
        if not kfs["pickup"] and not kfs["return"]:
            continue
        panel_marks = {role: [m for m in marks if m.panel_id == panel.panel_id and m.role == role] for role in ROLES}
        truncated_all = {role: bool(kfs[role]) and all(kf.marks_status == MarksStatus.TRUNCATED for kf in kfs[role]) for role in ROLES}
        if kfs["pickup"] and kfs["return"]:
            kf_p, kf_r, _ = pair_keyframes(kfs["pickup"], kfs["return"])
            mask_iou = _mask_iou(run_id, kf_p, kf_r)
            residual = handle_residual(kf_p, kf_r)
            gates = panel_gates(panel, kf_p, kf_r, mask_iou=mask_iou, residual=residual, lane_status=lane_status, marks_truncated_all=truncated_all, calibration=calibration, th=th)
            pm = match_panel(panel, panel_marks["pickup"], panel_marks["return"], whitened, (kf_p, kf_r), gates, calibration, th)
            pm = pm.model_copy(update={"mask_iou": mask_iou})
        else:
            gates = panel_gates(panel, kfs["pickup"][0] if kfs["pickup"] else None, kfs["return"][0] if kfs["return"] else None, mask_iou=None, residual=None, lane_status=lane_status, marks_truncated_all=truncated_all, calibration=calibration, th=th)
            pm = PanelMatch(
                panel_id=panel.panel_id,
                state=MatchState.NOT_COMPARABLE,
                reasons=gates,
                unmatched_pickup=[m.mark_id for m in panel_marks["pickup"]],
                unmatched_return=[m.mark_id for m in panel_marks["return"]],
            )
        looks: dict[tuple[str, str], LookVerdict] = {}
        if second_look is not None and pm.state == MatchState.COMPARABLE:
            kf_status = {kf.kf_id: kf.marks_status for kf in keyframes if kf.panel_id == pm.panel_id}
            for pair in pm.pairs:
                a, b = marks_by_id.get(pair.pickup_mark), marks_by_id.get(pair.return_mark)
                if a is None or b is None or looks_used >= looks_budget:
                    continue
                if _provisional_already_there(pair, a, b, mark_tracks, kf_status):
                    looks[(pair.pickup_mark, pair.return_mark)] = second_look(pair)
                    looks_used += 1
        matches.append(pm)
        verdicts.extend(verdicts_for(pm, marks_by_id, mark_tracks, looks, kf_by_id, calibration, trace_base, th))
    write_json(paths.matches_json(run_id), matches)
    write_json(paths.verdicts_json(run_id), verdicts)
    return matches, verdicts, calibration
