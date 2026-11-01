"""Cross-walk matching and verdicts (docs/ARCHITECTURE.md 5.8, 5.10 and 6, docs/CONTRACTS.md M9). Pure, no API.

Deterministic code owns every label. verdict() is the single entry point and takes numbers and booleans only; a
model answer (the second look, K7; a wet, dirty or glare view, K15) can only lower a label to needs_your_eye, never
raise one above what the distance gives. Every rule id printed here exists in walkaround.rules. Calibration follows
5.10: tau_pre0 = max(floor, 2.5 sqrt(2) sigma_within), the pseudo pair may raise tau_pre to its 90th percentile up
to the cap, tau_eye = 2 tau_pre, tau_fix = max(0.08, 3 sigma_frame), and the band widens with the radius: tau(r) =
tau0 (1 + 0.5 r^2).

Lead decisions of 11/1, each behind a thresholds leaf that runs frozen before it lack (so they rebuild as they were):
- matching.tight_pair_second_look (rule 1): on a pair inside tau_pre_r, K4 (type dirt, reflection or unsure) and K12
  (Spark confidence low) are notes, not caps, and the pair goes to the second look, which may lower it (K7). With no
  usable answer (not asked, or every engine failed) they cap as before. An already-there call rests on geometry, the
  same spot in both walks, not on what the mark is; the second look compares the two crops and is the veto.
- matching.next_keyframe_pair (rule 2): when the chosen keyframe pair fails a view gate (G3, G4, G5, G8, G9), the
  panel's other keyframe pairs are tried in a fixed order and the first that passes every gate is compared on; every
  pair tried is recorded with its gates (PanelMatch.kf_pairs_tried, how_checked). G1, G6, G7 and G10 stay panel-level.
"""

from __future__ import annotations

import json
import math
import re
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
    KeyframePairTry,
    LaneRecord,
    LaneStatus,
    LookVerdict,
    Mark,
    MarkGroup,
    MarkKind,
    MarkSource,
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
CAP_IDS_FROM_MARK = ("K2", "K3", "K4", "K8", "K9", "K10", "K11", "K12", "K14")
LOWERABLE = (VerdictLabel.ALREADY_THERE, VerdictLabel.NEW_OR_UNSEEN, VerdictLabel.NOT_SEEN_AT_RETURN)
VIEW_GATES = ("G3", "G4", "G5", "G8", "G9")  # gates of the keyframe pair: another pair of the panel may pass them (rule 2)
PANEL_GATES = ("G1", "G6", "G7", "G10")  # gates of the panel: no other keyframe pair changes them, so none is tried
TYPE_CAPS = ("K4", "K12")  # what Muse Spark typed and how sure it was: notes on a tight pair the second look answered (rule 1)


def tight_pair_look(th: dict | None) -> bool:
    """Rule 1 of 11/1 is on (thresholds matching.tight_pair_second_look); off when absent, as in runs frozen before."""
    return bool((th or {}).get("matching", {}).get("tight_pair_second_look", False))


def next_keyframe_pair(th: dict | None) -> bool:
    """Rule 2 of 11/1 is on (thresholds matching.next_keyframe_pair); off when absent, as in runs frozen before."""
    return bool((th or {}).get("matching", {}).get("next_keyframe_pair", False))


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


def pair_keyframes(kfs_pickup: list[Keyframe], kfs_return: list[Keyframe], *, th: dict | None = None) -> tuple[Keyframe, Keyframe, float]:
    """The pair with the smallest |ln(ar_pickup / ar_return)| among the candidates (a yaw proxy). A walk with a view
    flag on one of its keyframes offers only its clean usable keyframes when it has one (prefer_clean, lead 10/31)."""
    if not kfs_pickup or not kfs_return:
        raise ValueError("both walks need at least one keyframe")
    kfs_pickup, kfs_return = prefer_clean(kfs_pickup, th), prefer_clean(kfs_return, th)
    best: tuple[Keyframe, Keyframe, float] | None = None
    for a in kfs_pickup:
        for b in kfs_return:
            diff = _aspect_diff(a, b)
            if best is None or diff < best[2]:
                best = (a, b, diff)
    assert best is not None
    return best


# View flags (lead decision 10/31: Muse Spark's wet, dirty and glare are model judgments, so they only lower a label
# through K15 and never gate a panel; G2 is retired) -------------------------------------------------------------------

VIEW_FLAGS = ("wet", "dirty", "glare")  # dark is not one: it never gated and caps nothing
_COULD_BE = {"wet": "water", "dirty": "dirt", "glare": "a reflection"}


def view_flags(kf: Keyframe | None) -> list[str]:
    """The wet, dirty and glare flags Muse Spark set on a keyframe crop, in that order."""
    return [] if kf is None else [f for f in VIEW_FLAGS if getattr(kf.conditions, f)]


def _usable_clean(kf: Keyframe, th: dict | None) -> bool:
    """Spark answered the keyframe in full and flagged nothing, and it passes the per-keyframe gates (G3 sharpness, G6
    front_sign) with a crop and moments to whiten in. Without th the sharpness is not checked."""
    if kf.marks_status != MarksStatus.COMPLETE or view_flags(kf) or kf.crop is None or kf.moments is None or kf.front_sign is None:
        return False
    return th is None or kf.score.sharpness >= float(th["matching"]["sharpness_rel_min"])


def prefer_clean(kfs: list[Keyframe], th: dict | None = None) -> list[Keyframe]:
    """One walk's keyframe candidates for the pair: when one of them carries a view flag and a clean usable one exists,
    only the clean usable ones; otherwise all of them, so a walk with no flag (or no clean usable keyframe) pairs
    exactly as before 10/31. The choice is made here: panels picks keyframes before Muse Spark has seen them."""
    if not any(view_flags(kf) for kf in kfs):
        return kfs
    clean = [kf for kf in kfs if _usable_clean(kf, th)]
    return clean or kfs


_KF_INDEX = re.compile(r"_kf(\d+)$")


def _kf_index(kf: Keyframe) -> tuple[int, str]:
    """The keyframe's number within its walk (panels names keyframes <role>_<panel>_kf<n>, best score first), then the
    id, so every tie has one order whatever order keyframes.json lists them in."""
    found = _KF_INDEX.search(kf.kf_id)
    return (int(found.group(1)) if found else 1 << 30, kf.kf_id)


def keyframe_pair_order(kfs_pickup: list[Keyframe], kfs_return: list[Keyframe], *, th: dict | None = None) -> list[tuple[Keyframe, Keyframe]]:
    """Every keyframe pair of a panel in the order match tries them (lead 11/1, rule 2): the pair pair_keyframes chooses
    first, then the others with clean views first (how many of the two keyframes prefer_clean would not offer: 0, 1
    or 2), then the product of the two keyframe scores, highest first (a pair needs both views good; a sum would rank
    one sharp view with one poor view above two fair ones), then the keyframe numbers, pickup first."""
    first_p, first_r, _ = pair_keyframes(kfs_pickup, kfs_return, th=th)
    clean_p = {kf.kf_id for kf in prefer_clean(kfs_pickup, th)}
    clean_r = {kf.kf_id for kf in prefer_clean(kfs_return, th)}
    others = [(a, b) for a in kfs_pickup for b in kfs_return if (a.kf_id, b.kf_id) != (first_p.kf_id, first_r.kf_id)]
    others.sort(key=lambda ab: ((ab[0].kf_id not in clean_p) + (ab[1].kf_id not in clean_r), -(ab[0].score.total * ab[1].score.total), _kf_index(ab[0]), _kf_index(ab[1])))
    return [(first_p, first_r), *others]


def _joined(items: list[str], last: str) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {last} " + items[-1]


def _flag_words(flags: list[str]) -> str:
    return _joined(flags, "and")


def _or_words(items: list[str]) -> str:
    return _joined(items, "or")


def one_walk_note(role: str, flags: dict[str, list[str]]) -> str | None:
    """K15 in words for a mark found in one walk only: "return view flagged dirty: could be dirt" for the view it was
    found in, "pickup view flagged glare: could have hidden the mark" for the view it is missing from; None when
    neither view is flagged."""
    other = "return" if role == "pickup" else "pickup"
    parts = []
    if flags.get(role):
        parts.append(f"{role} view flagged {_flag_words(flags[role])}: could be {_or_words([_COULD_BE[f] for f in flags[role]])}")
    if flags.get(other):
        parts.append(f"{other} view flagged {_flag_words(flags[other])}: could have hidden the mark")
    return "; ".join(parts) or None


def both_walks_note(flags: dict[str, list[str]], *, same_place: bool) -> str | None:
    """The note on a pair seen in both walks when a view is flagged; the flag lowers nothing there (lead 10/31 (d)): a
    mark in the same place in both walks was there at pickup, damage or dirt. None when no view is flagged."""
    views = ", ".join(f"{role} view flagged {_flag_words(flags[role])}" for role in ROLES if flags.get(role))
    if not views:
        return None
    if same_place:
        return f"{views}: the mark is in the same place in both walks, so it was there at pickup, whatever it is"
    return f"{views}: a mark shows near this place in both walks"


# Tight pairs (lead 11/1, rule 1: K4 and K12 are notes on a pair inside tau_pre_r once the second look answered) -------


def _spark_words(a: Mark, b: Mark, deferred: list[str], roles: tuple[str, str] = ROLES) -> str:
    """What K4 and K12 stand for on this pair, in words with their ids. roles names the two marks (a claim passes
    ("pickup", "claim"))."""
    typed = None
    if "K4" in deferred:
        typed = f"both marks {a.type.value}" if a.type == b.type else f"the {roles[0]} mark {a.type.value} and the {roles[1]} mark {b.type.value}"
    low = [role for role, m in zip(roles, (a, b)) if "K12" in m.caps]
    where = f"the {low[0]} mark" if len(low) == 1 else "both marks"
    if typed and "K12" in deferred:
        return f"Muse Spark typed {typed} (K4), with low confidence on {where} (K12)"
    if typed:
        return f"Muse Spark typed {typed} (K4)"
    return f"Muse Spark's confidence was low on {where} (K12)"


def tight_pair_flags(
    pair: MatchCandidate, a: Mark, b: Mark, flags: VerdictFlags, look: LookVerdict | None, *, asked: bool, roles: tuple[str, str] = ROLES
) -> tuple[VerdictFlags, list[str], str | None]:
    """Rule 1 on a pair inside tau_pre_r (the caller checks d < tau_pre_r): once the second look answered (same,
    different or unsure), K4 and K12 are notes and lower nothing, so only the look (K7) or another cap can lower the
    pair; with no usable answer they cap as before. asked says the look was put to a model (its answer may be None:
    every engine failed or the look could not be built) rather than not asked at all (spend ceiling, deadline, look
    budget, the look turned off). Returns the flags for verdict(), the note ids and the type note; ([], None) when
    neither K4 nor K12 is set. roles names the two marks in the note: a claim passes ("pickup", "claim") (integration
    11/1: claims follow rule 1 under the same leaf)."""
    deferred = [k for k in TYPE_CAPS if getattr(flags, k)]
    if not deferred:
        return flags, [], None
    what = _spark_words(a, b, deferred, roles)
    if look is not None:
        if look == LookVerdict.SAME:
            place = "the mark sits in the same place in both walks" if tuple(roles) == ROLES else f"the {roles[1]} mark sits where the {roles[0]} mark is"
            tail = f"a note here, since {place} and the second look compared the two crops and said same"
        else:
            said = "said different" if look == LookVerdict.DIFFERENT else "was unsure"
            tail = f"a note here; the second look compared the two crops and {said} (K7)"
        return flags.model_copy(update={k: False for k in deferred}), deferred, f"{what}: {tail}"
    ids, verb = _joined(deferred, "and"), ("caps" if len(deferred) == 1 else "cap")
    others = flags.model_copy(update={**{k: False for k in deferred}, "K7": False})
    if verdict(pair.d, pair.tau_pre_r, pair.tau_eye_r, others)[0] != VerdictLabel.ALREADY_THERE:
        tail = f"another cap already lowers this pair, so no second look was asked and {ids} {verb} it too"
    elif asked:
        tail = f"the second look gave no usable answer, so {ids} {verb} the pair as before"
    else:
        tail = f"the second look was not asked, so {ids} {verb} the pair as before"
    return flags, [], f"{what}: {tail}"


# The sheet's how-checked line (lead 11/1, rule 2) -------------------------------------------------------------------


def _gate_words(gate: str, tried: KeyframePairTry) -> str:
    """A failed gate in words: G4 and G8 with the IoU, or why none was taken (a keyframe whose marks are compared
    turned its panel axis too far from the pair's pickup keyframe, or there was no mask or frame to take it in)."""
    label = rules.rule(gate).label
    if gate in ("G4", "G8"):
        if tried.mask_iou is not None:
            return f"{gate} {label} {tried.mask_iou:.2f}"
        return f"{gate} {label}: {'not measured' if tried.frames_agree else 'panel axes disagree'}"
    return f"{gate} {label}"


def how_checked(pm: PanelMatch) -> str | None:
    """The sheet's how-it-was-checked line for a panel: the keyframe pair it was compared on and every pair tried
    before it with the gates each failed; for a panel no pair passed, every pair tried. None when the match recorded
    no tries (keyframes in one walk only, or a run frozen before 11/1)."""
    tries = pm.kf_pairs_tried
    if not tries:
        return None

    def pair_words(t: KeyframePairTry) -> str:
        return f"{t.kf_pair[0]} with {t.kf_pair[1]}"

    def tried(t: KeyframePairTry) -> str:
        return f"{pair_words(t)} ({', '.join(_gate_words(g, t) for g in t.gates)})"

    used = next((t for t in tries if not t.gates), None)
    if used is not None:
        before = [t for t in tries if t is not used]
        if not before:
            return f"compared on keyframes {pair_words(used)}, the first pair tried"
        return f"compared on keyframes {pair_words(used)}; tried first: " + "; ".join(tried(t) for t in before)
    panel_level = [g for g in tries[0].gates if g in PANEL_GATES]
    if panel_level:
        which = f"{panel_level[0]} is a panel gate" if len(panel_level) == 1 else f"{_joined(panel_level, 'and')} are panel gates"
        return f"not compared: keyframes {tried(tries[0])}; {which}, so no other pair was tried"
    if len(tries) == 1:
        return f"not compared: keyframes {tried(tries[0])}, and the panel has no other keyframe pair"
    return f"not compared: none of the {len(tries)} keyframe pairs passed every gate; tried: " + "; ".join(tried(t) for t in tries)


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


# Panel gates (section 6, G1 and G3 to G10; G2 retired 10/31) ----------------------------------------------------------


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
    """The failing gate ids in order G1, G3..G10; an empty list means the panel is comparable. G2 (a wet, dirty or
    glare flag on either crop) is no gate since 10/31: a flagged view caps the marks that depend on it (K15)."""
    m = th["matching"]
    failing: list[str] = []
    states = [panel.per_role.get(role).state if panel.per_role.get(role) is not None else CoverageState.NOT_SEEN for role in ROLES]
    kfs = (kf_pickup, kf_return)
    if any(s != CoverageState.FILMED for s in states) or any(kf is None for kf in kfs):
        failing.append("G1")
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
        return geometry.wheel_distance((wa.u, wa.v), (wb.u, wb.v))  # the metric marks' dedup uses on wheels
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


def mutual_nn(cands: list[MatchCandidate], th: dict, *, tau_pre: float | None = None) -> list[MatchCandidate]:
    """Mutual nearest neighbours by d. On a side with two or more candidates the Lowe ratio (second / nearest >=
    ratio_test_min) and the margin (second - nearest >= margin_min_tau_pre_factor x tau_pre) must hold, tau_pre
    being the calibrated value of 5.8 item 5, not the band at the pair's radius (that rejected clear matches near
    the corners and printed an old mark as new). Without tau_pre it is read back from the pair as tau_pre_r / (1
    + band_radius_factor r_mean^2), which keeps the 1.5 fallback factor on a lower bumper. A side with one
    candidate passes by rule. Returned pairs carry mutual=True and the binding ratio and margin."""
    m = th["matching"]
    ratio_min = float(m["ratio_test_min"])
    margin_factor = float(m["margin_min_tau_pre_factor"])
    band = float(m["band_radius_factor"])
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
        base = tau_pre if tau_pre is not None else best.tau_pre_r / (1.0 + band * best.r_mean * best.r_mean)
        for side in (by_pickup[pickup_id], by_return[best.return_mark]):
            if len(side) < 2:
                continue
            first, second = side[0].d, side[1].d
            side_ratio = second / max(first, 1e-9)
            side_margin = second - first
            passes = passes and side_ratio >= ratio_min and side_margin >= margin_factor * base
            ratio = side_ratio if ratio is None else min(ratio, side_ratio)
            margin = side_margin if margin is None else min(margin, side_margin)
        if passes:
            out.append(best.model_copy(update={"mutual": True, "ratio": ratio, "margin": margin}))
    return out


def seeded_ids(*walks: list[Mark]) -> frozenset[str]:
    """The narration_seed marks among these: marks Muse Spark boxed because the renter named them."""
    return frozenset(m.mark_id for marks in walks for m in marks if m.source == MarkSource.NARRATION_SEED)


def seeds_last_nn(cands: list[MatchCandidate], seeded: frozenset[str], th: dict, *, tau_pre: float | None = None) -> list[MatchCandidate]:
    """mutual_nn over the box marks, then over the seed candidates among the marks still free: a seed (K14 caps it)
    can lower a label but never unsettles a box pair through the ratio test (5 step 11). Without seeds: mutual_nn."""
    box = [c for c in cands if c.pickup_mark not in seeded and c.return_mark not in seeded]
    pairs = mutual_nn(box, th, tau_pre=tau_pre)
    taken = {p.pickup_mark for p in pairs} | {p.return_mark for p in pairs}
    rest = [c for c in cands if (c.pickup_mark in seeded or c.return_mark in seeded) and c.pickup_mark not in taken and c.return_mark not in taken]
    return [*pairs, *mutual_nn(rest, th, tau_pre=tau_pre)] if rest else pairs


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
    """The return marks with the handle residual subtracted; r follows the shift so the band is taken where the
    comparison happens."""
    if residual is None:
        return dict(whitened)
    du, dv = float(residual[0]), float(residual[1])
    return {k: w.model_copy(update={"u": w.u - du, "v": w.v - dv, "r": math.hypot(w.u - du, w.v - dv), "residual_applied": [du, dv]}) for k, w in whitened.items()}


def _xy(whitened: dict[str, WhitenedMark]) -> dict[str, tuple[float, float]]:
    return {k: (w.u, w.v) for k, w in whitened.items()}


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _plural(kind: MarkKind, n: int) -> str:
    name = kind.value
    if kind in (MarkKind.DIRT, MarkKind.UNSURE):
        return f"{name} mark" if n == 1 else f"{name} marks"
    if n == 1:
        return name
    return name + "es" if name.endswith("ch") else name + "s"


Members = dict[MarkKind, tuple[list[str], list[str]]]


def _group_members(
    marks_pickup: list[Mark],
    marks_return: list[Mark],
    pos_pickup: dict[str, tuple[float, float]],
    pos_return: dict[str, tuple[float, float]],
    *,
    dist: Callable[[tuple[float, float], tuple[float, float]], float],
    th: dict,
) -> Members:
    """Per type with a group (5.6) on either walk, the member ids of each walk in mark order: a walk's grouped
    marks, plus the same-type marks of the other walk within group_radius_units of one of them. Each walk forms
    its groups alone and Spark's mark set swings between runs (10/26 probe), so three chips grouped at pickup and
    two of them found at return are one group, not three gone and two new. Positions are the compared ones (the
    return side shifted by the handle residual, or by the wheel's circular shift)."""
    radius_units = float(th["dedup"]["group_radius_units"])

    def near(mark_id: str, pos: dict[str, tuple[float, float]], seeds: list[Mark], seed_pos: dict[str, tuple[float, float]]) -> bool:
        p = pos.get(mark_id)
        return p is not None and any(s.mark_id in seed_pos and dist(p, seed_pos[s.mark_id]) <= radius_units for s in seeds)

    out: Members = {}
    for kind in sorted({m.type for m in (*marks_pickup, *marks_return) if m.cluster_id is not None}, key=lambda k: k.value):
        seeds_p = [m for m in marks_pickup if m.type == kind and m.cluster_id is not None]
        seeds_r = [m for m in marks_return if m.type == kind and m.cluster_id is not None]
        members_p = [m.mark_id for m in marks_pickup if m.type == kind and (m.cluster_id is not None or near(m.mark_id, pos_pickup, seeds_r, pos_return))]
        members_r = [m.mark_id for m in marks_return if m.type == kind and (m.cluster_id is not None or near(m.mark_id, pos_return, seeds_p, pos_pickup))]
        out[kind] = (members_p, members_r)
    return out


def _group_lines(members: Members, pairs: list[MatchCandidate]) -> list[MarkGroup]:
    """One line per type with a group on either walk ("4 chips here at pickup, 5 at return, 1 new or unseen").
    New or unseen counts the return members with no pair inside tau_eye_r, the ones verdicts_for labels new or
    unseen before any cap, so the line adds up with the per-mark labels."""
    paired_return = {p.return_mark for p in pairs if p.d < p.tau_eye_r}
    out: list[MarkGroup] = []
    for kind, (mp, mr) in members.items():
        new = sum(1 for i in mr if i not in paired_return)
        text = f"{len(mp)} {_plural(kind, len(mp))} here at pickup, {len(mr)} at return, {new} new or unseen"
        out.append(MarkGroup(type=kind, n_pickup=len(mp), n_return=len(mr), text=text))
    return out


def _group_pairs(cands: list[MatchCandidate], pairs: list[MatchCandidate], members: Members, seeded: frozenset[str] = frozenset()) -> list[MatchCandidate]:
    """Inside a group every ratio is near 1 (5.8 item 9), so the ratio test is waived there: still-unmatched
    members of one type pair greedily by cost while d < tau_eye_r, and the band labels them (already_there, or
    needs_your_eye with K1) instead of a confident not seen plus new. A wrong pairing inside one cluster is the
    same cluster; two clusters apart stay unmatched. Group pairs keep mutual False (they did not pass the mutual
    NN test), which is how matches.json and verdicts_for tell them apart. Box candidates go before seed candidates."""
    # TODO(contracts): the waiver is not in CONTRACTS.md or 5.8; it needs a rule id the trace can carry.
    pick = {i: kind for kind, (mp, _) in members.items() for i in mp}
    ret = {i: kind for kind, (_, mr) in members.items() for i in mr}
    matched = {p.pickup_mark for p in pairs} | {p.return_mark for p in pairs}
    extra: list[MatchCandidate] = []
    for c in sorted(cands, key=lambda c: (c.pickup_mark in seeded or c.return_mark in seeded, c.d, c.pickup_mark, c.return_mark)):
        if c.pickup_mark in matched or c.return_mark in matched:
            continue
        kind = pick.get(c.pickup_mark)
        if kind is None or kind != ret.get(c.return_mark) or c.d >= c.tau_eye_r:
            continue
        extra.append(c.model_copy(update={"mutual": False, "ratio": None, "margin": None}))
        matched |= {c.pickup_mark, c.return_mark}
    return extra


def _unmatched(marks: list[Mark], matched: set[str]) -> list[str]:
    # TODO(contracts): a mark with no whitened position (its keyframe had no moments, or a wheel keyframe lost its
    # mask) lands here and prints L1 or L2 with no cap when its keyframe is not the paired one; no K id says
    # "could not be placed" yet.
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
    """One PanelMatch: state and reasons from the gates, the keyframe pair and aspect_diff (None when not finite),
    the handle residual subtracted from the return marks, pairs by cost and mutual_nn (pairs beyond tau_eye_r stay
    listed so verdicts_for can print L5) plus the group pairs, the unmatched lists and the group lines. mask_iou
    is filled by the caller that saw the masks."""
    kf_p, kf_r = kf_pair
    aspect_diff = _finite(_aspect_diff(kf_p, kf_r))
    if panel.panel_id in WHEEL_PANELS:
        pm = wheel_match(panel, marks_pickup, marks_return, whitened, th=th, calibration=calibration)
        state = MatchState.NOT_COMPARABLE if gates else MatchState.COMPARABLE
        update: dict[str, Any] = {"state": state, "reasons": [*gates, *pm.reasons], "kf_pair": [kf_p.kf_id, kf_r.kf_id], "aspect_diff": aspect_diff}
        if gates:
            update.update({"pairs": [], "unmatched_pickup": [m.mark_id for m in marks_pickup], "unmatched_return": [m.mark_id for m in marks_return], "groups": []})
        return pm.model_copy(update=update)
    residual = handle_residual(kf_p, kf_r)
    # TODO(5.5): hoods are matched in the plain moment frame and geometry.hood_coords (lateral axis plus depth
    # thirds) is not wired. A synthetic grazing-view hood kept every residual under tau_pre_r up to a 0.75 m / 3 m
    # stance change and 20 degrees of yaw, where mark_matching.md expects 0.5 to 0.8 units along depth. D1 must
    # measure the depth residual on a real hood pair; switch hoods to FrameUsed.HOOD_AXIS only if it is that large.
    # TODO(D1): medoids come from any of the 3 keyframes of a walk while the handle residual is measured on the
    # aspect-paired keyframes. The cross-keyframe part is the within-walk view spread that sigma_view puts into
    # tau_pre (5.10); D1 (E8) must confirm that the handle residual between keyframes of one walk stays near
    # sigma_view, or the residual has to be taken per keyframe.
    pm = PanelMatch(
        panel_id=panel.panel_id,
        state=MatchState.NOT_COMPARABLE if gates else MatchState.COMPARABLE,
        reasons=list(gates),
        kf_pair=[kf_p.kf_id, kf_r.kf_id],
        aspect_diff=aspect_diff,
        handle_residual=residual,
        frame_used=FrameUsed.MOMENT,
    )
    if gates:
        return pm.model_copy(update={"unmatched_pickup": [m.mark_id for m in marks_pickup], "unmatched_return": [m.mark_id for m in marks_return]})
    w_pickup = {m.mark_id: whitened[m.mark_id] for m in marks_pickup if m.mark_id in whitened}
    w_return = _shift_return({m.mark_id: whitened[m.mark_id] for m in marks_return if m.mark_id in whitened}, residual)
    cands = [cost(a, b, w_pickup[a.mark_id], w_return[b.mark_id], calibration, th) for a in marks_pickup if a.mark_id in w_pickup for b in marks_return if b.mark_id in w_return]
    seeded = seeded_ids(marks_pickup, marks_return)
    pairs = seeds_last_nn(cands, seeded, th, tau_pre=calibration.tau_pre)
    members = _group_members(marks_pickup, marks_return, _xy(w_pickup), _xy(w_return), dist=math.dist, th=th)
    pairs = [*pairs, *_group_pairs(cands, pairs, members, seeded=seeded)]
    matched = {p.pickup_mark for p in pairs} | {p.return_mark for p in pairs}
    return pm.model_copy(
        update={
            "pairs": pairs,
            "unmatched_pickup": _unmatched(marks_pickup, matched),
            "unmatched_return": _unmatched(marks_return, matched),
            "groups": _group_lines(members, pairs),
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
    marks at the lowest cost, then cost and mutual_nn run as usual, plus the group pairs at that shift. With
    exactly one mark on a side, that mark pairs only with the single candidate passing the type, radius and size
    rule; otherwise the wheel's unmatched marks carry K10 (listed in reasons) and verdicts_for caps them. Angles
    alone never match: without a shift the group lines compare radii only."""
    cal = calibration or calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=th)
    pm = PanelMatch(panel_id=panel.panel_id, state=MatchState.COMPARABLE, frame_used=FrameUsed.WHEEL)
    w_p = {m.mark_id: whitened[m.mark_id] for m in marks_pickup if m.mark_id in whitened}
    w_r = {m.mark_id: whitened[m.mark_id] for m in marks_return if m.mark_id in whitened}
    seeded = seeded_ids(marks_pickup, marks_return)  # seeds stay unmatched on a wheel (K14 caps them)
    placed_p = [m for m in marks_pickup if m.mark_id in w_p and m.mark_id not in seeded]
    placed_r = [m for m in marks_return if m.mark_id in w_r and m.mark_id not in seeded]
    pairs: list[MatchCandidate] = []
    reasons: list[str] = []
    pos_p = {k: (w.u, 0.0) for k, w in w_p.items()}
    pos_r = {k: (w.u, 0.0) for k, w in w_r.items()}
    shift_cands: list[MatchCandidate] = []
    min_for_shift = int(th["matching"]["wheel_min_marks_for_shift"])
    if len(placed_p) >= min_for_shift and len(placed_r) >= min_for_shift:
        best: tuple[tuple[int, float], list[MatchCandidate], dict[str, WhitenedMark], list[MatchCandidate]] | None = None
        for a in placed_p:
            for b in placed_r:
                shift = w_p[a.mark_id].v - w_r[b.mark_id].v
                shifted = {k: w.model_copy(update={"v": _wrap_angle(w.v + shift)}) for k, w in w_r.items()}
                cands = [cost(x, y, w_p[x.mark_id], shifted[y.mark_id], cal, th) for x in placed_p for y in placed_r]
                found = mutual_nn(cands, th, tau_pre=cal.tau_pre)
                in_band = [c for c in found if c.d < c.tau_eye_r]
                score = (len(in_band), -sum(c.d for c in in_band))
                if best is None or score > best[0]:
                    best = (score, found, shifted, cands)
        assert best is not None
        _, pairs, shifted, shift_cands = best
        pos_p, pos_r = _xy(w_p), _xy(shifted)
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
    members = _group_members(marks_pickup, marks_return, pos_p, pos_r, dist=geometry.wheel_distance, th=th)
    if shift_cands:
        pairs = [*pairs, *_group_pairs(shift_cands, pairs, members)]
    matched = {p.pickup_mark for p in pairs} | {p.return_mark for p in pairs}
    return pm.model_copy(
        update={
            "reasons": reasons,
            "pairs": pairs,
            "unmatched_pickup": _unmatched(marks_pickup, matched),
            "unmatched_return": _unmatched(marks_return, matched),
            "groups": _group_lines(members, pairs),
        }
    )


# Flags, verdicts, traces ---------------------------------------------------------------------------------------------


def _staysput_cap(mt: MarkTrack | None) -> bool:
    return mt is not None and (mt.verdict in (StaysPutVerdict.SLIDING, StaysPutVerdict.INCONCLUSIVE) or mt.curvature_cap or mt.highlight_cap)


def staysput_caps(mark_tracks: dict[str, MarkTrack], th: dict) -> dict[str, MarkTrack]:
    """The Stays Put rows K6 may read: all of them while thresholds staysput.cap_enabled is on, none while it is off
    (docs/D1_MEMO.md note 2: not calibrated). Traces keep reading the full rows for their request ids."""
    return mark_tracks if th["staysput"].get("cap_enabled", True) else {}


def flags_for(
    pair: MatchCandidate | None,
    mark: Mark,
    other: Mark | None,
    mark_tracks: dict[str, MarkTrack],
    look: LookVerdict | None,
    kf_status: dict[str, MarksStatus],
    *,
    return_mark: bool,
    view_flagged: bool = False,
) -> VerdictFlags:
    """K1..K15 as booleans: K1 from the pair's band, K5 from the two types, K6 from Stays Put, K7 from the second
    look, K13 from any truncated keyframe in kf_status, K15 from view_flagged (verdicts_for sets it for a mark found
    in one walk only when a view it depends on is flagged), the rest from the caps of the mark and its partner."""
    caps = set(mark.caps) | (set(other.caps) if other is not None else set())
    values: dict[str, bool] = {k: k in caps for k in CAP_IDS_FROM_MARK}
    values["K1"] = pair is not None and pair.tau_pre_r <= pair.d < pair.tau_eye_r
    compatible = {MarkKind.SCRATCH, MarkKind.SCUFF, MarkKind.CHIP}
    values["K5"] = pair is not None and other is not None and mark.type != other.type and not (mark.type in compatible and other.type in compatible)
    values["K6"] = any(_staysput_cap(mark_tracks.get(m.mark_id)) for m in (mark, other) if m is not None)
    values["K7"] = look in (LookVerdict.DIFFERENT, LookVerdict.UNSURE)
    values["K13"] = any(status == MarksStatus.TRUNCATED for status in kf_status.values())
    values["K15"] = view_flagged
    return VerdictFlags(return_mark=return_mark, **values)


def _trace(trace_base: Trace, rule_ids: list[str], marks: list[Mark], mark_tracks: dict[str, MarkTrack], thresholds: dict[str, Any], note_ids: list[str] | None = None) -> Trace:
    spark_ids = sorted({m.spark_request_id for m in marks if m.spark_request_id})
    sam_ids = sorted({mark_tracks[m.mark_id].request_id for m in marks if m.mark_id in mark_tracks and mark_tracks[m.mark_id].request_id})
    return Trace(
        rule_ids=rule_ids,
        thresholds=thresholds,
        sam_request_ids=sorted({*trace_base.sam_request_ids, *sam_ids}),
        spark_request_ids=sorted({*trace_base.spark_request_ids, *spark_ids}),
        event_seqs=list(trace_base.event_seqs),
        config_hash=trace_base.config_hash,
        note_ids=list(note_ids or []),
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
    looks: dict[tuple[str, str], LookVerdict | None],
    keyframes: dict[str, Keyframe],
    calibration: Calibration,
    trace_base: Trace,
    th: dict,
) -> list[Verdict]:
    """One Verdict per pair (two when the pair lies beyond tau_eye_r, L5) and per unmatched mark, every label
    through verdict(); a not_comparable panel gives one L6 verdict naming its gates. distance_cm_est = d_pos x
    cm_per_unit_for_print.

    View flags (lead 10/31, K15): a walk's view is its paired keyframe plus the keyframe each involved mark was placed
    in. A mark found in one walk only (L1, L2, and both marks of an L5 pair) carries K15 when a view of either walk
    is flagged wet, dirty or glare; a pair inside tau_eye_r gets no K15 (a mark in the same place in both walks was
    there at pickup) and only the note. view_note says which view and why.

    Tight pairs (lead 11/1, rule 1, while thresholds matching.tight_pair_second_look is on): on a pair inside tau_pre_r
    K4 and K12 are notes (trace.note_ids, type_note) when looks holds an answer for it, and cap as before when it holds
    none. A key in looks with the value None is a look put to a model that gave no usable answer; a missing key is a
    look never asked. Marks seen in one walk only and pairs in the band keep K4 and K12 as caps."""
    cm_per_unit = float(th["matching"]["cm_per_unit_for_print"])
    panel_marks = [marks[i] for i in (*(p.pickup_mark for p in pm.pairs), *(p.return_mark for p in pm.pairs), *pm.unmatched_pickup, *pm.unmatched_return) if i in marks]
    kf_status = {kf.kf_id: kf.marks_status for kf in keyframes.values() if kf.panel_id == pm.panel_id}
    caps = staysput_caps(mark_tracks, th)
    out: list[Verdict] = []

    def vid() -> str:
        return f"{pm.panel_id.value}_v{len(out) + 1}"

    paired = {role: keyframes.get(kf_id) for role, kf_id in zip(ROLES, pm.kf_pair)} if pm.kf_pair else {}

    def flags_by_role(involved: list[Mark]) -> dict[str, list[str]]:
        """Per walk, the view flags of its paired keyframe and of the keyframes the involved marks were placed in."""
        out_flags: dict[str, list[str]] = {}
        for role in ROLES:
            views = [paired.get(role), *(keyframes.get(m.kf_id) for m in involved if m.role == role)]
            found = {f for kf in views for f in view_flags(kf)}
            out_flags[role] = [f for f in VIEW_FLAGS if f in found]
        return out_flags

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

    def emit(
        label: VerdictLabel, ids: list[str], mark_ids: list[str], d_pos: float | None, pair: MatchCandidate | None, view_note: str | None = None, *, type_note: str | None = None, note_ids: list[str] | None = None
    ) -> None:
        involved = [marks[i] for i in mark_ids if i in marks]
        thresholds = _thresholds_used(calibration, th, tau_pre_r=pair.tau_pre_r if pair else None, tau_eye_r=pair.tau_eye_r if pair else None)
        if pair is not None:
            thresholds.update({"d": pair.d, "d_pos": pair.d_pos, "type_cost": pair.type_cost, "size_cost": pair.size_cost, "r_mean": pair.r_mean})
            if not pair.mutual:
                thresholds["group_pairing"] = True  # paired inside a group with the ratio test waived
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
                trace=_trace(trace_base, ids, involved, mark_tracks, thresholds, note_ids),
                view_note=view_note,
                type_note=type_note,
            )
        )

    tight = tight_pair_look(th)
    for pair in pm.pairs:
        a, b = marks.get(pair.pickup_mark), marks.get(pair.return_mark)
        if a is None or b is None:
            continue
        key = (pair.pickup_mark, pair.return_mark)
        look = looks.get(key)
        if pair.d < pair.tau_eye_r:
            flags = flags_for(pair, a, b, caps, look, kf_status, return_mark=False)
            notes: list[str] = []
            type_note = None
            if tight and pair.d < pair.tau_pre_r:
                flags, notes, type_note = tight_pair_flags(pair, a, b, flags, look, asked=key in looks)
            label, ids = verdict(pair.d, pair.tau_pre_r, pair.tau_eye_r, flags)
            emit(label, ids, [a.mark_id, b.mark_id], pair.d_pos, pair, both_walks_note(flags_by_role([a, b]), same_place=pair.d < pair.tau_pre_r), type_note=type_note, note_ids=notes)
        else:
            for mark, is_return in ((a, False), (b, True)):
                view = flags_by_role([mark])
                flags = flags_for(pair, mark, None, caps, None, kf_status, return_mark=is_return, view_flagged=any(view.values()))
                label, ids = verdict(pair.d, pair.tau_pre_r, pair.tau_eye_r, flags)
                emit(label, ids, [mark.mark_id], None, pair, one_walk_note("return" if is_return else "pickup", view))
    unmatched = set(pm.unmatched_pickup) | set(pm.unmatched_return)
    for mark_id, is_return in (*((i, False) for i in pm.unmatched_pickup), *((i, True) for i in pm.unmatched_return)):
        mark = marks.get(mark_id)
        if mark is None:
            continue
        mark = _with_k10(mark, pm, unmatched)
        r = mark.r if mark.r is not None else 0.0
        tau_pre_r = tau_r(calibration.tau_pre, r, th)
        tau_eye_r = tau_r(calibration.tau_eye, r, th)
        view = flags_by_role([mark])
        flags = flags_for(None, mark, None, caps, None, kf_status, return_mark=is_return, view_flagged=any(view.values()))
        label, ids = verdict(None, tau_pre_r, tau_eye_r, flags)
        emit(label, ids, [mark.mark_id], None, None, one_walk_note("return" if is_return else "pickup", view))
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


def _mask_iou(run_id: str, kf_p: Keyframe, kf_r: Keyframe, masks: dict[str, np.ndarray | None] | None = None) -> float | None:
    """The whitened mask IoU of a keyframe pair; masks caches each keyframe's mask across the pairs of a panel."""
    if kf_p.moments is None or kf_r.moments is None or kf_p.front_sign is None or kf_r.front_sign is None:
        return None

    def load(kf: Keyframe) -> np.ndarray | None:
        if masks is None:
            return _load_mask(run_id, kf)
        if kf.kf_id not in masks:
            masks[kf.kf_id] = _load_mask(run_id, kf)
        return masks[kf.kf_id]

    a, b = load(kf_p), load(kf_r)
    if a is None or b is None:
        return None
    try:
        return geometry.whitened_mask_iou(a, kf_p.moments, kf_p.front_sign, b, kf_r.moments, kf_r.front_sign)
    except ValueError:
        return None


def _frames_agree(kf_ref: Keyframe, kfs: list[Keyframe], calibration: Calibration, th: dict) -> bool:
    """Every keyframe whose marks are compared took axis 1 within the rotation the band absorbs of the paired
    pickup keyframe's. A frame turned by dtheta moves a mark at radius r by about r dtheta, and r dtheta <=
    tau_pre (1 + k r^2) holds at every r only while dtheta <= 2 tau_pre sqrt(k) (k = band_radius_factor). The
    90-degree jump of the nearest-horizontal axis rule (principal axes near 45 degrees) and a noise-driven theta on
    a near-isotropic mask both fail it, while the whitened IoU of a symmetric shape stays high through either."""
    if kf_ref.moments is None:
        return True
    limit = 2.0 * calibration.tau_pre * math.sqrt(float(th["matching"]["band_radius_factor"]))
    ref = geometry.axis_angle(kf_ref.moments)
    return all(abs(geometry.axis_angle(kf.moments) - ref) <= limit for kf in kfs if kf.moments is not None)


def _compared_keyframes(kf_p: Keyframe, kf_r: Keyframe, kfs: dict[str, list[Keyframe]], panel_marks: dict[str, list[Mark]]) -> list[Keyframe]:
    """The paired keyframes plus every keyframe a placed mark of the panel was whitened in (medoids come from any)."""
    used = {kf_p.kf_id, kf_r.kf_id} | {m.kf_id for role in ROLES for m in panel_marks[role] if m.w is not None}
    return [kf for role in ROLES for kf in kfs[role] if kf.kf_id in used]


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


def _provisional_already_there(pair: MatchCandidate, a: Mark, b: Mark, mark_tracks: dict[str, MarkTrack], kf_status: dict[str, MarksStatus], *, deferred: tuple[str, ...] = ()) -> bool:
    """Already there before the second look, with the deferred caps (K4 and K12 under rule 1) left out: they wait for
    the look, so they alone never keep a pair from it."""
    flags = flags_for(pair, a, b, mark_tracks, None, kf_status, return_mark=False)
    if deferred:
        flags = flags.model_copy(update={k: False for k in deferred})
    label, _ = verdict(pair.d, pair.tau_pre_r, pair.tau_eye_r, flags)
    return label == VerdictLabel.ALREADY_THERE


def _pair_gates(
    run_id: str,
    panel: Panel,
    kf_p: Keyframe,
    kf_r: Keyframe,
    kfs: dict[str, list[Keyframe]],
    panel_marks: dict[str, list[Mark]],
    *,
    lane_status: dict[str, dict[Noun, LaneStatus]],
    truncated_all: dict[str, bool],
    calibration: Calibration,
    th: dict,
    masks: dict[str, np.ndarray | None],
) -> KeyframePairTry:
    """One keyframe pair of the panel put through the gates: the failing ids, the whitened mask IoU (None when the
    frames disagree) and whether they agreed."""
    mask_iou = _mask_iou(run_id, kf_p, kf_r, masks)
    agree = panel.panel_id in WHEEL_PANELS or _frames_agree(kf_p, _compared_keyframes(kf_p, kf_r, kfs, panel_marks), calibration, th)
    if not agree:
        # TODO(contracts): G4 stands in for "the whitened frames disagree" until that gate has its own id;
        # the IoU is dropped because it cannot vouch for frames that picked different axes.
        mask_iou = None
    residual = handle_residual(kf_p, kf_r)
    gates = panel_gates(panel, kf_p, kf_r, mask_iou=mask_iou, residual=residual, lane_status=lane_status, marks_truncated_all=truncated_all, calibration=calibration, th=th)
    return KeyframePairTry(kf_pair=[kf_p.kf_id, kf_r.kf_id], gates=gates, mask_iou=mask_iou, frames_agree=agree)


def derive_match(run_id: str, *, th: dict, second_look: Callable[[MatchCandidate], LookVerdict | None] | None = None) -> tuple[list[PanelMatch], list[Verdict], Calibration]:
    """Read panels, keyframes, marks, mark_tracks and calibration; gate, pair and match every panel with a keyframe
    in either walk; ask the optional second look on provisional already_there pairs (at most
    second_looks_per_pair_max); write matches.json and verdicts.json. Under a second without the second look.

    With thresholds matching.next_keyframe_pair (rule 2) a panel whose chosen keyframe pair fails only view gates is
    compared on the first pair in keyframe_pair_order that passes every gate, and kf_pairs_tried records each pair
    tried; when none passes, the panel keeps the chosen pair and its gates. With matching.tight_pair_second_look
    (rule 1) K4 and K12 alone never keep a pair inside tau_pre_r from the second look."""
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
    retry = next_keyframe_pair(th)
    deferred = TYPE_CAPS if tight_pair_look(th) else ()
    masks: dict[str, np.ndarray | None] = {}
    matches: list[PanelMatch] = []
    verdicts: list[Verdict] = []
    for panel in panels:
        kfs = {role: [kf for kf in keyframes if kf.panel_id == panel.panel_id and kf.role == role] for role in ROLES}
        if not kfs["pickup"] and not kfs["return"]:
            continue
        panel_marks = {role: [m for m in marks if m.panel_id == panel.panel_id and m.role == role] for role in ROLES}
        truncated_all = {role: bool(kfs[role]) and all(kf.marks_status == MarksStatus.TRUNCATED for kf in kfs[role]) for role in ROLES}
        if kfs["pickup"] and kfs["return"]:
            order = keyframe_pair_order(kfs["pickup"], kfs["return"], th=th)
            inputs: dict[str, Any] = {"lane_status": lane_status, "truncated_all": truncated_all, "calibration": calibration, "th": th, "masks": masks}
            kf_p, kf_r = order[0]
            used = _pair_gates(run_id, panel, kf_p, kf_r, kfs, panel_marks, **inputs)
            tries = [used]
            if retry and used.gates and set(used.gates) <= set(VIEW_GATES):
                for cand_p, cand_r in order[1:]:
                    cand = _pair_gates(run_id, panel, cand_p, cand_r, kfs, panel_marks, **inputs)
                    tries.append(cand)
                    if not cand.gates:
                        kf_p, kf_r, used = cand_p, cand_r, cand
                        break
            pm = match_panel(panel, panel_marks["pickup"], panel_marks["return"], whitened, (kf_p, kf_r), list(used.gates), calibration, th)
            pm = pm.model_copy(update={"mask_iou": used.mask_iou, "kf_pairs_tried": tries if retry else []})
        else:
            gates = panel_gates(panel, kfs["pickup"][0] if kfs["pickup"] else None, kfs["return"][0] if kfs["return"] else None, mask_iou=None, residual=None, lane_status=lane_status, marks_truncated_all=truncated_all, calibration=calibration, th=th)
            pm = PanelMatch(
                panel_id=panel.panel_id,
                state=MatchState.NOT_COMPARABLE,
                reasons=gates,
                unmatched_pickup=[m.mark_id for m in panel_marks["pickup"]],
                unmatched_return=[m.mark_id for m in panel_marks["return"]],
            )
        looks: dict[tuple[str, str], LookVerdict | None] = {}  # None: asked, no usable answer
        if second_look is not None and pm.state == MatchState.COMPARABLE:
            kf_status = {kf.kf_id: kf.marks_status for kf in keyframes if kf.panel_id == pm.panel_id}
            for pair in pm.pairs:
                a, b = marks_by_id.get(pair.pickup_mark), marks_by_id.get(pair.return_mark)
                if a is None or b is None or looks_used >= looks_budget:
                    continue
                if _provisional_already_there(pair, a, b, staysput_caps(mark_tracks, th), kf_status, deferred=deferred):
                    looks[(pair.pickup_mark, pair.return_mark)] = second_look(pair)
                    looks_used += 1
        matches.append(pm)
        verdicts.extend(verdicts_for(pm, marks_by_id, mark_tracks, looks, kf_by_id, calibration, trace_base, th))
    write_json(paths.matches_json(run_id), matches)
    write_json(paths.verdicts_json(run_id), verdicts)
    return matches, verdicts, calibration


# Older runs: why verify differs --------------------------------------------------------------------------------------

RULE_CHANGE_1031 = (
    "written before the Oct 31 rule change (a wet, dirty or glare view no longer makes a panel not comparable, G2 is "
    "retired; it caps a mark seen in one walk only, K15, and the keyframe pair prefers a clean view), and verify "
    "rebuilds with today's code"
)


def _stored(path: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stored_rows(path: Any) -> list[dict[str, Any]]:
    doc = _stored(path)
    return [row for row in doc if isinstance(row, dict)] if isinstance(doc, list) else []


def _stored_keyframes(root: Any) -> list[Keyframe]:
    out: list[Keyframe] = []
    for row in _stored_rows(root / "derived" / "keyframes.json"):
        try:
            out.append(Keyframe.model_validate(row))
        except ValueError:
            continue
    return out


def _paired_before_1031(root: Any, matches: list[dict[str, Any]]) -> bool:
    """A stored keyframe pair that today's clean-view preference would not pick (keyframes.json is frozen). Under the
    next-pair rule (11/1) the pair chosen is the first one tried, not the pair compared on."""
    keyframes = _stored_keyframes(root)
    run = _stored(root / "run.json")
    th = run.get("thresholds") if isinstance(run, dict) else None
    for pm in matches:
        tried = pm.get("kf_pairs_tried")
        first = tried[0] if isinstance(tried, list) and tried and isinstance(tried[0], dict) else None
        stored = first.get("kf_pair") if first is not None else pm.get("kf_pair")
        walks = [[kf for kf in keyframes if kf.panel_id.value == pm.get("panel_id") and kf.role == role] for role in ROLES]
        if isinstance(stored, list) and len(stored) == 2 and all(walks):
            kf_p, kf_r, _ = pair_keyframes(walks[0], walks[1], th=th or None)
            if [kf_p.kf_id, kf_r.kf_id] != stored:
                return True
    return False


def _flagged_without_note(root: Any, verdicts: list[dict[str, Any]]) -> bool:
    """A stored verdict on a flagged view without view_note: today's code writes one on every such verdict."""
    flagged = {kf.kf_id for kf in _stored_keyframes(root) if view_flags(kf)}
    kf_of = {m.get("mark_id"): m.get("kf_id") for m in _stored_rows(root / "derived" / "marks.json")}
    pairs = {pm.get("panel_id"): pm.get("kf_pair") or [] for pm in _stored_rows(root / "derived" / "matches.json")}
    for v in verdicts:
        if v.get("label") == VerdictLabel.NOT_COMPARABLE.value or "view_note" in v:
            continue
        if flagged & {*pairs.get(v.get("panel_id"), []), *(kf_of.get(i) for i in v.get("mark_ids") or [])}:
            return True
    return False


def why_rebuild_differs(run_id: str, name: str) -> str | None:
    """Why a run's stored file differs from today's rebuild when the 10/31 view-flag change explains it, read from the
    run's own files: derived/matches.json naming G2 or holding a keyframe pair the clean-view preference would not
    pick; derived/verdicts.json naming G2 or holding a verdict on a flagged view without view_note; sheet/sheet.json
    naming G2 in its not-comparable list or with a glossary that does not name K15. None otherwise, so a tampered or
    otherwise changed file keeps the bare "rebuild differs" line. (A run with no view flag keeps its matches and
    verdicts bytes: view_note is left out of the file when there is none.)"""
    root = paths.runs_root() / run_id
    if name == "derived/matches.json":
        rows = _stored_rows(root / name)
        old = any("G2" in (pm.get("reasons") or []) for pm in rows) or _paired_before_1031(root, rows)
    elif name == "derived/verdicts.json":
        rows = _stored_rows(root / name)
        old = any("G2" in ((v.get("trace") or {}).get("rule_ids") or []) for v in rows) or _flagged_without_note(root, rows)
    elif name == "sheet/sheet.json":
        doc = _stored(root / name)
        doc = doc if isinstance(doc, dict) else {}
        listed = [nc for nc in doc.get("not_comparable") or [] if isinstance(nc, dict)]
        glossary = (doc.get("appendices") or {}).get("B glossary") if doc else None
        old = bool(doc) and (any("G2" in (nc.get("rule_ids") or []) for nc in listed) or "K15" not in str(glossary or ""))
    else:
        old = False
    return RULE_CHANGE_1031 if old else None
