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
- matching.fixtures_not_damage (lead decision 4): a mark whose box sits on a factory fixture the run already has (a
  fixture landmark Muse Spark pointed at in the same crop, or the SAM door-handle point of its keyframe) by at least
  matching.fixture_overlap_min of its box is left out of matching and named in how_checked (PanelMatch.fixtures).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import geometry, paths, rules
from .contracts import (
    LOWER_BUMPERS,
    WHEEL_PANELS,
    ByEyeLine,
    Calibration,
    CopyNote,
    Confidence,
    CoverageState,
    FixtureDrop,
    FrameUsed,
    Keyframe,
    KeyframePairTry,
    ReshapeCheck,
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
    OffPanelNote,
    OwnContrast,
    Panel,
    PanelId,
    PanelMatch,
    PanelTrack,
    PrivacyDrop,
    RegistrationFit,
    RegistrationVouch,
    RegistrationMark,
    SameWalkNote,
    SpotCheck,
    SparkPurpose,
    SparkRequestRecord,
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
VIEW_GATES = ("G3", "G4", "G5", "G8", "G9", "G11", "G12", "G13")  # gates of the keyframe pair: another pair of the panel may pass them (rule 2)
PANEL_GATES = ("G1", "G6", "G7", "G10")  # gates of the panel: no other keyframe pair changes them, so none is tried
TYPE_CAPS = ("K4", "K12")  # what Muse Spark typed and how sure it was: notes on a tight pair the second look answered (rule 1)
TYPE_CAPS_K5 = ("K4", "K5", "K12")  # rule 1 with matching.tight_pair_k5 (lead 11/1, second): the two types disagreeing waits too
PANEL_VERDICT_RULES = ("L6", "L7")  # one verdict for the whole panel, naming every mark of it: no size, no mark pair


def tight_pair_look(th: dict | None) -> bool:
    """Rule 1 of 11/1 is on (thresholds matching.tight_pair_second_look); off when absent, as in runs frozen before."""
    return bool((th or {}).get("matching", {}).get("tight_pair_second_look", False))


def tight_pair_deferrable(th: dict | None) -> tuple[str, ...]:
    """The caps rule 1 defers on a pair inside tau_pre_r: none while it is off; K4 and K12 (runs frozen with
    tight_pair_second_look but without tight_pair_k5, pairC_core3 among them); K4, K5 and K12 with both leaves on."""
    if not tight_pair_look(th):
        return ()
    return TYPE_CAPS_K5 if (th or {}).get("matching", {}).get("tight_pair_k5", False) else TYPE_CAPS


def reshaped_panel(th: dict | None) -> bool:
    """The reshaped-panel rule (thresholds matching.reshaped_panel, lead 11/1, second); off when absent."""
    return bool((th or {}).get("matching", {}).get("reshaped_panel", False))


def one_walk_counter_look(th: dict | None) -> bool:
    """The one-walk counter-look (thresholds matching.one_walk_counter_look, lead 11/1, third); off when absent, as in
    every run frozen before it."""
    return bool((th or {}).get("matching", {}).get("one_walk_counter_look", False))


def counter_looks_max(th: dict | None) -> int:
    """At most this many counter-looks per run (thresholds matching.counter_looks_max), surest marks first."""
    return int((th or {}).get("matching", {}).get("counter_looks_max", 0))


def counter_confirms_mark(th: dict | None) -> bool:
    """Lead decision 5 of 11/1 (thresholds matching.counter_confirms_mark): the counter-look also asks whether the mark
    is a damage mark at all, so new needs a confirmed mark. Only with one_walk_counter_look; off when absent."""
    return one_walk_counter_look(th) and bool((th or {}).get("matching", {}).get("counter_confirms_mark", False))


PAIR_SCALE_DEFAULTS: dict[str, Any] = {"enabled": True, "max_ratio": 2.0, "prefer_ratio": 1.5}
PRIVACY_BLUR_DEFAULTS: dict[str, Any] = {"enabled": True, "view_max": 0.2, "mark_max": 0.5}


TWIN_PANELS = (PanelId.HEADLIGHTS, PanelId.TAIL_LIGHTS, PanelId.MIRRORS)
TWIN_MIRROR_DEFAULTS: dict[str, Any] = {"enabled": True, "margin": 0.1, "grid": 64}


def twin_mirror(th: dict | None) -> dict[str, Any] | None:
    """thresholds matching.twin_mirror (R27) with defaults filled; None when absent or off (runs frozen before it)."""
    raw = (th or {}).get("matching", {}).get("twin_mirror")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    return {**TWIN_MIRROR_DEFAULTS, **raw}


def mirror_ious(mask_p: np.ndarray | None, mask_r: np.ndarray | None, grid: int = 64) -> tuple[float, float] | None:
    """(IoU as they are, IoU with the return mask mirrored left to right), each mask cut to its box and scaled to a
    grid x grid square. A left and a right headlight are mirror images: on hero_focus4 the same lamp gave 0.80 and 0.73
    as they are (0.35 and 0.34 mirrored), the other lamp 0.45 as it is and 0.91 mirrored."""
    import cv2

    if mask_p is None or mask_r is None or not mask_p.any() or not mask_r.any():
        return None

    def norm(m: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(m)
        sub = m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8)
        return cv2.resize(sub, (grid, grid), interpolation=cv2.INTER_NEAREST) > 0

    a, b = norm(mask_p), norm(mask_r)

    def iou(x: np.ndarray, y: np.ndarray) -> float:
        return float((x & y).sum()) / float(max(1, (x | y).sum()))

    return round(iou(a, b), 4), round(iou(a, b[:, ::-1]), 4)


def pair_scale(th: dict | None) -> dict[str, Any] | None:
    """thresholds matching.pair_scale (R27) with defaults filled; None when absent or off (runs frozen before it)."""
    raw = (th or {}).get("matching", {}).get("pair_scale")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    return {**PAIR_SCALE_DEFAULTS, **raw}


def privacy_blur(th: dict | None) -> dict[str, Any] | None:
    """thresholds matching.privacy_blur (R27) with defaults filled; None when absent or off (runs frozen before it)."""
    raw = (th or {}).get("matching", {}).get("privacy_blur")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    return {**PRIVACY_BLUR_DEFAULTS, **raw}


def fixtures_not_damage(th: dict | None) -> bool:
    """Lead decision 4 of 11/1 (thresholds matching.fixtures_not_damage): factory fixtures are not damage, in the marks
    prompt and in matching; off when absent, as in every run frozen before it."""
    return bool((th or {}).get("matching", {}).get("fixtures_not_damage", False))


def seams_not_marks(th: dict | None) -> bool:
    """Lead 11/1 (thresholds matching.seams_not_marks): the marks prompt adds spark_client.SEAMS_SENTENCE (seams, shading
    and reflections are not marks); off when absent, as in every run frozen before it."""
    return bool((th or {}).get("matching", {}).get("seams_not_marks", False))


def colour_on_paint(th: dict | None) -> bool:
    """Lead Nov 2 (thresholds matching.colour_on_paint): the marks prompt also asks for a stripe or patch of a clearly
    different colour on the paint (paint transfer, tape, a sticker) typed scuff, and the two-question counter-look
    counts one as a mark in the box. On hero_focus1 the old prompt (scratch, scuff, chip, dent and crack only) boxed none
    of the tape strips Sanat put on the paint in the pickup, and the one it boxed at return came back unsure. Off when
    absent."""
    leaf = (th or {}).get("matching", {}).get("colour_on_paint")
    return bool(leaf.get("enabled")) if isinstance(leaf, dict) else bool(leaf)


def conditions_on_panel(th: dict | None) -> bool:
    """Lead Nov 2 (thresholds matching.conditions_on_panel): the marks prompt asks for the wet, dirty, glare and dark
    flags of the named panel itself, not of the whole crop, so leaves on the windshield in a door crop do not flag the
    door (K15 caps that door's one-walk marks). Off when absent."""
    leaf = (th or {}).get("matching", {}).get("conditions_on_panel")
    return bool(leaf.get("enabled")) if isinstance(leaf, dict) else bool(leaf)


def inside_other_box(th: dict | None) -> bool:
    """Nov 2 (thresholds matching.inside_other_box): a mark seen in one walk only whose position lies inside a box the
    other walk drew in its compared keyframe is capped (K19). Off when absent."""
    leaf = (th or {}).get("matching", {}).get("inside_other_box")
    return bool(leaf.get("enabled")) if isinstance(leaf, dict) else bool(leaf)


def _compared_frame_box(m: Mark, kf: Keyframe | None, residual: Point2 | None) -> list[tuple[float, float]] | None:
    """The four corners of the mark's box whitened in its own keyframe, in the frame the match compares in."""
    if m.bbox_crop is None or kf is None or kf.moments is None or kf.front_sign is None:
        return None
    du, dv = (float(residual[0]), float(residual[1])) if residual is not None and m.role == "return" else (0.0, 0.0)
    x1, y1, x2, y2 = (float(v) for v in m.bbox_crop)
    try:
        corners = [geometry.whiten(p, kf.moments, kf.front_sign) for p in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
    except (ValueError, np.linalg.LinAlgError):
        return None
    return [(float(u) - du, float(v) - dv) for u, v in corners]


COVERED_SHARE_MIN = 0.5  # K19: at least this share of the mark's whitened box lies inside the other walk's box


def _covered_share(poly: list[tuple[float, float]], other: list[tuple[float, float]]) -> float:
    """The share of poly's area inside other (both whitened boxes, convex quadrilaterals)."""
    import cv2

    a = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    b = np.asarray(other, dtype=np.float32).reshape(-1, 1, 2)
    own = abs(float(cv2.contourArea(a)))
    if own <= 0.0:
        return 0.0
    if cv2.contourArea(a, oriented=True) < 0:
        a = a[::-1].copy()
    if cv2.contourArea(b, oriented=True) < 0:
        b = b[::-1].copy()
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(inter) / own


def covered_by_other_walk(pm: PanelMatch, marks: dict[str, Mark], keyframes: dict[str, Keyframe]) -> dict[str, str]:
    """Pure (Nov 2, K19): each mark of the panel seen in one walk only (unmatched, or in a pair beyond tau_eye_r) whose
    whitened box, in the frame the match compares in, lies at least COVERED_SHARE_MIN inside the whitened box of a
    mark the other walk has in its compared keyframe, mapped to that mark's id. On eval_nhtsa15759_r16 with the colour
    sentence, the pickup's tape stripes on the left front door had no return box of their own because the return
    answer boxed the whole crush as one mark over them, and each came out not seen at return. The share, not the
    centre, decides: the crush itself, whose centre falls on a thin pickup tape box, stays new."""
    if pm.state != MatchState.COMPARABLE or not pm.kf_pair:
        return {}
    residual = pm.handle_residual
    one_walk = [*pm.unmatched_pickup, *pm.unmatched_return, *(i for p in pm.pairs if p.d >= p.tau_eye_r for i in (p.pickup_mark, p.return_mark))]
    panel_ids = {*pm.unmatched_pickup, *pm.unmatched_return, *(i for p in pm.pairs for i in (p.pickup_mark, p.return_mark))}
    compared = dict(zip(ROLES, pm.kf_pair))
    boxes: dict[str, list[tuple[str, list[tuple[float, float]]]]] = {role: [] for role in ROLES}
    for i in sorted(panel_ids):
        m = marks.get(i)
        if m is None or m.kf_id != compared.get(m.role):
            continue
        poly = _compared_frame_box(m, keyframes.get(m.kf_id), residual)
        if poly is not None:
            boxes[m.role].append((i, poly))
    out: dict[str, str] = {}
    for i in one_walk:
        m = marks.get(i)
        own = _compared_frame_box(m, keyframes.get(m.kf_id), residual) if m is not None else None
        if own is None:
            continue
        other = "return" if m.role == "pickup" else "pickup"
        hit = next((j for j, poly in boxes[other] if _covered_share(own, poly) >= COVERED_SHARE_MIN), None)
        if hit is not None:
            out[i] = hit
    return out


STANDS_OUT_DEFAULTS: dict[str, Any] = {
    "min_contrast": 3.0,  # register.DEFAULTS saliency_min: under this the presence test calls a box too faint to test
    "max_panel_share": 0.10,  # a box over this share of its panel's mask is an area (a crush, a dirt film), not tested
}


def stands_out_config(th: dict | None) -> dict[str, Any] | None:
    """Nov 2 (thresholds matching.stands_out, K20): the leaf over STANDS_OUT_DEFAULTS, with the presence test's blur
    and ring (matching.registration when on, else register.DEFAULTS); None when absent or off."""
    from . import register

    leaf = (th or {}).get("matching", {}).get("stands_out")
    if not isinstance(leaf, dict) or not leaf.get("enabled", False):
        return None
    return {**(register.config(th) or register.DEFAULTS), **STANDS_OUT_DEFAULTS, **leaf}


def one_walk_ids(pm: PanelMatch) -> list[str]:
    """The panel's marks seen in one walk only: unmatched, or in a pair beyond tau_eye_r (L5)."""
    return [*pm.unmatched_pickup, *pm.unmatched_return, *(i for p in pm.pairs if p.d >= p.tau_eye_r for i in (p.pickup_mark, p.return_mark))]


def stands_out_step(pm: PanelMatch, placed: dict[str, Mark], cache: "RegistrationCache", cfg: dict[str, Any]) -> PanelMatch:
    """Nov 2 (matching.stands_out, K20), on a compared panel after the match and registration: each mark seen in one
    walk only is tested on its own crop and box as Muse Spark placed them (placed: the run's marks before the same-walk
    step re-placed any), by register.own_contrast. Under min_contrast, or a box too small to measure, is faint (K20);
    a box over max_panel_share of its keyframe's panel mask is an area, not tested; a crop or mask that cannot be read
    is not_measured. On hero_focus2 a stray pickup box on clean rear bumper paint stood out by 0.95 and came out not
    seen at return; the tape strips it sat near stand out by 15 to 28. Pixels only, never a model; it lowers only."""
    from . import register

    if pm.state != MatchState.COMPARABLE:
        return pm
    out: list[OwnContrast] = []
    for i in dict.fromkeys(one_walk_ids(pm)):
        m = placed.get(i)
        if m is None:
            continue
        px = cache.pixels(m.kf_id) if m.bbox_crop is not None else None
        if px is None:
            out.append(OwnContrast(mark_id=i, kf_id=m.kf_id, result="not_measured", why="no box, crop or mask to read"))
            continue
        img, mask = px
        box = tuple(float(v) for v in m.bbox_crop)  # type: ignore[union-attr]
        h, w = img.shape[:2]
        bw, bh = max(0.0, min(float(w), box[2]) - max(0.0, box[0])), max(0.0, min(float(h), box[3]) - max(0.0, box[1]))
        area = float(np.count_nonzero(mask))
        share = round(bw * bh / area, 4) if area > 0 else None
        if share is not None and share > float(cfg["max_panel_share"]):
            out.append(OwnContrast(mark_id=i, kf_id=m.kf_id, result="area", panel_share=share))
            continue
        c = register.own_contrast(img, box, cfg)  # type: ignore[arg-type]
        if c is None:
            out.append(OwnContrast(mark_id=i, kf_id=m.kf_id, result="faint", panel_share=share, why="the box is too small to measure"))
            continue
        c = round(c, 2)
        out.append(OwnContrast(mark_id=i, kf_id=m.kf_id, result="stands_out" if c >= float(cfg["min_contrast"]) else "faint", contrast=c, panel_share=share))
    return pm.model_copy(update={"own_contrast": out}) if out else pm


def stands_out_words(pm: PanelMatch) -> str | None:
    """how_checked's part for K20: each faint mark with its number."""
    faint = [r for r in pm.own_contrast if r.result == "faint"]
    if not faint:
        return None
    off = set(pm.left_off)
    return "; ".join(
        f"{r.mark_id}: its box does not stand out from the paint around it in its own crop"
        + (f" (contrast {r.contrast:.1f})" if r.contrast is not None else "")
        + (", so it is left off the list (K20)" if r.mark_id in off else ", so needs your eye (K20)")
        for r in faint
    )


def same_walk_config(th: dict | None) -> dict[str, Any] | None:
    """Nov 2 (thresholds matching.same_walk_register): the registration numbers (the matching.registration leaf with
    register.DEFAULTS filled, or the defaults alone) when the leaf is on; None when absent or off."""
    leaf = (th or {}).get("matching", {}).get("same_walk_register")
    if not (leaf.get("enabled") if isinstance(leaf, dict) else leaf):
        return None
    from . import register

    return register.config(th) or {**register.DEFAULTS}


def same_walk_step(
    panel_marks: dict[str, list[Mark]], compared: dict[str, Keyframe], cache: "RegistrationCache", th: dict
) -> tuple[dict[str, list[Mark]], list[Mark], list[SameWalkNote]]:
    """Nov 2 (matching.same_walk_register), on a compared non-wheel panel before the match: in each walk, every mark of
    another keyframe of that walk is carried onto the walk's compared keyframe through a registration fit of the two
    crops (register.align, the same fit as across walks; within one walk the light and view barely change). A carried
    box whose centre lands inside, or within register.coincide_px of, a mark already on the compared keyframe is that
    mark seen again and leaves matching (merged_into); any other is re-placed there (kf_id, box and point mapped,
    whitened in the compared keyframe's frame) so every mark of the walk is compared in one frame. A mark whose fit is
    refused, or whose box lands off the compared crop, stays as it was. On hero_focus1 one tape strip sat 0.4 to 0.9
    panel units apart in the three pickup keyframes of the rear bumper (its SAM mask changed shape between them), far
    over the 0.2 merge radius, so the copies stayed apart and none paired. A mark another keyframe's copy merged into
    loses K2 (single frame): it shows in two keyframes. Returns the walks' marks, the marks changed (re-placed, or K2
    lifted) and the notes."""
    from . import register
    from .marks import whiten_marks

    out: dict[str, list[Mark]] = {role: list(panel_marks.get(role, [])) for role in ROLES}
    moved: list[Mark] = []
    notes: list[SameWalkNote] = []
    for role in ROLES:
        k = compared.get(role)
        if k is None or k.moments is None or k.front_sign is None or k.crop is None:
            continue
        kept = [m for m in out[role] if m.kf_id == k.kf_id]
        anchors = [(m.mark_id, tuple(float(v) for v in m.bbox_crop)) for m in kept if m.bbox_crop is not None]
        seen_again: set[str] = set()
        others = sorted({m.kf_id for m in out[role] if m.kf_id != k.kf_id}, key=lambda i: (_kf_index(cache.kf_by_id[i]) if i in cache.kf_by_id else (1 << 30, i)))
        for kf_id in others:
            ms = [m for m in out[role] if m.kf_id == kf_id]
            al = cache.fit(kf_id, k.kf_id)
            if al is None or not al.ok or al.H is None:
                kept.extend(ms)
                continue
            reproj = None if not math.isfinite(al.reproj_px) else round(float(al.reproj_px), 3)
            for m in ms:
                if m.bbox_crop is None:
                    kept.append(m)
                    continue
                box = register.map_box(al.H, tuple(float(v) for v in m.bbox_crop))  # type: ignore[arg-type]
                cx, cy = _centre(box)
                if not (0.0 <= cx < k.crop.w and 0.0 <= cy < k.crop.h):
                    kept.append(m)
                    continue
                hit = None
                best = float("inf")
                for aid, ab in anchors:
                    ax, ay = _centre(ab)
                    dist = math.hypot(cx - ax, cy - ay)
                    inside = ab[0] <= cx <= ab[2] and ab[1] <= cy <= ab[3]
                    if (inside or dist <= register.coincide_px(ab, cache.cfg)) and dist < best:
                        hit, best = aid, dist
                if hit is not None:
                    notes.append(SameWalkNote(mark_id=m.mark_id, from_kf=kf_id, to_kf=k.kf_id, merged_into=hit, model=str(al.model), reproj_px=reproj))
                    seen_again.add(hit)
                    continue
                clipped = [max(0.0, box[0]), max(0.0, box[1]), min(float(k.crop.w), box[2]), min(float(k.crop.h), box[3])]
                placed = m.model_copy(update={"kf_id": k.kf_id, "bbox_crop": [round(v, 3) for v in clipped], "point_crop": [round(cx, 3), round(cy, 3)]})
                [placed], _ = whiten_marks([placed], k, th=th)
                kept.append(placed)
                moved.append(placed)
                anchors.append((placed.mark_id, tuple(clipped)))  # type: ignore[arg-type]
                notes.append(SameWalkNote(mark_id=m.mark_id, from_kf=kf_id, to_kf=k.kf_id, model=str(al.model), reproj_px=reproj))
        # K2 (single frame) said the mark showed in one keyframe only; a copy registered onto it shows it in two
        lifted = [m.model_copy(update={"caps": [c for c in m.caps if c != "K2"]}) if m.mark_id in seen_again and "K2" in m.caps else m for m in kept]
        moved.extend(m for m, was in zip(lifted, kept) if m is not was and m not in moved)
        out[role] = lifted
    return out, moved, notes


def same_walk_words(pm: PanelMatch) -> str | None:
    """how_checked's part for the same-walk step: each mark merged or carried, with the fit."""
    if not pm.same_walk:
        return None
    parts = []
    for n in pm.same_walk:
        if n.merged_into:
            parts.append(f"{n.mark_id} is {n.merged_into} seen again in {n.from_kf} (registered onto {n.to_kf}, {n.model})")
        else:
            parts.append(f"{n.mark_id} carried from {n.from_kf} onto {n.to_kf} ({n.model})")
    return "same walk: " + "; ".join(parts)


GLASS_LINE = "glass: check by eye; reflections make marks on glass unreliable"
WHEELS_LINE = "wheels: check by eye; rims and tires are not checked mark by mark"
DEBRIS_LINE = "debris on this panel; check by eye"


def glass_by_eye(th: dict | None) -> frozenset[PanelId]:
    """Lead Nov 2 (thresholds matching.glass_by_eye {"enabled", "panels"}): the glass panels whose marks are left out
    of matching and that get one check-by-eye line (GLASS_LINE) instead of verdicts. Side windows are part of the door
    crops, whose masks cut the window out (keyframe window_subtracted), so the on-panel rule already leaves their boxes
    out. Empty when absent or off."""
    leaf = (th or {}).get("matching", {}).get("glass_by_eye")
    if not isinstance(leaf, dict) or not leaf.get("enabled"):
        return frozenset()
    return frozenset(PanelId(p) for p in leaf.get("panels", [PanelId.WINDSHIELD.value, PanelId.REAR_WINDOW.value]))


def wheels_by_eye(th: dict | None) -> frozenset[PanelId]:
    """R30 (lead decision of Nov 3, car body only; thresholds matching.wheels_by_eye {"enabled", "panels"}): the wheel
    panels (rims and tires) whose marks are left out of matching and that get one check-by-eye line (WHEELS_LINE)
    instead of verdicts, as glass does. Wheel tracking, keyframes and crops stay: the fenders are built from the wheels.
    Empty when absent or off."""
    leaf = (th or {}).get("matching", {}).get("wheels_by_eye")
    if not isinstance(leaf, dict) or not leaf.get("enabled"):
        return frozenset()
    return frozenset(PanelId(p) for p in leaf.get("panels", [p.value for p in WHEEL_PANELS]))


def by_eye_panels(th: dict | None) -> dict[PanelId, tuple[str, str]]:
    """The panels that get one check-by-eye line and no verdict (glass, then the wheels when R30's leaf is on): panel
    to (line kind, line text)."""
    out = {p: ("glass", GLASS_LINE) for p in glass_by_eye(th)}
    out.update({p: ("wheels", WHEELS_LINE) for p in wheels_by_eye(th) if p not in out})
    return out


def debris_by_eye(th: dict | None) -> frozenset[MarkKind]:
    """Lead Nov 2 (thresholds matching.debris_by_eye {"enabled", "types"}): the mark types whose needs-your-eye lines on a
    flagged view fold into one DEBRIS_LINE per panel (debris_split). Empty when absent or off."""
    leaf = (th or {}).get("matching", {}).get("debris_by_eye")
    if not isinstance(leaf, dict) or not leaf.get("enabled"):
        return frozenset()
    return frozenset(MarkKind(t) for t in leaf.get("types", [MarkKind.DIRT.value]))


def debris_split(pm: PanelMatch, verdicts: list[Verdict], marks: dict[str, Mark], keyframes: dict[str, Keyframe], th: dict) -> tuple[PanelMatch, list[Verdict]]:
    """Pure (lead Nov 2, matching.debris_by_eye): on a compared panel, a needs-your-eye verdict whose every mark Muse Spark
    typed one of the leaf's types (dirt) and whose views are flagged (the views K15 reads: the compared keyframe pair
    and the keyframes its marks were placed in, wet, dirty or glare) leaves the verdict list and its marks go into one
    ByEyeLine of kind debris on the panel. A verdict holding any other type (a damage type or unsure) stays, so no mark
    Spark called damage or could not tell is hidden. The kept verdicts are numbered again in order."""
    types = debris_by_eye(th)
    if not types or pm.state != MatchState.COMPARABLE or not verdicts:
        return pm, verdicts
    paired = [keyframes.get(k) for k in (pm.kf_pair or [])]

    any_view = debris_any_view(th)

    def folds(v: Verdict) -> bool:
        involved = [marks.get(i) for i in v.mark_ids]
        if v.label != VerdictLabel.NEEDS_YOUR_EYE or not involved or any(m is None or m.type not in types for m in involved):
            return False
        if any_view and len(involved) == 1:
            return True  # R28: a mark seen in one walk only folds whether or not its views are flagged
        views = [*paired, *(keyframes.get(m.kf_id) for m in involved if m is not None)]
        return any(view_flags(kf) for kf in views)

    folded = [v for v in verdicts if folds(v)]
    if not folded:
        return pm, verdicts
    kept = [v.model_copy(update={"verdict_id": f"{pm.panel_id.value}_v{n}"}) for n, v in enumerate((v for v in verdicts if not folds(v)), 1)]
    line = ByEyeLine(kind="debris", text=DEBRIS_LINE, mark_ids=sorted({i for v in folded for i in v.mark_ids}), any_view=True if any_view else None)
    return pm.model_copy(update={"by_eye": [*pm.by_eye, line]}), kept


def by_eye_words(pm: PanelMatch | None) -> list[str]:
    """The sheet's check-by-eye lines for a panel: the line, then how many boxes it stands for."""
    out = []
    for line in pm.by_eye if pm is not None else []:
        n = len(line.mark_ids)
        if line.kind in ("glass", "wheels"):
            out.append(f"{line.text} ({n} spot{'' if n == 1 else 's'} Muse Spark boxed on it, not judged)" if n else line.text)
        elif line.any_view:
            out.append(f"{line.text} ({n} spot{'' if n == 1 else 's'} Muse Spark typed dirt, not listed one by one)")
        else:
            out.append(f"{line.text} ({n} spot{'' if n == 1 else 's'} Muse Spark typed dirt on a flagged view, not listed one by one)")
    return out


def debris_any_view(th: dict | None) -> bool:
    """R28 (lead decision 2 of 11/2, matching.debris_by_eye any_view): a mark seen in one walk only that Muse Spark typed
    dirt folds into the debris line whether or not its views are flagged. False when absent."""
    leaf = (th or {}).get("matching", {}).get("debris_by_eye")
    return isinstance(leaf, dict) and bool(leaf.get("enabled")) and bool(leaf.get("any_view"))


# R28: same-walk copies, faint marks left off, the mirrors' sides ---------------------------------------------------

SIDE_SIGN = {"left": -1, "right": 1}  # the lateral rule: facing the left side the car's front shows image-left


def registration_defaults() -> dict[str, Any]:
    from . import register

    return {**register.DEFAULTS}


def same_walk_copies(th: dict | None) -> dict[str, Any] | None:
    """R28 (lead decision 1 of 11/2, thresholds matching.same_walk_copies): the leaf, or None when absent or off."""
    leaf = (th or {}).get("matching", {}).get("same_walk_copies")
    if not isinstance(leaf, dict) or not leaf.get("enabled"):
        return None
    return {"into_labels": [VerdictLabel.ALREADY_THERE.value], "refused_fit_min_mask_iou": 0.85, "refused_fit_min_hits": 2, **leaf}


def _renumbered(pm: PanelMatch, kept: list[Verdict]) -> list[Verdict]:
    return [v.model_copy(update={"verdict_id": f"{pm.panel_id.value}_v{n}"}) for n, v in enumerate(kept, 1)]


def _box_hit(box: tuple[float, float, float, float], target: list[float], cfg: dict[str, Any]) -> float | None:
    """The distance from box's centre to target's centre when the centre lies inside target or within coincide_px."""
    from . import register

    t = (float(target[0]), float(target[1]), float(target[2]), float(target[3]))
    cx, cy = _centre(box)
    tx, ty = _centre(t)
    dist = math.hypot(cx - tx, cy - ty)
    inside = t[0] <= cx <= t[2] and t[1] <= cy <= t[3]
    return dist if inside or dist <= register.coincide_px(t, cfg) else None


def copies_step(
    pm: PanelMatch, verdicts: list[Verdict], marks: dict[str, Mark], whitened: dict[str, WhitenedMark], cache: "RegistrationCache | None", th: dict
) -> tuple[PanelMatch, list[Verdict]]:
    """R28 (lead decision 1 of 11/2, matching.same_walk_copies), after the verdicts: a mark seen in one walk only, on
    another keyframe of the same panel in the same walk as a mark of a pair labelled one of into_labels, that maps onto
    that pair's mark is the pair's mark seen again. Its line leaves the list and a CopyNote names it in how checked. It
    maps through the panel frame (whitened distance within dedup.cluster_radius_units) or the registration (the crop
    fit of its keyframe onto the pair mark's keyframe, same walk first, then the other walk; the box centre inside the
    pair mark's box or within register.coincide_px). A fit that was refused after an outline fit with mask IoU at least
    refused_fit_min_mask_iou counts only when it puts at least refused_fit_min_hits marks of the keyframe on distinct
    decided marks: the marks themselves then confirm the map. Pure apart from the cache's fits; only removes lines."""
    cfg = same_walk_copies(th)
    if cfg is None or pm.state != MatchState.COMPARABLE or pm.panel_id in WHEEL_PANELS or not verdicts:
        return pm, verdicts
    labels = {VerdictLabel(x) for x in cfg["into_labels"]}
    anchors: list[tuple[Mark, Mark]] = []  # (the decided mark, its partner in the other walk)
    for v in verdicts:
        pair = [marks.get(i) for i in v.mark_ids]
        if v.label in labels and len(pair) == 2 and pair[0] is not None and pair[1] is not None and pair[0].role != pair[1].role:
            anchors += [(pair[0], pair[1]), (pair[1], pair[0])]
    if not anchors:
        return pm, verdicts
    radius = float(th["dedup"]["cluster_radius_units"])
    reg_cfg = cache.cfg if cache is not None else None
    found: dict[str, CopyNote] = {}
    refused: dict[tuple[str, str], list[tuple[str, CopyNote]]] = {}  # (copy kf, target kf) -> (target mark, note)
    for v in verdicts:
        if len(v.mark_ids) != 1 or v.label == VerdictLabel.NOT_COMPARABLE:
            continue
        m = marks.get(v.mark_ids[0])
        if m is None:
            continue
        mine = [(a, b) for a, b in anchors if a.role == m.role and a.kf_id != m.kf_id]
        if not mine:
            continue

        def note(a: Mark, b: Mark, target: Mark, via: str, dist: float, fit: str | None = None, was_refused: bool = False) -> CopyNote:
            pick, ret = (a, b) if a.role == ROLES[0] else (b, a)
            return CopyNote(mark_id=m.mark_id, kf_id=m.kf_id, into_mark=target.mark_id, into_pair=[pick.mark_id, ret.mark_id], via=via, distance=round(dist, 3), fit=fit, refused_fit=was_refused)  # type: ignore[arg-type]  # noqa: B023

        wm = whitened.get(m.mark_id)
        near = []
        for a, b in mine:
            wa = whitened.get(a.mark_id)
            if wm is not None and wa is not None and wm.frame_used == wa.frame_used == FrameUsed.MOMENT:
                d = math.hypot(wm.u - wa.u, wm.v - wa.v)
                if d <= radius:
                    near.append((d, a.mark_id, a, b))
        if near:
            d, _, a, b = min(near)
            found[m.mark_id] = note(a, b, a, "panel_frame", d)
            continue
        if cache is None or reg_cfg is None or m.bbox_crop is None:
            continue
        box = tuple(float(x) for x in m.bbox_crop)
        hits = []
        for a, b in mine:
            for target in (a, b):
                if target.bbox_crop is None or target.kf_id == m.kf_id:
                    continue
                al = cache.fit(m.kf_id, target.kf_id)
                if al is None or al.H is None:
                    continue
                from . import register

                dist = _box_hit(register.map_box(al.H, box), target.bbox_crop, reg_cfg)  # type: ignore[arg-type]
                if dist is None:
                    continue
                if al.ok:
                    hits.append((0, dist, target.mark_id, note(a, b, target, "registration", dist, str(al.model))))
                elif str(al.model) == "outline" and float(al.mask_iou) >= float(cfg["refused_fit_min_mask_iou"]):
                    refused.setdefault((m.kf_id, target.kf_id), []).append((target.mark_id, note(a, b, target, "registration", dist, "outline", True)))
        if hits:
            found[m.mark_id] = min(hits, key=lambda h: (h[0], h[1], h[2]))[3]
    for group in refused.values():
        best: dict[str, tuple[str, CopyNote]] = {}
        for target_id, n in group:
            if n.mark_id in found:
                continue
            if n.mark_id not in best or (n.distance or 0.0) < (best[n.mark_id][1].distance or 0.0):
                best[n.mark_id] = (target_id, n)
        if len(best) >= int(cfg["refused_fit_min_hits"]) and len({t for t, _ in best.values()}) >= int(cfg["refused_fit_min_hits"]):
            for mark_id, (_, n) in best.items():
                found.setdefault(mark_id, n)
    if not found:
        return pm, verdicts
    kept = [v for v in verdicts if not (len(v.mark_ids) == 1 and v.mark_ids[0] in found)]
    notes = [found[i] for i in sorted(found)]
    return pm.model_copy(update={"copies": [*pm.copies, *notes]}), _renumbered(pm, kept)


def copies_words(pm: PanelMatch) -> str | None:
    """how_checked's part for same-walk copies (R28)."""
    if not pm.copies:
        return None
    parts = []
    for n in pm.copies:
        if n.via == "panel_frame":
            how = f"within {n.distance:.2f} panel units in the panel frame"
        elif n.refused_fit:
            how = f"by an outline fit that was refused on its own but put this and another mark of {n.kf_id} inside the paired marks ({n.distance:.0f} px)"
        else:
            how = f"by a {n.fit} fit ({n.distance:.0f} px)"
        parts.append(f"{n.mark_id} is {n.into_mark} seen again in {n.kf_id}, {how}, so it has no line of its own (already decided with {' and '.join(i for i in n.into_pair if i != n.into_mark)})")
    return "copies: " + "; ".join(parts)


def faint_left_off(th: dict | None) -> bool:
    """R28 (lead decision 3 of 11/2, thresholds matching.faint_left_off)."""
    leaf = (th or {}).get("matching", {}).get("faint_left_off")
    return bool(leaf.get("enabled")) if isinstance(leaf, dict) else bool(leaf)


def faint_step(pm: PanelMatch, verdicts: list[Verdict], th: dict) -> tuple[PanelMatch, list[Verdict]]:
    """R28 (matching.faint_left_off): a line of one mark seen in one walk only that K20 capped (its box does not stand
    out from its own surroundings) leaves the renter's list; pm.left_off keeps its id so how checked says so with its
    contrast. Pure; only removes lines."""
    if not faint_left_off(th) or not verdicts:
        return pm, verdicts
    off = [v for v in verdicts if len(v.mark_ids) == 1 and "K20" in (v.trace.rule_ids if v.trace is not None else [])]
    if not off:
        return pm, verdicts
    gone = {v.verdict_id for v in off}
    kept = [v for v in verdicts if v.verdict_id not in gone]
    return pm.model_copy(update={"left_off": sorted({*pm.left_off, *(v.mark_ids[0] for v in off)})}), _renumbered(pm, kept)


def side_from_track(th: dict | None) -> dict[str, Any] | None:
    """R28 (lead decision 4 of 11/2, thresholds matching.side_from_track): the leaf, or None when absent or off."""
    leaf = (th or {}).get("matching", {}).get("side_from_track")
    if not isinstance(leaf, dict) or not leaf.get("enabled"):
        return None
    return {"panels": [PanelId.MIRRORS.value], "min_shared_frames": 15, "min_share": 0.8, **leaf}


def orientation_free_panels(th: dict | None) -> frozenset[PanelId]:
    cfg = side_from_track(th)
    return frozenset(PanelId(p) for p in cfg["panels"]) if cfg is not None else frozenset()


def track_sides(tracks: list[PanelTrack], cfg: dict[str, Any]) -> dict[str, str]:
    """For each track of an orientation-free panel, its side from the side-decided door tracks of its walk it shares
    frames with: the side holding at least min_share of those frames and at least min_shared_frames, else none."""
    free = {PanelId(p) for p in cfg["panels"]}
    doors: dict[str, list[tuple[str, set[int]]]] = {}
    for t in tracks:
        if t.noun == Noun.CAR_DOOR and t.side.value in SIDE_SIGN:
            doors.setdefault(t.role, []).append((t.side.value, {f.frame for f in t.per_frame}))
    out: dict[str, str] = {}
    for t in tracks:
        if t.panel_id not in free:
            continue
        frames = {f.frame for f in t.per_frame}
        counts = {s: 0 for s in SIDE_SIGN}
        for side, door_frames in doors.get(t.role, []):
            counts[side] += len(frames & door_frames)
        total = sum(counts.values())
        side, n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        out[t.track_id] = side if total and n >= int(cfg["min_shared_frames"]) and n >= float(cfg["min_share"]) * total else "none"
    return out


def assign_sides(run_id: str, keyframes: list[Keyframe], marks: list[Mark], th: dict) -> tuple[list[Keyframe], list[Mark], dict[str, str]]:
    """R28 (matching.side_from_track): a keyframe of an orientation-free panel with no front sign takes the sign of its
    track's side (right +1, left -1, the lateral rule) for the match, and its marks are whitened in that frame (in
    memory; marks.json keeps them as Muse Spark placed them). Returns the keyframes, the marks and each such keyframe's
    side ("none" when its track got none, and it keeps G6)."""
    cfg = side_from_track(th)
    free = orientation_free_panels(th)
    todo = [kf for kf in keyframes if kf.panel_id in free and kf.front_sign is None]
    if cfg is None or not todo:
        return keyframes, marks, {}
    path = paths.tracks_json(run_id)
    sides = track_sides(_read_list(path, PanelTrack), cfg) if path.is_file() else {}
    from .marks import whiten_marks

    given: dict[str, str] = {}
    signed: dict[str, Keyframe] = {}
    for kf in todo:
        side = sides.get(kf.track_id or "", "none")
        given[kf.kf_id] = side
        if side in SIDE_SIGN:
            signed[kf.kf_id] = kf.model_copy(update={"front_sign": SIDE_SIGN[side]})
    keyframes = [signed.get(kf.kf_id, kf) for kf in keyframes]
    out = []
    for m in marks:
        kf = signed.get(m.kf_id)
        out.append(whiten_marks([m], kf, th=th)[0][0] if kf is not None and m.w is None else m)
    return keyframes, out, given


def side_words(pm: PanelMatch) -> str | None:
    """how_checked's part for an orientation-free panel (R28): each keyframe's side from its own track."""
    if not pm.sides:
        return None
    return "no front sign on this panel, so each keyframe's side comes from its own track: " + ", ".join(f"{k} {v}" for k, v in sorted(pm.sides.items()))


def registration_config(th: dict | None) -> dict[str, Any] | None:
    """Thresholds matching.registration (11/2) with its defaults filled, or None when absent or not enabled."""
    from . import register

    return register.config(th)


VOUCH_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "models": ["features"],  # an outline fit rests on the two masks agreeing, which is what G4 doubts
    "min_inliers": 40,
    "min_spread": 0.3,
    "max_reproj_px": 2.0,
    "min_fit_iou": 0.85,
}


def registration_vouches(th: dict | None) -> dict[str, Any] | None:
    """Thresholds matching.registration_vouches (lead decision 1 of 11/2) with its defaults filled, or None when absent
    or not enabled, as in every run frozen before it."""
    raw = (th or {}).get("matching", {}).get("registration_vouches")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    return {**VOUCH_DEFAULTS, **raw}


GEOMETRIC_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "min_inliers": 20,
    "min_spread": 0.30,
    "max_reproj_px": 2.0,
    "min_mask_iou": 0.90,
    "outline_min_mask_iou": 0.95,
    "max_uncertainty_px": 4.0,
    "caps_ignored": ["K2", "K3", "K4", "K12", "K14"],
    "types_excluded": ["dirt", "reflection"],
    "wheels": True,
    "looks_max": 6,
}


def geometric_already_there(th: dict | None) -> dict[str, Any] | None:
    """thresholds matching.geometric_already_there (R29) over GEOMETRIC_DEFAULTS, or None when off or absent (every run
    frozen before it)."""
    raw = (th or {}).get("matching", {}).get("geometric_already_there")
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return None
    return {**GEOMETRIC_DEFAULTS, **raw}


def strong_fit(al: Any, pr: Any, cfg: dict[str, Any]) -> bool:
    """A crop fit good enough for code to call a mark already there from its presence test (R29): accepted, the map
    sure at the mark within max_uncertainty_px, and either fitted on matched edges (min_inliers agreeing points spread
    over min_spread of the panel, reprojection at most max_reproj_px, mask IoU after the warp at least min_mask_iou)
    or on the outline alone with the two outlines all but the same (mask IoU at least outline_min_mask_iou)."""
    if al is None or not al.ok or pr is None or pr.uncertainty_px is None or float(pr.uncertainty_px) > float(cfg["max_uncertainty_px"]):
        return False
    if al.model == "features":
        return (
            al.n_inliers >= int(cfg["min_inliers"])
            and al.spread >= float(cfg["min_spread"])
            and al.reproj_px <= float(cfg["max_reproj_px"])
            and al.mask_iou >= float(cfg["min_mask_iou"])
        )
    return al.model == "outline" and al.mask_iou >= float(cfg["outline_min_mask_iou"])


def geometric_ids(pm: PanelMatch) -> dict[str, RegistrationMark]:
    return {r.mark_id: r for r in pm.registration_marks if r.used == "already_there"}


def geometric_spot(r: RegistrationMark, kf_other: Keyframe | None, min_px: float = 24.0) -> tuple[float, float, float, float] | None:
    """The found box of a mark called already there by registration, grown to at least min_px a side and clipped to
    the other crop: the spot its look outlines."""
    if r.found_box is None or kf_other is None or kf_other.crop is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in r.found_box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = max((x2 - x1) / 2.0, min_px / 2.0), max((y2 - y1) / 2.0, min_px / 2.0)
    w_px, h_px = float(kf_other.crop.w), float(kf_other.crop.h)
    box = (round(max(0.0, cx - hw), 2), round(max(0.0, cy - hh), 2), round(min(w_px, cx + hw), 2), round(min(h_px, cy + hh), 2))
    return box if box[2] - box[0] >= 1.0 and box[3] - box[1] >= 1.0 else None


def geometric_asks(pm: PanelMatch, marks: dict[str, Mark], keyframes: dict[str, Keyframe]) -> list[CounterAsk]:
    """One one-question look per mark called already there by registration (R29): the mark's own crop beside the other
    walk's crop with the found spot outlined. A different or unsure answer lowers it (K7), as a look lowers a pair."""
    out: list[CounterAsk] = []
    for mid, r in geometric_ids(pm).items():
        mark = marks.get(mid)
        spot = geometric_spot(r, keyframes.get(r.other_kf))
        if mark is None or spot is None or mark.role not in ROLES:
            continue
        w = (float(mark.w[0]), float(mark.w[1])) if mark.w is not None else (0.0, 0.0)
        out.append(CounterAsk(mark_id=mid, role=mark.role, panel_id=pm.panel_id, other_kf=r.other_kf, spot=spot, w=w, radius_units=0.0, confirm_mark=False))
    return out


def geometric_look(mark: Mark, flags: VerdictFlags, counters: dict[str, LookVerdict | CounterAnswer | None]) -> tuple[VerdictFlags, str | None]:
    """K7 and the note for a mark called already there by registration: the look's spot answer (a one-question
    verdict, or part 2 of a two-question answer stored for the same mark) different or unsure lowers it."""
    if mark.mark_id not in counters:
        return flags, None
    other = "pickup" if mark.role == "return" else "return"
    look = counters[mark.mark_id]
    spot = look.spot if isinstance(look, CounterAnswer) else look
    what = f"the second look compared this mark with the spot found in the {other} crop"
    if spot == LookVerdict.SAME:
        return flags, f"{what} and said same, so it stays already there"
    if spot == LookVerdict.DIFFERENT:
        return flags.model_copy(update={"K7": True}), f"{what} and said different, so needs your eye (K7)"
    if spot == LookVerdict.UNSURE:
        return flags.model_copy(update={"K7": True}), f"{what} and was unsure, so needs your eye (K7)"
    return flags, f"{what} and gave no usable answer, so the label stands"


def vouch_fit_config(th: dict | None) -> dict[str, Any]:
    """The register numbers the vouch fits with: matching.registration's when that leaf is on (so the fit is shared
    with the registration step), register.DEFAULTS otherwise."""
    from . import register

    return registration_config(th) or {**register.DEFAULTS, "enabled": True}


def vouch_check(al: Any, cfg: dict[str, Any]) -> RegistrationVouch:
    """Whether a registration fit is strong enough to stand in for G4's mask IoU check: accepted by register.align,
    fitted on matched edges (cfg models), with at least min_inliers agreeing points spread over min_spread of the
    panel's box, a median reprojection of at most max_reproj_px and a mask IoU after the warp of at least min_fit_iou.
    G4 asks whether both views show the same region of the panel; a fit like that shows it directly."""
    short: list[str] = []
    if not al.ok:
        short.append(f"the fit was refused ({al.why})")
    else:
        if al.model not in cfg["models"]:
            short.append("fitted on the outline only")
        if al.n_inliers < int(cfg["min_inliers"]):
            short.append(f"{al.n_inliers} matched points agree, under {int(cfg['min_inliers'])}")
        if al.spread < float(cfg["min_spread"]):
            short.append(f"they span {al.spread:.2f} of the panel, under {float(cfg['min_spread']):.2f}")
        if not al.reproj_px <= float(cfg["max_reproj_px"]):
            short.append(f"reprojection {al.reproj_px:.1f} px, over {float(cfg['max_reproj_px']):.1f}")
        if al.mask_iou < float(cfg["min_fit_iou"]):
            short.append(f"mask IoU after the warp {al.mask_iou:.2f}, under {float(cfg['min_fit_iou']):.2f}")
    return RegistrationVouch(fit=_fit_record(al), vouched=not short, why="; ".join(short) or None)


def vouch_words(v: RegistrationVouch) -> str:
    """The vouch in words for the how-checked line."""
    f = v.fit
    if v.vouched:
        reproj = f", reprojection {f.reproj_px:.1f} px" if f.reproj_px is not None else ""
        return f"G4 passed on the registration instead: {f.n_inliers} of {f.n_matches} matched points agree over {f.spread:.2f} of the panel{reproj}"
    return f"the registration did not vouch: {v.why}"


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


def keyframe_pair_order(kfs_pickup: list[Keyframe], kfs_return: list[Keyframe], *, th: dict | None = None, scales: dict[str, float | None] | None = None) -> list[tuple[Keyframe, Keyframe]]:
    """Every keyframe pair of a panel in the order match tries them (lead 11/1, rule 2): the pair pair_keyframes chooses
    first, then the others with clean views first (how many of the two keyframes prefer_clean would not offer: 0, 1
    or 2), then the product of the two keyframe scores, highest first (a pair needs both views good; a sum would rank
    one sharp view with one poor view above two fair ones), then the keyframe numbers, pickup first."""
    first_p, first_r, _ = pair_keyframes(kfs_pickup, kfs_return, th=th)
    clean_p = {kf.kf_id for kf in prefer_clean(kfs_pickup, th)}
    clean_r = {kf.kf_id for kf in prefer_clean(kfs_return, th)}
    others = [(a, b) for a in kfs_pickup for b in kfs_return if (a.kf_id, b.kf_id) != (first_p.kf_id, first_r.kf_id)]
    others.sort(key=lambda ab: ((ab[0].kf_id not in clean_p) + (ab[1].kf_id not in clean_r), -(ab[0].score.total * ab[1].score.total), _kf_index(ab[0]), _kf_index(ab[1])))
    order = [(first_p, first_r), *others]
    ps = pair_scale(th)
    if ps is not None and scales is not None:  # R27: pairs at a similar scale first, then the rest, each in the order above
        def band(ab: tuple[Keyframe, Keyframe]) -> int:
            r = scale_ratio(scales.get(ab[0].kf_id), scales.get(ab[1].kf_id))
            return 1 if r is None else 0 if r <= float(ps["prefer_ratio"]) else 1 if r <= float(ps["max_ratio"]) else 2
        order = sorted(order, key=band)
    return order


def scale_ratio(a: float | None, b: float | None) -> float | None:
    """The larger of two keyframe scales over the smaller; None when either is unknown."""
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return max(a, b) / min(a, b)


def keyframe_scales(run_id: str, keyframes: list[Keyframe], masks: dict[str, np.ndarray | None] | None = None) -> dict[str, float | None]:
    """Each keyframe's scale in source px (R27): the long side of its panel mask's box, mask px over the crop's scale;
    for a fender built from a wheel, the wheel track's box long side at that frame (the wheel size the fender was built
    from), proxy px times the keyframe's source-over-proxy factor. On hero_focus4 the return right rear wheel's mask held
    both wheels of that side (1900 px against 476 px at pickup) and the fender built from it spanned the whole side."""
    tracks: dict[str, Any] | None = None
    out: dict[str, float | None] = {}
    for kf in keyframes:
        value: float | None = None
        if "_fender_from_wheel_" in kf.track_id:
            if tracks is None:
                tp = paths.tracks_json(run_id)
                tracks = {t["track_id"]: t for t in json.loads(tp.read_text(encoding="utf-8"))} if tp.is_file() else {}
            fender, wheel = tracks.get(kf.track_id), tracks.get(kf.track_id.replace("_fender_from_wheel_", "_wheel_", 1))
            fb = next((r["box"] for r in (fender or {}).get("per_frame", []) if r["frame"] == kf.f_proxy), None)
            wb = next((r["box"] for r in (wheel or {}).get("per_frame", []) if r["frame"] == kf.f_proxy), None)
            if fb is not None and wb is not None and fb[2] > fb[0]:
                factor = (kf.box_src[2] - kf.box_src[0]) / (fb[2] - fb[0])
                value = max(wb[2] - wb[0], wb[3] - wb[1]) * factor
        if value is None and kf.crop is not None and kf.crop.scale > 0:
            if masks is not None and kf.kf_id in masks:
                m = masks[kf.kf_id]
            else:
                m = _load_mask(run_id, kf)
                if masks is not None:
                    masks[kf.kf_id] = m
            if m is not None and m.any():
                ys, xs = np.nonzero(m)
                value = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)) / kf.crop.scale
        out[kf.kf_id] = round(value, 1) if value is not None else None
    return out


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
    """What K4, K5 and K12 stand for on this pair, in words with their ids. roles names the two marks (a claim passes
    ("pickup", "claim"))."""
    typed = None
    if "K4" in deferred or "K5" in deferred:
        typed = f"both marks {a.type.value}" if a.type == b.type else f"the {roles[0]} mark {a.type.value} and the {roles[1]} mark {b.type.value}"
        if "K5" in deferred:
            typed = f"{typed} (K4), two kinds that disagree (K5)" if "K4" in deferred else f"{typed}, two kinds that disagree (K5)"
            if "K12" in deferred:
                return f"Muse Spark typed {typed}, with low confidence on {_low_where(a, b, roles)} (K12)"
            return f"Muse Spark typed {typed}"
    where = _low_where(a, b, roles)
    if typed and "K12" in deferred:
        return f"Muse Spark typed {typed} (K4), with low confidence on {where} (K12)"
    if typed:
        return f"Muse Spark typed {typed} (K4)"
    return f"Muse Spark's confidence was low on {where} (K12)"


def _low_where(a: Mark, b: Mark, roles: tuple[str, str]) -> str:
    low = [role for role, m in zip(roles, (a, b)) if "K12" in m.caps]
    return f"the {low[0]} mark" if len(low) == 1 else "both marks"


def tight_pair_flags(
    pair: MatchCandidate,
    a: Mark,
    b: Mark,
    flags: VerdictFlags,
    look: LookVerdict | None,
    *,
    asked: bool,
    roles: tuple[str, str] = ROLES,
    deferrable: tuple[str, ...] = TYPE_CAPS,
    tally: tuple[int, int] | None = None,
) -> tuple[VerdictFlags, list[str], str | None]:
    """Rule 1 on a pair inside tau_pre_r (the caller checks d < tau_pre_r): once the second look answered (same,
    different or unsure), K4 and K12 are notes and lower nothing, so only the look (K7) or another cap can lower the
    pair; with no usable answer they cap as before. asked says the look was put to a model (its answer may be None:
    every engine failed or the look could not be built) rather than not asked at all (spend ceiling, deadline, look
    budget, the look turned off). Returns the flags for verdict(), the note ids and the type note; ([], None) when
    neither K4 nor K12 is set. roles names the two marks in the note: a claim passes ("pickup", "claim") (integration
    11/1: claims follow rule 1 under the same leaf). deferrable is tight_pair_deferrable(th): K5 joins K4 and K12
    under matching.tight_pair_k5 (lead 11/1, second). tally (votes, answers) is the majority look's count
    (matching.look_majority, lead decision 2 of 11/2), named when more than one answer stands behind the look."""
    deferred = [k for k in deferrable if getattr(flags, k)]
    if not deferred:
        return flags, [], None
    what = _spark_words(a, b, deferred, roles)
    of = f" ({tally[0]} of {tally[1]} answers)" if tally is not None and tally[1] > 1 else ""
    if look is not None:
        if look == LookVerdict.SAME:
            place = "the mark sits in the same place in both walks" if tuple(roles) == ROLES else f"the {roles[1]} mark sits where the {roles[0]} mark is"
            tail = f"a note here, since {place} and the second look compared the two crops and said same{of}"
        else:
            said = "said different" if look == LookVerdict.DIFFERENT else "was unsure"
            tail = f"a note here; the second look compared the two crops and {said}{of} (K7)"
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


# The one-walk counter-look (lead 11/1, third: K16) -------------------------------------------------------------------

COUNTER_DISC_STEPS = 32  # points on the whitened circle of radius tau_eye_r mapped into the other crop
CONFIDENCE_RANK = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}


@dataclass(frozen=True)
class CounterAsk:
    """One counter-look: a mark seen in one walk only whose label is new_or_unseen or not_seen_at_return with no cap,
    and the other walk's keyframe of the pair compared on, with the spot where the mark would sit outlined. spot is
    x1, y1, x2, y2 in that keyframe's crop pixels: the box around the mark's own box carried over and every point
    within radius_units of w, the mark's position carried into the other walk's whitened frame (the handle residual
    undone). radius_units is tau_eye_r at the mark: a mark found anywhere within it in the other walk would have
    matched this one or put it in the band (needs your eye), so the look asks whether Spark missed one there."""

    mark_id: str
    role: str
    panel_id: PanelId
    other_kf: str
    spot: tuple[float, float, float, float]
    w: tuple[float, float]
    radius_units: float
    confirm_mark: bool = False  # the two-question look (lead decision 5 of 11/1): also ask whether the mark is real

    @property
    def other_role(self) -> str:
        return "pickup" if self.role == "return" else "return"

    @property
    def key(self) -> tuple[str, str]:
        """(pickup field, return field) of its looks.json row: the mark in its own walk's field, the other walk's
        keyframe id in the other field."""
        return (self.other_kf, self.mark_id) if self.role == "return" else (self.mark_id, self.other_kf)


YES_NO_UNSURE = ("yes", "no", "unsure")


@dataclass(frozen=True)
class CounterAnswer:
    """The two-question counter-look's answer (lead decision 5 of 11/1). mark: is there a damage mark (scratch, scuff,
    chip, dent, crack) inside the outlined box on the mark's own crop, "yes", "no" or "unsure". spot: does the other
    walk's crop show the same mark at the outlined spot, as a look verdict (yes same, no different, unsure unsure).
    Only ("yes", different) keeps the label; part 1 not yes is K17, part 2 not different is K16."""

    mark: str
    spot: LookVerdict
    # the majority look (matching.look_majority, lead decision 2 of 11/2): how many usable answers it stands on and how
    # many of them took each part's winning side; 1, 1, 1 for a single answer
    answers: int = 1
    mark_votes: int = 1
    spot_votes: int = 1

    def __post_init__(self) -> None:
        if self.mark not in YES_NO_UNSURE:
            raise ValueError(f"mark must be one of {', '.join(YES_NO_UNSURE)}, not {self.mark!r}")


LOOK_MAJORITY_DEFAULTS: dict[str, Any] = {"enabled": False, "asks_max": 3, "agree": 2}


def look_majority(th: dict | None) -> dict[str, Any] | None:
    """Thresholds matching.look_majority (lead decision 2 of 11/2) with its defaults filled, or None when absent or off,
    as in every run frozen before it: then each look is asked once."""
    raw = (th or {}).get("matching", {}).get("look_majority")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    return {**LOOK_MAJORITY_DEFAULTS, **raw}


def _side(keep: int, lower: int, lowered: str, other: str, lowered_n: int, other_n: int, kept: Any) -> Any:
    """The winning side of one yes-or-no part: kept when its answers outnumber the rest; otherwise the lowering answer
    given most (other on a tie between the two lowering answers, and on a tie between keep and lower: split is unsure)."""
    if keep > lower:
        return kept
    if lower > keep and lowered_n > other_n:
        return lowered
    return other


def majority_pair_look(answers: list[LookVerdict | None]) -> tuple[LookVerdict | None, int, int]:
    """The pair look's majority (K7, lead decision 2 of 11/2) over the answers in the order asked: same keeps the pair
    only when same outnumbers different and unsure together; otherwise different when it outnumbers unsure, else
    unsure (a split look did not confirm the pair, and a model only lowers). Returns the verdict, the answers on its
    side and the usable answers; (None, 0, 0) when none was usable."""
    usable = [a for a in answers if a is not None]
    if not usable:
        return None, 0, 0
    same = sum(a == LookVerdict.SAME for a in usable)
    diff = sum(a == LookVerdict.DIFFERENT for a in usable)
    unsure = len(usable) - same - diff
    out = _side(same, diff + unsure, LookVerdict.DIFFERENT, LookVerdict.UNSURE, diff, unsure, LookVerdict.SAME)
    votes = same if out == LookVerdict.SAME else (diff + unsure)
    return out, votes, len(usable)


def majority_counter(answers: list[LookVerdict | CounterAnswer | None]) -> CounterAnswer | None:
    """The two-question counter-look's majority (K17 and K16, lead decision 2 of 11/2), each part on its own over the
    same answers: part 1 is yes only when yes outnumbers no and unsure together, else no when it outnumbers unsure,
    else unsure; part 2 is different only when different outnumbers same and unsure, else same when it outnumbers
    unsure, else unsure. None when no answer was a usable CounterAnswer."""
    usable = [a for a in answers if isinstance(a, CounterAnswer)]
    if not usable:
        return None
    yes = sum(a.mark == "yes" for a in usable)
    no = sum(a.mark == "no" for a in usable)
    m_unsure = len(usable) - yes - no
    diff = sum(a.spot == LookVerdict.DIFFERENT for a in usable)
    same = sum(a.spot == LookVerdict.SAME for a in usable)
    s_unsure = len(usable) - diff - same
    mark = _side(yes, no + m_unsure, "no", "unsure", no, m_unsure, "yes")
    spot = _side(diff, same + s_unsure, LookVerdict.SAME, LookVerdict.UNSURE, same, s_unsure, LookVerdict.DIFFERENT)
    return CounterAnswer(
        mark=mark,
        spot=spot,
        answers=len(usable),
        mark_votes=yes if mark == "yes" else no + m_unsure,
        spot_votes=diff if spot == LookVerdict.DIFFERENT else same + s_unsure,
    )


def _settled(answers: list[Any], agree: int) -> bool:
    """Every yes-or-no part has agree answers on one side (stop asking), or the last ask gave no usable answer (every
    engine failed or the look could not be built; asking again would not help)."""
    if answers and answers[-1] is None:
        return True
    usable = [a for a in answers if a is not None]
    if isinstance(usable[-1] if usable else None, CounterAnswer):
        yes = sum(a.mark == "yes" for a in usable)
        diff = sum(a.spot == LookVerdict.DIFFERENT for a in usable)
        return max(yes, len(usable) - yes) >= agree and max(diff, len(usable) - diff) >= agree
    same = sum(a == LookVerdict.SAME for a in usable)
    return max(same, len(usable) - same) >= agree


def ask_majority(ask: Callable[[], Any], cfg: dict[str, Any] | None) -> list[Any]:
    """Every answer of one look in order: asked once without cfg; with it up to asks_max times, stopping as soon as
    agree answers take the same side of each part (_settled). A one-question counter answer (a LookVerdict on a run
    whose counter-look asks one question) is asked once: the lead's decision covers K17 and K7."""
    first = ask()
    answers = [first]
    if cfg is None:
        return answers
    while len(answers) < int(cfg["asks_max"]) and not _settled(answers, int(cfg["agree"])):
        answers.append(ask())
    return answers


def counter_result(answers: list[LookVerdict | CounterAnswer | None]) -> LookVerdict | CounterAnswer | None:
    """What a counter-look's answers come to: the majority of the two-question answers (majority_counter); a stored
    one-question answer (a LookVerdict) stands alone, as it was asked once."""
    if any(isinstance(a, LookVerdict) for a in answers):
        return answers[0]
    return majority_counter(answers) if len(answers) > 1 else answers[0]


def counter_spot(
    mark: Mark, kf_own: Keyframe | None, kf_other: Keyframe, residual: Point2 | None, radius: float
) -> tuple[tuple[float, float], tuple[float, float, float, float]] | None:
    """Where a mark seen in one walk would sit in the other walk's keyframe crop. The match compares return marks with
    the handle residual subtracted, so a return mark's spot in the pickup frame is w - residual and a pickup mark's
    spot in the return frame is w + residual. The box covers the whitened circle of radius around that point and the
    mark's own box (its corners whitened in kf_own, shifted the same way) mapped into crop pixels, clipped to the crop.
    None when the mark has no whitened position, the keyframe has no moments, front sign or crop, or the spot's centre
    falls outside the crop."""
    if mark.w is None or kf_other.moments is None or kf_other.front_sign is None or kf_other.crop is None:
        return None
    du, dv = (float(residual[0]), float(residual[1])) if residual is not None else (0.0, 0.0)
    sign = -1.0 if mark.role == "return" else 1.0
    u, v = float(mark.w[0]) + sign * du, float(mark.w[1]) + sign * dv
    try:
        cx, cy = geometry.unwhiten((u, v), kf_other.moments, kf_other.front_sign)
        ring = [geometry.unwhiten((u + radius * math.cos(t), v + radius * math.sin(t)), kf_other.moments, kf_other.front_sign) for t in (2 * math.pi * k / COUNTER_DISC_STEPS for k in range(COUNTER_DISC_STEPS))]
        if mark.bbox_crop is not None and kf_own is not None and kf_own.moments is not None and kf_own.front_sign is not None:
            x1, y1, x2, y2 = (float(c) for c in mark.bbox_crop)
            for corner in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                cu, cv = geometry.whiten(corner, kf_own.moments, kf_own.front_sign)
                ring.append(geometry.unwhiten((cu + sign * du, cv + sign * dv), kf_other.moments, kf_other.front_sign))
    except (ValueError, np.linalg.LinAlgError):
        return None
    w_px, h_px = float(kf_other.crop.w), float(kf_other.crop.h)
    if not (0.0 <= cx < w_px and 0.0 <= cy < h_px):
        return None
    xs, ys = [p[0] for p in ring] + [cx], [p[1] for p in ring] + [cy]
    box = (round(max(0.0, min(xs)), 2), round(max(0.0, min(ys)), 2), round(min(w_px, max(xs)), 2), round(min(h_px, max(ys)), 2))
    if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
        return None
    return (round(u, 6), round(v, 6)), box


def counter_asks(pm: PanelMatch, verdicts: list[Verdict], marks: dict[str, Mark], keyframes: dict[str, Keyframe], calibration: Calibration, th: dict) -> list[CounterAsk]:
    """The panel's marks seen in one walk only (L1, L2, and either mark of an L5 pair) whose label is new_or_unseen or
    not_seen_at_return, so no cap applied, each with its spot in the other walk's keyframe of the pair compared on.
    Only a comparable panel in the moment frame has one: a wheel's frame is radius and angle on a rim that turns, so
    no spot is drawn there, and a mark whose spot cannot be placed (see counter_spot) is not asked."""
    if pm.state != MatchState.COMPARABLE or pm.frame_used != FrameUsed.MOMENT or not pm.kf_pair:
        return []
    confirm = counter_confirms_mark(th)
    out: list[CounterAsk] = []
    for v in verdicts:
        if v.label not in (VerdictLabel.NEW_OR_UNSEEN, VerdictLabel.NOT_SEEN_AT_RETURN) or len(v.mark_ids) != 1:
            continue
        mark = marks.get(v.mark_ids[0])
        if mark is None or mark.role not in ROLES:
            continue
        kf_other = keyframes.get(pm.kf_pair[0] if mark.role == "return" else pm.kf_pair[1])
        if kf_other is None:
            continue
        radius = tau_r(calibration.tau_eye, mark.r if mark.r is not None else 0.0, th)
        placed = counter_spot(mark, keyframes.get(mark.kf_id), kf_other, pm.handle_residual, radius)
        if placed is None:
            continue
        w, spot = placed
        out.append(CounterAsk(mark_id=mark.mark_id, role=mark.role, panel_id=pm.panel_id, other_kf=kf_other.kf_id, spot=spot, w=w, radius_units=round(radius, 6), confirm_mark=confirm))
    return out


def rank_counter_asks(asks: list[CounterAsk], marks: dict[str, Mark]) -> list[CounterAsk]:
    """Surest first: Muse Spark's confidence on the mark (high, medium, low), then the order the panels were matched
    in (asks arrive in it), then the mark id."""
    order = {a.mark_id: i for i, a in enumerate(asks)}
    return sorted(asks, key=lambda a: (CONFIDENCE_RANK.get(marks[a.mark_id].confidence, 3) if a.mark_id in marks else 3, order[a.mark_id], a.mark_id))


def counter_flags(
    mark: Mark, flags: VerdictFlags, counters: dict[str, LookVerdict | CounterAnswer | None], skipped: dict[str, int], *, confirm: bool = False
) -> tuple[VerdictFlags, str | None]:
    """K16, K17 and the counter note on a mark seen in one walk only. counters maps a mark asked to the answer (None:
    no usable answer); skipped maps a mark the run's cap left out to the cap. A one-question answer (a LookVerdict):
    same or unsure sets K16 (needs your eye). A two-question answer (CounterAnswer, lead decision 5 of 11/1): part 1
    not yes sets K17, part 2 not different sets K16, so only (yes, different) keeps the label. No answer or not asked
    leave the label as the distance and the other caps gave it. confirm says the run asks two questions (only the
    notes for no answer and not asked change)."""
    other = "pickup" if mark.role == "return" else "return"
    if mark.mark_id in counters and isinstance(counters[mark.mark_id], CounterAnswer):
        return two_question_flags(flags, counters[mark.mark_id], other)  # type: ignore[arg-type]
    if confirm and mark.mark_id in counters and counters[mark.mark_id] is None:
        return flags, f"the second look at the mark and the same spot in the {other} walk gave no usable answer, so the label stands"
    if confirm and mark.mark_id not in counters and mark.mark_id in skipped:
        n = skipped[mark.mark_id]
        return flags, f"the second look at the mark and the same spot in the {other} walk was not asked: the run's {n} counter-look{'' if n == 1 else 's'} went to surer marks first, so the label stands"
    if mark.mark_id in counters:
        look = counters[mark.mark_id]
        what = f"the second look compared this mark with the same spot in the {other} walk"
        if look == LookVerdict.SAME:
            return flags.model_copy(update={"K16": True}), f"{what} and said same: the {other} walk may show it (K16)"
        if look == LookVerdict.UNSURE:
            return flags.model_copy(update={"K16": True}), f"{what} and was unsure: the {other} walk may show it (K16)"
        if look == LookVerdict.DIFFERENT:
            return flags, f"{what} and said different, so the label stands"
        return flags, f"the second look at the same spot in the {other} walk gave no usable answer, so the label stands"
    if mark.mark_id in skipped:
        n = skipped[mark.mark_id]
        return flags, f"the second look at the same spot in the {other} walk was not asked: the run's {n} counter-look{'' if n == 1 else 's'} went to surer marks first, so the label stands"
    return flags, None


def two_question_flags(flags: VerdictFlags, answer: CounterAnswer, other: str) -> tuple[VerdictFlags, str]:
    """The two-question counter-look's caps and note (lead decision 5 of 11/1). Part 1 (a damage mark in the box on
    the mark's own crop) no or unsure sets K17; part 2 (the same mark at the other walk's spot) same or unsure sets
    K16; the note names every part that failed, or says the label stands on (yes, different)."""
    failed: list[str] = []
    update: dict[str, bool] = {}
    n = answer.answers
    m_of = f" ({answer.mark_votes} of {n} answers)" if n > 1 else ""
    s_of = f" ({answer.spot_votes} of {n} answers)" if n > 1 else ""
    if answer.mark != "yes":
        update["K17"] = True
        failed.append(f"saw no damage in the box{m_of} (K17)" if answer.mark == "no" else f"was unsure there is damage in the box{m_of} (K17)")
    if answer.spot == LookVerdict.SAME:
        update["K16"] = True
        failed.append(f"said the same mark shows at the same spot in the {other} walk{s_of}: the {other} walk may show it (K16)")
    elif answer.spot != LookVerdict.DIFFERENT:
        update["K16"] = True
        failed.append(f"was unsure about the same spot in the {other} walk{s_of}: the {other} walk may show it (K16)")
    if not failed:
        return flags, f"the second look saw damage in the box{m_of} and not at the same spot in the {other} walk{s_of}, so the label stands"
    return flags.model_copy(update=update), "the second look " + " and ".join(failed)


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
    before it with the gates each failed; for a panel no pair passed, every pair tried; then the reshaped-panel test and
    the marks left out as factory fixtures, when there are any. None when the match recorded none of these (keyframes
    in one walk only, or a run frozen before 11/1)."""
    parts = [
        _how_checked_tries(pm),
        reshape_words(pm.reshape) if pm.reshape is not None else None,
        fixture_words(pm.fixtures) if pm.fixtures else None,
        privacy_drop_words(pm.privacy_drops) if pm.privacy_drops else None,
        off_panel_words(pm) if pm.off_panel else None,
        "; ".join(f"second marks pass: {w}" for w in pm.second_pass) if pm.second_pass else None,
        registration_words(pm),
        same_walk_words(pm),
        copies_words(pm),
        stands_out_words(pm),
        side_words(pm),
        spot_words(pm),
    ]
    line = "; ".join(p for p in parts if p)
    return line or None


def _how_checked_tries(pm: PanelMatch) -> str | None:
    tries = pm.kf_pairs_tried
    if not tries:
        return None

    def pair_words(t: KeyframePairTry) -> str:
        return f"{t.kf_pair[0]} with {t.kf_pair[1]}"

    def tried(t: KeyframePairTry) -> str:
        vouch = f", {vouch_words(t.vouch)}" if t.vouch is not None else ""
        return f"{pair_words(t)} ({', '.join(_gate_words(g, t) for g in t.gates)}{vouch})"

    used = next((t for t in tries if not t.gates), None)
    if used is not None:
        at = tries.index(used)
        before, after = tries[:at], tries[at + 1 :]  # after: pairs tried before the registration vouched for this one
        how = f" ({_gate_words('G4', used)}; {vouch_words(used.vouch)})" if used.vouch is not None and used.vouch.vouched else ""
        also = ("; also tried: " + "; ".join(tried(t) for t in after)) if after else ""
        if not before:
            return f"compared on keyframes {pair_words(used)}{how}, the first pair tried{also}"
        return f"compared on keyframes {pair_words(used)}{how}; tried first: " + "; ".join(tried(t) for t in before) + also
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


# Factory fixtures (lead decision 4 of 11/1, thresholds matching.fixtures_not_damage) -----------------------------------

FIXTURE_WORDS = {
    "sensor": "a parking sensor",
    "badge": "a badge",
    "handle": "the door handle",
    "lock": "the door lock",
    "washer_nozzle": "a washer nozzle",
    "valve": "the tire valve",
    "hub": "the wheel hub",
    "plate_corner_tl": "a plate corner",
    "plate_corner_tr": "a plate corner",
    "plate_corner_bl": "a plate corner",
    "plate_corner_br": "a plate corner",
}
FIXTURE_SOURCE_WORDS = {"spark": "Muse Spark's landmark", "sam_handle": "SAM's door handle"}


@dataclass(frozen=True)
class FixtureZone:
    """A fixture's box on Spark's 0 to 1000 grid of one keyframe crop: its point plus the name's half width and height
    from thresholds matching.fixture_half_grid."""

    name: str
    source: str  # "spark" or "sam_handle"
    box: tuple[float, float, float, float]


def fixture_halves(th: dict | None) -> dict[str, tuple[float, float]]:
    raw = (th or {}).get("matching", {}).get("fixture_half_grid") or {}
    return {str(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def _zone(name: str, source: str, x: float, y: float, half: tuple[float, float]) -> FixtureZone:
    return FixtureZone(name, source, (x - half[0], y - half[1], x + half[0], y + half[1]))


def fixture_zones(kf: Keyframe, records: list[SparkRequestRecord], th: dict) -> list[FixtureZone]:
    """The fixtures the run already has on this keyframe's crop, in Spark's grid of that crop: every fixture landmark
    of a valid marks answer on the crop (read as find_marks reads it, so only the panel's own list), then the SAM
    door-handle point carried from source pixels into the same grid. A name without a half size is not a fixture."""
    from . import spark_client
    from .marks import landmarks_for

    if kf.crop is None:
        return []
    halves = fixture_halves(th)
    names = landmarks_for(kf.panel_id, th)
    zones: list[FixtureZone] = []
    for rec in records:
        if rec.purpose != SparkPurpose.MARKS or len(rec.input_refs) != 1 or rec.input_refs[0].path != kf.crop.path:
            continue
        for lm in spark_client.answer_landmarks(rec, names):
            if lm["name"] in halves:
                zones.append(_zone(lm["name"], "spark", float(lm["point"]["x"]), float(lm["point"]["y"]), halves[lm["name"]]))
    if kf.handle.point_src is not None and "handle" in halves and kf.crop.w > 0 and kf.crop.h > 0:
        px, py = geometry.src_to_crop(kf.handle.point_src, kf.crop)
        gx, gy = px / kf.crop.w * 1000.0, py / kf.crop.h * 1000.0
        if 0.0 <= gx <= 1000.0 and 0.0 <= gy <= 1000.0:
            zones.append(_zone("handle", "sam_handle", gx, gy, halves["handle"]))
    return zones


def fixture_overlap(box: tuple[float, float, float, float], zone: FixtureZone) -> float:
    """The share of the mark's box inside the fixture's box; a box with no area counts as its centre point."""
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone.box
    if x2 <= x1 or y2 <= y1:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return 1.0 if zx1 <= cx <= zx2 and zy1 <= cy <= zy2 else 0.0
    ix = max(0.0, min(x2, zx2) - max(x1, zx1))
    iy = max(0.0, min(y2, zy2) - max(y1, zy1))
    return ix * iy / ((x2 - x1) * (y2 - y1))


def fixture_drops(marks: list[Mark], keyframes: list[Keyframe], records: list[SparkRequestRecord], th: dict) -> dict[str, FixtureDrop]:
    """Every mark whose box sits on a fixture zone of its own crop by at least matching.fixture_overlap_min of its box,
    by mark id; the fixture covering most of the box names it (the first one on a tie). A mark the renter named
    (narration seed) is never dropped: the renter pointed at it, so the sheet answers it. Empty with the leaf off."""
    if not fixtures_not_damage(th):
        return {}
    overlap_min = float(th["matching"]["fixture_overlap_min"])
    kf_by_id = {kf.kf_id: kf for kf in keyframes}
    zones_by_kf: dict[str, list[FixtureZone]] = {}
    drops: dict[str, FixtureDrop] = {}
    for m in marks:
        kf = kf_by_id.get(m.kf_id)
        if m.source == MarkSource.NARRATION_SEED or m.grid_bbox is None or kf is None:
            continue
        if kf.kf_id not in zones_by_kf:
            zones_by_kf[kf.kf_id] = fixture_zones(kf, records, th)
        box = (float(m.grid_bbox.x_min), float(m.grid_bbox.y_min), float(m.grid_bbox.x_max), float(m.grid_bbox.y_max))
        best: tuple[float, FixtureZone] | None = None
        for zone in zones_by_kf[kf.kf_id]:
            share = fixture_overlap(box, zone)
            if best is None or share > best[0]:
                best = (share, zone)
        if best is not None and best[0] >= overlap_min:
            drops[m.mark_id] = FixtureDrop(mark_id=m.mark_id, fixture=best[1].name, source=best[1].source, overlap=round(best[0], 4))  # type: ignore[arg-type]
    return drops


def fixture_words(drops: list[FixtureDrop]) -> str:
    """how_checked's words for the marks left out as fixtures, one clause each."""
    return "; ".join(
        f"ignored: {d.mark_id} sits on {FIXTURE_WORDS.get(d.fixture, d.fixture.replace('_', ' '))} "
        f"({FIXTURE_SOURCE_WORDS.get(d.source, d.source)}, {math.floor(d.overlap * 100 + 0.5)}% of its box), so it was not matched"
        for d in drops
    )


def privacy_drops(run_id: str, marks: list[Mark], kf_by_id: dict[str, Keyframe], th: dict) -> dict[str, PrivacyDrop]:
    """Every mark whose box has more than privacy_blur's mark_max of its pixels in the part of its crop blurred for
    privacy (the keyframe's blur_png), by mark id. Empty with the leaf off or for keyframes made before R27."""
    pb = privacy_blur(th)
    if pb is None:
        return {}
    import cv2

    blur_masks: dict[str, np.ndarray | None] = {}
    out: dict[str, PrivacyDrop] = {}
    for m in marks:
        kf = kf_by_id.get(m.kf_id)
        if kf is None or not kf.blur_png or m.grid_bbox is None or kf.crop is None:
            continue
        if kf.kf_id not in blur_masks:
            img = cv2.imread(str(paths.run_dir(run_id) / kf.blur_png), cv2.IMREAD_GRAYSCALE)
            blur_masks[kf.kf_id] = None if img is None else img > 0
        bm = blur_masks[kf.kf_id]
        if bm is None:
            continue
        h, w = bm.shape[:2]
        x1, y1 = int(math.floor(m.grid_bbox.x_min * w / 1000)), int(math.floor(m.grid_bbox.y_min * h / 1000))
        x2, y2 = max(x1 + 1, int(math.ceil(m.grid_bbox.x_max * w / 1000))), max(y1 + 1, int(math.ceil(m.grid_bbox.y_max * h / 1000)))
        sub = bm[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        share = float(sub.mean()) if sub.size else 0.0
        if share > float(pb["mark_max"]):
            out[m.mark_id] = PrivacyDrop(mark_id=m.mark_id, kf_id=kf.kf_id, share=round(share, 4))
    return out


def privacy_drop_words(drops: list[PrivacyDrop]) -> str:
    """how_checked's words for the marks left out under the privacy blur."""
    return "; ".join(
        f"left out: {d.mark_id} sits where {d.kf_id} is blurred for privacy ({math.floor(d.share * 100 + 0.5)}% of its box), so it was not matched"
        for d in drops
    )


def marks_on_panel(th: dict | None) -> bool:
    """thresholds dedup.marks_on_panel (friend run 1, 11/2): a mark off its own panel mask was moved or left out by the
    marks stage, and match leaves the left-out ones out; off when absent."""
    from .marks import on_panel_frac

    return on_panel_frac(th) is not None


def _panel_words(pid: PanelId) -> str:
    return pid.value.replace("_", " ")


def off_panel_words(pm: PanelMatch) -> str:
    """how_checked's words for the marks the on-panel rule moved onto, moved off or left off this panel."""
    out = []
    for n in pm.off_panel:
        pct = f"{n.off_frac * 100:.1f}% of the crop's long side off the {_panel_words(n.from_panel)}"
        if n.to_panel is None:
            out.append(f"left out: {n.mark_id} sits {pct} and on no other panel tracked in that frame, so it was not matched")
        elif n.to_panel == pm.panel_id:
            out.append(f"moved here: {n.mark_id} was found on the {_panel_words(n.from_panel)} keyframe but sits on this panel ({pct})")
        else:
            out.append(f"moved: {n.mark_id} sits {pct} and on the {_panel_words(n.to_panel)}, so it was matched there")
    return "; ".join(out)


def off_panel_notes(marks: list[Mark]) -> dict[PanelId, list[OffPanelNote]]:
    """The on-panel rule's notes per panel: a left-out mark on its own panel, a moved one on both panels."""
    notes: dict[PanelId, list[OffPanelNote]] = {}
    for m in marks:
        op = m.off_panel
        if op is None:
            continue
        to = m.panel_id if op.moved_to_kf is not None else None
        note = OffPanelNote(mark_id=m.mark_id, from_panel=op.from_panel, to_panel=to, off_frac=op.off_frac)
        notes.setdefault(op.from_panel, []).append(note)
        if to is not None:
            notes.setdefault(to, []).append(note)
    return notes


# Panel gates (section 6, G1 and G3 to G10; G2 retired 10/31) ----------------------------------------------------------


def glimpsed_compare(th: dict | None) -> bool:
    """thresholds matching.glimpsed_compare (R29): a panel one walk only glimpsed is compared when both walks have a
    keyframe and the pair passes the view gates; absent from every run frozen before it."""
    cfg = (th or {}).get("matching", {}).get("glimpsed_compare")
    return isinstance(cfg, dict) and bool(cfg.get("enabled"))


def glimpsed_pair(panel: Panel) -> bool:
    """True when either walk's coverage of the panel is not filmed (glimpsed, or missing from per_role)."""
    return any(panel.per_role.get(role) is None or panel.per_role[role].state != CoverageState.FILMED for role in ROLES)


def spot_checks(
    pm: PanelMatch,
    panel_marks: dict[str, list[Mark]],
    kf_by_id: dict[str, Keyframe],
    *,
    run_id: str,
    cache: "RegistrationCache | None",
    masks: dict[str, np.ndarray | None],
) -> list[SpotCheck]:
    """R29, on a pair compared though a walk only glimpsed the panel: for each mark seen in one walk only (unmatched,
    or either mark of a pair beyond tau_eye_r) that registration did not find in both crops, where it lands in the other
    walk's keyframe of the pair. Through the accepted crop fit when there is one, else through the moment-whitened panel
    frame (the handle residual undone, as the match compares); inside when that point is on the other keyframe's panel
    mask. A mark that cannot be placed gets via none, inside None."""
    from . import register

    if pm.state != MatchState.COMPARABLE or not pm.kf_pair:
        return []
    by_id = {m.mark_id: m for role in ROLES for m in panel_marks[role]}
    found = {r.mark_id for r in pm.registration_marks if r.used in ("already_there", "paired", "pair_kept")}
    one_walk = [*pm.unmatched_pickup, *pm.unmatched_return]
    for pair in pm.pairs:
        if pair.d >= pair.tau_eye_r:
            one_walk.extend([pair.pickup_mark, pair.return_mark])
    out: list[SpotCheck] = []
    for mid in dict.fromkeys(one_walk):
        m = by_id.get(mid)
        if m is None or mid in found or m.role not in ROLES:
            continue
        kf_o = kf_by_id.get(pm.kf_pair[1] if m.role == "pickup" else pm.kf_pair[0])
        if kf_o is None:
            continue
        point: tuple[float, float] | None = None
        via = "none"
        if cache is not None and m.bbox_crop is not None:
            al = cache.fit(m.kf_id, kf_o.kf_id) if m.role == "pickup" else cache.fit(kf_o.kf_id, m.kf_id)
            if al is not None and al.ok and al.H is not None:
                if m.role != "pickup":
                    al = register.inverse(al)
                x, y = register.apply(al.H, np.array([_centre(m.bbox_crop)], dtype=float))[0]
                point, via = (float(x), float(y)), "registration"
        if point is None and m.w is not None and kf_o.moments is not None and kf_o.front_sign is not None:
            du, dv = (float(pm.handle_residual[0]), float(pm.handle_residual[1])) if pm.handle_residual is not None else (0.0, 0.0)
            sign = -1.0 if m.role == "return" else 1.0
            try:
                point = geometry.unwhiten((float(m.w[0]) + sign * du, float(m.w[1]) + sign * dv), kf_o.moments, kf_o.front_sign)
                via = "panel_frame"
            except (ValueError, np.linalg.LinAlgError):
                point = None
        if point is None:
            out.append(SpotCheck(mark_id=mid, other_kf=kf_o.kf_id, via="none"))
            continue
        if kf_o.kf_id not in masks:
            masks[kf_o.kf_id] = _load_mask(run_id, kf_o)
        mask = masks[kf_o.kf_id]
        x, y = int(math.floor(point[0])), int(math.floor(point[1]))
        inside = mask is not None and 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])
        out.append(SpotCheck(mark_id=mid, other_kf=kf_o.kf_id, via=via, point=(round(point[0], 2), round(point[1], 2)), inside=inside))  # type: ignore[arg-type]
    return out


def spot_words(pm: PanelMatch) -> str | None:
    """The glimpsed pair's spot checks in words for how_checked (R29)."""
    if not pm.spots:
        return None
    parts = []
    for sc in pm.spots:
        how = {"registration": "through the crop fit", "panel_frame": "through the panel frame", "none": ""}[sc.via]
        if sc.inside:
            parts.append(f"{sc.mark_id} lands on the panel in {sc.other_kf} ({how})")
        elif sc.via == "none":
            parts.append(f"{sc.mark_id} could not be placed in {sc.other_kf}, so needs your eye (K21)")
        else:
            parts.append(f"{sc.mark_id} lands off the panel in {sc.other_kf} ({how}), so needs your eye (K21)")
    return "one walk only glimpsed this panel; " + "; ".join(parts)


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
    scale_ratio: float | None = None,
    mirrored: tuple[float, float] | None = None,
) -> list[str]:
    """The failing gate ids in order G1, G3..G12; an empty list means the panel is comparable. G2 (a wet, dirty or
    glare flag on either crop) is no gate since 10/31: a flagged view caps the marks that depend on it (K15). G11 (a
    scale gap over pair_scale's max_ratio, scale_ratio measured by the caller) and G12 (a crop whose panel is more than
    privacy_blur's view_max blurred for privacy) exist only with their thresholds leaves (R27)."""
    m = th["matching"]
    failing: list[str] = []
    states = [panel.per_role.get(role).state if panel.per_role.get(role) is not None else CoverageState.NOT_SEEN for role in ROLES]
    kfs = (kf_pickup, kf_return)
    if any(kf is None for kf in kfs) or (any(s != CoverageState.FILMED for s in states) and not glimpsed_compare(th)):
        failing.append("G1")  # R29 (matching.glimpsed_compare): a keyframe in each walk is enough; the view gates decide
    if any(kf is not None and kf.score.sharpness < float(m["sharpness_rel_min"]) for kf in kfs):
        failing.append("G3")
    paired = kf_pickup is not None and kf_return is not None
    if paired and (mask_iou is None or mask_iou < float(m["mask_iou_min"])):
        failing.append("G4")
    if residual is not None and math.hypot(residual[0], residual[1]) > calibration.tau_pre:
        failing.append("G5")
    free = panel.panel_id in orientation_free_panels(th)  # R28: the mirrors carry no front sign; their side is their own
    if panel.side_uncertain or (panel.front_sign is None and not free) or any(kf is not None and kf.front_sign is None for kf in kfs):
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
    ps = pair_scale(th)
    if ps is not None and paired and scale_ratio is not None and scale_ratio > float(ps["max_ratio"]):
        failing.append("G11")
    pb = privacy_blur(th)
    if pb is not None and any(kf is not None and (kf.privacy_blurred_frac or 0.0) > float(pb["view_max"]) for kf in kfs):
        failing.append("G12")
    tm = twin_mirror(th)
    if tm is not None and paired and panel.panel_id in TWIN_PANELS and mirrored is not None and mirrored[1] > mirrored[0] + float(tm["margin"]):
        failing.append("G13")
    elif free and paired and kf_pickup.front_sign is not None and kf_return.front_sign is not None and kf_pickup.front_sign != kf_return.front_sign:  # type: ignore[union-attr]
        failing.append("G13")  # R28: the two keyframes' tracks were given different sides, so they show the two mirrors
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
    (docs/evidence/D1_MEMO.md note 2: not calibrated). Traces keep reading the full rows for their request ids."""
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
    counters: dict[str, LookVerdict | CounterAnswer | None] | None = None,
    counter_skipped: dict[str, int] | None = None,
    look_tallies: dict[tuple[str, str], tuple[int, int]] | None = None,
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
    look never asked. Marks seen in one walk only and pairs in the band keep K4 and K12 as caps.

    The counter-look (lead 11/1, third): counters maps a mark seen in one walk only that went to it (derive_match picks
    them through counter_asks) to the answer, None when no usable answer came; counter_skipped maps a mark the run's
    cap left out to the cap. counter_flags sets K16 on same or unsure and writes counter_note. With
    matching.counter_confirms_mark (lead decision 5 of 11/1) the answer is a CounterAnswer: K17 unless part 1 is yes,
    K16 unless part 2 is different."""
    counters, counter_skipped = counters or {}, counter_skipped or {}
    confirm = counter_confirms_mark(th)
    reg_capped = registration_capped(pm)
    covered = covered_by_other_walk(pm, marks, keyframes) if inside_other_box(th) else {}
    so_cfg = stands_out_config(th)
    contrast = {r.mark_id: r for r in pm.own_contrast} if so_cfg is not None else {}
    faint = {i for i, r in contrast.items() if r.result == "faint"}
    geo_cfg = geometric_already_there(th)
    geo = geometric_ids(pm) if geo_cfg is not None else {}
    off_view = {sc.mark_id for sc in pm.spots if not sc.inside}  # R29, K21

    def contrast_used(mark_id: str) -> dict[str, Any] | None:
        r = contrast.get(mark_id)
        if r is None or so_cfg is None:
            return None
        return {"own_contrast": r.contrast, "own_contrast_result": r.result, "min_contrast": so_cfg["min_contrast"]}
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

    if pm.state == MatchState.NOT_COMPARABLE and pm.reshape is not None and pm.reshape.called:
        # lead 11/1, second: a panel whose outline changed in place is a person's call, never already there or new
        rc = pm.reshape
        thresholds = _thresholds_used(calibration, th, tau_pre_r=None, tau_eye_r=None)
        thresholds.update(
            {
                "mask_iou": rc.mask_iou,
                "aspect_off": rc.aspect_off,
                "blob_share": rc.blob_share,
                **{k: th["matching"][k] for k in ("mask_iou_min", "reshape_iou_min", "reshape_aspect_tol", "reshape_open_px", "reshape_blob_share_min")},
            }
        )
        out.append(
            Verdict(
                verdict_id=vid(),
                panel_id=pm.panel_id,
                label=VerdictLabel.NEEDS_YOUR_EYE,
                rule_id="L7",
                mark_ids=sorted({m.mark_id for m in panel_marks}),
                reasons=rules.describe(["L7", "G4"]),
                trace=_trace(trace_base, ["L7", "G4"], panel_marks, mark_tracks, thresholds),
            )
        )
        return out
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
        label: VerdictLabel,
        ids: list[str],
        mark_ids: list[str],
        d_pos: float | None,
        pair: MatchCandidate | None,
        view_note: str | None = None,
        *,
        type_note: str | None = None,
        note_ids: list[str] | None = None,
        counter_note: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        involved = [marks[i] for i in mark_ids if i in marks]
        thresholds = _thresholds_used(calibration, th, tau_pre_r=pair.tau_pre_r if pair else None, tau_eye_r=pair.tau_eye_r if pair else None)
        if extra:
            thresholds.update(extra)
        if pair is not None:
            thresholds.update({"d": pair.d, "d_pos": pair.d_pos, "type_cost": pair.type_cost, "size_cost": pair.size_cost, "r_mean": pair.r_mean})
            if not pair.mutual:
                thresholds["group_pairing"] = True  # paired inside a group with the ratio test waived
            if pair.registered:
                thresholds["registered"] = True  # measured at the registered spot (matching.registration, 11/2)
                if pair.d_moment is not None:
                    thresholds["d_moment"] = pair.d_moment
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
                counter_note=counter_note,
            )
        )

    def emit_geometric(mark: Mark, is_return: bool, pair: MatchCandidate | None) -> None:
        """R29: a mark seen in one walk only whose patch the presence test found at its spot on a strong fit is already
        there (L8). The caps that only doubt the box is a damage mark (caps_ignored) do not lower it, nor K15 (it is in
        the same place in both walks); the look on the found spot and every other cap still may."""
        r = geo[mark.mark_id]
        flags = flags_for(None, mark, None, caps, None, kf_status, return_mark=is_return)
        flags = flags.model_copy(update={k: False for k in geo_cfg["caps_ignored"] if k in type(flags).model_fields})  # type: ignore[index]
        flags, note = geometric_look(mark, flags, counters)
        label, ids = verdict(0.0, 1.0, 2.0, flags)
        ids = ["L8" if i == "L3" else i for i in ids]
        extra = {"registered_present": True, "ncc": r.ncc, "uncertainty_px": r.uncertainty_px, "other_kf": r.other_kf}
        emit(label, ids, [mark.mark_id], None, pair, both_walks_note(flags_by_role([mark]), same_place=True), counter_note=note, extra=extra)

    tight, deferrable = tight_pair_look(th), tight_pair_deferrable(th)
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
                flags, notes, type_note = tight_pair_flags(pair, a, b, flags, look, asked=key in looks, deferrable=deferrable, tally=(look_tallies or {}).get(key))
            label, ids = verdict(pair.d, pair.tau_pre_r, pair.tau_eye_r, flags)
            emit(label, ids, [a.mark_id, b.mark_id], pair.d_pos, pair, both_walks_note(flags_by_role([a, b]), same_place=pair.d < pair.tau_pre_r), type_note=type_note, note_ids=notes)
        else:
            for mark, is_return in ((a, False), (b, True)):
                if mark.mark_id in geo:
                    emit_geometric(mark, is_return, pair)
                    continue
                view = flags_by_role([mark])
                flags = flags_for(pair, mark, None, caps, None, kf_status, return_mark=is_return, view_flagged=any(view.values()))
                if mark.mark_id in off_view:
                    flags = flags.model_copy(update={"K21": True})
                if mark.mark_id in reg_capped:
                    flags = flags.model_copy(update={"K18": True})
                if mark.mark_id in covered:
                    flags = flags.model_copy(update={"K19": True})
                if mark.mark_id in faint:
                    flags = flags.model_copy(update={"K20": True})
                flags, counter_note = counter_flags(mark, flags, counters, counter_skipped, confirm=confirm)
                label, ids = verdict(pair.d, pair.tau_pre_r, pair.tau_eye_r, flags)
                emit(label, ids, [mark.mark_id], None, pair, one_walk_note("return" if is_return else "pickup", view), counter_note=counter_note, extra=contrast_used(mark.mark_id))
    unmatched = set(pm.unmatched_pickup) | set(pm.unmatched_return)
    for mark_id, is_return in (*((i, False) for i in pm.unmatched_pickup), *((i, True) for i in pm.unmatched_return)):
        mark = marks.get(mark_id)
        if mark is None:
            continue
        mark = _with_k10(mark, pm, unmatched)
        if mark.mark_id in geo:
            emit_geometric(mark, is_return, None)
            continue
        r = mark.r if mark.r is not None else 0.0
        tau_pre_r = tau_r(calibration.tau_pre, r, th)
        tau_eye_r = tau_r(calibration.tau_eye, r, th)
        view = flags_by_role([mark])
        flags = flags_for(None, mark, None, caps, None, kf_status, return_mark=is_return, view_flagged=any(view.values()))
        if mark.mark_id in off_view:
            flags = flags.model_copy(update={"K21": True})
        if mark.mark_id in reg_capped:
            flags = flags.model_copy(update={"K18": True})
        if mark.mark_id in covered:
            flags = flags.model_copy(update={"K19": True})
        if mark.mark_id in faint:
            flags = flags.model_copy(update={"K20": True})
        flags, counter_note = counter_flags(mark, flags, counters, counter_skipped, confirm=confirm)
        label, ids = verdict(None, tau_pre_r, tau_eye_r, flags)
        emit(label, ids, [mark.mark_id], None, None, one_walk_note("return" if is_return else "pickup", view), counter_note=counter_note, extra=contrast_used(mark.mark_id))
    return out


# Registration (11/2, thresholds matching.registration) -------------------------------------------------------------


class RegistrationCache:
    """Crops, masks and fits of one run, so a keyframe pair is aligned once per derive_match."""

    def __init__(self, run_id: str, kf_by_id: dict[str, Keyframe], marks: list[Mark], cfg: dict[str, Any]) -> None:
        self.run_id, self.kf_by_id, self.cfg = run_id, kf_by_id, cfg
        self.boxes: dict[str, list[tuple[float, float, float, float]]] = {}
        for m in marks:
            if m.bbox_crop is not None:
                self.boxes.setdefault(m.kf_id, []).append(tuple(float(v) for v in m.bbox_crop))  # type: ignore[arg-type]
        self._px: dict[str, Any] = {}
        self._feats: dict[str, Any] = {}
        self._fits: dict[tuple[str, str], Any] = {}

    def features(self, kf_id: str) -> Any:
        from . import register

        if kf_id not in self._feats:
            px = self.pixels(kf_id)
            self._feats[kf_id] = None if px is None else register.features(px[0], px[1], self.boxes.get(kf_id, []), self.cfg)
        return self._feats[kf_id]

    def pixels(self, kf_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        if kf_id not in self._px:
            import cv2

            kf = self.kf_by_id.get(kf_id)
            img = mask = None
            if kf is not None and kf.crop is not None and kf.mask_png:
                img = cv2.imread(str(paths.run_dir(self.run_id) / kf.crop.path), cv2.IMREAD_COLOR)
                raw = cv2.imread(str(paths.run_dir(self.run_id) / kf.mask_png), cv2.IMREAD_GRAYSCALE)
                mask = None if raw is None else raw > 0
            self._px[kf_id] = None if img is None or mask is None or img.shape[:2] != mask.shape else (img, mask)
        return self._px[kf_id]

    def fit(self, kf_pickup: str, kf_return: str) -> Any:
        """register.Aligned from the pickup keyframe's crop onto the return keyframe's, or None without pixels."""
        from . import register

        key = (kf_pickup, kf_return)
        if key not in self._fits:
            pa, pb = self.pixels(kf_pickup), self.pixels(kf_return)
            ka, kb = self.kf_by_id.get(kf_pickup), self.kf_by_id.get(kf_return)
            if pa is None or pb is None or ka is None or kb is None:
                self._fits[key] = None
            else:
                prior = register.prior_map(ka.moments, ka.front_sign, kb.moments, kb.front_sign)
                self._fits[key] = register.align(
                    kf_pickup, pa[0], pa[1], kf_return, pb[0], pb[1], self.cfg, prior=prior, feats_a=self.features(kf_pickup), feats_b=self.features(kf_return)
                )
        return self._fits[key]


def _fit_record(al: Any) -> RegistrationFit:
    s = al.summary()
    return RegistrationFit(
        kf_pair=[al.kf_a, al.kf_b],
        model=s["model"],
        ok=s["ok"],
        why=s["why"],
        n_matches=s["n_matches"],
        n_inliers=s["n_inliers"],
        spread=s["spread"],
        mask_iou=s["mask_iou"],
        reproj_px=s["reproj_px"],
        H=None if al.H is None else [[round(float(v), 9) for v in row] for row in al.H],
    )


def _centre(box: Any) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def _reg_record(mark_id: str, other_kf: str, pr: Any, used: str = "none", paired_with: str | None = None) -> RegistrationMark:
    return RegistrationMark(
        mark_id=mark_id,
        other_kf=other_kf,
        result=pr.result,
        why=pr.why,
        mapped_box=None if pr.mapped_box is None else list(pr.mapped_box),
        found_box=None if pr.found_box is None else list(pr.found_box),
        ncc=pr.ncc,
        contrast_own=pr.contrast_a,
        contrast_other=pr.contrast_b,
        uncertainty_px=pr.uncertainty_px,
        used=used,  # type: ignore[arg-type]
        paired_with=paired_with,
    )


def registration_step(pm: PanelMatch, panel_marks: dict[str, list[Mark]], cache: RegistrationCache, calibration: Calibration, th: dict) -> PanelMatch:
    """The registration rules on one compared panel (thresholds matching.registration, 11/2). Pixels only, no model.

    The chosen keyframe pair, and each other keyframe that holds a mark with the other walk's keyframe it is tested in,
    are aligned (register.align). Then, for a mark at its mapped spot in the other walk's crop, register.presence says
    present, absent or cannot tell. Code decides and no label moves up:
    - a moment pair outside tau_pre_r whose spots coincide (one mark's patch is found in the other crop within
      coincide_px of the other mark) is measured there: d and d_pos from the registered spot, d_moment kept (pair_kept);
    - a mark seen in one walk only whose patch is present at its spot pairs with an unmatched mark of the other walk
      found there (paired), nearest first and one to one; with none there it is capped at needs your eye (K18, capped);
      both marks of a pair beyond tau_eye_r count as seen in one walk only;
    - absent and cannot tell change nothing; a refused alignment gives cannot tell everywhere.
    Every test is recorded on PanelMatch.registration_marks and every fit on PanelMatch.registration."""
    from . import register

    cfg = cache.cfg
    if pm.state != MatchState.COMPARABLE or pm.panel_id in WHEEL_PANELS or not pm.kf_pair:
        return pm
    kp, kr = pm.kf_pair
    by_id = {m.mark_id: m for role in ROLES for m in panel_marks[role]}
    used_fits: dict[tuple[str, str], Any] = {}

    def fit(kf_p: str, kf_r: str) -> Any:
        al = cache.fit(kf_p, kf_r)
        if al is not None:
            used_fits.setdefault((kf_p, kf_r), al)
        return al

    fit(kp, kr)

    def test(mark: Mark, other_kf: str) -> Any:
        if mark.bbox_crop is None:
            return None
        own, other = cache.pixels(mark.kf_id), cache.pixels(other_kf)
        if own is None or other is None:
            return None
        al = fit(mark.kf_id, other_kf) if mark.role == "pickup" else fit(other_kf, mark.kf_id)
        if al is None:
            return None
        if mark.role != "pickup":
            al = register.inverse(al)
        ys, xs = np.nonzero(own[1])
        diag = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min())) if xs.size else 1.0
        return register.presence(al, own[0], other[0], other[1], tuple(float(v) for v in mark.bbox_crop), cfg, diag_a=diag)  # type: ignore[arg-type]

    def coincide(pr: Any, other: Mark) -> float | None:
        if pr is None or pr.result != register.PRESENT or pr.found_box is None or other.bbox_crop is None:
            return None
        dist = math.dist(_centre(pr.found_box), _centre(other.bbox_crop))
        return dist if dist <= register.coincide_px(tuple(float(v) for v in other.bbox_crop), cfg) else None  # type: ignore[arg-type]

    def registered_pair(a: Mark, b: Mark, tested: Mark, pr: Any, d_moment: float | None) -> MatchCandidate | None:
        """The pair measured at the registered spot: the tested mark's found centre whitened in the other mark's
        keyframe frame against the other mark's own whitened position."""
        other = b if tested is a else a
        kf_o = cache.kf_by_id.get(other.kf_id)
        if kf_o is None or kf_o.moments is None or kf_o.front_sign is None or other.w is None or other.r is None:
            return None
        try:
            u, v = geometry.whiten(_centre(pr.found_box), kf_o.moments, kf_o.front_sign)
        except (ValueError, np.linalg.LinAlgError):
            return None
        w_other = WhitenedMark(mark_id=other.mark_id, role=other.role, panel_id=other.panel_id, kf_id=other.kf_id, u=other.w[0], v=other.w[1], r=other.r)
        w_tested = WhitenedMark(mark_id=tested.mark_id, role=tested.role, panel_id=tested.panel_id, kf_id=other.kf_id, u=u, v=v, r=geometry.radius((u, v)))
        wa, wb = (w_tested, w_other) if tested is a else (w_other, w_tested)
        cand = cost(a, b, wa, wb, calibration, th)
        return cand.model_copy(update={"registered": True, "d_moment": d_moment, "mutual": True})

    records: dict[str, RegistrationMark] = {}
    pairs = list(pm.pairs)
    one_walk: list[Mark] = []
    for i, pair in enumerate(pairs):
        a, b = by_id.get(pair.pickup_mark), by_id.get(pair.return_mark)
        if a is None or b is None or pair.d < pair.tau_pre_r:
            continue
        done = False
        for tested, other in ((a, b), (b, a)):
            pr = test(tested, other.kf_id)
            if pr is None:
                continue
            if coincide(pr, other) is not None:
                cand = registered_pair(a, b, tested, pr, pair.d)
                if cand is not None:
                    pairs[i] = cand
                    records[tested.mark_id] = _reg_record(tested.mark_id, other.kf_id, pr, "pair_kept", other.mark_id)
                    done = True
                    break
            records.setdefault(tested.mark_id, _reg_record(tested.mark_id, other.kf_id, pr))
        if not done and pair.d >= pair.tau_eye_r:
            one_walk.extend([a, b])
    unmatched = {role: [by_id[i] for i in (pm.unmatched_pickup if role == "pickup" else pm.unmatched_return) if i in by_id] for role in ROLES}
    proposals: list[tuple[float, str, str, str, Any]] = []
    present_at: dict[str, tuple[str, Any]] = {}
    for m in [*unmatched["pickup"], *unmatched["return"], *one_walk]:
        other_role = "return" if m.role == "pickup" else "pickup"
        chosen = kr if m.role == "pickup" else kp
        others = unmatched[other_role]
        kfs = [chosen, *sorted({o.kf_id for o in others} - {chosen})]
        for kf_o in kfs:
            pr = test(m, kf_o)
            if pr is None:
                continue
            if kf_o == chosen and m.mark_id not in records:
                records[m.mark_id] = _reg_record(m.mark_id, kf_o, pr)
            if pr.result != register.PRESENT:
                continue
            present_at.setdefault(m.mark_id, (kf_o, pr))
            if m in one_walk:
                continue  # a pair beyond tau_eye_r keeps its two marks; only the cap applies
            for o in others:
                if o.kf_id != kf_o:
                    continue
                dist = coincide(pr, o)
                if dist is not None:
                    proposals.append((round(dist, 6), m.mark_id, o.mark_id, kf_o, pr))
    taken: set[str] = set()
    new_pairs: list[MatchCandidate] = []
    for dist, mid, oid, kf_o, pr in sorted(proposals, key=lambda t: (t[0], t[1], t[2])):
        if mid in taken or oid in taken:
            continue
        m, o = by_id[mid], by_id[oid]
        a, b = (m, o) if m.role == "pickup" else (o, m)
        cand = registered_pair(a, b, m, pr, None)
        if cand is None:
            continue
        taken.update({mid, oid})
        new_pairs.append(cand)
        records[mid] = _reg_record(mid, kf_o, pr, "paired", oid)
        if oid in records and records[oid].used == "none":
            records[oid] = records[oid].model_copy(update={"paired_with": mid})
    geo = geometric_already_there(th)
    for m in [*unmatched["pickup"], *unmatched["return"], *one_walk]:
        if m.mark_id in taken or m.mark_id not in present_at:
            continue
        kf_o, pr = present_at[m.mark_id]
        al = fit(m.kf_id, kf_o) if m.role == "pickup" else fit(kf_o, m.kf_id)
        # R29: code calls it already there from the pixels; dirt and reflections seen twice stay capped for the debris line
        strong = geo is not None and m.type.value not in geo["types_excluded"] and strong_fit(al, pr, geo)
        records[m.mark_id] = _reg_record(m.mark_id, kf_o, pr, "already_there" if strong else "capped")
    order = [i for i in (*(p.pickup_mark for p in pm.pairs), *(p.return_mark for p in pm.pairs), *pm.unmatched_pickup, *pm.unmatched_return) if i in records]
    return pm.model_copy(
        update={
            "pairs": [*pairs, *new_pairs],
            "unmatched_pickup": [i for i in pm.unmatched_pickup if i not in taken],
            "unmatched_return": [i for i in pm.unmatched_return if i not in taken],
            "registration": [_fit_record(al) for al in used_fits.values()],
            "registration_marks": [records[i] for i in dict.fromkeys(order)],
        }
    )


def wheel_geometric_step(pm: PanelMatch, panel_marks: dict[str, list[Mark]], cache: RegistrationCache, cfg: dict[str, Any]) -> PanelMatch:
    """R29 on a compared wheel (matching.geometric_already_there wheels): the registration rules skip wheels, since a rim
    turns between walks and its outline cannot say how far. Here each mark seen in one walk only is tested at its spot in
    the other walk's keyframe of the pair, and only a fit on matched edges counts as strong (the outline gate drops edges
    that turned with the rim, so a turned wheel gets no such fit). Present on a strong fit: already there (L8). Nothing
    else changes: no cap, no pair. Pixels only."""
    from . import register

    if pm.state != MatchState.COMPARABLE or not pm.kf_pair:
        return pm
    kp, kr = pm.kf_pair
    by_id = {m.mark_id: m for role in ROLES for m in panel_marks[role]}
    al_pr = cache.fit(kp, kr)
    if al_pr is None:
        return pm
    records: list[RegistrationMark] = []
    for mid in (*pm.unmatched_pickup, *pm.unmatched_return):
        m = by_id.get(mid)
        if m is None or m.bbox_crop is None or m.role not in ROLES:
            continue
        other_kf = kr if m.role == "pickup" else kp
        own, other = cache.pixels(m.kf_id), cache.pixels(other_kf)
        al = cache.fit(m.kf_id, other_kf) if m.role == "pickup" else cache.fit(other_kf, m.kf_id)
        if own is None or other is None or al is None:
            continue
        al_m = al if m.role == "pickup" else register.inverse(al)
        ys, xs = np.nonzero(own[1])
        diag = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min())) if xs.size else 1.0
        pr = register.presence(al_m, own[0], other[0], other[1], tuple(float(v) for v in m.bbox_crop), cache.cfg, diag_a=diag)  # type: ignore[arg-type]
        strong = pr.result == register.PRESENT and al.model == "features" and m.type.value not in cfg["types_excluded"] and strong_fit(al, pr, cfg)
        records.append(_reg_record(mid, other_kf, pr, "already_there" if strong else "none"))
    if not records:
        return pm
    return pm.model_copy(update={"registration": [_fit_record(al_pr)], "registration_marks": records})


def registration_capped(pm: PanelMatch) -> set[str]:
    return {r.mark_id for r in pm.registration_marks if r.used == "capped"}


def registration_words(pm: PanelMatch) -> str | None:
    """The registration step in words for how_checked: each fit and what each tested mark showed. A panel with no
    mark tested prints nothing (its fit is still in matches.json)."""
    if not pm.registration or not pm.registration_marks:
        return None
    parts = []
    for f in pm.registration:
        how = "the panel outline" if f.model == "outline" else "matched edges and the outline"
        scores = f"mask IoU {f.mask_iou:.2f}, {f.n_inliers} of {f.n_matches} matched points agree"
        parts.append(f"registration {f.kf_pair[0]} onto {f.kf_pair[1]} by {how} ({scores})" + ("" if f.ok else f", refused: {f.why}"))
    unsure = 0
    for r in pm.registration_marks:
        other = "return" if r.other_kf.startswith("return") else "pickup"
        ncc = f"NCC {r.ncc:.2f}" if r.ncc is not None else "no NCC"
        if r.used == "capped":
            parts.append(f"{r.mark_id}: the {other} crop shows the same patch at its spot ({ncc}) and no mark there, so needs your eye (K18)")
        elif r.used == "already_there":
            parts.append(f"{r.mark_id}: the {other} crop shows the same patch at its spot ({ncc}) on a strong fit, so already there (L8)")
        elif r.used == "paired":
            parts.append(f"{r.mark_id}: paired with {r.paired_with}, the same patch at the same spot in the {other} crop ({ncc})")
        elif r.used == "pair_kept":
            parts.append(f"{r.mark_id}: its pair with {r.paired_with} sits at the same spot once the crops are aligned ({ncc}), so measured there")
        elif r.result == "present":
            parts.append(f"{r.mark_id}: the same patch is at its spot in the {other} crop ({ncc})")
        elif r.result == "absent":
            parts.append(f"{r.mark_id}: not at its spot in the {other} crop ({ncc}); recorded only")
        else:
            unsure += 1
    if unsure:
        parts.append(f"{unsure} mark{'s' if unsure != 1 else ''} could not be checked at {'their' if unsure != 1 else 'its'} spot")
    return "; ".join(parts)


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
    scales: dict[str, float | None] | None = None,
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
    ratio = scale_ratio(scales.get(kf_p.kf_id), scales.get(kf_r.kf_id)) if scales is not None else None
    tm = twin_mirror(th)
    mirrored = None
    if tm is not None and panel.panel_id in TWIN_PANELS:
        for kf in (kf_p, kf_r):
            if kf.kf_id not in masks:
                masks[kf.kf_id] = _load_mask(run_id, kf)
        mirrored = mirror_ious(masks[kf_p.kf_id], masks[kf_r.kf_id], int(tm["grid"]))
    gates = panel_gates(panel, kf_p, kf_r, mask_iou=mask_iou, residual=residual, lane_status=lane_status, marks_truncated_all=truncated_all, calibration=calibration, th=th, scale_ratio=ratio, mirrored=mirrored)
    return KeyframePairTry(kf_pair=[kf_p.kf_id, kf_r.kf_id], gates=gates, mask_iou=mask_iou, frames_agree=agree)


def reshape_check(run_id: str, tries: list[KeyframePairTry], kf_by_id: dict[str, Keyframe], th: dict, masks: dict[str, np.ndarray | None] | None = None) -> ReshapeCheck | None:
    """The reshaped-panel test (thresholds matching.reshaped_panel, lead 11/1, second). Its precondition: G4 is the only
    failing gate on every keyframe pair tried and the best measured whitened mask IoU (the first pair on a tie) is in
    [reshape_iou_min, mask_iou_min). Then on that pair: the two keyframes' box aspect ratios agree within
    reshape_aspect_tol (a turned camera changes the aspect) and the largest blob of the opened mask difference holds at
    least reshape_blob_share_min of it (a turned camera spreads the difference along the outline). None when the
    precondition fails; called says both tests passed. It compares no marks, so it never says already there or new."""
    m = th["matching"]
    if not tries or any(t.gates != ["G4"] for t in tries):
        return None
    measured = [t for t in tries if t.mask_iou is not None]
    if not measured:
        return None
    best = max(measured, key=lambda t: t.mask_iou)  # type: ignore[arg-type, return-value]
    if not float(m["reshape_iou_min"]) <= float(best.mask_iou) < float(m["mask_iou_min"]):  # type: ignore[arg-type]
        return None
    kf_p, kf_r = kf_by_id.get(best.kf_pair[0]), kf_by_id.get(best.kf_pair[1])
    if kf_p is None or kf_r is None or kf_p.moments is None or kf_r.moments is None or kf_p.front_sign is None or kf_r.front_sign is None:
        return None
    ra, rb = float(kf_p.aspect_ratio), float(kf_r.aspect_ratio)
    off = abs(ra - rb) / max(ra, rb) if max(ra, rb) > 0 else 1.0
    masks = {} if masks is None else masks
    for kf in (kf_p, kf_r):
        if kf.kf_id not in masks:
            masks[kf.kf_id] = _load_mask(run_id, kf)
    ma, mb = masks[kf_p.kf_id], masks[kf_r.kf_id]
    share = None
    if ma is not None and mb is not None:
        share = geometry.mask_difference_blob_share(ma, kf_p.moments, kf_p.front_sign, mb, kf_r.moments, kf_r.front_sign, open_px=int(m["reshape_open_px"]))
    called = off <= float(m["reshape_aspect_tol"]) and share is not None and share >= float(m["reshape_blob_share_min"])
    return ReshapeCheck(kf_pair=list(best.kf_pair), mask_iou=float(best.mask_iou), aspect_ratios=[ra, rb], aspect_off=off, blob_share=share, called=called)  # type: ignore[arg-type]


def reshape_words(rc: ReshapeCheck) -> str:
    """The reshaped-panel test in words for the how-checked line (lib/lines.ts reshapeWords mirrors it)."""
    head = "outline changed in place (L7)" if rc.called else "not called reshaped"
    blob = "nothing left after the opening" if rc.blob_share is None else f"largest blob {rc.blob_share * 100:.0f}% of the mask difference"
    return (
        f"{head}: keyframes {rc.kf_pair[0]} with {rc.kf_pair[1]}, IoU {rc.mask_iou:.2f}, box aspect ratios "
        f"{rc.aspect_ratios[0]:.3f} and {rc.aspect_ratios[1]:.3f} ({rc.aspect_off * 100:.1f}% apart), {blob}"
    )


def derive_match(
    run_id: str,
    *,
    th: dict,
    second_look: Callable[[MatchCandidate], LookVerdict | None] | None = None,
    counter_look: Callable[[CounterAsk], LookVerdict | CounterAnswer | None] | None = None,
) -> tuple[list[PanelMatch], list[Verdict], Calibration]:
    """Read panels, keyframes, marks, mark_tracks and calibration; gate, pair and match every panel with a keyframe
    in either walk; ask the optional second look on provisional already_there pairs (at most
    second_looks_per_pair_max); write matches.json and verdicts.json. Under a second without the second look.

    With thresholds matching.next_keyframe_pair (rule 2) a panel whose chosen keyframe pair fails only view gates is
    compared on the first pair in keyframe_pair_order that passes every gate, and kf_pairs_tried records each pair
    tried; when none passes, the panel keeps the chosen pair and its gates. With matching.tight_pair_second_look
    (rule 1) K4 and K12 alone never keep a pair inside tau_pre_r from the second look.

    With matching.one_walk_counter_look (lead 11/1, third), once every panel is matched, each mark seen in one walk only
    whose label came out new_or_unseen or not_seen_at_return with no cap goes to counter_look (default: the second
    look's own counter method, when it has one), surest first and at most matching.counter_looks_max per run; same or
    unsure caps it at needs your eye (K16). With matching.counter_confirms_mark (lead decision 5 of 11/1) each ask
    carries confirm_mark and the look also says whether the box on the mark's own crop holds a damage mark: anything
    but yes there caps it too (K17).

    With matching.fixtures_not_damage (lead decision 4 of 11/1) a mark whose box sits on a factory fixture of its own
    crop (fixture_drops) is left out of every panel's marks before the gates and the match, so it gets no verdict, and is
    recorded on PanelMatch.fixtures, which how_checked prints.

    With matching.registration_vouches (lead decision 1 of 11/2), when no keyframe pair of a panel passes every gate, the
    first tried pair whose only failing gate is G4 (on a measured IoU, frames agreeing) and whose registration fit passes
    vouch_check is compared instead; every pair asked records the vouch on its KeyframePairTry. With
    matching.look_majority (lead decision 2 of 11/2) each pair look and each two-question counter-look is asked up to
    asks_max times (ask_majority) and the label follows majority_pair_look or majority_counter."""
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
    keyframes, marks, kf_sides = assign_sides(run_id, keyframes, marks, th)  # R28: the mirrors' sides from their tracks
    whitened = whitened_from_marks(marks)
    marks_by_id = {m.mark_id: m for m in marks}
    kf_by_id = {kf.kf_id: kf for kf in keyframes}
    fixtures = fixture_drops(marks, keyframes, _read_list(paths.spark_json(run_id), SparkRequestRecord), th) if fixtures_not_damage(th) else {}
    blurred = privacy_drops(run_id, marks, kf_by_id, th)
    on_panel = marks_on_panel(th)
    left_out = {m.mark_id for m in marks if on_panel and m.off_panel is not None and m.off_panel.moved_to_kf is None}
    off_notes = off_panel_notes(marks) if on_panel else {}
    passes = second_pass_words(run_id, marks, th)
    lane_status = _lane_status(run_id)
    trace_base = Trace(config_hash=config_hash(th))
    looks_budget = int(th["spark"]["second_looks_per_pair_max"])
    looks_used = 0
    retry = next_keyframe_pair(th)
    reshaped = reshaped_panel(th)
    deferred = tight_pair_deferrable(th)
    reg_cfg = registration_config(th)
    reg_cache = RegistrationCache(run_id, kf_by_id, marks, reg_cfg) if reg_cfg is not None else None
    vouch_cfg = registration_vouches(th)
    majority = look_majority(th)
    tallies: dict[tuple[str, str], tuple[int, int]] = {}
    vouch_cache = (reg_cache if reg_cache is not None else RegistrationCache(run_id, kf_by_id, marks, vouch_fit_config(th))) if vouch_cfg is not None else None
    by_eye = by_eye_panels(th)
    glass = frozenset(by_eye)  # glass and, with R30's leaf, the wheels: one check-by-eye line, no verdict
    sw_cfg = same_walk_config(th)
    sw_cache = (reg_cache if reg_cache is not None and reg_cache.cfg == sw_cfg else RegistrationCache(run_id, kf_by_id, marks, sw_cfg)) if sw_cfg is not None else None
    so_cfg = stands_out_config(th)
    so_cache = (reg_cache or sw_cache or RegistrationCache(run_id, kf_by_id, marks, so_cfg)) if so_cfg is not None else None  # pixels only
    copy_cache = (reg_cache or sw_cache or RegistrationCache(run_id, kf_by_id, marks, registration_config(th) or registration_defaults())) if same_walk_copies(th) is not None else None
    as_placed = {m.mark_id: m for m in marks}  # before the same-walk step re-places any
    masks: dict[str, np.ndarray | None] = {}
    scales = keyframe_scales(run_id, keyframes, masks) if pair_scale(th) is not None else None
    matches: list[PanelMatch] = []
    verdicts: list[Verdict] = []
    staged: list[tuple[PanelMatch, dict[tuple[str, str], LookVerdict | None]]] = []
    for panel in panels:
        kfs = {role: [kf for kf in keyframes if kf.panel_id == panel.panel_id and kf.role == role] for role in ROLES}
        if not kfs["pickup"] and not kfs["return"]:
            continue
        panel_marks = {role: [m for m in marks if m.panel_id == panel.panel_id and m.role == role and m.mark_id not in fixtures and m.mark_id not in left_out and m.mark_id not in blurred] for role in ROLES}
        glass_ids: list[str] = []
        if panel.panel_id in glass:  # lead Nov 2: glass gets one check-by-eye line, its marks no verdict (R30: wheels too)
            glass_ids = sorted(m.mark_id for role in ROLES for m in panel_marks[role])
            panel_marks = {role: [] for role in ROLES}
        truncated_all = {role: bool(kfs[role]) and all(kf.marks_status == MarksStatus.TRUNCATED for kf in kfs[role]) for role in ROLES}
        if kfs["pickup"] and kfs["return"]:
            order = keyframe_pair_order(kfs["pickup"], kfs["return"], th=th, scales=scales)
            inputs: dict[str, Any] = {"lane_status": lane_status, "truncated_all": truncated_all, "calibration": calibration, "th": th, "masks": masks}
            if scales is not None:
                inputs["scales"] = scales
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
            if vouch_cache is not None and vouch_cfg is not None and used.gates and panel.panel_id not in WHEEL_PANELS and panel.panel_id not in glass:
                # lead decision 1 of 11/2: no pair passed every gate; where G4 alone failed on a measured IoU, a strong
                # registration fit may pass it (a native pass is preferred, so this runs only after the loop above)
                for i, t in enumerate(tries):
                    if t.gates != ["G4"] or t.mask_iou is None or not t.frames_agree:
                        continue
                    al = vouch_cache.fit(t.kf_pair[0], t.kf_pair[1])
                    if al is None:
                        continue
                    check = vouch_check(al, vouch_cfg)
                    tries[i] = t.model_copy(update={"vouch": check, "gates": [] if check.vouched else list(t.gates)})
                    if check.vouched:
                        kf_p, kf_r, used = kf_by_id[t.kf_pair[0]], kf_by_id[t.kf_pair[1]], tries[i]
                        break
            sw_notes: list[SameWalkNote] = []
            if sw_cache is not None and not used.gates and panel.panel_id not in WHEEL_PANELS:
                panel_marks, placed, sw_notes = same_walk_step(panel_marks, {"pickup": kf_p, "return": kf_r}, sw_cache, th)
                for m in placed:
                    marks_by_id[m.mark_id] = m
                    whitened[m.mark_id] = whitened_from_marks([m])[m.mark_id]
            pm = match_panel(panel, panel_marks["pickup"], panel_marks["return"], whitened, (kf_p, kf_r), list(used.gates), calibration, th)
            if sw_notes:
                pm = pm.model_copy(update={"same_walk": sw_notes})
            pm = pm.model_copy(update={"mask_iou": used.mask_iou, "kf_pairs_tried": tries if retry or any(t.vouch is not None for t in tries) else []})
            if reshaped and pm.state == MatchState.NOT_COMPARABLE:
                pm = pm.model_copy(update={"reshape": reshape_check(run_id, tries, kf_by_id, th, masks)})
            geo_wheels = geometric_already_there(th)
            if reg_cache is not None and geo_wheels is not None and geo_wheels.get("wheels") and panel.panel_id in WHEEL_PANELS and panel.panel_id not in glass:
                try:  # R29: a wheel found in both crops on an edge fit
                    pm = wheel_geometric_step(pm, panel_marks, reg_cache, geo_wheels)
                except Exception as exc:  # pixels only: a failure here must not stop the match
                    failed = RegistrationFit(kf_pair=list(pm.kf_pair or [kf_p.kf_id, kf_r.kf_id]), model="none", ok=False, why=f"registration failed: {type(exc).__name__}")
                    pm = pm.model_copy(update={"registration": [failed]})
            if reg_cache is not None and panel.panel_id not in glass:
                try:
                    pm = registration_step(pm, panel_marks, reg_cache, calibration, th)
                except Exception as exc:  # pixels only and never a verdict: a failure here must not stop the match
                    failed = RegistrationFit(kf_pair=list(pm.kf_pair or [kf_p.kf_id, kf_r.kf_id]), model="none", ok=False, why=f"registration failed: {type(exc).__name__}")
                    pm = pm.model_copy(update={"registration": [failed]})
            if so_cache is not None and so_cfg is not None and panel.panel_id not in glass:
                pm = stands_out_step(pm, as_placed, so_cache, so_cfg)
            if glimpsed_compare(th) and glimpsed_pair(panel) and panel.panel_id not in glass and panel.panel_id not in WHEEL_PANELS:
                checks = spot_checks(pm, panel_marks, kf_by_id, run_id=run_id, cache=reg_cache or sw_cache or so_cache, masks=masks)
                if checks:
                    pm = pm.model_copy(update={"spots": checks})
        else:
            gates = panel_gates(panel, kfs["pickup"][0] if kfs["pickup"] else None, kfs["return"][0] if kfs["return"] else None, mask_iou=None, residual=None, lane_status=lane_status, marks_truncated_all=truncated_all, calibration=calibration, th=th)
            pm = PanelMatch(
                panel_id=panel.panel_id,
                state=MatchState.NOT_COMPARABLE,
                reasons=gates,
                unmatched_pickup=[m.mark_id for m in panel_marks["pickup"]],
                unmatched_return=[m.mark_id for m in panel_marks["return"]],
            )
        sided = {k: side for k, side in kf_sides.items() if k in kf_by_id and kf_by_id[k].panel_id == panel.panel_id}
        if sided:
            pm = pm.model_copy(update={"sides": sided})
        dropped = [fixtures[m.mark_id] for m in marks if m.panel_id == panel.panel_id and m.mark_id in fixtures]
        if dropped:
            pm = pm.model_copy(update={"fixtures": dropped})
        hidden = [blurred[m.mark_id] for m in marks if m.panel_id == panel.panel_id and m.mark_id in blurred]
        if hidden:
            pm = pm.model_copy(update={"privacy_drops": hidden})
        if off_notes.get(panel.panel_id):
            pm = pm.model_copy(update={"off_panel": off_notes[panel.panel_id]})
        if passes.get(panel.panel_id):
            pm = pm.model_copy(update={"second_pass": passes[panel.panel_id]})
        if panel.panel_id in glass:
            kind, text = by_eye[panel.panel_id]
            pm = pm.model_copy(update={"by_eye": [ByEyeLine(kind=kind, text=text, mark_ids=glass_ids)]})
        looks: dict[tuple[str, str], LookVerdict | None] = {}  # None: asked, no usable answer
        if second_look is not None and pm.state == MatchState.COMPARABLE:
            kf_status = {kf.kf_id: kf.marks_status for kf in keyframes if kf.panel_id == pm.panel_id}
            for pair in pm.pairs:
                a, b = marks_by_id.get(pair.pickup_mark), marks_by_id.get(pair.return_mark)
                if a is None or b is None or looks_used >= looks_budget:
                    continue
                if _provisional_already_there(pair, a, b, staysput_caps(mark_tracks, th), kf_status, deferred=deferred):
                    key = (pair.pickup_mark, pair.return_mark)
                    if majority is None:
                        looks[key] = second_look(pair)
                    else:  # lead decision 2 of 11/2: up to asks_max answers, the label follows their majority
                        looks[key], votes, usable = majority_pair_look(ask_majority(lambda: second_look(pair), majority))  # noqa: B023
                        tallies[key] = (votes, usable)
                    looks_used += 1
        staged.append((pm, looks))
    counter = counter_look if counter_look is not None else getattr(second_look, "counter", None)
    counters: dict[str, LookVerdict | CounterAnswer | None] = {}
    skipped: dict[str, int] = {}
    if one_walk_counter_look(th) and counter is not None:
        asks = [a for pm, looks in staged for a in counter_asks(pm, verdicts_for(pm, marks_by_id, mark_tracks, looks, kf_by_id, calibration, trace_base, th), marks_by_id, kf_by_id, calibration, th)]
        cap = counter_looks_max(th)
        ranked = rank_counter_asks(asks, marks_by_id)
        for ask in ranked[:cap]:
            if majority is not None and ask.confirm_mark:  # lead decision 2 of 11/2: K17 (and K16 with it) by majority
                counters[ask.mark_id] = counter_result(ask_majority(lambda: counter(ask), majority))  # noqa: B023
            else:
                counters[ask.mark_id] = counter(ask)
        skipped = {ask.mark_id: cap for ask in ranked[cap:]}
    geo_cfg = geometric_already_there(th)
    if geo_cfg is not None and counter is not None and int(geo_cfg["looks_max"]) > 0:
        # R29: a mark code called already there from the pixels gets its own look at the found spot, after the counter-
        # looks and on its own budget, so the counter-looks a run asked (and stores) are the same with or without it
        geo_asks = [a for pm, _looks in staged for a in geometric_asks(pm, marks_by_id, kf_by_id)]
        for ask in rank_counter_asks(geo_asks, marks_by_id)[: int(geo_cfg["looks_max"])]:
            counters[ask.mark_id] = counter(ask)
    for pm, looks in staged:
        if pm.panel_id in glass:
            matches.append(pm)  # the glass (or wheels) line stands for the panel: no verdict, not even not comparable
            continue
        panel_verdicts = verdicts_for(pm, marks_by_id, mark_tracks, looks, kf_by_id, calibration, trace_base, th, counters, skipped, tallies)
        pm, panel_verdicts = copies_step(pm, panel_verdicts, marks_by_id, whitened, copy_cache, th)  # R28, decision 1
        pm, panel_verdicts = faint_step(pm, panel_verdicts, th)  # R28, decision 3
        pm, panel_verdicts = debris_split(pm, panel_verdicts, marks_by_id, kf_by_id, th)
        matches.append(pm)
        verdicts.extend(panel_verdicts)
    write_json(paths.matches_json(run_id), matches)
    write_json(paths.verdicts_json(run_id), verdicts)
    return matches, verdicts, calibration


def second_pass_words(run_id: str, marks: list[Mark], th: dict) -> dict[PanelId, list[str]]:
    """The second marks pass per panel in words, from the stored asks (thresholds matching.second_marks_pass): the crop
    looked at, the other walk's mark whose spot it was, and the mark found or why none was used. Runs whose leaf lacks
    the on_panel key keep their words ("found <id>" or "no mark there", from the marks they hold)."""
    from . import second_pass

    s = second_pass.settings(th)
    if s is None:
        return {}
    if s.get("on_panel"):
        return second_pass.outcome_words(run_id, th=th)
    found = {(m.kf_id, m.second_pass.other_mark): m.mark_id for m in marks if m.second_pass is not None}
    out: dict[PanelId, list[str]] = {}
    for a in second_pass.read_asks(run_id):
        got = found.get((a.kf_id, a.other_mark))
        out.setdefault(a.panel_id, []).append(f"looked again at {a.kf_id} around {a.other_mark}: " + (f"found {got}" if got else "no mark there"))
    return out


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
