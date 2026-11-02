"""Registration of one panel's pickup and return crops (11/2, thresholds matching.registration). Pixels only.

align() maps one crop onto the other with a homography (or an affine when the homography is badly conditioned) fitted
to SIFT matches on the panel's rigid structure: the band around the SAM panel mask's outline (panel gaps, the edges of
the neighbours, handles and lights at the edge) and, inside the mask, only pixels away from the open paint, where
reflections live. Every mark box of either walk is cut out of the feature region so a mark never aligns itself. The
moment frame (the whitened map of the two masks) is the prior: a match farther than gate_frac of the panel from where
the prior puts it is dropped before RANSAC. The fit is scored by its inliers, how much of the panel they span, the mask
IoU after the warp and the inliers' reprojection error; any score under its floor refuses the alignment and nothing downstream changes.

presence() maps a mark's box through an accepted alignment into the other crop and compares the warped patch with
what is there: normalized cross-correlation over a small search window, and how much the box stands out from its own
surroundings in both crops (a colour distance, Lab). It says present, absent or cannot tell.

Determinism: keypoints are sorted before descriptors are taken, matching is brute force, and OpenCV's RANSAC uses a
fixed seed, so a rebuild gives the same numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import geometry
from .contracts import Moments

Box = tuple[float, float, float, float]

PRESENT, ABSENT, CANNOT_TELL = "present", "absent", "cannot_tell"
ECC_MAX_SCALE = 0.5  # the finest outline pass runs on half-size masks: four times faster, under a pixel on the synth truth


def config(th: dict | None) -> dict[str, Any] | None:
    """The thresholds leaf matching.registration as a plain dict, or None when the leaf is absent or not enabled."""
    raw = (th or {}).get("matching", {}).get("registration")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    return {**DEFAULTS, **raw}


DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "band_frac": 0.08,  # half width of the outline band, over sqrt(mask area)
    "paint_frac": 0.10,  # inside the mask, pixels farther than this (over sqrt(area)) from strong structure are paint
    "structure_grad": 40.0,  # Sobel magnitude (0..255 gray) that counts as rigid structure inside the mask
    "mark_pad_px": 12.0,  # mark boxes grown by this before they are cut out of the feature region
    "max_features": 4000,
    "sift_contrast": 0.02,  # half OpenCV's default: panels are smooth, the gate and RANSAC drop the extra noise
    "ratio": 0.8,
    "ecc_sigmas": [16.0, 6.0, 2.0],
    "gate_px": 25.0,  # a match farther than this from where the outline fit puts it is dropped
    "gate_frac": 0.03,  # or this share of the other mask's diagonal, whichever is larger
    "ransac_px": 4.0,
    "min_inliers": 12,  # a feature fit needs this many RANSAC inliers ...
    "min_spread": 0.15,  # ... spanning this share of the mask's bounding box
    "min_mask_iou": 0.85,  # mask IoU after a feature fit
    "min_mask_iou_outline": 0.90,  # mask IoU after an outline fit: the two outlines must agree closely
    "max_reproj_px": 3.0,
    "scale_range": [0.4, 2.5],
    "max_perspective": 0.002,
    "local_radius_frac": 0.35,  # inliers within this (over the mask diagonal) of a mark fit its local affine
    "local_min_inliers": 8,
    "patch_min_px": 24.0,
    "patch_grow": 1.0,  # context added on each side of the box, as a share of its size
    "max_uncertainty_px": 50.0,  # farther than this between the two outline fits at the mark: cannot tell
    "search_margin_px": 10.0,  # the search window spans both outline fits' spots plus this a side
    "ring_px": 10.0,
    "residual_sigma_px": 6.0,  # the high-pass blur: this or residual_sigma_box of the box's long side, larger wins
    "residual_sigma_box": 1.0,
    "residual_sigma_max": 120.0,
    "saliency_min": 3.0,  # the box must stand out by this (ring spreads) in its own crop before anything is tested
    "saliency_floor": 2.5,  # and the found spot by at least this for present
    "ncc_present": 0.6,  # present: the surroundings match this well and the spot keeps keep_frac of the saliency ...
    "keep_frac": 0.25,
    "ncc_present_kept": 0.45,  # ... or they match this well, the spot keeps keep_frac_kept and the box itself lines up
    "keep_frac_kept": 0.5,
    "ncc_mark_min": 0.5,
    "mark_pad_corr_px": 3.0,
    "ncc_absent": 0.5,  # absent: the surroundings match worse than this, no spot keeps gone_frac, the mark is strong
    "gone_frac": 0.15,
    "saliency_absent_min": 6.0,
    "coincide_px": 12.0,
    "coincide_box_frac": 0.5,
}


@dataclass
class Aligned:
    kf_a: str
    kf_b: str
    model: str  # homography, affine or none
    H: np.ndarray | None  # 3x3, crop a pixels to crop b pixels
    H_alt: np.ndarray | None = None  # outline fits: the affine fit, whose disagreement with H is the point's uncertainty
    n_matches: int = 0
    n_inliers: int = 0
    inlier_frac: float = 0.0
    spread: float = 0.0
    mask_iou: float = 0.0
    reproj_px: float = float("inf")
    ok: bool = False
    why: str | None = None  # why it was refused
    inliers_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    inliers_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    def summary(self) -> dict[str, Any]:
        return {
            "kf_a": self.kf_a,
            "kf_b": self.kf_b,
            "model": self.model,
            "ok": self.ok,
            "why": self.why,
            "n_matches": self.n_matches,
            "n_inliers": self.n_inliers,
            "inlier_frac": round(self.inlier_frac, 3),
            "spread": round(self.spread, 3),
            "mask_iou": round(self.mask_iou, 3),
            "reproj_px": None if not math.isfinite(self.reproj_px) else round(self.reproj_px, 2),
        }


@dataclass
class Presence:
    result: str
    why: str | None
    mapped_box: Box | None = None  # in crop b
    found_box: Box | None = None  # the best NCC position in crop b
    ncc: float | None = None
    contrast_a: float | None = None
    contrast_b: float | None = None
    shift_px: float | None = None
    local: bool = False  # mapped by the local affine of the inliers near the mark
    uncertainty_px: float | None = None
    ncc_mark: float | None = None  # correlation over the mark's own box at the found spot


# --------------------------------------------------------------------------------------------------------------
# Feature region and alignment
# --------------------------------------------------------------------------------------------------------------


def _disk(r: float) -> np.ndarray:
    k = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def _gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def feature_region(img: np.ndarray, mask: np.ndarray, boxes: list[Box], cfg: dict[str, Any]) -> np.ndarray:
    """uint8 0/255: the outline band of the mask plus the mask's rigid structure, minus open paint and mark boxes."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    area = float(m.sum())
    if area < 100:
        return np.zeros_like(m)
    side = math.sqrt(area)
    band = float(cfg["band_frac"]) * side
    d_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)  # distance to the mask from outside
    d_in = cv2.distanceTransform(m, cv2.DIST_L2, 5)  # distance to the outline from inside
    outer = (d_out <= band).astype(np.uint8)
    inner = (d_in > band).astype(np.uint8)
    region = outer & (1 - inner)
    gray = _gray(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    strong = (np.hypot(gx, gy) >= float(cfg["structure_grad"]) * 4.0).astype(np.uint8)
    # inside the mask keep only compact strong structure (handles, badges, trim, lights): a strong edge whose
    # neighbourhood within paint_frac is mostly strong too; long smooth reflection gradients rarely pass
    density = cv2.blur(strong.astype(np.float32), (9, 9))
    rigid = ((density >= 0.35) & (inner > 0)).astype(np.uint8)
    rigid = cv2.dilate(rigid, _disk(3))
    region = region | rigid
    pad = float(cfg["mark_pad_px"])
    h, w = m.shape
    for x1, y1, x2, y2 in boxes:
        a, b = max(0, int(math.floor(x1 - pad))), max(0, int(math.floor(y1 - pad)))
        c, d = min(w, int(math.ceil(x2 + pad))), min(h, int(math.ceil(y2 + pad)))
        region[b:d, a:c] = 0
    return (region > 0).astype(np.uint8) * 255


def features(img: np.ndarray, mask: np.ndarray, boxes: list[Box], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    """SIFT points and descriptors of one crop's feature region; a caller aligning one crop many times keeps these."""
    return _features(img, feature_region(img, np.asarray(mask) > 0, boxes, cfg), cfg)


def _features(img: np.ndarray, region: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    sift = cv2.SIFT_create(nfeatures=int(cfg["max_features"]), contrastThreshold=float(cfg["sift_contrast"]))
    gray = _gray(img)
    kps = sift.detect(gray, region)
    kps = sorted(kps, key=lambda k: (-round(k.response, 9), round(k.pt[0], 3), round(k.pt[1], 3), round(k.size, 3), round(k.angle, 3)))
    if not kps:
        return np.zeros((0, 2)), None
    kps, desc = sift.compute(gray, kps)
    pts = np.array([k.pt for k in kps], dtype=np.float64)
    return pts, desc


def prior_map(mom_a: Moments | None, sign_a: int | None, mom_b: Moments | None, sign_b: int | None) -> np.ndarray | None:
    """3x3 affine a -> b that makes the two moment frames agree; None without moments or a front sign."""
    if mom_a is None or mom_b is None or sign_a not in (1, -1) or sign_b not in (1, -1):
        return None
    try:
        wa = geometry.whitening_matrix(mom_a, sign_a)
        wb = geometry.whitening_matrix(mom_b, sign_b)
        lin = np.linalg.solve(wb, wa)
    except (ValueError, np.linalg.LinAlgError):
        return None
    t = np.array([mom_b.cx, mom_b.cy]) - lin @ np.array([mom_a.cx, mom_a.cy])
    out = np.eye(3)
    out[:2, :2] = lin
    out[:2, 2] = t
    return out


def apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    z = q[:, 2:3]
    z = np.where(np.abs(z) < 1e-12, 1e-12, z)
    return q[:, :2] / z


def map_box(H: np.ndarray, box: Box) -> Box:
    x1, y1, x2, y2 = box
    q = apply(H, np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]))
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def _conditioned(H: np.ndarray, cfg: dict[str, Any]) -> bool:
    lin = H[:2, :2] / H[2, 2]
    det = float(np.linalg.det(lin))
    lo, hi = (float(v) for v in cfg["scale_range"])
    if det <= 0 or not (lo * lo <= det <= hi * hi):
        return False
    sv = np.linalg.svd(lin, compute_uv=False)
    if sv[1] <= 0 or sv[0] / sv[1] > 3.0:
        return False
    return float(np.abs(H[2, :2] / H[2, 2]).max()) <= float(cfg["max_perspective"])


def _hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))


def _ecc(ma: np.ndarray, mb: np.ndarray, H: np.ndarray, motion: int, sigma: float) -> np.ndarray | None:
    """One ECC pass on the two masks blurred by sigma, from H (a -> b), run at a scale where the blur is about 2 px
    (the coarse passes are cheap). OpenCV warps b's frame into a's, so the inverse goes in and comes out."""
    f = min(float(ECC_MAX_SCALE), 2.0 / sigma)
    S = np.diag([f, f, 1.0])
    if f < 1.0:
        ma = cv2.resize(ma.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        mb = cv2.resize(mb.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(ma.astype(np.float32), (0, 0), sigma * f)
    b = cv2.GaussianBlur(mb.astype(np.float32), (0, 0), sigma * f)
    W = np.linalg.inv(S @ H @ np.linalg.inv(S))
    W = W / W[2, 2]
    W = (W[:2] if motion == cv2.MOTION_AFFINE else W).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, W = cv2.findTransformECC(b, a, W, motion, crit, None, 5)
    except cv2.error:
        return None
    W = W.astype(np.float64)
    if W.shape[0] == 2:
        W = np.vstack([W, [0.0, 0.0, 1.0]])
    try:
        out = np.linalg.inv(S) @ np.linalg.inv(W) @ S
    except np.linalg.LinAlgError:
        return None
    return out / out[2, 2]


def mask_fit(ma: np.ndarray, mb: np.ndarray, prior: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The masks' outlines aligned: affine ECC coarse to fine from the moment prior, then a homography from that
    affine. Returns (affine, homography); a step that fails or leaves the conditioning bounds is skipped."""
    H = prior
    for sigma in cfg["ecc_sigmas"]:
        nxt = _ecc(ma, mb, H, cv2.MOTION_AFFINE, float(sigma))
        if nxt is not None and _conditioned(nxt, cfg):
            H = nxt
    Hh = H
    for sigma in cfg["ecc_sigmas"][1:]:
        nxt = _ecc(ma, mb, Hh, cv2.MOTION_HOMOGRAPHY, float(sigma))
        if nxt is not None and _conditioned(nxt, cfg):
            Hh = nxt
    return H, Hh


def _iou_after(ma: np.ndarray, mb: np.ndarray, H: np.ndarray) -> float:
    """Mask IoU after the warp, counted only where the warped crop a covers crop b: a panel cut off by the edge of
    one crop is not a disagreement of the outlines."""
    hb, wb = mb.shape
    warped = cv2.warpPerspective(ma.astype(np.uint8) * 255, H, (wb, hb), flags=cv2.INTER_NEAREST) > 127
    cover = cv2.warpPerspective(np.full(ma.shape, 255, np.uint8), H, (wb, hb), flags=cv2.INTER_NEAREST) > 127
    cover = cv2.erode(cover.astype(np.uint8), _disk(3)) > 0
    a, b = warped & cover, mb & cover
    union = float((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def align(
    kf_a: str,
    img_a: np.ndarray,
    mask_a: np.ndarray,
    kf_b: str,
    img_b: np.ndarray,
    mask_b: np.ndarray,
    cfg: dict[str, Any],
    *,
    prior: np.ndarray | None = None,
    boxes_a: list[Box] | None = None,
    boxes_b: list[Box] | None = None,
    feats_a: tuple[np.ndarray, np.ndarray | None] | None = None,
    feats_b: tuple[np.ndarray, np.ndarray | None] | None = None,
) -> Aligned:
    """Fit crop a -> crop b. The masks' outlines first (affine and homography ECC from the moment prior), then SIFT
    matches on the rigid structure gated around where either outline fit puts them, fitted by RANSAC. The feature fit
    is used when it has enough inliers spread over the panel and keeps the outlines together. Otherwise the outline
    homography is used and the affine is kept beside it: where the two disagree at a point, that is how unsure the
    map is there (uncertainty_px). Never raises on bad pixels: a weak fit comes back with ok False and why."""
    out = Aligned(kf_a=kf_a, kf_b=kf_b, model="none", H=None)
    ma, mb = np.asarray(mask_a) > 0, np.asarray(mask_b) > 0
    if ma.sum() < 100 or mb.sum() < 100:
        out.why = "no panel mask"
        return out
    if prior is None:
        out.why = "no moment frame to start from"
        return out
    H_aff, H_hom = mask_fit(ma, mb, prior, cfg)
    ys, xs = np.nonzero(mb)
    diag_b = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
    box_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    pa, da = feats_a if feats_a is not None else features(img_a, ma, boxes_a or [], cfg)
    pb, db = feats_b if feats_b is not None else features(img_b, mb, boxes_b or [], cfg)
    src = dst = all_src = all_dst = np.zeros((0, 2))
    if da is not None and db is not None and len(pa) >= 2 and len(pb) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        ratio = float(cfg["ratio"])
        knn = matcher.knnMatch(da, db, k=2)
        pairs = [(m[0].queryIdx, m[0].trainIdx) for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
        back = {m[0].queryIdx: m[0].trainIdx for m in matcher.knnMatch(db, da, k=1) if m}
        pairs = [(i, j) for i, j in pairs if back.get(j) == i]
        if pairs:
            src = all_src = np.array([pa[i] for i, _ in pairs]).reshape(-1, 2)
            dst = all_dst = np.array([pb[j] for _, j in pairs]).reshape(-1, 2)
            gate = max(float(cfg["gate_px"]), float(cfg["gate_frac"]) * diag_b)
            near = np.minimum(np.linalg.norm(apply(H_aff, src) - dst, axis=1), np.linalg.norm(apply(H_hom, src) - dst, axis=1))
            src, dst = src[near <= gate], dst[near <= gate]
    out.n_matches = int(len(src))
    thr = float(cfg["ransac_px"])
    H, alt, model = H_hom, H_aff, "outline"
    if len(src) >= int(cfg["min_inliers"]):
        Hf, inl = cv2.findHomography(src, dst, cv2.RANSAC, thr, maxIters=5000, confidence=0.999)
        if Hf is not None and _conditioned(Hf, cfg):
            inl = inl.ravel().astype(bool)
            spread = _hull_area(dst[inl]) / max(1.0, box_area)
            if int(inl.sum()) >= int(cfg["min_inliers"]) and spread >= float(cfg["min_spread"]) and _iou_after(ma, mb, Hf) >= float(cfg["min_mask_iou"]):
                H, alt, model = Hf / Hf[2, 2], None, "features"
    contradicted = None
    if model == "outline" and len(all_src) >= int(cfg["min_inliers"]):
        # the edges, ungated, may agree on a rigid motion the outline fit does not: a mask cut short or swollen in one
        # walk can be fitted by squeezing the other. A strong feature motion that lands far from the outline fit
        # refuses it (synthetic half mask: the squeeze kept a high IoU).
        Hu, inl = cv2.findHomography(all_src, all_dst, cv2.RANSAC, thr, maxIters=5000, confidence=0.999)
        if Hu is not None and _conditioned(Hu, cfg):
            inl = inl.ravel().astype(bool)
            if int(inl.sum()) >= int(cfg["min_inliers"]) and _hull_area(all_dst[inl]) / max(1.0, box_area) >= float(cfg["min_spread"]):
                ysa, xsa = np.nonzero(ma)
                step = max(1, len(xsa) // 400)
                sample = np.stack([xsa[::step], ysa[::step]], axis=1).astype(np.float64)
                gap = float(np.median(np.linalg.norm(apply(Hu, sample) - apply(H, sample), axis=1)))
                if gap > max(float(cfg["gate_px"]), float(cfg["gate_frac"]) * diag_b):
                    contradicted = gap
    out.model, out.H, out.H_alt = model, H, alt
    if len(src):
        err = np.linalg.norm(apply(H, src) - dst, axis=1)
        inl = err <= thr
        out.inliers_a, out.inliers_b = src[inl], dst[inl]
        out.n_inliers = int(inl.sum())
        out.inlier_frac = out.n_inliers / len(src)
        if out.n_inliers:
            out.reproj_px = float(np.median(err[inl]))
        out.spread = _hull_area(out.inliers_b) / max(1.0, box_area)
    out.mask_iou = _iou_after(ma, mb, H)
    if model == "features":
        checks = [
            (out.mask_iou >= float(cfg["min_mask_iou"]), f"mask IoU after the warp {out.mask_iou:.2f}"),
            (out.reproj_px <= float(cfg["max_reproj_px"]), f"reprojection {out.reproj_px:.1f} px"),
        ]
    else:
        iou_aff = _iou_after(ma, mb, H_aff)
        out.mask_iou = min(out.mask_iou, iou_aff)
        checks = [
            (out.mask_iou >= float(cfg["min_mask_iou_outline"]), f"mask IoU after the warp {out.mask_iou:.2f}"),
            (contradicted is None, f"matched edges put the panel {contradicted or 0:.0f} px from the outline fit"),
        ]
    failed = [words for passed, words in checks if not passed]
    out.ok = not failed
    out.why = None if out.ok else "; ".join(failed)
    return out


def inverse(al: Aligned) -> Aligned:
    """The same fit read the other way, crop b onto crop a."""
    out = Aligned(kf_a=al.kf_b, kf_b=al.kf_a, model=al.model, H=None if al.H is None else _inv(al.H), H_alt=None if al.H_alt is None else _inv(al.H_alt))
    out.n_matches, out.n_inliers, out.inlier_frac, out.spread, out.mask_iou = al.n_matches, al.n_inliers, al.inlier_frac, al.spread, al.mask_iou
    out.reproj_px, out.ok, out.why = al.reproj_px, al.ok, al.why
    out.inliers_a, out.inliers_b = al.inliers_b, al.inliers_a
    return out


def _inv(H: np.ndarray) -> np.ndarray:
    out = np.linalg.inv(H)
    return out / out[2, 2]


def uncertainty_px(al: Aligned, point_a: tuple[float, float]) -> float:
    """How unsure the map is at a point of crop a, in crop b pixels: twice the inliers' reprojection error for a
    feature fit; for an outline fit, how far the affine and the homography outline fits land apart."""
    assert al.H is not None
    if al.H_alt is None:
        return 2.0 * (al.reproj_px if math.isfinite(al.reproj_px) else 2.0)
    p = np.array([point_a])
    return float(np.linalg.norm(apply(al.H, p)[0] - apply(al.H_alt, p)[0]))


def local_map(al: Aligned, point_a: tuple[float, float], diag_a: float, cfg: dict[str, Any]) -> tuple[np.ndarray, bool]:
    """The map to use at a point: the affine of the inliers within local_radius_frac of the panel diagonal when there
    are enough of them and they fit, else the global model. Curved panels bend away from one homography."""
    assert al.H is not None
    r = float(cfg["local_radius_frac"]) * diag_a
    near = np.linalg.norm(al.inliers_a - np.array(point_a), axis=1) <= r
    if int(near.sum()) >= int(cfg["local_min_inliers"]):
        A, inl = cv2.estimateAffine2D(al.inliers_a[near], al.inliers_b[near], method=cv2.LMEDS)
        if A is not None:
            H = np.vstack([A, [0.0, 0.0, 1.0]])
            # the local fit must agree with the global one at the point within a few reprojection errors, or it
            # was fitted to a cluster on one side and is not trusted
            g = apply(al.H, np.array([point_a]))[0]
            loc = apply(H, np.array([point_a]))[0]
            if float(np.linalg.norm(g - loc)) <= max(8.0, 0.05 * diag_a):
                return H, True
    return al.H, False


# --------------------------------------------------------------------------------------------------------------
# Presence
# --------------------------------------------------------------------------------------------------------------


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) * np.array([100.0 / 255.0, 1.0, 1.0], dtype=np.float32) - np.array([0.0, 128.0, 128.0], dtype=np.float32)


def residual(img: np.ndarray, sigma: float) -> np.ndarray:
    """Lab minus its Gaussian blur, as a Delta E magnitude per pixel: the smooth shading and most of a soft
    reflection go, a mark's edges stay. Both walks' lighting differ, so marks are compared on this, not on colour.
    A blur wider than 8 px is taken on a copy shrunk to make it 8 px and enlarged back: the background it removes is
    smooth, so this costs little accuracy and keeps a long scratch's blur cheap."""
    lab = _lab(img)
    if sigma > 8.0:
        k = sigma / 8.0
        h, w = lab.shape[:2]
        small = cv2.resize(lab, (max(1, int(round(w / k))), max(1, int(round(h / k)))), interpolation=cv2.INTER_AREA)
        bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 8.0), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        bg = cv2.GaussianBlur(lab, (0, 0), sigma)
    res = lab - bg
    return np.sqrt((res * res).sum(axis=2))


def saliency(res: np.ndarray, box: Box, ring_px: float) -> float | None:
    """How much the box stands out from a ring around it on the residual: (90th percentile in the box minus the
    ring's median) over the ring's spread (1.4826 MAD, floored at 1 Delta E). None when the ring leaves the image."""
    h, w = res.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    r = int(round(ring_px))
    if x2 - x1 < 2 or y2 - y1 < 2 or x1 - r < 0 or y1 - r < 0 or x2 + r > w or y2 + r > h:
        return None
    outer = res[y1 - r : y2 + r, x1 - r : x2 + r]
    sel = np.ones(outer.shape, dtype=bool)
    sel[r : r + (y2 - y1), r : r + (x2 - x1)] = False
    ring = outer[sel]
    inner = res[y1:y2, x1:x2].ravel()
    med = float(np.median(ring))
    spread = max(1.0, 1.4826 * float(np.median(np.abs(ring - med))))
    return float((np.percentile(inner, 90) - med) / spread)


def box_sigma(box: Box, cfg: dict[str, Any]) -> float:
    """The high-pass blur presence uses for a box: residual_sigma_px or residual_sigma_box of its long side, larger
    wins, at most residual_sigma_max."""
    long_side = max(box[2] - box[0], box[3] - box[1], 4.0)
    return min(float(cfg["residual_sigma_max"]), max(float(cfg["residual_sigma_px"]), float(cfg["residual_sigma_box"]) * long_side))


def own_contrast(img: np.ndarray, box: Box, cfg: dict[str, Any]) -> float | None:
    """How much the box stands out from its own surroundings in its own crop, no fit needed (11/2, K20): presence's
    contrast_a (saliency on the high-passed residual, the same blur and ring), with the ring clipped at the crop edge
    where the box sits near it. Equal to contrast_a wherever presence measures one. None when the box keeps under 2 px
    a side inside the crop or the clipped ring keeps under ring_px squared pixels."""
    ring = float(cfg["ring_px"])
    sigma = box_sigma(box, cfg)
    wa = _window(img.shape, box, ring + 1.0)
    if wa is not None:
        return saliency(_residual_at(img, wa, sigma), _shift(box, wa[0], wa[1]), ring)
    h, w = img.shape[:2]
    win = (max(0, int(math.floor(box[0] - ring - 1.0))), max(0, int(math.floor(box[1] - ring - 1.0))), min(w, int(math.ceil(box[2] + ring + 1.0))), min(h, int(math.ceil(box[3] + ring + 1.0))))
    if win[2] - win[0] < 3 or win[3] - win[1] < 3:
        return None
    res = _residual_at(img, win, sigma)
    rh, rw = res.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in _shift(box, win[0], win[1]))
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(rw, x2), min(rh, y2)
    r = int(round(ring))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    ox1, oy1, ox2, oy2 = max(0, x1 - r), max(0, y1 - r), min(rw, x2 + r), min(rh, y2 + r)
    sel = np.ones((oy2 - oy1, ox2 - ox1), dtype=bool)
    sel[y1 - oy1 : y2 - oy1, x1 - ox1 : x2 - ox1] = False
    ring_px = res[oy1:oy2, ox1:ox2][sel]
    if ring_px.size < r * r:
        return None
    inner = res[y1:y2, x1:x2].ravel()
    med = float(np.median(ring_px))
    spread = max(1.0, 1.4826 * float(np.median(np.abs(ring_px - med))))
    return float((np.percentile(inner, 90) - med) / spread)


def _box_corr(res_a: np.ndarray, res_b: np.ndarray, box_a: Box, box_b: Box, pad: float) -> float | None:
    """Correlation of the two residual images over the mark's own box (padded a little), box_a in res_a against the
    same-size box_b in res_b: does the mark itself line up, not just its surroundings."""
    h, w = res_b.shape[:2]
    ax1, ay1 = int(round(box_a[0] - pad)), int(round(box_a[1] - pad))
    bx1, by1 = int(round(box_b[0] - pad)), int(round(box_b[1] - pad))
    bw = int(round(box_a[2] - box_a[0] + 2 * pad))
    bh = int(round(box_a[3] - box_a[1] + 2 * pad))
    if min(ax1, ay1, bx1, by1) < 0 or max(ax1, bx1) + bw > w or max(ay1, by1) + bh > h or bw < 3 or bh < 3:
        return None
    a = res_a[ay1 : ay1 + bh, ax1 : ax1 + bw].ravel().astype(np.float64)
    b = res_b[by1 : by1 + bh, bx1 : bx1 + bw].ravel().astype(np.float64)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def _grow(box: Box, cfg: dict[str, Any]) -> Box:
    x1, y1, x2, y2 = box
    g = float(cfg["patch_grow"])
    half_w = max((x2 - x1) * (0.5 + g), float(cfg["patch_min_px"]) / 2.0)
    half_h = max((y2 - y1) * (0.5 + g), float(cfg["patch_min_px"]) / 2.0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _window(shape: tuple[int, ...], box: Box, pad: float) -> tuple[int, int, int, int] | None:
    """The integer window of box grown by pad, or None when it leaves an image of this shape."""
    h, w = shape[:2]
    x1, y1 = int(math.floor(box[0] - pad)), int(math.floor(box[1] - pad))
    x2, y2 = int(math.ceil(box[2] + pad)), int(math.ceil(box[3] + pad))
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None
    return x1, y1, x2, y2


def _residual_at(img: np.ndarray, win: tuple[int, int, int, int], sigma: float) -> np.ndarray:
    """residual() of the window only, computed on the window padded by three blurs (clipped at the image edge)."""
    h, w = img.shape[:2]
    pad = int(math.ceil(3.0 * sigma))
    X1, Y1, X2, Y2 = max(0, win[0] - pad), max(0, win[1] - pad), min(w, win[2] + pad), min(h, win[3] + pad)
    res = residual(img[Y1:Y2, X1:X2], sigma)
    return res[win[1] - Y1 : win[3] - Y1, win[0] - X1 : win[2] - X1]


def _shift(box: Box, ox: float, oy: float) -> Box:
    return box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy


def presence(al: Aligned, img_a: np.ndarray, img_b: np.ndarray, mask_b: np.ndarray, box_a: Box, cfg: dict[str, Any], *, diag_a: float) -> Presence:
    """Is the thing in box_a (crop a) at its mapped spot in crop b? present, absent or cannot tell, with the numbers.

    Crop a is warped into b's frame. The patch around the mapped box (the box plus patch_grow of its size a side) is
    slid over b, on the residual images, across a window from where H puts the box to where the other outline fit
    puts it plus search_margin_px a side (ncc is the best match). Then the box's saliency in its own crop is compared
    with the saliency at the found spot. present: the surroundings match (ncc_present) and the spot keeps keep_frac
    of the saliency, or they match less well (ncc_present_kept) and the spot keeps keep_frac_kept and the box itself
    lines up (ncc_mark_min). absent: ncc under ncc_absent, no spot (found, H's, the other fit's) keeps gone_frac, and
    the mark stands out by saliency_absent_min in its own crop. A mark that barely stands out in its own crop, a spot
    off b's mask or crop, a clipped patch or an unsure map: cannot tell."""
    if not al.ok or al.H is None:
        return Presence(CANNOT_TELL, f"alignment refused ({al.why})")
    x1, y1, x2, y2 = box_a
    centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    u = uncertainty_px(al, centre)
    if u > float(cfg["max_uncertainty_px"]):
        return Presence(CANNOT_TELL, f"the map is unsure there by {u:.0f} px", uncertainty_px=round(u, 2))
    H, local = local_map(al, centre, diag_a, cfg) if al.model == "features" else (al.H, False)
    hb, wb = img_b.shape[:2]
    mapped = map_box(H, box_a)
    out = Presence(CANNOT_TELL, None, mapped_box=_round_box(mapped), local=local, uncertainty_px=round(u, 2))
    mcx, mcy = (mapped[0] + mapped[2]) / 2.0, (mapped[1] + mapped[3]) / 2.0
    if not (0 <= mcx < wb and 0 <= mcy < hb):
        out.why = "the mapped spot is outside the other crop"
        return out
    if not (np.asarray(mask_b) > 0)[int(mcy), int(mcx)]:
        out.why = "the mapped spot is off the other walk's panel mask"
        return out
    ring = float(cfg["ring_px"])
    sigma = box_sigma(box_a, cfg)
    wa = _window(img_a.shape, box_a, ring + 1.0)
    if wa is None:
        out.why = "the box is at the edge of its own crop"
        return out
    sa = saliency(_residual_at(img_a, wa, sigma), _shift(box_a, wa[0], wa[1]), ring)
    if sa is None:
        out.why = "the box is at the edge of its own crop"
        return out
    out.contrast_a = round(sa, 2)
    px1, py1, px2, py2 = (int(round(v)) for v in _grow(mapped, cfg))
    # the window runs from where H puts the box to where the other outline fit puts it, plus a margin: on the synth
    # truth the mark sat near one of the two, so the search covers both ends and the line between, not a square of
    # radius u
    alt = map_box(al.H_alt, box_a) if (al.H_alt is not None and not local) else mapped
    adx, ady = (alt[0] + alt[2] - mapped[0] - mapped[2]) / 2.0, (alt[1] + alt[3] - mapped[1] - mapped[3]) / 2.0
    m = float(cfg["search_margin_px"])
    lx, rx = int(round(m - min(0.0, adx))), int(round(m + max(0.0, adx)))
    ty, by = int(round(m - min(0.0, ady))), int(round(m + max(0.0, ady)))
    if px1 - lx < 0 or py1 - ty < 0 or px2 + rx > wb or py2 + by > hb:
        out.why = "the mapped patch reaches the edge of the other crop"
        return out
    gray_b = cv2.cvtColor(img_b[py1 - ty : py2 + by, px1 - lx : px2 + rx], cv2.COLOR_BGR2GRAY)
    if float(((gray_b >= 253) | (gray_b <= 2)).mean()) > 0.3:
        out.why = "the other crop is blown out or black there"
        return out
    # one window of b holds the search, every spot tested and their rings; crop a is warped into that window only
    wx1 = max(0, min(px1 - lx, int(math.floor(alt[0] - ring - 1))))
    wy1 = max(0, min(py1 - ty, int(math.floor(alt[1] - ring - 1))))
    wx2 = min(wb, max(px2 + rx, int(math.ceil(alt[2] + ring + 1))))
    wy2 = min(hb, max(py2 + by, int(math.ceil(alt[3] + ring + 1))))
    pad = int(math.ceil(3.0 * sigma))
    X1, Y1, X2, Y2 = max(0, wx1 - pad), max(0, wy1 - pad), min(wb, wx2 + pad), min(hb, wy2 + pad)
    T = np.array([[1.0, 0.0, -X1], [0.0, 1.0, -Y1], [0.0, 0.0, 1.0]])
    size = (X2 - X1, Y2 - Y1)
    warped = cv2.warpPerspective(img_a, T @ H, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    covered = cv2.warpPerspective(np.full(img_a.shape[:2], 255, np.uint8), T @ H, size, flags=cv2.INTER_NEAREST)
    if covered[py1 - Y1 : py2 - Y1, px1 - X1 : px2 - X1].min() == 0:
        out.why = "the mapped patch leaves the warped crop"
        return out
    res_w = residual(warped, sigma)
    res_b = residual(img_b[Y1:Y2, X1:X2], sigma)
    tpl = res_w[py1 - Y1 : py2 - Y1, px1 - X1 : px2 - X1]
    win = res_b[py1 - ty - Y1 : py2 + by - Y1, px1 - lx - X1 : px2 + rx - X1]
    if float(tpl.std()) < 1e-3 or float(win.std()) < 1e-3:
        out.why = "the patch is flat"
        return out
    score = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, best, _, loc = cv2.minMaxLoc(score)
    dx, dy = loc[0] - lx, loc[1] - ty
    out.ncc = round(float(best), 3)
    out.shift_px = round(math.hypot(dx, dy), 2)
    found = (mapped[0] + dx, mapped[1] + dy, mapped[2] + dx, mapped[3] + dy)
    out.found_box = _round_box(found)
    spots = [saliency(res_b, _shift(b, X1, Y1), ring) if _window(img_b.shape, b, ring) is not None else None for b in (found, mapped, alt)]
    if spots[0] is None or spots[1] is None:
        out.why = "the mapped box is at the edge of the other crop"
        return out
    sb_found = spots[0]
    out.contrast_b = round(sb_found, 2)
    out.ncc_mark = _box_corr(res_w, res_b, _shift(mapped, X1, Y1), _shift(found, X1, Y1), float(cfg["mark_pad_corr_px"]))
    if sa < float(cfg["saliency_min"]):
        out.why = f"the mark stands out by only {sa:.1f} in its own crop"
        return out
    keep = sb_found / sa
    strong = best >= float(cfg["ncc_present"]) and keep >= float(cfg["keep_frac"])
    kept = best >= float(cfg["ncc_present_kept"]) and keep >= float(cfg["keep_frac_kept"]) and (out.ncc_mark or -1.0) >= float(cfg["ncc_mark_min"])
    if sb_found >= float(cfg["saliency_floor"]) and (strong or kept):
        out.result = PRESENT
        return out
    gone = max(v for v in spots if v is not None) / sa
    if best < float(cfg["ncc_absent"]) and gone < float(cfg["gone_frac"]) and sa >= float(cfg["saliency_absent_min"]):
        out.result = ABSENT
        return out
    out.why = "between the present and absent bands"
    return out


def coincide_px(box: Box, cfg: dict[str, Any]) -> float:
    """How close the found spot and the other walk's mark must be to count as one spot: coincide_px or half the
    box's long side, whichever is larger. The found spot is already corrected by the patch search, so the map's own
    uncertainty is not added again."""
    long_side = max(box[2] - box[0], box[3] - box[1])
    return max(float(cfg["coincide_px"]), float(cfg["coincide_box_frac"]) * long_side)


def _round_box(b: Box) -> Box:
    return (round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2))
