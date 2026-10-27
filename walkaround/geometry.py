"""Panel-normalized geometry (docs/ARCHITECTURE.md 5.5 and 5.8, docs/CONTRACTS.md M7). Pure numpy and cv2.

Whitening. A panel mask in crop pixels has a centroid c and a covariance Sigma built from its central moments
(mu20, mu11, mu02, each divided by the pixel count). A point p maps to w = D^(-1/2) R^T (p - c): the principal
frame of the mask, one unit per standard deviation of the mask along each axis (about 35 cm along a 1.2 x 1.0 m
door, 29 cm vertically). Axis 1 is the principal axis nearest the image horizontal, signed so it points toward the
car's front (front_sign); axis 2 is perpendicular to it and points image-down. |w| is the Mahalanobis distance and
is unchanged by any affine map of the mask. The pair (u, v) is unchanged by similarity maps and by scaling along the
principal axes; a change of viewpoint (perspective) leaves a residual the frame does not remove.

Measured on the synthetic door of tests/test_geometry.py (1.2 x 1.0 m, rounded corners, 2 m away): the keystone of a
yawed view pulls the mask centroid toward the near edge, so the residual is close to a translation of the frame,
about 0.012 units per degree of yaw mismatch at the centre (0.12 at 10 degrees, 0.24 at 20, just inside tau_pre),
smaller at the side edges and in between at the corners; the whitened IoU stays above 0.9, so G4 does not catch it.
Two things in the design absorb it: pairing keyframes by aspect ratio (a yaw proxy, match.pair_keyframes) keeps the
mismatch small, and the door-handle residual subtraction of 5.8 item 2 removes the translation (0.07 at the centre
and 0.11 to 0.15 at the corners for 0 against 20 degrees, within the section 13 bounds). The band still widens with
r (match.tau_r) for what remains.

Nothing here reads a file or calls an API. A keyframe whose front_sign is null cannot be whitened (G6).
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from .contracts import Crop, Moments

Box4 = Sequence[float]
Point2 = Sequence[float]

IOU_EXTENT_UNITS = 4.0  # the IoU grid covers [-4, 4] whitened units on each axis; a rectangle reaches sqrt(3)


# Moments ---------------------------------------------------------------------------------------------------------


def _moments_from_cov(cx: float, cy: float, mu20: float, mu02: float, mu11: float) -> Moments:
    """Build the Moments record (theta and the symmetric Sigma^(-1/2)) from a centroid and a covariance."""
    det = mu20 * mu02 - mu11 * mu11
    if mu20 <= 0.0 or mu02 <= 0.0 or det <= 0.0:
        raise ValueError("degenerate mask: the covariance is not positive definite")
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    c, s = math.cos(theta), math.sin(theta)
    l1 = mu20 * c * c + 2.0 * mu11 * s * c + mu02 * s * s  # variance along the principal axis
    l2 = mu20 * s * s - 2.0 * mu11 * s * c + mu02 * c * c  # variance across it
    a, b = 1.0 / math.sqrt(l1), 1.0 / math.sqrt(l2)
    sigma_inv_half = [
        [c * c * a + s * s * b, c * s * (a - b)],
        [c * s * (a - b), s * s * a + c * c * b],
    ]
    return Moments(cx=float(cx), cy=float(cy), mu20=float(mu20), mu02=float(mu02), mu11=float(mu11), theta=float(theta), sigma_inv_half=sigma_inv_half)


def moments_from_points(points: np.ndarray) -> Moments:
    """The same record from an (N, 2) point cloud of x, y positions (the pixel centres of a mask, or a warped copy)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError("fewer than 3 points")
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    n = float(pts.shape[0])
    return _moments_from_cov(cx, cy, float(np.dot(dx, dx) / n), float(np.dot(dy, dy) / n), float(np.dot(dx, dy) / n))


def moments_from_mask(mask: np.ndarray) -> Moments:
    """Centroid, normalised central moments, theta = 0.5 atan2(2 mu11, mu20 - mu02) and Sigma^(-1/2) of a mask."""
    ys, xs = np.nonzero(np.asarray(mask))
    if xs.size < 3:
        raise ValueError("mask has fewer than 3 pixels")
    return moments_from_points(np.stack([xs, ys], axis=1))


def _frame(moments: Moments) -> tuple[float, float, float, float]:
    """(cos, sin, var_u, var_v): axis 1 is the principal axis nearest the image horizontal."""
    t = moments.theta
    if t > math.pi / 4.0:
        t -= math.pi / 2.0
    elif t <= -math.pi / 4.0:
        t += math.pi / 2.0
    c, s = math.cos(t), math.sin(t)
    var_u = moments.mu20 * c * c + 2.0 * moments.mu11 * s * c + moments.mu02 * s * s
    var_v = moments.mu20 * s * s - 2.0 * moments.mu11 * s * c + moments.mu02 * c * c
    return c, s, var_u, var_v


def _check_front_sign(front_sign: int) -> int:
    if isinstance(front_sign, bool) or front_sign not in (1, -1):
        raise ValueError(f"front_sign must be 1 or -1, got {front_sign!r}; the panel is not comparable (G6)")
    return int(front_sign)


def whitening_matrix(moments: Moments, front_sign: int) -> np.ndarray:
    """The 2x2 matrix W with w = W (p - c): rows are axis 1 (toward the front) and axis 2 (image-down)."""
    sign = _check_front_sign(front_sign)
    c, s, var_u, var_v = _frame(moments)
    if var_u <= 0.0 or var_v <= 0.0:
        raise ValueError("degenerate moments")
    return np.array([[sign * c / math.sqrt(var_u), sign * s / math.sqrt(var_u)], [-s / math.sqrt(var_v), c / math.sqrt(var_v)]])


def whiten(point: tuple[float, float], moments: Moments, front_sign: int) -> tuple[float, float]:
    """Whitened (u, v) of a crop-pixel point. ValueError when front_sign is not 1 or -1."""
    w = whitening_matrix(moments, front_sign) @ np.array([point[0] - moments.cx, point[1] - moments.cy], dtype=np.float64)
    return float(w[0]), float(w[1])


def whiten_many(points: np.ndarray, moments: Moments, front_sign: int) -> np.ndarray:
    """(N, 2) crop-pixel points to (N, 2) whitened coordinates."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return (pts - np.array([moments.cx, moments.cy])) @ whitening_matrix(moments, front_sign).T


def unwhiten(w: tuple[float, float], moments: Moments, front_sign: int) -> tuple[float, float]:
    """The inverse map: whitened (u, v) back to crop pixels."""
    p = np.linalg.solve(whitening_matrix(moments, front_sign), np.array(w, dtype=np.float64))
    return float(p[0] + moments.cx), float(p[1] + moments.cy)


def radius(w: tuple[float, float]) -> float:
    return math.hypot(float(w[0]), float(w[1]))


def carry_moments(moments: Moments, box_from: Box4, box_to: Box4) -> Moments:
    """Keyframe moments carried to another frame by the affine that maps box_from onto box_to (translation and
    per-axis scale), so a clip frame needs no mask decode."""
    fw, fh = box_from[2] - box_from[0], box_from[3] - box_from[1]
    tw, th_ = box_to[2] - box_to[0], box_to[3] - box_to[1]
    if fw <= 0 or fh <= 0 or tw <= 0 or th_ <= 0:
        raise ValueError("boxes must have positive width and height")
    sx, sy = tw / fw, th_ / fh
    cx = box_to[0] + (moments.cx - box_from[0]) * sx
    cy = box_to[1] + (moments.cy - box_from[1]) * sy
    return _moments_from_cov(cx, cy, moments.mu20 * sx * sx, moments.mu02 * sy * sy, moments.mu11 * sx * sy)


# Crop, source and proxy maps ------------------------------------------------------------------------------------------


def crop_geometry(box_src: Box4, media_size: tuple[int, int], *, pad_frac: float, long_edge: int) -> Crop:
    """The padded source box behind a panel crop: pad_frac of the box per side, clamped to the frame, integer
    source edges, long edge at most long_edge (never upscaled). path is "" for the caller to fill."""
    width, height = media_size
    x1, y1, x2, y2 = (float(v) for v in box_src)
    px, py = pad_frac * (x2 - x1), pad_frac * (y2 - y1)
    sx1 = max(0, math.floor(x1 - px))
    sy1 = max(0, math.floor(y1 - py))
    sx2 = min(int(width), math.ceil(x2 + px))
    sy2 = min(int(height), math.ceil(y2 + py))
    bw, bh = sx2 - sx1, sy2 - sy1
    if bw <= 0 or bh <= 0:
        raise ValueError(f"box {list(box_src)} is empty inside a {width}x{height} frame")
    scale = min(1.0, float(long_edge) / float(max(bw, bh)))
    return Crop(path="", w=max(1, round(bw * scale)), h=max(1, round(bh * scale)), scale=scale, dx=float(sx1), dy=float(sy1))


def crop_to_src(p: Point2, crop: Crop) -> list[float]:
    return [crop.dx + float(p[0]) / crop.scale, crop.dy + float(p[1]) / crop.scale]


def src_to_crop(p: Point2, crop: Crop) -> list[float]:
    return [(float(p[0]) - crop.dx) * crop.scale, (float(p[1]) - crop.dy) * crop.scale]


def box_crop_to_src(b: Box4, crop: Crop) -> list[float]:
    return [*crop_to_src((b[0], b[1]), crop), *crop_to_src((b[2], b[3]), crop)]


def box_src_to_crop(b: Box4, crop: Crop) -> list[float]:
    return [*src_to_crop((b[0], b[1]), crop), *src_to_crop((b[2], b[3]), crop)]


def proxy_to_src(box: Box4, proxy_size: tuple[int, int], src_size: tuple[int, int]) -> list[float]:
    sx, sy = src_size[0] / proxy_size[0], src_size[1] / proxy_size[1]
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def src_to_proxy(box: Box4, proxy_size: tuple[int, int], src_size: tuple[int, int]) -> list[float]:
    sx, sy = proxy_size[0] / src_size[0], proxy_size[1] / src_size[1]
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def sent_to_crop(p_sent: tuple[int, int], sent_size: tuple[int, int], crop_size: tuple[int, int]) -> list[float]:
    """Undo spark_client.prepare_image's resize: sent pixels to crop pixels."""
    return [float(p_sent[0]) * crop_size[0] / sent_size[0], float(p_sent[1]) * crop_size[1] / sent_size[1]]


# Box rules ---------------------------------------------------------------------------------------------------------------


def _area(b: Box4) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _intersection(a: Box4, b: Box4) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def box_iou(a: Box4, b: Box4) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: Box4, outer: Box4) -> float:
    """Share of inner's area that lies inside outer."""
    area = _area(inner)
    return _intersection(inner, outer) / area if area > 0 else 0.0


# Whitened mask IoU ------------------------------------------------------------------------------------------------------


def whitened_mask(mask: np.ndarray, moments: Moments, front_sign: int, *, grid: int = 256) -> np.ndarray:
    """The mask resampled onto a grid x grid raster of whitened coordinates covering +/- IOU_EXTENT_UNITS."""
    scale = grid / (2.0 * IOU_EXTENT_UNITS)
    w = whitening_matrix(moments, front_sign) * scale
    offset = np.array([grid / 2.0, grid / 2.0]) - w @ np.array([moments.cx, moments.cy])
    affine = np.hstack([w, offset.reshape(2, 1)]).astype(np.float64)
    src = (np.asarray(mask) > 0).astype(np.uint8) * 255
    out = cv2.warpAffine(src, affine, (grid, grid), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out >= 128


def whitened_mask_iou(mask_a: np.ndarray, mom_a: Moments, sign_a: int, mask_b: np.ndarray, mom_b: Moments, sign_b: int, *, grid: int = 256) -> float:
    """IoU of two panel masks after each is whitened onto one grid (G4, G8); 0.0 when either mask is empty."""
    if not np.asarray(mask_a).any() or not np.asarray(mask_b).any():
        return 0.0
    a = whitened_mask(mask_a, mom_a, sign_a, grid=grid)
    b = whitened_mask(mask_b, mom_b, sign_b, grid=grid)
    union = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / union if union else 0.0


# Secondary frames: wheels, bumpers, hoods ------------------------------------------------------------------------


def _as_u8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


def _clamp_index(p: Point2, mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape[:2]
    return min(max(int(round(float(p[0]))), 0), w - 1), min(max(int(round(float(p[1]))), 0), h - 1)


def inscribed_radius(mask: np.ndarray) -> float:
    """Radius of the largest inscribed circle (the maximum of the distance transform), in pixels."""
    m = _as_u8(mask)
    if not m.any():
        raise ValueError("empty mask")
    return float(cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).max())


def wheel_coords(point_crop: Point2, mask: np.ndarray) -> tuple[float, float]:
    """(radius, angle) on a wheel mask: radius = 1 - dt(p) / inscribed radius, so 0 at the centre of the largest
    inscribed circle and 1 at the rim or outside; angle in radians about the mask centroid, kept for the shift."""
    m = _as_u8(mask)
    if not m.any():
        raise ValueError("empty mask")
    dt = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    big = float(dt.max())
    x, y = _clamp_index(point_crop, m)
    rad = min(max(1.0 - float(dt[y, x]) / big, 0.0), 1.0)
    ys, xs = np.nonzero(m)
    angle = math.atan2(float(point_crop[1]) - float(ys.mean()), float(point_crop[0]) - float(xs.mean()))
    return rad, angle


def bumper_arc(point_crop: Point2, anchors: tuple[Point2, Point2] | None, mask: np.ndarray) -> tuple[float, float] | None:
    """(arc, height): arc = position of the point between the two anchors (0 at the first, 1 at the second), height
    = position inside the mask's vertical extent at the point's column (0 top, 1 bottom). None without anchors."""
    if anchors is None or anchors[0] is None or anchors[1] is None:
        return None
    a = np.array(anchors[0][:2], dtype=np.float64)
    b = np.array(anchors[1][:2], dtype=np.float64)
    p = np.array(point_crop[:2], dtype=np.float64)
    ab = b - a
    norm2 = float(ab @ ab)
    if norm2 <= 0.0:
        return None
    arc = float((p - a) @ ab / norm2)
    m = _as_u8(mask)
    if not m.any():
        return None
    cols = np.nonzero(m.any(axis=0))[0]
    x, _ = _clamp_index(point_crop, m)
    col = int(cols[np.argmin(np.abs(cols - x))])
    ys = np.nonzero(m[:, col])[0]
    top, bottom = float(ys.min()), float(ys.max())
    height = 0.5 if bottom <= top else (float(point_crop[1]) - top) / (bottom - top)
    return arc, height


def hood_coords(point_crop: Point2, moments: Moments, front_sign: int, mask: np.ndarray) -> tuple[float, int]:
    """(lateral whitened coordinate, depth third from the windshield edge: 0 nearest the windshield, 2 the front)."""
    u, v = whiten((float(point_crop[0]), float(point_crop[1])), moments, front_sign)
    ys, xs = np.nonzero(_as_u8(mask))
    if xs.size == 0:
        raise ValueError("empty mask")
    vs = whiten_many(np.stack([xs, ys], axis=1), moments, front_sign)[:, 1]
    lo, hi = float(vs.min()), float(vs.max())
    third = 1 if hi <= lo else int(math.floor((v - lo) / (hi - lo) * 3.0))
    return u, min(max(third, 0), 2)


def boundary_distance_units(point_crop: Point2, mask: np.ndarray, moments: Moments, front_sign: int) -> float:
    """Whitened distance from the point to the nearest mask boundary pixel (the curvature cap of 5.9 item 6)."""
    m = _as_u8(mask)
    if not m.any():
        raise ValueError("empty mask")
    eroded = cv2.erode(m, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.nonzero(m & (1 - eroded))
    boundary = whiten_many(np.stack([xs, ys], axis=1), moments, front_sign)
    w = np.array(whiten((float(point_crop[0]), float(point_crop[1])), moments, front_sign))
    return float(np.min(np.linalg.norm(boundary - w, axis=1)))
