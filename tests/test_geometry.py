"""Offline tests for walkaround.geometry (M7): moments, whitening, the crop maps, the whitened mask IoU and the
secondary frames, on synthetic masks. No file under runs/ and no model call.

The two-viewpoint tests build a 1.2 x 1.0 m door with rounded corners, plant marks on it and look at it twice: once
through an affine warp (the invariance the moment frame promises) and once through a pinhole camera at 2 m with 0,
10 and 20 degrees of yaw (the perspective residual the frame does not remove). The numbers asserted there were
measured on this synthetic door and are repeated in the docstrings so a change in the frame shows up as a number.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from walkaround import geometry
from walkaround.contracts import Crop, load_thresholds

TH = load_thresholds()
TAU_PRE = float(TH["matching"]["tau_pre0_floor"])
TAU_EYE = float(TH["matching"]["tau_eye_factor"]) * TAU_PRE
BAND = float(TH["matching"]["band_radius_factor"])


def tau_r(tau0: float, r: float) -> float:
    return tau0 * (1.0 + BAND * r * r)


# Synthetic shapes -----------------------------------------------------------------------------------------------


def rect_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = True
    return m


def rounded_rect(w: float, h: float, r: float, n: int = 24) -> list[tuple[float, float]]:
    """A rounded rectangle centred at the origin as a polygon, corners traversed TL, TR, BR, BL."""
    corners = [(-w / 2 + r, -h / 2 + r, math.pi), (w / 2 - r, -h / 2 + r, -math.pi / 2), (w / 2 - r, h / 2 - r, 0.0), (-w / 2 + r, h / 2 - r, math.pi / 2)]
    pts = []
    for cx, cy, start in corners:
        for k in range(n + 1):
            a = start + (math.pi / 2) * k / n
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def fill(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
    return m > 0


# The synthetic door: 1.2 x 1.0 m at Z = 2 m, f = 1200 px, so the frontal door is 720 x 600 px.
F, W, H, Z = 1200.0, 1600, 1200, 2.0
DOOR = rounded_rect(1.2, 1.0, 0.1)
MARKS = {
    "centre": (0.0, 0.0),
    "tl": (-0.5, -0.4),
    "tr": (0.5, -0.4),
    "bl": (-0.5, 0.4),
    "br": (0.5, 0.4),
    "left": (-0.55, 0.0),
    "right": (0.55, 0.0),
    "top": (0.0, -0.45),
    "bottom": (0.0, 0.45),
}
HANDLE = (0.35, -0.05)  # where a door handle sits, for the handle self-test


def project(points: list[tuple[float, float]], yaw_deg: float) -> np.ndarray:
    """Pinhole projection of door-plane points after a yaw about the vertical axis."""
    a = math.radians(yaw_deg)
    out = []
    for x, y in points:
        depth = Z + x * math.sin(a)
        out.append((W / 2 + F * (x * math.cos(a)) / depth, H / 2 + F * y / depth))
    return np.array(out, dtype=np.float64)


def door_view(yaw_deg: float) -> tuple[np.ndarray, geometry.Moments, dict[str, tuple[float, float]], tuple[float, float]]:
    """(mask, moments, whitened planted marks, whitened handle) of the door at one yaw."""
    mask = fill(project(DOOR, yaw_deg), (H, W))
    mom = geometry.moments_from_mask(mask)
    whitened = {k: geometry.whiten(tuple(project([v], yaw_deg)[0]), mom, 1) for k, v in MARKS.items()}
    handle = geometry.whiten(tuple(project([HANDLE], yaw_deg)[0]), mom, 1)
    return mask, mom, whitened, handle


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# Moments and the axis convention -----------------------------------------------------------------------------------


def test_rectangle_moments_match_the_closed_form() -> None:
    m = rect_mask((400, 600), (100, 100, 500, 300))  # 400 wide, 200 tall, centred at (299.5, 199.5)
    mom = geometry.moments_from_mask(m)
    assert mom.cx == pytest.approx(299.5) and mom.cy == pytest.approx(199.5)
    assert mom.mu20 == pytest.approx((400 * 400 - 1) / 12.0) and mom.mu02 == pytest.approx((200 * 200 - 1) / 12.0)
    assert mom.mu11 == pytest.approx(0.0, abs=1e-9) and mom.theta == pytest.approx(0.0, abs=1e-9)
    s = np.array(mom.sigma_inv_half)
    assert s[0][1] == pytest.approx(0.0, abs=1e-12) and s[0][0] == pytest.approx(1.0 / math.sqrt(mom.mu20))


def test_axis_one_points_to_the_front_and_axis_two_image_down() -> None:
    m = rect_mask((400, 600), (100, 100, 500, 300))
    mom = geometry.moments_from_mask(m)
    half_u = 200.0 / math.sqrt(mom.mu20)  # the right edge in units, about sqrt(3)
    u, v = geometry.whiten((499.5, 199.5), mom, 1)
    assert u == pytest.approx(half_u, rel=1e-6) and v == pytest.approx(0.0, abs=1e-9)
    assert half_u == pytest.approx(math.sqrt(3.0), rel=0.01)
    u, v = geometry.whiten((499.5, 199.5), mom, -1)
    assert u == pytest.approx(-half_u, rel=1e-6)
    u, v = geometry.whiten((299.5, 299.5), mom, 1)
    assert u == pytest.approx(0.0, abs=1e-9) and v == pytest.approx(100.0 / math.sqrt(mom.mu02), rel=1e-6)
    u, v = geometry.whiten((299.5, 299.5), mom, -1)  # the sign never flips the vertical axis
    assert v > 0


def test_tall_mask_keeps_axis_one_nearest_horizontal() -> None:
    tall = rect_mask((600, 400), (100, 50, 300, 550))  # 200 wide, 500 tall: the principal axis is vertical
    mom = geometry.moments_from_mask(tall)
    assert abs(mom.theta) == pytest.approx(math.pi / 2, abs=1e-6)
    u, v = geometry.whiten((299.0, 299.5), mom, 1)  # the right edge midpoint, 99.5 px from the centre: along axis 1
    assert v == pytest.approx(0.0, abs=1e-9) and u == pytest.approx(99.5 / math.sqrt(mom.mu20), rel=1e-6)


def test_whiten_refuses_a_bad_front_sign() -> None:
    mom = geometry.moments_from_mask(rect_mask((100, 100), (10, 10, 90, 60)))
    for bad in (0, None, 2, True):
        with pytest.raises(ValueError):
            geometry.whiten((50.0, 30.0), mom, bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        geometry.moments_from_mask(np.zeros((10, 10), dtype=bool))
    with pytest.raises(ValueError):
        geometry.moments_from_mask(rect_mask((10, 10), (2, 2, 3, 6)))  # a 1 px column: degenerate covariance


def test_whiten_many_unwhiten_and_radius_agree_with_whiten() -> None:
    mom = geometry.moments_from_mask(rect_mask((300, 500), (20, 40, 420, 260)))
    pts = np.array([[50.0, 60.0], [220.0, 150.0], [400.0, 250.0]])
    many = geometry.whiten_many(pts, mom, -1)
    for p, w in zip(pts, many):
        single = geometry.whiten((float(p[0]), float(p[1])), mom, -1)
        assert w[0] == pytest.approx(single[0]) and w[1] == pytest.approx(single[1])
        back = geometry.unwhiten(single, mom, -1)
        assert back[0] == pytest.approx(p[0], abs=1e-9) and back[1] == pytest.approx(p[1], abs=1e-9)
        assert geometry.radius(single) == pytest.approx(math.hypot(*single))


# Affine invariance -------------------------------------------------------------------------------------------------


def test_similarity_and_axis_scaling_leave_whitened_coordinates_unchanged_within_1e6() -> None:
    """Point-cloud moments (no raster error): rotation under 45 degrees, per-axis scale and translation change
    neither u nor v by more than whiten_affine_invariance_tol."""
    tol = float(TH["geometry"]["whiten_affine_invariance_tol"])
    rng = np.random.default_rng(7)
    ys, xs = np.nonzero(fill(project(DOOR, 0.0), (H, W)))
    cloud = np.stack([xs, ys], axis=1).astype(np.float64)
    mom = geometry.moments_from_points(cloud)
    marks = project(list(MARKS.values()), 0.0)
    w0 = geometry.whiten_many(marks, mom, 1)
    for _ in range(5):
        ang = rng.uniform(-0.6, 0.6)
        sx, sy = rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0)
        rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        a = rot @ np.diag([sx, sy])  # scale along the panel axes, then rotate
        t = rng.uniform(-300, 300, size=2)
        mom2 = geometry.moments_from_points(cloud @ a.T + t)
        w1 = geometry.whiten_many(marks @ a.T + t, mom2, 1)
        assert np.abs(w1 - w0).max() < tol


def test_any_affine_map_leaves_the_whitened_radius_unchanged() -> None:
    """Under a general affine map (shear included) the frame may rotate, but |w| is a true invariant."""
    tol = float(TH["geometry"]["whiten_affine_invariance_tol"])
    rng = np.random.default_rng(11)
    ys, xs = np.nonzero(fill(project(DOOR, 0.0), (H, W)))
    cloud = np.stack([xs, ys], axis=1).astype(np.float64)
    marks = project(list(MARKS.values()), 0.0)
    r0 = np.linalg.norm(geometry.whiten_many(marks, geometry.moments_from_points(cloud), 1), axis=1)
    for _ in range(5):
        a = rng.uniform(-1.0, 1.0, size=(2, 2)) + np.eye(2)
        if np.linalg.det(a) <= 0.1:
            a = a + np.eye(2)
        r1 = np.linalg.norm(geometry.whiten_many(marks @ a.T, geometry.moments_from_points(cloud @ a.T), 1), axis=1)
        assert np.abs(r1 - r0).max() < tol


def test_two_affine_viewpoints_of_the_door_agree_and_the_band_widens_at_the_corners() -> None:
    """The raster version of the invariance: a second view is cv2.warpAffine of the frontal door (rotation, scale,
    translation). Planted marks land within 0.02 units of their frontal position everywhere; tau(r) at a corner
    (r about 2.0) is about 3 times the centre band, which is what 'the band widens with the radius' means."""
    front = fill(project(DOOR, 0.0), (H, W))
    mom0 = geometry.moments_from_mask(front)
    marks0 = project(list(MARKS.values()), 0.0)
    w0 = geometry.whiten_many(marks0, mom0, 1)
    ang, s = math.radians(12.0), 0.8
    a = np.array([[s * math.cos(ang), -s * math.sin(ang), 140.0], [s * math.sin(ang), s * math.cos(ang), 60.0]])
    second = cv2.warpAffine(front.astype(np.uint8) * 255, a, (W, H), flags=cv2.INTER_NEAREST) > 0
    mom1 = geometry.moments_from_mask(second)
    marks1 = marks0 @ a[:, :2].T + a[:, 2]
    w1 = geometry.whiten_many(marks1, mom1, 1)
    residual = np.linalg.norm(w1 - w0, axis=1)
    assert residual.max() < 0.02
    names = list(MARKS)
    r_centre = float(np.linalg.norm(w0[names.index("centre")]))
    r_corner = float(np.linalg.norm(w0[names.index("tl")]))
    assert r_centre < 0.02 and 1.9 < r_corner < 2.1
    assert tau_r(TAU_PRE, r_corner) > 2.5 * tau_r(TAU_PRE, r_centre)
    for name, res in zip(names, residual):
        assert res < tau_r(TAU_PRE, float(np.linalg.norm(w0[names.index(name)])))


# Perspective: what the moment frame does not remove -----------------------------------------------------------------


def test_perspective_residual_at_20_degrees_measured_on_the_synthetic_door() -> None:
    """Frontal against 20 degrees of yaw through the moment frame alone. Measured: centre 0.236 units, top and
    bottom 0.237, corners 0.16 to 0.17, left and right edges 0.085. The keystone pulls the mask centroid toward the
    near edge, so the error is a near-translation of the frame: largest on the vertical centreline, smallest at
    the side edges, and NOT the 0.06 the research note computed for a correspondence-based affine fit. It stays
    inside tau_pre at the centre and inside tau(r) everywhere, and the whitened IoU (0.93) does not gate it."""
    mask0, mom0, w0, _ = door_view(0.0)
    mask20, mom20, w20, _ = door_view(20.0)
    res = {k: dist(w0[k], w20[k]) for k in MARKS}
    assert 0.20 < res["centre"] < TAU_PRE
    assert res["left"] < 0.1 and res["right"] < 0.1
    assert max(res["tl"], res["tr"], res["bl"], res["br"]) < 0.2
    for k, r in res.items():
        assert r < tau_r(TAU_PRE, math.hypot(*w0[k]))
    assert geometry.whitened_mask_iou(mask0, mom0, 1, mask20, mom20, 1) > float(TH["matching"]["mask_iou_min"])


def test_handle_residual_subtraction_meets_the_section_13_bounds_at_20_degrees() -> None:
    """The handle self-test of 5.8 item 2 subtracts the whitened handle difference from the second view. With it the
    frontal-vs-20-degree residual is 0.067 at the centre (bound 0.08) and 0.11 to 0.15 at the corners (bound
    0.25): the bounds of section 13 hold for a door with a visible handle, not for the bare moment frame."""
    centre_max = float(TH["geometry"]["perspective_residual_center_max"])
    corner_max = float(TH["geometry"]["perspective_residual_corner_max"])
    _, _, w0, h0 = door_view(0.0)
    _, _, w20, h20 = door_view(20.0)
    shift = (h20[0] - h0[0], h20[1] - h0[1])
    assert 0.15 < math.hypot(*shift) < TAU_PRE  # under tau_pre, so G5 does not fire and the shift is applied
    corrected = {k: (w20[k][0] - shift[0], w20[k][1] - shift[1]) for k in MARKS}
    res = {k: dist(w0[k], corrected[k]) for k in MARKS}
    assert res["centre"] < centre_max
    assert max(res["tl"], res["tr"], res["bl"], res["br"]) < corner_max
    assert res["centre"] < res["tl"]  # after the correction the corners carry the larger residual


def test_a_ten_degree_yaw_mismatch_stays_inside_tau_pre_without_any_correction() -> None:
    """Pairing keyframes by aspect ratio (5.8 item 1) leaves a small yaw mismatch between walks. Measured at 10
    degrees: centre 0.119, corners 0.08, edges 0.04; the residual grows about 0.012 units per degree."""
    _, _, w0, _ = door_view(0.0)
    _, _, w10, _ = door_view(10.0)
    _, _, w5, _ = door_view(5.0)
    res10 = {k: dist(w0[k], w10[k]) for k in MARKS}
    res5 = {k: dist(w0[k], w5[k]) for k in MARKS}
    assert max(res10.values()) < 0.15 and max(res5.values()) < 0.08
    assert all(res10[k] >= res5[k] for k in MARKS)


def test_opposite_yaws_land_in_the_needs_your_eye_band_at_the_centre() -> None:
    """+20 against -20 degrees doubles the residual (measured 0.47 at the centre): between tau_pre and tau_eye, so
    the band design prints needs_your_eye rather than a false new or unseen; the handle correction brings it to
    0.12. The aspect ratios differ (1.04 against 1.04, mirror images), which is why pairing prefers equal yaw."""
    _, _, wp, hp = door_view(20.0)
    _, _, wn, hn = door_view(-20.0)
    centre = dist(wp["centre"], wn["centre"])
    assert TAU_PRE <= centre < TAU_EYE
    shift = (hn[0] - hp[0], hn[1] - hp[1])
    assert dist(wp["centre"], (wn["centre"][0] - shift[0], wn["centre"][1] - shift[1])) < 0.15


# Carried moments ---------------------------------------------------------------------------------------------------


def test_carried_moments_match_a_full_decode_on_a_synthetic_clip() -> None:
    """A keyframe box carried to five clip frames that translate and scale the panel; whitening a mark with the
    carried moments stays within carried_moments_residual_max_units of the moments of the decoded frame mask."""
    limit = float(TH["geometry"]["carried_moments_residual_max_units"])
    shape = (900, 1400)
    key_box = (300, 250, 780, 610)  # 480 x 360
    key_mask = rect_mask(shape, key_box)
    key_mom = geometry.moments_from_mask(key_mask)
    mark = (520.0, 400.0)
    w_key = geometry.whiten(mark, key_mom, -1)
    for f, (dx, dy, sx, sy) in enumerate([(40, -20, 1.0, 1.0), (90, 10, 1.1, 1.08), (-60, 30, 0.9, 0.94), (150, -40, 1.25, 1.2), (0, 60, 0.8, 0.85)]):
        x1 = key_box[0] + dx
        y1 = key_box[1] + dy
        w_px = int(round((key_box[2] - key_box[0]) * sx))
        h_px = int(round((key_box[3] - key_box[1]) * sy))
        frame_box = (x1, y1, x1 + w_px, y1 + h_px)
        decoded = geometry.moments_from_mask(rect_mask(shape, frame_box))
        carried = geometry.carry_moments(key_mom, list(key_box), list(frame_box))
        mark_f = (x1 + (mark[0] - key_box[0]) * w_px / (key_box[2] - key_box[0]), y1 + (mark[1] - key_box[1]) * h_px / (key_box[3] - key_box[1]))
        w_carried = geometry.whiten(mark_f, carried, -1)
        w_decoded = geometry.whiten(mark_f, decoded, -1)
        assert dist(w_carried, w_decoded) < limit, f"frame {f}"
        assert dist(w_carried, w_key) < limit, f"frame {f}: the mark moved with the panel"
    with pytest.raises(ValueError):
        geometry.carry_moments(key_mom, [0, 0, 10, 10], [0, 0, 0, 10])


# Crop, source, proxy and sent maps ----------------------------------------------------------------------------------


def test_crop_geometry_pads_clamps_and_limits_the_long_edge() -> None:
    crop = geometry.crop_geometry([1000.0, 500.0, 3000.0, 1700.0], (3840, 2160), pad_frac=0.1, long_edge=1280)
    assert (crop.dx, crop.dy) == (800.0, 380.0)  # 10% of 2000 and of 1200 per side
    assert crop.scale == pytest.approx(1280 / 2400) and (crop.w, crop.h) == (1280, 768)
    assert crop.path == ""
    edge = geometry.crop_geometry([0.0, 0.0, 500.0, 400.0], (3840, 2160), pad_frac=0.1, long_edge=1280)
    assert (edge.dx, edge.dy) == (0.0, 0.0) and edge.scale == 1.0 and (edge.w, edge.h) == (550, 440)  # never upscaled
    with pytest.raises(ValueError):
        geometry.crop_geometry([3900.0, 10.0, 3950.0, 20.0], (3840, 2160), pad_frac=0.1, long_edge=1280)


def test_crop_source_box_and_proxy_maps_round_trip() -> None:
    crop = Crop(path="", w=1280, h=960, scale=0.5, dx=100.0, dy=50.0)
    p_src = geometry.crop_to_src([640.0, 480.0], crop)
    assert p_src == [1380.0, 1010.0]
    assert geometry.src_to_crop(p_src, crop) == [640.0, 480.0]
    box = [10.0, 20.0, 110.0, 220.0]
    assert geometry.box_src_to_crop(geometry.box_crop_to_src(box, crop), crop) == pytest.approx(box)
    proxy_box = [96.0, 54.0, 192.0, 108.0]
    src_box = geometry.proxy_to_src(proxy_box, (960, 540), (3840, 2160))
    assert src_box == [384.0, 216.0, 768.0, 432.0]
    assert geometry.src_to_proxy(src_box, (960, 540), (3840, 2160)) == pytest.approx(proxy_box)
    assert geometry.sent_to_crop((320, 240), (640, 480), (1280, 960)) == [640.0, 480.0]


def test_box_iou_and_containment() -> None:
    a = [0.0, 0.0, 10.0, 10.0]
    b = [5.0, 0.0, 15.0, 10.0]
    assert geometry.box_iou(a, b) == pytest.approx(50.0 / 150.0)
    assert geometry.box_iou(a, a) == 1.0 and geometry.box_iou(a, [20.0, 20.0, 30.0, 30.0]) == 0.0
    assert geometry.containment([2.0, 2.0, 4.0, 4.0], a) == 1.0
    assert geometry.containment(b, a) == pytest.approx(0.5)
    assert geometry.containment([3.0, 3.0, 3.0, 3.0], a) == 0.0


# Whitened mask IoU -------------------------------------------------------------------------------------------------


def test_whitened_mask_iou_is_high_for_the_same_panel_and_lower_for_a_different_shape() -> None:
    front = fill(project(DOOR, 0.0), (H, W))
    mom = geometry.moments_from_mask(front)
    a = np.array([[0.7 * math.cos(0.3), -0.7 * math.sin(0.3), 200.0], [0.7 * math.sin(0.3), 0.7 * math.cos(0.3), 100.0]])
    warped = cv2.warpAffine(front.astype(np.uint8) * 255, a, (W, H), flags=cv2.INTER_NEAREST) > 0
    same = geometry.whitened_mask_iou(front, mom, 1, warped, geometry.moments_from_mask(warped), 1)
    assert same > 0.95
    notched = front.copy()
    notched[: H // 2, : W // 2] = False  # a quarter missing: a different mask definition
    other = geometry.whitened_mask_iou(front, mom, 1, notched, geometry.moments_from_mask(notched), 1)
    assert other < same and other < 0.8
    assert geometry.whitened_mask_iou(front, mom, 1, np.zeros_like(front), mom, 1) == 0.0


# Secondary frames ----------------------------------------------------------------------------------------------------


def disk_mask(shape: tuple[int, int], centre: tuple[int, int], radius: int) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    cv2.circle(m, centre, radius, 255, -1)
    return m > 0


def test_wheel_coords_radius_and_angle() -> None:
    wheel = disk_mask((500, 500), (250, 250), 200)
    assert geometry.inscribed_radius(wheel) == pytest.approx(200.0, abs=2.0)
    r0, _ = geometry.wheel_coords([250.0, 250.0], wheel)
    assert r0 == pytest.approx(0.0, abs=0.01)
    r_rim, ang = geometry.wheel_coords([450.0, 250.0], wheel)
    assert r_rim == pytest.approx(1.0, abs=0.02) and ang == pytest.approx(0.0, abs=0.01)
    r_mid, ang = geometry.wheel_coords([250.0, 350.0], wheel)
    assert r_mid == pytest.approx(0.5, abs=0.02) and ang == pytest.approx(math.pi / 2, abs=0.01)
    r_out, _ = geometry.wheel_coords([490.0, 490.0], wheel)
    assert r_out == 1.0
    with pytest.raises(ValueError):
        geometry.wheel_coords([1.0, 1.0], np.zeros((5, 5), dtype=bool))


def test_bumper_arc_between_anchors_and_height_in_the_mask() -> None:
    bumper = rect_mask((300, 1000), (50, 100, 950, 250))
    assert geometry.bumper_arc([500.0, 175.0], None, bumper) is None
    assert geometry.bumper_arc([500.0, 175.0], ([100.0, 175.0], None), bumper) is None  # type: ignore[arg-type]
    anchors = ([100.0, 175.0], [900.0, 175.0])
    assert geometry.bumper_arc([100.0, 100.0], anchors, bumper) == pytest.approx((0.0, 0.0), abs=1e-6)
    arc, height = geometry.bumper_arc([900.0, 249.0], anchors, bumper)
    assert arc == pytest.approx(1.0) and height == pytest.approx(1.0)
    arc, height = geometry.bumper_arc([500.0, 175.0], anchors, bumper)
    assert arc == pytest.approx(0.5) and height == pytest.approx(0.5, abs=0.01)
    assert geometry.bumper_arc([500.0, 175.0], ([100.0, 175.0], [100.0, 175.0]), bumper) is None  # coincident anchors


def test_hood_coords_lateral_axis_and_depth_thirds() -> None:
    hood = rect_mask((400, 800), (100, 100, 700, 340))  # 600 wide, 240 deep
    mom = geometry.moments_from_mask(hood)
    u, third = geometry.hood_coords([399.5, 120.0], mom, 1, hood)
    assert u == pytest.approx(0.0, abs=1e-6) and third == 0
    u, third = geometry.hood_coords([700.0, 220.0], mom, 1, hood)
    assert u > 1.5 and third == 1
    u, third = geometry.hood_coords([100.0, 330.0], mom, -1, hood)
    assert u > 1.5 and third == 2
    _, third = geometry.hood_coords([400.0, 10000.0], mom, 1, hood)
    assert third == 2


def test_boundary_distance_units() -> None:
    panel = rect_mask((500, 500), (100, 100, 400, 400))
    mom = geometry.moments_from_mask(panel)
    centre = geometry.boundary_distance_units([249.5, 249.5], panel, mom, 1)
    assert centre == pytest.approx(math.sqrt(3.0), rel=0.02)
    edge = geometry.boundary_distance_units([102.0, 250.0], panel, mom, 1)
    assert edge < 0.05
    assert centre > float(TH["staysput"]["curvature_boundary_units"]) > edge
