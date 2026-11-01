"""walkaround.staysput: windows, prompts, the analysis on synthetic tracks, and derive_staysput end to end on synthetic
walks with only the network replaced. The fake Responses API answers an image check with the saved zero-match stream
(or, when asked to look, with a box on a yellow plate or a magenta face it finds in the image) and a clip request by
tracking colour blobs in the clip it was sent (red paint, white highlight), so the real cut_clip, privacy, sam_client
and sam_wire code runs. No key, no socket."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from walkaround import config, geometry, ledger, paths, privacy, sam_client, staysput, video
from walkaround import contracts as c

ROOT = Path(__file__).resolve().parent
ZERO = [json.loads(line) for line in (ROOT / "fixtures/sam_zero_match.events.ndjson").read_text().splitlines() if line.strip()]
MASK_TOKEN = "<|mask;x=0;y=0;c=0.9;data=33,27,!!!!!CFj28bL3[E2bt`E^7j`4++{J.K:4E8h+v1gm!!|>"  # never decoded here
MASK_11 = "<|mask;x=0;y=0;data=11,11,~!!!!r!kCq#$.pF[_cZIw.?wO3R8-&NJofiCI$_J0aA.1VC*YyK_'z*(LEG8!+H9evY.82Z-4.@C[.9C@T86o$3E=-&8w3W=]xS.|>"
SRC, PROXY = (640, 360), (320, 180)
N_SRC = 150  # 5 s at 30 fps
KF_PROXY = 37  # shows source frame 74
CLIP = 252
FACE_BOX = (270, 110, 282, 122)  # still in the frame, source pixels: inside the paint's window at the prompt frame


def thresholds() -> dict[str, Any]:
    th = c.load_thresholds()
    th["media"]["clip_size_px"] = CLIP  # a small clip keeps the encode fast; the geometry is the same
    th["staysput"]["window_min_px"] = 120  # the synthetic walk is 640 px wide
    return th


TH = thresholds()


def square(cx: float, cy: float, side: float) -> list[float]:
    return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]


def centre(b: list[float]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of the luminance: near 1 when b still shows a's texture, low once it is
    pixelated (on raw RGB the fixed channel offsets of a coloured patch would dominate)."""
    x, y = a.astype(np.float64).mean(axis=-1).ravel(), b.astype(np.float64).mean(axis=-1).ravel()
    x, y = x - x.mean(), y - y.mean()
    d = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d else 0.0


# ----------------------------------------------------------------------------------------------------------------
# Synthetic tracks for analyse (no video)
# ----------------------------------------------------------------------------------------------------------------


def moving_panel(i: int) -> list[float]:
    """A 300 x 160 panel crossing the frame at 3 px per frame (the camera walks along the car)."""
    x1 = 180 - 3.0 * (i - 45)
    return [x1, 100.0, x1 + 300.0, 260.0]


def synthetic(panel: Any, mark: Any, *, n: int = 91, k: int = 45, side: float = 120.0, drop: set[int] = frozenset()) -> dict[str, Any]:
    """What analyse sees for one object: the clip lane's boxes, panel boxes, windows following the panel, and the
    keyframe's crop and moments (a full rectangle mask)."""
    panels = [panel(i) for i in range(n)]
    kf_box = panels[k]
    crop = geometry.crop_geometry(kf_box, SRC, pad_frac=0.1, long_edge=1280)
    mask = np.zeros((crop.h, crop.w), dtype=bool)
    x1, y1, x2, y2 = (int(round(v)) for v in geometry.box_src_to_crop(kf_box, crop))
    mask[y1:y2, x1:x2] = True
    moments = geometry.moments_from_mask(mask)
    mc = centre(mark(k))
    windows = []
    for b in panels:
        sx, sy = (b[2] - b[0]) / (kf_box[2] - kf_box[0]), (b[3] - b[1]) / (kf_box[3] - kf_box[1])
        windows.append(square(b[0] + (mc[0] - kf_box[0]) * sx, b[1] + (mc[1] - kf_box[1]) * sy, side))
    frames = []
    for i in range(n):
        if i in drop:
            continue
        m, w = mark(i), windows[i]
        s = CLIP / side
        box = [int(round((m[0] - w[0]) * s)), int(round((m[1] - w[1]) * s)), int(round((m[2] - w[0]) * s)), int(round((m[3] - w[1]) * s))]
        frames.append(c.IndexFrame(frame=i, objects=[c.IndexObject(id="1", box=box, c=0.9)]))
    return {"frames": frames, "panels": panels, "windows": windows, "kf_box": kf_box, "crop": crop, "moments": moments}


def run_analyse(s: dict[str, Any], *, direction: int | None = None, th: dict[str, Any] = TH, **kw: Any) -> c.MarkTrack:
    return staysput.analyse(
        s["frames"], "m1", s["panels"], s["moments"], 1, direction, th=th, object_id="1", windows_src=s["windows"],
        clip_size=CLIP, kf_box_src=s["kf_box"], crop=s["crop"], src_size=SRC, proxy_size=PROXY, role="pickup", **kw,
    )


def paint(i: int) -> list[float]:
    p = moving_panel(i)
    return [p[0] + 60, p[1] + 40, p[0] + 68, p[1] + 48]


def reflection(i: int) -> list[float]:
    """Lags the panel by 0.8 px per frame: it slides along the panel in the direction the camera travels."""
    p = moving_panel(i)
    x = p[0] + 200 + 0.8 * (i - 45)
    return [x, p[1] + 70, x + 14, p[1] + 80]


def test_paint_moving_with_the_panel_is_fixed() -> None:
    mt = run_analyse(synthetic(moving_panel, paint))
    assert mt.verdict == c.StaysPutVerdict.FIXED, mt
    assert mt.drift_rms is not None and mt.drift_rms < 0.02
    assert mt.b1 == pytest.approx(270 / 640, abs=1e-9) and mt.b2 == pytest.approx(0.0, abs=1e-9)
    assert mt.b_pass and not mt.panned and len(mt.track) == 91
    assert all(p.inside_panel and p.w is not None and p.u_rel is not None for p in mt.track)
    assert mt.sign_agreement is not None and mt.sign_agreement < 0.8  # paint has no direction of its own


def test_a_highlight_moving_with_the_camera_slides() -> None:
    mt = run_analyse(synthetic(moving_panel, reflection))
    assert mt.verdict == c.StaysPutVerdict.SLIDING, mt
    assert mt.drift_rms is not None and mt.drift_rms > 2 * staysput.tau_fix(TH)
    assert mt.sign_agreement == 1.0 and mt.pairs_n >= 8


def test_large_drift_without_the_camera_sign_is_inconclusive_not_sliding() -> None:
    rng = np.random.default_rng(3)
    jitter = rng.normal(0, 25, size=91)

    def noisy(i: int) -> list[float]:
        b = paint(i)
        return [b[0] + jitter[i], b[1], b[2] + jitter[i], b[3]]

    mt = run_analyse(synthetic(moving_panel, noisy))
    assert mt.drift_rms is not None and mt.drift_rms > 2 * staysput.tau_fix(TH)
    assert mt.sign_agreement is not None and mt.sign_agreement < 0.8
    assert mt.verdict == c.StaysPutVerdict.INCONCLUSIVE


def test_a_still_camera_is_inconclusive() -> None:
    still = lambda i: [180.0, 100.0, 480.0, 260.0]  # noqa: E731
    mt = run_analyse(synthetic(still, lambda i: [240.0, 140.0, 248.0, 148.0]))
    assert mt.b1 == 0.0 and mt.b2 == 0.0 and not mt.b_pass and not mt.panned
    assert mt.verdict == c.StaysPutVerdict.INCONCLUSIVE and mt.drift_rms is not None and mt.drift_rms < 0.01
    assert mt.sign_agreement is None and mt.pairs_n == 0  # no camera motion and no walk direction: no pairs


def test_the_walk_direction_signs_pairs_when_the_panel_barely_moves() -> None:
    slow = lambda i: [180.0 - 0.1 * (i - 45), 100.0, 480.0 - 0.1 * (i - 45), 260.0]  # noqa: E731
    drifting = lambda i: [240.0 + 0.5 * (i - 45), 140.0, 248.0 + 0.5 * (i - 45), 148.0]  # noqa: E731
    with_dir = run_analyse(synthetic(slow, drifting), direction=-1)  # scene drifts left, so the camera walks right
    assert with_dir.pairs_n == 18 and with_dir.sign_agreement == 1.0  # 0-5, 5-10, ..., 85-90: no pair overlaps another
    against = run_analyse(synthetic(slow, drifting), direction=1)
    assert against.sign_agreement == 0.0


def test_sign_pairs_keep_their_spacing_in_time_at_24_fps() -> None:
    slow = lambda i: [180.0 - 0.1 * (i - 36), 100.0, 480.0 - 0.1 * (i - 36), 260.0]  # noqa: E731
    drifting = lambda i: [240.0 + 0.5 * (i - 36), 140.0, 248.0 + 0.5 * (i - 36), 148.0]  # noqa: E731
    s = synthetic(slow, drifting, n=73, k=36)
    at_24 = run_analyse(s, direction=-1, fps=24.0)
    assert at_24.pairs_n == 18 and at_24.sign_agreement == 1.0  # pairs 4 frames apart: 1/6 s, as 5 frames at 30 fps
    assert run_analyse(s, direction=-1).pairs_n == 14  # read as 30 fps the same clip gives 5-frame pairs


def test_u_is_measured_only_while_the_mark_is_on_its_panel() -> None:
    def leaving_mark(i: int) -> list[float]:
        p = moving_panel(i)
        x = p[2] - 40 + 1.5 * (i - 45)  # crosses the panel's right edge after frame 70
        return [x, 140.0, x + 8, 148.0]

    mt = run_analyse(synthetic(moving_panel, leaving_mark))
    off = [p for p in mt.track if not p.inside_panel]
    assert off and all(p.u_rel is None and p.w is None for p in off)
    assert all(p.u_rel is not None for p in mt.track if p.inside_panel)


def test_a_panned_clip_is_flagged_and_still_measured() -> None:
    def panned_panel(i: int) -> list[float]:
        w = 300.0 - 0.6 * (i - 45)  # the panel turns away while the camera keeps it centred
        return [320 - w / 2, 100.0, 320 + w / 2, 260.0]

    def mark(i: int) -> list[float]:
        p = panned_panel(i)
        x = p[0] + 0.2 * (p[2] - p[0])
        return [x, 140.0, x + 8, 148.0]

    mt = run_analyse(synthetic(panned_panel, mark))
    assert mt.b1 is not None and mt.b1 < TH["staysput"]["b1_floor"]
    assert mt.b2 is not None and mt.b2 > TH["staysput"]["b2_floor"]
    assert mt.panned and mt.b_pass and mt.verdict == c.StaysPutVerdict.FIXED


def test_a_lost_track_is_no_track() -> None:
    mt = run_analyse(synthetic(moving_panel, paint, drop=set(range(0, 25)) | set(range(70, 91))))
    assert len(mt.track) == 45 and mt.verdict == c.StaysPutVerdict.NO_TRACK


def test_the_box_share_counts_the_propagated_frames() -> None:
    kept = set(range(0, 18))  # 73 frames left, the prompt frame 45 among them
    s = synthetic(moving_panel, paint, drop=kept)
    assert run_analyse(s).verdict == c.StaysPutVerdict.FIXED  # 73 of 91 frames: 0.802
    assert run_analyse(s, prompt_frame=45).verdict == c.StaysPutVerdict.FIXED  # 72 of 90 propagated frames: 0.800
    s = synthetic(moving_panel, paint, drop=kept | {18})
    assert run_analyse(s, prompt_frame=45).verdict == c.StaysPutVerdict.NO_TRACK  # 71 of 90


def test_frames_with_the_panel_cut_by_the_frame_edge_are_left_out() -> None:
    def leaving(i: int) -> list[float]:
        x1 = 300 + 3.0 * (i - 45)  # crosses the right edge (640) after frame 58
        return [x1, 100.0, x1 + 300.0, 260.0]

    def mark(i: int) -> list[float]:
        p = leaving(i)
        return [p[0] + 60, 140.0, p[0] + 68, 148.0]

    mt = run_analyse(synthetic(leaving, mark))
    used = [p.f for p in mt.track if p.w is not None]
    assert used and max(used) < 60 and all(p.u_rel is None for p in mt.track if p.f > 60)
    assert mt.verdict == c.StaysPutVerdict.FIXED  # paint on the frames that hold the whole panel


def test_tau_fix_reads_its_own_sigma_frame_not_the_matchers() -> None:
    th = copy.deepcopy(TH)
    assert staysput.tau_fix(th) == th["staysput"]["tau_fix_floor"]
    th["matching"]["sigma_frame"] = 0.1689  # D1's E8 number, which would make tau_fix 0.507 (D1 note 2: not adopted)
    assert staysput.tau_fix(th) == th["staysput"]["tau_fix_floor"]
    assert run_analyse(synthetic(moving_panel, reflection), th=th).verdict == c.StaysPutVerdict.SLIDING
    th["staysput"]["sigma_frame"] = 0.1  # a recalibrated Stays Put leaf does move it
    assert staysput.tau_fix(th) == pytest.approx(0.3)
    assert run_analyse(synthetic(moving_panel, reflection), th=th).verdict == c.StaysPutVerdict.FIXED  # drift 0.24 < 0.3


def test_analyse_is_deterministic() -> None:
    s = synthetic(moving_panel, reflection)
    assert run_analyse(s).to_json() == run_analyse(s).to_json()


# ----------------------------------------------------------------------------------------------------------------
# Windows, spans, prompts, glare
# ----------------------------------------------------------------------------------------------------------------


def track_of(panel: Any, n_proxy: int, times: list[float]) -> c.PanelTrack:
    rows = []
    for f in range(n_proxy):
        b = [v / 2 for v in panel(2 * f)]
        rows.append(c.FrameObs(frame=f, pts=times[2 * f], box=b, box_area_frac=0.1, in_car_box=True, c=0.9))
    return c.PanelTrack(track_id="pickup_hood_00", role="pickup", noun=c.Noun.HOOD, sam_object_ids=["0"],
                        fragments=[c.TrackFragment(object_id="0", first_frame=0, last_frame=n_proxy - 1)],
                        c_present_share=1.0, per_frame=rows, panel_id=c.PanelId.HOOD)


def test_span_slides_inside_the_walk() -> None:
    times_us = [int(round(i / 30 * 1e6)) for i in range(150)]
    assert staysput._span(times_us, 74 / 30, 91) == (29, 91, 45)
    assert staysput._span(times_us, 0.5, 91) == (0, 91, 15)  # 1.5 s before would run past the start
    assert staysput._span(times_us, 4.9, 91) == (59, 91, 88)
    assert staysput._span(times_us[:40], 0.5, 91) == (0, 40, 15)  # a walk shorter than a clip gives what it has


def test_a_clip_is_three_seconds_whatever_the_frame_rate() -> None:
    def times(fps: float, n: int = 400) -> list[float]:
        return [i / fps for i in range(n)]

    assert staysput.clip_frames(times(30), TH) == 91
    assert staysput.clip_frames(times(1479 / 50), TH) == 89  # the uCO3D laps
    assert staysput.clip_frames(times(24), TH) == 73  # the Pexels S550 clips
    assert staysput.clip_frames(times(60), TH) == 181
    assert staysput.clip_frames(times(120), TH) == 181  # capped: a slow-motion walk gets 1.5 s
    assert staysput.clip_frames([0.0], TH) == TH["media"]["clip_frames"]


def test_panel_boxes_interpolate_between_proxy_frames_and_stop_at_gaps() -> None:
    times = [i / 30 for i in range(150)]
    track = track_of(lambda i: [330 - 2.0 * i, 100.0, 630 - 2.0 * i, 260.0], 75, times)
    kf = c.Keyframe(kf_id="k", role="pickup", panel_id=c.PanelId.HOOD, track_id=track.track_id, f_proxy=37, pts=times[74],
                    score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=1.0,
                    box_src=[182.0, 100.0, 482.0, 260.0])
    boxes = staysput.panel_boxes(track, kf, times, 29, 91, src_size=SRC, proxy_size=PROXY, th=TH)
    for i, b in enumerate(boxes):  # rows sit on even source frames; odd frames are halfway between
        assert b == pytest.approx([330 - 2.0 * (29 + i), 100.0, 630 - 2.0 * (29 + i), 260.0])
    gappy = track.model_copy(update={"per_frame": [r for r in track.per_frame if not 40 <= r.frame < 50]})  # 0.67 s gap
    boxes = staysput.panel_boxes(gappy, kf, times, 29, 91, src_size=SRC, proxy_size=PROXY, th=TH)
    assert boxes[80 - 29] is None and boxes[78 - 29] is not None and boxes[45] == kf.box_src


def item(mark_id: str, box: list[float], *, control: bool = False) -> Any:
    return staysput._Item(mark_id=mark_id, kf_id="k", panel_id=c.PanelId.HOOD, type=c.MarkKind.CHIP, is_control=control, box_src=tuple(box),
                          point_crop=None, bbox_crop=None, rank=0)


def test_overlapping_windows_share_one_clip_up_to_16() -> None:
    near = [item("a", [100, 100, 110, 110]), item("b", [140, 120, 150, 130]), item("c", [400, 100, 410, 110])]
    groups = staysput._group(near, 120.0, TH)
    assert [[it.mark_id for it in g[0]] for g in groups] == [["a", "b"], ["c"]]
    (cx, cy), side = groups[0][1], groups[0][2]
    assert (cx, cy) == (125.0, 115.0) and side == 120.0
    many = [item(f"m{i:02d}", [100 + i, 100, 104 + i, 104]) for i in range(20)]
    assert [len(g[0]) for g in staysput._group(many, 120.0, TH)] == [16, 4]
    long_mark = staysput._group([item("s", [100, 100, 300, 110])], 120.0, TH)  # a long scratch widens its window
    assert long_mark[0][2] == pytest.approx(1.5 * 200)


def chip(mark_id: str, **where: Any) -> c.Mark:
    return c.Mark(mark_id=mark_id, role="pickup", panel_id=c.PanelId.HOOD, kf_id="k", source=c.MarkSource.SPARK_BOX, type=c.MarkKind.CHIP,
                  confidence=c.Confidence.HIGH, size_class=c.SizeClass.S, **where)


def test_prompt_for_builds_one_object_per_mark_on_the_prompt_frame() -> None:
    window = [100.0, 100.0, 220.0, 220.0]
    marks = [chip("k_m1", bbox_src=[150.0, 150.0, 160.0, 160.0]), chip("k_m2", bbox_src=[130.0, 130.0, 130.0, 130.0])]  # m2 collapsed
    text = staysput.prompt_for(marks, window, CLIP, TH)
    assert text.startswith("<45f>1<|box;") and ",2<|box;" in text
    boxes = re.findall(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=252;h=252\|>", text)
    assert [b[0] for b in boxes] == ["1", "2"]
    x1, y1, x2, y2 = (int(v) for v in boxes[0][1:])
    assert (x1, y1, x2 + 1, y2 + 1) == (103, 103, 128, 128)  # 2.1 px per source px, padded 2 px, half-open on the wire
    x1, _, x2, _ = (int(v) for v in boxes[1][1:])
    assert x2 + 1 - x1 >= 2  # a collapsed box pads to at least 2 px and builds
    assert staysput.prompt_for(marks[:1], window, CLIP, TH, prompt_frame=12).startswith("<12f>1<|box;")
    # with the walk's size it grows boxes exactly as the live path does: 12 px at 4K is 24 px on a 7680 px walk
    grown = staysput.prompt_for(marks[:1], window, CLIP, TH, src_size=(7680, 4320))
    x1, x2 = (int(v) for v in re.findall(r"x1=(\d+);y1=\d+;x2=(\d+)", grown)[0])
    assert abs((x2 + 1 - x1) - (24 * 2.1 + 2 * TH["panels"]["prompt_box_pad_px"])) <= 1.5


def test_a_mark_under_the_size_floor_gets_no_clip() -> None:
    box, why = staysput._mark_box(chip("a", bbox_src=[100.0, 100.0, 110.0, 105.0]), 12.0)
    assert box is None and why is not None and why.startswith("too small for a clip") and "section 14 item 17" in why
    box, why = staysput._mark_box(chip("b", bbox_src=[100.0, 100.0, 130.0, 102.0]), 12.0)  # a thin 30 px scratch
    assert why is None and box == pytest.approx([100.0, 95.0, 130.0, 107.0])  # grown to 12 px across, not refused
    box, why = staysput._mark_box(chip("c", point_src=[50.0, 50.0]), 12.0)  # a point of unknown size is not refused
    assert why is None and box == pytest.approx([44.0, 44.0, 56.0, 56.0])
    assert staysput._mark_box(chip("d", point_src=[50.0, 50.0], size_px_src=8.0), 12.0)[0] is None
    assert staysput._mark_box(chip("e"), 12.0) == (None, "the mark has no source position")


def hood_hsv(base_v: int, base_s: int) -> tuple[np.ndarray, np.ndarray]:
    hsv = np.zeros((100, 200, 3), dtype=np.uint8)
    hsv[..., 2], hsv[..., 1] = base_v, base_s
    mask = np.zeros((100, 200), dtype=bool)
    mask[10:90, 10:190] = True
    return mask, hsv


def test_glare_blob_takes_the_largest_highlight_and_refuses_sunlit_paint() -> None:
    mask, hsv = hood_hsv(150, 120)  # saturated mid-tone paint
    hsv[20:30, 40:60, 2], hsv[20:30, 40:60, 1] = 255, 0  # 200 px highlight
    hsv[50:53, 100:103, 2], hsv[50:53, 100:103, 1] = 255, 0  # 9 px speck
    box, point = staysput.glare_blob(mask, hsv, TH)
    assert box == [40.0, 20.0, 60.0, 30.0] and point == pytest.approx([49.5, 24.5])
    hsv[10:90, 10:190, 2] = 255  # the whole panel in sun: no highlight to stand for a reflection
    assert staysput.glare_blob(mask, hsv, TH) is None
    assert staysput.glare_blob(None, hsv, TH) is None


def test_white_paint_gives_no_reflection_control() -> None:
    mask, hsv = hood_hsv(240, 12)  # a white lid in sun: the glare rule (V > 250, S < 40) picks plain paint
    hsv[30:50, 40:120, 2], hsv[30:50, 40:120, 1] = 255, 8  # a broad bright stretch, 11% of the panel
    assert staysput.glare_blob(mask, hsv, TH) is None  # not above the panel's 95th percentile of V, so no highlight
    mask, hsv = hood_hsv(60, 120)  # dark paint with a crisp highlight, 2.8% of the panel
    hsv[30:40, 40:80, 2], hsv[30:40, 40:80, 1] = 255, 8
    box, point = staysput.glare_blob(mask, hsv, TH)
    assert box == [40.0, 30.0, 80.0, 40.0] and point == pytest.approx([59.5, 34.5])
    hsv[30:50, 40:120, 2], hsv[30:50, 40:120, 1] = 255, 8  # the same stretch at 11%: no longer a top-5% highlight
    assert staysput.glare_blob(mask, hsv, TH) is None


# ----------------------------------------------------------------------------------------------------------------
# Synthetic walks and the fake API
# ----------------------------------------------------------------------------------------------------------------


class Scene:
    """A textured blue panel on a noisy background. paint: a red square fixed on the panel. highlight: a white blob that
    lags the panel by 0.8 px per frame (a reflection). panned: the panel stays centred and turns away instead. plate:
    a yellow plate with a random texture, fixed on the panel. face: a magenta face that stays still in the frame."""

    def __init__(self, *, panned: bool = False, highlight: bool = True, paint_frames: range | None = None, plate: bool = False, face: bool = False, parts: bool = False) -> None:
        rng = np.random.default_rng(7)
        self.panned, self.highlight, self.paint_frames, self.with_plate, self.with_face = panned, highlight, paint_frames, plate, face
        self.with_parts = parts
        texture = rng.integers(0, 41, size=(160, 400, 3), dtype=np.uint8)
        self.texture = texture + np.array([40, 60, 130], dtype=np.uint8)  # blue paint: saturated, never red or white
        bg = rng.integers(25, 70, size=(SRC[1], SRC[0], 3), dtype=np.uint8)
        bg[..., 1] = rng.integers(80, 131, size=(SRC[1], SRC[0]), dtype=np.uint8)
        self.background = bg
        noise = np.random.default_rng(11)
        self.plate_tex = np.stack([noise.integers(190, 256, (10, 24)), noise.integers(150, 226, (10, 24)), noise.integers(0, 41, (10, 24))], axis=-1).astype(np.uint8)
        fh, fw = FACE_BOX[3] - FACE_BOX[1], FACE_BOX[2] - FACE_BOX[0]
        self.face_tex = np.stack([noise.integers(170, 256, (fh, fw)), noise.integers(0, 41, (fh, fw)), noise.integers(170, 256, (fh, fw))], axis=-1).astype(np.uint8)

    def panel(self, i: int) -> list[float]:
        if self.panned:
            w = 300.0 + 0.6 * (74 - i)
            return [320 - w / 2, 100.0, 320 + w / 2, 260.0]
        return [330 - 2.0 * i, 100.0, 630 - 2.0 * i, 260.0]

    def paint(self, i: int) -> list[float] | None:
        if self.paint_frames is not None and i not in self.paint_frames:
            return None
        p = self.panel(i)
        x = p[0] + 0.2 * (p[2] - p[0])
        return [x, 140.0, x + 8, 148.0]

    def glint(self, i: int) -> list[float] | None:
        if not self.highlight:
            return None
        p = self.panel(i)
        x = p[0] + 200 + 0.8 * (i - 74)
        return [x, 170.0, x + 14, 180.0]

    def plate(self, i: int) -> list[float]:
        p = self.panel(i)
        return [p[0] + 75, 180.0, p[0] + 99, 190.0]

    def parts(self, i: int) -> list[list[float]]:
        """Eight green 8 px squares fixed on the panel, four around the paint and four around the glint's path (never
        on it): rigid neighbours."""
        p = self.panel(i)
        spots = [(30, 115), (95, 120), (35, 160), (90, 158), (175, 150), (235, 150), (175, 195), (235, 195)]
        return [[p[0] + dx, float(y), p[0] + dx + 8, float(y + 8)] for dx, y in spots]

    def frame(self, i: int) -> np.ndarray:
        rgb = self.background.copy()
        x1, y1, x2, y2 = (int(round(v)) for v in self.panel(i))
        tex = cv2.resize(self.texture[:, :300], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        rgb[y1:y2, x1:x2] = tex
        if self.with_parts:
            for b in self.parts(i):
                bx1, by1, bx2, by2 = (int(round(v)) for v in b)
                rgb[by1:by2, bx1:bx2] = (30, 220, 40)
        if (b := self.paint(i)) is not None:
            bx1, by1, bx2, by2 = (int(round(v)) for v in b)
            rgb[by1:by2, bx1:bx2] = (230, 20, 20)
        if (b := self.glint(i)) is not None:
            bx1, by1, bx2, by2 = (int(round(v)) for v in b)
            rgb[by1:by2, bx1:bx2] = (255, 255, 255)
        if self.with_plate:
            bx1, by1, bx2, by2 = (int(round(v)) for v in self.plate(i))
            rgb[by1:by2, bx1:bx2] = self.plate_tex[: by2 - by1, : bx2 - bx1]
        if self.with_face:
            rgb[FACE_BOX[1] : FACE_BOX[3], FACE_BOX[0] : FACE_BOX[2]] = self.face_tex
        return rgb


def encode(scene: Scene, dst: Path, fps: int = 30) -> Path:
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SRC[0]}x{SRC[1]}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p", "-g", "30", "-movflags", "+faststart", str(dst)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(N_SRC):
        proc.stdin.write(np.ascontiguousarray(scene.frame(i)).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return dst


@pytest.fixture(scope="module")
def walks(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Scene, Path]]:
    root = tmp_path_factory.mktemp("staysput_walks")
    scenes = {
        "moving": (Scene(), 30),
        "panned": (Scene(panned=True, highlight=False), 30),
        "lost": (Scene(highlight=False, paint_frames=range(54, 95)), 30),
        "moving24": (Scene(), 24),
        "plate": (Scene(highlight=False, plate=True), 30),
        "face": (Scene(highlight=False, face=True), 30),
        "parts": (Scene(parts=True), 30),
    }
    return {name: (scene, encode(scene, root / f"{name}.mp4", fps)) for name, (scene, fps) in scenes.items()}


def set_privacy_lanes(run_id: str, role: str, **status: c.LaneStatus | None) -> None:
    """The walk's license plate and face lane summaries (zero matches unless given); None leaves a lane missing."""
    for noun in c.PRIVACY_NOUNS:
        slug = c.noun_slug(noun)
        path = paths.lane_summary_path(run_id, role, noun)
        state = status.get(slug, c.LaneStatus.ZERO_MATCHES)
        if state is None:
            path.unlink(missing_ok=True)
            continue
        c.write_json(path, c.LaneRecord(lane_id=slug, role=role, noun=noun, prompt_text=noun.value, media_size=list(PROXY), status=state,
                                        zero_matches=state == c.LaneStatus.ZERO_MATCHES))


def add_privacy_track(run_id: str, role: str, noun: c.Noun, box: Any) -> None:
    """A plate or face track as panels.privacy_tracks writes it: one row per proxy frame, in proxy pixels."""
    pts = json.loads(paths.pts_map_path(run_id, role).read_text())
    rows = [c.FrameObs(frame=f, pts=pts[f], box=[v / 2 for v in box(2 * f)], box_area_frac=0.01, in_car_box=True, c=0.9) for f in range(len(pts))]
    track = c.PanelTrack(track_id=f"{role}_{c.noun_slug(noun)}_00", role=role, noun=noun, sam_object_ids=["0"],
                         fragments=[c.TrackFragment(object_id="0", first_frame=0, last_frame=len(pts) - 1)], c_present_share=1.0, per_frame=rows)
    c.write_json(paths.tracks_json(run_id), c.read_json_list(paths.tracks_json(run_id), c.PanelTrack) + [track])


def build_walk(run_id: str, role: str, scene: Scene, src: Path, *, marks: bool = True, fps: int = 30) -> None:
    """The files derive_staysput reads for one walk: the original in quarantine, the video record and pts map, the
    hood track, one hood keyframe with its crop, mask and moments, the paint mark Spark would have found, and plate
    and face lanes that answered with nothing."""
    root = paths.run_dir(run_id)
    shutil.copyfile(src, paths.quarantine_original(run_id, role))
    times = video.frame_times(src)
    pts = [times[2 * f] for f in range(N_SRC // 2)]
    c.write_json(paths.pts_map_path(run_id, role), pts)
    record = c.VideoRecord(
        role=role, original_filename=src.name, bytes=src.stat().st_size,
        ffprobe=c.FfprobeInfo(codec="h264", container="mp4", width=SRC[0], height=SRC[1], display_width=SRC[0], display_height=SRC[1], fps_avg=float(fps),
                              duration_s=N_SRC / fps, has_audio=False),
        proxy=c.ProxyRecord(path=f"media/{role}.proxy.mp4", fps=fps / 2, width=PROXY[0], height=PROXY[1], frames=len(pts), pts_map_path=paths.rel(run_id, paths.pts_map_path(run_id, role))),
    )
    c.write_json(paths.video_json(run_id, role), record)
    rows = [c.FrameObs(frame=f, pts=pts[f], box=[v / 2 for v in scene.panel(2 * f)], box_area_frac=0.2, in_car_box=True, c=0.9) for f in range(len(pts))]
    track = c.PanelTrack(track_id=f"{role}_hood_00", role=role, noun=c.Noun.HOOD, sam_object_ids=["0"], fragments=[c.TrackFragment(object_id="0", first_frame=0, last_frame=len(pts) - 1)],
                         c_present_share=1.0, per_frame=rows, panel_id=c.PanelId.HOOD)
    kf_id = f"{role}_hood_kf1"
    box_src = [v * 2 for v in rows[KF_PROXY].box]
    crop = geometry.crop_geometry(box_src, SRC, pad_frac=0.1, long_edge=1280)
    crop_file = paths.crop_path(run_id, role, kf_id)
    sha = video.write_jpeg(video.cut_crop(scene.frame(2 * KF_PROXY), crop), crop_file, quality=95)
    mask = np.zeros((crop.h, crop.w), dtype=bool)
    x1, y1, x2, y2 = (int(round(v)) for v in geometry.box_src_to_crop(box_src, crop))
    mask[y1:y2, x1:x2] = True
    mask_file = paths.crop_mask_png_path(run_id, role, kf_id)
    video.write_png_mask(mask, mask_file)
    kf = c.Keyframe(kf_id=kf_id, role=role, panel_id=c.PanelId.HOOD, track_id=track.track_id, f_proxy=KF_PROXY, pts=pts[KF_PROXY],
                    score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=1.9, box_src=box_src,
                    crop=crop.model_copy(update={"path": paths.rel(run_id, crop_file), "sha256": sha}), mask_png=paths.rel(run_id, mask_file),
                    moments=geometry.moments_from_mask(mask), front_sign=1)
    found = []
    if marks and (b := scene.paint(2 * KF_PROXY)) is not None:
        bbox_crop = geometry.box_src_to_crop(b, crop)
        found.append(c.Mark(mark_id=f"{kf_id}_m1", role=role, panel_id=c.PanelId.HOOD, kf_id=kf_id, source=c.MarkSource.SPARK_BOX, type=c.MarkKind.CHIP,
                            confidence=c.Confidence.HIGH, size_class=c.SizeClass.S, bbox_src=b, point_src=list(centre(b)), bbox_crop=bbox_crop,
                            point_crop=list(centre(bbox_crop))))
    for path, rows_new, model in ((paths.tracks_json(run_id), [track], c.PanelTrack), (paths.keyframes_json(run_id), [kf], c.Keyframe), (paths.marks_json(run_id), found, c.Mark)):
        old = c.read_json_list(path, model) if path.is_file() else []
        c.write_json(path, [r for r in old if r.role != role] + rows_new)
    set_privacy_lanes(run_id, role)
    assert root.is_dir()


class FakeStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self) -> Any:
        return self._gen()

    async def _gen(self) -> Any:
        for event in self.events:
            yield SimpleNamespace(model_dump=lambda event=event: copy.deepcopy(event))


def zero_events(rid: str, *, complete: bool = True) -> list[dict[str, Any]]:
    events = copy.deepcopy(ZERO)
    for e in events:
        if isinstance(e.get("response"), dict):
            e["response"]["store"] = False
            e["response"]["id"] = rid
    return events if complete else events[:2]


def text_events(rid: str, lines: list[str]) -> list[dict[str, Any]]:
    """A completed stream whose text is the given lines: one delta per line, the text in every finalizer."""
    created, in_progress, item_added, part_added, part_done, item_done, completed = zero_events(rid)
    text = "".join(lines)
    item_id = item_added["item"]["id"]
    deltas = [{"content_index": 0, "delta": line, "item_id": item_id, "logprobs": [], "output_index": 0, "sequence_number": 4 + i, "type": "response.output_text.delta"} for i, line in enumerate(lines)]
    part_done["part"]["text"] = text
    item_done["item"]["content"][0]["text"] = text
    completed["response"]["output"][0]["content"][0]["text"] = text
    return [created, in_progress, item_added, part_added, *deltas, part_done, item_done, completed]


def clip_events(rid: str, lines: list[str], frames: int) -> list[dict[str, Any]]:
    """A completed SAM video stream: text_events plus usage and the metadata echo on response.completed."""
    events = text_events(rid, lines)
    events[-1]["response"]["usage"]["video_frames_processed"] = frames
    events[-1]["response"]["metadata"] = {"include_confidence": "true", "propagation_direction": "both"}
    return events


def colour_mask(rgb: np.ndarray, kind: str) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    if kind == "red":
        return (r > 150) & (g < 110) & (b < 110)
    if kind == "yellow":
        return (r > 170) & (g > 130) & (b < 90)
    if kind == "magenta":
        return (r > 150) & (g < 100) & (b > 150)
    if kind == "green":
        return (r < 90) & (g > 180) & (b < 90)
    return (r > 215) & (g > 215) & (b > 215)


def image_answer(data_url: str, concept: str) -> list[str]:
    """The fake image check: an 11 x 11 box on the largest blob of the concept's colour (yellow plate, magenta face)."""
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB"))
    hit = colour_mask(rgb, "yellow" if concept == "license plate" else "magenta")
    count, _, stats, cents = cv2.connectedComponentsWithStats(hit.astype(np.uint8), connectivity=8)
    blobs = [i for i in range(1, count) if stats[i, 4] >= 6]
    if not blobs:
        return []
    h, w = hit.shape
    cx, cy = cents[max(blobs, key=lambda i: stats[i, 4])]
    x, y = min(max(int(round(cx)) - 5, 0), w - 11), min(max(int(round(cy)) - 5, 0), h - 11)
    return [f"<0f>0<|box;x1={x};y1={y};x2={x + 10};y2={y + 10};w={w};h={h};c=0.9|>{MASK_11}\n"]


def track_blobs(frames: list[np.ndarray], prompt: str) -> tuple[list[str], int]:
    """The fake tracker: each prompted box names a colour (red paint or a white highlight) by its pixels on the
    prompt frame, and is followed blob to blob from there, forward then backward, like propagation both."""
    k = int(re.match(r"<(\d+)f>", prompt).group(1))
    objects = re.findall(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>", prompt)
    size = frames[0].shape[1], frames[0].shape[0]
    found: dict[int, dict[str, list[int]]] = {}
    for oid, *nums in objects:
        x1, y1, x2, y2 = (int(v) for v in nums[:4])
        patch = frames[k][y1 : y2 + 1, x1 : x2 + 1]
        kind = next((k for k in ("red", "white", "green") if colour_mask(patch, k).mean() > 0.2), None)
        if kind is None:
            continue
        start = ((x1 + x2) / 2, (y1 + y2) / 2)
        for order in (range(k, len(frames)), range(k - 1, -1, -1)):
            last = centre(found[k][oid]) if order.start != k and oid in found.get(k, {}) else start
            for f in order:
                count, _, stats, cents = cv2.connectedComponentsWithStats(colour_mask(frames[f], kind).astype(np.uint8), connectivity=8)
                cands = [i for i in range(1, count) if stats[i, 4] >= 4 and math.dist(cents[i], last) < 40]
                if not cands:
                    continue
                best = min(cands, key=lambda i: math.dist(cents[i], last))
                bx, by, bw, bh = (int(v) for v in stats[best, :4])
                found.setdefault(f, {})[oid] = [bx, by, bx + bw, by + bh]
                last = (float(cents[best][0]), float(cents[best][1]))
    lines = []
    for f in [*range(k, len(frames)), *range(k - 1, -1, -1)]:
        if f in found:
            recs = [f"{oid}<|box;x1={b[0]};y1={b[1]};x2={b[2] - 1};y2={b[3] - 1};w={size[0]};h={size[1]};c=0.9|>{MASK_TOKEN}" for oid, b in sorted(found[f].items())]
            lines.append(f"<{f}f>" + ",".join(recs) + "\n")
    return lines, len(frames)


class FakeApi:
    def __init__(self, calls: list[tuple[str, str]], *, fail_images: bool = False, fail_clips: bool = False, look: bool = False) -> None:
        self.calls = calls
        self.fail_images, self.fail_clips, self.look = fail_images, fail_clips, look
        self.responses = SimpleNamespace(create=self.create)

    async def __aenter__(self) -> FakeApi:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def create(self, **body: Any) -> FakeStream:
        assert body["store"] is False and body["stream"] is True and body["model"] == "sam-3.1"
        text, media = body["input"][0]["content"][0]["text"], body["input"][0]["content"][1]
        rid = f"resp_fake_{len(self.calls):04d}"
        if media["type"] == "input_image":
            self.calls.append(("image", text))
            if self.fail_images:
                return FakeStream(zero_events(rid, complete=False))
            lines = image_answer(media["image_url"], text) if self.look else []
            return FakeStream(text_events(rid, lines) if lines else zero_events(rid))
        self.calls.append(("video", text))
        assert body["metadata"] == {"include_confidence": "true", "propagation_direction": "both"}
        if self.fail_clips:
            return FakeStream(zero_events(rid, complete=False))
        data = base64.b64decode(media["video_url"].split(",", 1)[1])
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            clip.write_bytes(data)
            frames = [rgb for _, _, rgb in video.iter_frames(clip)]
        lines, n = track_blobs(frames, text)
        return FakeStream(clip_events(rid, lines, n))


@pytest.fixture
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    monkeypatch.setenv("MODEL_API_KEY", "LLM|test-team|test-secret")
    state: dict[str, Any] = {"calls": [], "root": root, "fail_images": False, "fail_clips": False, "look": False}
    monkeypatch.setattr(sam_client, "client", lambda: FakeApi(state["calls"], fail_images=state["fail_images"], fail_clips=state["fail_clips"], look=state["look"]))
    return state


def derive(run_id: str, role: str = "pickup", th: dict[str, Any] = TH) -> list[c.MarkTrack]:
    return asyncio.run(staysput.derive_staysput(run_id, role, th=th))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kinds(calls: list[tuple[str, str]]) -> list[str]:
    return [k for k, _ in calls]


# ----------------------------------------------------------------------------------------------------------------
# derive_staysput end to end
# ----------------------------------------------------------------------------------------------------------------


def test_paint_stays_put_and_the_hood_highlight_slides(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    build_walk("sp1", "pickup", scene, src)
    rows = derive("sp1")
    by_id = {r.mark_id: r for r in rows}
    assert set(by_id) == {"pickup_hood_kf1_m1", "pickup_hood_kf1_ctrl"}
    chip_row, ctrl = by_id["pickup_hood_kf1_m1"], by_id["pickup_hood_kf1_ctrl"]
    assert chip_row.verdict == c.StaysPutVerdict.FIXED and not chip_row.is_control, chip_row
    assert chip_row.drift_rms is not None and chip_row.drift_rms < 0.04 and chip_row.b_pass and not chip_row.panned
    assert ctrl.is_control and ctrl.verdict == c.StaysPutVerdict.SLIDING, ctrl
    assert ctrl.sign_agreement is not None and ctrl.sign_agreement > 0.8 and ctrl.pairs_n >= 8
    assert ctrl.highlight_cap and not chip_row.highlight_cap  # the photometric rule agrees the glint is a highlight
    for r in rows:
        assert r.clip is not None and (r.clip.w, r.clip.h, r.clip.frames, r.clip.f0_src) == (CLIP, CLIP, 91, 29)
        assert (paths.run_dir("sp1") / r.clip.path).is_file() and r.clip.path.startswith("clips/pickup/")
        assert r.request_id and r.prompt_text.startswith("<45f>1<|box;") and len(r.track) >= 80
    # privacy order: each clip's two image checks were asked before its clip request, and every call has a ledger row
    calls = offline["calls"]
    assert kinds(calls).count("video") == 2 and kinds(calls).count("image") == 4
    for i, (kind, _) in enumerate(calls):
        if kind == "video":
            assert sum(1 for k, _ in calls[:i] if k == "image") >= 2 * (1 + sum(1 for k, _ in calls[:i] if k == "video"))
    ledger_kinds = [r["kind"] for r in ledger.rows()]
    assert ledger_kinds.count("sam_video") == 2 and ledger_kinds.count("sam_image") == 4
    assert all(r["note"].startswith("sp_pickup_hood_kf1_sp") for r in ledger.rows() if r["kind"] == "sam_video")
    stored = c.read_json_list(paths.mark_tracks_json("sp1"), c.MarkTrack)
    assert [r.to_json() for r in stored] == [r.to_json() for r in rows]
    plans = staysput._read_plans("sp1", "pickup")
    assert plans is not None and plans.complete and staysput.role_done("sp1", "pickup") and not staysput.role_done("sp1", "return")
    assert all(p.status == "answered" and p.check is not None and len(p.check.request_ids) == 2 for p in plans.clips)
    assert len(staysput.request_ids("sp1")) == 6  # 4 checks and 2 lanes, for the run-end deletion list
    assert not list(paths.clips_dir("sp1", "pickup").glob(".*")) and not list(paths.derived_dir("sp1").glob(".*"))  # no temp file left


def test_a_24_fps_walk_gets_three_second_clips(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving24"]
    build_walk("sp24", "pickup", scene, src, fps=24)
    rows = {r.mark_id: r for r in derive("sp24")}
    for r in rows.values():
        assert r.clip is not None and (r.clip.frames, r.clip.f0_src) == (73, 38) and r.prompt_text.startswith("<36f>1<|box;")
    assert rows["pickup_hood_kf1_m1"].verdict == c.StaysPutVerdict.FIXED
    ctrl = rows["pickup_hood_kf1_ctrl"]
    assert ctrl.verdict == c.StaysPutVerdict.SLIDING and ctrl.pairs_n == 18  # 73 frames in 4-frame pairs


def test_rebuild_is_byte_identical_without_the_original_or_the_network(offline: dict[str, Any], walks: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scene, src = walks["moving"]
    build_walk("sp2", "pickup", scene, src)
    derive("sp2")
    live = paths.mark_tracks_json("sp2").read_bytes()
    paths.mark_tracks_json("sp2").unlink()
    n_calls = len(offline["calls"])
    rebuilt = staysput.rebuild_staysput("sp2", th=TH)
    assert paths.mark_tracks_json("sp2").read_bytes() == live and len(rebuilt) == 2
    # in a copy with no quarantine (what walkaround rebuild and verify make), under another runs root
    other = tmp_path / "other_runs"
    shutil.copytree(paths.run_dir("sp2"), other / "sp2", ignore=shutil.ignore_patterns("quarantine"))
    (other / "sp2" / "derived" / "mark_tracks.json").unlink()
    monkeypatch.setattr(paths, "RUNS_DIR", other)
    staysput.rebuild_staysput("sp2", th=TH)
    assert (other / "sp2" / "derived" / "mark_tracks.json").read_bytes() == live
    assert not (other / "sp2" / "quarantine").exists() and len(offline["calls"]) == n_calls


def test_a_resumed_walk_asks_nothing_twice(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    build_walk("sp3", "pickup", scene, src)
    derive("sp3")
    first = paths.mark_tracks_json("sp3").read_bytes()
    n_calls, n_rows = len(offline["calls"]), len(ledger.rows())
    derive("sp3")
    assert len(offline["calls"]) == n_calls and len(ledger.rows()) == n_rows
    assert paths.mark_tracks_json("sp3").read_bytes() == first


def test_a_byte_identical_return_walk_reuses_every_answer(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    build_walk("sp4", "pickup", scene, src)
    build_walk("sp4", "return", scene, src)
    pickup = derive("sp4", "pickup")
    n_calls = len(offline["calls"])
    back = derive("sp4", "return")
    assert len(offline["calls"]) == n_calls  # same pixels, same clip bytes, same prompt: nothing asked
    assert [r.mark_id.replace("return_", "pickup_") for r in back] == [r.mark_id for r in pickup]
    for a, b in zip(pickup, back):
        assert (a.verdict, a.drift_rms, a.b1, a.b2, a.sign_agreement, a.request_id) == (b.verdict, b.drift_rms, b.b1, b.b2, b.sign_agreement, b.request_id)
        assert a.clip is not None and b.clip is not None and a.clip.sha256 == b.clip.sha256
    plans = staysput._read_plans("sp4", "return")
    assert plans is not None and all(p.lane_reused_from and p.check and p.check.reused_from for p in plans.clips)
    summary = c.read_json(paths.lane_summary_path("sp4", "return", plans.clips[0].lane_id), c.LaneRecord)
    assert summary.dollars == 0.0 and summary.diagnostics[-1].code == staysput.REUSED_CODE
    all_rows = c.read_json_list(paths.mark_tracks_json("sp4"), c.MarkTrack)
    assert [r.role for r in all_rows] == ["pickup", "pickup", "return", "return"]
    assert len(staysput.request_ids("sp4")) == 6  # the return walk added no request of its own


def test_a_panned_walk_is_measured_and_flagged(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["panned"]
    build_walk("sp5", "pickup", scene, src)
    rows = derive("sp5")
    assert [r.mark_id for r in rows] == ["pickup_hood_kf1_m1"]  # no glare on this hood: no control clip
    row = rows[0]
    assert row.panned and row.b_pass and row.b1 is not None and row.b1 < 0.01
    assert row.verdict == c.StaysPutVerdict.FIXED, row


def test_a_mark_the_tracker_loses_is_no_track(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["lost"]
    build_walk("sp6", "pickup", scene, src)
    rows = derive("sp6")
    assert [r.mark_id for r in rows] == ["pickup_hood_kf1_m1"]
    assert rows[0].verdict == c.StaysPutVerdict.NO_TRACK and 30 <= len(rows[0].track) <= 45
    assert not rows[0].curvature_cap and not rows[0].highlight_cap  # read from the keyframe: paint mid-panel, not bright


def test_a_failed_image_check_cuts_no_clip(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    build_walk("sp7", "pickup", scene, src)
    offline["fail_images"] = True
    rows = derive("sp7")
    assert {r.verdict for r in rows} == {c.StaysPutVerdict.NOT_RUN} and all(r.clip is None and r.request_id is None for r in rows)
    assert not list(paths.clips_dir("sp7", "pickup").glob("*.mp4"))
    assert "video" not in kinds(offline["calls"])
    plans = staysput._read_plans("sp7", "pickup")
    assert plans is not None and all(p.status == "failed" and p.error.startswith("image check did not complete") for p in plans.clips)
    offline["fail_images"] = False
    rows = derive("sp7")  # the next attempt asks again and completes
    assert {r.verdict for r in rows} == {c.StaysPutVerdict.FIXED, c.StaysPutVerdict.SLIDING}


def test_a_lane_that_never_completes_keeps_the_keyframe_caps(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    build_walk("sp8", "pickup", scene, src)
    offline["fail_clips"] = True
    rows = {r.mark_id: r for r in derive("sp8")}
    assert {r.verdict for r in rows.values()} == {c.StaysPutVerdict.NOT_RUN}
    assert all(r.clip is not None for r in rows.values())  # the blurred clip exists; the lane did not answer
    assert rows["pickup_hood_kf1_ctrl"].highlight_cap and not rows["pickup_hood_kf1_m1"].highlight_cap  # no model needed
    plans = staysput._read_plans("sp8", "pickup")
    assert plans is not None and all(p.status == "failed" for p in plans.clips)
    asked = staysput.request_ids("sp8")
    assert len(asked) == 6  # 4 checks, 2 lanes that never completed
    offline["fail_clips"] = False
    derive("sp8")  # asked again: the replaced plans' ids stay on the deletion list
    again = staysput.request_ids("sp8")
    assert len(again) == 12 and again[:6] == asked


def test_clips_held_back_at_the_deadline_wait_for_a_resume(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    build_walk("sp18", "pickup", scene, src)
    rows = asyncio.run(staysput.derive_staysput("sp18", "pickup", th=TH, stop=lambda: True))
    assert offline["calls"] == [] and {r.verdict for r in rows} == {c.StaysPutVerdict.NOT_RUN}
    plans = staysput._read_plans("sp18", "pickup")
    assert plans is not None and all(p.status == "planned" for p in plans.clips) and not staysput.role_done("sp18", "pickup")
    rows = derive("sp18")
    assert staysput.role_done("sp18", "pickup") and {r.verdict for r in rows} == {c.StaysPutVerdict.FIXED, c.StaysPutVerdict.SLIDING}


def test_a_mark_over_the_clip_cap_gets_a_not_run_row(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    build_walk("sp11", "pickup", scene, src)
    th = copy.deepcopy(TH)
    th["staysput"]["max_clips_per_walk"] = 1  # the control ranks first
    rows = derive("sp11", th=th)
    assert [(r.mark_id, r.verdict) for r in rows] == [("pickup_hood_kf1_ctrl", c.StaysPutVerdict.SLIDING), ("pickup_hood_kf1_m1", c.StaysPutVerdict.NOT_RUN)]
    assert rows[1].clip is None and rows[1].request_id is None and not rows[1].highlight_cap
    plans = staysput._read_plans("sp11", "pickup")
    assert plans is not None and [(s.mark_id, s.reason) for s in plans.unclipped] == [("pickup_hood_kf1_m1", "over the 1-clip cap for one walk")]
    live = paths.mark_tracks_json("sp11").read_bytes()
    staysput.rebuild_staysput("sp11", th=th)
    assert paths.mark_tracks_json("sp11").read_bytes() == live


def test_a_walk_whose_face_lane_failed_gets_no_clip(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    build_walk("sp12", "pickup", scene, src)
    set_privacy_lanes("sp12", "pickup", face=c.LaneStatus.FAILED)
    rows = derive("sp12")
    assert offline["calls"] == [] and not list(paths.clips_dir("sp12", "pickup").glob("*.mp4"))
    assert [(r.mark_id, r.verdict, r.clip) for r in rows] == [("pickup_hood_kf1_m1", c.StaysPutVerdict.NOT_RUN, None)]
    plans = staysput._read_plans("sp12", "pickup")
    assert plans is not None and plans.complete and plans.clips == []
    assert plans.unclipped[0].reason == "its face lane is failed, so a clip cannot be blurred"
    assert plans.skipped == ["pickup_hood_kf1_ctrl, the reflection control: its face lane is failed, so a clip cannot be blurred"]
    build_walk("sp13", "pickup", scene, src)
    set_privacy_lanes("sp13", "pickup", license_plate=None)  # a lane with no summary is unknown too
    assert {r.verdict for r in derive("sp13")} == {c.StaysPutVerdict.NOT_RUN} and offline["calls"] == []


def test_clips_a_later_derive_drops_are_removed(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    build_walk("sp14", "pickup", scene, src)
    derive("sp14")
    assert len(list(paths.clips_dir("sp14", "pickup").glob("*.mp4"))) == 2
    asked = staysput.request_ids("sp14")
    set_privacy_lanes("sp14", "pickup", face=c.LaneStatus.INCOMPLETE)  # as if the face lane were found wanting later
    rows = derive("sp14")
    assert not list(paths.clips_dir("sp14", "pickup").glob("*.mp4"))  # nothing served that the plan no longer vouches for
    assert all(r.clip is None for r in rows) and staysput.request_ids("sp14") == asked


def test_a_plate_with_a_track_is_pixelated_on_every_clip_frame(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["plate"]
    build_walk("sp15", "pickup", scene, src)
    set_privacy_lanes("sp15", "pickup", license_plate=c.LaneStatus.COMPLETED)
    add_privacy_track("sp15", "pickup", c.Noun.LICENSE_PLATE, scene.plate)
    offline["look"] = True
    rows = derive("sp15")
    assert [(r.mark_id, r.verdict) for r in rows] == [("pickup_hood_kf1_m1", c.StaysPutVerdict.FIXED)]
    plan = staysput._read_plans("sp15", "pickup").clips[0]
    assert plan.status == "answered" and len(plan.check.plate_boxes) == 1 and plan.check.face_boxes == []
    clip_frames = [rgb for _, _, rgb in video.iter_frames(paths.run_dir("sp15") / plan.clip.path)]
    src_frames = {i: rgb for i, _, rgb in video.iter_frames(src)}
    plate_ncc, texture_ncc = [], []
    for f, win in enumerate(plan.windows_src):
        clean = staysput._cut_window(src_frames[plan.f0_src + f], win, plan.size)
        x1, y1, x2, y2 = (int(round(v)) for v in staysput._to_clip(scene.plate(plan.f0_src + f), win, plan.size))
        plate_ncc.append(ncc(clip_frames[f][y1 + 2 : y2 - 2, x1 + 2 : x2 - 2], clean[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2]))
        texture_ncc.append(ncc(clip_frames[f][10:60, 10:60], clean[10:60, 10:60]))  # panel paint, never blurred
    assert len(plate_ncc) == 91 and max(plate_ncc) < 0.3, plate_ncc
    assert min(texture_ncc) > 0.6, texture_ncc  # the measure itself reads unblurred texture as readable


def test_a_face_only_the_check_finds_refuses_the_clip(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["face"]
    build_walk("sp16", "pickup", scene, src)  # the face lane answered with nothing: it missed this face
    offline["look"] = True
    rows = derive("sp16")
    assert [(r.mark_id, r.verdict, r.clip) for r in rows] == [("pickup_hood_kf1_m1", c.StaysPutVerdict.NOT_RUN, None)]
    assert kinds(offline["calls"]) == ["image", "image"] and not list(paths.clips_dir("sp16", "pickup").glob("*.mp4"))
    plan = staysput._read_plans("sp16", "pickup").clips[0]
    assert plan.status == "refused" and plan.clip is None and len(plan.check.face_boxes) == 1 and "face" in plan.error
    derive("sp16")
    assert len(offline["calls"]) == 2  # a refusal is not asked again
    # the same face with a face track: the clip is cut and the face is pixelated on every frame it is in the window
    build_walk("sp17", "pickup", scene, src)
    set_privacy_lanes("sp17", "pickup", face=c.LaneStatus.COMPLETED)
    add_privacy_track("sp17", "pickup", c.Noun.FACE, lambda i: list(FACE_BOX))
    rows = derive("sp17")
    assert rows[0].verdict == c.StaysPutVerdict.FIXED
    plan = staysput._read_plans("sp17", "pickup").clips[0]
    clip_frames = [rgb for _, _, rgb in video.iter_frames(paths.run_dir("sp17") / plan.clip.path)]
    src_frames = {i: rgb for i, _, rgb in video.iter_frames(src)}
    scores = []
    for f, win in enumerate(plan.windows_src):
        x1, y1, x2, y2 = (int(round(v)) for v in staysput._to_clip(FACE_BOX, win, plan.size))
        if x1 < 2 or y1 < 2 or x2 > plan.size - 2 or y2 > plan.size - 2:
            continue
        clean = staysput._cut_window(src_frames[plan.f0_src + f], win, plan.size)
        scores.append(ncc(clip_frames[f][y1 + 2 : y2 - 2, x1 + 2 : x2 - 2], clean[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2]))
    assert len(scores) >= 40 and max(scores) < 0.3, scores


def test_a_walk_with_nothing_to_track_writes_an_empty_role(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["panned"]  # no glare, and no mark
    build_walk("sp9", "pickup", scene, src, marks=False)
    assert derive("sp9") == [] and offline["calls"] == []
    assert c.read_json_list(paths.mark_tracks_json("sp9"), c.MarkTrack) == [] and staysput.role_done("sp9", "pickup")
    assert staysput.rebuild_staysput("sp9", th=TH) == []
    assert staysput.rebuild_staysput("never_ran", th=TH) == [] and not (paths.run_dir("never_ran") / "derived" / "mark_tracks.json").exists()


def test_windows_follow_the_panel_on_the_real_timeline(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    build_walk("sp10", "pickup", scene, src)
    kf = c.read_json_list(paths.keyframes_json("sp10"), c.Keyframe)[0]
    track = c.read_json_list(paths.tracks_json("sp10"), c.PanelTrack)[0]
    mark = c.read_json_list(paths.marks_json("sp10"), c.Mark)[0]
    times = video.frame_times(src)
    [(ids, windows)] = staysput.windows_for([mark], kf, track, times, th=TH, src_size=SRC, proxy_size=PROXY)
    assert ids == [mark.mark_id] and len(windows) == 91
    for i, w in enumerate(windows):  # the paint sits at the window centre on every clip frame
        b = scene.paint(29 + i)
        assert centre(w) == pytest.approx(centre(b), abs=1e-3) and w[2] - w[0] == pytest.approx(120.0)


def test_the_blurred_clip_and_its_lane_are_served_and_the_plan_is_not() -> None:
    run = c.RunManifest(run_id="r1", created_at="2026-10-31T20:00:00Z")
    assert privacy.served_file_allowed(run, "clips/pickup/pickup_hood_kf1_sp01.mp4", run_owner=False)
    assert privacy.served_file_allowed(run, "sam/pickup/sp_pickup_hood_kf1_sp01.ndjson.gz", run_owner=False)
    assert privacy.served_file_allowed(run, "derived/mark_tracks.json", run_owner=False)
    assert not privacy.served_file_allowed(run, "clips/pickup/staysput.json", run_owner=True)
    assert not privacy.served_file_allowed(run, "clips/pickup/.staysput.json.123.456.tmp", run_owner=True)


# ----------------------------------------------------------------------------------------------------------------
# Stays Put by neighbours (docs/STAYS_PUT_ANCHORS.md)
# ----------------------------------------------------------------------------------------------------------------

NEIGHBOUR_FIXTURES = ROOT / "fixtures" / "staysput"  # boxes only, from the anchor_exp_1031 lanes (uCO3D CC BY 4.0, Pexels License)
S550_PARTS = [("3", "door handle", "front"), ("4", "door handle", "rear"), ("5", "side mirror", "m"), ("6", "wheel", "rear"), ("7", "tail light", "t"), ("8", "wheel", "front")]


def stored_lane(name: str) -> list[c.IndexFrame]:
    return c.read_json_list(NEIGHBOUR_FIXTURES / f"{name}.index.json", c.IndexFrame)


def test_the_neighbour_test_gives_the_leads_numbers_on_the_two_stored_clips() -> None:
    """docs/STAYS_PUT_ANCHORS.md: paint on the pair C trunk lid 0.6 and 0.8x its neighbours, the reflection on the black
    S550's rear door 4.8 and 3.8x, the crease on its rear quarter 0.7 and 0.3x (over the clip, within 0.5 s)."""
    trunk = staysput.neighbour_test(stored_lane("neighbours_trunk"), "1", [(str(i), "mark", f"a{i}") for i in range(2, 7)], n_frames=91, prompt_frame=45, fps=1479 / 50, th=TH)
    assert trunk.verdict == c.StaysPutVerdict.FIXED and trunk.used == 5 and trunk.near_frames == 15
    assert (round(trunk.ratio_full, 1), round(trunk.ratio_near, 1)) == (0.6, 0.8)
    assert (round(trunk.mark_near_px, 1), [nb.frames for nb in trunk.neighbours]) == (14.1, [91, 91, 83, 91, 91])
    reflection = staysput.neighbour_test(stored_lane("neighbours_s550"), "1", S550_PARTS, n_frames=90, prompt_frame=45, fps=30.0, th=TH)
    crease = staysput.neighbour_test(stored_lane("neighbours_s550"), "2", S550_PARTS, n_frames=90, prompt_frame=45, fps=30.0, th=TH)
    assert reflection.verdict == c.StaysPutVerdict.SLIDING and (round(reflection.ratio_full, 1), round(reflection.ratio_near, 1)) == (4.8, 3.8)
    assert (round(reflection.mark_full_px, 1), round(reflection.mark_near_px, 1), round(reflection.neighbours_full_px, 2)) == (27.4, 7.9, 5.75)
    assert crease.verdict == c.StaysPutVerdict.FIXED and (round(crease.ratio_full, 1), round(crease.ratio_near, 1)) == (0.7, 0.3)
    assert all(nb.used and nb.frames == 90 and len(nb.centres) == 90 for nb in reflection.neighbours)
    assert [round(nb.residual_full_px, 1) for nb in reflection.neighbours] == [1.8, 4.4, 1.8, 9.1, 10.9, 7.1]  # handles and mirror about 1x


def camera(f: int, x0: float, y0: float) -> tuple[float, float]:
    """A camera that rotates, zooms and pans a little each frame (an affine of the prompt frame 45)."""
    d = f - 45
    s, a = 1.0 + 0.004 * d, 0.003 * d
    x, y = x0 - 500.0, y0 - 400.0
    return (500.0 + s * (math.cos(a) * x - math.sin(a) * y) + 3.0 * d, 400.0 + s * (math.sin(a) * x + math.cos(a) * y) + 1.5 * d)


def synthetic_lane(objects: dict[str, Any], *, n: int = 91, drop: dict[str, set[int]] | None = None, jitter: float = 0.8, seed: int = 5) -> list[c.IndexFrame]:
    """Boxes of 9 px around each object's centre per frame, with SAM-like jitter, rounded to whole pixels."""
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(n):
        objs = []
        for oid, at in objects.items():
            x, y = at(f)
            x, y = x + rng.normal(0, jitter), y + rng.normal(0, jitter)
            if drop and f in drop.get(oid, set()):
                continue
            objs.append(c.IndexObject(id=oid, box=[int(round(x - 4.5)), int(round(y - 4.5)), int(round(x + 4.5)), int(round(y + 4.5))], c=0.9))
        frames.append(c.IndexFrame(frame=f, objects=objs))
    return frames


SPOTS = [(380.0, 300.0), (640.0, 320.0), (420.0, 520.0), (610.0, 500.0), (300.0, 420.0), (700.0, 410.0)]


def rigid_scene() -> dict[str, Any]:
    objects: dict[str, Any] = {"paint": lambda f: camera(f, 520.0, 430.0), "glint": lambda f: camera(f, 480.0 + 1.2 * (f - 45), 380.0)}
    for i, (x, y) in enumerate(SPOTS):
        objects[f"n{i}"] = lambda f, x=x, y=y: camera(f, x, y)
    return objects


def run_neighbours(frames: list[c.IndexFrame], oid: str, ids: list[str], **kw: Any) -> c.NeighbourTest:
    return staysput.neighbour_test(frames, oid, [(i, "wheel", i) for i in ids], n_frames=kw.pop("n", 91), prompt_frame=45, fps=30.0, th=kw.pop("th", TH), **kw)


def test_paint_moves_with_its_neighbours_and_a_glint_slides_against_them() -> None:
    frames = synthetic_lane(rigid_scene())
    ids = [f"n{i}" for i in range(6)]
    paint, glint = run_neighbours(frames, "paint", ids), run_neighbours(frames, "glint", ids)
    assert paint.verdict == c.StaysPutVerdict.FIXED and paint.ratio_full < 1.5 and paint.ratio_near < 1.5 and paint.used == 6, paint
    assert glint.verdict == c.StaysPutVerdict.SLIDING and glint.ratio_full > 2.5 and glint.ratio_near > 2.5, glint
    assert glint.ratio_full > glint.ratio_near  # the slide adds up over the whole clip
    assert run_neighbours(frames, "glint", ids).to_json() == glint.to_json()  # pure


def test_fewer_than_four_neighbours_through_the_clip_is_inconclusive() -> None:
    frames = synthetic_lane(rigid_scene(), drop={"n3": set(range(0, 30)), "n4": set(range(60, 91)), "n5": {45}})
    nt = run_neighbours(frames, "glint", [f"n{i}" for i in range(6)])
    assert nt.verdict == c.StaysPutVerdict.INCONCLUSIVE and nt.used == 3 and nt.ratio_full is None
    assert nt.reason == "3 neighbours tracked through the clip, 4 needed"
    assert [nb.used for nb in nt.neighbours] == [True, True, True, False, False, False]  # 61 frames, 60 frames, no prompt frame
    assert [nb.frames for nb in nt.neighbours] == [91, 91, 91, 61, 60, 90]


def test_a_neighbour_on_most_of_the_clip_still_counts_and_a_lost_mark_is_no_track() -> None:
    drop = set(range(0, 18))  # 73 of 91 frames: just over 80%
    frames = synthetic_lane(rigid_scene(), drop={"n0": drop, "paint": set(range(0, 20))})
    nt = run_neighbours(frames, "glint", [f"n{i}" for i in range(6)])
    assert nt.used == 6 and nt.verdict == c.StaysPutVerdict.SLIDING
    lost = run_neighbours(frames, "paint", [f"n{i}" for i in range(6)])
    assert lost.verdict == c.StaysPutVerdict.NO_TRACK and lost.ratio_full is None and lost.used == 6


def test_neighbours_that_do_not_move_keep_the_ratio_finite() -> None:
    still: dict[str, Any] = {"paint": lambda f: (520.0, 430.0)}
    still.update({f"n{i}": (lambda f, x=x, y=y: (x, y)) for i, (x, y) in enumerate(SPOTS)})
    nt = run_neighbours(synthetic_lane(still, jitter=0.0), "paint", [f"n{i}" for i in range(6)])
    assert nt.verdict == c.StaysPutVerdict.FIXED and nt.ratio_full == pytest.approx(0.0, abs=1e-9) and nt.neighbours_full_px == pytest.approx(0.0, abs=1e-9)


def test_neighbours_on_one_line_fix_no_affine() -> None:
    line = {"paint": lambda f: camera(f, 520.0, 430.0)}
    line.update({f"n{i}": (lambda f, x=x: camera(f, x, 300.0)) for i, x in enumerate((300.0, 400.0, 500.0, 600.0, 700.0))})
    nt = run_neighbours(synthetic_lane(line, jitter=0.0), "paint", [f"n{i}" for i in range(5)])
    assert nt.verdict == c.StaysPutVerdict.INCONCLUSIVE and nt.ratio_full is None and nt.reason == "too few frames with three neighbours in view"


def part(kind: str, ref: str, box: list[float]) -> Any:
    return staysput._Cand(kind, ref, (box[0], box[1], box[2], box[3]), staysput._long(box))


def row_of_parts(*dx: float) -> list[Any]:
    return [part("wheel", f"w{i}", [500 + d - 5, 495, 500 + d + 5, 505]) for i, d in enumerate(dx)]


PANEL = [0.0, 0.0, 2000.0, 1000.0]
TH_1008 = c.load_thresholds()  # the product's 1008 px clip: the mark floor below is in its pixels


def test_the_window_takes_what_it_holds_then_grows_to_six_neighbours() -> None:
    # needs: 2 x (|dx| + 5) x 1.15 = 126.5, 149.5, 287.5, 356.5 (inside the 400 px base), 471.5, 609.5, 701.5, 931.5
    cands = row_of_parts(50, -60, 120, -150, 200, -260, 300, 400)
    side, chosen = staysput.pick_neighbours([[490, 490, 510, 510]], ["m"], (500.0, 500.0), 400.0, PANEL, cands, TH_1008)
    assert [c_.ref for c_ in chosen] == ["w0", "w1", "w2", "w3", "w4", "w5"] and side == pytest.approx(609.5)
    # a mark of 8 px caps the window at 1008 x 8 / 16 = 504 px, so it stops at five
    side, chosen = staysput.pick_neighbours([[496, 496, 504, 504]], ["m"], (500.0, 500.0), 400.0, PANEL, cands, TH_1008)
    assert len(chosen) == 5 and side == pytest.approx(471.5)
    # the panel caps it too: 1.5 x 300 px
    side, chosen = staysput.pick_neighbours([[490, 490, 510, 510]], ["m"], (500.0, 500.0), 400.0, [0, 0, 300, 200], cands, TH_1008)
    assert len(chosen) == 4 and side == 400.0  # 450 px holds no fifth: the next needs 471.5


def test_a_window_that_cannot_reach_four_neighbours_stays_sharp() -> None:
    cands = row_of_parts(50, 300, -330)  # one inside the base window, two far out
    side, chosen = staysput.pick_neighbours([[490, 490, 510, 510]], ["m"], (500.0, 500.0), 400.0, PANEL, cands, TH_1008)
    assert side == 400.0 and [c_.ref for c_ in chosen] == ["w0"]
    # the clip's own marks are neighbours of each other: with three marks of its own, two more reach four
    side, chosen = staysput.pick_neighbours([[490, 490, 510, 510]] * 3, ["a", "b", "c"], (500.0, 500.0), 400.0, PANEL, cands, TH_1008, member_rigid=3)
    assert side == pytest.approx(2 * 335 * 1.15) and len(chosen) == 3


def test_parts_that_repeat_or_are_too_big_are_skipped_and_the_object_cap_holds() -> None:
    cands = row_of_parts(50, -60, 120, -150)
    cands.append(part("rear window", "dup", [546, 495, 556, 505]))  # the first wheel seen again by another lane
    cands.append(part("windshield", "big", [300, 300, 700, 700]))  # 400 px: longer than half any window it fits in
    cands.append(part("mark", "m9", [520, 520, 540, 540]))  # a mark is never deduped against parts
    side, chosen = staysput.pick_neighbours([[490, 490, 510, 510]], ["m"], (500.0, 500.0), 400.0, PANEL, cands, TH_1008)
    assert "dup" not in [c_.ref for c_ in chosen] and "big" not in [c_.ref for c_ in chosen] and "m9" in [c_.ref for c_ in chosen]
    assert chosen[0].ref == "m9"  # nearest first
    side, chosen = staysput.pick_neighbours([[490, 490, 510, 510]] * 14, [f"k{i}" for i in range(14)], (500.0, 500.0), 400.0, PANEL, cands, TH_1008)
    assert len(chosen) == 2  # 14 marks leave room for 2 objects of the 16
    th = copy.deepcopy(TH_1008)
    th["staysput"]["neighbour_window_max_growth"] = 1.0  # an explicit growth cap keeps the base window
    side, chosen = staysput.pick_neighbours([[490, 490, 510, 510]], ["m"], (500.0, 500.0), 400.0, PANEL, row_of_parts(50, -60, 120, -150, 200, -260), th)
    assert side == 400.0 and len(chosen) == 4


def neighbour_kf(**kw: Any) -> c.Keyframe:
    base = dict(kf_id="k", role="pickup", panel_id=c.PanelId.LEFT_FRONT_DOOR, track_id="pickup_car_door_00", f_proxy=10, pts=0.6667,
                score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=1.5, box_src=[100.0, 100.0, 500.0, 360.0])
    return c.Keyframe(**{**base, **kw})


def lane_track(track_id: str, noun: c.Noun, box_proxy: list[float], *, frame: int = 10, border: float = 0.0) -> c.PanelTrack:
    row = c.FrameObs(frame=frame, pts=frame / 15, box=box_proxy, box_area_frac=0.01, in_car_box=True, c=0.9, border_frac=border)
    return c.PanelTrack(track_id=track_id, role="pickup", noun=noun, sam_object_ids=["0"], fragments=[c.TrackFragment(object_id="0", first_frame=frame, last_frame=frame)], per_frame=[row])


def test_neighbour_candidates_are_rigid_marks_lane_parts_and_the_handle() -> None:
    marks = [
        chip("k_m1", bbox_src=[200.0, 200.0, 220.0, 214.0]).model_copy(update={"panel_id": c.PanelId.LEFT_FRONT_DOOR}),
        chip("k_m2", bbox_src=[300.0, 200.0, 320.0, 214.0]).model_copy(update={"type": c.MarkKind.REFLECTION}),
        chip("k_m3", bbox_src=[300.0, 250.0, 320.0, 264.0]).model_copy(update={"type": c.MarkKind.UNSURE}),
        chip("k_m4", bbox_src=[300.0, 300.0, 303.0, 302.0]),  # under the 2 px floor of a 640 px walk? no: 3 px long side passes
        chip("other_m1", bbox_src=[300.0, 300.0, 320.0, 314.0]).model_copy(update={"kf_id": "other"}),
    ]
    tracks = [
        lane_track("pickup_wheel_00", c.Noun.WHEEL, [50.0, 160.0, 90.0, 200.0]),
        lane_track("pickup_side_mirror_00", c.Noun.SIDE_MIRROR, [30.0, 40.0, 45.0, 52.0], border=0.2),  # cut by the frame edge
        lane_track("pickup_car_door_00", c.Noun.CAR_DOOR, [50.0, 50.0, 250.0, 180.0]),  # the keyframe's own panel
        lane_track("pickup_car_door_01", c.Noun.CAR_DOOR, [250.0, 50.0, 300.0, 180.0]),  # a panel is never a neighbour
        lane_track("pickup_face_00", c.Noun.FACE, [10.0, 10.0, 20.0, 20.0]),
        lane_track("pickup_tail_light_00", c.Noun.TAIL_LIGHT, [280.0, 60.0, 300.0, 80.0], frame=11),  # not seen on the keyframe
    ]
    kf = neighbour_kf(handle=c.HandleRecord(point_src=[400.0, 200.0], c=0.9))
    cands = staysput.neighbour_candidates(marks, tracks, kf, th=TH, src_size=SRC, proxy_size=PROXY)
    assert [(x.kind, x.ref) for x in cands] == [("mark", "k_m1"), ("mark", "k_m4"), ("wheel", "pickup_wheel_00"), ("door handle", "k_handle")]
    assert cands[2].box == (100.0, 320.0, 180.0, 400.0)  # proxy to source
    assert cands[3].box == pytest.approx((368.0, 192.0, 432.0, 208.0))  # 0.08 and 0.02 of the 400 px door around the point


def test_a_group_whose_marks_ride_as_neighbours_gets_no_clip_of_its_own() -> None:
    marks = [chip(f"k_m{i}", bbox_src=[x, 200.0, x + 12.0, 212.0]) for i, x in enumerate((100.0, 160.0, 300.0, 350.0), start=1)]
    kf = neighbour_kf(panel_id=c.PanelId.HOOD, track_id="pickup_hood_00", box_src=[50.0, 100.0, 550.0, 340.0])
    parts = [lane_track("pickup_wheel_00", c.Noun.WHEEL, [60.0, 125.0, 70.0, 135.0]), lane_track("pickup_side_mirror_00", c.Noun.SIDE_MIRROR, [40.0, 75.0, 50.0, 82.5])]
    walk = staysput._Walk(role="pickup", src=Path("unused"), src_size=SRC, proxy_size=PROXY, proxy_fps=15.0, proxy_pts=[], tracks={t.track_id: t for t in parts},
                          privacy_tracks=[], keyframes={"k": kf}, marks=marks)
    th = copy.deepcopy(TH_1008)
    th["staysput"]["window_min_px"] = 120
    capped = copy.deepcopy(th)  # thresholds.json caps the growth at 2x the base side (integrator 10/31)
    th["staysput"]["neighbour_window_max_growth"] = 1000.0  # staysput.DEFAULTS: no cap, the lane's own geometry
    items = [staysput._item(m, staysput._mark_box(m, 2.0)[0]) for m in marks]
    groups = staysput._group(items, staysput._base_side(kf, th), th)
    assert [[it.mark_id for it in g[0]] for g in groups] == [["k_m1", "k_m2"], ["k_m3", "k_m4"]]
    base = staysput._base_side(kf, th)
    assert all(side <= 2.0 * base + 1e-6 for _g, _members, _centre, side, _near in staysput._neighbour_layout(walk, kf, groups, capped))
    layout = staysput._neighbour_layout(walk, kf, groups, th)
    assert len(layout) == 1  # the first clip grew to hold k_m3 and k_m4 as neighbours, so they are tested there
    g, members, _centre, side, near = layout[0]
    assert g == 1 and [it.mark_id for it in members] == ["k_m1", "k_m2", "k_m3", "k_m4"]
    assert [x.ref for x in near] == ["pickup_side_mirror_00", "pickup_wheel_00"] and side == pytest.approx(2 * 226 * 1.15)
    th["staysput"]["neighbours_min"] = 6  # four neighbours are then too few: the window stays at its base and both groups keep a clip
    assert [len(x[1]) for x in staysput._neighbour_layout(walk, kf, groups, th)] == [2, 2]


def add_part_tracks(run_id: str, role: str, scene: Scene) -> None:
    """The scene's green squares as lane tracks (wheel, side mirror, headlight, tail light in turn), proxy pixels."""
    pts = json.loads(paths.pts_map_path(run_id, role).read_text())
    nouns = [c.Noun.WHEEL, c.Noun.SIDE_MIRROR, c.Noun.HEADLIGHT, c.Noun.TAIL_LIGHT]
    tracks = c.read_json_list(paths.tracks_json(run_id), c.PanelTrack)
    for j in range(len(scene.parts(0))):
        rows = [c.FrameObs(frame=f, pts=pts[f], box=[v / 2 for v in scene.parts(2 * f)[j]], box_area_frac=0.001, in_car_box=True, c=0.9) for f in range(len(pts))]
        tracks.append(c.PanelTrack(track_id=f"{role}_part_{j:02d}", role=role, noun=nouns[j % 4], sam_object_ids=[str(j)],
                                   fragments=[c.TrackFragment(object_id=str(j), first_frame=0, last_frame=len(pts) - 1)], c_present_share=1.0, per_frame=rows))
    c.write_json(paths.tracks_json(run_id), tracks)


def test_derive_tracks_neighbours_in_the_same_request_and_rebuilds_them(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    from walkaround import sheet

    scene, src = walks["parts"]
    build_walk("spn1", "pickup", scene, src)
    add_part_tracks("spn1", "pickup", scene)
    rows = {r.mark_id: r for r in derive("spn1")}
    plans = staysput._read_plans("spn1", "pickup")
    assert plans is not None and len(plans.clips) == 2 and kinds(offline["calls"]).count("video") == 2  # no extra request
    for plan in plans.clips:
        assert plan.neighbour_rule == staysput.NEIGHBOUR_RULE and len(plan.objects) == 1 and len(plan.neighbours) == 4
        assert [nb.object_id for nb in plan.neighbours] == ["2", "3", "4", "5"] and plan.prompt_text.count("<|box;") == 5
        assert plan.window_px == pytest.approx(120.0)  # four neighbours inside the base window: no zoom out
    paint, ctrl = rows["pickup_hood_kf1_m1"], rows["pickup_hood_kf1_ctrl"]
    assert paint.verdict == c.StaysPutVerdict.FIXED and ctrl.verdict == c.StaysPutVerdict.SLIDING  # the drift test as before
    pn, cn = paint.neighbour_test, ctrl.neighbour_test
    assert pn is not None and pn.verdict == c.StaysPutVerdict.FIXED and pn.used == 4 and pn.ratio_full < 1.5 and pn.ratio_near < 1.5, pn
    assert cn is not None and cn.verdict == c.StaysPutVerdict.SLIDING and cn.used == 4 and cn.ratio_full > 2.5 and cn.ratio_near > 2.5, cn
    assert {nb.ref for nb in pn.neighbours} == {"pickup_part_00", "pickup_part_01", "pickup_part_02", "pickup_part_03"}
    assert all(len(nb.centres) >= 80 for nb in pn.neighbours)
    assert sheet.neighbour_line(paint).startswith("neighbour test: moves with its 4 neighbours, ") and sheet.neighbour_line(paint).endswith("; fixed")
    assert sheet.neighbour_line(ctrl).startswith("neighbour test: slides against its 4 neighbours, ")
    line = sheet.staysput_line(paint, calibrated=False)
    assert line is not None and "neighbour test: moves with" in line and line.endswith(sheet.STAYSPUT_UNCALIBRATED)
    live = paths.mark_tracks_json("spn1").read_bytes()
    paths.mark_tracks_json("spn1").unlink()
    staysput.rebuild_staysput("spn1", th=TH)
    assert paths.mark_tracks_json("spn1").read_bytes() == live and kinds(offline["calls"]).count("video") == 2


def test_with_the_neighbour_test_off_clips_and_rows_are_as_before(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["parts"]
    build_walk("spn2", "pickup", scene, src)
    add_part_tracks("spn2", "pickup", scene)
    th = copy.deepcopy(TH)
    th["staysput"]["neighbours_enabled"] = False
    rows = derive("spn2", th=th)
    plans = staysput._read_plans("spn2", "pickup")
    assert all(p.neighbour_rule is None and p.neighbours == [] and p.prompt_text.count("<|box;") == 1 for p in plans.clips)
    assert all(r.neighbour_test is None for r in rows)
    assert b"neighbour_test" not in paths.mark_tracks_json("spn2").read_bytes()  # a row without the test keeps the old shape
    assert "neighbour_test" not in c.MarkTrack(mark_id="m", role="pickup").to_json()


def test_a_clip_lane_that_never_answers_lists_its_planned_neighbours(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["parts"]
    build_walk("spn3", "pickup", scene, src)
    add_part_tracks("spn3", "pickup", scene)
    offline["fail_clips"] = True
    rows = derive("spn3")
    for r in rows:
        assert r.verdict == c.StaysPutVerdict.NOT_RUN and r.neighbour_test is not None
        assert r.neighbour_test.verdict == c.StaysPutVerdict.NOT_RUN and len(r.neighbour_test.neighbours) == 4
        assert all(nb.frames == 0 and not nb.used for nb in r.neighbour_test.neighbours)


# ----------------------------------------------------------------------------------------------------------------
# The clip cap's order (lead 11/1): paired across the walks, then named by a narration seed, then confidence
# ----------------------------------------------------------------------------------------------------------------


SPREAD = [(60.0, 80.0), (260.0, 80.0), (460.0, 80.0), (60.0, 260.0), (260.0, 260.0)]  # 200 px apart: no shared window
CONFIDENCE = [c.Confidence.HIGH, c.Confidence.HIGH, c.Confidence.MEDIUM, c.Confidence.LOW, c.Confidence.LOW]


def cap_walk() -> tuple[staysput._Walk, list[Any], dict[str, Any]]:
    """Five chips on one still hood keyframe of the synthetic walk, one clip each, and a cap of two clips."""
    times = [i / 30 for i in range(N_SRC)]
    track = track_of(lambda i: [20.0, 20.0, 620.0, 340.0], N_SRC // 2, times)
    kf = c.Keyframe(kf_id="k", role="pickup", panel_id=c.PanelId.HOOD, track_id=track.track_id, f_proxy=37, pts=times[74],
                    score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=1.9,
                    box_src=[20.0, 20.0, 620.0, 340.0])
    marks = [chip(f"k_m{i}", bbox_src=[x - 5, y - 5, x + 5, y + 5]).model_copy(update={"confidence": conf})
             for i, ((x, y), conf) in enumerate(zip(SPREAD, CONFIDENCE), start=1)]
    walk = staysput._Walk(role="pickup", src=Path("unused"), src_size=SRC, proxy_size=PROXY, proxy_fps=15.0, proxy_pts=[], tracks={track.track_id: track},
                          privacy_tracks=[], keyframes={"k": kf}, marks=marks, times=times)
    th = copy.deepcopy(TH)
    th["staysput"]["neighbours_enabled"] = False  # the order alone: no clip folds into another
    th["staysput"]["max_clips_per_walk"] = 2
    return walk, [staysput._item(m, staysput._mark_box(m, 2.0)[0]) for m in marks], th


def kept(walk: staysput._Walk, items: list[Any], th: dict[str, Any], *, paired: tuple[str, ...] = (), seeded: tuple[str, ...] = ()) -> list[str]:
    walk.paired, walk.seeded = frozenset(paired), frozenset(seeded)
    plans, dropped = staysput._plans(walk, items, th)
    assert all(why == "over the 2-clip cap for one walk" for _it, why in dropped) and len(plans) + len(dropped) == len(items)
    return sorted(o.mark_id for p in plans for o in p.objects)


def test_the_clip_cap_takes_paired_marks_then_seeded_marks_then_confidence() -> None:
    walk, items, th = cap_walk()
    assert len(staysput._group(items, staysput._base_side(walk.keyframes["k"], th), th)) == 5
    assert kept(walk, items, th) == ["k_m1", "k_m2"]  # nothing ranked first: Spark's confidence, as before 11/1
    assert kept(walk, items, th, paired=("k_m5",)) == ["k_m1", "k_m5"]  # a low-confidence mark in a cross-walk pair
    assert kept(walk, items, th, seeded=("k_m4",)) == ["k_m1", "k_m4"]  # a mark the renter named
    assert kept(walk, items, th, paired=("k_m5",), seeded=("k_m4",)) == ["k_m4", "k_m5"]
    assert kept(walk, items, th, paired=("k_m3", "k_m4", "k_m5"), seeded=("k_m5",)) == ["k_m3", "k_m5"]  # paired and seeded first of all
    th["staysput"]["max_clips_per_walk"] = 16
    walk.paired = walk.seeded = frozenset()
    assert len(staysput._plans(walk, items, th)[0]) == 5  # under the cap the order changes nothing


def test_the_glare_control_still_ranks_first() -> None:
    walk, items, th = cap_walk()
    control = staysput._Item(mark_id="k_ctrl", kf_id="k", panel_id=c.PanelId.HOOD, type=c.MarkKind.REFLECTION, is_control=True,
                             box_src=(455.0, 255.0, 465.0, 265.0), point_crop=None, bbox_crop=None, rank=-1)
    walk.paired = walk.seeded = frozenset({"k_m5"})
    plans, _dropped = staysput._plans(walk, [*items, control], th)
    assert sorted(o.mark_id for p in plans for o in p.objects) == ["k_ctrl", "k_m5"]


def pair_panel(panel_id: c.PanelId, noun: c.Noun, *, glimpsed: str | None = None) -> c.Panel:
    roles = {role: c.PanelRole(state=c.CoverageState.GLIMPSED if role == glimpsed else c.CoverageState.FILMED) for role in ("pickup", "return")}
    return c.Panel(panel_id=panel_id, noun=noun, front_sign=1, per_role=roles)


def pair_kf(role: str, panel_id: c.PanelId) -> c.Keyframe:
    return c.Keyframe(kf_id=f"{role}_{panel_id.value}_kf1", role=role, panel_id=panel_id, track_id=f"{role}_{panel_id.value}_00", f_proxy=10, pts=0.7,
                      score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=1.5,
                      box_src=[100.0, 100.0, 500.0, 300.0], front_sign=1, marks_status=c.MarksStatus.COMPLETE)


def placed(mark_id: str, role: str, panel_id: c.PanelId, w: tuple[float, float], **kw: Any) -> c.Mark:
    return c.Mark(mark_id=mark_id, role=role, panel_id=panel_id, kf_id=f"{role}_{panel_id.value}_kf1", source=kw.pop("source", c.MarkSource.SPARK_BOX),
                  type=kw.pop("type", c.MarkKind.CHIP), confidence=c.Confidence.LOW, size_class=c.SizeClass.S, w=list(w), r=math.hypot(*w), **kw)


def test_cross_walk_pairs_are_the_closest_candidates_inside_the_eye_band(offline: dict[str, Any]) -> None:
    hood, lid, door = c.PanelId.HOOD, c.PanelId.TRUNK_LID, c.PanelId.LEFT_FRONT_DOOR
    c.write_json(paths.panels_json("xw1"), [pair_panel(hood, c.Noun.HOOD), pair_panel(lid, c.Noun.TRUNK_LID), pair_panel(door, c.Noun.CAR_DOOR, glimpsed="return")])
    c.write_json(paths.keyframes_json("xw1"), [pair_kf(role, pid) for role in ("pickup", "return") for pid in (hood, lid, door)])
    marks = [
        placed("p_hood_a", "pickup", hood, (0.10, 0.20)), placed("r_hood_a", "return", hood, (0.13, 0.22)),  # the same chip, 0.04 units apart
        placed("p_hood_b", "pickup", hood, (-0.60, 0.10)),  # nothing near it at return
        placed("r_hood_far", "return", hood, (0.90, -0.70)),
        placed("p_lid", "pickup", lid, (0.0, 0.0)), placed("r_lid", "return", lid, (2.5, 0.0)),  # nearest to each other, past tau_eye_r
        placed("p_door", "pickup", door, (0.2, 0.0)), placed("r_door", "return", door, (0.2, 0.0)),  # G1: glimpsed at return, never compared
    ]
    c.write_json(paths.marks_json("xw1"), marks)
    assert staysput.cross_walk_pairs("xw1", th=TH) == [("p_hood_a", "r_hood_a")]
    assert not paths.calibration_json("xw1").exists()  # the match's calibration, worked out and not written
    c.write_json(paths.marks_json("xw1"), [m for m in marks if m.role == "pickup"])
    assert staysput.cross_walk_pairs("xw1", th=TH) == []  # one walk has no pair


def test_seeded_marks_are_the_seed_rows_and_the_marks_a_seed_landed_on(offline: dict[str, Any]) -> None:
    hood = c.PanelId.HOOD
    c.write_json(paths.marks_json("sd1"), [placed("pickup_hood_kf1_m1", "pickup", hood, (0.1, 0.1)), placed("pickup_hood_kf1_seed_t2", "pickup", hood, (0.5, 0.1), source=c.MarkSource.NARRATION_SEED)])
    assert staysput.seeded_marks("sd1") == {"pickup_hood_kf1_seed_t2"}
    turns = [c.NarrationTurn(turn_id="t1", start_ms=0, end_ms=900, text="hood two chips", seed_mark_id="pickup_hood_kf1_m1"), c.NarrationTurn(turn_id="t2", start_ms=900, end_ms=1500, text="and the bumper")]
    c.write_json(paths.narration_json("sd1"), c.Narration(per_role={"pickup": c.NarrationRole(turns=turns)}))
    assert staysput.seeded_marks("sd1") == {"pickup_hood_kf1_m1", "pickup_hood_kf1_seed_t2"}


def test_derive_records_what_the_cap_ranked_first(offline: dict[str, Any], walks: dict[str, Any]) -> None:
    scene, src = walks["moving"]
    for role in ("pickup", "return"):
        build_walk("spr1", role, scene, src)
    marks = c.read_json_list(paths.marks_json("spr1"), c.Mark)
    c.write_json(paths.marks_json("spr1"), [m.model_copy(update={"w": [0.10, 0.05], "r": 0.112}) for m in marks])
    c.write_json(paths.panels_json("spr1"), [pair_panel(c.PanelId.HOOD, c.Noun.HOOD)])
    derive("spr1")
    plans = staysput._read_plans("spr1", "pickup")
    assert plans is not None and plans.paired == ["pickup_hood_kf1_m1"] and plans.seeded == [] and plans.skipped == []
    text = (paths.run_dir("spr1") / "clips" / "pickup" / staysput.PLAN_NAME).read_text()
    assert '"paired":["pickup_hood_kf1_m1"]' in text and '"seeded"' not in text  # an empty list stays out of the file
    assert "paired" not in staysput.RolePlans(role="pickup").to_json()  # a walk planned before 11/1 keeps its bytes


SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "pairC_core2"


@pytest.mark.skipif(not (SAMPLE / "derived" / "marks.json").is_file(), reason="no pairC_core2 sample in this checkout")
def test_on_pair_c_the_trunk_chips_get_a_clip_in_both_walks(offline: dict[str, Any]) -> None:
    """The sample's own files (marks with the seed rows, keyframes, panels, the lane summaries, the pts maps) and a
    29.58 fps timeline like the uCO3D original's: the narrated chip and its return twin were over the cap on 10/31."""
    run = paths.run_dir("pc2")
    shutil.copytree(SAMPLE, run, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.mp4", "*.ndjson.gz", "*.pdf", "*.m4a", "*.jpg", "bundle", "sheet", "clips", "chips"))
    th = json.loads((run / "run.json").read_text())["thresholds"]
    pairs = staysput.cross_walk_pairs("pc2", th=th)
    assert ("pickup_trunk_lid_kf1_m1", "return_trunk_lid_kf1_m2") in pairs  # the match's own pair on 10/31 (matches.json)
    stored = [(p["pickup_mark"], p["return_mark"]) for pm in json.loads((run / "derived" / "matches.json").read_text()) for p in pm["pairs"] if p["d"] < p["tau_eye_r"]]
    assert set(stored) <= set(pairs)
    assert staysput.seeded_marks("pc2") == {"pickup_right_front_door_kf1_seed_t2", "pickup_trunk_lid_kf1_m1"}
    for role, chip_id in (("pickup", "pickup_trunk_lid_kf1_m1"), ("return", "return_trunk_lid_kf1_m2")):
        walk = staysput._load_walk("pc2", role)
        assert walk is not None
        probe = json.loads((run / "videos" / f"{role}.json").read_text())["ffprobe"]
        walk.times = [i / probe["fps_avg"] for i in range(int(probe["duration_s"] * probe["fps_avg"]))]
        items = [staysput._item(m, box) for m in walk.marks if staysput._why_not(walk.keyframes.get(m.kf_id), walk) is None and (box := staysput._mark_box(m, staysput._min_box_px(th, walk.src_size))[0]) is not None]
        before = {o.mark_id for p in staysput._plans(walk, items, th)[0] for o in p.objects}
        walk.paired, walk.seeded, notes = staysput._priorities("pc2", walk, th)
        plans, _dropped = staysput._plans(walk, items, th)
        after = {o.mark_id for p in plans for o in p.objects}
        assert notes == [] and chip_id not in before and chip_id in after and len(plans) == 16
        if role == "pickup":
            assert {"pickup_right_front_door_kf1_seed_t2", "pickup_trunk_lid_kf1_m2"} & after == {"pickup_right_front_door_kf1_seed_t2"}  # the door seed in, a dirt mark out


# ----------------------------------------------------------------------------------------------------------------
# The neighbour reason's words (11/1)
# ----------------------------------------------------------------------------------------------------------------


def test_the_reason_counts_one_neighbour_in_the_singular_on_new_plans_only() -> None:
    frames = synthetic_lane(rigid_scene(), drop={f"n{i}": {45} for i in range(1, 6)})  # only n0 is on the prompt frame
    nt = run_neighbours(frames, "glint", [f"n{i}" for i in range(6)])
    assert nt.used == 1 and nt.reason == "1 neighbour tracked through the clip, 4 needed"
    assert run_neighbours(frames, "glint", [f"n{i}" for i in range(6)], rule=1).reason == "1 neighbours tracked through the clip, 4 needed"  # frozen words
    none = synthetic_lane(rigid_scene(), drop={f"n{i}": {45} for i in range(6)})
    assert run_neighbours(none, "glint", [f"n{i}" for i in range(6)]).reason == "0 neighbours tracked through the clip, 4 needed"
    assert staysput.NEIGHBOUR_RULE == 2
