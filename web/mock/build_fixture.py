"""Builds web/mock/fixture/<run>/: one sample pair in the exact shapes of walkaround/contracts.py, for `npm run dev`
without the backend. Run from web/: ../.venv/bin/python mock/build_fixture.py

Every record goes through the pydantic models and contracts.write_json, the event log through contracts.chain_hash,
the lanes through sam_wire.index_from_events and the mark positions through geometry, so the mock cannot drift from
the contract silently. What is real: the cookbook ping pong clip (data/test/pingpong.mp4, served as the blurred walk by
the mock server, never copied into the fixture) and the two saved SAM 3.1 streams of 10/26 (tests/fixtures/sam).
Everything else is hand-made dev data: the panels, crops (drawn here, not photos), Stays Put clips (a synthetic test
pattern, so no face or plate from the clip is ever committed), marks, verdicts, narration and the sheet. Output is
deterministic.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

WEB = Path(__file__).resolve().parent.parent
REPO = WEB.parent
sys.path.insert(0, str(REPO))

from walkaround import contracts as c  # noqa: E402
from walkaround import geometry, sam_wire  # noqa: E402

RUN_ID = "mock-sample-pair"
OUT = WEB / "mock" / "fixture" / RUN_ID
FIX = REPO / "tests" / "fixtures"
CLIP = REPO / "data" / "test" / "pingpong.mp4"
T0 = datetime(2026, 10, 27, 15, 2, 11, tzinfo=timezone.utc)
FPS = 24.0
FRAMES = 141
SIZE = (1280, 720)
ROLES = ("pickup", "return")
CAMERA = {"pickup": T0 - timedelta(minutes=6, seconds=41), "return": T0 + timedelta(minutes=38, seconds=15)}
FAKE_SHA = {name: hashlib.sha256(name.encode()).hexdigest() for name in ("pickup-original", "return-original")}


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(OUT))


def put_json(name: str, value: Any) -> str:
    return c.write_json(OUT / name, value)


# Lanes --------------------------------------------------------------------------------------------------------------


def events_of(name: str) -> list[dict[str, Any]]:
    return list(sam_wire.read_events(FIX / name))


def truncated(events: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """A stream that ended early: the first `keep` events, no response.completed."""
    return [e for e in events[:keep] if e.get("type") != "response.completed"]


TEXT = events_of("sam/pingpong_text.events.ndjson")
BOX = events_of("sam/pingpong_box.events.ndjson")
ZERO = events_of("sam_zero_match.events.ndjson")

# role -> lane id -> (raw events or None for a lane that never answered, status)
LANES: dict[str, dict[str, tuple[list[dict[str, Any]] | None, c.LaneStatus]]] = {}
for role in ROLES:
    lanes: dict[str, tuple[list[dict[str, Any]] | None, c.LaneStatus]] = {}
    for noun in c.LANE_NOUNS:
        lanes[c.noun_slug(noun)] = (ZERO, c.LaneStatus.ZERO_MATCHES)
    lanes["car_door"] = (TEXT if role == "pickup" else BOX, c.LaneStatus.COMPLETED)
    LANES[role] = lanes
LANES["pickup"]["roof"] = (None, c.LaneStatus.FAILED)
LANES["return"]["bumper"] = (truncated(TEXT, 70), c.LaneStatus.INCOMPLETE)


def write_lanes() -> dict[str, dict[str, tuple[c.LaneRecord, list[c.IndexFrame]]]]:
    out: dict[str, dict[str, tuple[c.LaneRecord, list[c.IndexFrame]]]] = {}
    for role, lanes in LANES.items():
        out[role] = {}
        for lane, (events, status) in lanes.items():
            noun = c.Noun(lane.replace("_", " "))
            record = c.LaneRecord(lane_id=lane, role=role, noun=noun, prompt_text=noun.value, media_size=list(SIZE), status=status, frames_sent=FRAMES)
            frames: list[c.IndexFrame] = []
            if events is not None:
                raw = "".join(json.dumps(e, sort_keys=True) + "\n" for e in events).encode()
                gz = OUT / "sam" / role / f"{lane}.ndjson.gz"
                gz.parent.mkdir(parents=True, exist_ok=True)
                gz.write_bytes(gzip.compress(raw, mtime=0))
                frames, outcome = sam_wire.index_from_events(events, media="video")
                assert outcome.status == status, (role, lane, outcome.status, status)
                ids = sorted({o.id for f in frames for o in f.objects})
                record = record.model_copy(
                    update={
                        "lane_path": rel(gz),
                        "lane_sha256": sha(gz),
                        "frames_returned": len(frames),
                        "ids": ids,
                        "zero_matches": status == c.LaneStatus.ZERO_MATCHES,
                        "video_frames_processed": outcome.video_frames_processed,
                        "request_id": outcome.response_id,
                        "parse_cpu_s": 0.004 * max(1, len(frames)),
                        "dollars": round(0.0002 * FRAMES, 6) if outcome.video_frames_processed else 0.0,
                    }
                )
            else:
                record = record.model_copy(update={"error": "429 after 6 tries"})
            out[role][lane] = (record, frames)
    return out


# Pictures -----------------------------------------------------------------------------------------------------------

CROP_W, CROP_H = 1280, 860
PANEL_PAINT = {"right_rear_door": (88, 104, 120), "left_front_door": (88, 104, 120), "hood": (96, 110, 126)}


def font(size: int) -> ImageFont.ImageFont:
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def panel_polygon(panel: str, shift: tuple[int, int]) -> list[tuple[int, int]]:
    dx, dy = shift
    if panel == "hood":
        pts = [(170, 250), (1110, 250), (1220, 760), (60, 760)]
    else:
        pts = [(150, 120), (1090, 120), (1160, 210), (1160, 760), (120, 760), (110, 200)]
    return [(x + dx, y + dy) for x, y in pts]


def draw_crop(panel: str, role: str, shift: tuple[int, int], marks: list[dict[str, Any]], *, glare: bool, stem: str | None = None) -> tuple[Path, Path, np.ndarray]:
    """A drawn stand-in for a blurred 1280 px panel crop, plus its one-bit panel mask. Not a photo."""
    img = Image.new("RGB", (CROP_W, CROP_H), (46, 49, 48))
    d = ImageDraw.Draw(img)
    for y in range(CROP_H):
        g = 40 + int(18 * y / CROP_H)
        d.line([(0, y), (CROP_W, y)], fill=(g, g + 3, g + 2))
    poly = panel_polygon(panel, shift)
    paint = PANEL_PAINT[panel]
    d.polygon(poly, fill=paint)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    if panel != "hood":
        top = min(ys)
        d.polygon([(xs[0] + 40, top + 18), (xs[1] - 40, top + 18), (xs[1] - 10, top + 180), (xs[0] + 30, top + 180)], fill=(34, 40, 46))
        d.rounded_rectangle([max(xs) - 260, top + 300, max(xs) - 110, top + 336], radius=12, fill=(52, 60, 68))
        d.line([(min(xs) + 20, top + 214), (max(xs) - 20, top + 214)], fill=(70, 84, 98), width=3)
    if glare:
        # A soft sky highlight, clipped to the panel.
        light = Image.new("L", (CROP_W, CROP_H), 0)
        ImageDraw.Draw(light).ellipse([420, 300, 880, 500], fill=210)
        light = light.filter(ImageFilter.GaussianBlur(70))
        clip = Image.new("L", (CROP_W, CROP_H), 0)
        ImageDraw.Draw(clip).polygon(poly, fill=255)
        light = ImageChops.multiply(light, clip)
        img.paste(Image.new("RGB", (CROP_W, CROP_H), (236, 240, 244)), (0, 0), light)
        d = ImageDraw.Draw(img)
    for m in marks:
        x1, y1, x2, y2 = m["bbox"]
        if m["type"] == "scuff":
            for k in range(7):
                d.line([(x1 + k * 4, y1 + 6 + k * 2), (x2 - 6 + k, y2 - 8 + k)], fill=(170, 176, 180), width=2)
        elif m["type"] == "chip":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(196, 198, 196))
            d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(28, 30, 32))
        elif m["type"] == "scratch":
            d.rectangle([x1, y1, x2, y2], fill=(59, 125, 216))
        elif m["type"] == "reflection":
            d.ellipse([x1, y1, x2, y2], outline=(230, 232, 236), width=4)
    d.text((24, CROP_H - 44), f"mock crop, drawn, not a photo: {panel.replace('_', ' ')}, {role}", fill=(214, 210, 200), font=font(24))
    stem = stem or f"crops/{role}/kf_{role}_{panel}"
    crop = OUT / f"{stem}.jpg"
    crop.parent.mkdir(parents=True, exist_ok=True)
    img.save(crop, quality=78, optimize=True)
    mask_img = Image.new("1", (CROP_W, CROP_H), 0)
    ImageDraw.Draw(mask_img).polygon(poly, fill=1)
    mask_png = OUT / f"{stem}.mask.png"
    mask_img.save(mask_png, optimize=True)
    return crop, mask_png, np.array(mask_img, dtype=bool)


def silent_chip(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", f"{seconds}", "-c:a", "aac", "-b:a", "32k", "-fflags", "+bitexact", "-map_metadata", "-1", str(path)],
        check=True,
    )


def clip_from_walk(path: Path, x: int, y: int, size: int) -> None:
    """A stand-in for a Stays Put clip: 3 s of a synthetic test pattern at 504 x 504, offset by (x, y) so the two clips
    differ. The walk itself has people's faces in it, and nothing unblurred from it belongs in the repo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pattern = f"testsrc2=size={size}x{size}:rate=30:duration=3,hue=h={(x + y) % 360},scale=504:504"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", pattern,
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-fflags", "+bitexact", "-map_metadata", "-1", str(path),
        ],
        check=True,
    )


# Records ------------------------------------------------------------------------------------------------------------


def video_record(role: str) -> c.VideoRecord:
    cam = CAMERA[role]
    received = T0 if role == "pickup" else T0 + timedelta(minutes=41)
    ffprobe = c.FfprobeInfo(codec="h264", container="mp4", width=SIZE[0], height=SIZE[1], display_width=SIZE[0], display_height=SIZE[1], fps_avg=FPS, duration_s=FRAMES / FPS, creation_time=iso(cam), device="mock", has_audio=role == "pickup")
    upload = received + timedelta(seconds=30)
    return c.VideoRecord(
        role=role,
        original_filename=f"{role}.mp4",
        bytes=CLIP.stat().st_size,
        sha256_browser=FAKE_SHA[f"{role}-original"],
        sha256_server=FAKE_SHA[f"{role}-original"],
        hash_match=True,
        received_at=iso(received),
        chunks_expected=1,
        chunks_received=1,
        ffprobe=ffprobe,
        proxy=c.ProxyRecord(path=f"media/{role}.proxy.mp4", fps=FPS, width=SIZE[0], height=SIZE[1], frames=FRAMES, pts_map_path=f"media/{role}.pts.json"),
        proxy_blurred=c.BlurredProxyRecord(path=f"media/{role}.proxy_blurred.mp4", sha256=sha(CLIP)),
        files_api=c.FilesApiRecord(file_id=f"file-mock-{role}", uploaded_at=iso(upload), expires_at=iso(upload + timedelta(hours=1)), deleted_at=iso(upload + timedelta(minutes=9)), delete_response={"id": f"file-mock-{role}", "object": "file", "deleted": True}),
        quarantine_path=f"quarantine/{role}.orig",
        quarantine_deleted_at=iso(upload + timedelta(minutes=9)),
    )


def fragments_of(frames: list[c.IndexFrame], object_id: str) -> list[c.TrackFragment]:
    seen = [f.frame for f in frames if any(o.id == object_id for o in f.objects)]
    out: list[c.TrackFragment] = []
    start = prev = seen[0]
    for f in seen[1:]:
        if f - prev > 1:
            out.append(c.TrackFragment(object_id=object_id, first_frame=start, last_frame=prev))
            start = f
        prev = f
    out.append(c.TrackFragment(object_id=object_id, first_frame=start, last_frame=prev))
    return out


def track_for(role: str, frames: list[c.IndexFrame]) -> c.PanelTrack:
    object_id = frames[0].objects[0].id
    rows = []
    for f in frames:
        o = next(o for o in f.objects if o.id == object_id)
        x1, y1, x2, y2 = o.box
        rows.append(c.FrameObs(frame=f.frame, pts=f.frame / FPS, box=[float(v) for v in o.box], box_area_frac=(x2 - x1) * (y2 - y1) / (SIZE[0] * SIZE[1]), in_car_box=True, c=o.c))
    frags = fragments_of(frames, object_id)
    merges = [c.MergeRecord(rule=c.MergeRule.VELOCITY_GAP, gap_s=(b.first_frame - a.last_frame) / FPS, pred_dist=0.12, area_ratio=1.04) for a, b in zip(frags, frags[1:])]
    return c.PanelTrack(
        track_id=f"{role}_car_door_t0",
        role=role,
        noun=c.Noun.CAR_DOOR,
        sam_object_ids=[object_id],
        fragments=frags,
        merges=merges,
        c_present_share=1.0,
        per_frame=rows,
        panel_id=c.PanelId.RIGHT_REAR_DOOR,
        side=c.Side.RIGHT,
        position=c.Position.REAR,
    )


SHIFT = {"pickup": (0, 0), "return": (-36, 14)}
MARKS: dict[tuple[str, str], list[dict[str, Any]]] = {
    ("pickup", "right_rear_door"): [
        {"id": "m1", "type": "scuff", "bbox": (612, 448, 690, 488), "size": "S", "conf": "medium", "note": "light scuff below the handle line"},
        {"id": "m2", "type": "chip", "bbox": (402, 560, 430, 588), "size": "S", "conf": "low", "note": "small chip"},
    ],
    ("return", "right_rear_door"): [
        {"id": "m1", "type": "scuff", "bbox": (579, 464, 655, 505), "size": "S", "conf": "medium", "note": "scuff below the handle line"},
        {"id": "m2", "type": "chip", "bbox": (371, 571, 399, 599), "size": "S", "conf": "low", "note": "chip"},
    ],
    ("pickup", "left_front_door"): [],
    ("return", "left_front_door"): [{"id": "m1", "type": "scratch", "bbox": (520, 470, 650, 500), "size": "M", "conf": "high", "note": "blue strip, likely tape"}],
    ("pickup", "hood"): [{"id": "m1", "type": "reflection", "bbox": (560, 360, 720, 440), "size": "M", "conf": "low", "note": "bright patch, likely sky"}],
    ("return", "hood"): [],
}
KF_FRAME = {"right_rear_door": 40, "left_front_door": 96, "hood": 120}


def build_keyframes_and_marks() -> tuple[list[c.Keyframe], list[c.Mark]]:
    keyframes: list[c.Keyframe] = []
    marks: list[c.Mark] = []
    for (role, panel), specs in MARKS.items():
        glare = panel == "hood" and role == "return"
        crop_path, mask_path, mask = draw_crop(panel, role, SHIFT[role], specs, glare=glare)
        moments = geometry.moments_from_mask(mask)
        crop = c.Crop(path=rel(crop_path), sha256=sha(crop_path), w=CROP_W, h=CROP_H, scale=0.5, dx=900.0, dy=820.0)
        kf_id = f"kf_{role}_{panel}"
        f = KF_FRAME[panel] + (3 if role == "return" else 0)
        conditions = c.Conditions(glare=glare)
        plate = [[1180.0, 1820.0, 1420.0, 1880.0]] if panel == "right_rear_door" else []
        keyframes.append(
            c.Keyframe(
                kf_id=kf_id,
                role=role,
                panel_id=c.PanelId(panel),
                track_id=f"{role}_car_door_t0" if panel == "right_rear_door" else f"{role}_{panel}_mock",
                f_proxy=f,
                pts=f / FPS,
                score=c.KeyframeScore(area=0.92, completeness=0.97, sharpness=0.81, glare=0.4 if glare else 0.03, frontality=0.88, total=0.71),
                aspect_ratio=1.38,
                box_src=[900.0, 820.0, 900.0 + CROP_W / 0.5, 820.0 + CROP_H / 0.5],
                crop=crop,
                mask_png=rel(mask_path),
                moments=moments,
                front_sign=1,
                blur=c.BlurRecord(plate_boxes=plate, request_ids=["resp_mock_plate"], plate_ruler_px=236.0 if plate else None, plate_ruler_c=0.91 if plate else None),
                conditions=conditions,
                marks_status=c.MarksStatus.COMPLETE,
            )
        )
        for spec in specs:
            x1, y1, x2, y2 = spec["bbox"]
            point = ((x1 + x2) / 2, (y1 + y2) / 2)
            w = geometry.whiten(point, moments, 1)
            caps = []
            if spec["conf"] == "low":
                caps.append("K12")
            if spec["type"] in ("reflection", "dirt", "unsure"):
                caps.append("K4")
            marks.append(
                c.Mark(
                    mark_id=f"{kf_id}_{spec['id']}",
                    role=role,
                    panel_id=c.PanelId(panel),
                    kf_id=kf_id,
                    source=c.MarkSource.SPARK_BOX,
                    type=c.MarkKind(spec["type"]),
                    confidence=c.Confidence(spec["conf"]),
                    size_class=c.SizeClass(spec["size"]),
                    description=spec["note"],
                    grid_point=c.GridPoint(x=round(point[0] / CROP_W * 1000), y=round(point[1] / CROP_H * 1000)),
                    grid_bbox=c.GridBox(x_min=round(x1 / CROP_W * 1000), y_min=round(y1 / CROP_H * 1000), x_max=round(x2 / CROP_W * 1000), y_max=round(y2 / CROP_H * 1000)),
                    bbox_crop=[float(v) for v in spec["bbox"]],
                    point_crop=[point[0], point[1]],
                    bbox_src=[crop.dx + x1 / crop.scale, crop.dy + y1 / crop.scale, crop.dx + x2 / crop.scale, crop.dy + y2 / crop.scale],
                    point_src=[crop.dx + point[0] / crop.scale, crop.dy + point[1] / crop.scale],
                    w=[w[0], w[1]],
                    r=float(geometry.radius(w)),
                    size_units=0.08 if spec["size"] == "S" else 0.2,
                    size_px_src=float(max(x2 - x1, y2 - y1) / crop.scale),
                    spark_request_id=f"resp_mock_spark_{role}_{panel}",
                    caps=caps,
                )
            )
    return keyframes, marks


def build_panels(tracks: list[c.PanelTrack]) -> tuple[list[c.Panel], c.Coverage]:
    states: dict[str, dict[str, tuple[c.CoverageState, float, str]]] = {
        "pickup": {
            "right_rear_door": (c.CoverageState.FILMED, 5.6, ""),
            "left_front_door": (c.CoverageState.FILMED, 4.1, ""),
            "hood": (c.CoverageState.GLIMPSED, 1.4, "median c under the run's 25th percentile"),
            "front_bumper": (c.CoverageState.FILMED, 3.2, ""),
            "rear_bumper": (c.CoverageState.GLIMPSED, 0.8, "0.8 s in frame"),
            "roof": (c.CoverageState.NOT_RUN, 0.0, "the roof lane got no answer after 6 tries"),
            "left_front_wheel": (c.CoverageState.FILMED, 2.4, ""),
            "right_rear_wheel": (c.CoverageState.FILMED, 2.1, ""),
        },
        "return": {
            "right_rear_door": (c.CoverageState.FILMED, 5.6, ""),
            "left_front_door": (c.CoverageState.FILMED, 3.8, ""),
            "hood": (c.CoverageState.FILMED, 3.0, ""),
            "front_bumper": (c.CoverageState.FILMED, 2.9, ""),
            "rear_bumper": (c.CoverageState.NOT_RUN, 0.0, "the bumper track ended early"),
            "roof": (c.CoverageState.NOT_FILMED, 0.0, "the roof lane completed with no match"),
            "left_front_wheel": (c.CoverageState.GLIMPSED, 0.9, ""),
            "right_rear_wheel": (c.CoverageState.FILMED, 2.0, ""),
        },
    }
    panels: list[c.Panel] = []
    per_role_cov: dict[str, dict[c.PanelId, c.PanelCoverage]] = {r: {} for r in ROLES}
    for pid in c.PanelId:
        per_role: dict[str, c.PanelRole] = {}
        for role in ROLES:
            state, seconds, reason = states[role].get(pid.value, (c.CoverageState.NOT_SEEN, 0.0, "no track in either walk"))
            track_ids = [t.track_id for t in tracks if t.role == role and t.panel_id == pid]
            windows = [[t.fragments[0].first_frame / FPS, (t.fragments[-1].last_frame + 1) / FPS] for t in tracks if t.role == role and t.panel_id == pid]
            if not windows and state in (c.CoverageState.FILMED, c.CoverageState.GLIMPSED):
                windows = [[1.0, 1.0 + seconds]]
            kfs = [f"kf_{role}_{pid.value}"] if pid.value in KF_FRAME else []
            per_role[role] = c.PanelRole(track_ids=track_ids, windows=windows, seconds_visible=seconds, state=state, reason=reason, keyframe_ids=kfs)
            per_role_cov[role][pid] = c.PanelCoverage(state=state, seconds=seconds, best_kf=kfs[0] if kfs else None, reason=reason)
        side = c.Side.LEFT if pid.value.startswith("left") else c.Side.RIGHT if pid.value.startswith("right") else c.Side.NONE
        position = c.Position.FRONT if "front" in pid.value else c.Position.REAR if "rear" in pid.value else c.Position.NONE
        comparable = pid.value in ("right_rear_door", "left_front_door")
        panels.append(
            c.Panel(
                panel_id=pid,
                noun=c.PANEL_NOUN[pid],
                side=side,
                position=position,
                instance_from=c.InstanceFrom.TRACKS if pid.value in KF_FRAME else c.InstanceFrom.CANONICAL,
                votes=c.SideVotes(R1=0.6, R2=0.3) if side != c.Side.NONE else c.SideVotes(),
                front_sign=1 if pid.value in KF_FRAME else None,
                per_role=per_role,
                comparable=comparable,
                not_comparable_reasons=[] if comparable else (["G2"] if pid == c.PanelId.HOOD else ["G1"]),
            )
        )
    coverage = c.Coverage(
        per_role=per_role_cov,
        missing=[c.PanelId.REAR_BUMPER, c.PanelId.ROOF],
        topup_text="rear bumper: 0.8 s, add 10 seconds? roof: not run, add 5 seconds?",
        nouns_to_rerun=[c.Noun.BUMPER, c.Noun.ROOF, c.Noun.CAR, c.Noun.LICENSE_PLATE, c.Noun.FACE],
        odometer=c.Odometer(asked=True, filmed=False),
    )
    return panels, coverage


def build_match(marks: list[c.Mark], calib: c.Calibration) -> tuple[list[c.PanelMatch], list[c.Verdict]]:
    def mark(kf: str, n: str) -> c.Mark:
        return next(m for m in marks if m.mark_id == f"{kf}_{n}")

    def cand(a: c.Mark, b: c.Mark) -> c.MatchCandidate:
        d_pos = float(np.hypot(a.w[0] - b.w[0], a.w[1] - b.w[1]))
        r_mean = ((a.r or 0) + (b.r or 0)) / 2
        return c.MatchCandidate(pickup_mark=a.mark_id, return_mark=b.mark_id, d=d_pos + 0.1, d_pos=d_pos, type_cost=0.1, size_cost=0.0, r_mean=r_mean, tau_pre_r=calib.tau_pre * (1 + 0.25 * r_mean**2), tau_eye_r=calib.tau_eye * (1 + 0.25 * r_mean**2), ratio=None, margin=None, mutual=True)

    p1, r1 = mark("kf_pickup_right_rear_door", "m1"), mark("kf_return_right_rear_door", "m1")
    p2, r2 = mark("kf_pickup_right_rear_door", "m2"), mark("kf_return_right_rear_door", "m2")
    tape = mark("kf_return_left_front_door", "m1")
    pairs = [cand(p1, r1), cand(p2, r2)]
    trace = lambda ids, seqs: c.Trace(rule_ids=ids, thresholds={"tau_pre": calib.tau_pre, "tau_eye": calib.tau_eye}, sam_request_ids=["resp_mock_door"], spark_request_ids=["resp_mock_spark_pickup_right_rear_door", "resp_mock_spark_return_right_rear_door"], event_seqs=seqs, config_hash=CONFIG_HASH)
    matches = [
        c.PanelMatch(panel_id=c.PanelId.RIGHT_REAR_DOOR, state=c.MatchState.COMPARABLE, kf_pair=["kf_pickup_right_rear_door", "kf_return_right_rear_door"], aspect_diff=0.02, mask_iou=0.91, handle_residual=[0.01, -0.02], pairs=pairs),
        c.PanelMatch(panel_id=c.PanelId.LEFT_FRONT_DOOR, state=c.MatchState.COMPARABLE, kf_pair=["kf_pickup_left_front_door", "kf_return_left_front_door"], aspect_diff=0.03, mask_iou=0.88, unmatched_return=[tape.mark_id]),
        c.PanelMatch(panel_id=c.PanelId.HOOD, state=c.MatchState.NOT_COMPARABLE, reasons=["G2"], kf_pair=["kf_pickup_hood", "kf_return_hood"]),
    ]
    verdicts = [
        c.Verdict(verdict_id="v1", panel_id=c.PanelId.RIGHT_REAR_DOOR, label=c.VerdictLabel.ALREADY_THERE, rule_id="L3", mark_ids=[p1.mark_id, r1.mark_id], distance_units=pairs[0].d_pos, distance_cm_est=pairs[0].d_pos * 35, trace=trace(["L3"], [120, 410])),
        c.Verdict(verdict_id="v2", panel_id=c.PanelId.RIGHT_REAR_DOOR, label=c.VerdictLabel.NEEDS_YOUR_EYE, rule_id="L4", mark_ids=[p2.mark_id, r2.mark_id], distance_units=pairs[1].d_pos, distance_cm_est=pairs[1].d_pos * 35, reasons=["K12"], trace=trace(["L4", "K12"], [121, 411])),
        c.Verdict(
            verdict_id="v3",
            panel_id=c.PanelId.LEFT_FRONT_DOOR,
            label=c.VerdictLabel.NEW_OR_UNSEEN,
            rule_id="L2",
            mark_ids=[tape.mark_id],
            trace=trace(["L2"], [412]),
            size=c.SizeEstimate(refused_reason="the plate is not in this frame", threshold_type=c.ThresholdType.BODY_SCRATCH),
        ),
        c.Verdict(verdict_id="v4", panel_id=c.PanelId.HOOD, label=c.VerdictLabel.NOT_COMPARABLE, rule_id="L6", reasons=["G2"], trace=trace(["L6", "G2"], [413])),
    ]
    return matches, verdicts


def build_mark_tracks(marks: list[c.Mark]) -> list[c.MarkTrack]:
    fixed_clip = OUT / "clips" / "pickup" / "kf_pickup_right_rear_door_m1.mp4"
    control_clip = OUT / "clips" / "pickup" / "kf_pickup_hood_control.mp4"
    clip_from_walk(fixed_clip, 420, 180, 504)
    clip_from_walk(control_clip, 700, 120, 504)
    still = [c.TrackPoint(f=f, box_clip=[226.0 + (f % 3) * 0.4, 238.0, 278.0 + (f % 3) * 0.4, 262.0], c=0.88) for f in range(91)]
    slide = [c.TrackPoint(f=f, box_clip=[140.0 + f * 2.2, 200.0, 230.0 + f * 2.2, 250.0], c=0.71) for f in range(91)]
    return [
        c.MarkTrack(mark_id="kf_pickup_right_rear_door_m1", role="pickup", clip=c.ClipRecord(path=rel(fixed_clip), sha256=sha(fixed_clip), f0_src=1155, frames=91, window_px=504, w=504, h=504), request_id="resp_mock_staysput_m1", prompt_text="(object prompt)", track=still, b1=0.21, b2=0.02, b_pass=True, drift_rms=0.03, sign_agreement=0.94, pairs_n=88, verdict=c.StaysPutVerdict.FIXED),
        c.MarkTrack(mark_id="kf_pickup_hood_m1", role="pickup", clip=c.ClipRecord(path=rel(control_clip), sha256=sha(control_clip), f0_src=3500, frames=91, window_px=504, w=504, h=504), request_id="resp_mock_staysput_ctrl", prompt_text="(object prompt)", track=slide, b1=0.19, b2=0.03, b_pass=True, drift_rms=0.61, sign_agreement=0.97, pairs_n=90, verdict=c.StaysPutVerdict.SLIDING, highlight_cap=True, is_control=True),
    ]


# Events -------------------------------------------------------------------------------------------------------------

CONFIG_HASH = c.config_hash(c.load_thresholds())


class Chain:
    def __init__(self, genesis: str) -> None:
        self.prev = genesis
        self.events: list[c.Event] = []
        self.t = T0

    def emit(self, kind: c.EventKind, payload: dict[str, Any], dt: float = 0.2) -> None:
        self.t += timedelta(seconds=dt)
        line = {"seq": len(self.events), "ts": self.t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{self.t.microsecond // 1000:03d}Z", "run_id": RUN_ID, "kind": kind.value, "payload": json.loads(c.canonical_json(payload)), "prev": self.prev}
        event = c.Event(**line, hash=c.chain_hash(self.prev, line))
        self.events.append(event)
        self.prev = event.hash

    def stage(self, status: c.RunStatus, pct: int, dt: float = 0.3) -> None:
        self.emit(c.EventKind.STAGE, {"name": status.value, "pct": pct}, dt)


def build_events(genesis: str, manifest: c.SealManifest, videos: dict[str, c.VideoRecord], lanes, derived_sha: dict[str, str], marks: list[c.Mark], deletions: c.Deletions, sheet_sha: dict[str, str], coverage: c.Coverage) -> list[c.Event]:
    E = c.EventKind
    ch = Chain(genesis)
    ch.emit(E.RUN_CREATED, {"kind": "pair", "mode": "car", "keep_plate": False, "config_hash": CONFIG_HASH}, 0)
    for role in ROLES:
        v = videos[role]
        ch.emit(E.UPLOAD_STARTED, {"role": role, "upload_id": f"mock-upload-{role}", "chunks": 1}, 4.0)
        ch.emit(E.UPLOAD_DONE, {"role": role, "bytes": v.bytes, "sha256": v.sha256_server, "filename": v.original_filename}, 6.0)
    ch.stage(c.RunStatus.SEALING, 5)
    ch.emit(E.SEAL_MANIFEST, {"sha256": genesis, "files": manifest.files}, 1.2)
    ch.emit(E.SEAL_TSR, {"tsr_time": manifest.stamps.tsr_time, "serial": manifest.stamps.tsa_serial, "issuer": manifest.stamps.tsa_issuer}, 1.6)
    ch.emit(E.SEAL_OTS, {"status": "pending", "pickup": "pending", "return": "pending"}, 0.8)
    ch.stage(c.RunStatus.SEALED, 10)
    ch.stage(c.RunStatus.CONVERTING, 15)
    for role in ROLES:
        ch.emit(E.MEDIA_PROBED, {"role": role, "ffprobe": videos[role].ffprobe}, 1.1)
    for role in ROLES:
        if not videos[role].ffprobe.has_audio:
            ch.emit(E.VOICE_DONE, {"role": role, "mode_applied": False, "text": "no audio: this walk has no audio track"}, 0.1)
        ch.emit(E.MEDIA_PROXY_DONE, {"role": role, "proxy": videos[role].proxy}, 9.0)
    ch.stage(c.RunStatus.CONVERTING, 20)
    for role in ROLES:
        ch.emit(E.FILES_UPLOADED, {"role": role, "file_id": videos[role].files_api.file_id, "expires_at": videos[role].files_api.expires_at}, 9.9)
    ch.stage(c.RunStatus.TRACKING, 45)
    for role in ROLES:
        role_lanes = lanes[role]
        for lane, (record, _frames) in role_lanes.items():
            if record.status != c.LaneStatus.FAILED:
                ch.emit(E.SAM_REQUEST, {"request_id": record.request_id, "role": role, "noun": lane, "params": {"include_confidence": "true", "mask_encoding": "one_bit"}, "store": False}, 0.05)
        streaming = [(lane, frames) for lane, (record, frames) in role_lanes.items() if frames]
        for lane, _ in streaming:
            ch.emit(E.SAM_FIRST_DELTA, {"role": role, "noun": lane, "ms": 3600 if lane == "car_door" else 3900}, 1.8)
        for lane, (record, _frames) in role_lanes.items():
            if record.status == c.LaneStatus.ZERO_MATCHES:
                ch.emit(E.SAM_LANE_DONE, lane_done(role, lane, record, []), 0.15)
        longest = max(len(f) for _, f in streaming)
        for i in range(longest):
            for lane, frames in streaming:
                if i < len(frames):
                    fr = frames[i]
                    ch.emit(E.SAM_FRAME, {"role": role, "noun": lane, "frame": fr.frame, "objects": [{"id": o.id, "box": o.box, "c": o.c} for o in fr.objects]}, 0.03)
        for lane, (record, frames) in role_lanes.items():
            if record.status == c.LaneStatus.INCOMPLETE:
                ch.emit(E.SAM_FAILED, {"role": role, "noun": lane, "status": "incomplete", "error": "the stream ended before response.completed", "request_id": record.request_id}, 0.2)
                ch.emit(E.SAM_LANE_DONE, lane_done(role, lane, record, frames), 0.05)
            elif record.status == c.LaneStatus.COMPLETED:
                ch.emit(E.SAM_LANE_DONE, lane_done(role, lane, record, frames), 0.3)
            elif record.status == c.LaneStatus.FAILED:
                ch.emit(E.SAM_FAILED, {"role": role, "noun": lane, "status": "failed", "error": record.error, "request_id": None}, 0.2)
    ch.emit(E.LEDGER_ROW, {"dollars_so_far": 0.94}, 0.4)
    ch.stage(c.RunStatus.TRACKING, 55)
    ch.emit(E.PANELS_DERIVED, {"sha256": {n: derived_sha[n] for n in ("tracks", "panels", "keyframes")}}, 2.5)
    ch.emit(E.COVERAGE_UPDATED, {"sha256": derived_sha["coverage"], "missing": coverage.missing}, 0.1)
    ch.emit(E.TOPUP_REQUESTED, {"missing": coverage.missing, "topup_text": coverage.topup_text, "nouns_to_rerun": coverage.nouns_to_rerun}, 0.1)
    ch.emit(E.LEDGER_ROW, {"dollars_so_far": 1.21}, 0.2)
    ch.stage(c.RunStatus.BLURRING, 60)
    for role in ROLES:
        ch.emit(E.BLUR_DONE, {"role": role, "sha256": videos[role].proxy_blurred.sha256, "path": videos[role].proxy_blurred.path}, 6.0)
    ch.stage(c.RunStatus.MARKING, 70)
    for kf in sorted({m.kf_id for m in marks} | {"kf_pickup_left_front_door", "kf_return_hood"}):
        role = "pickup" if "_pickup_" in kf else "return"
        request_id = f"resp_mock_spark_{kf[3:]}"
        # orchestrator._emit_spark: spark.request then spark.response for each stored call.
        ch.emit(E.SPARK_REQUEST, {"role": role, "request_id": request_id, "purpose": "marks", "max_output_tokens": 4000, "effort": "low", "store": False}, 0.1)
        ch.emit(E.SPARK_RESPONSE, {"role": role, "request_id": request_id, "status": "completed", "incomplete_reason": None, "truncated": False, "sha256": hashlib.sha256(kf.encode()).hexdigest()}, 1.3)
    ch.emit(E.MARKS_DERIVED, {"sha256": derived_sha["marks"], "marks": len(marks)}, 0.3)
    ch.emit(E.LEDGER_ROW, {"dollars_so_far": 1.38}, 0.2)
    ch.stage(c.RunStatus.STAYSPUT, 78)
    ch.emit(E.STAYSPUT_DERIVED, {"sha256": derived_sha["mark_tracks"], "mark_tracks": 2}, 14.0)
    ch.emit(E.LEDGER_ROW, {"dollars_so_far": 1.52}, 0.2)
    ch.stage(c.RunStatus.STAYSPUT, 82)
    ch.emit(E.VOICE_DONE, {"role": "pickup", "mode_applied": True, "turns": 1}, 5.3)
    ch.emit(E.NARRATION_DERIVED, {"sha256": derived_sha["narration"], "mode_applied": True}, 0.4)
    ch.emit(E.LEDGER_ROW, {"dollars_so_far": 1.53}, 0.1)
    ch.stage(c.RunStatus.MATCHING, 88)
    ch.emit(E.MATCH_DERIVED, {"sha256": {"matches": derived_sha["matches"], "verdicts": derived_sha["verdicts"]}, "matches": 3, "verdicts": 4}, 0.6)
    ch.stage(c.RunStatus.SHEET, 95)
    ch.emit(E.SHEET_WRITTEN, {"sha256": sheet_sha}, 3.0)
    ch.stage(c.RunStatus.SHEET, 98)
    for row in deletions.rows:
        if row.kind == c.DeletionKind.FILES_API:
            ch.emit(E.FILES_DELETED, {"file_id": row.target, "response": row.api_response, "reason": row.reason}, 0.6)
    ch.emit(E.QUARANTINE_DELETED, {"files": [r.target for r in deletions.rows if r.kind == c.DeletionKind.QUARANTINE]}, 0.3)
    ch.stage(c.RunStatus.DONE, 100)
    ch.emit(E.RUN_COMPLETED, {"status": "done", "api_spend_cents": 153}, 0.05)
    return ch.events


def lane_done(role: str, lane: str, record: c.LaneRecord, frames: list[c.IndexFrame]) -> dict[str, Any]:
    return {
        "role": role,
        "noun": lane,
        "lane_path": record.lane_path,
        "sha256": record.lane_sha256,
        "frames": len(frames),
        "ids": record.ids,
        "status": record.status.value,
        "frames_sent": record.frames_sent,
        "video_frames_processed": record.video_frames_processed,
        "parse_cpu_s": record.parse_cpu_s,
        "dollars": record.dollars,
    }


SHEET_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Mock sheet</title>
<style>body{font-family:'IBM Plex Sans',system-ui,sans-serif;max-width:46em;margin:2em auto;padding:0 1em;line-height:1.5;color:#1d2321;background:#fff}
h1{font-size:1.4em}code{font-family:'IBM Plex Mono',monospace}.page{border-bottom:1px solid #d8d2c4;padding-bottom:2em;margin-bottom:2em;min-height:60vh}</style></head>
<body>
<div class="page"><h1>Walkaround, mock sheet</h1>
<p>This page stands in for the sheet the sheet lane renders (walkaround/sheet.py). The mock server serves it so the sheet page, its frame and
the kiosk scroll can be built without the backend. Nothing on it describes a real car.</p>
<p>run <code>mock-sample-pair</code>; evidence support, not legal advice.</p>
<p>The real page 1 carries the three clocks (sealed at, first model call at, sealed N minutes after the camera time), both originals with their
full SHA-256, the plate line, the privacy line and the five-line note.</p></div>
<div class="page"><h2>Right rear door</h2><p>Pickup left, return right, the marks drawn, "already there", the distance, the Stays Put line and the narration chip.</p></div>
<div class="page"><h2>Method and limits</h2><p>Thresholds with the config hash, the eval numbers with N, what was blurred and what was deleted.</p></div>
</body></html>
"""


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    lanes = write_lanes()
    videos = {role: video_record(role) for role in ROLES}
    for role, v in videos.items():
        put_json(f"videos/{role}.json", v)
    stamps = c.Stamps(manifest_ots="pending", pickup_ots="pending", return_ots="pending", tsr_time=iso(T0 + timedelta(seconds=58)), tsa_serial="0x5c1f", tsa_issuer="freetsa")
    manifest = c.SealManifest(run_id=RUN_ID, files=[c.ManifestEntry(path=f"quarantine/{r}.orig", sha256=videos[r].sha256_server, bytes=videos[r].bytes) for r in ROLES], stamps=stamps, config_hash=CONFIG_HASH)
    genesis = put_json("bundle/manifest.json", manifest)

    tracks = [track_for(role, lanes[role]["car_door"][1]) for role in ROLES]
    keyframes, marks = build_keyframes_and_marks()
    panels, coverage = build_panels(tracks)
    calib = c.Calibration(sigma_frame=0.02, sigma_spark=0.03, sigma_view=0.04, sigma_within=0.05, tau_pre=0.18, tau_eye=0.36, tau_fix=0.08, calibrated=True)
    matches, verdicts = build_match(marks, calib)
    mark_tracks = build_mark_tracks(marks)
    chip = OUT / "chips" / "pickup" / "t1.m4a"
    silent_chip(chip, 1.7)
    narration = c.Narration(
        per_role={
            "pickup": c.NarrationRole(turns=[c.NarrationTurn(turn_id="t1", start_ms=400, end_ms=2100, text="small scuff, right rear door", keywords_hit=["scuff", "door"], pinned_panel=c.PanelId.RIGHT_REAR_DOOR, pin_seconds=1.7, pin_kind=c.PinKind.OVERLAP, chip_path=rel(chip))]),
            "return": c.NarrationRole(turns=[]),
        },
        mode_applied=True,
    )
    claim_img = OUT / "claim" / "claim_1.jpg"
    claim_draw, claim_mask, _ = draw_crop("right_rear_door", "claim", (40, -20), MARKS[("pickup", "right_rear_door")][:1], glare=False, stem="claim/claim_1_full")
    Image.open(claim_draw).resize((960, 645)).save(claim_img, quality=78)
    claim_draw.unlink()
    claim_mask.unlink()
    claim = c.Claim(
        image=c.ClaimImage(sha256=sha(claim_img), received_at=iso(T0 + timedelta(days=2)), w=960, h=645),
        text="scratch, right rear door",
        roi=c.ClaimRoi(box=[430.0, 300.0, 560.0, 380.0], source=c.RoiSource.DRAWN),
        naming=c.ClaimNaming(spark=c.PanelId.RIGHT_REAR_DOOR, sam_votes=[{"noun": "car door", "matches": 1, "c": 0.9}], agreed=True),
        panel_id=c.PanelId.RIGHT_REAR_DOOR,
        side_source="text",
        coverage_state=c.CoverageState.FILMED,
        placement=c.ClaimPlacement(method=c.PlacementMethod.FULL_PANEL, w_claim=[0.31, 0.12], r_claim=0.33),
        size=c.SizeEstimate(inches=1.1, bound_lo=0.9, bound_hi=1.3, ruler=c.RulerKind.PLATE, ruler_source_kf="kf_pickup_right_rear_door", threshold_type=c.ThresholdType.BODY_SCRATCH),
        paragraph="Mock claim: the right rear door was filmed at pickup, and the marked area falls on the scuff seen in that walk.",
    )

    derived_sha = {
        "tracks": put_json("derived/tracks.json", tracks),
        "panels": put_json("derived/panels.json", panels),
        "keyframes": put_json("derived/keyframes.json", keyframes),
        "marks": put_json("derived/marks.json", marks),
        "mark_tracks": put_json("derived/mark_tracks.json", mark_tracks),
        "matches": put_json("derived/matches.json", matches),
        "verdicts": put_json("derived/verdicts.json", verdicts),
        "coverage": put_json("derived/coverage.json", coverage),
        "narration": put_json("derived/narration.json", narration),
        "calibration": put_json("derived/calibration.json", calib),
        "claim": put_json("derived/claim.json", claim),
    }
    end = T0 + timedelta(minutes=9)
    rows = [c.DeletionRecord(target=f"media/{r}.proxy.mp4", kind=c.DeletionKind.FILE, sha256=FAKE_SHA[f"{r}-original"], bytes=3_100_000, deleted_at=iso(T0 + timedelta(minutes=4)), reason="unblurred proxy, deleted at blur.done") for r in ROLES]
    rows += [c.DeletionRecord(target=videos[r].files_api.file_id, kind=c.DeletionKind.FILES_API, deleted_at=iso(end), reason="run ended", api_response=videos[r].files_api.delete_response) for r in ROLES]
    rows += [c.DeletionRecord(target=f"quarantine/{r}.orig", kind=c.DeletionKind.QUARANTINE, sha256=videos[r].sha256_server, bytes=videos[r].bytes, deleted_at=iso(end), reason="run ended") for r in ROLES]
    deletions = c.Deletions(run_id=RUN_ID, rows=rows)
    put_json("deletions.json", deletions)
    (OUT / "sheet").mkdir(parents=True, exist_ok=True)
    (OUT / "sheet" / "sheet.html").write_text(SHEET_HTML, encoding="ascii")
    # Keyed like sheet.write_sheet: run-relative paths. No evidence.pdf, so the pages show "PDF retry" as on a PDF failure.
    sheet_sha = {"sheet/sheet.html": sha(OUT / "sheet" / "sheet.html")}

    events = build_events(genesis, manifest, videos, lanes, derived_sha, marks, deletions, sheet_sha, coverage)
    (OUT / "events.ndjson").write_text("".join(e.to_json() + "\n" for e in events), encoding="ascii")
    run = c.RunManifest(
        run_id=RUN_ID,
        created_at=iso(T0),
        status=c.RunStatus.DONE,
        live_or_replay=c.LiveOrReplay.REPLAY,
        keep=True,
        config_hash=CONFIG_HASH,
        fps=FPS,
        api_spend_cents=153,
        clocks=c.ThreeClocks(sealed_at=iso(T0 + timedelta(seconds=58)), first_model_call_at=iso(T0 + timedelta(seconds=94)), sealed_minutes_after_camera=7, camera_time=iso(CAMERA["pickup"])),
        responses_store=c.ResponsesStore.FALSE_ACCEPTED,
        events_head_hash=events[-1].hash,
        counts=c.RunCounts(panels=22, marks=len(marks), matches=3),
    )
    put_json("run.json", run)
    # The chain must verify the way eventlog.verify_chain does: every hash recomputes from the manifest's sha256.
    prev = genesis
    for i, e in enumerate(events):
        assert e.seq == i and e.prev == prev and e.check(), i
        prev = e.hash
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {len(files)} files, {sum(p.stat().st_size for p in files) // 1024} KB, {len(events)} events to {OUT.relative_to(WEB)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
