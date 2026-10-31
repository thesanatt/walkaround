"""claim (M11, 5.11): the receipt and the privacy order on the photo, the scanner test and the drawn box, the naming
votes, every exit E1 to E8 (E2b as both sides shown), the pickup side's own gates, the door's window cut like the
pickup keyframes', the placement in the whitened frame and the comparison through match.verdict, the second look
lowering only, the renter's tap reusing every earlier answer, and rebuild_claim reproducing the claim with no network
call. Every photo here is synthetic: a red panel with a short dark line drawn on it, a grayscale scan-like copy with an
orange box, and a photo with no car."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from meta_sam_parser import SegmentationMask
from meta_sam_parser._mask_codec import _encode_raster
from PIL import Image

from walkaround import claim, config, geometry, ledger, paths, rules, ruler, sam_client, spark_client
from walkaround import contracts as c
from walkaround.panels import END_ON_FRONT_SIGN

TH = c.load_thresholds()
for _leaf in ("sigma_frame", "sigma_spark"):  # the band at its floor (tau_pre 0.25); D1's sigma_spark sits it at the cap
    TH["matching"].pop(_leaf, None)
W, H = 800, 600
PANEL = (100, 100, 700, 500)  # the trunk lid on the synthetic photo
KF = "pickup_trunk_lid_kf0"
NOW = "2026-10-27T15:00:00Z"
CAMERA = "2026-10-27T14:37:00Z"
PICKUP_SHA = hashlib.sha256(b"pickup walk").hexdigest()


# Photos and masks ---------------------------------------------------------------------------------------------------


def photo(*, gray: bool = False, drawn: tuple[int, int, int, int] | None = None, plate: tuple[int, int, int, int] | None = None, size: tuple[int, int] = (W, H)) -> np.ndarray:
    w, h = size
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (60, 140, 60)  # grass
    x1, y1, x2, y2 = (round(v * w / W) if i % 2 == 0 else round(v * h / H) for i, v in enumerate(PANEL))
    rgb[y1:y2, x1:x2] = (180, 40, 40)  # red paint
    cv2.line(rgb, (round(395 * w / W), round(295 * h / H)), (round(425 * w / W), round(305 * h / H)), (20, 20, 20), 3)  # the mark
    if plate is not None:
        px1, py1, px2, py2 = plate
        rgb[py1:py2, px1:px2] = 250
        for k in range(5):  # characters, so pixelation is visible
            rgb[py1 + 12 : py2 - 12, px1 + 10 + 18 * k : px1 + 18 + 18 * k] = 10
    if gray:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb = np.stack([g, g, g], axis=-1)
    if drawn is not None:
        cv2.rectangle(rgb, drawn[:2], drawn[2:], (255, 120, 0), 4)
    return rgb


def landscape(size: tuple[int, int] = (W, H)) -> np.ndarray:
    w, h = size
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[: h // 2] = (90, 150, 230)  # sky
    rgb[h // 2 :] = (70, 150, 60)  # field
    return rgb


def jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def rect_mask(box: tuple[int, int, int, int], size: tuple[int, int] = (W, H)) -> np.ndarray:
    m = np.zeros((size[1], size[0]), bool)
    m[box[1] : box[3], box[0] : box[2]] = True
    return m


_ENCODED: dict[str, tuple[tuple[int, int, int, int], str]] = {}  # the pure-Python codec is slow; one encode per mask


def detection(mask: np.ndarray, object_id: str, confidence: float | None) -> sam_client.Detection:
    key = sha_of(mask)
    if key not in _ENCODED:
        ys, xs = np.nonzero(mask)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        local = mask[y1:y2, x1:x2].astype(np.uint8)
        _ENCODED[key] = ((x1, y1, x2, y2), _encode_raster(bytes(local.flatten().tolist()), x2 - x1, y2 - y1))
    (x1, y1, x2, y2), payload = _ENCODED[key]
    seg = SegmentationMask(encoding="one_bit", payload=payload, width=x2 - x1, height=y2 - y1)
    return sam_client.Detection(object_id=object_id, frame_index=None, box=sam_client.Box(x1, y1, x2, y2), mask=seg, confidence=confidence)


def sha_of(image: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def default_dets() -> dict[str, list[tuple[np.ndarray, float | None]]]:
    return {"trunk lid": [(rect_mask(PANEL), 0.9)], "car": [(rect_mask((50, 50, 750, 550)), 0.95)]}


class FakeSam:
    """Stands in for sam_client.segment_image: answers each concept with the given masks and records every call."""

    def __init__(self, dets: dict[str, list[tuple[np.ndarray, float | None]]] | None = None) -> None:
        self.dets = default_dets() if dets is None else dets
        self.calls: list[dict[str, Any]] = []
        self.fail: set[str] = set()
        self._encoded: dict[str, list[sam_client.Detection]] = {}  # the pure-Python codec is slow; encode once

    async def segment_image(self, image, concept: str, *, run_id, role=None, tag="", include_confidence=True):
        image = np.asarray(image)
        self.calls.append({"concept": concept, "run_id": run_id, "role": role, "tag": tag, "sha": sha_of(image), "image": image.copy()})
        run = sam_client.SegmentationRun(concept=concept, media="image", media_width=int(image.shape[1]), media_height=int(image.shape[0]))
        run.response_id = f"resp_{concept.replace(' ', '_')}_{len(self.calls)}"
        if concept in self.fail:
            run.outcome = "incomplete"
            raise sam_client.SamRequestError(f"The {concept!r} image check ended incomplete", run)
        name = "box" if concept.startswith("<0f>") else concept  # an image box prompt on the claim mark
        if name not in self._encoded:
            self._encoded[name] = [detection(mask, str(i), conf) for i, (mask, conf) in enumerate(self.dets.get(name, []))]
        found = list(self._encoded[name])
        if found:
            run.frames[0] = found
        return run

    def concepts(self) -> list[str]:
        return [call["concept"] for call in self.calls]


def completed(payload: Any, *, rid: str) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "id": rid,
        "status": "completed",
        "incomplete_details": None,
        "error": None,
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 1400, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 500, "output_tokens_details": {"reasoning_tokens": 400}},
    }


INCOMPLETE = "incomplete"


class FakeSpark:
    """A Responses client answering by the schema name the request asks for."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.requests: list[tuple[str, dict]] = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **request: Any) -> dict:
        name = request["text"]["format"]["name"]
        self.requests.append((name, request))
        payload = self.answers[name]
        if isinstance(payload, list):
            payload = payload.pop(0) if len(payload) > 1 else payload[0]
        rid = f"resp_{name}_{len(self.requests)}"
        if payload == INCOMPLETE:
            data = completed("", rid=rid)
            data["status"] = "incomplete"
            data["incomplete_details"] = {"reason": "max_output_tokens"}
            return data
        return completed(copy.deepcopy(payload), rid=rid)

    def names(self) -> list[str]:
        return [name for name, _ in self.requests]


def naming(panel: str = "trunk lid", side: str = "none", position: str = "rear") -> dict:
    return {"panel": panel, "side": side, "position": position, "front_direction": "unknown", "panel_bbox": None, "context": "the rear of a red car", "landmarks": []}


def marks_answer(points: tuple[tuple[int, int], ...] = ((512, 500),), *, kind: str = "scratch", conf: str = "high", glare: bool = False, half: tuple[int, int] = (31, 25)) -> dict:
    hx, hy = half
    marks = [
        {"type": kind, "point": {"x": x, "y": y}, "bbox": {"x_min": x - hx, "y_min": y - hy, "x_max": x + hx, "y_max": y + hy}, "size_class": "S", "confidence": conf, "note": "a short dark line"}
        for x, y in points
    ]
    return {"marks": marks, "conditions": {"wet": False, "dirty": False, "glare": glare, "dark": False}, "landmarks": [], "neighbour_car_visible": False}


def answers(**over: Any) -> dict[str, Any]:
    base = {
        "claim_name_v1": naming(),
        "claim_box_v1": {"found": False, "bbox": None},
        "marks_v1": marks_answer(),
        "look_v1": {"verdict": "same", "reason": "the same short line"},
    }
    base.update(over)
    return base


# The run a claim is placed against ----------------------------------------------------------------------------------


def front_sign(pid: c.PanelId) -> int | None:
    if pid in END_ON_FRONT_SIGN:
        return END_ON_FRONT_SIGN[pid]
    if pid.value.startswith("right_"):
        return 1
    if pid.value.startswith("left_"):
        return -1
    return None


def claim_w(box: tuple[int, int, int, int] = (0, 0, W, H), grid: tuple[int, int] = (512, 500), panel: tuple[int, int, int, int] = PANEL, sign: int = -1, mask: np.ndarray | None = None) -> tuple[float, float]:
    """The whitened position the claim code computes for a mark Spark puts at `grid` on the crop `box`."""
    cw, ch = box[2] - box[0], box[3] - box[1]
    sw, sh = spark_client.sent_size(cw, ch)
    px = spark_client.grid_to_pixels(grid[0], grid[1], sw, sh)
    pc = geometry.sent_to_crop(px, (sw, sh), (cw, ch))
    return geometry.whiten((box[0] + pc[0], box[1] + pc[1]), geometry.moments_from_mask(rect_mask(panel) if mask is None else mask), sign)


def make_run(run_id: str, *, states: dict[c.PanelId, c.CoverageState] | None = None, offsets: tuple[tuple[float, float], ...] = (), kind: c.MarkKind = c.MarkKind.SCRATCH, coverage: bool = True, closed: bool = False) -> None:
    c.write_json(paths.run_json(run_id), c.RunManifest(run_id=run_id, created_at=NOW))
    states = states if states is not None else {c.PanelId.TRUNK_LID: c.CoverageState.FILMED}
    if coverage:
        per = {pid: c.PanelCoverage(state=st, seconds=6.6 if st == c.CoverageState.FILMED else 1.2, best_kf=KF if pid == c.PanelId.TRUNK_LID else None) for pid, st in states.items()}
        c.write_json(paths.coverage_json(run_id), c.Coverage(per_role={"pickup": per}))
    c.write_json(paths.panels_json(run_id), [c.Panel(panel_id=pid, noun=c.PANEL_NOUN[pid], front_sign=front_sign(pid)) for pid in c.PanelId])
    crop = c.Crop(path=f"crops/pickup/{KF}.jpg", w=400, h=300, scale=1.0, dx=0.0, dy=0.0)
    Image.fromarray(np.full((300, 400, 3), 150, np.uint8)).save(paths.crop_path(run_id, "pickup", KF), "JPEG")
    score = c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1)
    kf = c.Keyframe(kf_id=KF, role="pickup", panel_id=c.PanelId.TRUNK_LID, track_id="t1", f_proxy=219, pts=14.6, score=score, aspect_ratio=1.5, box_src=[400.0, 300.0, 1500.0, 1000.0], crop=crop, front_sign=-1, marks_status=c.MarksStatus.COMPLETE)
    c.write_json(paths.keyframes_json(run_id), [kf])
    u, v = claim_w()
    marks = []
    for i, (du, dv) in enumerate(offsets, 1):
        w = [u + du, v + dv]
        marks.append(
            c.Mark(
                mark_id=f"{KF}_m{i}", role="pickup", panel_id=c.PanelId.TRUNK_LID, kf_id=KF, source=c.MarkSource.SPARK_BOX, type=kind, confidence=c.Confidence.HIGH,
                size_class=c.SizeClass.S, bbox_crop=[100.0 + 40 * i, 100.0, 130.0 + 40 * i, 110.0], point_crop=[115.0 + 40 * i, 105.0], w=w, r=math.hypot(*w), size_px_src=30.0,
            )
        )
    c.write_json(paths.marks_json(run_id), marks)
    probe = c.FfprobeInfo(codec="h264", container="mp4", width=1920, height=1080, display_width=1920, display_height=1080, fps_avg=30.0, duration_s=20.0, creation_time=CAMERA)
    c.write_json(paths.video_json(run_id, "pickup"), c.VideoRecord(role="pickup", original_filename="pickup.mp4", bytes=1, sha256_server=PICKUP_SHA, ffprobe=probe))
    if closed:
        paths.manifest_run_json(run_id).write_text("{}\n")


def upload(run_id: str, rgb: np.ndarray, n: int = 1) -> Path:
    target = paths.claim_image_path(run_id, f"claim_{n}")
    target.write_bytes(jpeg(rgb))
    return target


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    yield root
    spark_client.set_client(None)


def install(monkeypatch: pytest.MonkeyPatch, sam: FakeSam | None = None, spark: FakeSpark | None = None) -> tuple[FakeSam, FakeSpark]:
    sam = sam or FakeSam()
    spark = spark or FakeSpark(answers())
    monkeypatch.setattr(sam_client, "segment_image", sam.segment_image)
    spark_client.set_client(spark)
    return sam, spark


def derive(run_id: str, rgb: np.ndarray, *, n: int = 1, text: str = "", tire: str | None = None, panel: str | None = None) -> c.Claim:
    image = upload(run_id, rgb, n)
    return asyncio.run(claim.derive_claim(run_id, image=image, text=text, tire_code=tire, th=TH, panel_id=panel))


# Words, saturation, the drawn box ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, side, position",
    [
        ("scratch, passenger-side front door", c.Side.RIGHT, c.Position.FRONT),
        ("Driver's rear door dent", c.Side.LEFT, c.Position.REAR),
        ("RR door", c.Side.RIGHT, c.Position.REAR),
        ("LT RR DOOR", c.Side.LEFT, c.Position.REAR),  # an explicit side beats the corner; RR reads as rear
        ("RT FRT FENDER", c.Side.RIGHT, c.Position.FRONT),
        ("drivers side rear door", c.Side.LEFT, c.Position.REAR),
        ("LH front door", c.Side.LEFT, c.Position.FRONT),
        ("chip on the back bumper", None, c.Position.REAR),
        ("left and right doors", None, None),
        ("", None, None),
    ],
)
def test_the_words_on_the_bill_name_side_and_position(text: str, side, position) -> None:
    assert claim.side_from_text(text, TH) == side
    assert claim.position_from_text(text) == position


def test_passenger_side_follows_the_threshold() -> None:
    th = copy.deepcopy(TH)
    th["claim"]["passenger_side_is"] = "left"
    assert claim.side_from_text("passenger door", th) == c.Side.LEFT
    assert claim.side_from_text("driver door", th) == c.Side.RIGHT


def test_scanner_style_is_a_saturation_test() -> None:
    assert claim.mean_saturation(photo()) > TH["claim"]["scanner_style_saturation_max"]
    assert claim.mean_saturation(photo(gray=True)) < TH["claim"]["scanner_style_saturation_max"]
    dark = np.zeros((50, 50, 3), np.uint8)
    dark[..., 0] = 3  # near-black JPEG noise is left out
    assert claim.mean_saturation(dark) == 0.0


def test_the_drawn_box_is_measured_from_the_pixels_near_sparks_box() -> None:
    rect = (300, 200, 500, 380)
    img = photo(gray=True, drawn=rect)
    roi, stats = claim.verify_drawn_box(img, (312, 214, 516, 392), TH)  # Spark a few pixels off the line, as on D1
    assert roi is not None and stats["perimeter_saturated_share"] >= 0.3 and stats["inside_saturated_share"] <= 0.1
    assert roi == pytest.approx([rect[0] - 2, rect[1] - 2, rect[2] + 3, rect[3] + 3], abs=2)
    assert claim.verify_drawn_box(photo(gray=True), (312, 214, 516, 392), TH)[0] is None  # nothing drawn
    filled = photo(gray=True)
    filled[200:380, 300:500] = (255, 120, 0)
    assert claim.verify_drawn_box(filled, (300, 200, 500, 380), TH)[0] is None  # a coloured patch is not an outline


def test_border_share() -> None:
    assert claim.border_share(rect_mask(PANEL)) == 0.0
    assert claim.border_share(rect_mask((0, 100, 700, 500))) > TH["claim"]["full_panel_border_max"]
    assert claim.border_share(np.zeros((10, 10), bool)) == 1.0


def test_decide_names_every_answer_it_needs() -> None:
    a = claim.ClaimAnswers(claim_id="claim_1", run_id="r", image_sha256="0" * 64, received_at=NOW, w=W, h=H, work_w=W, work_h=H, blurred_sha256="0" * 64)
    view = SimpleNamespace(run_id="r")
    with pytest.raises(claim.NeedAnswers) as need:
        claim.decide(view, a, photo(gray=True), TH)
    keys = [x.key for x in need.value.asks]
    assert keys[:2] == ["claim_name", "claim_box"] and keys[2:] == [n.value for n in claim.VOTE_NOUNS]


# Receipt, privacy, files ---------------------------------------------------------------------------------------------


def test_receipt_privacy_and_files(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    plate_box = (480, 400, 600, 460)
    dets = default_dets()
    dets["license plate"] = [(rect_mask(plate_box), 0.9)]
    line = np.zeros((H, W), np.uint8)
    cv2.line(line, (400, 300), (420, 300), 1, 3)
    dets["box"] = [(line.astype(bool), 0.8)]  # SAM's mask of the mark from the box prompt on Spark's box
    sam, spark = install(monkeypatch, FakeSam(dets))
    original = photo(plate=plate_box)
    raw = jpeg(original)
    image = paths.claim_image_path("r", "claim_1")
    image.write_bytes(raw)
    out = asyncio.run(claim.derive_claim("r", image=image, text="scratch on the trunk", tire_code=None, th=TH))

    assert out.image.sha256 == hashlib.sha256(raw).hexdigest() and (out.image.w, out.image.h) == (W, H) and not out.image.low_res
    assert not image.exists()  # the photo as uploaded is gone; only the pixelated copy stays
    blurred_path = paths.claim_blurred_path("r", "claim_1")
    blurred = np.asarray(Image.open(blurred_path).convert("RGB"))
    region = (slice(plate_box[1], plate_box[3]), slice(plate_box[0], plate_box[2]))
    assert np.abs(blurred[region].astype(int) - original[region].astype(int)).mean() > 25  # the characters are gone
    rec = claim.ClaimAnswers.model_validate_json(claim.answers_path("r", "claim_1").read_text())
    assert rec.blur.plate_ruler_px == pytest.approx(120.0) and rec.blur.plate_tilt_hw == pytest.approx(0.5)  # measured before the pixelation
    assert rec.deletions and rec.deletions[0].target == "claim/claim_1.jpg" and rec.deletions[0].sha256 == out.image.sha256
    assert rec.blurred_sha256 == hashlib.sha256(blurred_path.read_bytes()).hexdigest()

    # the two privacy checks saw the unblurred copy first; every later model call saw the stored pixelated bytes
    assert sam.concepts()[:2] == ["license plate", "face"] or sam.concepts()[:2] == ["face", "license plate"]
    assert all(call["sha"] != sha_of(blurred) for call in sam.calls[:2])
    assert all(call["sha"] == sha_of(blurred) for call in sam.calls[2:]) and len(sam.calls) == 2 + len(claim.VOTE_NOUNS) + 1
    assert sam.calls[-1]["concept"].startswith("<0f>1<|box;")  # the plate qualified, so the mark's own mask was asked
    assert {call["run_id"] for call in sam.calls} == {"r.claim_1"} and {call["role"] for call in sam.calls} == {None}
    sent = spark.requests[0][1]["input"][1]["content"][1]["image_url"]
    seen = np.asarray(Image.open(io.BytesIO(base64.b64decode(sent.split(",", 1)[1]))).convert("RGB"))
    assert np.abs(seen[region].astype(int) - blurred[region].astype(int)).mean() < 6
    rows = ledger.rows(ledger.LEDGER_PATH)
    assert rows and {r["run_id"] for r in rows} == {"r.claim_1"}  # claim calls never touch the run's own totals

    assert out.claim_id == "claim_1" and paths.claim_record_path("r", "claim_1").read_bytes() == c.json_bytes(out)
    # 5.12 on the claim photo: the plate measured before the pixelation, the mark from its own SAM mask
    expected = ruler.extent_from_mask(dets["box"][0][0]).long_px / 10.0  # the plate is 120 px long: 10 px per inch
    assert out.size.ruler == c.RulerKind.PLATE and out.size.inches == pytest.approx(expected)
    assert out.size.threshold_type == c.ThresholdType.BODY_SCRATCH and ruler.compare_hertz(out.size, TH) == "under"
    assert "(ruler: the license plate, 12 in); under Hertz's example threshold of 3.5 in for a body scratch along its long axis (REV 7.2025, illustrative only)." in out.paragraph
    assert paths.claim_json("r").read_bytes() == paths.claim_record_path("r", "claim_1").read_bytes()


def test_a_closed_run_keeps_its_derived_files(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),), closed=True)
    install(monkeypatch)
    out = derive("r", photo())
    assert out.exit is None and paths.claim_record_path("r", "claim_1").is_file()
    assert not paths.claim_json("r").is_file()


# The comparison ---------------------------------------------------------------------------------------------------------


def test_already_there_with_the_dated_pickup_frame(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    sam, spark = install(monkeypatch)
    out = derive("r", photo())
    assert out.exit is None and out.panel_id == c.PanelId.TRUNK_LID and out.coverage_state == c.CoverageState.FILMED
    assert out.naming.agreed and out.naming.spark == c.PanelId.TRUNK_LID and len(out.naming.sam_votes) == len(claim.VOTE_NOUNS)
    assert [row["noun"] for row in out.naming.sam_votes if row["winner"]] == ["trunk lid"]
    assert out.roi.source == c.RoiSource.WHOLE and not out.scanner_style
    assert out.placement.method == c.PlacementMethod.FULL_PANEL and out.placement.r_claim == TH["claim"]["r_claim_full_panel"]
    assert out.placement.w_claim == pytest.approx(list(claim_w()), abs=1e-6)
    assert out.match.reasons == ["L3"] and out.match.kf_pair == [KF, "claim_1"]
    pair = out.match.pairs[0]
    assert pair.pickup_mark == f"{KF}_m1" and pair.d_pos == pytest.approx(0.05, abs=1e-6)
    assert pair.tau_pre_r == pytest.approx(match_band(0.05) + TH["claim"]["r_claim_full_panel"], abs=1e-6)
    assert "Already there at pickup: the claim mark sits 0.05 units (about 2 cm) from a scratch on the pickup frame at 0:15 (camera time 2026-10-27 14:37:15 UTC)." in out.paragraph
    assert "Trunk lid. Filmed at pickup for 6.6 s" in out.paragraph and PICKUP_SHA in out.paragraph and out.image.sha256 in out.paragraph
    # no plate on the photo: no ruler, so the mark's own mask is never asked for and never blamed
    assert out.size.refused_reason == ruler.no_ruler_reason(TH) and "Size not stated: no ruler within 0.6 m of this mark." in out.paragraph
    assert not any(call["concept"].startswith("<0f>") for call in sam.calls)
    assert spark.names() == ["claim_name_v1", "marks_v1", "look_v1"]


def match_band(du: float) -> float:
    u, v = claim_w()
    r_mean = 0.5 * (math.hypot(u, v) + math.hypot(u + du, v))
    from walkaround import match

    cal = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=TH)
    return match.tau_r(cal.tau_pre, r_mean, TH)


def test_the_second_look_can_only_lower(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    install(monkeypatch, spark=FakeSpark(answers(look_v1={"verdict": "different", "reason": "not the same line"})))
    out = derive("r", photo())
    assert out.match.reasons == ["L4", "L3", "K7"]
    assert out.paragraph.count("Needs your eye: a scratch on the pickup frame at 0:15") == 1 and "second look doubts" in out.paragraph


def test_needs_your_eye_inside_the_band(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.45, 0.0),))
    _, spark = install(monkeypatch)
    out = derive("r", photo())
    assert out.match.reasons == ["L4", "K1"] and "inside the band where Walkaround does not decide" in out.paragraph
    assert "look_v1" not in spark.names()  # the second look is asked only of an already-there pair


def test_no_pickup_mark_at_that_spot(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((1.2, 0.0),))
    install(monkeypatch)
    far = derive("r", photo())
    assert far.match.reasons == ["L5"] and far.match.unmatched_return == ["claim_1_m1"]
    assert "No pickup mark at that spot. The nearest pickup mark, a scratch, is 1.20 units (about 42 cm) away. Look yourself." in far.paragraph
    make_run("r2")
    none = derive("r2", photo())
    assert none.match.reasons == ["L2"] and none.match.pairs == [] and "No pickup mark was found on this panel." in none.paragraph


def test_a_far_pickup_marks_caps_never_lower_the_claim(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((1.2, 0.0),))
    marks = c.read_json_list(paths.marks_json("r"), c.Mark)
    c.write_json(paths.marks_json("r"), [m.model_copy(update={"caps": ["K2"], "single_frame": True}) for m in marks])
    install(monkeypatch)
    out = derive("r", photo())
    assert out.match.reasons == ["L5"]  # as match.verdicts_for: a pair beyond tau_eye_r passes no partner
    assert "No pickup mark at that spot. The nearest pickup mark, a scratch, is 1.20 units (about 42 cm) away. Look yourself." in out.paragraph


def test_no_pair_still_names_the_nearest_pickup_mark(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((1.2, 0.0), (-1.25, 0.0)))  # both far and about as far: the ratio test fails
    install(monkeypatch)
    out = derive("r", photo())
    assert out.match.reasons == ["L2"] and out.match.pairs == []
    assert "The nearest pickup mark, a scratch, is 1.20 units (about 42 cm) away." in out.paragraph and "No pickup mark was found" not in out.paragraph


def test_a_mark_of_unsure_type_reads_as_such(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.45, 0.0),), kind=c.MarkKind.UNSURE)
    install(monkeypatch)
    out = derive("r", photo())
    assert "a mark of unsure type" in out.paragraph and "a unsure" not in out.paragraph


def test_two_pickup_marks_about_as_close_are_never_more_than_needs_your_eye(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0), (-0.07, 0.0)))  # the ratio test fails
    install(monkeypatch)
    out = derive("r", photo())
    assert out.match.reasons[0] == "L4" and "K1" in out.match.reasons
    assert "two pickup marks lie about as close to the claim mark" in out.paragraph


# Exits ------------------------------------------------------------------------------------------------------------------


def test_E1_no_car_panel(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r")
    sam, spark = install(monkeypatch, FakeSam({}), FakeSpark(answers(claim_name_v1=naming("none", "none", "none"))))
    out = derive("r", landscape())
    assert out.exit == c.ClaimExit.E1 and out.panel_id is None and out.match is None
    assert "does not show a car panel" in out.paragraph and "SAM 3.1 found no car in it." in out.paragraph
    assert spark.names() == ["claim_name_v1"] and len(sam.calls) == 2 + len(claim.VOTE_NOUNS)


def test_E2_naming_disagreed_then_the_tap_reuses_every_answer(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    dets = default_dets()
    dets["bumper"] = [(rect_mask((80, 80, 720, 520)), 0.95)]  # holds the centre with a larger area x c
    sam, spark = install(monkeypatch, FakeSam(dets))
    first = derive("r", photo())
    assert first.exit == c.ClaimExit.E2 and not first.naming.agreed and first.panel_id is None
    assert "Muse Spark named the trunk lid; the SAM 3.1 vote named the rear bumper." in first.paragraph
    calls_before, sparks_before = len(sam.calls), len(spark.requests)
    tapped = derive("r", photo(), n=2, panel="trunk_lid")
    assert tapped.panel_id == c.PanelId.TRUNK_LID and tapped.exit is None and tapped.match.reasons == ["L3"]
    assert tapped.paragraph.startswith("Panel picked by you: the trunk lid. Trunk lid. Filmed at pickup")
    assert len(sam.calls) == calls_before  # the same photo bytes: privacy checks and the vote are reused
    assert spark.names()[sparks_before:] == ["marks_v1", "look_v1"]
    rec = claim.ClaimAnswers.model_validate_json(claim.answers_path("r", "claim_2").read_text())
    assert rec.reused_from == "claim_1" and rec.panel_id == c.PanelId.TRUNK_LID


def test_a_twin_mask_places_the_panel_after_the_tap(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    hood_only = {"hood": [(rect_mask(PANEL), 0.9)], "car": [(rect_mask((50, 50, 750, 550)), 0.95)]}  # SAM called the trunk lid a hood (10/31 smoke)
    install(monkeypatch, FakeSam(hood_only))
    first = derive("r", photo())
    assert first.exit == c.ClaimExit.E2 and "the SAM 3.1 vote named the hood (the same shape from the other end)." in first.paragraph
    tapped = derive("r", photo(), n=2, panel="trunk_lid")
    assert tapped.placement.method == c.PlacementMethod.FULL_PANEL and tapped.match.reasons == ["L3"]


def test_an_unfinished_vote_never_counts_as_agreement(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r")
    sam = FakeSam()
    sam.fail = {"fender"}
    install(monkeypatch, sam)
    out = derive("r", photo())
    assert out.exit == c.ClaimExit.E2 and "1 of the SAM 3.1 requests did not finish" in out.paragraph
    assert [row["status"] for row in out.naming.sam_votes if row["noun"] == "fender"] == ["incomplete"]


def test_a_bill_and_a_photo_that_disagree_on_the_side_show_both(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", states={c.PanelId.LEFT_FRONT_DOOR: c.CoverageState.FILMED, c.PanelId.RIGHT_FRONT_DOOR: c.CoverageState.FILMED})
    door = {"car door": [(rect_mask(PANEL), 0.9)], "car": [(rect_mask((50, 50, 750, 550)), 0.95)]}
    install(monkeypatch, FakeSam(door), FakeSpark(answers(claim_name_v1=naming("car door", "left", "front"))))
    clash = derive("r", photo(), text="scratch, passenger-side door")
    assert clash.side_source == "both_shown" and clash.panel_id is None
    assert "The photo shows a car door, but the words on the bill say the right side and Muse Spark sees the left side." in clash.paragraph
    agree = derive("r", photo(), n=2, text="driver door")
    assert agree.panel_id == c.PanelId.LEFT_FRONT_DOOR and agree.side_source == "text"


def test_E2b_both_sides_shown_and_the_side_from_the_text(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", states={c.PanelId.LEFT_FRONT_DOOR: c.CoverageState.FILMED, c.PanelId.RIGHT_FRONT_DOOR: c.CoverageState.NOT_FILMED})
    door = {"car door": [(rect_mask(PANEL), 0.9)], "car": [(rect_mask((50, 50, 750, 550)), 0.95)]}
    install(monkeypatch, FakeSam(door), FakeSpark(answers(claim_name_v1=naming("car door", "none", "front"))))
    both = derive("r", photo())
    assert both.side_source == "both_shown" and both.panel_id is None and both.exit is None
    assert "Left front door: filmed at pickup for 6.6 s. Right front door: not filmed at pickup. Pick one." in both.paragraph
    told = derive("r", photo(), n=2, text="scratch, passenger-side door")
    assert told.panel_id == c.PanelId.RIGHT_FRONT_DOOR and told.side_source == "text" and told.exit == c.ClaimExit.E3


@pytest.mark.parametrize(
    "state, exit_, words",
    [
        (c.CoverageState.NOT_FILMED, c.ClaimExit.E3, "Trunk lid. Not filmed at pickup"),
        (c.CoverageState.NOT_SEEN, c.ClaimExit.E3, "Trunk lid. Not filmed at pickup"),
        (c.CoverageState.GLIMPSED, c.ClaimExit.E4, "Trunk lid. Only glimpsed at pickup (1.2 s), pickup frame at 0:15 of the pickup video"),
        (c.CoverageState.NOT_RUN, None, "The pickup run did not finish checking this panel"),
    ],
)
def test_coverage_exits(runs: Path, monkeypatch: pytest.MonkeyPatch, state, exit_, words: str) -> None:
    make_run("r", states={c.PanelId.TRUNK_LID: state})
    _, spark = install(monkeypatch)
    out = derive("r", photo())
    assert out.exit == exit_ and out.coverage_state == state and out.match is None and words in out.paragraph
    assert spark.names() == ["claim_name_v1"]  # nothing is grounded on a panel the pickup does not show


@pytest.mark.parametrize(
    "change, words",
    [
        ("not_run", "The pickup marks of this panel were not read, so nothing was compared."),
        ("glare", "The pickup frame shows glare, water or dirt, so Walkaround does not compare marks on it. Look at that frame yourself."),
        ("truncated", "The mark search on the pickup frames of this panel was cut short, so nothing was compared."),
        ("blurry", "The pickup frames of this panel are too blurry to compare marks on."),
        ("unplaced", "1 pickup mark of this panel could not be placed in its frame"),
        ("side", "The pickup walk could not tell which side this panel is on"),
        ("no_keyframe", "The pickup walk kept no frame of this panel to search for marks"),
    ],
)
def test_pickup_marks_that_prove_nothing_are_never_no_pickup_mark(runs: Path, monkeypatch: pytest.MonkeyPatch, change: str, words: str) -> None:
    make_run("r", offsets=((0.05, 0.0),) if change == "unplaced" else ())
    kfs = c.read_json_list(paths.keyframes_json("r"), c.Keyframe)
    update = {
        "not_run": {"marks_status": c.MarksStatus.NOT_RUN},
        "glare": {"conditions": c.Conditions(glare=True)},
        "truncated": {"marks_status": c.MarksStatus.TRUNCATED},
        "blurry": {"score": kfs[0].score.model_copy(update={"sharpness": 0.2})},
    }.get(change, {})
    c.write_json(paths.keyframes_json("r"), [] if change == "no_keyframe" else [k.model_copy(update=update) for k in kfs])
    if change == "not_run":
        paths.marks_json("r").unlink()  # a claim sent before the marks stage
    if change == "unplaced":
        marks = c.read_json_list(paths.marks_json("r"), c.Mark)
        c.write_json(paths.marks_json("r"), [m.model_copy(update={"w": None, "r": None}) for m in marks])
    if change == "side":
        panels = c.read_json_list(paths.panels_json("r"), c.Panel)
        c.write_json(paths.panels_json("r"), [p.model_copy(update={"side_uncertain": p.panel_id == c.PanelId.TRUNK_LID}) for p in panels])
    _, spark = install(monkeypatch)
    out = derive("r", photo())
    assert out.exit is None and out.match is None and out.placement.method == c.PlacementMethod.NONE
    assert words in out.paragraph and "No pickup mark" not in out.paragraph
    assert spark.names() == ["claim_name_v1"]  # nothing is grounded when nothing can be compared


def test_a_top_up_frame_prints_its_own_clip_and_time(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    kfs = c.read_json_list(paths.keyframes_json("r"), c.Keyframe)
    c.write_json(paths.keyframes_json("r"), [k.model_copy(update={"role": "topup_1", "pts": 3.0}) for k in kfs])
    marks = c.read_json_list(paths.marks_json("r"), c.Mark)
    c.write_json(paths.marks_json("r"), [m.model_copy(update={"role": "topup_1"}) for m in marks])
    cov = c.read_json(paths.coverage_json("r"), c.Coverage)
    c.write_json(paths.coverage_json("r"), c.Coverage(per_role={"topup_1": cov.per_role["pickup"]}))
    probe = c.FfprobeInfo(codec="h264", container="mp4", width=1920, height=1080, display_width=1920, display_height=1080, fps_avg=30.0, duration_s=8.0, creation_time="2026-10-27T15:17:00Z")
    c.write_json(paths.video_json("r", "topup_1"), c.VideoRecord(role="topup_1", original_filename="topup.mp4", bytes=1, sha256_server="1" * 64, ffprobe=probe))
    install(monkeypatch)
    out = derive("r", photo())
    assert "Filmed at pickup for 6.6 s, pickup frame at 0:03 of top-up clip 1 (camera time 2026-10-27 15:17:03 UTC)." in out.paragraph
    assert "from a scratch on the pickup frame at 0:03 of top-up clip 1 (camera time 2026-10-27 15:17:03 UTC)." in out.paragraph
    assert f"Pickup video sha256 {PICKUP_SHA}. Top-up clip 1 sha256 {'1' * 64}." in out.paragraph and "14:37" not in out.paragraph


WINDOW = (140, 100, 660, 250)  # the glass in the top of the door on the synthetic photo
DOOR_KF = "pickup_left_front_door_kf1"


def door_run(run_id: str, *, window_subtracted: bool) -> None:
    """A pickup walk whose left front door keyframe was scored with or without its window cut out, with one pickup
    mark 0.05 units from where the claim code places the photo's mark on the door with its window cut out."""
    make_run(run_id, states={c.PanelId.LEFT_FRONT_DOOR: c.CoverageState.FILMED})
    score = c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1)
    kf = c.Keyframe(kf_id=DOOR_KF, role="pickup", panel_id=c.PanelId.LEFT_FRONT_DOOR, track_id="t2", f_proxy=60, pts=4.0, score=score, aspect_ratio=1.5, box_src=[0.0, 0.0, 900.0, 600.0], front_sign=-1, window_subtracted=window_subtracted, marks_status=c.MarksStatus.COMPLETE)
    c.write_json(paths.keyframes_json(run_id), [kf])
    u, v = claim_w(mask=rect_mask(PANEL) & ~rect_mask(WINDOW))
    mark = c.Mark(mark_id=f"{DOOR_KF}_m1", role="pickup", panel_id=c.PanelId.LEFT_FRONT_DOOR, kf_id=DOOR_KF, source=c.MarkSource.SPARK_BOX, type=c.MarkKind.SCRATCH, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S, w=[u + 0.05, v], r=math.hypot(u + 0.05, v), size_px_src=30.0)
    c.write_json(paths.marks_json(run_id), [mark])


def door_dets() -> dict[str, list[tuple[np.ndarray, float | None]]]:
    return {"car door": [(rect_mask(PANEL), 0.9)], "car window": [(rect_mask(WINDOW), 0.9)], "car": [(rect_mask((50, 50, 750, 550)), 0.95)]}


def sam_answer(concept: str, masks: list[np.ndarray]) -> claim.SamAnswer:
    dets = [detection(m, str(i), 0.9) for i, m in enumerate(masks)]
    return claim.SamAnswer(
        concept=concept,
        detections=[
            claim.SamDetection(object_id=d.object_id, box=[d.box.left, d.box.top, d.box.right, d.box.bottom], c=d.confidence, mask_encoding=d.mask.encoding, mask_payload=d.mask.payload, mask_w=d.mask.width, mask_h=d.mask.height)
            for d in dets
        ],
    )


def test_cut_windows_follows_the_panels_rule() -> None:
    door = rect_mask(PANEL)
    cut, done = claim.cut_windows(sam_answer("car window", [rect_mask(WINDOW)]), door, claim._Masks((W, H)), TH)
    assert done and np.array_equal(cut, door & ~rect_mask(WINDOW))
    tiny = rect_mask((380, 280, 420, 320))  # under panels' 2% area floor: a stray detection cuts nothing
    away = rect_mask((710, 10, 790, 90))  # outside the door
    kept, done = claim.cut_windows(sam_answer("car window", [tiny, away]), door, claim._Masks((W, H)), TH)
    assert not done and np.array_equal(kept, door)


def test_a_door_claim_cuts_the_window_like_the_pickup_keyframes(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    door_run("r", window_subtracted=True)
    install(monkeypatch, FakeSam(door_dets()), FakeSpark(answers(claim_name_v1=naming("car door", "left", "front"))))
    out = derive("r", photo())
    assert out.panel_id == c.PanelId.LEFT_FRONT_DOOR and out.exit is None and out.match.reasons == ["L3"]
    assert out.placement.w_claim == pytest.approx(list(claim_w(mask=rect_mask(PANEL) & ~rect_mask(WINDOW))), abs=1e-6)
    assert out.match.pairs[0].d_pos == pytest.approx(0.05, abs=1e-6)


def test_a_door_whose_window_was_cut_on_one_frame_only_is_E5(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    door_run("r", window_subtracted=False)  # G9: the pickup keyframe kept its window, the photo's door did not
    install(monkeypatch, FakeSam(door_dets()), FakeSpark(answers(claim_name_v1=naming("car door", "left", "front"))))
    out = derive("r", photo())
    assert out.exit == c.ClaimExit.E5 and out.match is None and out.placement.method == c.PlacementMethod.NONE
    assert "window cut out on one of the two frames and not on the other" in out.paragraph


def test_a_door_whose_window_check_did_not_finish_is_E5(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    door_run("r", window_subtracted=True)
    sam = FakeSam(door_dets())
    sam.fail = {"car window"}
    install(monkeypatch, sam, FakeSpark(answers(claim_name_v1=naming("car door", "left", "front"))))
    first = derive("r", photo())
    assert first.exit == c.ClaimExit.E2  # an unfinished vote is never agreement
    tapped = derive("r", photo(), n=2, panel="left_front_door")  # the failed request is asked again and fails again
    assert tapped.exit == c.ClaimExit.E5 and "the window check on the photo did not finish" in tapped.paragraph


def test_a_mark_off_the_panel_is_never_judged(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    install(monkeypatch, spark=FakeSpark(answers(marks_v1=marks_answer(((50, 50),), half=(10, 10)))))  # on the grass
    out = derive("r", photo())
    assert out.exit == c.ClaimExit.E6 and out.match is None and out.placement.method == c.PlacementMethod.NONE
    assert "Muse Spark found marks in the claimed area, but none on the panel itself." in out.paragraph


def test_E5_cannot_place_a_partial_panel(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    install(monkeypatch, FakeSam({"trunk lid": [(rect_mask((0, 100, 700, 500)), 0.9)], "car": [(rect_mask((0, 50, 750, 550)), 0.95)]}))
    out = derive("r", photo())
    assert out.exit == c.ClaimExit.E5 and out.placement.method == c.PlacementMethod.NONE and out.match is None
    assert "only part of the panel" in out.paragraph


def test_E6_no_mark_in_the_claimed_area(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    install(monkeypatch, spark=FakeSpark(answers(marks_v1=marks_answer(()))))
    out = derive("r", photo())
    assert out.exit == c.ClaimExit.E6 and out.match is None and "found no mark it can name" in out.paragraph


def test_marks_cut_short_are_never_no_marks(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    install(monkeypatch, spark=FakeSpark(answers(marks_v1=INCOMPLETE)))
    out = derive("r", photo())
    assert out.exit is None and out.match is None and "was cut short" in out.paragraph


def test_E7_a_drawn_box_over_most_of_the_panel(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    box = {"found": True, "bbox": {"x_min": 197, "y_min": 260, "x_max": 822, "y_max": 760}}  # off the line by a few px
    _, spark = install(monkeypatch, spark=FakeSpark(answers(claim_box_v1=box)))
    out = derive("r", photo(gray=True, drawn=(150, 150, 650, 450)))
    assert out.scanner_style and out.roi.source == c.RoiSource.DRAWN
    assert out.exit == c.ClaimExit.E7 and out.match is None and "treated as a region" in out.paragraph
    assert sorted(spark.names()) == ["claim_box_v1", "claim_name_v1"]  # asked together, in threads


def test_the_roi_is_in_the_working_copy_the_web_draws_it_in(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r")
    box = {"found": True, "bbox": {"x_min": 400, "y_min": 400, "x_max": 600, "y_max": 600}}
    install(monkeypatch, spark=FakeSpark(answers(claim_box_v1=box)))
    out = derive("r", photo(gray=True, drawn=(1600, 1200, 2400, 1800), size=(4000, 3000)))
    assert (out.image.w, out.image.h) == (4000, 3000) and out.roi.source == c.RoiSource.DRAWN
    rec = claim.ClaimAnswers.model_validate_json(claim.answers_path("r", "claim_1").read_text())
    assert (rec.work_w, rec.work_h) == (2048, 1536)  # what claimWorkSize in web/src/lib/claim.ts computes from 4000x3000
    s = 2048 / 4000
    assert out.roi.box == pytest.approx([1600 * s, 1200 * s, 2400 * s, 1800 * s], abs=6)


def test_a_small_drawn_box_is_the_roi_spark_grounds_in(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r")
    box = {"found": True, "bbox": {"x_min": 455, "y_min": 435, "x_max": 580, "y_max": 570}}
    _, spark = install(monkeypatch, spark=FakeSpark(answers(claim_box_v1=box, marks_v1=marks_answer(((500, 500),), half=(200, 150)))))
    out = derive("r", photo(gray=True, drawn=(360, 260, 460, 340)))
    assert out.roi.source == c.RoiSource.DRAWN and out.roi.box == pytest.approx([358, 258, 463, 343], abs=2)
    sent = next(r for name, r in spark.requests if name == "marks_v1")["input"][1]["content"][1]["image_url"]
    crop = Image.open(io.BytesIO(base64.b64decode(sent.split(",", 1)[1])))
    rx1, ry1, rx2, ry2 = out.roi.box
    assert crop.size == pytest.approx(((rx2 - rx1) * 1.5, (ry2 - ry1) * 1.5), abs=3)  # padded 25% per side
    assert out.match.reasons == ["L2"]


def test_E8_glare_on_the_claim_photo_caps_the_answer(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    _, spark = install(monkeypatch, spark=FakeSpark(answers(marks_v1=marks_answer(glare=True))))
    out = derive("r", photo())
    assert out.exit == c.ClaimExit.E8 and out.match.reasons == ["L4", "L3", "K11"]
    assert "glare, water or dirt on the claim photo" in out.paragraph and out.size.refused_reason == ruler.CROP_FLAG
    assert "look_v1" not in spark.names()


def test_every_exit_is_a_rule_id() -> None:
    for e in c.ClaimExit:
        assert rules.rule(e.value).id == e.value


# Refusals and the rebuild ----------------------------------------------------------------------------------------------


def test_refusals_delete_the_upload_too(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    make_run("r", coverage=False)
    image = upload("r", photo())
    with pytest.raises(claim.ClaimRefused, match="no coverage yet"):
        asyncio.run(claim.derive_claim("r", image=image, text="", tire_code=None, th=TH))
    assert not image.exists()
    make_run("r2")
    bad = paths.claim_image_path("r2", "claim_1")
    bad.write_bytes(b"not an image")
    with pytest.raises(claim.ClaimRefused, match="could not be read"):
        asyncio.run(claim.derive_claim("r2", image=bad, text="", tire_code=None, th=TH))
    assert not bad.exists()
    tapped = upload("r2", photo())
    with pytest.raises(claim.ClaimRefused, match="not a panel id"):
        asyncio.run(claim.derive_claim("r2", image=tapped, text="", tire_code=None, th=TH, panel_id="nonsense"))
    assert not tapped.exists()  # refused before the privacy step, and still deleted
    with pytest.raises(claim.ClaimRefused, match="named claim_<n>"):
        claim.claim_id_for(Path("photo.jpg"))
    with pytest.raises(claim.ClaimRefused, match="no run"):
        asyncio.run(claim.derive_claim("missing", image=Path("claim_1.jpg"), text="", tire_code=None, th=TH))


def test_privacy_checks_that_do_not_finish_refuse_the_photo(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r")
    sam = FakeSam()
    sam.fail = {"face"}
    _, spark = install(monkeypatch, sam)
    image = upload("r", photo())
    with pytest.raises(claim.ClaimRefused, match="did not complete"):
        asyncio.run(claim.derive_claim("r", image=image, text="", tire_code=None, th=TH))
    assert not image.exists() and not paths.claim_blurred_path("r", "claim_1").exists() and spark.requests == []


def test_a_small_photo_is_low_resolution(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r")
    install(monkeypatch, FakeSam({}), FakeSpark(answers(claim_name_v1=naming("none", "none", "none"))))
    out = derive("r", landscape((400, 300)))
    assert out.image.low_res and out.exit == c.ClaimExit.E1


def test_rebuild_reproduces_the_claim_with_no_network(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    install(monkeypatch)
    derive("r", photo())
    before = paths.claim_record_path("r", "claim_1").read_bytes()

    class Refuse:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError("no model call in a rebuild")

    async def no_sam(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no SAM call in a rebuild")

    monkeypatch.setattr(sam_client, "segment_image", no_sam)
    spark_client.set_client(Refuse())
    rebuilt = claim.rebuild_claim("r", "claim_1", th=TH)
    assert c.json_bytes(rebuilt) == before == paths.claim_record_path("r", "claim_1").read_bytes()
    result = claim.rebuild_claims("r", th=TH)
    assert result["claim/claim_1.json"] == hashlib.sha256(before).hexdigest() == result["claim.json"]
    paths.claim_blurred_path("r", "claim_1").write_bytes(jpeg(landscape()))
    with pytest.raises(claim.ClaimRebuildError, match="missing or changed"):
        claim.rebuild_claim("r", "claim_1", th=TH)


def test_a_rebuild_keeps_the_claim_derived_claim_json_holds(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    install(monkeypatch)
    derive("r", photo())
    held = paths.claim_json("r").read_bytes()
    paths.manifest_run_json("r").write_text("{}\n")  # the run closes
    later = photo()
    later[0:5, 0:5] = 0  # another photo, sent after the close
    derive("r", later, n=2)
    assert paths.claim_json("r").read_bytes() == held
    out = claim.rebuild_claims("r", th=TH)
    assert set(out) == {"claim/claim_1.json", "claim/claim_2.json", "claim.json"}
    assert paths.claim_json("r").read_bytes() == held and out["claim.json"] == hashlib.sha256(held).hexdigest()


def test_decide_runs_off_the_event_loop(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r", offsets=((0.05, 0.0),))
    install(monkeypatch)
    on_main: list[bool] = []
    real = claim.decide

    def spy(*args: Any, **kwargs: Any) -> c.Claim:
        on_main.append(threading.current_thread() is threading.main_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(claim, "decide", spy)
    derive("r", photo())
    assert on_main and not any(on_main)


def test_the_landmark_fit_flag_fails_loudly_while_the_fit_is_not_built() -> None:
    th = copy.deepcopy(TH)
    th["claim"]["align_landmark_fit"] = True
    a = claim.ClaimAnswers(claim_id="claim_1", run_id="r", image_sha256="0" * 64, received_at=NOW, w=W, h=H, work_w=W, work_h=H, blurred_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="align_landmark_fit"):
        claim.decide(SimpleNamespace(run_id="r"), a, photo(gray=True), th)


def wheel_run(run_id: str) -> np.ndarray:
    """A pickup walk with one scratch on the left rear wheel; returns the wheel's mask on the photo."""
    make_run(run_id, states={c.PanelId.LEFT_REAR_WHEEL: c.CoverageState.FILMED})
    disc = np.zeros((H, W), np.uint8)
    cv2.circle(disc, (400, 300), 150, 1, -1)
    disc = disc.astype(bool)
    point = geometry.sent_to_crop(spark_client.grid_to_pixels(512, 500, W, H), (W, H), (W, H))
    u, _ = geometry.wheel_coords(point, disc)
    score = c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1)
    kf = c.Keyframe(kf_id="pickup_wheel_kf", role="pickup", panel_id=c.PanelId.LEFT_REAR_WHEEL, track_id="t3", f_proxy=90, pts=6.0, score=score, aspect_ratio=1.0, box_src=[0.0, 0.0, 300.0, 300.0], front_sign=-1, marks_status=c.MarksStatus.COMPLETE)
    c.write_json(paths.keyframes_json(run_id), [kf])
    mark = c.Mark(mark_id="pickup_wheel_kf_m1", role="pickup", panel_id=c.PanelId.LEFT_REAR_WHEEL, kf_id="pickup_wheel_kf", source=c.MarkSource.SPARK_BOX, type=c.MarkKind.SCRATCH, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S, w=[u + 0.02, 2.5], r=u + 0.02)
    c.write_json(paths.marks_json(run_id), [mark])
    return disc


def test_a_wheel_claim_is_capped_and_sized_from_the_wheel_of_typed_size(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    disc = wheel_run("r")
    line = np.zeros((H, W), np.uint8)
    cv2.line(line, (400, 300), (420, 300), 1, 3)
    dets = {"wheel": [(disc, 0.9)], "car": [(rect_mask((50, 50, 750, 550)), 0.95)], "box": [(line.astype(bool), 0.8)]}
    sam, spark = install(monkeypatch, FakeSam(dets), FakeSpark(answers(claim_name_v1=naming("wheel", "left", "rear"))))
    out = derive("r", photo(), tire="P225/60 R16 98H")
    assert out.panel_id == c.PanelId.LEFT_REAR_WHEEL and out.side_source == "spark" and out.match.frame_used == c.FrameUsed.WHEEL
    assert out.match.reasons == ["L4", "L3", "K10"]  # one mark on a wheel: the radius alone never says already there
    assert "look_v1" not in spark.names()
    assert out.size.ruler == c.RulerKind.WHEEL and out.size.tire_code == "225/60R16" and out.size.threshold_type == c.ThresholdType.WHEEL_MARK
    assert "(ruler: the wheel, 26.6 in from 225/60R16, typed by you)" in out.paragraph
    wheel_run("r2")
    bare = derive("r2", photo())  # 5.12: no typed code, no wheel ruler, and no mask asked for the mark
    assert bare.size.inches is None and bare.size.refused_reason == ruler.TIRE_NOT_GIVEN
    assert "Size not stated: tire size not given." in bare.paragraph
    assert not any(call["concept"].startswith("<0f>") for call in sam.calls if call["run_id"] == "r2.claim_1")


def test_an_unusable_spark_answer_is_not_a_panel_name(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run("r")
    install(monkeypatch, spark=FakeSpark(answers(claim_name_v1="not json")))
    out = derive("r", photo())
    assert out.exit == c.ClaimExit.E2 and out.paragraph.startswith("Muse Spark gave no usable answer; the SAM 3.1 vote named the trunk lid.")
