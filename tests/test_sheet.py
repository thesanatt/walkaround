"""sheet (M13): the sentences of section 7, a render from a synthetic SheetModel that must carry every required
string, every label, exit and refusal reason, the footer on every page, and write_sheet on a synthetic run dir
(the keep_plate owner frames, the stamps read from the chain, and a rebuild after run end that prints the same
sheet.json)."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from markupsafe import escape

from walkaround import config
from walkaround import contracts as c
from walkaround import ledger, paths, privacy, rules, seal, sheet

TH = c.load_thresholds()
NOW = "2026-10-27T15:00:00Z"
CAMERA = "2026-10-27T14:37:00Z"
SHA_P = hashlib.sha256(b"pickup").hexdigest()
SHA_R = hashlib.sha256(b"return").hexdigest()
LFD = c.PanelId.LEFT_FRONT_DOOR


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    return root


def _size(**kw) -> c.SizeEstimate:
    return c.SizeEstimate(**kw)


def _model() -> c.SheetModel:
    """A SheetModel that exercises every label, every refusal reason, the Ordinary Wear line and the clocks."""
    verdicts = [
        c.SheetVerdictLine(verdict_id="v1", label=c.VerdictLabel.ALREADY_THERE, rule_ids=["L3"], mark_ids=["p_m1", "r_m1"], type=c.MarkKind.SCRATCH, size_class=c.SizeClass.M, distance_units=0.12, distance_cm_est=4.2,
                           staysput_line=sheet.staysput_line(c.MarkTrack(mark_id="r_m1", role="return", verdict=c.StaysPutVerdict.FIXED, b1=0.2, b2=0.01, drift_rms=0.03)),
                           size_line=sheet.size_line(_size(inches=0.9, bound_lo=0.6, bound_hi=1.3, ruler=c.RulerKind.PLATE, threshold_type=c.ThresholdType.BODY_SCRATCH))),
        c.SheetVerdictLine(verdict_id="v2", label=c.VerdictLabel.NEEDS_YOUR_EYE, rule_ids=["L4", "K1", "K12"], mark_ids=["p_m3", "r_m2"], type=c.MarkKind.SCUFF, size_class=c.SizeClass.S, distance_units=0.33, distance_cm_est=11.6,
                           staysput_line=sheet.staysput_line(c.MarkTrack(mark_id="p_m3", role="pickup", verdict=c.StaysPutVerdict.INCONCLUSIVE, panned=True)),
                           size_line=sheet.size_line(_size(refused_reason="any crop flag is set")), ordinary_wear_line=sheet.ordinary_wear_line(c.MarkKind.SCUFF, LFD)),
        c.SheetVerdictLine(verdict_id="v3", label=c.VerdictLabel.NOT_SEEN_AT_RETURN, rule_ids=["L1"], mark_ids=["p_m2"], type=c.MarkKind.DENT, size_class=c.SizeClass.L,
                           size_line=sheet.size_line(_size(inches=1.4, bound_lo=1.1, bound_hi=1.8, ruler=c.RulerKind.WHEEL, tire_code="225/60R16", threshold_type=c.ThresholdType.DENT))),
        c.SheetVerdictLine(verdict_id="v4", label=c.VerdictLabel.NEW_OR_UNSEEN, rule_ids=["L2", "K3"], mark_ids=["r_m3"], type=c.MarkKind.CHIP, size_class=c.SizeClass.S,
                           size_line=sheet.size_line(_size(refused_reason="tire size not given"))),
        c.SheetVerdictLine(verdict_id="v5", label=c.VerdictLabel.NOT_COMPARABLE, rule_ids=["L6", "G1"], mark_ids=[], size_line=sheet.size_line(None)),
    ]
    door = c.SheetPanelBlock(
        panel_id=LFD, display_name="left front door",
        pickup=c.SheetFrame(kf_id="pickup_left_front_door_kf1", path="sheet/frames/pickup_left_front_door_kf1.jpg", video_time_s=12.4, wall_clock="2026-10-27T14:37:12Z", mark_ids=["p_m1", "p_m2", "p_m3"]),
        return_=c.SheetFrame(kf_id="return_left_front_door_kf1", path="sheet/frames/return_left_front_door_kf1.jpg", video_time_s=9.1, mark_ids=["r_m1", "r_m2", "r_m3"]),
        coverage_seconds={"pickup": 3.2, "return": 2.8}, coverage_states={"pickup": c.CoverageState.FILMED, "return": c.CoverageState.FILMED},
        verdicts=verdicts, group_lines=["4 chips here at pickup, 5 at return, 1 new or unseen"],
        chips=[c.SheetChip(turn_id="t1", start_ms=1200, end_ms=3400, text="scratch on the driver door")],
    )
    wheel = c.SheetPanelBlock(panel_id=c.PanelId.LEFT_FRONT_WHEEL, display_name="left front wheel", lead_with_frames=True,
                              pickup=c.SheetFrame(kf_id="pickup_left_front_wheel_kf1", path="", video_time_s=20.0),
                              coverage_seconds={"pickup": 0.4}, coverage_states={"pickup": c.CoverageState.GLIMPSED}, odometer_path="keyframes/pickup/odometer.jpg")
    rest = [c.SheetPanelBlock(panel_id=p, display_name=sheet.display_name(p), coverage_states={"pickup": c.CoverageState.NOT_SEEN}) for p in c.PanelId if p not in (LFD, c.PanelId.LEFT_FRONT_WHEEL)]
    thresholds = dict(TH)
    thresholds["calibration"] = c.Calibration(sigma_frame=0.01, sigma_spark=0.02, sigma_view=0.03, sigma_within=0.037, tau_pre=0.25, tau_eye=0.5, tau_fix=0.08, calibrated=True).model_dump(mode="json")
    thresholds["ledger_totals"] = {"calls": 3, "usd": 0.1234, "by_kind": {"sam_video": {"calls": 1, "usd": 0.1, "quantity": "141 frames"}, "spark": {"calls": 2, "usd": 0.0234, "quantity": "3000 tokens"}}}
    deletions = [c.DeletionRecord(target="file_abc", kind=c.DeletionKind.FILES_API, sha256="ab" * 32, deleted_at="2026-10-27T15:09:00Z", reason="run end (pickup proxy)", api_response={"id": "file_abc", "deleted": True})]
    model = c.SheetModel(
        run_id="01JRUN", app_commit="abc1234", config_hash=c.config_hash(TH), events_head_hash="cd" * 32,
        clocks=c.ThreeClocks(sealed_at=NOW, first_model_call_at="2026-10-27T15:00:40Z", sealed_minutes_after_camera=23, camera_time=CAMERA),
        deletion_clocks=c.DeletionClocks(files_deleted_at="2026-10-27T15:09:00Z", quarantine_deleted_at="2026-10-27T15:09:01Z"),
        originals=[
            c.SheetOriginal(role="pickup", name="IMG_0001.MOV", bytes=123456789, sha256=SHA_P, camera_time=CAMERA, received_at="2026-10-27T14:59:00Z", hash_match=True, ots_status="confirmed:921000", tsr_time="2026-10-27T15:00:02Z", bitcoin_block=921000),
            c.SheetOriginal(role="return", name="IMG_0002.MOV", bytes=98765432, sha256=SHA_R, camera_time="not present", received_at="2026-10-29T18:02:00Z", hash_match=True, ots_status="pending"),
        ],
        agreement_number="H1234567", plate_line="plate blurred",
        privacy_line=privacy.privacy_line(c.RunManifest(run_id="01JRUN", created_at=NOW, responses_store=c.ResponsesStore.FALSE_ACCEPTED), {"pickup": True, "return": False}),
        coverage_summary={"pickup": ["left front door: filmed, 3.2 s", "left front wheel: glimpsed, 0.4 s", "missing at pickup: left front wheel"], "return": ["left front door: filmed, 2.8 s"]},
        panels=[door, wheel, *rest],
        not_comparable=[c.SheetNotComparable(panel_id=c.PanelId.HOOD, rule_ids=["G1"], reasons=rules.describe(["G1"]))],
        method=c.SheetMethod(config_hash=c.config_hash(TH), thresholds=thresholds, eval_lines=["false-new 2 of 40 (N=40)"], blurred=["pickup: SAM image checks on 6 keyframes found a plate on 2 and faces on 0"],
                             deletions=deletions, storage_line="Model requests were sent with storage off and the models accepted it; nothing was kept after the run.",
                             rebuild_status="rebuild identical", what_seal_does_not_prove=list(sheet.WHAT_SEAL_DOES_NOT_PROVE), hertz_threshold_note=sheet.HERTZ_NOTE),
        footer_hash_prefixes=[SHA_P[:12], SHA_R[:12]],
    )
    model.note_lines = sheet.note_lines(model)
    model.appendices = {"A certification": sheet.certification_md(model), "B glossary": sheet.glossary_md(), "C verify it yourself": sheet.verify_md(model)}
    model.letter_md = sheet.letter_md(model)
    return model


# The sentences ---------------------------------------------------------------------------------------------------------------


def test_size_line_variants() -> None:
    assert sheet.size_line(_size(inches=0.9, bound_lo=0.6, bound_hi=1.3, ruler=c.RulerKind.PLATE)) == "about 0.9 in, between 0.6 and 1.3 in (ruler: the license plate, 12 in)"
    assert sheet.size_line(_size(inches=1.4, bound_lo=1.1, bound_hi=1.8, ruler=c.RulerKind.WHEEL, tire_code="225/60R16")) == "about 1.4 in, between 1.1 and 1.8 in (ruler: the wheel, 26.6 in from 225/60R16, typed by you)"
    assert sheet.size_line(_size(inches=1.4, ruler=c.RulerKind.WHEEL, tire_code="bogus")) == "about 1.4 in (ruler: the wheel)"
    assert sheet.size_line(None) == "size not stated: no ruler within 0.6 m of this mark"
    assert sheet.size_line(_size()) == "size not stated: no ruler qualified"
    for reason in ("tire size not given", "no ruler within 0.6 m of this mark", "mark mask under 8 px", "any crop flag is set", "a drawn box overlaps the mark", "proxy frame", "ruler on a non-adjacent panel", "plate tilt over 30 degrees"):
        assert sheet.size_line(_size(refused_reason=reason)) == f"size not stated: {reason}"


def test_ots_note_words() -> None:
    assert sheet.ots_note("confirmed:921003") == "Bitcoin block 921003"
    assert sheet.ots_note("pending").startswith("OpenTimestamps pending at the calendars")
    assert sheet.ots_note(None) == "OpenTimestamps not stamped"
    assert sheet.ots_note("failed:FileNotFoundError: the OpenTimestamps client is not installed; pip install x") == "OpenTimestamps not stamped (the OpenTimestamps client is not installed on the server)"
    assert sheet.ots_note("failed:RuntimeError: Failed to create timestamp: all calendars unreachable") == "OpenTimestamps not stamped (Failed to create timestamp: all calendars unreachable)"


def test_tire_diameter_in() -> None:
    assert abs(sheet.tire_diameter_in("225/60R16") - 26.63) < 0.02
    assert abs(sheet.tire_diameter_in(" 205/55 R 16 ") - 24.88) < 0.02
    assert sheet.tire_diameter_in("P215") is None and sheet.tire_diameter_in(None) is None


def test_threshold_line() -> None:
    under = sheet.threshold_line(_size(inches=0.9, bound_lo=0.6, bound_hi=1.3, threshold_type=c.ThresholdType.BODY_SCRATCH), TH)
    assert under == "under Hertz's example threshold of 3.5 in for a body scratch along its long axis (REV 7.2025, illustrative only)"
    over = sheet.threshold_line(_size(inches=1.4, bound_lo=1.1, bound_hi=1.8, threshold_type=c.ThresholdType.DENT), TH)
    assert over.startswith("over Hertz's example threshold of 1 in for a dent")
    straddle = sheet.threshold_line(_size(inches=1.0, bound_lo=0.8, bound_hi=1.3, threshold_type=c.ThresholdType.WHEEL_MARK), TH)
    assert straddle.startswith("threshold not compared: the size bound straddles 1 in")
    assert sheet.threshold_line(_size(inches=2.0, threshold_type=c.ThresholdType.BODY_SCUFF), TH) is None
    assert sheet.threshold_line(_size(refused_reason="x"), TH) is None and sheet.threshold_line(None, TH) is None


def test_ordinary_wear_line() -> None:
    line = sheet.ordinary_wear_line(c.MarkKind.SCUFF, LFD)
    assert line == "Hertz's page lists scuffs that do not break paint as ordinary wear (REV 7.2025, illustrative only; page saved 2026-10-26)."
    assert sheet.ordinary_wear_line(c.MarkKind.SCRATCH, LFD) is None
    assert sheet.ordinary_wear_line(c.MarkKind.SCUFF, c.PanelId.LEFT_FRONT_WHEEL) is None
    assert sheet.ordinary_wear_line(None, LFD) is None


def test_staysput_line_variants() -> None:
    fixed = c.MarkTrack(mark_id="m", role="pickup", verdict=c.StaysPutVerdict.FIXED, b1=0.2, b2=0.01, drift_rms=0.03)
    assert sheet.staysput_line(fixed) == "drift 0.03, b1 0.20, b2 0.01, not panned; fixed"
    sliding = c.MarkTrack(mark_id="m", role="pickup", verdict=c.StaysPutVerdict.SLIDING, b1=0.3, b2=0.02, drift_rms=0.21, panned=False, curvature_cap=True, highlight_cap=True)
    assert sheet.staysput_line(sliding) == "drift 0.21, b1 0.30, b2 0.02, not panned; sliding; curvature cap; highlight cap"
    assert sheet.staysput_line(c.MarkTrack(mark_id="m", role="pickup", verdict=c.StaysPutVerdict.INCONCLUSIVE)) == "inconclusive (camera did not move enough)"
    assert sheet.staysput_line(c.MarkTrack(mark_id="m", role="pickup", verdict=c.StaysPutVerdict.INCONCLUSIVE, panned=True)) == "inconclusive (the camera panned: b1 under its floor, b2 over)"
    assert sheet.staysput_line(c.MarkTrack(mark_id="m", role="pickup", verdict=c.StaysPutVerdict.NO_TRACK)).startswith("no track")
    assert sheet.staysput_line(c.MarkTrack(mark_id="m", role="pickup")) is None and sheet.staysput_line(None) is None


def test_note_lines_follow_the_section_7_order() -> None:
    lines = sheet.note_lines(_model())
    assert len(lines) == 5
    assert "dated pickup frames" in lines[0] and CAMERA in lines[0] and SHA_P[:12] in lines[0] and "2026-10-27T14:59:00Z" in lines[0]
    assert "full damage report" in lines[1] and "request more images" in lines[1]
    assert "alternative written estimate" in lines[2] and "10 business days" in lines[2] and "115%" in lines[2]
    assert "executive" in lines[3] and "state consumer" in lines[3]
    assert "card dispute" in lines[4] and "60 days" in lines[4] and "120 days" in lines[4] and "$50" in lines[4] and "collections" in lines[4]
    assert all(line.isascii() for line in lines)


def test_verify_md_lists_only_what_the_run_has() -> None:
    bare = c.SheetModel(run_id="01JBARE", originals=[c.SheetOriginal(role="pickup", name="a.mov", bytes=1, sha256=SHA_P)])
    text = sheet.verify_md(bare)
    assert "ots info" not in text and "openssl ts -verify" not in text  # no proof, no token: no command for them
    assert text.startswith("Verify it yourself (2 steps;") and "1. Hash the video file" in text and "2. Re-hash every file" in text
    failed = bare.model_copy(deep=True)
    failed.method.thresholds["stamps"] = c.Stamps(manifest_ots="failed:FileNotFoundError: no client", tsr_time="2026-10-27T15:00:02Z", tsa_issuer="DigiCert").model_dump(mode="json")
    text = sheet.verify_md(failed)
    assert "ots info" not in text and "DigiCert" in text and "-CAfile /etc/ssl/cert.pem" in text and "(3 steps;" in text
    freetsa = bare.model_copy(deep=True)
    freetsa.method.thresholds["stamps"] = c.Stamps(manifest_ots="pending", tsr_time="2026-10-27T15:00:02Z", tsa_issuer="freetsa").model_dump(mode="json")
    text = sheet.verify_md(freetsa)
    assert "ots verify bundle/manifest.json.ots" in text and "https://freetsa.org/files/cacert.pem" in text and "(5 steps;" in text
    assert text.isascii()


def test_appendix_texts() -> None:
    model = _model()
    for text in (sheet.letter_md(model), sheet.verify_md(model), sheet.certification_md(model), sheet.glossary_md()):
        assert text.isascii()
    assert "H1234567" in sheet.letter_md(model) and SHA_P in sheet.letter_md(model) and "Visa Core Rules 13.3 and 13.5" in sheet.letter_md(model)
    verify = sheet.verify_md(model)
    for cmd in ("shasum -a 256", "ots info", "ots verify", "openssl ts -verify", "walkaround verify 01JRUN"):
        assert cmd in verify
    assert "signed statement, not a notarized affidavit" in sheet.certification_md(model)
    glossary = sheet.glossary_md()
    for term in ("already there", "needs your eye", "new or unseen", "not seen at return", "not comparable", "tau_pre", "Stays Put", "config hash"):
        assert term in glossary


# Rendering from a synthetic SheetModel ----------------------------------------------------------------------------------------


def test_render_contains_every_required_string() -> None:
    model = _model()
    html = sheet.render_html(model)
    required = [
        "01JRUN", SHA_P, SHA_R, "evidence support, not legal advice", "abc1234", c.config_hash(TH), "cd" * 32,
        NOW, "2026-10-27T15:00:40Z", "sealed 23 minutes after the camera time",  # the three clocks
        CAMERA, "not present", "2026-10-27T14:59:00Z", "2026-10-29T18:02:00Z",  # capture and receipt times
        "IMG_0001.MOV", "123456789", "Bitcoin block 921000", "OpenTimestamps pending", "RFC 3161 token",
        "H1234567", "plate blurred", "Return walk: no plate detected",
        "left front door: filmed, 3.2 s", "missing at pickup: left front wheel",  # the coverage list
        "video time 12.40 s", "wall clock 2026-10-27T14:37:12Z", "video time 9.10 s",
        "4 chips here at pickup, 5 at return, 1 new or unseen", "scratch on the driver door", "1200 to 3400 ms",
        "0.12 units, about 4 cm", "drift 0.03, b1 0.20, b2 0.01, not panned; fixed", "inconclusive (the camera panned",
        "about 0.9 in, between 0.6 and 1.3 in (ruler: the license plate, 12 in)", "26.6 in from 225/60R16, typed by you",
        "size not stated: tire size not given", "size not stated: any crop flag is set", "size not stated: no ruler within 0.6 m of this mark",
        sheet.ORDINARY_WEAR_LINE, "REV 7.2025", "2026-10-26", "illustrative only", "2 in x 3.5 in",
        "false-new 2 of 40 (N=40)", "rebuild identical", "sam_video", "0.1234", "141 frames",
        "file_abc", "'deleted': True", "2026-10-27T15:09:00Z", "Quarantine (unblurred originals) deleted: 2026-10-27T15:09:01Z",
        "which car was filmed", "FRE 901(b)(9)", "panels.crop_long_edge_px", "tau_pre", "0.25", "0.037",
        "G1 not filmed", "Appendix A certification", "Appendix B glossary", "Appendix C verify it yourself", "Write to the company first",
        "Processed with Meta's Model API in Standard mode",
    ]
    for text in required:
        assert str(escape(text)) in html, text
    for line in model.note_lines:
        assert str(escape(line)) in html, line[:40]
    order = [html.index(line.split(":")[0][:30]) for line in model.note_lines]
    assert order == sorted(order)
    for label, text in sheet.LABEL_TEXT.items():
        assert f'class="label {label}">{text}</span>' in html, label
    for rid in ("L1", "L2", "L3", "L4", "L6", "K1", "K3", "K12", "G1"):
        assert f">{rid}</span>" in html or f"{rid} " in html, rid


def test_render_footer_on_every_page_ascii_and_self_contained() -> None:
    html = sheet.render_html(_model())
    pages = html.count('<section class="page"')
    assert pages >= 7  # page 1, two panel pages, the frameless table, the method page, three appendices, the letter
    footers = re.findall(r'<div class="footer">(.*?)</div>', html, flags=re.S)
    assert len(footers) == pages
    for footer in footers:
        assert "run 01JRUN" in footer and SHA_P[:12] in footer and SHA_R[:12] in footer
    assert html.isascii()
    for banned in ("<script", "<link", "@import", 'src="http', "src='http", 'href="http', "url(", "font-face"):
        assert banned not in html, banned
    assert 'src="../sheet/frames/pickup_left_front_door_kf1.jpg"' in html
    assert "no return frame for this panel" in html  # the wheel block has a pickup frame only
    assert "not available" in html  # a frame record with no pixels says so instead of a broken image
    assert "Panels without frames" in html and "hood" in html


def test_render_every_exit_and_refusal_reason() -> None:
    for exit_id in c.ClaimExit:
        claim = c.Claim(image=c.ClaimImage(sha256="ef" * 32, received_at=NOW, w=800, h=600), text="scratch", panel_id=LFD, exit=exit_id,
                        size=c.SizeEstimate(refused_reason=f"refused for {exit_id.value}"), paragraph=f"paragraph for {exit_id.value}")
        text = sheet._claim_text(claim, TH)
        r = rules.rule(exit_id.value)
        assert f"exit {r.id} {r.label}: {r.text}" in text and f"size not stated: refused for {exit_id.value}" in text and "paragraph for" in text
    model = _model()
    model.appendices["D claim check"] = sheet._claim_text(c.Claim(image=c.ClaimImage(sha256="ef" * 32, received_at=NOW, w=800, h=600), exit=c.ClaimExit.E6, size=c.SizeEstimate(refused_reason="mark mask under 8 px")), TH)
    html = sheet.render_html(model)
    assert "Appendix D claim check" in html and "exit E6 no marks in ROI" in html and "size not stated: mark mask under 8 px" in html


def test_render_empty_model_and_strict_undefined() -> None:
    html = sheet.render_html(c.SheetModel(run_id="empty"))
    assert "run empty" in html and "no originals recorded" in html and "coverage not computed" in html and html.isascii()
    assert "Every canonical panel has a frame." in html  # no blocks at all


def test_pin_pdf_dates_keeps_the_width_and_pins_both_dates(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    raw = b"<</Producer (Skia/PDF m153)\n/CreationDate (D:20261031175842+00'00')\n/ModDate (D:20261031175843+00'00')>>"
    pdf.write_bytes(raw)
    sheet.pin_pdf_dates(pdf, "20261027150002")
    out = pdf.read_bytes()
    assert len(out) == len(raw) and out.count(b"D:20261027150002+00'00'") == 2


def test_two_prints_of_one_page_are_byte_identical(tmp_path: Path) -> None:
    pytest.importorskip("playwright.sync_api")
    html = tmp_path / "p.html"
    html.write_text("<html><body><p>sheet</p></body></html>", encoding="ascii")
    try:
        a = sheet.render_pdf(html, tmp_path / "a.pdf", timeout_s=60, stamp="20261027150002").read_bytes()
        time.sleep(1.1)  # the print time moves on by a second
        b = sheet.render_pdf(html, tmp_path / "b.pdf", timeout_s=60, stamp="20261027150002").read_bytes()
    except Exception as exc:  # playwright without its browser
        pytest.skip(f"Chromium not available: {exc}")
    assert a == b and b"D:20261027150002" in a


def test_render_pdf_without_playwright_raises_runtime_error(tmp_path: Path) -> None:
    if importlib.util.find_spec("playwright") is not None:
        pytest.skip("playwright is installed; the PDF path is a live check")
    html = tmp_path / "sheet.html"
    html.write_text("<html><body>x</body></html>", encoding="ascii")
    with pytest.raises(RuntimeError):
        sheet.render_pdf(html, tmp_path / "evidence.pdf", timeout_s=5)


# write_sheet on a synthetic run dir ---------------------------------------------------------------------------------------


def _make_clip(path: Path, *, w: int = 320, h: int = 180) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=15", "-t", "0.5",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", "-f", "mp4", str(path)], check=True)


def _keyframe(role: str, panel: c.PanelId, run_id: str) -> c.Keyframe:
    kf_id = f"{role}_{panel.value}_kf1"
    crop_file = paths.crop_path(run_id, role, kf_id)
    rng = np.random.default_rng(1 if role == "pickup" else 2)
    pixels = rng.integers(0, 256, (240, 240, 3), dtype=np.uint8)
    cv2.imwrite(str(crop_file), pixels)
    sha = hashlib.sha256(crop_file.read_bytes()).hexdigest()
    return c.Keyframe(
        kf_id=kf_id, role=role, panel_id=panel, track_id=f"{role}_car_door_0", f_proxy=3, pts=0.2,
        score=c.KeyframeScore(area=0.1, completeness=0.9, sharpness=120.0, glare=0.01, frontality=0.8, total=0.7), aspect_ratio=1.2,
        box_src=[100.0, 50.0, 220.0, 170.0], crop=c.Crop(path=paths.rel(run_id, crop_file), sha256=sha, w=240, h=240, scale=1.0, dx=100.0, dy=50.0),
        blur=c.BlurRecord(plate_boxes=[[10.0, 10.0, 50.0, 30.0]], request_ids=["resp_p", "resp_f"], plate_ruler_px=40.0, plate_ruler_c=0.9, plate_tilt_hw=0.5),
        front_sign=1, marks_status=c.MarksStatus.COMPLETE,
    )


def _mark(role: str, n: int, kind: c.MarkKind, *, conf: c.Confidence = c.Confidence.HIGH, caps: list[str] | None = None) -> c.Mark:
    return c.Mark(mark_id=f"{role}_left_front_door_kf1_m{n}", role=role, panel_id=LFD, kf_id=f"{role}_left_front_door_kf1", source=c.MarkSource.SPARK_BOX,
                  type=kind, confidence=conf, size_class=c.SizeClass.M, bbox_crop=[20.0 + 40 * n, 30.0, 50.0 + 40 * n, 60.0], point_crop=[35.0 + 40 * n, 45.0], caps=caps or [])


def _synthetic_run(run_id: str, *, keep_plate: bool) -> tuple[str, str]:
    paths.ensure_run_layout(run_id)
    pickup_orig = paths.quarantine_original(run_id, "pickup")
    _make_clip(pickup_orig)
    ret_orig = paths.quarantine_original(run_id, "return")
    ret_orig.write_bytes(b"return bytes " * 300)
    sha_p, sha_r = seal.sha256_file(pickup_orig), seal.sha256_file(ret_orig)
    c.write_json(paths.run_json(run_id), c.RunManifest(
        run_id=run_id, created_at=NOW, status=c.RunStatus.SHEET, keep_plate=keep_plate, app_commit="abc1234", config_hash=c.config_hash(TH), thresholds=TH, fps=15.0,
        clocks=c.ThreeClocks(sealed_at=NOW, first_model_call_at="2026-10-27T15:00:40Z", sealed_minutes_after_camera=23, camera_time=CAMERA),
        responses_store=c.ResponsesStore.FALSE_ACCEPTED, events_head_hash="cd" * 32,
    ))
    probe = c.FfprobeInfo(codec="h264", container="mov", width=320, height=180, display_width=320, display_height=180, fps_avg=15.0, duration_s=0.5, creation_time=CAMERA)
    proxy = c.ProxyRecord(path="media/pickup.proxy.mp4", fps=15.0, width=320, height=180, frames=8, pts_map_path="media/pickup.pts.json")
    c.write_json(paths.video_json(run_id, "pickup"), c.VideoRecord(role="pickup", original_filename="IMG_0001.MOV", bytes=pickup_orig.stat().st_size, sha256_browser=sha_p, sha256_server=sha_p, hash_match=True, received_at="2026-10-27T14:59:00Z", ffprobe=probe, proxy=proxy))
    c.write_json(paths.video_json(run_id, "return"), c.VideoRecord(role="return", original_filename="IMG_0002.MOV", bytes=ret_orig.stat().st_size, sha256_server=sha_r, hash_match=True, received_at="2026-10-29T18:02:00Z", ffprobe=probe.model_copy(update={"creation_time": "not present"})))
    seal.build_manifest(run_id, [pickup_orig, ret_orig])

    kf_p, kf_r = _keyframe("pickup", LFD, run_id), _keyframe("return", LFD, run_id)
    c.write_json(paths.keyframes_json(run_id), [kf_p, kf_r])
    c.write_json(paths.panels_json(run_id), [
        c.Panel(panel_id=LFD, noun=c.Noun.CAR_DOOR, side=c.Side.LEFT, position=c.Position.FRONT, instance_from=c.InstanceFrom.TRACKS, front_sign=1, comparable=True,
                per_role={"pickup": c.PanelRole(track_ids=["pickup_car_door_0"], seconds_visible=3.2, state=c.CoverageState.FILMED, keyframe_ids=[kf_p.kf_id]),
                          "return": c.PanelRole(track_ids=["return_car_door_0"], seconds_visible=2.8, state=c.CoverageState.FILMED, keyframe_ids=[kf_r.kf_id])}),
        c.Panel(panel_id=c.PanelId.HOOD, noun=c.Noun.HOOD, comparable=False, not_comparable_reasons=["G1"]),
    ])
    marks = [_mark("pickup", 1, c.MarkKind.SCRATCH), _mark("pickup", 2, c.MarkKind.DENT), _mark("pickup", 3, c.MarkKind.SCUFF, conf=c.Confidence.LOW, caps=["K12"]),
             _mark("return", 1, c.MarkKind.SCRATCH), _mark("return", 2, c.MarkKind.SCUFF), _mark("return", 3, c.MarkKind.CHIP, caps=["K3"])]
    c.write_json(paths.marks_json(run_id), marks)
    pair = c.MatchCandidate(pickup_mark=marks[0].mark_id, return_mark=marks[3].mark_id, d=0.12, d_pos=0.1, type_cost=0.0, size_cost=0.02, r_mean=0.5, tau_pre_r=0.3, tau_eye_r=0.6, mutual=True)
    band = c.MatchCandidate(pickup_mark=marks[2].mark_id, return_mark=marks[4].mark_id, d=0.33, d_pos=0.31, type_cost=0.0, size_cost=0.02, r_mean=0.6, tau_pre_r=0.3, tau_eye_r=0.6, mutual=True)
    c.write_json(paths.matches_json(run_id), [
        c.PanelMatch(panel_id=LFD, state=c.MatchState.COMPARABLE, kf_pair=[kf_p.kf_id, kf_r.kf_id], aspect_diff=0.01, mask_iou=0.85, pairs=[pair, band],
                     unmatched_pickup=[marks[1].mark_id], unmatched_return=[marks[5].mark_id], groups=[c.MarkGroup(type=c.MarkKind.CHIP, n_pickup=4, n_return=5, text="4 chips here at pickup, 5 at return, 1 new or unseen")]),
        c.PanelMatch(panel_id=c.PanelId.HOOD, state=c.MatchState.NOT_COMPARABLE, reasons=["G1"]),
    ])
    trace = c.Trace(config_hash=c.config_hash(TH))
    c.write_json(paths.verdicts_json(run_id), [
        c.Verdict(verdict_id="v1", panel_id=LFD, label=c.VerdictLabel.ALREADY_THERE, rule_id="L3", mark_ids=[marks[0].mark_id, marks[3].mark_id], distance_units=0.12, distance_cm_est=4.2, trace=trace.model_copy(update={"rule_ids": ["L3"]}),
                  size=c.SizeEstimate(inches=0.9, bound_lo=0.6, bound_hi=1.3, ruler=c.RulerKind.PLATE, ruler_source_kf=kf_p.kf_id, threshold_type=c.ThresholdType.BODY_SCRATCH)),
        c.Verdict(verdict_id="v2", panel_id=LFD, label=c.VerdictLabel.NEEDS_YOUR_EYE, rule_id="L4", mark_ids=[marks[2].mark_id, marks[4].mark_id], distance_units=0.33, distance_cm_est=11.6, trace=trace.model_copy(update={"rule_ids": ["L4", "K1", "K12"]})),
        c.Verdict(verdict_id="v3", panel_id=LFD, label=c.VerdictLabel.NOT_SEEN_AT_RETURN, rule_id="L1", mark_ids=[marks[1].mark_id], trace=trace.model_copy(update={"rule_ids": ["L1"]})),
        c.Verdict(verdict_id="v4", panel_id=LFD, label=c.VerdictLabel.NEW_OR_UNSEEN, rule_id="L2", mark_ids=[marks[5].mark_id], trace=trace.model_copy(update={"rule_ids": ["L2", "K3"]}), size=c.SizeEstimate(refused_reason="tire size not given")),
        c.Verdict(verdict_id="v5", panel_id=c.PanelId.HOOD, label=c.VerdictLabel.NOT_COMPARABLE, rule_id="L6", trace=trace.model_copy(update={"rule_ids": ["L6", "G1"]})),
    ])
    c.write_json(paths.mark_tracks_json(run_id), [
        c.MarkTrack(mark_id=marks[3].mark_id, role="return", verdict=c.StaysPutVerdict.FIXED, b1=0.2, b2=0.01, drift_rms=0.03),
        c.MarkTrack(mark_id=marks[2].mark_id, role="pickup", verdict=c.StaysPutVerdict.INCONCLUSIVE, panned=True),
    ])
    odometer = paths.keyframes_dir(run_id, "pickup") / "odometer.jpg"
    cv2.imwrite(str(odometer), np.full((60, 80, 3), 90, dtype=np.uint8))
    c.write_json(paths.coverage_json(run_id), c.Coverage(
        per_role={"pickup": {LFD: c.PanelCoverage(state=c.CoverageState.FILMED, seconds=3.2, best_kf=kf_p.kf_id), c.PanelId.HOOD: c.PanelCoverage(state=c.CoverageState.NOT_FILMED), c.PanelId.LEFT_FRONT_WHEEL: c.PanelCoverage(state=c.CoverageState.GLIMPSED, seconds=0.4)},
                  "return": {LFD: c.PanelCoverage(state=c.CoverageState.FILMED, seconds=2.8, best_kf=kf_r.kf_id)}},
        missing=[c.PanelId.LEFT_FRONT_WHEEL], topup_text="film the left front wheel", odometer=c.Odometer(asked=True, filmed=True, frame_path=paths.rel(run_id, odometer)),
    ))
    c.write_json(paths.narration_json(run_id), c.Narration(per_role={"pickup": c.NarrationRole(turns=[c.NarrationTurn(turn_id="t1", start_ms=1200, end_ms=3400, text="scratch on the driver door", keywords_hit=["scratch"], pinned_panel=LFD, pin_kind=c.PinKind.OVERLAP)])}))
    c.write_json(paths.calibration_json(run_id), c.Calibration(sigma_frame=0.01, sigma_spark=0.02, sigma_view=0.03, sigma_within=0.037, tau_pre=0.25, tau_eye=0.5, tau_fix=0.08, calibrated=True))
    c.write_json(paths.claim_json(run_id), c.Claim(image=c.ClaimImage(sha256="ef" * 32, received_at=NOW, w=800, h=600), text="scratch on the left front door", panel_id=LFD, exit=c.ClaimExit.E6, size=c.SizeEstimate(refused_reason="mark mask under 8 px"), paragraph="No mark was found in the region you drew."))
    c.write_json(paths.deletions_json(run_id), c.Deletions(run_id=run_id, rows=[
        c.DeletionRecord(target="file_abc", kind=c.DeletionKind.FILES_API, sha256="ab" * 32, deleted_at="2026-10-27T15:09:00Z", reason="run end (pickup proxy)", api_response={"id": "file_abc", "deleted": True}),
    ], keep_plate=keep_plate, keep_plate_chosen_by="renter at upload" if keep_plate else None))
    ledger.record("sam_video", "sam-3.1", note="car", run_id=run_id, frames=141, unit="frames", quantity=141)
    ledger.record("sam_image", "sam-3.1", note="license plate f000003", run_id=run_id, images=1)
    ledger.record("spark", "muse-spark-1.3", note="marks", run_id=run_id, input_tokens=1400, output_tokens=600, cached_tokens=0, reasoning_tokens=400)
    ledger.record("spark", "muse-spark-1.3", note="other run", run_id="other", input_tokens=99999, output_tokens=1)
    return sha_p, sha_r


def test_write_sheet_from_a_synthetic_run(runs: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    run_id = "01JTEST"
    sha_p, sha_r = _synthetic_run(run_id, keep_plate=False)
    model, hashes = sheet.write_sheet(run_id, th=TH)
    expected_files = {"sheet/sheet.json", "sheet/sheet.html", "bundle/letter.md", "bundle/verify.md", "bundle/note.md"}
    if importlib.util.find_spec("playwright") is None:
        assert set(hashes) == expected_files and not paths.sheet_pdf(run_id).exists()
    else:
        assert expected_files <= set(hashes)
    for rel, digest in hashes.items():
        assert seal.sha256_file(paths.run_dir(run_id) / rel) == digest
    assert c.read_json(paths.sheet_json(run_id), c.SheetModel) == model and hashes["sheet/sheet.json"] == model.digest()

    # Page 1.
    assert [o.sha256 for o in model.originals] == [sha_p, sha_r] and model.footer_hash_prefixes == [sha_p[:12], sha_r[:12]]
    assert model.originals[0].camera_time == CAMERA and model.originals[1].camera_time == "not present"
    assert model.clocks.sealed_minutes_after_camera == 23 and model.plate_line == "plate blurred"
    assert model.deletion_clocks.files_deleted_at == "2026-10-27T15:09:00Z" and model.deletion_clocks.quarantine_deleted_at is None
    assert "Model requests were sent with storage off." in model.privacy_line and "no plate detected" not in model.privacy_line
    assert len(model.note_lines) == 5
    assert model.coverage_summary["pickup"][:1] == ["left front door: filmed, 3.2 s"]
    assert "missing at pickup: left front wheel" in model.coverage_summary["pickup"] and "top-up asked: film the left front wheel" in model.coverage_summary["pickup"]

    # Panel blocks: every canonical slot, the door with both frames and the marks drawn, the wheel leading with frames.
    assert [b.panel_id for b in model.panels] == list(c.PanelId)
    door = next(b for b in model.panels if b.panel_id == LFD)
    assert door.pickup and door.return_ and door.pickup.kf_id == "pickup_left_front_door_kf1" and door.return_.kf_id == "return_left_front_door_kf1"
    assert door.pickup.wall_clock == "2026-10-27T14:37:00Z" and door.return_.wall_clock is None  # 0.2 s after a camera time, rounded to the second
    assert door.pickup.mark_ids == ["pickup_left_front_door_kf1_m1", "pickup_left_front_door_kf1_m2", "pickup_left_front_door_kf1_m3"]
    assert door.coverage_seconds == {"pickup": 3.2, "return": 2.8} and door.coverage_states["return"] == c.CoverageState.FILMED
    assert [v.label for v in door.verdicts] == [c.VerdictLabel.ALREADY_THERE, c.VerdictLabel.NEEDS_YOUR_EYE, c.VerdictLabel.NOT_SEEN_AT_RETURN, c.VerdictLabel.NEW_OR_UNSEEN]
    assert door.verdicts[1].rule_ids == ["L4", "K1", "K12"] and door.verdicts[1].ordinary_wear_line == sheet.ORDINARY_WEAR_LINE
    assert door.verdicts[0].size_line.startswith("about 0.9 in, between 0.6 and 1.3 in (ruler: the license plate, 12 in); under Hertz")
    assert door.verdicts[0].staysput_line == "drift 0.03, b1 0.20, b2 0.01, not panned; fixed"
    assert door.verdicts[3].size_line == "size not stated: tire size not given"
    assert door.group_lines == ["4 chips here at pickup, 5 at return, 1 new or unseen"] and door.chips[0].text == "scratch on the driver door"
    assert door.odometer_path == "keyframes/pickup/odometer.jpg"
    for frame in (door.pickup, door.return_):
        drawn = paths.run_dir(run_id) / frame.path
        assert drawn.is_file() and frame.path.startswith("sheet/frames/")
        crop = cv2.imread(str(paths.crop_path(run_id, frame.kf_id.split("_")[0], frame.kf_id)))
        assert not np.array_equal(cv2.imread(str(drawn)), crop)  # marks were drawn on the crop
    wheel = next(b for b in model.panels if b.panel_id == c.PanelId.LEFT_FRONT_WHEEL)
    assert wheel.lead_with_frames and wheel.pickup is None and wheel.coverage_states == {"pickup": c.CoverageState.GLIMPSED}
    assert model.not_comparable == [c.SheetNotComparable(panel_id=c.PanelId.HOOD, rule_ids=["G1"], reasons=rules.describe(["G1"]))]
    hood = next(b for b in model.panels if b.panel_id == c.PanelId.HOOD)
    assert [v.label for v in hood.verdicts] == [c.VerdictLabel.NOT_COMPARABLE] and hood.verdicts[0].rule_ids == ["L6", "G1"]

    # The method page.
    assert model.method.config_hash == c.config_hash(TH) and model.method.thresholds["calibration"]["tau_pre"] == 0.25
    totals = model.method.thresholds["ledger_totals"]
    assert totals["calls"] == 3 and set(totals["by_kind"]) == {"sam_image", "sam_video", "spark"} and totals["by_kind"]["spark"]["quantity"] == "2000 tokens"
    assert abs(totals["usd"] - (round(141 * 0.0002, 6) + 0.0025 + round(1400 * 1.25e-6 + 600 * 4.25e-6, 6))) < 1e-4
    assert model.method.deletions[0].api_response == {"id": "file_abc", "deleted": True}
    assert model.method.rebuild_status.startswith("event chain verified to the head on page 1")
    assert "the models accepted it" in model.method.storage_line and model.method.hertz_threshold_note == sheet.HERTZ_NOTE
    assert model.method.blurred[0].startswith("pickup: SAM image checks on 1 keyframes found a plate on 1 and faces on 0")
    assert "D claim check" in model.appendices and "exit E6" in model.appendices["D claim check"]

    assert model.method.storage_line.endswith(sheet.RUN_END_NOTE)  # the quarantine is not deleted yet
    assert model.method.thresholds["stamps"] == c.Stamps().model_dump(mode="json")  # no chain, no stamps
    html = paths.sheet_html(run_id).read_text(encoding="ascii")
    for text in (run_id, sha_p, sha_r, "sealed 23 minutes after the camera time", "evidence support, not legal advice", "REV 7.2025", "2026-10-26", sheet.ORDINARY_WEAR_LINE, "G1 not filmed", "sam_video", sheet.RUN_END_NOTE):
        assert str(escape(text)) in html, text
    # frames are embedded, so the page shows them when served at /api/runs/<id>/sheet, where ../ paths break
    assert 'src="../' not in html and html.count('src="data:image/jpeg;base64,') == 3  # two door frames and the odometer
    note = paths.bundle_file(run_id, "note.md").read_text(encoding="ascii")
    assert note.startswith("1. ") and note.count("\n") == 5


def test_write_sheet_keep_plate_serves_the_owner_variant_only(runs: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    run_id = "01JKEEP"
    _synthetic_run(run_id, keep_plate=True)
    model, _ = sheet.write_sheet(run_id, th=TH)
    assert model.plate_line == "plate kept on this sheet at the renter's request at upload"
    assert any("keep_plate was chosen at upload" in line for line in model.method.blurred)
    door = next(b for b in model.panels if b.panel_id == LFD)
    assert door.pickup.path == f"sheet/frames/{door.pickup.kf_id}.owner.jpg"
    assert door.return_.path == f"sheet/frames/{door.return_.kf_id}.jpg"  # the return original is not a decodable video
    owner = cv2.imread(str(paths.run_dir(run_id) / door.pickup.path))
    assert owner is not None and owner.shape == (240, 240, 3)
    public_crop = cv2.imread(str(paths.crop_path(run_id, "pickup", door.pickup.kf_id)))
    assert not np.array_equal(owner, public_crop)  # the owner's frame came from the original, not the public crop
    public_frame = paths.sheet_dir(run_id) / "frames" / f"{door.pickup.kf_id}.jpg"
    assert public_frame.is_file()  # the blurred frame with the marks is still written beside it
    run = c.read_json(paths.run_json(run_id), c.RunManifest)
    crop_path = paths.rel(run_id, paths.crop_path(run_id, "pickup", door.pickup.kf_id))
    for rel in (door.pickup.path, crop_path, paths.rel(run_id, public_frame)):
        assert privacy.served_file_allowed(run, rel, run_owner=True)
        assert not privacy.served_file_allowed(run, rel, run_owner=False)
    assert privacy.served_file_allowed(run, "media/pickup.proxy_blurred.mp4", run_owner=False)


def test_owner_frame_keeps_only_the_plate_on_its_own_panel(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    run_id = "01JNEIGH"
    _synthetic_run(run_id, keep_plate=True)
    keyframes = c.read_json_list(paths.keyframes_json(run_id), c.Keyframe)
    on_panel, neighbour = [110.0, 60.0, 150.0, 80.0], [250.0, 100.0, 300.0, 130.0]  # the door's box is 100, 50, 220, 170
    keyframes[0].blur.plate_boxes = [on_panel, neighbour]
    keyframes[0].blur.face_boxes = [[230.0, 20.0, 260.0, 60.0]]
    c.write_json(paths.keyframes_json(run_id), keyframes)
    calls: list[dict] = []
    real = privacy.blur_frame

    def spy(frame, **kwargs):
        calls.append(kwargs)
        return real(frame, **kwargs)

    monkeypatch.setattr(sheet.privacy, "blur_frame", spy)
    sheet.write_sheet(run_id, th=TH)
    assert len(calls) == 1 and calls[0]["keep_plate_region"] is True
    boxes = calls[0]["track_boxes_src"]
    assert neighbour in boxes and [230.0, 20.0, 260.0, 60.0] in boxes  # a neighbour's plate and every face are pixelated
    assert on_panel not in boxes  # the rented car's own plate stays for the owner


def _append_event(run_id: str, kind: c.EventKind, payload: dict, ts: str) -> c.Event:
    """One chained events.ndjson line the way the orchestrator writes it (prev of seq 0 is manifest.json's sha256)."""
    path = paths.events_path(run_id)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
    if lines:
        last = c.Event.model_validate_json(lines[-1])
        seq, prev = last.seq + 1, last.hash
    else:
        seq, prev = 0, seal.sha256_file(paths.manifest_json(run_id))
    line = {"seq": seq, "ts": ts, "run_id": run_id, "kind": kind.value, "payload": payload, "prev": prev}
    event = c.Event(**line, hash=c.chain_hash(prev, line))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(event.to_json() + "\n")
    return event


def _freeze_head(run_id: str, head: c.Event) -> None:
    run = c.read_json(paths.run_json(run_id), c.RunManifest)
    c.write_json(paths.run_json(run_id), run.model_copy(update={"events_head_hash": head.hash}))


def _pipeline_until_sheet(run_id: str, *, issuer: str = "freetsa", return_ots: str = "pending") -> c.Event:
    """The events a run has when the sheet stage starts, the blur-stage deletion row, and the frozen head."""
    blur_row = c.DeletionRecord(target="media/pickup.proxy.mp4", kind=c.DeletionKind.FILE, sha256="cd" * 32, bytes=1000, deleted_at="2026-10-27T15:02:00Z", reason="unblurred proxy deleted at blur.done")
    # privacy stamps this row with the wall clock; blur.done announces it, so a skewed clock cannot drop it
    late_clock = c.DeletionRecord(target="media/return.proxy.mp4", kind=c.DeletionKind.FILE, sha256="ce" * 32, bytes=900, deleted_at="2026-10-31T09:00:00Z", reason="unblurred proxy deleted at blur.done")
    c.write_json(paths.deletions_json(run_id), c.Deletions(run_id=run_id, rows=[blur_row, late_clock]))
    _append_event(run_id, c.EventKind.RUN_CREATED, {}, "2026-10-27T14:59:00.100Z")
    _append_event(run_id, c.EventKind.SEAL_MANIFEST, {"sha256": seal.sha256_file(paths.manifest_json(run_id))}, "2026-10-27T15:00:00.200Z")
    _append_event(run_id, c.EventKind.SEAL_TSR, {"tsr_time": "2026-10-27T15:00:02Z", "serial": "0x1f", "issuer": issuer}, "2026-10-27T15:00:02.300Z")
    _append_event(run_id, c.EventKind.SEAL_OTS, {"status": "pending", "pickup": "pending", "return": return_ots}, "2026-10-27T15:00:03.400Z")
    _append_event(run_id, c.EventKind.BLUR_DONE, {"role": "pickup", "sha256": "ef" * 32, "path": "media/pickup.proxy_blurred.mp4"}, "2026-10-27T15:02:00.500Z")
    _append_event(run_id, c.EventKind.BLUR_DONE, {"role": "return", "sha256": "ee" * 32, "path": "media/return.proxy_blurred.mp4"}, "2026-10-27T15:02:01.500Z")
    head = _append_event(run_id, c.EventKind.STAGE, {"name": "sheet", "pct": 95}, "2026-10-27T15:06:00.000Z")
    _freeze_head(run_id, head)
    return head


def _close_the_run(run_id: str, *, keep_plate: bool) -> list[c.DeletionRecord]:
    """The orchestrator's close stage after the sheet: receipts, the quarantine, deletions.json, manifest_run.json,
    run.completed, then an hourly ots upgrade."""
    _append_event(run_id, c.EventKind.SHEET_WRITTEN, {}, "2026-10-27T15:06:05.000Z")
    now = "2026-10-27T15:06:06.000Z"
    receipt = c.DeletionRecord(target="file_abc", kind=c.DeletionKind.FILES_API, sha256="ab" * 32, deleted_at=now, reason="run end (pickup proxy)", api_response={"id": "file_abc", "deleted": True})
    local = privacy.delete_quarantine(run_id, now=now)
    _append_event(run_id, c.EventKind.FILES_DELETED, {"file_id": "file_abc", "response": receipt.api_response}, "2026-10-27T15:06:06.100Z")
    _append_event(run_id, c.EventKind.QUARANTINE_DELETED, {"files": sorted(r.target for r in local)}, "2026-10-27T15:06:06.200Z")
    rows = c.read_json(paths.deletions_json(run_id), c.Deletions).rows + [receipt, *local]
    privacy.close_deletions(run_id, rows, keep_plate=keep_plate, chosen_by="renter at upload" if keep_plate else None)
    head = seal.events_through(run_id, None)[-1]
    seal.write_manifest_run(run_id, deletions=rows, events_head_hash=head.hash, config_hash=c.config_hash(TH), app_commit="abc1234", pdf_sha256=None)
    _append_event(run_id, c.EventKind.RUN_COMPLETED, {"status": "done"}, "2026-10-27T15:06:07.000Z")
    _append_event(run_id, c.EventKind.SEAL_OTS, {"status": "confirmed:921003"}, "2026-10-27T18:00:00.000Z")
    return rows


@pytest.mark.parametrize("keep_plate", [False, True])
def test_a_rebuild_after_run_end_prints_the_same_sheet(runs: Path, keep_plate: bool) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    run_id = f"01JREB{int(keep_plate)}"
    _synthetic_run(run_id, keep_plate=keep_plate)
    _pipeline_until_sheet(run_id)
    model, _ = sheet.write_sheet(run_id, th=TH)
    written = paths.sheet_json(run_id).read_bytes()
    frames = {p.name: p.read_bytes() for p in (paths.sheet_dir(run_id) / "frames").glob("*.jpg")}
    assert model.deletion_clocks == c.DeletionClocks()  # the run-end deletions come after this sheet
    assert [d.target for d in model.method.deletions] == ["media/pickup.proxy.mp4", "media/return.proxy.mp4"]
    assert model.method.storage_line.endswith(sheet.RUN_END_NOTE)
    assert model.method.rebuild_status.startswith("event chain verified to the head on page 1")
    assert model.originals[0].ots_status == "pending" and model.originals[0].tsr_time == "2026-10-27T15:00:02Z"
    if keep_plate:
        assert model.plate_line.startswith("plate kept") and any(name.endswith(".owner.jpg") for name in frames)

    _close_the_run(run_id, keep_plate=keep_plate)
    assert not [p for p in paths.quarantine_dir(run_id).rglob("*") if p.is_file()]
    rebuilt, _ = sheet.write_sheet(run_id, th=TH)  # walkaround rebuild in place, or a resumed sheet stage
    assert paths.sheet_json(run_id).read_bytes() == written
    assert rebuilt.plate_line == model.plate_line and rebuilt.originals[0].ots_status == "pending"  # the upgrade came after the head
    assert {p.name: p.read_bytes() for p in (paths.sheet_dir(run_id) / "frames").glob("*.jpg")} == frames  # owner frames kept
    assert seal.verify_run(run_id) == []

    # Written after close instead (the order section 8 item 7 wants), the sheet carries the receipts and still rebuilds.
    _freeze_head(run_id, seal.events_through(run_id, None)[-1])
    after, _ = sheet.write_sheet(run_id, th=TH)
    assert after.deletion_clocks.files_deleted_at == "2026-10-27T15:06:06.000Z" and after.deletion_clocks.quarantine_deleted_at == "2026-10-27T15:06:06.000Z"
    assert not after.method.storage_line.endswith(sheet.RUN_END_NOTE) and after.originals[0].bitcoin_block == 921003
    again = paths.sheet_json(run_id).read_bytes()
    _append_event(run_id, c.EventKind.SEAL_OTS, {"status": "confirmed:921002"}, "2026-10-27T19:00:00.000Z")
    assert c.json_bytes(sheet.build_sheet(run_id, th=TH)) == again


def test_sheet_time_check_names_a_tampered_original_or_a_broken_chain(runs: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    run_id = "01JTAMP"
    _synthetic_run(run_id, keep_plate=False)
    head = _pipeline_until_sheet(run_id)
    original = paths.quarantine_original(run_id, "return")
    data = bytearray(original.read_bytes())
    data[5] ^= 0x01
    original.write_bytes(bytes(data))
    assert sheet.build_sheet(run_id, th=TH).method.rebuild_status == "verify at sheet time: hash differs: quarantine/return.orig"
    original.unlink()  # deleted at run end: skipped here, checked by walkaround verify against manifest_run.json
    assert sheet.build_sheet(run_id, th=TH).method.rebuild_status.startswith("event chain verified to the head on page 1")
    lines = paths.events_path(run_id).read_text(encoding="utf-8").splitlines()
    lines[2] = lines[2].replace('"freetsa"', '"DigiCert"')
    paths.events_path(run_id).write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert sheet.build_sheet(run_id, th=TH).method.rebuild_status == "verify at sheet time: event chain broken at seq 2"
    assert head.seq == 6


def test_page_1_reads_the_stamps_from_the_chain_and_names_the_issuer(runs: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    run_id = "01JSTAMP"
    _synthetic_run(run_id, keep_plate=False)
    _pipeline_until_sheet(run_id, issuer="DigiCert", return_ots="failed:FileNotFoundError: no client")
    _append_event(run_id, c.EventKind.SEAL_OTS, {"status": "confirmed:921003"}, "2026-10-27T18:00:00.000Z")  # after the head
    model, _ = sheet.write_sheet(run_id, th=TH)
    pickup, ret = model.originals
    assert (pickup.ots_status, pickup.tsr_time, pickup.bitcoin_block) == ("pending", "2026-10-27T15:00:02Z", None)
    assert ret.ots_status == "pending"  # its own stamp failed; the manifest's proof holds its sha256
    assert model.method.thresholds["stamps"]["tsa_issuer"] == "DigiCert"
    html = paths.sheet_html(run_id).read_text(encoding="ascii")
    assert "2026-10-27T15:00:02Z (RFC 3161 token from DigiCert)" in html and "OpenTimestamps pending at the calendars" in html
    verify = model.appendices["C verify it yourself"]
    assert "from DigiCert" in verify and "ots info bundle/manifest.json.ots" in verify
    assert paths.bundle_file(run_id, "verify.md").read_text(encoding="ascii") == verify


def test_asset_src_inlines_only_files_inside_the_run(tmp_path: Path) -> None:
    (tmp_path / "sheet" / "frames").mkdir(parents=True)
    (tmp_path / "sheet" / "frames" / "k.jpg").write_bytes(b"\xff\xd8jpeg")
    (tmp_path.parent / "outside.jpg").write_bytes(b"\xff\xd8secret")
    src = sheet._asset_src(tmp_path)
    assert src("sheet/frames/k.jpg") == "data:image/jpeg;base64," + __import__("base64").b64encode(b"\xff\xd8jpeg").decode("ascii")
    assert src("../outside.jpg") == "../../outside.jpg" and src("sheet/frames/missing.jpg") == "../sheet/frames/missing.jpg"
    assert sheet._asset_src(None)("sheet/frames/k.jpg") == "../sheet/frames/k.jpg"


def test_build_sheet_on_a_bare_run(runs: Path) -> None:
    run_id = "bare"
    paths.ensure_run_layout(run_id)
    c.write_json(paths.run_json(run_id), c.RunManifest(run_id=run_id, created_at=NOW, thresholds=TH, config_hash=c.config_hash(TH)))
    model = sheet.build_sheet(run_id, th=TH)
    assert model.originals == [] and model.footer_hash_prefixes == [] and len(model.panels) == len(c.PanelId)
    assert model.method.rebuild_status == "verify at sheet time: bundle/manifest.json missing"
    assert model.clocks.sealed_at is None
    html = sheet.render_html(model)
    assert "not sealed" in html and "camera time not present" not in html and "unknown" in html and html.isascii()
