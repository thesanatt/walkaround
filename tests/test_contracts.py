"""Round-trip one of every contract model through canonical JSON; check paths, rules and thresholds load."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from walkaround import contracts as c
from walkaround import paths, rules

TS = "2026-10-27T15:04:05Z"
SHA = "a" * 64


def _instances() -> list[c.Contract]:
    clocks = c.ThreeClocks(sealed_at=TS, first_model_call_at=TS, sealed_minutes_after_camera=6, camera_time=TS)
    crop = c.Crop(path="crops/pickup/kf1.jpg", sha256=SHA, w=1280, h=900, scale=0.333333333, dx=100.5, dy=200.25)
    moments = c.Moments(cx=1.0, cy=2.0, mu20=3.0, mu02=4.0, mu11=0.5, theta=0.1234567891, sigma_inv_half=[[1.0, 0.0], [0.0, 1.0]])
    deletion = c.DeletionRecord(target="quarantine/pickup.orig", kind=c.DeletionKind.QUARANTINE, sha256=SHA, bytes=10, deleted_at=TS, reason="run end")
    mark = c.Mark(
        mark_id="m1", role="pickup", panel_id=c.PanelId.LEFT_FRONT_DOOR, kf_id="kf1", source=c.MarkSource.SPARK_BOX,
        type=c.MarkKind.CHIP, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S, description="small chip",
        grid_point=c.GridPoint(x=512, y=480), grid_bbox=c.GridBox(x_min=500, y_min=470, x_max=524, y_max=490),
        bbox_crop=[640.0, 423.0, 670.7, 441.0], point_crop=[655.4, 432.0], w=[0.12345678, -0.5], r=0.51, caps=["K3"],
    )
    event = c.Event(seq=1, ts=TS, run_id="r1", kind=c.EventKind.SEAL_MANIFEST, payload={"sha256": SHA}, prev=SHA, hash="0" * 64)
    lane = c.LaneRecord(
        lane_id="car_door", role="pickup", noun=c.Noun.CAR_DOOR, prompt_text="car door", media_size=[960, 540],
        params={"include_confidence": "true", "mask_encoding": "one_bit"}, status=c.LaneStatus.COMPLETED,
        frames_sent=600, frames_returned=580, video_frames_processed=600, ids=["0", "1"], dollars=0.12,
        diagnostics=[c.Diagnostic(severity="warning", code="ignored_field", message="x", line=3)],
    )
    return [
        clocks,
        c.DeletionClocks(files_deleted_at=TS),
        c.RunError(stage="tracking", text="429 after 6 tries"),
        c.RunCounts(panels=12, marks=9, matches=4),
        c.RunManifest(run_id="r1", created_at=TS, clocks=clocks, thresholds={"a": 1}, config_hash=SHA, fps=15.0),
        c.FfprobeInfo(codec="h264", container="mov,mp4", width=3840, height=2160, rotation=90, display_width=2160, display_height=3840, fps_avg=29.97, vfr=True, duration_s=40.5, creation_time=TS),
        c.ProxyRecord(path="media/pickup.proxy.mp4", fps=15.0, width=540, height=960, frames=608, pts_map_path="media/pickup.pts.json", sha256=SHA),
        c.BlurredProxyRecord(path="media/pickup.proxy_blurred.mp4", sha256=SHA),
        c.WavRecord(path="media/pickup.wav", duration_s=40.5, sha256=SHA),
        c.FilesApiRecord(file_id="file-123", uploaded_at=TS, expires_at=TS, deleted_at=TS, delete_response={"deleted": True}),
        c.VideoRecord(role="pickup", original_filename="IMG_0001.MOV", bytes=123456, sha256_browser=SHA, sha256_server=SHA, hash_match=True, received_at=TS, chunks_expected=3, chunks_received=3),
        c.ManifestEntry(path="quarantine/pickup.orig", sha256=SHA, bytes=5),
        c.Stamps(manifest_ots="pending", tsr_time=TS, tsa_serial="1", tsa_issuer="freetsa"),
        c.SealManifest(run_id="r1", files=[c.ManifestEntry(path="x", sha256=SHA, bytes=1)], deletions=[deletion], events_head_hash=SHA),
        event,
        c.Diagnostic(severity="error", code="bad_mask", message="m", line=1),
        c.MaskRef(h=20, w=30, encoding=c.MaskEncoding.ONE_BIT, payload_offset=120, payload_len=44),
        c.IndexObject(id="0", box=[1, 2, 30, 40], c=0.73, mask=c.MaskRef(h=38, w=29, encoding=c.MaskEncoding.LOSSLESS, payload_offset=1, payload_len=2)),
        c.IndexFrame(frame=7, objects=[c.IndexObject(id="1", box=[0, 0, 1, 1])]),
        lane,
        c.InputRef(path="crops/pickup/kf1.jpg", sha256=SHA),
        c.SparkUsage(input=1400, output=900, reasoning=700, cached=0),
        c.SparkRequestRecord(request_id="resp_1", purpose=c.SparkPurpose.MARKS, model="muse-spark-1.3", schema_name="marks_v1", raw_text="{}", parsed={"marks": []}, valid=True, dollars=0.0071, ms=6100),
        c.VoiceRequest(mode=c.VoiceMode.ENDPOINTING, model="muse-voice-transcribe-1.0", keywords=["scuff"]),
        c.VoiceTurn(turnId=0, startMs=500, endMs=15300, transcript="hi", extra_server_key="kept"),
        c.VoiceResponse(sessionId="s", transcript="hi", audioDurationMs=16600, turns=[c.VoiceTurn(startMs=1, endMs=2, transcript="x")]),
        c.VoiceRecord(role="pickup", request=c.VoiceRequest(mode=c.VoiceMode.ENDPOINTING, model="m"), response=c.VoiceResponse(transcript="t"), wav_sha256=SHA, mode_applied=False, dollars=0.004),
        c.TrackFragment(object_id="3", first_frame=10, last_frame=55),
        c.MergeRecord(rule=c.MergeRule.VELOCITY_GAP, gap_s=0.4, pred_dist=0.2, area_ratio=1.1),
        c.AnchorReprompt(request_id="resp_2", frame=90, box=[10, 10, 200, 300], negative_points=[[50.0, 60.0]]),
        c.FrameObs(frame=3, pts=0.2, box=[1.0, 2.0, 3.0, 4.0], box_area_frac=0.05, in_car_box=True, c=None, border_frac=0.0),
        c.PanelTrack(track_id="t1", role="pickup", noun=c.Noun.CAR_DOOR, sam_object_ids=["0"], fragments=[c.TrackFragment(object_id="0", first_frame=0, last_frame=9)], per_frame=[c.FrameObs(frame=0, pts=0.0, box=[0, 0, 10, 10], box_area_frac=0.1, in_car_box=True)], panel_id=c.PanelId.LEFT_FRONT_DOOR, side=c.Side.LEFT, position=c.Position.FRONT),
        c.TwinCheck(lid=["return_hood_00", "return_trunk_lid_00"], votes={"plate": -1, "mirror": -1, "lights": 0, "glass": 0}, numbers={"plate_on_lid_share": 0.91, "mirror_gap_lid_sizes": 2.57}, score=-4.0, decided=True),
        c.DoorCheck(was=c.PanelId.RIGHT_REAR_DOOR, votes={"lid_side": -1, "mirror_side": 0, "mirror": 1, "wheel": 0, "pair": 1, "walk": 1}, numbers={"mirror_gap_door_widths": 0.02}, side_score=-2.0, position_score=4.0),
        c.PartCheck(kind="side", was=c.PanelId.RIGHT_FRONT_WHEEL, votes={"lid_side": 0, "light_side": -1, "door_side": -1}, numbers={"door_side_frames": 30.0}, score=-4.0),
        c.PartCheck(kind="end", was=c.PanelId.WINDSHIELD, votes={"lights": -1, "bumper": 0, "lid": 0, "mirror": 1}, score=-1.0, decided=False),
        c.OffPanel(from_panel=c.PanelId.HOOD, from_kf="pickup_hood_kf1", off_frac=0.04, moved_to_kf="pickup_left_front_fender_kf1"),
        c.SecondPass(other_mark="return_hood_kf1_m1", window_crop=[10, 20, 181, 116], expected_crop=[95.5, 68.0], boxes=2, spark_request_id="resp_1"),
        c.HalvesFind(half=1, window_crop=[576, 0, 1280, 486], boxes=3, spark_request_id="resp_1"),
        c.HalvesMerge(halves_mark_id="pickup_hood_kf1_h1", half=0, full_bbox_crop=[600.0, 400.0, 640.0, 420.0], gap_patches=0.4, overlap=True, spark_request_id="resp_1"),
        c.RegistrationFit(kf_pair=["pickup_hood_kf1", "return_hood_kf1"], model="outline", ok=True, n_matches=9, n_inliers=4, spread=0.1, mask_iou=0.95, reproj_px=2.1, H=[[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]),
        c.ByEyeLine(kind="debris", text="debris on this panel; check by eye", mark_ids=["pickup_front_bumper_kf1_m1"]),
        c.OwnContrast(mark_id="pickup_rear_bumper_kf2_m2", kf_id="pickup_rear_bumper_kf2", result="faint", contrast=0.95, panel_share=0.003),
        c.RegistrationVouch(fit=c.RegistrationFit(kf_pair=["pickup_rear_bumper_kf1", "return_rear_bumper_kf2"], model="features", ok=True, n_matches=179, n_inliers=110, spread=0.665, mask_iou=0.858, reproj_px=1.32), vouched=True),
        c.KeyframePairTry(kf_pair=["pickup_rear_bumper_kf1", "return_rear_bumper_kf2"], gates=["G4"], mask_iou=0.68, vouch=c.RegistrationVouch(fit=c.RegistrationFit(kf_pair=["pickup_rear_bumper_kf1", "return_rear_bumper_kf2"], model="outline", ok=False, why="mask IoU after the warp 0.86"), vouched=False, why="the fit was refused (mask IoU after the warp 0.86)")),
        c.SameWalkNote(mark_id="pickup_rear_bumper_kf3_m1", from_kf="pickup_rear_bumper_kf3", to_kf="pickup_rear_bumper_kf2", merged_into="pickup_rear_bumper_kf2_m1", model="features", reproj_px=0.87),
        c.CopyNote(mark_id="pickup_rear_bumper_kf2_m1", kf_id="pickup_rear_bumper_kf2", into_mark="pickup_rear_bumper_kf1_m1", into_pair=["pickup_rear_bumper_kf1_m1", "return_rear_bumper_kf1_m1"], via="registration", distance=3.6, fit="outline", refused_fit=True),
        c.Rederived(from_run="r0", from_events_head_hash=SHA, stored_on="Nov 2, 2026", recomputed_on="Nov 2, 2026", text="verdicts recomputed on Nov 2, 2026 with this version's rules (code abc1234) from the model answers stored on Nov 2, 2026 by run r0; no new model calls"),
        c.RegistrationMark(mark_id="return_hood_kf1_m1", other_kf="pickup_hood_kf1", result="present", mapped_box=[10.0, 10.0, 30.0, 20.0], found_box=[12.0, 10.0, 32.0, 20.0], ncc=0.8, contrast_own=12.0, contrast_other=10.0, uncertainty_px=4.0, used="capped"),
        c.RegistrationMark(mark_id="return_mirrors_kf2_m3", other_kf="pickup_mirrors_kf3", result="present", found_box=[146.81, 65.15, 152.1, 68.94], ncc=0.846, uncertainty_px=3.06, used="already_there"),
        c.SpotCheck(mark_id="pickup_roof_kf1_m1", other_kf="return_roof_kf2", via="registration", point=[120.5, 44.25], inside=True),
        c.SpotCheck(mark_id="pickup_roof_kf1_m2", other_kf="return_roof_kf2", via="none"),
        c.OffPanelNote(mark_id="pickup_hood_kf1_m2", from_panel=c.PanelId.HOOD, off_frac=0.06),
        c.SideVotes(R1=2, R2=1),
        c.PanelRole(track_ids=["t1"], windows=[[8.0, 14.0]], seconds_visible=6.0, state=c.CoverageState.FILMED, keyframe_ids=["kf1"]),
        c.Panel(panel_id=c.PanelId.LEFT_FRONT_DOOR, noun=c.Noun.CAR_DOOR, side=c.Side.LEFT, position=c.Position.FRONT, instance_from=c.InstanceFrom.TRACKS, front_sign=1, per_role={"pickup": c.PanelRole(state=c.CoverageState.FILMED)}, comparable=True),
        c.KeyframeScore(area=0.1, completeness=0.9, sharpness=0.8, glare=0.01, frontality=0.7, total=0.05),
        crop,
        moments,
        c.BlurRecord(plate_boxes=[[1.0, 2.0, 3.0, 4.0]], request_ids=["resp_3"], plate_ruler_px=140.5, plate_ruler_c=0.9, plate_tilt_hw=0.5),
        c.HandleRecord(point_src=[100.0, 200.0], c=0.8, request_id="resp_4"),
        c.Conditions(glare=True),
        c.Keyframe(kf_id="kf1", role="pickup", panel_id=c.PanelId.LEFT_FRONT_DOOR, track_id="t1", f_proxy=120, pts=8.0, score=c.KeyframeScore(area=0.1, completeness=0.9, sharpness=0.8, glare=0.0, frontality=0.7, total=0.05), aspect_ratio=1.3, box_src=[0, 0, 100, 100], crop=crop, moments=moments, front_sign=1, marks_status=c.MarksStatus.COMPLETE),
        c.GridPoint(x=1000, y=0),
        c.GridBox(x_min=0, y_min=0, x_max=10, y_max=10),
        c.Landmark(name="handle", point=c.GridPoint(x=1, y=2)),
        c.SparkMark(type=c.MarkKind.SCRATCH, point=c.GridPoint(x=1, y=2), bbox=c.GridBox(x_min=0, y_min=0, x_max=3, y_max=4), size_class=c.SizeClass.M, confidence=c.Confidence.LOW, note="n"),
        c.SparkMarksAnswer(marks=[], conditions=c.Conditions(), landmarks=[], neighbour_car_visible=True),
        mark,
        c.BoxRefine(used="sam", spark_bbox_crop=[600.0, 400.0, 640.0, 420.0], spark_point_crop=[620.0, 410.0], sam_bbox_crop=[610.0, 404.0, 630.0, 418.0], sam_c=0.61, mask_px=180, inside=1.0, area_ratio=0.35, sam_request_id="resp_1"),
        mark.model_copy(update={"box_refine": c.BoxRefine(used="spark", spark_bbox_crop=[600.0, 400.0, 640.0, 420.0], note="SAM found nothing in the box")}),
        c.ZoomRefine(used="zoom", spark_bbox_crop=[600.0, 400.0, 640.0, 420.0], spark_point_crop=[620.0, 410.0], window_crop=[524, 362, 716, 458], zoom_bbox_crop=[628.0, 380.0, 640.0, 390.0], zoom_type=c.MarkKind.CHIP, zoom_boxes=2, moved_bbox_crop=[614.0, 375.0, 654.0, 395.0], spark_zoom_request_id="resp_2"),
        mark.model_copy(update={"zoom_refine": c.ZoomRefine(used="spark", spark_bbox_crop=[600.0, 400.0, 640.0, 420.0], window_crop=[524, 362, 716, 458], zoom_boxes=0, note="the zoomed look found no mark near the box")}),
        c.WhitenedMark(mark_id="m1", role="pickup", panel_id=c.PanelId.HOOD, kf_id="kf1", u=0.1, v=0.2, r=0.2236, frame_used=c.FrameUsed.HOOD_AXIS),
        c.ClipRecord(path="clips/pickup/m1.mp4", f0_src=1200, window_px=400, windows_src=[[0, 0, 400, 400]]),
        c.TrackPoint(f=45, box_clip=[500, 500, 540, 530], c=0.89, inside_panel=True, w=[0.1, 0.2], u_rel=0.01),
        c.MarkTrack(mark_id="m1", role="pickup", b1=0.2, b2=0.01, b_pass=True, panned=False, drift_rms=0.03, sign_agreement=0.5, pairs_n=12, verdict=c.StaysPutVerdict.FIXED),
        c.StaysPutNeighbour(object_id="3", kind="side mirror", ref="pickup_side_mirror_00", frames=90, used=True, residual_full_px=1.8, residual_near_px=0.9, centres=[[45.0, 232.5, 249.5]]),
        c.NeighbourTest(verdict=c.StaysPutVerdict.SLIDING, ratio_full=4.77, ratio_near=3.83, mark_full_px=27.4, used=6, near_frames=15, neighbours=[c.StaysPutNeighbour(object_id="3", kind="wheel", ref="t")]),
        c.MatchCandidate(pickup_mark="m1", return_mark="m2", d=0.18, d_pos=0.08, type_cost=0.1, size_cost=0.0, r_mean=0.5, tau_pre_r=0.28, tau_eye_r=0.56, ratio=None, margin=None, mutual=True),
        c.MarkGroup(type=c.MarkKind.CHIP, n_pickup=4, n_return=5, text="4 chips here at pickup, 5 at return, 1 new or unseen"),
        c.PanelMatch(panel_id=c.PanelId.LEFT_FRONT_DOOR, state=c.MatchState.COMPARABLE, kf_pair=["kf1", "kf9"], aspect_diff=0.02, mask_iou=0.81, handle_residual=[0.01, -0.02]),
        c.KeyframePairTry(kf_pair=["kf1", "kf8"], gates=["G4"], mask_iou=None, frames_agree=False),  # match lane 11/1 (rule 2)
        c.ReshapeCheck(kf_pair=["kf3", "kf1"], mask_iou=0.699, aspect_ratios=[0.988, 0.984], aspect_off=0.004, blob_share=0.64, called=True),  # lead 11/1 (second)
        c.FixtureDrop(mark_id="return_rear_bumper_kf1_m1", fixture="sensor", source="spark", overlap=0.92),  # lead decision 4 of 11/1
        c.PrivacyDrop(mark_id="pickup_hood_kf1_m2", kf_id="pickup_hood_kf1", share=0.81),  # R27
        c.VerdictFlags(return_mark=True, K2=True, K6=True),
        c.Trace(rule_ids=["L3", "K2"], thresholds={"tau_pre": 0.25}, spark_request_ids=["resp_1"], event_seqs=[3, 4], config_hash=SHA),
        c.SizeEstimate(inches=0.9, bound_lo=0.6, bound_hi=1.3, ruler=c.RulerKind.PLATE, ruler_source_kf="kf1", threshold_type=c.ThresholdType.DENT),
        c.Verdict(verdict_id="v1", panel_id=c.PanelId.LEFT_FRONT_DOOR, label=c.VerdictLabel.NEEDS_YOUR_EYE, rule_id="L4", mark_ids=["m1", "m2"], distance_units=0.3, distance_cm_est=10.5, trace=c.Trace(rule_ids=["L4", "K1"])),
        c.Calibration(sigma_frame=0.03, sigma_spark=0.05, sigma_view=0.04, sigma_within=0.0707, tau_pre=0.25, tau_eye=0.5, tau_fix=0.09, calibrated=True),
        c.PanelCoverage(state=c.CoverageState.GLIMPSED, state_basis=c.StateBasis.AREA_ONLY, seconds=0.8, best_kf="kf2", reason="0.8 s"),
        c.Odometer(asked=True),
        c.Coverage(per_role={"pickup": {c.PanelId.REAR_BUMPER: c.PanelCoverage(state=c.CoverageState.NOT_FILMED)}}, missing=[c.PanelId.LEFT_REAR_WHEEL, c.PanelId.REAR_BUMPER], topup_text="film the left rear wheel", nouns_to_rerun=[c.Noun.WHEEL, c.Noun.BUMPER, c.Noun.CAR, c.Noun.LICENSE_PLATE, c.Noun.FACE], rerun_seconds=41.2),
        c.NarrationTurn(turn_id="0", start_ms=1200, end_ms=3400, text="small chip, right rear door", keywords_hit=["chip", "car door"], pinned_panel=c.PanelId.RIGHT_REAR_DOOR, pin_seconds=2.2, pin_kind=c.PinKind.OVERLAP),
        c.NarrationRole(turns=[]),
        c.Narration(per_role={"pickup": c.NarrationRole()}, mode_applied=True),
        c.ClaimImage(sha256=SHA, received_at=TS, w=1200, h=800),
        c.ClaimRoi(box=[0, 0, 1200, 800], source=c.RoiSource.WHOLE),
        c.ClaimNaming(spark=c.PanelId.HOOD, sam_votes=[{"panel": "hood", "score": 0.9}], agreed=True),
        c.ClaimPlacement(method=c.PlacementMethod.FULL_PANEL, w_claim=[0.1, 0.2], r_claim=0.1),
        c.NotePrefill(fields={"panel": "hood"}, verified_against_typed=True),
        c.Claim(image=c.ClaimImage(sha256=SHA, received_at=TS, w=1, h=1), text="passenger-side door", exit=c.ClaimExit.E3, paragraph="p"),
        c.SheetOriginal(role="pickup", name="IMG_0001.MOV", bytes=1, sha256=SHA, camera_time=TS, received_at=TS, hash_match=True, ots_status="pending", tsr_time=TS),
        c.SheetFrame(kf_id="kf1", path="keyframes/pickup/kf1.frame.jpg", video_time_s=8.0, wall_clock=TS, mark_ids=["m1"]),
        c.SheetVerdictLine(verdict_id="v1", label=c.VerdictLabel.ALREADY_THERE, rule_ids=["L3"], mark_ids=["m1"], type=c.MarkKind.CHIP, size_class=c.SizeClass.S, distance_units=0.1, distance_cm_est=3.5, staysput_line="drift 0.03, b1 0.2, b2 0.01, not panned"),
        c.SheetChip(turn_id="0", start_ms=1, end_ms=2, text="t"),
        c.SheetPanelBlock(panel_id=c.PanelId.LEFT_FRONT_DOOR, display_name="left front door", pickup=c.SheetFrame(kf_id="kf1", path="p", video_time_s=1.0), **{"return": c.SheetFrame(kf_id="kf9", path="q", video_time_s=2.0)}, coverage_seconds={"pickup": 6.0}),
        c.SheetNotComparable(panel_id=c.PanelId.HOOD, rule_ids=["G8"], reasons=["hood IoU 0.6"]),
        c.SheetMethod(config_hash=SHA, eval_lines=["false-new 3 of 41 (7%)"], deletions=[deletion], storage_line="s", rebuild_status="rebuild identical"),
        c.SheetModel(run_id="r1", clocks=clocks, originals=[], note_lines=["a", "b", "c", "d", "e"], footer_hash_prefixes=[SHA[:12]]),
        c.LedgerTokens(**{"in": 1400, "out": 900, "reasoning": 700, "cached": 0}),
        c.LedgerLatency(upload=9900, first_delta=3600, done=8400),
        c.LedgerRow(ts=TS, kind="sam_video", model="sam-3.1", run_id="r1", note="car door", usd=0.12, unit=c.LedgerUnit.FRAMES, quantity=600, frames=600, latency_ms=c.LedgerLatency(done=8400)),
        deletion,
        c.Deletions(run_id="r1", rows=[deletion], keep_plate=False),
        c.ReplayPayload(run_id="r1", mode=c.ReplayMode.FIXED, recorded_on="2026-10-29", delays_ms={"tracking": 4000}, events=[event]),
    ]


@pytest.mark.parametrize("instance", _instances(), ids=lambda i: type(i).__name__)
def test_round_trip(instance: c.Contract) -> None:
    text = instance.to_json()
    back = type(instance).from_json(text)
    assert back.to_json() == text, type(instance).__name__
    data = json.loads(text)
    assert list(data) == sorted(data)  # sorted keys at the top level
    assert text.isascii()


def test_every_contract_model_is_covered() -> None:
    covered = {type(i) for i in _instances()}
    models = {
        obj for name, obj in vars(c).items()
        if isinstance(obj, type) and issubclass(obj, c.Contract) and obj not in (c.Contract, c.Verbatim)
    }
    missing = sorted(m.__name__ for m in models - covered)
    assert not missing, f"models without a round-trip case: {missing}"


def test_canonical_json_rounds_floats_and_sorts() -> None:
    text = c.canonical_json({"b": 1 / 3, "a": [2.0000000001, -0.0], "e": c.Side.LEFT, "p": Path("x/y")})
    assert text == '{"a":[2.0,0.0],"b":0.333333,"e":"left","p":"x/y"}'
    with pytest.raises(ValueError):
        c.canonical_json({"x": math.nan})


def test_write_read_and_hash(tmp_path: Path) -> None:
    kf = _instances()[0]
    path = tmp_path / "derived" / "x.json"
    digest = c.write_json(path, kf)
    assert digest == c.sha256_json(kf) == kf.digest()
    assert path.read_bytes() == c.json_bytes(kf)
    assert c.read_json(path, type(kf)) == kf
    marks = [m for m in _instances() if isinstance(m, c.Mark)]
    c.write_json(tmp_path / "marks.json", marks)
    back = c.read_json_list(tmp_path / "marks.json", c.Mark)
    # Floats are rounded to 6 places on the way out, so compare canonical text, not objects.
    assert [m.to_json() for m in back] == [m.to_json() for m in marks]


def test_event_chain_hash() -> None:
    prev = c.sha256_json({"files": []})
    line = {"seq": 0, "ts": TS, "run_id": "r1", "kind": "run.created", "payload": {}, "prev": prev}
    event = c.Event(**line, hash=c.chain_hash(prev, line))
    assert event.check()
    tampered = event.model_copy(update={"payload": {"x": 1}})
    assert not tampered.check()


def test_verdict_flags_list_caps() -> None:
    flags = c.VerdictFlags(return_mark=True, K2=True, K13=True, K6=True)
    assert flags.set_caps() == ["K13", "K2", "K6"]


def test_extra_keys_are_errors_except_verbatim() -> None:
    with pytest.raises(Exception):
        c.Mark.model_validate({"mark_id": "m", "role": "pickup", "panel_id": "hood", "kf_id": "k", "source": "spark_box", "type": "chip", "confidence": "low", "size_class": "S", "bogus": 1})
    row = c.LedgerRow.model_validate({"ts": TS, "kind": "spark", "model": "m", "usd": 0.01, "seconds": 5.4, "reasoning_tokens": 700})
    assert json.loads(row.to_json())["reasoning_tokens"] == 700


def test_enumerations() -> None:
    assert len(c.LANE_NOUNS) == 16
    assert len(c.PANEL_NOUNS) == 14
    assert set(c.PRIVACY_NOUNS) == {c.Noun.LICENSE_PLATE, c.Noun.FACE}
    assert len(c.BODY_PANELS) == 13 and len(c.WHEEL_PANELS) == 4 and len(c.LIGHT_AND_GLASS_PANELS) == 5
    assert set(c.PANEL_NOUN) == set(c.PanelId)
    assert c.noun_slug(c.Noun.LICENSE_PLATE) == "license_plate"
    assert [v.value for v in c.VerdictLabel] == ["already_there", "needs_your_eye", "new_or_unseen", "not_seen_at_return", "not_comparable"]
    assert [v.value for v in c.CoverageState] == ["filmed", "glimpsed", "not_filmed", "not_run", "not_seen"]
    with pytest.raises(Exception):
        c.VideoRecord(role="topup", original_filename="x", bytes=1)  # topup needs a number


def test_thresholds_load_and_hash() -> None:
    values = c.load_thresholds()
    assert values["media"]["proxy_fps"] == 15
    assert values["matching"]["tau_pre0_floor"] == 0.25
    assert values["spark"]["max_output_tokens"] == 4000
    assert "_meta" not in values
    raw = json.loads(paths.thresholds_path().read_text())
    for group, leaves in raw.items():
        if group.startswith("_"):
            continue
        for name, leaf in leaves.items():
            assert set(leaf) >= {"value", "source"}, f"{group}.{name} lacks value or source"
            assert leaf["source"], f"{group}.{name} has an empty source"
    assert len(c.config_hash(values)) == 64
    assert c.config_hash(values) == c.config_hash(json.loads(json.dumps(values)))


def test_rules_table() -> None:
    assert rules.gate_ids() == [f"G{i}" for i in range(1, 14)]  # G11 to G13 since R27
    assert rules.cap_ids() == [f"K{i}" for i in range(1, 22)]  # K21 since R29
    assert rules.rule("K7").label == "second look"
    assert rules.describe(["G1"])[0].startswith("G1 not filmed: ")
    with pytest.raises(KeyError):
        rules.rule("K99")
    flags = c.VerdictFlags()
    assert set(flags.set_caps()) <= set(rules.cap_ids())
    assert {f for f in type(flags).model_fields if f.startswith("K")} == set(rules.cap_ids())


def test_paths_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    root = paths.ensure_run_layout("r1")
    assert root == tmp_path / "runs" / "r1"
    assert (root / "sam" / "pickup").is_dir() and (root / "quarantine").is_dir()
    lane = paths.lane_events_path("r1", "pickup", c.Noun.LICENSE_PLATE)
    assert lane == root / "sam" / "pickup" / "license_plate.ndjson"
    assert paths.lane_index_path("r1", "return", "car door").name == "car_door.index.json"
    assert paths.tracks_json("r1") == root / "derived" / "tracks.json"
    assert paths.crop_path("r1", "pickup", "kf1").parent.is_dir()
    assert paths.rel("r1", paths.sheet_json("r1")) == "sheet/sheet.json"
    assert paths.ledger_path() == tmp_path / "runs" / "ledger.jsonl"
    assert set(paths.FROZEN_DERIVED_FILES) <= set(paths.DERIVED_FILES)
