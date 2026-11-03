# Lead notes on this memo (Oct 31, 4:50 PM). Read these before acting on anything below.

1. E2.tracks failed because of how it is measured, not because SAM fragments tracks. The check divides one id's frames
   by the frames where any door (or bumper) is visible, and a full lap shows four doors and two bumpers. Each door came
   back as one unbroken track (E4: nothing to join), and three repeats of "car door" agree at mean best IoU 0.97 to
   0.99. Decision: keep hosted SAM 3.1 at 15 fps. Do NOT turn the anchor pass on for every panel, and do not switch
   to fal. The check gets fixed to measure each id against its own visible window (follow-up).
2. E5, Stays Put, is not calibrated, and until it is it must NOT cap verdicts (no K6 from Stays Put). Measured at
   5:10 PM on real paint, the pair C trunk chips at 16.4 s (staysput_redo/, $0.04): whitened drift 0.218 (mark clip)
   and 0.211 (panel clip), labelled "sliding" at tau_fix 0.08; the carried-moments residual is 0.48 units, so the
   whitened frame itself is the noise. The memo's "reflection" control (auto_hood_glare, drift 0.34) sits on the white
   trunk lid, where the glare rule (V > 250, S < 40) picks plain white paint, so it is no reflection test. Panel-box
   coordinates do not separate them either (paint 0.063 rms, the control 0.074), and an affine fitted to five
   co-tracked anchors on the same lid leaves paint 12 to 14 px off within +/- 0.5 s (8 px on 5-frame steps;
   scratchpad anchor_exp, $0.02), mostly SAM box jitter on 40 px objects. Do NOT adopt tau_fix 0.507 or any
   threshold from these numbers. Decision: Stays Put runs, records drift, b1, b2 and its label, writes the blurred
   clips and the trails for the UI, and prints them on the method page, but match.py ignores it for the cap (a
   thresholds flag such as staysput.cap_enabled = false, read by match). Recalibrate on dark paint with a real
   reflection (Pexels 4228872, the black S550; or the friend's walk) before turning the cap back on.
   A neighbour-based test that does separate paint from a reflection on two clips is in docs/STAYS_PUT_ANCHORS.md
   (paint 0.6 to 0.8x its neighbours' residual, a reflection on the black S550 3.8 to 4.8x).
3. E6: the 6.5% jitter (10% at medium) and Jaccard 0.5 come from the runner's automatic crops (a door half out of
   frame at the clip's last frame, a wide bumper). Re-run at 5:00 PM on the real marks (spark_redo/gate_spark.json,
   $0.12, effort low, 5 calls each): trunk chips in both walks sigma 2.0% of crop width (0.073 and 0.090 units),
   Jaccard 1.00 and 0.80, 6 of 8 hand-marked chips hit; the Alto hood dent sigma 2.4% (0.117 units), Jaccard 0.72,
   the dent hit. tau_pre0 from sigma_spark is 0.415. Keep effort low (cheaper and steadier), raise
   max_output_tokens to 8000 (one incomplete answer at medium), keep the band calibrate computed (tau_pre0 0.4,
   tau_eye 0.8). SAM box-prompt refinement of Spark marks stays a nice-to-have, not a fix.
4. E8.cross_walk 0.82 is side-agnostic (doors were not matched by side), so it says nothing about the matcher yet.
5. E11: claim naming agreed on 2 of 3 photos. Ship layer 1 with the one-tap panel chooser as the default.
6. E0 and E12: ots is installed now (walkaround/.venv/bin/ots; the seal finds it since commit e39353e), so seals stamp.
7. E17: S2 (Spark timeline over the proxy) is dropped (5 of 22 panels within 1 s). E18 note prefill passed.
8. E10 failed here only because the licensed pickup has no audio track. The Voice gate ran on our narrated copy
   (voice_narrated/gate_voice.json: 3 turns, every noun-naming turn lands).
9. Cost: about $4.33 per pair at 15 fps from the parts (E13). D1 itself cost $9.69 because of the 10 and 30 fps sweeps.

# D1 decision memo

Run `d1_20261031T195329Z`, started 2026-10-31T19:53:29Z, finished 2026-10-31T20:21:20Z, folder runs/d1/uco3d_pairC.
Spend on ledger rows with this prefix: $9.6949 of the $15.00 ceiling; the whole ledger stands at $10.1761.
Thresholds config hash 4d209b15dedecf0856e2c8f4f9646b1e6f70a8b6cf004c49ed2888740793c57e. Plan: none (picks below come from gate_sam's lanes).

Criteria: 27 pass, 12 fail, 25 not run, 8 measured with no bar.

## Recommendation

**Use hosted SAM 3.1 with the anchor pass on every panel by default, then re-run E2**

- E2.tracks failed at every fps tried (15 fps fail; 10 fps fail; 30 fps fail; at 15 fps: pickup car door 0.58/1 frags, pickup bumper 0.54/1 frags, return car door 0.72/1 frags, return bumper 0.61/1 frags); E4 measured no join (info: nothing to join: no door track came back in 2 or more fragments), so section 11's first branch stands.
- Section 11's branch: run the anchor pass on every panel by default and re-test; if tracks still fail 3 hours after the footage landed, switch to SAM_ENGINE=fal and re-run E2 through it.
- Named fallback during the build or judging: SAM_ENGINE=fal flips every video lane without a deploy (section 14 item 33).

## Decisions

The list section 11 asks the memo for, each with the criteria it comes from.

| Item | Decision | From |
|---|---|---|
| fps | none passed E2.tracks (15 fps fails; 10 fps fails; 30 fps fails) | E2.tracks |
| noun list | the 16 lanes of section 5; door handle as image requests only; empty lanes this run: zero_matches: D1 E2 face, D1 E2 roof; none incomplete | E2.zero_matches |
| concurrency policy | 16 at once per video | E2.concurrency, E2.no_429_loss |
| SAM engine | Use hosted SAM 3.1 with the anchor pass on every panel by default, then re-run E2 | E2.tracks, E4.joins |
| decode path | Python process pool | E2.decode |
| store false or the deletion path | store false on every request, accepted by both models | E12.store_false_sam, E12.store_false_spark |
| Stays Put input type and state | not measured | E5.demo_mark, E5.kill |
| b floors and the panned share | b1 observed 0.297 to 0.297 against the 0.15 floor; b2 observed 0.437 to 0.437 against 0.05; panned: no panned clip in the plan | E5 |
| Spark effort, tokens and truncation | effort low; in 1300, out 1243 of which reasoning 1080, $0.0061 per call, 6102 ms; truncation signal seen yes, none at 4000 no | E6 |
| tau_pre0, tau_fix, sigma_frame | tau_pre0 0.400, tau_eye 0.800, tau_fix 0.507, sigma_frame 0.1689, sigma_spark 0.1095 (match.calibrate) | E6, E8 |
| count mode and the second look | count mode undecided (no targets); second look not measured | E7.count, E6.second_look_* |
| the ruler | alive: license plate and wheel | E11.plate_ruler, E11.wheel_ruler |
| pickup-check wording | check coverage before you drive off (pickup-only 51 s from parts) | E13.pickup_wall |
| capture path | path (a), screencapture of the Claude Chrome window, until E14 runs by hand | E14 |
| S1, S2 and S2b for Mon 11/2 | S1: not measured here (gate_s1.py); S2: drop; S2b: schedule | E16, E17, E18 |

## What failed, and section 11's branch for it

| Criterion | Measured | Section 11's branch |
|---|---|---|
| E0.tools | missing: ots, modal, modal token, railway whoami | Sanat installs what is missing (uv pip install opentimestamps-client modal; railway login) |
| E0.ots | 0 .ots files; freetsa yes | stamps print pending; the freetsa token stands alone until ots works |
| E2.tracks | 15 fps fail; 10 fps fail; 30 fps fail; at 15 fps: pickup car door 0.58/1 frags, pickup bumper 0.54/1 frags, return car door 0.72/1 frags, return bumper 0.61/1 frags | anchor pass on every panel by default and re-test; still failing 3 h after the footage: SAM_ENGINE=fal; fal failing too: One Walk |
| E5.control | auto_hood_glare mark: drift 0.341, sign 0.58 on 77 pairs; auto_hood_glare panel: drift 0.311, sign 0.39 on 33 pairs | the moment loses its reflection inset |
| E5.carried_moments | 2 frames, max residual 0.2725 units | decode the panel mask on every clip frame |
| E6.sigma | low 6.48% of crop width (0.1095 units), medium 10.08% | 2 to 3% with medium meeting 2%: medium becomes the default; over 3% at both: tau_pre past the 0.4 cap, hoods and corners go needs your eye |
| E6.jaccard | low worst crop median 0.500, pooled 1.000; medium 0.500 | dedup, count union and the second look carry more weight; README says so |
| E6.no_truncation | low: 0 incomplete, 0 truncated, medium: 1 incomplete, 0 truncated | max_output_tokens 8000 by default |
| E8.sigma_frame | 0.1102, 0.1689 | the band widens; tau_fix and tau_pre0 follow |
| E8.cross_walk | 0.8165 units (mirrored; 4K door keyframes, mean of up to 3 per walk) | the band widens and the method page says why |
| E11.naming | 2 of 3 agree | layer 1 ships with the one-tap chooser as the default |
| E17.timeline | 5 of 22 panels within 1 s; {'cached': 0, 'input': 5216, 'output': 2583, 'reasoning': 1890} tokens, $0.017498 | S2 dropped |

## Inputs

| Role | File | sha256, first 16 | Camera time | Video |
|---|---|---|---|---|
| pickup | data/candidates/uco3d_car_automobile/uco3d_7336-29949-54766.mp4 | 0afacdbbd4799b35 | not present | 1920x1080 at 29.58 fps, 19.7 s |
| return | data/candidates/uco3d_car_automobile/uco3d_50152-5241-46685.mp4 | f49b432f2932d4a9 | not present | 1920x1080 at 29.58 fps, 20.2 s |

Picked by the runner where the plan named nothing:

- second look: auto pairs (weak), the plan names no pair
- claims: the blurred crops stand in for staged claim photos
- E6 crop auto_door: uco3d_7336-29949-54766.mp4 at 19.6 s, box [672, 0, 928, 752]
- E6 crop auto_bumper: uco3d_7336-29949-54766.mp4 at 2.6667 s, box [0, 454, 1681, 1030]
- E6 crop auto_door_return: uco3d_50152-5241-46685.mp4 at 6.8 s, box [562, 0, 1330, 845]
- E5 auto_mark: uco3d_7336-29949-54766.mp4 at 19.6 s, mark [754.0, 492.5, 802.5, 545.0] (dent seen in 4 of 5 calls on auto_door)
- E5 auto_hood_glare: uco3d_50152-5241-46685.mp4 at 4.0667 s, mark [336.0, 176.0, 474.0, 280.0] (the brightest unsaturated blob on the return walk's largest hood frame (5.9 item 4))

## Every criterion, by row

### E0 Setup (20 min, $0): fail

Step `E0`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E0.tools | every command answers: ots --version, modal token, railway whoami, node --version, ffmpeg -version with libx265 | missing: ots, modal, modal token, railway whoami | fail | uv pip install opentimestamps-client modal (into walkaround/.venv; Sanat runs it, this runner never installs) |
| E0.ots | ots and freetsa smoke on a text file: two .ots files exist | 0 .ots files; freetsa yes | fail | FileNotFoundError: the OpenTimestamps client is not installed; pip install opentimestamps-client==0.7.2 (the `ots` binary); FileNotFoundError: the OpenTimestamps client is not installed; pip install opentimestamps-client==0.7.2 (the `ots` binary) |
| E0.hashes | the originals are hashed before any conversion | pickup 0afacdbbd479, return f49b432f2932 | pass | hashed before the proxies |
| E0.proxies | proxies of the pickup walk at 15, 10 and 30 fps | 15 fps 296 frames in 1.2 s, 10 fps 198 frames in 1.0 s, 30 fps 592 frames in 1.8 s | pass |  |

### E1 Upload and proxy timing (30 min, $0): partial

Step `E1`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E1.upload_wifi | 300 MB through the chunk route to the deploy from the Mac on wifi under 60 s, and the phone on LTE |  | not run | needs the deploy and the chunk route; run by hand from the Mac on wifi and the phone on LTE |
| E1.proxy_mac | the 4K30 take to the 960 px 15 fps proxy on the M4 Max at preset medium and veryfast | medium 1.3 s, veryfast 0.6 s for a 19.7 s source | info | the 90 s container bar is about a shared vCPU; this Mac number is a lower bound |
| E1.proxy_container | container proxy under 90 s at medium |  | not run | needs the Railway container |

### E2 Panel sweep (60 min, about $7 at 15 and 10 fps on both walks): fail

Step `sam`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E2.tracks | door and bumper tracks >= 70% of visible frames with <= 3 fragments after 5.1 at the chosen fps | 15 fps fail; 10 fps fail; 30 fps fail; at 15 fps: pickup car door 0.58/1 frags, pickup bumper 0.54/1 frags, return car door 0.72/1 frags, return bumper 0.61/1 frags | fail | no fps passed |
| E2.completion | p50 completion under 120 s at 400 frames and under 180 s at 600 | p50 14.0 s against 120 s at 304 frames | pass |  |
| E2.wire_parse | wire parse under 5 s per lane | max 0.022 s CPU per lane | pass |  |
| E2.decode | keyframe decodes under 60 s per video on the M4 Max | 96 masks in 39.41 s CPU | pass |  |
| E2.no_429_loss | no request lost to 429 after backoff | lost to 429: none; failed otherwise: none; retries 1 | pass | sam_client counts 429 backoffs with the one retry of a broken stream; the propagation_direction probe is read under E2.prop_both |
| E2.concurrency | 16 concurrent accepted | 38 lanes started, failed none; first delta grows with request index: no | pass |  |
| E2.zero_matches | a completed lane with no delta reads as zero_matches, never incomplete | zero_matches: D1 E2 face, D1 E2 roof; none incomplete | pass |  |
| E2.frames_billed | usage.video_frames_processed equals frames_sent, the difference logged | equal on every lane | pass |  |
| E2.oracle | the wire reader agrees with meta_sam_parser on one lane (boxes, ids, c, frames) | D1 E2 car door: agree yes, reader 0.013 s vs parser 34.411 s; D1 E2 car door: agree yes, reader 0.012 s vs parser 31.126 s | pass |  |
| E2.overlay | raster placement checked by overlay on 3 frames | 55 masks drawn, 0 misplaced, 172 frames pixelated; 62 masks drawn, 0 misplaced, 176 frames pixelated | pass |  |
| E2.prop_both | one text lane with propagation_direction both: accepted, ignored or a 400 | status completed, metadata echo {'include_confidence': 'true', 'mask_encoding': 'one_bit', 'propagation_direction': 'both'}, 225 frames; status completed, metadata echo {'include_confidence': 'true', 'mask_encoding': 'one_bit', 'propagation_direction': 'both'}, 247 frames | info |  |
| E2.variance | 'car door' three times for run-to-run variance | mean best IoU 0.991 on 226 shared frames, mean best IoU 0.974 on 224 shared frames, mean best IoU 0.976 on 224 shared frames; mean best IoU 0.968 on 247 shared frames, mean best IoU 0.990 on 247 shared frames, mean best IoU 0.977 on 247 shared frames | info |  |

### E3 c semantics and states (15 min, in E2's data): pass

Step `sam`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E3.states | states agree across the three 'car door' runs on >= 11 of 12 panels, c basis | pickup c basis 1.00 (5 of 5), area-only 1.00; return c basis 1.00 (6 of 6), area-only 1.00 | pass | pass at >= 11/12 of the tracks matched across runs |
| E3.c_semantics | c on first, mid and propagated frames per noun; c_present_share; whether c varies per frame | c varies per frame on 0 of 32 text lanes; c present on 100.0% of records at worst | info | 10/26 measured c constant per track on text prompts |

### E4 Anchor bridge (20 min, $0.08): partial

Step `sam`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E4.joins | the bridge joins the fragments | nothing to join: no door track came back in 2 or more fragments | info | nothing to join: no door track had 2 or more fragments, so the bridge ran on one track (two prompts 2 s apart) for the glass overlap and the negative point only; nothing to join: 1 fragment(s) on the bridged track |
| E4.negative_point | the negative glass point lowers glass overlap | -, - | not run | needs a car window inside the door box at the first prompt frame |

### E5 Stays Put (45 min, about $0.20): fail

Step `staysput`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E5.demo_mark | the prompted id returns a box on >= 80% of the 90 frames and lies inside the panel on >= 80% of complete-panel frames with RMS drift under 0.08 |  | not run | refused: auto_mark: 91 frames around 19.600 s run past the ends of uco3d_7336-29949-54766.mp4 |
| E5.control | the reflection drifts over 0.16 with the sign test over 80% on >= 8 pairs | auto_hood_glare mark: drift 0.341, sign 0.58 on 77 pairs; auto_hood_glare panel: drift 0.311, sign 0.39 on 33 pairs | fail |  |
| E5.kill | kill when both demo-mark clips fail on both clip types |  | not run | no demo-mark clip; refused: auto_mark: 91 frames around 19.600 s run past the ends of uco3d_7336-29949-54766.mp4 |
| E5.carried_moments | the carried-moments check of 5.5 on 5 frames of one clip, under 0.02 units | 2 frames, max residual 0.2725 units | fail |  |
| E5.panned | inconclusive share on the deliberately panned pass |  | not run | no panned clip in the plan |
| E5.scuff_text | 'scuff' as a text prompt once |  | not run | no scuff prompt |

### E6 Spark jitter and the second look (30 min, about $0.70): fail

Step `spark`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E6.sigma | sigma <= 2% of crop width (about 0.07 units) at effort low | low 6.48% of crop width (0.1095 units), medium 10.08% | fail | effort default: low |
| E6.jaccard | median Jaccard of the mark sets >= 0.7 at low | low worst crop median 0.500, pooled 1.000; medium 0.500 | fail |  |
| E6.no_truncation | no completed call truncated at 4000 | low: 0 incomplete, 0 truncated, medium: 1 incomplete, 0 truncated | fail |  |
| E6.truncation_signal | one call at max_output_tokens 600 shows status incomplete, reason max_output_tokens | reason max_output_tokens | pass |  |
| E6.second_look_valid | the second look returns valid JSON |  | not run | no data |
| E6.second_look_control | the second look says unsure or different on the control |  | not run | no data |
| E6.false_marks | false marks on the clean crop |  | not run | no crop labelled clean |
| E6.targets | error on the hand-measured targets |  | not run | no hand-measured targets |

### E7 Count mode (10 min, $0.08): measured

Step `spark`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E7.count | count mode finds a target the box pass missed on the 3 crops; kill when it finds none | 1 new points beyond the box pass | info | no hand-measured targets, so the kill rule cannot be read |

### E8 Door handle and the cross-take residual (25 min, $0.13): fail

Step `sam`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E8.sigma_frame | sigma_frame under 0.05 from one 'door handle' video request over 10 consecutive frames | 0.1102, 0.1689 | fail | the bar is an estimate |
| E8.cross_take | cross-take residual under 0.15 between takes |  | not run | no take 2 given (--take2) |
| E8.cross_walk | residual under 0.25 across walks | 0.8165 units (mirrored; 4K door keyframes, mean of up to 3 per walk) | fail | the bar is an estimate; side-agnostic because the doors are not matched by side |

### E9 Blur lanes (15 min, in E2's data plus $0.02): pass

Step `sam`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E9.plate_coverage | plate-track coverage >= 90% where the plate is visible, with the 0.5 s carry-over | pickup raw 1.00, carried 1.00 over 2 visible window(s) [[0.667, 5.0], [13.133, 19.733]]; return raw 1.00, carried 1.00 over 2 visible window(s) [[0.0, 6.133], [14.6, 19.133]] | pass | visible windows split at gaps over 2 s between detections (an estimate); check each split gap on the overlay |
| E9.image_checks | image checks on 3 keyframes, run on the unblurred frame first, plate extent recorded | pickup: 3 frames, extents [[341.74554443359375, 0.7153343475332635, 45.655266617297436], [336.9970703125, 0.7397785147137662, 47.477633301401625], [332.981689453125, 0.7515195015298121, 48.29321789795592]]; return: 3 frames, extents [[318.9500427246094, 0.6232960223817868, 36.66053890431913], [321.4217834472656, 0.613697713586969, 35.43899437643678], [318.30511474609375, 0.6304392658181197, 37.524043739905466]] | info |  |

### E10 Voice (10 min, $0.01): not run

Step `voice`: failed (exit 1).

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E10.turns | ENDPOINTING turns non-empty, the mode string validated |  | not run | step failed: exit 1 |
| E10.mode_applied | no 'mode not applied' (empty turns with a transcript) |  | not run | step failed: exit 1 |
| E10.pins | >= 80% of noun-naming turns land where SAM shows that panel |  | not run | step failed: exit 1 |
| E10.diarization | DIARIZATION as the comparison |  | not run | step failed: exit 1 |

### E11 Claim-first probes (30 min, $0.15): fail

Step `spark`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E11.naming | Spark naming agrees with the SAM vote on 3 of 3 claim photos | 2 of 3 agree | fail |  |
| E11.plate_ruler | plate mask over 60 px with c >= 0.5 on a 4K frame | 341.8 px at c 0.88 | pass |  |
| E11.wheel_ruler | wheel major axis over 120 px | major axis 179.0 px | pass |  |
| E11.ppi | ppi from plate and wheel within 15% of each other (estimate) |  | not run | needs both rulers and a tire code |
| E11.pixelation | plate ppi unblurred against the same frame pixelated, over 20% apart (the step 7 order) | no plate found on the pixelated frame | pass |  |
| E11.door_pair | the same door whitened from a walk frame and the close photo (r_claim) |  | not run | no door pair, or the handle not found in both |

### E12 Seal, Files and storage (15 min, $0): partial

Step `E12`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E12.ots_latency | ots stamp under 5 s |  | not run | FileNotFoundError: the OpenTimestamps client is not installed; pip install opentimestamps-client==0.7.2 (the `ots` binary); FileNotFoundError: the OpenTimestamps client is not installed; pip install opentimestamps-client==0.7.2 (the `ots` binary) |
| E12.ots_upgrade | whether ots upgrade lands within the evening |  | not run | no ots stamp from E0 to upgrade |
| E12.freetsa | the freetsa round trip | 0.3 s, time 2026-10-31T19:53:35Z | pass |  |
| E12.expires_after | expires_after accepted by the Files API | expires_at 1793480027; expires_at 1793480271 | pass |  |
| E12.files_delete | DELETE /v1/files returns a printable body, stored | {"deleted": true, "id": "file-2913621379038970", "object": "file"}; {"deleted": true, "id": "file-1779825103277912", "object": "file"} | pass |  |
| E12.store_false_sam | sam-3.1 accepts store false (the response echo) | store false echoed on every lane | pass |  |
| E12.store_false_spark | muse-spark-1.3 accepts store false (the response echo) | {'False': 49} over 49 answers | pass |  |
| E12.responses_delete | whether DELETE /v1/responses/{id} exists and what it returns, stored | {"body": {"deleted": false, "error": "Error code: 404 - The requested response was not found.", "id": "resp_6ae6474f9c68c3316e2c48c8", "status_code": 404}, "req | info |  |

### E13 Wall clock and dollars (30 min, about $9): pass

Step `E13`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E13.pickup_wall | pickup-only check under 3 min keeps 'check coverage before you drive off' | 50.5 s (proxy 1.3 + Files 12.9 + lanes 36.4) | pass | computed from parts, not an end-to-end run; the browser upload is not in it |
| E13.pair_dollars | a pair under $7 | $4.33 at 15 fps | pass | computed from parts, not an end-to-end run |
| E13.pickup_dollars | a pickup-only check under $3 | $1.72 | pass | computed from parts, not an end-to-end run |

### E14 Capture path and browsers (20 min, $0): not run

Step `-`: not run here (needs the replay route in a browser and screencapture of the Claude Chrome window; run by hand).

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E14.capture | capture path (a) at 60 fps |  | not run | needs the replay route in a browser and screencapture of the Claude Chrome window; run by hand |
| E14.browsers | @meta-sam/video on the H.264 proxy in Safari 18 and on the iPhone; the chunk upload from the phone |  | not run | needs the replay route in a browser and screencapture of the Claude Chrome window; run by hand |

### E15 Lane size and worker parse (10 min, $0): partial

Step `sam`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E15.lane_bytes | the 14 display nouns under about 30 MB per pair (estimate) | 5.92 MB for 14 display nouns per pair | pass |  |
| E15.worker | no main-thread task over 50 ms while the Web Worker parses |  | not run | a browser measurement (the Web Worker on the E2 fixture lanes) |

### E16 S1 probe, Spark Open on Baseten (30 min, about $0.10): not run

Step `-`: not run here (needs Baseten (Muse Spark Open); scripts/gate_s1.py in another lane owns it).

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E16.s1 | Spark Open on Baseten: sigma <= 2% and >= 90% valid JSON first try; the second look valid |  | not run | needs Baseten (Muse Spark Open); scripts/gate_s1.py in another lane owns it |

### E17 S2 probe, timeline over the blurred proxy (15 min, under $0.50): fail

Step `sam`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E17.timeline | >= 70% of panels within 1 s of SAM's windows | 5 of 22 panels within 1 s; {'cached': 0, 'input': 5216, 'output': 2583, 'reasoning': 1890} tokens, $0.017498 | fail |  |

### E18 S2b probe, note prefill (15 min, under $0.20): pass

Step `spark`: done.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E18.note_prefill | 5 of 5 fields agree on 2 of 2 synthetic agreements | fields agreed per agreement [5, 5] | pass |  |

### E19 fal.ai adapter ($0 until used): not run

Step `-`: not run here (needs the fal.ai adapter; scripts/gate_fal.py in another lane owns it).

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
| E19.fal | the fal adapter round-trips its recorded fixture |  | not run | needs the fal.ai adapter; scripts/gate_fal.py in another lane owns it |

## E13 dollars from the parts

measured unit prices times section 5's counts; the browser upload of E1 is not in the wall clock; gate_wallclock.py through the orchestrator replaces this

| Pair item | Count | Unit $ | $ | Basis |
|---|---|---|---|---|
| 16 lanes per walk, both walks (ledger) | 1 | 1.9792 | 1.9792 | measured |
| anchor bridges | 12 | 0.0338 | 0.4056 | measured |
| keyframe blur image checks (78 keyframes x 2) | 156 | 0.0025 | 0.3900 | price |
| door handle image requests | 24 | 0.0025 | 0.0600 | price |
| Stays Put clips | 24 | 0.0182 | 0.4368 | measured |
| clip blur checks (24 clips x 2) | 48 | 0.0025 | 0.1200 | price |
| Spark box marks (78 crops) | 78 | 0.0061 | 0.4795 | measured |
| Spark count mode (78 crops) | 78 | 0.0044 | 0.3420 | measured |
| second looks | 12 | 0.0061 | 0.0738 | estimate |
| narration seeds | 6 | 0.0061 | 0.0369 | measured |
| voice, both walks | 2 | 0.0025 | 0.0050 | estimate |

| Pickup-only item | Count | Unit $ | $ | Basis |
|---|---|---|---|---|
| 16 lanes on the pickup walk (ledger) | 1 | 1.0064 | 1.0064 | measured |
| keyframe blur image checks (39 keyframes x 2) | 78 | 0.0025 | 0.1950 | price |
| Spark box marks (39 crops) | 39 | 0.0061 | 0.2398 | measured |
| Stays Put clips on one walk | 12 | 0.0182 | 0.2184 | measured |
| clip blur checks (12 clips x 2) | 24 | 0.0025 | 0.0600 | price |

## Steps and spend

| Step | Rows | Status | Seconds | Planned $ | Spent $ | Reason |
|---|---|---|---|---|---|---|
| E0 | E0 | done | 5.4 | 0.00 | 0.0000 |  |
| E1 | E1 | done | 1.8 | 0.00 | 0.0000 |  |
| sam | E2, E3, E4, E8, E9, E15, E17 | done | 1212.6 | 4.29 | 9.1571 |  |
| spark | E6, E7, E11, E18 | done | 381.6 | 0.72 | 0.4939 |  |
| staysput | E5 | done | 68.9 | 0.11 | 0.0439 |  |
| voice | E10 | failed | 0.2 | 0.00 | 0.0000 | exit 1 |
| E12 | E12 | done | 0.0 | 0.00 | 0.0000 |  |
| E13 | E13 | done | 0.0 | 0.00 | 0.0000 |  |

## Files

- sam: sam/gate_sam.json
- spark: spark/gate_spark.json
- staysput: staysput/gate_stays_put.json
- results.json holds every number above with the gate JSONs it came from.

Unblurred intermediates deleted at the end of the run:

- sam/proxies/pickup_10fps.mp4 (13344344 bytes, sha256 2adcddfeece0)
- sam/proxies/pickup_15fps.mp4 (16408949 bytes, sha256 91b587fb0840)
- sam/proxies/pickup_30fps.mp4 (20714970 bytes, sha256 75207b7c5a1c)
- sam/proxies/return_10fps.mp4 (12452508 bytes, sha256 4f6d666db2ba)
- sam/proxies/return_15fps.mp4 (15180664 bytes, sha256 095b907b280c)
- sam/proxies/return_30fps.mp4 (19236698 bytes, sha256 f21175e8b2a6)
- sam/clips/pickup_bridge.mp4 (5555509 bytes, sha256 177c31100125)
- sam/clips/return_bridge.mp4 (5046842 bytes, sha256 021f1cfe8c96)

Next: the README's measured rows are drafted from this memo, each number with the script and the run that produced it.
