"""walkaround.narration offline: the measured Muse Voice turns (D1, Oct 31: ENDPOINTING on our own Piper narration of
the uCO3D pair C pickup) split, pinned and seeded against canned panels, keyframes and Spark answers; chips cut by
ffmpeg from a synthetic WAV; the rebuild replaying it all with no call. The fake Spark client sits under the real
spark_client.seed_mark, so the seed_v1 parsing and the replay from spark.json are both exercised. One live smoke runs
only with LIVE=1 (about $0.01: one Voice call on the narrated pickup and one seed_v1 call on a blurred trunk-lid crop)."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np
import pytest
from pydantic import Field

from walkaround import config, contracts as c, geometry, ledger, marks, match, narration, paths, spark_client, video
from walkaround.contracts import MarkKind, Noun, PanelId, PinKind, Position, Side, SizeClass, VerdictLabel

REPO = Path(__file__).resolve().parent.parent
TH = c.load_thresholds()
for _leaf in ("sigma_frame", "sigma_spark"):  # the band at its floor (tau_pre 0.25); D1's sigma_spark sits it at the cap
    TH["matching"].pop(_leaf, None)
KEYWORDS = list(TH["nouns"]["voice_keywords"])

# ENDPOINTING on data/derived/uco3d_7336_narrated.mp4, D1 Oct 31 (runs/d1/uco3d_pairC/voice_narrated): three sentences
# parted by pauses under about 3 s came back as one 11 s turn, and "light scratch" as "like scratch".
MEASURED = {
    "sessionId": "voyager/transcribe/CBC438A14E0C47908D115929B7A65708",
    "transcript": "pick up work starting at the front bumper hood looks clean front door like scratch near the handle trunk lid two small chips by the edge rear bumper looks fine looks fine",
    "audioDurationMs": 19760,
    "turns": [
        {"turnId": 0, "startMs": 0, "endMs": 11180, "transcript": "pick up work starting at the front bumper hood looks clean front door like scratch near the handle"},
        {"turnId": 1, "startMs": 13100, "endMs": 16460, "transcript": "trunk lid two small chips by the edge"},
        {"turnId": 2, "startMs": 16860, "endMs": 18460, "transcript": "rear bumper looks fine looks fine"},
    ],
}
# The D1 15 fps pickup lanes of the same walk, one window per object, filed into slots by hand (the side of the door
# lane is arbitrary here; nothing in the words names a side).
WINDOWS: dict[PanelId, list[list[float]]] = {
    PanelId.FRONT_BUMPER: [[0.0, 7.8]],
    PanelId.HOOD: [[0.0, 9.667]],
    PanelId.RIGHT_FRONT_DOOR: [[0.0, 2.467]],
    PanelId.LEFT_FRONT_DOOR: [[4.867, 13.0]],
    PanelId.LEFT_REAR_DOOR: [[5.733, 14.933]],
    PanelId.TRUNK_LID: [[13.067, 19.733]],
    PanelId.REAR_BUMPER: [[9.933, 18.533]],
}
CROP_W, CROP_H = 640, 400
CHIP_SPOT = (320, 120)  # crop pixels of the dark spot on the synthetic trunk-lid crop
DOOR_MARK = "pickup_left_front_door_kf1_m1"


# Fixtures and builders ------------------------------------------------------------------------------------------------


class FakeSpark:
    """responses.create with a seed_v1 answer per prompt; answer(prompt) returns a dict, raw text or an exception."""

    def __init__(self, answer: Callable[[str], Any]) -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **request: Any) -> dict[str, Any]:
        assert request["store"] is False and request["text"]["format"]["name"] == "seed_v1"
        prompt = request["input"][1]["content"][0]["text"]
        self.prompts.append(prompt)
        answer = self.answer(prompt)
        if isinstance(answer, Exception):
            raise answer
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return reply(f"resp_seed_{len(self.prompts)}", text)


def reply(response_id: str, text: str) -> dict[str, Any]:
    usage = {"input_tokens": 1400, "output_tokens": 300, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 250}}
    return {"id": response_id, "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}], "usage": usage}


def found(box: dict[str, int], kind: str) -> dict[str, Any]:
    return {"found": True, "bbox": box, "type": kind}


NOT_FOUND = {"found": False, "bbox": None, "type": None}
TRUNK_BOX = {"x_min": 490, "y_min": 285, "x_max": 510, "y_max": 315}  # around CHIP_SPOT on a 640 x 400 crop


def default_answer(prompt: str) -> Any:
    if "of the trunk lid" in prompt:
        return found(TRUNK_BOX, "chip")
    if "of the left front door" in prompt:
        return found({"x_min": 405, "y_min": 405, "x_max": 465, "y_max": 465}, "scratch")  # on the box pass's mark
    return NOT_FOUND


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    monkeypatch.setenv("MODEL_API_KEY", "LLM|test-team|test-secret")
    yield root
    spark_client.set_client(None)


def fake(answer: Callable[[str], Any] = default_answer) -> FakeSpark:
    client = FakeSpark(answer)
    spark_client.set_client(client)
    return client


def write_wav(path: Path, seconds: float = 20.0, rate: int = 24000) -> str:
    t = np.arange(int(seconds * rate)) / rate
    pcm = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return video.sha256_file(path)


def voice_record(role: str, response: dict[str, Any], *, mode: str = "ENDPOINTING", wav_sha: str | None = None) -> c.VoiceRecord:
    turns = response.get("turns") or []
    applied = not (mode != "PUSH_TO_TALK" and not turns and str(response.get("transcript", "")).strip())
    request = c.VoiceRequest(mode=mode, model="muse-voice-transcribe-1.0", keywords=KEYWORDS)
    return c.VoiceRecord(role=role, request=request, response=c.VoiceResponse.model_validate(response), wav_sha256=wav_sha, mode_applied=applied, dollars=0.00095)


def add_video(run_id: str, role: str) -> str:
    wav = paths.wav_path(run_id, role)
    sha = write_wav(wav)
    c.write_json(paths.video_json(run_id, role), c.VideoRecord(role=role, original_filename=f"{role}.mp4", bytes=1, wav=c.WavRecord(path=paths.rel(run_id, wav), duration_s=20.0, sha256=sha)))
    return sha


def front_sign(pid: PanelId) -> int:
    side = pid.value.split("_")[0]
    return {"left": -1, "right": 1}.get(side, -1 if pid in (PanelId.TRUNK_LID, PanelId.REAR_BUMPER, PanelId.REAR_WINDOW) else 1)


def add_keyframe(run_id: str, role: str, pid: PanelId, n: int, *, sharpness: float) -> c.Keyframe:
    img = np.full((CROP_H, CROP_W, 3), 200, np.uint8)
    cv2.circle(img, CHIP_SPOT, 6, (60, 60, 60), -1)
    kf_id = f"{role}_{pid.value}_kf{n}"
    crop_file = paths.crop_path(run_id, role, kf_id)
    sha = video.write_jpeg(img, crop_file, quality=90)
    mask = np.zeros((CROP_H, CROP_W), bool)
    mask[40 : CROP_H - 40, 40 : CROP_W - 40] = True
    mask_file = paths.crop_mask_png_path(run_id, role, kf_id)
    video.write_png_mask(mask, mask_file)
    return c.Keyframe(
        kf_id=kf_id,
        role=role,
        panel_id=pid,
        track_id=f"{role}_{c.PANEL_NOUN[pid].value.replace(' ', '_')}_01",
        f_proxy=100 + n,
        pts=7.0 + n,
        score=c.KeyframeScore(area=0.3, completeness=1.0, sharpness=sharpness, glare=0.0, frontality=1.0, total=sharpness),
        aspect_ratio=1.6,
        box_src=[0.0, 0.0, float(CROP_W), float(CROP_H)],
        crop=c.Crop(path=paths.rel(run_id, crop_file), sha256=sha, w=CROP_W, h=CROP_H, scale=1.0, dx=0.0, dy=0.0),
        mask_png=paths.rel(run_id, mask_file),
        moments=geometry.moments_from_mask(mask),
        front_sign=front_sign(pid),
        marks_status=c.MarksStatus.COMPLETE,
    )


def grid_mark(kf: c.Keyframe, gx: int, gy: int = 430) -> c.Mark:
    """The marks pass's own scratch at grid (gx, gy) on a crop, mapped and whitened by walkaround.marks."""
    box = c.GridBox(x_min=gx - 30, y_min=gy - 30, x_max=gx + 30, y_max=gy + 30)
    spark_mark = c.SparkMark(type=MarkKind.SCRATCH, point=c.GridPoint(x=gx, y=gy), bbox=box, size_class=SizeClass.S, confidence=c.Confidence.MEDIUM)
    result = SimpleNamespace(sent_width=CROP_W, sent_height=CROP_H, record=SimpleNamespace(request_id=f"resp_box_{kf.role}"))
    found_marks = marks.marks_from_answer(c.SparkMarksAnswer(marks=[spark_mark]), result, kf, kf.crop, th=TH)  # type: ignore[arg-type]
    return marks.whiten_marks(found_marks, kf, th=TH)[0][0]


def make_run(run_id: str, *, roles: tuple[str, ...] = ("pickup",), response: dict[str, Any] = MEASURED, mode: str = "ENDPOINTING", keyframes: bool = True) -> None:
    records, kfs, box_marks = [], [], []
    for role in roles:
        records.append(voice_record(role, response, mode=mode, wav_sha=add_video(run_id, role)))
        if keyframes:
            kfs += [add_keyframe(run_id, role, PanelId.TRUNK_LID, 1, sharpness=0.9), add_keyframe(run_id, role, PanelId.TRUNK_LID, 2, sharpness=0.5)]
            door = add_keyframe(run_id, role, PanelId.LEFT_FRONT_DOOR, 1, sharpness=0.8)
            kfs.append(door)
            box_marks.append(grid_mark(door, 430))
    c.write_json(paths.voice_json(run_id), records)
    panels = [
        c.Panel(panel_id=pid, noun=c.PANEL_NOUN[pid], per_role={role: c.PanelRole(windows=ws, seconds_visible=sum(b - a for a, b in ws), state=c.CoverageState.FILMED) for role in roles})
        for pid, ws in WINDOWS.items()
    ]
    c.write_json(paths.panels_json(run_id), panels)
    c.write_json(paths.keyframes_json(run_id), kfs)
    if box_marks:
        c.write_json(paths.marks_json(run_id), box_marks)


def make_door_pair(run_id: str, *, return_gx: int, seed_gx: int) -> None:
    """Pickup and return door keyframes with byte-identical crops and a box-pass scratch on each (the return one at
    return_gx); only the pickup walk narrates ("scratch on the front door") and Spark boxes it at seed_gx."""
    kfs, box_marks, records = [], [], []
    for role in ("pickup", "return"):
        sha = add_video(run_id, role)
        door = add_keyframe(run_id, role, PanelId.LEFT_FRONT_DOOR, 1, sharpness=0.8)
        kfs.append(door)
        box_marks.append(grid_mark(door, 430 if role == "pickup" else return_gx))
        if role == "pickup":
            said = {"transcript": "scratch on the front door", "audioDurationMs": 20000, "turns": [{"turnId": 0, "startMs": 6000, "endMs": 8000, "transcript": "scratch on the front door"}]}
            records.append(voice_record(role, said, wav_sha=sha))
    c.write_json(paths.voice_json(run_id), records)
    per_role = {role: c.PanelRole(windows=[[4.867, 13.0]], seconds_visible=8.133, state=c.CoverageState.FILMED) for role in ("pickup", "return")}
    door_panel = c.Panel(panel_id=PanelId.LEFT_FRONT_DOOR, noun=Noun.CAR_DOOR, side=Side.LEFT, position=Position.FRONT, front_sign=-1, per_role=per_role)
    c.write_json(paths.panels_json(run_id), [door_panel])
    c.write_json(paths.keyframes_json(run_id), kfs)
    c.write_json(paths.marks_json(run_id), box_marks)
    fake(lambda prompt: found({"x_min": seed_gx - 30, "y_min": 400, "x_max": seed_gx + 30, "y_max": 460}, "scratch"))


def derive(run_id: str, th: dict = TH) -> c.Narration:
    return asyncio.run(narration.derive_narration(run_id, th=th))


def turns(run_id: str, role: str = "pickup") -> dict[str, c.NarrationTurn]:
    doc = c.read_json(paths.narration_json(run_id), c.Narration)
    return {t.turn_id: t for t in doc.per_role[role].turns}


def piece_rows(run_id: str) -> dict[tuple[str, str], narration.PieceRow]:
    return {(r.role, r.turn_id): r for r in c.read_json_list(narration.pieces_json(run_id), narration.PieceRow)}


def seed_marks(run_id: str) -> list[c.Mark]:
    return [m for m in c.read_json_list(paths.marks_json(run_id), c.Mark) if m.source == c.MarkSource.NARRATION_SEED]


def verdicts(run_id: str) -> list[tuple[VerdictLabel, list[str]]]:
    _matches, found_verdicts, _calibration = match.derive_match(run_id, th=TH, second_look=None)
    return [(v.label, v.mark_ids) for v in found_verdicts]


def duration_s(path: Path) -> float:
    out = subprocess.run([video.FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def piece(text: str, start_ms: int, end_ms: int, **kw: Any) -> narration.Piece:
    pieces = narration.split_turn(text, start_ms, end_ms, turn=1, keywords=KEYWORDS, **kw)
    assert len(pieces) == 1, pieces
    return pieces[0]


def split(text: str) -> list[tuple[str, Noun | None, tuple[MarkKind | None, ...]]]:
    return [(p.text, p.noun, p.kinds) for p in narration.split_turn(text, 0, 6000, turn=1, keywords=KEYWORDS)]


# Words ----------------------------------------------------------------------------------------------------------------


def test_the_measured_11_s_turn_splits_at_each_panel_with_spans_by_character_offset():
    text = MEASURED["turns"][0]["transcript"]
    pieces = narration.split_turn(text, 0, 11180, turn=1, keywords=KEYWORDS)
    assert [p.text for p in pieces] == ["pick up work starting at the front bumper", "hood looks clean", "front door like scratch near the handle"]
    assert [p.turn_id for p in pieces] == ["t1_1", "t1_2", "t1_3"]
    assert all(p.estimated and p.turn_ms == (0, 11180) for p in pieces)
    # cut at chars 42 and 59 of 98: 11180 x 42 / 98 and 11180 x 59 / 98, the last piece ends on the turn's end
    assert [(p.start_ms, p.end_ms) for p in pieces] == [(0, 4791), (4791, 6731), (6731, 11180)]
    assert [(p.noun, p.position) for p in pieces] == [(Noun.BUMPER, Position.FRONT), (Noun.HOOD, None), (Noun.CAR_DOOR, Position.FRONT)]
    assert [p.kinds for p in pieces] == [(), (), (MarkKind.SCRATCH,)]
    assert pieces[2].keywords == ("car door", "scratch")
    whole = narration.split_turn(MEASURED["turns"][1]["transcript"], 13100, 16460, turn=2, keywords=KEYWORDS)
    assert len(whole) == 1 and whole[0].turn_id == "t2" and not whole[0].estimated
    assert (whole[0].start_ms, whole[0].end_ms, whole[0].kinds, whole[0].size) == (13100, 16460, (MarkKind.CHIP,), SizeClass.S)


def test_a_mark_phrase_before_on_the_panel_goes_with_that_panel():
    pieces = narration.split_turn("front bumper is fine a small dent on the hood", 0, 5000, turn=1, keywords=KEYWORDS)
    assert [p.text for p in pieces] == ["front bumper is fine", "a small dent on the hood"]
    assert [p.kinds for p in pieces] == [(), (MarkKind.DENT,)]
    assert pieces[1].size == SizeClass.S
    pieces = narration.split_turn("scratch on the door and a dent on the hood", 0, 5000, turn=1, keywords=KEYWORDS)
    assert [p.text for p in pieces] == ["scratch on the door", "and a dent on the hood"]
    assert [p.noun for p in pieces] == [Noun.CAR_DOOR, Noun.HOOD]
    pieces = narration.split_turn("roof is fine 2 small chips on the trunk lid", 0, 4000, turn=1, keywords=KEYWORDS)
    assert [p.text for p in pieces] == ["roof is fine", "2 small chips on the trunk lid"]


def test_a_mark_word_that_modifies_the_next_panel_goes_with_that_panel():
    assert split("hood looks clean dented front bumper") == [("hood looks clean", Noun.HOOD, ()), ("dented front bumper", Noun.BUMPER, (MarkKind.DENT,))]
    assert split("roof is clean chipped windshield") == [("roof is clean", Noun.ROOF, ()), ("chipped windshield", Noun.WINDSHIELD, (MarkKind.CHIP,))]
    assert split("hood is fine a badly scratched door on the left") == [("hood is fine", Noun.HOOD, ()), ("a badly scratched door on the left", Noun.CAR_DOOR, (MarkKind.SCRATCH,))]
    assert piece("scratched door", 0, 1000).kinds == (MarkKind.SCRATCH,)
    # after a copula, or right after the previous panel, the word says how that panel is
    assert split("the hood is dented front bumper is fine") == [("the hood is dented", Noun.HOOD, (MarkKind.DENT,)), ("front bumper is fine", Noun.BUMPER, ())]
    assert split("the hood is a bit dented front bumper fine") == [("the hood is a bit dented", Noun.HOOD, (MarkKind.DENT,)), ("front bumper fine", Noun.BUMPER, ())]
    assert split("front door scratched rear door fine") == [("front door scratched", Noun.CAR_DOOR, (MarkKind.SCRATCH,)), ("rear door fine", Noun.CAR_DOOR, ())]
    # a noun form there stays where it is: "the hood has a scratch" then "front bumper is fine"
    assert split("the hood has a scratch front bumper is fine") == [("the hood has a scratch", Noun.HOOD, (MarkKind.SCRATCH,)), ("front bumper is fine", Noun.BUMPER, ())]
    # joined by and: the pair goes where its first word goes
    assert split("the hood is scratched and dented front bumper fine") == [("the hood is scratched and dented", Noun.HOOD, (MarkKind.SCRATCH, MarkKind.DENT)), ("front bumper fine", Noun.BUMPER, ())]
    assert split("hood fine a scratched and dented front bumper") == [("hood fine", Noun.HOOD, ()), ("a scratched and dented front bumper", Noun.BUMPER, (MarkKind.SCRATCH, MarkKind.DENT))]
    assert split("the door has a scratch and a dent on the hood") == [("the door has a scratch", Noun.CAR_DOOR, (MarkKind.SCRATCH,)), ("and a dent on the hood", Noun.HOOD, (MarkKind.DENT,))]


def test_repeated_mentions_of_one_panel_stay_one_piece_and_different_ones_split():
    assert len(narration.split_turn("door is fine the door handle has a ding", 0, 4000, turn=1, keywords=KEYWORDS)) == 1
    pieces = narration.split_turn("front bumper rear bumper", 0, 2000, turn=1, keywords=KEYWORDS)
    assert [p.position for p in pieces] == [Position.FRONT, Position.REAR]


def test_side_and_position_words_before_or_after_the_noun():
    p = piece("the door on the passenger side has a long scrat", 0, 3000)
    assert (p.noun, p.side, p.position, p.kinds, p.size) == (Noun.CAR_DOOR, Side.RIGHT, None, (MarkKind.SCRATCH,), SizeClass.L)
    p = piece("left rear wheel has curb rash", 0, 3000)
    assert (p.noun, p.side, p.position, p.kinds) == (Noun.WHEEL, Side.LEFT, Position.REAR, (MarkKind.SCUFF,))
    p = piece("bumper at the back is scuffed", 0, 3000)
    assert (p.noun, p.position, p.kinds) == (Noun.BUMPER, Position.REAR, (MarkKind.SCUFF,))
    p = piece("came back and the hood is clean", 0, 3000)  # "back" two words off the noun is not a position
    assert (p.noun, p.position) == (Noun.HOOD, None)
    # passenger and driver follow thresholds claim.passenger_side_is
    assert TH["claim"]["passenger_side_is"] == "right"
    assert piece("passenger door scratch", 0, 1000, passenger=Side.LEFT).side == Side.LEFT
    assert piece("driver's door scratch", 0, 1000, passenger=Side.LEFT).side == Side.RIGHT


def test_mark_words_match_loosely_and_negated_ones_do_not_count():
    assert piece("front door like scrat near the handle", 0, 1000).kinds == (MarkKind.SCRATCH,)
    assert piece("hood has a ding", 0, 1000).kinds == (MarkKind.DENT,)
    assert piece("windshield is chipped", 0, 1000).kinds == (MarkKind.CHIP,)
    assert piece("back window cracked", 0, 1000).kinds == (MarkKind.CRACK,)
    assert piece("paint transfer on the rear bumper", 0, 1000).kinds == (MarkKind.SCUFF,)
    assert piece("some damage on the roof", 0, 1000).kinds == (None,)
    assert piece("hood looks clean no scratches", 0, 1000).kinds == ()
    assert narration._phrase('front door like scrat, "near" the handle') == "front door like scratch, 'near' the handle"
    assert piece("I don't see any dents on the front door", 0, 1000).kinds == ()
    assert piece("the paint looks good on the hood", 0, 1000).kinds == ()  # paint alone is not a mark
    # a negator spent before a clause break, or in "never mind", negates nothing
    assert piece("never mind small dent on the hood", 0, 1000).kinds == (MarkKind.DENT,)
    assert piece("no wait there's a dent on the hood", 0, 1000).kinds == (MarkKind.DENT,)


def test_words_inside_a_panel_name_are_not_size_words_and_a_rear_view_mirror_is_no_panel():
    p = piece("tail light crack", 0, 1000)
    assert (p.noun, p.kinds, p.size) == (Noun.TAIL_LIGHT, (MarkKind.CRACK,), None)
    assert piece("light scratch on the hood", 0, 1000).size == SizeClass.S
    p = piece("rear view mirror is cracked", 0, 1000)
    assert (p.noun, p.kinds, p.keywords) == (None, (MarkKind.CRACK,), ("crack",))
    assert piece("side mirror is scratched", 0, 1000).noun == Noun.SIDE_MIRROR


def test_keywords_heard_are_spelled_as_the_voice_list_spells_them():
    assert narration.keywords_heard("pick up work starting at the front bumper", KEYWORDS) == ["bumper"]
    assert narration.keywords_heard("the car door and the car", KEYWORDS) == ["car door", "car"]
    assert narration.keywords_heard("back window, mileage, scrat and curb rash", KEYWORDS) == ["rear window", "odometer", "scratch", "scuff"]
    assert narration.keywords_heard("some damage", KEYWORDS) == []


# Pins -----------------------------------------------------------------------------------------------------------------


def test_candidates_follow_side_and_position_words():
    assert narration.candidates(Noun.CAR_DOOR, None, Position.FRONT) == [PanelId.LEFT_FRONT_DOOR, PanelId.RIGHT_FRONT_DOOR]
    assert narration.candidates(Noun.CAR_DOOR, Side.RIGHT, None) == [PanelId.RIGHT_FRONT_DOOR, PanelId.RIGHT_REAR_DOOR]
    assert narration.candidates(Noun.BUMPER, Side.LEFT, Position.REAR) == [PanelId.REAR_BUMPER]
    assert narration.candidates(Noun.HOOD, Side.LEFT, Position.REAR) == [PanelId.HOOD]


def test_a_piece_pins_where_sam_shows_the_named_panel_else_the_slot_of_that_kind_on_screen():
    assert TH["voice"]["pin_fallback"] == "turn_start_panel"
    windows = {PanelId.LEFT_FRONT_DOOR: [(5.0, 10.0)], PanelId.RIGHT_FRONT_DOOR: [(25.0, 30.0)]}
    pin = narration.pin_piece(piece("front door scratch", 6000, 8000), windows)
    assert pin == narration.Pin(PanelId.LEFT_FRONT_DOOR, PinKind.OVERLAP, 2.0)
    assert narration.pin_piece(piece("front door scratch", 26000, 29000), windows).panel == PanelId.RIGHT_FRONT_DOOR
    # the words name the right door while SAM shows the left one: turn_start_panel pins "about" the door on screen
    said = piece("passenger front door scratch", 6000, 8000)
    assert narration.pin_piece(said, windows) == narration.Pin(PanelId.LEFT_FRONT_DOOR, PinKind.ABOUT, None)
    # named_panel_nearest keeps the words: "about" the right door, which SAM shows later
    assert narration.pin_piece(said, windows, fallback="named_panel_nearest") == narration.Pin(PanelId.RIGHT_FRONT_DOOR, PinKind.ABOUT, None)
    with pytest.raises(ValueError, match="pin_fallback"):
        narration.pin_piece(said, windows, fallback="turn_start")
    # the slot on screen at the piece's start wins over one shown longer from later on
    windows = {PanelId.LEFT_FRONT_DOOR: [(6.5, 20.0)], PanelId.LEFT_REAR_DOOR: [(4.0, 7.0)]}
    assert narration.pin_piece(piece("right door scratch", 6000, 10000), windows) == narration.Pin(PanelId.LEFT_REAR_DOOR, PinKind.ABOUT, None)


def test_about_pins_and_pieces_that_pin_nowhere():
    windows = {PanelId.LEFT_FRONT_DOOR: [(4.867, 13.0)]}
    assert narration.pin_piece(piece("front door", 15000, 16000), windows) == narration.Pin(PanelId.LEFT_FRONT_DOOR, PinKind.ABOUT, None)
    assert narration.pin_piece(piece("small dent on the roof", 2000, 4000), windows) == narration.Pin(PanelId.ROOF, PinKind.ABOUT, None)
    assert narration.pin_piece(piece("small dent on the roof", 2000, 4000), {}) == narration.Pin(PanelId.ROOF, PinKind.ABOUT, None)
    assert narration.pin_piece(piece("rear door scratch", 2000, 4000), windows) == narration.Pin()  # two rear doors, neither filmed
    assert narration.pin_piece(piece("small scratch here", 6000, 7000), windows) == narration.Pin()  # no noun, no pin


# The whole walk -----------------------------------------------------------------------------------------------------


def test_derive_pins_cuts_chips_and_seeds_the_measured_walk(runs: Path):
    make_run("m1")
    client = fake()
    result = derive("m1")
    assert result.mode_applied is True
    t = turns("m1")
    assert list(t) == ["t1_1", "t1_2", "t1_3", "t2", "t3"]
    # every piece keeps the span Voice measured for its turn; no word time is invented
    assert {k: (v.start_ms, v.end_ms) for k, v in t.items()} == {"t1_1": (0, 11180), "t1_2": (0, 11180), "t1_3": (0, 11180), "t2": (13100, 16460), "t3": (16860, 18460)}
    # pins chosen on the estimated piece spans; pin_seconds is what SAM shows of the pinned panel during the Voice turn
    pins = {k: (v.pinned_panel, v.pin_kind, round(v.pin_seconds, 3) if v.pin_seconds is not None else None) for k, v in t.items()}
    assert pins == {
        "t1_1": (PanelId.FRONT_BUMPER, PinKind.OVERLAP, 7.8),
        "t1_2": (PanelId.HOOD, PinKind.OVERLAP, 9.667),
        "t1_3": (PanelId.LEFT_FRONT_DOOR, PinKind.OVERLAP, 6.313),
        "t2": (PanelId.TRUNK_LID, PinKind.OVERLAP, 3.36),
        "t3": (PanelId.REAR_BUMPER, PinKind.OVERLAP, 1.6),
    }
    assert t["t2"].keywords_hit == ["trunk lid", "chip"]
    # two seed questions (the door and the trunk lid); the bumper and hood pieces name no mark
    assert len(client.prompts) == 2
    assert any('"trunk lid two small chips by the edge"' in p and "of the trunk lid" in p for p in client.prompts)
    rows = piece_rows("m1")
    assert {k: rows[("pickup", k)].est_ms for k in t} == {"t1_1": [0, 4791], "t1_2": [4791, 6731], "t1_3": [6731, 11180], "t2": None, "t3": None}
    door, trunk = rows[("pickup", "t1_3")].seed, rows[("pickup", "t2")].seed
    assert door is not None and door.status == "on_existing_mark" and door.mark_id == DOOR_MARK and "inside the match band" in door.reason
    assert trunk is not None and trunk.status == "found" and trunk.mark_id == "pickup_trunk_lid_kf1_seed_t2" and trunk.kf_id == "pickup_trunk_lid_kf1"
    assert (trunk.type_from, trunk.size_from) == ("spark", "words")
    assert t["t1_3"].seed_mark_id == door.mark_id and t["t2"].seed_mark_id == trunk.mark_id and t["t1_2"].seed_mark_id is None
    assert rows[("pickup", "t1_2")].seed is None and rows[("pickup", "t1_2")].estimated and not rows[("pickup", "t2")].estimated
    # the seed mark: mapped and whitened like a box-pass mark, the renter's words as its label, capped
    (seed,) = seed_marks("m1")
    assert (seed.mark_id, seed.role, seed.panel_id, seed.kf_id) == ("pickup_trunk_lid_kf1_seed_t2", "pickup", PanelId.TRUNK_LID, "pickup_trunk_lid_kf1")
    assert (seed.type, seed.size_class) == (MarkKind.CHIP, SizeClass.S)
    assert seed.confidence == c.Confidence.LOW and narration.seed_cap() in seed.caps and seed.description == "trunk lid two small chips by the edge"
    assert seed.w is not None and seed.r is not None and seed.spark_request_id == trunk.request_id
    assert abs(seed.point_crop[0] - CHIP_SPOT[0]) <= 3 and abs(seed.point_crop[1] - CHIP_SPOT[1]) <= 3
    # the box-pass mark is untouched and comes first
    stored = c.read_json_list(paths.marks_json("m1"), c.Mark)
    assert [m.mark_id for m in stored] == [DOOR_MARK, "pickup_trunk_lid_kf1_seed_t2"]
    # every attempt in spark.json, every question in seeds.json, a ledger row per call
    records = c.read_json_list(paths.spark_json("m1"), c.SparkRequestRecord)
    assert [r.purpose for r in records] == [c.SparkPurpose.SEED, c.SparkPurpose.SEED] and all(r.valid for r in records)
    asks = c.read_json_list(narration.seeds_json("m1"), narration.SeedAsk)
    assert sorted(a.turn_id for a in asks) == ["t1_3", "t2"] and all(a.crop.path.startswith("crops/pickup/") for a in asks)
    assert [r["note"] for r in ledger.rows()] == ["seed", "seed"]
    assert [(role, r.request_id) for role, r in narration.seed_records("m1")] == [("pickup", r.request_id) for r in records]


def test_every_turn_gets_a_chip_holding_its_whole_voice_turn(runs: Path):
    make_run("m2")
    fake()
    derive("m2")
    t, rows = turns("m2"), piece_rows("m2")
    assert {k: v.chip_path for k, v in t.items()} == {k: f"chips/pickup/{k}.m4a" for k in t}
    spans = {k: rows[("pickup", k)].chip_ms for k in t}
    assert spans == {"t1_1": [0, 11180], "t1_2": [0, 11180], "t1_3": [0, 11180], "t2": [13100, 16460], "t3": [16860, 18460]}
    for k, (a, b) in spans.items():
        assert [t[k].start_ms, t[k].end_ms] == [a, b]  # the web shows end - start as the chip's length
        assert abs(duration_s(paths.run_dir("m2") / t[k].chip_path) - (b - a) / 1000) < 0.1


def test_rebuild_replays_narration_and_seed_marks_with_no_call(runs: Path):
    make_run("m3")
    fake()
    derive("m3")
    files = [paths.narration_json("m3"), narration.pieces_json("m3"), paths.marks_json("m3"), paths.chip_path("m3", "pickup", "t2")]
    before = [f.read_bytes() for f in files]
    blown = fake(lambda prompt: AssertionError("a rebuild never calls Spark"))
    narration.rebuild_narration("m3", th=TH)
    assert [f.read_bytes() for f in files] == before and blown.prompts == []
    # a walk's rows rewritten without its seeds: merge_seed_marks puts the same bytes back
    c.write_json(paths.marks_json("m3"), [m for m in c.read_json_list(paths.marks_json("m3"), c.Mark) if m.source != c.MarkSource.NARRATION_SEED])
    assert [m.mark_id for m in narration.merge_seed_marks("m3", th=TH)] == ["pickup_trunk_lid_kf1_seed_t2"]
    assert [f.read_bytes() for f in files] == before
    # a second derive finds every answer stored and asks nothing
    derive("m3")
    assert blown.prompts == [] and [f.read_bytes() for f in files] == before
    # a chip no turn names is removed by the next cut
    stray = paths.chip_path("m3", "pickup", "t9")
    stray.write_bytes(b"old")
    narration.rebuild_narration("m3", th=TH)
    assert not stray.exists() and [f.read_bytes() for f in files] == before


class MarksAndSeedSpark:
    """responses.create for both passes: marks_v1 finds one scratch at grid (430, 430), seed_v1 boxes the chip spot."""

    def __init__(self) -> None:
        self.responses = SimpleNamespace(create=self.create)
        self.calls: list[str] = []

    def create(self, **request: Any) -> dict[str, Any]:
        name = request["text"]["format"]["name"]
        self.calls.append(name)
        if name == "seed_v1":
            body: dict[str, Any] = found(TRUNK_BOX, "chip")
        else:
            box = {"x_min": 400, "y_min": 400, "x_max": 460, "y_max": 460}
            scratch = {"type": "scratch", "point": {"x": 430, "y": 430}, "bbox": box, "size_class": "S", "confidence": "medium", "note": ""}
            body = {"marks": [scratch], "conditions": {"wet": False, "dirty": False, "glare": False, "dark": False}, "landmarks": [], "neighbour_car_visible": False}
        return reply(f"resp_{name}_{len(self.calls)}", json.dumps(body))


def test_marks_rebuild_then_narration_rebuild_gives_the_run_bytes_back(runs: Path):
    """The rebuild order the orchestrator needs: marks.rebuild_marks drops a walk's seeds, rebuild_narration right after
    it (before the match) writes the same marks.json, narration.json and narration_pieces.json as the run."""
    sha = add_video("rb", "pickup")
    said = {"transcript": "small chip on the trunk lid", "audioDurationMs": 20000, "turns": [{"turnId": 0, "startMs": 14000, "endMs": 16000, "transcript": "small chip on the trunk lid"}]}
    c.write_json(paths.voice_json("rb"), [voice_record("pickup", said, wav_sha=sha)])
    kf = add_keyframe("rb", "pickup", PanelId.TRUNK_LID, 1, sharpness=0.9)
    c.write_json(paths.keyframes_json("rb"), [kf])
    c.write_json(paths.panels_json("rb"), [c.Panel(panel_id=PanelId.TRUNK_LID, noun=Noun.TRUNK_LID, per_role={"pickup": c.PanelRole(windows=[[13.0, 19.7]], state=c.CoverageState.FILMED)})])
    client = MarksAndSeedSpark()
    spark_client.set_client(client)
    asyncio.run(marks.derive_marks("rb", "pickup", [kf], th=TH))
    derive("rb")
    assert client.calls == ["marks_v1", "seed_v1"]
    files = [paths.marks_json("rb"), paths.narration_json("rb"), narration.pieces_json("rb"), paths.calibration_json("rb")]
    before = [f.read_bytes() for f in files]
    assert [m.source for m in c.read_json_list(paths.marks_json("rb"), c.Mark)] == [c.MarkSource.SPARK_BOX, c.MarkSource.NARRATION_SEED]
    marks.rebuild_marks("rb", "pickup", th=TH)
    assert seed_marks("rb") == []
    narration.rebuild_narration("rb", th=TH)
    assert [f.read_bytes() for f in files] == before and client.calls == ["marks_v1", "seed_v1"]


def test_merge_seed_marks_does_nothing_before_the_narration_stage(runs: Path):
    make_run("m3b")
    before = paths.marks_json("m3b").read_bytes()
    assert narration.merge_seed_marks("m3b", th=TH) == []
    assert not paths.narration_json("m3b").exists() and paths.marks_json("m3b").read_bytes() == before


def test_a_seed_that_is_not_found_is_recorded_never_invented(runs: Path):
    make_run("m4")
    fake(lambda prompt: NOT_FOUND)
    derive("m4")
    seed = piece_rows("m4")[("pickup", "t2")].seed
    assert seed is not None and seed.status == "not_found" and seed.mark_id is None and seed.request_id
    assert seed.reason == "Spark found no mark matching these words on the trunk lid"
    assert turns("m4")["t2"].seed_mark_id is None and seed_marks("m4") == []
    assert [r.request_id for r in c.read_json_list(paths.spark_json("m4"), c.SparkRequestRecord)] != []


def test_a_box_off_the_panel_mask_is_not_a_mark(runs: Path):
    make_run("m5")
    fake(lambda prompt: found({"x_min": 10, "y_min": 10, "x_max": 40, "y_max": 40}, "chip") if "trunk lid" in prompt else NOT_FOUND)
    derive("m5")
    seed = piece_rows("m5")[("pickup", "t2")].seed
    assert seed is not None and seed.status == "off_panel" and seed.mark_id is None and seed.bbox_crop is not None
    assert seed_marks("m5") == []


def test_an_unusable_answer_is_failed_and_asked_again_next_time(runs: Path):
    make_run("m6")
    fake(lambda prompt: "not json" if "trunk lid" in prompt else default_answer(prompt))
    derive("m6")
    seed = piece_rows("m6")[("pickup", "t2")].seed
    assert seed is not None and seed.status == "failed" and seed.reason.startswith("no usable answer")
    trunk_ask = next(a for a in c.read_json_list(narration.seeds_json("m6"), narration.SeedAsk) if a.turn_id == "t2")
    assert len(trunk_ask.request_ids) == 2  # the schema miss and its one retry, both kept
    client = fake()
    derive("m6")
    assert len(client.prompts) == 1 and "trunk lid" in client.prompts[0]  # the valid door answer is not asked again
    assert piece_rows("m6")[("pickup", "t2")].seed.status == "found"


def test_a_call_that_raises_costs_only_that_seed_and_its_error_is_redacted(runs: Path):
    make_run("m7")
    fake(lambda prompt: RuntimeError("upstream said LLM|1234567|abcdefghijkl") if "trunk lid" in prompt else default_answer(prompt))
    result = derive("m7")
    assert result.mode_applied
    seed = piece_rows("m7")[("pickup", "t2")].seed
    assert seed is not None and seed.status == "failed" and "abcdefghijkl" not in seed.reason and "[redacted]" in seed.reason
    assert piece_rows("m7")[("pickup", "t1_3")].seed.status == "on_existing_mark"


def test_a_turn_naming_a_panel_sam_never_showed_pins_about_it_and_asks_no_seed(runs: Path):
    response = {"transcript": "small dent on the roof", "audioDurationMs": 6000, "turns": [{"turnId": 0, "startMs": 2000, "endMs": 4000, "transcript": "small dent on the roof"}]}
    make_run("m8", response=response)
    client = fake()
    derive("m8")
    turn = turns("m8")["t1"]
    assert (turn.pinned_panel, turn.pin_kind, turn.pin_seconds, turn.seed_mark_id) == (PanelId.ROOF, PinKind.ABOUT, None, None)
    assert turn.chip_path == "chips/pickup/t1.m4a"
    seed = piece_rows("m8")[("pickup", "t1")].seed
    assert seed is not None and seed.status == "not_asked" and seed.reason == "SAM did not find the roof in this walk"
    assert client.prompts == []
    # a lane that never ran says so instead
    panels = c.read_json_list(paths.panels_json("m8"), c.Panel)
    roof = c.Panel(panel_id=PanelId.ROOF, noun=Noun.ROOF, per_role={"pickup": c.PanelRole(state=c.CoverageState.NOT_RUN)})
    c.write_json(paths.panels_json("m8"), [*panels, roof])
    narration.rebuild_narration("m8", th=TH)
    assert piece_rows("m8")[("pickup", "t1")].seed.reason == "SAM did not run for the roof in this walk"


def test_a_turn_with_no_panel_noun_pins_nowhere_and_still_gets_its_chip(runs: Path):
    response = {"transcript": "okay small scratch here", "audioDurationMs": 6000, "turns": [{"turnId": 0, "startMs": 1000, "endMs": 3000, "transcript": "okay small scratch here"}]}
    make_run("m9", response=response)
    client = fake()
    derive("m9")
    turn = turns("m9")["t1"]
    assert (turn.pinned_panel, turn.pin_kind, turn.chip_path, turn.keywords_hit) == (None, None, "chips/pickup/t1.m4a", ["scratch"])
    assert piece_rows("m9")[("pickup", "t1")].seed is None and client.prompts == []


def test_turns_empty_under_endpointing_is_mode_not_applied_with_no_chips(runs: Path):
    response = {"transcript": "front door small scratch", "audioDurationMs": 5000, "turns": []}
    make_run("m10", response=response)
    client = fake()
    result = derive("m10")
    assert result.mode_applied is False and turns("m10") == {}
    assert not any(paths.chips_dir("m10", "pickup").iterdir()) and client.prompts == []
    # PUSH_TO_TALK never has turns: applied, nothing to pin
    make_run("m11", response=response, mode="PUSH_TO_TALK")
    assert derive("m11").mode_applied is True and turns("m11") == {}


def test_a_transcript_with_no_turns_rides_along_once_the_contract_has_the_field(runs: Path, monkeypatch: pytest.MonkeyPatch):
    """Section 14 item 16b: the transcript shown without pins, never "no narration found". contracts.NarrationRole
    has no transcript field yet; the lane fills it as soon as it does (stood in for here by a subclass)."""

    class RoleWithTranscript(c.NarrationRole):
        transcript: str | None = None

    class NarrationWithTranscript(c.Narration):
        per_role: dict[c.Role, RoleWithTranscript] = Field(default_factory=dict)

    monkeypatch.setattr(narration, "NarrationRole", RoleWithTranscript)
    monkeypatch.setattr(narration, "Narration", NarrationWithTranscript)
    response = {"transcript": "front door small scratch", "audioDurationMs": 5000, "turns": []}
    make_run("m10t", response=response)
    fake()
    result = derive("m10t")
    assert result.mode_applied is False and result.per_role["pickup"].turns == []
    assert json.loads(paths.narration_json("m10t").read_text())["per_role"]["pickup"]["transcript"] == "front door small scratch"
    make_run("m10u")  # a walk with turns carries no second copy of its words
    derive("m10u")
    assert json.loads(paths.narration_json("m10u").read_text())["per_role"]["pickup"].get("transcript") is None


def test_a_walk_with_no_voice_record_is_left_out(runs: Path):
    make_run("m12")
    paths.voice_json("m12").write_text("[]\n")
    fake()
    result = derive("m12")
    assert result.per_role == {} and result.mode_applied is True


def test_the_same_question_on_the_same_crop_bytes_is_asked_once(runs: Path):
    make_run("m13", roles=("pickup", "return"))  # identical words and byte-identical crops in both walks
    client = fake()
    derive("m13")
    assert len(client.prompts) == 2
    assert [m.mark_id for m in seed_marks("m13")] == ["pickup_trunk_lid_kf1_seed_t2", "return_trunk_lid_kf1_seed_t2"]
    assert turns("m13", "return")["t1_3"].seed_mark_id == "return_left_front_door_kf1_m1"
    a, b = (next(m for m in seed_marks("m13") if m.role == role) for role in ("pickup", "return"))
    assert a.w == b.w and a.spark_request_id == b.spark_request_id


def test_a_walk_asks_at_most_the_seed_cap(runs: Path):
    th = json.loads(json.dumps(TH))
    th["voice"]["seeds_per_walk_max"] = 1  # the optional leaf; narration.SEEDS_PER_WALK_MAX (16) without it
    make_run("m14")
    client = fake()
    derive("m14", th)
    assert len(client.prompts) == 1
    seed = piece_rows("m14")[("pickup", "t2")].seed
    assert seed is not None and seed.status == "not_asked" and seed.reason == "over the 1 seeds one walk may ask"


def test_the_sharpest_crop_of_the_panel_is_the_one_asked(runs: Path):
    make_run("m15")
    client = fake()
    derive("m15")
    asks = {a.turn_id: a for a in c.read_json_list(narration.seeds_json("m15"), narration.SeedAsk)}
    assert asks["t2"].kf_id == "pickup_trunk_lid_kf1" and asks["t2"].crop.path == "crops/pickup/pickup_trunk_lid_kf1.jpg"
    assert len(client.prompts) == 2


# Seeds against the match ----------------------------------------------------------------------------------------------


def test_a_seed_on_a_mark_the_box_pass_found_links_to_it_and_the_pair_stays_already_there(runs: Path):
    """The review's case: box pair A-R about 0.10 units apart, the seed box about 0.23 from A and 0.12 from R. As a
    second mark the seed failed R's ratio test and turned already_there into not_seen_at_return plus new_or_unseen;
    inside the match band of A it is A."""
    make_door_pair("b1", return_gx=456, seed_gx=487)
    pair = (VerdictLabel.ALREADY_THERE, [DOOR_MARK, "return_left_front_door_kf1_m1"])
    assert verdicts("b1") == [pair]
    derive("b1")
    seed = piece_rows("b1")[("pickup", "t1")].seed
    assert seed is not None and seed.status == "on_existing_mark" and seed.mark_id == DOOR_MARK
    assert seed.reason.startswith("the box is 0.23 units from pickup_left_front_door_kf1_m1, inside the match band")
    assert turns("b1")["t1"].seed_mark_id == DOOR_MARK and seed_marks("b1") == []
    assert verdicts("b1") == [pair]


def test_the_link_band_is_the_one_calibration_json_gives_the_match(runs: Path):
    make_door_pair("b2", return_gx=456, seed_gx=487)
    c.write_json(paths.calibration_json("b2"), c.Calibration(tau_pre=0.05, tau_eye=0.1, tau_fix=0.08))
    derive("b2")
    seed = piece_rows("b2")[("pickup", "t1")].seed
    assert seed is not None and seed.status == "found" and (seed.type_from, seed.size_from) == ("spark", "default")
    (mark,) = seed_marks("b2")
    assert mark.size_class == SizeClass.M and mark.confidence == c.Confidence.LOW and mark.caps[-1] == narration.seed_cap()


def test_the_seed_cap_is_k14_once_match_reads_it_from_the_marks(monkeypatch: pytest.MonkeyPatch):
    without = tuple(k for k in match.CAP_IDS_FROM_MARK if k != "K14")
    monkeypatch.setattr(match, "CAP_IDS_FROM_MARK", without)
    assert narration.seed_cap() == "K12"
    monkeypatch.setattr(match, "CAP_IDS_FROM_MARK", (*without, "K14"))
    assert narration.seed_cap() == "K14"


def test_the_orchestrator_signature():
    assert inspect.iscoroutinefunction(narration.derive_narration)
    params = inspect.signature(narration.derive_narration).parameters
    assert list(params) == ["run_id", "th"] and params["th"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(narration.rebuild_narration).parameters) == ["run_id", "th"]


# Live (LIVE=1 only) ---------------------------------------------------------------------------------------------------

NARRATED = REPO / "data" / "derived" / "uco3d_7336_narrated.mp4"
D1_LANES = REPO / "runs" / "d1_20261031T195329Z_E2_15fps" / "sam" / "pickup"


def _lane_frame(name: str, frame: int) -> list[dict[str, Any]]:
    index = json.loads((D1_LANES / f"d1_e2_{name}.index.json").read_text())
    return next(f["objects"] for f in index if f["frame"] == frame)


def _live_trunk_keyframe(run_id: str, th: dict) -> c.Keyframe:
    """The trunk lid at proxy frame 225 (15.0 s) of the D1 15 fps lanes: the plate lane's boxes pixelated on the full
    frame first (the rear plate, and a dealer sticker on the bumper), then the crop and the lane's own mask."""
    from walkaround import privacy, sam_wire

    frame, lane_size = 225, (960, 540)
    rgb = video.read_frame_at(NARRATED, frame / 15)
    h, w = rgb.shape[:2]
    sx, sy = w / lane_size[0], h / lane_size[1]

    def to_src(b: list[float]) -> list[float]:
        return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]

    plates = [to_src(o["box"]) for o in _lane_frame("license_plate", frame)]
    blurred = privacy.blur_frame(rgb, track_boxes_src=plates, detections=[], keep_plate_region=False, th=th)
    trunk = _lane_frame("trunk_lid", frame)[0]
    crop = geometry.crop_geometry(to_src(trunk["box"]), (w, h), pad_frac=float(th["panels"]["crop_pad_frac"]), long_edge=int(th["panels"]["crop_long_edge_px"]))
    kf_id = "pickup_trunk_lid_kf1"
    crop_file = paths.crop_path(run_id, "pickup", kf_id)
    crop_rgb = video.cut_crop(blurred, crop)
    sha = video.write_jpeg(crop_rgb, crop_file, quality=int(th["spark"]["jpeg_quality"]))
    ref = c.MaskRef.model_validate(trunk["mask"])
    text = sam_wire.lane_text(sam_wire.read_events(D1_LANES / "d1_e2_trunk_lid.ndjson.gz"))
    full = sam_wire.paste_mask(sam_wire.decode_mask(ref, sam_wire.mask_payload(text, ref)), trunk["box"], ref, lane_size)
    full_src = cv2.resize(full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    region = full_src[int(crop.dy) : int(crop.dy) + round(crop.h / crop.scale), int(crop.dx) : int(crop.dx) + round(crop.w / crop.scale)]
    mask = cv2.resize(region, (crop.w, crop.h), interpolation=cv2.INTER_NEAREST) > 0
    mask_file = paths.crop_mask_png_path(run_id, "pickup", kf_id)
    video.write_png_mask(mask, mask_file)
    sharp = video.sharpness(crop_rgb, mask=mask)
    return c.Keyframe(
        kf_id=kf_id, role="pickup", panel_id=PanelId.TRUNK_LID, track_id="pickup_trunk_lid_01", f_proxy=frame, pts=frame / 15,
        score=c.KeyframeScore(area=float(mask.mean()), completeness=1.0, sharpness=sharp, glare=0.0, frontality=1.0, total=sharp),
        aspect_ratio=crop.w / crop.h, box_src=to_src(trunk["box"]),
        crop=crop.model_copy(update={"path": paths.rel(run_id, crop_file), "sha256": sha}), mask_png=paths.rel(run_id, mask_file),
        moments=geometry.moments_from_mask(mask), front_sign=-1, marks_status=c.MarksStatus.COMPLETE,
    )  # fmt: skip


@pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="LIVE=1 only: one Voice call and one seed_v1 call, about $0.01")
def test_live_narrated_pickup_pins_and_one_trunk_lid_seed(monkeypatch: pytest.MonkeyPatch):
    from walkaround import voice_client

    if not NARRATED.is_file() or not D1_LANES.is_dir():
        pytest.skip("the narrated pair C pickup or the D1 15 fps lanes are not on this machine")
    root = REPO / "runs" / "_m10_smoke"  # kept for a look (an underscore folder: no run listing, no sweep)
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths, "RUNS_DIR", root)  # the ledger rows still go to runs/ledger.jsonl
    monkeypatch.setattr(config, "RUNS_DIR", root)
    spark_client.set_client(None)
    run_id = "m10_smoke_" + time.strftime("%Y%m%dT%H%M%S")
    th = c.load_thresholds()
    wav = video.extract_wav(NARRATED, paths.wav_path(run_id, "pickup"), rate=int(th["media"]["wav_rate"]))
    wav_sha = video.sha256_file(wav)
    with wave.open(str(wav), "rb") as fh:
        seconds = fh.getnframes() / fh.getframerate()
    wav_record = c.WavRecord(path=paths.rel(run_id, wav), duration_s=seconds, sha256=wav_sha)
    c.write_json(paths.video_json(run_id, "pickup"), c.VideoRecord(role="pickup", original_filename=NARRATED.name, bytes=NARRATED.stat().st_size, wav=wav_record))
    transcript = voice_client.transcribe_wav(wav, mode=th["voice"]["mode"], keywords=list(th["nouns"]["voice_keywords"]), run_id=run_id)
    c.write_json(paths.voice_json(run_id), [voice_client.voice_record(transcript, role="pickup", wav_sha256=wav_sha)])
    panels = [c.Panel(panel_id=pid, noun=c.PANEL_NOUN[pid], per_role={"pickup": c.PanelRole(windows=ws, state=c.CoverageState.FILMED)}) for pid, ws in WINDOWS.items()]
    c.write_json(paths.panels_json(run_id), panels)
    kf = _live_trunk_keyframe(run_id, th)
    c.write_json(paths.keyframes_json(run_id), [kf])
    result = asyncio.run(narration.derive_narration(run_id, th=th))
    rows = piece_rows(run_id)
    print(f"\nLIVE voice: {len(transcript.turns)} turns, mode_applied {transcript.mode_applied}, ${transcript.usd:.5f}, text {transcript.text!r}")
    for turn in result.per_role.get("pickup", c.NarrationRole()).turns:
        row = rows[("pickup", turn.turn_id)]
        seed = row.seed.model_dump(mode="json", exclude_none=True) if row.seed else None
        pin = f"{turn.pinned_panel.value} {turn.pin_kind.value} {turn.pin_seconds}" if turn.pinned_panel and turn.pin_kind else "no pin"
        print(f"  {turn.turn_id}: {turn.start_ms}-{turn.end_ms} ms {f'(pinned on {row.est_ms}) ' if row.estimated else ''}{turn.text!r} -> {pin}, chip {turn.chip_path}, seed {seed}")
    for m in seed_marks(run_id):
        print(f"  seed mark {m.mark_id}: {m.type.value} {m.size_class.value} bbox_crop {m.bbox_crop} grid {m.grid_bbox} w {m.w} caps {m.caps}")
    spend = [r for r in ledger.rows() if r.get("run_id") == run_id]
    print(f"  ledger rows for {run_id}: {[(r['kind'], r.get('note'), r['usd']) for r in spend]}, ${sum(r['usd'] for r in spend):.5f}")
    print(f"  crop sent: {paths.run_dir(run_id) / kf.crop.path}")
    assert transcript.turns and result.mode_applied
    trunk = [r.seed for r in rows.values() if r.seed is not None and r.seed.kf_id == kf.kf_id]
    assert trunk and trunk[0].request_id and trunk[0].status in ("found", "on_existing_mark", "not_found", "off_panel")
    assert len([r for r in spend if r["kind"] == "spark"]) <= 2 and sum(r["usd"] for r in spend) < 0.05
