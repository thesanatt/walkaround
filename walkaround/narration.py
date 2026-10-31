"""Narration (M10, backlog B6) and the seed hand-off (B11): the renter's own words, pinned to panels, as evidence, and
a spoken mark turned into a Muse Spark box that the match and Stays Put use like any mark.

Muse Voice gives turn spans only, no word times, and ENDPOINTING merges sentences parted by pauses under about 3 s
(D1, Oct 31: one 11 s turn named the front bumper, the hood and a door). So a turn is cut where it names a new panel,
and every piece keeps the Voice turn's own span in narration.json: no word time is invented, and each chip holds the
whole turn it came from. Only the pin is chosen on an estimate, each piece spread over its turn in proportion to its
character offsets and recorded as one (est_ms) in derived/narration_pieces.json. A piece pins to the slot it names
(side and position words from the text, else the side vote that named the slots in derived/panels.json) whose SAM
window overlaps it most ("overlap"). When SAM shows none of the named slots during the piece, the pin is "about":
under thresholds voice.pin_fallback turn_start_panel (ARCHITECTURE 11 E10) the slot of the named kind that SAM shows
at the piece's start, side words set aside (the words name the kind of panel, the view names which one), else the
named slot SAM shows nearest in time, or the one slot the words name when SAM never showed it. A piece that names no
panel pins nowhere. Every turn gets an AAC chip cut from the walk's WAV.

Seeds: a pinned piece that names a mark (loose forms such as "scrat", "chipped" or "ding" count; negated words do
not) asks spark_client.seed_mark once on that walk's sharpest blurred crop of the panel. A found box that touches the
panel mask becomes a narration_seed mark in marks.json, mapped and whitened like a box-pass mark. seed_v1 gives no
confidence, so code records confidence low and caps the mark (seed_cap: K14 once match reads it, else K12): a seed can
pull a label down to needs your eye but never makes "already there" on its own. A box the match would call the same
mark as one this walk already has on the panel (match.cost d under tau_eye_r, from calibration.json) links the words
to that mark instead, so no twin competes with it in the match. Not found is recorded, never invented. Seed outcomes
are recorded in narration_pieces.json; every attempt goes to spark.json and every question to seeds.json at the run
root, so rebuild_narration replays narration.json, narration_pieces.json and the seed marks with no model call.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import math
import os
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from . import marks, match, paths, spark_client, video
from .config import redact
from .contracts import (
    CAPPED_MARK_KINDS,
    PANEL_NOUN,
    WHEEL_PANELS,
    Box4,
    Calibration,
    Confidence,
    Contract,
    CoverageState,
    GridBox,
    GridPoint,
    InputRef,
    Keyframe,
    Mark,
    MarkKind,
    MarkSource,
    MatchCandidate,
    Narration,
    NarrationRole,
    NarrationTurn,
    Noun,
    Panel,
    PanelId,
    PinKind,
    Position,
    Role,
    Side,
    SizeClass,
    SparkMark,
    SparkRequestRecord,
    VideoRecord,
    VoiceMode,
    VoiceRecord,
    read_json,
    read_json_list,
    write_json,
)

# Defaults for optional thresholds leaves voice.chip_min_ms and voice.seeds_per_walk_max (estimates, TODO(E10)).
CHIP_MIN_MS = 100  # a shorter turn gets no chip
SEEDS_PER_WALK_MAX = 16  # distinct seed questions one walk may ask (about $0.005 each)
SEED_CONFIDENCE = Confidence.LOW  # seed_v1 has no confidence field
PIN_FALLBACKS = ("turn_start_panel", "named_panel_nearest")  # thresholds voice.pin_fallback

# Spoken forms of the panel nouns; at one spot the longest form wins. car and car window are not panel slots.
PANEL_FORMS: dict[Noun, tuple[str, ...]] = {
    Noun.CAR_DOOR: ("car door", "car doors", "door", "doors"),
    Noun.BUMPER: ("bumper", "bumpers"),
    Noun.WHEEL: ("wheel", "wheels", "rim", "rims", "tire", "tires", "tyre", "tyres", "hubcap", "hubcaps", "alloy", "alloys"),
    Noun.HOOD: ("hood", "bonnet"),
    Noun.TRUNK_LID: ("trunk lid", "trunk", "boot lid", "boot", "tailgate", "liftgate", "hatch"),
    Noun.FENDER: ("fender", "fenders", "quarter panel", "quarter panels", "wheel arch", "wheel arches", "wheel well", "wheel wells"),
    Noun.ROOF: ("roof",),
    Noun.HEADLIGHT: ("headlight", "headlights", "head light", "head lights", "headlamp", "headlamps"),
    Noun.TAIL_LIGHT: ("tail light", "tail lights", "taillight", "taillights", "tail lamp", "tail lamps", "brake light", "brake lights"),
    Noun.SIDE_MIRROR: ("side mirror", "side mirrors", "wing mirror", "wing mirrors", "mirror", "mirrors"),
    Noun.WINDSHIELD: ("windshield", "windscreen"),
    Noun.REAR_WINDOW: ("rear window", "back window", "rear windscreen", "back glass"),
}
NOT_PANELS = ("rear view mirror", "rear view mirrors", "rearview mirror", "rearview mirrors", "interior mirror")  # inside the car
# Mark words: spoken forms (ASR slips seen or likely included), the MarkKind they name and their Voice keyword.
MARK_FORMS: tuple[tuple[tuple[str, ...], MarkKind | None, str | None], ...] = (
    (("scratch", "scratches", "scratched", "scratchy", "scrat", "scratc", "scrach", "gouge", "gouges", "gouged"), MarkKind.SCRATCH, "scratch"),
    (("scuff", "scuffs", "scuffed", "scuffing", "scrape", "scrapes", "scraped", "curb rash", "kerb rash", "curbed", "kerbed", "paint transfer"), MarkKind.SCUFF, "scuff"),
    (("chip", "chips", "chipped", "chipping", "nick", "nicks", "nicked"), MarkKind.CHIP, "chip"),
    (("dent", "dents", "dented", "ding", "dings", "dinged"), MarkKind.DENT, "dent"),
    (("crack", "cracks", "cracked"), MarkKind.CRACK, "crack"),
    (("paint damage", "damaged paint", "missing paint", "paint missing", "peeling paint", "paint peeling", "mark", "marks", "damage", "damaged"), None, None),
)
OTHER_FORMS: dict[str, tuple[str, ...]] = {
    "car": ("car", "cars"),
    "car window": ("car window", "car windows", "window", "windows"),
    "tape": ("tape", "taped"),
    "odometer": ("odometer", "mileage", "odo"),
}
# Truncated mark words fixed in the phrase Spark reads (the transcript itself is kept as heard).
ASR_FIXES = {"scrat": "scratch", "scratc": "scratch", "scrach": "scratch"}

SIDE_TERMS = ("left", "right", "driver", "driver's", "drivers", "passenger", "passenger's", "passengers")
POSITION_WORDS = {"front": Position.FRONT, "rear": Position.REAR, "back": Position.REAR}
SIZE_WORDS = {
    "small": SizeClass.S, "tiny": SizeClass.S, "little": SizeClass.S, "minor": SizeClass.S, "light": SizeClass.S,
    "slight": SizeClass.S, "faint": SizeClass.S, "hairline": SizeClass.S,
    "big": SizeClass.L, "large": SizeClass.L, "huge": SizeClass.L, "long": SizeClass.L, "deep": SizeClass.L,
    "major": SizeClass.L, "massive": SizeClass.L,
}  # fmt: skip
NEGATORS = frozenset(
    "no not never without zero none nothing free cannot isn't isnt doesn't doesnt don't dont didn't didnt wasn't wasnt "
    "aren't arent hasn't hasnt haven't havent can't cant won't wont".split()
)
CLAUSE_BREAKS = frozenset("but except though although however wait actually".split())  # a negator before one is spent
DETERMINERS = frozenset("the a an this that my its his her their our".split())
CONNECTORS = frozenset("and then also plus next now so okay ok".split())
PANEL_PREPOSITIONS = frozenset("on in at along across".split())  # "a dent on the hood": the mark before them is the hood's
COUNT_WORDS = frozenset("one two three four five six several some few couple of another multiple many lots lot".split())
DEGREE_WORDS = frozenset("very really pretty quite bit kind sort of so too all".split())
COPULAS = frozenset("is was are were be been being looks look looked seems seem seemed appears appear appeared got gets it's that's thats".split())
_QUALIFIERS = frozenset(SIDE_TERMS) | frozenset(POSITION_WORDS) | {"side"} | DETERMINERS
_LEADERS = DETERMINERS | CONNECTORS | PANEL_PREPOSITIONS
_MARK_PHRASE = frozenset(SIZE_WORDS) | COUNT_WORDS | DETERMINERS | CONNECTORS | DEGREE_WORDS
_POST_LEADERS = frozenset({"on", "at", "in", "the"})
_SIDE_POSITION = frozenset(SIDE_TERMS) | frozenset(POSITION_WORDS) | {"side"}
_SIDED = (Noun.CAR_DOOR, Noun.FENDER, Noun.WHEEL)
_KINDS = frozenset(k.value for k in MarkKind)

SeedStatus = Literal["found", "on_existing_mark", "not_found", "off_panel", "failed", "not_asked"]
LINKED: tuple[str, ...] = ("found", "on_existing_mark")  # the statuses whose mark the turn's seed_mark_id names


def side_words(passenger: Side = Side.RIGHT) -> dict[str, Side]:
    """Spoken side words; passenger and driver follow thresholds claim.passenger_side_is (right in the US)."""
    driver = Side.LEFT if passenger == Side.RIGHT else Side.RIGHT
    return {"left": Side.LEFT, "right": Side.RIGHT} | {w: passenger for w in SIDE_TERMS if w.startswith("passenger")} | {w: driver for w in SIDE_TERMS if w.startswith("driver")}


def seed_cap() -> str:
    """The cap every seed mark carries: K14 (boxed because the renter named it) once match reads it from the marks,
    else K12, the nearest rule until then."""
    return "K14" if "K14" in match.CAP_IDS_FROM_MARK else "K12"


# Records ---------------------------------------------------------------------------------------------------------------


class SeedAsk(Contract):
    """One seeds.json row (frozen, run root): a seed question as sent and every Spark record that answered it."""

    role: Role
    turn_id: str
    panel_id: PanelId
    kf_id: str
    crop: InputRef
    phrase: str
    request_ids: list[str] = Field(default_factory=list)  # first to last; the last one stands
    error: str | None = None  # redacted, when the call raised before Spark answered


class SeedOutcome(Contract):
    """What a piece's seed came to; mark_id is the new seed mark (found) or the mark the box is (on_existing_mark)."""

    status: SeedStatus
    reason: str = ""
    phrase: str
    kf_id: str | None = None
    request_id: str | None = None
    mark_id: str | None = None
    type: MarkKind | None = None
    type_from: Literal["spark", "words", "unsure"] | None = None  # found only: where the seed mark's type came from
    size_from: Literal["words", "default"] | None = None  # found only: seed_v1 gives no size class
    bbox_crop: Box4 | None = None


class PieceRow(Contract):
    """One derived/narration_pieces.json row: what a narration turn carries beyond contracts.NarrationTurn."""

    role: Role
    turn_id: str
    voice_turn: int  # 1-based position of the Voice turn the piece was cut from
    estimated: bool  # cut from a longer Voice turn: the pin was chosen on est_ms
    est_ms: list[int] | None = None  # the piece's span spread by character position inside the turn (an estimate)
    chip_ms: list[int] | None = None  # the audio span the chip holds
    seed: SeedOutcome | None = None


def seeds_json(run_id: str) -> Path:
    """seeds.json at the run root, next to spark.json and voice.json; never rewritten by a rebuild."""
    return paths.run_dir(run_id) / "seeds.json"


def pieces_json(run_id: str) -> Path:
    return paths.derived_dir(run_id) / "narration_pieces.json"


def _read_list(path: Path, model: Any) -> list[Any]:
    return read_json_list(path, model) if path.is_file() and path.stat().st_size > 0 else []


def _sha256(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _role_key(role: str) -> tuple[int, int, str]:
    if role in ("pickup", "return"):
        return (0 if role == "pickup" else 1, 0, role)
    head, _, n = role.partition("_")
    return (2 if head == "topup" else 3, int(n) if n.isdigit() else 0, role)


# Words: panel mentions, mark words, keywords -----------------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    word: str  # lower case


_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def _tokens(text: str) -> list[_Token]:
    return [_Token(m.start(), m.end(), m.group().lower()) for m in _TOKEN.finditer(text)]


def _pattern(forms: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(f).replace(r"\ ", r"\s+") for f in sorted(forms, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


_NOT_PANEL = _pattern(NOT_PANELS)
_PANEL_PATTERNS = [(_pattern(forms), noun) for noun, forms in PANEL_FORMS.items()] + [(_NOT_PANEL, None)]
_MARK_PATTERNS = [(_pattern(forms), (kind, keyword)) for forms, kind, keyword in MARK_FORMS]
_KEYWORD_PATTERNS = (
    [(_pattern(forms), noun.value) for noun, forms in PANEL_FORMS.items()]
    + [(_pattern(forms), keyword) for forms, _, keyword in MARK_FORMS if keyword]
    + [(_pattern(forms), keyword) for keyword, forms in OTHER_FORMS.items()]
    + [(_NOT_PANEL, None)]
)
_ASR_FIX = re.compile(r"\b(" + "|".join(ASR_FIXES) + r")\b", re.IGNORECASE)


def _spans(text: str, patterns: list[tuple[re.Pattern[str], Any]]) -> list[tuple[int, int, Any]]:
    """Non-overlapping matches in text order; at one spot the longest match wins."""
    found = sorted(((m.start(), m.end(), payload) for pattern, payload in patterns for m in pattern.finditer(text)), key=lambda s: (s[0], -(s[1] - s[0])))
    taken: list[tuple[int, int, Any]] = []
    for span in found:
        if not taken or span[0] >= taken[-1][1]:
            taken.append(span)
    return taken


def _token_index(tokens: list[_Token], offset: int) -> int:
    """The token holding char offset (the nearest one before it otherwise)."""
    return max(0, bisect.bisect_right([t.start for t in tokens], offset) - 1)


def keywords_heard(text: str, keywords: list[str]) -> list[str]:
    """The Voice keywords heard in text, spelled as the list spells them, in order of first occurrence."""
    allowed = set(keywords)
    out: list[str] = []
    for _, _, keyword in _spans(text, _KEYWORD_PATTERNS):
        if keyword in allowed and keyword not in out:
            out.append(keyword)
    return out


def _phrase(text: str) -> str:
    """The words as Spark reads them inside the prompt's quotes: truncated mark words fixed, double quotes made single."""
    return _ASR_FIX.sub(lambda m: ASR_FIXES[m.group(1).lower()], text).replace('"', "'")


@dataclass(frozen=True)
class _MarkWord:
    first: int  # token indices
    last: int
    kind: MarkKind | None


@dataclass(frozen=True)
class _Mention:
    noun: Noun
    first: int  # token indices of the noun words
    last: int
    run: int  # token index where a piece about this mention begins
    side: Side | None
    position: Position | None


def _agree(a: Any, b: Any) -> Any:
    """One qualifier from two readings; a disagreement is no reading."""
    if a is None or b is None:
        return a if b is None else b
    return a if a == b else None


def _mark_words(text: str, tokens: list[_Token]) -> list[_MarkWord]:
    return [_MarkWord(_token_index(tokens, s), _token_index(tokens, e - 1), payload[0]) for s, e, payload in _spans(text, _MARK_PATTERNS)]


def _modifier(tokens: list[_Token], k: int) -> bool:
    """A size or degree word ("small", "badly", "a bit") that can stand inside a mark phrase."""
    word = tokens[k].word
    if word in ("a", "an"):
        return k + 1 < len(tokens) and tokens[k + 1].word in ("bit", "little")
    return word in SIZE_WORDS or word in DEGREE_WORDS or word.endswith("ly")


def _attributive(tokens: list[_Token], w: _MarkWord, bound: int, by_last: dict[int, _MarkWord]) -> _MarkWord | None:
    """The first mark word of an attributive phrase ending in w, or None. A mark word ending in -ed right before a
    panel noun modifies it ("a dented front bumper", "roof is clean chipped windshield") unless a copula or the previous
    noun stands before it ("the hood is dented", "door scratched"); after and or or it goes with the word before it
    ("is scratched and dented" stays, "a scratched and dented front bumper" moves whole)."""
    last = tokens[w.last].word
    if not (last.endswith("ed") or last == "scratchy"):
        return None
    k = w.first - 1
    while k >= bound and _modifier(tokens, k):
        k -= 1
    if k < bound:
        return w if bound == 0 else None  # first words of the turn; otherwise the previous panel noun stands before it
    if tokens[k].word in ("and", "or") and k - 1 >= bound and k - 1 in by_last:
        return _attributive(tokens, by_last[k - 1], bound, by_last)
    return None if tokens[k].word in COPULAS or tokens[k].word in _SIDE_POSITION else w


def _phrase_start(tokens: list[_Token], first: int, bound: int) -> int:
    """The first token of the mark phrase whose mark word starts at token `first` ("and two small chips")."""
    j = first - 1
    while j >= bound and (tokens[j].word in _MARK_PHRASE or tokens[j].word.endswith("ly") or tokens[j].word.isdigit()):
        j -= 1
    return j + 1


def _pre_run(tokens: list[_Token], first: int, bound: int, by_last: dict[int, _MarkWord], sides: dict[str, Side]) -> tuple[int, Side | None, Position | None]:
    """Where a piece about the noun at token `first` begins, and the side and position words right before the noun.
    Leaders (the, and, on ...) join the piece, and so does the mark phrase before a panel preposition ("a small dent on
    the hood") or right before the noun when its mark word modifies the noun ("a dented front bumper")."""
    j = first - 1
    side = position = None
    while j >= bound and tokens[j].word in _QUALIFIERS:
        side = side or sides.get(tokens[j].word)
        position = position or POSITION_WORDS.get(tokens[j].word)
        j -= 1
    head = _attributive(tokens, by_last[j], bound, by_last) if j >= bound and j in by_last else None
    if head is not None:
        return _phrase_start(tokens, head.first, bound), side, position
    via_preposition = False
    while j >= bound and tokens[j].word in _LEADERS:
        via_preposition = via_preposition or tokens[j].word in PANEL_PREPOSITIONS
        j -= 1
    if via_preposition and j >= bound and j in by_last:
        return _phrase_start(tokens, by_last[j].first, bound), side, position
    return j + 1, side, position


def _post_qualifiers(tokens: list[_Token], k: int, stop: int, sides: dict[str, Side]) -> tuple[Side | None, Position | None]:
    """Side and position words right after a noun: 'door on the left', 'bumper at the back', 'door front left'."""
    while k < stop and tokens[k].word in _POST_LEADERS:
        k += 1
    side = position = None
    while k < stop and tokens[k].word in _SIDE_POSITION:
        side = side or sides.get(tokens[k].word)
        position = position or POSITION_WORDS.get(tokens[k].word)
        k += 1
    return side, position


def _mentions(text: str, tokens: list[_Token], words: list[_MarkWord], sides: dict[str, Side]) -> list[_Mention]:
    """Panel nouns in text order with their side and position words and where a piece about each would begin. A
    rear view mirror is inside the car: no panel."""
    by_last = {w.last: w for w in words}
    raw: list[tuple[Noun, int, int, int, Side | None, Position | None]] = []
    bound = 0
    for start, end, noun in _spans(text, _PANEL_PATTERNS):
        first, last = _token_index(tokens, start), _token_index(tokens, end - 1)
        if noun is not None:
            run, side, position = _pre_run(tokens, first, bound, by_last, sides)
            raw.append((noun, first, last, run, side, position))
        bound = last + 1
    out: list[_Mention] = []
    for i, (noun, first, last, run, side, position) in enumerate(raw):
        stop = raw[i + 1][3] if i + 1 < len(raw) else len(tokens)
        post_side, post_position = _post_qualifiers(tokens, last + 1, stop, sides)
        out.append(_Mention(noun, first, last, run, _agree(side, post_side), _agree(position, post_position)))
    return out


def _same_panel(a: _Mention, b: _Mention) -> bool:
    return a.noun == b.noun and (a.side is None or b.side is None or a.side == b.side) and (a.position is None or b.position is None or a.position == b.position)


def _merged(a: _Mention, b: _Mention) -> _Mention:
    return replace(a, side=a.side or b.side, position=a.position or b.position)


def _negated(tokens: list[_Token], w: _MarkWord, lo: int) -> bool:
    """A negator among the three tokens before the mark word with no clause break after it ("no scratches", "I don't
    see any dents"; not "never mind, a small dent" or "no, wait, a dent")."""
    for k in range(w.first - 1, max(lo, w.first - 3) - 1, -1):
        word = tokens[k].word
        if word in CLAUSE_BREAKS:
            return False
        if word in NEGATORS and tokens[k + 1].word != "mind":
            return True
    return False


def _marks_between(tokens: list[_Token], words: list[_MarkWord], lo: int, hi: int, panel_tokens: set[int]) -> tuple[tuple[MarkKind | None, ...], SizeClass | None]:
    """The mark words of tokens lo..hi-1 that are not negated, and the size word among the three tokens before the
    first of them (light in "tail light" names the panel, not a size)."""
    kinds: list[MarkKind | None] = []
    size: SizeClass | None = None
    for w in words:
        if not lo <= w.first < hi or _negated(tokens, w, lo):
            continue
        if not kinds:
            near = range(w.first - 1, max(lo, w.first - 3) - 1, -1)
            size = next((SIZE_WORDS[tokens[k].word] for k in near if tokens[k].word in SIZE_WORDS and k not in panel_tokens), None)
        kinds.append(w.kind)
    return tuple(kinds), size


@dataclass(frozen=True)
class Piece:
    """A narration turn as pinned: one Voice turn, or the part of one that names one panel. start_ms and end_ms are
    the turn's own span for a whole turn and an estimate for a piece cut from a longer one; turn_ms is Voice's span."""

    turn: int  # 1-based Voice turn
    index: int  # 1-based piece of that turn
    count: int  # pieces cut from that turn
    start_ms: int
    end_ms: int
    turn_ms: tuple[int, int]
    text: str
    noun: Noun | None
    side: Side | None
    position: Position | None
    kinds: tuple[MarkKind | None, ...]  # mark words heard and not negated, in order
    size: SizeClass | None
    keywords: tuple[str, ...]

    @property
    def turn_id(self) -> str:
        return f"t{self.turn}" if self.count == 1 else f"t{self.turn}_{self.index}"

    @property
    def estimated(self) -> bool:
        return self.count > 1


def split_turn(text: str, start_ms: int, end_ms: int, *, turn: int, keywords: list[str], passenger: Side = Side.RIGHT) -> list[Piece]:
    """Cut one Voice turn where it names a panel the current piece does not name, each piece's estimated span taking
    the turn's time in proportion to its character offsets. A turn naming one panel or none comes back whole. A piece
    names one slot: "front and rear bumpers have scuffs" is about the rear bumper."""
    text = text.strip()
    tokens = _tokens(text)
    words = _mark_words(text, tokens)
    mentions = _mentions(text, tokens, words, side_words(passenger))
    panel_tokens = {k for m in mentions for k in range(m.first, m.last + 1)}
    cuts: list[tuple[int, _Mention | None]] = []
    for m in mentions:
        if cuts and cuts[-1][1] is not None and _same_panel(cuts[-1][1], m):
            cuts[-1] = (cuts[-1][0], _merged(cuts[-1][1], m))
        else:
            cuts.append((0 if not cuts else m.run, m))
    cuts = cuts or [(0, None)]
    total, span = len(tokens), end_ms - start_ms
    chars = len(text) or 1
    pieces: list[Piece] = []
    for i, (lo, mention) in enumerate(cuts):
        hi = cuts[i + 1][0] if i + 1 < len(cuts) else total
        a = tokens[lo].start if lo < total and i else 0
        b = tokens[hi].start if hi < total else len(text)
        kinds, size = _marks_between(tokens, words, lo, hi, panel_tokens)
        piece_text = text[a:b].strip()
        pieces.append(
            Piece(
                turn=turn,
                index=i + 1,
                count=len(cuts),
                start_ms=start_ms + round(span * a / chars),
                end_ms=end_ms if i + 1 == len(cuts) else start_ms + round(span * b / chars),
                turn_ms=(start_ms, end_ms),
                text=piece_text,
                noun=mention.noun if mention else None,
                side=mention.side if mention else None,
                position=mention.position if mention else None,
                kinds=kinds,
                size=size,
                keywords=tuple(keywords_heard(piece_text, keywords)),
            )
        )
    return pieces


# Pins ------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pin:
    panel: PanelId | None = None
    kind: PinKind | None = None
    seconds: float | None = None  # seconds SAM shows the panel during the Voice turn (overlap pins only)


def _slot(pid: PanelId) -> tuple[Side | None, Position | None]:
    parts = pid.value.split("_")
    side = Side(parts[0]) if parts[0] in ("left", "right") else None
    position = Position.FRONT if "front" in parts else Position.REAR if "rear" in parts else None
    return side, position


def candidates(noun: Noun, side: Side | None, position: Position | None) -> list[PanelId]:
    """The panel slots a noun with these words can name, in enum order. Side words only narrow doors, fenders and
    wheels; position words narrow those and the bumpers."""
    out: list[PanelId] = []
    for pid in PanelId:
        if PANEL_NOUN[pid] != noun:
            continue
        s, p = _slot(pid)
        if noun in _SIDED and side is not None and s != side:
            continue
        if noun in (*_SIDED, Noun.BUMPER) and position is not None and p != position:
            continue
        out.append(pid)
    return out


def role_windows(panels: list[Panel], role: str) -> dict[PanelId, list[tuple[float, float]]]:
    """Per panel, the windows in seconds where SAM shows it in this walk (derived/panels.json)."""
    out: dict[PanelId, list[tuple[float, float]]] = {}
    for p in panels:
        pr = p.per_role.get(role)
        if pr is not None and pr.windows:
            out[p.panel_id] = [(float(a), float(b)) for a, b in pr.windows]
    return out


def role_states(panels: list[Panel], role: str) -> dict[PanelId, CoverageState]:
    """Per panel, its coverage state in this walk (derived/panels.json)."""
    return {p.panel_id: p.per_role[role].state for p in panels if role in p.per_role}


def _gap(t0: float, t1: float, a: float, b: float) -> float:
    return max(a - t1, t0 - b, 0.0)


def _overlap(windows: dict[PanelId, list[tuple[float, float]]], pid: PanelId, t0: float, t1: float) -> float:
    return sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in windows.get(pid, []))


def _longest(pids: list[PanelId], windows: dict[PanelId, list[tuple[float, float]]], t0: float, t1: float, *, at: float | None = None) -> PanelId | None:
    """The slot SAM shows longest during t0..t1, one shown at `at` first when given; a tie goes to enum order."""
    scored = [(at is not None and any(a <= at < b for a, b in windows.get(pid, [])), _overlap(windows, pid, t0, t1), -i, pid) for i, pid in enumerate(pids)]
    shown = [s for s in scored if s[1] > 0]
    return max(shown)[3] if shown else None


def pin_piece(piece: Piece, windows: dict[PanelId, list[tuple[float, float]]], *, fallback: str = "turn_start_panel") -> Pin:
    """overlap: the named slot SAM shows longest during the piece. about, when SAM shows none of them then: under
    turn_start_panel the slot of the named kind SAM shows at the piece's start (side words set aside), else the named
    slot SAM shows nearest in time, else the one slot the words name. No noun, no pin."""
    if fallback not in PIN_FALLBACKS:
        raise ValueError(f"voice.pin_fallback {fallback!r} is not one of {', '.join(PIN_FALLBACKS)}")
    if piece.noun is None:
        return Pin()
    named = candidates(piece.noun, piece.side, piece.position)
    t0, t1 = piece.start_ms / 1000, piece.end_ms / 1000
    best = _longest(named, windows, t0, t1)
    if best is not None:
        return Pin(best, PinKind.OVERLAP, _overlap(windows, best, piece.turn_ms[0] / 1000, piece.turn_ms[1] / 1000))
    if fallback == "turn_start_panel":
        # TODO(E10): no D1 turn reached this branch (every noun-naming turn landed, gate_voice.json pin_fallback null);
        # a walk where the renter's side words and the side vote disagree must confirm that the view names the side.
        on_screen = _longest([pid for pid in candidates(piece.noun, None, piece.position) if pid not in named], windows, t0, t1, at=t0)
        if on_screen is not None:
            return Pin(on_screen, PinKind.ABOUT)
    shown = [(min(_gap(t0, t1, a, b) for a, b in windows[pid]), pid) for pid in named if windows.get(pid)]
    if shown:
        return Pin(min(shown, key=lambda s: s[0])[1], PinKind.ABOUT)
    return Pin(named[0], PinKind.ABOUT) if len(named) == 1 else Pin()


# Walks and seed questions ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One seed question: a pinned piece that names a mark, the crop it goes to, or why it is not asked."""

    role: str
    piece: Piece
    panel: PanelId
    phrase: str
    kf: Keyframe | None = None
    crop_sha256: str | None = None
    reason: str = ""  # empty when the question can be asked

    @property
    def key(self) -> tuple[str, str, str]:
        """Same crop bytes, panel and words: one answer for the run (a walk given twice asks once)."""
        return (self.crop_sha256 or "", self.panel.value, self.phrase)


@dataclass
class Walk:
    record: VoiceRecord
    applied: bool
    pieces: list[Piece]
    pins: list[Pin]
    questions: list[Question | None]  # one per piece; None where no seed is wanted


def mode_applied(record: VoiceRecord) -> bool:
    """False when a speech mode came back with a transcript and no turns (section 14 item 16b), whatever was stored."""
    silent = record.request.mode != VoiceMode.PUSH_TO_TALK and not record.response.turns and bool(record.response.transcript.strip())
    return record.mode_applied and not silent


def _voice_records(run_id: str) -> list[VoiceRecord]:
    """The last voice.json record of each walk, in role order."""
    by_role: dict[str, VoiceRecord] = {}
    for record in _read_list(paths.voice_json(run_id), VoiceRecord):
        by_role[record.role] = record
    return [by_role[role] for role in sorted(by_role, key=_role_key)]


def _sharpest(keyframes: list[Keyframe], role: str, pid: PanelId, root: Path) -> Keyframe | None:
    kfs = [k for k in keyframes if k.role == role and k.panel_id == pid and k.crop is not None and (root / k.crop.path).is_file()]
    return min(kfs, key=lambda k: (-k.score.sharpness, -k.score.total, k.kf_id), default=None)


def _no_crop(pid: PanelId, windows: dict[PanelId, list[tuple[float, float]]], states: dict[PanelId, CoverageState]) -> str:
    name = spark_client.panel_display_name(pid)
    if windows.get(pid):
        return f"no crop of the {name} in this walk"
    if states.get(pid) == CoverageState.NOT_RUN:
        return f"SAM did not run for the {name} in this walk"
    return f"SAM did not find the {name} in this walk"


def _questions(root: Path, role: str, pieces: list[Piece], pins: list[Pin], keyframes: list[Keyframe], windows: dict[PanelId, list[tuple[float, float]]], states: dict[PanelId, CoverageState], cap: int) -> list[Question | None]:
    out: list[Question | None] = []
    keys: set[tuple[str, str, str]] = set()
    for piece, pin in zip(pieces, pins):
        if pin.panel is None or not piece.kinds:
            out.append(None)
            continue
        phrase = _phrase(piece.text)
        kf = _sharpest(keyframes, role, pin.panel, root)
        if kf is None or kf.crop is None:
            out.append(Question(role, piece, pin.panel, phrase, reason=_no_crop(pin.panel, windows, states)))
            continue
        q = Question(role, piece, pin.panel, phrase, kf, kf.crop.sha256 or _sha256(root / kf.crop.path))
        if q.key not in keys and len(keys) >= cap:
            q = replace(q, reason=f"over the {cap} seeds one walk may ask")
        else:
            keys.add(q.key)
        out.append(q)
    return out


def walks(run_id: str, *, th: dict) -> list[Walk]:
    """Every walk in voice.json split, pinned against derived/panels.json and given its seed questions."""
    root = paths.run_dir(run_id)
    panels = _read_list(paths.panels_json(run_id), Panel)
    keyframes = _read_list(paths.keyframes_json(run_id), Keyframe)
    keywords = [str(k) for k in th["nouns"]["voice_keywords"]]
    passenger = Side(str(th["claim"]["passenger_side_is"]))
    fallback = str(th["voice"]["pin_fallback"])
    cap = int(th["voice"].get("seeds_per_walk_max", SEEDS_PER_WALK_MAX))
    out: list[Walk] = []
    for record in _voice_records(run_id):
        applied = mode_applied(record)
        pieces: list[Piece] = []
        if applied:
            for i, turn in enumerate(record.response.turns, 1):
                if turn.transcript.strip():
                    start = int(turn.startMs)
                    pieces += split_turn(turn.transcript, start, max(start, int(turn.endMs)), turn=i, keywords=keywords, passenger=passenger)
        windows = role_windows(panels, record.role)
        pins = [pin_piece(p, windows, fallback=fallback) for p in pieces]
        questions = _questions(root, record.role, pieces, pins, keyframes, windows, role_states(panels, record.role), cap)
        out.append(Walk(record, applied, pieces, pins, questions))
    return out


# Seed answers and seed marks ----------------------------------------------------------------------------------------


def _latest_asks(run_id: str) -> dict[tuple[str, str, str], SeedAsk]:
    """seeds.json by question key; a later row of the same key wins."""
    return {(a.crop.sha256, a.panel_id.value, a.phrase): a for a in _read_list(seeds_json(run_id), SeedAsk)}


def _grid_box(raw: Any) -> GridBox | None:
    if not isinstance(raw, dict):
        return None
    v = [spark_client.clamp_grid(raw.get(k)) for k in ("x_min", "y_min", "x_max", "y_max")]
    if any(x is None for x in v):
        return None
    x1, y1, x2, y2 = v  # type: ignore[misc]
    return GridBox(x_min=min(x1, x2), y_min=min(y1, y2), x_max=max(x1, x2), y_max=max(y1, y2))


def seed_answer(record: SparkRequestRecord) -> tuple[bool, GridBox | None, MarkKind | None]:
    """found, box and type from a stored seed_v1 record, read as spark_client.seed_mark reads a fresh one."""
    parsed = record.parsed if record.valid else None
    if not parsed:
        return False, None, None
    box = _grid_box(parsed.get("bbox"))
    found = bool(parsed.get("found")) and box is not None
    kind = parsed.get("type")
    return found, box if found else None, MarkKind(kind) if found and kind in _KINDS else None


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


def _calibration(run_id: str, th: dict) -> Calibration:
    """calibration.json as the marks stage left it (the bands the match will use), else what match would compute."""
    path = paths.calibration_json(run_id)
    if path.is_file() and path.stat().st_size > 0:
        return read_json(path, Calibration)
    sigma_frame, sigma_spark = match.measured_sigmas(th)
    return match.calibrate(sigma_frame=sigma_frame, sigma_spark=sigma_spark, sigma_view=None, pseudo_pair_d=None, th=th)


def _seed_mark(root: Path, q: Question, record: SparkRequestRecord, box: GridBox, kind: MarkKind | None, th: dict) -> tuple[Mark, np.ndarray | None]:
    """The found box as a narration_seed Mark: grid to crop to source like a box-pass mark, then whitened."""
    kf = q.kf
    assert kf is not None and kf.crop is not None
    with Image.open(root / kf.crop.path) as img:
        sent_w, sent_h = spark_client.sent_size(*img.size)  # the size seed_mark sent the crop at
    mark_type = kind or next((k for k in q.piece.kinds if k is not None), MarkKind.UNSURE)
    size = q.piece.size or SizeClass.M  # seed_v1 has no size class; the renter's size word, else the middle class
    point = GridPoint(x=(box.x_min + box.x_max) // 2, y=(box.y_min + box.y_max) // 2)
    sm = SparkMark(type=mark_type, point=point, bbox=box, size_class=size, confidence=SEED_CONFIDENCE, note=q.phrase)
    bbox_crop, point_crop, bbox_src, point_src = marks.map_spark_mark(sm, SimpleNamespace(sent_width=sent_w, sent_height=sent_h), kf.crop)  # type: ignore[arg-type]
    size_px = max(bbox_src[2] - bbox_src[0], bbox_src[3] - bbox_src[1])
    caps = [cap for cap, on in (("K3", size_px < float(th["dedup"]["small_mark_px"])), ("K4", mark_type in CAPPED_MARK_KINDS), (seed_cap(), True)) if on]
    mark = Mark(
        mark_id=f"{kf.kf_id}_seed_{q.piece.turn_id}",
        role=q.role,
        panel_id=kf.panel_id,
        kf_id=kf.kf_id,
        source=MarkSource.NARRATION_SEED,
        type=mark_type,
        confidence=SEED_CONFIDENCE,
        size_class=size,
        description=q.phrase,
        grid_point=point,
        grid_bbox=box,
        bbox_crop=bbox_crop,
        point_crop=point_crop,
        bbox_src=bbox_src,
        point_src=point_src,
        size_px_src=size_px,
        spark_request_id=record.request_id,
        caps=caps,
    )
    mask = _read_mask(root / kf.mask_png) if kf.mask_png else None
    whitened, _ = marks.whiten_marks([mark], kf, th=th, mask=mask if kf.panel_id in WHEEL_PANELS else None)
    return whitened[0], mask


def _touches(mask: np.ndarray, box: Box4) -> bool:
    h, w = mask.shape[:2]
    x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x2, y2 = min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))
    return x2 > x1 and y2 > y1 and bool(mask[y1:y2, x1:x2].any())


def _inside(p: list[float], box: list[float]) -> bool:
    return box[0] <= p[0] < box[2] and box[1] <= p[1] < box[3]


def _linked_mark(mark: Mark, pool: list[Mark], cal: Calibration, th: dict) -> tuple[Mark, MatchCandidate | None] | None:
    """The mark of this walk and panel the seed box is, by the match's own cost: the nearest with d under tau_eye_r,
    where the match would call the two the same mark or look twice. Without whitened positions, a mark on the same crop
    whose box holds the seed's point."""
    others = [m for m in pool if m.role == mark.role and m.panel_id == mark.panel_id]
    seed_w = marks.whitened_from_marks([mark]).get(mark.mark_id)
    if seed_w is not None:
        rows = marks.whitened_from_marks(others)
        by_id = {m.mark_id: m for m in others}
        near = [match.cost(mark, m, seed_w, rows[m.mark_id], cal, th) for m in others if m.mark_id in rows]
        best = min((c for c in near if c.d < c.tau_eye_r), key=lambda c: (c.d, c.return_mark), default=None)
        if best is not None:
            return by_id[best.return_mark], best
    for m in sorted(others, key=lambda m: m.mark_id):
        if (seed_w is None or m.w is None) and m.kf_id == mark.kf_id and m.bbox_crop is not None and mark.point_crop is not None and _inside(mark.point_crop, m.bbox_crop):
            return m, None
    return None


def _outcome(root: Path, q: Question, asks: dict[tuple[str, str, str], SeedAsk], records: dict[str, SparkRequestRecord], pool: list[Mark], cal: Calibration, th: dict) -> tuple[SeedOutcome, Mark | None]:
    """The seed's outcome from the stored answer, and the new mark when the box is a mark this walk does not have yet."""
    info: dict[str, Any] = {"phrase": q.phrase, "kf_id": q.kf.kf_id if q.kf is not None else None}
    if q.reason:
        return SeedOutcome(status="not_asked", reason=q.reason, **info), None
    ask = asks.get(q.key)
    if ask is None:
        return SeedOutcome(status="not_asked", reason="no stored answer; a rebuild asks no model", **info), None
    record = records.get(ask.request_ids[-1]) if ask.request_ids else None
    if record is None:
        return SeedOutcome(status="failed", reason=ask.error or "the answer is missing from spark.json", **info), None
    info["request_id"] = record.request_id
    if not record.valid:
        return SeedOutcome(status="failed", reason=f"no usable answer ({record.incomplete_reason or 'it did not fit seed_v1'})", **info), None
    name = spark_client.panel_display_name(q.panel)
    found, box, kind = seed_answer(record)
    if not found or box is None:
        return SeedOutcome(status="not_found", reason=f"Spark found no mark matching these words on the {name}", **info), None
    mark, mask = _seed_mark(root, q, record, box, kind, th)
    info.update(type=mark.type, bbox_crop=mark.bbox_crop)
    if mask is not None and not _touches(mask, mark.bbox_crop or []):
        return SeedOutcome(status="off_panel", reason=f"the box Spark drew is off the {name}", **info), None
    link = _linked_mark(mark, pool, cal, th)
    if link is not None:
        same, cand = link
        reason = f"the box's centre is inside the box of {same.mark_id} on the same crop" if cand is None else f"the box is {cand.d:.2f} units from {same.mark_id}, inside the match band ({cand.tau_eye_r:.2f})"
        return SeedOutcome(status="on_existing_mark", reason=reason, mark_id=same.mark_id, **info), None
    type_from = "spark" if kind is not None else "words" if any(k is not None for k in q.piece.kinds) else "unsure"
    return SeedOutcome(status="found", mark_id=mark.mark_id, type_from=type_from, size_from="words" if q.piece.size else "default", **info), mark


# Chips -----------------------------------------------------------------------------------------------------------------


def _wav(run_id: str, record: VoiceRecord) -> Path | None:
    """The WAV Voice heard (its sha256 checked against voice.json), or None."""
    vpath = paths.videos_dir(run_id) / f"{record.role}.json"
    if not vpath.is_file():
        return None
    wav_record = read_json(vpath, VideoRecord).wav
    wav = paths.run_dir(run_id) / wav_record.path if wav_record is not None else None
    if wav is None or not wav.is_file() or (record.wav_sha256 and _sha256(wav) != record.wav_sha256):
        return None
    return wav


def chip_span(piece: Piece, duration_ms: int) -> tuple[int, int]:
    """The Voice turn's own span, cut at the end of the audio: a piece cut from a longer turn holds the whole turn,
    since Voice gives no word times."""
    a, b = piece.turn_ms
    if duration_ms > 0:
        b = min(b, duration_ms)
    return max(0, a), b


def _cut_chip(wav: Path, dst: Path, start_ms: int, end_ms: int) -> bool:
    """AAC in .m4a of wav from start_ms to end_ms, bitexact so the same cut gives the same bytes on one machine."""
    tmp = dst.with_name(dst.stem + ".part.m4a")
    cmd = [
        video.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_ms / 1000:.3f}", "-t", f"{(end_ms - start_ms) / 1000:.3f}", "-i", str(wav),
        "-map", "0:a:0", "-ac", "1", "-c:a", "aac", "-b:a", "64k", "-map_metadata", "-1",
        "-fflags", "+bitexact", "-flags:a", "+bitexact", "-movflags", "+faststart", "-f", "mp4", str(tmp),
    ]  # fmt: skip
    try:
        done = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        return False
    if done.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, dst)
    return True


def _chip(run_id: str, role: str, piece: Piece, duration_ms: int, wav: Path | None, min_ms: int) -> tuple[str | None, list[int] | None]:
    """Cut the piece's chip when the WAV is given; the chip file then on disk, if any, is the one narration names."""
    dst = paths.chip_path(run_id, role, piece.turn_id)
    a, b = chip_span(piece, duration_ms)
    if wav is not None and not (b - a >= min_ms and _cut_chip(wav, dst, a, b)):
        dst.unlink(missing_ok=True)
    if not dst.is_file():
        return None, None
    return paths.rel(run_id, dst), [a, b]


def _drop_unnamed_chips(run_id: str, role: str, named: set[str]) -> None:
    """A chip no turn names any more (or a cut that died half way) is removed, so the file route cannot serve it."""
    folder = paths.run_dir(run_id) / "chips" / role
    for path in sorted(folder.glob("*.m4a")) if folder.is_dir() else []:
        if path.name not in named:
            path.unlink()


# Settling: narration.json, narration_pieces.json and the seed rows of marks.json ----------------------------------------


def _role_doc(turns: list[NarrationTurn], record: VoiceRecord) -> NarrationRole:
    """The walk's turns. A transcript that came back with no turns rides along as heard (section 14 item 16b: shown
    without pins, never "no narration found") once contracts.NarrationRole carries a transcript field."""
    if not turns and record.response.transcript.strip() and "transcript" in NarrationRole.model_fields:
        return NarrationRole(turns=turns, transcript=record.response.transcript)  # type: ignore[call-arg]
    return NarrationRole(turns=turns)


def _settle(run_id: str, th: dict, all_walks: list[Walk], *, cut_chips: bool) -> Narration:
    """Everything derived from the stored answers, no model call: turns, pins, chips, seed outcomes, seed marks."""
    root = paths.run_dir(run_id)
    asks = _latest_asks(run_id)
    records = {r.request_id: r for r in _read_list(paths.spark_json(run_id), SparkRequestRecord)}
    cal = _calibration(run_id, th)
    chip_min = int(th["voice"].get("chip_min_ms", CHIP_MIN_MS))
    marks_path = paths.marks_json(run_id)
    stored = _read_list(marks_path, Mark)
    pool = [m for m in stored if m.source != MarkSource.NARRATION_SEED]
    base = len(pool)
    per_role: dict[str, NarrationRole] = {}
    rows: list[PieceRow] = []
    for walk in all_walks:
        role, duration_ms = walk.record.role, int(walk.record.response.audioDurationMs)
        wav = _wav(run_id, walk.record) if cut_chips and walk.pieces else None
        turns: list[NarrationTurn] = []
        for piece, pin, question in zip(walk.pieces, walk.pins, walk.questions):
            outcome = None
            if question is not None:
                outcome, mark = _outcome(root, question, asks, records, pool, cal, th)
                if mark is not None:
                    pool.append(mark)
            chip, chip_ms = _chip(run_id, role, piece, duration_ms, wav, chip_min)
            turns.append(
                NarrationTurn(
                    turn_id=piece.turn_id,
                    start_ms=piece.turn_ms[0],
                    end_ms=piece.turn_ms[1],
                    text=piece.text,
                    keywords_hit=list(piece.keywords),
                    pinned_panel=pin.panel,
                    pin_seconds=pin.seconds,
                    pin_kind=pin.kind,
                    seed_mark_id=outcome.mark_id if outcome is not None and outcome.status in LINKED else None,
                    chip_path=chip,
                )
            )
            est_ms = [piece.start_ms, piece.end_ms] if piece.estimated else None
            rows.append(PieceRow(role=role, turn_id=piece.turn_id, voice_turn=piece.turn, estimated=piece.estimated, est_ms=est_ms, chip_ms=chip_ms, seed=outcome))
        per_role[role] = _role_doc(turns, walk.record)
        if cut_chips:
            _drop_unnamed_chips(run_id, role, {Path(t.chip_path).name for t in turns if t.chip_path})
    if len(pool) > base or len(stored) != base:
        write_json(marks_path, pool)
    write_json(pieces_json(run_id), rows)
    narration = Narration(per_role=per_role, mode_applied=all(w.applied for w in all_walks))
    write_json(paths.narration_json(run_id), narration)
    return narration


def rebuild_narration(run_id: str, *, th: dict) -> Narration:
    """narration.json, narration_pieces.json, chips and the seed rows of marks.json from the stored answers, no model
    call. Run it after marks.rebuild_marks, which drops the seed rows of the walk it rewrites."""
    return _settle(run_id, th, walks(run_id, th=th), cut_chips=True)


def merge_seed_marks(run_id: str, *, th: dict) -> list[Mark]:
    """The stored seeds merged back into marks.json (no call, no chip cut, nothing before the narration stage ran), for
    a caller that reran marks and cannot run the narration stage. narration.json is rewritten in step, so that caller
    emits narration.derived again."""
    if not paths.narration_json(run_id).is_file():
        return []
    _settle(run_id, th, walks(run_id, th=th), cut_chips=False)
    return [m for m in _read_list(paths.marks_json(run_id), Mark) if m.source == MarkSource.NARRATION_SEED]


def _open_questions(run_id: str, all_walks: list[Walk]) -> list[Question]:
    """Askable questions with no valid stored answer, one per key."""
    asks = _latest_asks(run_id)
    records = {r.request_id: r for r in _read_list(paths.spark_json(run_id), SparkRequestRecord)}
    out: list[Question] = []
    seen: set[tuple[str, str, str]] = set()
    for walk in all_walks:
        for q in walk.questions:
            if q is None or q.reason or q.key in seen:
                continue
            seen.add(q.key)
            ask = asks.get(q.key)
            record = records.get(ask.request_ids[-1]) if ask is not None and ask.request_ids else None
            if record is None or not record.valid:
                out.append(q)
    return out


async def _ask(run_id: str, questions: list[Question], th: dict) -> None:
    """One seed_mark per question (Spark concurrency capped like the marks pass); every attempt appended to
    spark.json, then one seeds.json row per question."""
    root = paths.run_dir(run_id)
    gate = asyncio.Semaphore(int(th["orchestrator"]["spark_semaphore"]))
    effort = str(th["spark"]["effort"])

    def ref(q: Question) -> InputRef:
        assert q.kf is not None and q.kf.crop is not None and q.crop_sha256 is not None  # open questions have a crop
        return InputRef(path=q.kf.crop.path, sha256=q.crop_sha256)

    async def one(q: Question) -> tuple[Question, list[SparkRequestRecord], str | None]:
        async with gate:
            try:
                rgb = await asyncio.to_thread(_read_rgb, root / ref(q).path)
                result = await asyncio.to_thread(spark_client.seed_mark, rgb, q.panel, q.phrase, run_id=run_id, input_ref=ref(q), effort=effort)
            except Exception as exc:  # a missing key or an unreadable crop costs this seed, not the walk's narration
                return q, [], redact(f"{type(exc).__name__}: {exc}")[:300]
        return q, list(result.attempts) or [result.record], None

    answers = await asyncio.gather(*(one(q) for q in questions))
    new_records = [r for _, recs, _ in answers for r in recs]
    if new_records:
        write_json(paths.spark_json(run_id), [*_read_list(paths.spark_json(run_id), SparkRequestRecord), *new_records])
    rows = [
        SeedAsk(role=q.role, turn_id=q.piece.turn_id, panel_id=q.panel, kf_id=q.kf.kf_id if q.kf else "", crop=ref(q), phrase=q.phrase, request_ids=[r.request_id for r in recs], error=error)
        for q, recs, error in answers
    ]
    write_json(seeds_json(run_id), [*_read_list(seeds_json(run_id), SeedAsk), *rows])


def seed_records(run_id: str) -> list[tuple[str, SparkRequestRecord]]:
    """(role, record) for every Spark attempt seeds.json names, in ask order: what the orchestrator emits spark.request
    and spark.response for."""
    records = {r.request_id: r for r in _read_list(paths.spark_json(run_id), SparkRequestRecord)}
    return [(a.role, records[i]) for a in _read_list(seeds_json(run_id), SeedAsk) for i in a.request_ids if i in records]


async def derive_narration(run_id: str, *, th: dict) -> Narration:
    """Pins, chips and seeds for every walk in voice.json. Spark is asked only the seed questions no valid stored
    answer covers; everything is then written by rebuild_narration, so a rebuild gives the same bytes."""
    all_walks = await asyncio.to_thread(walks, run_id, th=th)
    questions = await asyncio.to_thread(_open_questions, run_id, all_walks)
    if questions:
        await _ask(run_id, questions, th)
    return await asyncio.to_thread(rebuild_narration, run_id, th=th)


__all__ = [
    "LINKED",
    "MARK_FORMS",
    "NOT_PANELS",
    "PANEL_FORMS",
    "PIN_FALLBACKS",
    "Piece",
    "PieceRow",
    "Pin",
    "Question",
    "SeedAsk",
    "SeedOutcome",
    "Walk",
    "candidates",
    "chip_span",
    "derive_narration",
    "keywords_heard",
    "merge_seed_marks",
    "mode_applied",
    "pieces_json",
    "pin_piece",
    "rebuild_narration",
    "role_states",
    "role_windows",
    "seed_answer",
    "seed_cap",
    "seed_records",
    "seeds_json",
    "side_words",
    "split_turn",
    "walks",
]
