"""Contract models for every file under runs/<run_id>/ (docs/ARCHITECTURE.md section 4).

Rules every lane follows:
- Derived JSON is written with canonical_json(): sorted keys, floats rounded to 6 places, compact separators,
  ASCII only, no NaN. The same bytes hash the same, so a rebuild is byte-identical.
- Enumerations here are the only spellings of nouns, panel ids, sides, labels, states and kinds.
- Importing this module has no side effects. Nothing here reads a file or calls a model.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_serializer

FLOAT_PLACES = 6

# ----------------------------------------------------------------------------------------------------------------
# Canonical JSON
# ----------------------------------------------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Recursively turn models, enums, paths and floats into JSON-ready plain values."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON cannot hold NaN or infinity")
        rounded = round(value, FLOAT_PLACES)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json", by_alias=True))
    if isinstance(value, dict):
        return {str(_plain(k)): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_plain(v) for v in value]
        return sorted(items) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    """Sorted keys, 6-place floats, compact separators, ASCII. One spelling per value."""
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def json_bytes(value: Any) -> bytes:
    """The exact bytes write_json puts on disk: canonical JSON plus one newline."""
    return (canonical_json(value) + "\n").encode("ascii")


def sha256_json(value: Any) -> str:
    """Hex SHA-256 of json_bytes(value), so a file hash equals the hash of its model."""
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> str:
    """Write json_bytes(value) to path, creating parents. Returns the hex SHA-256 of the bytes written."""
    data = json_bytes(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


M = TypeVar("M", bound="Contract")


def read_json(path: Path, model: type[M]) -> M:
    """Validate the JSON file at path as model."""
    return model.model_validate_json(Path(path).read_text(encoding="utf-8"))


def read_json_list(path: Path, model: type[M]) -> list[M]:
    """Validate a JSON file holding a list of model."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} does not hold a JSON list")
    return [model.model_validate(item) for item in raw]


def chain_hash(prev: str, line_without_hash: dict[str, Any]) -> str:
    """events.ndjson rule: hash = sha256(prev + canonical_json(line without hash))."""
    return hashlib.sha256((prev + canonical_json(line_without_hash)).encode("ascii")).hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# Thresholds (config/thresholds.json)
# ----------------------------------------------------------------------------------------------------------------


def threshold_values(doc: dict[str, Any]) -> dict[str, Any]:
    """Strip the {value, source} leaves of config/thresholds.json down to plain nested values."""
    if isinstance(doc, dict) and set(doc) >= {"value", "source"} and set(doc) <= {"value", "source", "unit", "note"}:
        return doc["value"]
    if isinstance(doc, dict):
        return {k: threshold_values(v) for k, v in doc.items() if not k.startswith("_")}
    return doc


def config_hash(values: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of the plain threshold values; frozen into run.json and every trace."""
    return hashlib.sha256(canonical_json(values).encode("ascii")).hexdigest()


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    """Plain nested threshold values from config/thresholds.json (or path). Called, never run at import."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "thresholds.json"
    return threshold_values(json.loads(Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------------------------------------------------
# Base model and shared aliases
# ----------------------------------------------------------------------------------------------------------------


class Contract(BaseModel):
    """Base for every record: unknown keys are errors, enums stay enums, aliases round-trip."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    def to_json(self) -> str:
        return canonical_json(self)

    @classmethod
    def from_json(cls: type[M], text: str) -> M:
        return cls.model_validate_json(text)

    def digest(self) -> str:
        """Hex SHA-256 of json_bytes(self); the hash a file written with write_json would have."""
        return sha256_json(self)


class Verbatim(Contract):
    """A record that stores an API answer as it came: unknown keys are kept."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


Box4 = Annotated[list[float], Field(min_length=4, max_length=4, description="x1, y1, x2, y2; right and bottom half-open")]
IntBox4 = Annotated[list[int], Field(min_length=4, max_length=4, description="integer x1, y1, x2, y2; half-open")]
Point2 = Annotated[list[float], Field(min_length=2, max_length=2, description="x, y")]
Size2 = Annotated[list[int], Field(min_length=2, max_length=2, description="width, height after rotation")]
Window = Annotated[list[float], Field(min_length=2, max_length=2, description="t0, t1 seconds")]
Timestamp = Annotated[str, Field(description="ISO 8601 UTC, e.g. 2026-10-27T15:04:05Z")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Role = Annotated[str, Field(pattern=r"^(pickup|return|topup_[0-9]+|claim_[0-9]+)$")]
WalkRole = Literal["pickup", "return"]
CAMERA_TIME_NOT_PRESENT = "camera time not present"


# ----------------------------------------------------------------------------------------------------------------
# Enumerations
# ----------------------------------------------------------------------------------------------------------------


class Noun(str, Enum):
    """SAM concept nouns. The first 16 are the video lanes of section 5 step 5, in request order."""

    CAR = "car"
    LICENSE_PLATE = "license plate"
    FACE = "face"
    CAR_DOOR = "car door"
    BUMPER = "bumper"
    WHEEL = "wheel"
    HOOD = "hood"
    TRUNK_LID = "trunk lid"
    FENDER = "fender"
    ROOF = "roof"
    HEADLIGHT = "headlight"
    TAIL_LIGHT = "tail light"
    SIDE_MIRROR = "side mirror"
    WINDSHIELD = "windshield"
    REAR_WINDOW = "rear window"
    CAR_WINDOW = "car window"
    DOOR_HANDLE = "door handle"  # image requests on door keyframes and one D1 video request (E8), never a lane


LANE_NOUNS: tuple[Noun, ...] = (
    Noun.CAR,
    Noun.LICENSE_PLATE,
    Noun.FACE,
    Noun.CAR_DOOR,
    Noun.BUMPER,
    Noun.WHEEL,
    Noun.HOOD,
    Noun.TRUNK_LID,
    Noun.FENDER,
    Noun.ROOF,
    Noun.HEADLIGHT,
    Noun.TAIL_LIGHT,
    Noun.SIDE_MIRROR,
    Noun.WINDSHIELD,
    Noun.REAR_WINDOW,
    Noun.CAR_WINDOW,
)
PRIVACY_NOUNS: tuple[Noun, ...] = (Noun.LICENSE_PLATE, Noun.FACE)
PANEL_NOUNS: tuple[Noun, ...] = tuple(n for n in LANE_NOUNS if n not in PRIVACY_NOUNS)  # the 14 of section 16 item 3
SMALL_NOUNS: tuple[Noun, ...] = (Noun.SIDE_MIRROR, Noun.HEADLIGHT, Noun.TAIL_LIGHT, Noun.LICENSE_PLATE, Noun.FACE)
TOPUP_ALWAYS_NOUNS: tuple[Noun, ...] = (Noun.CAR, Noun.LICENSE_PLATE, Noun.FACE)


def noun_slug(noun: Noun | str) -> str:
    """File-safe lane id: 'license plate' -> 'license_plate'."""
    text = noun.value if isinstance(noun, Noun) else str(noun)
    return "_".join(text.strip().lower().split())


class PanelId(str, Enum):
    """Canonical slots of derived/panels.json. 13 body panels, 4 wheels, 5 light and glass groups."""

    LEFT_FRONT_DOOR = "left_front_door"
    LEFT_REAR_DOOR = "left_rear_door"
    RIGHT_FRONT_DOOR = "right_front_door"
    RIGHT_REAR_DOOR = "right_rear_door"
    LEFT_FRONT_FENDER = "left_front_fender"
    LEFT_REAR_FENDER = "left_rear_fender"
    RIGHT_FRONT_FENDER = "right_front_fender"
    RIGHT_REAR_FENDER = "right_rear_fender"
    HOOD = "hood"
    ROOF = "roof"
    TRUNK_LID = "trunk_lid"
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    LEFT_FRONT_WHEEL = "left_front_wheel"
    LEFT_REAR_WHEEL = "left_rear_wheel"
    RIGHT_FRONT_WHEEL = "right_front_wheel"
    RIGHT_REAR_WHEEL = "right_rear_wheel"
    HEADLIGHTS = "headlights"
    TAIL_LIGHTS = "tail_lights"
    MIRRORS = "mirrors"
    WINDSHIELD = "windshield"
    REAR_WINDOW = "rear_window"


BODY_PANELS: tuple[PanelId, ...] = (
    PanelId.LEFT_FRONT_DOOR,
    PanelId.LEFT_REAR_DOOR,
    PanelId.RIGHT_FRONT_DOOR,
    PanelId.RIGHT_REAR_DOOR,
    PanelId.LEFT_FRONT_FENDER,
    PanelId.LEFT_REAR_FENDER,
    PanelId.RIGHT_FRONT_FENDER,
    PanelId.RIGHT_REAR_FENDER,
    PanelId.HOOD,
    PanelId.ROOF,
    PanelId.TRUNK_LID,
    PanelId.FRONT_BUMPER,
    PanelId.REAR_BUMPER,
)
WHEEL_PANELS: tuple[PanelId, ...] = (
    PanelId.LEFT_FRONT_WHEEL,
    PanelId.LEFT_REAR_WHEEL,
    PanelId.RIGHT_FRONT_WHEEL,
    PanelId.RIGHT_REAR_WHEEL,
)
LIGHT_AND_GLASS_PANELS: tuple[PanelId, ...] = (
    PanelId.HEADLIGHTS,
    PanelId.TAIL_LIGHTS,
    PanelId.MIRRORS,
    PanelId.WINDSHIELD,
    PanelId.REAR_WINDOW,
)
LOWER_BUMPERS: tuple[PanelId, ...] = (PanelId.FRONT_BUMPER, PanelId.REAR_BUMPER)

PANEL_NOUN: dict[PanelId, Noun] = {
    **{p: Noun.CAR_DOOR for p in BODY_PANELS if p.value.endswith("_door")},
    **{p: Noun.FENDER for p in BODY_PANELS if p.value.endswith("_fender")},
    PanelId.HOOD: Noun.HOOD,
    PanelId.ROOF: Noun.ROOF,
    PanelId.TRUNK_LID: Noun.TRUNK_LID,
    PanelId.FRONT_BUMPER: Noun.BUMPER,
    PanelId.REAR_BUMPER: Noun.BUMPER,
    **{p: Noun.WHEEL for p in WHEEL_PANELS},
    PanelId.HEADLIGHTS: Noun.HEADLIGHT,
    PanelId.TAIL_LIGHTS: Noun.TAIL_LIGHT,
    PanelId.MIRRORS: Noun.SIDE_MIRROR,
    PanelId.WINDSHIELD: Noun.WINDSHIELD,
    PanelId.REAR_WINDOW: Noun.REAR_WINDOW,
}


class Side(str, Enum):
    LEFT = "left"  # US driver side
    RIGHT = "right"
    NONE = "none"


class Position(str, Enum):
    FRONT = "front"
    REAR = "rear"
    NONE = "none"


class RunMode(str, Enum):
    CAR = "car"
    APARTMENT = "apartment"


class RunKind(str, Enum):
    PAIR = "pair"
    PICKUP_CHECK = "pickup_check"
    TOPUP = "topup"
    CLAIM = "claim"


class RunStatus(str, Enum):
    CREATED = "created"
    SEALING = "sealing"
    SEALED = "sealed"
    CONVERTING = "converting"
    TRACKING = "tracking"
    BLURRING = "blurring"
    MARKING = "marking"
    STAYSPUT = "staysput"
    MATCHING = "matching"
    SHEET = "sheet"
    DONE = "done"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    PARTIAL = "partial"


class LiveOrReplay(str, Enum):
    LIVE = "live"
    REPLAY = "replay"


class ResponsesStore(str, Enum):
    FALSE_ACCEPTED = "false_accepted"
    DELETED_PER_REQUEST = "deleted_per_request"
    UNKNOWN = "unknown"


class SamEngine(str, Enum):
    META_MODEL_API = "meta-model-api"
    FAL = "fal"
    MODAL = "modal"


class SparkEngine(str, Enum):
    META_MODEL_API = "meta-model-api"
    BASETEN = "baseten"


class LaneStatus(str, Enum):
    """completed: records came; zero_matches: completed with no delta (a valid empty lane, coverage not_filmed);
    incomplete: ended before response.completed or parser error diagnostics (G7); failed: no answer after retries."""

    COMPLETED = "completed"
    ZERO_MATCHES = "zero_matches"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class MaskEncoding(str, Enum):
    LOSSLESS = "lossless"  # payload marker ~
    ONE_BIT = "one_bit"  # payload marker !


class SparkPurpose(str, Enum):
    MARKS = "marks"
    COUNT = "count"
    SECOND_LOOK = "second_look"
    CLAIM_NAME = "claim_name"
    CLAIM_BOX = "claim_box"
    SEED = "seed"
    TIMELINE = "timeline"
    NOTE_PREFILL = "note_prefill"


class SparkStatus(str, Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


class ReasoningEffort(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class VoiceMode(str, Enum):
    PUSH_TO_TALK = "PUSH_TO_TALK"
    ENDPOINTING = "ENDPOINTING"
    DIARIZATION = "DIARIZATION"


class MarkKind(str, Enum):
    """marks_v1 type enum (section 5.7). dirt, reflection and unsure are capped by K4."""

    SCRATCH = "scratch"
    SCUFF = "scuff"
    CHIP = "chip"
    DENT = "dent"
    CRACK = "crack"
    DIRT = "dirt"
    REFLECTION = "reflection"
    UNSURE = "unsure"


COMPATIBLE_MARK_KINDS: frozenset[MarkKind] = frozenset({MarkKind.SCRATCH, MarkKind.SCUFF, MarkKind.CHIP})
CAPPED_MARK_KINDS: frozenset[MarkKind] = frozenset({MarkKind.DIRT, MarkKind.REFLECTION, MarkKind.UNSURE})


class Confidence(str, Enum):
    """An enum so no sheet prints a model number as a probability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SizeClass(str, Enum):
    S = "S"
    M = "M"
    L = "L"


class MarkSource(str, Enum):
    SPARK_BOX = "spark_box"
    SPARK_COUNT = "spark_count"
    NARRATION_SEED = "narration_seed"


class MarksStatus(str, Enum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    NOT_RUN = "not_run"


class LookVerdict(str, Enum):
    """look_v1: the second look can only lower (different or unsure become K7)."""

    SAME = "same"
    DIFFERENT = "different"
    UNSURE = "unsure"


class CoverageState(str, Enum):
    FILMED = "filmed"
    GLIMPSED = "glimpsed"
    NOT_FILMED = "not_filmed"
    NOT_RUN = "not_run"
    NOT_SEEN = "not_seen"


class StateBasis(str, Enum):
    C = "c"
    AREA_ONLY = "area_only"


class InstanceFrom(str, Enum):
    TRACKS = "tracks"
    CANONICAL = "canonical"


class MergeRule(str, Enum):
    VELOCITY_GAP = "velocity_gap"  # 5.1 item 4
    ANCHOR_ABSORB = "anchor_absorb"  # 5.1 item 6
    MOMENT_ORIENTATION = "moment_orientation"  # STRATEGY 4.3, keyframes only


class VerdictLabel(str, Enum):
    ALREADY_THERE = "already_there"
    NEEDS_YOUR_EYE = "needs_your_eye"
    NEW_OR_UNSEEN = "new_or_unseen"
    NOT_SEEN_AT_RETURN = "not_seen_at_return"
    NOT_COMPARABLE = "not_comparable"


class MatchState(str, Enum):
    COMPARABLE = "comparable"
    NOT_COMPARABLE = "not_comparable"


class FrameUsed(str, Enum):
    MOMENT = "moment"
    BUMPER_ARC = "bumper_arc"
    HOOD_AXIS = "hood_axis"
    WHEEL = "wheel"


class StaysPutVerdict(str, Enum):
    FIXED = "fixed"
    SLIDING = "sliding"
    INCONCLUSIVE = "inconclusive"
    NO_TRACK = "no_track"
    NOT_RUN = "not_run"


class PinKind(str, Enum):
    OVERLAP = "overlap"
    ABOUT = "about"


class ClaimExit(str, Enum):
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    E6 = "E6"
    E7 = "E7"
    E8 = "E8"


class RoiSource(str, Enum):
    DRAWN = "drawn"
    WHOLE = "whole"


class PlacementMethod(str, Enum):
    FULL_PANEL = "full_panel"
    LANDMARK_FIT = "landmark_fit"
    NONE = "none"


class RulerKind(str, Enum):
    PLATE = "plate"
    WHEEL = "wheel"


class ThresholdType(str, Enum):
    """Which Hertz example threshold (REV 7.2025, illustrative only) the size line compares against."""

    BODY_SCRATCH = "body_scratch"  # 3.5 in long axis (the 2 x 3.5 in card)
    DENT = "dent"  # 1 in diameter
    WHEEL_MARK = "wheel_mark"  # 1 in
    BODY_SCUFF = "body_scuff"  # ordinary wear sentence, no size comparison


class DeletionKind(str, Enum):
    FILE = "file"
    FILES_API = "files_api"
    RESPONSE = "response"
    QUARANTINE = "quarantine"


class LedgerKind(str, Enum):
    """ledger.record kinds; cost_for in ledger.py prices the first four and spark_open (Muse Spark Open on Baseten, at
    Baseten's list price: the calls run on Baseten credits, not Meta API spend)."""

    SAM_VIDEO = "sam_video"
    SAM_IMAGE = "sam_image"
    SPARK = "spark"
    VOICE = "voice"
    SPARK_OPEN = "spark_open"
    BASETEN = "baseten"
    FAL = "fal"


class LedgerUnit(str, Enum):
    FRAMES = "frames"
    TOKENS = "tokens"
    IMAGES = "images"
    SECONDS = "seconds"


class ReplayMode(str, Enum):
    FIXED = "fixed"
    RECORDED = "recorded"
    INSTANT = "instant"


class EventKind(str, Enum):
    """events.ndjson kinds (section 4). The reducer in web/ switches on these exact strings."""

    RUN_CREATED = "run.created"
    UPLOAD_STARTED = "upload.started"
    UPLOAD_DONE = "upload.done"
    SEAL_MANIFEST = "seal.manifest"
    SEAL_OTS = "seal.ots"
    SEAL_TSR = "seal.tsr"
    MEDIA_PROBED = "media.probed"
    MEDIA_PROXY_DONE = "media.proxy_done"
    FILES_UPLOADED = "files.uploaded"
    SAM_REQUEST = "sam.request"
    SAM_FIRST_DELTA = "sam.first_delta"
    SAM_FRAME = "sam.frame"
    SAM_LANE_DONE = "sam.lane_done"
    SAM_FAILED = "sam.failed"
    BLUR_DONE = "blur.done"
    PANELS_DERIVED = "panels.derived"
    COVERAGE_UPDATED = "coverage.updated"
    TOPUP_REQUESTED = "topup.requested"
    SPARK_REQUEST = "spark.request"
    SPARK_RESPONSE = "spark.response"
    MARKS_DERIVED = "marks.derived"
    VOICE_DONE = "voice.done"
    NARRATION_DERIVED = "narration.derived"
    STAYSPUT_DERIVED = "staysput.derived"
    MATCH_DERIVED = "match.derived"
    CLAIM_DERIVED = "claim.derived"
    SHEET_WRITTEN = "sheet.written"
    FILES_DELETED = "files.deleted"
    RESPONSES_DELETED = "responses.deleted"
    QUARANTINE_DELETED = "quarantine.deleted"
    LEDGER_ROW = "ledger.row"
    STAGE = "stage"
    RUN_PARTIAL = "run.partial"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"


# ----------------------------------------------------------------------------------------------------------------
# Clocks, run.json, videos/<role>.json, bundle/manifest.json, events
# ----------------------------------------------------------------------------------------------------------------


class ThreeClocks(Contract):
    """The three clocks of page 1: sealed at, first model call at, sealed N minutes after the camera time."""

    sealed_at: Timestamp | None = None
    first_model_call_at: Timestamp | None = None
    sealed_minutes_after_camera: int | Literal["camera time not present"] | None = None
    camera_time: Timestamp | None = None  # ffprobe creation_time of the pickup original, when present


class DeletionClocks(Contract):
    """When the Files API upload, the stored responses and the quarantine were deleted."""

    files_deleted_at: Timestamp | None = None
    responses_deleted_at: Timestamp | None = None
    quarantine_deleted_at: Timestamp | None = None


class RunError(Contract):
    stage: str
    text: str


class RunCounts(Contract):
    panels: int = 0
    marks: int = 0
    matches: int = 0


class RunManifest(Contract):
    """run.json. Timestamps are allowed here; it is not a derived file."""

    run_id: str
    created_at: Timestamp
    mode: RunMode = RunMode.CAR
    kind: RunKind = RunKind.PAIR
    status: RunStatus = RunStatus.CREATED
    live_or_replay: LiveOrReplay = LiveOrReplay.LIVE
    keep: bool = False  # sample and eval runs survive the sweeper
    keep_plate: bool = False  # the renter's explicit choice at upload (section 8)
    footage_credit: str | None = None  # a sample run's footage credit, shown under the walk player
    tire_code: str | None = None  # walkaround run --tire-code, normalized (225/60R16); the wheel ruler reads it (polish lane)
    app_commit: str | None = None
    config_hash: str | None = None
    thresholds: dict[str, Any] = Field(default_factory=dict)  # the whole plain dict, frozen at creation
    fps: float | None = None
    noun_list: list[Noun] = Field(default_factory=lambda: list(LANE_NOUNS))
    # Test clips only: the text a lane is sent with instead of its noun ({"hood": "person"}); the lane keeps its
    # noun's slot, file names and rules. Empty for every real walk.
    prompts: dict[str, str] = Field(default_factory=dict)
    retention_deadline: Timestamp | None = None
    api_spend_cents: int = 0
    spend_ceiling_cents: int = 800
    clocks: ThreeClocks = Field(default_factory=ThreeClocks)
    responses_store: ResponsesStore = ResponsesStore.UNKNOWN
    sam_engine: SamEngine = SamEngine.META_MODEL_API
    events_head_hash: str | None = None
    counts: RunCounts = Field(default_factory=RunCounts)
    errors: list[RunError] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _omit_absent_tire_code(self, handler: Any) -> Any:
        """A run with no tire code typed serializes as it did before the field existed (10/31)."""
        data = handler(self)
        if isinstance(data, dict) and data.get("tire_code") is None:
            data.pop("tire_code", None)
        return data


class FfprobeInfo(Contract):
    codec: str
    container: str
    width: int  # stored size
    height: int
    rotation: int = 0
    display_width: int  # after rotation; the w every SAM prompt token carries
    display_height: int
    fps_avg: float
    vfr: bool = False
    duration_s: float
    creation_time: Timestamp | Literal["not present"] = "not present"
    device: str | None = None
    hdr: bool = False
    has_audio: bool = True


class ProxyRecord(Contract):
    """The H.264 proxy SAM sees: longest edge <= max_edge, constant fps, rotation baked in, no audio."""

    path: str
    fps: float
    width: int
    height: int
    frames: int
    pts_map_path: str  # JSON list: proxy frame index -> source pts seconds
    sha256: Sha256 | None = None


class BlurredProxyRecord(Contract):
    path: str
    sha256: Sha256 | None = None


class WavRecord(Contract):
    path: str
    duration_s: float
    sha256: Sha256 | None = None


class FilesApiRecord(Contract):
    file_id: str
    uploaded_at: Timestamp
    expires_at: Timestamp | None = None
    deleted_at: Timestamp | None = None
    delete_response: dict[str, Any] | None = None  # the DELETE body verbatim (deleted true, section API notes)


class VideoRecord(Contract):
    """videos/<role>.json."""

    role: Role
    original_filename: str
    bytes: int
    sha256_browser: Sha256 | None = None
    sha256_server: Sha256 | None = None
    hash_match: bool | None = None
    received_at: Timestamp | None = None
    chunks_expected: int = 0
    chunks_received: int = 0
    ffprobe: FfprobeInfo | None = None
    proxy: ProxyRecord | None = None
    proxy_blurred: BlurredProxyRecord | None = None
    wav: WavRecord | None = None
    files_api: FilesApiRecord | None = None
    quarantine_path: str | None = None
    quarantine_deleted_at: Timestamp | None = None


class ManifestEntry(Contract):
    path: str
    sha256: Sha256
    bytes: int


class Stamps(Contract):
    manifest_ots: str | None = None  # pending | confirmed:<block> | failed:<reason>
    pickup_ots: str | None = None
    return_ots: str | None = None
    tsr_time: Timestamp | None = None
    tsa_serial: str | None = None
    tsa_issuer: str | None = None  # freetsa, or DigiCert as the fallback (section 14 item 24)


class SealManifest(Contract):
    """bundle/manifest.json at seal time; manifest_run.json adds the run-end fields."""

    run_id: str
    files: list[ManifestEntry]
    stamps: Stamps = Field(default_factory=Stamps)
    deletions: list[DeletionRecord] = Field(default_factory=list)
    events_head_hash: str | None = None
    config_hash: str | None = None
    app_commit: str | None = None
    pdf_sha256: Sha256 | None = None


class Event(Contract):
    """One events.ndjson line. prev of seq 0 is the sha256 of manifest.json."""

    seq: int
    ts: Timestamp
    run_id: str
    kind: EventKind
    payload: dict[str, Any] = Field(default_factory=dict)
    prev: str
    hash: str

    def line_without_hash(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"hash"})

    def check(self) -> bool:
        return chain_hash(self.prev, self.line_without_hash()) == self.hash


# ----------------------------------------------------------------------------------------------------------------
# SAM lanes: summary, index, diagnostics
# ----------------------------------------------------------------------------------------------------------------


class Diagnostic(Contract):
    severity: Literal["warning", "error"]
    code: str
    message: str
    line: int | None = None


class MaskRef(Contract):
    """Where one mask's payload sits in the lane, so a decode never re-parses the stream.

    payload_offset counts characters into the lane's output text (the deltas concatenated in order; ASCII,
    so chars equal bytes) and points at the encoding marker (~ or !). payload_len runs to the closing |>.
    """

    x: int = 0
    y: int = 0
    h: int
    w: int
    encoding: MaskEncoding
    payload_offset: int
    payload_len: int


class IndexObject(Contract):
    id: str
    box: IntBox4
    c: float | None = None  # None means absent, never zero
    mask: MaskRef | None = None


class IndexFrame(Contract):
    frame: int  # 0 for an image
    objects: list[IndexObject]


class LaneRecord(Contract):
    """sam/<role>/<lane_id>.summary.json: one SAM concept or object-prompt request."""

    lane_id: str  # noun_slug(noun) or the prompt id
    role: Role
    kind: Literal["text", "object"] = "text"
    media: Literal["video", "image"] = "video"
    noun: Noun | None = None  # None for object prompts
    prompt_text: str
    media_size: Size2
    params: dict[str, str] = Field(default_factory=dict)  # the metadata sent
    store: bool = False
    store_echo: bool | None = None  # what response.created said
    request_id: str | None = None  # the response id
    request_index: int = 0
    status: LaneStatus = LaneStatus.COMPLETED
    lane_path: str | None = None  # the gz ndjson of raw SSE events
    lane_sha256: Sha256 | None = None
    index_path: str | None = None
    upload_s: float | None = None
    first_delta_s: float | None = None
    completed_s: float | None = None
    frames_sent: int = 0
    frames_returned: int = 0
    video_frames_processed: int | None = None  # usage.video_frames_processed via model_dump()
    ids: list[str] = Field(default_factory=list)
    zero_matches: bool = False
    parse_cpu_s: float = 0.0
    masks_decoded: int = 0
    decode_cpu_s: float = 0.0
    dollars: float = 0.0
    retries: int = 0
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    error: str | None = None


# ----------------------------------------------------------------------------------------------------------------
# Spark and Voice records (verbatim model-era facts)
# ----------------------------------------------------------------------------------------------------------------


class InputRef(Contract):
    path: str
    sha256: Sha256


class SparkUsage(Contract):
    input: int = 0
    output: int = 0
    reasoning: int = 0
    cached: int = 0


class SparkRequestRecord(Contract):
    """One spark.json entry. raw_text is stored before parsing; the call cost money even if the JSON is bad."""

    request_id: str
    purpose: SparkPurpose
    model: str
    engine: SparkEngine = SparkEngine.META_MODEL_API
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW
    max_output_tokens: int = 4000
    store: bool = False
    input_refs: list[InputRef] = Field(default_factory=list)
    schema_name: str
    raw_text: str = ""
    parsed: dict[str, Any] | None = None
    valid: bool = False
    status: SparkStatus = SparkStatus.COMPLETED
    incomplete_reason: str | None = None
    truncated: bool = False
    retry: Literal[0, 1] = 0
    usage: SparkUsage = Field(default_factory=SparkUsage)
    dollars: float = 0.0
    ms: int = 0


class VoiceRequest(Contract):
    mode: VoiceMode
    model: str
    audioEncoding: str = "WAV"
    keywords: list[str] = Field(default_factory=list)
    languageBias: list[str] | None = None


class VoiceTurn(Verbatim):
    turnId: int | str | None = None
    startMs: int = 0
    endMs: int = 0
    transcript: str = ""
    speaker: str | None = None


class VoiceResponse(Verbatim):
    sessionId: str | None = None
    transcript: str = ""
    audioDurationMs: int = 0
    turns: list[VoiceTurn] = Field(default_factory=list)


class VoiceRecord(Contract):
    """voice.json: request part and response verbatim. mode_applied is false when ENDPOINTING was sent and
    turns is empty while transcript is not (ledger flag 'mode not applied', no chips)."""

    role: Role
    request: VoiceRequest
    response: VoiceResponse
    wav_sha256: Sha256 | None = None
    mode_applied: bool = True
    dollars: float = 0.0


# ----------------------------------------------------------------------------------------------------------------
# derived/tracks.json, panels.json, keyframes.json
# ----------------------------------------------------------------------------------------------------------------


class TrackFragment(Contract):
    object_id: str
    first_frame: int
    last_frame: int


class MergeRecord(Contract):
    rule: MergeRule
    gap_s: float
    pred_dist: float | None = None  # in box widths
    area_ratio: float | None = None
    angle: float | None = None  # degrees, moment orientation test only


class AnchorReprompt(Contract):
    request_id: str
    frame: int
    box: IntBox4
    negative_points: list[Point2] = Field(default_factory=list)


class FrameObs(Contract):
    """One per_frame row of a track: boxes and c only; a mask is decoded only where mask_decoded is true."""

    frame: int
    pts: float
    box: Box4
    box_area_frac: float
    in_car_box: bool
    c: float | None = None
    border_frac: float = 0.0  # share of the box perimeter on the frame edge
    mask_decoded: bool = False
    area_frac: float | None = None
    complete: bool | None = None


class DoorCheck(Contract):
    """The door check (11/1, thresholds panels.door_check): which side and which door (front or rear) a car door track
    is, from cues the walk already tracked. votes: side cues lid_side and mirror_side are +1 right, -1 left; position
    cues mirror, wheel, pair and walk are +1 front, -1 rear; 0 is silent. numbers holds what each vote was measured
    from. was is the slot the side vote gave; side_decided or position_decided false means the cues disagreed, so the
    track keeps a placeholder slot and its panel is side_uncertain (G6)."""

    was: PanelId | None = None
    votes: dict[str, int] = Field(default_factory=dict)
    numbers: dict[str, float] = Field(default_factory=dict)
    side_score: float = 0.0
    position_score: float = 0.0
    side_decided: bool = True
    position_decided: bool = True


class TwinCheck(Contract):
    """The twin check (panels lane, 11/1): which end of the car a hood or trunk-lid track's lid is on, and for a light
    or glass track the end of the placed lid it sits beside. votes are +1 for the front, -1 for the rear and 0 when a
    cue is silent (plate, mirror, lights, glass; next_to_lid for lights and glass); numbers holds what each vote was
    measured from. decided false (lids only) means the evidence was split: the track keeps its lane's slot as a
    placeholder and its panel is side_uncertain (G6)."""

    lid: list[str]  # the lid tracks judged to show one lid (box IoU on shared frames), or for a light or glass track the lid it sat beside
    votes: dict[str, int] = Field(default_factory=dict)
    numbers: dict[str, float] = Field(default_factory=dict)
    score: float = 0.0  # the weighted sum of votes
    decided: bool = True


class PartCheck(Contract):
    """The side check of a wheel or fender-lane track and the end check of a glass track (lexi run 1, 11/2, thresholds
    panels.part_check), in the style of DoorCheck. kind is "side" (votes +1 right, -1 left: lid_side, light_side,
    door_side) or "end" (votes +1 front, -1 rear: lights, bumper, lid, mirror). was is the slot before the check;
    decided false means the cues disagreed with the slot the vote gave and did not reach the margin, so the track keeps a
    placeholder slot and its panel is side_uncertain (G6)."""

    kind: Literal["side", "end"]
    was: PanelId | None = None
    votes: dict[str, int] = Field(default_factory=dict)
    numbers: dict[str, float] = Field(default_factory=dict)
    score: float = 0.0
    decided: bool = True


class PanelTrack(Contract):
    """One derived/tracks.json entry: a merged track with its canonical panel id and side once voted."""

    track_id: str
    role: Role
    noun: Noun
    sam_object_ids: list[str]
    fragments: list[TrackFragment]
    merges: list[MergeRecord] = Field(default_factory=list)
    anchor_reprompts: list[AnchorReprompt] = Field(default_factory=list)
    c_present_share: float = 0.0
    per_frame: list[FrameObs]
    panel_id: PanelId | None = None  # filled by the side vote; None until then or for car, car window, plate, face
    side: Side = Side.NONE
    position: Position = Position.NONE
    twin_check: TwinCheck | None = None  # lid, light and glass tracks the twin check placed (thresholds panels.twin_check)
    door_check: DoorCheck | None = None  # door tracks the door check judged (thresholds panels.door_check)
    edge_on_frames: list[int] = Field(default_factory=list)  # keyframe candidates under panels.edge_on_floor
    part_check: PartCheck | None = None  # wheel, fender-lane and glass tracks the part check judged (panels.part_check)

    @model_serializer(mode="wrap")
    def _omit_absent_twin_check(self, handler: Any) -> Any:
        """A track the twin check, the door check or the edge-on floor did not touch serializes as it did before those
        fields existed (11/1)."""
        data = handler(self)
        if isinstance(data, dict):
            for key in ("twin_check", "door_check", "part_check"):
                if data.get(key) is None:
                    data.pop(key, None)
            if not data.get("edge_on_frames"):
                data.pop("edge_on_frames", None)
        return data


class SideVotes(Contract):
    R1: float = 0.0
    R2: float = 0.0
    R3: float = 0.0
    R4: float = 0.0


class PanelRole(Contract):
    track_ids: list[str] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    seconds_visible: float = 0.0
    state: CoverageState = CoverageState.NOT_SEEN
    state_basis: StateBasis = StateBasis.C
    reason: str = ""
    keyframe_ids: list[str] = Field(default_factory=list)


class Panel(Contract):
    """One derived/panels.json entry."""

    panel_id: PanelId
    noun: Noun
    side: Side = Side.NONE
    position: Position = Position.NONE
    instance_from: InstanceFrom = InstanceFrom.CANONICAL
    votes: SideVotes = Field(default_factory=SideVotes)
    side_uncertain: bool = False
    front_sign: Literal[1, -1] | None = None
    per_role: dict[Role, PanelRole] = Field(default_factory=dict)
    comparable: bool = False
    not_comparable_reasons: list[str] = Field(default_factory=list)  # rule ids G1..G10


class KeyframeScore(Contract):
    area: float
    completeness: float
    sharpness: float
    glare: float  # glare_frac; the factor is (1 - glare)
    frontality: float
    total: float


class Crop(Contract):
    """A blurred crop cut from the source frame. crop pixel (u, v) maps to source (dx + u / scale, dy + v / scale);
    the crop covers source box [dx, dy, dx + w / scale, dy + h / scale]."""

    path: str
    sha256: Sha256 | None = None
    w: int
    h: int
    scale: float
    dx: float
    dy: float


class Moments(Contract):
    """Centroid and central moments of a panel mask in crop pixels, plus the whitening matrix (section 5.5)."""

    cx: float
    cy: float
    mu20: float
    mu02: float
    mu11: float
    theta: float  # 0.5 atan2(2 mu11, mu20 - mu02), radians
    sigma_inv_half: Annotated[list[Annotated[list[float], Field(min_length=2, max_length=2)]], Field(min_length=2, max_length=2)]


class BlurRecord(Contract):
    plate_boxes: list[Box4] = Field(default_factory=list)
    face_boxes: list[Box4] = Field(default_factory=list)
    request_ids: list[str] = Field(default_factory=list)
    plate_ruler_px: float | None = None  # long side of the plate mask's min-area rectangle on the unblurred frame
    plate_ruler_c: float | None = None
    plate_tilt_hw: float | None = None  # apparent h / w
    _blurred: Any = PrivateAttr(default=None)  # privacy.mask_blur (R27): bool mask of the pixels blurred; never serialized

    def set_blurred(self, mask: Any) -> None:
        self._blurred = mask

    def blurred_mask(self) -> Any:
        return self._blurred


class HandleRecord(Contract):
    point_src: Point2 | None = None
    c: float | None = None
    request_id: str | None = None


class Conditions(Contract):
    wet: bool = False
    dirty: bool = False
    glare: bool = False
    dark: bool = False

    def any_flag(self) -> bool:
        return self.wet or self.dirty or self.glare or self.dark


class Keyframe(Contract):
    """One derived/keyframes.json entry (frozen model-era facts plus the crop and moments)."""

    kf_id: str
    role: Role
    panel_id: PanelId
    track_id: str
    f_proxy: int
    pts: float
    score: KeyframeScore
    aspect_ratio: float
    box_src: Box4
    crop: Crop | None = None
    mask_png: str | None = None  # one-bit PNG in crop pixels
    moments: Moments | None = None
    front_sign: Literal[1, -1] | None = None
    window_subtracted: bool = False
    blur: BlurRecord = Field(default_factory=BlurRecord)
    handle: HandleRecord = Field(default_factory=HandleRecord)
    conditions: Conditions = Field(default_factory=Conditions)
    marks_status: MarksStatus = MarksStatus.NOT_RUN
    privacy_blurred_frac: float | None = None  # share of the crop's panel mask blurred for privacy (mask_blur, R27)
    blur_png: str | None = None  # one-bit PNG in crop pixels of what was blurred, when anything was (R27)

    @model_serializer(mode="wrap")
    def _omit_absent_privacy(self, handler: Any) -> Any:
        """A keyframe made before R27 serializes as it did."""
        data = handler(self)
        if isinstance(data, dict):
            for key in ("privacy_blurred_frac", "blur_png"):
                if data.get(key) is None:
                    data.pop(key, None)
        return data


# ----------------------------------------------------------------------------------------------------------------
# Marks (Spark grid output, derived marks, whitened marks)
# ----------------------------------------------------------------------------------------------------------------


class GridPoint(Contract):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class GridBox(Contract):
    x_min: int = Field(ge=0, le=1000)
    y_min: int = Field(ge=0, le=1000)
    x_max: int = Field(ge=0, le=1000)
    y_max: int = Field(ge=0, le=1000)


class Landmark(Contract):
    name: str  # from the panel's closed list in thresholds spark.landmarks
    point: GridPoint


class SparkMark(Contract):
    """One marks_v1 mark as Spark emitted it, after the 1001 clamp, in the 0..1000 grid of the sent image."""

    type: MarkKind
    point: GridPoint
    bbox: GridBox
    size_class: SizeClass
    confidence: Confidence
    note: str = ""


class SparkMarksAnswer(Contract):
    """The parsed marks_v1 object."""

    marks: list[SparkMark] = Field(default_factory=list)
    conditions: Conditions = Field(default_factory=Conditions)
    landmarks: list[Landmark] = Field(default_factory=list)
    neighbour_car_visible: bool = False


class BoxRefine(Contract):
    """How a Spark box mark got the box it uses (thresholds dedup.sam_box_refine, box_fix 11/1): SAM 3.1's tight box
    of its mask when that mask is small and sits on Spark's box, else Spark's own box with the reason. Spark's box and
    point are kept here, so both boxes stay on the row; the mark's bbox_crop, point_crop, bbox_src, point_src and
    size_px_src are those of the box used."""

    used: Literal["sam", "spark"]
    spark_bbox_crop: Box4
    spark_point_crop: Point2 | None = None
    sam_bbox_crop: Box4 | None = None  # the tight box of SAM's mask, when SAM answered with one
    sam_c: float | None = None
    mask_px: int | None = None
    inside: float | None = None  # share of SAM's tight box inside Spark's box grown by the margin
    area_ratio: float | None = None  # SAM's tight box area over Spark's box area
    sam_request_id: str | None = None
    note: str = ""  # why Spark's box was kept, in words; empty when SAM's box is used


class ZoomRefine(Contract):
    """How a Spark box mark was moved by the zoomed look (thresholds dedup.spark_zoom_refine, zoom refine 11/1): Spark's
    box, same size, moved so its centre is the centre of the zoom box nearest it, or Spark's box with the reason.
    Spark's box and point and the zoom's own box are kept here; the mark's bbox_crop, point_crop, bbox_src, point_src
    and size_px_src are those of the box used (before any SAM tighten, which then starts from the moved box)."""

    used: Literal["zoom", "spark"]
    spark_bbox_crop: Box4
    spark_point_crop: Point2 | None = None
    window_crop: IntBox4 | None = None  # the crop window that was enlarged and sent
    zoom_bbox_crop: Box4 | None = None  # the zoom box nearest Spark's box centre, as Spark drew it, in crop pixels
    zoom_type: MarkKind | None = None  # what the zoomed look called it; the mark keeps Spark's first type
    zoom_boxes: int | None = None  # how many boxes the zoomed look returned
    moved_bbox_crop: Box4 | None = None
    spark_zoom_request_id: str | None = None
    note: str = ""  # why Spark's box was kept, in words; empty when the moved box is used


class OffPanel(Contract):
    """A Spark mark whose point sat off its keyframe's panel mask by more than the claim path's on_panel_frac of the
    crop's long side (lexi run 1, 11/2, thresholds dedup.marks_on_panel). When the same walk has a keyframe of another
    panel cut from the same frame whose mask holds the point, the mark was moved there (moved_to_kf; the row then
    carries that panel, keyframe and crop coordinates); otherwise it is left out of matching and named in how_checked."""

    from_panel: PanelId
    from_kf: str
    off_frac: float  # distance from the point to the panel mask over the crop's long side
    moved_to_kf: str | None = None


class SecondPass(Contract):
    """A mark found by the second marks pass (lexi run 1, 11/2, thresholds matching.second_marks_pass): on a panel that
    passed every gate where this walk had no mark and the other walk had, Muse Spark was asked again on a zoomed window
    of this walk's compared crop around the other walk's mark mapped through the panel frame."""

    other_mark: str  # the other walk's mark whose spot was looked at
    window_crop: IntBox4
    expected_crop: Point2  # where the other walk's mark maps in this crop
    boxes: int  # how many boxes the zoomed look returned
    spark_request_id: str | None = None


class HalvesFind(Contract):
    """A mark found by the halves pass (11/2, thresholds dedup.spark_halves_pass): Muse Spark's box on an enlarged half
    of the crop, not the same mark as any box already kept on that crop."""

    half: int  # 0 the left or top half, 1 the right or bottom
    window_crop: IntBox4
    boxes: int  # how many boxes that half's answer returned
    spark_request_id: str | None = None


class HalvesMerge(Contract):
    """A full-crop mark that took the box of a halves mark close to it once both were moved by the zoom refine and the SAM
    step (11/2, thresholds dedup.spark_halves_pass merge_patches): one mark, the halves box, the full crop's type and
    confidence."""

    halves_mark_id: str
    half: int
    full_bbox_crop: Box4  # the full-crop mark's own box after its zoom move and SAM step, which the halves box replaced
    gap_patches: float  # centre gap in vision patches of the full crop
    overlap: bool  # the two boxes overlapped
    spark_request_id: str | None = None  # the halves answer


class Mark(Contract):
    """One derived/marks.json entry. Grid coords are what Spark said; crop, src and w are code's mappings."""

    mark_id: str
    role: Role
    panel_id: PanelId
    kf_id: str  # the crop id: the keyframe whose crop Spark saw
    source: MarkSource
    type: MarkKind
    confidence: Confidence
    size_class: SizeClass
    description: str = ""  # Spark's note
    grid_point: GridPoint | None = None
    grid_bbox: GridBox | None = None
    bbox_crop: Box4 | None = None
    point_crop: Point2 | None = None
    bbox_src: Box4 | None = None
    point_src: Point2 | None = None
    w: Point2 | None = None  # whitened [u, v]
    r: float | None = None
    size_units: float | None = None
    size_px_src: float | None = None
    single_frame: bool = False
    cluster_id: str | None = None
    merged_from: list[str] = Field(default_factory=list)
    spark_request_id: str | None = None
    caps: list[str] = Field(default_factory=list)  # K ids set upstream
    box_refine: BoxRefine | None = None  # left out of the JSON when absent, so marks of runs without the leaf keep their bytes
    zoom_refine: ZoomRefine | None = None  # the same rule
    off_panel: OffPanel | None = None  # the same rule (dedup.marks_on_panel)
    second_pass: SecondPass | None = None  # the same rule (matching.second_marks_pass)
    halves: HalvesFind | None = None  # the same rule (dedup.spark_halves_pass)
    halves_merge: HalvesMerge | None = None  # the same rule (dedup.spark_halves_pass merge_patches)

    @model_serializer(mode="wrap")
    def _omit_box_refine(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict):
            for key in ("box_refine", "zoom_refine", "off_panel", "second_pass", "halves", "halves_merge"):
                if data.get(key) is None:
                    data.pop(key, None)
        return data


class WhitenedMark(Contract):
    """A mark's position in the panel-normalized frame (section 5.5)."""

    mark_id: str
    role: Role
    panel_id: PanelId
    kf_id: str
    u: float
    v: float
    r: float
    frame_used: FrameUsed = FrameUsed.MOMENT
    residual_applied: Point2 | None = None  # the handle residual subtracted (return marks only)


# ----------------------------------------------------------------------------------------------------------------
# Stays Put (derived/mark_tracks.json)
# ----------------------------------------------------------------------------------------------------------------


class ClipRecord(Contract):
    path: str
    sha256: Sha256 | None = None
    f0_src: int
    frames: int = 91
    window_px: int
    windows_src: list[Box4] = Field(default_factory=list)  # one window per clip frame in source pixels
    w: int = 1008
    h: int = 1008


class TrackPoint(Contract):
    f: int
    box_clip: Box4
    c: float | None = None
    inside_panel: bool = True
    w: Point2 | None = None
    u_rel: float | None = None


class StaysPutNeighbour(Contract):
    """One rigid neighbour SAM 3.1 tracked in the mark's clip (docs/STAYS_PUT_ANCHORS.md): a lane part (door handle,
    side mirror, wheel, light, plate, window) or another mark on the panel. Residuals are its own leave-one-out medians
    in clip pixels; centres are its box centres as [f, x, y] in clip pixels, for the page's markers."""

    object_id: str
    kind: str  # "mark" or the lane noun
    ref: str  # the mark id, the track id or "<kf_id>_handle"
    frames: int = 0
    used: bool = False  # tracked on at least staysput.neighbour_track_share_min of the clip, so it is in the fit
    residual_full_px: float | None = None
    residual_near_px: float | None = None
    centres: list[Annotated[list[float], Field(min_length=3, max_length=3)]] = Field(default_factory=list)


class NeighbourTest(Contract):
    """Stays Put by neighbours for one mark: the median distance in clip pixels between the mark and where an affine
    fit to its neighbours puts it, divided by the neighbours' own leave-one-out median, over the whole clip (full) and
    within near_s of the prompt frame (near). fixed when both ratios are under 1.5, sliding when both are over 2.5,
    else inconclusive; fewer than 4 neighbours tracked through the clip is inconclusive."""

    verdict: StaysPutVerdict = StaysPutVerdict.NOT_RUN
    ratio_full: float | None = None
    ratio_near: float | None = None
    mark_full_px: float | None = None
    mark_near_px: float | None = None
    neighbours_full_px: float | None = None
    neighbours_near_px: float | None = None
    near_s: float = 0.5
    near_frames: int = 0
    used: int = 0  # neighbours in the fit
    reason: str | None = None
    neighbours: list[StaysPutNeighbour] = Field(default_factory=list)


class MarkTrack(Contract):
    mark_id: str
    role: Role
    clip: ClipRecord | None = None
    request_id: str | None = None
    prompt_text: str = ""
    track: list[TrackPoint] = Field(default_factory=list)
    b1: float | None = None
    b2: float | None = None
    b_pass: bool = False
    panned: bool = False
    drift_rms: float | None = None
    sign_agreement: float | None = None
    pairs_n: int = 0
    verdict: StaysPutVerdict = StaysPutVerdict.NOT_RUN
    curvature_cap: bool = False
    highlight_cap: bool = False
    is_control: bool = False
    neighbour_test: NeighbourTest | None = None  # None: the clip was planned before the neighbour test existed

    @model_serializer(mode="wrap")
    def _omit_absent_neighbour_test(self, handler: Any) -> Any:
        """A row without a neighbour test serializes as it did before the field existed, so older runs rebuild
        byte-identical."""
        data = handler(self)
        if isinstance(data, dict) and data.get("neighbour_test") is None:
            data.pop("neighbour_test", None)
        return data


# ----------------------------------------------------------------------------------------------------------------
# Match (derived/matches.json) and verdicts (derived/verdicts.json)
# ----------------------------------------------------------------------------------------------------------------


class MatchCandidate(Contract):
    """One pair row of matches.json. d is the full cost; d_pos the whitened distance alone."""

    pickup_mark: str
    return_mark: str
    d: float
    d_pos: float
    type_cost: float
    size_cost: float
    r_mean: float
    tau_pre_r: float
    tau_eye_r: float
    ratio: float | None = None
    margin: float | None = None
    mutual: bool = False
    # registration (thresholds matching.registration, 11/2): the pair's spots coincide once the pickup crop is mapped
    # onto the return crop, so d and d_pos are measured there; d_moment keeps the whitened-moment d it replaced (None
    # when the moment match had not paired the two)
    registered: bool = False
    d_moment: float | None = None

    @model_serializer(mode="wrap")
    def _omit_unregistered(self, handler: Any) -> Any:
        """A pair the registration did not touch serializes as it did before the fields existed (11/2)."""
        data = handler(self)
        if isinstance(data, dict) and not data.get("registered"):
            data.pop("registered", None)
            data.pop("d_moment", None)
        return data


class RegistrationFit(Contract):
    """One keyframe pair the registration step aligned (thresholds matching.registration, 11/2): crop pickup mapped onto
    crop return by the panel outline (ECC on the two SAM masks) or by SIFT matches on the panel's rigid structure, and
    the scores that accepted or refused it."""

    kf_pair: Annotated[list[str], Field(min_length=2, max_length=2)]  # [kf_pickup, kf_return]
    model: Literal["features", "outline", "none"]
    ok: bool
    why: str | None = None  # why it was refused
    n_matches: int = 0  # SIFT matches near where the outline fit puts them
    n_inliers: int = 0  # of those, within ransac_px of the fit
    spread: float = 0.0
    mask_iou: float = 0.0  # mask IoU after the warp, where the two crops overlap
    reproj_px: float | None = None
    H: list[list[float]] | None = None  # 3x3, pickup crop pixels to return crop pixels


class OwnContrast(Contract):
    """Does a mark seen in one walk only stand out from its own surroundings in plain pixels (thresholds
    matching.stands_out, 11/2, K20)? register.own_contrast on the mark's own crop and box, as Muse Spark placed it.
    stands_out keeps the label; faint caps it (K20); area is a box over max_panel_share of its panel, not tested;
    not_measured is a mark whose crop or mask could not be read, not capped."""

    mark_id: str
    kf_id: str
    result: Literal["stands_out", "faint", "area", "not_measured"]
    contrast: float | None = None
    panel_share: float | None = None
    why: str | None = None


class RegistrationMark(Contract):
    """The presence test of one mark at its mapped spot in the other walk's crop (11/2), and what the match did with it:
    capped (K18, the same patch is there and no mark of the other walk is), paired (with an unmatched mark of the other
    walk at that spot), pair_kept (a moment pair whose spots coincide, now measured by registration) or none."""

    mark_id: str
    other_kf: str
    result: Literal["present", "absent", "cannot_tell"]
    why: str | None = None
    mapped_box: Box4 | None = None  # in the other crop
    found_box: Box4 | None = None  # where the patch search put it
    ncc: float | None = None
    contrast_own: float | None = None  # Delta E the box stands out by in its own crop
    contrast_other: float | None = None  # and at the found spot in the other crop
    uncertainty_px: float | None = None
    used: Literal["capped", "paired", "pair_kept", "none"] = "none"
    paired_with: str | None = None


class MarkGroup(Contract):
    type: MarkKind
    n_pickup: int
    n_return: int
    text: str


class RegistrationVouch(Contract):
    """The registration's word on a keyframe pair whose only failing gate was G4 (lead decision 1 of 11/2, thresholds
    matching.registration_vouches): the fit of the pickup crop onto the return crop and whether it was strong enough to
    stand in for the mask IoU check. vouched True drops G4 from the pair's gates; why says which score fell short."""

    fit: RegistrationFit
    vouched: bool
    why: str | None = None


class KeyframePairTry(Contract):
    """One keyframe pair match.derive_match tried on a panel (lead 11/1, rule 2): the gates it failed (empty when it
    passed every gate) and its whitened mask IoU (None when there was none to take)."""

    kf_pair: Annotated[list[str], Field(min_length=2, max_length=2)]  # [kf_pickup, kf_return]
    gates: list[str] = Field(default_factory=list)
    mask_iou: float | None = None
    # False: a keyframe whose marks are compared turned its panel axis too far from this pair's pickup keyframe, so the
    # IoU was not taken and G4 fires (match._frames_agree)
    frames_agree: bool = True
    # the registration asked to vouch for the views where G4 was the only failing gate (matching.registration_vouches,
    # 11/2); left out of the JSON when it was not asked, so a run frozen before it keeps its bytes
    vouch: RegistrationVouch | None = None

    @model_serializer(mode="wrap")
    def _omit_no_vouch(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict) and data.get("vouch") is None:
            data.pop("vouch", None)
        return data


class ReshapeCheck(Contract):
    """The reshaped-panel test (thresholds matching.reshaped_panel, lead 11/1, second), recorded on a panel G4 alone kept
    from being compared on every keyframe pair tried with the best whitened mask IoU in [reshape_iou_min,
    mask_iou_min). called is True when both tests passed: the panel is needs your eye (L7, outline changed in place)."""

    kf_pair: Annotated[list[str], Field(min_length=2, max_length=2)]  # the pair with the best IoU, [kf_pickup, kf_return]
    mask_iou: float
    aspect_ratios: Annotated[list[float], Field(min_length=2, max_length=2)]  # the two keyframes' box aspect ratios
    aspect_off: float  # |a - b| / max(a, b)
    blob_share: float | None = None  # the largest blob's share of the opened difference; None when the opening left nothing
    called: bool = False


class PrivacyDrop(Contract):
    """A mark left out of matching because its box sits in the part of its crop blurred for privacy (thresholds
    matching.privacy_blur, R27): share is the part of the box under the blur. Printed in how_checked."""

    mark_id: str
    kf_id: str
    share: float


class FixtureDrop(Contract):
    """A mark left out of matching because its box sits on a factory fixture the run already has (thresholds
    matching.fixtures_not_damage, lead decision 4 of 11/1): a fixture landmark Muse Spark pointed at in the same crop
    (source "spark"), or the SAM door-handle point of the mark's keyframe (source "sam_handle"). overlap is the share of
    the mark's box inside the fixture's box, on Spark's 0 to 1000 grid of that crop. Printed in how_checked, never
    dropped silently."""

    mark_id: str
    fixture: str  # the landmark name (sensor, badge, handle, ...) or "handle" for the SAM door handle
    source: Literal["spark", "sam_handle"]
    overlap: float


class ByEyeLine(Contract):
    """One check-by-eye line on a panel in place of mark-level verdicts (lead Nov 2, hero_focus1). kind "glass": the
    panel is glass (thresholds matching.glass_by_eye), its marks are left out of matching and it gets no verdict;
    kind "debris": the needs-your-eye lines whose every mark Muse Spark typed dirt on a flagged view (thresholds
    matching.debris_by_eye), folded into this one line. mark_ids are the marks the line stands for, never dropped
    silently."""

    kind: Literal["glass", "debris"]
    text: str
    mark_ids: list[str] = Field(default_factory=list)


class SameWalkNote(Contract):
    """A mark of a non-compared keyframe carried onto its walk's compared keyframe by registration (thresholds
    matching.same_walk_register, 11/2): merged_into names the compared keyframe's mark it landed on (the two are one
    mark, this one leaves matching); None means it was carried, re-placed and whitened in the compared keyframe."""

    mark_id: str
    from_kf: str
    to_kf: str
    merged_into: str | None = None
    model: str = "none"  # the fit used: features or outline
    reproj_px: float | None = None


class OffPanelNote(Contract):
    """A mark moved onto or left off this panel by the on-panel rule (thresholds dedup.marks_on_panel), for how_checked."""

    mark_id: str
    from_panel: PanelId
    to_panel: PanelId | None = None  # None: left out of matching
    off_frac: float


class PanelMatch(Contract):
    panel_id: PanelId
    state: MatchState
    reasons: list[str] = Field(default_factory=list)  # rule ids
    kf_pair: Annotated[list[str], Field(min_length=2, max_length=2)] | None = None  # [kf_pickup, kf_return]
    aspect_diff: float | None = None
    mask_iou: float | None = None
    handle_residual: Point2 | None = None
    frame_used: FrameUsed = FrameUsed.MOMENT
    pairs: list[MatchCandidate] = Field(default_factory=list)
    unmatched_pickup: list[str] = Field(default_factory=list)
    unmatched_return: list[str] = Field(default_factory=list)
    groups: list[MarkGroup] = Field(default_factory=list)
    # every keyframe pair tried in order, the chosen pair first (thresholds matching.next_keyframe_pair, lead 11/1)
    kf_pairs_tried: list[KeyframePairTry] = Field(default_factory=list)
    # the reshaped-panel test where its precondition held (thresholds matching.reshaped_panel, lead 11/1, second)
    reshape: ReshapeCheck | None = None
    # marks left out of matching as factory fixtures (thresholds matching.fixtures_not_damage, lead decision 4 of 11/1)
    fixtures: list[FixtureDrop] = Field(default_factory=list)
    # marks left out of matching because their box sits in a blurred part of the crop (matching.privacy_blur, R27)
    privacy_drops: list[PrivacyDrop] = Field(default_factory=list)
    # marks moved onto or left off this panel because they sat off their own panel mask (dedup.marks_on_panel, 11/2)
    off_panel: list[OffPanelNote] = Field(default_factory=list)
    # the second marks pass on this panel: the other walk's marks whose spot was looked at again (11/2)
    second_pass: list[str] = Field(default_factory=list)
    # registration (thresholds matching.registration, 11/2): the keyframe pairs aligned and each mark's presence test
    registration: list[RegistrationFit] = Field(default_factory=list)
    registration_marks: list[RegistrationMark] = Field(default_factory=list)
    # check-by-eye lines (thresholds matching.glass_by_eye and matching.debris_by_eye, Nov 2)
    by_eye: list[ByEyeLine] = Field(default_factory=list)
    # marks of the walk's other keyframes carried onto the compared keyframe (matching.same_walk_register, Nov 2)
    same_walk: list[SameWalkNote] = Field(default_factory=list)
    # one-walk marks tested for standing out from their own surroundings (matching.stands_out, 11/2, K20)
    own_contrast: list[OwnContrast] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _omit_no_tries(self, handler: Any) -> Any:
        """A panel matched without the next-pair rule (or the reshaped-panel rule, or the fixture rule) serializes as it
        did before the field existed (11/1), so every run frozen before it rebuilds byte-identical."""
        data = handler(self)
        if isinstance(data, dict) and not data.get("kf_pairs_tried"):
            data.pop("kf_pairs_tried", None)
        if isinstance(data, dict) and data.get("reshape") is None:
            data.pop("reshape", None)
        if isinstance(data, dict) and not data.get("fixtures"):
            data.pop("fixtures", None)
        for key in ("off_panel", "second_pass", "registration", "registration_marks", "by_eye", "same_walk", "own_contrast", "privacy_drops"):
            if isinstance(data, dict) and not data.get(key):
                data.pop(key, None)
        return data


class VerdictFlags(Contract):
    """The booleans verdict() takes. One field per K cap of section 6 plus which walk an unmatched mark is from."""

    return_mark: bool = False  # d None: True -> new_or_unseen, False -> not_seen_at_return
    K1: bool = False
    K2: bool = False
    K3: bool = False
    K4: bool = False
    K5: bool = False
    K6: bool = False
    K7: bool = False
    K8: bool = False
    K9: bool = False
    K10: bool = False
    K11: bool = False
    K12: bool = False
    K13: bool = False
    K14: bool = False
    K15: bool = False  # a flagged view caps a mark seen in one walk only (replaces gate G2, lead 10/31)
    K16: bool = False  # the counter-look at the other walk's same spot said same or unsure (lead 11/1, third)
    K17: bool = False  # the two-question counter-look did not say yes, a damage mark is in the box (lead decision 5 of 11/1)
    K18: bool = False  # registration: the other walk's crop shows the same patch at this mark's mapped spot (11/2)
    K19: bool = False  # the mark's spot lies inside a box the other walk drew in its compared view (11/2)
    K20: bool = False  # the mark's box does not stand out from its own surroundings in its own crop (11/2)

    def set_caps(self) -> list[str]:
        return [name for name in sorted(type(self).model_fields) if name.startswith("K") and getattr(self, name)]


class Trace(Contract):
    rule_ids: list[str] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    sam_request_ids: list[str] = Field(default_factory=list)
    spark_request_ids: list[str] = Field(default_factory=list)
    event_seqs: list[int] = Field(default_factory=list)
    config_hash: str | None = None
    # K ids that applied as notes and lowered nothing: K4 and K12 on a tight pair the second look answered (lead 11/1)
    note_ids: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _omit_no_notes(self, handler: Any) -> Any:
        """A trace with no note id serializes as it did before the field existed (11/1)."""
        data = handler(self)
        if isinstance(data, dict) and not data.get("note_ids"):
            data.pop("note_ids", None)
        return data


class SizeEstimate(Contract):
    inches: float | None = None
    bound_lo: float | None = None
    bound_hi: float | None = None
    ruler: RulerKind | None = None
    ruler_source_kf: str | None = None
    tire_code: str | None = None
    threshold_type: ThresholdType | None = None
    refused_reason: str | None = None
    ruler_notes: list[str] = Field(default_factory=list)  # why each other ruler next to the mark did not size it (polish lane)

    @model_serializer(mode="wrap")
    def _omit_empty_ruler_notes(self, handler: Any) -> Any:
        """A size with no note on another ruler serializes as it did before the field existed (10/31)."""
        data = handler(self)
        if isinstance(data, dict) and not data.get("ruler_notes"):
            data.pop("ruler_notes", None)
        return data


class Verdict(Contract):
    """One derived/verdicts.json entry. rule_id is the label rule (L1..L5); trace.rule_ids holds every G and K."""

    verdict_id: str
    panel_id: PanelId
    label: VerdictLabel
    rule_id: str
    mark_ids: list[str] = Field(default_factory=list)
    distance_units: float | None = None
    distance_cm_est: float | None = None
    reasons: list[str] = Field(default_factory=list)
    trace: Trace = Field(default_factory=Trace)
    size: SizeEstimate | None = None
    view_note: str | None = None  # the flagged view in words: why K15 capped it, or the note on a pair seen in both walks
    # K4 and K12 on a pair inside tau_pre_r in words (lead 11/1): what Muse Spark typed and how sure it was, and what the
    # second look made of it (a note when it answered, the reason they still cap when it did not)
    type_note: str | None = None
    # a mark seen in one walk only that went to the counter-look (lead 11/1, third): what the look at the other walk's
    # same spot said and what it did to the label (K16 on same or unsure; the label stands on different or no answer)
    counter_note: str | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_view_note(self, handler: Any) -> Any:
        """A verdict with no flagged view and no type note serializes as it did before the fields existed (10/31 and
        11/1), so a run with neither rebuilds byte-identical."""
        data = handler(self)
        if isinstance(data, dict):
            for key in ("view_note", "type_note", "counter_note"):
                if data.get(key) is None:
                    data.pop(key, None)
        return data


class Calibration(Contract):
    """Section 5.10: the thresholds that freeze into run.json after the pseudo pair."""

    sigma_frame: float | None = None
    sigma_spark: float | None = None
    sigma_view: float | None = None
    sigma_within: float | None = None
    tau_pre: float
    tau_eye: float
    tau_fix: float
    calibrated: bool = False  # false when tau_pre hit the cap; the README then says so


# ----------------------------------------------------------------------------------------------------------------
# Coverage (derived/coverage.json), narration, claim
# ----------------------------------------------------------------------------------------------------------------


class PanelCoverage(Contract):
    state: CoverageState
    state_basis: StateBasis = StateBasis.C
    seconds: float = 0.0
    best_kf: str | None = None
    reason: str = ""


class Odometer(Contract):
    asked: bool = False
    filmed: bool = False
    frame_path: str | None = None


class Coverage(Contract):
    per_role: dict[Role, dict[PanelId, PanelCoverage]] = Field(default_factory=dict)
    missing: list[PanelId] = Field(default_factory=list)  # wheels first, then lower bumpers, then the rest
    topup_text: str = ""
    nouns_to_rerun: list[Noun] = Field(default_factory=lambda: list(TOPUP_ALWAYS_NOUNS))
    rerun_seconds: float | None = None
    odometer: Odometer = Field(default_factory=Odometer)


class NarrationTurn(Contract):
    turn_id: str
    start_ms: int
    end_ms: int
    text: str
    keywords_hit: list[str] = Field(default_factory=list)
    pinned_panel: PanelId | None = None
    pin_seconds: float | None = None
    pin_kind: PinKind | None = None
    seed_mark_id: str | None = None
    chip_path: str | None = None


class NarrationRole(Contract):
    turns: list[NarrationTurn] = Field(default_factory=list)
    transcript: str | None = None  # Voice's words as heard when they came back with no turns (section 14 item 16b)


class Narration(Contract):
    per_role: dict[Role, NarrationRole] = Field(default_factory=dict)
    mode_applied: bool = True


class ClaimImage(Contract):
    sha256: Sha256
    received_at: Timestamp
    w: int
    h: int
    low_res: bool = False


class ClaimRoi(Contract):
    box: Box4
    source: RoiSource


class ClaimNaming(Contract):
    spark: PanelId | None = None
    sam_votes: list[dict[str, Any]] = Field(default_factory=list)
    agreed: bool = False


class ClaimPlacement(Contract):
    method: PlacementMethod = PlacementMethod.NONE
    w_claim: Point2 | None = None
    r_claim: float | None = None


class NotePrefill(Contract):
    fields: dict[str, str] = Field(default_factory=dict)
    verified_against_typed: bool = False


class Claim(Contract):
    """derived/claim.json (the latest claim); every claim also at claim/<claim_id>.json."""

    claim_id: str | None = None  # "claim_<n>": the upload claim/<claim_id>.jpg (never served), its blurred copy claim/<claim_id>.blurred.jpg
    image: ClaimImage
    text: str = ""
    scanner_style: bool = False
    roi: ClaimRoi | None = None
    naming: ClaimNaming = Field(default_factory=ClaimNaming)
    panel_id: PanelId | None = None
    side_source: Literal["text", "spark", "both_shown"] | None = None
    coverage_state: CoverageState | None = None
    placement: ClaimPlacement = Field(default_factory=ClaimPlacement)
    match: PanelMatch | None = None
    size: SizeEstimate | None = None
    exit: ClaimExit | None = None
    paragraph: str = ""
    note_prefill: NotePrefill | None = None


# ----------------------------------------------------------------------------------------------------------------
# Sheet (sheet/sheet.json)
# ----------------------------------------------------------------------------------------------------------------


class SheetOriginal(Contract):
    role: Role
    name: str
    bytes: int
    sha256: Sha256
    camera_time: Timestamp | Literal["not present"] = "not present"
    received_at: Timestamp | None = None
    hash_match: bool | None = None
    ots_status: str | None = None
    tsr_time: Timestamp | None = None
    bitcoin_block: int | None = None


class SheetFrame(Contract):
    kf_id: str
    path: str
    video_time_s: float
    wall_clock: Timestamp | None = None  # capture time plus pts
    mark_ids: list[str] = Field(default_factory=list)


class SheetVerdictLine(Contract):
    verdict_id: str
    label: VerdictLabel
    rule_ids: list[str] = Field(default_factory=list)
    mark_ids: list[str] = Field(default_factory=list)
    type: MarkKind | None = None
    size_class: SizeClass | None = None
    distance_units: float | None = None
    distance_cm_est: float | None = None
    staysput_line: str | None = None  # drift, b1, b2, panned
    size_line: str | None = None  # "about 0.9 in, between 0.6 and 1.3 in (ruler: ...)" or "size not stated: ..."
    ordinary_wear_line: str | None = None
    view_line: str | None = None  # Verdict.view_note: a flagged view named (K15, or the note on a pair seen in both walks)
    type_line: str | None = None  # Verdict.type_note: K4 and K12 on a tight pair and what the second look made of them (11/1)
    counter_line: str | None = None  # Verdict.counter_note: the counter-look on a mark seen in one walk only (11/1, third)

    @model_serializer(mode="wrap")
    def _omit_absent_view_line(self, handler: Any) -> Any:
        """A line with no flagged view and no type note serializes as it did before the fields existed (10/31, 11/1)."""
        data = handler(self)
        if isinstance(data, dict):
            for key in ("view_line", "type_line", "counter_line"):
                if data.get(key) is None:
                    data.pop(key, None)
        return data


class SheetChip(Contract):
    turn_id: str
    start_ms: int
    end_ms: int
    text: str
    chip_path: str | None = None


class SheetPanelBlock(Contract):
    panel_id: PanelId
    display_name: str
    lead_with_frames: bool = False  # wheels and bumpers (5.8 item 11)
    pickup: SheetFrame | None = None
    return_: SheetFrame | None = Field(default=None, alias="return")
    coverage_seconds: dict[Role, float] = Field(default_factory=dict)
    coverage_states: dict[Role, CoverageState] = Field(default_factory=dict)
    verdicts: list[SheetVerdictLine] = Field(default_factory=list)
    group_lines: list[str] = Field(default_factory=list)
    chips: list[SheetChip] = Field(default_factory=list)
    odometer_path: str | None = None
    how_checked: str | None = None  # match.how_checked: the keyframe pair compared on and each pair tried with its gates (11/1)
    by_eye_lines: list[str] = Field(default_factory=list)  # match.by_eye_words: glass and debris lines (Nov 2)

    @model_serializer(mode="wrap")
    def _omit_absent_how_checked(self, handler: Any) -> Any:
        """A block with no recorded keyframe pair tries serializes as it did before the field existed (11/1); one with
        no check-by-eye line as before Nov 2."""
        data = handler(self)
        if isinstance(data, dict) and data.get("how_checked") is None:
            data.pop("how_checked", None)
        if isinstance(data, dict) and not data.get("by_eye_lines"):
            data.pop("by_eye_lines", None)
        return data


class SheetNotComparable(Contract):
    panel_id: PanelId
    rule_ids: list[str]
    reasons: list[str] = Field(default_factory=list)


class SheetMethod(Contract):
    config_hash: str | None = None
    thresholds: dict[str, Any] = Field(default_factory=dict)
    eval_lines: list[str] = Field(default_factory=list)  # each with N, copied from eval/run.py output
    blurred: list[str] = Field(default_factory=list)
    deletions: list[DeletionRecord] = Field(default_factory=list)
    storage_line: str = ""
    rebuild_status: str = ""  # "rebuild identical" or "rebuild differs: <file>"
    what_seal_does_not_prove: list[str] = Field(default_factory=list)
    hertz_threshold_note: str = ""


class SheetModel(Contract):
    """sheet.json: everything the Jinja template renders. The sheet module never recomputes a verdict."""

    product: str = "Walkaround"
    run_id: str
    app_commit: str | None = None
    config_hash: str | None = None
    events_head_hash: str | None = None
    disclaimer: str = "evidence support, not legal advice"
    stand_in_line: str = ""  # page 1, when the lanes were sent with stand-in prompts (a test clip, not a car)
    clocks: ThreeClocks = Field(default_factory=ThreeClocks)
    deletion_clocks: DeletionClocks = Field(default_factory=DeletionClocks)
    originals: list[SheetOriginal] = Field(default_factory=list)
    agreement_number: str | None = None
    plate_line: str = "plate blurred"
    privacy_line: str = ""
    note_lines: list[str] = Field(default_factory=list)  # the five-line note in section 7 order
    coverage_summary: dict[Role, list[str]] = Field(default_factory=dict)
    panels: list[SheetPanelBlock] = Field(default_factory=list)
    not_comparable: list[SheetNotComparable] = Field(default_factory=list)
    method: SheetMethod = Field(default_factory=SheetMethod)
    appendices: dict[str, str] = Field(default_factory=dict)  # A certification, B glossary, C verify-it-yourself
    letter_md: str = ""
    footer_hash_prefixes: list[str] = Field(default_factory=list)  # first 12 hex of each original's sha256
    # run.footage_credit and its license link, printed on page 1, in every page footer and in the letter when thresholds
    # sheet.footage_credit is on (11/1; CC BY 4.0 asks for both wherever the modified frames are shared)
    footage_credit: str | None = None
    footage_license_url: str | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_footage(self, handler: Any) -> Any:
        """A sheet with no credit printed serializes as it did before the fields existed (11/1), so every run frozen
        before the rule rebuilds byte-identical."""
        data = handler(self)
        if isinstance(data, dict):
            for key in ("footage_credit", "footage_license_url"):
                if data.get(key) is None:
                    data.pop(key, None)
        return data


# ----------------------------------------------------------------------------------------------------------------
# Ledger and deletions
# ----------------------------------------------------------------------------------------------------------------


class LedgerTokens(Contract):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    in_: int = Field(default=0, alias="in")
    out: int = 0
    reasoning: int = 0
    cached: int = 0


class LedgerLatency(Contract):
    upload: int | None = None
    first_delta: int | None = None
    done: int | None = None


class LedgerRow(Verbatim):
    """One runs/ledger.jsonl line as walkaround.ledger.record writes it (usd is the dollars column of section 4).
    Extra keys from record(**usage) are kept, so older rows still parse."""

    ts: str
    kind: str
    model: str
    run_id: str | None = None
    note: str = ""
    usd: float = 0.0
    engine: str | None = None
    request_id: str | None = None
    unit: LedgerUnit | None = None
    quantity: float | None = None
    tokens: LedgerTokens | None = None
    latency_ms: LedgerLatency | None = None
    parse_cpu_ms: int | None = None
    status: str | None = None
    retries: int | None = None


class DeletionRecord(Contract):
    target: str  # path, file_id or request_id
    kind: DeletionKind
    sha256: Sha256 | None = None
    bytes: int | None = None
    deleted_at: Timestamp
    reason: str = ""
    api_response: dict[str, Any] | None = None


class Deletions(Contract):
    """deletions.json."""

    run_id: str
    rows: list[DeletionRecord] = Field(default_factory=list)
    keep_plate: bool = False
    keep_plate_chosen_by: str | None = None  # "renter at upload" or None


# ----------------------------------------------------------------------------------------------------------------
# Replay payload (GET /api/runs/<id>/replay)
# ----------------------------------------------------------------------------------------------------------------


class ReplayPayload(Contract):
    run_id: str
    mode: ReplayMode
    recorded_on: str  # date for the badge "replayed from a run on <date>"
    delays_ms: dict[str, int] = Field(default_factory=dict)  # per stage, from web/kiosk/delays.json in fixed mode
    events: list[Event] = Field(default_factory=list)


SealManifest.model_rebuild()
SheetMethod.model_rebuild()

__all__ = [name for name in globals() if not name.startswith("_")]
