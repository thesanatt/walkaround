"""The runs/<run_id>/ layout (docs/ARCHITECTURE.md section 4) as functions.

A function that names a directory creates it; a function that names a file creates its parent. Nothing else
happens here: no reads, no model calls. The root is config.RUNS_DIR (WALKAROUND_RUNS or <repo>/runs; /data/runs on
Railway). Every path is returned absolute. Lane ids are contracts.noun_slug(noun) or a prompt id.
"""

from __future__ import annotations

from pathlib import Path

from .config import REPO_DIR, RUNS_DIR
from .contracts import Noun, noun_slug


def _dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def lane_id(noun_or_prompt: Noun | str) -> str:
    """'license plate' -> 'license_plate'; a prompt id is returned as is."""
    return noun_slug(noun_or_prompt)


# Repo-level ---------------------------------------------------------------------------------------------------


def runs_root() -> Path:
    return _dir(Path(RUNS_DIR))


def ledger_path() -> Path:
    """runs/ledger.jsonl, the global ledger walkaround.ledger writes."""
    return _file(runs_root() / "ledger.jsonl")


def uploads_cache_path() -> Path:
    """runs/uploads.json: sha256 -> Files API id cache used by sam_client.upload_file."""
    return _file(runs_root() / "uploads.json")


def thresholds_path() -> Path:
    return Path(REPO_DIR) / "config" / "thresholds.json"


def samples_dir() -> Path:
    """samples/<run_id>/ in the repo: the demo run's blurred artifacts, copied to the volume at boot."""
    return _dir(Path(REPO_DIR) / "samples")


def sample_run_dir(run_id: str) -> Path:
    return _dir(samples_dir() / run_id)


def web_dist_dir() -> Path:
    return Path(REPO_DIR) / "web" / "dist"


def kiosk_delays_path() -> Path:
    return Path(REPO_DIR) / "web" / "kiosk" / "delays.json"


# Run root and top-level files ----------------------------------------------------------------------------------


def run_dir(run_id: str) -> Path:
    return _dir(runs_root() / run_id)


def run_json(run_id: str) -> Path:
    return _file(run_dir(run_id) / "run.json")


WORK_LOCK_NAME = ".work.lock"  # pid, hostname and start time of the process working on the run; never served or sealed


def work_lock_path(run_id: str) -> Path:
    """The run's work lock; no mkdir, so releasing it never recreates a run folder that is gone."""
    return runs_root() / run_id / WORK_LOCK_NAME


def events_path(run_id: str) -> Path:
    """events.ndjson, the hash-chained event log (M3)."""
    return _file(run_dir(run_id) / "events.ndjson")


def events_gz_path(run_id: str) -> Path:
    return _file(run_dir(run_id) / "events.ndjson.gz")


def spark_json(run_id: str) -> Path:
    return _file(run_dir(run_id) / "spark.json")


def sam_marks_json(run_id: str) -> Path:
    """SAM 3.1 image answers on mark boxes (thresholds dedup.sam_box_refine), a frozen fact like spark.json."""
    return _file(run_dir(run_id) / "sam_marks.json")


def second_pass_json(run_id: str) -> Path:
    """Muse Spark answers of the second marks pass (thresholds matching.second_marks_pass), a frozen fact."""
    return _file(run_dir(run_id) / "second_pass.json")


def halves_marks_json(run_id: str) -> Path:
    """Muse Spark answers on the two enlarged halves of each crop (thresholds dedup.spark_halves_pass), a frozen fact."""
    return _file(run_dir(run_id) / "halves_marks.json")


def zoom_marks_json(run_id: str) -> Path:
    """Muse Spark answers on zoomed windows around mark boxes (thresholds dedup.spark_zoom_refine), a frozen fact."""
    return _file(run_dir(run_id) / "zoom_marks.json")


def voice_json(run_id: str) -> Path:
    return _file(run_dir(run_id) / "voice.json")


def deletions_json(run_id: str) -> Path:
    return _file(run_dir(run_id) / "deletions.json")


# Uploads (chunks before finalize) ------------------------------------------------------------------------------


def uploads_dir(run_id: str, upload_id: str) -> Path:
    return _dir(run_dir(run_id) / "uploads" / upload_id)


def upload_chunk_path(run_id: str, upload_id: str, index: int) -> Path:
    return _file(uploads_dir(run_id, upload_id) / "chunks" / f"{index:06d}")


# Quarantine (unblurred originals; never served) ---------------------------------------------------------------


def quarantine_dir(run_id: str) -> Path:
    return _dir(run_dir(run_id) / "quarantine")


def quarantine_original(run_id: str, role: str) -> Path:
    """quarantine/<role>.orig: the uploaded bytes, cut into the proxy, deleted at run end."""
    return _file(quarantine_dir(run_id) / f"{role}.orig")


def quarantine_temp(run_id: str, name: str) -> Path:
    """Any unblurred intermediate (the sweeper removes files here older than 30 minutes)."""
    return _file(quarantine_dir(run_id) / "tmp" / name)


# Videos and media -----------------------------------------------------------------------------------------------


def videos_dir(run_id: str) -> Path:
    return _dir(run_dir(run_id) / "videos")


def video_json(run_id: str, role: str) -> Path:
    return _file(videos_dir(run_id) / f"{role}.json")


def media_dir(run_id: str) -> Path:
    return _dir(run_dir(run_id) / "media")


def proxy_path(run_id: str, role: str) -> Path:
    """The unblurred proxy sent to the Files API; deleted with a deletions.json row at blur.done."""
    return _file(media_dir(run_id) / f"{role}.proxy.mp4")


def proxy_blurred_path(run_id: str, role: str) -> Path:
    return _file(media_dir(run_id) / f"{role}.proxy_blurred.mp4")


def pts_map_path(run_id: str, role: str) -> Path:
    """JSON list: proxy frame index -> source pts in seconds."""
    return _file(media_dir(run_id) / f"{role}.pts.json")


def wav_path(run_id: str, role: str) -> Path:
    return _file(media_dir(run_id) / f"{role}.wav")


def overlay_path(run_id: str, role: str) -> Path:
    """Safari-only fallback overlay MP4."""
    return _file(media_dir(run_id) / f"{role}.overlay.mp4")


# SAM lanes ------------------------------------------------------------------------------------------------------


def sam_dir(run_id: str, role: str) -> Path:
    return _dir(run_dir(run_id) / "sam" / role)


def lane_events_path(run_id: str, role: str, lane: Noun | str) -> Path:
    """sam/<role>/<lane>.ndjson: raw Responses SSE events while streaming."""
    return _file(sam_dir(run_id, role) / f"{lane_id(lane)}.ndjson")


def lane_events_gz_path(run_id: str, role: str, lane: Noun | str) -> Path:
    """The same lane gzipped at lane_done; what the browser fetches and the hash in sam.lane_done."""
    return _file(sam_dir(run_id, role) / f"{lane_id(lane)}.ndjson.gz")


def lane_summary_path(run_id: str, role: str, lane: Noun | str) -> Path:
    return _file(sam_dir(run_id, role) / f"{lane_id(lane)}.summary.json")


def lane_index_path(run_id: str, role: str, lane: Noun | str) -> Path:
    return _file(sam_dir(run_id, role) / f"{lane_id(lane)}.index.json")


def image_requests_dir(run_id: str, role: str) -> Path:
    """SAM image requests (plate, face, door handle checks): <request_id>.json verbatim."""
    return _dir(sam_dir(run_id, role) / "images")


# Keyframes, crops, clips, chips -------------------------------------------------------------------------------


def keyframes_dir(run_id: str, role: str) -> Path:
    return _dir(run_dir(run_id) / "keyframes" / role)


def keyframe_frame_path(run_id: str, role: str, kf_id: str) -> Path:
    """The blurred full frame, cropped to the car box (section 8 item 4)."""
    return _file(keyframes_dir(run_id, role) / f"{kf_id}.frame.jpg")


def crops_dir(run_id: str, role: str) -> Path:
    return _dir(run_dir(run_id) / "crops" / role)


def crop_path(run_id: str, role: str, kf_id: str) -> Path:
    """The blurred 1280 px panel crop Spark saw."""
    return _file(crops_dir(run_id, role) / f"{kf_id}.jpg")


def crop_mask_png_path(run_id: str, role: str, kf_id: str) -> Path:
    """One-bit panel mask in crop pixels."""
    return _file(crops_dir(run_id, role) / f"{kf_id}.mask.png")


def crop_blur_png_path(run_id: str, role: str, kf_id: str) -> Path:
    """One-bit mask of the crop pixels blurred for privacy (privacy.mask_blur, R27); written only when any are."""
    return _file(crops_dir(run_id, role) / f"{kf_id}.blur.png")


def clips_dir(run_id: str, role: str) -> Path:
    return _dir(run_dir(run_id) / "clips" / role)


def clip_path(run_id: str, role: str, clip_id: str) -> Path:
    """Stays Put 1008 px blurred clip; clip_id is a mark id or a shared-window id."""
    return _file(clips_dir(run_id, role) / f"{clip_id}.mp4")


def chips_dir(run_id: str, role: str) -> Path:
    return _dir(run_dir(run_id) / "chips" / role)


def chip_path(run_id: str, role: str, turn_id: str) -> Path:
    """Per-turn AAC narration chip."""
    return _file(chips_dir(run_id, role) / f"{turn_id}.m4a")


def claim_dir(run_id: str) -> Path:
    return _dir(run_dir(run_id) / "claim")


def claim_image_path(run_id: str, claim_role: str) -> Path:
    return _file(claim_dir(run_id) / f"{claim_role}.jpg")


# Derived JSON ---------------------------------------------------------------------------------------------------


def derived_dir(run_id: str) -> Path:
    return _dir(run_dir(run_id) / "derived")


def tracks_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "tracks.json")


def panels_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "panels.json")


def keyframes_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "keyframes.json")


def marks_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "marks.json")


def mark_tracks_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "mark_tracks.json")


def matches_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "matches.json")


def verdicts_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "verdicts.json")


def coverage_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "coverage.json")


def narration_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "narration.json")


def claim_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "claim.json")


def claim_record_path(run_id: str, claim_id: str) -> Path:
    """claim/<claim_id>.json, one contracts.Claim per claim; derived/claim.json mirrors the latest."""
    return _file(claim_dir(run_id) / f"{claim_id}.json")


def claim_blurred_path(run_id: str, claim_id: str) -> Path:
    """The claim photo with plates and faces pixelated; the only copy of the photo the app serves."""
    return _file(claim_dir(run_id) / f"{claim_id}.blurred.jpg")


def claim_open_weights_log(run_id: str, claim_id: str) -> Path:
    """claim/<claim_id>.open_weights.ndjson: the claim's Muse Spark Open lines. A claim comes after the run is sealed,
    so its lines never go into the run's own open_weights.ndjson (sealed in bundle/manifest_run.json); not served."""
    return _file(claim_dir(run_id) / f"{claim_id}.open_weights.ndjson")


def looks_json(run_id: str) -> Path:
    """derived/looks.json: one row per second look, sorted by (pickup_mark, return_mark)."""
    return _file(derived_dir(run_id) / "looks.json")


def open_weights_log(run_id: str) -> Path:
    """open_weights.ndjson at the run root: one line per Muse Spark Open call on Baseten, append-only provenance."""
    return _file(run_dir(run_id) / "open_weights.ndjson")


def calibration_json(run_id: str) -> Path:
    return _file(derived_dir(run_id) / "calibration.json")


DERIVED_FILES = (
    "tracks.json",
    "panels.json",
    "keyframes.json",
    "marks.json",
    "mark_tracks.json",
    "matches.json",
    "verdicts.json",
    "coverage.json",
    "narration.json",
    "narration_pieces.json",
    "claim.json",
    "calibration.json",
)
FROZEN_DERIVED_FILES = ("tracks.json", "keyframes.json")  # model-era facts rebuild does not regenerate (section 7)


# Sheet and bundle -----------------------------------------------------------------------------------------------


def sheet_dir(run_id: str) -> Path:
    return _dir(run_dir(run_id) / "sheet")


def sheet_json(run_id: str) -> Path:
    return _file(sheet_dir(run_id) / "sheet.json")


def sheet_html(run_id: str) -> Path:
    return _file(sheet_dir(run_id) / "sheet.html")


def sheet_pdf(run_id: str) -> Path:
    return _file(sheet_dir(run_id) / "evidence.pdf")


def bundle_dir(run_id: str) -> Path:
    return _dir(run_dir(run_id) / "bundle")


def manifest_json(run_id: str) -> Path:
    return _file(bundle_dir(run_id) / "manifest.json")


def manifest_run_json(run_id: str) -> Path:
    return _file(bundle_dir(run_id) / "manifest_run.json")


def ots_path(run_id: str, name: str) -> Path:
    """bundle/<name>.ots for manifest.json or an original (name is 'manifest.json', 'pickup', 'return')."""
    return _file(bundle_dir(run_id) / f"{name}.ots")


def tsq_path(run_id: str, name: str = "manifest.json") -> Path:
    return _file(bundle_dir(run_id) / f"{name}.tsq")


def tsr_path(run_id: str, name: str = "manifest.json") -> Path:
    return _file(bundle_dir(run_id) / f"{name}.tsr")


def bundle_file(run_id: str, name: str) -> Path:
    """cacert.pem, tsa.crt, verify.md, letter.md, note.md, evidence.pdf copies."""
    return _file(bundle_dir(run_id) / name)


def bundle_zip(run_id: str) -> Path:
    return _file(sheet_dir(run_id) / "bundle.zip")


# Whole layout ---------------------------------------------------------------------------------------------------


def ensure_run_layout(run_id: str, roles: tuple[str, ...] = ("pickup", "return")) -> Path:
    """Create every standard directory of a run and return the run directory."""
    root = run_dir(run_id)
    quarantine_dir(run_id)
    videos_dir(run_id)
    media_dir(run_id)
    derived_dir(run_id)
    sheet_dir(run_id)
    bundle_dir(run_id)
    claim_dir(run_id)
    for role in roles:
        sam_dir(run_id, role)
        keyframes_dir(run_id, role)
        crops_dir(run_id, role)
        clips_dir(run_id, role)
        chips_dir(run_id, role)
    return root


def rel(run_id: str, path: Path) -> str:
    """Path relative to the run directory, the form every JSON record stores."""
    return str(Path(path).resolve().relative_to(run_dir(run_id).resolve()))
