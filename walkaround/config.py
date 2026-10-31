"""Model ids, prices and paths. Prices are from docs/API_NOTES.md in the private workspace
(https://dev.meta.ai/docs/pricing-rate-limits, read 2026-10-26)."""

from __future__ import annotations

import os
import re
from pathlib import Path

BASE_URL = "https://api.meta.ai/v1"
SPARK_MODEL = "muse-spark-1.3"
SAM_MODEL = "sam-3.1"
VOICE_MODEL = "muse-voice-transcribe-1.0"
VOICE_TRANSCRIBE_URL = "https://api.meta.ai/v1/asr/transcribe"

# Dollars. Token prices are per token (price per million / 1e6).
PRICES = {
    "spark_input_token": 1.25 / 1e6,
    "spark_output_token": 4.25 / 1e6,
    "spark_cached_token": 0.15 / 1e6,
    "sam_video_frame": 0.20 / 1000,
    "sam_video_min_frames": 50,
    "sam_image": 2.50 / 1000,
    "voice_second": 0.18 / 3600,
}

REPO_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("WALKAROUND_RUNS", REPO_DIR / "runs"))

# Car panels. One SAM request per noun. The last two are only used for blurring.
CAR_PANELS = [
    "car door",
    "front bumper",
    "rear bumper",
    "hood",
    "trunk lid",
    "roof",
    "wheel",
    "headlight",
    "tail light",
    "side mirror",
    "windshield",
    "rear window",
]
PRIVACY_NOUNS = ["license plate", "face"]

# Apartment mode is the same engine with a different noun list.
ROOM_PANELS = [
    "wall",
    "floor",
    "door",
    "window",
    "countertop",
    "cabinet",
    "carpet",
    "ceiling",
    "appliance",
]


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_env() -> None:
    """Read walkaround/.env, then ~/meta2026/.keys. Existing environment wins."""
    _load_env_file(REPO_DIR / ".env")
    _load_env_file(Path.home() / "meta2026" / ".keys")


_KEY_SHAPES = re.compile(r"\bLLM[_|][A-Za-z0-9_|-]{6,}|\bhf_[A-Za-z0-9]{10,}")  # LLM_{digits}_{secret}, HF tokens
_BEARER = re.compile(r"(?i)\b(bearer\s+)\S+")
_SECRET_ENV = ("MODEL_API_KEY", "BASETEN_API_KEY", "HF_TOKEN")


def redact(text: str) -> str:
    """Text with every configured key value and key-shaped token replaced, before it is stored, emitted or served."""
    for name in _SECRET_ENV:
        value = os.environ.get(name) or ""
        if len(value) >= 8:
            text = text.replace(value, "[redacted]")
    return _BEARER.sub(r"\1[redacted]", _KEY_SHAPES.sub("[redacted]", text))


def api_key() -> str:
    load_env()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set. Put it in walkaround/.env (see .env.example).")
    return key
