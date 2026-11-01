"""config: redact replaces every configured key and the key shapes (Meta's LLM_, hf_, Baseten's b10_ keys), an empty
value in an env file never hides a later file's value, and .env.example names the hackathon key with no value."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from walkaround import config

REPO = Path(__file__).resolve().parent.parent
B10 = "b10_" + "Ab3dE6gH" + "." + "0123456789abcdefABCDEF0123456789"  # the shape of a Baseten key, not a real one


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in config._SECRET_ENV:
        monkeypatch.setenv(name, "")  # records the state before the test, so whatever load_env sets here is undone
        monkeypatch.delenv(name)


def test_a_baseten_shaped_key_is_redacted_wherever_it_sits(no_keys: None) -> None:
    for text in (f"Incorrect API key provided: {B10}.", f"key={B10}", f"({B10})", f"{B10}extra"):
        out = config.redact(text)
        assert "b10_" not in out and "[redacted]" in out
    assert config.redact(f"Bearer {B10}") == "Bearer [redacted]"


def test_other_text_is_left_alone(no_keys: None) -> None:
    for text in ("meta-models/muse-spark-open-1026", "b10_short.abc", "a" * 64, "chatcmpl-0123456789abcdef", "sha256 0f" + "e" * 62):
        assert config.redact(text) == text


def test_the_hackathon_key_is_a_configured_secret(no_keys: None, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "BASETEN_HACKATHON_API_KEY" in config._SECRET_ENV and "BASETEN_API_KEY" in config._SECRET_ENV
    monkeypatch.setenv("BASETEN_HACKATHON_API_KEY", "plain-secret-not-shaped-42")
    assert config.redact("sent plain-secret-not-shaped-42 to Baseten") == "sent [redacted] to Baseten"


def test_an_empty_value_in_dot_env_does_not_hide_the_keys_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_keys: None) -> None:
    repo, home = tmp_path / "repo", tmp_path / "home"
    (home / "meta2026").mkdir(parents=True)
    repo.mkdir()
    (repo / ".env").write_text("MODEL_API_KEY=LLM_1_from_dot_env\nBASETEN_HACKATHON_API_KEY=\nBASETEN_API_KEY=''\n", encoding="utf-8")
    (home / "meta2026" / ".keys").write_text("export BASETEN_HACKATHON_API_KEY=from_keys_file\nMODEL_API_KEY=LLM_2_from_keys\n", encoding="utf-8")
    monkeypatch.setattr(config, "REPO_DIR", repo)
    monkeypatch.setenv("HOME", str(home))
    config.load_env()
    assert os.environ["BASETEN_HACKATHON_API_KEY"] == "from_keys_file"  # the empty line in .env set nothing
    assert os.environ["MODEL_API_KEY"] == "LLM_1_from_dot_env"  # .env still comes first
    assert "BASETEN_API_KEY" not in os.environ  # '' is empty too


def test_env_example_names_the_hackathon_key_with_no_value() -> None:
    lines = (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
    assert "BASETEN_HACKATHON_API_KEY=" in lines
    assert all(line.isascii() for line in lines)
