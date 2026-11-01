"""scripts/second_look_again.py on the committed evidence (docs/evidence/second_look), offline: the stored looks find
their crops by sha256, nothing is asked without LIVE=1, and with LIVE=1 a fake Baseten client gets one request per look
with both crops, the run id second_look_again goes on the ledger rows and provenance lines, and the stored answer is
printed beside the new one."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from walkaround import config, ledger, open_weights, paths

REPO = Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "docs" / "evidence" / "second_look"
MODEL = "meta-models/muse-spark-open"


def script():
    spec = importlib.util.spec_from_file_location("second_look_again", REPO / "scripts" / "second_look_again.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["second_look_again"] = module
    spec.loader.exec_module(module)
    return module


class FakeChat:
    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = list(verdicts)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        text = json.dumps({"verdict": self.verdicts.pop(0), "reason": "One plain sentence from the fake."})
        return {
            "id": f"chatcmpl-{len(self.requests)}",
            "object": "chat.completion",
            "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 1800, "completion_tokens": 200, "total_tokens": 2000, "completion_tokens_details": {"reasoning_tokens": 150}},
        }


class FakeBaseten:
    def __init__(self, verdicts: list[str]) -> None:
        self.models = SimpleNamespace(list=lambda: {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        self.chat = SimpleNamespace(completions=FakeChat(verdicts))


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    monkeypatch.setattr(open_weights, "_sleep", lambda s: None)
    return root


def test_the_evidence_holds_three_looks_with_their_crops() -> None:
    todo = script().plan(EVIDENCE)
    assert sorted((row["return_mark"], row["verdict"]) for row, _ in todo) == [("r_leaf", "different"), ("r_lowleft", "different"), ("r_paw", "same")]
    assert all(crops[0].name == "pickup_p_paw.look.jpg" and crops[1].name.startswith("return_") for _, crops in todo)


def test_without_live_nothing_is_asked(runs_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("LIVE", raising=False)
    assert script().main([str(EVIDENCE)]) == 2
    assert "would ask trunk_lid: p_paw vs r_paw" in capsys.readouterr().out
    assert not (runs_dir / "ledger.jsonl").exists()


def test_live_asks_each_look_once_and_logs_it(runs_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LIVE", "1")
    monkeypatch.setenv("BASETEN_API_KEY", "bt-test-key-0123456789abcdef")
    client = FakeBaseten(["different", "different", "same"])
    open_weights.reset()
    open_weights.set_client(client)
    try:
        assert script().main([str(EVIDENCE)]) == 0
    finally:
        open_weights.set_client(None)
        open_weights.reset()
    out = capsys.readouterr().out
    assert "p_paw vs r_paw: now same (meta-models/muse-spark-open), stored same" in out
    assert out.count("  hackathon key, ") == 3  # the slot that asked and the latency, under each answer
    requests = client.chat.completions.requests
    assert len(requests) == 3 and {r["model"] for r in requests} == {"meta-models/muse-spark-open-1026"}
    images = [part for part in requests[0]["messages"][-1]["content"] if part.get("type") == "image_url"]
    assert len(images) == 2
    rows = [json.loads(line) for line in (runs_dir / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and {r["run_id"] for r in rows} == {"second_look_again"} and {r["kind"] for r in rows} == {"spark_open"}
    lines = [json.loads(line) for line in (runs_dir / "second_look_again" / "open_weights.ndjson").read_text().splitlines()]
    assert [(line["pickup_mark"], line["return_mark"], line["verdict"]) for line in lines] == [("p_paw", "r_leaf", "different"), ("p_paw", "r_lowleft", "different"), ("p_paw", "r_paw", "same")]
