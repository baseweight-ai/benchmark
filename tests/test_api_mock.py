"""Layer 2 — API mocking: test eval_api.py with openai mocked.

openai and aiohttp are stub-injected into sys.modules so these
tests run without installing those packages.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tests._api_stubs  # noqa: F401 — injects openai/aiohttp/tqdm stubs
import eval_api
from eval_api import TaskConfig, run_eval

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _task_cfg(task_id="fpb", answer_mode="direct", task_type="classification"):
    return TaskConfig(task_id=task_id, max_output_tokens=32,
                      task_type=task_type, answer_mode=answer_mode)


def _setup_prepared_dir(tmp_path: Path, n: int = 5):
    """Write toy test.jsonl and train.jsonl into tmp_path/data/prepared/fpb/."""
    from tests.conftest import make_test_prompts, make_chat_rows, write_jsonl

    prep = tmp_path / "data" / "prepared" / "fpb"
    prep.mkdir(parents=True)
    write_jsonl(make_test_prompts(n), prep / "test.jsonl")
    write_jsonl(make_chat_rows(5), prep / "train.jsonl")
    return prep


# ── eval_api tests ─────────────────────────────────────────────────────────────

def test_run_eval_openai_zero_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def _mock_call(*args, **kwargs):
        return "positive", 100, 10, 0, 150.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=_mock_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-5.4-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    out = tmp_path / "results" / "predictions" / "api" / "gpt-5.4-nano" / "fpb" / "zero-shot.jsonl"
    assert out.exists()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 5
    assert all(r["output"] == "positive" for r in rows)
    assert all(r["model"] == "gpt-5.4-nano" for r in rows)


def test_run_eval_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)

    out = tmp_path / "results" / "predictions" / "api" / "gpt-5.4-nano" / "fpb" / "zero-shot.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"id":"x"}\n')

    call_count = 0

    async def counting_call(*a, **kw):
        nonlocal call_count
        call_count += 1
        return "positive", 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=counting_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-5.4-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    assert call_count == 0


def test_run_eval_resumes_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    out = tmp_path / "results" / "predictions" / "api" / "gpt-5.4-nano" / "fpb" / "zero-shot.jsonl"
    partial = out.with_name(out.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)

    # Pre-write 3 rows as already done
    already_done = [{"id": f"toy_test_{i:04d}", "output": "positive"} for i in range(3)]
    partial.write_text("\n".join(json.dumps(r) for r in already_done) + "\n")

    async def tracking_call(*a, **kw):
        return "negative", 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=tracking_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-5.4-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    # Final file should have all 5 rows (3 from partial + 2 newly run)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 5
    # Partial file must be gone
    assert not partial.exists()


def test_run_eval_dry_run_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)

    asyncio.run(run_eval("gpt-5.4-nano", "fpb", "zero-shot", _task_cfg(), dry_run=True))

    out = tmp_path / "results" / "predictions" / "api" / "gpt-5.4-nano" / "fpb" / "zero-shot.jsonl"
    assert not out.exists()


def test_run_eval_missing_data_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    # No data directory created — should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        asyncio.run(run_eval("gpt-5.4-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))


def test_run_eval_disables_reasoning_for_capable_model(tmp_path, monkeypatch):
    """Reasoning-capable models are sent reasoning_effort='none' so the benchmark
    holds every model to the same (non-reasoning) compute regime."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(eval_api, "REASONING_CAPABLE", {"gpt-5.4-mini": True})
    monkeypatch.setattr(eval_api, "OPENAI_MODELS", {"gpt-5.4-mini": "gpt-5.4-mini"})
    _setup_prepared_dir(tmp_path, n=2)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    received_effort = []

    async def capture_call(client, model_str, messages, max_tokens, semaphore, **kwargs):
        received_effort.append(kwargs.get("reasoning_effort"))
        return "positive", 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=capture_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-5.4-mini", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    assert received_effort and all(e == "none" for e in received_effort)


def test_run_eval_omits_reasoning_for_non_capable_model(tmp_path, monkeypatch):
    """A model flagged reasoning_capable=False is not sent reasoning_effort —
    a non-reasoning model would 400 on the parameter."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(eval_api, "REASONING_CAPABLE", {"gpt-5.4-nano": False})
    monkeypatch.setattr(eval_api, "OPENAI_MODELS", {"gpt-5.4-nano": "gpt-5.4-nano"})
    _setup_prepared_dir(tmp_path, n=2)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    received_effort = []

    async def capture_call(client, model_str, messages, max_tokens, semaphore, **kwargs):
        received_effort.append(kwargs.get("reasoning_effort"))
        return "positive", 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=capture_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-5.4-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    assert received_effort and all(e is None for e in received_effort)


def test_run_eval_records_reasoning_tokens_per_row(tmp_path, monkeypatch):
    """reasoning_tokens must land in every prediction row, even when zero, so
    the dashboard can always decompose output_tokens."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=3)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def call(client, model_str, messages, max_tokens, semaphore, **kwargs):
        return "positive", 100, 12, 7, 100.0, 50.0, None  # 7 reasoning tokens

    with patch("eval_api.call_openai", side_effect=call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-5.4-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    out = tmp_path / "results" / "predictions" / "api" / "gpt-5.4-nano" / "fpb" / "zero-shot.jsonl"
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert all(r["reasoning_tokens"] == 7 for r in rows)
    assert all(r["output_tokens"] == 12 for r in rows)


def test_run_eval_5shot_builds_messages(tmp_path, monkeypatch):
    """Verify that 5-shot condition actually passes few-shot context."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=2)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    captured_messages = []

    async def capture_call(client, model_str, messages, max_tokens, semaphore, **kw):
        captured_messages.append(messages)
        return "positive", 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=capture_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-5.4-nano", "fpb", "5-shot", _task_cfg(), dry_run=False))

    assert len(captured_messages) == 2
    # 5-shot: should have more than 2 messages (system + few-shot turns + user)
    assert len(captured_messages[0]) > 2


# ── response_format constrained decoding ───────────────────────────────────────

def test_run_eval_constrained_decoding_parses_json_label(tmp_path, monkeypatch):
    """For a direct classification task with labels.json, eval_api sends a
    response_format pinned to that label set and unwraps {"label": X} back to
    the bare label X — matching the local (guided_choice) output shape."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    prep = _setup_prepared_dir(tmp_path, n=3)
    (prep / "labels.json").write_text(json.dumps(["positive", "negative", "neutral"]))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    received_rf = []

    async def capture_call(client, model_str, messages, max_tokens, semaphore, **kwargs):
        received_rf.append(kwargs.get("response_format"))
        return '{"label": "negative"}', 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=capture_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-5.4-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    # response_format was passed and constrains to the labels.json set
    assert received_rf and all(rf is not None for rf in received_rf)
    enum = received_rf[0]["json_schema"]["schema"]["properties"]["label"]["enum"]
    assert set(enum) == {"positive", "negative", "neutral"}
    # The stored output is the bare label, not the JSON envelope
    out = tmp_path / "results" / "predictions" / "api" / "gpt-5.4-nano" / "fpb" / "zero-shot.jsonl"
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert all(r["output"] == "negative" for r in rows)


def test_run_eval_tagged_task_is_unconstrained(tmp_path, monkeypatch):
    """answer_mode='tagged' tasks (CoT) get NO response_format — a free-form
    chain-of-thought cannot be pinned to a label set."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    prep = _setup_prepared_dir(tmp_path, n=2)
    (prep / "labels.json").write_text(json.dumps(["A", "B", "C", "D"]))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    received_rf = []

    async def capture_call(client, model_str, messages, max_tokens, semaphore, **kwargs):
        received_rf.append(kwargs.get("response_format"))
        return "<thinking>x</thinking><answer>A</answer>", 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=capture_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-5.4-nano", "fpb", "zero-shot",
                                 _task_cfg(answer_mode="tagged"), dry_run=False))

    assert received_rf and all(rf is None for rf in received_rf)
