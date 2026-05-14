"""Layer 2 — API mocking: test eval_api.py and train_api.py with openai mocked.

openai and aiohttp are stub-injected into sys.modules so these
tests run without installing those packages.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tests._api_stubs  # noqa: F401 — injects openai/aiohttp/tqdm stubs
import eval_api
import train_api
from eval_api import TaskConfig, run_eval

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _task_cfg(task_id="fpb"):
    return TaskConfig(task_id=task_id, max_output_tokens=32, task_type="classification")


def _setup_prepared_dir(tmp_path: Path, n: int = 5):
    """Write toy test.jsonl and train.jsonl into tmp_path/data/prepared/fpb/."""
    from tests.conftest import make_test_prompts, make_chat_rows, write_jsonl

    prep = tmp_path / "data" / "prepared" / "fpb"
    prep.mkdir(parents=True)
    write_jsonl(make_test_prompts(n), prep / "test.jsonl")
    write_jsonl(make_chat_rows(5), prep / "train.jsonl")
    return prep


def _write_sft_metadata(tmp_path: Path, model_id: str = "gpt-4.1-nano", task_id: str = "fpb",
                         ft_model_id: str = "ft:gpt-4.1-nano-2025-04-14:test:abc123"):
    meta = tmp_path / "results" / "training" / "api" / model_id / task_id / "api-sft" / "metadata.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"ft_model_id": ft_model_id, "trained_tokens": 1000,
                                "training_cost": 0.025, "n_train": 5}))
    return meta


# ── Mock call_* functions directly ────────────────────────────────────────────

def _mock_call(response_text="positive"):
    """Return an async mock that behaves like call_openai.

    Tuple shape: (text, input_tokens, output_tokens, reasoning_tokens, latency_ms,
                  ttft_ms, avg_logprob).
    reasoning_tokens=0 because the benchmark policy disables reasoning everywhere.
    avg_logprob=None because mocked calls don't simulate logprobs.
    """
    async def _fn(*args, **kwargs):
        return response_text, 100, 10, 0, 150.0, 50.0, None
    return _fn


# ── eval_api tests ─────────────────────────────────────────────────────────────

def test_run_eval_openai_zero_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("eval_api.call_openai", side_effect=_mock_call("positive")):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-4.1-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    out = tmp_path / "results" / "predictions" / "api" / "gpt-4.1-nano" / "fpb" / "zero-shot.jsonl"
    assert out.exists()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 5
    assert all(r["output"] == "positive" for r in rows)
    assert all(r["model"] == "gpt-4.1-nano" for r in rows)


def test_run_eval_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)

    out = tmp_path / "results" / "predictions" / "api" / "gpt-4.1-nano" / "fpb" / "zero-shot.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"id":"x"}\n')

    call_count = 0

    async def counting_call(*a, **kw):
        nonlocal call_count
        call_count += 1
        return "positive", 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=counting_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-4.1-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    assert call_count == 0


def test_run_eval_resumes_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    out = tmp_path / "results" / "predictions" / "api" / "gpt-4.1-nano" / "fpb" / "zero-shot.jsonl"
    partial = out.with_name(out.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)

    # Pre-write 3 rows as already done
    already_done = [{"id": f"toy_test_{i:04d}", "output": "positive"} for i in range(3)]
    partial.write_text("\n".join(json.dumps(r) for r in already_done) + "\n")

    async def tracking_call(*a, **kw):
        return "negative", 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=tracking_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-4.1-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    # Final file should have all 5 rows (3 from partial + 2 newly run)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 5
    # Partial file must be gone
    assert not partial.exists()


def test_run_eval_dry_run_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)

    asyncio.run(run_eval("gpt-4.1-nano", "fpb", "zero-shot", _task_cfg(), dry_run=True))

    out = tmp_path / "results" / "predictions" / "api" / "gpt-4.1-nano" / "fpb" / "zero-shot.jsonl"
    assert not out.exists()


def test_run_eval_missing_data_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    # No data directory created — should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        asyncio.run(run_eval("gpt-4.1-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))


def test_run_eval_disables_reasoning_for_capable_model(tmp_path, monkeypatch):
    """Reasoning-capable models (e.g. gpt-5.5) must be sent reasoning_effort=minimal
    so the benchmark stays apples-to-apples with non-reasoning models."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(eval_api, "REASONING_CAPABLE", {"gpt-5.5": True, "gpt-4.1-nano": False})
    _setup_prepared_dir(tmp_path, n=2)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    received_effort = []

    async def capture_call(client, model_str, messages, max_tokens, semaphore, **kwargs):
        received_effort.append(kwargs.get("reasoning_effort"))
        return "positive", 10, 5, 0, 100.0, 50.0, None

    # gpt-5.5 is reasoning-capable → effort should be "minimal"
    monkeypatch.setattr(eval_api, "OPENAI_MODELS", {"gpt-5.5": "gpt-5.5", "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14"})
    with patch("eval_api.call_openai", side_effect=capture_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-5.5", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    assert received_effort and all(e == "minimal" for e in received_effort)


def test_run_eval_omits_reasoning_for_non_capable_model(tmp_path, monkeypatch):
    """Non-reasoning models (gpt-4.1) must not receive the reasoning_effort param —
    OpenAI would 400 on it."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(eval_api, "REASONING_CAPABLE", {"gpt-4.1-nano": False})
    _setup_prepared_dir(tmp_path, n=2)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    received_effort = []

    async def capture_call(client, model_str, messages, max_tokens, semaphore, **kwargs):
        received_effort.append(kwargs.get("reasoning_effort"))
        return "positive", 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=capture_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-4.1-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    assert received_effort and all(e is None for e in received_effort)


def test_run_eval_records_reasoning_tokens_per_row(tmp_path, monkeypatch):
    """reasoning_tokens must land in every prediction row, even when zero, so
    the dashboard can always decompose output_tokens."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(eval_api, "REASONING_CAPABLE", {"gpt-4.1-nano": False})
    _setup_prepared_dir(tmp_path, n=3)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def call(client, model_str, messages, max_tokens, semaphore, **kwargs):
        return "positive", 100, 12, 7, 100.0, 50.0, None  # 7 reasoning tokens

    with patch("eval_api.call_openai", side_effect=call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-4.1-nano", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    out = tmp_path / "results" / "predictions" / "api" / "gpt-4.1-nano" / "fpb" / "zero-shot.jsonl"
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
            asyncio.run(run_eval("gpt-4.1-nano", "fpb", "5-shot", _task_cfg(), dry_run=False))

    assert len(captured_messages) == 2
    # 5-shot: should have more than 2 messages (system + few-shot turns + user)
    assert len(captured_messages[0]) > 2


# ── api-sft condition tests ────────────────────────────────────────────────────

def test_run_eval_api_sft_reads_ft_model_from_metadata(tmp_path, monkeypatch):
    """run_eval with api-sft reads ft_model_id from training metadata and uses it."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=3)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _write_sft_metadata(tmp_path, ft_model_id="ft:gpt-4.1-nano-2025-04-14:test:abc123")

    used_model_strs = []

    async def capture_call(client, model_str, messages, max_tokens, semaphore, **kw):
        used_model_strs.append(model_str)
        return "positive", 10, 5, 0, 100.0, 50.0, None

    with patch("eval_api.call_openai", side_effect=capture_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-4.1-nano", "fpb", "api-sft", _task_cfg(), dry_run=False))

    out = tmp_path / "results" / "predictions" / "api" / "gpt-4.1-nano" / "fpb" / "api-sft.jsonl"
    assert out.exists()
    assert len(used_model_strs) == 3
    # The fine-tuned model string, not the base model, must be used for inference
    assert all(s == "ft:gpt-4.1-nano-2025-04-14:test:abc123" for s in used_model_strs)


def test_run_eval_api_sft_skips_if_no_metadata(tmp_path, monkeypatch):
    """run_eval with api-sft skips gracefully when no training metadata exists."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=3)
    # No metadata written

    asyncio.run(run_eval("gpt-4.1-nano", "fpb", "api-sft", _task_cfg(), dry_run=False))

    out = tmp_path / "results" / "predictions" / "api" / "gpt-4.1-nano" / "fpb" / "api-sft.jsonl"
    assert not out.exists()


# ── train_api tests ────────────────────────────────────────────────────────────

def test_run_sft_train_skips_if_metadata_exists(tmp_path, monkeypatch):
    """run_sft_train skips immediately if metadata.json already exists."""
    monkeypatch.setattr(train_api, "REPO_ROOT", tmp_path)

    sft_path = tmp_path / "data" / "prepared" / "fpb" / "train.jsonl"
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.write_text('{"messages":[]}\n')

    meta = _write_sft_metadata(tmp_path, ft_model_id="ft:gpt-4.1-nano-2025-04-14:org:existing")
    original_mtime = meta.stat().st_mtime

    # Should return without touching openai — skip happens before the lazy import
    train_api.run_sft_train("gpt-4.1-nano", "fpb", dry_run=False, smoke_test=False, force=False)

    # Metadata must be unchanged (not overwritten)
    assert meta.stat().st_mtime == original_mtime
    assert json.loads(meta.read_text())["ft_model_id"] == "ft:gpt-4.1-nano-2025-04-14:org:existing"


def test_run_sft_train_dry_run_no_files_created(tmp_path, monkeypatch):
    """run_sft_train in dry-run mode never creates metadata or calls the API."""
    monkeypatch.setattr(train_api, "REPO_ROOT", tmp_path)

    sft_path = tmp_path / "data" / "prepared" / "fpb" / "train.jsonl"
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.write_text('{"messages":[]}\n')

    train_api.run_sft_train("gpt-4.1-nano", "fpb", dry_run=True, smoke_test=False, force=False)

    meta = tmp_path / "results" / "training" / "api" / "gpt-4.1-nano" / "fpb" / "api-sft" / "metadata.json"
    assert not meta.exists()
