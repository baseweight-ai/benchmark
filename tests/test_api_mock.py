"""Layer 2 — API mocking: test eval_api.py with openai mocked.

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
from eval_api import TaskConfig, run_eval, run_sft

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _task_cfg(task_id="fpb"):
    return TaskConfig(task_id=task_id, max_output_tokens=32, task_type="classification")


def _setup_prepared_dir(tmp_path: Path, n: int = 5):
    """Write toy test.jsonl and few_shot_5.jsonl into tmp_path/data/prepared/fpb/."""
    from tests.conftest import make_test_prompts, make_chat_rows, write_jsonl

    prep = tmp_path / "data" / "prepared" / "fpb"
    prep.mkdir(parents=True)
    write_jsonl(make_test_prompts(n), prep / "test.jsonl")
    write_jsonl(make_chat_rows(5), prep / "few_shot_5.jsonl")
    return prep


# ── Mock call_* functions directly ────────────────────────────────────────────

def _mock_call(response_text="positive"):
    """Return an async mock that behaves like call_openai."""
    async def _fn(*args, **kwargs):
        return response_text, 100, 10, 150.0
    return _fn


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_run_eval_openai_zero_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("eval_api.call_openai", side_effect=_mock_call("positive")):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-4.1", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    out = tmp_path / "results" / "predictions" / "gpt-4.1" / "fpb" / "zero-shot.jsonl"
    assert out.exists()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 5
    assert all(r["output"] == "positive" for r in rows)
    assert all(r["model"] == "gpt-4.1" for r in rows)



def test_run_eval_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)

    out = tmp_path / "results" / "predictions" / "gpt-4.1" / "fpb" / "zero-shot.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"id":"x"}\n')

    call_count = 0

    async def counting_call(*a, **kw):
        nonlocal call_count
        call_count += 1
        return "positive", 10, 5, 100.0

    with patch("eval_api.call_openai", side_effect=counting_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-4.1", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    assert call_count == 0


def test_run_eval_resumes_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    out = tmp_path / "results" / "predictions" / "gpt-4.1" / "fpb" / "zero-shot.jsonl"
    partial = out.with_name(out.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)

    # Pre-write 3 rows as already done
    already_done = [{"id": f"toy_test_{i:04d}", "output": "positive"} for i in range(3)]
    partial.write_text("\n".join(json.dumps(r) for r in already_done) + "\n")

    called_ids = []

    async def tracking_call(*a, **kw):
        return "negative", 10, 5, 100.0

    with patch("eval_api.call_openai", side_effect=tracking_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-4.1", "fpb", "zero-shot", _task_cfg(), dry_run=False))

    # Final file should have all 5 rows (3 from partial + 2 newly run)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 5
    # Partial file must be gone
    assert not partial.exists()


def test_run_eval_dry_run_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=5)

    asyncio.run(run_eval("gpt-4.1", "fpb", "zero-shot", _task_cfg(), dry_run=True))

    out = tmp_path / "results" / "predictions" / "gpt-4.1" / "fpb" / "zero-shot.jsonl"
    assert not out.exists()


def test_run_eval_missing_data_skips_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    # No data directory created — should skip without crashing
    asyncio.run(run_eval("gpt-4.1", "fpb", "zero-shot", _task_cfg(), dry_run=False))


def test_run_eval_5shot_builds_messages(tmp_path, monkeypatch):
    """Verify that 5-shot condition actually passes few-shot context."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=2)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    captured_messages = []

    async def capture_call(client, model_str, messages, max_tokens, semaphore):
        captured_messages.append(messages)
        return "positive", 10, 5, 100.0

    with patch("eval_api.call_openai", side_effect=capture_call):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            asyncio.run(run_eval("gpt-4.1", "fpb", "5-shot", _task_cfg(), dry_run=False))

    assert len(captured_messages) == 2
    # 5-shot: should have more than 2 messages (system + few-shot turns + user)
    assert len(captured_messages[0]) > 2


def test_run_sft_uses_cached_ft_model(tmp_path, monkeypatch):
    """If metadata.json already has ft_model_id, skip job creation."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", tmp_path)
    _setup_prepared_dir(tmp_path, n=3)

    # Pre-write the SFT training data
    sft_path = tmp_path / "data" / "prepared" / "fpb" / "openai_sft_500.jsonl"
    sft_path.write_text('{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}\n' * 3)

    # Pre-write metadata so it skips the fine-tuning job
    meta_path = tmp_path / "results" / "training" / "gpt-4.1-sft" / "fpb" / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"ft_model_id": "ft:gpt-4.1:test:abc123"}))

    # Patch OPENAI_MODELS so gpt-4.1-sft maps to the fake ft model
    with patch("eval_api.call_openai", side_effect=_mock_call("positive")):
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            with patch("openai.OpenAI", return_value=MagicMock()):
                asyncio.run(run_sft("fpb", _task_cfg(), dry_run=False))

    out = tmp_path / "results" / "predictions" / "gpt-4.1-sft" / "fpb" / "api-sft-500.jsonl"
    assert out.exists()
