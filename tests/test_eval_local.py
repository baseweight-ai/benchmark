"""Tests for eval_local.py — dry-run, config loading, and data helpers (no GPU/vLLM needed)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import tests._api_stubs  # noqa: F401 — stubs aiohttp/tqdm
import eval_local
from eval_local import (
    ModelConfig,
    TaskConfig,
    get_few_shot,
    load_model_config,
    load_task_config,
    load_test_rows,
    run_eval,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_test_data(root: Path, task_id: str = "fpb", n: int = 5, with_labels: bool = False) -> None:
    prep = root / "data" / "prepared" / task_id
    prep.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": f"{task_id}_test_{i:04d}", "messages": [{"role": "user", "content": f"Q{i}"}]}
        for i in range(n)
    ]
    with open(prep / "test.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if with_labels:
        label_rows = [
            {"id": r["id"], "label": ["positive", "negative", "neutral"][i % 3]}
            for i, r in enumerate(rows)
        ]
        with open(prep / "test_labels.jsonl", "w") as f:
            for r in label_rows:
                f.write(json.dumps(r) + "\n")


def _write_train_data(root: Path, task_id: str = "fpb", n: int = 8) -> None:
    prep = root / "data" / "prepared" / task_id
    prep.mkdir(parents=True, exist_ok=True)
    rows = [
        {"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": "A"}]}
        for i in range(n)
    ]
    with open(prep / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _model_cfg(model_short: str = "qwen2.5-0.5b") -> ModelConfig:
    return ModelConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_short=model_short,
        max_seq_length=256,
    )


def _task_cfg(task_id: str = "fpb") -> TaskConfig:
    return TaskConfig(task_id=task_id, max_output_tokens=32, task_type="classification")


# ── Config loading ──────────────────────────────────────────────────────────────

def test_load_task_config_fpb():
    cfg = load_task_config("fpb")
    assert cfg.task_id == "fpb"
    assert cfg.max_output_tokens > 0
    assert cfg.task_type in ("classification", "extraction", "code")


def test_load_task_config_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_task_config("nonexistent_task_xyz")


def test_load_model_config_smoke_default():
    """qwen2.5-0.5b is the --smoke-test default model; its config must exist and parse."""
    cfg = load_model_config("qwen2.5-0.5b")
    assert cfg.model_short == "qwen2.5-0.5b"
    assert cfg.max_seq_length > 0


def test_load_model_config_all_prod_models():
    """All models in ALL_MODELS must have a training config. Fails if a config is missing."""
    missing = []
    for mid in eval_local.ALL_MODELS:
        path = REPO_ROOT / "configs" / "training" / f"{mid}.yaml"
        if not path.exists():
            missing.append(mid)
    assert not missing, f"Missing training configs for: {missing}"


# ── Data helpers ────────────────────────────────────────────────────────────────

def test_load_test_rows_basic(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=4)
    rows = load_test_rows("fpb", smoke_test=False)
    assert len(rows) == 4
    assert all("id" in r for r in rows)


def test_load_test_rows_joins_labels(tmp_path, monkeypatch):
    """Labels from a separate *_labels.jsonl are joined onto each row by id."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=4, with_labels=True)
    rows = load_test_rows("fpb", smoke_test=False)
    assert all("label" in r for r in rows)
    assert all(r["label"] in ("positive", "negative", "neutral") for r in rows)


def test_load_test_rows_no_labels_file(tmp_path, monkeypatch):
    """Missing labels file is handled gracefully — rows load without a label field."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=3, with_labels=False)
    rows = load_test_rows("fpb", smoke_test=False)
    assert len(rows) == 3


def test_get_few_shot_returns_at_most_five(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_train_data(tmp_path, n=10)
    rows = get_few_shot("fpb", "qwen2.5-0.5b", smoke_test=False)
    assert len(rows) == 5


def test_get_few_shot_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    rows = get_few_shot("fpb", "qwen2.5-0.5b", smoke_test=False)
    assert rows == []


def test_get_few_shot_fewer_than_five_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_train_data(tmp_path, n=3)
    rows = get_few_shot("fpb", "qwen2.5-0.5b", smoke_test=False)
    assert len(rows) == 3


# ── run_eval dry-run ────────────────────────────────────────────────────────────

def test_run_eval_dry_run_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=5)

    asyncio.run(run_eval(
        _model_cfg(), "fpb", "zero-shot", _task_cfg(),
        model_name="Qwen/Qwen2.5-0.5B-Instruct", dry_run=True,
    ))

    out = tmp_path / "results" / "predictions" / "local" / "qwen2.5-0.5b" / "fpb" / "zero-shot.jsonl"
    assert not out.exists()


def test_run_eval_dry_run_missing_data_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError):
        asyncio.run(run_eval(
            _model_cfg(), "fpb", "zero-shot", _task_cfg(),
            model_name="Qwen/Qwen2.5-0.5B-Instruct", dry_run=True,
        ))


def test_run_eval_skips_existing(tmp_path, monkeypatch):
    """If the output file already exists, run_eval returns without making any API calls."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=5)

    out = tmp_path / "results" / "predictions" / "local" / "qwen2.5-0.5b" / "fpb" / "zero-shot.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"id":"existing"}\n')

    asyncio.run(run_eval(
        _model_cfg(), "fpb", "zero-shot", _task_cfg(),
        model_name="Qwen/Qwen2.5-0.5B-Instruct", dry_run=False,
    ))

    assert out.read_text() == '{"id":"existing"}\n'


def test_run_eval_dry_run_all_tasks(tmp_path, monkeypatch):
    """dry-run succeeds for every registered task (catches missing/broken task configs)."""
    # Load real configs before redirecting REPO_ROOT to tmp_path
    task_cfgs = {tid: load_task_config(tid) for tid in eval_local.ALL_TASKS}
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    for tid, cfg in task_cfgs.items():
        _write_test_data(tmp_path, task_id=tid, n=2)
        asyncio.run(run_eval(
            _model_cfg(), tid, "zero-shot", cfg,
            model_name="Qwen/Qwen2.5-0.5B-Instruct", dry_run=True,
        ))
