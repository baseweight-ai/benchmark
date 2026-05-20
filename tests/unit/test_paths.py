"""Unit tests for pipeline.paths — output paths must namespace smoke runs so
they never clobber real-run outputs; read-only paths must stay shared."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path("/repo")


def test_pred_path_real_vs_smoke():
    from pipeline.paths import pred_path
    assert pred_path(ROOT, "local", "qwen3-8b", "fpb", "lora") == \
        Path("/repo/results/predictions/local/qwen3-8b/fpb/lora.jsonl")
    assert pred_path(ROOT, "local", "qwen3-8b", "fpb", "lora", smoke=True) == \
        Path("/repo/results/smoke/predictions/local/qwen3-8b/fpb/lora.jsonl")


def test_classified_path_smoke_namespaces():
    from pipeline.paths import classified_path
    assert classified_path(ROOT, "api", "gpt-5.4-mini", "fpb", "5-shot", smoke=True) == \
        Path("/repo/results/smoke/classified/api/gpt-5.4-mini/fpb/5-shot.jsonl")


def test_summary_path_smoke_namespaces():
    from pipeline.paths import summary_path
    assert summary_path(ROOT, "local", "qwen3-8b", "fpb", "lora", smoke=True) == \
        Path("/repo/results/smoke/summaries/local/qwen3-8b/fpb/lora.json")


def test_training_meta_path_smoke_namespaces():
    from pipeline.paths import training_meta_path
    assert training_meta_path(ROOT, "local", "qwen3-8b", "fpb", "lora", smoke=True) == \
        Path("/repo/results/smoke/training/local/qwen3-8b/fpb/lora/metadata.json")


def test_adapter_path_smoke_namespaces():
    from pipeline.paths import adapter_path
    assert adapter_path(ROOT, "qwen3-8b", "fpb", "lora", smoke=True) == \
        Path("/repo/results/smoke/adapters/local/qwen3-8b/fpb/lora")


def test_prepared_path_smoke_namespaces():
    from pipeline.paths import prepared_path
    assert prepared_path(ROOT, "fpb") == Path("/repo/data/prepared/fpb")
    assert prepared_path(ROOT, "fpb", smoke=True) == Path("/repo/data/smoke/prepared/fpb")


def test_raw_path_smoke_namespaces():
    """Smoke and non-smoke raw datasets live in separate trees — a truncated
    smoke download must never clobber the real raw artifact."""
    from pipeline.paths import raw_path
    assert raw_path(ROOT, "fpb") == Path("/repo/data/raw/fpb")
    assert raw_path(ROOT, "fpb", smoke=True) == Path("/repo/data/smoke/raw/fpb")


def test_snapshot_path_smoke_namespaces():
    from pipeline.paths import snapshot_path
    assert snapshot_path(ROOT, "run-123", smoke=True) == \
        Path("/repo/results/smoke/snapshots/run-123/results.json")


def test_run_manifest_path_smoke_namespaces():
    from pipeline.paths import run_manifest_path
    assert run_manifest_path(ROOT, "run-123") == Path("/repo/runs/run-123.json")
    assert run_manifest_path(ROOT, "run-123", smoke=True) == Path("/repo/runs/smoke/run-123.json")


def test_test_full_path_smoke_namespaces():
    from pipeline.paths import test_full_path
    assert test_full_path(ROOT, "fpb") == Path("/repo/data/prepared/fpb/test_full.jsonl")
    assert test_full_path(ROOT, "fpb", smoke=True) == \
        Path("/repo/data/smoke/prepared/fpb/smoke_test_full.jsonl")


def test_config_paths_are_not_smoke_namespaced():
    """Read-only paths (configs, prompts) stay shared across smoke and real."""
    from pipeline.paths import task_config_path, model_config_path, prompt_path
    assert task_config_path(ROOT, "fpb") == Path("/repo/configs/tasks/fpb.yaml")
    assert model_config_path(ROOT, "qwen3-8b") == Path("/repo/configs/training/qwen3-8b.yaml")
    assert prompt_path(ROOT, "fpb") == Path("/repo/prompts/fpb.json")


def test_checkpoint_dir_smoke_namespaces():
    from checkpoint_utils import checkpoint_dir
    real = checkpoint_dir("qwen3-8b", "fpb", "lora")
    smoke = checkpoint_dir("qwen3-8b", "fpb", "lora", smoke=True)
    assert "/checkpoints/smoke/" in str(smoke)
    assert "/checkpoints/smoke/" not in str(real)
