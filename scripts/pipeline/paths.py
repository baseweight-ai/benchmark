"""Centralized path registry. All functions take root: Path as first argument
so callers can pass REPO_ROOT and tests can monkeypatch just that one variable."""
from __future__ import annotations
from pathlib import Path


def pred_path(root: Path, source: str, model_short: str, task_id: str, condition: str) -> Path:
    return root / "results" / "predictions" / source / model_short / task_id / f"{condition}.jsonl"


def classified_path(root: Path, source: str, model_short: str, task_id: str, condition: str) -> Path:
    return root / "results" / "classified" / source / model_short / task_id / f"{condition}.jsonl"


def summary_path(root: Path, source: str, model_short: str, task_id: str, condition: str) -> Path:
    return root / "results" / "summaries" / source / model_short / task_id / f"{condition}.json"


def training_meta_path(root: Path, source: str, model_id: str, task_id: str, condition: str) -> Path:
    return root / "results" / "training" / source / model_id / task_id / condition / "metadata.json"


def adapter_path(root: Path, model_short: str, task_id: str, condition: str) -> Path:
    return root / "results" / "adapters" / "local" / model_short / task_id / condition


def prepared_path(root: Path, task_id: str) -> Path:
    return root / "data" / "prepared" / task_id


def task_config_path(root: Path, task_id: str) -> Path:
    return root / "configs" / "tasks" / f"{task_id}.yaml"


def model_config_path(root: Path, model_id: str) -> Path:
    return root / "configs" / "training" / f"{model_id}.yaml"


def prompt_path(root: Path, task_id: str) -> Path:
    return root / "prompts" / f"{task_id}.json"


def run_manifest_path(root: Path, run_id: str) -> Path:
    return root / "runs" / f"{run_id}.json"
