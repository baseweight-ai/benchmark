"""Centralized path registry. All functions take root: Path as first argument
so callers can pass REPO_ROOT and tests can monkeypatch just that one variable.

Output paths take an optional smoke: bool — when True, a "smoke/" segment is
inserted after the top-level directory so smoke runs never clobber real outputs.
Read-only paths (configs, prompts) are not smoke-namespaced — they're shared.
"""
from __future__ import annotations
from pathlib import Path


def smoke_seg(smoke: bool) -> str:
    """Returns "smoke" when smoke, "" otherwise. Path-joining absorbs an empty
    segment, so non-smoke paths remain byte-for-byte unchanged."""
    return "smoke" if smoke else ""


def pred_path(root: Path, source: str, model_short: str, task_id: str, condition: str,
              smoke: bool = False) -> Path:
    return root / "results" / smoke_seg(smoke) / "predictions" / source / model_short / task_id / f"{condition}.jsonl"


def classified_path(root: Path, source: str, model_short: str, task_id: str, condition: str,
                    smoke: bool = False) -> Path:
    return root / "results" / smoke_seg(smoke) / "classified" / source / model_short / task_id / f"{condition}.jsonl"


def summary_path(root: Path, source: str, model_short: str, task_id: str, condition: str,
                 smoke: bool = False) -> Path:
    return root / "results" / smoke_seg(smoke) / "summaries" / source / model_short / task_id / f"{condition}.json"


def training_meta_path(root: Path, source: str, model_id: str, task_id: str, condition: str,
                       smoke: bool = False) -> Path:
    return root / "results" / smoke_seg(smoke) / "training" / source / model_id / task_id / condition / "metadata.json"


def adapter_path(root: Path, model_short: str, task_id: str, condition: str,
                 smoke: bool = False) -> Path:
    return root / "results" / smoke_seg(smoke) / "adapters" / "local" / model_short / task_id / condition


def prepared_path(root: Path, task_id: str, smoke: bool = False) -> Path:
    return root / "data" / smoke_seg(smoke) / "prepared" / task_id


def raw_path(root: Path, task_id: str, smoke: bool = False) -> Path:
    """Raw downloaded dataset for a task. Smoke routes to data/smoke/raw/ so a
    truncated smoke download cannot clobber the real raw artifact."""
    return root / "data" / smoke_seg(smoke) / "raw" / task_id


def task_config_path(root: Path, task_id: str) -> Path:
    return root / "configs" / "tasks" / f"{task_id}.yaml"


def model_config_path(root: Path, model_id: str) -> Path:
    return root / "configs" / "training" / f"{model_id}.yaml"


def prompt_path(root: Path, task_id: str) -> Path:
    return root / "prompts" / f"{task_id}.json"


def run_manifest_path(root: Path, run_id: str, smoke: bool = False) -> Path:
    return root / "runs" / smoke_seg(smoke) / f"{run_id}.json"


def test_full_path(root: Path, task_id: str, smoke: bool = False) -> Path:
    # Smoke prepared data lives under data/smoke/prepared/<task>/; the legacy
    # smoke_ filename prefix is retained for compatibility with existing
    # readers even though the dir alone already disambiguates.
    prefix = "smoke_" if smoke else ""
    return root / "data" / smoke_seg(smoke) / "prepared" / task_id / f"{prefix}test_full.jsonl"


def snapshot_path(root: Path, run_id: str, smoke: bool = False) -> Path:
    return root / "results" / smoke_seg(smoke) / "snapshots" / run_id / "results.json"
