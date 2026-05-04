"""Shared fixtures and sys.path setup for all tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ── Toy data factories ─────────────────────────────────────────────────────────

def make_prediction_rows(n: int = 6, task_type: str = "classification") -> list[dict]:
    labels = ["positive", "negative", "neutral"]
    rows = []
    for i in range(n):
        gt = labels[i % 3]
        pred = gt if i % 4 != 0 else labels[(i + 1) % 3]
        rows.append({
            "id": f"test_{i:04d}",
            "model": "mock-model",
            "condition": "zero-shot",
            "input": f"Classify: sentence {i}",
            "output": pred,
            "ground_truth": gt,
            "input_tokens": 100 + i,
            "output_tokens": 5,
            "latency_ms": 100.0 + i * 10,
            "ttft_ms": 50.0 + i * 5,
            "timestamp": "2025-01-01T00:00:00Z",
        })
    return rows


def make_chat_rows(n: int = 5, with_assistant: bool = True) -> list[dict]:
    system = "Classify sentiment: positive, negative, or neutral."
    rows = []
    for i in range(n):
        msgs: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Classify: sentence {i}"},
        ]
        if with_assistant:
            msgs.append({"role": "assistant", "content": ["positive", "negative", "neutral"][i % 3]})
        rows.append({"id": f"row_{i:04d}", "messages": msgs})
    return rows


def make_test_prompts(n: int = 5) -> list[dict]:
    system = "Classify sentiment: positive, negative, or neutral."
    return [
        {
            "id": f"toy_test_{i:04d}",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Classify: sentence {i}"},
            ],
        }
        for i in range(n)
    ]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def toy_predictions():
    return make_prediction_rows(n=6)


@pytest.fixture
def toy_chat_rows():
    return make_chat_rows(n=5)


@pytest.fixture
def toy_test_prompts():
    return make_test_prompts(n=5)


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    from pipeline.providers import _openai_cb, _vllm_cb
    yield
    _openai_cb.reset()
    _vllm_cb.reset()


@pytest.fixture
def tmp_network_volume(tmp_path, monkeypatch):
    """Redirect NETWORK_VOLUME to a temp dir so checkpoint tests don't touch /workspace."""
    import checkpoint_utils
    nv = tmp_path / "workspace"
    nv.mkdir()
    monkeypatch.setattr(checkpoint_utils, "NETWORK_VOLUME", nv)
    return nv


@pytest.fixture
def tmp_repo_root(tmp_path, monkeypatch):
    """Redirect REPO_ROOT in pipeline scripts to tmp_path."""
    for d in [
        "data/prepared/toy",
        "results/predictions/mock-model/toy",
        "results/classified/mock-model/toy",
        "results/summaries/mock-model/toy",
        "results/training/mock-model/toy/zero-shot",
        "results/adapters",
        "configs/tasks",
        "configs/training",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    import classify_errors
    import generate_dashboard_data
    import sync_artifacts
    for mod in [classify_errors, generate_dashboard_data, sync_artifacts]:
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    return tmp_path
