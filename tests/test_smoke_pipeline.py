"""Layer 3 — Smoke pipeline: full end-to-end run with toy data and mocked APIs.

This test exercises every stage of the pipeline in sequence:
  1. Write toy prepared data (bypassing download/prepare)
  2. Run eval_api with a mocked OpenAI call
  3. Run classify_errors on the predictions
  4. Run generate_dashboard_data to assemble results.json
  5. Assert all expected output files exist with correct structure

No GPU, no real API keys, no HuggingFace downloads required.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

import tests._api_stubs  # noqa: F401 — injects openai/aiohttp/tqdm stubs
import classify_errors
import eval_api
import generate_dashboard_data
from eval_api import TaskConfig as EvalTaskConfig
from tests.conftest import write_jsonl

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).parent.parent

TASK_ID = "fpb"
MODEL_ID = "gpt-4.1-nano"
CONDITION = "zero-shot"
N = 8  # toy dataset size


# ── Toy data setup ─────────────────────────────────────────────────────────────

def _write_toy_prepared_data(root: Path) -> None:
    prep = root / "data" / "prepared" / TASK_ID
    prep.mkdir(parents=True, exist_ok=True)

    system = "Classify the financial statement sentiment. Respond with exactly one word: positive, negative, or neutral."
    labels = ["positive", "negative", "neutral"]

    test_rows = [
        {
            "id": f"fpb_test_{i:04d}",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Classify the sentiment:\n\nSentence {i}"},
            ],
        }
        for i in range(N)
    ]
    train_rows = [
        {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Classify the sentiment:\n\nFS Sentence {i}"},
                {"role": "assistant", "content": labels[i % 3]},
            ]
        }
        for i in range(5)
    ]

    write_jsonl(test_rows, prep / "test.jsonl")
    write_jsonl(train_rows, prep / "train.jsonl")


@pytest.fixture
def toy_repo(tmp_path):
    """Minimal repo layout: prepared data + task and pricing configs."""
    _write_toy_prepared_data(tmp_path)

    dst_tasks = tmp_path / "configs" / "tasks"
    dst_tasks.mkdir(parents=True, exist_ok=True)
    for cfg in (REPO_ROOT / "configs" / "tasks").glob("*.yaml"):
        shutil.copy(cfg, dst_tasks / cfg.name)

    dst_configs = tmp_path / "configs"
    shutil.copy(REPO_ROOT / "configs" / "pricing.yaml", dst_configs / "pricing.yaml")

    return tmp_path


# ── Stage helpers ──────────────────────────────────────────────────────────────

def _stage_eval(root: Path, monkeypatch) -> Path:
    """Stage 2: run eval_api with mocked OpenAI responses."""
    monkeypatch.setattr(eval_api, "REPO_ROOT", root)

    labels = ["positive", "negative", "neutral"]
    call_count = [0]

    async def counting_call(*args, **kwargs):
        label = labels[call_count[0] % 3]
        call_count[0] += 1
        return label, 100, 10, 120.0, 50.0

    cfg = EvalTaskConfig(task_id=TASK_ID, max_output_tokens=32, task_type="classification")
    with patch("eval_api.call_openai", side_effect=counting_call):
        with patch("openai.AsyncOpenAI", return_value=None):
            asyncio.run(eval_api.run_eval(MODEL_ID, TASK_ID, CONDITION, cfg, dry_run=False))

    return root / "results" / "predictions" / "api" / MODEL_ID / TASK_ID / f"{CONDITION}.jsonl"


def _stage_classify(root: Path, monkeypatch) -> Path:
    """Stage 3: classify predictions."""
    monkeypatch.setattr(classify_errors, "REPO_ROOT", root)
    cfg = classify_errors.load_task_config(TASK_ID)
    valid_labels = classify_errors.get_valid_labels(TASK_ID)
    classify_errors.process_model_task_condition(
        MODEL_ID, TASK_ID, CONDITION, cfg, valid_labels, dry_run=False, source="api"
    )
    return root / "results" / "summaries" / "api" / MODEL_ID / TASK_ID / f"{CONDITION}.json"


def _stage_dashboard(root: Path, monkeypatch) -> Path:
    """Stage 4: generate dashboard data."""
    from datetime import datetime, timezone
    monkeypatch.setattr(generate_dashboard_data, "REPO_ROOT", root)
    data = generate_dashboard_data.build_dashboard_data(daily_volume=1000)

    out = root / "dashboard-data" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    return out


# ── Smoke test ─────────────────────────────────────────────────────────────────

def test_smoke_pipeline(toy_repo, monkeypatch):
    """Full pipeline: toy data → eval → classify → dashboard."""
    root = toy_repo

    # Stage 2: eval
    pred_path = _stage_eval(root, monkeypatch)
    assert pred_path.exists(), "Predictions file not created"
    pred_rows = [json.loads(l) for l in pred_path.read_text().splitlines()]
    assert len(pred_rows) == N
    assert all(r["output"] in ("positive", "negative", "neutral") for r in pred_rows)
    assert all("id" in r and "latency_ms" in r for r in pred_rows)

    # Stage 3: classify
    summary_path = _stage_classify(root, monkeypatch)
    assert summary_path.exists(), "Summary JSON not created"
    summary = json.loads(summary_path.read_text())
    assert summary["model"] == MODEL_ID
    assert summary["task_id"] == TASK_ID
    assert summary["n_predictions"] == N
    assert summary["metric_value"] is not None
    assert 0.0 <= summary["metric_value"] <= 1.0
    assert sum(summary["error_counts"].values()) == N
    assert summary["per_class_metrics"] is not None
    for cls_data in summary["per_class_metrics"].values():
        assert {"correct", "total", "accuracy"} <= cls_data.keys()

    classified_path = root / "results" / "classified" / "api" / MODEL_ID / TASK_ID / f"{CONDITION}.jsonl"
    assert classified_path.exists()
    classified_rows = [json.loads(l) for l in classified_path.read_text().splitlines()]
    assert len(classified_rows) == N
    assert all("error_category" in r for r in classified_rows)

    # Stage 4: dashboard
    results_path = _stage_dashboard(root, monkeypatch)
    assert results_path.exists(), "Dashboard results.json not created"
    data = json.loads(results_path.read_text())
    assert "results" in data
    assert "generated_at" in data
    assert "tasks_won_by_oss" in data
    assert "comparisons" in data
    assert "cost_summary" in data
    assert isinstance(data["tasks_won_by_oss"], int)
    assert isinstance(data["cost_summary"], dict)
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0

    matching = [
        r for r in data["results"]
        if r["model_id"] == MODEL_ID and r["task_id"] == TASK_ID
    ]
    assert len(matching) == 1
    assert matching[0]["metric_value"] == pytest.approx(summary["metric_value"], rel=1e-4)
