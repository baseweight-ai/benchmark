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
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tests._api_stubs  # noqa: F401 — injects openai/aiohttp/tqdm stubs
import classify_errors
import eval_api
import generate_dashboard_data
from eval_api import TaskConfig as EvalTaskConfig

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).parent.parent


# ── Toy fixtures ───────────────────────────────────────────────────────────────

TASK_ID = "fpb"
MODEL_ID = "gpt-4.1-nano"
CONDITION = "zero-shot"
N = 8  # toy dataset size


def _write_toy_prepared_data(root: Path) -> Path:
    """Write minimal prepared data for fpb."""
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
    few_shot_rows = [
        {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Classify the sentiment:\n\nFS Sentence {i}"},
                {"role": "assistant", "content": labels[i % 3]},
            ]
        }
        for i in range(5)
    ]

    def _wj(rows, path):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    _wj(test_rows, prep / "test.jsonl")
    _wj(few_shot_rows, prep / "few_shot_5.jsonl")
    return prep


def _write_toy_task_config(root: Path) -> None:
    """Copy all task configs into the tmp repo (build_efficiency_points reads them)."""
    import shutil
    src_dir = REPO_ROOT / "configs" / "tasks"
    dst = root / "configs" / "tasks"
    dst.mkdir(parents=True, exist_ok=True)
    for cfg in src_dir.glob("*.yaml"):
        shutil.copy(cfg, dst / cfg.name)


def _write_toy_pricing_config(root: Path) -> None:
    import shutil
    src = REPO_ROOT / "configs" / "pricing.yaml"
    dst = root / "configs"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst / "pricing.yaml")


# ── Stage helpers ──────────────────────────────────────────────────────────────

def _stage_eval(root: Path) -> Path:
    """Stage 2: run eval_api with mocked OpenAI responses."""
    labels = ["positive", "negative", "neutral"]

    async def mock_call(client, model_str, messages, max_tokens, semaphore):
        idx = len(list((root / "results" / "predictions" / MODEL_ID / TASK_ID).glob("*.partial")))
        return labels[idx % 3], 100, 10, 120.0

    call_count = [0]

    async def counting_call(client, model_str, messages, max_tokens, semaphore):
        label = labels[call_count[0] % 3]
        call_count[0] += 1
        return label, 100, 10, 120.0, 50.0

    from eval_api import TaskConfig as TC
    cfg = TC(task_id=TASK_ID, max_output_tokens=32, task_type="classification")

    with patch("eval_api.call_openai", side_effect=counting_call):
        with patch("openai.AsyncOpenAI", return_value=None):
            asyncio.run(run_eval_patched(MODEL_ID, TASK_ID, CONDITION, cfg, root))

    out = root / "results" / "predictions" / "api" / MODEL_ID / TASK_ID / f"{CONDITION}.jsonl"
    return out


async def run_eval_patched(model_id, task_id, condition, cfg, root):
    """Thin wrapper that patches REPO_ROOT inside the running coroutine."""
    import eval_api as _ea
    original = _ea.REPO_ROOT
    _ea.REPO_ROOT = root
    try:
        await eval_api.run_eval(model_id, task_id, condition, cfg, dry_run=False)
    finally:
        _ea.REPO_ROOT = original


def _stage_classify(root: Path) -> Path:
    """Stage 3: classify predictions."""
    import classify_errors as ce
    original = ce.REPO_ROOT
    ce.REPO_ROOT = root
    try:
        cfg = ce.load_task_config(TASK_ID)
        valid_labels = ce.get_valid_labels(TASK_ID)
        ce.process_model_task_condition(MODEL_ID, TASK_ID, CONDITION, cfg, valid_labels, dry_run=False, source="api")
    finally:
        ce.REPO_ROOT = original

    return root / "results" / "summaries" / "api" / MODEL_ID / TASK_ID / f"{CONDITION}.json"


def _stage_dashboard(root: Path) -> Path:
    """Stage 4: generate dashboard data."""
    import generate_dashboard_data as gdd
    original = gdd.REPO_ROOT
    gdd.REPO_ROOT = root
    try:
        data = gdd.build_dashboard_data(daily_volume=1000)
    finally:
        gdd.REPO_ROOT = original

    out = root / "dashboard-data" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    data["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    return out


# ── Smoke test ─────────────────────────────────────────────────────────────────

def test_smoke_pipeline(tmp_path, monkeypatch):
    """Full pipeline: toy data → eval → classify → dashboard."""
    # Stage 1: write toy data and configs
    _write_toy_prepared_data(tmp_path)
    _write_toy_task_config(tmp_path)
    _write_toy_pricing_config(tmp_path)

    # Stage 2: eval
    pred_path = _stage_eval(tmp_path)
    assert pred_path.exists(), "Predictions file not created"
    pred_rows = [json.loads(l) for l in pred_path.read_text().splitlines()]
    assert len(pred_rows) == N
    assert all(r["output"] in ("positive", "negative", "neutral") for r in pred_rows)
    assert all("id" in r and "latency_ms" in r for r in pred_rows)

    # Stage 3: classify
    summary_path = _stage_classify(tmp_path)
    assert summary_path.exists(), "Summary JSON not created"
    summary = json.loads(summary_path.read_text())
    assert summary["model"] == MODEL_ID
    assert summary["task_id"] == TASK_ID
    assert summary["n_predictions"] == N
    assert summary["metric_value"] is not None
    assert 0.0 <= summary["metric_value"] <= 1.0
    assert sum(summary["error_counts"].values()) == N

    classified_path = tmp_path / "results" / "classified" / "api" / MODEL_ID / TASK_ID / f"{CONDITION}.jsonl"
    assert classified_path.exists()
    classified_rows = [json.loads(l) for l in classified_path.read_text().splitlines()]
    assert len(classified_rows) == N
    assert all("error_category" in r for r in classified_rows)

    # Stage 4: dashboard
    results_path = _stage_dashboard(tmp_path)
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

    # The fpb/gpt-4.1-nano/zero-shot result should have our metric value.
    # results use display labels ("Zero-shot"), not raw condition slugs.
    matching = [
        r for r in data["results"]
        if r["model_id"] == MODEL_ID and r["task_id"] == TASK_ID
    ]
    assert len(matching) == 1
    assert matching[0]["metric_value"] == pytest.approx(summary["metric_value"], rel=1e-4)
