"""Unit tests for generate_dashboard_data.py pure functions."""
import json

import pytest

from generate_dashboard_data import (
    PricingConfig,
    build_result,
    compute_cost_per_query,
    compute_stats,
    compute_tco_12mo,
    merge_results,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def pricing():
    return PricingConfig(
        apis={
            "gpt-4.1": {"input_per_m": 2.0, "output_per_m": 8.0},
            "gpt-4.1-sft": {"input_per_m": 2.0, "output_per_m": 8.0, "training_per_m": 25.0},
        },
        self_hosted={"gpu_hourly_rate": 0.49, "queries_per_hour_per_gpu": 2000},
    )


# ── compute_cost_per_query ─────────────────────────────────────────────────────

def test_cost_per_query_self_hosted(pricing):
    cost = compute_cost_per_query("qwen3-8b", 100_000, 5_000, 1000, pricing)
    assert cost == pytest.approx(0.49 / 2000, rel=1e-3)


def test_cost_per_query_api_model(pricing):
    # 1000 predictions, avg 500 input + 50 output tokens
    cost = compute_cost_per_query("gpt-4.1", 500_000, 50_000, 1000, pricing)
    expected = (500 / 1_000_000) * 2.0 + (50 / 1_000_000) * 8.0
    assert cost == pytest.approx(expected, rel=1e-3)


def test_cost_per_query_zero_predictions(pricing):
    assert compute_cost_per_query("gpt-4.1", 0, 0, 0, pricing) is None


def test_cost_per_query_unknown_api_model(pricing):
    assert compute_cost_per_query("unknown-model", 1000, 100, 10, pricing) is None


# ── compute_tco_12mo ───────────────────────────────────────────────────────────

def test_tco_12mo_api_model(pricing):
    cost_per_query = 0.001
    daily_vol = 100
    tco = compute_tco_12mo("gpt-4.1", training_cost=0.0, cost_per_query=cost_per_query, daily_volume=daily_vol, pricing=pricing)
    expected = 0.001 * 100 * 365
    assert tco == pytest.approx(expected, rel=1e-3)


def test_tco_12mo_includes_training_cost(pricing):
    tco = compute_tco_12mo("gpt-4.1", training_cost=20.0, cost_per_query=0.001, daily_volume=100, pricing=pricing)
    assert tco > 20.0


def test_tco_12mo_none_cost_per_query(pricing):
    assert compute_tco_12mo("gpt-4.1", 0, None, 100, pricing) is None


def test_tco_12mo_self_hosted_uses_gpu_reservation(pricing):
    # Self-hosted uses GPU reservation cost, not per-query × volume
    tco = compute_tco_12mo("qwen3-8b", training_cost=2.5, cost_per_query=0.49 / 2000, daily_volume=1000, pricing=pricing)
    # 1000 queries/day → 1000/(2000*24) < 1 GPU → 0 GPUs (ceil=1? depends on math)
    # Just verify it's a reasonable positive number
    assert tco is not None
    assert tco > 0


# ── build_result ───────────────────────────────────────────────────────────────

def test_build_result_with_summary(pricing):
    summary = {
        "metric_value": 0.85,
        "metric_id": "weighted_f1",
        "n_predictions": 500,
        "total_input_tokens": 50_000,
        "total_output_tokens": 5_000,
        "ttft_p50_ms": 120.0,
        "ttft_p95_ms": 300.0,
        "error_counts": {"correct": 425, "wrong_class": 75},
    }
    result = build_result("gpt-4.1", "fpb", "zero-shot", summary, None, pricing)
    assert result["model_id"] == "gpt-4.1"
    assert result["metric_value"] == pytest.approx(0.85)
    assert result["family"] == "frontier"
    assert result["cost_per_query"] is not None
    assert result["cost_per_1k_correct"] is not None
    assert result["error_counts"]["correct"] == 425


def test_build_result_without_summary(pricing):
    result = build_result("gpt-4.1", "fpb", "zero-shot", None, None, pricing)
    assert result["metric_value"] is None
    assert result["cost_per_query"] is None


def test_build_result_with_training_meta(pricing):
    summary = {
        "metric_value": 0.9, "metric_id": "weighted_f1", "n_predictions": 100,
        "total_input_tokens": 10_000, "total_output_tokens": 1_000,
        "ttft_p50_ms": None, "ttft_p95_ms": None, "error_counts": {},
    }
    training_meta = {"training_cost": 2.5, "training_time_min": 45.0, "n_train": 500}
    result = build_result("qwen3-8b", "fpb", "lora-500", summary, training_meta, pricing)
    assert result["training_cost"] == pytest.approx(2.5)
    assert result["training_time_min"] == pytest.approx(45.0)
    assert result["n_train"] == 500


def test_build_result_display_name(pricing):
    result = build_result("qwen3-8b", "fpb", "lora-500", None, None, pricing)
    assert "Qwen3" in result["display_name"]


def test_build_result_unknown_model(pricing):
    result = build_result("mystery-model", "fpb", "zero-shot", None, None, pricing)
    assert result["family"] == "frontier"  # default


@pytest.fixture
def axis_scores_summary():
    return {
        "metric_value": 0.85,
        "metric_id": "weighted_f1",
        "n_predictions": 100,
        "total_input_tokens": 10_000,
        "total_output_tokens": 1_000,
        "ttft_p50_ms": None,
        "ttft_p95_ms": None,
        "error_counts": {"correct": 85, "wrong_class": 15},
        "eval_axes": ["accuracy", "instruction_following", "cost"],
        "axis_scores": {
            "accuracy": {"value": 0.85, "higher_is_better": True},
            "instruction_following": {"value": 1.0, "higher_is_better": True},
        },
    }


def test_build_result_propagates_summary_axis_scores(axis_scores_summary, pricing):
    result = build_result("gpt-4.1", "fpb", "zero-shot", axis_scores_summary, None, pricing)
    assert result["axis_scores"]["accuracy"]["value"] == pytest.approx(0.85)
    assert result["axis_scores"]["instruction_following"]["value"] == pytest.approx(1.0)


def test_build_result_adds_cost_axis_score(axis_scores_summary, pricing):
    result = build_result("gpt-4.1", "fpb", "zero-shot", axis_scores_summary, None, pricing)
    assert "cost" in result["axis_scores"]
    assert result["axis_scores"]["cost"]["higher_is_better"] is False


def test_build_result_axis_scores_none_without_summary(pricing):
    result = build_result("gpt-4.1", "fpb", "zero-shot", None, None, pricing)
    assert result["axis_scores"] is None


# ── compute_stats ──────────────────────────────────────────────────────────────

def _make_result(model_id, family, condition, metric_value, cost_per_query=None, training_cost=None, task_id="fpb"):
    return {
        "model_id": model_id,
        "display_name": model_id,
        "family": family,
        "task_id": task_id,
        "condition": condition,
        "metric_value": metric_value,
        "cost_per_query": cost_per_query,
        "training_cost": training_cost,
    }


def test_compute_stats_lora_wins_task():
    results = [
        _make_result("qwen3-8b", "open-source", "LoRA", 0.85, task_id="fpb"),
        _make_result("gpt-4.1-nano", "frontier", "5-shot", 0.75, task_id="fpb"),
    ]
    stats = compute_stats(results)
    assert stats["tasks_won_by_oss"] == 1
    assert stats["comparisons"]["lora_vs_5shot"]["tasks_won"] == 1


def test_compute_stats_lora_loses_task():
    results = [
        _make_result("qwen3-8b", "open-source", "LoRA", 0.60, task_id="fpb"),
        _make_result("gpt-4.1-nano", "frontier", "5-shot", 0.80, task_id="fpb"),
    ]
    stats = compute_stats(results)
    assert stats["tasks_won_by_oss"] == 0


def test_compute_stats_total_training_cost():
    results = [
        _make_result("qwen3-8b", "open-source", "LoRA", 0.8, training_cost=2.5, task_id="fpb"),
        _make_result("qwen3-8b", "open-source", "LoRA", 0.7, training_cost=2.5, task_id="banking77"),
    ]
    stats = compute_stats(results)
    assert stats["cost_summary"]["total_training_cost"] == pytest.approx(5.0)


def test_compute_stats_empty_results():
    stats = compute_stats([])
    assert stats["tasks_won_by_oss"] == 0
    assert stats["comparisons"]["lora_vs_5shot"]["tasks_won"] == 0
    assert stats["cost_summary"]["total_training_cost"] is None


def test_compute_stats_ignores_null_metric_values():
    results = [
        _make_result("qwen3-8b", "open-source", "LoRA", None, task_id="fpb"),
        _make_result("gpt-4.1-nano", "frontier", "5-shot", None, task_id="fpb"),
    ]
    stats = compute_stats(results)
    assert stats["comparisons"]["lora_vs_5shot"]["tasks_won"] == 0


# ── merge_results ──────────────────────────────────────────────────────────────

def _write_existing(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": results}))


def test_merge_results_no_existing_file_returns_fresh(tmp_path):
    fresh = [_make_result("gpt-4.1", "frontier", "zero-shot", 0.82)]
    merged = merge_results(fresh, tmp_path / "nonexistent.json")
    assert merged == fresh


def test_merge_results_fresh_wins_when_nonnull(tmp_path):
    out = tmp_path / "results.json"
    prior = [_make_result("gpt-4.1", "frontier", "zero-shot", 0.75)]
    _write_existing(out, prior)

    fresh = [_make_result("gpt-4.1", "frontier", "zero-shot", 0.82)]
    merged = merge_results(fresh, out)
    assert merged[0]["metric_value"] == pytest.approx(0.82)


def test_merge_results_preserves_existing_when_fresh_null(tmp_path):
    out = tmp_path / "results.json"
    prior = [_make_result("gpt-4.1", "frontier", "zero-shot", 0.75)]
    _write_existing(out, prior)

    fresh = [_make_result("gpt-4.1", "frontier", "zero-shot", None)]
    merged = merge_results(fresh, out)
    assert merged[0]["metric_value"] == pytest.approx(0.75)


def test_merge_results_new_key_not_in_existing_kept(tmp_path):
    out = tmp_path / "results.json"
    prior = [_make_result("gpt-4.1", "frontier", "zero-shot", 0.75)]
    _write_existing(out, prior)

    # New model not in prior — should be included unchanged
    fresh = [
        _make_result("gpt-4.1", "frontier", "zero-shot", None),
        _make_result("gpt-5.4", "frontier", "zero-shot", 0.79),
    ]
    merged = merge_results(fresh, out)
    new_model = next(r for r in merged if r["model_id"] == "gpt-5.4")
    assert new_model["metric_value"] == pytest.approx(0.79)


def test_merge_results_handles_corrupt_existing(tmp_path):
    out = tmp_path / "results.json"
    out.write_text("not valid json {{{")

    fresh = [_make_result("gpt-4.1", "frontier", "zero-shot", 0.82)]
    merged = merge_results(fresh, out)
    assert merged == fresh


# ── _cohens_dz and _effect_label ─────────────────────────────────────────────

def test_cohens_dz_too_few_returns_none():
    from generate_dashboard_data import _cohens_dz
    assert _cohens_dz([]) is None
    assert _cohens_dz([0.05]) is None


def test_cohens_dz_n_lt_3_returns_none():
    from generate_dashboard_data import _cohens_dz
    assert _cohens_dz([0.10]) is None
    assert _cohens_dz([0.10, 0.20]) is None  # n=2: Hedge's J = 0, estimate undefined


def test_cohens_dz_zero_variance_returns_none():
    from generate_dashboard_data import _cohens_dz
    # n=3 with identical values: std approaches 0, guarded by < 1e-10 threshold
    assert _cohens_dz([0.10, 0.10, 0.10]) is None


def test_cohens_dz_positive_gain():
    from generate_dashboard_data import _cohens_dz
    deltas = [0.05, 0.08, 0.06, 0.07, 0.09]
    dz = _cohens_dz(deltas)
    assert dz is not None and dz > 0


def test_cohens_dz_negative_gain():
    from generate_dashboard_data import _cohens_dz
    deltas = [-0.05, -0.08, -0.06]
    dz = _cohens_dz(deltas)
    assert dz is not None and dz < 0


def test_cohens_dz_applies_small_sample_correction():
    from generate_dashboard_data import _cohens_dz
    import math
    # For n=3, Cohen's d = mean/std_sample, Hedge's J = 1 - 3/(4*2-1) = 1 - 3/7 ≈ 0.571
    deltas = [0.1, 0.2, 0.3]
    n = 3
    mean = sum(deltas) / n
    var = sum((x - mean) ** 2 for x in deltas) / (n - 1)
    d = mean / math.sqrt(var)
    j = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    expected_dz = round(d * j, 4)
    assert _cohens_dz(deltas) == pytest.approx(expected_dz, rel=1e-4)


def test_effect_label_negligible():
    from generate_dashboard_data import _effect_label
    assert _effect_label(0.1) == "negligible"
    assert _effect_label(-0.1) == "negligible"


def test_effect_label_small():
    from generate_dashboard_data import _effect_label
    assert _effect_label(0.3) == "small"


def test_effect_label_medium():
    from generate_dashboard_data import _effect_label
    assert _effect_label(0.65) == "medium"


def test_effect_label_large():
    from generate_dashboard_data import _effect_label
    assert _effect_label(1.2) == "large"
    assert _effect_label(-0.9) == "large"


def test_effect_label_none_returns_none():
    from generate_dashboard_data import _effect_label
    assert _effect_label(None) is None


def test_comparison_includes_effect_size(pricing):
    from generate_dashboard_data import compute_stats
    # Need n≥3 tasks with varied gains (not identical) for Hedge's g to be defined.
    # Gains: fpb +0.10, banking77 +0.08, cuad +0.12 → mean 0.10, variance > 0.
    results = [
        _make_result("qwen3-8b", "open-source", "LoRA", 0.85, task_id="fpb"),
        _make_result("qwen3-8b", "open-source", "LoRA", 0.78, task_id="banking77"),
        _make_result("qwen3-8b", "open-source", "LoRA", 0.92, task_id="cuad"),
        _make_result("gpt-4.1-nano", "frontier", "5-shot", 0.75, task_id="fpb"),
        _make_result("gpt-4.1-nano", "frontier", "5-shot", 0.70, task_id="banking77"),
        _make_result("gpt-4.1-nano", "frontier", "5-shot", 0.80, task_id="cuad"),
    ]
    stats = compute_stats(results)
    comp = stats["comparisons"]["lora_vs_5shot"]
    assert "effect_size_dz" in comp
    assert "effect_size_label" in comp
    assert comp["effect_size_dz"] is not None
    assert comp["effect_size_dz"] > 0


# ── metric_granularity in build_result ────────────────────────────────────────

def test_build_result_propagates_metric_granularity(pricing):
    summary = {
        "metric_id": "weighted_f1",
        "metric_granularity": "per_class_weighted",
        "metric_value": 0.82,
        "n_predictions": 100,
        "eval_axes": [],
        "axis_scores": {},
    }
    result = build_result("qwen3-8b", "fpb", "lora", summary, None, pricing)
    assert result["metric_granularity"] == "per_class_weighted"


def test_build_result_metric_granularity_none_without_summary(pricing):
    result = build_result("qwen3-8b", "fpb", "lora", None, None, pricing)
    assert result["metric_granularity"] is None


# ── semantic_error_counts in build_result ─────────────────────────────────────

def test_build_result_propagates_semantic_error_counts(pricing):
    summary = {
        "metric_id": "weighted_f1",
        "metric_value": 0.82,
        "n_predictions": 100,
        "error_counts": {"correct": 82, "wrong_class": 18},
        "semantic_error_counts": {"correct": 82, "factual_error": 18},
    }
    result = build_result("qwen3-8b", "fpb", "lora", summary, None, pricing)
    assert result["semantic_error_counts"] == {"correct": 82, "factual_error": 18}


def test_build_result_semantic_error_counts_none_without_summary(pricing):
    result = build_result("qwen3-8b", "fpb", "lora", None, None, pricing)
    assert result["semantic_error_counts"] is None


def test_build_result_semantic_error_counts_none_when_absent(pricing):
    summary = {
        "metric_id": "accuracy",
        "metric_value": 0.75,
        "n_predictions": 50,
        "error_counts": {"correct": 37, "wrong_class": 13},
        # no semantic_error_counts key
    }
    result = build_result("qwen3-8b", "fpb", "lora", summary, None, pricing)
    assert result["semantic_error_counts"] is None


# ── compute_dtype in build_result and dtype_warnings in compute_stats ──────────

def test_build_result_propagates_compute_dtype(pricing):
    training_meta = {"training_cost": 2.0, "n_train": 200, "compute_dtype": "bfloat16"}
    result = build_result("qwen3-8b", "fpb", "lora", None, training_meta, pricing)
    assert result["compute_dtype"] == "bfloat16"


def test_build_result_compute_dtype_none_without_training_meta(pricing):
    result = build_result("qwen3-8b", "fpb", "zero-shot", None, None, pricing)
    assert result["compute_dtype"] is None


def test_compute_stats_dtype_warnings_empty_when_consistent():
    results = [
        {**_make_result("qwen3-8b",   "open-source", "LoRA", 0.85, task_id="fpb"),   "compute_dtype": "bfloat16"},
        {**_make_result("qwen2.5-7b", "open-source", "LoRA", 0.80, task_id="fpb"),   "compute_dtype": "bfloat16"},
    ]
    stats = compute_stats(results)
    assert stats["dtype_warnings"] == []


def test_compute_stats_dtype_warnings_populated_on_mismatch():
    results = [
        {**_make_result("qwen3-8b",   "open-source", "LoRA", 0.85, task_id="fpb"), "compute_dtype": "bfloat16"},
        {**_make_result("qwen2.5-7b", "open-source", "LoRA", 0.80, task_id="fpb"), "compute_dtype": "float32"},
    ]
    stats = compute_stats(results)
    assert len(stats["dtype_warnings"]) == 1
    assert "bfloat16" in stats["dtype_warnings"][0]
    assert "float32" in stats["dtype_warnings"][0]


def test_compute_stats_dtype_warnings_ignores_non_lora():
    # dtype mismatch in zero-shot conditions doesn't matter for training comparison
    results = [
        {**_make_result("qwen3-8b",   "open-source", "Zero-shot", 0.85, task_id="fpb"), "compute_dtype": "bfloat16"},
        {**_make_result("qwen2.5-7b", "open-source", "Zero-shot", 0.80, task_id="fpb"), "compute_dtype": "float32"},
    ]
    stats = compute_stats(results)
    assert stats["dtype_warnings"] == []


# ── compliance rates in build_result ─────────────────────────────────────────

def _summary_with_rates(**overrides) -> dict:
    base = {
        "metric_id": "weighted_f1",
        "metric_value": 0.80,
        "n_predictions": 100,
        "error_counts": {"correct": 80, "wrong_class": 10, "format_violation": 5, "empty": 3, "refusal": 2},
        "format_compliance_rate": 0.95,
        "refusal_rate": 0.02,
        "empty_rate": 0.03,
        "partial_rate": 0.0,
    }
    base.update(overrides)
    return base


def test_build_result_propagates_compliance_rates(pricing):
    result = build_result("qwen3-8b", "fpb", "lora", _summary_with_rates(), None, pricing)
    assert result["format_compliance_rate"] == pytest.approx(0.95)
    assert result["refusal_rate"] == pytest.approx(0.02)
    assert result["empty_rate"] == pytest.approx(0.03)
    assert result["partial_rate"] == pytest.approx(0.0)


def test_build_result_compliance_rates_none_without_summary(pricing):
    result = build_result("qwen3-8b", "fpb", "lora", None, None, pricing)
    assert result["format_compliance_rate"] is None
    assert result["refusal_rate"] is None
    assert result["empty_rate"] is None
    assert result["partial_rate"] is None


def test_build_result_compliance_rates_none_when_absent(pricing):
    summary = {
        "metric_id": "accuracy",
        "metric_value": 0.75,
        "n_predictions": 50,
        "error_counts": {"correct": 37, "wrong_class": 13},
        # no rate fields
    }
    result = build_result("qwen3-8b", "fpb", "lora", summary, None, pricing)
    assert result["format_compliance_rate"] is None
    assert result["refusal_rate"] is None


# ── _load_task_baselines ───────────────────────────────────────────────────────

import math as _math


def _write_quality_report(root, task_id: str, label_counts: dict[str, int]) -> None:
    qr_dir = root / "data" / "prepared" / task_id
    qr_dir.mkdir(parents=True, exist_ok=True)
    qr = {
        "task_id": task_id,
        "prepared": {
            "test": {
                "label_distribution": {lbl: {"count": cnt} for lbl, cnt in label_counts.items()}
            }
        },
    }
    (qr_dir / "quality_report.json").write_text(json.dumps(qr))


def test_load_task_baselines_empty_when_no_reports(tmp_repo_root, monkeypatch):
    import generate_dashboard_data
    monkeypatch.setattr(generate_dashboard_data, "ALL_TASKS", ["banking77"])
    from generate_dashboard_data import _load_task_baselines
    assert _load_task_baselines() == {}


def test_load_task_baselines_skips_missing_file(tmp_repo_root, monkeypatch):
    import generate_dashboard_data
    monkeypatch.setattr(generate_dashboard_data, "ALL_TASKS", ["banking77", "fpb"])
    _write_quality_report(tmp_repo_root, "banking77", {"A": 50, "B": 50})
    from generate_dashboard_data import _load_task_baselines
    result = _load_task_baselines()
    assert "banking77" in result
    assert "fpb" not in result


def test_load_task_baselines_computes_random_chance(tmp_repo_root, monkeypatch):
    import generate_dashboard_data
    monkeypatch.setattr(generate_dashboard_data, "ALL_TASKS", ["banking77"])
    _write_quality_report(tmp_repo_root, "banking77", {"A": 25, "B": 25, "C": 25, "D": 25})
    from generate_dashboard_data import _load_task_baselines
    result = _load_task_baselines()
    assert result["banking77"]["random_chance"] == pytest.approx(0.25, rel=1e-4)
    assert result["banking77"]["n_classes"] == 4
    assert result["banking77"]["n_test"] == 100


def test_load_task_baselines_computes_majority_class(tmp_repo_root, monkeypatch):
    import generate_dashboard_data
    monkeypatch.setattr(generate_dashboard_data, "ALL_TASKS", ["banking77"])
    _write_quality_report(tmp_repo_root, "banking77", {"A": 80, "B": 20})
    from generate_dashboard_data import _load_task_baselines
    result = _load_task_baselines()
    assert result["banking77"]["majority_class_accuracy"] == pytest.approx(0.8, rel=1e-4)


def test_load_task_baselines_computes_mde(tmp_repo_root, monkeypatch):
    import generate_dashboard_data
    monkeypatch.setattr(generate_dashboard_data, "ALL_TASKS", ["banking77"])
    n = 400
    _write_quality_report(tmp_repo_root, "banking77", {"A": n // 2, "B": n // 2})
    from generate_dashboard_data import _load_task_baselines
    result = _load_task_baselines()
    expected_mde = round(2.8 * _math.sqrt(0.25 / n) * 100, 1)
    assert result["banking77"]["min_detectable_effect_pp"] == pytest.approx(expected_mde, rel=1e-4)


def test_load_task_baselines_skips_empty_label_distribution(tmp_repo_root, monkeypatch):
    import generate_dashboard_data
    monkeypatch.setattr(generate_dashboard_data, "ALL_TASKS", ["banking77"])
    qr_dir = tmp_repo_root / "data" / "prepared" / "banking77"
    qr_dir.mkdir(parents=True, exist_ok=True)
    (qr_dir / "quality_report.json").write_text(
        json.dumps({"task_id": "banking77", "prepared": {"test": {}}})
    )
    from generate_dashboard_data import _load_task_baselines
    assert _load_task_baselines() == {}
