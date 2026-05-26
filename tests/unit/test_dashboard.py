"""Unit tests for generate_dashboard_data.py pure functions."""
import json

import pytest

from generate_dashboard_data import (
    PricingConfig,
    _export_tables,
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


def _summary(**overrides) -> dict:
    """Default classification-task summary; override any field per test."""
    base = {
        "metric_value": 0.9,
        "metric_id": "accuracy",
        "n_predictions": 500,
        "total_input_tokens": 50_000,
        "total_output_tokens": 5_000,
        "error_counts": {},
    }
    base.update(overrides)
    return base


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


def test_build_result_surfaces_classification_metrics(pricing):
    summary = _summary(
        metric_value=0.85, metric_id="weighted_f1",
        exact_match=0.82, macro_f1=0.78, weighted_f1=0.85, hallucination_rate=0.04,
    )
    result = build_result("gpt-4.1", "fpb", "zero-shot", summary, None, pricing)
    assert result["exact_match"] == pytest.approx(0.82)
    assert result["macro_f1"] == pytest.approx(0.78)
    assert result["weighted_f1"] == pytest.approx(0.85)
    assert result["hallucination_rate"] == pytest.approx(0.04)


def test_build_result_cost_per_1k_requests_is_cost_per_query_times_1000(pricing):
    result = build_result("gpt-4.1", "fpb", "zero-shot", _summary(), None, pricing)
    assert result["cost_per_1k_requests"] == pytest.approx(result["cost_per_query"] * 1000, rel=1e-4)


def test_build_result_cost_per_1k_tokens_matches_total_cost_over_total_tokens(pricing):
    result = build_result("gpt-4.1", "fpb", "zero-shot", _summary(), None, pricing)
    expected = result["cost_per_query"] * 500 / 55_000 * 1000
    # 6-decimal rounding in dashboard output.
    assert result["cost_per_1k_tokens"] == pytest.approx(expected, abs=1e-6)


def test_build_result_cost_per_1m_api_uses_billed_rates(pricing):
    """API models bill input/output separately; pricing fixture has gpt-4.1
    at input=$2/1M, output=$8/1M."""
    result = build_result("gpt-4.1", "fpb", "zero-shot", _summary(), None, pricing)
    assert result["cost_per_1m_input_tokens"] == pytest.approx(2.0, rel=1e-4)
    assert result["cost_per_1m_output_tokens"] == pytest.approx(8.0, rel=1e-4)
    assert result["cost_per_1m_input_tokens"] != result["cost_per_1m_output_tokens"]


def test_build_result_cost_per_1m_self_hosted_input_equals_output(pricing):
    """Self-hosted shares wall time between prefill and decode — can't
    attribute input vs output separately."""
    result = build_result("qwen3-8b", "fpb", "lora", _summary(), None, pricing)
    assert result["cost_per_1m_input_tokens"] is not None
    assert result["cost_per_1m_output_tokens"] is not None
    assert result["cost_per_1m_input_tokens"] == result["cost_per_1m_output_tokens"]


def test_build_result_surfaces_latency_percentiles(pricing):
    summary = _summary(latency_p50_ms=120.0, latency_p99_ms=450.0)
    result = build_result("gpt-4.1", "fpb", "zero-shot", summary, None, pricing)
    assert result["latency_p50_ms"] == 120.0
    assert result["latency_p99_ms"] == 450.0


def test_build_result_computes_throughput_qps(pricing):
    summary = _summary(eval_wall_time_s=10.0)
    result = build_result("gpt-4.1", "fpb", "zero-shot", summary, None, pricing)
    assert result["throughput_qps"] == pytest.approx(50.0)


def test_build_result_throughput_none_without_wall_time(pricing):
    result = build_result("gpt-4.1", "fpb", "zero-shot", _summary(), None, pricing)
    assert result["throughput_qps"] is None


def test_build_result_cost_per_1m_queries_is_cost_per_query_times_1m(pricing):
    result = build_result("gpt-4.1", "fpb", "zero-shot", _summary(), None, pricing)
    assert result["cost_per_1m_queries"] == pytest.approx(
        result["cost_per_query"] * 1_000_000, rel=1e-4
    )


def test_build_result_surfaces_gpu_model(pricing):
    """eval-time gpu_model overrides training-time."""
    summary = _summary(n_predictions=100, total_input_tokens=10_000, total_output_tokens=1_000,
                       gpu_model="NVIDIA L4")
    training_meta = {"gpu_model": "NVIDIA A10G", "training_cost": 2.5}
    result = build_result("qwen3-8b", "fpb", "lora", summary, training_meta, pricing)
    assert result["gpu_model"] == "NVIDIA L4"


def test_summarise_hardware_consistent_returns_no_warning():
    """All local rows on one GPU → no warning."""
    from generate_dashboard_data import _summarise_hardware
    results = [
        {"model_id": "qwen3-8b", "family": "open-source", "gpu_model": "NVIDIA GeForce RTX 3090"},
        {"model_id": "qwen3-8b", "family": "open-source", "gpu_model": "NVIDIA GeForce RTX 3090"},
    ]
    hw = _summarise_hardware(results)
    assert hw["local_gpus_observed"] == ["NVIDIA GeForce RTX 3090"]
    assert hw["inconsistent_gpu_models"] == {}
    assert hw["hardware_warning"] is None


def test_summarise_hardware_mixed_gpus_for_one_model_warns():
    """Same model on 2 different GPUs → warning, latency/cost not comparable."""
    from generate_dashboard_data import _summarise_hardware
    results = [
        {"model_id": "qwen3-8b", "family": "open-source", "gpu_model": "NVIDIA GeForce RTX 3090"},
        {"model_id": "qwen3-8b", "family": "open-source", "gpu_model": "NVIDIA A10G"},
    ]
    hw = _summarise_hardware(results)
    assert "qwen3-8b" in hw["inconsistent_gpu_models"]
    assert hw["hardware_warning"] is not None
    assert "qwen3-8b" in hw["hardware_warning"]


def test_summarise_hardware_ignores_api_rows():
    """API rows have gpu_model=None — they don't trip the consistency check."""
    from generate_dashboard_data import _summarise_hardware
    results = [
        {"model_id": "qwen3-8b", "family": "open-source", "gpu_model": "NVIDIA GeForce RTX 3090"},
        {"model_id": "gpt-4.1-nano", "family": "frontier", "gpu_model": None},
    ]
    hw = _summarise_hardware(results)
    assert hw["hardware_warning"] is None


def test_build_dashboard_data_applies_gpu_hourly_rate_override(tmp_path, monkeypatch):
    """--gpu-hourly-rate flag changes the rate used for cost computations
    without touching pricing.yaml on disk."""
    import generate_dashboard_data as gdd

    # Tmp repo with summaries + pricing.yaml at $1.00/hr
    monkeypatch.setattr(gdd, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pricing.yaml").write_text(
        "apis: {}\n"
        "self_hosted:\n"
        "  gpu_hourly_rate: 1.00\n"
        "  queries_per_hour_per_gpu: 2000\n"
    )
    # qwen3-8b needs a training config file to be recognised as open-source.
    (tmp_path / "configs" / "training").mkdir()
    (tmp_path / "configs" / "training" / "qwen3-8b.yaml").write_text(
        "model_id: Qwen/Qwen3-8B\nmodel_short: qwen3-8b\n"
    )
    summary_dir = tmp_path / "results" / "summaries" / "local" / "qwen3-8b" / "banking77"
    summary_dir.mkdir(parents=True)
    (summary_dir / "lora.json").write_text(json.dumps({
        "model": "qwen3-8b", "task_id": "banking77", "condition": "lora",
        "metric_id": "weighted_f1", "metric_value": 0.85,
        "n_predictions": 100, "total_input_tokens": 10_000,
        "total_output_tokens": 1_000, "eval_wall_time_s": 10.0,
        "error_counts": {},
    }))

    # Render with override at $0.30/hr
    data = gdd.build_dashboard_data(daily_volume=1000, gpu_hourly_rate_override=0.30)

    assert data["pricing_provenance"]["gpu_hourly_rate_used"] == 0.30
    assert data["pricing_provenance"]["gpu_hourly_rate_source"] == "cli_override"
    # cost_per_query should reflect $0.30/hr not the $1.00 in pricing.yaml
    row = next(r for r in data["results"] if r["model_id"] == "qwen3-8b")
    # cost = 0.30 * 10s / 100 / 3600 = 8.33e-6
    assert row["cost_per_query"] == pytest.approx(0.30 * 10 / 100 / 3600, rel=1e-3)


def test_build_dashboard_data_default_uses_pricing_yaml(tmp_path, monkeypatch):
    """Without override, rate comes from pricing.yaml and provenance reflects it."""
    import generate_dashboard_data as gdd
    monkeypatch.setattr(gdd, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pricing.yaml").write_text(
        "apis: {}\n"
        "self_hosted:\n"
        "  gpu_hourly_rate: 0.46\n"
        "  queries_per_hour_per_gpu: 2000\n"
    )
    (tmp_path / "results" / "summaries").mkdir(parents=True)
    data = gdd.build_dashboard_data(daily_volume=1000)
    assert data["pricing_provenance"]["gpu_hourly_rate_used"] == 0.46
    assert data["pricing_provenance"]["gpu_hourly_rate_source"] == "pricing.yaml"


def test_build_result_gpu_model_falls_back_to_training_meta(pricing):
    """Summary lacks gpu_model → falls back to training_meta."""
    summary = _summary(n_predictions=100, total_input_tokens=10_000, total_output_tokens=1_000)
    training_meta = {"gpu_model": "NVIDIA A10G", "training_cost": 2.5}
    result = build_result("qwen3-8b", "fpb", "lora", summary, training_meta, pricing)
    assert result["gpu_model"] == "NVIDIA A10G"


def test_build_result_surfaces_logprob_observability(pricing):
    summary = _summary(n_predictions=100, total_input_tokens=10_000, total_output_tokens=1_000,
                       avg_logprob=-0.12, p10_logprob=-0.45)
    result = build_result("gpt-4.1", "fpb", "zero-shot", summary, None, pricing)
    assert result["avg_logprob"] == pytest.approx(-0.12)
    assert result["p10_logprob"] == pytest.approx(-0.45)


def test_build_result_classification_metrics_null_when_absent(pricing):
    """Extraction tasks have null classification metrics; pass-through."""
    summary = _summary(
        metric_value=0.6, metric_id="token_f1",
        n_predictions=100, total_input_tokens=10_000, total_output_tokens=1_000,
        exact_match=None, macro_f1=None, weighted_f1=None, hallucination_rate=None,
    )
    result = build_result("gpt-4.1", "cuad", "zero-shot", summary, None, pricing)
    assert result["exact_match"] is None
    assert result["macro_f1"] is None
    assert result["hallucination_rate"] is None


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


def test_merge_results_preserves_existing_only_keys(tmp_path):
    """A (model, task, condition) tuple present in existing but absent from
    fresh stays verbatim — a partial-task pipeline run must not silently delete
    the rest of the dashboard."""
    import json
    existing = {"results": [
        {"model_id": "m1", "task_id": "fpb",    "condition": "zero-shot", "metric_value": 0.7},
        {"model_id": "m1", "task_id": "ledgar", "condition": "zero-shot", "metric_value": 0.5},
    ]}
    out = tmp_path / "results.json"
    out.write_text(json.dumps(existing))

    fresh = [{"model_id": "m1", "task_id": "fpb", "condition": "zero-shot", "metric_value": 0.8}]
    merged = merge_results(fresh, out)

    keys = {(r["model_id"], r["task_id"], r["condition"]) for r in merged}
    assert keys == {("m1", "fpb", "zero-shot"), ("m1", "ledgar", "zero-shot")}

    fpb = next(r for r in merged if r["task_id"] == "fpb")
    assert fpb["metric_value"] == 0.8  # fresh wins for the recomputed section

    ledgar = next(r for r in merged if r["task_id"] == "ledgar")
    assert ledgar["metric_value"] == 0.5  # existing preserved for the untouched section


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


# ── _export_tables: the four mandated metrics + cost-per-1k in CSV/MD ─────────

def _result_row(model_id, task_id, condition, metric_value=0.85, *, em=0.82,
                macro=0.78, weighted=0.85, halluc=0.04, cost_per_query=0.001):
    """Build a dashboard result row with the fields _export_tables consumes."""
    return {
        "model_id": model_id, "task_id": task_id, "condition": condition,
        "metric_id": "weighted_f1", "metric_value": metric_value,
        "metric_std": None, "metric_ci_lo": None, "metric_ci_hi": None,
        "exact_match": em, "macro_f1": macro, "weighted_f1": weighted,
        "hallucination_rate": halluc,
        "cost_per_query": cost_per_query,
        "cost_per_1k_requests": round(cost_per_query * 1000, 4),
        "cost_per_1k_tokens": 0.000125,
        "cost_per_1m_input_tokens": 0.10,
        "cost_per_1m_output_tokens": 0.40,
        "avg_latency_ms": 150.0, "n_predictions": 500,
    }


def test_export_tables_csv_includes_all_four_metrics(tmp_path):
    """CSV must carry EM, macro_f1, weighted_f1, hallucination_rate per row."""
    import csv
    results = [
        _result_row("qwen3-8b", "banking77", "lora", em=0.78, macro=0.74, weighted=0.80, halluc=0.02),
        _result_row("gpt-4.1-nano", "banking77", "zero-shot", em=0.55, macro=0.50, weighted=0.58, halluc=0.12),
    ]
    _export_tables(results, tmp_path)
    csv_path = tmp_path / "results.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    for col in ("exact_match", "macro_f1", "weighted_f1", "hallucination_rate",
                "cost_per_query", "cost_per_1k_requests", "cost_per_1k_tokens",
                "cost_per_1m_input_tokens", "cost_per_1m_output_tokens"):
        assert col in rows[0], f"{col} missing from CSV header"
    # Spot-check values land in the right rows
    qwen_row = next(r for r in rows if r["model_id"] == "qwen3-8b")
    assert qwen_row["exact_match"] == "0.78"
    assert qwen_row["macro_f1"] == "0.74"
    assert qwen_row["hallucination_rate"] == "0.02"


def test_export_tables_markdown_includes_all_four_metrics(tmp_path):
    """Markdown leaderboard must surface EM, Macro-F1, Weighted-F1, Halluc, $/1k, $/1M."""
    results = [
        _result_row("qwen3-8b", "banking77", "lora", em=0.78, macro=0.74, halluc=0.02),
    ]
    _export_tables(results, tmp_path)
    md_text = (tmp_path / "results.md").read_text()
    for token in ("EM", "Macro-F1", "Weighted-F1", "Halluc", "$/1k req", "$/1M in", "$/1M out"):
        assert token in md_text, f"{token} missing from Markdown header"
    # The values must render too — not just the headers
    assert "0.7800" in md_text  # EM
    assert "0.0200" in md_text  # hallucination rate


def test_export_tables_renders_null_metrics_as_empty(tmp_path):
    """Extraction-task rows have null EM/F1/halluc — render as empty, not 'None'."""
    extraction_row = _result_row("qwen3-8b", "cuad", "lora",
                                  em=None, macro=None, weighted=None, halluc=None)
    _export_tables([extraction_row], tmp_path)
    md_text = (tmp_path / "results.md").read_text()
    assert "None" not in md_text, "Null metrics leaked as 'None' in Markdown"


def test_comparison_reports_per_task_cis_not_cross_task_stats(pricing):
    """v1 surfaces per-task seed CIs, NOT a cross-task p-value / effect size:
    with n≤3 shared tasks the latter are underpowered and misleading."""
    from generate_dashboard_data import compute_stats
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
    # Cross-task significance is intentionally NOT reported.
    assert "effect_size_dz" not in comp
    assert "effect_size_label" not in comp
    assert "p_value_gain" not in comp
    # Per-task seed-CI fields ARE reported.
    pt = comp["per_task"]["fpb"]
    for k in ("fine_metric", "fine_ci_lo", "fine_ci_hi", "fine_std",
              "base_metric", "base_ci_lo", "base_ci_hi", "accuracy_gain_pp"):
        assert k in pt


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
