"""Assemble results.json for the benchmark dashboard from summaries and metadata."""
from __future__ import annotations

import csv
import json
import math
import os
import random
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Optional

os.environ.setdefault("LITELLM_LOG", "ERROR")

import click
import litellm
import yaml
from pydantic import BaseModel

from checkpoint_utils import atomic_write_json
from pipeline.paths import smoke_seg, training_meta_path

litellm.suppress_debug_info = True

REPO_ROOT = Path(__file__).parent.parent
ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa"]

# Known API vendor prefixes → canonical capitalisation.
_VENDOR_CAPS: dict[str, str] = {
    "gpt": "GPT",
    "gemini": "Gemini",
    "o1": "O1",
    "o3": "O3",
}

# Suffixes to strip from HF base model names (case-insensitive).
_INSTRUCT_SUFFIXES = ("-instruct", "-it", "-chat", "-hf")

# Condition display names (filesystem names → dashboard labels).
_CONDITION_LABELS: dict[str, str] = {
    "zero-shot": "Zero-shot",
    "5-shot": "5-shot",
    "lora": "LoRA",
}


def _condition_label(condition: str) -> str:
    return _CONDITION_LABELS.get(condition, condition)


def _format_api_display_name(model_id: str) -> str:
    """'gpt-5.4-mini' → 'GPT 5.4 Mini'."""
    parts = model_id.split("-")
    out = []
    for p in parts:
        out.append(_VENDOR_CAPS.get(p.lower(), p.capitalize()))
    return " ".join(out)


def _strip_instruct_suffix(name: str) -> str:
    lower = name.lower()
    for sfx in _INSTRUCT_SUFFIXES:
        if lower.endswith(sfx):
            return name[: -len(sfx)]
    return name


@lru_cache(maxsize=64)
def _get_model_meta(model_id: str) -> dict:
    """Return {"display_name": ..., "family": ...} for any model_id."""
    config_path = REPO_ROOT / "configs" / "training" / f"{model_id}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        hf_id = cfg.get("model_id", model_id)
        # Strip org prefix ("Qwen/Qwen3-8B" → "Qwen3-8B")
        short = hf_id.split("/")[-1]
        display = _strip_instruct_suffix(short)
        display = display[:1].upper() + display[1:] if display else display
        return {"display_name": display, "family": "open-source"}
    return {"display_name": _format_api_display_name(model_id), "family": "frontier"}

# GPU cost for self-hosted models (loaded from pricing.yaml)
GPU_HOURLY = 0.49  # Default GPU hourly rate — override via pricing.yaml
QUERIES_PER_HOUR = 2000


class PricingConfig(BaseModel):
    apis: dict[str, dict]
    self_hosted: dict


def load_pricing() -> PricingConfig:
    path = REPO_ROOT / "configs" / "pricing.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return PricingConfig(**data)


@lru_cache(maxsize=64)
def _api_cost_per_token(model_id: str) -> tuple[float, float]:
    """Return (input_$/token, output_$/token) via litellm for any API model.

    For fine-tuned models (ft:BASE:ORG::HASH), falls back to the versioned
    base model then the unversioned base model since OpenAI charges the same
    inference rate as the base.
    """
    import re

    candidates = [model_id]
    if model_id.startswith("ft:"):
        base_versioned = model_id.split(":")[1]
        candidates.append(base_versioned)
        base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", base_versioned)
        if base != base_versioned:
            candidates.append(base)

    for candidate in candidates:
        try:
            info = litellm.get_model_info(candidate)
            in_cost = info.get("input_cost_per_token") or 0.0
            out_cost = info.get("output_cost_per_token") or 0.0
            if in_cost or out_cost:
                return in_cost, out_cost
        except Exception:
            continue
    return 0.0, 0.0


def compute_cost_per_query(
    model_id: str,
    total_input_tokens: int,
    total_output_tokens: int,
    n_predictions: int,
    pricing: PricingConfig,
    eval_wall_time_s: Optional[float] = None,
) -> Optional[float]:
    """Cost per query in USD."""
    if n_predictions == 0:
        return None

    meta = _get_model_meta(model_id)
    if meta.get("family") == "open-source":
        gpu_hourly = pricing.self_hosted.get("gpu_hourly_rate", GPU_HOURLY)
        if eval_wall_time_s:
            # Actual throughput from the eval run: total wall time / queries.
            # Captures real concurrency and batching behaviour of vLLM.
            return gpu_hourly * eval_wall_time_s / n_predictions / 3600
        qph = pricing.self_hosted.get("queries_per_hour_per_gpu", QUERIES_PER_HOUR)
        return gpu_hourly / qph

    # API models: token usage × live prices from litellm.
    avg_input = total_input_tokens / n_predictions
    avg_output = total_output_tokens / n_predictions
    in_per_tok, out_per_tok = _api_cost_per_token(model_id)
    if not in_per_tok and not out_per_tok:
        return None
    return avg_input * in_per_tok + avg_output * out_per_tok


def compute_tco_12mo(
    model_id: str,
    training_cost: float,
    cost_per_query: float,
    daily_volume: int,
    pricing: PricingConfig,
    eval_wall_time_s: Optional[float] = None,
    n_predictions: Optional[int] = None,
) -> Optional[float]:
    """12-month TCO: training + inference + (for self-hosted) GPU reservation."""
    if cost_per_query is None:
        return None

    meta = _get_model_meta(model_id)
    annual_queries = daily_volume * 365
    inference_cost = cost_per_query * annual_queries

    if meta.get("family") == "open-source":
        gpu_hourly = pricing.self_hosted.get("gpu_hourly_rate", GPU_HOURLY)
        if eval_wall_time_s and n_predictions:
            qph = n_predictions / (eval_wall_time_s / 3600)
        else:
            qph = pricing.self_hosted.get("queries_per_hour_per_gpu", QUERIES_PER_HOUR)
        gpus_needed = math.ceil(daily_volume / (qph * 24))
        gpu_annual = gpus_needed * gpu_hourly * 24 * 365
        return (training_cost or 0.0) + gpu_annual
    else:
        return (training_cost or 0.0) + inference_cost


def load_summary(source: str, model_short: str, task_id: str, condition: str,
                 smoke: bool = False) -> Optional[dict]:
    summaries_root = REPO_ROOT / "results" / smoke_seg(smoke) / "summaries" / source / model_short / task_id
    # Prefer aggregated multi-seed summary when it exists.
    agg_path = summaries_root / f"{condition}_agg.json"
    if agg_path.exists():
        with open(agg_path) as f:
            return json.load(f)
    base_path = summaries_root / f"{condition}.json"
    if base_path.exists():
        with open(base_path) as f:
            return json.load(f)
    return None


def discover_summaries(
    smoke: bool = False,
    allowed_models: Optional[set[str]] = None,
) -> list[tuple[str, str, str, str]]:
    """Return (source, model, task, condition) for base condition files only.

    Skips seed-specific (*_seedN.json) and aggregated (*_agg.json) files —
    load_summary() will transparently return the agg variant when it exists.

    allowed_models: when set, only summaries whose model_id is in the set are
    returned. Used to scope the dashboard to the current pipeline cohort —
    stale summaries on disk from earlier configurations (e.g. an old smoke
    model run before the namespace split) are then ignored. None disables
    filtering (back-compat for callers that just want everything).
    """
    from pipeline.config import get_prod_model_ids, get_smoke_model_ids
    if allowed_models is None:
        allowed_models = get_smoke_model_ids() if smoke else get_prod_model_ids()

    # Defensive: an explicitly-empty cohort would have walked every directory
    # only to skip every model. Short-circuit before touching the filesystem.
    if not allowed_models:
        return []

    summaries_root = REPO_ROOT / "results" / smoke_seg(smoke) / "summaries"
    found = []
    if not summaries_root.exists():
        return found
    for source_dir in sorted(summaries_root.iterdir()):
        if not source_dir.is_dir():
            continue
        for model_dir in sorted(source_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if allowed_models and model_dir.name not in allowed_models:
                continue
            for task_dir in sorted(model_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                for f in sorted(task_dir.glob("*.json")):
                    stem = f.stem
                    if "_seed" in stem or stem.endswith("_agg"):
                        continue
                    found.append((source_dir.name, model_dir.name, task_dir.name, stem))
    return found


def load_training_meta(source: str, model_short: str, task_id: str, condition: str,
                       smoke: bool = False) -> Optional[dict]:
    path = training_meta_path(REPO_ROOT, source, model_short, task_id, condition, smoke=smoke)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def build_result(
    model_id: str,
    task_id: str,
    condition: str,
    summary: Optional[dict],
    training_meta: Optional[dict],
    pricing: PricingConfig,
    daily_volume: int = 10000,
) -> dict:
    """Build a single Result object for the dashboard schema."""
    meta = _get_model_meta(model_id)

    metric_value = summary["metric_value"] if summary else None
    n_predictions = summary["n_predictions"] if summary else None
    total_input = summary.get("total_input_tokens", 0) if summary else 0
    total_output = summary.get("total_output_tokens", 0) if summary else 0
    # Legacy summaries lack these — fall back so answer + reasoning = output.
    total_reasoning = summary.get("total_reasoning_tokens", 0) or 0 if summary else 0
    total_answer = summary.get("total_answer_tokens", total_output) if summary else 0
    # Envelope-aware accounting (tiktoken-counted bare label for API rows that
    # used response_format). None for legacy summaries that never recorded it
    # and for local rows that don't use a JSON envelope. The complement is
    # surfaced as `envelope_overhead_tokens` so cost comparisons can subtract
    # the JSON wrapper cost when the user wants apples-to-apples.
    total_answer_only = summary.get("total_answer_only_tokens") if summary else None
    total_envelope_overhead = summary.get("total_envelope_overhead_tokens") if summary else None
    avg_latency_ms = summary.get("avg_latency_ms") if summary else None
    eval_wall_time_s = summary.get("eval_wall_time_s") if summary else None
    ttft_p50 = summary.get("ttft_p50_ms") if summary else None
    ttft_p95 = summary.get("ttft_p95_ms") if summary else None
    latency_p50_ms = summary.get("latency_p50_ms") if summary else None
    latency_p99_ms = summary.get("latency_p99_ms") if summary else None
    error_counts = dict(summary.get("error_counts", {})) if summary else {}
    # not_applicable is a correct abstention: gold and prediction both signal "no
    # clause present". The classifier keeps it as a distinct fine-grained category
    # (and already maps it to semantic "correct"), but for the public breakdown it
    # should count as correct so the correctness rate is not understated. The
    # fine-grained split stays in the summaries and in the answer_detection_* metrics.
    _abstentions = error_counts.pop("not_applicable", 0)
    if _abstentions:
        error_counts["correct"] = error_counts.get("correct", 0) + _abstentions
    semantic_error_counts = summary.get("semantic_error_counts", {}) if summary else {}
    format_compliance_rate = summary.get("format_compliance_rate") if summary else None
    refusal_rate = summary.get("refusal_rate") if summary else None
    empty_rate = summary.get("empty_rate") if summary else None
    partial_rate = summary.get("partial_rate") if summary else None

    training_cost = training_meta.get("training_cost") if training_meta else None
    training_time_min = training_meta.get("training_time_min") if training_meta else None
    n_train = training_meta.get("n_train") if training_meta else None
    gpu_hours = training_meta.get("gpu_hours") if training_meta else None
    peak_gpu_mem_mb = training_meta.get("peak_gpu_mem_mb") if training_meta else None
    avg_gpu_util_pct = training_meta.get("avg_gpu_util_pct") if training_meta else None
    loss_history = training_meta.get("loss_history") if training_meta else None
    hyperparams = training_meta.get("hyperparams") if training_meta else None
    training_diagnostics = training_meta.get("training_diagnostics") if training_meta else None
    compute_dtype = training_meta.get("compute_dtype") if training_meta else None
    # Prefer eval-time GPU (where latency was measured); API rows have none.
    gpu_model = None
    if summary and summary.get("gpu_model"):
        gpu_model = summary["gpu_model"]
    elif training_meta and training_meta.get("gpu_model"):
        gpu_model = training_meta["gpu_model"]
    per_class_metrics = summary.get("per_class_metrics") if summary else None

    # Decomposed cost inputs — stored so the site can recalculate or display assumptions.
    n = n_predictions or 1
    avg_input_tokens = round(total_input / n, 1) if summary else None
    avg_output_tokens = round(total_output / n, 1) if summary else None
    avg_reasoning_tokens = round(total_reasoning / n, 1) if summary else None
    avg_answer_tokens = round(total_answer / n, 1) if summary else None
    gpu_hourly_rate = pricing.self_hosted.get("gpu_hourly_rate", GPU_HOURLY) if meta["family"] == "open-source" else None
    in_per_tok, out_per_tok = (_api_cost_per_token(model_id) if meta["family"] == "frontier" and summary else (None, None))

    cost_per_query = compute_cost_per_query(
        model_id, total_input, total_output, n, pricing, eval_wall_time_s
    ) if summary else None

    cost_per_1k_correct: Optional[float] = None
    if cost_per_query is not None and metric_value and metric_value > 0:
        cost_per_1k_correct = (cost_per_query * 1000) / metric_value

    cost_per_1k_requests: Optional[float] = (
        round(cost_per_query * 1000, 4) if cost_per_query is not None else None
    )
    cost_per_1m_queries: Optional[float] = (
        round(cost_per_query * 1_000_000, 2) if cost_per_query is not None else None
    )
    # throughput_qps: queries served per second at the eval's observed
    # Queries-per-second at the eval's observed concurrency — also the
    # denominator for self-hosted cost: cost_per_query = gpu_hourly / 3600 / qps.
    throughput_qps: Optional[float] = None
    if eval_wall_time_s and eval_wall_time_s > 0 and n > 0:
        throughput_qps = round(n / eval_wall_time_s, 2)
    total_tokens = total_input + total_output
    cost_per_1k_tokens: Optional[float] = None
    cost_per_1m_input_tokens: Optional[float] = None
    cost_per_1m_output_tokens: Optional[float] = None
    if cost_per_query is not None and total_tokens > 0 and n > 0:
        total_cost = cost_per_query * n
        cost_per_1k_tokens = round(total_cost / total_tokens * 1000, 6)
        if meta["family"] == "frontier" and in_per_tok is not None and out_per_tok is not None:
            cost_per_1m_input_tokens = round(in_per_tok * 1_000_000, 4)
            cost_per_1m_output_tokens = round(out_per_tok * 1_000_000, 4)
        else:
            # Self-hosted shares wall time between prefill and decode — can't
            # attribute input vs output separately, so report the same rate.
            effective = round(total_cost / total_tokens * 1_000_000, 4)
            cost_per_1m_input_tokens = effective
            cost_per_1m_output_tokens = effective

    tco_12mo = compute_tco_12mo(model_id, training_cost, cost_per_query or 0, daily_volume, pricing, eval_wall_time_s, n_predictions) if cost_per_query is not None else None

    metric_std = summary.get("metric_std") if summary else None
    metric_cv = summary.get("metric_cv") if summary else None
    metric_ci_lo = summary.get("metric_ci_lo") if summary else None
    metric_ci_hi = summary.get("metric_ci_hi") if summary else None
    n_seeds = summary.get("n_seeds") if summary else None
    metric_granularity = summary.get("metric_granularity") if summary else None

    exact_match = summary.get("exact_match") if summary else None
    macro_f1 = summary.get("macro_f1") if summary else None
    weighted_f1 = summary.get("weighted_f1") if summary else None
    hallucination_rate = summary.get("hallucination_rate") if summary else None
    api_error_rate = summary.get("api_error_rate") if summary else None
    # Extraction-task metrics (None for classification tasks).
    answer_detection_f1 = summary.get("answer_detection_f1") if summary else None
    precision_at_80_recall = summary.get("precision_at_80_recall") if summary else None
    aupr = summary.get("aupr") if summary else None
    avg_logprob = summary.get("avg_logprob") if summary else None
    p10_logprob = summary.get("p10_logprob") if summary else None

    # Per-axis scores: start from what classify_errors.py computed, then add cost.
    eval_axes: list[str] = summary.get("eval_axes", []) if summary else []
    axis_scores: dict[str, dict] = dict(summary.get("axis_scores", {})) if summary else {}
    if "cost" in eval_axes and cost_per_query is not None:
        axis_scores["cost"] = {"value": round(cost_per_query, 8), "higher_is_better": False}

    return {
        # Identity
        "model_id": model_id,
        "display_name": meta["display_name"],
        "family": meta["family"],
        "task_id": task_id,
        "condition": _condition_label(condition),
        # Accuracy (mean when multi-seed, single value otherwise)
        "metric_id": summary["metric_id"] if summary else None,
        "metric_granularity": metric_granularity,
        "metric_value": metric_value,
        "metric_std": round(metric_std, 4) if metric_std is not None else None,
        "metric_cv": round(metric_cv, 4) if metric_cv is not None else None,
        "metric_ci_lo": round(metric_ci_lo, 4) if metric_ci_lo is not None else None,
        "metric_ci_hi": round(metric_ci_hi, 4) if metric_ci_hi is not None else None,
        "n_seeds": n_seeds,
        "n_predictions": n_predictions,
        "exact_match": exact_match,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "hallucination_rate": hallucination_rate,
        "api_error_rate": api_error_rate,
        "answer_detection_f1": answer_detection_f1,
        "precision_at_80_recall": precision_at_80_recall,
        "aupr": aupr,
        "avg_logprob": avg_logprob,
        "p10_logprob": p10_logprob,
        "cost_per_query": round(cost_per_query, 8) if cost_per_query is not None else None,
        "cost_per_1k_requests": cost_per_1k_requests,
        "cost_per_1m_queries": cost_per_1m_queries,
        "cost_per_1k_tokens": cost_per_1k_tokens,
        "cost_per_1m_input_tokens": cost_per_1m_input_tokens,
        "cost_per_1m_output_tokens": cost_per_1m_output_tokens,
        "cost_per_1k_correct": round(cost_per_1k_correct, 4) if cost_per_1k_correct is not None else None,
        "tco_12mo": round(tco_12mo, 2) if tco_12mo is not None else None,
        "avg_latency_ms": round(avg_latency_ms, 1) if avg_latency_ms is not None else None,
        "latency_p50_ms": latency_p50_ms,
        "latency_p99_ms": latency_p99_ms,
        "ttft_p50_ms": ttft_p50,
        "ttft_p95_ms": ttft_p95,
        "throughput_qps": throughput_qps,
        # Training
        "training_cost": round(training_cost, 4) if training_cost is not None else None,
        "training_time_min": training_time_min,
        "gpu_hours": gpu_hours,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "avg_gpu_util_pct": avg_gpu_util_pct,
        "n_train": n_train,
        # Per-axis scores
        "axis_scores": axis_scores or None,
        # Error breakdown
        "error_counts": error_counts,
        "semantic_error_counts": semantic_error_counts or None,
        "per_class_metrics": per_class_metrics,
        # Pre-computed compliance rates (derivable from error_counts / n_predictions,
        # included here so dashboard consumers don't need to recompute)
        "format_compliance_rate": format_compliance_rate,
        "refusal_rate": refusal_rate,
        "empty_rate": empty_rate,
        "partial_rate": partial_rate,
        # Training details
        "loss_history": loss_history,
        "hyperparams": hyperparams,
        "training_diagnostics": training_diagnostics,
        "compute_dtype": compute_dtype,
        "gpu_model": gpu_model,
        "total_input_tokens": total_input if summary else None,
        "total_output_tokens": total_output if summary else None,
        "total_reasoning_tokens": total_reasoning if summary else None,
        "total_answer_tokens": total_answer if summary else None,
        # Envelope-aware counts (currently populated by eval_api when
        # response_format is set; None elsewhere). Lets cost comparisons
        # subtract the JSON wrapper overhead the API adds.
        "total_answer_only_tokens": total_answer_only if summary else None,
        "total_envelope_overhead_tokens": total_envelope_overhead if summary else None,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_reasoning_tokens": avg_reasoning_tokens,
        "avg_answer_tokens": avg_answer_tokens,
        "input_cost_per_token": in_per_tok if in_per_tok else None,
        "output_cost_per_token": out_per_tok if out_per_tok else None,
        "gpu_hourly_rate": gpu_hourly_rate,
        "eval_wall_time_s": eval_wall_time_s,
    }



# NOTE: _cohens_dz / _effect_label / _sign_flip_p_value compute CROSS-TASK
# significance. They are intentionally NOT surfaced in the v1 comparison —
# with n≤3 shared tasks such statistics are underpowered and misleading, so the
# dashboard reports per-task seed CIs instead (see _comparison). Retained as
# tested utilities for a future benchmark with enough tasks to use them.
def _cohens_dz(deltas: list[float]) -> Optional[float]:
    """Cohen's d_z (paired effect size) with Hedge's small-sample correction.

    Measures how large the mean gain is relative to the cross-task spread, in
    units of the sample standard deviation. The Hedge's correction factor
    J = 1 - 3/(4*(n-1)-1) de-biases Cohen's d for small n; at n=5 it is ~0.80,
    so the correction is substantial. Returns None for n < 2 or zero variance.

    Interpretation (Cohen 1988): |d_z| < 0.2 negligible, 0.2–0.5 small,
    0.5–0.8 medium, > 0.8 large.
    """
    n = len(deltas)
    if n < 3:
        # For n=2, Hedge's J = 1 - 3/(4×1-1) = 0, which collapses d to 0.
        # With fewer than 3 tasks the estimate is not interpretable.
        return None
    mean = sum(deltas) / n
    variance = sum((x - mean) ** 2 for x in deltas) / (n - 1)
    if variance == 0:
        return None
    std = math.sqrt(variance)
    if std < 1e-10:  # near-zero std: all deltas identical, effect size undefined
        return None
    d = mean / std
    j = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    return round(d * j, 4)


def _effect_label(g: Optional[float]) -> Optional[str]:
    """Map |Cohen's d_z| to a Cohen-convention interpretation label."""
    if g is None:
        return None
    abs_g = abs(g)
    if abs_g < 0.2:
        return "negligible"
    if abs_g < 0.5:
        return "small"
    if abs_g < 0.8:
        return "medium"
    return "large"


def _sign_flip_p_value(gains: list[float], n_perm: int = 5000) -> Optional[float]:
    """One-sided sign-flip permutation p-value: P(mean gain ≥ observed | H0: no effect).

    H0 assumes gains are symmetric around 0. We flip each gain's sign randomly and
    ask what fraction of permutations produce a mean as large as observed. Small
    p-value → fine-tuned model is significantly better.
    """
    if len(gains) < 2:
        return None
    rng = random.Random(42)
    obs = sum(gains) / len(gains)
    count = sum(
        1 for _ in range(n_perm)
        if sum(g * rng.choice((-1, 1)) for g in gains) / len(gains) >= obs
    )
    return round(count / n_perm, 4)


# Minimum eval-set size for a row to count toward headline comparisons.
# Anchored to prepare_datasets.SMOKE_TEST_N so the threshold tracks the
# smoke-row size automatically: any row with at least 5× a smoke run is a
# real eval (smallest production test set is ≥150).
from prepare_datasets import SMOKE_TEST_N as _SMOKE_TEST_N
_MIN_REAL_EVAL_N = _SMOKE_TEST_N * 5


def _comparison(results: list[dict], fine_family: str, fine_cond: str, base_family: str, base_cond: str) -> dict:
    """Compute tasks_won, cost_per_correct_ratio, accuracy gain for one comparison pair.

    Rows with `n_predictions < _MIN_REAL_EVAL_N` are excluded — they're smoke
    artefacts or partial runs and would otherwise contribute spurious 0.0-vs-0.0
    pairs to averages. The per-task breakdown likewise only contains tasks
    where both sides have a real evaluation.
    """
    best_fine: dict[str, float] = {}
    best_base: dict[str, float] = {}
    fine_ci: dict[str, dict] = {}   # per-task seed CI for the winning fine row
    base_ci: dict[str, dict] = {}
    fine_cp1k: dict[str, list[float]] = {}
    base_cp1k: dict[str, list[float]] = {}

    def _ci(r: dict) -> dict:
        return {"ci_lo": r.get("metric_ci_lo"), "ci_hi": r.get("metric_ci_hi"),
                "std": r.get("metric_std"), "n_seeds": r.get("n_seeds")}

    for r in results:
        if r["metric_value"] is None:
            continue
        # Skip smoke / partial rows: a real eval has >> _MIN_REAL_EVAL_N rows
        # (≥150 for our smallest task), smoke runs have ≤10. Filter only when
        # n_predictions is explicitly recorded — back-compat for unit tests
        # and legacy result rows that don't carry the field.
        n_pred = r.get("n_predictions")
        if n_pred is not None and n_pred < _MIN_REAL_EVAL_N:
            continue
        tid = r["task_id"]
        mv = r["metric_value"]
        if r["family"] == fine_family and r["condition"] == fine_cond:
            if mv > best_fine.get(tid, -1):
                best_fine[tid] = mv
                fine_ci[tid] = _ci(r)
            if r.get("cost_per_1k_correct") is not None:
                fine_cp1k.setdefault(tid, []).append(r["cost_per_1k_correct"])
        if r["family"] == base_family and r["condition"] == base_cond:
            if mv > best_base.get(tid, -1):
                best_base[tid] = mv
                base_ci[tid] = _ci(r)
            if r.get("cost_per_1k_correct") is not None:
                base_cp1k.setdefault(tid, []).append(r["cost_per_1k_correct"])

    shared_tasks = sorted(set(best_fine) & set(best_base))
    tasks_won = sum(1 for t in shared_tasks if best_fine[t] > best_base[t])

    cp1k_ratios = []
    for tid in set(fine_cp1k) & set(base_cp1k):
        fine_avg = sum(fine_cp1k[tid]) / len(fine_cp1k[tid])
        base_avg = sum(base_cp1k[tid]) / len(base_cp1k[tid])
        if fine_avg > 0:
            cp1k_ratios.append(base_avg / fine_avg)

    acc_deltas = [best_fine[t] - best_base[t] for t in shared_tasks]

    def _mean(xs): return sum(xs) / len(xs) if xs else None
    def _median(xs):
        if not xs: return None
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    # No cross-task significance test: with n≤3 shared tasks a p-value / d_z is
    # underpowered and misleading. The meaningful uncertainty is the per-task
    # seed CI (mean ± multi-seed spread), surfaced per task below.
    return {
        "tasks_won": tasks_won,
        "tasks_total": len(shared_tasks),
        "cost_per_correct_ratio": round(_mean(cp1k_ratios), 1) if cp1k_ratios else None,
        "avg_accuracy_gain_pp": round(_mean(acc_deltas) * 100, 1) if acc_deltas else None,
        "median_accuracy_gain_pp": round(_median(acc_deltas) * 100, 1) if acc_deltas else None,
        "per_task": {
            t: {
                "fine_metric": round(best_fine[t], 4),
                "fine_ci_lo": fine_ci.get(t, {}).get("ci_lo"),
                "fine_ci_hi": fine_ci.get(t, {}).get("ci_hi"),
                "fine_std": fine_ci.get(t, {}).get("std"),
                "fine_n_seeds": fine_ci.get(t, {}).get("n_seeds"),
                "base_metric": round(best_base[t], 4),
                "base_ci_lo": base_ci.get(t, {}).get("ci_lo"),
                "base_ci_hi": base_ci.get(t, {}).get("ci_hi"),
                "base_std": base_ci.get(t, {}).get("std"),
                "base_n_seeds": base_ci.get(t, {}).get("n_seeds"),
                "accuracy_gain_pp": round((best_fine[t] - best_base[t]) * 100, 1),
            }
            for t in shared_tasks
        },
    }


def compute_stats(results: list[dict]) -> dict:
    """Compute all headline stats and comparison breakdowns."""
    # Scope
    task_ids = sorted({r["task_id"] for r in results})
    model_ids = sorted({r["model_id"] for r in results})
    conditions = sorted({r["condition"] for r in results})

    # Cost summary
    training_costs = [r["training_cost"] for r in results if r.get("training_cost") is not None]
    training_times = [r["training_time_min"] for r in results if r.get("training_time_min") is not None]
    n_trains = [r["n_train"] for r in results if r.get("n_train") is not None]

    # Comparison pairs: open-source LoRA against the API baseline conditions.
    lora_vs_5shot      = _comparison(results, "open-source", "LoRA", "frontier", "5-shot")
    lora_vs_zero_shot  = _comparison(results, "open-source", "LoRA", "frontier", "Zero-shot")

    # Average cost_per_query by condition (across all tasks/models with data)
    cpq_by_cond: dict[str, list[float]] = {}
    for r in results:
        if r.get("cost_per_query") is not None:
            cpq_by_cond.setdefault(r["condition"], []).append(r["cost_per_query"])
    avg_cost_per_query_by_condition = {
        cond: round(sum(v) / len(v), 8) for cond, v in cpq_by_cond.items()
    }

    # Precision mismatch warnings: comparing LoRA runs trained at different compute_dtype
    # can introduce systematic differences unrelated to the task (e.g. bfloat16 vs float32
    # produce slightly different weight updates, affecting convergence and final accuracy).
    dtype_warnings: list[str] = []
    lora_dtypes: dict[str, str] = {}
    for r in results:
        if r.get("condition") == "LoRA" and r.get("compute_dtype"):
            lora_dtypes[r["model_id"]] = r["compute_dtype"]
    if len(set(lora_dtypes.values())) > 1:
        dtype_warnings.append(
            "Comparing LoRA runs with mixed compute_dtype: "
            + ", ".join(f"{m}={d}" for m, d in sorted(lora_dtypes.items()))
        )

    headline = lora_vs_5shot  # the primary comparison

    return {
        # Headline stats (flat, for dashboard backward-compat)
        "tasks_won_by_oss": headline["tasks_won"],
        "cost_per_correct_ratio": headline["cost_per_correct_ratio"],
        "avg_accuracy_gain_pp": headline["avg_accuracy_gain_pp"],
        # Scope
        "scope": {
            "n_tasks": len(task_ids),
            "n_models": len(model_ids),
            "n_conditions": len(conditions),
            "task_ids": task_ids,
            "model_ids": model_ids,
            "conditions": conditions,
        },
        # All comparisons
        "comparisons": {
            "lora_vs_5shot": lora_vs_5shot,
            "lora_vs_zero_shot": lora_vs_zero_shot,
        },
        # Cost summary
        "cost_summary": {
            "total_training_cost": round(sum(training_costs), 4) if training_costs else None,
            "avg_training_cost": round(sum(training_costs) / len(training_costs), 4) if training_costs else None,
            "avg_training_time_min": round(sum(training_times) / len(training_times), 1) if training_times else None,
            "avg_n_train": round(sum(n_trains) / len(n_trains)) if n_trains else None,
            "avg_cost_per_query_by_condition": avg_cost_per_query_by_condition,
        },
        # Non-empty when runs being compared used different precisions — a confound
        # that can produce systematic accuracy differences unrelated to the task.
        "dtype_warnings": dtype_warnings,
    }


def _print_run_report(data: dict) -> None:
    """Print a human-readable benchmark summary after dashboard generation."""
    sep = "  " + "─" * 64
    scope = data.get("scope", {})
    comps = data.get("comparisons", {})
    cost_sum = data.get("cost_summary", {})

    click.echo(
        f"\n  Benchmark Report  "
        f"({scope.get('n_tasks', '?')} tasks | "
        f"{scope.get('n_models', '?')} models | "
        f"{scope.get('n_conditions', '?')} conditions)"
    )
    click.echo(sep)

    comp_labels = [
        ("lora_vs_5shot",       "LoRA vs 5-shot"),
        ("lora_vs_zero_shot",   "LoRA vs zero-shot"),
    ]
    for key, label in comp_labels:
        c = comps.get(key, {})
        if not c or not c.get("tasks_total"):
            continue
        won, total = c["tasks_won"], c["tasks_total"]
        gain    = c.get("avg_accuracy_gain_pp")
        ratio   = c.get("cost_per_correct_ratio")

        gain_str  = f"{gain:+.1f}pp" if gain is not None else "n/a"
        ratio_str = f"  cost_ratio={ratio:.1f}x" if ratio is not None else ""

        # Per-task seed CIs replace any cross-task p-value (underpowered at n≤3 tasks).
        click.echo(f"  {label:<26} {won}/{total} tasks  {gain_str:<10}{ratio_str}")
        for tid, t in sorted(c.get("per_task", {}).items()):
            sign = "+" if t["accuracy_gain_pp"] >= 0 else ""
            flo, fhi = t.get("fine_ci_lo"), t.get("fine_ci_hi")
            blo, bhi = t.get("base_ci_lo"), t.get("base_ci_hi")
            fci = f" [{flo:.3f},{fhi:.3f}]" if flo is not None and fhi is not None else ""
            bci = f" [{blo:.3f},{bhi:.3f}]" if blo is not None and bhi is not None else ""
            click.echo(
                f"    {tid:<22}  {t['fine_metric']:.3f}{fci} vs {t['base_metric']:.3f}{bci}"
                f"  {sign}{t['accuracy_gain_pp']:.1f}pp"
            )

    # ── Baseline reference ─────────────────────────────────────────────────
    task_baselines = data.get("task_baselines", {})
    if task_baselines:
        click.echo(sep)
        click.echo("  Reference baselines (for contextualising absolute scores):")
        for tid in sorted(task_baselines):
            b = task_baselines[tid]
            rc  = b.get("random_chance")
            mjr = b.get("majority_class_accuracy")
            mde = b.get("min_detectable_effect_pp")
            parts = []
            if rc is not None:
                parts.append(f"random={rc:.3f}")
            if mjr is not None:
                parts.append(f"majority={mjr:.3f}")
            if mde is not None:
                parts.append(f"MDE≈±{mde:.1f}pp")
            if parts:
                click.echo(f"    {tid:<20}  {' '.join(parts)}")

    # ── Non-converged training warnings ───────────────────────────────────
    non_converged = [
        r for r in data.get("results", [])
        if (r.get("training_diagnostics") or {}).get("converged") is False
    ]
    if non_converged:
        click.echo(sep)
        click.echo(f"  WARNING: {len(non_converged)} run(s) did not converge during training "
                   f"(loss improved <5%) — eval metrics may reflect an undertrained model:")
        for r in non_converged:
            diag = r.get("training_diagnostics", {})
            imp = diag.get("loss_improvement_pct")
            imp_str = f"{imp:.1f}%" if imp is not None else "?"
            click.echo(f"    {r['model_id']}/{r['task_id']}/{r['condition']}  "
                       f"loss_improvement={imp_str}")

    # ── Overfitting warnings ───────────────────────────────────────────────
    overfitting = [
        r for r in data.get("results", [])
        if (r.get("training_diagnostics") or {}).get("overfitting_detected") is True
    ]
    if overfitting:
        click.echo(sep)
        click.echo(f"  WARNING: {len(overfitting)} run(s) showed overfitting "
                   f"(val_loss rose while train_loss fell):")
        for r in overfitting:
            click.echo(f"    {r['model_id']}/{r['task_id']}/{r['condition']}")

    click.echo(sep)

    cpq = cost_sum.get("avg_cost_per_query_by_condition", {})
    if cpq:
        cost_parts = "  ".join(f"{cond}=${v:.6f}" for cond, v in sorted(cpq.items()))
        click.echo(f"  Avg cost/query:     {cost_parts}")
    total_tc = cost_sum.get("total_training_cost")
    if total_tc is not None:
        click.echo(f"  Total training cost:  ${total_tc:.4f}")

    click.echo(sep)


def _write_snapshot(data: dict, repo_root: Path, run_id: str, smoke: bool = False) -> Path:
    """Write an immutable snapshot of results.json for this run."""
    from pipeline.paths import snapshot_path
    snap = snapshot_path(repo_root, run_id, smoke=smoke)
    snap.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(data, snap)
    return snap


def _export_tables(results: list[dict], out_dir: Path) -> None:
    """Write CSV and Markdown summary tables to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-task accuracy table: rows = models, cols = conditions
    tasks = sorted({r["task_id"] for r in results})

    csv_path = out_dir / "results.csv"
    fieldnames = ["model_id", "task_id", "condition", "metric_id", "metric_value",
                  "metric_std", "metric_ci_lo", "metric_ci_hi",
                  "exact_match", "macro_f1", "weighted_f1", "hallucination_rate",
                  "cost_per_query", "cost_per_1k_requests", "cost_per_1k_tokens",
                  "cost_per_1m_input_tokens", "cost_per_1m_output_tokens",
                  "avg_latency_ms", "n_predictions"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["model_id"], x["task_id"], x["condition"])):
            w.writerow({k: r.get(k) for k in fieldnames})

    md_lines = ["# Benchmark Results\n"]
    for task in tasks:
        task_rows = [r for r in results if r["task_id"] == task and r.get("metric_value") is not None]
        if not task_rows:
            continue
        md_lines.append(f"\n## {task}\n")
        md_lines.append(
            "| Model | Condition | Metric | Value | Std | CI | EM | Macro-F1 | "
            "Weighted-F1 | Halluc | $/1k req | $/1M in | $/1M out |"
        )
        md_lines.append(
            "|-------|-----------|--------|-------|-----|-----|----|----------|"
            "-------------|--------|----------|---------|----------|"
        )
        for r in sorted(task_rows, key=lambda x: -(x["metric_value"] or 0)):
            std = f"±{r['metric_std']:.4f}" if r.get("metric_std") is not None else ""
            ci = (f"[{r['metric_ci_lo']:.4f},{r['metric_ci_hi']:.4f}]"
                  if r.get("metric_ci_lo") is not None else "")
            em = f"{r['exact_match']:.4f}" if r.get("exact_match") is not None else ""
            mf1 = f"{r['macro_f1']:.4f}" if r.get("macro_f1") is not None else ""
            wf1 = f"{r['weighted_f1']:.4f}" if r.get("weighted_f1") is not None else ""
            halluc = f"{r['hallucination_rate']:.4f}" if r.get("hallucination_rate") is not None else ""
            cpr = f"${r['cost_per_1k_requests']:.4f}" if r.get("cost_per_1k_requests") is not None else ""
            cin = f"${r['cost_per_1m_input_tokens']:.4f}" if r.get("cost_per_1m_input_tokens") is not None else ""
            cout = f"${r['cost_per_1m_output_tokens']:.4f}" if r.get("cost_per_1m_output_tokens") is not None else ""
            md_lines.append(
                f"| {r['model_id']} | {r['condition']} | {r.get('metric_id','')} "
                f"| {r['metric_value']:.4f} | {std} | {ci} "
                f"| {em} | {mf1} | {wf1} | {halluc} | {cpr} | {cin} | {cout} |"
            )

    md_path = out_dir / "results.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    click.echo(f"  Tables written to {out_dir}")


def _load_task_baselines() -> dict[str, dict]:
    """Compute per-task reference baselines from quality reports.

    Returns a dict keyed by task_id with:
      random_chance: 1/n_classes (accuracy of uniform random predictor)
      majority_class_accuracy: accuracy of always-predicting-the-most-common-class
      min_detectable_effect_pp: approximate MDE (pp) at α=0.05, 80% power for two-proportion z-test
      n_test: number of test examples used for evaluation
      n_classes: number of distinct classes
    """
    baselines: dict[str, dict] = {}
    for task_id in ALL_TASKS:
        qr_path = REPO_ROOT / "data" / "prepared" / task_id / "quality_report.json"
        # NB: build_dashboard_data's caller (main) sets smoke when needed; the
        # qr_path here is shared with non-smoke (test_full.jsonl quality report
        # is task-level metadata, not a run output) so it stays in data/prepared/.
        try:
            with open(qr_path) as f:
                qr = json.load(f)
        except FileNotFoundError:
            continue

        test_dist = (qr.get("prepared", {}).get("test", {}) or {}).get("label_distribution", {})
        if not test_dist:
            continue

        total = sum(v["count"] for v in test_dist.values())
        if total == 0:
            continue

        n_classes = len(test_dist)
        majority_count = max(v["count"] for v in test_dist.values())

        # Minimum detectable effect (pp) for a two-proportion z-test:
        # MDE ≈ 2.8 × sqrt(p(1-p)/n) where p ≈ 0.5 (most sensitive near 50%)
        # This is an approximation; the true MDE depends on baseline rate.
        mde_pp = round(2.8 * math.sqrt(0.25 / total) * 100, 1) if total > 0 else None

        baselines[task_id] = {
            "random_chance": round(1.0 / n_classes, 4),
            "majority_class_accuracy": round(majority_count / total, 4),
            "min_detectable_effect_pp": mde_pp,
            "n_test": total,
            "n_classes": n_classes,
        }
    return baselines


def merge_results(
    fresh: list[dict],
    existing_path: Path,
    allowed_models: Optional[set[str]] = None,
) -> list[dict]:
    """Merge fresh results into existing ones, keyed by (model_id, task_id, condition).

    Per key:
      - fresh value wins if non-null (overwrites that section);
      - else existing non-null value is preserved.
    Plus: keys present in existing but absent from fresh are kept verbatim, so a
    pipeline that only recomputes some (m,t,c) sections does not delete the rest.

    allowed_models: when set, rows from either side whose model_id is outside
    the cohort are skipped — keeps stale rows from earlier pipeline
    configurations (e.g. renamed smoke models) out of the published JSON.
    """
    try:
        with open(existing_path) as f:
            existing = json.load(f)
    except Exception:
        return [r for r in fresh if allowed_models is None or r["model_id"] in allowed_models]

    def _in_cohort(r: dict) -> bool:
        return allowed_models is None or r.get("model_id") in allowed_models

    existing_map = {
        (r["model_id"], r["task_id"], r["condition"]): r
        for r in existing.get("results", []) if _in_cohort(r)
    }
    fresh_keys = {(r["model_id"], r["task_id"], r["condition"]) for r in fresh if _in_cohort(r)}

    merged = []
    for r in fresh:
        if not _in_cohort(r):
            continue
        key = (r["model_id"], r["task_id"], r["condition"])
        prior = existing_map.get(key)
        if r["metric_value"] is None and prior is not None and prior.get("metric_value") is not None:
            merged.append(prior)
        else:
            merged.append(r)

    # Sections this run did not recompute stay verbatim — partial-task pipelines
    # (a single-model retrain, an eval-api-only refresh) must not silently
    # delete the rest of the benchmark.
    for key, prior in existing_map.items():
        if key not in fresh_keys:
            merged.append(prior)

    return merged


def _summarise_hardware(results: list[dict]) -> dict:
    """Walk local result rows and surface GPU consistency state.

    Latency / throughput / cost are GPU-bound. If different local rows ran on
    different silicon, comparing them on those axes is invalid. This block
    makes the situation visible in the dashboard JSON so consumers can either
    filter or raise the issue.
    """
    by_model: dict[str, set] = {}
    for r in results:
        if r.get("family") != "open-source":
            continue
        gpu = r.get("gpu_model")
        if not gpu:
            continue
        by_model.setdefault(r["model_id"], set()).add(gpu)

    inconsistent = {m: sorted(gpus) for m, gpus in by_model.items() if len(gpus) > 1}
    all_local_gpus = sorted({g for gpus in by_model.values() for g in gpus})
    return {
        "local_gpus_observed": all_local_gpus,
        # Models where >1 GPU was seen across runs — latency/throughput/cost
        # comparisons within those rows are not apples-to-apples.
        "inconsistent_gpu_models": inconsistent,
        "hardware_warning": (
            f"Multiple GPUs detected for the same model_id: {inconsistent}. "
            f"Latency/throughput/cost columns are not directly comparable. "
            f"Re-run on a single GPU for publication."
        ) if inconsistent else None,
    }


def build_dashboard_data(
    daily_volume: int = 10000,
    gpu_hourly_rate_override: Optional[float] = None,
    smoke: bool = False,
) -> dict:
    """Assemble full BenchmarkData JSON.

    gpu_hourly_rate_override: if set, replaces pricing.yaml's gpu_hourly_rate
    for this render. Lets you re-generate results.json with an ICP-realistic
    rate (e.g. AWS A10G on-demand) without re-running eval — the cost formulas
    are linear in the rate, so a fresh dashboard render is enough.

    smoke: when True, read summaries from results/smoke/summaries/ so a smoke
    pipeline dashboard renders against its own (throwaway) data rather than
    the published benchmark.
    """
    # Reset per-render so monkeypatch'd REPO_ROOT in tests doesn't see
    # cached values from a previous invocation with a different repo root.
    _get_model_meta.cache_clear()
    _api_cost_per_token.cache_clear()
    pricing = load_pricing()
    if gpu_hourly_rate_override is not None:
        pricing = PricingConfig(
            apis=pricing.apis,
            self_hosted={**pricing.self_hosted, "gpu_hourly_rate": gpu_hourly_rate_override},
        )

    results = []
    for source, model_id, task_id, condition in discover_summaries(smoke=smoke):
        summary = load_summary(source, model_id, task_id, condition, smoke=smoke)
        training_meta = None
        if condition == "lora":
            training_meta = load_training_meta(source, model_id, task_id, condition, smoke=smoke)
        result = build_result(model_id, task_id, condition, summary, training_meta, pricing, daily_volume)
        results.append(result)

    stats = compute_stats(results)
    task_baselines = _load_task_baselines()
    hardware = _summarise_hardware(results)
    pricing_provenance = {
        "gpu_hourly_rate_used": pricing.self_hosted.get("gpu_hourly_rate"),
        "gpu_hourly_rate_source": (
            "cli_override" if gpu_hourly_rate_override is not None else "pricing.yaml"
        ),
        "gpu_pricing_notes": pricing.self_hosted.get("gpu_pricing_notes"),
    }

    return {
        "generated_at": None,  # filled at write time
        **stats,
        "task_baselines": task_baselines,
        "hardware": hardware,
        "pricing_provenance": pricing_provenance,
        "results": results,
    }


@click.command()
@click.option("--daily-volume", default=10000, help="Daily query volume for TCO calc")
@click.option("--out", default=None, help="Output path (default: data/benchmark/results.json in site)")
@click.option("--run-id", "run_id", default=None, help="Run ID for immutable snapshot (from pipeline manifest)")
@click.option("--also-benchmark-repo", is_flag=True, help="Also write to dashboard-data/results.json in benchmark repo")
@click.option("--replace", is_flag=True,
              help="Replace results.json entirely. Default is to merge new results into existing per (model, task, condition).")
@click.option("--export-tables", is_flag=True, help="Also write CSV and Markdown tables to results/tables/")
@click.option("--gpu-hourly-rate", "gpu_hourly_rate", default=None, type=float,
              help="Override self_hosted.gpu_hourly_rate for this render only — useful for re-rendering under an alternative ICP-realistic rate (e.g. AWS spot). Pricing.yaml is unchanged.")
@click.option("--dry-run", is_flag=True)
@click.option("--smoke-test", is_flag=True,
              help="Render dashboard for a smoke pipeline; routes outputs (including dashboard-data/results.json) under a smoke namespace.")
def main(
    daily_volume: int,
    out: Optional[str],
    run_id: Optional[str],
    also_benchmark_repo: bool,
    replace: bool,
    export_tables: bool,
    gpu_hourly_rate: Optional[float],
    dry_run: bool,
    smoke_test: bool,
) -> None:
    """Generate dashboard results.json from summaries."""
    from datetime import datetime, timezone
    data = build_dashboard_data(daily_volume, gpu_hourly_rate_override=gpu_hourly_rate, smoke=smoke_test)

    # Surface hardware-consistency warning prominently — easy to miss in JSON.
    hw_warning = data.get("hardware", {}).get("hardware_warning")
    if hw_warning:
        click.echo(f"  WARNING: {hw_warning}", err=True)
    pp = data.get("pricing_provenance", {})
    rate_used = pp.get("gpu_hourly_rate_used")
    rate_src = pp.get("gpu_hourly_rate_source")
    if rate_used is not None:
        click.echo(f"  GPU hourly rate used: ${rate_used} (source: {rate_src})")

    for w in data.get("dtype_warnings", []):
        click.echo(f"  WARNING: {w}", err=True)

    # Default output: the site's data/benchmark/results.json
    site_root = REPO_ROOT.parent / "baseweight-site"
    default_out = site_root / "data" / "benchmark" / "results.json"
    out_path = Path(out) if out else default_out

    # Merge is the default — existing per-(m,t,c) sections are preserved unless
    # this run produces new data for them; --replace forces a clean rebuild.
    merge = not replace
    if merge:
        from pipeline.config import get_prod_model_ids, get_smoke_model_ids
        allowed_models = get_smoke_model_ids() if smoke_test else get_prod_model_ids()
        fresh = data["results"]
        merged = merge_results(fresh, out_path, allowed_models=allowed_models)
        fresh_map = {(r["model_id"], r["task_id"], r["condition"]): r for r in fresh}
        n_preserved = sum(1 for r in merged if fresh_map.get((r["model_id"], r["task_id"], r["condition"])) is not r)
        if n_preserved:
            click.echo(f"  Merge: preserved {n_preserved} existing result(s) with no new data")
        data["results"] = merged
        merged_stats = compute_stats(merged)
        data.update(merged_stats)
        for w in merged_stats.get("dtype_warnings", []):
            click.echo(f"  WARNING: {w}", err=True)

    data["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if run_id:
        data["run_id"] = run_id

    _print_run_report(data)

    if dry_run:
        n_results = len(data["results"])
        n_with_data = sum(1 for r in data["results"] if r["metric_value"] is not None)
        click.echo(f"  [dry-run] Would write {n_results} results ({n_with_data} with data) to {out_path}")
        if run_id:
            click.echo(f"  [dry-run] Would write snapshot for run {run_id}")
        return

    atomic_write_json(data, out_path)
    click.echo(f"  Written to {out_path}")

    # Always attempt to write to the site repo; skip silently if it doesn't exist.
    if out_path != default_out and default_out.parent.exists():
        atomic_write_json(data, default_out)
        click.echo(f"  Also written to {default_out}")

    if also_benchmark_repo:
        # Smoke routes to dashboard-data/smoke/ so a smoke render cannot
        # overwrite the published (committed) benchmark dashboard.
        repo_out = REPO_ROOT / "dashboard-data" / smoke_seg(smoke_test) / "results.json"
        atomic_write_json(data, repo_out)
        click.echo(f"  Also written to {repo_out}")

    if run_id:
        snap = _write_snapshot(data, REPO_ROOT, run_id, smoke=smoke_test)
        click.echo(f"  Snapshot written to {snap}")

    if export_tables:
        _export_tables(data["results"], REPO_ROOT / "results" / "tables")


if __name__ == "__main__":
    main()
