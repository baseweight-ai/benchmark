"""Assemble results.json for the benchmark dashboard from summaries and metadata."""
from __future__ import annotations

import csv
import json
import math
import os
import random
from io import StringIO
from pathlib import Path
from typing import Optional

os.environ.setdefault("LITELLM_LOG", "ERROR")

import click
import litellm
import yaml
from pydantic import BaseModel

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
    "api-sft": "API SFT",
}


def _condition_label(condition: str) -> str:
    return _CONDITION_LABELS.get(condition, condition)


def _format_api_display_name(model_id: str) -> str:
    """'gpt-4.1-nano' → 'GPT 4.1 Nano'."""
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


def load_summary(source: str, model_short: str, task_id: str, condition: str) -> Optional[dict]:
    summaries_root = REPO_ROOT / "results" / "summaries" / source / model_short / task_id
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


def discover_summaries() -> list[tuple[str, str, str, str]]:
    """Return (source, model, task, condition) for base condition files only.

    Skips seed-specific (*_seedN.json) and aggregated (*_agg.json) files —
    load_summary() will transparently return the agg variant when it exists.
    """
    summaries_root = REPO_ROOT / "results" / "summaries"
    found = []
    if not summaries_root.exists():
        return found
    for source_dir in sorted(summaries_root.iterdir()):
        if not source_dir.is_dir():
            continue
        for model_dir in sorted(source_dir.iterdir()):
            if not model_dir.is_dir():
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


def load_training_meta(source: str, model_short: str, task_id: str, condition: str) -> Optional[dict]:
    path = REPO_ROOT / "results" / "training" / source / model_short / task_id / condition / "metadata.json"
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
    avg_latency_ms = summary.get("avg_latency_ms") if summary else None
    eval_wall_time_s = summary.get("eval_wall_time_s") if summary else None
    ttft_p50 = summary.get("ttft_p50_ms") if summary else None
    ttft_p95 = summary.get("ttft_p95_ms") if summary else None
    error_counts = summary.get("error_counts", {}) if summary else {}

    training_cost = training_meta.get("training_cost") if training_meta else None
    training_time_min = training_meta.get("training_time_min") if training_meta else None
    n_train = training_meta.get("n_train") if training_meta else None
    gpu_hours = training_meta.get("gpu_hours") if training_meta else None
    peak_gpu_mem_mb = training_meta.get("peak_gpu_mem_mb") if training_meta else None
    avg_gpu_util_pct = training_meta.get("avg_gpu_util_pct") if training_meta else None
    loss_history = training_meta.get("loss_history") if training_meta else None
    hyperparams = training_meta.get("hyperparams") if training_meta else None
    per_class_metrics = summary.get("per_class_metrics") if summary else None

    # Decomposed cost inputs — stored so the site can recalculate or display assumptions.
    n = n_predictions or 1
    avg_input_tokens = round(total_input / n, 1) if summary else None
    avg_output_tokens = round(total_output / n, 1) if summary else None
    gpu_hourly_rate = pricing.self_hosted.get("gpu_hourly_rate", GPU_HOURLY) if meta["family"] == "open-source" else None
    in_per_tok, out_per_tok = (_api_cost_per_token(model_id) if meta["family"] == "frontier" and summary else (None, None))

    cost_per_query = compute_cost_per_query(
        model_id, total_input, total_output, n, pricing, eval_wall_time_s
    ) if summary else None

    cost_per_1k_correct: Optional[float] = None
    if cost_per_query is not None and metric_value and metric_value > 0:
        cost_per_1k_correct = (cost_per_query * 1000) / metric_value

    tco_12mo = compute_tco_12mo(model_id, training_cost, cost_per_query or 0, daily_volume, pricing, eval_wall_time_s, n_predictions) if cost_per_query is not None else None

    metric_std = summary.get("metric_std") if summary else None
    metric_ci_lo = summary.get("metric_ci_lo") if summary else None
    metric_ci_hi = summary.get("metric_ci_hi") if summary else None
    n_seeds = summary.get("n_seeds") if summary else None

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
        "metric_value": metric_value,
        "metric_std": round(metric_std, 4) if metric_std is not None else None,
        "metric_ci_lo": round(metric_ci_lo, 4) if metric_ci_lo is not None else None,
        "metric_ci_hi": round(metric_ci_hi, 4) if metric_ci_hi is not None else None,
        "n_seeds": n_seeds,
        "n_predictions": n_predictions,
        # Cost (derived)
        "cost_per_query": round(cost_per_query, 8) if cost_per_query is not None else None,
        "cost_per_1k_correct": round(cost_per_1k_correct, 4) if cost_per_1k_correct is not None else None,
        "tco_12mo": round(tco_12mo, 2) if tco_12mo is not None else None,
        # Latency
        "avg_latency_ms": round(avg_latency_ms, 1) if avg_latency_ms is not None else None,
        "ttft_p50_ms": ttft_p50,
        "ttft_p95_ms": ttft_p95,
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
        "per_class_metrics": per_class_metrics,
        # Training details
        "loss_history": loss_history,
        "hyperparams": hyperparams,
        # Cost inputs (decomposed for transparency / recalculation)
        "total_input_tokens": total_input if summary else None,
        "total_output_tokens": total_output if summary else None,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "input_cost_per_token": in_per_tok if in_per_tok else None,
        "output_cost_per_token": out_per_tok if out_per_tok else None,
        "gpu_hourly_rate": gpu_hourly_rate,
        "eval_wall_time_s": eval_wall_time_s,
    }



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


def _comparison(results: list[dict], fine_family: str, fine_cond: str, base_family: str, base_cond: str) -> dict:
    """Compute tasks_won, cost_per_correct_ratio, accuracy gain for one comparison pair."""
    best_fine: dict[str, float] = {}
    best_base: dict[str, float] = {}
    fine_cp1k: dict[str, list[float]] = {}
    base_cp1k: dict[str, list[float]] = {}

    for r in results:
        if r["metric_value"] is None:
            continue
        tid = r["task_id"]
        mv = r["metric_value"]
        if r["family"] == fine_family and r["condition"] == fine_cond:
            if mv > best_fine.get(tid, -1):
                best_fine[tid] = mv
            if r.get("cost_per_1k_correct") is not None:
                fine_cp1k.setdefault(tid, []).append(r["cost_per_1k_correct"])
        if r["family"] == base_family and r["condition"] == base_cond:
            if mv > best_base.get(tid, -1):
                best_base[tid] = mv
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

    p_value = _sign_flip_p_value(acc_deltas) if acc_deltas else None
    return {
        "tasks_won": tasks_won,
        "tasks_total": len(shared_tasks),
        "cost_per_correct_ratio": round(_mean(cp1k_ratios), 1) if cp1k_ratios else None,
        "avg_accuracy_gain_pp": round(_mean(acc_deltas) * 100, 1) if acc_deltas else None,
        "median_accuracy_gain_pp": round(_median(acc_deltas) * 100, 1) if acc_deltas else None,
        "p_value_gain": p_value,
        "per_task": {
            t: {
                "fine_metric": round(best_fine[t], 4),
                "base_metric": round(best_base[t], 4),
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

    # All four comparison pairs
    lora_vs_5shot      = _comparison(results, "open-source", "LoRA",    "frontier", "5-shot")
    lora_vs_zero_shot  = _comparison(results, "open-source", "LoRA",    "frontier", "Zero-shot")
    api_sft_vs_5shot   = _comparison(results, "frontier",    "API SFT", "frontier", "5-shot")
    api_sft_vs_zero    = _comparison(results, "frontier",    "API SFT", "frontier", "Zero-shot")

    # Average cost_per_query by condition (across all tasks/models with data)
    cpq_by_cond: dict[str, list[float]] = {}
    for r in results:
        if r.get("cost_per_query") is not None:
            cpq_by_cond.setdefault(r["condition"], []).append(r["cost_per_query"])
    avg_cost_per_query_by_condition = {
        cond: round(sum(v) / len(v), 8) for cond, v in cpq_by_cond.items()
    }

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
            "api_sft_vs_5shot": api_sft_vs_5shot,
            "api_sft_vs_zero_shot": api_sft_vs_zero,
        },
        # Cost summary
        "cost_summary": {
            "total_training_cost": round(sum(training_costs), 4) if training_costs else None,
            "avg_training_cost": round(sum(training_costs) / len(training_costs), 4) if training_costs else None,
            "avg_training_time_min": round(sum(training_times) / len(training_times), 1) if training_times else None,
            "avg_n_train": round(sum(n_trains) / len(n_trains)) if n_trains else None,
            "avg_cost_per_query_by_condition": avg_cost_per_query_by_condition,
        },
    }


def _write_snapshot(data: dict, repo_root: Path, run_id: str) -> Path:
    """Write an immutable snapshot of results.json for this run."""
    from pipeline.paths import snapshot_path
    from checkpoint_utils import atomic_write_json
    snap = snapshot_path(repo_root, run_id)
    snap.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(data, snap)
    return snap


def _export_tables(results: list[dict], out_dir: Path) -> None:
    """Write CSV and Markdown summary tables to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-task accuracy table: rows = models, cols = conditions
    tasks = sorted({r["task_id"] for r in results})

    # CSV: one row per (model, task, condition)
    csv_path = out_dir / "results.csv"
    fieldnames = ["model_id", "task_id", "condition", "metric_id", "metric_value",
                  "metric_std", "metric_ci_lo", "metric_ci_hi",
                  "cost_per_query", "avg_latency_ms", "n_predictions"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["model_id"], x["task_id"], x["condition"])):
            w.writerow({k: r.get(k) for k in fieldnames})

    # Markdown: accuracy leaderboard per task
    md_lines = ["# Benchmark Results\n"]
    for task in tasks:
        task_rows = [r for r in results if r["task_id"] == task and r.get("metric_value") is not None]
        if not task_rows:
            continue
        md_lines.append(f"\n## {task}\n")
        md_lines.append("| Model | Condition | Metric | Value | Std | CI |")
        md_lines.append("|-------|-----------|--------|-------|-----|-----|")
        for r in sorted(task_rows, key=lambda x: -(x["metric_value"] or 0)):
            std = f"±{r['metric_std']:.4f}" if r.get("metric_std") is not None else ""
            ci = (f"[{r['metric_ci_lo']:.4f},{r['metric_ci_hi']:.4f}]"
                  if r.get("metric_ci_lo") is not None else "")
            md_lines.append(
                f"| {r['model_id']} | {r['condition']} | {r.get('metric_id','')} "
                f"| {r['metric_value']:.4f} | {std} | {ci} |"
            )

    md_path = out_dir / "results.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    click.echo(f"  Tables written to {out_dir}")


def merge_results(fresh: list[dict], existing_path: Path) -> list[dict]:
    """Merge fresh results with existing ones.

    For each (model_id, task_id, condition), the fresh value wins if it has
    metric_value; otherwise the existing non-null value is preserved.
    """
    try:
        with open(existing_path) as f:
            existing = json.load(f)
    except Exception:
        return fresh

    existing_map = {
        (r["model_id"], r["task_id"], r["condition"]): r
        for r in existing.get("results", [])
    }
    merged = []
    for r in fresh:
        key = (r["model_id"], r["task_id"], r["condition"])
        prior = existing_map.get(key)
        if r["metric_value"] is None and prior is not None and prior.get("metric_value") is not None:
            merged.append(prior)
        else:
            merged.append(r)
    return merged


def build_dashboard_data(daily_volume: int = 10000) -> dict:
    """Assemble full BenchmarkData JSON."""
    pricing = load_pricing()

    results = []
    for source, model_id, task_id, condition in discover_summaries():
        summary = load_summary(source, model_id, task_id, condition)
        training_meta = None
        if condition in ("lora", "api-sft"):
            training_meta = load_training_meta(source, model_id, task_id, condition)
        result = build_result(model_id, task_id, condition, summary, training_meta, pricing, daily_volume)
        results.append(result)

    stats = compute_stats(results)

    return {
        "generated_at": None,  # filled at write time
        **stats,
        "results": results,
    }


@click.command()
@click.option("--daily-volume", default=10000, help="Daily query volume for TCO calc")
@click.option("--out", default=None, help="Output path (default: data/benchmark/results.json in site)")
@click.option("--run-id", "run_id", default=None, help="Run ID for immutable snapshot (from pipeline manifest)")
@click.option("--also-benchmark-repo", is_flag=True, help="Also write to dashboard-data/results.json in benchmark repo")
@click.option("--merge", is_flag=True, help="Preserve existing results where new run has no data (matched by model+task+condition)")
@click.option("--export-tables", is_flag=True, help="Also write CSV and Markdown tables to results/tables/")
@click.option("--dry-run", is_flag=True)
def main(
    daily_volume: int,
    out: Optional[str],
    run_id: Optional[str],
    also_benchmark_repo: bool,
    merge: bool,
    export_tables: bool,
    dry_run: bool,
) -> None:
    """Generate dashboard results.json from summaries."""
    from datetime import datetime, timezone
    data = build_dashboard_data(daily_volume)

    # Default output: the site's data/benchmark/results.json
    site_root = REPO_ROOT.parent / "baseweight-site"
    default_out = site_root / "data" / "benchmark" / "results.json"
    out_path = Path(out) if out else default_out

    if merge:
        fresh = data["results"]
        merged = merge_results(fresh, out_path)
        fresh_map = {(r["model_id"], r["task_id"], r["condition"]): r for r in fresh}
        n_preserved = sum(1 for r in merged if fresh_map.get((r["model_id"], r["task_id"], r["condition"])) is not r)
        if n_preserved:
            click.echo(f"  Merge: preserved {n_preserved} existing result(s) with no new data")
        data["results"] = merged
        data.update(compute_stats(merged))

    data["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if run_id:
        data["run_id"] = run_id

    if dry_run:
        n_results = len(data["results"])
        n_with_data = sum(1 for r in data["results"] if r["metric_value"] is not None)
        click.echo(f"  [dry-run] Would write {n_results} results ({n_with_data} with data) to {out_path}")
        if run_id:
            click.echo(f"  [dry-run] Would write snapshot for run {run_id}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    click.echo(f"  Written to {out_path}")

    # Always attempt to write to the site repo; skip silently if it doesn't exist.
    if out_path != default_out and default_out.parent.exists():
        with open(default_out, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"  Also written to {default_out}")

    if also_benchmark_repo:
        repo_out = REPO_ROOT / "dashboard-data" / "results.json"
        repo_out.parent.mkdir(parents=True, exist_ok=True)
        with open(repo_out, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"  Also written to {repo_out}")

    if run_id:
        snap = _write_snapshot(data, REPO_ROOT, run_id)
        click.echo(f"  Snapshot written to {snap}")

    if export_tables:
        _export_tables(data["results"], REPO_ROOT / "results" / "tables")


if __name__ == "__main__":
    main()
