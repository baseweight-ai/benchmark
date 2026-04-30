"""Assemble results.json for the benchmark dashboard from summaries and metadata."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import click
import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).parent.parent
ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa"]

# Model display names and families
MODEL_META = {
    # open-source (QLoRA fine-tuned)
    "qwen3-8b":         {"display_name": "Qwen3-8B (LoRA)",         "family": "open-source"},
    "gemma3-4b":        {"display_name": "Gemma 3 4B (LoRA)",       "family": "open-source"},
    "phi4-mini":        {"display_name": "Phi-4 Mini (LoRA)",        "family": "open-source"},
    # OpenAI frontier
    "gpt-5.4":          {"display_name": "GPT-5.4",                  "family": "frontier"},
    "gpt-4.1":          {"display_name": "GPT-4.1",                  "family": "frontier"},
    "gpt-4.1-mini":     {"display_name": "GPT-4.1 Mini",             "family": "frontier"},
    "gpt-4.1-nano":     {"display_name": "GPT-4.1 Nano",             "family": "frontier"},
    # Anthropic frontier
    "claude-sonnet-4":  {"display_name": "Claude Sonnet 4",          "family": "frontier"},
    # Google frontier
    "gemini-2.5-flash": {"display_name": "Gemini 2.5 Flash",         "family": "frontier"},
    # API fine-tuned
    "gpt-4.1-sft":      {"display_name": "GPT-4.1 (SFT-500)",       "family": "api-finetuned"},
}

# Conditions per model
MODEL_CONDITIONS = {
    "qwen3-8b":         ["lora-500", "lora-full"],
    "gemma3-4b":        ["lora-500", "lora-full"],
    "phi4-mini":        ["lora-500", "lora-full"],
    "gpt-5.4":          ["zero-shot", "5-shot"],
    "gpt-4.1":          ["zero-shot", "5-shot"],
    "gpt-4.1-mini":     ["zero-shot", "5-shot"],
    "gpt-4.1-nano":     ["zero-shot", "5-shot"],
    "claude-sonnet-4":  ["zero-shot", "5-shot"],
    "gemini-2.5-flash": ["zero-shot", "5-shot"],
    "gpt-4.1-sft":      ["api-sft-500"],
}

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


def compute_cost_per_query(
    model_id: str,
    total_input_tokens: int,
    total_output_tokens: int,
    n_predictions: int,
    pricing: PricingConfig,
) -> Optional[float]:
    """Cost per query in USD."""
    if n_predictions == 0:
        return None

    # Self-hosted models
    meta = MODEL_META.get(model_id, {})
    if meta.get("family") == "open-source":
        cost_per_hour = pricing.self_hosted.get("gpu_hourly_rate", GPU_HOURLY)
        qph = pricing.self_hosted.get("queries_per_hour_per_gpu", QUERIES_PER_HOUR)
        return cost_per_hour / qph

    # API models
    model_pricing = pricing.apis.get(model_id)
    if not model_pricing:
        return None

    input_cost = (total_input_tokens / n_predictions / 1_000_000) * model_pricing.get("input_per_m", 0)
    output_cost = (total_output_tokens / n_predictions / 1_000_000) * model_pricing.get("output_per_m", 0)
    return input_cost + output_cost


def compute_tco_12mo(
    model_id: str,
    training_cost: float,
    cost_per_query: float,
    daily_volume: int,
    pricing: PricingConfig,
) -> Optional[float]:
    """12-month TCO: training + inference + (for self-hosted) GPU reservation."""
    if cost_per_query is None:
        return None

    meta = MODEL_META.get(model_id, {})
    annual_queries = daily_volume * 365
    inference_cost = cost_per_query * annual_queries

    if meta.get("family") == "open-source":
        # GPU reservation cost
        qph = pricing.self_hosted.get("queries_per_hour_per_gpu", QUERIES_PER_HOUR)
        gpu_hourly = pricing.self_hosted.get("gpu_hourly_rate", GPU_HOURLY)
        gpus_needed = math.ceil(daily_volume / (qph * 24))
        gpu_annual = gpus_needed * gpu_hourly * 24 * 365
        return training_cost + gpu_annual
    else:
        return training_cost + inference_cost


def load_summary(model_short: str, task_id: str, condition: str) -> Optional[dict]:
    for source in ("local", "api"):
        path = REPO_ROOT / "results" / "summaries" / source / model_short / task_id / f"{condition}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def load_training_meta(model_short: str, task_id: str, condition: str) -> Optional[dict]:
    path = REPO_ROOT / "results" / "training" / model_short / task_id / condition / "metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_sft_training_meta(task_id: str) -> Optional[dict]:
    path = REPO_ROOT / "results" / "training" / "gpt-4.1-sft" / task_id / "metadata.json"
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
    meta = MODEL_META.get(model_id, {"display_name": model_id, "family": "frontier"})

    metric_value = summary["metric_value"] if summary else None
    n_predictions = summary["n_predictions"] if summary else None
    total_input = summary.get("total_input_tokens", 0) if summary else 0
    total_output = summary.get("total_output_tokens", 0) if summary else 0
    ttft_p50 = summary.get("ttft_p50_ms") if summary else None
    ttft_p95 = summary.get("ttft_p95_ms") if summary else None
    error_counts = summary.get("error_counts", {}) if summary else {}

    training_cost = training_meta.get("training_cost", 0.0) if training_meta else 0.0
    training_time_min = training_meta.get("training_time_min") if training_meta else None
    n_train = training_meta.get("n_train") if training_meta else None

    cost_per_query = compute_cost_per_query(
        model_id, total_input, total_output, n_predictions or 1, pricing
    ) if summary else None

    cost_per_1k_correct: Optional[float] = None
    if cost_per_query is not None and metric_value and metric_value > 0:
        cost_per_1k_correct = (cost_per_query * 1000) / metric_value

    tco_12mo = compute_tco_12mo(model_id, training_cost, cost_per_query or 0, daily_volume, pricing) if cost_per_query is not None else None

    return {
        "model_id": model_id,
        "display_name": meta["display_name"],
        "family": meta["family"],
        "task_id": task_id,
        "condition": condition,
        "metric_id": summary["metric_id"] if summary else None,
        "metric_value": metric_value,
        "cost_per_query": round(cost_per_query, 8) if cost_per_query is not None else None,
        "cost_per_1k_correct": round(cost_per_1k_correct, 4) if cost_per_1k_correct is not None else None,
        "ttft_p50_ms": ttft_p50,
        "ttft_p95_ms": ttft_p95,
        "training_cost": round(training_cost, 4) if training_cost else None,
        "training_time_min": training_time_min,
        "n_train": n_train,
        "tco_12mo": round(tco_12mo, 2) if tco_12mo is not None else None,
        "error_counts": error_counts,
    }


def build_efficiency_points(task_id: str) -> list[dict]:
    """Build efficiency curve data for qwen3-8b (all efficiency sizes)."""
    points = []
    task_path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(task_path) as f:
        task_data = yaml.safe_load(f)
    sizes = task_data.get("efficiency_curve_sizes", [])

    for n in sizes:
        cond = f"lora-{n}"
        summary = load_summary("qwen3-8b", task_id, cond)
        training_meta = load_training_meta("qwen3-8b", task_id, cond)
        n_train_actual = training_meta.get("n_train", n) if training_meta else n
        metric_value = summary["metric_value"] if summary else None
        training_cost = training_meta.get("training_cost", 0.0) if training_meta else 0.0
        points.append({
            "n_train": n_train_actual,
            "condition": cond,
            "metric_value": metric_value,
            "training_cost": round(training_cost, 4),
        })

    return points


def compute_stats(results: list[dict]) -> tuple[dict, int, float]:
    """Compute headline_stats, tasks_won_by_oss, and total_cost from a results list."""
    best_oss_per_task: dict[str, float] = {}
    best_frontier_per_task: dict[str, float] = {}

    best_oss: Optional[dict] = None
    best_frontier: Optional[dict] = None
    best_sft: Optional[dict] = None

    seen_training: set[tuple] = set()
    total_cost = 0.0

    for r in results:
        key = (r["model_id"], r["condition"])
        if key not in seen_training:
            seen_training.add(key)
            total_cost += r["training_cost"] or 0

        if r["metric_value"] is None:
            continue
        mv = r["metric_value"]
        tid = r["task_id"]
        if r["family"] == "open-source" and r["condition"] == "lora-500":
            if best_oss is None or mv > best_oss["metric_value"]:
                best_oss = r
            if mv > best_oss_per_task.get(tid, -1):
                best_oss_per_task[tid] = mv
        if r["family"] == "frontier" and r["condition"] == "5-shot":
            if best_frontier is None or mv > best_frontier["metric_value"]:
                best_frontier = r
            if mv > best_frontier_per_task.get(tid, -1):
                best_frontier_per_task[tid] = mv
        if r["family"] == "api-finetuned":
            if best_sft is None or mv > best_sft["metric_value"]:
                best_sft = r

    tasks_won = sum(
        1 for tid in best_oss_per_task
        if best_oss_per_task[tid] > best_frontier_per_task.get(tid, -1)
    )

    headline_stats = {
        "best_oss_lora500_vs_frontier": {
            "oss_model": best_oss["display_name"] if best_oss else None,
            "oss_metric": best_oss["metric_value"] if best_oss else None,
            "frontier_model": best_frontier["display_name"] if best_frontier else None,
            "frontier_metric": best_frontier["metric_value"] if best_frontier else None,
            "delta": round(best_oss["metric_value"] - best_frontier["metric_value"], 4) if best_oss and best_frontier else None,
        },
        "oss_cost_reduction": {
            "oss_cost_per_query": best_oss["cost_per_query"] if best_oss else None,
            "frontier_cost_per_query": best_frontier["cost_per_query"] if best_frontier else None,
            "reduction_factor": round((best_frontier["cost_per_query"] or 1) / (best_oss["cost_per_query"] or 1), 1) if best_oss and best_frontier and best_oss["cost_per_query"] else None,
        },
        "sft_vs_base": {
            "sft_model": best_sft["display_name"] if best_sft else None,
            "sft_metric": best_sft["metric_value"] if best_sft else None,
        },
    }

    return headline_stats, tasks_won, round(total_cost, 2)


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
    efficiency_data: dict[str, list] = {}

    for task_id in ALL_TASKS:
        for model_id, conditions in MODEL_CONDITIONS.items():
            for condition in conditions:
                training_meta = None
                if condition.startswith("lora-"):
                    training_meta = load_training_meta(model_id, task_id, condition)
                elif condition == "api-sft-500":
                    training_meta = load_sft_training_meta(task_id)
                summary = load_summary(model_id, task_id, condition)
                result = build_result(model_id, task_id, condition, summary, training_meta, pricing, daily_volume)
                results.append(result)

        try:
            efficiency_data[task_id] = build_efficiency_points(task_id)
        except Exception:
            efficiency_data[task_id] = []

    headline_stats, tasks_won_by_oss, total_cost = compute_stats(results)

    return {
        "generated_at": None,  # filled at write time
        "daily_volume_assumption": daily_volume,
        "total_cost": total_cost,
        "tasks_won_by_oss": tasks_won_by_oss,
        "headline_stats": headline_stats,
        "results": results,
        "efficiency_data": efficiency_data,
    }


@click.command()
@click.option("--daily-volume", default=10000, help="Daily query volume for TCO calc")
@click.option("--out", default=None, help="Output path (default: data/benchmark/results.json in site)")
@click.option("--also-benchmark-repo", is_flag=True, help="Also write to dashboard-data/results.json in benchmark repo")
@click.option("--merge", is_flag=True, help="Preserve existing results where new run has no data (matched by model+task+condition)")
@click.option("--dry-run", is_flag=True)
def main(daily_volume: int, out: Optional[str], also_benchmark_repo: bool, merge: bool, dry_run: bool) -> None:
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
        headline_stats, tasks_won_by_oss, total_cost = compute_stats(merged)
        data["results"] = merged
        data["headline_stats"] = headline_stats
        data["tasks_won_by_oss"] = tasks_won_by_oss
        data["total_cost"] = total_cost

    data["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    if dry_run:
        n_results = len(data["results"])
        n_with_data = sum(1 for r in data["results"] if r["metric_value"] is not None)
        click.echo(f"  [dry-run] Would write {n_results} results ({n_with_data} with data) to {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    click.echo(f"  Written to {out_path}")

    if also_benchmark_repo:
        repo_out = REPO_ROOT / "dashboard-data" / "results.json"
        repo_out.parent.mkdir(parents=True, exist_ok=True)
        with open(repo_out, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"  Also written to {repo_out}")


if __name__ == "__main__":
    main()
