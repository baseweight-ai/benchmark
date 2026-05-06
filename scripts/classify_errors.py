"""Classify prediction errors and compute primary metrics per task."""
from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import click
import yaml
from pydantic import BaseModel

from checkpoint_utils import atomic_write_json
from utils import load_jsonl, write_jsonl as _write_jsonl
from pipeline.config import get_tasks
from pipeline.paths import classified_path, pred_path, summary_path

REPO_ROOT = Path(__file__).parent.parent
ALL_TASKS: list[str] = get_tasks()


@lru_cache(maxsize=1)
def _get_axis_definitions() -> dict[str, dict]:
    path = REPO_ROOT / "configs" / "eval_axes.yaml"
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def compute_axis_scores(
    summary: dict,
    eval_axes: list[str],
    axis_defs: dict[str, dict],
) -> dict[str, dict]:
    """Compute per-axis scores from summary data for all summary-source axes.

    Cost axis (source: dashboard) is intentionally excluded — it requires token
    pricing not available at classify time.  generate_dashboard_data.py adds it.
    """
    scores: dict[str, dict] = {}
    n = summary.get("n_predictions", 0)

    for axis in eval_axes:
        defn = axis_defs.get(axis)
        if defn is None or defn.get("source") != "summary":
            continue

        higher = defn.get("higher_is_better", True)

        if axis in ("accuracy", "token_f1"):
            val = summary.get("metric_value")

        elif axis == "instruction_following":
            if n:
                counts = summary.get("error_counts", {})
                non_compliant = (counts.get("empty", 0) + counts.get("refusal", 0)
                                 + counts.get("format_violation", 0))
                val = round(max(0.0, 1.0 - non_compliant / n), 4)
            else:
                val = None

        elif axis == "latency":
            val = summary.get("avg_latency_ms")

        else:
            val = None

        scores[axis] = {"value": val, "higher_is_better": higher}

    return scores


class TaskConfig(BaseModel):
    task_id: str
    task_type: str
    metric_id: str
    eval_axes: list[str] = []


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})



# ── Metric helpers ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for loose matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 (whitespace tokenization) for extraction tasks."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = defaultdict(int)
    gold_set = defaultdict(int)
    for t in pred_tokens:
        pred_set[t] += 1
    for t in gold_tokens:
        gold_set[t] += 1
    common = sum(min(pred_set[t], gold_set[t]) for t in pred_set if t in gold_set)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_empty(text: str) -> bool:
    return not text.strip()


def is_refusal(text: str) -> bool:
    lower = text.lower()
    refusal_phrases = [
        "i cannot", "i can't", "i'm not able", "i am not able",
        "i'm unable", "i am unable", "i won't", "i will not",
        "i refuse", "as an ai", "i don't feel comfortable",
        "i'm sorry, but i can", "sorry, i can",
    ]
    return any(p in lower for p in refusal_phrases)


def is_format_violation(text: str, valid_labels: Optional[list[str]]) -> bool:
    """Check if output is not in the valid label set."""
    if valid_labels is None:
        return False
    norm = normalize_text(text)
    for label in valid_labels:
        if normalize_text(label) == norm:
            return False
    return True


# ── Classification task error classification ───────────────────────────────

def classify_classification(prediction: str, ground_truth: str, valid_labels: Optional[list[str]] = None) -> str:
    """Priority: empty > refusal > format_violation > correct > wrong_class."""
    if is_empty(prediction):
        return "empty"
    if is_refusal(prediction):
        return "refusal"
    if valid_labels and is_format_violation(prediction, valid_labels):
        return "format_violation"
    if normalize_text(prediction) == normalize_text(ground_truth):
        return "correct"
    return "wrong_class"


# ── Extraction task error classification ──────────────────────────────────

def classify_extraction(prediction: str, ground_truth: str, f1_threshold_partial: float = 0.5) -> str:
    """Priority: empty > format_violation > correct > partial > hallucinated > not_applicable."""
    if is_empty(prediction):
        return "empty"

    # format_violation: longer than 5× the expected or contains structured tokens not in ground truth
    if len(prediction.split()) > max(5 * len(ground_truth.split()), 100) and ground_truth != "Not found.":
        return "format_violation"

    # not_applicable: ground truth is "Not found." and model also says not found
    gt_norm = normalize_text(ground_truth)
    pred_norm = normalize_text(prediction)
    not_found_phrases = ["not found", "no answer", "n/a", "none", "not applicable", "not mentioned"]
    gt_is_not_found = any(p in gt_norm for p in not_found_phrases) or gt_norm in ("not found.", "not found")
    pred_is_not_found = any(p in pred_norm for p in not_found_phrases)

    if gt_is_not_found and pred_is_not_found:
        return "not_applicable"

    # hallucinated: ground truth is "Not found." but model gives an answer
    if gt_is_not_found and not pred_is_not_found and len(prediction.strip()) > 5:
        return "hallucinated"

    f1 = token_f1(prediction, ground_truth)
    if f1 >= 0.9:
        return "correct"
    if f1 >= f1_threshold_partial:
        return "partial"
    return "hallucinated"



# ── Bootstrap CI and multi-seed aggregation ───────────────────────────────

def bootstrap_ci(
    values: list[float], n_boot: int = 5000, ci: float = 0.95
) -> tuple[float, float]:
    """Return (lower, upper) percentile bootstrap CI for the mean."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v
    rng = random.Random(0)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_boot)
    )
    alpha = (1 - ci) / 2
    return means[int(alpha * n_boot)], means[int((1 - alpha) * n_boot)]


def aggregate_seed_summaries(summaries: list[dict]) -> dict:
    """Aggregate summaries from multiple eval seeds into mean ± std + CI."""
    metric_values = [s["metric_value"] for s in summaries if s.get("metric_value") is not None]
    if not metric_values:
        return {}
    n = len(metric_values)
    mean = sum(metric_values) / n
    variance = sum((v - mean) ** 2 for v in metric_values) / n
    std = math.sqrt(variance)
    ci_lo, ci_hi = bootstrap_ci(metric_values)
    # Aggregate error counts by summing across seeds
    all_counts: dict[str, int] = defaultdict(int)
    for s in summaries:
        for k, v in s.get("error_counts", {}).items():
            all_counts[k] += v
    base = summaries[0]  # representative row for metadata

    eval_axes = base.get("eval_axes", [])
    agg_axis_scores: dict[str, dict] = {}
    axis_defs = _get_axis_definitions()
    for axis in eval_axes:
        vals = [
            s["axis_scores"][axis]["value"]
            for s in summaries
            if s.get("axis_scores", {}).get(axis, {}).get("value") is not None
        ]
        if vals:
            higher = axis_defs.get(axis, {}).get("higher_is_better", True)
            agg_axis_scores[axis] = {"value": round(sum(vals) / len(vals), 4), "higher_is_better": higher}

    return {
        "model": base.get("model"),
        "task_id": base.get("task_id"),
        "condition": base.get("condition"),
        "n_seeds": n,
        "seed_metric_values": metric_values,
        "metric_id": base.get("metric_id"),
        "metric_value": round(mean, 4),    # mean used as the primary value
        "metric_mean": round(mean, 4),
        "metric_std": round(std, 4),
        "metric_ci_lo": round(ci_lo, 4),
        "metric_ci_hi": round(ci_hi, 4),
        "n_predictions": sum(s.get("n_predictions", 0) for s in summaries),
        "error_counts": dict(all_counts),
        "prompt_sha": base.get("prompt_sha"),
        "few_shot_hash": base.get("few_shot_hash"),
        "eval_axes": eval_axes,
        "axis_scores": agg_axis_scores,
    }


# ── Primary metric computation ─────────────────────────────────────────────

def compute_metric(task_cfg: TaskConfig, classified_rows: list[dict]) -> Optional[float]:
    """Compute primary metric value from classified predictions."""
    metric = task_cfg.metric_id

    if metric in ("macro_f1", "weighted_f1"):
        from sklearn.metrics import f1_score
        average = "weighted" if metric == "weighted_f1" else "macro"
        y_true, y_pred = [], []
        for r in classified_rows:
            y_true.append(normalize_text(r["ground_truth"]))
            y_pred.append(r["predicted_clean"])
        try:
            score = f1_score(y_true, y_pred, average=average, zero_division=0)
        except Exception:
            score = None
        return score

    if metric == "accuracy":
        correct = sum(1 for r in classified_rows if r["error_category"] == "correct")
        return correct / len(classified_rows) if classified_rows else 0.0

    if metric == "token_f1":
        scores = [r.get("token_f1", 0.0) for r in classified_rows]
        return sum(scores) / len(scores) if scores else 0.0

    return None


# ── Main classification loop ───────────────────────────────────────────────

def classify_predictions(
    predictions: list[dict],
    task_cfg: TaskConfig,
    valid_labels: Optional[list[str]] = None,
) -> tuple[list[dict], dict]:
    """Classify all predictions and return (enriched_rows, summary_counts)."""
    classified = []
    counts: dict[str, int] = defaultdict(int)

    for row in predictions:
        pred = row.get("output", "")
        gt = row.get("ground_truth", "")
        enriched = dict(row)

        if task_cfg.task_type == "classification":
            cat = classify_classification(pred, gt, valid_labels)
            enriched["error_category"] = cat
            enriched["predicted_clean"] = normalize_text(pred) if cat not in ("empty", "refusal", "format_violation") else "__INVALID__"

        elif task_cfg.task_type == "extraction":
            cat = classify_extraction(pred, gt)
            enriched["error_category"] = cat
            enriched["token_f1"] = token_f1(pred, gt)

        else:
            cat = "unknown"
            enriched["error_category"] = cat

        counts[cat] += 1
        classified.append(enriched)

    return classified, dict(counts)


def process_model_task_condition(
    model_short: str,
    task_id: str,
    condition: str,
    task_cfg: TaskConfig,
    valid_labels: Optional[list[str]],
    dry_run: bool,
    source: str = "local",
) -> Optional[dict]:
    """Classify one predictions file and write summary.

    condition is the filename stem, which may include a _seedN suffix.
    """
    input_path = pred_path(REPO_ROOT, source, model_short, task_id, condition)
    # Logical condition name strips any _seedN suffix for storage in summary.
    base_condition = condition.split("_seed")[0] if "_seed" in condition else condition
    label = f"{source}/{model_short}/{task_id}/{condition}"
    if not input_path.exists():
        return None

    predictions = load_jsonl(input_path)
    if not predictions:
        click.echo(f"  SKIP [{label}]: empty predictions file")
        return None

    if dry_run:
        click.echo(f"  [dry-run] Would classify {len(predictions)} predictions for {label}")
        return {}

    classified, counts = classify_predictions(predictions, task_cfg, valid_labels)

    classified_out = classified_path(REPO_ROOT, source, model_short, task_id, condition)
    _write_jsonl(classified, classified_out)

    metric_value = compute_metric(task_cfg, classified)

    latencies = [r["latency_ms"] for r in predictions if r.get("latency_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in predictions if r.get("ttft_ms", 0) > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    ttft_p50 = sorted(ttfts)[len(ttfts) // 2] if ttfts else None
    ttft_p95 = sorted(ttfts)[int(len(ttfts) * 0.95)] if ttfts else None

    total_input_tokens = sum(r.get("input_tokens", 0) for r in predictions)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in predictions)

    # Wall time: prefer the sidecar written by eval_local.py (vLLM batch processing
    # collapses per-row timestamps to the same millisecond, making the span useless).
    # Fall back to timestamp-derived span for API predictions which lack a sidecar.
    wall_sidecar = input_path.with_suffix(".wall.json")
    eval_wall_time_s = None
    try:
        with open(wall_sidecar) as _wf:
            eval_wall_time_s = json.load(_wf).get("eval_wall_time_s")
    except FileNotFoundError:
        from datetime import datetime, timezone as tz
        timestamps = [r["timestamp"] for r in predictions if r.get("timestamp")]
        if len(timestamps) >= 2:
            def _parse(ts: str) -> datetime:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elapsed = (_parse(max(timestamps)) - _parse(min(timestamps))).total_seconds()
            rounded = round(elapsed, 1)
            eval_wall_time_s = rounded if rounded > 0 else None

    n = len(predictions)
    # Derived compliance/error rates
    format_violation_n = counts.get("format_violation", 0)
    refusal_n = counts.get("refusal", 0)
    empty_n = counts.get("empty", 0)
    partial_n = counts.get("partial", 0)
    format_compliance_rate = round(1 - format_violation_n / n, 4) if n else None
    refusal_rate = round(refusal_n / n, 4) if n else None
    empty_rate = round(empty_n / n, 4) if n else None
    partial_rate = round(partial_n / n, 4) if n else None

    # Per-class breakdown (classification only — used for reporting without rerunning)
    per_class_metrics: Optional[dict] = None
    if task_cfg.task_type == "classification":
        _per_class: dict = defaultdict(lambda: {"correct": 0, "total": 0})
        for row in classified:
            gt = str(row.get("ground_truth", ""))
            _per_class[gt]["total"] += 1
            if row.get("error_category") == "correct":
                _per_class[gt]["correct"] += 1
        per_class_metrics = {
            cls: {
                "correct": d["correct"],
                "total": d["total"],
                "accuracy": round(d["correct"] / d["total"], 4) if d["total"] else None,
            }
            for cls, d in sorted(_per_class.items())
        }

    # Propagate reproducibility fields from first prediction row
    first = predictions[0] if predictions else {}
    prompt_sha = first.get("prompt_sha")
    few_shot_hash = first.get("few_shot_hash")
    eval_seed = first.get("eval_seed", 0)

    summary = {
        "model": model_short,
        "task_id": task_id,
        "condition": base_condition,
        "eval_seed": eval_seed,
        "prompt_sha": prompt_sha,
        "few_shot_hash": few_shot_hash,
        "n_predictions": n,
        "metric_id": task_cfg.metric_id,
        "metric_value": round(metric_value, 4) if metric_value is not None else None,
        "error_counts": counts,
        "format_compliance_rate": format_compliance_rate,
        "refusal_rate": refusal_rate,
        "empty_rate": empty_rate,
        "partial_rate": partial_rate,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
        "ttft_p50_ms": round(ttft_p50, 1) if ttft_p50 is not None else None,
        "ttft_p95_ms": round(ttft_p95, 1) if ttft_p95 is not None else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "eval_wall_time_s": eval_wall_time_s,
        "per_class_metrics": per_class_metrics,
    }

    summary["eval_axes"] = task_cfg.eval_axes
    summary["axis_scores"] = compute_axis_scores(summary, task_cfg.eval_axes, _get_axis_definitions())

    # condition_key is the filename stem (may include _seedN suffix)
    summary_out = summary_path(REPO_ROOT, source, model_short, task_id, condition)
    atomic_write_json(summary, summary_out)
    metric_str = f"{metric_value:.4f}" if metric_value is not None else "N/A"
    click.echo(f"  [{label}] {task_cfg.metric_id}={metric_str} counts={counts}")
    return summary


def get_valid_labels(task_id: str) -> Optional[list[str]]:
    """Return valid output labels for classification tasks, or None (no format check)."""
    label_map: dict[str, Optional[list[str]]] = {
        "banking77":  None,  # 77 classes — too many to enumerate
        "cuad":       None,  # extraction — no fixed label set
        "ledgar":     None,  # many provision types — skip format check
        "fpb":        ["positive", "negative", "neutral"],
        "medmcqa":    ["A", "B", "C", "D"],
    }
    return label_map.get(task_id)


@click.command()
@click.option("--model", default="all", help="Model short name or 'all' (ignored for api source)")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="Condition or 'all'")
@click.option("--source", default="all", help="Prediction source: local|api|all")
@click.option("--dry-run", is_flag=True)
def main(model: str, task: str, condition: str, source: str, dry_run: bool) -> None:
    """Classify prediction errors and compute primary metrics."""
    pred_root = REPO_ROOT / "results" / "predictions"
    sources = ["local", "api"] if source == "all" else [source]

    task_ids = ALL_TASKS if task == "all" else [task]

    if condition == "all":
        conditions = ["zero-shot", "5-shot", "lora", "api-sft"]
    else:
        conditions = [condition]

    failures = []
    processed = 0

    for src in sources:
        src_root = pred_root / src

        if model == "all":
            model_shorts = sorted(d.name for d in src_root.iterdir() if d.is_dir()) if src_root.exists() else []
        else:
            model_shorts = [model]

        for ms in model_shorts:
            for tid in task_ids:
                try:
                    task_cfg = load_task_config(tid)
                    valid_labels = get_valid_labels(tid)
                except Exception as exc:
                    click.echo(f"  ERROR: could not load task config {tid}: {exc}", err=True)
                    failures.append((f"{src}/{ms}/{tid}", str(exc)))
                    continue

                # Collect base conditions plus any seed-specific files in the predictions dir
                pred_dir = pred_root / src / ms / tid
                seed_conds: list[str] = []
                if pred_dir.exists():
                    for f in sorted(pred_dir.glob("*_seed*.jsonl")):
                        stem = f.stem
                        if stem not in seed_conds:
                            seed_conds.append(stem)

                all_conds = list(conditions) + [c for c in seed_conds if c not in conditions]

                # Track summaries by base condition for aggregation
                seed_summaries: dict[str, list[dict]] = defaultdict(list)

                for cond in all_conds:
                    try:
                        result = process_model_task_condition(
                            ms, tid, cond, task_cfg, valid_labels, dry_run, source=src
                        )
                        if result is not None:
                            processed += 1
                            # Collect seed summaries for aggregation
                            base_cond = cond.split("_seed")[0] if "_seed" in cond else None
                            if base_cond:
                                seed_summaries[base_cond].append(result)
                    except Exception as exc:
                        click.echo(f"  ERROR [{src}/{ms}/{tid}/{cond}]: {exc}", err=True)
                        import traceback; traceback.print_exc()
                        failures.append((f"{src}/{ms}/{tid}/{cond}", str(exc)))

                # Aggregate multi-seed summaries per base condition
                for base_cond, seed_summs in seed_summaries.items():
                    if len(seed_summs) < 2:
                        continue
                    if dry_run:
                        continue
                    try:
                        agg = aggregate_seed_summaries(seed_summs)
                        agg["condition"] = base_cond
                        agg_out = summary_path(REPO_ROOT, src, ms, tid, f"{base_cond}_agg")
                        atomic_write_json(agg, agg_out)
                        click.echo(
                            f"  [{src}/{ms}/{tid}/{base_cond}] aggregated {len(seed_summs)} seeds → "
                            f"mean={agg['metric_mean']:.4f} ±{agg['metric_std']:.4f} "
                            f"[{agg['metric_ci_lo']:.4f}, {agg['metric_ci_hi']:.4f}]"
                        )
                    except Exception as exc:
                        click.echo(f"  WARN: seed aggregation failed for {src}/{ms}/{tid}/{base_cond}: {exc}", err=True)

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    if processed == 0:
        click.echo("ERROR: no prediction files found — nothing was classified.", err=True)
        sys.exit(1)
    click.echo(f"\nClassified {processed} prediction file(s).")


if __name__ == "__main__":
    main()
