"""Classify prediction errors and compute primary metrics per task."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import click
import yaml
from pydantic import BaseModel, Field

from utils import load_jsonl, write_jsonl as _write_jsonl

REPO_ROOT = Path(__file__).parent.parent
ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa", "mbpp"]


class TaskConfig(BaseModel):
    task_id: str
    task_type: str
    metric_id: str


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


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


# ── Code task error classification ────────────────────────────────────────

def classify_code(prediction: str, ground_truth: str, mbpp_passed: Optional[bool] = None, mbpp_error: Optional[str] = "") -> str:
    """Priority: pass > syntax_error > runtime_error > logic_error > incomplete."""
    if mbpp_passed is True:
        return "pass"

    if is_empty(prediction):
        return "incomplete"

    # Try to detect syntax errors
    code = prediction.strip()
    if "```" in code:
        # strip fences
        if "```python" in code:
            start = code.find("```python") + len("```python")
            end = code.find("```", start)
            code = code[start:end].strip() if end != -1 else code[start:].strip()
        else:
            start = code.find("```") + 3
            end = code.find("```", start)
            code = code[start:end].strip() if end != -1 else code[start:].strip()

    try:
        import ast
        ast.parse(code)
        syntax_ok = True
    except SyntaxError:
        syntax_ok = False

    if not syntax_ok:
        return "syntax_error"

    err = (mbpp_error or "").lower()
    if err and any(kw in err for kw in ["syntaxerror", "indentation"]):
        return "syntax_error"
    if err and any(kw in err for kw in ["typeerror", "nameerror", "attributeerror", "valueerror",
                                         "zerodivisionerror", "indexerror", "keyerror", "timeout"]):
        return "runtime_error"

    # If tests ran but failed (mbpp_passed is False)
    if mbpp_passed is False:
        return "logic_error"

    # Code looks syntactically valid but incomplete (no function def)
    if "def " not in code:
        return "incomplete"

    return "logic_error"


# ── Primary metric computation ─────────────────────────────────────────────

def compute_metric(task_cfg: TaskConfig, classified_rows: list[dict]) -> Optional[float]:
    """Compute primary metric value from classified predictions."""
    metric = task_cfg.metric_id

    if metric in ("macro_f1", "weighted_f1"):
        from sklearn.metrics import f1_score
        average = "weighted" if metric == "weighted_f1" else "macro"
        y_true, y_pred = [], []
        for r in classified_rows:
            y_true.append(r["ground_truth"])
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

    if metric == "pass_at_1":
        passed = sum(1 for r in classified_rows if r["error_category"] == "pass")
        return passed / len(classified_rows) if classified_rows else 0.0

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

        elif task_cfg.task_type == "code":
            mbpp_passed = row.get("mbpp_passed")
            mbpp_error = row.get("mbpp_error", "")
            cat = classify_code(pred, gt, mbpp_passed, mbpp_error)
            enriched["error_category"] = cat

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
    """Classify one predictions file and write summary."""
    pred_path = REPO_ROOT / "results" / "predictions" / source / model_short / task_id / f"{condition}.jsonl"
    if not pred_path.exists():
        click.echo(f"  SKIP [{source}/{model_short}/{task_id}/{condition}]: predictions not found")
        return None

    predictions = load_jsonl(pred_path)
    if not predictions:
        click.echo(f"  SKIP [{model_short}/{task_id}/{condition}]: empty predictions file")
        return None

    if dry_run:
        click.echo(f"  [dry-run] Would classify {len(predictions)} predictions for {model_short}/{task_id}/{condition}")
        return None

    classified, counts = classify_predictions(predictions, task_cfg, valid_labels)

    # Write classified predictions
    classified_path = REPO_ROOT / "results" / "classified" / source / model_short / task_id / f"{condition}.jsonl"
    _write_jsonl(classified, classified_path)

    # Compute metric
    metric_value = compute_metric(task_cfg, classified)

    # Compute average latency and TTFT
    latencies = [r["latency_ms"] for r in predictions if r.get("latency_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in predictions if r.get("ttft_ms", 0) > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    ttft_p50 = sorted(ttfts)[len(ttfts) // 2] if ttfts else None
    ttft_p95 = sorted(ttfts)[int(len(ttfts) * 0.95)] if ttfts else None

    # Token counts for cost estimation
    total_input_tokens = sum(r.get("input_tokens", 0) for r in predictions)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in predictions)

    summary = {
        "model": model_short,
        "task_id": task_id,
        "condition": condition,
        "n_predictions": len(predictions),
        "metric_id": task_cfg.metric_id,
        "metric_value": round(metric_value, 4) if metric_value is not None else None,
        "error_counts": counts,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
        "ttft_p50_ms": round(ttft_p50, 1) if ttft_p50 is not None else None,
        "ttft_p95_ms": round(ttft_p95, 1) if ttft_p95 is not None else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

    summary_path = REPO_ROOT / "results" / "summaries" / source / model_short / task_id / f"{condition}.json"
    write_json(summary, summary_path)
    metric_str = f"{metric_value:.4f}" if metric_value is not None else "N/A"
    click.echo(
        f"  [{model_short}/{task_id}/{condition}] "
        f"{task_cfg.metric_id}={metric_str} counts={counts}"
    )
    return summary


def get_valid_labels(task_id: str) -> Optional[list[str]]:
    """Return valid output labels for classification tasks, or None (no format check)."""
    label_map: dict[str, Optional[list[str]]] = {
        "banking77":  None,  # 77 classes — too many to enumerate
        "cuad":       None,  # extraction — no fixed label set
        "ledgar":     None,  # many provision types — skip format check
        "fpb":        ["positive", "negative", "neutral"],
        "medmcqa":    ["A", "B", "C", "D"],
        "mbpp":       None,  # code generation — no fixed label set
    }
    return label_map.get(task_id)


@click.command()
@click.option("--model", default="all", help="Model short name or 'all'")
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
        conditions = ["zero-shot", "5-shot", "lora-500", "lora-full", "api-sft-500"]
    else:
        conditions = [condition]

    failures = []
    for src in sources:
        src_root = pred_root / src
        if model == "all":
            model_shorts = sorted(d.name for d in src_root.iterdir() if d.is_dir()) if src_root.exists() else []
        else:
            model_shorts = [model]

        for model_short in model_shorts:
            for tid in task_ids:
                try:
                    task_cfg = load_task_config(tid)
                    valid_labels = get_valid_labels(tid)
                except Exception as exc:
                    click.echo(f"  ERROR: could not load task config {tid}: {exc}", err=True)
                    failures.append((f"{src}/{model_short}/{tid}", str(exc)))
                    continue

                for cond in conditions:
                    try:
                        process_model_task_condition(
                            model_short, tid, cond, task_cfg, valid_labels, dry_run, source=src
                        )
                    except Exception as exc:
                        click.echo(f"  ERROR [{src}/{model_short}/{tid}/{cond}]: {exc}", err=True)
                        import traceback; traceback.print_exc()
                        failures.append((f"{src}/{model_short}/{tid}/{cond}", str(exc)))

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll error classification completed.")


if __name__ == "__main__":
    main()
