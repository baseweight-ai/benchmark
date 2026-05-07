"""Evaluate models via lm-evaluation-harness for tasks with lm_eval_task_id set.

Uses log-probability scoring (argmax over class continuations) rather than
autoregressive generation.  Results are written as predictions JSONL in the
same format as eval_local.py so classify_errors.py can process them normally:

    python scripts/classify_errors.py --source lm_eval

lm-eval evaluation differs from our generative eval in two key ways:
  1. Scoring: log P(label|context) argmax vs free-form generation
  2. Dataset scope: lm-eval uses the task's own full split; our custom eval
     uses a stratified sample capped by test_sampling.total_cap

lm-eval scores are directly comparable to published benchmarks and provide a
discriminative upper bound that isolates instruction-following overhead.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

REPO_ROOT = Path(__file__).parent.parent


# ── Task config helpers ────────────────────────────────────────────────────────

def _load_task_cfg(task_id: str) -> dict:
    with open(REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml") as f:
        return yaml.safe_load(f)


def _get_lm_task_map(task_ids: list[str]) -> dict[str, str]:
    """Return {lm_eval_task_id: our_task_id} for tasks with lm_eval_task_id set."""
    mapping: dict[str, str] = {}
    for tid in task_ids:
        cfg = _load_task_cfg(tid)
        lm_id = cfg.get("lm_eval_task_id")
        if lm_id:
            mapping[lm_id] = tid
        else:
            click.echo(f"  {tid}: lm_eval_task_id not configured — skipping", err=True)
    return mapping


def _verify_tasks(lm_task_map: dict[str, str]) -> dict[str, str]:
    """Return only tasks that can be loaded by the current lm-eval installation."""
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    verified: dict[str, str] = {}
    for lm_id, our_id in lm_task_map.items():
        try:
            tm._load_individual_task_or_group(lm_id)
            verified[lm_id] = our_id
        except Exception as exc:
            click.echo(
                f"  {our_id} ({lm_id}): cannot load — {exc!s:.120}\n"
                f"    Hint: some tasks require `pip install unitxt`",
                err=True,
            )
    return verified


# ── Prediction extraction ──────────────────────────────────────────────────────

def _extract_prediction(sample: dict) -> str:
    """Argmax of per-class log-probabilities → predicted class label string.

    For multiple_choice / loglikelihood tasks, filtered_resps is a list
    parallel to arguments — one (logprob, is_greedy) tuple per choice.
    The continuation of arguments[argmax] is the predicted label.
    """
    args = sample.get("arguments", [])
    resps = sample.get("filtered_resps", [])
    if not args or not resps:
        return ""
    logprobs: list[float] = []
    for r in resps:
        # r can be (lp, is_greedy) or [(lp, is_greedy)] depending on lm-eval version
        inner = r[0] if isinstance(r, (list, tuple)) and r else r
        lp = inner[0] if isinstance(inner, (list, tuple)) else inner
        logprobs.append(float(lp))
    pred_idx = max(range(len(logprobs)), key=lambda i: logprobs[i])
    return args[pred_idx][1].strip()


def _extract_ground_truth(sample: dict) -> str:
    """Gold class label from lm-eval's target field.

    target is either:
      - int: index into arguments[], continuation gives the label string
      - str: the label directly
    """
    target = sample.get("target")
    args = sample.get("arguments", [])
    if isinstance(target, int) and 0 <= target < len(args):
        return args[target][1].strip()
    return str(target).strip() if target is not None else ""


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--model-path", required=True,
              help="HuggingFace model ID or local path to the model")
@click.option("--model-short", required=True,
              help="Short name used in result paths (matches eval_local.py convention)")
@click.option("--tasks", "task_spec", default="all",
              help="Comma-separated task IDs or 'all'")
@click.option("--num-fewshot", default=0, type=int, show_default=True)
@click.option("--dtype", default="bfloat16", show_default=True)
@click.option("--gpu-memory-utilization", default=0.85, type=float, show_default=True)
@click.option("--max-samples", default=None, type=int,
              help="Cap examples per task — useful for smoke testing")
@click.option("--out-dir", type=Path, default=None,
              help="Override output root (default: results/predictions/lm_eval/)")
def main(
    model_path: str,
    model_short: str,
    task_spec: str,
    num_fewshot: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_samples: int | None,
    out_dir: Path | None,
) -> None:
    """Log-prob evaluation via lm-evaluation-harness.

    Writes per-sample predictions to results/predictions/lm_eval/{model_short}/
    in the same JSONL format as eval_local.py.  Run afterwards:

        python scripts/classify_errors.py --source lm_eval

    Only tasks with lm_eval_task_id set in their configs/tasks/*.yaml are
    evaluated.  Tasks whose lm-eval definition cannot be loaded are skipped
    with a warning (e.g. unitxt-backed tasks with a datasets version conflict).
    """
    try:
        import lm_eval
    except ImportError:
        raise SystemExit(
            "lm-eval not installed.  Run:  pip install lm-eval\n"
            "Then recreate the conda env or add it to environment.yml."
        )

    from pipeline.config import get_tasks

    all_task_ids = get_tasks()
    task_ids = all_task_ids if task_spec == "all" else [t.strip() for t in task_spec.split(",")]
    lm_task_map = _get_lm_task_map(task_ids)
    if not lm_task_map:
        click.echo("No tasks with lm_eval_task_id configured.  Nothing to evaluate.")
        return

    # Filter to tasks that can actually be loaded in the current environment
    loadable = _verify_tasks(lm_task_map)
    if not loadable:
        click.echo("No tasks could be loaded.  Exiting.", err=True)
        return

    condition = "zero-shot" if num_fewshot == 0 else f"{num_fewshot}-shot"
    click.echo(
        f"lm-eval  model={model_short}  condition={condition}"
        f"  tasks={list(loadable.keys())}"
    )

    model_args = ",".join([
        f"pretrained={model_path}",
        f"dtype={dtype}",
        f"gpu_memory_utilization={gpu_memory_utilization}",
    ])

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=list(loadable.keys()),
        num_fewshot=num_fewshot,
        log_samples=True,
        limit=max_samples,
    )

    samples_by_task: dict[str, list[dict]] = results.get("samples", {})
    timestamp = datetime.now(timezone.utc).isoformat()
    out_root = out_dir or (REPO_ROOT / "results" / "predictions" / "lm_eval")

    for lm_task_id, our_task_id in loadable.items():
        samples = samples_by_task.get(lm_task_id, [])
        if not samples:
            click.echo(f"  [{our_task_id}] no samples returned by lm-eval", err=True)
            continue

        pred_dir = out_root / model_short / our_task_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / f"{condition}.jsonl"

        predictions: list[dict] = []
        for s in samples:
            predictions.append({
                "id": f"{our_task_id}_lmeval_{s.get('doc_id', len(predictions)):04d}",
                "model": model_short,
                "condition": condition,
                "lm_eval_task": lm_task_id,
                "input": None,          # context omitted — can be very long
                "output": _extract_prediction(s),
                "ground_truth": _extract_ground_truth(s),
                "input_tokens": None,   # not available from log-prob scoring
                "output_tokens": None,
                "latency_ms": None,
                "ttft_ms": None,
                "timestamp": timestamp,
            })

        with pred_file.open("w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        n_correct = sum(
            1 for p in predictions
            if p["output"].lower() == p["ground_truth"].lower()
        )
        naive_acc = n_correct / len(predictions) if predictions else 0.0

        # lm-eval's own aggregate metrics for reference
        lm_metrics = results.get("results", {}).get(lm_task_id, {})
        native_str = "  ".join(
            f"{k}={v:.4f}" for k, v in sorted(lm_metrics.items())
            if isinstance(v, float) and "stderr" not in k
        )

        click.echo(
            f"  [{our_task_id}] n={len(predictions)}  naive_acc≈{naive_acc:.3f}"
            + (f"  lm-eval: {native_str}" if native_str else "")
            + f"  → {pred_file}"
        )


if __name__ == "__main__":
    main()
