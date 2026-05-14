"""Evaluate frontier API models on benchmark tasks."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import traceback as _tb

import click
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from checkpoint_utils import append_jsonl, finalize_partial, load_partial_ids, partial_path
from utils import build_messages, load_jsonl, rows_hash as _rows_hash, read_prompt_sha as _read_prompt_sha, seed_sample as _seed_sample
from pipeline.config import get_model_conditions, get_openai_models, get_reasoning_capable, get_tasks
from pipeline.log import configure, get_logger
from pipeline.paths import pred_path, training_meta_path
from pipeline.providers import call_openai  # noqa: F401  # re-exported for test patching

_log = get_logger("eval-api")

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

ALL_TASKS: list[str] = get_tasks()
OPENAI_MODELS: dict[str, Optional[str]] = get_openai_models()
MODEL_CONDITIONS: dict[str, list[str]] = get_model_conditions()
REASONING_CAPABLE: dict[str, bool] = get_reasoning_capable()

# OpenAI's lowest-effort reasoning tier — effectively disables the loop.
REASONING_EFFORT_OFF = "minimal"

SMOKE_MODELS = ["gpt-4.1-nano"]
PROD_MODELS  = ["gpt-4.1-mini", "gpt-5.5"]
ALL_API_MODELS = SMOKE_MODELS + PROD_MODELS

MAX_CONCURRENCY = 5


class TaskConfig(BaseModel):
    task_id: str
    max_output_tokens: int
    task_type: str
    skip_conditions: list[str] = []


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})


async def _load_sft_model_id(model_id: str, task_id: str) -> Optional[str]:
    """Return the fine-tuned model ID from training metadata.

    If the job is still pending (submitted via --submit-only), polls OpenAI until
    it completes and updates metadata.json with the final ft_model_id.
    Returns None if not trained or if the job fails.
    """
    mp = training_meta_path(REPO_ROOT, "api", model_id, task_id, "api-sft")
    if not mp.exists():
        return None
    with open(mp) as f:
        meta = json.load(f)

    ft_model_id = meta.get("ft_model_id")
    if ft_model_id:
        return ft_model_id

    job_id = meta.get("job_id")
    if not job_id:
        return None

    from openai import OpenAI
    from checkpoint_utils import atomic_write_json
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    key = f"{model_id}/{task_id}/api-sft"
    click.echo(f"  [{key}] Training job {job_id} still pending, waiting for completion...")

    poll_start = last_log = time.time()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        if job.status == "succeeded":
            with open(REPO_ROOT / "configs" / "pricing.yaml") as f:
                pricing = yaml.safe_load(f)
            training_per_m = pricing.get("apis", {}).get(model_id, {}).get("training_per_m", 25.0)
            trained_tokens = job.trained_tokens or 0
            training_time_min = (
                round((job.finished_at - job.created_at) / 60, 1)
                if job.finished_at and job.created_at else None
            )
            updated = {
                "ft_model_id": job.fine_tuned_model,
                "job_id": job_id,
                "trained_tokens": trained_tokens,
                "training_cost": trained_tokens * training_per_m / 1_000_000,
                "training_time_min": training_time_min,
                "n_train": meta.get("n_train", 0),
            }
            atomic_write_json(updated, mp)
            click.echo(f"  [{key}] Training complete → {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status in ("failed", "cancelled"):
            click.echo(f"  [{key}] Training job ended with status: {job.status}", err=True)
            return None

        now = time.time()
        if now - last_log >= 60:
            click.echo(f"  [{key}/{job.status}] waiting... ({int(now - poll_start)}s elapsed)")
            last_log = now
        await asyncio.sleep(15)


async def run_eval(
    model_id: str,
    task_id: str,
    condition: str,
    task_cfg: TaskConfig,
    dry_run: bool,
    smoke_test: bool = False,
    eval_seed: int = 0,
) -> None:
    """Evaluate one model/task/condition combination."""
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    suffix = "smoke_" if smoke_test else ""
    base_test_path = prepared / f"{suffix}test.jsonl"
    full_test_path = prepared / "test_full.jsonl"
    few_shot_path = prepared / ("smoke_train.jsonl" if smoke_test else "train.jsonl")

    if not base_test_path.exists():
        raise FileNotFoundError(f"test data not found: {base_test_path}")

    # For seed > 0, resample from the full test set when it's available.
    if eval_seed > 0 and full_test_path.exists() and not smoke_test:
        meta_path = prepared / "dataset_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                n_test = json.load(f)["n_test"]
        else:
            n_test = len(load_jsonl(base_test_path))
        full_rows = load_jsonl(full_test_path)
        test_rows = _seed_sample(full_rows, n_test, eval_seed)
    else:
        test_rows = load_jsonl(base_test_path)

    labels_path = prepared / f"{suffix}test_labels.jsonl"
    full_labels_path = prepared / "test_full_labels.jsonl"
    if eval_seed > 0 and full_labels_path.exists() and not smoke_test:
        label_map = {r["id"]: r["label"] for r in load_jsonl(full_labels_path)}
    elif labels_path.exists():
        label_map = {r["id"]: r["label"] for r in load_jsonl(labels_path)}
    else:
        label_map = {}
    for row in test_rows:
        row["label"] = label_map.get(row["id"], "")

    # Prefer the curated few-shot pool emitted by prepare_datasets (covers
    # distinct labels) over the legacy "first 5 of train.jsonl" fallback.
    curated_few_shot = prepared / "few_shot.jsonl"
    if not smoke_test and curated_few_shot.exists():
        few_shot = load_jsonl(curated_few_shot)
    elif few_shot_path.exists():
        few_shot = load_jsonl(few_shot_path)[:5]
    else:
        few_shot = []
    prompt_sha = _read_prompt_sha(prepared)
    few_shot_hash = _rows_hash(few_shot) if few_shot else None

    # Resolve the exact model string to use — task-specific for api-sft, base otherwise.
    # Never mutate the module-level OPENAI_MODELS dict; keep model_str local so
    # different tasks in the same run each get the right ft model.
    if condition == "api-sft":
        model_str = await _load_sft_model_id(model_id, task_id)
        if not model_str:
            click.echo(f"  SKIP [{model_id}/{task_id}/api-sft]: no training metadata — run train_api.py first")
            return
    else:
        model_str = OPENAI_MODELS.get(model_id)
        if not model_str:
            raise ValueError(f"Model string not set for {model_id}")

    if dry_run:
        seed_label = f" seed={eval_seed}" if eval_seed > 0 else ""
        click.echo(f"  [dry-run] Would eval {model_id} on {task_id}/{condition}{seed_label} ({len(test_rows)} examples)")
        return

    cond_key = condition if eval_seed == 0 else f"{condition}_seed{eval_seed}"
    out_path = pred_path(REPO_ROOT, "api", model_id, task_id, cond_key)
    if out_path.exists():
        click.echo(f"  SKIP [{model_id}/{task_id}/{condition}]: already exists")
        _log.info("eval skip", model=model_id, task=task_id, condition=condition,
                  event="stage_skip", reason="already exists")
        return

    pp = partial_path(out_path)
    completed_ids = load_partial_ids(pp)
    pending_rows = [r for r in test_rows if r.get("id", "") not in completed_ids]

    if completed_ids:
        click.echo(f"  Resuming: {len(completed_ids)}/{len(test_rows)} rows already done")

    if not pending_rows:
        finalize_partial(pp, out_path)
        click.echo(f"  All {len(test_rows)} rows complete, finalized to {out_path.relative_to(REPO_ROOT)}")
        return
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    totals = [0, 0]  # [input_tokens, output_tokens]

    # Non-capable models reject the param entirely — only send when supported.
    reasoning_effort = REASONING_EFFORT_OFF if REASONING_CAPABLE.get(model_id, False) else None

    async def process_row(row: dict) -> None:
        msgs = build_messages(row, few_shot, condition)
        ground_truth = row.get("label", "")
        try:
            text, in_tok, out_tok, reasoning_tok, lat, ttft, avg_logprob = await call_openai(
                client, model_str, msgs, task_cfg.max_output_tokens, semaphore,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            text, in_tok, out_tok, reasoning_tok, lat, ttft, avg_logprob = (
                f"ERROR: {exc}", 0, 0, 0, 0, 0.0, None
            )
        totals[0] += in_tok
        totals[1] += out_tok
        result = {
            "id": row.get("id", ""),
            "model": model_id,
            "condition": condition,
            "eval_seed": eval_seed,
            "prompt_sha": prompt_sha,
            "few_shot_hash": few_shot_hash,
            "input": msgs[-1]["content"] if msgs else "",
            "output": text,
            "ground_truth": ground_truth,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "reasoning_tokens": reasoning_tok,
            "latency_ms": lat,
            "ttft_ms": ttft,
            "avg_logprob": avg_logprob,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        append_jsonl(result, pp)

    # No API warmup: TLS handshake elevates TTFT on only the first row out of
    # the full eval set, which is negligible signal vs. the cost of dummy
    # billed requests. Local vLLM warmup is different — first ~20 requests
    # pay JIT + CUDA graph capture, so it's warranted there.
    seed_label = f" seed={eval_seed}" if eval_seed > 0 else ""
    _log.info("evaluating", model=model_id, task=task_id, condition=condition, eval_seed=eval_seed,
              n_rows=len(pending_rows), n_total=len(test_rows))
    click.echo(f"  Evaluating {model_id}/{task_id}/{condition}{seed_label} ({len(pending_rows)}/{len(test_rows)} rows)...")
    from tqdm.asyncio import tqdm
    await tqdm.gather(*[process_row(r) for r in pending_rows], desc=f"{model_id}/{task_id}")

    finalize_partial(pp, out_path)
    click.echo(f"  Saved {len(test_rows)} predictions to {out_path.relative_to(REPO_ROOT)}")
    _log.info("eval complete", model=model_id, task=task_id, condition=condition,
              event="stage_complete", n_rows=len(test_rows),
              total_input_tokens=totals[0], total_output_tokens=totals[1])


@click.command()
@click.option("--model", default="all", help=f"Model ID or 'all'. Choices: {', '.join(ALL_API_MODELS)}")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="zero-shot|5-shot|api-sft|all")
@click.option("--eval-seed", "eval_seed", default=0, type=int,
              help="Evaluation seed (0 = deterministic sample; >0 resamples from test_full.jsonl)")
@click.option("--dry-run", is_flag=True)
@click.option("--smoke-test", is_flag=True)
def main(model: str, task: str, condition: str, eval_seed: int, dry_run: bool, smoke_test: bool) -> None:
    """Evaluate frontier OpenAI models on benchmark tasks.

    For api-sft conditions, training metadata must already exist (run train_api.py first).
    """
    configure(REPO_ROOT)
    default_models = SMOKE_MODELS if smoke_test else PROD_MODELS
    model_ids = default_models if model == "all" else [model]

    if not dry_run and any(m in OPENAI_MODELS for m in model_ids):
        if not os.environ.get("OPENAI_API_KEY"):
            click.echo("  WARNING: OPENAI_API_KEY not set", err=True)
            _log.warning("OPENAI_API_KEY not set")
    task_ids = ALL_TASKS if task == "all" else [task]
    failures = []

    async def run_all() -> None:
        for mid in model_ids:
            for tid in task_ids:
                try:
                    task_cfg = load_task_config(tid)
                    supported = MODEL_CONDITIONS.get(mid, [])
                    conditions_to_run = (
                        supported if condition == "all"
                        else [condition] if condition in supported
                        else []
                    )
                    conditions_to_run = [c for c in conditions_to_run if c not in task_cfg.skip_conditions]
                    if not conditions_to_run:
                        click.echo(f"  SKIP [{mid}/{tid}/{condition}]: not supported for {mid}")
                        continue
                    for cond in conditions_to_run:
                        await run_eval(mid, tid, cond, task_cfg, dry_run, smoke_test, eval_seed)
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}]: {exc}", err=True)
                    _tb.print_exc()
                    _log.error(f"eval failed: {type(exc).__name__}: {exc}",
                               model=mid, task=tid,
                               exc=str(exc), traceback=_tb.format_exc())
                    failures.append((f"{mid}/{tid}", str(exc)))

    asyncio.run(run_all())

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll API evaluations completed.")


if __name__ == "__main__":
    main()
