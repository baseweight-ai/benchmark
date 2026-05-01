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

import click
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from checkpoint_utils import append_jsonl, finalize_partial, load_partial_ids, partial_path
from utils import build_messages, load_jsonl

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa"]

OPENAI_MODELS: dict[str, Optional[str]] = {
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.5":      "gpt-5.5",
}

# Models that support SFT — maps model_id to the versioned base model required by OpenAI
SFT_BASE_MODELS: dict[str, str] = {
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-5.4-mini": "gpt-5.4-mini-2026-03-17",
}

MODEL_CONDITIONS: dict[str, list[str]] = {
    "gpt-4.1-nano": ["zero-shot", "5-shot", "api-sft"],
    "gpt-5.4-mini": ["zero-shot", "api-sft"],
    "gpt-5.5":      ["5-shot"],
}

SMOKE_MODELS = ["gpt-4.1-nano"]
PROD_MODELS  = ["gpt-5.4-mini", "gpt-5.5"]
ALL_API_MODELS = SMOKE_MODELS + PROD_MODELS

_SFT_SUFFIX_MAX_LEN = 18  # OpenAI fine-tuning suffix character limit

MAX_CONCURRENCY = 5
MAX_RETRIES = 5


class TaskConfig(BaseModel):
    task_id: str
    max_output_tokens: int
    task_type: str


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})




async def call_openai(
    client, model_str: str, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore
) -> tuple[str, int, int, float, float]:
    """Stream one request. Returns (text, in_tokens, out_tokens, latency_ms, ttft_ms)."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                ttft_ms = 0.0
                first_token = True
                chunks: list[str] = []
                in_tok = out_tok = 0

                stream = await client.chat.completions.create(
                    model=model_str, messages=messages, temperature=0, max_tokens=max_tokens,
                    stream=True, stream_options={"include_usage": True},
                )
                async for chunk in stream:
                    if chunk.usage:
                        in_tok = chunk.usage.prompt_tokens
                        out_tok = chunk.usage.completion_tokens
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        if first_token:
                            ttft_ms = (time.time() - t0) * 1000
                            first_token = False
                        chunks.append(content)

                return "".join(chunks), in_tok, out_tok, (time.time() - t0) * 1000, ttft_ms

            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0, 0, 0, 0.0



async def run_eval(model_id: str, task_id: str, condition: str, task_cfg: TaskConfig, dry_run: bool, smoke_test: bool = False) -> None:
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    suffix = "smoke_" if smoke_test else ""
    test_path = prepared / f"{suffix}test.jsonl"
    few_shot_path = prepared / ("smoke_train.jsonl" if smoke_test else "train.jsonl")

    if not test_path.exists():
        raise FileNotFoundError(f"test data not found: {test_path}")

    test_rows = load_jsonl(test_path)
    labels_path = prepared / f"{suffix}test_labels.jsonl"
    if labels_path.exists():
        label_map = {r["id"]: r["label"] for r in load_jsonl(labels_path)}
        for row in test_rows:
            row["label"] = label_map.get(row["id"], "")
    few_shot = load_jsonl(few_shot_path)[:5] if few_shot_path.exists() else []

    if dry_run:
        click.echo(f"  [dry-run] Would eval {model_id} on {task_id}/{condition} ({len(test_rows)} examples)")
        return

    out_path = REPO_ROOT / "results" / "predictions" / "api" / model_id / task_id / f"{condition}.jsonl"
    if out_path.exists():
        click.echo(f"  SKIP [{model_id}/{task_id}/{condition}]: already exists")
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

    model_str = OPENAI_MODELS.get(model_id)
    if not model_str:
        raise ValueError(f"Model string not set for {model_id}")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process_row(row: dict) -> None:
        msgs = build_messages(row, few_shot, condition)
        ground_truth = row.get("label", "")
        try:
            text, in_tok, out_tok, lat, ttft = await call_openai(
                client, model_str, msgs, task_cfg.max_output_tokens, semaphore
            )
        except Exception as exc:
            text, in_tok, out_tok, lat, ttft = f"ERROR: {exc}", 0, 0, 0, 0.0
        result = {
            "id": row.get("id", ""),
            "model": model_id,
            "condition": condition,
            "input": msgs[-1]["content"] if msgs else "",
            "output": text,
            "ground_truth": ground_truth,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": lat,
            "ttft_ms": ttft,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        append_jsonl(result, pp)

    click.echo(f"  Evaluating {model_id}/{task_id}/{condition} ({len(pending_rows)}/{len(test_rows)} rows)...")
    from tqdm.asyncio import tqdm
    await tqdm.gather(*[process_row(r) for r in pending_rows], desc=f"{model_id}/{task_id}")

    finalize_partial(pp, out_path)
    click.echo(f"  Saved {len(test_rows)} predictions to {out_path.relative_to(REPO_ROOT)}")


def _find_existing_sft_job(client, model_id: str, task_id: str, smoke_test: bool):
    """Search OpenAI for a completed or in-progress fine-tuning job matching this run."""
    active = {"validating_files", "queued", "running"}
    after = None
    while True:
        page = client.fine_tuning.jobs.list(limit=100, after=after).data
        for job in page:
            if job.status not in active and job.status != "succeeded":
                continue
            meta = job.metadata or {}
            if (meta.get("model_id") == model_id
                    and meta.get("task_id") == task_id
                    and meta.get("smoke_test") == str(smoke_test)):
                return job
        if len(page) < 100:
            break
        after = page[-1].id
    return None


async def run_sft(model_id: str, task_id: str, task_cfg: TaskConfig, dry_run: bool, smoke_test: bool = False) -> None:
    """Upload training data, create fine-tuning job for the given model, then evaluate."""
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    sft_path = prepared / ("smoke_train.jsonl" if smoke_test else "train.jsonl")
    base_model = SFT_BASE_MODELS[model_id]

    if not sft_path.exists():
        raise FileNotFoundError(f"SFT training data not found: {sft_path}")

    if dry_run:
        click.echo(f"  [dry-run] Would upload {sft_path} and create {base_model} fine-tuning job")
        return

    import time as _time
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    meta_path = REPO_ROOT / "results" / "training" / "api" / model_id / task_id / "api-sft" / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        ft_model = meta.get("ft_model_id")
        click.echo(f"  Using cached fine-tuned model: {ft_model}")
    else:
        # Check OpenAI for an existing completed or in-progress job before creating a new one
        job = _find_existing_sft_job(client, model_id, task_id, smoke_test)
        if job and job.status == "succeeded":
            ft_model = job.fine_tuned_model
            click.echo(f"  Found completed job {job.id} on OpenAI → {ft_model}")
            _tt = round((job.finished_at - job.created_at) / 60, 1) if job.finished_at and job.created_at else None
            _n = sum(1 for line in open(sft_path) if line.strip())
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w") as f:
                json.dump({"ft_model_id": ft_model, "job_id": job.id, "trained_tokens": job.trained_tokens or 0,
                           "training_time_min": _tt, "n_train": _n}, f)
            OPENAI_MODELS[model_id] = ft_model
            await run_eval(model_id, task_id, "api-sft", task_cfg, dry_run, smoke_test)
            return
        elif job:
            click.echo(f"  Found in-progress job {job.id} (status={job.status}), attaching...")
        else:
            click.echo(f"  Uploading training file for {task_id}...")
            with open(sft_path, "rb") as f:
                train_file_obj = client.files.create(file=f, purpose="fine-tune")

            n_epochs = 1 if smoke_test else 3
            batch_size = 1 if smoke_test else 8
            job = client.fine_tuning.jobs.create(
                training_file=train_file_obj.id,
                model=base_model,
                suffix=(f"bw-{task_id}-sm" if smoke_test else f"bw-{task_id}")[:_SFT_SUFFIX_MAX_LEN],
                metadata={
                    "model_id": model_id,
                    "task_id": task_id,
                    "base_model": base_model,
                    "n_epochs": str(n_epochs),
                    "smoke_test": str(smoke_test),
                },
                hyperparameters={
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate_multiplier": "auto",
                },
            )
            click.echo(f"  Fine-tuning job created: {job.id} (epochs={n_epochs}). Waiting for completion...")

        seen_event_ids: set[str] = set()
        job_start = last_event_time = _time.time()
        while job.status not in ("succeeded", "failed", "cancelled"):
            _time.sleep(15)
            job = client.fine_tuning.jobs.retrieve(job.id)
            events = client.fine_tuning.jobs.list_events(job.id, limit=10)
            new_events = [e for e in reversed(events.data) if e.id not in seen_event_ids]
            for event in new_events:
                seen_event_ids.add(event.id)
                elapsed = int(_time.time() - job_start)
                click.echo(f"  [{job.status}] {event.message} (+{elapsed}s)")
                last_event_time = _time.time()
            if not new_events:
                since_last = int(_time.time() - last_event_time)
                total = int(_time.time() - job_start)
                click.echo(f"  [{job.status}] waiting... ({since_last}s since last event, {total}s total)")

        if job.status != "succeeded":
            raise RuntimeError(f"Fine-tuning job failed: {job.status}")

        ft_model = job.fine_tuned_model
        trained_tokens = job.trained_tokens or 0
        with open(REPO_ROOT / "configs" / "pricing.yaml") as f:
            pricing = yaml.safe_load(f)
        training_per_m = pricing.get("apis", {}).get(model_id, {}).get("training_per_m", 25.0)
        training_cost = trained_tokens * training_per_m / 1_000_000

        training_time_min = None
        if job.finished_at and job.created_at:
            training_time_min = round((job.finished_at - job.created_at) / 60, 1)

        n_train = sum(1 for line in open(sft_path) if line.strip())

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({
                "ft_model_id": ft_model, "job_id": job.id,
                "trained_tokens": trained_tokens, "training_cost": training_cost,
                "training_time_min": training_time_min, "n_train": n_train,
            }, f)
        click.echo(f"  Fine-tuned model: {ft_model}, cost: ${training_cost:.3f}")

    OPENAI_MODELS[model_id] = ft_model
    await run_eval(model_id, task_id, "api-sft", task_cfg, dry_run, smoke_test)


@click.command()
@click.option("--model", default="all", help=f"Model ID or 'all'. Choices: {', '.join(ALL_API_MODELS)}")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="zero-shot|5-shot|api-sft|all")
@click.option("--dry-run", is_flag=True)
@click.option("--smoke-test", is_flag=True)
def main(model: str, task: str, condition: str, dry_run: bool, smoke_test: bool) -> None:
    """Evaluate frontier OpenAI models on benchmark tasks."""
    default_models = SMOKE_MODELS if smoke_test else PROD_MODELS
    model_ids = default_models if model == "all" else [model]

    if not dry_run and any(m in OPENAI_MODELS for m in model_ids):
        if not os.environ.get("OPENAI_API_KEY"):
            click.echo("  WARNING: OPENAI_API_KEY not set", err=True)
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
                    if not conditions_to_run:
                        click.echo(f"  SKIP [{mid}/{tid}/{condition}]: not supported for {mid}")
                        continue
                    for cond in conditions_to_run:
                        if cond == "api-sft":
                            await run_sft(mid, tid, task_cfg, dry_run, smoke_test)
                        else:
                            await run_eval(mid, tid, cond, task_cfg, dry_run, smoke_test)
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}]: {exc}", err=True)
                    import traceback; traceback.print_exc()
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
