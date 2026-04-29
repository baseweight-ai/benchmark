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

ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa", "mbpp"]

OPENAI_MODELS: dict[str, Optional[str]] = {
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-sft":  None,          # set at runtime after fine-tuning job completes
}

# Anthropic models
ANTHROPIC_MODELS: dict[str, str] = {
    "claude-sonnet-4": "claude-sonnet-4-5",
}

# Google models
GOOGLE_MODELS: dict[str, str] = {
    "gemini-2.5-flash": "gemini-2.5-flash",
}

MODEL_CONDITIONS: dict[str, list[str]] = {
    "gpt-4.1-mini":     ["zero-shot", "5-shot"],
    "gpt-4.1-sft":      ["api-sft-500"],
    "claude-sonnet-4":  ["zero-shot", "5-shot"],
    "gemini-2.5-flash": ["zero-shot", "5-shot"],
}

ALL_API_MODELS = [
    "gpt-4.1-mini", "gpt-4.1-sft",
    "claude-sonnet-4", "gemini-2.5-flash",
]

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


def load_claude_auth_token() -> Optional[str]:
    """Read the Claude Code OAuth access token from ~/.claude/.credentials.json."""
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if not creds_path.exists():
        return None
    with open(creds_path) as f:
        data = json.load(f)
    return data.get("claudeAiOauth", {}).get("accessToken")




async def call_openai(
    client, model_str: str, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore
) -> tuple[str, int, int, float]:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = await client.chat.completions.create(
                    model=model_str, messages=messages, temperature=0, max_tokens=max_tokens,
                )
                latency_ms = int((time.time() - t0) * 1000)
                text = resp.choices[0].message.content or ""
                return text, resp.usage.prompt_tokens, resp.usage.completion_tokens, latency_ms
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0, 0, 0


async def call_anthropic(
    client, model_str: str, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore
) -> tuple[str, int, int, float]:
    """Call Anthropic API. Strips system message and passes it via the system param."""
    async with semaphore:
        system_content = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                chat_messages.append(m)

        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                kwargs: dict = dict(
                    model=model_str,
                    messages=chat_messages,
                    temperature=0,
                    max_tokens=max_tokens,
                )
                if system_content:
                    kwargs["system"] = system_content
                resp = await client.messages.create(**kwargs)
                latency_ms = int((time.time() - t0) * 1000)
                text = resp.content[0].text if resp.content else ""
                in_tok = resp.usage.input_tokens
                out_tok = resp.usage.output_tokens
                return text, in_tok, out_tok, latency_ms
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0, 0, 0


async def call_gemini(
    client, model_str: str, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore
) -> tuple[str, int, int, float]:
    """Call Google Gemini API via the OpenAI-compatible client."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = await client.chat.completions.create(
                    model=model_str, messages=messages, temperature=0, max_tokens=max_tokens,
                )
                latency_ms = int((time.time() - t0) * 1000)
                text = resp.choices[0].message.content or ""
                in_tok = resp.usage.prompt_tokens if resp.usage else 0
                out_tok = resp.usage.completion_tokens if resp.usage else 0
                return text, in_tok, out_tok, latency_ms
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0, 0, 0


async def run_eval(model_id: str, task_id: str, condition: str, task_cfg: TaskConfig, dry_run: bool, smoke_test: bool = False) -> None:
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    test_path = prepared / ("smoke_test.jsonl" if smoke_test else "test.jsonl")
    few_shot_path = prepared / ("smoke_train.jsonl" if smoke_test else "few_shot_5.jsonl")

    if not test_path.exists():
        raise FileNotFoundError(f"test data not found: {test_path}")

    test_rows = load_jsonl(test_path)
    few_shot = load_jsonl(few_shot_path) if few_shot_path.exists() else []

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

    if model_id in ANTHROPIC_MODELS:
        model_str = ANTHROPIC_MODELS[model_id]
        from anthropic import AsyncAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        auth_token = None if api_key else load_claude_auth_token()
        client = AsyncAnthropic(api_key=api_key, auth_token=auth_token)
        call_fn = call_anthropic
    elif model_id in GOOGLE_MODELS:
        model_str = GOOGLE_MODELS[model_id]
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        call_fn = call_gemini
    else:
        model_str = OPENAI_MODELS.get(model_id)
        if not model_str:
            raise ValueError(f"Model string not set for {model_id}")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        call_fn = call_openai

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process_row(row: dict) -> None:
        msgs = build_messages(row, few_shot, condition)
        ground_truth = row.get("label", "")
        try:
            text, in_tok, out_tok, lat = await call_fn(
                client, model_str, msgs, task_cfg.max_output_tokens, semaphore
            )
        except Exception as exc:
            text, in_tok, out_tok, lat = f"ERROR: {exc}", 0, 0, 0
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
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        append_jsonl(result, pp)

    click.echo(f"  Evaluating {model_id}/{task_id}/{condition} ({len(pending_rows)}/{len(test_rows)} rows)...")
    from tqdm.asyncio import tqdm
    await tqdm.gather(*[process_row(r) for r in pending_rows], desc=f"{model_id}/{task_id}")

    finalize_partial(pp, out_path)
    click.echo(f"  Saved {len(test_rows)} predictions to {out_path.relative_to(REPO_ROOT)}")


async def run_sft(task_id: str, task_cfg: TaskConfig, dry_run: bool, smoke_test: bool = False) -> None:
    """Upload training data, create GPT-4.1 fine-tuning job, then evaluate."""
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    sft_path = prepared / "openai_sft_500.jsonl"

    if not sft_path.exists():
        raise FileNotFoundError(f"SFT training data not found: {sft_path}")

    if dry_run:
        click.echo(f"  [dry-run] Would upload {sft_path} and create GPT-4.1 fine-tuning job")
        return

    import time as _time
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    meta_path = REPO_ROOT / "results" / "training" / "gpt-4.1-sft" / task_id / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        ft_model = meta.get("ft_model_id")
        click.echo(f"  Using cached fine-tuned model: {ft_model}")
    else:
        click.echo(f"  Uploading training file for {task_id}...")
        with open(sft_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="fine-tune")

        job = client.fine_tuning.jobs.create(training_file=file_obj.id, model="gpt-4.1")
        click.echo(f"  Fine-tuning job created: {job.id}. Waiting for completion...")

        while job.status not in ("succeeded", "failed", "cancelled"):
            _time.sleep(60)
            job = client.fine_tuning.jobs.retrieve(job.id)
            click.echo(f"  Status: {job.status}")

        if job.status != "succeeded":
            raise RuntimeError(f"Fine-tuning job failed: {job.status}")

        ft_model = job.fine_tuned_model
        trained_tokens = job.trained_tokens or 0
        with open(REPO_ROOT / "configs" / "pricing.yaml") as f:
            pricing = yaml.safe_load(f)
        training_per_m = pricing.get("apis", {}).get("gpt-4.1-sft", {}).get("training_per_m", 25.0)
        training_cost = trained_tokens * training_per_m / 1_000_000

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({
                "ft_model_id": ft_model, "job_id": job.id,
                "trained_tokens": trained_tokens, "training_cost": training_cost,
            }, f)
        click.echo(f"  Fine-tuned model: {ft_model}, cost: ${training_cost:.3f}")

    OPENAI_MODELS["gpt-4.1-sft"] = ft_model
    await run_eval("gpt-4.1-sft", task_id, "api-sft-500", task_cfg, dry_run, smoke_test)


@click.command()
@click.option("--model", default="all", help=f"Model ID or 'all'. Choices: {', '.join(ALL_API_MODELS)}")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="zero-shot|5-shot|api-sft-500|all")
@click.option("--dry-run", is_flag=True)
@click.option("--smoke-test", is_flag=True)
def main(model: str, task: str, condition: str, dry_run: bool, smoke_test: bool) -> None:
    """Evaluate frontier API models (OpenAI, Anthropic, Google) on benchmark tasks."""
    model_ids = ALL_API_MODELS if model == "all" else [model]

    if not dry_run:
        needs_openai = any(m in OPENAI_MODELS for m in model_ids)
        needs_anthropic = any(m in ANTHROPIC_MODELS for m in model_ids)
        needs_google = any(m in GOOGLE_MODELS for m in model_ids)
        if needs_openai and not os.environ.get("OPENAI_API_KEY"):
            click.echo("  WARNING: OPENAI_API_KEY not set", err=True)
        if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
            if load_claude_auth_token():
                click.echo("  INFO: ANTHROPIC_API_KEY not set, falling back to Claude Code auth token")
            else:
                click.echo("  WARNING: ANTHROPIC_API_KEY not set and no Claude Code credentials found", err=True)
        if needs_google and not os.environ.get("GOOGLE_API_KEY"):
            click.echo("  WARNING: GOOGLE_API_KEY not set", err=True)
    task_ids = ALL_TASKS if task == "all" else [task]
    failures = []

    async def run_all() -> None:
        for mid in model_ids:
            for tid in task_ids:
                try:
                    task_cfg = load_task_config(tid)
                    if mid == "gpt-4.1-sft":
                        await run_sft(tid, task_cfg, dry_run, smoke_test)
                    else:
                        supported = MODEL_CONDITIONS.get(mid, ["zero-shot", "5-shot"])
                        conditions_to_run = (
                            supported if condition == "all"
                            else [condition] if condition in supported
                            else []
                        )
                        if not conditions_to_run:
                            click.echo(f"  SKIP [{mid}/{tid}/{condition}]: not supported for {mid}")
                            continue
                        for cond in conditions_to_run:
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
