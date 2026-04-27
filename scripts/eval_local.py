"""Evaluate fine-tuned local models via vLLM server."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
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
ALL_MODELS = ["qwen3-8b", "gemma3-4b", "phi4-mini"]

MAX_CONCURRENCY = 4
MAX_RETRIES = 3
VLLM_HOST = "http://localhost:8000"
VLLM_HEALTH_TIMEOUT = 300  # seconds to wait for vLLM to be ready
VLLM_HEALTH_INTERVAL = 5


class TaskConfig(BaseModel):
    task_id: str
    max_output_tokens: int
    task_type: str
    max_seq_length: Optional[int] = None


class ModelConfig(BaseModel):
    model_id: str
    model_short: str
    max_seq_length: int = 2048
    enable_thinking: Optional[bool] = None
    fallback_model_id: Optional[str] = None


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})


def load_model_config(model_id: str) -> ModelConfig:
    path = REPO_ROOT / "configs" / "training" / f"{model_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelConfig(**{k: v for k, v in data.items() if k in ModelConfig.model_fields})


def get_test_path(task_id: str, smoke_test: bool) -> Path:
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    return prepared / ("smoke_test.jsonl" if smoke_test else "test.jsonl")


def get_few_shot(task_id: str, model_short: str, smoke_test: bool) -> list[dict]:
    """Return first 5 rows of the appropriate train file."""
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    if smoke_test:
        train_path = prepared / "smoke_train.jsonl"
    else:
        train_path = prepared / "train.jsonl"
        meta_path = REPO_ROOT / "results" / "training" / model_short / task_id / "lora" / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            override = meta.get("train_data_path")
            if override and Path(override).exists():
                train_path = Path(override)
    if not train_path.exists():
        return []
    rows = []
    with open(train_path) as f:
        for line in f:
            if len(rows) >= 5:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows



def start_vllm_server(
    base_model: str,
    adapter_path: Optional[Path],
    max_seq_length: int,
    enable_thinking: Optional[bool] = None,
) -> subprocess.Popen:
    """Start vLLM server process."""
    cmd = [
        "vllm", "serve", base_model,
        "--dtype", "bfloat16",
        "--max-model-len", str(max_seq_length),
        "--gpu-memory-utilization", "0.9",
        "--port", "8000",
    ]
    # Pass enable_thinking=False for Qwen3 to suppress chain-of-thought tokens
    if enable_thinking is False:
        cmd += ["--chat-template-kwargs", '{"enable_thinking": false}']
    if adapter_path and adapter_path.exists():
        cmd += [
            "--enable-lora",
            "--lora-modules", f"adapter={adapter_path}",
        ]

    click.echo(f"  Starting vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    return proc


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """Gracefully stop vLLM server."""
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    click.echo("  vLLM server stopped.")


async def wait_for_vllm(timeout: int = VLLM_HEALTH_TIMEOUT) -> bool:
    """Poll /health endpoint until server is ready."""
    import aiohttp
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{VLLM_HOST}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        click.echo("  vLLM server is ready.")
                        return True
        except Exception:
            pass
        await asyncio.sleep(VLLM_HEALTH_INTERVAL)
    return False


async def call_vllm(
    session: "aiohttp.ClientSession",
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, float, float]:
    """Stream one request. Returns (text, latency_ms, ttft_ms)."""
    import aiohttp

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                ttft_ms = 0.0
                chunks: list[str] = []
                first_token = True

                async with session.post(
                    f"{VLLM_HOST}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            if first_token:
                                ttft_ms = (time.time() - t0) * 1000
                                first_token = False
                            chunks.append(content)

                return "".join(chunks), (time.time() - t0) * 1000, ttft_ms

            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0.0, 0.0


def run_eval_tiny_sync(
    model_cfg: ModelConfig,
    task_id: str,
    condition: str,
    task_cfg: TaskConfig,
    adapter_path: Optional[Path],
    smoke_test: bool = False,
) -> None:
    """CPU inference via transformers pipeline — no vLLM or GPU required."""
    import time
    import torch
    from datetime import datetime, timezone
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline
    from peft import PeftModel

    test_path = get_test_path(task_id, smoke_test)
    few_shot = get_few_shot(task_id, model_cfg.model_short, smoke_test)

    if not test_path.exists():
        click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{condition}]: {test_path.name} not found")
        return

    test_rows = load_jsonl(test_path)

    out_path = (
        REPO_ROOT / "results" / "predictions"
        / model_cfg.model_short / task_id / f"{condition}.jsonl"
    )

    if out_path.exists():
        click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{condition}]: already exists")
        return

    pp = partial_path(out_path)
    completed_ids = load_partial_ids(pp)
    pending_rows = [r for r in test_rows if r.get("id", "") not in completed_ids]

    if not pending_rows:
        finalize_partial(pp, out_path)
        return

    click.echo(f"  [tiny] Loading {model_cfg.model_id} on CPU...")
    base = AutoModelForCausalLM.from_pretrained(model_cfg.model_id, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(model_cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if adapter_path and adapter_path.exists():
        click.echo(f"  [tiny] Applying LoRA adapter from {adapter_path.name}/")
        base = PeftModel.from_pretrained(base, str(adapter_path))

    pipe = hf_pipeline(
        "text-generation",
        model=base,
        tokenizer=tok,
        max_new_tokens=task_cfg.max_output_tokens,
        do_sample=False,
        temperature=None,
        pad_token_id=tok.eos_token_id,
    )

    click.echo(f"  [tiny] Evaluating {len(pending_rows)} rows on CPU (no vLLM)...")
    for row in pending_rows:
        msgs = build_messages(row, few_shot, condition)
        t0 = time.time()
        result = pipe(msgs)
        lat = (time.time() - t0) * 1000
        generated = result[0]["generated_text"]
        # Chat pipeline returns full conversation; last entry is the assistant turn
        output = generated[-1]["content"] if isinstance(generated, list) else str(generated)
        rec = {
            "id": row.get("id", ""),
            "model": model_cfg.model_short,
            "condition": condition,
            "input": msgs[-1]["content"] if msgs else "",
            "output": output,
            "ground_truth": row.get("label", ""),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": round(lat, 1),
            "ttft_ms": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        append_jsonl(rec, pp)

    finalize_partial(pp, out_path)
    click.echo(f"  Saved {len(test_rows)} predictions to {out_path.relative_to(REPO_ROOT)}")


async def run_eval(
    model_cfg: ModelConfig,
    task_id: str,
    condition: str,
    task_cfg: TaskConfig,
    adapter_path: Optional[Path],
    dry_run: bool,
    smoke_test: bool = False,
) -> None:
    import aiohttp

    test_path = get_test_path(task_id, smoke_test)
    few_shot = get_few_shot(task_id, model_cfg.model_short, smoke_test)

    if not test_path.exists():
        click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{condition}]: {test_path.name} not found")
        return

    test_rows = load_jsonl(test_path)

    out_path = (
        REPO_ROOT / "results" / "predictions"
        / model_cfg.model_short / task_id / f"{condition}.jsonl"
    )

    if dry_run:
        click.echo(f"  [dry-run] Would eval {model_cfg.model_short} on {task_id}/{condition} ({len(test_rows)} examples)")
        return

    if out_path.exists():
        click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{condition}]: already exists")
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

    model_name = "adapter" if adapter_path and adapter_path.exists() else model_cfg.model_id
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process_row(row: dict, session: "aiohttp.ClientSession") -> None:
        msgs = build_messages(row, few_shot, condition)
        try:
            text, lat, ttft = await call_vllm(
                session, model_name, msgs, task_cfg.max_output_tokens, semaphore
            )
        except Exception as exc:
            text, lat, ttft = f"ERROR: {exc}", 0.0, 0.0
        result = {
            "id": row.get("id", ""),
            "model": model_cfg.model_short,
            "condition": condition,
            "input": msgs[-1]["content"] if msgs else "",
            "output": text,
            "ground_truth": row.get("label", ""),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": lat,
            "ttft_ms": ttft,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        append_jsonl(result, pp)

    click.echo(f"  Evaluating {model_cfg.model_short}/{task_id}/{condition} ({len(pending_rows)}/{len(test_rows)} rows)...")
    from tqdm.asyncio import tqdm
    async with aiohttp.ClientSession() as session:
        await tqdm.gather(
            *[process_row(r, session) for r in pending_rows],
            desc=f"{model_cfg.model_short}/{task_id}",
        )

    finalize_partial(pp, out_path)
    click.echo(f"  Saved {len(test_rows)} predictions to {out_path.relative_to(REPO_ROOT)}")


@click.command()
@click.option("--model", default="qwen3-8b", help="Model ID or 'all'")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="zero-shot|5-shot|lora|all")
@click.option("--dry-run", is_flag=True, help="Validate without running inference")
@click.option("--smoke-test", is_flag=True, help="CPU test mode: transformers pipeline, smoke test data, no vLLM or GPU")
def main(model: str, task: str, condition: str, dry_run: bool, smoke_test: bool) -> None:
    """Evaluate local fine-tuned models via vLLM server."""
    if smoke_test and model in ALL_MODELS + ["all"]:
        model = "tiny"  # use CPU-safe config automatically
    model_ids = ALL_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]

    failures = []

    for mid in model_ids:
        model_cfg = load_model_config(mid)

        conditions = ["zero-shot", "5-shot", "lora"] if condition == "all" else [condition]

        base_conditions = [c for c in conditions if c in ("zero-shot", "5-shot")]
        lora_conditions = [c for c in conditions if c == "lora"]

        for tid in task_ids:
            try:
                task_cfg = load_task_config(tid)
            except Exception as exc:
                click.echo(f"  ERROR: could not load task config for {tid}: {exc}", err=True)
                failures.append((f"{mid}/{tid}", str(exc)))
                continue

            seq_len = task_cfg.max_seq_length or model_cfg.max_seq_length

            if smoke_test:
                # CPU path: direct transformers inference, no vLLM server.
                for cond in conditions:
                    if dry_run:
                        asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, None, dry_run=True, smoke_test=True))
                        continue
                    adapter_path = None
                    if cond == "lora":
                        adapter_path = REPO_ROOT / "results" / "adapters" / model_cfg.model_short / tid / cond
                        if not adapter_path.exists():
                            click.echo(f"  SKIP [{mid}/{tid}/{cond}]: adapter not found at {adapter_path}")
                            continue
                    try:
                        run_eval_tiny_sync(model_cfg, tid, cond, task_cfg, adapter_path, smoke_test=True)
                    except Exception as exc:
                        click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                        failures.append((f"{mid}/{tid}/{cond}", str(exc)))
                continue  # skip the vLLM block below

            # --- Base model (zero-shot, 5-shot) ---
            if base_conditions:
                if dry_run:
                    for cond in base_conditions:
                        asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, None, dry_run=True))
                else:
                    proc = start_vllm_server(
                        model_cfg.model_id, None, seq_len, model_cfg.enable_thinking
                    )
                    try:
                        ready = asyncio.run(wait_for_vllm())
                        if not ready:
                            raise RuntimeError("vLLM server did not become ready in time")
                        for cond in base_conditions:
                            try:
                                asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, None, dry_run=False))
                            except Exception as exc:
                                click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                                failures.append((f"{mid}/{tid}/{cond}", str(exc)))
                    finally:
                        stop_vllm_server(proc)

            # --- LoRA adapter ---
            for cond in lora_conditions:
                adapter_path = (
                    REPO_ROOT / "results" / "adapters" / model_cfg.model_short / tid / cond
                )
                if not adapter_path.exists():
                    click.echo(f"  SKIP [{mid}/{tid}/{cond}]: adapter not found at {adapter_path}")
                    continue

                if dry_run:
                    asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, adapter_path, dry_run=True))
                    continue

                proc = start_vllm_server(
                    model_cfg.model_id, adapter_path, seq_len, model_cfg.enable_thinking
                )
                try:
                    ready = asyncio.run(wait_for_vllm())
                    if not ready:
                        raise RuntimeError("vLLM server did not become ready in time")
                    asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, adapter_path, dry_run=False))
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                    failures.append((f"{mid}/{tid}/{cond}", str(exc)))
                finally:
                    stop_vllm_server(proc)

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll local evaluations completed.")


if __name__ == "__main__":
    main()
