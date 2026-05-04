"""Evaluate fine-tuned local models via vLLM server."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from checkpoint_utils import append_jsonl, atomic_write_json, finalize_partial, load_partial_ids, partial_path
from utils import build_messages, load_jsonl

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa"]
ALL_MODELS = ["qwen3-8b", "gemma3-4b", "phi4-mini"]

MAX_CONCURRENCY = 4
MAX_RETRIES = 3
VLLM_HOST = "http://localhost:8000"
VLLM_HEALTH_TIMEOUT_GPU = 300
VLLM_HEALTH_TIMEOUT_SMOKE = 300
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
    vllm_task: str = "auto"
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


def load_test_rows(task_id: str, smoke_test: bool) -> list[dict]:
    """Load test prompts joined with their labels by id."""
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    suffix = "smoke_" if smoke_test else ""
    prompts = load_jsonl(prepared / f"{suffix}test.jsonl")
    labels_path = prepared / f"{suffix}test_labels.jsonl"
    if labels_path.exists():
        label_map = {r["id"]: r["label"] for r in load_jsonl(labels_path)}
        for row in prompts:
            row["label"] = label_map.get(row["id"], "")
    return prompts


def _pred_path(model_short: str, task_id: str, condition: str) -> Path:
    return (
        REPO_ROOT / "results" / "predictions" / "local"
        / model_short / task_id / f"{condition}.jsonl"
    )


def get_few_shot(task_id: str, model_short: str, smoke_test: bool) -> list[dict]:
    """Return first 5 rows of the appropriate train file."""
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    if smoke_test:
        train_path = prepared / "smoke_train.jsonl"
    else:
        train_path = prepared / "train.jsonl"
        meta_path = REPO_ROOT / "results" / "training" / "local" / model_short / task_id / "lora" / "metadata.json"
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



def _stream_vllm_output(proc: subprocess.Popen) -> None:
    for line in proc.stdout:
        click.echo(f"  [vllm] {line.decode(errors='replace').rstrip()}")


def _check_vllm_package() -> None:
    """Verify vllm is installed. Raises RuntimeError if setup.sh has not been run."""
    try:
        importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError("vllm is not installed. Run scripts/setup.sh.")


def start_vllm_server(
    base_model: str,
    adapter_path: Optional[Path],
    max_seq_length: int,
    enable_thinking: Optional[bool] = None,
    vllm_task: str = "auto",
) -> subprocess.Popen:
    # Kill any lingering vLLM processes (server + engine subprocess) to avoid
    # orphaned GPU contexts causing OOM on the next run
    killed = any(
        subprocess.run(["pkill", "-9", "-f", p], capture_output=True).returncode == 0
        for p in ["vllm.entrypoints.openai.api_server", "vllm.engine.multiprocessing"]
    )
    if killed:
        time.sleep(3)

    env = os.environ.copy()
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"

    _check_vllm_package()
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--task", vllm_task,
        "--dtype", "auto",
        "--max-model-len", str(max_seq_length),
        "--port", "8000",
        "--gpu-memory-utilization", "0.9",
        "--swap-space", "0",
        "--max-num-seqs", str(MAX_CONCURRENCY * 2),
    ]
    if enable_thinking is False:
        cmd += ["--default-chat-template-kwargs", '{"enable_thinking": false}']
    if adapter_path and adapter_path.exists():
        cmd += [
            "--enable-lora",
            "--lora-modules", f"adapter={adapter_path}",
        ]

    click.echo(f"  Starting vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env,
    )
    threading.Thread(target=_stream_vllm_output, args=(proc,), daemon=True).start()
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


def _vllm_health_timeout(smoke_test: bool) -> int:
    return VLLM_HEALTH_TIMEOUT_SMOKE if smoke_test else VLLM_HEALTH_TIMEOUT_GPU


async def wait_for_vllm(proc: subprocess.Popen, timeout: int = VLLM_HEALTH_TIMEOUT_GPU) -> bool:
    """Poll /health until ready, failing immediately if the process exits."""
    import aiohttp
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            if proc.poll() is not None:
                click.echo(f"  vLLM process exited (code {proc.returncode}).", err=True)
                return False
            try:
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
) -> tuple[str, int, int, float, float]:
    """Stream one request. Returns (text, in_tokens, out_tokens, latency_ms, ttft_ms)."""
    import aiohttp

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                ttft_ms = 0.0
                chunks: list[str] = []
                first_token = True
                in_tok = out_tok = 0

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
                        usage = data.get("usage") or {}
                        if usage.get("prompt_tokens"):
                            in_tok = usage["prompt_tokens"]
                        if usage.get("completion_tokens"):
                            out_tok = usage["completion_tokens"]
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
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
    return "", 0, 0, 0.0, 0.0



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
        raise FileNotFoundError(f"test data not found: {test_path}")

    test_rows = load_test_rows(task_id, smoke_test)

    out_path = (
        REPO_ROOT / "results" / "predictions" / "local"
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
            text, in_tok, out_tok, lat, ttft = await call_vllm(
                session, model_name, msgs, task_cfg.max_output_tokens, semaphore
            )
        except Exception as exc:
            text, in_tok, out_tok, lat, ttft = f"ERROR: {exc}", 0, 0, 0.0, 0.0
        result = {
            "id": row.get("id", ""),
            "model": model_cfg.model_short,
            "condition": condition,
            "input": msgs[-1]["content"] if msgs else "",
            "output": text,
            "ground_truth": row.get("label", ""),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": lat,
            "ttft_ms": ttft,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        append_jsonl(result, pp)

    click.echo(f"  Evaluating {model_cfg.model_short}/{task_id}/{condition} ({len(pending_rows)}/{len(test_rows)} rows)...")
    from tqdm.asyncio import tqdm
    t_wall_start = time.time()
    async with aiohttp.ClientSession() as session:
        await tqdm.gather(
            *[process_row(r, session) for r in pending_rows],
            desc=f"{model_cfg.model_short}/{task_id}",
        )
    eval_wall_time_s = round(time.time() - t_wall_start, 1) or None

    finalize_partial(pp, out_path)

    # Write wall time sidecar — vLLM batches cause all per-row timestamps to
    # collapse to the same millisecond, making timestamp-derived wall time useless.
    wall_path = out_path.with_suffix(".wall.json")
    atomic_write_json({"eval_wall_time_s": eval_wall_time_s}, wall_path)

    click.echo(f"  Saved {len(test_rows)} predictions to {out_path.relative_to(REPO_ROOT)}")


@click.command()
@click.option("--model", default=None, help="Model ID or 'all'. Defaults to 'qwen2.5-0.5b' with --smoke-test, 'qwen3-8b' otherwise.")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="zero-shot|5-shot|lora|all")
@click.option("--dry-run", is_flag=True, help="Validate without running inference")
@click.option("--smoke-test", is_flag=True, help="Use smoke test data and model; mirrors train_local.py --smoke-test")
def main(model: Optional[str], task: str, condition: str, dry_run: bool, smoke_test: bool) -> None:
    """Evaluate local fine-tuned models via vLLM server."""
    if model is None:
        model = "qwen2.5-0.5b" if smoke_test else "qwen3-8b"
    model_ids = ALL_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]

    failures = []
    health_timeout = _vllm_health_timeout(smoke_test)

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

            # model_cfg.max_seq_length is a training cap (e.g. 256 for smoke test).
            # For eval the vLLM server needs room for both the prompt and the
            # full output budget, so floor at max_output_tokens + 512.
            seq_len = max(
                task_cfg.max_seq_length or model_cfg.max_seq_length,
                task_cfg.max_output_tokens + 512,
            )

            # --- Base model (zero-shot, 5-shot) ---
            if base_conditions:
                if dry_run:
                    for cond in base_conditions:
                        asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, None, dry_run=True, smoke_test=smoke_test))
                else:
                    pending_base = [c for c in base_conditions if not _pred_path(model_cfg.model_short, tid, c).exists()]
                    if not pending_base:
                        click.echo(f"  SKIP [{mid}/{tid}]: all base conditions already complete")
                    else:
                        proc = start_vllm_server(
                            model_cfg.model_id, None, seq_len, model_cfg.enable_thinking, model_cfg.vllm_task
                        )
                        try:
                            ready = asyncio.run(wait_for_vllm(proc, timeout=health_timeout))
                            if not ready:
                                raise RuntimeError("vLLM server did not become ready in time")
                            for cond in pending_base:
                                try:
                                    asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, None, dry_run=False, smoke_test=smoke_test))
                                except Exception as exc:
                                    click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                                    failures.append((f"{mid}/{tid}/{cond}", str(exc)))
                        finally:
                            stop_vllm_server(proc)

            # --- LoRA adapter ---
            for cond in lora_conditions:
                adapter_path = (
                    REPO_ROOT / "results" / "adapters" / "local" / model_cfg.model_short / tid / cond
                )
                if not adapter_path.exists():
                    click.echo(f"  SKIP [{mid}/{tid}/{cond}]: adapter not found at {adapter_path}")
                    continue

                if not dry_run and _pred_path(model_cfg.model_short, tid, cond).exists():
                    click.echo(f"  SKIP [{mid}/{tid}/{cond}]: already complete")
                    continue

                if dry_run:
                    asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, adapter_path, dry_run=True, smoke_test=smoke_test))
                    continue

                proc = start_vllm_server(
                    model_cfg.model_id, adapter_path, seq_len, model_cfg.enable_thinking, model_cfg.vllm_task
                )
                health_timeout = _vllm_health_timeout(smoke_test)
                try:
                    ready = asyncio.run(wait_for_vllm(proc, timeout=health_timeout))
                    if not ready:
                        raise RuntimeError("vLLM server did not become ready in time")
                    asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, adapter_path, dry_run=False, smoke_test=smoke_test))
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
