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

import traceback as _tb

import click
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from checkpoint_utils import append_jsonl, atomic_write_json, finalize_partial, load_partial_ids, partial_path
from utils import build_messages, load_jsonl, load_label_set as _load_label_set, question_id, rows_hash as _rows_hash, read_prompt_sha as _read_prompt_sha, seed_sample_questions
from pipeline.config import get_local_models, get_tasks
from pipeline.hardware import check_allowed_gpu, get_current_gpu_name
from pipeline.log import configure, get_logger
from pipeline.cache import code_closure_hash, dict_hash, record_fingerprint, reuse_is_valid, tree_hash
from pipeline.paths import adapter_path, pred_path
from pipeline.providers import call_vllm  # noqa: F401  # re-exported for test patching
from pipeline.validation import reject_test_path, require_jsonl

_log = get_logger("eval-local")

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

ALL_TASKS: list[str] = get_tasks()
ALL_MODELS: list[str] = [m["id"] for m in get_local_models()]

MAX_CONCURRENCY = 4
# Discarded warmup requests — removes cold-start (JIT, CUDA graphs, KV cache).
WARMUP_REQUESTS = 20
VLLM_HOST = "http://localhost:8000"
VLLM_HEALTH_TIMEOUT_GPU = 1200
VLLM_HEALTH_TIMEOUT_SMOKE = 300
VLLM_HEALTH_INTERVAL = 5


class TaskConfig(BaseModel):
    task_id: str
    max_output_tokens: int
    task_type: str
    max_seq_length: Optional[int] = None
    # direct → the raw output IS the label: guided_choice constrains decoding.
    # tagged → output is a CoT around <answer>X</answer> (medmcqa): unconstrained.
    answer_mode: str = "direct"
    skip_conditions: list[str] = []


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


def load_label_set(task_id: str) -> Optional[list[str]]:
    """Return the closed answer set written by prepare_datasets, or None.

    Whether guided decoding actually applies is a separate decision — run_eval
    constrains only classification tasks with answer_mode == "direct" (medmcqa
    has a closed A/B/C/D answer set but emits a free-form CoT, so it is not
    constrained).
    """
    return _load_label_set(REPO_ROOT, task_id)


def load_test_rows(task_id: str, smoke_test: bool, eval_seed: int = 0) -> list[dict]:
    """Load test prompts joined with their labels by id.

    When eval_seed > 0 and test_full.jsonl exists, resamples the full set.
    """
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    suffix = "smoke_" if smoke_test else ""
    base_path = prepared / f"{suffix}test.jsonl"
    full_path = prepared / "test_full.jsonl"
    full_labels_path = prepared / "test_full_labels.jsonl"

    base_prompts = load_jsonl(base_path)

    if eval_seed > 0 and full_path.exists() and not smoke_test:
        full_prompts = load_jsonl(full_path)
        # Resample whole questions, not rows: chunked tasks (CUAD) have many rows
        # per question, and a question's windows must stay together. Degenerates
        # to row-level resampling for unchunked tasks (one row == one question).
        n_questions = len({question_id(r["id"]) for r in base_prompts})
        prompts = seed_sample_questions(full_prompts, n_questions, eval_seed)
        if full_labels_path.exists():
            label_map = {r["id"]: r["label"] for r in load_jsonl(full_labels_path)}
        else:
            label_map = {}
    else:
        prompts = base_prompts
        labels_path = prepared / f"{suffix}test_labels.jsonl"
        label_map = {r["id"]: r["label"] for r in load_jsonl(labels_path)} if labels_path.exists() else {}

    for row in prompts:
        row["label"] = label_map.get(row["id"], "")
    return prompts


def _pred_path(model_short: str, task_id: str, condition: str) -> Path:
    return pred_path(REPO_ROOT, "local", model_short, task_id, condition)


def get_few_shot(task_id: str, model_short: str, smoke_test: bool) -> list[dict]:
    """Return up to 5 few-shot examples.

    Prefers the curated few_shot.jsonl emitted by prepare_datasets (covers
    distinct classes deterministically) over the legacy "first 5 of train.jsonl"
    fallback. The curated file is reproducibly built at prepare time and
    versioned via prompt_sha + train_sha, so few_shot_hash changes track upstream
    data changes correctly.
    """
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    if not smoke_test:
        curated = prepared / "few_shot.jsonl"
        if curated.exists():
            return load_jsonl(curated)

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
    reject_test_path(train_path)
    rows = []
    with open(train_path) as f:
        for line in f:
            if len(rows) >= 5:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows



_HF_NOISE = ("429 Client Error", "rate limit", "Too Many Requests", "We had to rate limit")
_ALWAYS_SUPPRESS = ("ProcessGroupNCCL.cpp", "process group has NOT been destroyed", "destroy_process_group")


def _stream_vllm_output(proc: subprocess.Popen) -> None:
    in_atexit_tb = False
    for line in proc.stdout:
        text = line.decode(errors='replace').rstrip()
        if any(p in text for p in _HF_NOISE):
            continue
        if any(p in text for p in _ALWAYS_SUPPRESS):
            continue
        if "Exception ignored in" in text:
            in_atexit_tb = True
            continue
        if in_atexit_tb:
            if "KeyboardInterrupt: MQLLMEngine terminated" in text:
                in_atexit_tb = False
            continue
        click.echo(f"  [vllm] {text}")


def _check_vllm_package() -> None:
    """Verify vllm is installed. Raises RuntimeError if setup.sh has not been run."""
    try:
        importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError("vllm is not installed. Run scripts/setup.sh.")


def start_vllm_server(
    base_model: str,
    lora_modules: dict[str, Path],
    max_seq_length: int,
    vllm_task: str = "auto",
    smoke_test: bool = False,
    max_concurrent: int = 64,
) -> subprocess.Popen:
    """Start a vLLM server.

    lora_modules maps adapter name → adapter path. All adapters are loaded at
    startup so the server handles base-model and all LoRA requests without
    restarts. Pass an empty dict for base-model-only eval.
    """
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
    env["TRANSFORMERS_VERBOSITY"] = "error"

    dtype = "bfloat16"
    gpu_mem_util = "0.2" if smoke_test else "0.9"

    _check_vllm_package()
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--task", vllm_task,
        "--dtype", dtype,
        "--max-model-len", str(max_seq_length),
        "--port", "8000",
        "--gpu-memory-utilization", gpu_mem_util,
        "--swap-space", "0",
        # Sized so the highest batch-size profile (typically 32) can actually
        # batch. Lower client concurrency just under-utilises this ceiling.
        "--max-num-seqs", str(max_concurrent),
    ]
    if smoke_test:
        cmd += ["--enforce-eager"]
    if lora_modules:
        cmd += ["--enable-lora", "--max-loras", str(len(lora_modules))]
        for name, path in lora_modules.items():
            cmd += ["--lora-modules", f"{name}={path}"]

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


async def run_eval(
    model_cfg: ModelConfig,
    task_id: str,
    condition: str,
    task_cfg: TaskConfig,
    model_name: str,
    dry_run: bool,
    smoke_test: bool = False,
    eval_seed: int = 0,
    concurrency: int = MAX_CONCURRENCY,
) -> None:
    """Run eval for one (task, condition). model_name is passed directly to vLLM:
    use model_cfg.model_id for base conditions, 'adapter_{task_id}' for LoRA.

    concurrency: client-side asyncio semaphore limit. The default (4) preserves
    legacy behaviour. Other values (e.g. 1 for single-user latency, 32 for
    high-traffic throughput) get suffixed in the output path as `_bs{N}` so
    the regimes are stored side-by-side and don't overwrite each other.
    """
    import aiohttp

    test_path = get_test_path(task_id, smoke_test)
    few_shot = get_few_shot(task_id, model_cfg.model_short, smoke_test)
    prompt_sha = _read_prompt_sha(REPO_ROOT / "data" / "prepared" / task_id)
    few_shot_hash = _rows_hash(few_shot) if few_shot else None

    if not test_path.exists():
        raise FileNotFoundError(f"test data not found: {test_path}")
    require_jsonl(test_path, min_rows=1, check_chat_format=True, require_assistant_completion=False)

    test_rows = load_test_rows(task_id, smoke_test, eval_seed)

    # Suffix order: seed first, batch-size second (orthogonal).
    cond_key = condition if eval_seed == 0 else f"{condition}_seed{eval_seed}"
    if concurrency != MAX_CONCURRENCY:
        cond_key = f"{cond_key}_bs{concurrency}"
    out_path = (
        REPO_ROOT / "results" / "predictions" / "local"
        / model_cfg.model_short / task_id / f"{cond_key}.jsonl"
    )

    if dry_run:
        click.echo(f"  [dry-run] Would eval {model_cfg.model_short} on {task_id}/{condition} ({len(test_rows)} examples)")
        return

    # Skip-if-unchanged: reuse the prediction file only when its fingerprint
    # still matches. A stale file — changed test data, adapter, prompt, eval
    # code, or generation config — is discarded and regenerated.
    fingerprint = dict_hash({
        "code": code_closure_hash(Path(__file__)),
        "test_rows": _rows_hash(test_rows),
        "prompt_sha": prompt_sha,
        "few_shot_hash": few_shot_hash,
        "label_set": load_label_set(task_id),
        "condition": condition,
        "eval_seed": eval_seed,
        "concurrency": concurrency,
        "model": model_cfg.model_id,
        "adapter": (tree_hash(adapter_path(REPO_ROOT, model_cfg.model_short, task_id, "lora"))
                    if condition == "lora" else None),
        "max_output_tokens": task_cfg.max_output_tokens,
        "task_type": task_cfg.task_type,
        "answer_mode": task_cfg.answer_mode,
        "enable_thinking": model_cfg.enable_thinking,
    })
    pp = partial_path(out_path)
    if reuse_is_valid(out_path, pp, fingerprint):
        click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{condition}]: up-to-date")
        _log.info("eval skip", model=model_cfg.model_short, task=task_id, condition=condition,
                  event="stage_skip", reason="fingerprint match")
        return
    record_fingerprint(out_path, fingerprint)

    completed_ids = load_partial_ids(pp)
    pending_rows = [r for r in test_rows if r.get("id", "") not in completed_ids]

    if completed_ids:
        click.echo(f"  Resuming: {len(completed_ids)}/{len(test_rows)} rows already done")

    if not pending_rows:
        finalize_partial(pp, out_path)
        click.echo(f"  All {len(test_rows)} rows complete, finalized to {out_path.relative_to(REPO_ROOT)}")
        return

    semaphore = asyncio.Semaphore(concurrency)
    totals = [0, 0]  # [input_tokens, output_tokens]
    chat_template_kwargs = (
        {"enable_thinking": False} if model_cfg.enable_thinking is False else None
    )
    # Constrained decoding: classification tasks whose raw output IS the label
    # (answer_mode == "direct": banking77, fpb, ledgar) get vLLM's guided_choice,
    # which pins decoding to the labels.json set. medmcqa is a tagged CoT task
    # and cuad is free-form extraction — both stay unconstrained.
    guided_choice = (
        load_label_set(task_id)
        if task_cfg.task_type == "classification" and task_cfg.answer_mode == "direct"
        else None
    )
    if guided_choice:
        click.echo(f"  guided_choice: {len(guided_choice)} labels for {task_id}")

    async def process_row(row: dict, session: "aiohttp.ClientSession") -> None:
        msgs = build_messages(row, few_shot, condition)
        try:
            text, in_tok, out_tok, reasoning_tok, lat, ttft, avg_logprob = await call_vllm(
                session, model_name, msgs, task_cfg.max_output_tokens, semaphore,
                chat_template_kwargs=chat_template_kwargs,
                guided_choice=guided_choice,
            )
        except Exception as exc:
            text, in_tok, out_tok, reasoning_tok, lat, ttft, avg_logprob = (
                f"ERROR: {exc}", 0, 0, 0, 0.0, 0.0, None
            )
        totals[0] += in_tok
        totals[1] += out_tok
        result = {
            "id": row.get("id", ""),
            "model": model_cfg.model_short,
            "condition": condition,
            "eval_seed": eval_seed,
            "prompt_sha": prompt_sha,
            "few_shot_hash": few_shot_hash,
            "input": msgs[-1]["content"] if msgs else "",
            "output": text,
            "ground_truth": row.get("label", ""),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "reasoning_tokens": reasoning_tok,
            "latency_ms": lat,
            "ttft_ms": ttft,
            "avg_logprob": avg_logprob,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        append_jsonl(result, pp)

    async def _warmup(session: "aiohttp.ClientSession") -> None:
        """Send WARMUP_REQUESTS dummy requests and discard their latency.

        Uses the first test row as the warmup prompt — same model/adapter,
        same prompt structure, so the warmup exercises the actual hot path
        (LoRA load, CUDA graphs, KV cache). Errors are swallowed: a warmup
        failure shouldn't block the real eval.
        """
        if not pending_rows:
            return
        warm_msg = build_messages(pending_rows[0], few_shot, condition)
        n = min(WARMUP_REQUESTS, max(5, len(pending_rows)))
        click.echo(f"  Warming up: {n} requests through {model_name}...")

        async def _one() -> None:
            try:
                await call_vllm(
                    session, model_name, warm_msg, task_cfg.max_output_tokens, semaphore,
                    chat_template_kwargs=chat_template_kwargs,
                    guided_choice=guided_choice,
                )
            except Exception:
                pass

        await asyncio.gather(*[_one() for _ in range(n)])

    seed_label = f" seed={eval_seed}" if eval_seed > 0 else ""
    _log.info("evaluating", model=model_cfg.model_short, task=task_id, condition=condition, eval_seed=eval_seed,
              n_rows=len(pending_rows), n_total=len(test_rows))
    click.echo(f"  Evaluating {model_cfg.model_short}/{task_id}/{condition}{seed_label} ({len(pending_rows)}/{len(test_rows)} rows)...")
    from tqdm.asyncio import tqdm
    async with aiohttp.ClientSession() as session:
        await _warmup(session)
        # Wall-time clock starts AFTER warmup so cold-start cost doesn't
        # pollute throughput / cost-per-query downstream.
        t_wall_start = time.time()
        await tqdm.gather(
            *[process_row(r, session) for r in pending_rows],
            desc=f"{model_cfg.model_short}/{task_id}",
        )
    eval_wall_time_s = round(time.time() - t_wall_start, 1) or None

    finalize_partial(pp, out_path)

    # vLLM batches collapse per-row timestamps to the same millisecond, so
    # wall time can't be derived from the timestamp range — record it here.
    wall_path = out_path.with_suffix(".wall.json")
    atomic_write_json(
        {
            "eval_wall_time_s": eval_wall_time_s,
            "gpu_model": get_current_gpu_name(),
            "concurrency": concurrency,
        },
        wall_path,
    )

    total_toks = totals[0] + totals[1]
    tok_per_s = round(total_toks / eval_wall_time_s) if eval_wall_time_s and total_toks else None
    ms_per_row = round(eval_wall_time_s * 1000 / len(test_rows), 1) if eval_wall_time_s and test_rows else None
    stats_str = ""
    if ms_per_row is not None:
        stats_str += f"  ~{ms_per_row}ms/row"
    if tok_per_s is not None:
        stats_str += f"  {tok_per_s} tok/s"
    click.echo(f"  Saved {len(test_rows)} predictions to {out_path.relative_to(REPO_ROOT)}{stats_str}")
    _log.info("eval complete", model=model_cfg.model_short, task=task_id, condition=condition,
              event="stage_complete", n_rows=len(test_rows),
              total_input_tokens=totals[0], total_output_tokens=totals[1])


@click.command()
@click.option("--model", default=None, help="Model ID or 'all'. Defaults to 'qwen2.5-0.5b' with --smoke-test, 'qwen3-8b' otherwise.")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="zero-shot|5-shot|lora|all")
@click.option("--eval-seed", "eval_seed", default=0, type=int,
              help="Evaluation seed (0 = deterministic sample; >0 resamples from test_full.jsonl)")
@click.option("--serving-profile", "serving_profile", default=None,
              help="Comma-separated batch sizes to sweep (e.g. '1,32'). Each value runs a separate eval at that client-concurrency; outputs are written to {condition}_bs{N}.jsonl. Default: single run at MAX_CONCURRENCY.")
@click.option("--dry-run", is_flag=True, help="Validate without running inference")
@click.option("--smoke-test", is_flag=True, help="Use smoke test data and model; mirrors train_local.py --smoke-test")
def main(model: Optional[str], task: str, condition: str, eval_seed: int,
         serving_profile: Optional[str], dry_run: bool, smoke_test: bool) -> None:
    """Evaluate local fine-tuned models via vLLM server.

    Starts one vLLM server per model with all available LoRA adapters loaded
    upfront, eliminating per-task server restarts. Zero-shot, 5-shot, and all
    LoRA conditions are served from the same running instance.
    """
    configure(REPO_ROOT)
    check_allowed_gpu(skip=smoke_test or dry_run)
    if model is None:
        model = "qwen2.5-0.5b" if smoke_test else "qwen3-8b"
    model_ids = ALL_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]
    conditions = ["zero-shot", "5-shot", "lora"] if condition == "all" else [condition]

    if serving_profile:
        concurrencies = [int(n) for n in serving_profile.split(",") if n.strip()]
        if not concurrencies:
            raise click.UsageError(f"--serving-profile parsed to empty list from {serving_profile!r}")
    else:
        concurrencies = [MAX_CONCURRENCY]

    failures = []

    for mid in model_ids:
        model_cfg = load_model_config(mid)

        # Pre-load all task configs so we can compute seq_len and detect errors early.
        task_cfgs: dict[str, TaskConfig] = {}
        for tid in task_ids:
            try:
                task_cfgs[tid] = load_task_config(tid)
            except Exception as exc:
                click.echo(f"  ERROR: could not load task config for {tid}: {exc}", err=True)
                failures.append((f"{mid}/{tid}", str(exc)))

        if not task_cfgs:
            continue

        # Compute max seq_len across all tasks so one server covers every prompt.
        # model_cfg.max_seq_length is a training cap; floor at max_output_tokens+512
        # so there's always room for both prompt and full output budget.
        seq_len = max(
            max(tc.max_seq_length or model_cfg.max_seq_length, tc.max_output_tokens + 512)
            for tc in task_cfgs.values()
        )

        # Collect LoRA adapters that exist on disk.
        lora_modules: dict[str, Path] = {}
        if "lora" in conditions:
            for tid in task_cfgs:
                ap = adapter_path(REPO_ROOT, model_cfg.model_short, tid, "lora")
                if ap.exists():
                    lora_modules[f"adapter_{tid}"] = ap
                else:
                    click.echo(f"  SKIP [{mid}/{tid}/lora]: adapter not found at {ap}")

        def _cond_key_for(cond: str, conc: int) -> str:
            k = cond if eval_seed == 0 else f"{cond}_seed{eval_seed}"
            if conc != MAX_CONCURRENCY:
                k = f"{k}_bs{conc}"
            return k

        pending: list[tuple[str, str, TaskConfig, str, int]] = []
        for tid, task_cfg in task_cfgs.items():
            for cond in conditions:
                if cond in task_cfg.skip_conditions:
                    click.echo(f"  SKIP [{mid}/{tid}/{cond}]: condition excluded for this task")
                    continue
                if cond == "lora" and f"adapter_{tid}" not in lora_modules:
                    continue  # already reported above
                model_name = f"adapter_{tid}" if cond == "lora" else model_cfg.model_id
                for conc in concurrencies:
                    if not dry_run and _pred_path(model_cfg.model_short, tid, _cond_key_for(cond, conc)).exists():
                        click.echo(f"  SKIP [{mid}/{tid}/{_cond_key_for(cond, conc)}]: already complete")
                        continue
                    pending.append((tid, cond, task_cfg, model_name, conc))

        if dry_run:
            for tid, cond, task_cfg, model_name, conc in pending:
                try:
                    asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, model_name,
                                         dry_run=True, smoke_test=smoke_test,
                                         eval_seed=eval_seed, concurrency=conc))
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}/{cond}/bs{conc}]: {exc}", err=True)
                    _log.error(f"eval failed: {type(exc).__name__}: {exc}",
                               model=mid, task=tid, condition=cond,
                               exc=str(exc), traceback=_tb.format_exc())
                    failures.append((f"{mid}/{tid}/{cond}/bs{conc}", str(exc)))
            continue

        if not pending:
            click.echo(f"  SKIP [{mid}]: all conditions already complete")
            continue

        # Start one server for ALL tasks and conditions of this model.
        # Base-model requests use model_cfg.model_id; LoRA requests use adapter_{tid}.
        # max_concurrent sized to the largest profile so high-BS conditions
        # actually batch (a profile of [1, 32] needs --max-num-seqs ≥ 32).
        proc = start_vllm_server(
            model_cfg.model_id, lora_modules, seq_len,
            model_cfg.vllm_task, smoke_test,
            max_concurrent=max(max(concurrencies) * 2, 8),
        )
        try:
            ready = asyncio.run(wait_for_vllm(proc, timeout=_vllm_health_timeout(smoke_test)))
            if not ready:
                raise RuntimeError("vLLM server did not become ready in time")
            for tid, cond, task_cfg, model_name, conc in pending:
                try:
                    asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, model_name,
                                         dry_run=False, smoke_test=smoke_test,
                                         eval_seed=eval_seed, concurrency=conc))
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}/{cond}/bs{conc}]: {exc}", err=True)
                    _log.error(f"eval failed: {type(exc).__name__}: {exc}",
                               model=mid, task=tid, condition=cond,
                               exc=str(exc), traceback=_tb.format_exc())
                    failures.append((f"{mid}/{tid}/{cond}/bs{conc}", str(exc)))
        except Exception as exc:
            click.echo(f"  ERROR [{mid}]: {exc}", err=True)
            _log.error(f"vLLM server error: {type(exc).__name__}: {exc}",
                       model=mid, exc=str(exc), traceback=_tb.format_exc())
            failures.append((mid, str(exc)))
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
