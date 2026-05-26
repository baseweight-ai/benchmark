"""QLoRA fine-tuning orchestrator.

This module owns the cross-run concerns of a fine-tune — config loading, path
layout, hardware/hyperparameter resolution, input hashing, skip-if-unchanged
and resume decisions, metadata writing, auto-upload — and delegates the
framework-specific training loop to a `Trainer` adapter selected from
`pipeline.trainers`. The Unsloth + TRL fast path is the default backend
(`unsloth-lora`); a pure HF stub (`hf-lora`) and a future Axolotl backend
plug in through the same interface without touching the orchestrator.
"""
from __future__ import annotations

import faulthandler
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Any, Optional

faulthandler.enable()

import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_PADDING_FREE", "1")

from dotenv import load_dotenv
load_dotenv()

import click
import yaml
from pydantic import BaseModel, Field

from checkpoint_utils import (
    atomic_write_json,
    checkpoint_dir,
    find_hf_resume_checkpoint,
    load_train_state,
    save_train_state,
    training_log,
)
from utils import load_jsonl, write_jsonl
from pipeline.cache import code_closure_hash, training_inputs_hash
from pipeline.config import get_local_models, get_tasks
from pipeline.hardware import check_allowed_gpu
from pipeline.log import configure, get_logger
from pipeline.paths import adapter_path, prepared_path, training_meta_path
from pipeline.trainers import CheckpointCallback, TrainSpec, get_trainer
from pipeline.validation import reject_test_path, require_jsonl
from pipeline.versioning import git_sha as _git_sha

_log = get_logger("train-local")

REPO_ROOT = Path(__file__).parent.parent


def _echo(ctx: str, msg: str) -> None:
    prefix = f"[{ctx}] " if ctx else ""
    click.echo(f"  {prefix}{msg}")


ALL_TASKS: list[str] = get_tasks()
ALL_MODELS: list[str] = [m["id"] for m in get_local_models()]
GPU_HOURLY = 0.49  # Default GPU hourly rate — override via pricing.yaml
CONDITION = "lora"

# Default Trainer adapter. ModelConfig.trainer_id overrides this per-model so
# a config can swap backends (unsloth-lora / hf-lora / axolotl-lora / ...)
# without code changes — the pipeline never branches on backend name.
DEFAULT_TRAINER_ID = "unsloth-lora"


class ModelConfig(BaseModel):
    model_id: str
    model_short: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: str = "bfloat16"
    enable_thinking: Optional[bool] = None
    fallback_model_id: Optional[str] = None
    # Pinned base-model HF revision (commit SHA or tag). Forwarded to the loader so a
    # run binds to an exact base checkpoint, and recorded into metadata for provenance.
    revision: Optional[str] = None
    # Trainer backend identifier (matches a key in pipeline.trainers.TRAINER_REGISTRY).
    # Defaults to the Unsloth fast path; set this in the YAML config to switch
    # backends without touching code.
    trainer_id: str = DEFAULT_TRAINER_ID
    # Chat-template turn markers used to mask prompt tokens for completion-only
    # loss. Defaults are ChatML (Qwen, etc.); override per model when its
    # chat template uses different turn delimiters.
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"
    lora: dict = Field(default_factory=dict)
    training: dict = Field(default_factory=dict)


class TaskConfig(BaseModel):
    task_id: str
    max_seq_length: Optional[int] = None  # overrides model max_seq_length when set
    # Per-task overrides merged into model_cfg.training before SFTConfig is
    # built. Use for task-specific stability tweaks (e.g. lower LR on tiny
    # datasets where the default overfits).
    training_overrides: Optional[dict] = None
    # Per-task overrides merged into model_cfg.lora before the LoRA adapter is
    # attached. Use for task-specific capacity tweaks (e.g. lower rank on tiny
    # datasets prone to overfitting). Mirrors training_overrides semantics.
    lora_overrides: Optional[dict] = None


def load_model_config(model_id: str) -> ModelConfig:
    path = REPO_ROOT / "configs" / "training" / f"{model_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})


def get_epochs(n_examples: int) -> int:
    """Upper bound on epochs by dataset size — a cap, not the trained length.

    When a validation split exists, EarlyStoppingCallback usually ends the run
    well before this; it is the safety ceiling for the no-overfit case.
    """
    if n_examples <= 200:
        return 10
    if n_examples <= 1000:
        return 5
    return 3


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def get_or_create_cap(data_dir: Path, n: int, ctx: str = "") -> Path:
    """Return path to a fixed random-sample of n rows from train.jsonl, creating it if needed."""
    cap_path = data_dir / f"train_cap{n}.jsonl"
    if cap_path.exists():
        return cap_path
    src = data_dir / "train.jsonl"
    with open(src) as f:
        rows = [json.loads(line) for line in f]
    sample = random.Random(42).sample(rows, min(n, len(rows)))
    write_jsonl(sample, cap_path)
    _echo(ctx, f"Cap: wrote {len(sample)} rows to {cap_path.name}")
    return cap_path


# ---------------------------------------------------------------------------
# Hardware/hyperparameter configuration
# ---------------------------------------------------------------------------

@dataclass
class HardwareConfig:
    device: str
    load_dtype: Any           # torch.dtype | None
    load_in_4bit: bool
    use_grad_ckpt: bool | str
    lora_rank: int
    lora_alpha: int
    seq_len: int
    sft_extra: dict = field(default_factory=dict)


class ConfigFactory:
    """Centralises all hardware and hyperparameter decisions.

    Callers receive a HardwareConfig and never branch on smoke_test themselves.
    """

    @staticmethod
    def build(model_cfg: ModelConfig, task_cfg: TaskConfig, smoke_test: bool) -> HardwareConfig:
        base_seq = task_cfg.max_seq_length or model_cfg.max_seq_length
        cfg = HardwareConfig(
            device="cuda",
            load_dtype=torch.bfloat16,
            load_in_4bit=model_cfg.load_in_4bit,
            use_grad_ckpt="unsloth",
            lora_rank=model_cfg.lora.get("rank", 16),
            lora_alpha=model_cfg.lora.get("alpha", 32),
            seq_len=base_seq,
            # save_strategy / eval_strategy come from the model config — they
            # must agree for load_best_model_at_end, so keep only the disk cap
            # here rather than overriding the configured save strategy.
            sft_extra={"save_total_limit": 3},
        )
        if smoke_test:
            # Smoke shrinks the model and LoRA for speed, but NOT seq_len:
            # task prompts run 500–1200 tokens (banking77's 77-label list,
            # cuad's contract window), so a shorter limit truncates the
            # assistant answer off the end and completion-only loss collapses
            # to all-masked. seq_len stays at the real per-task value.
            cfg = dc_replace(
                cfg,
                load_dtype=torch.float32,
                load_in_4bit=False,
                use_grad_ckpt=False,
                lora_rank=4,
                lora_alpha=8,
                sft_extra={"bf16": False, "save_strategy": "no"},
            )
        return cfg


# ---------------------------------------------------------------------------
# Resume decision
# ---------------------------------------------------------------------------

def _resume_decision(prior_state: Optional[dict], current_input_hash: str) -> str:
    """Decide what to do with an existing train_state given the current inputs.

    Returns one of:
      "fresh"  — no prior state; start training (no resume, no discard).
      "skip"   — prior run completed with matching input_hash; can no-op.
      "resume" — prior run in-progress with matching input_hash; safe to resume.
      "stale"  — prior input_hash missing or different; discard checkpoints and
                 retrain from scratch. Treating "missing" as stale keeps legacy
                 train_state.json files (pre-input_hash) on the conservative
                 side: when we can't prove inputs match, retrain.
    """
    if not prior_state:
        return "fresh"
    if prior_state.get("input_hash") != current_input_hash:
        return "stale"
    return "skip" if prior_state.get("status") == "complete" else "resume"


def _discard_stale_checkpoints(ckpt_dir: Path) -> int:
    """Remove HF `checkpoint-*` dirs so a retrain cannot resume a superseded run.

    Resume-from-checkpoint is for crash recovery within a single run. Once the
    inputs change, the old checkpoints belong to a different model — and a
    *finished* checkpoint is the trap: resuming it leaves zero steps to run, so
    the "retrain" silently completes without training. Returns the count removed.
    """
    removed = 0
    for d in sorted(ckpt_dir.glob("checkpoint-*")):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _metadata_from_result(
    result,
    model_cfg: ModelConfig,
    task_cfg: TaskConfig,
    hw_cfg: HardwareConfig,
    data_path: Path,
    epochs: int,
    input_hash: str,
    smoke_test: bool,
) -> dict:
    """Merge TrainResult with run-level fields to produce the metadata.json shape.

    The Trainer adapter knows everything framework-coupled (dtypes, history,
    diagnostics); the orchestrator stamps in identifiers (model_id, task_id,
    git_sha, input_hash) and run-level economics (training_cost = gpu_hours *
    rate). This split keeps the on-disk schema stable across backends.
    """
    training_cost = 0.0 if smoke_test else result.gpu_hours * GPU_HOURLY
    return {
        "model_id": model_cfg.model_short,
        "task_id": task_cfg.task_id,
        "condition": CONDITION,
        "train_data_path": str(data_path),
        "n_train": result.n_train,
        "n_val": result.n_val,
        "epochs": epochs,
        "epochs_completed": result.epochs_completed,
        "early_stopped": result.early_stopped,
        "seq_len": hw_cfg.seq_len,
        "load_dtype": result.load_dtype,
        "compute_dtype": result.compute_dtype,
        "weight_dtype": result.weight_dtype,
        "training_cost": round(training_cost, 4),
        "training_time_min": result.elapsed_min,
        "gpu_hours": result.gpu_hours,
        "gpu_model": result.gpu_model,
        "peak_gpu_mem_mb": result.peak_gpu_mem_mb,
        "avg_gpu_util_pct": result.avg_gpu_util_pct,
        "eval_loss": result.eval_loss,
        "train_loss": result.train_loss,
        "model_used": result.model_used,
        # The pinned base revision actually bound this run, unless we fell back to a
        # different checkpoint (then the pin no longer applies).
        "model_revision": None if result.substituted else getattr(model_cfg, "revision", None),
        "substituted": result.substituted,
        "git_sha": _git_sha(),
        "input_hash": input_hash,
        "loss_history": result.loss_history,
        "eval_loss_history": result.eval_loss_history,
        "hyperparams": result.hyperparams,
        "training_diagnostics": result.training_diagnostics,
        "trainer_id": getattr(model_cfg, "trainer_id", DEFAULT_TRAINER_ID),
    }


def train_one(
    model_cfg: ModelConfig,
    task_cfg: TaskConfig,
    data_path: Path,
    dry_run: bool,
    auto_upload: bool = False,
    smoke_test: bool = False,
    ctx: str = "",
) -> dict:
    """Orchestrate a single model/task run. Returns the metadata dict."""
    echo = lambda msg: _echo(ctx, msg)
    task_id = task_cfg.task_id
    n_train = count_jsonl(data_path)

    # Merge task_cfg.lora_overrides into model_cfg.lora once, here, so every
    # downstream consumer (ConfigFactory, the Trainer adapter, the input hash)
    # sees the same effective LoRA config. Task-level overrides take priority
    # over model-level defaults; absent overrides leave model_cfg unchanged.
    if task_cfg.lora_overrides:
        effective_lora = {**model_cfg.lora, **task_cfg.lora_overrides}
        model_cfg = model_cfg.model_copy(update={"lora": effective_lora})
        echo(f"Task LoRA overrides applied: {task_cfg.lora_overrides}")

    hw_cfg = ConfigFactory.build(model_cfg, task_cfg, smoke_test)

    adapter_dir = adapter_path(REPO_ROOT, model_cfg.model_short, task_id, CONDITION, smoke=smoke_test)
    log_dir = training_meta_path(REPO_ROOT, "local", model_cfg.model_short, task_id, CONDITION, smoke=smoke_test).parent
    ckpt_dir = checkpoint_dir(model_cfg.model_short, task_id, CONDITION, smoke=smoke_test)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = 1 if smoke_test else get_epochs(n_train)
    echo(f"n={n_train}, epochs={epochs}, seq_len={hw_cfg.seq_len}")

    # Merge training_overrides into the hashed config so per-task LR/decay
    # tweaks trigger a retrain via inputs_changed.
    hashed_training = dict(model_cfg.training)
    if task_cfg.training_overrides:
        hashed_training.update(task_cfg.training_overrides)
    input_hash = training_inputs_hash(data_path, {
        "epochs": epochs,
        "smoke_test": smoke_test,
        "lora": model_cfg.lora,
        "training": hashed_training,
        "seq_len": hw_cfg.seq_len,
        "load_in_4bit": hw_cfg.load_in_4bit,
        # Turn markers drive completion-only loss masking, so changing them
        # changes the trained weights and must invalidate the cached run.
        "instruction_part": model_cfg.instruction_part,
        "response_part": model_cfg.response_part,
        # Trainer backend choice can change tokenisation, masking, optimizer
        # specifics — bust the hash when it changes.
        "trainer_id": getattr(model_cfg, "trainer_id", DEFAULT_TRAINER_ID),
        # Closure of training code: a change to the loss, LoRA wiring, or any
        # module train_local imports busts the hash and forces a retrain.
        "code": code_closure_hash(Path(__file__)),
    })
    meta_path = log_dir / "metadata.json"

    prior_state = load_train_state(model_cfg.model_short, task_id, CONDITION, smoke=smoke_test)
    # train_state.input_hash is the source of truth for "what produced these
    # checkpoints". Compare against the current input_hash *before* resume — if
    # they differ (or no hash was recorded — legacy state), discard the
    # checkpoints and retrain. Otherwise a finished checkpoint can silently
    # no-op a retrain whose inputs in fact changed.
    decision = _resume_decision(prior_state, input_hash)
    if decision == "skip":
        echo("SKIP: already complete")
        _log.info("training skip", model=model_cfg.model_short, task=task_id, condition=CONDITION,
                  event="stage_skip")
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}
    if decision == "stale":
        n_cleared = _discard_stale_checkpoints(ckpt_dir)
        reason = ("no input_hash on prior state"
                  if prior_state and not prior_state.get("input_hash")
                  else "inputs changed")
        echo(f"RETRAIN: {reason}"
             + (f" — discarded {n_cleared} stale checkpoint(s)" if n_cleared else ""))

    if dry_run:
        echo(f"[dry-run] Would train {model_cfg.model_id} on {data_path.name}")
        meta = {
            "model_id": model_cfg.model_short, "task_id": task_id, "condition": CONDITION,
            "train_data_path": str(data_path),
            "n_train": n_train, "epochs": epochs, "seq_len": hw_cfg.seq_len,
            "training_cost": 0, "training_time_min": 0,
            "eval_loss": None, "model_used": model_cfg.model_id, "substituted": False,
        }
        atomic_write_json(meta, log_dir / "metadata.json")
        return meta

    resume_ckpt = find_hf_resume_checkpoint(model_cfg.model_short, task_id, CONDITION, smoke=smoke_test)
    if resume_ckpt:
        echo(f"Resuming from checkpoint: {resume_ckpt.name}")
    save_train_state(model_cfg.model_short, task_id, CONDITION, {
        "status": "in_progress",
        "epoch": 0,
        "global_step": 0,
        "input_hash": input_hash,
    }, smoke=smoke_test)

    # ── Build spec, look up Trainer adapter, run training ──────────────────
    val_path_candidate = data_path.parent / "val.jsonl"
    val_path = val_path_candidate if (not smoke_test and val_path_candidate.exists()) else None
    callback = CheckpointCallback(
        echo, model_cfg.model_short, task_id, CONDITION, input_hash, smoke=smoke_test
    )
    spec = TrainSpec(
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        hw_cfg=hw_cfg,
        data_path=data_path,
        val_path=val_path,
        epochs=epochs,
        resume_ckpt=resume_ckpt,
        ckpt_dir=ckpt_dir,
        adapter_dir=adapter_dir,
        smoke_test=smoke_test,
        callbacks=[callback],
        echo=echo,
    )

    trainer_id = getattr(model_cfg, "trainer_id", DEFAULT_TRAINER_ID)
    TrainerCls = get_trainer(trainer_id)
    echo(f"Trainer backend: {trainer_id}")

    with training_log(ckpt_dir):
        result = TrainerCls().train(spec)

    meta = _metadata_from_result(
        result, model_cfg, task_cfg, hw_cfg, data_path, epochs, input_hash, smoke_test
    )
    atomic_write_json(meta, log_dir / "metadata.json")
    atomic_write_json(meta, ckpt_dir / "metadata.json")
    save_train_state(model_cfg.model_short, task_id, CONDITION, {
        "status": "complete",
        "eval_loss": result.eval_loss,
        "train_loss": result.train_loss,
        "training_time_min": result.elapsed_min,
        "training_cost": meta["training_cost"],
        "input_hash": input_hash,
    }, smoke=smoke_test)

    _log.info("training complete", model=model_cfg.model_short, task=task_id, condition=CONDITION,
              event="stage_complete", training_cost=meta.get("training_cost"),
              training_time_min=meta.get("training_time_min"), gpu_hours=meta.get("gpu_hours"),
              peak_gpu_mem_mb=meta.get("peak_gpu_mem_mb"), avg_gpu_util_pct=meta.get("avg_gpu_util_pct"),
              eval_loss=meta.get("eval_loss"), n_train=meta.get("n_train"))

    if auto_upload and not smoke_test:
        _upload_adapter(model_cfg.model_short, task_id, CONDITION, ctx=ctx)
    elif auto_upload and smoke_test:
        _echo(ctx, "Skipping auto-upload: smoke runs must not push adapters to HF.")

    return meta


def _upload_adapter(model_short: str, task_id: str, condition: str, ctx: str = "") -> None:
    import subprocess
    _echo(ctx, "Auto-uploading to HuggingFace...")
    result = subprocess.run(
        ["python", str(REPO_ROOT / "scripts" / "upload_artifacts.py"),
         "--model", model_short, "--task", task_id, "--condition", condition],
        capture_output=False,
    )
    if result.returncode != 0:
        _echo(ctx, f"WARNING: upload failed (exit {result.returncode})")


@click.command()
@click.option("--model", default=None, help="Model config ID or 'all'. Defaults to 'qwen2.5-0.5b' with --smoke-test, 'qwen3-8b' otherwise.")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--cap", default=None, type=int, help="Cap training data to N rows (writes train_capN.jsonl for reproducibility)")
@click.option("--auto-upload", is_flag=True, help="Upload adapter to HuggingFace after each run (persistence)")
@click.option("--dry-run", is_flag=True, help="Validate configs without training")
@click.option("--smoke-test", is_flag=True, help="Smoke test: reduced seq_len/rank/alpha, 4 threads.")
def main(model: Optional[str], task: str, cap: Optional[int], auto_upload: bool, dry_run: bool, smoke_test: bool) -> None:
    """QLoRA fine-tune one or more model/task combinations."""
    configure(REPO_ROOT)
    check_allowed_gpu(skip=smoke_test or dry_run)
    if smoke_test:
        torch.set_num_threads(4)
        click.echo("Smoke-test mode: 4 threads, seq_len=256, r=4.")

    if model is None:
        model = "qwen2.5-0.5b" if smoke_test else "qwen3-8b"

    model_ids = ALL_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]

    failures = []
    for mid in model_ids:
        model_cfg = load_model_config(mid)
        for tid in task_ids:
            task_cfg = load_task_config(tid)
            ctx = f"{mid}/{tid}"

            src_name = "smoke_train.jsonl" if smoke_test else "train.jsonl"
            src = prepared_path(REPO_ROOT, tid, smoke=smoke_test) / src_name
            if not src.exists():
                click.echo(f"  [{ctx}] SKIP: {src_name} not found", err=True)
                if not dry_run:
                    failures.append((f"{mid}/{tid}", "data file missing"))
                continue
            data_file = get_or_create_cap(src.parent, cap, ctx=ctx) if cap is not None and not smoke_test else src
            if not dry_run:
                try:
                    reject_test_path(data_file)
                    require_jsonl(data_file, min_rows=1, check_chat_format=True)
                except Exception as exc:
                    click.echo(f"  [{ctx}] ERROR: input validation failed: {exc}", err=True)
                    failures.append((f"{mid}/{tid}", str(exc)))
                    continue

            try:
                train_one(model_cfg, task_cfg, data_file, dry_run, auto_upload=auto_upload, smoke_test=smoke_test, ctx=ctx)
            except Exception as exc:
                click.echo(f"  [{ctx}] ERROR: {exc}", err=True)
                traceback.print_exc()
                _log.error(f"training failed: {type(exc).__name__}: {exc}",
                           model=mid, task=tid, condition=CONDITION,
                           exc=str(exc), traceback=traceback.format_exc())
                failures.append((f"{mid}/{tid}", str(exc)))

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        os._exit(1)
    click.echo("\nAll training jobs completed.")
    os._exit(0)


if __name__ == "__main__":
    main()
