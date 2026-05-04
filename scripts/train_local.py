"""QLoRA fine-tuning for open-source models via Unsloth."""
from __future__ import annotations

import ctypes
import faulthandler
import gc
import json
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

# ML libs imported lazily inside run_training_task — keeps module importable on CPU for tests

import click
import yaml
from pydantic import BaseModel, Field

from checkpoint_utils import (
    NETWORK_VOLUME,
    atomic_write_json,
    checkpoint_dir,
    find_hf_resume_checkpoint,
    load_train_state,
    nv_prepared_dir,
    save_train_state,
    training_log,
)
from utils import write_jsonl
from pipeline.cache import inputs_changed, training_inputs_hash
from pipeline.config import get_local_models, get_tasks
from pipeline.paths import adapter_path, training_meta_path
from pipeline.versioning import git_sha as _git_sha

REPO_ROOT = Path(__file__).parent.parent
ALL_TASKS: list[str] = get_tasks()
ALL_MODELS: list[str] = [m["id"] for m in get_local_models()]
GPU_HOURLY = 0.49  # Default GPU hourly rate — override via pricing.yaml
CONDITION = "lora"

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def _rss_mb() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 // (1024 * 1024)
    except Exception:
        return 0


class ModelConfig(BaseModel):
    model_id: str
    model_short: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: str = "bfloat16"
    enable_thinking: Optional[bool] = None
    fallback_model_id: Optional[str] = None
    lora: dict = Field(default_factory=dict)
    training: dict = Field(default_factory=dict)


class TaskConfig(BaseModel):
    task_id: str
    training_cap: Optional[int] = None
    max_seq_length: Optional[int] = None  # overrides model max_seq_length when set


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


def _resolve_prepared(task_id: str, filename: str) -> Optional[Path]:
    """Return path to a prepared data file, falling back to the network volume."""
    p = REPO_ROOT / "data" / "prepared" / task_id / filename
    if p.exists():
        return p
    nv = nv_prepared_dir(task_id) / filename
    return nv if nv.exists() else None


def get_or_create_cap(data_dir: Path, n: int) -> Path:
    """Return path to a fixed random-sample of n rows from train.jsonl, creating it if needed."""
    cap_path = data_dir / f"train_cap{n}.jsonl"
    if cap_path.exists():
        return cap_path
    src = data_dir / "train.jsonl"
    with open(src) as f:
        rows = [json.loads(line) for line in f]
    sample = random.Random(42).sample(rows, min(n, len(rows)))
    write_jsonl(sample, cap_path)
    click.echo(f"  Cap: wrote {len(sample)} rows to {cap_path.name}")
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
            sft_extra={"save_strategy": "steps", "save_steps": 250, "save_total_limit": 3},
        )
        if smoke_test:
            cfg = dc_replace(
                cfg,
                load_dtype=torch.float32,
                load_in_4bit=False,
                use_grad_ckpt=False,
                lora_rank=4,
                lora_alpha=8,
                seq_len=256,
                sft_extra={"bf16": False, "save_strategy": "no"},
            )
        return cfg


# ---------------------------------------------------------------------------
# Core training logic (isolated for GC scoping)
# ---------------------------------------------------------------------------

def run_training_task(
    model_cfg: ModelConfig,
    task_cfg: TaskConfig,
    data_path: Path,
    hw_cfg: HardwareConfig,
    n_train: int,
    epochs: int,
    resume_ckpt: Optional[Path],
    ckpt_dir: Path,
    adapter_dir: Path,
    log_dir: Path,
    smoke_test: bool,
    input_hash: str = "",
) -> dict:
    """Load model, train, save adapter. Returns metadata dict.

    All heavy objects (model, trainer) are deleted before returning so the GC
    can reclaim memory before the next task starts.
    """
    # Heavy GPU libs imported here so the module is importable on CPU for tests
    import unsloth  # noqa: F401 — must come before transformers/peft
    from unsloth import FastModel
    import datasets as hf_datasets
    from transformers import TrainerCallback
    from trl import SFTTrainer, SFTConfig

    task_id = task_cfg.task_id
    model_id = model_cfg.model_id
    substituted = False

    click.echo(f"  Loading {model_id} on {hw_cfg.device.upper()}...")
    click.echo(f"  Config: load_in_4bit={hw_cfg.load_in_4bit} dtype={hw_cfg.load_dtype} grad_ckpt={hw_cfg.use_grad_ckpt}")
    click.echo(f"  SFT overrides: {hw_cfg.sft_extra or '(none)'}")

    def _load_model(mid: str):
        return FastModel.from_pretrained(
            model_name=mid,
            max_seq_length=hw_cfg.seq_len,
            load_in_4bit=hw_cfg.load_in_4bit,
            dtype=hw_cfg.load_dtype,
            device_map=hw_cfg.device,
        )

    try:
        model, tokenizer = _load_model(model_id)
    except Exception as exc:
        if model_cfg.fallback_model_id:
            click.echo(f"  WARNING: {model_id} failed ({exc}). Falling back to {model_cfg.fallback_model_id}")
            model_id = model_cfg.fallback_model_id
            substituted = True
            model, tokenizer = _load_model(model_id)
        else:
            raise

    click.echo("  Applying LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=hw_cfg.lora_rank,
        lora_alpha=hw_cfg.lora_alpha,
        lora_dropout=model_cfg.lora.get("dropout", 0.05),
        bias="none",
        use_rslora=model_cfg.lora.get("use_rslora", True),
        use_gradient_checkpointing=hw_cfg.use_grad_ckpt,
    )

    class _CheckpointCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            click.echo(f"  Training started: {state.max_steps} steps")

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1 and state.log_history:
                click.echo(f"  Step 1 complete — loss={state.log_history[-1].get('loss', '?'):.4f}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                step = state.global_step
                loss = logs.get("loss", logs.get("train_loss"))
                lr = logs.get("learning_rate")
                parts = [f"step {step}"]
                if loss is not None:
                    parts.append(f"loss={loss:.4f}")
                if lr is not None:
                    parts.append(f"lr={lr:.2e}")
                click.echo("  " + " | ".join(parts))

        def on_save(self, args, state, control, **kwargs):
            save_train_state(model_cfg.model_short, task_id, CONDITION, {
                "status": "in_progress",
                "epoch": state.epoch,
                "global_step": state.global_step,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
            })

    click.echo(f"  Loading dataset from {data_path} ...")
    with open(data_path) as f:
        rows = [json.loads(line) for line in f]
    click.echo(f"  {len(rows)} examples loaded")
    train_ds = hf_datasets.Dataset.from_list(rows)

    # Pre-apply the chat template so trl tokenizes a plain "text" field.
    # When enable_thinking is explicitly False, suppress <think> tokens.
    template_kwargs = {}
    if model_cfg.enable_thinking is False:
        template_kwargs["enable_thinking"] = False

    click.echo("  Applying chat template...")
    def apply_template(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            **template_kwargs,
        )}
    train_ds = train_ds.map(apply_template)
    click.echo(f"  Dataset ready: {len(train_ds)} rows")

    training_cfg = model_cfg.training
    sft_kwargs = dict(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=training_cfg.get("warmup_steps", 50),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        optim=training_cfg.get("optim", "adamw_8bit"),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        bf16=training_cfg.get("bf16", True),
        seed=training_cfg.get("seed", 42),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        eval_strategy=training_cfg.get("eval_strategy", "epoch"),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_cfg.get("greater_is_better", False),
        logging_steps=training_cfg.get("logging_steps", 10),
        report_to=training_cfg.get("report_to", "none"),
        max_seq_length=hw_cfg.seq_len,
        dataset_text_field="text",
        packing=False,
    )
    sft_kwargs.update(hw_cfg.sft_extra)
    click.echo(
        f"  SFTConfig: epochs={sft_kwargs['num_train_epochs']}"
        f" batch={sft_kwargs['per_device_train_batch_size']}"
        f" accum={sft_kwargs['gradient_accumulation_steps']}"
        f" lr={sft_kwargs['learning_rate']:.2e}"
        f" warmup_steps={sft_kwargs['warmup_steps']}"
    )
    sft_config = SFTConfig(**sft_kwargs)

    click.echo("  Building trainer...")
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds, args=sft_config,
        callbacks=[_CheckpointCallback()],
    )

    click.echo("  Starting trainer.train()...")
    t0 = time.time()
    result = trainer.train(
        resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None
    )
    elapsed_min = (time.time() - t0) / 60
    training_cost = 0.0 if smoke_test else (elapsed_min / 60) * GPU_HOURLY

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    nv_adapter_dir = ckpt_dir / "final_adapter"
    shutil.copytree(str(adapter_dir), str(nv_adapter_dir), dirs_exist_ok=True)

    m = result.metrics or {}
    eval_loss  = m.get("eval_loss")
    train_loss = round(m["train_loss"], 4) if "train_loss" in m else None

    meta = {
        "model_id": model_cfg.model_short,
        "task_id": task_id,
        "condition": CONDITION,
        "train_data_path": str(data_path),
        "n_train": n_train,
        "epochs": epochs,
        "seq_len": hw_cfg.seq_len,
        "training_cost": round(training_cost, 4),
        "training_time_min": round(elapsed_min, 1),
        "eval_loss": eval_loss,
        "train_loss": train_loss,
        "model_used": model_id,
        "substituted": substituted,
        "git_sha": _git_sha(),
        "input_hash": input_hash,
    }
    atomic_write_json(meta, log_dir / "metadata.json")
    atomic_write_json(meta, ckpt_dir / "metadata.json")
    save_train_state(model_cfg.model_short, task_id, CONDITION, {
        "status": "complete",
        "eval_loss": eval_loss,
        "train_loss": train_loss,
        "training_time_min": round(elapsed_min, 1),
        "training_cost": round(training_cost, 4),
    })

    loss_display = eval_loss if eval_loss is not None else train_loss
    click.echo(f"  Done: {elapsed_min:.1f} min, ${training_cost:.3f}, loss={loss_display}")

    rss_before = _rss_mb()
    del trainer, model, tokenizer, train_ds, result
    gc.collect()
    gc.collect()  # two passes: PyTorch cyclic refs may survive the first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    if _libc:
        _libc.malloc_trim(0)
    rss_after = _rss_mb()
    click.echo(f"  Memory released after {task_id}/{CONDITION}: {rss_before - rss_after} MB freed (RSS {rss_before}→{rss_after} MB).")

    return meta


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def train_one(
    model_cfg: ModelConfig,
    task_cfg: TaskConfig,
    data_path: Path,
    dry_run: bool,
    auto_upload: bool = False,
    smoke_test: bool = False,
) -> dict:
    """Orchestrate a single model/task run. Returns metadata dict."""
    task_id = task_cfg.task_id
    n_train = count_jsonl(data_path)
    hw_cfg = ConfigFactory.build(model_cfg, task_cfg, smoke_test)

    adapter_dir = adapter_path(REPO_ROOT, model_cfg.model_short, task_id, CONDITION)
    log_dir = training_meta_path(REPO_ROOT, "local", model_cfg.model_short, task_id, CONDITION).parent
    ckpt_dir = checkpoint_dir(model_cfg.model_short, task_id, CONDITION)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = 1 if smoke_test else get_epochs(n_train)
    click.echo(f"  [{model_cfg.model_short}/{task_id}/{CONDITION}] n={n_train}, epochs={epochs}, seq_len={hw_cfg.seq_len}")

    input_hash = training_inputs_hash(data_path, {
        "epochs": epochs,
        "smoke_test": smoke_test,
        "lora": model_cfg.lora,
        "training": model_cfg.training,
        "seq_len": hw_cfg.seq_len,
        "load_in_4bit": hw_cfg.load_in_4bit,
    })
    meta_path = log_dir / "metadata.json"

    prior_state = load_train_state(model_cfg.model_short, task_id, CONDITION)
    if prior_state and prior_state.get("status") == "complete":
        if not inputs_changed(input_hash, meta_path):
            click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{CONDITION}]: already complete")
            if meta_path.exists():
                with open(meta_path) as f:
                    return json.load(f)
            return {}
        click.echo(f"  RETRAIN [{model_cfg.model_short}/{task_id}/{CONDITION}]: inputs changed")

    if dry_run:
        click.echo(f"  [dry-run] Would train {model_cfg.model_id} on {data_path.name}")
        meta = {
            "model_id": model_cfg.model_short, "task_id": task_id, "condition": CONDITION,
            "train_data_path": str(data_path),
            "n_train": n_train, "epochs": epochs, "seq_len": hw_cfg.seq_len,
            "training_cost": 0, "training_time_min": 0,
            "eval_loss": None, "model_used": model_cfg.model_id, "substituted": False,
        }
        atomic_write_json(meta, log_dir / "metadata.json")
        return meta

    resume_ckpt = find_hf_resume_checkpoint(model_cfg.model_short, task_id, CONDITION)
    if resume_ckpt:
        click.echo(f"  Resuming from checkpoint: {resume_ckpt.name}")
    save_train_state(model_cfg.model_short, task_id, CONDITION, {
        "status": "in_progress",
        "epoch": 0,
        "global_step": 0,
    })

    with training_log(ckpt_dir):
        meta = run_training_task(
            model_cfg=model_cfg,
            task_cfg=task_cfg,
            data_path=data_path,
            hw_cfg=hw_cfg,
            n_train=n_train,
            epochs=epochs,
            resume_ckpt=resume_ckpt,
            ckpt_dir=ckpt_dir,
            adapter_dir=adapter_dir,
            log_dir=log_dir,
            smoke_test=smoke_test,
            input_hash=input_hash,
        )

    if auto_upload:
        _upload_adapter(model_cfg.model_short, task_id, CONDITION)

    return meta


def _upload_adapter(model_short: str, task_id: str, condition: str) -> None:
    import subprocess
    click.echo(f"  Auto-uploading {model_short}/{task_id}/{condition} to HuggingFace...")
    result = subprocess.run(
        ["python", str(REPO_ROOT / "scripts" / "upload_artifacts.py"),
         "--model", model_short, "--task", task_id, "--condition", condition],
        capture_output=False,
    )
    if result.returncode != 0:
        click.echo(f"  WARNING: upload failed (exit {result.returncode})", err=True)


@click.command()
@click.option("--model", default=None, help="Model config ID or 'all'. Defaults to 'qwen2.5-0.5b' with --smoke-test, 'qwen3-8b' otherwise.")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--cap", default=None, type=int, help="Cap training data to N rows (writes train_capN.jsonl for reproducibility)")
@click.option("--auto-upload", is_flag=True, help="Upload adapter to HuggingFace after each run (persistence)")
@click.option("--dry-run", is_flag=True, help="Validate configs without training")
@click.option("--smoke-test", is_flag=True, help="Smoke test: reduced seq_len/rank/alpha, 4 threads.")
def main(model: Optional[str], task: str, cap: Optional[int], auto_upload: bool, dry_run: bool, smoke_test: bool) -> None:
    """QLoRA fine-tune one or more model/task combinations."""
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

            src_name = "smoke_train.jsonl" if smoke_test else "train.jsonl"
            src = _resolve_prepared(tid, src_name)
            if src is None:
                click.echo(f"  SKIP [{mid}/{tid}]: {src_name} not found", err=True)
                if not dry_run:
                    failures.append((f"{mid}/{tid}", "data file missing"))
                continue
            data_file = get_or_create_cap(src.parent, cap) if cap is not None and not smoke_test else src

            try:
                train_one(model_cfg, task_cfg, data_file, dry_run, auto_upload=auto_upload, smoke_test=smoke_test)
            except Exception as exc:
                click.echo(f"  ERROR [{mid}/{tid}]: {exc}", err=True)
                traceback.print_exc()
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
