"""QLoRA fine-tuning for open-source models via Unsloth."""
from __future__ import annotations

import faulthandler
import gc
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

faulthandler.enable()

import torch

_SMOKE_TEST_EARLY = "--smoke-test" in sys.argv

# Smoke-test forces these; otherwise allow the environment to override.
if _SMOKE_TEST_EARLY:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["UNSLOTH_DISABLE_AUTO_PADDING_FREE"] = "1"
else:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_PADDING_FREE", "1")

# unsloth_zoo calls mem_get_info at import time; crashes on XPU devices that
# don't support free-memory queries. Patch before importing unsloth.
if hasattr(torch, "xpu") and torch.xpu.is_available():
    _orig_info = torch.xpu.mem_get_info

    def _safe_xpu_mem_get_info(device=None):
        try:
            return _orig_info(device)
        except Exception:
            total = torch.xpu.get_device_properties(0).total_memory
            return (total, total)

    torch.xpu.mem_get_info = _safe_xpu_mem_get_info
    torch.xpu.memory.mem_get_info = _safe_xpu_mem_get_info

import unsloth  # must be imported before transformers/peft

from dotenv import load_dotenv
load_dotenv()

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

REPO_ROOT = Path(__file__).parent.parent
ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa", "mbpp"]
ALL_MODELS = ["qwen3-8b"]
GPU_HOURLY = 0.49  # Default GPU hourly rate — override via pricing.yaml


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
    efficiency_curve_sizes: list[int] = Field(default_factory=list)


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

        if smoke_test:
            return HardwareConfig(
                device="cpu",
                load_dtype=torch.float32,
                load_in_4bit=False,
                use_grad_ckpt=True,
                lora_rank=4,
                lora_alpha=8,
                seq_len=256,
                sft_extra={"use_cpu": True, "bf16": False, "optim": "adamw_8bit"},
            )

        if torch.cuda.is_available():
            return HardwareConfig(
                device="cuda",
                load_dtype=torch.bfloat16,
                load_in_4bit=model_cfg.load_in_4bit,
                use_grad_ckpt="unsloth",
                lora_rank=model_cfg.lora.get("rank", 16),
                lora_alpha=model_cfg.lora.get("alpha", 32),
                seq_len=base_seq,
                sft_extra={},
            )

        return HardwareConfig(
            device="cpu",
            load_dtype=None,
            load_in_4bit=False,
            use_grad_ckpt=True,
            lora_rank=model_cfg.lora.get("rank", 16),
            lora_alpha=model_cfg.lora.get("alpha", 32),
            seq_len=base_seq,
            sft_extra={"use_cpu": True, "bf16": False, "optim": "adamw_torch"},
        )


# ---------------------------------------------------------------------------
# Core training logic (isolated for GC scoping)
# ---------------------------------------------------------------------------

def run_training_task(
    model_cfg: ModelConfig,
    task_cfg: TaskConfig,
    condition: str,
    data_path: Path,
    hw_cfg: HardwareConfig,
    n_train: int,
    epochs: int,
    resume_ckpt: Optional[Path],
    ckpt_dir: Path,
    adapter_dir: Path,
    log_dir: Path,
    smoke_test: bool,
) -> dict:
    """Load model, train, save adapter. Returns metadata dict.

    All heavy objects (model, trainer) are deleted before returning so the GC
    can reclaim memory before the next task starts.
    """
    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    import datasets as hf_datasets

    task_id = task_cfg.task_id
    model_id = model_cfg.model_id
    substituted = False

    click.echo(f"  Loading {model_id} on {hw_cfg.device.upper()}...")
    click.echo(f"  Config: load_in_4bit={hw_cfg.load_in_4bit} dtype={hw_cfg.load_dtype} grad_ckpt={hw_cfg.use_grad_ckpt}")
    click.echo(f"  SFT overrides: {hw_cfg.sft_extra or '(none)'}")

    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_id,
            max_seq_length=hw_cfg.seq_len,
            load_in_4bit=hw_cfg.load_in_4bit,
            dtype=hw_cfg.load_dtype,
            device_map=hw_cfg.device,
        )
    except Exception as exc:
        if model_cfg.fallback_model_id:
            click.echo(f"  WARNING: {model_id} failed ({exc}). Falling back to {model_cfg.fallback_model_id}")
            model_id = model_cfg.fallback_model_id
            substituted = True
            model, tokenizer = FastModel.from_pretrained(
                model_name=model_id,
                max_seq_length=hw_cfg.seq_len,
                load_in_4bit=hw_cfg.load_in_4bit,
                dtype=hw_cfg.load_dtype,
                device_map=hw_cfg.device,
            )
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

    if hw_cfg.device == "cpu":
        model = model.to("cpu")

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
            save_train_state(model_cfg.model_short, task_id, condition, {
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
        "condition": condition,
        "n_train": n_train,
        "epochs": epochs,
        "seq_len": hw_cfg.seq_len,
        "training_cost": round(training_cost, 4),
        "training_time_min": round(elapsed_min, 1),
        "eval_loss": eval_loss,
        "train_loss": train_loss,
        "model_used": model_id,
        "substituted": substituted,
    }
    atomic_write_json(meta, log_dir / "metadata.json")
    atomic_write_json(meta, ckpt_dir / "metadata.json")
    save_train_state(model_cfg.model_short, task_id, condition, {
        "status": "complete",
        "eval_loss": eval_loss,
        "train_loss": train_loss,
        "training_time_min": round(elapsed_min, 1),
        "training_cost": round(training_cost, 4),
    })

    loss_display = eval_loss if eval_loss is not None else train_loss
    click.echo(f"  Done: {elapsed_min:.1f} min, ${training_cost:.3f}, loss={loss_display}")

    # Release memory before the next task.
    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

    return meta


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def train_one(
    model_cfg: ModelConfig,
    task_cfg: TaskConfig,
    condition: str,
    data_path: Path,
    dry_run: bool,
    auto_upload: bool = False,
    smoke_test: bool = False,
) -> dict:
    """Orchestrate a single model/task/condition run. Returns metadata dict."""
    task_id = task_cfg.task_id
    n_train = count_jsonl(data_path)
    hw_cfg = ConfigFactory.build(model_cfg, task_cfg, smoke_test)

    adapter_dir = REPO_ROOT / "results" / "adapters" / model_cfg.model_short / task_id / condition
    log_dir = REPO_ROOT / "results" / "training" / model_cfg.model_short / task_id / condition
    ckpt_dir = checkpoint_dir(model_cfg.model_short, task_id, condition)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = 1 if smoke_test else get_epochs(n_train)
    click.echo(f"  [{model_cfg.model_short}/{task_id}/{condition}] n={n_train}, epochs={epochs}, seq_len={hw_cfg.seq_len}")

    prior_state = load_train_state(model_cfg.model_short, task_id, condition)
    if prior_state and prior_state.get("status") == "complete":
        click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{condition}]: already complete")
        meta_path = log_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}

    if dry_run:
        click.echo(f"  [dry-run] Would train {model_cfg.model_id} on {data_path.name}")
        meta = {
            "model_id": model_cfg.model_short, "task_id": task_id, "condition": condition,
            "n_train": n_train, "epochs": epochs, "seq_len": hw_cfg.seq_len,
            "training_cost": 0, "training_time_min": 0,
            "eval_loss": None, "model_used": model_cfg.model_id, "substituted": False,
        }
        atomic_write_json(meta, log_dir / "metadata.json")
        return meta

    resume_ckpt = find_hf_resume_checkpoint(model_cfg.model_short, task_id, condition)
    if resume_ckpt:
        click.echo(f"  Resuming from checkpoint: {resume_ckpt.name}")
    save_train_state(model_cfg.model_short, task_id, condition, {
        "status": "in_progress",
        "epoch": 0,
        "global_step": 0,
    })

    with training_log(ckpt_dir):
        meta = run_training_task(
            model_cfg=model_cfg,
            task_cfg=task_cfg,
            condition=condition,
            data_path=data_path,
            hw_cfg=hw_cfg,
            n_train=n_train,
            epochs=epochs,
            resume_ckpt=resume_ckpt,
            ckpt_dir=ckpt_dir,
            adapter_dir=adapter_dir,
            log_dir=log_dir,
            smoke_test=smoke_test,
        )

    if auto_upload:
        _upload_adapter(model_cfg.model_short, task_id, condition)

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
@click.option("--model", default=None, help="Model config ID or 'all'. Defaults to 'tiny' with --smoke-test, 'qwen3-8b' otherwise.")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="lora-500|lora-full|all")
@click.option("--auto-upload", is_flag=True, help="Upload adapter to HuggingFace after each run (persistence)")
@click.option("--dry-run", is_flag=True, help="Validate configs without training")
@click.option("--smoke-test", is_flag=True, help="CPU-only smoke test: reduced seq_len/rank/alpha, 4 threads.")
def main(model: Optional[str], task: str, condition: str, auto_upload: bool, dry_run: bool, smoke_test: bool) -> None:
    """QLoRA fine-tune one or more model/task/condition combinations."""
    if smoke_test:
        torch.set_num_threads(4)
        click.echo("Smoke-test mode: CPU only, 4 threads, seq_len=256, r=4.")

    if model is None:
        model = "tiny" if smoke_test else "qwen3-8b"

    model_ids = ALL_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]
    conditions = ["lora-500", "lora-full"] if condition == "all" else [condition]

    failures = []
    for mid in model_ids:
        model_cfg = load_model_config(mid)
        for tid in task_ids:
            task_cfg = load_task_config(tid)  # load once per task, not per condition
            prepared_dir = REPO_ROOT / "data" / "prepared" / tid
            for cond in conditions:
                filename = "train_500.jsonl" if cond == "lora-500" else "train_full.jsonl"
                data_file = prepared_dir / filename
                if not data_file.exists():
                    nv_file = nv_prepared_dir(tid) / filename
                    if nv_file.exists():
                        data_file = nv_file
                    else:
                        click.echo(f"  SKIP [{mid}/{tid}/{cond}]: {data_file} not found", err=True)
                        if not dry_run:
                            failures.append((f"{mid}/{tid}/{cond}", "data file missing"))
                        continue
                try:
                    train_one(model_cfg, task_cfg, cond, data_file, dry_run, auto_upload=auto_upload, smoke_test=smoke_test)
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                    traceback.print_exc()
                    failures.append((f"{mid}/{tid}/{cond}", str(exc)))

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll training jobs completed.")


if __name__ == "__main__":
    main()
