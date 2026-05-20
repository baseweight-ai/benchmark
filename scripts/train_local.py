"""QLoRA fine-tuning for open-source models via Unsloth."""
from __future__ import annotations

import ctypes
import faulthandler
import gc
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

# Heavy ML libs (unsloth, trl, datasets) imported lazily inside run_training_task
# to keep this module CPU-importable for tests. transformers is CPU-safe, so
# TrainerCallback can be a module-level base class for _CheckpointCallback below.
from transformers import EarlyStoppingCallback, TrainerCallback

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
from pipeline.hardware import check_allowed_gpu, get_current_gpu_name
from pipeline.log import configure, get_logger
from pipeline.paths import adapter_path, prepared_path, training_meta_path
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


def _dtype_str(dtype) -> str:
    """Convert a torch dtype to a canonical string like 'bfloat16' or 'float32'."""
    if dtype is None:
        return "auto"
    return str(dtype).split(".")[-1]  # "torch.bfloat16" → "bfloat16"


def _compute_dtype_str(model) -> str:
    """Infer the model's effective compute dtype.

    For bitsandbytes QLoRA, the authoritative source is quantization_config —
    parameter dtypes reflect storage format (4-bit/8-bit), not compute format.
    Falls back to the first floating-point parameter's dtype for non-quantized models.
    """
    qcfg = getattr(getattr(model, "config", None), "quantization_config", None)
    if qcfg is not None and hasattr(qcfg, "bnb_4bit_compute_dtype"):
        return _dtype_str(qcfg.bnb_4bit_compute_dtype)
    try:
        param = next(p for p in model.parameters() if p.dtype not in (torch.int8,))
        return _dtype_str(param.dtype)
    except StopIteration:
        return "unknown"


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


def eval_save_steps(n_train: int, eff_batch: int, evals_per_epoch: int) -> int:
    """Steps between evaluations: a fixed number of evals per epoch, so the
    validation optimum stays locatable for early stopping at any dataset size."""
    steps_per_epoch = math.ceil(n_train / max(1, eff_batch))
    return max(1, steps_per_epoch // max(1, evals_per_epoch))


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
# Training quality analysis
# ---------------------------------------------------------------------------

def _analyze_training(
    losses: list[float],
    anomalies: list[dict],
    echo,
    val_losses: Optional[list[float]] = None,
) -> dict:
    """Post-training diagnostics: convergence, plateau, divergence, and overfitting.

    val_losses: per-epoch validation losses from an in-training eval split, used to
    detect overfitting (train loss ↓ while val loss ↑). When None, overfitting
    cannot be assessed and overfitting_detected is set to None.
    """
    diag: dict = {
        "anomalies": anomalies,
        "converged": None,
        "plateaued": False,
        "diverged": False,
        "loss_improvement_pct": None,
        "overfitting_detected": None,
    }
    if not losses:
        return diag

    first, last = losses[0], losses[-1]
    improvement = (first - last) / max(first, 1e-8)
    diag["loss_improvement_pct"] = round(improvement * 100, 2)
    diag["converged"] = improvement > 0.05

    if not diag["converged"]:
        echo(f"WARNING: loss improved only {improvement * 100:.1f}% — model may not have trained "
             f"meaningfully (start={first:.4f}, end={last:.4f})")

    # Divergence: end-third mean > start-third mean by >5%
    if len(losses) >= 10:
        third = len(losses) // 3
        start_mean = sum(losses[:third]) / third
        end_mean = sum(losses[-third:]) / third
        if end_mean > start_mean * 1.05:
            diag["diverged"] = True
            echo(f"WARNING: loss diverging — early mean={start_mean:.4f}, late mean={end_mean:.4f}")

    # Plateau: Q4 mean within 1% of Q3 mean (and not diverging)
    if len(losses) >= 8 and not diag["diverged"]:
        q3_start = len(losses) // 2
        q4_start = 3 * len(losses) // 4
        q3_mean = sum(losses[q3_start:q4_start]) / max(1, q4_start - q3_start)
        q4_mean = sum(losses[q4_start:]) / max(1, len(losses) - q4_start)
        rel_change = abs(q3_mean - q4_mean) / max(q3_mean, 1e-8)
        if rel_change < 0.01:
            diag["plateaued"] = True
            echo(f"NOTE: loss plateaued — Q3={q3_mean:.4f}, Q4={q4_mean:.4f} ({rel_change * 100:.2f}% change)")

    # Overfitting: by the final epoch, validation loss has climbed >5% above its
    # best (minimum) value while training loss kept falling. The reference is the
    # minimum, not the first epoch — a U-shaped val curve whose best generalization
    # point is mid-run still overfits even when the last epoch sits below the
    # (noisy) first one. Only assessable when a validation split was reserved
    # during training (val_losses provided).
    if val_losses and len(val_losses) >= 2:
        best_val = min(val_losses)
        if val_losses[-1] > best_val * 1.05 and improvement > 0:
            diag["overfitting_detected"] = True
            echo(f"WARNING: overfitting — val_loss bottomed at {best_val:.4f} then rose "
                 f"to {val_losses[-1]:.4f} (+{(val_losses[-1] / best_val - 1) * 100:.1f}%) "
                 f"while train_loss fell {improvement * 100:.1f}%")
        else:
            diag["overfitting_detected"] = False
    elif val_losses is not None:
        # val_losses provided but too few epochs to assess
        diag["overfitting_detected"] = None

    return diag


def _verify_completion_masking(train_ds, echo) -> None:
    """Fail loudly if completion-only loss masking degenerated.

    train_on_responses_only masks prompt tokens to -100 and leaves assistant
    tokens supervised. If the configured instruction/response markers don't
    match the model's chat template, masking silently degenerates into one of
    two ruined runs: every token masked (zero loss, no learning) or no token
    masked (loss over the whole prompt). Catch both before training, not after.

    Masking is deterministic given the chat template, so example 0 is
    representative of the whole split.
    """
    labels = train_ds[0]["labels"]
    n = len(labels)
    n_supervised = sum(1 for x in labels if x != -100)
    if n_supervised == 0:
        raise RuntimeError(
            "Completion masking left no supervised tokens (all -100). The "
            "response_part marker likely does not match the chat template — "
            "check instruction_part/response_part in the model config."
        )
    if n_supervised == n:
        raise RuntimeError(
            "Completion masking left every token supervised — the prompt was "
            "not masked. The instruction_part marker likely does not match the "
            "chat template — check instruction_part/response_part in the model config."
        )
    echo(f"Completion masking verified: {n_supervised}/{n} tokens supervised "
         f"({100 * n_supervised / n:.0f}%) in example 0")


def _detect_loss_spike(loss: float, recent: list[float], threshold: float = 3.0) -> Optional[dict]:
    """Flag a loss value as a spike when > `threshold` × the mean of the last 5
    recorded losses. Returns the anomaly dict or None; needs 5 priors for a
    baseline (None before that) and ignores degenerate mean5 <= 0."""
    if len(recent) < 5:
        return None
    mean5 = sum(recent[-5:]) / 5
    if mean5 <= 0 or loss <= threshold * mean5:
        return None
    return {"type": "spike", "value": round(loss, 4), "mean5": round(mean5, 4)}


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


class _CheckpointCallback(TrainerCallback):
    """Trainer callback for progress logging, GPU-util sampling, anomaly
    detection (NaN/Inf, loss spikes), and periodic train_state.json saves.

    Module-level so its helpers (_detect_loss_spike) can be unit-tested
    independently of the heavy Unsloth/TRL imports the rest of run_training_task
    pulls in. The constructor takes the per-run identifiers it needs to write
    train_state.json on each checkpoint save, including the input_hash so the
    next run can detect stale checkpoints (see _resume_decision)."""

    def __init__(self, echo, model_short: str, task_id: str, input_hash: str, smoke: bool = False):
        self._echo = echo
        self._model_short = model_short
        self._task_id = task_id
        self._input_hash = input_hash
        self._smoke = smoke
        self.gpu_util_samples: list[float] = []
        self.loss_steps: list[tuple[int, float]] = []
        self.anomalies: list[dict] = []
        self._train_start: float = 0.0
        self._total_steps: int = 0
        self._last_heartbeat: float = 0.0
        self._last_logged_step: int = -1

    def on_train_begin(self, args, state, control, **kwargs):
        now = time.time()
        self._train_start = now
        self._last_heartbeat = now
        self._total_steps = state.max_steps
        self._echo(f"Training started: {state.max_steps} steps")

    def _eta_str(self, step: int) -> str:
        if step <= 0 or self._total_steps <= 0:
            return ""
        elapsed = time.time() - self._train_start
        secs_per_step = elapsed / step
        remaining = (self._total_steps - step) * secs_per_step
        if remaining < 60:
            return f"ETA ~{int(remaining)}s"
        return f"ETA ~{int(remaining / 60)}m"

    def _progress_header(self, step: int) -> list[str]:
        pct = int(100 * step / self._total_steps) if self._total_steps else 0
        parts = [f"step {step}/{self._total_steps} ({pct}%)"]
        eta = self._eta_str(step)
        if eta:
            parts.append(eta)
        return parts

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        now = time.time()
        if now - self._last_heartbeat >= 60 and step != self._last_logged_step:
            elapsed_min = (now - self._train_start) / 60
            parts = self._progress_header(step) + [f"elapsed {elapsed_min:.1f}m"]
            self._echo("... " + " | ".join(parts))
            self._last_heartbeat = now

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            self.gpu_util_samples.append(torch.cuda.utilization())
        except Exception:
            pass
        if not logs:
            return
        step = state.global_step
        self._last_logged_step = step
        self._last_heartbeat = time.time()
        # Only per-step training logs (key "loss") count as step losses.
        # The end-of-training summary log carries "train_loss" (the mean
        # over all steps) — counting that as a step loss trips a phantom
        # spike anomaly and skews the loss diagnostics, so ignore it here.
        loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        lr = logs.get("learning_rate")
        grad_norm = logs.get("grad_norm")

        # ── Gradient norm ─────────────────────────────────────────────
        if grad_norm is not None and (math.isnan(grad_norm) or math.isinf(grad_norm)):
            self._echo(f"FATAL: NaN/Inf grad_norm at step {step} — halting training")
            self.anomalies.append({"step": step, "type": "nan_grad_norm"})
            control.should_training_stop = True

        # ── Loss anomalies ────────────────────────────────────────────
        if loss is not None:
            if math.isnan(loss) or math.isinf(loss):
                self._echo(f"FATAL: NaN/Inf loss at step {step} — halting training")
                self.anomalies.append({"step": step, "type": "nan_loss"})
                control.should_training_stop = True
            else:
                spike = _detect_loss_spike(loss, [v for _, v in self.loss_steps[-5:]])
                if spike is not None:
                    self._echo(f"WARNING: loss spike at step {step}: "
                               f"{spike['value']:.4f} (5-step mean={spike['mean5']:.4f})")
                    self.anomalies.append({"step": step, **spike})
                self.loss_steps.append((step, loss))

        # ── Progress line ─────────────────────────────────────────────
        parts = self._progress_header(step)
        if loss is not None and not (math.isnan(loss) or math.isinf(loss)):
            parts.append(f"loss={loss:.4f}")
        if eval_loss is not None:
            parts.append(f"eval_loss={eval_loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        if grad_norm is not None and not (math.isnan(grad_norm) or math.isinf(grad_norm)):
            parts.append(f"grad={grad_norm:.3f}")
        self._echo(" | ".join(parts))

    def on_save(self, args, state, control, **kwargs):
        save_train_state(self._model_short, self._task_id, CONDITION, {
            "status": "in_progress",
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "input_hash": self._input_hash,
        }, smoke=self._smoke)


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
    ctx: str = "",
) -> dict:
    """Load model, train, save adapter. Returns metadata dict.

    All heavy objects (model, trainer) are deleted before returning so the GC
    can reclaim memory before the next task starts.
    """
    echo = lambda msg: _echo(ctx, msg)
    # Heavy GPU libs imported here so the module is importable on CPU for tests
    import unsloth  # noqa: F401 — must come before transformers/peft
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    import datasets as hf_datasets
    from trl import SFTTrainer, SFTConfig

    task_id = task_cfg.task_id
    model_id = model_cfg.model_id
    substituted = False

    echo(f"Loading {model_id} on {hw_cfg.device.upper()}...")
    echo(f"Config: load_in_4bit={hw_cfg.load_in_4bit} dtype={hw_cfg.load_dtype} grad_ckpt={hw_cfg.use_grad_ckpt}")
    echo(f"SFT overrides: {hw_cfg.sft_extra or '(none)'}")

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
            echo(f"WARNING: {model_id} failed ({exc}). Falling back to {model_cfg.fallback_model_id}")
            model_id = model_cfg.fallback_model_id
            substituted = True
            model, tokenizer = _load_model(model_id)
        else:
            raise

    echo("Applying LoRA adapters...")
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

    load_dtype_str = _dtype_str(hw_cfg.load_dtype)
    compute_dtype_str = _compute_dtype_str(model)
    weight_dtype_str = "4bit" if hw_cfg.load_in_4bit else load_dtype_str
    echo(f"Precision: load_dtype={load_dtype_str}, compute_dtype={compute_dtype_str}, weight_dtype={weight_dtype_str}")

    echo(f"Loading dataset from {data_path} ...")
    train_rows = load_jsonl(data_path)
    echo(f"{len(train_rows)} training examples loaded")

    # Validation split: a prepared, versioned, stratified artifact — val.jsonl
    # alongside train.jsonl (see prepare_datasets.py). Used for per-epoch eval
    # and overfitting detection. Smoke runs skip eval, so they skip the val load.
    val_rows: list[dict] = []
    val_path = data_path.parent / "val.jsonl"
    if not smoke_test and val_path.exists():
        val_rows = load_jsonl(val_path)
        echo(f"Validation split: {len(train_rows)} train + {len(val_rows)} val ({val_path.name})")
    val_n = len(val_rows)

    train_ds = hf_datasets.Dataset.from_list(train_rows)
    val_ds = hf_datasets.Dataset.from_list(val_rows) if val_rows else None

    # Render each conversation to a single `text` field with the model's chat
    # template. SFTTrainer tokenizes it; train_on_responses_only (below) then
    # masks prompt tokens so loss is computed on assistant tokens only.
    template_kwargs = {}
    if model_cfg.enable_thinking is False:
        template_kwargs["enable_thinking"] = False

    echo("Applying chat template...")
    def apply_template(example):
        msgs = example["messages"]
        if not any(m["role"] == "assistant" for m in msgs):
            raise ValueError(f"Training row has no assistant message: {msgs}")
        return {"text": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, **template_kwargs,
        )}

    train_ds = train_ds.map(apply_template, remove_columns=train_ds.column_names)
    if val_ds is not None:
        val_ds = val_ds.map(apply_template, remove_columns=val_ds.column_names)
    echo(f"Dataset ready: {len(train_ds)} train rows" + (f", {len(val_ds)} val rows" if val_ds else ""))

    training_cfg = dict(model_cfg.training)
    if task_cfg.training_overrides:
        training_cfg.update(task_cfg.training_overrides)
        echo(f"Task overrides applied: {task_cfg.training_overrides}")
    # Evaluation cadence + early stopping. A held-out val split makes the epoch
    # count empirical: evaluate several times per epoch so the val optimum is
    # locatable, and let EarlyStoppingCallback (added below) end the run once
    # val loss stops improving — num_train_epochs is then only a cap. Smoke
    # runs have no val split, so eval and early stopping are disabled.
    has_val = val_ds is not None
    eval_strategy = training_cfg.get("eval_strategy", "steps") if has_val else "no"
    do_eval = eval_strategy != "no"
    eval_steps = None
    if do_eval:
        eff_batch = (training_cfg.get("per_device_train_batch_size", 4)
                     * training_cfg.get("gradient_accumulation_steps", 4))
        eval_steps = eval_save_steps(n_train, eff_batch,
                                     training_cfg.get("evals_per_epoch", 3))

    sft_kwargs = dict(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        optim=training_cfg.get("optim", "adamw_8bit"),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        bf16=training_cfg.get("bf16", True),
        seed=training_cfg.get("seed", 42),
        save_strategy=eval_strategy if do_eval else training_cfg.get("save_strategy", "epoch"),
        eval_strategy=eval_strategy,
        load_best_model_at_end=do_eval and training_cfg.get("load_best_model_at_end", False),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_cfg.get("greater_is_better", False),
        logging_steps=training_cfg.get("logging_steps", 10),
        logging_nan_inf_filter=training_cfg.get("logging_nan_inf_filter", False),
        report_to=training_cfg.get("report_to", "none"),
        max_seq_length=hw_cfg.seq_len,
        dataset_text_field="text",
        packing=False,
    )
    if do_eval:
        # Save in lockstep with eval so load_best_model_at_end can restore the
        # lowest-eval-loss checkpoint (HF requires save_steps % eval_steps == 0).
        sft_kwargs["eval_steps"] = eval_steps
        sft_kwargs["save_steps"] = eval_steps
    sft_kwargs.update(hw_cfg.sft_extra)
    # Pass warmup_ratio directly so Trainer computes steps from the real num_training_steps.
    # Fall back to an explicit warmup_steps only when ratio is absent.
    warmup_ratio = training_cfg.get("warmup_ratio")
    if warmup_ratio is not None:
        sft_kwargs["warmup_ratio"] = warmup_ratio
    else:
        sft_kwargs["warmup_steps"] = training_cfg.get("warmup_steps", 50)
    warmup_disp = (f"warmup_ratio={sft_kwargs['warmup_ratio']}"
                   if "warmup_ratio" in sft_kwargs
                   else f"warmup_steps={sft_kwargs.get('warmup_steps', 0)}")
    eval_disp = (f" eval_every={eval_steps}st patience={training_cfg.get('early_stopping_patience', 3)}"
                 if do_eval else " eval=off")
    echo(
        f"SFTConfig: max_epochs={sft_kwargs['num_train_epochs']}"
        f" batch={sft_kwargs['per_device_train_batch_size']}"
        f" accum={sft_kwargs['gradient_accumulation_steps']}"
        f" lr={sft_kwargs['learning_rate']:.2e}"
        f" sched={sft_kwargs['lr_scheduler_type']}"
        f" {warmup_disp}{eval_disp}"
    )
    sft_config = SFTConfig(**sft_kwargs)

    echo("Building trainer...")
    callback = _CheckpointCallback(echo, model_cfg.model_short, task_id, input_hash, smoke=smoke_test)
    callbacks = [callback]
    if do_eval:
        # transformers' built-in early stopping: end the run once eval_loss has
        # not improved for `patience` consecutive evals. patience > 0 so a
        # single noisy uptick does not stop training prematurely.
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=training_cfg.get("early_stopping_patience", 3),
            early_stopping_threshold=training_cfg.get("early_stopping_threshold", 0.0),
        ))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
        callbacks=callbacks,
    )

    # Completion-only loss: mask prompt tokens to -100 so gradients flow from
    # assistant tokens only. Unsloth's patched SFTTrainer ignores TRL's
    # `completion_only_loss` flag (its `_prepare_dataset` has no prompt/completion
    # path), so masking is applied here, after tokenization, via the chat
    # template's turn markers — and masks the eval split the same way.
    echo(f"Masking prompts (loss on response only): "
         f"instruction={model_cfg.instruction_part!r} response={model_cfg.response_part!r}")
    trainer = train_on_responses_only(
        trainer,
        instruction_part=model_cfg.instruction_part,
        response_part=model_cfg.response_part,
    )
    _verify_completion_masking(trainer.train_dataset, echo)

    echo("Starting trainer.train()...")
    t0 = time.time()
    result = trainer.train(
        resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None
    )
    elapsed_min = (time.time() - t0) / 60
    # num_train_epochs is a cap: record whether early stopping ended the run
    # short of it, and how many epochs actually ran. isinstance guards keep the
    # mocked-trainer unit tests JSON-serializable.
    _ms = getattr(trainer.state, "max_steps", 0)
    _gs = getattr(trainer.state, "global_step", 0)
    _ep = getattr(trainer.state, "epoch", 0.0)
    max_steps = int(_ms) if isinstance(_ms, (int, float)) else 0
    global_step = int(_gs) if isinstance(_gs, (int, float)) else 0
    epochs_completed = round(float(_ep), 2) if isinstance(_ep, (int, float)) else 0.0
    early_stopped = bool(do_eval and max_steps and global_step < max_steps)
    if early_stopped:
        echo(f"Early stopping fired: {epochs_completed}/{epochs} epochs run "
             f"({global_step}/{max_steps} steps)")
    gpu_hours = round(elapsed_min / 60, 4)
    training_cost = 0.0 if smoke_test else gpu_hours * GPU_HOURLY
    peak_gpu_mem_mb = round(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
    gpu_model = get_current_gpu_name()
    avg_gpu_util_pct = round(sum(callback.gpu_util_samples) / len(callback.gpu_util_samples)) if callback.gpu_util_samples else None

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    nv_adapter_dir = ckpt_dir / "final_adapter"
    shutil.copytree(str(adapter_dir), str(nv_adapter_dir), dirs_exist_ok=True)

    m = result.metrics or {}
    eval_loss  = m.get("eval_loss")
    train_loss = round(m["train_loss"], 4) if "train_loss" in m else None

    loss_history = [
        {"step": e.get("step", 0), "loss": round(e["loss"], 6), "lr": e.get("learning_rate")}
        for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e
    ]
    eval_loss_history = [
        {"step": e.get("step", 0), "epoch": round(e.get("epoch", 0), 2),
         "eval_loss": round(e["eval_loss"], 6)}
        for e in trainer.state.log_history if "eval_loss" in e
    ]
    val_losses = [e["eval_loss"] for e in eval_loss_history]
    # trainer.train() returns training metrics only — the eval loss lives in the
    # log history. Report the saved model's eval loss: trainer.state.best_metric
    # when load_best_model_at_end reloaded the best epoch, else the final epoch.
    if eval_loss is None and val_losses:
        best = getattr(trainer.state, "best_metric", None)
        eval_loss = round(best if best is not None else val_losses[-1], 6)
    training_diagnostics = _analyze_training(
        [v for _, v in callback.loss_steps], callback.anomalies, echo, val_losses=val_losses or None
    )
    hyperparams = {
        "lora_rank": hw_cfg.lora_rank,
        "lora_alpha": hw_cfg.lora_alpha,
        **{k: sft_kwargs[k] for k in (
            "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
            "lr_scheduler_type", "weight_decay", "optim",
        )},
        **({"warmup_ratio": sft_kwargs["warmup_ratio"]} if "warmup_ratio" in sft_kwargs
           else {"warmup_steps": sft_kwargs.get("warmup_steps", 0)}),
        **({"eval_steps": eval_steps,
            "early_stopping_patience": training_cfg.get("early_stopping_patience", 3)}
           if do_eval else {}),
    }

    meta = {
        "model_id": model_cfg.model_short,
        "task_id": task_id,
        "condition": CONDITION,
        "train_data_path": str(data_path),
        "n_train": n_train,
        "n_val": val_n,
        "epochs": epochs,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "seq_len": hw_cfg.seq_len,
        "load_dtype": load_dtype_str,
        "compute_dtype": compute_dtype_str,
        "weight_dtype": weight_dtype_str,
        "training_cost": round(training_cost, 4),
        "training_time_min": round(elapsed_min, 1),
        "gpu_hours": gpu_hours,
        "gpu_model": gpu_model,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "avg_gpu_util_pct": avg_gpu_util_pct,
        "eval_loss": eval_loss,
        "train_loss": train_loss,
        "model_used": model_id,
        "substituted": substituted,
        "git_sha": _git_sha(),
        "input_hash": input_hash,
        "loss_history": loss_history,
        "eval_loss_history": eval_loss_history,
        "hyperparams": hyperparams,
        "training_diagnostics": training_diagnostics,
    }
    atomic_write_json(meta, log_dir / "metadata.json")
    atomic_write_json(meta, ckpt_dir / "metadata.json")
    save_train_state(model_cfg.model_short, task_id, CONDITION, {
        "status": "complete",
        "eval_loss": eval_loss,
        "train_loss": train_loss,
        "training_time_min": round(elapsed_min, 1),
        "training_cost": round(training_cost, 4),
        "input_hash": input_hash,
    }, smoke=smoke_test)

    loss_display = eval_loss if eval_loss is not None else train_loss
    mem_str = f", peak_mem={peak_gpu_mem_mb}MB" if peak_gpu_mem_mb is not None else ""
    util_str = f", gpu_util={avg_gpu_util_pct}%" if avg_gpu_util_pct is not None else ""
    echo(f"Done: {elapsed_min:.1f} min ({gpu_hours:.4f} GPU-h), ${training_cost:.3f}, loss={loss_display}{mem_str}{util_str}")

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
    echo(f"Memory released: {rss_before - rss_after} MB freed (RSS {rss_before}→{rss_after} MB).")

    return meta


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

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


def train_one(
    model_cfg: ModelConfig,
    task_cfg: TaskConfig,
    data_path: Path,
    dry_run: bool,
    auto_upload: bool = False,
    smoke_test: bool = False,
    ctx: str = "",
) -> dict:
    """Orchestrate a single model/task run. Returns metadata dict."""
    echo = lambda msg: _echo(ctx, msg)
    task_id = task_cfg.task_id
    n_train = count_jsonl(data_path)
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
            ctx=ctx,
        )

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
