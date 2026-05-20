"""QLoRA SFT via Unsloth + TRL.

This is the fast path for the benchmark — Unsloth's `FastModel` patches
attention/MLP layers and ships a chat-template-aware `train_on_responses_only`
masker that the TRL `SFTTrainer` doesn't support natively (Unsloth's patched
`SFTTrainer._prepare_dataset` has no prompt/completion path, so TRL's
`completion_only_loss` flag is silently ignored). The adapter therefore:

  1. Loads the model with Unsloth's `FastModel.from_pretrained` (4-bit QLoRA
     via bitsandbytes when configured),
  2. Wraps it in a LoRA adapter via `FastModel.get_peft_model`,
  3. Applies the model's chat template to the prepared chat-JSONL data,
  4. Builds a TRL `SFTTrainer` + `SFTConfig` (this part is plain TRL/HF),
  5. Applies Unsloth's `train_on_responses_only` for completion-only loss,
  6. Runs the trainer, saves the LoRA adapter, releases GPU memory.

A pure HF/peft + TRL backend would do (1)–(2) with `AutoModelForCausalLM` +
`peft.get_peft_model`, and (5) with TRL's `DataCollatorForCompletionOnlyLM`.
That lives in `hf_lora.py` (stub).
"""
from __future__ import annotations

import ctypes
import gc
import math
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from pipeline.trainers.base import (
    TrainResult,
    TrainSpec,
    Trainer,
    analyze_training,
    eval_save_steps,
    register_trainer,
    verify_completion_masking,
)
from pipeline.trainers.callbacks import CheckpointCallback
from utils import load_jsonl

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


def _dtype_str(dtype: Any) -> str:
    """Convert a torch dtype to a canonical string like 'bfloat16' or 'float32'."""
    if dtype is None:
        return "auto"
    return str(dtype).split(".")[-1]


def _compute_dtype_str(model: Any) -> str:
    """Infer the model's effective compute dtype.

    For bitsandbytes QLoRA, the authoritative source is `quantization_config` —
    parameter dtypes reflect storage format (4-bit/8-bit), not compute format.
    Falls back to the first floating-point parameter's dtype for non-quantized
    models.
    """
    import torch
    qcfg = getattr(getattr(model, "config", None), "quantization_config", None)
    if qcfg is not None and hasattr(qcfg, "bnb_4bit_compute_dtype"):
        return _dtype_str(qcfg.bnb_4bit_compute_dtype)
    try:
        param = next(p for p in model.parameters() if p.dtype not in (torch.int8,))
        return _dtype_str(param.dtype)
    except StopIteration:
        return "unknown"


@register_trainer("unsloth-lora")
class UnslothLoRATrainer(Trainer):
    """LoRA SFT adapter backed by Unsloth's FastModel + TRL SFTTrainer."""

    name = "unsloth-lora"

    def train(self, spec: TrainSpec) -> TrainResult:
        # Deferred imports — keeps this module CPU-importable. Tests inject
        # mocks into sys.modules before train() is called.
        import torch
        import unsloth  # noqa: F401 — must come before transformers/peft
        from unsloth import FastModel
        from unsloth.chat_templates import train_on_responses_only
        import datasets as hf_datasets
        from trl import SFTTrainer, SFTConfig
        from transformers import EarlyStoppingCallback

        echo = spec.echo
        model_cfg = spec.model_cfg
        task_cfg = spec.task_cfg
        hw_cfg = spec.hw_cfg

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

        echo(f"Loading dataset from {spec.data_path} ...")
        train_rows = load_jsonl(spec.data_path)
        echo(f"{len(train_rows)} training examples loaded")

        val_rows: list[dict] = []
        if not spec.smoke_test and spec.val_path and spec.val_path.exists():
            val_rows = load_jsonl(spec.val_path)
            echo(f"Validation split: {len(train_rows)} train + {len(val_rows)} val ({spec.val_path.name})")
        val_n = len(val_rows)
        n_train = len(train_rows)

        train_ds = hf_datasets.Dataset.from_list(train_rows)
        val_ds = hf_datasets.Dataset.from_list(val_rows) if val_rows else None

        # Render each conversation to a single `text` field with the model's chat
        # template. SFTTrainer tokenizes it; train_on_responses_only (below) then
        # masks prompt tokens so loss is computed on assistant tokens only.
        template_kwargs: dict = {}
        if model_cfg.enable_thinking is False:
            template_kwargs["enable_thinking"] = False

        echo("Applying chat template...")

        def apply_template(example):
            msgs = example["messages"]
            if not any(m["role"] == "assistant" for m in msgs):
                raise ValueError(f"Training row has no assistant message: {msgs}")
            return {
                "text": tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False, **template_kwargs,
                )
            }

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
            eff_batch = (
                training_cfg.get("per_device_train_batch_size", 4)
                * training_cfg.get("gradient_accumulation_steps", 4)
            )
            eval_steps = eval_save_steps(n_train, eff_batch, training_cfg.get("evals_per_epoch", 3))

        sft_kwargs = dict(
            output_dir=str(spec.ckpt_dir),
            num_train_epochs=spec.epochs,
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
        # warmup_ratio if present; else explicit warmup_steps.
        warmup_ratio = training_cfg.get("warmup_ratio")
        if warmup_ratio is not None:
            sft_kwargs["warmup_ratio"] = warmup_ratio
        else:
            sft_kwargs["warmup_steps"] = training_cfg.get("warmup_steps", 50)
        warmup_disp = (
            f"warmup_ratio={sft_kwargs['warmup_ratio']}"
            if "warmup_ratio" in sft_kwargs
            else f"warmup_steps={sft_kwargs.get('warmup_steps', 0)}"
        )
        eval_disp = (
            f" eval_every={eval_steps}st patience={training_cfg.get('early_stopping_patience', 3)}"
            if do_eval else " eval=off"
        )
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
        callbacks = list(spec.callbacks)
        if do_eval:
            # HF's built-in early stopping: end once eval_loss has not improved
            # for `patience` consecutive evals.
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=training_cfg.get("early_stopping_patience", 3),
                    early_stopping_threshold=training_cfg.get("early_stopping_threshold", 0.0),
                )
            )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args=sft_config,
            callbacks=callbacks,
        )

        # Completion-only loss: mask prompt tokens to -100 so gradients flow from
        # assistant tokens only. Unsloth's patched SFTTrainer ignores TRL's
        # `completion_only_loss` flag (its `_prepare_dataset` has no prompt/completion
        # path), so masking is applied here, after tokenization, via the chat
        # template's turn markers — and masks the eval split the same way.
        echo(
            f"Masking prompts (loss on response only): "
            f"instruction={model_cfg.instruction_part!r} response={model_cfg.response_part!r}"
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part=model_cfg.instruction_part,
            response_part=model_cfg.response_part,
        )
        verify_completion_masking(trainer.train_dataset, echo)

        echo("Starting trainer.train()...")
        t0 = time.time()
        result = trainer.train(
            resume_from_checkpoint=str(spec.resume_ckpt) if spec.resume_ckpt else None
        )
        elapsed_min = (time.time() - t0) / 60

        # num_train_epochs is a cap: record whether early stopping ended the run
        # short of it. isinstance guards keep mocked-trainer tests JSON-serializable.
        _ms = getattr(trainer.state, "max_steps", 0)
        _gs = getattr(trainer.state, "global_step", 0)
        _ep = getattr(trainer.state, "epoch", 0.0)
        max_steps = int(_ms) if isinstance(_ms, (int, float)) else 0
        global_step = int(_gs) if isinstance(_gs, (int, float)) else 0
        epochs_completed = round(float(_ep), 2) if isinstance(_ep, (int, float)) else 0.0
        early_stopped = bool(do_eval and max_steps and global_step < max_steps)
        if early_stopped:
            echo(
                f"Early stopping fired: {epochs_completed}/{spec.epochs} epochs run "
                f"({global_step}/{max_steps} steps)"
            )
        gpu_hours = round(elapsed_min / 60, 4)
        peak_gpu_mem_mb = (
            round(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
        )
        gpu_model = _current_gpu_name()
        # Pull GPU util and anomalies off the orchestrator's CheckpointCallback.
        # It's framework-agnostic (HF TrainerCallback API), instantiated by the
        # caller, and threaded through TrainSpec — ducktyping on attributes
        # keeps adapters from needing to import the concrete class.
        metrics_cb = next((cb for cb in spec.callbacks if isinstance(cb, CheckpointCallback)), None)
        avg_gpu_util_pct: Optional[int] = None
        anomalies: list[dict] = []
        if metrics_cb is not None:
            samples = metrics_cb.gpu_util_samples
            avg_gpu_util_pct = round(sum(samples) / len(samples)) if samples else None
            anomalies = list(metrics_cb.anomalies)

        # Save adapter and mirror into the checkpoint dir so vLLM evals can
        # pick it up alongside HF Trainer's checkpoint-N artefacts.
        model.save_pretrained(str(spec.adapter_dir))
        tokenizer.save_pretrained(str(spec.adapter_dir))
        nv_adapter_dir = spec.ckpt_dir / "final_adapter"
        shutil.copytree(str(spec.adapter_dir), str(nv_adapter_dir), dirs_exist_ok=True)

        m = result.metrics or {}
        eval_loss = m.get("eval_loss")
        train_loss = round(m["train_loss"], 4) if "train_loss" in m else None

        loss_history = [
            {"step": e.get("step", 0), "loss": round(e["loss"], 6), "lr": e.get("learning_rate")}
            for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e
        ]
        eval_loss_history = [
            {
                "step": e.get("step", 0),
                "epoch": round(e.get("epoch", 0), 2),
                "eval_loss": round(e["eval_loss"], 6),
            }
            for e in trainer.state.log_history if "eval_loss" in e
        ]
        val_losses = [e["eval_loss"] for e in eval_loss_history]
        # trainer.train() returns training metrics only — the eval loss lives in the
        # log history. Report the saved model's eval loss: trainer.state.best_metric
        # when load_best_model_at_end reloaded the best epoch, else the final epoch.
        if eval_loss is None and val_losses:
            best = getattr(trainer.state, "best_metric", None)
            eval_loss = round(best if best is not None else val_losses[-1], 6)

        diagnostics = analyze_training(
            [v for _, v in (metrics_cb.loss_steps if metrics_cb else [])],
            anomalies,
            echo,
            val_losses=val_losses or None,
        )

        hyperparams = {
            "lora_rank": hw_cfg.lora_rank,
            "lora_alpha": hw_cfg.lora_alpha,
            **{
                k: sft_kwargs[k]
                for k in (
                    "learning_rate",
                    "per_device_train_batch_size",
                    "gradient_accumulation_steps",
                    "lr_scheduler_type",
                    "weight_decay",
                    "optim",
                )
            },
            **(
                {"warmup_ratio": sft_kwargs["warmup_ratio"]}
                if "warmup_ratio" in sft_kwargs
                else {"warmup_steps": sft_kwargs.get("warmup_steps", 0)}
            ),
            **(
                {
                    "eval_steps": eval_steps,
                    "early_stopping_patience": training_cfg.get("early_stopping_patience", 3),
                }
                if do_eval else {}
            ),
        }

        train_result = TrainResult(
            model_used=model_id,
            substituted=substituted,
            n_train=n_train,
            n_val=val_n,
            epochs_completed=epochs_completed,
            early_stopped=early_stopped,
            elapsed_min=round(elapsed_min, 1),
            gpu_hours=gpu_hours,
            peak_gpu_mem_mb=peak_gpu_mem_mb,
            avg_gpu_util_pct=avg_gpu_util_pct,
            gpu_model=gpu_model,
            train_loss=train_loss,
            eval_loss=eval_loss,
            load_dtype=load_dtype_str,
            compute_dtype=compute_dtype_str,
            weight_dtype=weight_dtype_str,
            loss_history=loss_history,
            eval_loss_history=eval_loss_history,
            hyperparams=hyperparams,
            training_diagnostics=diagnostics,
        )

        loss_display = eval_loss if eval_loss is not None else train_loss
        mem_str = f", peak_mem={peak_gpu_mem_mb}MB" if peak_gpu_mem_mb is not None else ""
        util_str = f", gpu_util={avg_gpu_util_pct}%" if avg_gpu_util_pct is not None else ""
        echo(f"Done: {elapsed_min:.1f} min ({gpu_hours:.4f} GPU-h), loss={loss_display}{mem_str}{util_str}")

        # Release GPU/CPU memory before returning so the next task starts clean.
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

        return train_result


def _current_gpu_name() -> Optional[str]:
    # Local import keeps this module CPU-cheap; pipeline.hardware avoids torch
    # at module load.
    try:
        from pipeline.hardware import get_current_gpu_name
        return get_current_gpu_name()
    except Exception:
        return None
