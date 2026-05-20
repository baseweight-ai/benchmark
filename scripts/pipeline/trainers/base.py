"""Trainer interface, dataclasses, and registry.

A `Trainer` adapter encapsulates the framework-specific work of fine-tuning a
model — model loading, PEFT wiring, completion masking, the training loop —
behind a uniform interface. The orchestrator (train_local.py) builds a
TrainSpec, calls trainer.train(spec), and lifts the returned TrainResult into
the metadata it writes to disk.

This module is framework-agnostic: it does not import torch, transformers,
trl, peft, unsloth, or axolotl. Concrete adapters in sibling modules pay that
cost lazily (inside `train`), so importing this module on CPU stays cheap.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

TRAINER_REGISTRY: dict[str, type["Trainer"]] = {}


def register_trainer(name: str) -> Callable[[type["Trainer"]], type["Trainer"]]:
    """Class decorator: register a Trainer adapter under a name.

    The name is the trainer identifier — backend + method combined, e.g.
    'unsloth-lora', 'hf-lora', 'axolotl-lora'. Backends register at import
    time (see pipeline.trainers.__init__) so `get_trainer(name)` works without
    callers wiring imports themselves.
    """
    def decorator(cls: type["Trainer"]) -> type["Trainer"]:
        if not hasattr(cls, "name") or not getattr(cls, "name"):
            cls.name = name
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator


def get_trainer(name: str) -> type["Trainer"]:
    """Look up a Trainer class by name. Caller instantiates."""
    if name not in TRAINER_REGISTRY:
        raise KeyError(
            f"Trainer {name!r} not registered. Available: {sorted(TRAINER_REGISTRY)}"
        )
    return TRAINER_REGISTRY[name]


@dataclass(frozen=True)
class TrainSpec:
    """Everything a Trainer adapter needs to run one fine-tune.

    The orchestrator builds this from model/task configs and computed paths
    (see train_local.train_one); the adapter consumes it and runs the
    framework-specific training loop. Framework-agnostic concerns
    (skip-if-unchanged, retrain decisions, metadata writing, input hashing)
    stay in the orchestrator and never enter this object.

    `callbacks` is a list of HuggingFace `TrainerCallback` instances. Adapters
    targeting an HF Trainer subclass (TRL's SFTTrainer, Axolotl, vanilla HF)
    pass them through; adapters with a different control flow may translate
    or ignore them.
    """
    model_cfg: Any                    # train_local.ModelConfig
    task_cfg: Any                     # train_local.TaskConfig
    hw_cfg: Any                       # train_local.HardwareConfig
    data_path: Path                   # train.jsonl
    val_path: Optional[Path]          # val.jsonl, or None if no validation split
    epochs: int                       # cap; early stopping may end the run sooner
    resume_ckpt: Optional[Path]       # for crash recovery within a run
    ckpt_dir: Path                    # HF Trainer output_dir (checkpoint-N/)
    adapter_dir: Path                 # final destination for the saved adapter
    smoke_test: bool
    callbacks: list = field(default_factory=list)
    echo: Callable[[str], None] = field(default=lambda msg: None)


@dataclass
class TrainResult:
    """What a Trainer adapter returns after training.

    Framework-agnostic; the orchestrator merges this with run-level metadata
    (input_hash, git_sha, training_cost from gpu_hours, paths) before writing
    metadata.json. Fields default to None when the adapter has no useful
    value (e.g. peak GPU memory on CPU-only runs).
    """
    model_used: str
    substituted: bool
    n_train: int
    n_val: int
    epochs_completed: float
    early_stopped: bool
    elapsed_min: float
    gpu_hours: float
    peak_gpu_mem_mb: Optional[int] = None
    avg_gpu_util_pct: Optional[int] = None
    gpu_model: Optional[str] = None
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    load_dtype: str = "unknown"
    compute_dtype: str = "unknown"
    weight_dtype: str = "unknown"
    loss_history: list[dict] = field(default_factory=list)
    eval_loss_history: list[dict] = field(default_factory=list)
    hyperparams: dict = field(default_factory=dict)
    training_diagnostics: dict = field(default_factory=dict)


class Trainer(ABC):
    """Adapter for one (backend, method) combination.

    Concrete adapters live in sibling modules:
      - UnslothLoRATrainer       (unsloth_lora.py) — FastModel + train_on_responses_only
      - HFLoRATrainer            (hf_lora.py)      — peft + TRL DataCollator (stub)
      - AxolotlLoRATrainer       (axolotl_lora.py) — to be added; subprocess Axolotl CLI

    Subclasses set `name` (e.g. 'unsloth-lora') and implement `train(spec)`.
    Use the `@register_trainer(...)` decorator to make the adapter discoverable.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def train(self, spec: TrainSpec) -> TrainResult:
        """Run one fine-tune end-to-end. Save the adapter to `spec.adapter_dir`.

        Heavy framework imports must be deferred to inside this method so
        importing the adapter module stays CPU-cheap (important for tests).
        """
        ...


# ── Framework-agnostic utilities reused by adapters ────────────────────────

def verify_completion_masking(train_ds: Any, echo: Callable[[str], None]) -> None:
    """Fail loudly if completion-only loss masking degenerated.

    A masking step (whether Unsloth's `train_on_responses_only`, TRL's
    `DataCollatorForCompletionOnlyLM`, or Axolotl's `train_on_inputs: false`)
    must leave SOME tokens supervised and at least one token masked. If the
    configured instruction/response markers don't match the model's chat
    template, masking silently degenerates into one of two ruined runs —
    every token masked (zero loss, no learning) or no token masked (loss
    over the whole prompt). Catch both before training, not after.

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
    echo(
        f"Completion masking verified: {n_supervised}/{n} tokens supervised "
        f"({100 * n_supervised / n:.0f}%) in example 0"
    )


def analyze_training(
    losses: list[float],
    anomalies: list[dict],
    echo: Callable[[str], None],
    val_losses: Optional[list[float]] = None,
) -> dict:
    """Post-training diagnostics: convergence, plateau, divergence, overfitting.

    Framework-agnostic — operates on a sequence of training loss values and an
    optional sequence of validation losses. Both Unsloth and HF adapters
    populate these the same way (from `trainer.state.log_history`).

    val_losses: per-epoch validation losses from an in-training eval split,
    used to detect overfitting (train loss ↓ while val loss ↑). When None,
    overfitting cannot be assessed and `overfitting_detected` is None.
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
        echo(
            f"WARNING: loss improved only {improvement * 100:.1f}% — model may not have trained "
            f"meaningfully (start={first:.4f}, end={last:.4f})"
        )

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

    # Overfitting: after hitting its best (minimum), validation loss drifted
    # back up by >5% while training loss kept falling.
    #
    # The frame is "what happened AFTER the best", not "last vs min". Two
    # consequences:
    #   - If the best is the LAST eval, the model is still improving (or just
    #     plateau'd at best); flagging that as overfit would be wrong.
    #   - The post-best tail is compared as a MEDIAN to suppress single-step
    #     noise — e.g. [0.04, 0.04, 0.04, 0.09, 0.04] is one spike that
    #     recovered, not sustained drift; median(last 3) = 0.04, no flag.
    # Only assessable when a validation split was reserved (val_losses given).
    if val_losses and len(val_losses) >= 2:
        best_val = min(val_losses)
        best_idx = val_losses.index(best_val)
        if best_idx >= len(val_losses) - 1:
            # Best is the most recent eval — no drift, no overfit.
            diag["overfitting_detected"] = False
        else:
            post_best = val_losses[best_idx + 1:]
            # Smoothing window: median over the last 3 post-best evals (or
            # all post-best evals if fewer than 3 exist).
            window = post_best[-min(3, len(post_best)):]
            smoothed = sorted(window)[len(window) // 2] if len(window) % 2 == 1 \
                else (sorted(window)[len(window) // 2 - 1] + sorted(window)[len(window) // 2]) / 2
            if smoothed > best_val * 1.05 and improvement > 0:
                diag["overfitting_detected"] = True
                echo(
                    f"WARNING: overfitting — val_loss bottomed at {best_val:.4f} at "
                    f"eval {best_idx + 1}/{len(val_losses)}, post-best median rose to "
                    f"{smoothed:.4f} (+{(smoothed / best_val - 1) * 100:.1f}%) "
                    f"while train_loss fell {improvement * 100:.1f}%"
                )
            else:
                diag["overfitting_detected"] = False
    elif val_losses is not None:
        # val_losses provided but too few epochs to assess
        diag["overfitting_detected"] = None

    return diag


def eval_save_steps(n_train: int, eff_batch: int, evals_per_epoch: int) -> int:
    """Steps between evaluations: a fixed number of evals per epoch, so the
    validation optimum stays locatable for early stopping at any dataset size.

    Framework-agnostic — uses the same arithmetic for any HF Trainer subclass
    (TRL SFTTrainer, vanilla HF Trainer, Axolotl-wrapped Trainer).
    """
    import math
    steps_per_epoch = math.ceil(n_train / max(1, eff_batch))
    return max(1, steps_per_epoch // max(1, evals_per_epoch))
