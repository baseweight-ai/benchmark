"""HuggingFace TrainerCallback adapters shared across Trainer backends.

Both Unsloth (via TRL's `SFTTrainer`) and a pure HF/peft setup use the HF
`TrainerCallback` API, so progress logging, GPU-util sampling, anomaly
detection, and `train_state.json` checkpointing live here — reused, not
duplicated per backend.

Heavy imports (`torch`, `transformers`) are deferred to inside the class so
this module stays CPU-importable for tests.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import time
from typing import Callable, Optional

# transformers.TrainerCallback is the only base class. It's CPU-safe to import
# at module load (no CUDA), so import here to make the callback subclassable.
from transformers import TrainerCallback

from checkpoint_utils import save_train_state


# Resolved once per process. None when nvidia-smi can't be located.
_NVIDIA_SMI = shutil.which("nvidia-smi")


def _sample_gpu_util() -> Optional[float]:
    """Best-effort GPU utilisation percentage (0-100), or None.

    Two paths, in order:
      1. `torch.cuda.utilization()` — fast but requires the pynvml package,
         which isn't a hard dep on this project. Returns None if missing.
      2. `nvidia-smi --query-gpu=utilization.gpu` — works wherever the NVIDIA
         driver is installed (always true on a CUDA host), at ~50ms per call.
         Cheap enough at logging cadence (every `logging_steps` steps).
    Either failure path silently returns None — sampling is best-effort
    observability and must never interrupt training.
    """
    try:
        import torch
        return float(torch.cuda.utilization())
    except Exception:
        pass
    if not _NVIDIA_SMI:
        return None
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode != 0:
            return None
        return float(out.stdout.splitlines()[0].strip())
    except Exception:
        return None


def detect_loss_spike(loss: float, recent: list[float], threshold: float = 3.0) -> Optional[dict]:
    """Flag a loss value as a spike when > `threshold` × the mean of the last 5
    recorded losses.

    Returns the anomaly dict or None. Needs 5 priors for a baseline (returns
    None before that) and ignores a degenerate mean5 <= 0.
    """
    if len(recent) < 5:
        return None
    mean5 = sum(recent[-5:]) / 5
    if mean5 <= 0 or loss <= threshold * mean5:
        return None
    return {"type": "spike", "value": round(loss, 4), "mean5": round(mean5, 4)}


class CheckpointCallback(TrainerCallback):
    """Per-step progress logging, GPU-util sampling, anomaly detection
    (NaN/Inf, loss spikes), and periodic `train_state.json` saves.

    Created once per run by the orchestrator (train_local.train_one) and
    passed through `TrainSpec.callbacks`. The Trainer adapter forwards the
    list to its HF Trainer subclass; after training, the orchestrator reads
    `gpu_util_samples`, `loss_steps`, and `anomalies` from this instance to
    populate training diagnostics.
    """

    def __init__(
        self,
        echo: Callable[[str], None],
        model_short: str,
        task_id: str,
        condition: str,
        input_hash: str,
        smoke: bool = False,
    ) -> None:
        self._echo = echo
        self._model_short = model_short
        self._task_id = task_id
        self._condition = condition
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
        util = _sample_gpu_util()
        if util is not None:
            self.gpu_util_samples.append(util)
        if not logs:
            return
        step = state.global_step
        self._last_logged_step = step
        self._last_heartbeat = time.time()
        # Only per-step training logs (key "loss") feed the diagnostics. The
        # end-of-training summary log uses key "train_loss" — it's the mean
        # over all steps, not a step value, so treating it as one would skew
        # loss-curve stats.
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
                spike = detect_loss_spike(loss, [v for _, v in self.loss_steps[-5:]])
                if spike is not None:
                    self._echo(
                        f"WARNING: loss spike at step {step}: "
                        f"{spike['value']:.4f} (5-step mean={spike['mean5']:.4f})"
                    )
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
        save_train_state(
            self._model_short,
            self._task_id,
            self._condition,
            {
                "status": "in_progress",
                "epoch": state.epoch,
                "global_step": state.global_step,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
                "input_hash": self._input_hash,
            },
            smoke=self._smoke,
        )
