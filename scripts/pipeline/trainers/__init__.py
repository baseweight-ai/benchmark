"""Trainer interface + backend registry.

Importing this package triggers backend registration (via the
`@register_trainer` decorator on each adapter class), so callers can rely on
the registry being populated:

    from pipeline.trainers import get_trainer
    trainer = get_trainer("unsloth-lora")()
    result = trainer.train(spec)

The public surface is intentionally small: the registry helpers, the
dataclasses, the abstract base class, and two framework-agnostic utilities
that any adapter is expected to call.
"""
from pipeline.trainers.base import (
    TRAINER_REGISTRY,
    TrainResult,
    TrainSpec,
    Trainer,
    analyze_training,
    eval_save_steps,
    get_trainer,
    register_trainer,
    verify_completion_masking,
)
from pipeline.trainers.callbacks import CheckpointCallback, detect_loss_spike

# Side-effect imports — register the adapters under their names.
from pipeline.trainers import unsloth_lora  # noqa: F401
from pipeline.trainers import hf_lora       # noqa: F401

__all__ = [
    "TRAINER_REGISTRY",
    "TrainResult",
    "TrainSpec",
    "Trainer",
    "CheckpointCallback",
    "analyze_training",
    "detect_loss_spike",
    "eval_save_steps",
    "get_trainer",
    "register_trainer",
    "verify_completion_masking",
]
