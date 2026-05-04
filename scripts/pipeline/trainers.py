"""Trainer interface and registry for post-training methods (SFT, LoRA, DPO, etc.)."""
from __future__ import annotations
from pathlib import Path
from typing import Protocol, runtime_checkable

TRAINER_REGISTRY: dict[str, type] = {}


def register_trainer(method: str):
    """Class decorator that registers a Trainer implementation under a method name."""
    def decorator(cls):
        TRAINER_REGISTRY[method] = cls
        return cls
    return decorator


def get_trainer(method: str) -> type:
    if method not in TRAINER_REGISTRY:
        raise KeyError(f"Trainer '{method}' not registered. Available: {list(TRAINER_REGISTRY)}")
    return TRAINER_REGISTRY[method]


@runtime_checkable
class Trainer(Protocol):
    method: str

    def train(
        self,
        model_id: str,
        task_id: str,
        data_path: Path,
        dry_run: bool,
        smoke_test: bool,
    ) -> dict:
        """Run training and return a metadata dict."""
        ...
