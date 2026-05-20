"""Unit tests for pipeline.trainers registry and dataclasses.

Covers the framework-agnostic seam: the Trainer ABC, the registry, the
shared dataclasses, and the HF stub's contract. The Unsloth adapter is
exercised through train_local end-to-end in test_training_state.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_populated_on_import():
    """Importing the package must register the bundled backends — callers
    rely on `get_trainer(name)` working without wiring imports themselves."""
    from pipeline.trainers import TRAINER_REGISTRY
    assert "unsloth-lora" in TRAINER_REGISTRY
    assert "hf-lora" in TRAINER_REGISTRY


def test_get_trainer_returns_class():
    from pipeline.trainers import get_trainer
    from pipeline.trainers.unsloth_lora import UnslothLoRATrainer
    assert get_trainer("unsloth-lora") is UnslothLoRATrainer


def test_get_trainer_unknown_raises():
    from pipeline.trainers import get_trainer
    with pytest.raises(KeyError, match="not registered"):
        get_trainer("does-not-exist")


def test_register_trainer_decorator_adds_to_registry():
    """Round-trip: a new decorator-registered class is reachable via get_trainer."""
    from pipeline.trainers import TRAINER_REGISTRY, Trainer, get_trainer, register_trainer

    name = "test-only-fake"

    @register_trainer(name)
    class _FakeTrainer(Trainer):
        def train(self, spec):  # pragma: no cover — never called
            raise NotImplementedError

    try:
        assert name in TRAINER_REGISTRY
        assert get_trainer(name) is _FakeTrainer
        # name attribute is set by the decorator when not pre-defined
        assert _FakeTrainer.name == name
    finally:
        TRAINER_REGISTRY.pop(name, None)


def test_trainer_subclasses_must_be_abstract():
    """Trainer is an ABC: instantiating without implementing train() must fail."""
    from pipeline.trainers import Trainer

    class _MissingTrain(Trainer):
        name = "broken"

    with pytest.raises(TypeError):
        _MissingTrain()  # type: ignore[abstract]


# ── HF stub contract ──────────────────────────────────────────────────────────

def test_hf_lora_stub_raises_with_helpful_message():
    """The stub is deliberately not implemented — but it should point callers
    at the working backend, not just dump a NotImplementedError."""
    from pipeline.trainers import TrainSpec, get_trainer

    trainer = get_trainer("hf-lora")()
    # The spec is irrelevant — the stub raises before touching it.
    spec = TrainSpec(
        model_cfg=None, task_cfg=None, hw_cfg=None,
        data_path=Path("/dev/null"), val_path=None,
        epochs=1, resume_ckpt=None,
        ckpt_dir=Path("/tmp/x"), adapter_dir=Path("/tmp/y"),
        smoke_test=True,
    )
    with pytest.raises(NotImplementedError, match="unsloth-lora"):
        trainer.train(spec)


# ── TrainSpec / TrainResult shapes ────────────────────────────────────────────

def test_train_spec_is_frozen():
    """Specs are run-scoped immutable inputs — accidental mutation would break
    the input_hash contract the orchestrator depends on."""
    from pipeline.trainers import TrainSpec

    spec = TrainSpec(
        model_cfg=None, task_cfg=None, hw_cfg=None,
        data_path=Path("/dev/null"), val_path=None,
        epochs=1, resume_ckpt=None,
        ckpt_dir=Path("/tmp/x"), adapter_dir=Path("/tmp/y"),
        smoke_test=False,
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        spec.epochs = 2  # type: ignore[misc]


def test_train_result_has_safe_defaults():
    """A minimal TrainResult must be constructible — adapters may omit optional
    fields (peak memory on CPU, GPU util when sampling fails)."""
    from pipeline.trainers import TrainResult

    result = TrainResult(
        model_used="x/y",
        substituted=False,
        n_train=10,
        n_val=0,
        epochs_completed=1.0,
        early_stopped=False,
        elapsed_min=1.0,
        gpu_hours=0.01,
    )
    assert result.peak_gpu_mem_mb is None
    assert result.avg_gpu_util_pct is None
    assert result.loss_history == []
    assert result.training_diagnostics == {}


# ── Adapter discoverability ───────────────────────────────────────────────────

def test_unsloth_adapter_module_is_cpu_importable():
    """Adapters must defer heavy framework imports to inside train() — the
    module itself must import without torch/unsloth at the top so unit tests
    that don't mock those libs still load the registry."""
    import importlib
    mod = importlib.import_module("pipeline.trainers.unsloth_lora")
    assert hasattr(mod, "UnslothLoRATrainer")


def test_hf_stub_module_is_cpu_importable():
    import importlib
    mod = importlib.import_module("pipeline.trainers.hf_lora")
    assert hasattr(mod, "HFLoRATrainer")
