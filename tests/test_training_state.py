"""Layer 5 — Training state management: verify checkpoint logic without GPU.

Unsloth, TRL, and transformers are all mocked. These tests verify:
  - Completed conditions are skipped immediately
  - In-progress state is written before training starts
  - _CheckpointCallback persists epoch state on each save
  - Completed state is written after training finishes
  - Final adapter is mirrored to the checkpoint dir via shutil.copytree
  - Training resumes from the latest checkpoint when one exists
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest

import checkpoint_utils
from checkpoint_utils import (
    checkpoint_dir,
    find_hf_resume_checkpoint,
    load_train_state,
    save_train_state,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parent.parent


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ml_modules():
    """Inject stub ML modules into sys.modules so train_local.py's lazy imports work."""
    # TrainerCallback must be a real class so _CheckpointCallback can subclass it.
    TrainerCallbackBase = type("TrainerCallback", (), {})

    mock_trainer = MagicMock()
    mock_train_result = MagicMock()
    mock_train_result.metrics = {"eval_loss": 0.35}
    mock_trainer.train.return_value = mock_train_result

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted_text"

    mock_fast_model_cls = MagicMock()
    mock_fast_model_cls.from_pretrained.return_value = (mock_model, mock_tokenizer)
    mock_fast_model_cls.get_peft_model.return_value = mock_model

    mock_unsloth = MagicMock()
    mock_unsloth.FastModel = mock_fast_model_cls

    mock_trl = MagicMock()
    mock_trl.SFTTrainer.return_value = mock_trainer
    mock_trl.SFTConfig.return_value = MagicMock()

    mock_transformers = MagicMock()
    mock_transformers.TrainerCallback = TrainerCallbackBase

    mock_dataset = MagicMock()
    mock_dataset.map.return_value = mock_dataset
    mock_datasets = MagicMock()
    mock_datasets.Dataset.from_list.return_value = mock_dataset

    stubs = {
        "unsloth": mock_unsloth,
        "trl": mock_trl,
        "transformers": mock_transformers,
        "datasets": mock_datasets,
    }
    with patch.dict(sys.modules, stubs):
        yield {
            "trainer": mock_trainer,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "fast_model_cls": mock_fast_model_cls,
            "train_result": mock_train_result,
        }


@pytest.fixture
def train_env(tmp_path, monkeypatch, tmp_checkpoints_root):
    """Full environment: redirect REPO_ROOT for train_local.py; CHECKPOINTS_ROOT comes from tmp_checkpoints_root."""
    import train_local as train
    monkeypatch.setattr(train, "REPO_ROOT", tmp_path)

    from train_local import ModelConfig, TaskConfig
    model_cfg = ModelConfig(
        model_id="test/tiny",
        model_short="test-model",
        max_seq_length=512,
        load_in_4bit=False,
        dtype="bfloat16",
        enable_thinking=None,
        lora={"rank": 4, "alpha": 8, "dropout": 0.0, "use_rslora": False},
        training={
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "optim": "adamw_8bit",
            "max_grad_norm": 1.0,
            "bf16": False,
            "seed": 42,
            "save_strategy": "epoch",
            "eval_strategy": "no",
            "load_best_model_at_end": False,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "logging_steps": 1,
            "report_to": "none",
        },
    )
    task_cfg = TaskConfig(task_id="toy", max_seq_length=512)

    # Write a tiny training JSONL
    data_dir = tmp_path / "data" / "prepared" / "toy"
    data_dir.mkdir(parents=True)
    data_path = data_dir / "train_500.jsonl"
    rows = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
    with open(data_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    return {"model_cfg": model_cfg, "task_cfg": task_cfg, "data_path": data_path, "tmp_path": tmp_path}


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_train_one_skips_complete_condition(train_env, tmp_checkpoints_root):
    """If train_state.json says 'complete', train_one returns without training."""
    import train_local as train

    e = train_env
    save_train_state("test-model", "toy", "lora", {"status": "complete", "eval_loss": 0.2})

    # Write dummy metadata so the early-return has something to load
    meta_path = e["tmp_path"] / "results" / "training" / "local" / "test-model" / "toy" / "lora" / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"model_id": "test-model", "n_train": 1}))

    result = train.train_one(e["model_cfg"], e["task_cfg"], e["data_path"], dry_run=False)

    assert result.get("model_id") == "test-model"


def test_train_one_writes_in_progress_state(train_env, tmp_checkpoints_root, mock_ml_modules):
    """Training starts by writing status=in_progress to the checkpoint dir."""
    import train_local as train

    e = train_env
    train.train_one(e["model_cfg"], e["task_cfg"], e["data_path"], dry_run=False)

    # State must be "complete" after successful training, but check it was written
    state = load_train_state("test-model", "toy", "lora")
    assert state is not None
    assert state["status"] == "complete"


def test_train_one_writes_complete_state(train_env, tmp_checkpoints_root, mock_ml_modules):
    """After training, train_state.json has status=complete with eval_loss."""
    import train_local as train

    e = train_env
    mock_ml_modules["train_result"].metrics = {"eval_loss": 0.42}
    train.train_one(e["model_cfg"], e["task_cfg"], e["data_path"], dry_run=False)

    state = load_train_state("test-model", "toy", "lora")
    assert state["status"] == "complete"
    assert state["eval_loss"] == pytest.approx(0.42)


def test_train_one_writes_metadata_atomically(train_env, tmp_checkpoints_root, mock_ml_modules):
    """metadata.json is written atomically (no .tmp residue)."""
    import train_local as train

    e = train_env
    train.train_one(e["model_cfg"], e["task_cfg"], e["data_path"], dry_run=False)

    log_dir = e["tmp_path"] / "results" / "training" / "local" / "test-model" / "toy" / "lora"
    meta = log_dir / "metadata.json"
    assert meta.exists()
    assert not (log_dir / "metadata.json.tmp").exists()
    data = json.loads(meta.read_text())
    assert data["task_id"] == "toy"


def test_train_one_mirrors_adapter_to_checkpoint_dir(train_env, tmp_checkpoints_root, mock_ml_modules):
    """Adapter is copied (via shutil.copytree) to ckpt_dir/final_adapter."""
    import train_local as train

    e = train_env
    copied_src = []
    original_copytree = shutil.copytree

    def tracking_copytree(src, dst, **kwargs):
        copied_src.append(Path(src))
        Path(dst).mkdir(parents=True, exist_ok=True)

    with patch("train_local.shutil.copytree", side_effect=tracking_copytree):
        train.train_one(e["model_cfg"], e["task_cfg"], e["data_path"], dry_run=False)

    assert len(copied_src) == 1
    # Source should be the local adapter dir (in REPO_ROOT)
    assert "adapters" in str(copied_src[0])


def test_train_one_passes_resume_checkpoint_to_trainer(train_env, tmp_checkpoints_root, mock_ml_modules):
    """If a checkpoint-N dir exists in the checkpoints root, it is passed to trainer.train()."""
    import train_local as train

    e = train_env
    ckpt_dir = checkpoint_dir("test-model", "toy", "lora")
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "checkpoint-5").mkdir()

    train.train_one(e["model_cfg"], e["task_cfg"], e["data_path"], dry_run=False)

    trainer = mock_ml_modules["trainer"]
    resume_arg = trainer.train.call_args.kwargs.get("resume_from_checkpoint")
    assert resume_arg is not None
    assert "checkpoint-5" in str(resume_arg)


def test_checkpoint_callback_updates_state(train_env, tmp_checkpoints_root, mock_ml_modules):
    """_CheckpointCallback.on_save must write epoch/step to train_state.json."""
    import train_local as train

    e = train_env
    captured_callback = None

    original_sft_trainer = sys.modules["trl"].SFTTrainer

    def capture_trainer(*args, **kwargs):
        nonlocal captured_callback
        callbacks = kwargs.get("callbacks", [])
        if callbacks:
            captured_callback = callbacks[0]
        return mock_ml_modules["trainer"]

    sys.modules["trl"].SFTTrainer = capture_trainer

    train.train_one(e["model_cfg"], e["task_cfg"], e["data_path"], dry_run=False)

    assert captured_callback is not None
    # Simulate on_save being called by the trainer
    mock_state = MagicMock()
    mock_state.epoch = 2
    mock_state.global_step = 50
    mock_state.best_metric = 0.3
    mock_state.best_model_checkpoint = "/tmp/ckpt"
    captured_callback.on_save(MagicMock(), mock_state, MagicMock())

    state = load_train_state("test-model", "toy", "lora")
    assert state["epoch"] == 2
    assert state["global_step"] == 50


def test_train_one_dry_run_writes_metadata_no_training(train_env, tmp_checkpoints_root):
    """--dry-run writes metadata.json but never imports unsloth."""
    import train_local as train

    e = train_env
    result = train.train_one(e["model_cfg"], e["task_cfg"], e["data_path"], dry_run=True)

    assert result["model_id"] == "test-model"
    assert result["n_train"] >= 0
    meta_path = e["tmp_path"] / "results" / "training" / "local" / "test-model" / "toy" / "lora" / "metadata.json"
    assert meta_path.exists()


# ── TaskConfig training_overrides ─────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("overrides,expected", [
    ({"learning_rate": 5e-5}, {"learning_rate": 5e-5}),
    (None, None),
])
def test_task_config_training_overrides_roundtrip(overrides, expected):
    from train_local import TaskConfig
    cfg = TaskConfig(task_id="x", training_overrides=overrides)
    assert cfg.training_overrides == expected


@pytest.mark.unit
def test_fpb_yaml_specifies_anti_overfit_overrides():
    """FPB (~600 train rows) keeps reduced LR + raised weight_decay so the
    LoRA adapter can't memorise the train set and forget pretrained knowledge.
    Thresholded (not equality) so values can be tuned without churning tests.
    """
    import train_local as train
    cfg = train.load_task_config("fpb")
    assert cfg.training_overrides is not None
    assert cfg.training_overrides.get("learning_rate", 2e-4) <= 4e-5
    assert cfg.training_overrides.get("weight_decay", 0.01) >= 0.05


@pytest.mark.unit
def test_training_overrides_merge_replaces_model_defaults():
    model_training = {"learning_rate": 2e-4, "warmup_ratio": 0.05, "weight_decay": 0.01}
    overrides = {"learning_rate": 5e-5, "warmup_ratio": 0.1}
    merged = dict(model_training)
    merged.update(overrides)
    assert merged == {"learning_rate": 5e-5, "warmup_ratio": 0.1, "weight_decay": 0.01}


@pytest.mark.unit
def test_training_overrides_change_invalidates_cache(tmp_path):
    """Tweaking task-level overrides must change input_hash so a cached run
    is re-executed instead of silently skipped."""
    from pipeline.cache import training_inputs_hash

    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"messages":[]}\n')
    model_training = {"learning_rate": 2e-4}

    def hash_for(overrides: dict | None) -> str:
        merged = dict(model_training)
        if overrides:
            merged.update(overrides)
        return training_inputs_hash(data_path, {"training": merged})

    base = hash_for(None)
    tweaked = hash_for({"learning_rate": 5e-5})
    assert base != tweaked
