"""Tests for eval_local.py — dry-run, config loading, and data helpers (no GPU/vLLM needed)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import tests._api_stubs  # noqa: F401 — stubs aiohttp/tqdm
import eval_local
from eval_local import (
    ModelConfig,
    TaskConfig,
    get_few_shot,
    load_label_set,
    load_model_config,
    load_task_config,
    load_test_rows,
    run_eval,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_test_data(root: Path, task_id: str = "fpb", n: int = 5, with_labels: bool = False) -> None:
    prep = root / "data" / "prepared" / task_id
    prep.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": f"{task_id}_test_{i:04d}", "messages": [{"role": "user", "content": f"Q{i}"}]}
        for i in range(n)
    ]
    with open(prep / "test.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if with_labels:
        label_rows = [
            {"id": r["id"], "label": ["positive", "negative", "neutral"][i % 3]}
            for i, r in enumerate(rows)
        ]
        with open(prep / "test_labels.jsonl", "w") as f:
            for r in label_rows:
                f.write(json.dumps(r) + "\n")


def _write_train_data(root: Path, task_id: str = "fpb", n: int = 8) -> None:
    prep = root / "data" / "prepared" / task_id
    prep.mkdir(parents=True, exist_ok=True)
    rows = [
        {"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": "A"}]}
        for i in range(n)
    ]
    with open(prep / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _model_cfg(model_short: str = "qwen2.5-0.5b") -> ModelConfig:
    return ModelConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_short=model_short,
        max_seq_length=256,
    )


def _task_cfg(task_id: str = "fpb") -> TaskConfig:
    return TaskConfig(task_id=task_id, max_output_tokens=32, task_type="classification")


# ── Config loading ──────────────────────────────────────────────────────────────

def test_load_task_config_fpb():
    cfg = load_task_config("fpb")
    assert cfg.task_id == "fpb"
    assert cfg.max_output_tokens > 0
    assert cfg.task_type in ("classification", "extraction")


def test_load_task_config_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_task_config("nonexistent_task_xyz")


def test_load_task_config_cuad_skip_conditions():
    cfg = load_task_config("cuad")
    assert "5-shot" in cfg.skip_conditions


def test_load_model_config_smoke_default():
    """qwen2.5-0.5b is the --smoke-test default model; its config must exist and parse."""
    cfg = load_model_config("qwen2.5-0.5b")
    assert cfg.model_short == "qwen2.5-0.5b"
    assert cfg.max_seq_length > 0


def test_load_model_config_all_prod_models():
    """All models in ALL_MODELS must have a training config. Fails if a config is missing."""
    missing = []
    for mid in eval_local.ALL_MODELS:
        path = REPO_ROOT / "configs" / "training" / f"{mid}.yaml"
        if not path.exists():
            missing.append(mid)
    assert not missing, f"Missing training configs for: {missing}"


# ── Data helpers ────────────────────────────────────────────────────────────────

def test_load_test_rows_basic(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=4)
    rows = load_test_rows("fpb", smoke_test=False)
    assert len(rows) == 4
    assert all("id" in r for r in rows)


def test_load_test_rows_joins_labels(tmp_path, monkeypatch):
    """Labels from a separate *_labels.jsonl are joined onto each row by id."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=4, with_labels=True)
    rows = load_test_rows("fpb", smoke_test=False)
    assert all("label" in r for r in rows)
    assert all(r["label"] in ("positive", "negative", "neutral") for r in rows)


def test_load_test_rows_no_labels_file(tmp_path, monkeypatch):
    """Missing labels file is handled gracefully — rows load without a label field."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=3, with_labels=False)
    rows = load_test_rows("fpb", smoke_test=False)
    assert len(rows) == 3


def test_load_test_rows_seed_resamples_whole_questions(tmp_path, monkeypatch):
    """For a chunked task (CUAD), eval_seed > 0 resamples whole questions from
    test_full.jsonl: every window of a chosen question is kept together, and the
    number of questions matches the seed-0 set."""
    from collections import Counter
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    prep = tmp_path / "data" / "prepared" / "cuad"
    prep.mkdir(parents=True)

    def _chunked(qids, chunks=4):
        return [
            {"id": f"cuad_test_{q:04d}_chunk{c:02d}",
             "messages": [{"role": "user", "content": f"q{q} window {c}"}]}
            for q in qids for c in range(chunks)
        ]

    def _dump(rows, path):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # seed-0 set: 5 questions; full set: 25 — all chunked into 4 windows each.
    _dump(_chunked(range(5)), prep / "test.jsonl")
    full = _chunked(range(25))
    _dump(full, prep / "test_full.jsonl")
    _dump([{"id": r["id"], "label": ["gold span"]} for r in full],
          prep / "test_full_labels.jsonl")

    rows = load_test_rows("cuad", smoke_test=False, eval_seed=2)

    per_q = Counter(r["id"].rsplit("_chunk", 1)[0] for r in rows)
    assert len(per_q) == 5                      # same question count as the seed-0 set
    assert all(c == 4 for c in per_q.values())  # every window of each question kept
    assert all(r["label"] == ["gold span"] for r in rows)  # multi-answer label joined by id


def test_load_label_set_present(tmp_path, monkeypatch):
    """labels.json drives guided_choice — return its list verbatim."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    task_dir = tmp_path / "data" / "prepared" / "fpb"
    task_dir.mkdir(parents=True)
    (task_dir / "labels.json").write_text(json.dumps(["positive", "negative", "neutral"]))
    assert load_label_set("fpb") == ["positive", "negative", "neutral"]


def test_load_label_set_absent_returns_none(tmp_path, monkeypatch):
    """Free-form tasks lack labels.json → no guided_choice constraint."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    (tmp_path / "data" / "prepared" / "cuad").mkdir(parents=True)
    assert load_label_set("cuad") is None


def test_run_eval_writes_bs_suffix_when_concurrency_set(tmp_path, monkeypatch):
    """Non-default concurrency suffixes the output filename with _bs{N}, so
    BS=1 and BS=32 runs are stored side-by-side instead of overwriting."""
    from unittest.mock import patch
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=3, with_labels=True)
    _write_train_data(tmp_path)

    async def mock_call(*args, **kwargs):
        from pipeline.providers import InferenceResult
        return InferenceResult("positive", 100, 5, 0, 100.0, 50.0, None)

    with patch("eval_local.call_vllm", side_effect=mock_call):
        asyncio.run(eval_local.run_eval(
            _model_cfg(), "fpb", "zero-shot", _task_cfg(),
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            dry_run=False, smoke_test=False, eval_seed=0,
            concurrency=1,  # non-default → expect _bs1 suffix
        ))

    out = tmp_path / "results" / "predictions" / "local" / "qwen2.5-0.5b" / "fpb" / "zero-shot_bs1.jsonl"
    assert out.exists(), f"Expected {out}, but it wasn't created"
    # Default-concurrency path must NOT exist (no overwrite)
    default = tmp_path / "results" / "predictions" / "local" / "qwen2.5-0.5b" / "fpb" / "zero-shot.jsonl"
    assert not default.exists()


def test_run_eval_default_concurrency_keeps_legacy_filename(tmp_path, monkeypatch):
    """concurrency=MAX_CONCURRENCY (default) → no suffix, preserving back-compat
    with existing predictions paths."""
    from unittest.mock import patch
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=3, with_labels=True)
    _write_train_data(tmp_path)

    async def mock_call(*args, **kwargs):
        from pipeline.providers import InferenceResult
        return InferenceResult("positive", 100, 5, 0, 100.0, 50.0, None)

    with patch("eval_local.call_vllm", side_effect=mock_call):
        asyncio.run(eval_local.run_eval(
            _model_cfg(), "fpb", "zero-shot", _task_cfg(),
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            dry_run=False, smoke_test=False, eval_seed=0,
            # concurrency omitted → defaults to MAX_CONCURRENCY
        ))

    out = tmp_path / "results" / "predictions" / "local" / "qwen2.5-0.5b" / "fpb" / "zero-shot.jsonl"
    assert out.exists()


def test_run_eval_guided_choice_active_for_direct_task(tmp_path, monkeypatch):
    """A direct classification task with labels.json IS guided_choice-constrained."""
    from unittest.mock import patch
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, task_id="fpb", n=3, with_labels=True)
    _write_train_data(tmp_path, task_id="fpb")
    (tmp_path / "data" / "prepared" / "fpb" / "labels.json").write_text(
        json.dumps(["positive", "negative", "neutral"]))

    seen = []

    async def mock_call(*args, **kwargs):
        from pipeline.providers import InferenceResult
        seen.append(kwargs.get("guided_choice"))
        return InferenceResult("positive", 100, 5, 0, 100.0, 50.0, None)

    with patch("eval_local.call_vllm", side_effect=mock_call):
        asyncio.run(eval_local.run_eval(
            _model_cfg(), "fpb", "zero-shot", _task_cfg("fpb"),
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            dry_run=False, smoke_test=False, eval_seed=0,
        ))
    assert seen and all(gc == ["positive", "negative", "neutral"] for gc in seen)


def test_run_eval_guided_choice_gated_off_for_tagged_task(tmp_path, monkeypatch):
    """A tagged (CoT) task must NOT be guided_choice-constrained, even though it
    still emits a labels.json — a chain-of-thought cannot be pinned to a set."""
    from unittest.mock import patch
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, task_id="medmcqa", n=3, with_labels=True)
    _write_train_data(tmp_path, task_id="medmcqa")
    (tmp_path / "data" / "prepared" / "medmcqa" / "labels.json").write_text(
        json.dumps(["A", "B", "C", "D"]))

    seen = []

    async def mock_call(*args, **kwargs):
        from pipeline.providers import InferenceResult
        seen.append(kwargs.get("guided_choice"))
        return InferenceResult("A", 100, 5, 0, 100.0, 50.0, None)

    tagged_cfg = TaskConfig(task_id="medmcqa", max_output_tokens=64,
                            task_type="classification", answer_mode="tagged")
    with patch("eval_local.call_vllm", side_effect=mock_call):
        asyncio.run(eval_local.run_eval(
            _model_cfg(), "medmcqa", "zero-shot", tagged_cfg,
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            dry_run=False, smoke_test=False, eval_seed=0,
        ))
    assert seen and all(gc is None for gc in seen)


def test_load_label_set_empty_list_returns_none(tmp_path, monkeypatch):
    """An empty list is not a valid constraint — treat as absent."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    task_dir = tmp_path / "data" / "prepared" / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "labels.json").write_text("[]")
    assert load_label_set("task") is None


def test_get_few_shot_returns_at_most_five(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_train_data(tmp_path, n=10)
    rows = get_few_shot("fpb", "qwen2.5-0.5b", smoke_test=False)
    assert len(rows) == 5


def test_get_few_shot_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    rows = get_few_shot("fpb", "qwen2.5-0.5b", smoke_test=False)
    assert rows == []


def test_get_few_shot_prefers_curated_file(tmp_path, monkeypatch):
    """When data/prepared/{task}/few_shot.jsonl exists, it must be used over
    train.jsonl[:5] — curated selection is deterministic and class-diverse."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    prep = tmp_path / "data" / "prepared" / "fpb"
    prep.mkdir(parents=True)
    # 5 dummy train rows, none of which should be picked
    train_rows = [
        {"messages": [{"role": "user", "content": f"DUMMY{i}"},
                      {"role": "assistant", "content": "negative"}]}
        for i in range(5)
    ]
    (prep / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    # Curated few-shot with distinct labels
    curated = [
        {"messages": [{"role": "user", "content": "CURATED1"},
                      {"role": "assistant", "content": "positive"}]},
        {"messages": [{"role": "user", "content": "CURATED2"},
                      {"role": "assistant", "content": "neutral"}]},
    ]
    (prep / "few_shot.jsonl").write_text("\n".join(json.dumps(r) for r in curated) + "\n")
    result = get_few_shot("fpb", "qwen3-8b", smoke_test=False)
    assert len(result) == 2
    assert result[0]["messages"][0]["content"] == "CURATED1"


def test_get_few_shot_falls_back_to_train_when_no_curated(tmp_path, monkeypatch):
    """No few_shot.jsonl → legacy behaviour (first 5 of train.jsonl)."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    prep = tmp_path / "data" / "prepared" / "fpb"
    prep.mkdir(parents=True)
    train_rows = [
        {"messages": [{"role": "user", "content": f"TRAIN{i}"},
                      {"role": "assistant", "content": "neutral"}]}
        for i in range(5)
    ]
    (prep / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    result = get_few_shot("fpb", "qwen3-8b", smoke_test=False)
    assert len(result) == 5
    assert result[0]["messages"][0]["content"] == "TRAIN0"


def test_get_few_shot_fewer_than_five_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_train_data(tmp_path, n=3)
    rows = get_few_shot("fpb", "qwen2.5-0.5b", smoke_test=False)
    assert len(rows) == 3


# ── run_eval dry-run ────────────────────────────────────────────────────────────

def test_run_eval_dry_run_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=5)

    asyncio.run(run_eval(
        _model_cfg(), "fpb", "zero-shot", _task_cfg(),
        model_name="Qwen/Qwen2.5-0.5B-Instruct", dry_run=True,
    ))

    out = tmp_path / "results" / "predictions" / "local" / "qwen2.5-0.5b" / "fpb" / "zero-shot.jsonl"
    assert not out.exists()


def test_run_eval_dry_run_missing_data_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError):
        asyncio.run(run_eval(
            _model_cfg(), "fpb", "zero-shot", _task_cfg(),
            model_name="Qwen/Qwen2.5-0.5B-Instruct", dry_run=True,
        ))


def test_run_eval_skips_existing(tmp_path, monkeypatch):
    """If the output file already exists, run_eval returns without making any API calls."""
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=5)

    out = tmp_path / "results" / "predictions" / "local" / "qwen2.5-0.5b" / "fpb" / "zero-shot.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"id":"existing"}\n')

    asyncio.run(run_eval(
        _model_cfg(), "fpb", "zero-shot", _task_cfg(),
        model_name="Qwen/Qwen2.5-0.5B-Instruct", dry_run=False,
    ))

    assert out.read_text() == '{"id":"existing"}\n'


def test_eval_fingerprint_changes_when_adapter_changes(tmp_path, monkeypatch):
    """Regression: a LoRA adapter rewrite must change the prediction fingerprint
    so the pre-filter invalidates stale predictions and re-evaluates against the
    new adapter. Without this, retraining a LoRA leaves the old predictions in
    place and downstream metrics silently report the old run's accuracy.
    """
    from eval_local import _eval_fingerprint

    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    _write_test_data(tmp_path, n=3)

    # Stub the prompt sidecar so the fingerprint has a stable prompt_sha.
    (tmp_path / "data" / "prepared" / "fpb" / "prompt_sha.txt").write_text("dummy_sha\n")

    adapter_dir = tmp_path / "results" / "adapters" / "local" / "qwen2.5-0.5b" / "fpb" / "lora"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter_v1_weights")
    fp1 = _eval_fingerprint(_model_cfg(), "fpb", "lora", _task_cfg(), eval_seed=0, concurrency=4)

    # Rewrite the adapter (simulates a retrain) — fingerprint must shift.
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter_v2_weights_DIFFERENT")
    fp2 = _eval_fingerprint(_model_cfg(), "fpb", "lora", _task_cfg(), eval_seed=0, concurrency=4)
    assert fp1 != fp2, "fingerprint must invalidate when the LoRA adapter changes"

    # The non-lora condition does NOT incorporate the adapter, so its
    # fingerprint stays stable across the same adapter rewrite.
    fp_zs = _eval_fingerprint(_model_cfg(), "fpb", "zero-shot", _task_cfg(),
                              eval_seed=0, concurrency=4)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter_v3_weights_DIFFERENT_AGAIN")
    fp_zs_again = _eval_fingerprint(_model_cfg(), "fpb", "zero-shot", _task_cfg(),
                                    eval_seed=0, concurrency=4)
    assert fp_zs == fp_zs_again, "zero-shot fingerprint must not depend on the lora adapter"


def test_run_eval_dry_run_all_tasks(tmp_path, monkeypatch):
    """dry-run succeeds for every registered task (catches missing/broken task configs)."""
    # Load real configs before redirecting REPO_ROOT to tmp_path
    task_cfgs = {tid: load_task_config(tid) for tid in eval_local.ALL_TASKS}
    monkeypatch.setattr(eval_local, "REPO_ROOT", tmp_path)
    for tid, cfg in task_cfgs.items():
        _write_test_data(tmp_path, task_id=tid, n=2)
        asyncio.run(run_eval(
            _model_cfg(), tid, "zero-shot", cfg,
            model_name="Qwen/Qwen2.5-0.5B-Instruct", dry_run=True,
        ))
