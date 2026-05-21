"""Unit tests for training quality analysis and sweep utilities."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

_noop = lambda msg: None  # silent echo for tests


# ── _analyze_training ──────────────────────────────────────────────────────────

def _diag(losses, anomalies=None):
    from pipeline.trainers import analyze_training as _analyze_training
    return _analyze_training(losses, anomalies or [], _noop)


def test_empty_losses_returns_no_verdict():
    d = _diag([])
    assert d["converged"] is None
    assert d["diverged"] is False
    assert d["plateaued"] is False
    assert d["loss_improvement_pct"] is None


def test_converged_large_improvement():
    losses = [2.0, 1.5, 1.1, 0.9, 0.8, 0.75, 0.7, 0.65]
    d = _diag(losses)
    assert d["converged"] is True
    assert d["loss_improvement_pct"] == pytest.approx(67.5, rel=0.01)


def test_not_converged_small_improvement():
    # ~3% improvement — below the 5% threshold
    losses = [1.0, 0.99, 0.985, 0.98, 0.975, 0.972, 0.970]
    d = _diag(losses)
    assert d["converged"] is False


def test_diverged_rising_end():
    # Clear divergence: early third ~0.83, late third ~1.50 (>5% higher)
    losses = [1.0, 0.8, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7]
    d = _diag(losses)
    assert d["diverged"] is True


def test_not_diverged_monotone_decrease():
    losses = [2.0, 1.6, 1.3, 1.1, 0.9, 0.8, 0.75, 0.72, 0.70, 0.69]
    d = _diag(losses)
    assert d["diverged"] is False


def test_plateaued():
    # Sharp drop early; flat tail
    losses = [3.0, 2.0, 1.5, 1.2, 1.01, 1.005, 1.003, 1.002, 1.001, 1.001, 1.000, 1.000]
    d = _diag(losses)
    assert d["plateaued"] is True
    assert d["converged"] is True  # still improved from start to end


def test_not_plateaued_still_improving():
    losses = [3.0, 2.5, 2.0, 1.6, 1.3, 1.1, 0.9, 0.75]
    d = _diag(losses)
    assert d["plateaued"] is False


def test_anomalies_passed_through():
    anomalies = [{"step": 5, "type": "spike", "value": 9.9, "mean5": 1.0}]
    d = _diag([1.0, 0.9, 0.8], anomalies)
    assert d["anomalies"] == anomalies


def test_short_sequence_skips_diverge_and_plateau():
    # 5 values: too short for divergence (need >=10) or plateau (need >=8)
    losses = [1.0, 0.9, 0.95, 1.0, 1.1]
    d = _diag(losses)
    assert d["diverged"] is False
    assert d["plateaued"] is False


# ── Overfitting detection (val_losses param) ───────────────────────────────────

def _diag_with_val(losses, val_losses, anomalies=None):
    from pipeline.trainers import analyze_training as _analyze_training
    return _analyze_training(losses, anomalies or [], _noop, val_losses=val_losses)


def test_overfitting_none_when_val_losses_not_provided():
    d = _diag([1.0, 0.8, 0.6])
    assert d["overfitting_detected"] is None


def test_overfitting_not_detected_when_val_loss_stable():
    # train loss drops 50%, val loss drops too → no overfitting
    d = _diag_with_val([1.0, 0.8, 0.6, 0.5], val_losses=[1.0, 0.9, 0.85, 0.8])
    assert d["overfitting_detected"] is False


def test_overfitting_detected_when_val_loss_rises():
    # train loss drops 50%, val loss rises >5% → overfitting
    d = _diag_with_val([1.0, 0.8, 0.6, 0.5], val_losses=[0.9, 1.0, 1.1, 1.2])
    assert d["overfitting_detected"] is True


def test_overfitting_detected_when_val_loss_rises_after_dip():
    # U-shaped val curve: loss bottoms mid-run, then climbs >5% above that
    # minimum while train loss keeps falling. The final value (0.044) still
    # sits below the noisy first epoch (0.046), so a first-vs-last check misses
    # it — only the minimum-based reference catches this. Regression guard.
    d = _diag_with_val([1.0, 0.8, 0.6, 0.5, 0.4],
                       val_losses=[0.046, 0.046, 0.038, 0.042, 0.044])
    assert d["overfitting_detected"] is True


def test_overfitting_not_detected_when_dip_recovers_within_5pct():
    # Val loss dips then drifts up, but the final value stays within 5% of the
    # minimum — within noise, not flagged as overfitting.
    d = _diag_with_val([1.0, 0.8, 0.6, 0.5],
                       val_losses=[0.050, 0.040, 0.041, 0.0415])
    assert d["overfitting_detected"] is False


def test_overfitting_not_detected_when_val_loss_rises_less_than_5pct():
    # val loss rises <5% — within noise, not flagged
    d = _diag_with_val([1.0, 0.8, 0.6, 0.5], val_losses=[1.0, 1.01, 1.02, 1.03])
    assert d["overfitting_detected"] is False


def test_overfitting_not_detected_single_noise_spike_at_end():
    """A single tall spike at the end that recovers (or hasn't recovered yet)
    is noise, not overfitting. The smoothed-tail check should suppress it —
    one outlier in three doesn't move the median. Regression guard for the
    'last-value-vs-min' brittleness the previous heuristic had."""
    d = _diag_with_val(
        [1.0, 0.8, 0.6, 0.5, 0.4, 0.3],
        val_losses=[0.04, 0.04, 0.04, 0.04, 0.10],
    )
    assert d["overfitting_detected"] is False


def test_overfitting_not_detected_when_best_is_last():
    """Monotonically-decreasing val_loss: the minimum is the LAST point. The
    model is still improving — must never be flagged as overfit, even though
    every earlier point is above the min. Regression guard for naive
    'tail-vs-min' framings (which would flag this incorrectly)."""
    d = _diag_with_val(
        [1.0, 0.8, 0.6, 0.5],
        val_losses=[1.0, 0.9, 0.85, 0.8],
    )
    assert d["overfitting_detected"] is False


def test_overfitting_detected_with_sustained_post_best_drift():
    """Two consecutive post-best values above the +5% threshold → sustained
    drift, not noise. This is the case the heuristic exists to catch."""
    d = _diag_with_val(
        [1.0, 0.8, 0.6, 0.5, 0.4, 0.3],
        val_losses=[0.046, 0.046, 0.038, 0.080, 0.055],
    )
    assert d["overfitting_detected"] is True


def test_overfitting_none_when_fewer_than_2_val_epochs():
    # Only 1 val_loss point — cannot assess direction
    d = _diag_with_val([1.0, 0.8, 0.6], val_losses=[0.9])
    assert d["overfitting_detected"] is None


# ── Sweep utilities ────────────────────────────────────────────────────────────

def test_grid_trials_count():
    from run_sweep import _make_trials
    cfg = {"search": "grid", "params": {"learning_rate": [1e-4, 2e-4], "lora_rank": [4, 8]}}
    assert len(_make_trials(cfg)) == 4


def test_grid_trials_all_combinations():
    from run_sweep import _make_trials
    cfg = {"search": "grid", "params": {"lr": [1e-4, 2e-4], "rank": [4, 8]}}
    trials = _make_trials(cfg)
    assert {"lr": 1e-4, "rank": 4} in trials
    assert {"lr": 2e-4, "rank": 8} in trials


def test_random_trials_count():
    from run_sweep import _make_trials
    cfg = {"search": "random", "n_trials": 3, "seed": 42,
           "params": {"lr": [1e-4, 2e-4, 5e-4], "rank": [4, 8, 16]}}
    assert len(_make_trials(cfg)) == 3


def test_random_trials_are_subset_of_grid():
    from run_sweep import _make_trials
    params = {"lr": [1e-4, 2e-4, 5e-4], "rank": [4, 8]}
    grid = _make_trials({"search": "grid", "params": params})
    rand = _make_trials({"search": "random", "n_trials": 3, "seed": 42, "params": params})
    for t in rand:
        assert t in grid


def test_random_trials_deterministic():
    from run_sweep import _make_trials
    cfg = {"search": "random", "n_trials": 3, "seed": 42,
           "params": {"lr": [1e-4, 2e-4, 5e-4], "rank": [4, 8, 16]}}
    assert _make_trials(cfg) == _make_trials(cfg)


def test_no_params_returns_single_base_trial():
    from run_sweep import _make_trials
    assert _make_trials({}) == [{}]
    assert _make_trials({"params": {}}) == [{}]


def test_apply_params_routes_lora_keys():
    from run_sweep import _apply_params
    import train_local
    base_model = train_local.load_model_config("qwen2.5-0.5b")
    base_task = train_local.TaskConfig(task_id="t")
    model_cfg, task_cfg = _apply_params(
        base_model, base_task, {"lora_rank": 16, "learning_rate": 5e-4}, "test-sw00"
    )
    # Sweep params land in task overrides — same mechanism as fpb's
    # lora_overrides — so train_one's merge precedence (sweep > task > model)
    # is uniform regardless of where the override came from.
    assert task_cfg.lora_overrides == {"rank": 16}
    assert task_cfg.training_overrides == {"learning_rate": 5e-4}
    assert model_cfg.model_short == "test-sw00"
    # Base model and base task unchanged
    assert base_model.lora["rank"] == 4
    assert base_model.model_short == "qwen2.5-0.5b"
    assert base_task.lora_overrides is None
    assert base_task.training_overrides is None


def test_apply_params_layers_onto_existing_task_overrides():
    """Sweep params merge with the task's own overrides; the sweep value wins
    when keys collide. This is what lets `fpb` carry alpha=16 in its task
    config while the sweep varies `rank` independently."""
    from run_sweep import _apply_params
    import train_local
    base_model = train_local.load_model_config("qwen2.5-0.5b")
    base_task = train_local.TaskConfig(
        task_id="t",
        lora_overrides={"rank": 8, "alpha": 16},
        training_overrides={"learning_rate": 2e-5, "weight_decay": 0.1},
    )
    _, task_cfg = _apply_params(
        base_model, base_task, {"lora_rank": 4, "learning_rate": 5e-5}, "test-sw01"
    )
    # Sweep's lora_rank=4 overrides task's rank=8; alpha=16 persists.
    assert task_cfg.lora_overrides == {"rank": 4, "alpha": 16}
    # Sweep's learning_rate overrides task's; weight_decay persists.
    assert task_cfg.training_overrides == {"learning_rate": 5e-5, "weight_decay": 0.1}


# ── _verify_completion_masking ─────────────────────────────────────────────────

def _verify_masking(labels):
    from pipeline.trainers import verify_completion_masking as _verify_completion_masking
    _verify_completion_masking([{"labels": labels}], _noop)


def test_completion_masking_ok_when_partially_masked():
    # prompt masked (-100), response tokens supervised — the healthy case
    _verify_masking([-100, -100, -100, 5, 6, 7])  # must not raise


def test_completion_masking_raises_when_all_masked():
    # every token -100 → zero loss; response marker missed the chat template
    with pytest.raises(RuntimeError, match="no supervised tokens"):
        _verify_masking([-100, -100, -100, -100])


def test_completion_masking_raises_when_nothing_masked():
    # no -100 → prompt not masked; instruction marker missed the chat template
    with pytest.raises(RuntimeError, match="every token supervised"):
        _verify_masking([3, 4, 5, 6])


# ── Chat-template turn markers ─────────────────────────────────────────────────

def test_model_config_marker_defaults_are_chatml():
    from train_local import ModelConfig
    cfg = ModelConfig(model_id="x", model_short="x")
    assert cfg.instruction_part == "<|im_start|>user\n"
    assert cfg.response_part == "<|im_start|>assistant\n"


@pytest.mark.parametrize("model_id", ["qwen3-8b", "qwen2.5-0.5b"])
def test_model_config_yaml_markers_parse_real_newlines(model_id):
    """The YAML must double-quote the markers so `\\n` is parsed as a real
    newline; a literal backslash-n would never match the tokenized template."""
    import train_local
    cfg = train_local.load_model_config(model_id)
    assert cfg.instruction_part == "<|im_start|>user\n"
    assert cfg.response_part == "<|im_start|>assistant\n"


# ── ConfigFactory smoke-test seq_len ───────────────────────────────────────────

def test_smoke_does_not_shrink_seq_len():
    """Smoke must keep the real per-task seq_len. Task prompts run 500-1200
    tokens, so a shrunken smoke limit truncates the assistant answer off the
    end and completion-only loss collapses to all-masked (the bug this guards)."""
    from train_local import ConfigFactory, load_model_config, load_task_config
    model_cfg = load_model_config("qwen2.5-0.5b")  # the smoke model
    for task_id in ("banking77", "cuad", "fpb", "ledgar", "medmcqa"):
        task_cfg = load_task_config(task_id)
        smoke = ConfigFactory.build(model_cfg, task_cfg, smoke_test=True)
        prod = ConfigFactory.build(model_cfg, task_cfg, smoke_test=False)
        assert smoke.seq_len == prod.seq_len, f"{task_id}: smoke shrank seq_len"
    # cuad needs its longer-window override (2560) — even under smoke
    cuad = ConfigFactory.build(model_cfg, load_task_config("cuad"), smoke_test=True)
    assert cuad.seq_len == 2560


def test_smoke_still_shrinks_model_and_lora():
    """seq_len is the only thing smoke must NOT shrink — the model/LoRA/dtype
    reductions for speed still apply."""
    from train_local import ConfigFactory, load_model_config, load_task_config
    smoke = ConfigFactory.build(load_model_config("qwen3-8b"),
                                load_task_config("banking77"), smoke_test=True)
    assert smoke.lora_rank == 4 and smoke.lora_alpha == 8
    assert smoke.load_in_4bit is False


# ── eval config ────────────────────────────────────────────────────────────────

def test_qwen3_eval_config_supports_load_best():
    """load_best_model_at_end requires eval and save strategies to agree, or HF
    Trainer raises at construction. Also guards that eval stays on so the
    held-out val split actually produces a loss (overfitting stays measurable)."""
    from train_local import load_model_config
    t = load_model_config("qwen3-8b").training
    assert t["eval_strategy"] != "no", "eval must run to record validation loss"
    if t.get("load_best_model_at_end"):
        assert t["eval_strategy"] == t["save_strategy"]


# ── eval cadence + early stopping ──────────────────────────────────────────────

def test_eval_save_steps_basic():
    from pipeline.trainers import eval_save_steps
    # 600 examples / effective batch 16 → 38 steps/epoch; 3 evals/epoch → 12.
    assert eval_save_steps(600, 16, 3) == 12


def test_eval_save_steps_floor_is_one():
    from pipeline.trainers import eval_save_steps
    # Tiny dataset: fewer steps than evals_per_epoch → at least 1, never 0.
    assert eval_save_steps(12, 16, 3) == 1


def test_eval_save_steps_scales_with_cadence():
    from pipeline.trainers import eval_save_steps
    # Same dataset, more evals/epoch → smaller eval_steps.
    assert eval_save_steps(1600, 16, 1) == 100
    assert eval_save_steps(1600, 16, 4) == 25


def test_qwen3_schedule_is_length_insensitive():
    """Early stopping keeps a mid-run checkpoint; the LR schedule must be flat
    after warmup so the kept checkpoint is not a half-decayed snapshot."""
    from train_local import load_model_config
    t = load_model_config("qwen3-8b").training
    assert t["lr_scheduler_type"] in ("constant", "constant_with_warmup")


def test_qwen3_early_stopping_configured():
    """Epoch count is empirical: rigorous setup needs sub-epoch eval +
    EarlyStoppingCallback patience, not a fixed num_train_epochs."""
    from train_local import load_model_config
    t = load_model_config("qwen3-8b").training
    assert t["eval_strategy"] == "steps", "eval per-epoch is too coarse for early stopping"
    assert t.get("evals_per_epoch", 0) >= 1
    assert t.get("early_stopping_patience", 0) >= 1


# ── Loss spike detection (extracted from _CheckpointCallback) ──────────────────

def test_detect_loss_spike_needs_5_priors():
    from pipeline.trainers import detect_loss_spike as _detect_loss_spike
    # Fewer than 5 priors → no baseline → no spike, even if loss is huge.
    assert _detect_loss_spike(99.0, [1.0, 1.0, 1.0, 1.0]) is None


def test_detect_loss_spike_fires_above_3x():
    from pipeline.trainers import detect_loss_spike as _detect_loss_spike
    # mean5 = 0.1, loss = 0.4 → 4× mean → spike.
    spike = _detect_loss_spike(0.4, [0.1, 0.1, 0.1, 0.1, 0.1])
    assert spike is not None
    assert spike["type"] == "spike"
    assert spike["value"] == 0.4
    assert spike["mean5"] == 0.1


def test_detect_loss_spike_no_fire_at_3x():
    from pipeline.trainers import detect_loss_spike as _detect_loss_spike
    # loss == 3 × mean5 exactly → strict >, so NOT a spike.
    assert _detect_loss_spike(0.3, [0.1, 0.1, 0.1, 0.1, 0.1]) is None


def test_detect_loss_spike_zero_mean_skips():
    from pipeline.trainers import detect_loss_spike as _detect_loss_spike
    # Degenerate: mean5 == 0 → can't compute a meaningful ratio, no spike.
    assert _detect_loss_spike(1.0, [0.0, 0.0, 0.0, 0.0, 0.0]) is None


# ── Resume decision (input_hash-aware) ─────────────────────────────────────────

def test_resume_decision_no_prior_state_is_fresh():
    from train_local import _resume_decision
    assert _resume_decision(None, "abc") == "fresh"
    assert _resume_decision({}, "abc") == "fresh"


def test_resume_decision_complete_matching_hash_skips():
    from train_local import _resume_decision
    state = {"status": "complete", "input_hash": "abc"}
    assert _resume_decision(state, "abc") == "skip"


def test_resume_decision_in_progress_matching_hash_resumes():
    from train_local import _resume_decision
    state = {"status": "in_progress", "input_hash": "abc"}
    assert _resume_decision(state, "abc") == "resume"


def test_resume_decision_stale_hash_forces_retrain():
    from train_local import _resume_decision
    state = {"status": "complete", "input_hash": "old"}
    assert _resume_decision(state, "new") == "stale"


def test_resume_decision_missing_hash_is_stale():
    """Legacy train_state.json files (pre-input_hash) must force a retrain —
    we can't prove inputs match, so conservatively assume they don't."""
    from train_local import _resume_decision
    state = {"status": "in_progress"}   # no input_hash recorded
    assert _resume_decision(state, "abc") == "stale"
