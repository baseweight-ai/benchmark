"""Unit tests for training quality analysis and sweep utilities."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

_noop = lambda msg: None  # silent echo for tests


# ── _analyze_training ──────────────────────────────────────────────────────────

def _diag(losses, anomalies=None):
    from train_local import _analyze_training
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
    from train_local import _analyze_training
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


def test_overfitting_not_detected_when_val_loss_rises_less_than_5pct():
    # val loss rises <5% — within noise, not flagged
    d = _diag_with_val([1.0, 0.8, 0.6, 0.5], val_losses=[1.0, 1.01, 1.02, 1.03])
    assert d["overfitting_detected"] is False


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
    base = train_local.load_model_config("qwen2.5-0.5b")
    result = _apply_params(base, {"lora_rank": 16, "learning_rate": 5e-4}, "test-sw00")
    assert result.lora["rank"] == 16
    assert result.training["learning_rate"] == 5e-4
    assert result.model_short == "test-sw00"
    # Base model unchanged
    assert base.lora["rank"] == 4
    assert base.model_short == "qwen2.5-0.5b"
