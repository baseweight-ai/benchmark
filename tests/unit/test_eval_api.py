"""Unit tests for eval_api.py — fingerprint sensitivity."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _fp(**overrides):
    """Build an _eval_fingerprint with a known baseline, optionally perturbed."""
    from eval_api import _eval_fingerprint
    base = dict(
        test_rows_hash="data-hash",
        prompt_sha="prompt-sha",
        few_shot_hash="fs-hash",
        label_set=["a", "b"],
        condition="zero-shot",
        eval_seed=0,
        model_str="gpt-5.4-mini",
        reasoning_capable=False,
        max_output_tokens=64,
        task_type="classification",
        answer_mode="direct",
    )
    base.update(overrides)
    return _eval_fingerprint(**base)


def test_eval_api_fingerprint_is_stable():
    """Same inputs → same hash (deterministic across calls)."""
    assert _fp() == _fp()


@pytest.mark.parametrize("key,new", [
    ("test_rows_hash",    "other-hash"),
    ("prompt_sha",        "other-sha"),
    ("few_shot_hash",     "other-fs"),
    ("label_set",         ["a", "b", "c"]),
    ("condition",         "5-shot"),
    ("eval_seed",         1),
    ("model_str",         "gpt-5.4-nano"),
    ("reasoning_capable", True),
    ("max_output_tokens", 128),
    ("task_type",         "extraction"),
    ("answer_mode",       "tagged"),
])
def test_eval_api_fingerprint_changes_per_input(key, new):
    """Each documented input must contribute to the fingerprint — perturbing
    any one of them changes the hash. Guards against silently-cached re-evals
    when only some inputs changed (the explicit ask: never run again unless
    something changed about the data, the model, or its properties)."""
    base = _fp()
    perturbed = _fp(**{key: new})
    assert base != perturbed, f"fingerprint unchanged when {key} changed"
