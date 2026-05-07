"""Unit tests for eval_lm_harness.py prediction extraction helpers.

These tests do not require lm-eval or a GPU — they only exercise the
sample-parsing logic that converts lm-eval's internal sample dicts into
our predictions JSONL format.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _extract():
    from eval_lm_harness import _extract_ground_truth, _extract_prediction
    return _extract_prediction, _extract_ground_truth


# ── _extract_prediction ───────────────────────────────────────────────────────

def test_extract_prediction_argmax():
    _extract_prediction, _ = _extract()
    # Highest logprob at index 2 → third continuation
    sample = {
        "arguments": [("ctx", " A"), ("ctx", " B"), ("ctx", " C"), ("ctx", " D")],
        "filtered_resps": [(-2.0, False), (-3.0, False), (-0.5, True), (-1.5, False)],
    }
    assert _extract_prediction(sample) == "C"


def test_extract_prediction_strips_whitespace():
    _extract_prediction, _ = _extract()
    sample = {
        "arguments": [("ctx", " positive"), ("ctx", " negative"), ("ctx", " neutral")],
        "filtered_resps": [(-1.0, False), (-0.3, True), (-2.0, False)],
    }
    assert _extract_prediction(sample) == "negative"


def test_extract_prediction_empty_args():
    _extract_prediction, _ = _extract()
    assert _extract_prediction({"arguments": [], "filtered_resps": []}) == ""


def test_extract_prediction_missing_keys():
    _extract_prediction, _ = _extract()
    assert _extract_prediction({}) == ""


def test_extract_prediction_nested_resps():
    """Handle lm-eval versions that wrap each response in an extra list."""
    _extract_prediction, _ = _extract()
    sample = {
        "arguments": [("ctx", " A"), ("ctx", " B")],
        "filtered_resps": [[(-1.0, False)], [(-0.2, True)]],
    }
    assert _extract_prediction(sample) == "B"


def test_extract_prediction_first_when_tied():
    """Deterministic: max() returns first maximum, so index 0 wins on tie."""
    _extract_prediction, _ = _extract()
    sample = {
        "arguments": [("ctx", " X"), ("ctx", " Y")],
        "filtered_resps": [(-1.0, False), (-1.0, False)],
    }
    assert _extract_prediction(sample) == "X"


# ── _extract_ground_truth ─────────────────────────────────────────────────────

def test_extract_ground_truth_int_maps_to_continuation():
    _, _extract_ground_truth = _extract()
    sample = {
        "arguments": [("ctx", " A"), ("ctx", " B"), ("ctx", " C"), ("ctx", " D")],
        "target": 2,
    }
    assert _extract_ground_truth(sample) == "C"


def test_extract_ground_truth_zero_index():
    _, _extract_ground_truth = _extract()
    sample = {
        "arguments": [("ctx", " A"), ("ctx", " B")],
        "target": 0,
    }
    assert _extract_ground_truth(sample) == "A"


def test_extract_ground_truth_string_returned_directly():
    _, _extract_ground_truth = _extract()
    sample = {"arguments": [], "target": "card_arrival"}
    assert _extract_ground_truth(sample) == "card_arrival"


def test_extract_ground_truth_string_stripped():
    _, _extract_ground_truth = _extract()
    sample = {"arguments": [], "target": "  legal_provision  "}
    assert _extract_ground_truth(sample) == "legal_provision"


def test_extract_ground_truth_int_out_of_range_falls_back_to_str():
    _, _extract_ground_truth = _extract()
    # int target beyond args length → str(target)
    sample = {"arguments": [("ctx", " A")], "target": 5}
    assert _extract_ground_truth(sample) == "5"


def test_extract_ground_truth_none_target():
    _, _extract_ground_truth = _extract()
    assert _extract_ground_truth({"target": None}) == ""


# ── medmcqa-shaped sample (integration shape check) ──────────────────────────

def test_medmcqa_sample_shape():
    """End-to-end shape test matching medmcqa's actual sample structure."""
    _extract_prediction, _extract_ground_truth = _extract()
    # medmcqa: cop=0 → correct answer is A; model scores B highest (wrong)
    sample = {
        "doc_id": 42,
        "doc": {"question": "...", "cop": 0},
        "arguments": [
            ("Q: ... A:", " A"),
            ("Q: ... A:", " B"),
            ("Q: ... A:", " C"),
            ("Q: ... A:", " D"),
        ],
        "filtered_resps": [(-2.1, False), (-0.9, True), (-3.4, False), (-2.8, False)],
        "target": 0,   # cop field — correct answer index
        "acc": 0.0,    # model predicted B, gold is A
    }
    pred = _extract_prediction(sample)
    gold = _extract_ground_truth(sample)
    assert pred == "B"
    assert gold == "A"
    assert pred != gold  # correctly identified as wrong
