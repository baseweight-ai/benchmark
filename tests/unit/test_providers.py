"""Unit tests for pipeline.providers — focused on the pure helpers and
InferenceResult schema. Network-touching paths are covered indirectly by
test_api_mock and test_eval_local."""
import pytest

from pipeline.providers import InferenceResult, _estimate_reasoning_tokens

pytestmark = pytest.mark.unit


# ── _estimate_reasoning_tokens ─────────────────────────────────────────────────

def test_no_think_tag_returns_zero():
    """When the model doesn't emit a <think> block, reasoning_tokens=0."""
    assert _estimate_reasoning_tokens("just an answer", 5) == 0


def test_empty_text_returns_zero():
    assert _estimate_reasoning_tokens("", 0) == 0


def test_only_think_block_attributes_most_tokens_to_reasoning():
    """A response that's almost entirely <think>...</think> attributes
    most tokens to reasoning."""
    text = "<think>" + "x" * 90 + "</think>" + "y"
    n = _estimate_reasoning_tokens(text, 100)
    assert n > 90  # the boundary is past the closing tag


def test_balanced_split_proportionally_attributes():
    """50/50 split between thinking and answer text → roughly half the tokens."""
    text = "a" * 50 + "</think>" + "b" * 50
    n = _estimate_reasoning_tokens(text, 100)
    # Boundary is at char 58 (50 + len("</think>")), so ~58% reasoning
    assert 50 <= n <= 65


def test_returns_int_not_float():
    text = "abc</think>xyz"
    n = _estimate_reasoning_tokens(text, 7)
    assert isinstance(n, int)


# ── InferenceResult schema ────────────────────────────────────────────────────

def test_inference_result_field_order_supports_tuple_destructure():
    """Tuple destructuring at every call site relies on this exact order."""
    r = InferenceResult("hi", 10, 5, 2, 150.0, 50.0, -0.5)
    text, in_tok, out_tok, reasoning_tok, lat, ttft, avg_lp = r
    assert text == "hi"
    assert in_tok == 10
    assert out_tok == 5
    assert reasoning_tok == 2
    assert lat == 150.0
    assert ttft == 50.0
    assert avg_lp == -0.5


def test_inference_result_field_access():
    r = InferenceResult("hi", 10, 5, 2, 150.0, 50.0, -0.5)
    assert r.reasoning_tokens == 2
    assert r.output_tokens == 5
    assert r.avg_logprob == -0.5
