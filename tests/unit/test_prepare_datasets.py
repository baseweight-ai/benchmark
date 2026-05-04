"""Unit tests for prepare_datasets.py helper functions."""
import pytest

from prepare_datasets import (
    format_assistant,
    format_user,
    stratified_sample,
    to_chat,
    truncate_context,
)

pytestmark = pytest.mark.unit


# ── format_user ────────────────────────────────────────────────────────────────

def test_format_user_single_field():
    prompt = {"user_template": "Classify: {sentence}", "text_field": "sentence"}
    row = {"sentence": "The stock rose sharply."}
    result = format_user(prompt, row)
    assert result == "Classify: The stock rose sharply."


def test_format_user_multiple_fields():
    prompt = {
        "user_template": "Q: {question}\nContext: {context}",
        "text_fields": ["question", "context"],
    }
    row = {"question": "What is the date?", "context": "The date is Jan 1."}
    result = format_user(prompt, row)
    assert "What is the date?" in result
    assert "Jan 1" in result


def test_format_user_missing_field_defaults_empty():
    prompt = {"user_template": "Classify: {sentence}", "text_field": "sentence"}
    result = format_user(prompt, {})
    assert result == "Classify: "


# ── format_assistant ───────────────────────────────────────────────────────────

def test_format_assistant_verbatim_str():
    prompt = {"label_format": "verbatim", "label_field": "label"}
    row = {"label": "positive"}
    assert format_assistant(prompt, row) == "positive"


def test_format_assistant_verbatim_int_with_label_names():
    prompt = {"label_format": "verbatim", "label_field": "label"}
    row = {"label": 1}
    label_names = ["negative", "neutral", "positive"]
    assert format_assistant(prompt, row, label_names) == "neutral"


def test_format_assistant_letter():
    prompt = {"label_format": "letter", "label_field": "cop"}
    row = {"cop": 2}
    assert format_assistant(prompt, row) == "C"


def test_format_assistant_code():
    prompt = {"label_format": "code", "label_field": "code"}
    row = {"code": "def foo(): return 42"}
    assert format_assistant(prompt, row) == "def foo(): return 42"


def test_format_assistant_extractive_with_answer():
    prompt = {"label_format": "extractive", "answer_field": "answers"}
    row = {"answers": {"text": ["January 2025", "Jan 2025"], "answer_start": [0, 0]}}
    assert format_assistant(prompt, row) == "January 2025"


def test_format_assistant_extractive_no_answer():
    prompt = {"label_format": "extractive", "answer_field": "answers"}
    row = {"answers": {"text": [], "answer_start": []}}
    assert format_assistant(prompt, row) == "Not found."


# ── to_chat ────────────────────────────────────────────────────────────────────

def test_to_chat_with_assistant():
    result = to_chat("System msg", "User msg", "Assistant msg")
    msgs = result["messages"]
    assert len(msgs) == 3
    assert msgs[0] == {"role": "system", "content": "System msg"}
    assert msgs[1] == {"role": "user", "content": "User msg"}
    assert msgs[2] == {"role": "assistant", "content": "Assistant msg"}


def test_to_chat_without_assistant():
    result = to_chat("System", "User")
    assert len(result["messages"]) == 2
    assert all(m["role"] != "assistant" for m in result["messages"])


# ── stratified_sample ──────────────────────────────────────────────────────────

def _make_rows(n_per_class: int, classes=("A", "B", "C")) -> list[dict]:
    rows = []
    for cls in classes:
        for i in range(n_per_class):
            rows.append({"label": cls, "id": f"{cls}{i}"})
    return rows


def test_stratified_sample_count():
    rows = _make_rows(20)
    result = stratified_sample(rows, "label", 15)
    assert len(result) == 15


def test_stratified_sample_class_balance():
    rows = _make_rows(100)
    result = stratified_sample(rows, "label", 30, seed=42)
    from collections import Counter
    counts = Counter(r["label"] for r in result)
    # Each class should have ~10 (30 // 3)
    for cls in ["A", "B", "C"]:
        assert 8 <= counts[cls] <= 12


def test_stratified_sample_deterministic():
    rows = _make_rows(50)
    r1 = stratified_sample(rows, "label", 20, seed=42)
    r2 = stratified_sample(rows, "label", 20, seed=42)
    assert [r["id"] for r in r1] == [r["id"] for r in r2]


def test_stratified_sample_capped_at_available():
    rows = _make_rows(2)  # 6 total
    result = stratified_sample(rows, "label", 100)
    assert len(result) <= 6


# ── truncate_context ───────────────────────────────────────────────────────────

def test_truncate_context_within_limit():
    text = "word " * 100
    result = truncate_context(text, 200)
    assert result == text.strip() or len(result.split()) <= 200


def test_truncate_context_at_limit():
    text = " ".join(["word"] * 150)
    result = truncate_context(text, 100)
    assert len(result.split()) == 100


def test_truncate_context_short_text():
    text = "short"
    assert truncate_context(text, 1000) == text
