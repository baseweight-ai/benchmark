"""Unit tests for prepare_datasets.py helper functions."""
from collections import Counter

import pytest

from prepare_datasets import (
    format_assistant,
    format_user,
    sample,
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


# ── sample helpers ─────────────────────────────────────────────────────────────

def _make_rows(n_per_class: int, classes=("A", "B", "C")) -> list[dict]:
    rows = []
    for cls in classes:
        for i in range(n_per_class):
            rows.append({"label": cls, "id": f"{cls}{i}"})
    return rows


# ── sample — error paths ──────────────────────────────────────────────────────

def test_sample_empty_data_returns_empty():
    assert sample([], strategy="stratified", stratify_by="label", total_cap=10) == []
    assert sample([], strategy="balanced", stratify_by="label", per_group_cap=5) == []


def test_sample_stratified_missing_total_cap_raises():
    with pytest.raises(ValueError, match="total_cap"):
        sample(_make_rows(5), strategy="stratified", stratify_by="label")


def test_sample_balanced_missing_per_group_cap_raises():
    with pytest.raises(ValueError, match="per_group_cap"):
        sample(_make_rows(5), strategy="balanced", stratify_by="label")


def test_sample_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown sampling strategy"):
        sample(_make_rows(5), strategy="reservoir", stratify_by="label", total_cap=10)


# ── sample — stratified ────────────────────────────────────────────────────────

def test_stratified_sample_total_count():
    rows = _make_rows(20)  # 60 total
    result = sample(rows, strategy="stratified", stratify_by="label", total_cap=15)
    assert len(result) == 15


def test_stratified_sample_proportional_balance():
    rows = _make_rows(100)  # equal classes → equal allocation
    result = sample(rows, strategy="stratified", stratify_by="label", total_cap=30, seed=42)
    counts = Counter(r["label"] for r in result)
    for cls in ["A", "B", "C"]:
        assert 8 <= counts[cls] <= 12


def test_stratified_sample_deterministic():
    rows = _make_rows(50)
    r1 = sample(rows, strategy="stratified", stratify_by="label", total_cap=20, seed=42)
    r2 = sample(rows, strategy="stratified", stratify_by="label", total_cap=20, seed=42)
    assert [r["id"] for r in r1] == [r["id"] for r in r2]


def test_stratified_sample_capped_at_available():
    rows = _make_rows(2)  # 6 total
    result = sample(rows, strategy="stratified", stratify_by="label", total_cap=100)
    assert len(result) <= 6


def test_stratified_sample_min_per_group():
    # Imbalanced: A has 1000, B has 5 — without min_per_group B might get 0 at small total
    rows = [{"label": "A", "id": f"A{i}"} for i in range(1000)]
    rows += [{"label": "B", "id": f"B{i}"} for i in range(5)]
    result = sample(rows, strategy="stratified", stratify_by="label", total_cap=20, min_per_group=2, seed=42)
    counts = Counter(r["label"] for r in result)
    assert counts["B"] >= 2


# ── sample — balanced ──────────────────────────────────────────────────────────

def test_balanced_sample_per_group_cap():
    rows = _make_rows(50)  # 50 per class
    result = sample(rows, strategy="balanced", stratify_by="label", per_group_cap=10, seed=42)
    counts = Counter(r["label"] for r in result)
    assert len(result) == 30
    for cls in ["A", "B", "C"]:
        assert counts[cls] == 10


def test_balanced_sample_capped_at_available():
    rows = _make_rows(3)  # only 3 per class
    result = sample(rows, strategy="balanced", stratify_by="label", per_group_cap=10, seed=42)
    counts = Counter(r["label"] for r in result)
    for cls in ["A", "B", "C"]:
        assert counts[cls] == 3


def test_balanced_sample_deterministic():
    rows = _make_rows(50)
    r1 = sample(rows, strategy="balanced", stratify_by="label", per_group_cap=10, seed=42)
    r2 = sample(rows, strategy="balanced", stratify_by="label", per_group_cap=10, seed=42)
    assert [r["id"] for r in r1] == [r["id"] for r in r2]


def test_balanced_sample_different_seeds_differ():
    rows = _make_rows(50)
    r1 = sample(rows, strategy="balanced", stratify_by="label", per_group_cap=20, seed=1)
    r2 = sample(rows, strategy="balanced", stratify_by="label", per_group_cap=20, seed=2)
    assert [r["id"] for r in r1] != [r["id"] for r in r2]


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
