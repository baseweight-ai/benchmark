"""Unit tests for prepare_datasets.py helper functions."""
from collections import Counter

import pytest

from prepare_datasets import (
    format_assistant,
    format_user,
    get_label_set,
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


# ── get_label_set ──────────────────────────────────────────────────────────────

def test_get_label_set_verbatim_returns_label_names():
    prompt = {"label_format": "verbatim"}
    assert get_label_set(prompt, ["negative", "neutral", "positive"]) == [
        "negative", "neutral", "positive"
    ]


def test_get_label_set_verbatim_without_names_returns_none():
    """verbatim labels stored as raw strings (e.g. FPB before custom_label_names)
    have no closed set knowable from prompt alone."""
    assert get_label_set({"label_format": "verbatim"}, None) is None


def test_get_label_set_letter_returns_choice_letters():
    prompt = {"label_format": "letter", "label_map": {"0": "A", "1": "B", "2": "C", "3": "D"}}
    assert get_label_set(prompt, None) == ["A", "B", "C", "D"]


def test_get_label_set_letter_dedups_preserving_order():
    prompt = {"label_format": "letter", "label_map": {"0": "Yes", "1": "No", "2": "Yes"}}
    assert get_label_set(prompt, None) == ["Yes", "No"]


def test_get_label_set_extractive_returns_none():
    """extraction tasks have no closed set — labels.json should not be emitted."""
    assert get_label_set({"label_format": "extractive"}, None) is None


# ── Curated few-shot selection ─────────────────────────────────────────────────

def _few_shot_from_train(train_rows: list[dict]) -> list[dict]:
    """Replicate the selection logic prepare_datasets uses for few_shot.jsonl —
    one example per distinct label, capped at 5, in the order they first appear."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in train_rows:
        msgs = r.get("messages", [])
        a = next((m for m in msgs if m["role"] == "assistant"), None)
        if a is None:
            continue
        label = a["content"]
        if label in seen:
            continue
        seen.add(label)
        out.append(r)
        if len(out) >= 5:
            break
    return out


def test_curated_few_shot_covers_distinct_classes_first():
    """Selection takes one row per distinct label until 5 are filled."""
    train = [
        {"messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "A"}]},
        {"messages": [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "A"}]},  # skip — already have A
        {"messages": [{"role": "user", "content": "u3"}, {"role": "assistant", "content": "B"}]},
        {"messages": [{"role": "user", "content": "u4"}, {"role": "assistant", "content": "C"}]},
    ]
    few = _few_shot_from_train(train)
    assert [r["messages"][1]["content"] for r in few] == ["A", "B", "C"]


def test_curated_few_shot_caps_at_five():
    """With ≥5 distinct labels available, exactly 5 are selected."""
    train = [
        {"messages": [{"role": "user", "content": f"u{i}"},
                      {"role": "assistant", "content": chr(ord("A") + i)}]}
        for i in range(10)
    ]
    few = _few_shot_from_train(train)
    assert len(few) == 5
    # First five labels (A..E) in order — selection is deterministic.
    assert [r["messages"][1]["content"] for r in few] == ["A", "B", "C", "D", "E"]


def test_curated_few_shot_handles_fewer_than_five_classes():
    """3-class task → 3 examples, not padded to 5."""
    train = [
        {"messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "positive"}]},
        {"messages": [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "negative"}]},
        {"messages": [{"role": "user", "content": "u3"}, {"role": "assistant", "content": "neutral"}]},
        {"messages": [{"role": "user", "content": "u4"}, {"role": "assistant", "content": "positive"}]},
    ]
    few = _few_shot_from_train(train)
    assert len(few) == 3
    assert sorted(r["messages"][1]["content"] for r in few) == ["negative", "neutral", "positive"]


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
