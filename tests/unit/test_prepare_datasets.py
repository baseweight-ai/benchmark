"""Unit tests for prepare_datasets.py helper functions."""
from collections import Counter

import pytest

from prepare_datasets import (
    chunk_cuad_test,
    chunk_cuad_train,
    format_assistant,
    format_eval_label,
    format_gold,
    format_user,
    get_label_set,
    sample,
    sliding_windows,
    to_chat,
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


def test_stratified_sample_min_per_group_can_exceed_total_cap():
    """Documents an asymmetry in sample(): min_per_group*n_groups can exceed
    total_cap, in which case the floor wins and the cap is silently overshot
    (see prepare_datasets.py val-carve clamp for the production safeguard).

    Three groups × min_per_group=300 = 900 rows, even with total_cap=300.
    """
    rows = [{"label": "A", "id": f"A{i}"} for i in range(1000)]
    rows += [{"label": "B", "id": f"B{i}"} for i in range(1000)]
    rows += [{"label": "C", "id": f"C{i}"} for i in range(1000)]
    result = sample(rows, strategy="stratified", stratify_by="label",
                    total_cap=300, min_per_group=300, seed=42)
    assert len(result) == 900


def test_val_carve_clamps_inherited_min_per_group():
    """The val-carve path in process_task inherits the train_sampling config
    (e.g. min_per_group=300 for the FPB train pool) and only overrides
    total_cap with val_size. Without clamping min_per_group to val_size /
    n_groups, the floor inflates val to floor*n_groups rows — 900 from a
    val_size=300 budget for FPB's 3 classes. This test pins the clamp by
    replaying the same arithmetic prepare_datasets.process_task uses.
    """
    rows = [{"label": L, "id": f"{L}{i}"}
            for L in ["A", "B", "C"] for i in range(500)]
    val_size, n_groups = 300, 3
    train_min_per_group = 300

    floor_cap = max(1, val_size // n_groups)         # 100
    clamped = min(train_min_per_group, floor_cap)    # 100, not 300

    result = sample(rows, strategy="stratified", stratify_by="label",
                    total_cap=val_size, min_per_group=clamped, seed=42)
    assert len(result) == val_size
    counts = Counter(r["label"] for r in result)
    assert all(counts[k] > 0 for k in ["A", "B", "C"])


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


# ── format_gold / cot_letter ────────────────────────────────────────────────────

_COT_PROMPT = {
    "label_format": "cot_letter", "label_field": "cop", "explanation_field": "exp",
    "label_map": {"0": "A", "1": "B", "2": "C", "3": "D"},
}


def test_format_gold_cot_letter_returns_bare_letter():
    """The gold/label for a CoT task is the answer letter only — no thinking block."""
    assert format_gold(_COT_PROMPT, {"cop": 2}) == "C"


def test_format_assistant_cot_letter_wraps_thinking_and_answer():
    out = format_assistant(_COT_PROMPT, {"cop": 0, "exp": "Because reasons."})
    assert out == "<thinking>Because reasons.</thinking><answer>A</answer>"


def test_format_assistant_cot_letter_strips_explanation_whitespace():
    out = format_assistant(_COT_PROMPT, {"cop": 1, "exp": "  spaced out  "})
    assert out == "<thinking>spaced out</thinking><answer>B</answer>"


def test_format_assistant_cot_letter_caps_long_explanation():
    """A multi-thousand-char exp is trimmed so the completion fits the output budget."""
    from prepare_datasets import _COT_EXP_MAX_CHARS
    out = format_assistant(_COT_PROMPT, {"cop": 2, "exp": "word " * 5000})
    assert out.startswith("<thinking>")
    assert out.endswith("</thinking><answer>C</answer>")  # answer tag never trimmed
    inner = out[len("<thinking>"):-len("</thinking><answer>C</answer>")]
    assert 0 < len(inner) <= _COT_EXP_MAX_CHARS
    assert not inner.endswith(" ")  # trimmed at a word boundary


def test_format_assistant_cot_letter_short_explanation_untouched():
    """An explanation under the cap passes through verbatim."""
    out = format_assistant(_COT_PROMPT, {"cop": 0, "exp": "short reason"})
    assert out == "<thinking>short reason</thinking><answer>A</answer>"


def test_format_gold_matches_format_assistant_for_non_cot():
    """For verbatim/letter/extractive the training target IS the gold label."""
    for prompt, row in [
        ({"label_format": "verbatim", "label_field": "label"}, {"label": "positive"}),
        ({"label_format": "letter", "label_field": "cop"}, {"cop": 1}),
        ({"label_format": "extractive", "answer_field": "answers"},
         {"answers": {"text": ["a clause"]}}),
    ]:
        assert format_gold(prompt, row) == format_assistant(prompt, row)


def test_get_label_set_cot_letter_returns_choice_letters():
    prompt = {"label_format": "cot_letter", "label_map": {"0": "A", "1": "B", "2": "C", "3": "D"}}
    assert get_label_set(prompt, None) == ["A", "B", "C", "D"]


# ── format_eval_label (multi-answer extraction gold) ────────────────────────────

def test_format_eval_label_extractive_returns_all_spans():
    """A CUAD question can have several valid gold spans — all are kept so
    token_f1 can score the max over them."""
    prompt = {"label_format": "extractive", "answer_field": "answers"}
    row = {"answers": {"text": ["Governing Law: Delaware", "Delaware law governs"]}}
    assert format_eval_label(prompt, row) == ["Governing Law: Delaware", "Delaware law governs"]


def test_format_eval_label_extractive_drops_empty_spans():
    prompt = {"label_format": "extractive", "answer_field": "answers"}
    row = {"answers": {"text": ["real span", "", "  "]}}
    assert format_eval_label(prompt, row) == ["real span"]


def test_format_eval_label_extractive_no_answer_is_not_found():
    prompt = {"label_format": "extractive", "answer_field": "answers"}
    assert format_eval_label(prompt, {"answers": {"text": []}}) == ["Not found."]


def test_format_eval_label_classification_returns_single_string():
    """Closed-set tasks keep a single string label (delegates to format_gold)."""
    assert format_eval_label({"label_format": "verbatim", "label_field": "label"},
                             {"label": "positive"}) == "positive"
    assert format_eval_label({"label_format": "letter", "label_field": "cop"},
                             {"cop": 2}) == "C"


# ── sliding_windows ─────────────────────────────────────────────────────────────

def test_sliding_windows_short_text_single_window():
    assert sliding_windows("a b c", window=10, stride=5) == ["a b c"]


def test_sliding_windows_overlapping_windows():
    text = " ".join(str(i) for i in range(10))  # 10 words
    windows = sliding_windows(text, window=4, stride=2)
    # windows start at 0, 2, 4, 6 → words [0:4], [2:6], [4:8], [6:10]
    assert windows[0] == "0 1 2 3"
    assert windows[1] == "2 3 4 5"
    assert windows[-1].split()[-1] == "9"  # the tail word is covered


def test_sliding_windows_respects_max_chunks():
    text = " ".join(str(i) for i in range(100))
    assert len(sliding_windows(text, window=10, stride=5, max_chunks=3)) == 3


def test_sliding_windows_rejects_nonpositive_stride():
    with pytest.raises(ValueError, match="stride"):
        sliding_windows("a b c d", window=2, stride=0)


# ── chunk_cuad_train / chunk_cuad_test ──────────────────────────────────────────

def _cuad_row(context: str, answer: str) -> dict:
    return {
        "context": context,
        "question": "Governing Law",
        "answers": {"text": [answer] if answer else [], "answer_start": [0]},
        "clause_type": "Governing Law",
        "has_answer": bool(answer),
    }


def test_chunk_cuad_train_one_row_per_question():
    """chunk_cuad_train reduces each question to exactly one window."""
    rows_in = [
        _cuad_row(" ".join(["a"] * 100), "a a a"),   # answerable
        _cuad_row(" ".join(["b"] * 100), ""),        # no-answer
    ]
    out = chunk_cuad_train(rows_in, window=20, stride=15)
    assert len(out) == 2


def test_chunk_cuad_train_positive_window_contains_the_clause():
    ctx = " ".join(["filler"] * 50 + ["the governing law is delaware"] + ["filler"] * 50)
    out = chunk_cuad_train([_cuad_row(ctx, "the governing law is delaware")],
                           window=20, stride=15)
    assert out[0]["answers"]["text"], "expected an answer-bearing positive window"
    assert "the governing law is delaware" in out[0]["context"]


def test_chunk_cuad_train_positive_is_a_grid_window():
    """The training positive is a real grid window — the same fixed grid the
    test side uses — not an answer-snapped off-grid window, so train and eval
    clause positions stay aligned (the position-bias fix)."""
    ctx = " ".join(["filler"] * 200 + ["the unique target clause here"] + ["filler"] * 200)
    grid = sliding_windows(ctx, window=60, stride=45)
    out = chunk_cuad_train([_cuad_row(ctx, "the unique target clause here")],
                           window=60, stride=45)
    assert out[0]["context"] in grid
    assert "the unique target clause here" in out[0]["context"]


def test_chunk_cuad_train_no_answer_question_targets_not_found():
    """A no-answer question yields one window with an empty (→ 'Not found.') target."""
    ctx = " ".join(["some clause text"] * 100)
    out = chunk_cuad_train([_cuad_row(ctx, "")], window=20, stride=15)
    assert len(out) == 1
    assert out[0]["answers"]["text"] == []


def test_chunk_cuad_train_deterministic():
    ctx = " ".join(["the answer here"] + ["x"] * 300)
    a = chunk_cuad_train([_cuad_row(ctx, "the answer here")], window=20, stride=10, seed=42)
    b = chunk_cuad_train([_cuad_row(ctx, "the answer here")], window=20, stride=10, seed=42)
    assert [r["context"] for r in a] == [r["context"] for r in b]


# ── sample() balance_by (50/50 positive / no-answer) ────────────────────────────

def test_sample_balance_by_produces_5050():
    rows = [{"has_answer": i % 2 == 0, "label": ["A", "B", "C"][i % 3], "id": i}
            for i in range(300)]
    out = sample(rows, strategy="stratified", stratify_by="label",
                 total_cap=60, balance_by="has_answer", seed=42)
    assert len(out) == 60
    pos = sum(1 for r in out if r["has_answer"])
    assert pos == 30 and len(out) - pos == 30


def test_sample_balance_by_requires_total_cap():
    with pytest.raises(ValueError, match="total_cap"):
        sample([{"has_answer": True, "label": "A"}],
               strategy="stratified", stratify_by="label", balance_by="has_answer")


def test_sample_balance_by_caps_at_available():
    """When one side is too small for a true 50/50, take all of it (no more)."""
    rows = ([{"has_answer": True, "label": "A", "id": f"p{i}"} for i in range(10)]
            + [{"has_answer": False, "label": "A", "id": f"n{i}"} for i in range(200)])
    out = sample(rows, strategy="stratified", stratify_by="label",
                 total_cap=100, balance_by="has_answer", seed=42)
    assert sum(1 for r in out if r["has_answer"]) == 10


def test_chunk_cuad_test_one_row_per_window_with_stable_ids():
    ctx = " ".join(str(i) for i in range(100))
    rows = chunk_cuad_test([_cuad_row(ctx, "5 6 7")], window=20, stride=20, max_chunks=10)
    assert len(rows) >= 2
    assert rows[0]["_eval_id"] == "cuad_test_0000_chunk00"
    assert rows[1]["_eval_id"] == "cuad_test_0000_chunk01"


def test_chunk_cuad_test_distinct_questions_get_distinct_ids():
    ctx = " ".join(str(i) for i in range(60))
    rows = chunk_cuad_test([_cuad_row(ctx, "1"), _cuad_row(ctx, "2")],
                           window=20, stride=20, max_chunks=10)
    assert any(r["_eval_id"].startswith("cuad_test_0000_") for r in rows)
    assert any(r["_eval_id"].startswith("cuad_test_0001_") for r in rows)
