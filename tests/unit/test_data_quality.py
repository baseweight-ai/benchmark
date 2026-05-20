"""Unit tests for pipeline/data_quality.py."""
from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit


def _dq():
    from pipeline.data_quality import (
        analyze_split,
        cross_split_near_dupes,
        cross_split_stats,
        find_exact_dupes,
        find_near_dupes,
        flag_extreme_length,
        kl_divergence,
        ks_test,
        length_stats,
    )
    return (
        analyze_split, cross_split_near_dupes, cross_split_stats,
        find_exact_dupes, find_near_dupes, flag_extreme_length,
        kl_divergence, ks_test, length_stats,
    )


# ── length_stats ──────────────────────────────────────────────────────────────

def test_length_stats_empty():
    from pipeline.data_quality import length_stats
    assert length_stats([]) == {"n": 0}


def test_length_stats_single():
    from pipeline.data_quality import length_stats
    s = length_stats([5])
    assert s["n"] == 1
    assert s["mean"] == 5.0
    assert s["std"] == 0.0
    assert s["min"] == s["max"] == 5


def test_length_stats_symmetric():
    from pipeline.data_quality import length_stats
    s = length_stats([1, 2, 3, 4, 5])
    assert s["n"] == 5
    assert s["mean"] == 3.0
    assert s["p50"] == 3
    assert s["min"] == 1
    assert s["max"] == 5


def test_length_stats_percentiles_ordered():
    from pipeline.data_quality import length_stats
    s = length_stats(list(range(1, 101)))
    assert s["p5"] <= s["p25"] <= s["p50"] <= s["p75"] <= s["p95"]


# ── find_exact_dupes ──────────────────────────────────────────────────────────

def test_exact_dupes_none():
    from pipeline.data_quality import find_exact_dupes
    assert find_exact_dupes(["alpha", "beta", "gamma"]) == []


def test_exact_dupes_removes_later_occurrences():
    from pipeline.data_quality import find_exact_dupes
    texts = ["apple", "banana", "apple", "cherry", "banana"]
    dupes = find_exact_dupes(texts)
    assert set(dupes) == {2, 4}


def test_exact_dupes_case_and_whitespace_normalized():
    from pipeline.data_quality import find_exact_dupes
    # "hello  world" and "Hello World" normalise to the same hash
    dupes = find_exact_dupes(["hello  world", "Hello World"])
    assert dupes == [1]


def test_exact_dupes_keeps_first():
    from pipeline.data_quality import find_exact_dupes
    texts = ["dup", "unique", "dup", "dup"]
    dupes = set(find_exact_dupes(texts))
    # indices 2 and 3 are duplicates of index 0
    assert 0 not in dupes
    assert {2, 3} <= dupes


# ── find_near_dupes ───────────────────────────────────────────────────────────

def test_near_dupes_empty():
    from pipeline.data_quality import find_near_dupes
    result = find_near_dupes([])
    assert result["n_near_dup_pairs"] == 0


def test_near_dupes_single():
    from pipeline.data_quality import find_near_dupes
    result = find_near_dupes(["only one sentence here"])
    assert result["n_near_dup_pairs"] == 0


def test_near_dupes_identical_pair():
    from pipeline.data_quality import find_near_dupes
    text = "the quick brown fox jumps over the lazy dog"
    # Need enough texts so the stop-shingle filter (>40% doc freq) doesn't eliminate
    # the shared shingles: with 10 texts, max_df=4, so shingles in only 2 docs are kept.
    others = [f"sentence {i} is entirely different from all others" for i in range(8)]
    texts = [text, text] + others
    result = find_near_dupes(texts)
    assert result["n_near_dup_pairs"] >= 1
    assert result["example_pairs"][0][2] == 1.0


def test_near_dupes_budget_cap_truncates(monkeypatch):
    """When the candidate-pair budget is exhausted the scan stops and flags
    `truncated` — guarding prepare against an unbounded run on a boilerplate-heavy
    corpus (e.g. windowed legal contracts)."""
    import pipeline.data_quality as dq
    monkeypatch.setattr(dq, "_NEAR_DUP_BUDGET", 1)
    text = "the quick brown fox jumps over the lazy dog"
    others = [f"sentence {i} is entirely different here" for i in range(8)]
    result = dq.find_near_dupes([text, text] + others)
    assert result["truncated"] is True


def test_near_dupes_not_truncated_under_budget():
    """A small corpus stays well under the budget — truncated is False."""
    from pipeline.data_quality import find_near_dupes
    text = "the quick brown fox jumps over the lazy dog"
    others = [f"sentence {i} is entirely different here" for i in range(8)]
    result = find_near_dupes([text, text] + others)
    assert result["truncated"] is False


def test_cross_split_near_dupes_budget_cap_truncates():
    """A caller-supplied budget caps the cross-split scan and flags truncated."""
    import pipeline.data_quality as dq
    shared = "the quick brown fox jumps over the lazy dog today"
    train = [shared] + [f"train doc {i} unrelated content" for i in range(8)]
    test = [shared] + [f"test doc {i} unrelated content" for i in range(8)]
    result = dq.cross_split_near_dupes(train, test, threshold=0.5, budget=1)
    assert result["truncated"] is True


def test_cross_split_near_dupes_uncapped_by_default():
    """Default (budget=None) runs the full scan — the leakage gate never truncates."""
    import pipeline.data_quality as dq
    train = [f"train doc {i} unrelated content" for i in range(40)]
    test = [f"test doc {i} unrelated content" for i in range(40)]
    result = dq.cross_split_near_dupes(train, test, threshold=0.5)
    assert result["truncated"] is False


def test_near_dupes_dissimilar():
    from pipeline.data_quality import find_near_dupes
    texts = [
        "apple banana cherry mango orange grape lemon lime",
        "python java scala rust golang erlang haskell lisp",
        "table chair desk lamp floor ceiling window door",
    ]
    result = find_near_dupes(texts, threshold=0.8)
    assert result["n_near_dup_pairs"] == 0


def test_near_dupes_threshold_respected():
    from pipeline.data_quality import find_near_dupes
    # Two texts sharing most words
    base = "the quick brown fox jumps over the lazy sleeping dog near the river"
    similar = "the quick brown fox jumps over the lazy sleeping dog near the pond"
    result_strict = find_near_dupes([base, similar], threshold=0.95)
    result_loose  = find_near_dupes([base, similar], threshold=0.5)
    # loose threshold should find it; strict may not
    assert result_loose["n_near_dup_pairs"] >= result_strict["n_near_dup_pairs"]


def test_near_dupes_example_pairs_capped_at_5():
    from pipeline.data_quality import find_near_dupes
    base = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    texts = [base + f" extra{i}" for i in range(20)]
    result = find_near_dupes(texts, threshold=0.5)
    assert len(result["example_pairs"]) <= 5


# ── cross_split_near_dupes ────────────────────────────────────────────────────

def test_cross_split_empty_inputs():
    from pipeline.data_quality import cross_split_near_dupes
    result = cross_split_near_dupes([], ["some test text here"])
    assert result["train_indices_to_filter"] == []
    assert result["total"] == 0


def test_cross_split_exact_match_flagged():
    from pipeline.data_quality import cross_split_near_dupes
    train = ["the quick brown fox jumps over the lazy dog", "unrelated training text"]
    test  = ["the quick brown fox jumps over the lazy dog", "another test example"]
    result = cross_split_near_dupes(train, test, threshold=0.9)
    assert 0 in result["train_indices_to_filter"]
    assert result["exact_count"] >= 1


def test_cross_split_does_not_filter_dissimilar():
    from pipeline.data_quality import cross_split_near_dupes
    train = ["machine learning gradient descent neural networks", "tokenisation vocabulary embeddings"]
    test  = ["completely unrelated content here cooking recipes"]
    result = cross_split_near_dupes(train, test, threshold=0.9)
    assert result["train_indices_to_filter"] == []


def test_cross_split_near_dup_not_exact_flagged():
    from pipeline.data_quality import cross_split_near_dupes
    base    = "the quick brown fox jumps over the lazy sleeping dog near the stream"
    variant = "the quick brown fox jumps over the lazy sleeping dog near the river"
    # Extra test texts ensure stop-shingle filter (>40% doc freq) doesn't wipe the index:
    # with 6 test texts, max_df=2.4, so shingles unique to the similar pair are kept.
    extra_test = [f"completely different test sentence number {i}" for i in range(5)]
    result = cross_split_near_dupes([base], [variant] + extra_test, threshold=0.8)
    assert 0 in result["train_indices_to_filter"]
    assert result["near_dup_count"] >= 1
    assert result["exact_count"] == 0


def test_cross_split_indices_unique():
    from pipeline.data_quality import cross_split_near_dupes
    # Each train example appears at most once in to_filter
    train = ["dup text alpha beta gamma delta epsilon", "unique training example here"]
    test  = ["dup text alpha beta gamma delta epsilon"]
    result = cross_split_near_dupes(train, test)
    indices = result["train_indices_to_filter"]
    assert len(indices) == len(set(indices))


# ── flag_extreme_length ───────────────────────────────────────────────────────

def test_flag_extreme_empty():
    from pipeline.data_quality import flag_extreme_length
    result = flag_extreme_length([])
    assert result["too_short"] == []
    assert result["too_long"] == []


def test_flag_extreme_too_short():
    from pipeline.data_quality import flag_extreme_length
    texts = ["hi", "a normal length sentence here for testing", "ok"]
    result = flag_extreme_length(texts, min_chars=10)
    short_indices = result["too_short"]
    assert 0 in short_indices  # "hi" = 2 chars
    assert 2 in short_indices  # "ok" = 2 chars
    assert 1 not in short_indices


def test_flag_extreme_too_long_tukey_fence():
    from pipeline.data_quality import flag_extreme_length
    # Texts alternate between 45 and 50 chars so IQR=5, fence=Q3+3*IQR=65.
    # An outlier at 500 chars exceeds the fence and must be flagged.
    # (If all normals were identical length, IQR=0 and the fence falls back to s[-1],
    #  which would equal the outlier — so the outlier wouldn't be flagged.)
    normal  = ["x" * (50 if i % 2 == 0 else 45) for i in range(50)]
    outlier = "x" * 500
    texts   = normal + [outlier]
    result  = flag_extreme_length(texts)
    assert len(texts) - 1 in result["too_long"]
    assert not any(i in result["too_long"] for i in range(len(normal)))


def test_flag_extreme_max_chars_override():
    from pipeline.data_quality import flag_extreme_length
    texts = ["a" * 100, "a" * 200, "a" * 50]
    result = flag_extreme_length(texts, max_chars=150)
    assert 1 in result["too_long"]
    assert 0 not in result["too_long"]
    assert 2 not in result["too_long"]
    assert result["max_chars"] == 150


def test_flag_extreme_uniform_lengths_no_too_long():
    from pipeline.data_quality import flag_extreme_length
    # When IQR=0, fence = last value → nothing is too long
    texts = ["hello world"] * 10
    result = flag_extreme_length(texts)
    assert result["too_long"] == []


# ── kl_divergence ─────────────────────────────────────────────────────────────

def test_kl_identical_distributions():
    from pipeline.data_quality import kl_divergence
    p = {"a": 10, "b": 10}
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)


def test_kl_non_negative():
    from pipeline.data_quality import kl_divergence
    p = {"pos": 80, "neg": 20}
    q = {"pos": 50, "neg": 50}
    assert kl_divergence(p, q) >= 0.0


def test_kl_empty_counts():
    from pipeline.data_quality import kl_divergence
    # Should not crash; empty totals treated as 1
    assert kl_divergence({}, {}) == 0.0


def test_kl_missing_key_in_q():
    from pipeline.data_quality import kl_divergence
    p = {"a": 5, "b": 5}
    q = {"a": 10}  # "b" missing from q — p_k > 0 but q_k = 0 → inf
    result = kl_divergence(p, q)
    assert math.isinf(result)


def test_kl_missing_key_in_p_only():
    from pipeline.data_quality import kl_divergence
    # "b" only in q, not p → 0 * log(0/q_k) = 0 by convention → finite
    p = {"a": 10}
    q = {"a": 5, "b": 5}
    result = kl_divergence(p, q)
    assert math.isfinite(result)
    assert result > 0.0


# ── ks_test ───────────────────────────────────────────────────────────────────

def test_ks_returns_dict():
    from pipeline.data_quality import ks_test
    result = ks_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result


def test_ks_identical_distributions():
    from pipeline.data_quality import ks_test
    a = [1.0, 2.0, 3.0, 4.0]
    result = ks_test(a, a)
    assert result["statistic"] == pytest.approx(0.0, abs=1e-6)
    assert result["p_value"] is not None


def test_ks_empty():
    from pipeline.data_quality import ks_test
    result = ks_test([], [1.0, 2.0])
    assert result["statistic"] == 0.0
    assert result["p_value"] is None


def test_ks_disjoint_ranges():
    from pipeline.data_quality import ks_test
    a = [1.0, 2.0, 3.0]
    b = [100.0, 200.0, 300.0]
    assert ks_test(a, b)["statistic"] > 0.9


def test_ks_between_zero_and_one():
    from pipeline.data_quality import ks_test
    a = [float(x) for x in range(50)]
    b = [float(x) for x in range(25, 75)]
    stat = ks_test(a, b)["statistic"]
    assert 0.0 <= stat <= 1.0


# ── analyze_split ─────────────────────────────────────────────────────────────

def test_analyze_split_basic_keys():
    from pipeline.data_quality import analyze_split
    texts = ["hello world today is a great day", "machine learning is fascinating indeed"]
    result = analyze_split(texts)
    assert "n" in result
    assert "char_length" in result
    assert "word_length" in result
    assert "within_near_dupes" in result
    assert result["n"] == 2


def test_analyze_split_with_labels():
    from pipeline.data_quality import analyze_split
    texts  = ["text one here is quite long", "text two is also long enough"]
    labels = ["pos", "neg"]
    result = analyze_split(texts, labels)
    assert "label_distribution" in result
    assert "n_classes" in result
    assert result["n_classes"] == 2
    assert result["label_distribution"]["pos"]["count"] == 1


def test_analyze_split_no_labels():
    from pipeline.data_quality import analyze_split
    result = analyze_split(["some text here indeed"])
    assert "label_distribution" not in result


def test_analyze_split_empty():
    from pipeline.data_quality import analyze_split
    result = analyze_split([])
    assert result["n"] == 0


# ── cross_split_stats ─────────────────────────────────────────────────────────

def test_cross_split_stats_keys():
    from pipeline.data_quality import cross_split_stats
    train = ["training text example one here", "another training example here"]
    test  = ["test text example here indeed", "another test example here now"]
    result = cross_split_stats(train, test, None, None)
    assert "exact_overlap" in result
    assert "near_dup_pairs" in result
    assert "length_ks_stat" in result
    assert "length_ks_p_value" in result
    assert "near_dup_threshold" in result


def test_cross_split_stats_with_labels():
    from pipeline.data_quality import cross_split_stats
    train = ["training example one", "training example two"]
    test  = ["test example one now", "test example two now"]
    result = cross_split_stats(train, test, ["pos", "neg"], ["pos", "neg"])
    assert "label_kl_divergence" in result
    assert result["label_kl_divergence"] == pytest.approx(0.0, abs=1e-4)


def test_cross_split_stats_exact_overlap_counted():
    from pipeline.data_quality import cross_split_stats
    shared = "the quick brown fox jumps over the lazy dog today"
    train = [shared, "unique training text here indeed now"]
    test  = [shared, "unique test text here indeed now"]
    result = cross_split_stats(train, test, None, None)
    assert result["exact_overlap"] >= 1


# ── validate_raw_counts — fails loud when raw is truncated ─────────────────────

def test_validate_raw_counts_passes_when_counts_meet_threshold():
    from pipeline.data_quality import validate_raw_counts
    # Dict-of-lists stands in for HuggingFace DatasetDict — len() works the same.
    ds = {"train": [None] * 4000, "test": [None] * 1000}
    validate_raw_counts(ds, "fpb")   # fpb threshold: train >= 3000; passes


def test_validate_raw_counts_raises_when_below_threshold():
    from pipeline.data_quality import validate_raw_counts
    ds = {"train": [None] * 12, "test": [None] * 5}  # smoke-sized
    with pytest.raises(RuntimeError, match="clobbered by a smoke download"):
        validate_raw_counts(ds, "fpb")


def test_validate_raw_counts_unknown_task_is_noop():
    from pipeline.data_quality import validate_raw_counts
    # Tasks without an EXPECTED_COUNTS entry get no validation — silently pass.
    validate_raw_counts({"train": [None] * 1}, "no_such_task")


def test_validate_raw_counts_missing_split_skipped():
    """A split listed in EXPECTED_COUNTS but absent from ds is skipped, not raised."""
    from pipeline.data_quality import validate_raw_counts
    ds = {"train": [None] * 4000}   # fpb expects train; no test split here
    validate_raw_counts(ds, "fpb")
