"""Unit tests for classify_errors.py."""
from pathlib import Path

import pytest

from classify_errors import (
    aggregate_chunk_predictions,
    aggregate_seed_summaries,
    classify_classification,
    classify_extraction,
    classify_predictions,
    compute_axis_scores,
    compute_metric,
    extract_tagged_answer,
    get_valid_labels,
    is_chunked,
    is_empty,
    is_format_violation,
    is_refusal,
    normalize_text,
    token_f1,
)

pytestmark = pytest.mark.unit


# ── normalize_text ─────────────────────────────────────────────────────────────

def test_normalize_text_lowercases():
    assert normalize_text("POSITIVE") == "positive"


def test_normalize_text_strips_punctuation():
    assert normalize_text("hello, world!") == "hello world"


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  a   b  ") == "a b"


def test_normalize_text_empty():
    assert normalize_text("") == ""


# ── token_f1 ───────────────────────────────────────────────────────────────────

def test_token_f1_perfect_match():
    assert token_f1("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)


def test_token_f1_empty_both():
    assert token_f1("", "") == pytest.approx(1.0)


def test_token_f1_empty_pred():
    assert token_f1("", "answer") == pytest.approx(0.0)


def test_token_f1_empty_gold():
    assert token_f1("answer", "") == pytest.approx(0.0)


def test_token_f1_partial_overlap():
    score = token_f1("the quick fox", "the quick brown fox")
    assert 0.0 < score < 1.0


def test_token_f1_no_overlap():
    assert token_f1("cat", "dog") == pytest.approx(0.0)


def test_token_f1_duplicate_tokens():
    # "a a" vs "a" — precision=0.5, recall=1.0 → F1=2/3
    score = token_f1("a a", "a")
    assert score == pytest.approx(2 / 3, rel=1e-3)


# ── is_empty / is_refusal / is_format_violation ───────────────────────────────

def test_is_empty_whitespace():
    assert is_empty("   ")
    assert is_empty("")


def test_is_empty_nonempty():
    assert not is_empty("positive")


@pytest.mark.parametrize("text", [
    "I cannot provide that information.",
    "I can't help with this.",
    "I'm not able to do that.",
    "I am not able to assist.",
    "I'm unable to answer.",
    "I am unable to help.",
    "I won't do that.",
    "I will not answer.",
    "I refuse to comply.",
    "As an AI, I...",
    "I don't feel comfortable with this.",
    "I'm sorry, but I cannot answer.",
    "Sorry, I cannot do that.",
])
def test_is_refusal_detected(text):
    assert is_refusal(text)


def test_is_refusal_normal_output():
    assert not is_refusal("positive")
    assert not is_refusal("The answer is negative.")


def test_is_format_violation_not_in_labels():
    assert is_format_violation("maybe", ["positive", "negative", "neutral"])


def test_is_format_violation_valid_label():
    assert not is_format_violation("positive", ["positive", "negative", "neutral"])


def test_is_format_violation_case_sensitive():
    """Constrained decoding produces exact label strings — case mismatches are
    real format violations and must be surfaced, not masked."""
    assert is_format_violation("POSITIVE", ["positive", "negative", "neutral"])
    assert is_format_violation("Positive", ["positive", "negative", "neutral"])


def test_is_format_violation_whitespace_sensitive():
    """Trailing newlines / surrounding whitespace also count as violations —
    they indicate the model didn't follow the exact-string contract."""
    assert is_format_violation("positive\n", ["positive", "negative", "neutral"])
    assert is_format_violation(" positive", ["positive", "negative", "neutral"])


def test_is_format_violation_no_labels():
    assert not is_format_violation("anything goes", None)


# ── classify_classification ────────────────────────────────────────────────────

def test_classify_classification_correct():
    assert classify_classification("positive", "positive") == "correct"


def test_classify_classification_wrong_class():
    assert classify_classification("negative", "positive") == "wrong_class"


def test_classify_classification_empty():
    assert classify_classification("", "positive") == "empty"


def test_classify_classification_refusal():
    assert classify_classification("I cannot answer this", "positive") == "refusal"


def test_classify_classification_format_violation():
    result = classify_classification("sure thing", "positive", ["positive", "negative", "neutral"])
    assert result == "format_violation"


def test_classify_classification_priority_empty_beats_refusal():
    assert classify_classification("", "positive", ["positive"]) == "empty"


def test_classify_classification_case_mismatch_is_format_violation_not_correct():
    """A case-mismatched label is a format violation, not a correct answer.
    Constrained generation should prevent this, but if it slips through, it
    surfaces as a violation rather than silently passing."""
    result = classify_classification("POSITIVE", "positive", ["positive", "negative", "neutral"])
    assert result == "format_violation"


def test_classify_classification_case_mismatch_without_label_set_is_wrong():
    """Without a label set we can't classify it as a format_violation, but
    strict matching still requires exact equality — case mismatch → wrong."""
    assert classify_classification("POSITIVE", "positive") == "wrong_class"


def test_classify_classification_trailing_whitespace_is_format_violation():
    """Trailing whitespace from API models without guided_choice → violation."""
    result = classify_classification("positive\n", "positive", ["positive", "negative"])
    assert result == "format_violation"


def test_classify_classification_banking77_exact_match():
    """The canonical PolyAI/banking77 label uses mixed case for some intents
    (e.g. 'Refund_not_showing_up'). Strict matching must respect that."""
    labels = ["Refund_not_showing_up", "card_arrival"]
    # Exact match: correct
    assert classify_classification("Refund_not_showing_up", "Refund_not_showing_up", labels) == "correct"
    # Lowercased: format_violation (not in the labels list literally)
    assert classify_classification("refund_not_showing_up", "Refund_not_showing_up", labels) == "format_violation"


# ── classify_extraction ────────────────────────────────────────────────────────

def test_classify_extraction_correct():
    gt = "The contract expires on January 1, 2025"
    assert classify_extraction(gt, gt) == "correct"


def test_classify_extraction_empty():
    assert classify_extraction("", "some answer") == "empty"


def test_classify_extraction_partial():
    pred = "The contract"
    gt = "The contract expires on January 1, 2025"
    result = classify_extraction(pred, gt)
    assert result in ("partial", "hallucinated")


@pytest.mark.parametrize("pred,gt", [
    ("Not found.", "Not found."),
    ("No answer.", "Not found."),
    ("None", "Not found."),
    ("not applicable", "not mentioned"),
    ("not mentioned", "not found"),
])
def test_classify_extraction_not_applicable_variants(pred, gt):
    assert classify_extraction(pred, gt) == "not_applicable"


def test_classify_extraction_hallucinated_gt_not_found():
    result = classify_extraction("The date is January 2025", "Not found.")
    assert result == "hallucinated"


def test_classify_extraction_format_violation_too_long():
    # Prediction much longer than ground truth → format_violation
    gt = "yes"
    pred = " ".join(["word"] * 200)
    result = classify_extraction(pred, gt)
    assert result == "format_violation"


# ── multi-answer extraction scoring (CUAD questions with several gold spans) ──

def test_token_f1_list_takes_max_over_golds():
    """A question with multiple valid spans → F1 is the best match, not the first."""
    pred = "the agreement is governed by delaware law"
    golds = ["completely unrelated text", "the agreement is governed by delaware law"]
    assert token_f1(pred, golds) == pytest.approx(1.0)


def test_token_f1_list_picks_best_partial():
    pred = "delaware law"
    golds = ["new york", "governed by delaware law here"]
    assert token_f1(pred, golds) == max(token_f1(pred, g) for g in golds)
    assert token_f1(pred, golds) > 0.0


def test_token_f1_empty_list_matches_empty_string():
    assert token_f1("anything", []) == token_f1("anything", "")


def test_classify_extraction_list_credits_a_non_first_gold():
    """The multi-answer fix: extracting the SECOND valid span must score
    'correct', not be penalised for not matching answers[0]."""
    pred = "either party may terminate on 30 days notice"
    golds = ["the first valid clause text here", "either party may terminate on 30 days notice"]
    assert classify_extraction(pred, golds) == "correct"


def test_classify_extraction_empty_gold_list_treated_as_string():
    assert classify_extraction("", []) == "empty"


# ── compute_metric ─────────────────────────────────────────────────────────────

def _make_task_cfg(task_type: str, metric_id: str, metric_granularity: str = "per_example",
                    task_id: str = "toy"):
    from classify_errors import TaskConfig
    return TaskConfig(task_id=task_id, task_type=task_type, metric_id=metric_id,
                      metric_granularity=metric_granularity)


@pytest.fixture
def summary_from_predictions(tmp_path, monkeypatch):
    """Write toy prediction rows then run classify and return the summary dict.

    Usage: summary = summary_from_predictions(rows, valid_labels=["positive"])
    """
    import classify_errors as ce

    def _build(rows: list[dict], valid_labels: list[str] | None = None) -> dict:
        import json as _json
        pred_dir = tmp_path / "results" / "predictions" / "local" / "m" / "fpb"
        pred_dir.mkdir(parents=True, exist_ok=True)
        (pred_dir / "lora.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in rows) + "\n"
        )
        monkeypatch.setattr(ce, "REPO_ROOT", tmp_path)
        cfg = _make_task_cfg("classification", "accuracy", task_id="fpb")
        return ce.process_model_task_condition(
            "m", "fpb", "lora", cfg, valid_labels, dry_run=False, source="local"
        )

    return _build


def test_compute_metric_accuracy():
    cfg = _make_task_cfg("classification", "accuracy")
    rows = [
        {"error_category": "correct"},
        {"error_category": "correct"},
        {"error_category": "wrong_class"},
        {"error_category": "correct"},
    ]
    assert compute_metric(cfg, rows) == pytest.approx(0.75)


def test_compute_metric_token_f1():
    cfg = _make_task_cfg("extraction", "token_f1")
    rows = [{"token_f1": 1.0}, {"token_f1": 0.5}, {"token_f1": 0.0}]
    assert compute_metric(cfg, rows) == pytest.approx(0.5)


def test_compute_metric_empty_rows():
    cfg = _make_task_cfg("classification", "accuracy")
    assert compute_metric(cfg, []) == pytest.approx(0.0)


def test_compute_metric_weighted_f1_strict_match():
    """F1 must compare raw labels — no normalisation. Mixed-case labels from
    PolyAI/banking77 (e.g. 'Refund_not_showing_up') stay intact through the
    sklearn call, so a lowercased prediction is correctly counted as wrong."""
    cfg = _make_task_cfg("classification", "weighted_f1", "per_class_weighted")
    rows = [
        {"ground_truth": "Refund_not_showing_up", "predicted_clean": "Refund_not_showing_up"},
        {"ground_truth": "Refund_not_showing_up", "predicted_clean": "refund_not_showing_up"},
        {"ground_truth": "card_arrival", "predicted_clean": "card_arrival"},
    ]
    # 2 of 3 correct → weighted F1 should be 0.8 (perfect on majority class,
    # zero on the lowercased mismatch).
    score = compute_metric(cfg, rows)
    assert score is not None
    assert 0.5 < score < 0.9


def test_compute_classification_metrics_perfect():
    """All-correct → EM=1.0, both F1s=1.0, hallucination=0."""
    from classify_errors import compute_classification_metrics
    rows = [
        {"error_category": "correct", "ground_truth": "positive", "predicted_clean": "positive"},
        {"error_category": "correct", "ground_truth": "negative", "predicted_clean": "negative"},
        {"error_category": "correct", "ground_truth": "neutral", "predicted_clean": "neutral"},
    ]
    out = compute_classification_metrics(rows, ["positive", "negative", "neutral"])
    assert out["exact_match"] == 1.0
    assert out["macro_f1"] == 1.0
    assert out["weighted_f1"] == 1.0
    assert out["hallucination_rate"] == 0.0


def test_compute_classification_metrics_hallucination_is_format_violation_only():
    """Hallucination_rate is the standard ML definition: only format_violation,
    not empty / refusal. Those are tracked separately as empty_rate / refusal_rate.

    A hallucination is the model confidently producing something outside the
    valid label set — refusals and empty outputs are different failure modes."""
    from classify_errors import compute_classification_metrics
    rows = [
        {"error_category": "correct", "ground_truth": "positive", "predicted_clean": "positive"},
        {"error_category": "empty", "ground_truth": "negative", "predicted_clean": "__INVALID__"},
        {"error_category": "refusal", "ground_truth": "neutral", "predicted_clean": "__INVALID__"},
        {"error_category": "format_violation", "ground_truth": "negative", "predicted_clean": "__INVALID__"},
        {"error_category": "format_violation", "ground_truth": "positive", "predicted_clean": "__INVALID__"},
        {"error_category": "wrong_class", "ground_truth": "positive", "predicted_clean": "negative"},
    ]
    out = compute_classification_metrics(rows, ["positive", "negative", "neutral"])
    # 2 of 6 are format_violations (not the broader 4 of 6 that includes empty + refusal)
    assert out["hallucination_rate"] == pytest.approx(2 / 6, abs=1e-4)


def test_compute_classification_metrics_no_valid_labels_returns_none_hallucination():
    """Extraction-style tasks have no closed label list → hallucination_rate is None."""
    from classify_errors import compute_classification_metrics
    rows = [{"error_category": "correct", "ground_truth": "x", "predicted_clean": "x"}]
    out = compute_classification_metrics(rows, None)
    assert out["hallucination_rate"] is None
    # EM and F1 still computable
    assert out["exact_match"] == 1.0


def test_compute_classification_metrics_empty_rows_returns_nones():
    from classify_errors import compute_classification_metrics
    out = compute_classification_metrics([], ["a", "b"])
    assert all(v is None for v in out.values())


def test_compute_classification_metrics_macro_vs_weighted_diverge_on_imbalanced():
    """Macro and weighted F1 should differ when classes are imbalanced and the
    model performs differently on rare vs common classes."""
    from classify_errors import compute_classification_metrics
    # "positive" is 4/5 of the data; model is perfect on positive, fails on negative.
    rows = (
        [{"error_category": "correct", "ground_truth": "positive", "predicted_clean": "positive"}] * 4
        + [{"error_category": "wrong_class", "ground_truth": "negative", "predicted_clean": "positive"}]
    )
    out = compute_classification_metrics(rows, ["positive", "negative"])
    # Weighted-F1 is dominated by the majority class → high. Macro-F1 averages
    # the per-class scores equally → much lower (negative-class F1 is 0).
    assert out["weighted_f1"] > out["macro_f1"]


def _pred_row(**overrides) -> dict:
    base = {
        "id": "r0", "model": "m", "condition": "lora",
        "output": "positive", "ground_truth": "positive",
        "input_tokens": 10, "output_tokens": 1, "reasoning_tokens": 0,
        "latency_ms": 100.0, "ttft_ms": 0.0,
        "timestamp": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_summary_includes_latency_p50_p99_from_predictions(summary_from_predictions):
    """Per-row latency_ms aggregates into p50 / p99 percentile fields."""
    rows = [_pred_row(id=f"r{i}", latency_ms=float(i + 1)) for i in range(100)]
    r = summary_from_predictions(rows, valid_labels=["positive"])
    # linear-interpolation percentile: 50th of 1..100 → 50.5; 99th → 99.01.
    assert r["latency_p50_ms"] == 50.5
    assert r["latency_p99_ms"] == 99.0


def test_summary_aggregates_avg_logprob(summary_from_predictions):
    """avg_logprob aggregates into summary mean + p10."""
    rows = [_pred_row(id=f"r{i}", avg_logprob=-1.0 + i * 0.1) for i in range(10)]
    r = summary_from_predictions(rows, valid_labels=["positive"])
    # Mean of -1.0 .. -0.1 step 0.1 = -0.55; p10 is the worst-confidence bucket.
    assert r["avg_logprob"] == pytest.approx(-0.55, abs=1e-3)
    assert r["p10_logprob"] is not None
    assert r["p10_logprob"] <= r["avg_logprob"]


def test_summary_handles_missing_logprob_gracefully(summary_from_predictions):
    """No prediction row has avg_logprob → summary fields stay None, not 0."""
    r = summary_from_predictions([_pred_row()], valid_labels=["positive"])
    assert r["avg_logprob"] is None
    assert r["p10_logprob"] is None


def test_compute_metric_weighted_f1_invalid_sentinel_counts_as_wrong():
    """Empty/refusal/format_violation rows are mapped to '__INVALID__' so they
    don't accidentally match any real label — they cleanly count as wrong."""
    cfg = _make_task_cfg("classification", "weighted_f1", "per_class_weighted")
    rows = [
        {"ground_truth": "positive", "predicted_clean": "positive"},
        {"ground_truth": "negative", "predicted_clean": "__INVALID__"},
    ]
    score = compute_metric(cfg, rows)
    # One correct, one wrong → F1 < 1.0
    assert score is not None and score < 1.0


# ── classify_predictions (full pipeline) ──────────────────────────────────────

def test_classify_predictions_classification(toy_predictions):
    cfg = _make_task_cfg("classification", "accuracy")
    classified, counts = classify_predictions(toy_predictions, cfg, ["positive", "negative", "neutral"])
    assert len(classified) == len(toy_predictions)
    assert "correct" in counts or "wrong_class" in counts
    assert all("error_category" in r for r in classified)
    assert all("predicted_clean" in r for r in classified)


def test_classify_predictions_sums_to_n(toy_predictions):
    cfg = _make_task_cfg("classification", "accuracy")
    _, counts = classify_predictions(toy_predictions, cfg)
    assert sum(counts.values()) == len(toy_predictions)


def test_classify_predictions_extraction():
    cfg = _make_task_cfg("extraction", "token_f1")
    rows = [
        {"id": "e1", "output": "The answer is yes", "ground_truth": "The answer is yes", "latency_ms": 100},
        {"id": "e2", "output": "", "ground_truth": "something", "latency_ms": 100},
    ]
    classified, counts = classify_predictions(rows, cfg)
    assert classified[0]["error_category"] == "correct"
    assert classified[1]["error_category"] == "empty"


# ── get_valid_labels ───────────────────────────────────────────────────────────

def _write_label_set(repo_root, task_id: str, labels: list[str]) -> None:
    import json
    d = repo_root / "data" / "prepared" / task_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "labels.json").write_text(json.dumps(labels))


def test_get_valid_labels_fpb(tmp_repo_root):
    _write_label_set(tmp_repo_root, "fpb", ["negative", "neutral", "positive"])
    labels = get_valid_labels("fpb")
    assert set(labels) == {"positive", "negative", "neutral"}


def test_get_valid_labels_medmcqa(tmp_repo_root):
    _write_label_set(tmp_repo_root, "medmcqa", ["A", "B", "C", "D"])
    labels = get_valid_labels("medmcqa")
    assert set(labels) == {"A", "B", "C", "D"}


# ── metric_granularity ────────────────────────────────────────────────────────

def test_task_config_default_granularity():
    from classify_errors import TaskConfig
    cfg = TaskConfig(task_id="toy", task_type="classification", metric_id="accuracy")
    assert cfg.metric_granularity == "per_example"


def test_task_config_explicit_granularity():
    from classify_errors import TaskConfig
    cfg = TaskConfig(task_id="toy", task_type="classification", metric_id="weighted_f1",
                     metric_granularity="per_class_weighted")
    assert cfg.metric_granularity == "per_class_weighted"


def test_expected_granularity_map_coverage():
    from classify_errors import _EXPECTED_GRANULARITY
    assert "macro_f1" in _EXPECTED_GRANULARITY
    assert "weighted_f1" in _EXPECTED_GRANULARITY
    assert "accuracy" in _EXPECTED_GRANULARITY
    assert "token_f1" in _EXPECTED_GRANULARITY


def test_granularity_mismatch_emits_warning(capsys):
    from classify_errors import TaskConfig, _EXPECTED_GRANULARITY
    # weighted_f1 implies per_class_weighted; declaring per_example is a mismatch
    cfg = TaskConfig(task_id="toy", task_type="classification", metric_id="weighted_f1",
                     metric_granularity="per_example")
    expected = _EXPECTED_GRANULARITY["weighted_f1"]
    assert expected == "per_class_weighted"
    assert cfg.metric_granularity != expected   # confirms the mismatch


def test_task_configs_granularity_consistent():
    """Real task configs must declare a metric_granularity matching their metric_id."""
    from classify_errors import TaskConfig, _EXPECTED_GRANULARITY, load_task_config
    for task_id in ["banking77", "fpb", "ledgar", "medmcqa", "cuad"]:
        cfg = load_task_config(task_id)
        expected = _EXPECTED_GRANULARITY.get(cfg.metric_id)
        assert expected is not None, f"{task_id}: metric_id {cfg.metric_id!r} not in _EXPECTED_GRANULARITY"
        assert cfg.metric_granularity == expected, (
            f"{task_id}: metric_id={cfg.metric_id!r} implies granularity {expected!r} "
            f"but config declares {cfg.metric_granularity!r}"
        )


# ── aggregate_seed_summaries: metric_cv ───────────────────────────────────────

def _make_seed_summary(metric_value: float) -> dict:
    return {
        "model": "qwen3-8b",
        "task_id": "fpb",
        "condition": "lora",
        "metric_id": "weighted_f1",
        "metric_granularity": "per_class_weighted",
        "metric_value": metric_value,
        "n_predictions": 150,
        "error_counts": {"correct": 100, "wrong_class": 50},
        "eval_axes": [],
        "axis_scores": {},
    }


def test_aggregate_seed_summaries_includes_metric_cv():
    import math
    vals = [0.80, 0.84, 0.82]
    summaries = [_make_seed_summary(v) for v in vals]
    agg = aggregate_seed_summaries(summaries)
    assert "metric_cv" in agg
    assert agg["metric_cv"] is not None
    # Recompute expected CV from raw values (both metric_cv and metric_std are
    # rounded independently, so comparing agg["metric_std"]/agg["metric_mean"]
    # would diverge from the directly-rounded metric_cv).
    n = len(vals)
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
    assert agg["metric_cv"] == round(std / mean, 4)


def test_aggregate_seed_summaries_cv_none_for_zero_mean():
    # If mean is 0 (pathological but possible), CV is undefined
    summaries = [_make_seed_summary(0.0), _make_seed_summary(0.0)]
    agg = aggregate_seed_summaries(summaries)
    assert agg["metric_cv"] is None


def test_aggregate_seed_summaries_sums_reasoning_and_answer_tokens():
    """Token decomposition must be summed across seeds, mirroring single-seed schema."""
    summaries = [
        {**_make_seed_summary(0.80), "total_reasoning_tokens": 0, "total_answer_tokens": 1500},
        {**_make_seed_summary(0.82), "total_reasoning_tokens": 50, "total_answer_tokens": 1450},
    ]
    agg = aggregate_seed_summaries(summaries)
    assert agg["total_reasoning_tokens"] == 50
    assert agg["total_answer_tokens"] == 2950


def test_aggregate_seed_summaries_averages_classification_metrics():
    """EM, macro_f1, weighted_f1, hallucination_rate are averaged across seeds."""
    summaries = [
        {**_make_seed_summary(0.80),
         "exact_match": 0.78, "macro_f1": 0.75, "weighted_f1": 0.80, "hallucination_rate": 0.05},
        {**_make_seed_summary(0.82),
         "exact_match": 0.82, "macro_f1": 0.79, "weighted_f1": 0.82, "hallucination_rate": 0.03},
    ]
    agg = aggregate_seed_summaries(summaries)
    assert agg["exact_match"] == pytest.approx(0.80, abs=1e-4)
    assert agg["macro_f1"] == pytest.approx(0.77, abs=1e-4)
    assert agg["weighted_f1"] == pytest.approx(0.81, abs=1e-4)
    assert agg["hallucination_rate"] == pytest.approx(0.04, abs=1e-4)


def test_aggregate_seed_summaries_missing_classification_metrics_returns_none():
    """Seeds that pre-date the new fields → averaged value is None, not 0."""
    summaries = [_make_seed_summary(0.80), _make_seed_summary(0.82)]
    agg = aggregate_seed_summaries(summaries)
    assert agg["exact_match"] is None
    assert agg["macro_f1"] is None
    assert agg["weighted_f1"] is None
    assert agg["hallucination_rate"] is None


def test_aggregate_seed_summaries_handles_missing_reasoning_fields():
    """Legacy summaries without reasoning fields default to 0 — backwards compat."""
    summaries = [_make_seed_summary(0.80), _make_seed_summary(0.82)]
    agg = aggregate_seed_summaries(summaries)
    assert agg["total_reasoning_tokens"] == 0
    assert agg["total_answer_tokens"] == 0


def test_aggregate_seed_summaries_propagates_granularity():
    summaries = [_make_seed_summary(0.80), _make_seed_summary(0.84)]
    agg = aggregate_seed_summaries(summaries)
    assert agg["metric_granularity"] == "per_class_weighted"


def test_get_valid_labels_reads_sidecar(tmp_repo_root):
    """labels.json content is returned verbatim when present."""
    import json
    task_dir = tmp_repo_root / "data" / "prepared" / "banking77"
    task_dir.mkdir(parents=True, exist_ok=True)
    labels = ["activate_my_card", "card_arrival", "lost_or_stolen_card"]
    (task_dir / "labels.json").write_text(json.dumps(labels))
    assert get_valid_labels("banking77") == labels


def test_get_valid_labels_missing_sidecar_none(tmp_repo_root):
    """Free-form tasks (no labels.json) → None, signalling no format check."""
    (tmp_repo_root / "data" / "prepared" / "cuad").mkdir(parents=True, exist_ok=True)
    assert get_valid_labels("cuad") is None


def test_get_valid_labels_malformed_sidecar_none(tmp_repo_root):
    """Defensive: a corrupt sidecar (not a non-empty list) is treated as absent."""
    task_dir = tmp_repo_root / "data" / "prepared" / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "labels.json").write_text("[]")
    assert get_valid_labels("task") is None
    (task_dir / "labels.json").write_text('{"not": "a list"}')
    assert get_valid_labels("task") is None


# ── compute_axis_scores ────────────────────────────────────────────────────────

_AXIS_DEFS = {
    "accuracy":             {"higher_is_better": True,  "source": "summary"},
    "token_f1":             {"higher_is_better": True,  "source": "summary"},
    "instruction_following":{"higher_is_better": True,  "source": "summary"},
    "latency":              {"higher_is_better": False, "source": "summary"},
    "cost":                 {"higher_is_better": False, "source": "dashboard"},
}


@pytest.mark.parametrize("axis,value", [
    ("accuracy", 0.85),
    ("token_f1", 0.72),
])
def test_compute_axis_scores_metric_value_axes(axis, value):
    summary = {"metric_value": value, "n_predictions": 100, "error_counts": {}}
    scores = compute_axis_scores(summary, [axis], _AXIS_DEFS)
    assert scores[axis]["value"] == pytest.approx(value)
    assert scores[axis]["higher_is_better"] is True


def test_compute_axis_scores_instruction_following():
    summary = {
        "metric_value": 0.85,
        "n_predictions": 100,
        "error_counts": {"correct": 80, "wrong_class": 10, "format_violation": 5, "empty": 3, "refusal": 2},
    }
    scores = compute_axis_scores(summary, ["instruction_following"], _AXIS_DEFS)
    # (5 + 3 + 2) = 10 non-compliant / 100 → 0.9
    assert scores["instruction_following"]["value"] == pytest.approx(0.9)
    assert scores["instruction_following"]["higher_is_better"] is True


def test_compute_axis_scores_instruction_following_perfect():
    summary = {"metric_value": 1.0, "n_predictions": 50, "error_counts": {"correct": 50}}
    scores = compute_axis_scores(summary, ["instruction_following"], _AXIS_DEFS)
    assert scores["instruction_following"]["value"] == pytest.approx(1.0)


def test_compute_axis_scores_instruction_following_clamps_zero():
    # More non-compliant than total is impossible but clamp to 0 defensively
    summary = {"metric_value": 0.0, "n_predictions": 10,
               "error_counts": {"empty": 6, "refusal": 5, "format_violation": 3}}
    scores = compute_axis_scores(summary, ["instruction_following"], _AXIS_DEFS)
    assert scores["instruction_following"]["value"] == pytest.approx(0.0)


def test_compute_axis_scores_latency():
    summary = {"metric_value": 0.85, "n_predictions": 100, "error_counts": {}, "avg_latency_ms": 423.5}
    scores = compute_axis_scores(summary, ["latency"], _AXIS_DEFS)
    assert scores["latency"]["value"] == pytest.approx(423.5)
    assert scores["latency"]["higher_is_better"] is False


@pytest.mark.parametrize("axis,defs", [
    pytest.param("cost", _AXIS_DEFS, id="dashboard_source_excluded"),
    pytest.param("unknown_axis", {}, id="not_in_defs"),
])
def test_compute_axis_scores_axis_excluded(axis, defs):
    summary = {"metric_value": 0.85, "n_predictions": 100, "error_counts": {}}
    assert axis not in compute_axis_scores(summary, [axis], defs)


def test_compute_axis_scores_multiple_axes():
    summary = {
        "metric_value": 0.9,
        "n_predictions": 50,
        "error_counts": {"correct": 45, "format_violation": 2, "empty": 1, "refusal": 2},
        "avg_latency_ms": 200.0,
    }
    scores = compute_axis_scores(summary, ["accuracy", "instruction_following", "latency"], _AXIS_DEFS)
    assert scores["accuracy"]["value"] == pytest.approx(0.9)
    # (2 + 1 + 2) = 5 non-compliant / 50 → 0.9
    assert scores["instruction_following"]["value"] == pytest.approx(0.9)
    assert scores["latency"]["value"] == pytest.approx(200.0)


@pytest.mark.parametrize("axis", ["instruction_following", "latency"])
def test_compute_axis_scores_zero_predictions_returns_none(axis):
    summary = {"metric_value": None, "n_predictions": 0, "error_counts": {}}
    scores = compute_axis_scores(summary, [axis], _AXIS_DEFS)
    assert scores[axis]["value"] is None


# ── _SEMANTIC_ERROR_TYPE / semantic_error_type field ─────────────────────────

def test_semantic_error_type_taxonomy_covers_all_known_categories():
    from classify_errors import _SEMANTIC_ERROR_TYPE
    expected_categories = {
        "correct", "not_applicable", "empty", "format_violation",
        "refusal", "wrong_class", "hallucinated", "partial",
    }
    assert expected_categories <= set(_SEMANTIC_ERROR_TYPE.keys())


@pytest.mark.parametrize("error_category,expected_semantic", [
    ("correct",          "correct"),
    ("not_applicable",   "correct"),
    ("empty",            "instruction_following_failure"),
    ("format_violation", "instruction_following_failure"),
    ("refusal",          "safety_or_alignment_refusal"),
    ("wrong_class",      "factual_error"),
    ("hallucinated",     "factual_error"),
    ("partial",          "extraction_mismatch"),
])
def test_semantic_error_type_mapping(error_category, expected_semantic):
    from classify_errors import _SEMANTIC_ERROR_TYPE
    assert _SEMANTIC_ERROR_TYPE[error_category] == expected_semantic


def test_classify_predictions_sets_semantic_error_type_on_each_row(toy_predictions):
    cfg = _make_task_cfg("classification", "accuracy")
    classified, _ = classify_predictions(toy_predictions, cfg)
    assert all("semantic_error_type" in r for r in classified)
    valid_types = {"correct", "instruction_following_failure",
                   "safety_or_alignment_refusal", "factual_error", "extraction_mismatch", "unknown"}
    assert all(r["semantic_error_type"] in valid_types for r in classified)


def test_classify_predictions_correct_rows_have_correct_semantic(toy_predictions):
    cfg = _make_task_cfg("classification", "accuracy")
    classified, _ = classify_predictions(toy_predictions, cfg)
    for row in classified:
        if row["error_category"] == "correct":
            assert row["semantic_error_type"] == "correct"


def test_classify_predictions_refusal_has_safety_semantic():
    cfg = _make_task_cfg("classification", "accuracy")
    rows = [{"id": "r1", "output": "I cannot answer this.", "ground_truth": "positive", "latency_ms": 100}]
    classified, _ = classify_predictions(rows, cfg)
    assert classified[0]["semantic_error_type"] == "safety_or_alignment_refusal"


# ── semantic_error_counts in summary and aggregation ─────────────────────────

def test_classify_predictions_semantic_counts_sum_to_n(toy_predictions):
    from classify_errors import classify_predictions, _SEMANTIC_ERROR_TYPE
    cfg = _make_task_cfg("classification", "accuracy")
    classified, _ = classify_predictions(toy_predictions, cfg)
    # semantic_error_type is set per-row; counts derived in process_model_task_condition.
    # Check that the per-row field values are consistent with the taxonomy.
    assert all(r["semantic_error_type"] in set(_SEMANTIC_ERROR_TYPE.values()) | {"unknown"}
               for r in classified)


def test_aggregate_seed_summaries_propagates_semantic_error_counts():
    summaries = [
        {**_make_seed_summary(0.80),
         "semantic_error_counts": {"correct": 80, "factual_error": 20}},
        {**_make_seed_summary(0.84),
         "semantic_error_counts": {"correct": 84, "factual_error": 16}},
    ]
    agg = aggregate_seed_summaries(summaries)
    assert "semantic_error_counts" in agg
    assert agg["semantic_error_counts"]["correct"] == 164
    assert agg["semantic_error_counts"]["factual_error"] == 36


def test_aggregate_seed_summaries_missing_semantic_counts_handled():
    # Summaries without semantic_error_counts should not crash aggregation.
    summaries = [_make_seed_summary(0.80), _make_seed_summary(0.84)]
    agg = aggregate_seed_summaries(summaries)
    assert "semantic_error_counts" in agg
    assert agg["semantic_error_counts"] == {}


# ── extract_tagged_answer (CoT <answer> parsing) ──────────────────────────────

def test_extract_tagged_answer_basic():
    assert extract_tagged_answer("<thinking>reasoning</thinking><answer>C</answer>") == "C"


def test_extract_tagged_answer_no_tag_returns_raw():
    """No <answer> tag → raw text passes through (and later fails the closed-set
    check, scoring as a format violation)."""
    assert extract_tagged_answer("the answer is C") == "the answer is C"


def test_extract_tagged_answer_strips_whitespace():
    assert extract_tagged_answer("<answer>  B  </answer>") == "B"


def test_extract_tagged_answer_last_tag_wins():
    """A stray <answer> inside the thinking block must not shadow the real one."""
    text = "<thinking>maybe <answer>A</answer></thinking><answer>D</answer>"
    assert extract_tagged_answer(text) == "D"


def test_extract_tagged_answer_case_insensitive_and_empty():
    assert extract_tagged_answer("<ANSWER>A</ANSWER>") == "A"
    assert extract_tagged_answer("") == ""


# ── classify_predictions with answer_mode="tagged" ────────────────────────────

def _tagged_cfg():
    from classify_errors import TaskConfig
    return TaskConfig(task_id="medmcqa", task_type="classification",
                      metric_id="accuracy", answer_mode="tagged")


def test_classify_predictions_tagged_scores_extracted_answer():
    rows = [
        {"id": "q0", "output": "<thinking>r</thinking><answer>C</answer>", "ground_truth": "C"},
        {"id": "q1", "output": "<thinking>r</thinking><answer>A</answer>", "ground_truth": "B"},
    ]
    classified, _ = classify_predictions(rows, _tagged_cfg(), ["A", "B", "C", "D"])
    assert classified[0]["error_category"] == "correct"
    assert classified[0]["parsed_answer"] == "C"
    assert classified[1]["error_category"] == "wrong_class"


def test_classify_predictions_tagged_missing_tag_is_format_violation():
    """A CoT output that never emits <answer> fails the closed-set check."""
    rows = [{"id": "q0", "output": "I think it is option C honestly", "ground_truth": "C"}]
    classified, _ = classify_predictions(rows, _tagged_cfg(), ["A", "B", "C", "D"])
    assert classified[0]["error_category"] == "format_violation"


# ── precision_at_1 ─────────────────────────────────────────────────────────────

def test_compute_classification_metrics_precision_at_1_equals_em():
    """Precision@1 ≡ EM by construction for a single-best-answer task."""
    from classify_errors import compute_classification_metrics
    rows = [
        {"error_category": "correct", "ground_truth": "A", "predicted_clean": "A"},
        {"error_category": "wrong_class", "ground_truth": "B", "predicted_clean": "A"},
        {"error_category": "correct", "ground_truth": "C", "predicted_clean": "C"},
    ]
    out = compute_classification_metrics(rows, ["A", "B", "C", "D"])
    assert out["precision_at_1"] == out["exact_match"] == pytest.approx(2 / 3, abs=1e-4)


def test_compute_classification_metrics_precision_at_1_none_when_empty():
    from classify_errors import compute_classification_metrics
    assert compute_classification_metrics([], ["A"])["precision_at_1"] is None


# ── sliding-window chunk aggregation ───────────────────────────────────────────

def _chunk(cid, output, gt="the gold clause", logprob=None, in_tok=100, out_tok=10, lat=50.0):
    return {"id": cid, "model": "m", "condition": "lora", "output": output,
            "ground_truth": gt, "input_tokens": in_tok, "output_tokens": out_tok,
            "reasoning_tokens": 0, "latency_ms": lat, "ttft_ms": 5.0,
            "avg_logprob": logprob, "timestamp": "2026-01-01T00:00:00Z"}


def test_is_chunked_detects_chunk_suffix():
    assert is_chunked([_chunk("cuad_test_0000_chunk00", "x")])
    assert not is_chunked([_chunk("fpb_test_0000", "x")])


def test_aggregate_chunk_predictions_groups_by_question():
    preds = [
        _chunk("cuad_test_0000_chunk00", "Not found."),
        _chunk("cuad_test_0000_chunk01", "the gold clause", logprob=-0.2),
        _chunk("cuad_test_0001_chunk00", "Not found."),
    ]
    agg = aggregate_chunk_predictions(preds)
    assert {r["id"] for r in agg} == {"cuad_test_0000", "cuad_test_0001"}


def test_aggregate_chunk_predictions_picks_extraction_over_not_found():
    preds = [
        _chunk("cuad_test_0000_chunk00", "Not found."),
        _chunk("cuad_test_0000_chunk01", "the gold clause", logprob=-0.1),
    ]
    assert aggregate_chunk_predictions(preds)[0]["output"] == "the gold clause"


def test_aggregate_chunk_predictions_picks_most_confident_extraction():
    preds = [
        _chunk("cuad_test_0000_chunk00", "wrong span", logprob=-2.0),
        _chunk("cuad_test_0000_chunk01", "the gold clause", logprob=-0.1),
    ]
    assert aggregate_chunk_predictions(preds)[0]["output"] == "the gold clause"


def test_aggregate_chunk_predictions_sums_tokens_and_latency():
    """Token and latency counts are summed across windows so downstream cost
    stays the true per-question cost of processing the whole contract."""
    preds = [
        _chunk("cuad_test_0000_chunk00", "Not found.", in_tok=100, out_tok=5, lat=40.0),
        _chunk("cuad_test_0000_chunk01", "the gold clause", in_tok=120, out_tok=15, lat=60.0),
    ]
    agg = aggregate_chunk_predictions(preds)[0]
    assert agg["input_tokens"] == 220
    assert agg["output_tokens"] == 20
    assert agg["latency_ms"] == 100.0
    assert agg["n_chunks"] == 2


def test_aggregate_chunk_predictions_all_not_found():
    preds = [
        _chunk("cuad_test_0000_chunk00", "Not found."),
        _chunk("cuad_test_0000_chunk01", "Not found."),
    ]
    agg = aggregate_chunk_predictions(preds)
    assert len(agg) == 1
    assert "not found" in agg[0]["output"].lower()


# ── compute_extraction_metrics (positive / no-answer mix) ─────────────────────

def _ext_row(golds, output, token_f1_val, logprob):
    return {"ground_truth": golds, "output": output,
            "token_f1": token_f1_val, "avg_logprob": logprob}


def test_compute_extraction_metrics_answer_detection():
    """Answer-detection P/R/F1 over the binary clause-present decision."""
    from classify_errors import compute_extraction_metrics
    rows = [
        _ext_row(["the clause"], "the clause", 1.0, -0.1),      # answerable, answered  → TP
        _ext_row(["another clause"], "Not found.", 0.0, -0.5),  # answerable, abstained → FN
        _ext_row(["Not found."], "Not found.", 1.0, -0.2),      # no-answer, abstained  → TN
        _ext_row(["Not found."], "made up clause", 0.0, -2.0),  # no-answer, answered   → FP
    ]
    m = compute_extraction_metrics(rows)
    assert m["answer_detection_precision"] == pytest.approx(0.5)  # TP 1 / (TP 1 + FP 1)
    assert m["answer_detection_recall"] == pytest.approx(0.5)     # TP 1 / (TP 1 + FN 1)
    assert m["answer_detection_f1"] == pytest.approx(0.5)


def test_compute_extraction_metrics_aupr_and_p80_perfect():
    """Every answerable question hit confidently, no false positives → AUPR = 1."""
    from classify_errors import compute_extraction_metrics
    rows = [
        _ext_row(["clause one"], "clause one", 1.0, -0.1),
        _ext_row(["clause two"], "clause two", 1.0, -0.2),
        _ext_row(["Not found."], "Not found.", 1.0, -0.3),
    ]
    m = compute_extraction_metrics(rows)
    assert m["aupr"] == pytest.approx(1.0)
    assert m["precision_at_80_recall"] == pytest.approx(1.0)


def test_compute_extraction_metrics_p80_none_when_recall_unreachable():
    """Two answerable questions, only one ever hit → recall caps at 0.5 → P@80%R undefined."""
    from classify_errors import compute_extraction_metrics
    rows = [
        _ext_row(["clause one"], "clause one", 1.0, -0.1),
        _ext_row(["clause two"], "wrong span", 0.0, -1.0),
    ]
    m = compute_extraction_metrics(rows)
    assert m["precision_at_80_recall"] is None


def test_compute_extraction_metrics_empty_returns_nones():
    from classify_errors import compute_extraction_metrics
    m = compute_extraction_metrics([])
    assert all(v is None for v in m.values())


@pytest.mark.parametrize("text,expected", [
    ("Not found.", True), ("not found", True), ("", True), ("None", True),
    ("no answer", True), ("the governing law clause", False), ("Delaware", False),
])
def test_reads_as_no_answer(text, expected):
    from classify_errors import _reads_as_no_answer
    assert _reads_as_no_answer(text) is expected
