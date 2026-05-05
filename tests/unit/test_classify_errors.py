"""Unit tests for classify_errors.py."""
import pytest

from classify_errors import (
    classify_classification,
    classify_extraction,
    classify_predictions,
    compute_axis_scores,
    compute_metric,
    get_valid_labels,
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


def test_is_format_violation_case_insensitive():
    assert not is_format_violation("POSITIVE", ["positive", "negative", "neutral"])


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


# ── compute_metric ─────────────────────────────────────────────────────────────

def _make_task_cfg(task_type: str, metric_id: str):
    from classify_errors import TaskConfig
    return TaskConfig(task_id="toy", task_type=task_type, metric_id=metric_id)


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

def test_get_valid_labels_fpb():
    labels = get_valid_labels("fpb")
    assert set(labels) == {"positive", "negative", "neutral"}


def test_get_valid_labels_medmcqa():
    labels = get_valid_labels("medmcqa")
    assert set(labels) == {"A", "B", "C", "D"}


def test_get_valid_labels_banking77_none():
    assert get_valid_labels("banking77") is None


def test_get_valid_labels_cuad_none():
    assert get_valid_labels("cuad") is None


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
