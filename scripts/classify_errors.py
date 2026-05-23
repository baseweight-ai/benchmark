"""Classify prediction errors and compute primary metrics per task."""
from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import click
import yaml
from pydantic import BaseModel

from checkpoint_utils import atomic_write_json
from utils import is_chunked, load_jsonl, load_label_set, question_id, write_jsonl as _write_jsonl
from pipeline.config import get_tasks
from pipeline.paths import classified_path, pred_path, smoke_seg, summary_path

REPO_ROOT = Path(__file__).parent.parent
ALL_TASKS: list[str] = get_tasks()


@lru_cache(maxsize=1)
def _get_axis_definitions() -> dict[str, dict]:
    path = REPO_ROOT / "configs" / "eval_axes.yaml"
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def compute_axis_scores(
    summary: dict,
    eval_axes: list[str],
    axis_defs: dict[str, dict],
) -> dict[str, dict]:
    """Compute per-axis scores from summary data for all summary-source axes.

    Cost axis (source: dashboard) is intentionally excluded — it requires token
    pricing not available at classify time.  generate_dashboard_data.py adds it.
    """
    scores: dict[str, dict] = {}
    n = summary.get("n_predictions", 0)

    for axis in eval_axes:
        defn = axis_defs.get(axis)
        if defn is None or defn.get("source") != "summary":
            continue

        higher = defn.get("higher_is_better", True)

        if axis in ("accuracy", "token_f1"):
            val = summary.get("metric_value")

        elif axis == "instruction_following":
            if n:
                counts = summary.get("error_counts", {})
                non_compliant = (counts.get("empty", 0) + counts.get("refusal", 0)
                                 + counts.get("format_violation", 0))
                val = round(max(0.0, 1.0 - non_compliant / n), 4)
            else:
                val = None

        elif axis == "latency":
            val = summary.get("avg_latency_ms")

        else:
            val = None

        scores[axis] = {"value": val, "higher_is_better": higher}

    return scores


# Maps metric_id → the metric_granularity it implies.
# Used to catch silent mismatches between metric_id and the declared granularity.
_EXPECTED_GRANULARITY: dict[str, str] = {
    "macro_f1":    "per_class_macro",
    "weighted_f1": "per_class_weighted",
    "accuracy":    "per_example",
    "token_f1":    "per_example",
}

# Maps per-example error_category → broad semantic failure type.
# Allows coarser cross-task analysis ("how often does the model confuse facts?"
# vs "how often does it refuse or ignore instructions?") without rerunning classify.
#
# api_error is distinct from instruction_following_failure: the model never got
# to respond — the request errored or the circuit breaker was open. Folding
# transport failures into hallucination/format violations would inflate those
# metrics and slander the model for an outage it never saw.
_SEMANTIC_ERROR_TYPE: dict[str, str] = {
    "correct":          "correct",
    "not_applicable":   "correct",            # model correctly identified no answer
    "api_error":        "provider_error",     # request errored before the model could respond
    "empty":            "instruction_following_failure",
    "format_violation": "instruction_following_failure",
    "refusal":          "safety_or_alignment_refusal",
    "wrong_class":      "factual_error",
    "hallucinated":     "factual_error",
    "partial":          "extraction_mismatch",
}


class TaskConfig(BaseModel):
    task_id: str
    task_type: str
    metric_id: str
    # Declared aggregation level for the primary metric. Must be consistent with
    # metric_id — mismatches are flagged because they silently change what is measured.
    # Values: per_class_weighted | per_class_macro | per_example
    metric_granularity: str = "per_example"
    # direct → the model output IS the label; compare it directly.
    # tagged → the output is a CoT around <answer>X</answer> (medmcqa); the
    #          <answer> payload is extracted before scoring.
    answer_mode: str = "direct"
    eval_axes: list[str] = []


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})



# ── Metric helpers ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for loose matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction: str, ground_truth) -> float:
    """Token-level F1 (whitespace tokenization) for extraction tasks.

    ground_truth may be a single gold string or a list of acceptable gold
    answers; for a list the max F1 over them is returned — the standard
    SQuAD/CUAD treatment of a question that has several equally-valid spans.
    """
    if isinstance(ground_truth, list):
        if not ground_truth:
            return token_f1(prediction, "")
        return max(token_f1(prediction, g) for g in ground_truth)
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = defaultdict(int)
    gold_set = defaultdict(int)
    for t in pred_tokens:
        pred_set[t] += 1
    for t in gold_tokens:
        gold_set[t] += 1
    common = sum(min(pred_set[t], gold_set[t]) for t in pred_set if t in gold_set)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_api_error(text: str) -> bool:
    """True when an output is a recorded exception, not a model response.

    eval_api / eval_local write `ERROR: ...` into the output field when a
    provider call raises (HTTP 4xx, transport failure, circuit breaker open).
    These rows must be distinguished from format violations or hallucinations —
    the model never got to answer, so blaming it for the failure is wrong.
    """
    s = (text or "").lstrip()
    return s.startswith("ERROR:") or s.startswith("ERROR ")


def is_empty(text: str) -> bool:
    return not text.strip()


def is_refusal(text: str) -> bool:
    lower = text.lower()
    refusal_phrases = [
        "i cannot", "i can't", "i'm not able", "i am not able",
        "i'm unable", "i am unable", "i won't", "i will not",
        "i refuse", "as an ai", "i don't feel comfortable",
        "i'm sorry, but i can", "sorry, i can",
    ]
    return any(p in lower for p in refusal_phrases)


def is_format_violation(text: str, valid_labels: Optional[list[str]]) -> bool:
    """Check if output is not in the valid label set — strict case-sensitive.

    Constrained decoding (guided_choice) forces the model to emit one of the
    valid labels verbatim, so any deviation (case, whitespace, extra tokens) is
    a real format violation worth surfacing rather than masking with
    normalisation.
    """
    if valid_labels is None:
        return False
    return text not in valid_labels


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_tagged_answer(text: str) -> str:
    """Pull the <answer>...</answer> payload out of a chain-of-thought output.

    Tagged tasks (medmcqa) train the model to emit
    `<thinking>...</thinking><answer>X</answer>`; scoring uses X, not the whole
    generation. When no <answer> tag is present the raw text is returned
    unchanged — it then fails the closed-set check and is scored as a format
    violation, the correct outcome for an output that ignored the required
    format. The last tag wins, so a stray <answer> inside the thinking block
    does not shadow the real answer.
    """
    matches = _ANSWER_TAG_RE.findall(text or "")
    return matches[-1].strip() if matches else (text or "")


# ── Classification task error classification ───────────────────────────────

def classify_classification(prediction: str, ground_truth: str, valid_labels: Optional[list[str]] = None) -> str:
    """Priority: api_error > empty > refusal > format_violation > correct > wrong_class.

    api_error comes first because a provider failure means the model never
    answered — counting that as a format violation would inflate
    hallucination_rate and instruction_following_failure with transport
    outages.

    Correctness for non-error rows is strict, case-sensitive equality. With
    guided_choice active, the model's output is constrained to one of the exact
    label strings, so `prediction == ground_truth` is the right test — no
    lowercasing, no punctuation stripping, no whitespace collapse.
    """
    if is_api_error(prediction):
        return "api_error"
    if is_empty(prediction):
        return "empty"
    if is_refusal(prediction):
        return "refusal"
    if valid_labels and is_format_violation(prediction, valid_labels):
        return "format_violation"
    if prediction == ground_truth:
        return "correct"
    return "wrong_class"


# ── Extraction task error classification ──────────────────────────────────

def classify_extraction(prediction: str, ground_truth, f1_threshold_partial: float = 0.5) -> str:
    """Priority: api_error > empty > format_violation > correct > partial > hallucinated > not_applicable.

    api_error comes first (provider failure, not a model output). The rest
    of the priority order matches single-answer extraction scoring.
    ground_truth may be a list of valid gold spans (a CUAD question can have
    several); the prediction is then classified against whichever span it
    matches best (max-F1), mirroring multi-answer extraction scoring.
    """
    if is_api_error(prediction):
        return "api_error"
    if isinstance(ground_truth, list):
        ground_truth = (max(ground_truth, key=lambda g: token_f1(prediction, g))
                        if ground_truth else "")
    if is_empty(prediction):
        return "empty"

    # format_violation: longer than 5× the expected or contains structured tokens not in ground truth
    if len(prediction.split()) > max(5 * len(ground_truth.split()), 100) and ground_truth != "Not found.":
        return "format_violation"

    # not_applicable: ground truth is "Not found." and model also says not found
    gt_norm = normalize_text(ground_truth)
    pred_norm = normalize_text(prediction)
    not_found_phrases = ["not found", "no answer", "n/a", "none", "not applicable", "not mentioned"]
    gt_is_not_found = any(p in gt_norm for p in not_found_phrases) or gt_norm in ("not found.", "not found")
    pred_is_not_found = any(p in pred_norm for p in not_found_phrases)

    if gt_is_not_found and pred_is_not_found:
        return "not_applicable"

    # hallucinated: ground truth is "Not found." but model gives an answer
    if gt_is_not_found and not pred_is_not_found and len(prediction.strip()) > 5:
        return "hallucinated"

    f1 = token_f1(prediction, ground_truth)
    if f1 >= 0.9:
        return "correct"
    if f1 >= f1_threshold_partial:
        return "partial"
    return "hallucinated"



# ── Bootstrap CI and multi-seed aggregation ───────────────────────────────

def bootstrap_ci(
    values: list[float], n_boot: int = 5000, ci: float = 0.95
) -> tuple[float, float]:
    """Return (lower, upper) percentile bootstrap CI for the mean."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v
    rng = random.Random(0)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_boot)
    )
    alpha = (1 - ci) / 2
    return means[int(alpha * n_boot)], means[int((1 - alpha) * n_boot)]


def aggregate_seed_summaries(summaries: list[dict]) -> dict:
    """Aggregate summaries from multiple eval seeds into mean ± std + CI."""
    metric_values = [s["metric_value"] for s in summaries if s.get("metric_value") is not None]
    if not metric_values:
        return {}
    n = len(metric_values)
    mean = sum(metric_values) / n
    variance = sum((v - mean) ** 2 for v in metric_values) / n
    std = math.sqrt(variance)
    ci_lo, ci_hi = bootstrap_ci(metric_values)
    # Aggregate error counts by summing across seeds
    all_counts: dict[str, int] = defaultdict(int)
    for s in summaries:
        for k, v in s.get("error_counts", {}).items():
            all_counts[k] += v
    all_semantic_counts: dict[str, int] = defaultdict(int)
    for s in summaries:
        for k, v in s.get("semantic_error_counts", {}).items():
            all_semantic_counts[k] += v
    base = summaries[0]  # representative row for metadata

    eval_axes = base.get("eval_axes", [])
    agg_axis_scores: dict[str, dict] = {}
    axis_defs = _get_axis_definitions()
    for axis in eval_axes:
        vals = [
            s["axis_scores"][axis]["value"]
            for s in summaries
            if s.get("axis_scores", {}).get(axis, {}).get("value") is not None
        ]
        if vals:
            higher = axis_defs.get(axis, {}).get("higher_is_better", True)
            agg_axis_scores[axis] = {"value": round(sum(vals) / len(vals), 4), "higher_is_better": higher}

    # Coefficient of variation = std/mean: how stable the metric is across evaluation
    # seeds. A stability measure, not an effect size. High CV (>0.05) suggests the
    # metric is sensitive to test-set sampling, weakening per-seed comparisons.
    metric_cv = round(std / mean, 4) if mean != 0 else None

    def _mean_field(key: str) -> Optional[float]:
        # Excludes seeds where the field is None (legacy summaries).
        vals = [s[key] for s in summaries if s.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _sum_optional(key: str) -> Optional[int]:
        # Sum across only the seeds that recorded the field. None when no
        # seed has it (e.g. legacy summaries, or local conditions without an
        # envelope-wrapping response_format).
        vals = [s[key] for s in summaries if s.get(key) is not None]
        return sum(vals) if vals else None

    return {
        "model": base.get("model"),
        "task_id": base.get("task_id"),
        "condition": base.get("condition"),
        "n_seeds": n,
        "seed_metric_values": metric_values,
        "metric_id": base.get("metric_id"),
        "metric_granularity": base.get("metric_granularity"),
        "metric_value": round(mean, 4),
        "metric_mean": round(mean, 4),
        "metric_std": round(std, 4),
        "metric_cv": metric_cv,
        "metric_ci_lo": round(ci_lo, 4),
        "metric_ci_hi": round(ci_hi, 4),
        "exact_match": _mean_field("exact_match"),
        "precision_at_1": _mean_field("precision_at_1"),
        "macro_f1": _mean_field("macro_f1"),
        "weighted_f1": _mean_field("weighted_f1"),
        "hallucination_rate": _mean_field("hallucination_rate"),
        "api_error_rate": _mean_field("api_error_rate"),
        "answer_detection_precision": _mean_field("answer_detection_precision"),
        "answer_detection_recall": _mean_field("answer_detection_recall"),
        "answer_detection_f1": _mean_field("answer_detection_f1"),
        "precision_at_80_recall": _mean_field("precision_at_80_recall"),
        "aupr": _mean_field("aupr"),
        "n_predictions": sum(s.get("n_predictions", 0) for s in summaries),
        # Token totals summed across seeds so the dashboard can still derive
        # cost from an aggregated (_agg) summary, exactly as for a single seed.
        "total_input_tokens": sum(s.get("total_input_tokens", 0) or 0 for s in summaries),
        "total_output_tokens": sum(s.get("total_output_tokens", 0) or 0 for s in summaries),
        "total_reasoning_tokens": sum(s.get("total_reasoning_tokens", 0) or 0 for s in summaries),
        "total_answer_tokens": sum(s.get("total_answer_tokens", 0) or 0 for s in summaries),
        "total_answer_only_tokens": _sum_optional("total_answer_only_tokens"),
        "total_envelope_overhead_tokens": _sum_optional("total_envelope_overhead_tokens"),
        # Summed across seeds so the dashboard derives self-hosted cost from
        # MEASURED throughput (gpu_hourly * wall / n). Summed ONLY when EVERY
        # seed recorded wall-time: n_predictions sums all seeds, so a partial
        # wall sum (one seed's .wall.json missing) over the full n would
        # understate cost. None → dashboard falls back to its flat estimate.
        "eval_wall_time_s": (
            sum(s["eval_wall_time_s"] for s in summaries)
            if all(s.get("eval_wall_time_s") is not None for s in summaries) else None
        ),
        "error_counts": dict(all_counts),
        "semantic_error_counts": dict(all_semantic_counts),
        "prompt_sha": base.get("prompt_sha"),
        "few_shot_hash": base.get("few_shot_hash"),
        "eval_axes": eval_axes,
        "axis_scores": agg_axis_scores,
    }


# ── Sliding-window chunk aggregation ───────────────────────────────────────

def _reads_as_no_answer(text: str) -> bool:
    """True when an extraction output (or gold) signals 'no clause is present'."""
    norm = normalize_text(str(text))
    if not norm:
        return True
    return (norm == "none"
            or any(p in norm for p in ("not found", "no answer", "not applicable", "not mentioned")))


def aggregate_chunk_predictions(predictions: list[dict]) -> list[dict]:
    """Collapse sliding-window chunk predictions into one row per question.

    Sliding-window tasks (CUAD) emit one prediction per context window, with ids
    `..._chunkNN`. This regroups windows by question and picks a single answer:
    the most confident window (highest avg_logprob) that returned a real
    extraction, falling back to "Not found." when no window did. Token, latency
    and reasoning counts are SUMMED across every window so downstream cost stays
    the true per-question cost of running the model over the whole contract.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for r in predictions:
        base = question_id(r.get("id", ""))
        if base not in groups:
            order.append(base)
        groups[base].append(r)

    def _errored(c: dict) -> bool:
        return is_api_error(c.get("output", ""))

    aggregated: list[dict] = []
    for base in order:
        chunks = groups[base]
        # Prefer a window that returned a real extraction; among those, the most
        # confident (highest avg_logprob). Else the first non-errored window
        # ("Not found."), else the first window.
        extracted = [c for c in chunks
                     if not _errored(c) and not _reads_as_no_answer(c.get("output", ""))]
        if extracted:
            chosen = max(
                extracted,
                key=lambda c: c["avg_logprob"] if c.get("avg_logprob") is not None else float("-inf"),
            )
        else:
            non_errored = [c for c in chunks if not _errored(c)]
            chosen = non_errored[0] if non_errored else chunks[0]

        agg = dict(chosen)
        agg["id"] = base
        agg["input_tokens"] = sum(c.get("input_tokens", 0) for c in chunks)
        agg["output_tokens"] = sum(c.get("output_tokens", 0) for c in chunks)
        agg["reasoning_tokens"] = sum(c.get("reasoning_tokens", 0) or 0 for c in chunks)
        agg["latency_ms"] = sum(c.get("latency_ms", 0) or 0 for c in chunks)
        agg["ttft_ms"] = chunks[0].get("ttft_ms", 0.0)
        agg["n_chunks"] = len(chunks)
        aggregated.append(agg)
    return aggregated


# ── Primary metric computation ─────────────────────────────────────────────

def compute_metric(task_cfg: TaskConfig, classified_rows: list[dict]) -> Optional[float]:
    """Compute primary metric value from classified predictions."""
    metric = task_cfg.metric_id

    if metric in ("macro_f1", "weighted_f1"):
        from sklearn.metrics import f1_score
        average = "weighted" if metric == "weighted_f1" else "macro"
        y_true, y_pred = [], []
        for r in classified_rows:
            # Strict match: both sides are raw label strings. predicted_clean
            # is the raw prediction for valid rows, "__INVALID__" sentinel for
            # empty/refusal/format_violation rows so they count as wrong without
            # accidentally colliding with a real label.
            y_true.append(r["ground_truth"])
            y_pred.append(r["predicted_clean"])
        try:
            score = f1_score(y_true, y_pred, average=average, zero_division=0)
        except Exception:
            score = None
        return score

    if metric == "accuracy":
        correct = sum(1 for r in classified_rows if r["error_category"] == "correct")
        return correct / len(classified_rows) if classified_rows else 0.0

    if metric == "token_f1":
        scores = [r.get("token_f1", 0.0) for r in classified_rows]
        return sum(scores) / len(scores) if scores else 0.0

    return None


def compute_classification_metrics(
    classified_rows: list[dict], valid_labels: Optional[list[str]]
) -> dict:
    """Compute EM, Precision@1, macro/weighted F1, and hallucination_rate.

    All are always computed for closed-set classification regardless of the
    task's primary metric_id, so the dashboard can surface them side-by-side.
    Returns Nones for tasks where the metric doesn't apply (e.g. extraction).

      EM             = correct / n_predictions (strict label equality).
      Precision@1    = correct / n_predictions. For a single-best-answer task
                       each example yields exactly one prediction and one gold
                       label, so "relevant items in the top-1 result" reduces to
                       the EM indicator — Precision@1 equals EM by construction.
                       Reported under its own name because medmcqa lists it as a
                       primary metric.
      Macro-F1       = unweighted mean of per-class F1 (rare classes equal vote).
      Weighted-F1    = per-class F1 weighted by class support in y_true.
      Hallucination  = format_violation / n — fraction of outputs that look
                       like an attempted answer but aren't in the target label
                       set. Standard ML usage: the model confidently invented
                       something outside the allowed vocabulary. Refusals
                       ("I cannot...") and empty outputs are tracked separately
                       in refusal_rate / empty_rate, not folded in here.
                       None when the task has no closed label set.
    """
    n = len(classified_rows)
    if n == 0:
        return {k: None for k in _CLASSIFICATION_METRIC_KEYS}

    correct = sum(1 for r in classified_rows if r.get("error_category") == "correct")
    exact_match = round(correct / n, 4)
    # Precision@1 ≡ EM for a single-best-answer task: one prediction, one gold.
    precision_at_1 = exact_match

    try:
        from sklearn.metrics import f1_score
        y_true = [r["ground_truth"] for r in classified_rows]
        y_pred = [r["predicted_clean"] for r in classified_rows]
        macro = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        weighted = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
    except Exception:
        macro = None
        weighted = None

    # hallucination_rate measures the model's tendency to invent labels —
    # transport failures (api_error) never reached the model, so they're
    # excluded from BOTH numerator and denominator. Reported separately as
    # api_error_rate so a broken provider stays visible.
    api_errors = sum(1 for r in classified_rows if r.get("error_category") == "api_error")
    api_error_rate = round(api_errors / n, 4)

    if valid_labels is None:
        hallucination = None
    else:
        format_violations = sum(
            1 for r in classified_rows
            if r.get("error_category") == "format_violation"
        )
        responded = n - api_errors
        hallucination = round(format_violations / responded, 4) if responded else None

    return {
        "exact_match": exact_match,
        "precision_at_1": precision_at_1,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "hallucination_rate": hallucination,
        "api_error_rate": api_error_rate,
    }


# ── Extraction metrics (positive / no-answer mix) ──────────────────────────

_EXTRACTION_METRIC_KEYS = (
    "answer_detection_precision", "answer_detection_recall", "answer_detection_f1",
    "precision_at_80_recall", "aupr",
)

# token-F1 at/above which an extraction counts as a real "hit" for Precision@Recall.
_EXTRACTION_HIT_THRESHOLD = 0.5


# Keys returned by compute_classification_metrics. Defined once so the
# zero-row early return and the non-empty path can't drift apart silently.
_CLASSIFICATION_METRIC_KEYS = (
    "exact_match", "precision_at_1", "macro_f1", "weighted_f1",
    "hallucination_rate", "api_error_rate",
)


def compute_extraction_metrics(classified_rows: list[dict]) -> dict:
    """Metrics for an extraction task with a positive / no-answer mix (CUAD).

    answer_detection_{precision,recall,f1}: the binary "is this clause present
      at all?" decision — the model "says present" when it returns a
      non-"Not found." extraction.
    precision_at_80_recall / aupr: rank questions by the model's confidence
      (avg_logprob), sweep an accept threshold down the ranking, and trace a
      precision-recall curve over the answerable questions. A question is a hit
      when the model attempts an extraction and scores token-F1 >= 0.5. AUPR is
      the area under that curve; precision_at_80_recall is the (interpolated)
      best precision at recall >= 0.8, or None if 0.8 recall is unreachable.
    Returns Nones for every key when there are no rows.
    """
    if not classified_rows:
        return {k: None for k in _EXTRACTION_METRIC_KEYS}

    # (gold_answerable, pred_answered, hit, confidence) per question.
    items: list[tuple[bool, bool, bool, Optional[float]]] = []
    for r in classified_rows:
        golds = r.get("ground_truth", [])
        if isinstance(golds, str):
            golds = [golds]
        gold_answerable = bool(golds) and not all(_reads_as_no_answer(g) for g in golds)
        pred_answered = not _reads_as_no_answer(r.get("output", ""))
        hit = gold_answerable and pred_answered and r.get("token_f1", 0.0) >= _EXTRACTION_HIT_THRESHOLD
        items.append((gold_answerable, pred_answered, hit, r.get("avg_logprob")))

    # ── answer-detection precision / recall / F1 ───────────────────────────
    tp = sum(1 for a, p, _, _ in items if a and p)
    fp = sum(1 for a, p, _, _ in items if not a and p)
    fn = sum(1 for a, p, _, _ in items if a and not p)
    det_p = tp / (tp + fp) if (tp + fp) else 0.0
    det_r = tp / (tp + fn) if (tp + fn) else 0.0
    det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) else 0.0

    # ── confidence-ranked Precision@Recall / AUPR ──────────────────────────
    total_answerable = sum(1 for a, _, _, _ in items if a)
    p_at_80: Optional[float] = None
    aupr: Optional[float] = None
    if total_answerable:
        # Rank by confidence (descending); missing confidence sorts last.
        ranked = sorted(
            items, reverse=True,
            key=lambda it: (it[3] is not None, it[3] if it[3] is not None else 0.0),
        )
        curve: list[tuple[float, float]] = []  # (recall, precision)
        accepted = accepted_hits = 0
        for _, pred_answered, hit, _ in ranked:
            if not pred_answered:
                continue  # an abstention is never an accepted extraction
            accepted += 1
            accepted_hits += int(hit)
            curve.append((accepted_hits / total_answerable, accepted_hits / accepted))
        if curve:
            at80 = [prec for rec, prec in curve if rec >= 0.80]
            p_at_80 = round(max(at80), 4) if at80 else None
            pts = [(0.0, curve[0][1])] + curve
            aupr = round(sum((pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2
                             for i in range(1, len(pts))), 4)
        else:
            aupr = 0.0

    return {
        "answer_detection_precision": round(det_p, 4),
        "answer_detection_recall": round(det_r, 4),
        "answer_detection_f1": round(det_f1, 4),
        "precision_at_80_recall": p_at_80,
        "aupr": aupr,
    }


# ── Main classification loop ───────────────────────────────────────────────

def classify_predictions(
    predictions: list[dict],
    task_cfg: TaskConfig,
    valid_labels: Optional[list[str]] = None,
) -> tuple[list[dict], dict]:
    """Classify all predictions and return (enriched_rows, summary_counts)."""
    classified = []
    counts: dict[str, int] = defaultdict(int)

    for row in predictions:
        pred = row.get("output", "")
        gt = row.get("ground_truth", "")
        enriched = dict(row)

        if task_cfg.task_type == "classification":
            # Tagged (CoT) tasks emit <thinking>...</thinking><answer>X</answer> —
            # score the extracted <answer>, not the whole generation.
            scored = extract_tagged_answer(pred) if task_cfg.answer_mode == "tagged" else pred
            cat = classify_classification(scored, gt, valid_labels)
            enriched["error_category"] = cat
            # predicted_clean is the (extracted) prediction for valid rows,
            # "__INVALID__" sentinel for everything that didn't yield a usable
            # label — empty / refusal / format_violation / api_error. F1 treats
            # the sentinel as a strict mismatch, so transport failures count
            # against the model in the headline metric (intentionally — a
            # benchmark that hides API downtime as "didn't happen" would be
            # misleading), but `hallucination_rate` excludes them via
            # compute_classification_metrics (api errors aren't hallucinations).
            enriched["predicted_clean"] = (
                scored if cat not in ("empty", "refusal", "format_violation", "api_error")
                else "__INVALID__"
            )
            if task_cfg.answer_mode == "tagged":
                enriched["parsed_answer"] = scored

        elif task_cfg.task_type == "extraction":
            cat = classify_extraction(pred, gt)
            enriched["error_category"] = cat
            enriched["token_f1"] = token_f1(pred, gt)

        else:
            cat = "unknown"
            enriched["error_category"] = cat

        enriched["semantic_error_type"] = _SEMANTIC_ERROR_TYPE.get(cat, "unknown")
        counts[cat] += 1
        classified.append(enriched)

    return classified, dict(counts)


def process_model_task_condition(
    model_short: str,
    task_id: str,
    condition: str,
    task_cfg: TaskConfig,
    valid_labels: Optional[list[str]],
    dry_run: bool,
    source: str = "local",
    smoke: bool = False,
) -> Optional[dict]:
    """Classify one predictions file and write summary.

    condition is the filename stem, which may include a _seedN suffix.
    """
    input_path = pred_path(REPO_ROOT, source, model_short, task_id, condition, smoke=smoke)
    # Logical condition name strips any _seedN suffix for storage in summary.
    base_condition = condition.split("_seed")[0] if "_seed" in condition else condition
    label = f"{source}/{model_short}/{task_id}/{condition}"
    if not input_path.exists():
        return None

    predictions = load_jsonl(input_path)
    if not predictions:
        click.echo(f"  SKIP [{label}]: empty predictions file")
        return None

    if dry_run:
        click.echo(f"  [dry-run] Would classify {len(predictions)} predictions for {label}")
        return {}

    # Sliding-window tasks (CUAD) emit one prediction per context window;
    # collapse them to one row per question before classifying so metrics and
    # cost are per-question, not per-window.
    if is_chunked(predictions):
        n_windows = len(predictions)
        predictions = aggregate_chunk_predictions(predictions)
        click.echo(f"  [{label}] aggregated {n_windows} window predictions → {len(predictions)} questions")

    classified, counts = classify_predictions(predictions, task_cfg, valid_labels)

    classified_out = classified_path(REPO_ROOT, source, model_short, task_id, condition, smoke=smoke)
    _write_jsonl(classified, classified_out)

    # Enforce consistency between metric_id and the declared metric_granularity.
    # A mismatch means the config says one thing but the computation does another,
    # which silently changes what is being measured across runs or comparisons.
    expected_gran = _EXPECTED_GRANULARITY.get(task_cfg.metric_id)
    if expected_gran and task_cfg.metric_granularity != expected_gran:
        click.echo(
            f"  WARNING [{task_id}]: metric_id={task_cfg.metric_id!r} implies "
            f"granularity {expected_gran!r} but config declares "
            f"metric_granularity={task_cfg.metric_granularity!r}. "
            "These measure different things — fix the task config.",
            err=True,
        )

    metric_value = compute_metric(task_cfg, classified)

    if task_cfg.task_type == "classification":
        classification_metrics = compute_classification_metrics(classified, valid_labels)
    else:
        classification_metrics = {
            "exact_match": None, "precision_at_1": None, "macro_f1": None,
            "weighted_f1": None, "hallucination_rate": None,
            "api_error_rate": round(
                sum(1 for r in classified if r.get("error_category") == "api_error")
                / max(1, len(classified)), 4
            ),
        }

    if task_cfg.task_type == "extraction":
        extraction_metrics = compute_extraction_metrics(classified)
    else:
        extraction_metrics = {k: None for k in _EXTRACTION_METRIC_KEYS}

    from pipeline.data_quality import _percentile

    def _pct(values: list[float], p: float) -> Optional[float]:
        return _percentile(sorted(values), p) if values else None

    latencies = [r["latency_ms"] for r in predictions if r.get("latency_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in predictions if r.get("ttft_ms", 0) > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    ttft_p50 = _pct(ttfts, 50)
    ttft_p95 = _pct(ttfts, 95)
    latency_p50 = _pct(latencies, 50)
    latency_p99 = _pct(latencies, 99)

    total_input_tokens = sum(r.get("input_tokens", 0) for r in predictions)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in predictions)
    total_reasoning_tokens = sum(r.get("reasoning_tokens", 0) or 0 for r in predictions)
    total_answer_tokens = total_output_tokens - total_reasoning_tokens
    # When the API wraps answers in a JSON envelope (response_format), the
    # API-reported output_tokens include the wrapper. answer_only_tokens
    # (counted via tiktoken in eval_api) holds the bare-label count; the
    # complement is the envelope overhead. Local (guided_choice) and free-form
    # outputs don't emit a JSON wrapper, so envelope_overhead is ~0 there.
    has_answer_only = any("answer_only_tokens" in r for r in predictions)
    total_answer_only_tokens = (
        sum(r.get("answer_only_tokens", 0) or 0 for r in predictions)
        if has_answer_only else None
    )
    total_envelope_overhead_tokens = (
        max(0, total_answer_tokens - total_answer_only_tokens)
        if total_answer_only_tokens is not None else None
    )

    logprobs = [r["avg_logprob"] for r in predictions if r.get("avg_logprob") is not None]
    avg_logprob = round(sum(logprobs) / len(logprobs), 4) if logprobs else None
    p10_logprob = round(_pct(logprobs, 10), 4) if logprobs else None

    # Wall time: prefer the sidecar written by eval_local.py (vLLM batch processing
    # collapses per-row timestamps to the same millisecond, making the span useless).
    # Fall back to timestamp-derived span for API predictions which lack a sidecar.
    wall_sidecar = input_path.with_suffix(".wall.json")
    eval_wall_time_s = None
    gpu_model = None
    try:
        with open(wall_sidecar) as _wf:
            sidecar = json.load(_wf)
            eval_wall_time_s = sidecar.get("eval_wall_time_s")
            gpu_model = sidecar.get("gpu_model")
    except FileNotFoundError:
        from datetime import datetime, timezone as tz
        timestamps = [r["timestamp"] for r in predictions if r.get("timestamp")]
        if len(timestamps) >= 2:
            def _parse(ts: str) -> datetime:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elapsed = (_parse(max(timestamps)) - _parse(min(timestamps))).total_seconds()
            rounded = round(elapsed, 1)
            eval_wall_time_s = rounded if rounded > 0 else None

    # Aggregate error categories into semantic failure types for coarser cross-task reporting
    semantic_counts: dict[str, int] = defaultdict(int)
    for row in classified:
        semantic_counts[row.get("semantic_error_type", "unknown")] += 1

    n = len(predictions)
    # Derived compliance/error rates. Denominator is responses (n − api_errors),
    # not total — a transport failure isn't a refusal/empty/format-violation by
    # the model. api_error_rate stays as fraction of total so the provider's
    # error rate stays visible.
    api_error_n = counts.get("api_error", 0)
    responded = max(1, n - api_error_n)
    format_violation_n = counts.get("format_violation", 0)
    refusal_n = counts.get("refusal", 0)
    empty_n = counts.get("empty", 0)
    partial_n = counts.get("partial", 0)
    format_compliance_rate = round(1 - format_violation_n / responded, 4) if n else None
    refusal_rate = round(refusal_n / responded, 4) if n else None
    empty_rate = round(empty_n / responded, 4) if n else None
    partial_rate = round(partial_n / responded, 4) if n else None
    api_error_rate = round(api_error_n / n, 4) if n else None

    # Per-class breakdown (classification only — used for reporting without rerunning)
    per_class_metrics: Optional[dict] = None
    if task_cfg.task_type == "classification":
        _per_class: dict = defaultdict(lambda: {"correct": 0, "total": 0})
        for row in classified:
            gt = str(row.get("ground_truth", ""))
            _per_class[gt]["total"] += 1
            if row.get("error_category") == "correct":
                _per_class[gt]["correct"] += 1
        per_class_metrics = {
            cls: {
                "correct": d["correct"],
                "total": d["total"],
                "accuracy": round(d["correct"] / d["total"], 4) if d["total"] else None,
            }
            for cls, d in sorted(_per_class.items())
        }

    # Propagate reproducibility fields from first prediction row
    first = predictions[0] if predictions else {}
    prompt_sha = first.get("prompt_sha")
    few_shot_hash = first.get("few_shot_hash")
    eval_seed = first.get("eval_seed", 0)

    summary = {
        "model": model_short,
        "task_id": task_id,
        "condition": base_condition,
        "eval_seed": eval_seed,
        "prompt_sha": prompt_sha,
        "few_shot_hash": few_shot_hash,
        "n_predictions": n,
        "metric_id": task_cfg.metric_id,
        "metric_granularity": task_cfg.metric_granularity,
        "metric_value": round(metric_value, 4) if metric_value is not None else None,
        # Classification-only fields below are None for extraction tasks.
        "exact_match": classification_metrics["exact_match"],
        "precision_at_1": classification_metrics["precision_at_1"],
        "macro_f1": classification_metrics["macro_f1"],
        "weighted_f1": classification_metrics["weighted_f1"],
        "hallucination_rate": classification_metrics["hallucination_rate"],
        "api_error_rate": api_error_rate,
        # Extraction-only fields below are None for classification tasks.
        "answer_detection_precision": extraction_metrics["answer_detection_precision"],
        "answer_detection_recall": extraction_metrics["answer_detection_recall"],
        "answer_detection_f1": extraction_metrics["answer_detection_f1"],
        "precision_at_80_recall": extraction_metrics["precision_at_80_recall"],
        "aupr": extraction_metrics["aupr"],
        "error_counts": counts,
        "semantic_error_counts": dict(semantic_counts),
        "format_compliance_rate": format_compliance_rate,
        "refusal_rate": refusal_rate,
        "empty_rate": empty_rate,
        "partial_rate": partial_rate,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
        "latency_p50_ms": round(latency_p50, 1) if latency_p50 is not None else None,
        "latency_p99_ms": round(latency_p99, 1) if latency_p99 is not None else None,
        "ttft_p50_ms": round(ttft_p50, 1) if ttft_p50 is not None else None,
        "ttft_p95_ms": round(ttft_p95, 1) if ttft_p95 is not None else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_answer_tokens": total_answer_tokens,
        "total_answer_only_tokens": total_answer_only_tokens,
        "total_envelope_overhead_tokens": total_envelope_overhead_tokens,
        "avg_logprob": avg_logprob,
        "p10_logprob": p10_logprob,
        "eval_wall_time_s": eval_wall_time_s,
        "gpu_model": gpu_model,
        "per_class_metrics": per_class_metrics,
    }

    summary["eval_axes"] = task_cfg.eval_axes
    summary["axis_scores"] = compute_axis_scores(summary, task_cfg.eval_axes, _get_axis_definitions())

    # condition_key is the filename stem (may include _seedN suffix)
    summary_out = summary_path(REPO_ROOT, source, model_short, task_id, condition, smoke=smoke)
    atomic_write_json(summary, summary_out)
    metric_str = f"{metric_value:.4f}" if metric_value is not None else "N/A"
    click.echo(f"  [{label}] {task_cfg.metric_id}={metric_str} counts={counts}")
    # Surface semantic failure breakdown when anything other than "correct" is present.
    semantic_failures = {k: v for k, v in semantic_counts.items() if k != "correct"}
    if semantic_failures:
        click.echo(f"  [{label}] semantic failures: {dict(semantic_failures)}")
    return summary


def get_valid_labels(task_id: str) -> Optional[list[str]]:
    """Return valid output labels for format-violation checking, or None.

    Present for every closed-set classification task (banking77, fpb, ledgar,
    medmcqa) and absent for free-form tasks (cuad).
    """
    return load_label_set(REPO_ROOT, task_id)


def _print_classify_summary(summaries: list[dict]) -> None:
    """Print a cross-task summary table after classify_errors finishes."""
    sep = "  " + "─" * 70
    click.echo(f"\n  Classification Summary  ({len(summaries)} run(s))")
    click.echo(sep)
    click.echo(f"  {'run':<40}  {'metric':>7}  {'n':>5}  issues")
    click.echo(sep)
    for s in sorted(summaries, key=lambda x: (x.get("task_id", ""), x.get("condition", ""), x.get("model", ""))):
        label = f"{s.get('model','?')}/{s.get('task_id','?')}/{s.get('condition','?')}"[:40]
        mv = s.get("metric_value")
        metric_str = f"{mv:.4f}" if mv is not None else "   n/a"
        n = s.get("n_predictions", 0)
        ec = s.get("error_counts", {})
        issues = " ".join(f"{k}={v}" for k, v in sorted(ec.items()) if k != "correct" and v > 0)
        click.echo(f"  {label:<40}  {metric_str:>7}  {n:>5}  {issues}")
    click.echo(sep)


@click.command()
@click.option("--model", default="all", help="Model short name or 'all' (ignored for api source)")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="Condition or 'all'")
@click.option("--source", default="all", help="Prediction source: local|api|all")
@click.option("--dry-run", is_flag=True)
@click.option("--smoke-test", is_flag=True,
              help="Classify smoke-namespaced predictions; outputs stay under results/smoke/.")
def main(model: str, task: str, condition: str, source: str, dry_run: bool, smoke_test: bool) -> None:
    """Classify prediction errors and compute primary metrics."""
    pred_root = REPO_ROOT / "results" / smoke_seg(smoke_test) / "predictions"
    sources = ["local", "api", "lm_eval"] if source == "all" else [source]

    task_ids = ALL_TASKS if task == "all" else [task]

    if condition == "all":
        conditions = ["zero-shot", "5-shot", "lora"]
    else:
        conditions = [condition]

    failures = []
    processed = 0
    all_summaries: list[dict] = []

    for src in sources:
        src_root = pred_root / src

        if model == "all":
            model_shorts = sorted(d.name for d in src_root.iterdir() if d.is_dir()) if src_root.exists() else []
        else:
            model_shorts = [model]

        for ms in model_shorts:
            for tid in task_ids:
                try:
                    task_cfg = load_task_config(tid)
                    valid_labels = get_valid_labels(tid)
                except Exception as exc:
                    click.echo(f"  ERROR: could not load task config {tid}: {exc}", err=True)
                    failures.append((f"{src}/{ms}/{tid}", str(exc)))
                    continue

                # Collect base conditions plus any seed-specific files in the predictions dir
                pred_dir = pred_root / src / ms / tid
                seed_conds: list[str] = []
                if pred_dir.exists():
                    for f in sorted(pred_dir.glob("*_seed*.jsonl")):
                        stem = f.stem
                        if stem not in seed_conds:
                            seed_conds.append(stem)

                all_conds = list(conditions) + [c for c in seed_conds if c not in conditions]

                # Base conditions that have ANY seed1+ resamples. For those bases
                # the unsuffixed result IS seed 0 and must be aggregated alongside
                # seed1+ — otherwise headline mean/std/CI are computed on
                # seeds 1+2 only (silently dropping seed 0).
                bases_with_resamples = {c.split("_seed")[0] for c in seed_conds}

                # Track summaries by base condition for aggregation
                seed_summaries: dict[str, list[dict]] = defaultdict(list)

                for cond in all_conds:
                    try:
                        result = process_model_task_condition(
                            ms, tid, cond, task_cfg, valid_labels, dry_run, source=src, smoke=smoke_test
                        )
                        if result is not None:
                            processed += 1
                            all_summaries.append(result)
                            # Seed1+ files contribute under their stripped base; the
                            # unsuffixed file contributes under itself only when seed
                            # resamples exist for that base (i.e. this is seed 0 of
                            # a multi-seed run, not a single-seed condition).
                            if "_seed" in cond:
                                seed_summaries[cond.split("_seed")[0]].append(result)
                            elif cond in bases_with_resamples:
                                seed_summaries[cond].append(result)
                    except Exception as exc:
                        click.echo(f"  ERROR [{src}/{ms}/{tid}/{cond}]: {exc}", err=True)
                        import traceback; traceback.print_exc()
                        failures.append((f"{src}/{ms}/{tid}/{cond}", str(exc)))

                # Aggregate multi-seed summaries per base condition
                for base_cond, seed_summs in seed_summaries.items():
                    if len(seed_summs) < 2:
                        continue
                    if dry_run:
                        continue
                    try:
                        agg = aggregate_seed_summaries(seed_summs)
                        agg["condition"] = base_cond
                        agg_out = summary_path(REPO_ROOT, src, ms, tid, f"{base_cond}_agg", smoke=smoke_test)
                        atomic_write_json(agg, agg_out)
                        click.echo(
                            f"  [{src}/{ms}/{tid}/{base_cond}] aggregated {len(seed_summs)} seeds → "
                            f"mean={agg['metric_mean']:.4f} ±{agg['metric_std']:.4f} "
                            f"[{agg['metric_ci_lo']:.4f}, {agg['metric_ci_hi']:.4f}]"
                        )
                    except Exception as exc:
                        click.echo(f"  WARN: seed aggregation failed for {src}/{ms}/{tid}/{base_cond}: {exc}", err=True)

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    if processed == 0:
        click.echo("ERROR: no prediction files found — nothing was classified.", err=True)
        sys.exit(1)

    if all_summaries and not dry_run:
        _print_classify_summary(all_summaries)

    click.echo(f"Classified {processed} prediction file(s).")


if __name__ == "__main__":
    main()
