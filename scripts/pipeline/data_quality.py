"""Data quality analysis: length stats, deduplication, near-dup detection, and report assembly."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np
import click
from scipy.special import rel_entr
from scipy.stats import ks_2samp


# ── Raw-dataset row count thresholds ───────────────────────────────────────────

# Minimum row counts for each task's raw splits, used by prepare_datasets to
# fail loudly when a real (non-smoke) prepare reads from a truncated raw
# artifact — e.g. a stale smoke download. Numbers are lower bounds set well
# below actual published dataset sizes.
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "banking77": {"train": 8000,  "test": 2000},
    "cuad":      {"train": 10000, "test": 1000},
    "ledgar":    {"train": 50000, "test": 5000},
    "fpb":       {"train": 3000},
    "medmcqa":   {"train": 100000, "test": 4000},
}


def validate_raw_counts(ds, task_id: str) -> None:
    """Raise loudly if any raw split is below its EXPECTED_COUNTS minimum.

    Use in prepare on a non-smoke run: if the raw artifact was overwritten by
    a smoke download (the silent-corruption bug), the row counts will be far
    below the published sizes. A no-op for tasks without a recorded expectation,
    and for splits not present in the dataset."""
    expected = EXPECTED_COUNTS.get(task_id)
    if not expected:
        return
    for split, n_min in expected.items():
        if split not in ds:
            continue
        actual = len(ds[split])
        if actual < n_min:
            raise RuntimeError(
                f"[{task_id}] Raw split '{split}' has only {actual} rows, "
                f"expected >= {n_min}. The raw dataset at data/raw/{task_id}/ "
                f"may have been clobbered by a smoke download. Re-run "
                f"`python scripts/download_data.py --task {task_id}` to restore it."
            )


# ── Length statistics ──────────────────────────────────────────────────────────

def _percentile(s: list[float], p: float) -> float:
    """Linear-interpolation percentile on a sorted list."""
    if not s:
        return 0.0
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def length_stats(values: list[int]) -> dict:
    """Descriptive statistics for integer lengths."""
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    return {
        "n": n,
        "mean": round(mean, 1),
        "std": round(math.sqrt(variance), 1),
        "min": s[0],
        "p5":  int(_percentile(s, 5)),
        "p25": int(_percentile(s, 25)),
        "p50": int(_percentile(s, 50)),
        "p75": int(_percentile(s, 75)),
        "p95": int(_percentile(s, 95)),
        "max": s[-1],
    }


# ── Normalization ──────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _md5(text: str) -> str:
    return hashlib.md5(_norm(text).encode()).hexdigest()


# ── Exact deduplication ────────────────────────────────────────────────────────

def find_exact_dupes(texts: list[str]) -> list[int]:
    """Return indices of duplicate texts to remove (keeping first occurrence)."""
    seen: dict[str, int] = {}
    remove: list[int] = []
    for i, t in enumerate(texts):
        h = _md5(t)
        if h in seen:
            remove.append(i)
        else:
            seen[h] = i
    return remove


# ── Near-duplicate detection ───────────────────────────────────────────────────

# Shingles appearing in > this fraction of docs are stop-shingles (excluded from index).
_STOP_FRAC = 0.4

# Work budget for near-dup detection. Shingle-Jaccard scans can blow up on long,
# boilerplate-heavy documents (e.g. windowed legal contracts) — past this many
# candidate pairs / comparisons the scan stops and the result is flagged
# `truncated`, a lower bound rather than an unbounded runtime.
_NEAR_DUP_BUDGET = 250_000


def _shingles(text: str, k: int = 3) -> frozenset[str]:
    words = _norm(text).split()
    if len(words) < k:
        return frozenset({" ".join(words)} if words else set())
    return frozenset(" ".join(words[i: i + k]) for i in range(len(words) - k + 1))


def _build_index(
    doc_shingles: list[frozenset[str]], max_df: float
) -> dict[str, list[int]]:
    """Inverted index from shingle → doc indices, skipping stop-shingles."""
    df: dict[str, int] = defaultdict(int)
    for shs in doc_shingles:
        for sh in shs:
            df[sh] += 1
    index: dict[str, list[int]] = defaultdict(list)
    for i, shs in enumerate(doc_shingles):
        for sh in shs:
            if df[sh] < max_df:
                index[sh].append(i)
    return index


def find_near_dupes(
    texts: list[str],
    threshold: float = 0.8,
    shingle_k: int = 3,
) -> dict[str, Any]:
    """Within-split near-dup detection via inverted shingle index + Jaccard.

    Returns n_near_dup_pairs, threshold, and up to 5 example pairs.
    """
    n = len(texts)
    if n < 2:
        return {"n_near_dup_pairs": 0, "threshold": threshold,
                "example_pairs": [], "truncated": False}

    doc_sh = [_shingles(t, shingle_k) for t in texts]
    index = _build_index(doc_sh, n * _STOP_FRAC)

    candidates: set[tuple[int, int]] = set()
    truncated = False
    for docs in index.values():
        if len(candidates) >= _NEAR_DUP_BUDGET:
            truncated = True  # boilerplate-heavy corpus — stop enumerating pairs
            break
        for a in range(len(docs)):
            for b in range(a + 1, len(docs)):
                candidates.add((docs[a], docs[b]))

    pairs: list[tuple[int, int, float]] = []
    for i, j in candidates:
        si, sj = doc_sh[i], doc_sh[j]
        u = si | sj
        jacc = len(si & sj) / len(u) if u else 1.0
        if jacc >= threshold:
            pairs.append((i, j, round(jacc, 3)))

    pairs.sort(key=lambda x: -x[2])
    return {
        "n_near_dup_pairs": len(pairs),
        "threshold": threshold,
        "example_pairs": pairs[:5],
        "truncated": truncated,
    }


def cross_split_near_dupes(
    train_texts: list[str],
    test_texts: list[str],
    threshold: float = 0.9,
    shingle_k: int = 3,
    budget: Optional[int] = None,
) -> dict[str, Any]:
    """Find train indices whose content is near-identical to a test example.

    Uses threshold=0.9 by default (stricter than within-split) to avoid
    over-filtering train examples that are merely topically similar.

    This is a correctness gate against train/eval leakage, so it runs the full
    near-dup scan by default (budget=None). Passing a `budget` caps the number
    of Jaccard comparisons for a lower-bound estimate; `truncated` then reports
    whether the cap was hit.
    Returns train_indices_to_filter, exact_count, near_dup_count, example_pairs.
    """
    if not train_texts or not test_texts:
        return {
            "train_indices_to_filter": [],
            "exact_count": 0, "near_dup_count": 0, "total": 0,
            "threshold": threshold, "example_pairs": [], "truncated": False,
        }

    test_hashes = {_md5(t): j for j, t in enumerate(test_texts)}
    test_sh = [_shingles(t, shingle_k) for t in test_texts]
    n_test = len(test_texts)
    max_df = n_test * _STOP_FRAC
    test_index = _build_index(test_sh, max_df)
    # Distinct "stop" shingles excluded from the index (too common to discriminate).
    # A shared stop-shingle is invisible to `overlap` below, so the prune bound
    # allows for up to n_stop of them — keeping the prune exact (no missed pair).
    _df: Counter = Counter()
    for shs in test_sh:
        _df.update(shs)
    n_stop = sum(1 for c in _df.values() if c >= max_df)

    to_filter: list[int] = []
    exact_count = near_dup_count = 0
    examples: list[tuple[int, int, float]] = []
    comparisons = 0
    truncated = False

    for i, t in enumerate(train_texts):
        # Exact-overlap detection (O(1) per row) always runs; a missed exact
        # cross-split dupe would survive into the train set and abort the run.
        h = _md5(t)
        if h in test_hashes:
            to_filter.append(i)
            exact_count += 1
            examples.append((i, test_hashes[h], 1.0))
            continue

        if budget is not None and comparisons >= budget:
            truncated = True  # caller opted into a capped (lower-bound) scan
            continue

        si = _shingles(t, shingle_k)
        n_si = len(si)
        # Index-visible shared-shingle count per candidate test doc.
        overlap: dict[int, int] = defaultdict(int)
        for sh in si:
            for j in test_index.get(sh, []):
                overlap[j] += 1

        for j, shared in overlap.items():
            # Jaccard >= t requires |A∩B| >= t*max(|A|,|B|); the true intersection
            # is at most `shared + n_stop`. Skip any candidate that cannot reach t
            # — survivors get an exact full-set Jaccard, so no near-dup is missed.
            if shared + n_stop < threshold * max(n_si, len(test_sh[j])):
                continue
            comparisons += 1
            sj = test_sh[j]
            inter = len(si & sj)
            union = n_si + len(sj) - inter
            jacc = inter / union if union else 1.0
            if jacc >= threshold:
                to_filter.append(i)
                near_dup_count += 1
                examples.append((i, j, round(jacc, 3)))
                break

    examples.sort(key=lambda x: -x[2])
    return {
        "train_indices_to_filter": to_filter,
        "exact_count": exact_count,
        "near_dup_count": near_dup_count,
        "total": exact_count + near_dup_count,
        "threshold": threshold,
        "example_pairs": examples[:5],
        "truncated": truncated,
    }


# ── Extreme-length flagging ────────────────────────────────────────────────────

def flag_extreme_length(
    texts: list[str],
    min_chars: int = 10,
    max_chars: Optional[int] = None,
) -> dict[str, Any]:
    """Flag indices that are too short or exceed Tukey's outer fence (Q3 + 3×IQR).

    max_chars overrides the computed fence when provided.
    """
    if not texts:
        return {"too_short": [], "too_long": [], "min_chars": min_chars, "max_chars": max_chars}

    char_lens = [len(t) for t in texts]
    s = sorted(char_lens)
    q1, q3 = _percentile(s, 25), _percentile(s, 75)
    iqr = q3 - q1
    fence = int(q3 + 3.0 * iqr) if iqr > 0 else s[-1]
    effective_max = max_chars if max_chars is not None else fence

    return {
        "too_short": [i for i, t in enumerate(texts) if len(t) < min_chars],
        "too_long":  [i for i, t in enumerate(texts) if len(t) > effective_max],
        "min_chars": min_chars,
        "max_chars": effective_max,
    }


# ── Distribution divergence ────────────────────────────────────────────────────

def kl_divergence(p: dict[str, int], q: dict[str, int]) -> float:
    """KL(P ∥ Q) for discrete label counts using scipy.special.rel_entr.

    Returns float('inf') when P assigns non-zero probability to a class with
    zero probability in Q — this is the mathematically correct value and signals
    that Q is missing a class present in P (a genuine data-quality concern).
    The old epsilon-smoothing approach hid this case with a finite but arbitrary
    large value; surfacing inf makes the problem visible.
    """
    if not p and not q:
        return 0.0
    pt = sum(p.values()) or 1
    qt = sum(q.values()) or 1
    all_keys = sorted(set(p) | set(q))
    p_arr = np.array([p.get(k, 0) / pt for k in all_keys])
    q_arr = np.array([q.get(k, 0) / qt for k in all_keys])
    result = float(rel_entr(p_arr, q_arr).sum())
    return round(result, 4) if math.isfinite(result) else result


def ks_test(a: list[float], b: list[float]) -> dict[str, Optional[float]]:
    """Two-sample KS test via scipy.stats.ks_2samp.

    Returns the KS statistic and an asymptotic p-value. The p-value is
    unreliable for very small samples (n < ~20) but still directionally useful.
    Returns p_value=None when either sample is empty.
    """
    if not a or not b:
        return {"statistic": 0.0, "p_value": None}
    result = ks_2samp(a, b)
    return {
        "statistic": round(float(result.statistic), 4),
        "p_value": round(float(result.pvalue), 4),
    }


# ── Split-level analysis ───────────────────────────────────────────────────────

def analyze_split(
    texts: list[str],
    labels: Optional[list[str]] = None,
    near_dup_threshold: float = 0.8,
) -> dict[str, Any]:
    """Compute length stats, label distribution, and within-split near-dupes."""
    char_lens = [len(t) for t in texts]
    word_lens = [len(t.split()) for t in texts]
    result: dict[str, Any] = {
        "n": len(texts),
        "char_length": length_stats(char_lens),
        "word_length": length_stats(word_lens),
        "within_near_dupes": find_near_dupes(texts, threshold=near_dup_threshold),
    }
    if labels is not None:
        n = len(labels)
        counts = Counter(labels)
        result["label_distribution"] = {
            k: {"count": v, "pct": round(100.0 * v / n, 1)}
            for k, v in sorted(counts.items(), key=str)
        }
        result["n_classes"] = len(counts)
    return result


def cross_split_stats(
    train_texts: list[str],
    test_texts: list[str],
    train_labels: Optional[list[str]],
    test_labels: Optional[list[str]],
) -> dict[str, Any]:
    """Cross-split overlap, label divergence, and length distribution shift."""
    nd = cross_split_near_dupes(train_texts, test_texts, threshold=0.8)
    stats: dict[str, Any] = {
        "exact_overlap": nd["exact_count"],
        "near_dup_pairs": nd["total"],
        "near_dup_threshold": 0.8,
        "near_dup_truncated": nd["truncated"],
        "example_pairs": nd["example_pairs"],
    }
    if train_labels and test_labels:
        p = Counter(train_labels)
        q = Counter(test_labels)
        stats["label_kl_divergence"] = kl_divergence(dict(p), dict(q))
    ks = ks_test(
        [len(t) for t in train_texts],
        [len(t) for t in test_texts],
    )
    stats["length_ks_stat"] = ks["statistic"]
    stats["length_ks_p_value"] = ks["p_value"]
    return stats


# ── Console summary ────────────────────────────────────────────────────────────

def _fmt_dist(dist: dict[str, Any]) -> str:
    items = list(dist.items())
    parts = [f"{k}:{v['count']}({v['pct']}%)" for k, v in items[:8]]
    suffix = f"  +{len(items) - 8} more" if len(items) > 8 else ""
    return "  ".join(parts) + suffix


def _fmt_len(s: dict) -> str:
    if not s or not s.get("n"):
        return "n/a"
    return f"p5={s['p5']}  p50={s['p50']}  p95={s['p95']}  max={s['max']}"


def print_quality_summary(report: dict) -> None:
    sep = "    " + "─" * 60

    def _line(msg: str, warn: bool = False) -> None:
        click.echo(f"{'  WARNING ' if warn else '    '}{msg}", err=warn)

    click.echo(f"\n  Quality Report [{report['task_id']}]")
    click.echo(sep)

    raw = report.get("raw", {})
    _line(f"Raw loaded:   train={raw.get('train_n', '?')}  test={raw.get('test_n', '?')}")

    # Filtering summary
    filt = report.get("filtering", {})
    n_removed = (
        filt.get("extreme_too_short", 0) + filt.get("extreme_too_long", 0)
        + filt.get("train_exact_dupes_removed", 0) + filt.get("cross_split_removed", 0)
        + filt.get("test_exact_dupes_removed", 0)
    )
    if n_removed:
        parts = []
        if filt.get("extreme_too_long"):
            parts.append(f"{filt['extreme_too_long']} extreme-long")
        if filt.get("extreme_too_short"):
            parts.append(f"{filt['extreme_too_short']} too-short")
        if filt.get("train_exact_dupes_removed"):
            parts.append(f"{filt['train_exact_dupes_removed']} train-exact-dupes")
        if filt.get("test_exact_dupes_removed"):
            parts.append(f"{filt['test_exact_dupes_removed']} test-exact-dupes")
        if filt.get("cross_split_removed"):
            parts.append(f"{filt['cross_split_removed']} cross-split-contamination")
        _line(f"Filtered:     {', '.join(parts)}", warn=True)
    else:
        _line("Filtered:     none removed")

    prep = report.get("prepared", {})
    tr, te = prep.get("train", {}), prep.get("test", {})
    _line(f"Prepared:     train={tr.get('n', '?')}  test={te.get('n', '?')}")

    for split, data in (("train", tr), ("test", te)):
        if data.get("label_distribution"):
            _line(f"Labels ({split}):  {_fmt_dist(data['label_distribution'])}")
    for split, data in (("train", tr), ("test", te)):
        if data.get("char_length"):
            _line(f"Char len ({split}): {_fmt_len(data['char_length'])}")

    tr_nd = tr.get("within_near_dupes", {})
    te_nd = te.get("within_near_dupes", {})
    n_tr_nd = tr_nd.get("n_near_dup_pairs", 0)
    n_te_nd = te_nd.get("n_near_dup_pairs", 0)
    thresh = tr_nd.get("threshold", 0.8)
    cap_note = "  [capped — lower bound]" if (tr_nd.get("truncated") or te_nd.get("truncated")) else ""
    if n_tr_nd or n_te_nd:
        _line(f"Near-dupes:   train={n_tr_nd}  test={n_te_nd}  (Jaccard≥{thresh}){cap_note}", warn=True)
    else:
        _line(f"Near-dupes:   none found (Jaccard≥{thresh})")

    cross = prep.get("cross_split", {})
    n_cross = cross.get("near_dup_pairs", 0)
    if n_cross:
        _line(f"Cross-split:  {n_cross} near-dup pair(s) remain after filtering", warn=True)
    kl = cross.get("label_kl_divergence")
    ks = cross.get("length_ks_stat")
    ks_p = cross.get("length_ks_p_value")
    if kl is not None:
        if math.isinf(kl):
            _line("Label KL(train→test): ∞  ← test set contains class absent from train", warn=True)
        else:
            _line(
                f"Label KL(train→test): {kl}{'  ← distribution shift' if kl > 0.2 else ''}",
                warn=kl > 0.2,
            )
    if ks is not None:
        p_str = f"  p={ks_p}" if ks_p is not None else ""
        _line(
            f"Length KS stat:       {ks}{p_str}{'  ← distribution shift' if ks > 0.2 else ''}",
            warn=ks > 0.2,
        )
    click.echo(sep)
