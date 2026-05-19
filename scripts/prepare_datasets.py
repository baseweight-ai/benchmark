"""Prepare datasets for training and evaluation: split, sample, format, save JSONL."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click
import yaml
from pydantic import BaseModel

# load_from_disk (via HuggingFace datasets) calls tqdm's thread_map internally,
# which modifies tqdm class-level lock state in a non-thread-safe way.
# Serializing calls with this lock prevents the AttributeError on concurrent loads.
_load_from_disk_lock = threading.Lock()

from checkpoint_utils import atomic_write_json
from pipeline.cache import (
    code_closure_hash, dict_hash, file_content_hash, inputs_changed, rows_sha, tree_hash,
)
from pipeline.config import get_tasks
from pipeline.paths import prompt_path
from pipeline.validation import check_contamination, reject_test_path, require_dir, validate_dataset

REPO_ROOT = Path(__file__).parent.parent
ALL_TASKS: list[str] = get_tasks()

SMOKE_TRAIN_N = 20
SMOKE_TEST_N = 10

# Regex to extract CUAD clause type from question text.
_CUAD_CLAUSE_RE = re.compile(r'related to "([^"]+)"', re.IGNORECASE)


# ── Config models ──────────────────────────────────────────────────────────

class SamplingConfig(BaseModel):
    strategy: str                       # "balanced" | "stratified"
    stratify_by: str                    # field name in the raw row dict
    total_cap: Optional[int] = None     # stratified: target total count
    per_group_cap: Optional[int] = None # balanced: max examples per group
    min_per_group: int = 1              # stratified: floor per group
    seed: int = 42
    # Optional boolean field to balance 50/50 on (e.g. has_answer): the sample
    # is split evenly between the two groups, each independently stratified.
    balance_by: Optional[str] = None


class TaskConfig(BaseModel):
    task_id: str
    task_name: str
    dataset_path: str
    dataset_config: Optional[str] = None
    task_type: str
    metric_id: str
    max_output_tokens: int
    text_field: Optional[str] = None
    label_field: Optional[str] = None
    label_type: Optional[str] = None
    custom_label_names: Optional[list[str]] = None
    split_ratios: Optional[list[float]] = None
    split_seed: Optional[int] = None
    # Sliding-window context management (CUAD): context_max_tokens is the window
    # size in words, context_stride_tokens the step between windows, and
    # max_chunks the cap on windows per test contract.
    context_max_tokens: Optional[int] = None
    context_stride_tokens: Optional[int] = None
    max_chunks: int = 12
    test_split: str = "test"
    train_sampling: Optional[SamplingConfig] = None
    test_sampling: Optional[SamplingConfig] = None
    # Fraction of the sampled training pool held out as a stratified validation
    # split (val.jsonl), used for early stopping and overfitting detection.
    val_ratio: float = 0.1
    val_seed: int = 42


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})


def load_prompt(task_id: str) -> tuple[dict, str]:
    """Return (prompt_dict, sha256[:16]) for reproducible prompt versioning."""
    raw = prompt_path(REPO_ROOT, task_id).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    return json.loads(raw), sha


# ── Formatting helpers ─────────────────────────────────────────────────────

def format_user(prompt: dict, row: dict) -> str:
    template = prompt["user_template"]
    if "text_fields" in prompt:
        fields = {f: row.get(f, "") for f in prompt["text_fields"]}
    else:
        field = prompt.get("text_field", "text")
        fields = {field: row.get(field, "")}
    return template.format(**fields)


def format_gold(prompt: dict, row: dict, label_names: Optional[list[str]] = None) -> str:
    """Return the bare gold answer/label for a row.

    This is the exact string a prediction is scored against (the eval label).
    For `cot_letter` it is the answer letter only — no <thinking> block — since
    that is what classify_errors extracts from the model output and compares.
    """
    lf = prompt.get("label_format")
    if lf in ("letter", "cot_letter"):
        label_map = prompt.get("label_map", {"0": "A", "1": "B", "2": "C", "3": "D"})
        val = row.get(prompt.get("label_field", "cop"), 0)
        return label_map.get(str(val), str(val))
    if lf == "extractive":
        answers = row.get(prompt.get("answer_field", "answers"), {})
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        return texts[0] if texts else "Not found."
    # verbatim
    val = row.get(prompt.get("label_field", "label"), "")
    if isinstance(val, int) and label_names:
        return label_names[val]
    return str(val)


# Char cap on the <thinking> explanation in a cot_letter target. The raw
# MedMCQA `exp` field has a heavy tail — some entries are multi-thousand-token
# textbook dumps. Left uncapped, those targets exceed the eval-time
# max_output_tokens budget (512), so generation is truncated before the
# <answer> tag and at train time the answer tag can fall past max_seq_length.
# ~1400 chars keeps the whole completion well under 512 tokens; only the long
# tail (~p90+) is trimmed, at a word boundary so the target stays clean.
_COT_EXP_MAX_CHARS = 1400

# Minimum <thinking> explanation length for a usable CoT target. MedMCQA's
# `exp` is frequently a placeholder (".") or a bare answer restatement
# ("Methyldopa", "Ans. C"); 15 chars is the observed ceiling of those
# non-reasoning values, so a shorter `exp` is dropped from training.
_COT_EXP_MIN_CHARS = 15


def format_assistant(prompt: dict, row: dict, label_names: Optional[list[str]] = None) -> str:
    """Return the assistant training target for a row.

    Identical to format_gold except for `cot_letter`, where the target is a
    chain-of-thought (`<thinking>{explanation}</thinking>`) followed by the
    answer tag (`<answer>{letter}</answer>`). Completion-only SFT then teaches
    the model to reason inside <thinking> before committing to <answer>. The
    explanation is length-capped (see _COT_EXP_MAX_CHARS).
    """
    if prompt.get("label_format") == "cot_letter":
        letter = format_gold(prompt, row, label_names)
        exp = str(row.get(prompt.get("explanation_field", "exp"), "") or "").strip()
        if len(exp) > _COT_EXP_MAX_CHARS:
            exp = exp[:_COT_EXP_MAX_CHARS].rsplit(" ", 1)[0].rstrip()
        return f"<thinking>{exp}</thinking><answer>{letter}</answer>"
    return format_gold(prompt, row, label_names)


def format_eval_label(prompt: dict, row: dict, label_names: Optional[list[str]] = None):
    """Return the gold answer(s) a prediction is scored against.

    For extraction tasks a single question can have several equally-valid gold
    spans, so this returns the full list of acceptable answers — token_f1 then
    takes the max over them (SQuAD/CUAD-style multi-answer scoring). For
    closed-set tasks it returns the single label string, same as format_gold.
    """
    if prompt.get("label_format") == "extractive":
        answers = row.get(prompt.get("answer_field", "answers"), {})
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        texts = [t for t in texts if t and str(t).strip()]
        return texts or ["Not found."]
    return format_gold(prompt, row, label_names)


def to_chat(system: str, user: str, assistant: Optional[str] = None) -> dict:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


def get_label_set(prompt: dict, label_names: Optional[list[str]]) -> Optional[list[str]]:
    """Return the closed answer set for this task, or None for free-form tasks.

    The set holds the valid answers, not the whole output: for `verbatim`/`letter`
    it is what the model emits directly; for `cot_letter` it is the valid
    <answer> values (A/B/C/D), since the model emits a CoT around them. Written
    to labels.json and consumed by classify_errors (format-violation checks) and,
    for answer_mode==direct tasks only, by eval as the constrained-decoding set.
    Order is preserved when known (ClassLabel.names ordering) as it can carry
    semantic meaning.
    """
    lf = prompt.get("label_format")
    if lf in ("letter", "cot_letter"):
        label_map = prompt.get("label_map", {"0": "A", "1": "B", "2": "C", "3": "D"})
        # label_map values are the rendered tokens (A/B/C/D); de-dup while preserving order.
        seen: dict[str, None] = {}
        for v in label_map.values():
            seen.setdefault(str(v), None)
        return list(seen)
    if lf == "verbatim" and label_names:
        return list(label_names)
    return None


# ── Sampling ───────────────────────────────────────────────────────────────

def sample(
    data: list[dict],
    strategy: str,
    stratify_by: str,
    seed: int = 42,
    total_cap: Optional[int] = None,
    per_group_cap: Optional[int] = None,
    min_per_group: int = 1,
    balance_by: Optional[str] = None,
) -> list[dict]:
    """Balanced: per_group_cap rows per group. Stratified: total_cap rows via LRM allocation.

    balance_by: when set, the sample is first split 50/50 on this boolean field
    (e.g. has_answer) — total_cap/2 rows from each side, each side independently
    sampled by `strategy`/`stratify_by` — then interleaved.
    """
    if not data:
        return []

    if balance_by is not None:
        if total_cap is None:
            raise ValueError("balance_by requires total_cap")
        partitions: dict[bool, list[dict]] = {True: [], False: []}
        for row in data:
            partitions[bool(row.get(balance_by))].append(row)
        half = total_cap // 2
        out: list[dict] = []
        for key in (True, False):
            out.extend(sample(
                partitions[key], strategy, stratify_by, seed,
                total_cap=half, per_group_cap=per_group_cap, min_per_group=min_per_group,
            ))
        random.Random(seed).shuffle(out)
        return out

    rng = random.Random(seed)

    groups: dict[Any, list[dict]] = defaultdict(list)
    for row in data:
        groups[row.get(stratify_by)].append(row)
    sorted_keys = sorted(groups.keys(), key=str)

    if strategy == "balanced":
        if per_group_cap is None:
            raise ValueError("balanced strategy requires per_group_cap")
        result: list[dict] = []
        for k in sorted_keys:
            pool = list(groups[k])
            rng.shuffle(pool)
            result.extend(pool[:per_group_cap])
        rng.shuffle(result)
        return result

    if strategy == "stratified":
        if total_cap is None:
            raise ValueError("stratified strategy requires total_cap")

        group_sizes = {k: len(groups[k]) for k in sorted_keys}
        total_available = sum(group_sizes.values())
        total = min(total_cap, total_available)

        targets = {k: total * (group_sizes[k] / total_available) for k in sorted_keys}
        allocs: dict[Any, int] = {k: max(min_per_group, math.floor(targets[k])) for k in sorted_keys}

        remainder = total - sum(allocs.values())

        if remainder > 0:
            # largest-remainder method: give +1 to groups with the biggest fractional part
            by_frac = sorted(sorted_keys, key=lambda k: targets[k] - math.floor(targets[k]), reverse=True)
            for k in by_frac[:remainder]:
                allocs[k] += 1
        elif remainder < 0:
            # min_per_group floors pushed us over total; trim from groups with most slack
            over = -remainder
            for k in sorted(sorted_keys, key=lambda k: allocs[k] - min_per_group, reverse=True):
                if over <= 0:
                    break
                slack = allocs[k] - min_per_group
                cut = min(slack, over)
                if cut > 0:
                    allocs[k] -= cut
                    over -= cut

        result = []
        for k in sorted_keys:
            pool = list(groups[k])
            rng.shuffle(pool)
            result.extend(pool[:min(allocs[k], group_sizes[k])])
        rng.shuffle(result)
        return result

    raise ValueError(f"Unknown sampling strategy: {strategy!r}. Expected 'balanced' or 'stratified'.")


# ── Context-window management: sliding-window chunking ─────────────────────

def sliding_windows(
    text: str, window: int, stride: int, max_chunks: Optional[int] = None
) -> list[str]:
    """Split text into overlapping fixed-size windows (whitespace-tokenised).

    Each window is `window` words; consecutive windows start `stride` words
    apart, so stride < window produces overlap. Returns at most `max_chunks`
    windows when set. Text shorter than one window yields a single window.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    words = text.split()
    if len(words) <= window:
        return [" ".join(words)]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + window]))
        if start + window >= len(words):
            break
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        start += stride
    return chunks


def _answer_window(context: str, answer: str, window: int) -> str:
    """Return a `window`-word slice of context positioned to contain `answer`.

    Used for CUAD training positives — the gold clause must sit inside the
    context for a completion-only SFT target to be learnable. Falls back to the
    head window when the answer text cannot be located verbatim.
    """
    words = context.split()
    if len(words) <= window:
        return " ".join(words)
    char_pos = context.find(answer)
    if char_pos < 0:
        return " ".join(words[:window])
    word_idx = len(context[:char_pos].split())
    # Start slightly before the clause so it keeps some left-context, while
    # keeping the whole window inside the document.
    start = max(0, min(word_idx - window // 8, len(words) - window))
    return " ".join(words[start:start + window])


def chunk_cuad_train(
    rows: list[dict], window: int, stride: int, seed: int = 42
) -> list[dict]:
    """Reduce each CUAD training row to a single sliding-window example.

    A positive question yields its answer-bearing window — a random grid window
    that contains the clause, drawn from the same fixed grid the test side uses
    so the clause sits at the same range of positions at train and eval (no
    answer-centred position bias); an answer-snapped window is the fallback when
    the clause fits no grid window. A no-answer question yields a random grid
    window with target "Not found.". Abstention is taught by these real
    no-answer questions — the dataset is balanced 50/50 by stratified sampling —
    not by synthetic answer-free windows.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for r in rows:
        context = r.get("context", "") or ""
        answers = r.get("answers", {})
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        gold = texts[0].strip() if texts and texts[0] else ""
        windows = sliding_windows(context, window, stride)
        if gold:
            # Windows are whitespace-normalised (" ".join of split words);
            # normalise the gold the same way so the substring test is reliable
            # and the SFT target matches the window text verbatim.
            gold_norm = " ".join(gold.split())
            containing = [w for w in windows if gold_norm in w]
            pos_window = rng.choice(containing) if containing else _answer_window(context, gold, window)
            out.append({**r, "context": pos_window,
                        "answers": {"text": [gold_norm], "answer_start": [0]}})
        else:
            out.append({**r, "context": rng.choice(windows),
                        "answers": {"text": [], "answer_start": []}})
    return out


def chunk_cuad_test(
    rows: list[dict], window: int, stride: int, max_chunks: int
) -> list[dict]:
    """Expand each CUAD test question into one row per sliding window.

    Each emitted row carries a stable `_eval_id` of the form
    `cuad_test_{q:04d}_chunk{c:02d}`. classify_errors strips the `_chunkNN`
    suffix to aggregate the per-window predictions back to one score per
    question.
    """
    out: list[dict] = []
    for q_idx, r in enumerate(rows):
        windows = sliding_windows(r.get("context", "") or "", window, stride, max_chunks)
        for c_idx, w in enumerate(windows):
            out.append({**r, "context": w,
                        "_eval_id": f"cuad_test_{q_idx:04d}_chunk{c_idx:02d}"})
    return out


# ── JSONL writing ──────────────────────────────────────────────────────────

def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    click.echo(f"  Written {len(rows)} rows to {path.relative_to(REPO_ROOT)}")


def _log_distribution(rows: list[dict], key: str, label: str) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key, "?"))
        counts[k] = counts.get(k, 0) + 1
    n = len(rows)
    if len(counts) <= 10:
        parts = " | ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        click.echo(f"    {label}: {parts}  (n={n})")
    else:
        mn, mx = min(counts.values()), max(counts.values())
        click.echo(f"    {label}: {len(counts)} groups, n={n}, min={mn}, max={mx}")


# ── Per-task preprocessing ─────────────────────────────────────────────────

def process_task(cfg: TaskConfig, dry_run: bool, smoke_test: bool = False) -> None:
    from datasets import load_from_disk  # lazy
    label = " (smoke)" if smoke_test else ""
    click.echo(f"\n[{cfg.task_id}] Processing {cfg.task_name}{label}...")

    raw_dir = REPO_ROOT / "data" / "raw" / cfg.task_id
    out_dir = REPO_ROOT / "data" / "prepared" / cfg.task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        click.echo(f"  [dry-run] Would process {cfg.task_id}")
        return

    require_dir(raw_dir, min_files=1, desc=f"raw data for {cfg.task_id}")
    prompt, prompt_sha = load_prompt(cfg.task_id)

    # Skip-if-unchanged: re-derive only when the raw data, this task's config,
    # the prompt, or the preparation code itself changed since the last run.
    task_yaml = REPO_ROOT / "configs" / "tasks" / f"{cfg.task_id}.yaml"
    fingerprint = dict_hash({
        "code": code_closure_hash(Path(__file__)),
        "raw": tree_hash(raw_dir),
        "config": file_content_hash(task_yaml) if task_yaml.is_file() else "",
        "prompt_sha": prompt_sha,
        "smoke_test": smoke_test,
    })
    out_file = out_dir / ("smoke_test.jsonl" if smoke_test else "test.jsonl")
    meta_file = out_dir / "dataset_meta.json"
    if out_file.exists() and not inputs_changed(fingerprint, meta_file):
        click.echo(f"  [{cfg.task_id}] SKIP: prepared data up-to-date (inputs unchanged)")
        return

    system = prompt["system"]
    click.echo(f"  [{cfg.task_id}] Loading dataset from disk...")
    with _load_from_disk_lock:
        ds = load_from_disk(str(raw_dir))
    click.echo(f"  [{cfg.task_id}] Dataset loaded — splits: {list(ds.keys())}")

    # ── Split into train / test ────────────────────────────────────────────
    if cfg.task_id == "fpb":
        # FPB has no test split on HF; derive train/test from the single train split.
        all_rows = list(ds["train"])
        label_names = cfg.custom_label_names or ["negative", "neutral", "positive"]
        rng = random.Random(cfg.split_seed or 42)
        rng.shuffle(all_rows)
        if smoke_test:
            half = len(all_rows) // 2
            train_rows = all_rows[:half]
            test_rows = all_rows[half:]
            click.echo(f"  FPB smoke split: train={len(train_rows)}, test={len(test_rows)}")
        else:
            n = len(all_rows)
            n_test = int(n * 0.15)
            train_rows = all_rows[:-n_test]  # 85% train
            test_rows = all_rows[-n_test:]
            click.echo(f"  FPB split: train={len(train_rows)}, test={len(test_rows)}")
    else:
        label_names = None
        split_name = cfg.test_split
        train_rows = list(ds["train"]) if "train" in ds else []
        if split_name in ds:
            test_rows = list(ds[split_name])
        else:
            test_rows = train_rows[-100:]

    # ── Label names for integer-mapped tasks ───────────────────────────────
    if cfg.label_type == "integer_mapped" and cfg.task_id != "fpb":
        try:
            split_key = "train" if "train" in ds else list(ds.keys())[0]
            feats = ds[split_key].features
            lf = cfg.label_field or "label"
            if hasattr(feats.get(lf, None), "names"):
                label_names = feats[lf].names
            else:
                label_names = cfg.custom_label_names
        except Exception:
            label_names = cfg.custom_label_names

    # ── MedMCQA: keep single-answer questions; CoT needs an explanation ────
    if cfg.task_id == "medmcqa":
        def _is_single(r: dict) -> bool:
            return str(r.get("choice_type", "")).strip().lower() == "single"
        n_tr, n_te = len(train_rows), len(test_rows)
        train_rows = [r for r in train_rows if _is_single(r)]
        test_rows  = [r for r in test_rows  if _is_single(r)]
        click.echo(f"  [medmcqa] choice_type=single filter: "
                   f"train {n_tr}→{len(train_rows)}, test {n_te}→{len(test_rows)}")
        # The CoT target embeds `exp` in the <thinking> block. An empty `exp`
        # teaches an empty thinking block; a too-short `exp` (placeholder ".",
        # bare answer restatement) teaches a no-op reasoning step — both are
        # degenerate CoT targets. Drop either from TRAIN only; the test set is
        # scored on the answer letter and never needs `exp`.
        exp_field = prompt.get("explanation_field", "exp")
        n_exp = len(train_rows)
        train_rows = [
            r for r in train_rows
            if len(str(r.get(exp_field, "") or "").strip()) >= _COT_EXP_MIN_CHARS
        ]
        if len(train_rows) != n_exp:
            click.echo(f"  [medmcqa] dropped {n_exp - len(train_rows)} train rows "
                       f"with no usable explanation (CoT needs an `exp` of "
                       f">= {_COT_EXP_MIN_CHARS} chars)")

    # ── CUAD: flatten SQuAD format and sliding-window chunk the train side ──
    if cfg.task_id == "cuad":
        click.echo(f"  [cuad] Flattening SQuAD format "
                   f"({len(train_rows)} train, {len(test_rows)} test rows)...")
        def flatten_squad(rows: list[dict]) -> list[dict]:
            out = []
            for row in rows:
                # Keep full context — truncating here would drop clauses;
                # sliding-window chunking handles the context budget instead.
                ctx = row.get("context", "") or ""
                answers = row.get("answers", {})
                texts = answers.get("text", []) if isinstance(answers, dict) else []
                question = row.get("question", "")
                # The CUAD question is a templated sentence carrying the clause
                # type in quotes (... related to "Governing Law" ...) — extract it.
                m = _CUAD_CLAUSE_RE.search(question)
                clause_type = m.group(1) if m else (question.strip() or "unknown")
                # All valid gold spans retained — a CUAD question can have
                # several equally-correct clause spans, and test scoring takes
                # the max F1 over them (see format_eval_label / token_f1).
                kept = [t for t in texts if t and t.strip()]
                out.append({
                    "context": ctx,
                    "question": question,
                    "clause_type": clause_type,
                    "answers": {"text": kept, "answer_start": [0] * len(kept)},
                    "id": row.get("id", ""),
                    # has_answer drives the 50/50 positive / no-answer balancing:
                    # CUAD is a full contract x clause grid and roughly half of
                    # the (contract, clause) pairs have no clause of that type.
                    "has_answer": bool(kept),
                })
            return out
        train_rows = flatten_squad(train_rows)
        test_rows = flatten_squad(test_rows)
        all_cuad = train_rows + test_rows
        unknown_count = sum(1 for r in all_cuad if r.get("clause_type") == "unknown")
        if all_cuad and unknown_count / len(all_cuad) > 0.05:
            click.echo(
                f"  WARNING [cuad]: {unknown_count}/{len(all_cuad)} rows "
                f"({unknown_count / len(all_cuad):.1%}) have unknown clause_type — check _CUAD_CLAUSE_RE",
                err=True,
            )
        n_pos = sum(1 for r in all_cuad if r["has_answer"])
        click.echo(f"  [cuad] {n_pos}/{len(all_cuad)} rows have an answer "
                   f"({100 * n_pos / max(len(all_cuad), 1):.0f}%); the rest are no-answer questions")

        # Sliding-window chunking, training side: one window per question — the
        # answer-bearing window for a positive, a random window for a no-answer
        # question. The 50/50 positive/no-answer balance is enforced by the
        # stratified sampler (train_sampling.balance_by: has_answer).
        cuad_window = cfg.context_max_tokens or 750
        cuad_stride = cfg.context_stride_tokens or cuad_window
        train_rows = chunk_cuad_train(train_rows, cuad_window, cuad_stride, seed=42)
        click.echo(f"  [cuad] Train chunking: one {cuad_window}w window per question "
                   f"({len(train_rows)} questions)")

    from pipeline.data_quality import (  # lazy — only needed at prepare time
        find_exact_dupes, flag_extreme_length, cross_split_near_dupes,
        analyze_split, cross_split_stats, print_quality_summary,
    )
    quality_report: dict[str, Any] = {"task_id": cfg.task_id}
    click.echo(f"  [{cfg.task_id}] Quality analysis — {len(train_rows)} train, {len(test_rows)} test rows...")
    raw_train_texts = [format_user(prompt, r) for r in train_rows]
    raw_test_texts  = [format_user(prompt, r) for r in test_rows]
    quality_report["raw"] = {"train_n": len(train_rows), "test_n": len(test_rows)}

    extreme = flag_extreme_length(raw_train_texts)
    drop    = set(extreme["too_short"] + extreme["too_long"])
    if drop:
        train_rows      = [r for i, r in enumerate(train_rows)      if i not in drop]
        raw_train_texts = [t for i, t in enumerate(raw_train_texts) if i not in drop]

    train_exact = set(find_exact_dupes(raw_train_texts))
    if train_exact:
        train_rows      = [r for i, r in enumerate(train_rows)      if i not in train_exact]
        raw_train_texts = [t for i, t in enumerate(raw_train_texts) if i not in train_exact]

    test_exact = set(find_exact_dupes(raw_test_texts))
    if test_exact:
        test_rows      = [r for i, r in enumerate(test_rows)      if i not in test_exact]
        raw_test_texts = [t for i, t in enumerate(raw_test_texts) if i not in test_exact]

    # Test is sampled first so cross-split checks against the actual eval rows,
    # and train sampling then draws from an already-clean pool (no data loss).
    click.echo(f"  [{cfg.task_id}] Sampling and formatting...")
    test_rows_full: Optional[list[dict]] = None
    if not smoke_test:
        if cfg.test_sampling:
            test_rows_full = test_rows
            test_rows = sample(test_rows, **cfg.test_sampling.model_dump())
            raw_test_texts = [format_user(prompt, r) for r in test_rows]
            _log_distribution(test_rows, cfg.test_sampling.stratify_by, f"Test ({cfg.test_sampling.strategy})")

    # CUAD: expand each test question into one prompt per sliding window. Both
    # the sampled eval set (test.jsonl) and the full set (test_full.jsonl, which
    # backs multi-seed resampling) are chunked; eval resamples whole questions
    # so a question's windows always stay together.
    if cfg.task_id == "cuad":
        cuad_window = cfg.context_max_tokens or 750
        cuad_stride = cfg.context_stride_tokens or cuad_window
        n_questions = len(test_rows)
        n_pos_q = sum(1 for r in test_rows if r.get("has_answer"))
        test_rows = chunk_cuad_test(test_rows, cuad_window, cuad_stride, cfg.max_chunks)
        raw_test_texts = [format_user(prompt, r) for r in test_rows]
        if test_rows_full is not None:
            # Balance the multi-seed pool 50/50 too, so each seed's question
            # resample keeps the same positive/no-answer ratio as seed 0.
            full_pos = [r for r in test_rows_full if r.get("has_answer")]
            full_neg = [r for r in test_rows_full if not r.get("has_answer")]
            k = min(len(full_pos), len(full_neg))
            brng = random.Random(42)
            brng.shuffle(full_pos)
            brng.shuffle(full_neg)
            test_rows_full = chunk_cuad_test(full_pos[:k] + full_neg[:k],
                                             cuad_window, cuad_stride, cfg.max_chunks)
        click.echo(f"  [cuad] Test sliding-window chunking: {n_questions} questions "
                   f"({n_pos_q} answerable / {n_questions - n_pos_q} no-answer) → "
                   f"{len(test_rows)} windowed prompts (window={cuad_window}w, "
                   f"stride={cuad_stride}w, max {cfg.max_chunks} windows/contract)")

    click.echo(f"  [{cfg.task_id}] Cross-split deduplication ({len(train_rows)} train × {len(test_rows)} test)...")
    cross_res  = cross_split_near_dupes(raw_train_texts, raw_test_texts, threshold=0.9)
    cross_drop = set(cross_res["train_indices_to_filter"])
    if cross_drop:
        train_rows = [r for i, r in enumerate(train_rows) if i not in cross_drop]

    quality_report["filtering"] = {
        "extreme_too_short":         len(extreme["too_short"]),
        "extreme_too_long":          len(extreme["too_long"]),
        "train_exact_dupes_removed": len(train_exact),
        "test_exact_dupes_removed":  len(test_exact),
        "cross_split_removed":       len(cross_drop),
    }

    if not smoke_test:
        if cfg.train_sampling:
            train_rows = sample(train_rows, **cfg.train_sampling.model_dump())
            _log_distribution(train_rows, cfg.train_sampling.stratify_by, f"Train ({cfg.train_sampling.strategy})")

    # ── FPB: assert no overlap between train and test ─────────────────────
    if cfg.task_id == "fpb" and not smoke_test:
        train_sentences = {r.get("sentence", "") for r in train_rows}
        overlap = [r for r in test_rows if r.get("sentence", "") in train_sentences]
        assert not overlap, f"FPB: {len(overlap)} rows overlap between train and test after sampling"

    # ── Carve a stratified validation split out of the training pool ───────
    # A versioned, stratified val.jsonl beats a positional tail-slice at train
    # time: every class is represented and the split is reproducible. Stratified
    # on the same field train sampling used (and balanced the same way, if any).
    val_rows: list[dict] = []
    if not smoke_test and cfg.train_sampling and len(train_rows) >= 20:
        val_n = round(len(train_rows) * cfg.val_ratio)
        if val_n >= 1:
            val_rows = sample(
                train_rows, strategy="stratified",
                stratify_by=cfg.train_sampling.stratify_by,
                total_cap=val_n, seed=cfg.val_seed,
                balance_by=cfg.train_sampling.balance_by,
            )
            # Partition by object identity: sample() returns the same dict
            # objects, so id() removes exactly the rows drawn.
            val_ids = {id(r) for r in val_rows}
            train_rows = [r for r in train_rows if id(r) not in val_ids]
            _log_distribution(val_rows, cfg.train_sampling.stratify_by, "Val (stratified)")

    # ── Smoke capping ─────────────────────────────────────────────────────
    src_train = train_rows[:SMOKE_TRAIN_N] if smoke_test else train_rows
    src_test  = test_rows[:SMOKE_TEST_N]  if smoke_test else test_rows

    # ── Format to chat JSONL ───────────────────────────────────────────────
    def fmt_rows(rows: list[dict], include_assistant: bool = True) -> list[dict]:
        out = []
        for r in rows:
            user = format_user(prompt, r)
            asst = format_assistant(prompt, r, label_names) if include_assistant else None
            out.append(to_chat(system, user, asst))
        return out

    def _eval_row_id(r: dict, i: int) -> str:
        # Chunked tasks (CUAD) precompute a stable per-window id so chunks of one
        # question stay linked; everything else uses positional enumeration.
        return r.get("_eval_id") or f"{cfg.task_id}_test_{i:04d}"

    def fmt_labels(rows: list[dict]) -> list[dict]:
        # format_eval_label, not format_assistant: the label is the gold answer
        # a prediction is scored against — the answer letter for CoT tasks (not
        # the <thinking> block), and the full list of valid spans for extraction.
        return [
            {"id": _eval_row_id(r, i), "label": format_eval_label(prompt, r, label_names)}
            for i, r in enumerate(rows)
        ]

    def fmt_test_prompts(rows: list[dict]) -> list[dict]:
        out = []
        for i, r in enumerate(rows):
            user = format_user(prompt, r)
            out.append({
                "id": _eval_row_id(r, i),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            })
        return out

    prefix = "smoke_" if smoke_test else ""
    fmt_train = fmt_rows(src_train)
    fmt_val   = fmt_rows(val_rows)
    fmt_test  = fmt_test_prompts(src_test)
    fmt_labs  = fmt_labels(src_test)

    train_and_val = fmt_train + fmt_val
    _, invalid_train = validate_dataset(train_and_val)
    if invalid_train:
        click.echo(f"  WARNING [{cfg.task_id}]: {len(invalid_train)} train/val rows failed validation", err=True)
        for row in invalid_train[:3]:
            click.echo(f"    {row['validation_error']}", err=True)

    contam_hits = check_contamination(train_and_val, fmt_test)
    if contam_hits:
        click.echo(f"  WARNING [{cfg.task_id}]: {len(contam_hits)} training example(s) overlap test set", err=True)
        for hit in contam_hits[:3]:
            click.echo(f"    {hit}", err=True)
        if not smoke_test:
            # The quality pipeline should have eliminated all exact overlap before this point.
            # A hit here means the pipeline has a bug — abort rather than silently produce
            # a contaminated dataset that invalidates all downstream eval metrics.
            raise RuntimeError(
                f"[{cfg.task_id}] Train/test contamination detected after quality filtering "
                f"({len(contam_hits)} example(s)). Re-check data_quality.py integration."
            )

    write_jsonl(fmt_train, out_dir / f"{prefix}train.jsonl")
    if fmt_val:
        write_jsonl(fmt_val, out_dir / "val.jsonl")
    write_jsonl(fmt_test,  out_dir / f"{prefix}test.jsonl")
    write_jsonl(fmt_labs,  out_dir / f"{prefix}test_labels.jsonl")

    # Save full unsampled test set when sampling reduced it — enables multi-seed resampling.
    if test_rows_full is not None and len(test_rows_full) > len(src_test):
        write_jsonl(fmt_test_prompts(test_rows_full), out_dir / "test_full.jsonl")
        write_jsonl(fmt_labels(test_rows_full),       out_dir / "test_full_labels.jsonl")

    # ── Quality report: prepared-data stats ───────────────────────────────
    prep_train_texts = [
        m["content"] for r in fmt_train for m in r["messages"] if m["role"] == "user"
    ]
    prep_test_texts = [
        m["content"] for r in fmt_test for m in r["messages"] if m["role"] == "user"
    ]
    # Label distributions are a classification concept. Extraction labels are
    # free-form spans (a list of acceptable answers for CUAD) — not hashable and
    # not a class vocabulary — so leave them out of the quality report.
    if cfg.task_type == "classification":
        # Use the eval label (the bare class — the answer letter for cot_letter
        # tasks like medmcqa), not the raw assistant completion. The completion
        # embeds the <thinking> CoT, which makes every train "label" unique and
        # forces label-KL to infinity; train and test label vocabularies must
        # match for that metric to mean anything.
        train_labels_prep = [format_eval_label(prompt, r, label_names) for r in src_train]
        test_labels_prep = [r["label"] for r in fmt_labs]
    else:
        train_labels_prep = None
        test_labels_prep = None

    quality_report["prepared"] = {
        "train": analyze_split(prep_train_texts, train_labels_prep or None),
        "test":  analyze_split(prep_test_texts,  test_labels_prep or None),
        "cross_split": cross_split_stats(
            prep_train_texts, prep_test_texts,
            train_labels_prep or None, test_labels_prep or None,
        ),
    }
    if not smoke_test:
        atomic_write_json(quality_report, out_dir / "quality_report.json")
    print_quality_summary(quality_report)

    if smoke_test:
        click.echo(f"  [{cfg.task_id}] Done — {len(src_test)} smoke_test, {len(src_train)} smoke_train")
    else:
        click.echo(f"  [{cfg.task_id}] Done — {len(src_test)} test, {len(src_train)} train")

    atomic_write_json(
        {
            "task_id": cfg.task_id,
            "input_hash": fingerprint,
            "prompt_sha": prompt_sha,
            "train_sha": rows_sha(fmt_train),
            "val_sha": rows_sha(fmt_val) if fmt_val else None,
            "test_sha": rows_sha(fmt_test),
            "n_train": len(fmt_train),
            "n_val": len(fmt_val),
            "n_test": len(fmt_test),
            "validation_errors": len(invalid_train),
            "contamination_hits": len(contam_hits),
            "smoke_test": smoke_test,
            "prepared_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        out_dir / "dataset_meta.json",
    )

    # Prompt versioning sidecar — records which prompt produced this prepared data.
    (out_dir / "prompt_sha.txt").write_text(prompt_sha + "\n")

    # Up to 5 deterministic examples, one per distinct label.
    if not smoke_test and fmt_train:
        seen_labels: set[str] = set()
        few_shot: list[dict] = []
        for row in fmt_train:
            msgs = row["messages"]
            assistant_msg = next((m for m in msgs if m["role"] == "assistant"), None)
            if assistant_msg is None:
                continue
            # Dedup on the answer, not the whole completion: a CoT target's
            # <thinking> block is unique per row, so keying on raw content would
            # never dedup and few-shot would lose its one-per-class coverage.
            content = assistant_msg["content"]
            answer_m = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.S)
            label = answer_m.group(1).strip() if answer_m else content
            if label in seen_labels:
                continue
            seen_labels.add(label)
            few_shot.append(row)
            if len(few_shot) >= 5:
                break
        if few_shot:
            write_jsonl(few_shot, out_dir / "few_shot.jsonl")
            click.echo(f"  [{cfg.task_id}] Curated few-shot: {len(few_shot)} examples spanning {len(seen_labels)} distinct labels")

    # Closed-set label sidecar — emitted only when the task has a finite output
    # vocabulary. Consumed by eval_local (guided_choice decoding) and by
    # classify_errors (format_violation checking). Absent file → free-form task.
    label_set = get_label_set(prompt, label_names)
    if label_set is not None:
        atomic_write_json(label_set, out_dir / "labels.json")


@click.command()
@click.option("--task", default=None, help="Task ID to prepare (required; use 'all' to prepare every task)")
@click.option("--dry-run", is_flag=True, help="Validate without processing")
@click.option("--smoke-test", is_flag=True, help="Write small smoke_train/smoke_test files instead of full train/test")
def main(task: str, dry_run: bool, smoke_test: bool) -> None:
    """Prepare datasets: split, sample, and format into chat JSONL.

    You must specify --task <id> or --task all. No default — raw data must
    have been downloaded first for each task you want to prepare.
    """
    if task is None:
        raise click.UsageError("--task is required. Pass a task ID or 'all' to prepare every downloaded task.")
    task_ids = ALL_TASKS if task == "all" else [t.strip() for t in task.split(",")]
    failures: list[tuple[str, str]] = []

    max_workers = min(len(task_ids), os.cpu_count() or 4, 4)
    if len(task_ids) > 1:
        click.echo(f"  Preparing {len(task_ids)} tasks with {max_workers} workers (output may interleave)")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_tid = {
            pool.submit(process_task, load_task_config(tid), dry_run, smoke_test=smoke_test): tid
            for tid in task_ids
        }
        for fut in as_completed(fut_to_tid):
            tid = fut_to_tid[fut]
            try:
                fut.result()
            except Exception as exc:
                click.echo(f"  ERROR [{tid}]: {exc}", err=True)
                traceback.print_exc()
                failures.append((tid, str(exc)))

    if failures:
        click.echo(f"\nFAILED ({len(failures)}): " + ", ".join(t for t, _ in failures))
        sys.exit(1)
    click.echo("\nAll tasks prepared successfully.")


if __name__ == "__main__":
    main()
