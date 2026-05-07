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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click
import yaml
from pydantic import BaseModel

from checkpoint_utils import atomic_write_json, nv_prepared_dir
from pipeline.cache import rows_sha
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
    context_max_tokens: Optional[int] = None
    test_split: str = "test"
    train_sampling: Optional[SamplingConfig] = None
    test_sampling: Optional[SamplingConfig] = None


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


def format_assistant(prompt: dict, row: dict, label_names: Optional[list[str]] = None) -> str:
    lf = prompt.get("label_format")
    if lf == "letter":
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


def to_chat(system: str, user: str, assistant: Optional[str] = None) -> dict:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


# ── Sampling ───────────────────────────────────────────────────────────────

def sample(
    data: list[dict],
    strategy: str,
    stratify_by: str,
    seed: int = 42,
    total_cap: Optional[int] = None,
    per_group_cap: Optional[int] = None,
    min_per_group: int = 1,
) -> list[dict]:
    """Balanced: per_group_cap rows per group. Stratified: total_cap rows via LRM allocation."""
    if not data:
        return []

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


# ── Context truncation ─────────────────────────────────────────────────────

def truncate_context(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


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
    system = prompt["system"]
    ds = load_from_disk(str(raw_dir))

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

    # ── CUAD: flatten SQuAD format and tag clause_type ────────────────────
    if cfg.task_id == "cuad":
        def flatten_squad(rows: list[dict]) -> list[dict]:
            out = []
            for row in rows:
                ctx = truncate_context(row.get("context", ""), cfg.context_max_tokens or 1500)
                answers = row.get("answers", {})
                texts = answers.get("text", []) if isinstance(answers, dict) else []
                question = row.get("question", "")
                m = _CUAD_CLAUSE_RE.search(question)
                clause_type = m.group(1) if m else "unknown"
                out.append({
                    "context": ctx,
                    "question": question,
                    "answers": {"text": [texts[0]] if texts else [], "answer_start": [0] if texts else []},
                    "id": row.get("id", ""),
                    "clause_type": clause_type,
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

    # ── Data quality: raw counts + filtering ──────────────────────────────
    from pipeline.data_quality import (  # lazy — only needed at prepare time
        find_exact_dupes, flag_extreme_length, cross_split_near_dupes,
        analyze_split, cross_split_stats, print_quality_summary,
    )
    quality_report: dict[str, Any] = {"task_id": cfg.task_id}
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

    # ── Sampling (non-smoke only) ──────────────────────────────────────────
    test_rows_full: Optional[list[dict]] = None
    if not smoke_test:
        if cfg.train_sampling:
            train_rows = sample(train_rows, **cfg.train_sampling.model_dump())
            _log_distribution(train_rows, cfg.train_sampling.stratify_by, f"Train ({cfg.train_sampling.strategy})")
        if cfg.test_sampling:
            test_rows_full = test_rows
            test_rows = sample(test_rows, **cfg.test_sampling.model_dump())
            _log_distribution(test_rows, cfg.test_sampling.stratify_by, f"Test ({cfg.test_sampling.strategy})")

    # ── FPB: assert no overlap between train and test ─────────────────────
    if cfg.task_id == "fpb" and not smoke_test:
        train_sentences = {r.get("sentence", "") for r in train_rows}
        overlap = [r for r in test_rows if r.get("sentence", "") in train_sentences]
        assert not overlap, f"FPB: {len(overlap)} rows overlap between train and test after sampling"

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

    def fmt_labels(rows: list[dict]) -> list[dict]:
        return [
            {"id": f"{cfg.task_id}_test_{i:04d}", "label": format_assistant(prompt, r, label_names)}
            for i, r in enumerate(rows)
        ]

    def fmt_test_prompts(rows: list[dict]) -> list[dict]:
        out = []
        for i, r in enumerate(rows):
            user = format_user(prompt, r)
            d: dict = {
                "id": f"{cfg.task_id}_test_{i:04d}",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            if cfg.task_id == "cuad":
                d["context"] = r.get("context", "")
            out.append(d)
        return out

    prefix = "smoke_" if smoke_test else ""
    fmt_train = fmt_rows(src_train)
    fmt_test  = fmt_test_prompts(src_test)
    fmt_labs  = fmt_labels(src_test)

    _, invalid_train = validate_dataset(fmt_train)
    if invalid_train:
        click.echo(f"  WARNING [{cfg.task_id}]: {len(invalid_train)} training rows failed validation", err=True)
        for row in invalid_train[:3]:
            click.echo(f"    {row['validation_error']}", err=True)

    contam_hits = check_contamination(fmt_train, fmt_test)
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
    train_labels_prep = [
        m["content"] for r in fmt_train for m in r["messages"] if m["role"] == "assistant"
    ]
    test_labels_prep = [r["label"] for r in fmt_labs]

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
            "prompt_sha": prompt_sha,
            "train_sha": rows_sha(fmt_train),
            "test_sha": rows_sha(fmt_test),
            "n_train": len(fmt_train),
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

    if os.environ.get("NETWORK_VOLUME"):
        nv_dir = nv_prepared_dir(cfg.task_id)
        shutil.copytree(str(out_dir), str(nv_dir), dirs_exist_ok=True)
        click.echo(f"  Mirrored to {nv_dir}")


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
    task_ids = ALL_TASKS if task == "all" else [task]
    failures = []
    for tid in task_ids:
        try:
            cfg = load_task_config(tid)
            process_task(cfg, dry_run, smoke_test=smoke_test)
        except Exception as exc:
            click.echo(f"  ERROR [{tid}]: {exc}", err=True)
            import traceback; traceback.print_exc()
            failures.append((tid, str(exc)))
    if failures:
        click.echo(f"\nFAILED ({len(failures)}): " + ", ".join(t for t, _ in failures))
        sys.exit(1)
    click.echo("\nAll tasks prepared successfully.")


if __name__ == "__main__":
    main()
