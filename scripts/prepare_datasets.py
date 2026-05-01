"""Prepare datasets for training and evaluation: split, sample, format, save JSONL."""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import click
import yaml
from pydantic import BaseModel, Field

from checkpoint_utils import nv_prepared_dir

REPO_ROOT = Path(__file__).parent.parent
SEED = 42
ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa"]

SMOKE_TRAIN_N = 20
SMOKE_TEST_N = 10


# ── Config models ──────────────────────────────────────────────────────────

class TaskConfig(BaseModel):
    task_id: str
    task_name: str
    dataset_path: str
    dataset_config: Optional[str] = None
    task_type: str
    metric_id: str
    max_output_tokens: int
    test_sample_size: Optional[int] = None
    training_cap: Optional[int] = None
    text_field: Optional[str] = None
    label_field: Optional[str] = None
    label_type: Optional[str] = None
    custom_label_names: Optional[list[str]] = None
    split_ratios: Optional[list[float]] = None
    split_seed: Optional[int] = None
    context_max_tokens: Optional[int] = None
    test_split: str = "test"


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})


def load_prompt(task_id: str) -> dict:
    path = REPO_ROOT / "prompts" / f"{task_id}.json"
    with open(path) as f:
        return json.load(f)


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
    if lf == "code":
        return str(row.get(prompt.get("label_field", "code"), ""))
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


# ── Stratified sampling ────────────────────────────────────────────────────

def stratified_sample(rows: list[dict], label_key: str, n: int, seed: int = 42) -> list[dict]:
    """Return stratified sample of n rows, preserving class distribution."""
    rng = random.Random(seed)
    by_label: dict[Any, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r.get(label_key)].append(r)
    classes = sorted(by_label.keys(), key=str)
    result = []
    per_class = max(1, n // len(classes))
    for cls in classes:
        pool = by_label[cls]
        k = min(per_class, len(pool))
        result.extend(rng.sample(pool, k))
    # top up if needed
    all_remaining = [r for r in rows if r not in result]
    rng.shuffle(all_remaining)
    result = result[:n]
    if len(result) < n:
        result.extend(all_remaining[:n - len(result)])
    rng.shuffle(result)
    return result[:n]



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


# ── Per-task preprocessing ─────────────────────────────────────────────────

def process_task(cfg: TaskConfig, dry_run: bool, smoke_test: bool = False) -> None:
    from datasets import load_from_disk, Dataset  # lazy
    label = " (smoke)" if smoke_test else ""
    click.echo(f"\n[{cfg.task_id}] Processing {cfg.task_name}{label}...")

    raw_dir = REPO_ROOT / "data" / "raw" / cfg.task_id
    out_dir = REPO_ROOT / "data" / "prepared" / cfg.task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        click.echo(f"  [dry-run] Would process {cfg.task_id}")
        return

    prompt = load_prompt(cfg.task_id)
    system = prompt["system"]
    ds = load_from_disk(str(raw_dir))

    # ── Task-specific split handling ───────────────────────────────────────
    if cfg.task_id == "fpb":
        all_rows = list(ds["train"])
        label_names = cfg.custom_label_names or ["negative", "neutral", "positive"]
        rng = random.Random(cfg.split_seed or 42)
        rng.shuffle(all_rows)
        if smoke_test:
            # Download provided 2x rows; split evenly into 12 train / 12 test
            half = len(all_rows) // 2
            train_rows = all_rows[:half]
            test_rows = all_rows[half:]
            val_rows = []
            click.echo(f"  FPB smoke split: train={len(train_rows)}, test={len(test_rows)}")
        else:
            # Prod: 70/15/15
            n = len(all_rows)
            n_train = int(n * 0.70)
            n_val = int(n * 0.15)
            train_rows = all_rows[:n_train]
            val_rows = all_rows[n_train:n_train + n_val]
            test_rows = all_rows[n_train + n_val:]
            click.echo(f"  FPB split: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
    else:
        label_names = None
        split_name = cfg.test_split
        if "train" in ds:
            train_rows = list(ds["train"])
        else:
            train_rows = []
        if split_name in ds:
            test_rows = list(ds[split_name])
        else:
            test_rows = train_rows[-100:]
        val_rows = list(ds.get("validation", ds.get("val", [])))

    # Label names for integer-mapped tasks
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

    # ── CUAD: flatten SQuAD format ─────────────────────────────────────────
    if cfg.task_id == "cuad":
        def flatten_squad(rows: list[dict]) -> list[dict]:
            out = []
            for row in rows:
                ctx = truncate_context(row.get("context", ""), cfg.context_max_tokens or 1500)
                answers = row.get("answers", {})
                texts = answers.get("text", []) if isinstance(answers, dict) else []
                out.append({
                    "context": ctx,
                    "question": row.get("question", ""),
                    "answers": {"text": [texts[0]] if texts else [], "answer_start": [0] if texts else []},
                    "id": row.get("id", ""),
                })
            return out
        train_rows = flatten_squad(train_rows)
        test_rows = flatten_squad(test_rows)
        val_rows = flatten_squad(val_rows)

    # ── Stratified test sample for ledgar, medmcqa ────────────────────────
    label_key = cfg.label_field or "label"
    if cfg.test_sample_size and len(test_rows) > cfg.test_sample_size:
        test_rows = stratified_sample(test_rows, label_key, cfg.test_sample_size, SEED)
        click.echo(f"  Test set sampled to {len(test_rows)}")

    # ── Cap training set ───────────────────────────────────────────────────
    if cfg.training_cap and len(train_rows) > cfg.training_cap:
        train_rows = stratified_sample(train_rows, label_key, cfg.training_cap, SEED)
        click.echo(f"  Training set capped at {len(train_rows)}")

    # ── Format to chat JSONL ───────────────────────────────────────────────
    def fmt_rows(rows: list[dict], include_assistant: bool = True) -> list[dict]:
        out = []
        for r in rows:
            user = format_user(prompt, r)
            asst = format_assistant(prompt, r, label_names) if include_assistant else None
            out.append(to_chat(system, user, asst))
        return out

    def fmt_labels(rows: list[dict]) -> list[dict]:
        out = []
        for i, r in enumerate(rows):
            asst = format_assistant(prompt, r, label_names)
            out.append({"id": f"{cfg.task_id}_test_{i:04d}", "label": asst})
        return out

    def fmt_test_prompts(rows: list[dict]) -> list[dict]:
        out = []
        for i, r in enumerate(rows):
            user = format_user(prompt, r)
            d = {"id": f"{cfg.task_id}_test_{i:04d}", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if cfg.task_id == "cuad":
                d["context"] = r.get("context", "")
            out.append(d)
        return out

    # Write files
    if smoke_test:
        smoke_train = train_rows[:SMOKE_TRAIN_N]
        smoke_test_rows = test_rows[:SMOKE_TEST_N]
        write_jsonl(fmt_rows(smoke_train), out_dir / "smoke_train.jsonl")
        write_jsonl(fmt_test_prompts(smoke_test_rows), out_dir / "smoke_test.jsonl")
        write_jsonl(fmt_labels(smoke_test_rows), out_dir / "smoke_test_labels.jsonl")
        click.echo(f"  [{cfg.task_id}] Done — {len(smoke_test_rows)} smoke_test, {len(smoke_train)} smoke_train")
    else:
        write_jsonl(fmt_rows(train_rows), out_dir / "train.jsonl")
        write_jsonl(fmt_test_prompts(test_rows), out_dir / "test.jsonl")
        write_jsonl(fmt_labels(test_rows), out_dir / "test_labels.jsonl")
        click.echo(f"  [{cfg.task_id}] Done — {len(test_rows)} test, {len(train_rows)} train")

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
