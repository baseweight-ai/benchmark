"""Shared utilities for benchmark pipeline scripts."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

# Sliding-window tasks (CUAD) emit one row per context window, id'd
# `<question-id>_chunkNN`. This is the single definition of that convention —
# prepare_datasets writes it, classify_errors groups by it, eval resamples by it.
_CHUNK_SUFFIX_RE = re.compile(r"_chunk\d+$")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rows_hash(rows: list[dict]) -> str:
    """Stable hash of a list of dicts (sort_keys for cross-source consistency)."""
    h = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    return h


def read_prompt_sha(prepared: Path) -> str | None:
    p = prepared / "prompt_sha.txt"
    return p.read_text().strip() if p.exists() else None


def load_label_set(repo_root: Path, task_id: str) -> list[str] | None:
    """Return the closed answer set written by prepare_datasets, or None.

    labels.json is the single source of truth for a task's finite answer
    vocabulary: eval_local reads it for vLLM guided_choice, eval_api to build
    the response_format schema, and classify_errors for format-violation
    checks. Whether decoding is actually constrained is a separate, per-task
    decision made by each caller.
    """
    p = repo_root / "data" / "prepared" / task_id / "labels.json"
    if not p.exists():
        return None
    labels = json.loads(p.read_text())
    if not isinstance(labels, list) or not labels:
        return None
    return [str(s) for s in labels]


def question_id(row_id: str) -> str:
    """Strip a trailing _chunkNN suffix to recover the underlying question id.

    Returns the id unchanged when there is no chunk suffix, so it is safe to
    call on rows from any task.
    """
    return _CHUNK_SUFFIX_RE.sub("", str(row_id))


def is_chunked(rows: list[dict]) -> bool:
    """True when any row id carries a _chunkNN suffix (sliding-window tasks)."""
    return any(_CHUNK_SUFFIX_RE.search(str(r.get("id", ""))) for r in rows)


def seed_sample_questions(rows: list[dict], n_questions: int, seed: int) -> list[dict]:
    """Reproducibly sample n_questions whole questions, keeping their rows together.

    Rows are grouped by question id (chunk suffix stripped); the questions are
    seed-shuffled and every row of the first n_questions is returned. For
    sliding-window tasks (CUAD) this keeps a question's windows together instead
    of splitting them across seeds. For unchunked data — one row per question —
    it is exactly a reproducible row-level sample.
    """
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        q = question_id(r.get("id", ""))
        if q not in groups:
            order.append(q)
            groups[q] = []
        groups[q].append(r)
    rng = random.Random(seed)
    rng.shuffle(order)
    out: list[dict] = []
    for q in order[:n_questions]:
        out.extend(groups[q])
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_messages(prompt_row: dict, few_shot: list[dict], condition: str) -> list[dict]:
    """Return message list for a test row, prepending few-shot turns when requested."""
    base = prompt_row["messages"]
    if condition == "5-shot" and few_shot:
        system = base[0]
        user = base[1]
        shots = []
        for ex in few_shot:
            msgs = ex.get("messages", [])
            if len(msgs) >= 3:
                shots.append(msgs[1])  # user turn
                shots.append(msgs[2])  # assistant turn
        return [system] + shots + [user]
    return base
