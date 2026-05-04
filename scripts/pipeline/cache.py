"""Content-hash utilities for skip-if-unchanged logic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from pipeline.versioning import file_sha256 as file_content_hash  # re-exported, no duplicate


def rows_sha(rows: list[dict]) -> str:
    """SHA-256 (first 16 hex) of rows serialized as JSONL (mirrors write_jsonl byte output)."""
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps(row, ensure_ascii=False) + "\n").encode())
    return h.hexdigest()[:16]


def dict_hash(d: dict) -> str:
    """SHA-256 (first 16 hex chars) of a stable JSON serialization (keys sorted)."""
    serialized = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def training_inputs_hash(data_path: Path, hyperparams: dict) -> str:
    """Combined hash of training data content and hyperparameters."""
    parts = [file_content_hash(data_path), dict_hash(hyperparams)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def read_stored_hash(meta_path: Path) -> Optional[str]:
    """Read input_hash from metadata.json. Returns None if file absent or key missing."""
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f).get("input_hash")
    except Exception:
        return None


def inputs_changed(current_hash: str, meta_path: Path) -> bool:
    """Return True only when a stored hash exists AND differs from current_hash.

    No stored hash → False, preserving backward-compatible skip for runs that
    pre-date content hashing.
    """
    stored = read_stored_hash(meta_path)
    if stored is None:
        return False
    return stored != current_hash
