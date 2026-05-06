"""Chat schema validation and training data contamination checks."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


# ── File-level input guards ────────────────────────────────────────────────────

class InputValidationError(RuntimeError):
    pass


def require_jsonl(
    path: Path,
    min_rows: int = 1,
    check_chat_format: bool = False,
    require_assistant_completion: bool = True,
    sample_size: int = 5,
) -> int:
    """Assert path exists, contains >= min_rows valid JSON lines, and optionally passes chat-format checks.

    Returns the total row count. Raises InputValidationError on any failure.
    Reads lazily: parses only the first max(min_rows, sample_size) rows, then
    counts remaining lines without parsing to keep large-file overhead minimal.
    """
    if not path.exists():
        raise InputValidationError(f"Required input file not found: {path}")
    stop_at = max(min_rows, sample_size)
    sample: list[dict] = []
    count = 0
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InputValidationError(
                    f"Invalid JSON on line {lineno} of {path}: {exc}"
                ) from exc
            count += 1
            if len(sample) < sample_size:
                sample.append(row)
            if count >= stop_at:
                count += sum(1 for line in fh if line.strip())
                break
    if count < min_rows:
        raise InputValidationError(
            f"Expected >= {min_rows} rows in {path}, found {count}"
        )
    if check_chat_format:
        for i, row in enumerate(sample):
            err = validate_chat_row(row, require_assistant_completion=require_assistant_completion)
            if err:
                raise InputValidationError(
                    f"Row {i} of {path} failed chat-format check: {err}"
                )
    return count


def require_dir(path: Path, min_files: int = 1, desc: str = "") -> int:
    """Assert directory exists and contains >= min_files entries.

    Returns the file count (exact when min_files > 1, otherwise 1). Raises InputValidationError on failure.
    """
    label = desc or str(path)
    if not path.exists():
        raise InputValidationError(f"Required directory not found: {label} ({path})")
    if not path.is_dir():
        raise InputValidationError(f"Expected a directory, got a file: {path}")
    if min_files <= 1:
        if next(path.iterdir(), None) is None:
            raise InputValidationError(f"Expected >= 1 file in {label}, directory is empty")
        return 1
    files = list(path.iterdir())
    if len(files) < min_files:
        raise InputValidationError(
            f"Expected >= {min_files} file(s) in {label}, found {len(files)}"
        )
    return len(files)

_VALID_ROLES = frozenset({"system", "user", "assistant"})


def validate_chat_row(row: dict, require_assistant_completion: bool = True) -> Optional[str]:
    """Return an error string if the row is malformed, or None if valid."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return "missing or empty messages"

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"message {i} is not a dict"
        role = msg.get("role")
        content = msg.get("content")
        if role not in _VALID_ROLES:
            return f"message {i} has invalid role: {role!r}"
        if not isinstance(content, str):
            return f"message {i} has non-string content"
        if role == "assistant" and not content.strip():
            return f"message {i} has empty assistant completion"

    non_system = [m for m in messages if m["role"] != "system"]
    if not non_system:
        return "no non-system messages"
    if non_system[0]["role"] != "user":
        return f"first non-system message has role {non_system[0]['role']!r}, expected 'user'"
    for i in range(1, len(non_system)):
        expected = "assistant" if i % 2 == 1 else "user"
        if non_system[i]["role"] != expected:
            return (
                f"role alternation violation at non-system[{i}]: "
                f"expected {expected!r}, got {non_system[i]['role']!r}"
            )

    if require_assistant_completion and non_system[-1]["role"] != "assistant":
        return f"last non-system message is {non_system[-1]['role']!r}, expected 'assistant'"

    return None


def validate_dataset(
    rows: list[dict],
    require_assistant_completion: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, invalid_rows). Invalid rows gain a 'validation_error' key."""
    valid: list[dict] = []
    invalid: list[dict] = []
    for row in rows:
        error = validate_chat_row(row, require_assistant_completion)
        if error is None:
            valid.append(row)
        else:
            invalid.append({**row, "validation_error": error})
    return valid, invalid


def _normalize_prompt(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return re.sub(r"\s+", " ", msg.get("content", "").lower().strip())
    return ""


def check_contamination(train_rows: list[dict], test_rows: list[dict]) -> list[str]:
    """Return error strings for training rows whose user prompt exactly matches a test prompt.

    Uses normalized exact-string matching (case-fold + whitespace-collapse) with O(n) lookup.
    """
    test_prompts = {
        _normalize_prompt(r.get("messages", []))
        for r in test_rows
        if r.get("messages")
    }
    test_prompts.discard("")

    hits: list[str] = []
    for i, row in enumerate(train_rows):
        prompt = _normalize_prompt(row.get("messages", []))
        if prompt and prompt in test_prompts:
            hits.append(f"train[{i}] matches test set: {prompt[:80]!r}")
    return hits
