"""Chat schema validation and training data contamination checks."""
from __future__ import annotations

import re
from typing import Optional

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
