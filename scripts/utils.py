"""Shared utilities for benchmark pipeline scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path



def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


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
