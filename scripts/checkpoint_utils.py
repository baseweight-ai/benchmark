"""Checkpoint, recovery, and logging utilities for training runs.

Checkpoint layout (under <repo>/checkpoints/<model>/<task>/<condition>/):
  checkpoint-N/      HF Trainer intermediate checkpoints (one per epoch)
  train_state.json   Lightweight state: epoch, step, loss, status
  adapter/           Final adapter copy (written atomically on job completion)

Partial eval layout:
  results/predictions/local/<model>/<task>/<condition>.jsonl.partial
  results/predictions/api/<model>/<task>/<condition>.jsonl.partial
    Written row-by-row as inference completes; renamed to .jsonl on completion.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"


# ── Atomic I/O ────────────────────────────────────────────────────────────────

def atomic_write_json(data: dict, path: Path) -> None:
    """Write JSON atomically: write to .tmp, fsync, os.replace (POSIX atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(row: dict, path: Path) -> None:
    """Append one JSONL row and fsync.

    Safe in a single-threaded asyncio event loop: f.write() does not yield,
    so concurrent coroutines cannot interleave within a single append call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def finalize_partial(partial_path: Path, final_path: Path) -> None:
    """Atomically promote a completed partial file to its final name."""
    os.replace(partial_path, final_path)


# ── Eval partial-progress helpers ─────────────────────────────────────────────

def partial_path(out_path: Path) -> Path:
    """Return the .partial sibling of a predictions output path."""
    return out_path.with_name(out_path.name + ".partial")


def load_partial_ids(pp: Path) -> set[str]:
    """Return set of row IDs already written to a partial predictions file."""
    if not pp.exists():
        return set()
    ids: set[str] = set()
    with open(pp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_id = json.loads(line).get("id", "")
                if row_id:
                    ids.add(row_id)
            except json.JSONDecodeError:
                continue
    return ids


# ── Training checkpoint helpers ───────────────────────────────────────────────

def checkpoint_dir(model_short: str, task_id: str, condition: str, smoke: bool = False) -> Path:
    base = (CHECKPOINTS_ROOT / "smoke") if smoke else CHECKPOINTS_ROOT
    return base / model_short / task_id / condition


def find_hf_resume_checkpoint(
    model_short: str, task_id: str, condition: str, smoke: bool = False
) -> Optional[Path]:
    """Return the latest HF Trainer checkpoint directory, or None."""
    ckpt_dir = checkpoint_dir(model_short, task_id, condition, smoke=smoke)
    if not ckpt_dir.exists():
        return None
    candidates = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    return candidates[-1] if candidates else None


def save_train_state(
    model_short: str, task_id: str, condition: str, state: dict, smoke: bool = False
) -> None:
    """Atomically write training state to the network volume."""
    path = checkpoint_dir(model_short, task_id, condition, smoke=smoke) / "train_state.json"
    atomic_write_json({**state, "saved_at": time.time()}, path)


def load_train_state(
    model_short: str, task_id: str, condition: str, smoke: bool = False
) -> Optional[dict]:
    path = checkpoint_dir(model_short, task_id, condition, smoke=smoke) / "train_state.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Training log ──────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, *streams):
        self._s = streams

    def write(self, data):
        for s in self._s:
            try:
                if getattr(s, "closed", False):
                    continue
                s.write(data)
            except (ValueError, OSError):
                pass
        if "\n" in data:
            self.flush()

    def flush(self):
        for s in self._s:
            try:
                if getattr(s, "closed", False):
                    continue
                s.flush()
            except (ValueError, OSError):
                pass

    def isatty(self):
        return False  # suppress ANSI escape codes and progress bars from tqdm/rich/click


@contextlib.contextmanager
def training_log(ckpt_dir: Path):
    """Tee stdout/stderr to ckpt_dir/train.log for the duration of the block."""
    log_path = ckpt_dir / "train.log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(orig_out, fh)
        sys.stderr = _Tee(orig_err, fh)
        try:
            yield log_path
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err
