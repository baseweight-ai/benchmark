"""Structured pipeline logger — machine-readable JSON side-channel.

Every record carries stage/model/task/condition tags so concurrent-task logs
are distinguishable in post. Records are appended to results/pipeline.log.jsonl.

Call configure(REPO_ROOT) once per script main() to activate file logging.
Records emitted before configure() are silently discarded (standard Python
logging behaviour when no handlers are installed).

Usage:
    from pipeline.log import configure, get_logger
    configure(REPO_ROOT)
    log = get_logger("eval-api")
    log.info("eval complete", model="gpt-5.4-mini", task="fpb",
             condition="zero-shot", event="stage_complete",
             n_rows=350, total_input_tokens=120_000, total_output_tokens=3_500)
    log.error("request failed", model="gpt-5.4-mini", task="fpb",
              exc=str(exc), traceback=traceback.format_exc())
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_EXTRA_KEYS = frozenset({
    "stage", "model", "task", "condition", "event",
    "cost_usd", "training_cost", "training_time_min", "wall_time_s",
    "metric_id", "metric_value", "n_rows", "n_train",
    "total_input_tokens", "total_output_tokens",
    "attempt", "max_attempts", "delay_s",
    "exc", "traceback",
})


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for key in _EXTRA_KEYS:
            val = getattr(record, key, None)
            if val is not None:
                d[key] = val
        return json.dumps(d, ensure_ascii=False)


def configure(repo_root: Path, log_path: Optional[Path] = None) -> None:
    """Activate JSON file logging. Idempotent for the same output path."""
    root = logging.getLogger("pipeline")
    if log_path is None:
        log_path = repo_root / "results" / "pipeline.log.jsonl"

    resolved = log_path.resolve()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == resolved:
            return  # already configured for this path

    # Remove stale handlers from a previous configure() call
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), encoding="utf-8", mode="a")
    fh.setFormatter(_JSONFormatter())
    root.setLevel(logging.DEBUG)
    root.propagate = False
    root.addHandler(fh)


class StageLogger:
    """Logger bound to a pipeline stage, accepting per-call context fields."""

    __slots__ = ("_stage", "_inner")

    def __init__(self, stage: str) -> None:
        self._stage = stage
        self._inner = logging.getLogger(f"pipeline.{stage}")

    def _emit(self, level: int, msg: str, **extra: Any) -> None:
        extra["stage"] = self._stage
        extra = {k: v for k, v in extra.items() if v is not None}
        self._inner.log(level, msg, extra=extra, stacklevel=3)

    def debug(self, msg: str, **kw: Any) -> None:
        self._emit(logging.DEBUG, msg, **kw)

    def info(self, msg: str, **kw: Any) -> None:
        self._emit(logging.INFO, msg, **kw)

    def warning(self, msg: str, **kw: Any) -> None:
        self._emit(logging.WARNING, msg, **kw)

    def error(self, msg: str, **kw: Any) -> None:
        self._emit(logging.ERROR, msg, **kw)


def get_logger(stage: str) -> StageLogger:
    """Return a StageLogger bound to the given pipeline stage name."""
    return StageLogger(stage)
