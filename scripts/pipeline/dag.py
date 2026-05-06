"""DAG-based pipeline stage execution.

Independent stages run concurrently within their compute pool:
  GPU   — max 1 worker  (GPU is exclusive)
  CPU   — max 4 workers
  CLOUD — max 10 workers (API calls)

Stages are skipped when ALL of their active dependencies have failed.
A single failed dependency does not block sibling stages.
"""
from __future__ import annotations

import concurrent.futures
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal

from pipeline.log import get_logger

_stdout_lock = threading.Lock()

_log = get_logger("dag")

_POOL_WORKERS: dict[str, int] = {
    "cpu": 4,
    "gpu": 1,
    "cloud": 10,
}


class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


_TERMINAL = frozenset({StageStatus.DONE, StageStatus.FAILED, StageStatus.SKIPPED})


@dataclass
class StageResult:
    stage_id: str
    status: StageStatus
    elapsed_s: float = 0.0
    error: str | None = None


@dataclass
class Stage:
    id: str
    cmd: list[str]
    depends_on: list[str]
    compute: Literal["cpu", "gpu", "cloud"]
    timeout_s: int | None = None
    requires_all_deps: bool = False  # if True, skip if any dep failed/skipped
    description: str = ""


class DAGRunner:
    """Execute Stages in dependency order, parallelizing by compute pool."""

    def __init__(
        self,
        stages: list[Stage],
        repo_root: Path,
        after_stage: Callable[[StageResult], bool] | None = None,
    ) -> None:
        self._stages = {s.id: s for s in stages}
        self._status: dict[str, StageStatus] = {s.id: StageStatus.PENDING for s in stages}
        self._results: dict[str, StageResult] = {}
        self._repo_root = repo_root
        self._after_stage = after_stage  # return False to abort remaining stages
        self._abort = False

    def run(self) -> dict[str, StageResult]:
        pools = {ct: ThreadPoolExecutor(max_workers=w) for ct, w in _POOL_WORKERS.items()}
        futures: dict[Future, str] = {}

        try:
            while True:
                if not self._abort:
                    self._submit_ready(pools, futures)

                if not futures:
                    break

                done, _ = concurrent.futures.wait(
                    list(futures), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    sid = futures.pop(f)
                    result: StageResult = f.result()
                    self._results[sid] = result
                    self._status[sid] = result.status
                    if self._after_stage and not self._after_stage(result):
                        self._abort = True

                if self._all_terminal():
                    break
        finally:
            for pool in pools.values():
                pool.shutdown(wait=False)

        for sid in list(self._stages):
            if self._status[sid] == StageStatus.PENDING:
                self._status[sid] = StageStatus.SKIPPED
                self._results[sid] = StageResult(sid, StageStatus.SKIPPED)

        return self._results

    def _submit_ready(
        self,
        pools: dict[str, ThreadPoolExecutor],
        futures: dict[Future, str],
    ) -> None:
        for sid, stage in self._stages.items():
            if self._status[sid] != StageStatus.PENDING:
                continue
            if not self._deps_terminal(stage):
                continue
            if self._should_skip(stage):
                self._status[sid] = StageStatus.SKIPPED
                self._results[sid] = StageResult(sid, StageStatus.SKIPPED)
                continue
            self._status[sid] = StageStatus.RUNNING
            future = pools[stage.compute].submit(self._run_stage, stage)
            futures[future] = sid

    def _run_stage(self, stage: Stage) -> StageResult:
        t0 = time.monotonic()
        with _stdout_lock:
            timeout_str = f"  timeout: {stage.timeout_s}s" if stage.timeout_s else ""
            meta = f"  (compute: {stage.compute}{timeout_str})"
            desc = f"  {stage.description}" if stage.description else ""
            print(f"\n  [{stage.id}] ━━━{desc}{meta} ━━━\n", flush=True)
        _log.info("stage start", stage=stage.id, event="stage_start")
        prefix = f"[{stage.id}] "
        try:
            proc = subprocess.Popen(
                stage.cmd,
                cwd=self._repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            def _emit(raw: bytes) -> None:
                line = raw.decode(errors="replace")
                # Simulate terminal \r overwrite: keep only the last segment
                if "\r" in line:
                    line = line.rsplit("\r", 1)[-1]
                line = line.rstrip()
                if line:
                    with _stdout_lock:
                        print(f"  {prefix}{line}", flush=True)

            def _stream() -> None:
                buf = b""
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        raw_line, buf = buf.split(b"\n", 1)
                        _emit(raw_line)
                if buf:
                    _emit(buf)

            reader = threading.Thread(target=_stream, daemon=True)
            reader.start()

            timed_out = False
            try:
                proc.wait(timeout=stage.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()

            reader.join()
            elapsed = time.monotonic() - t0

            if timed_out:
                msg = f"timeout after {stage.timeout_s}s"
                _log.error("stage timeout", stage=stage.id, wall_time_s=round(elapsed, 1), exc=msg)
                return StageResult(stage.id, StageStatus.FAILED, elapsed_s=elapsed, error=msg)

            if proc.returncode != 0:
                msg = f"exit code {proc.returncode}"
                _log.error("stage failed", stage=stage.id, wall_time_s=round(elapsed, 1), exc=msg)
                return StageResult(stage.id, StageStatus.FAILED, elapsed_s=elapsed, error=msg)

            _log.info("stage complete", stage=stage.id, wall_time_s=round(elapsed, 1),
                      event="stage_complete")
            return StageResult(stage.id, StageStatus.DONE, elapsed_s=elapsed)
        except Exception as e:
            elapsed = time.monotonic() - t0
            _log.error("stage error", stage=stage.id, wall_time_s=round(elapsed, 1), exc=str(e))
            return StageResult(stage.id, StageStatus.FAILED, elapsed_s=elapsed, error=str(e))

    def _deps_terminal(self, stage: Stage) -> bool:
        return all(
            self._status[d] in _TERMINAL
            for d in stage.depends_on
            if d in self._status
        )

    def _should_skip(self, stage: Stage) -> bool:
        if self._abort:
            return True
        active = [d for d in stage.depends_on if d in self._status]
        if not active:
            return False
        if stage.requires_all_deps:
            return any(
                self._status[d] in (StageStatus.FAILED, StageStatus.SKIPPED)
                for d in active
            )
        return all(
            self._status[d] in (StageStatus.FAILED, StageStatus.SKIPPED)
            for d in active
        )

    def _all_terminal(self) -> bool:
        return all(s in _TERMINAL for s in self._status.values())
