"""Async circuit breaker for external service calls.

In Python asyncio (single-threaded co-operative multitasking), state
transitions between non-await statements are atomic — no locks needed.

States:
  CLOSED    — normal operation, failures counted
  OPEN      — all requests rejected; waiting for cooldown_s before probing
  HALF_OPEN — one probe request in flight; trips back to OPEN on failure

Usage:
    cb = CircuitBreaker("openai", failure_threshold=5, cooldown_s=60)
    try:
        result = await cb.call(some_coroutine())
    except CircuitOpenError:
        ...  # circuit open, request was fast-rejected
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a request is fast-rejected because the circuit is OPEN."""


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        cooldown_s: float = 60.0,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._cooldown = cooldown_s
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float = 0.0
        self._probing = False  # True while a HALF_OPEN probe is awaited

    @property
    def state(self) -> CircuitState:
        return self._state

    def _open_msg(self) -> str:
        return (
            f"Circuit '{self.name}' OPEN — "
            f"{self._failures} consecutive failure(s), "
            f"cooldown {self._cooldown}s"
        )

    def raise_if_open(self) -> None:
        """Fast-fail if the circuit is firmly OPEN (cooldown not yet elapsed).

        Does NOT transition to HALF_OPEN — call call() for the full state machine.
        Useful to fast-reject before acquiring a semaphore slot.
        """
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at < self._cooldown:
                raise CircuitOpenError(self._open_msg())

    def _check(self) -> None:
        """Raise CircuitOpenError if this request should be rejected.

        Called synchronously (no await), so it is atomic in asyncio context.
        """
        if self._state == CircuitState.CLOSED:
            return

        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._cooldown and not self._probing:
                self._state = CircuitState.HALF_OPEN
                self._probing = True
                return  # allow the probe through
            raise CircuitOpenError(self._open_msg())

        if self._probing:
            raise CircuitOpenError(
                f"Circuit '{self.name}' HALF_OPEN — probe in progress"
            )

    def _on_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._probing = False

    def _on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold or self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            self._probing = False

    async def call(self, coro: Coroutine[Any, Any, T]) -> T:
        """Execute coro under circuit-breaker protection.

        Raises CircuitOpenError without awaiting coro when the circuit is OPEN.
        Records successes and failures to drive state transitions.
        """
        try:
            self._check()  # synchronous — atomic in asyncio
        except CircuitOpenError:
            coro.close()  # prevent unawaited-coroutine warnings on fast-reject
            raise
        try:
            result = await coro
        except CircuitOpenError:
            raise
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def reset(self) -> None:
        """Reset to CLOSED state. Intended for tests and manual recovery."""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._probing = False
        self._opened_at = 0.0
