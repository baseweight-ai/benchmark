"""Unit tests for pipeline.circuit_breaker."""
import asyncio
import time
import pytest
from pipeline.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState

pytestmark = pytest.mark.unit


async def _fail():
    raise RuntimeError("boom")


async def _ok():
    return "ok"


def _run(coro):
    return asyncio.run(coro)


class TestInitialState:
    def test_starts_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_raise_if_open_allows_when_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.raise_if_open()  # must not raise

    def test_returns_coroutine_result(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        async def produce(): return 42
        assert _run(cb.call(produce())) == 42


class TestFailureAccumulation:
    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_s=60)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                _run(cb.call(_fail()))
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_s=60)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                _run(cb.call(_fail()))
        assert cb.state == CircuitState.OPEN

    def test_open_fast_rejects_via_call(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_s=60)
        with pytest.raises(RuntimeError):
            _run(cb.call(_fail()))
        with pytest.raises(CircuitOpenError):
            _run(cb.call(_fail()))

    def test_raise_if_open_fast_rejects(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_s=60)
        with pytest.raises(RuntimeError):
            _run(cb.call(_fail()))
        with pytest.raises(CircuitOpenError):
            cb.raise_if_open()


class TestSuccessResets:
    def test_success_clears_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_s=60)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                _run(cb.call(_fail()))
        result = _run(cb.call(_ok()))
        assert result == "ok"
        assert cb._failures == 0
        assert cb.state == CircuitState.CLOSED


class TestHalfOpen:
    def test_probe_succeeds_closes_circuit(self, monkeypatch):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_s=10)
        with pytest.raises(RuntimeError):
            _run(cb.call(_fail()))
        assert cb.state == CircuitState.OPEN

        monkeypatch.setattr(time, "monotonic", lambda: cb._opened_at + 11)

        result = _run(cb.call(_ok()))
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_probe_failure_reopens_circuit(self, monkeypatch):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_s=10)
        with pytest.raises(RuntimeError):
            _run(cb.call(_fail()))

        monkeypatch.setattr(time, "monotonic", lambda: cb._opened_at + 11)

        with pytest.raises(RuntimeError):
            _run(cb.call(_fail()))
        assert cb.state == CircuitState.OPEN

    def test_raise_if_open_allows_after_cooldown(self, monkeypatch):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_s=10)
        with pytest.raises(RuntimeError):
            _run(cb.call(_fail()))

        monkeypatch.setattr(time, "monotonic", lambda: cb._opened_at + 11)

        cb.raise_if_open()  # cooldown elapsed — must not raise


class TestReset:
    def test_reset_closes_circuit(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_s=60)
        with pytest.raises(RuntimeError):
            _run(cb.call(_fail()))
        assert cb.state == CircuitState.OPEN

        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb._failures == 0
        cb.raise_if_open()  # must not raise after reset

    def test_reset_allows_new_calls(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_s=60)
        with pytest.raises(RuntimeError):
            _run(cb.call(_fail()))

        cb.reset()
        result = _run(cb.call(_ok()))
        assert result == "ok"
