"""Unit tests for pipeline.log."""
import json
import logging
import pytest
from pathlib import Path

from pipeline.log import configure, get_logger

pytestmark = pytest.mark.unit


def _read_records(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


@pytest.fixture(autouse=True)
def clean_handlers():
    yield
    root = logging.getLogger("pipeline")
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()


@pytest.fixture
def log_path(tmp_path):
    """Configured log file path, ready for a StageLogger to write to."""
    path = tmp_path / "out.jsonl"
    configure(tmp_path, path)
    return path


class TestConfigure:
    def test_creates_log_file_on_first_emit(self, tmp_path):
        path = tmp_path / "test.jsonl"
        configure(tmp_path, path)
        get_logger("t").info("ping")
        assert path.exists()

    def test_idempotent_same_path(self, tmp_path):
        path = tmp_path / "test.jsonl"
        configure(tmp_path, path)
        n = len(logging.getLogger("pipeline").handlers)
        configure(tmp_path, path)
        assert len(logging.getLogger("pipeline").handlers) == n

    def test_replaces_handler_on_path_change(self, tmp_path):
        configure(tmp_path, tmp_path / "a.jsonl")
        configure(tmp_path, tmp_path / "b.jsonl")
        assert len(logging.getLogger("pipeline").handlers) == 1


class TestRecordFormat:
    def test_required_fields_present(self, log_path):
        get_logger("stage1").info("hello")
        r = _read_records(log_path)[0]
        assert "ts" in r
        assert r["level"] == "INFO"
        assert r["msg"] == "hello"
        assert r["stage"] == "stage1"

    def test_extra_fields_included(self, log_path):
        get_logger("eval").info("done", model="gpt-4", task="fpb", condition="zero-shot",
                                event="stage_complete", n_rows=100)
        r = _read_records(log_path)[0]
        assert r["model"] == "gpt-4"
        assert r["task"] == "fpb"
        assert r["condition"] == "zero-shot"
        assert r["event"] == "stage_complete"
        assert r["n_rows"] == 100

    def test_none_values_excluded(self, log_path):
        get_logger("stage").info("msg", model=None, task="fpb")
        r = _read_records(log_path)[0]
        assert "model" not in r
        assert r["task"] == "fpb"

    def test_warning_level(self, log_path):
        get_logger("stage").warning("watch out")
        assert _read_records(log_path)[0]["level"] == "WARNING"

    def test_error_with_exc_and_traceback(self, log_path):
        get_logger("stage").error("failed", exc="RuntimeError: boom",
                                  traceback="Traceback (most recent call last)...")
        r = _read_records(log_path)[0]
        assert r["level"] == "ERROR"
        assert r["exc"] == "RuntimeError: boom"
        assert "Traceback" in r["traceback"]

    def test_multiple_records_appended(self, log_path):
        log = get_logger("stage")
        log.info("first")
        log.info("second")
        assert len(_read_records(log_path)) == 2

    def test_stage_bound_to_logger(self, log_path):
        get_logger("my-stage").info("hello")
        assert _read_records(log_path)[0]["stage"] == "my-stage"
