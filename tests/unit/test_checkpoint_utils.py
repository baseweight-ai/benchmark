"""Unit tests for checkpoint_utils.py — all pure I/O with no GPU deps."""
import json
import time

import pytest

import checkpoint_utils
from checkpoint_utils import (
    append_jsonl,
    atomic_write_json,
    checkpoint_dir,
    find_hf_resume_checkpoint,
    finalize_partial,
    load_partial_ids,
    load_train_state,
    partial_path,
    save_train_state,
)

pytestmark = pytest.mark.unit


# ── atomic_write_json ──────────────────────────────────────────────────────────

def test_atomic_write_json_creates_file(tmp_path):
    path = tmp_path / "out.json"
    atomic_write_json({"key": "value"}, path)
    assert path.exists()
    assert json.loads(path.read_text())["key"] == "value"


def test_atomic_write_json_no_tmp_residue(tmp_path):
    path = tmp_path / "out.json"
    atomic_write_json({"x": 1}, path)
    assert not (tmp_path / "out.json.tmp").exists()


def test_atomic_write_json_overwrites(tmp_path):
    path = tmp_path / "out.json"
    atomic_write_json({"v": 1}, path)
    atomic_write_json({"v": 2}, path)
    assert json.loads(path.read_text())["v"] == 2


def test_atomic_write_json_creates_parents(tmp_path):
    path = tmp_path / "a" / "b" / "c.json"
    atomic_write_json({"ok": True}, path)
    assert path.exists()


# ── append_jsonl ───────────────────────────────────────────────────────────────

def test_append_jsonl_creates_file(tmp_path):
    path = tmp_path / "rows.jsonl"
    append_jsonl({"id": "r1"}, path)
    assert path.exists()
    assert json.loads(path.read_text().strip())["id"] == "r1"


def test_append_jsonl_multiple_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    for i in range(3):
        append_jsonl({"id": f"r{i}", "v": i}, path)
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(lines) == 3
    assert [l["v"] for l in lines] == [0, 1, 2]


def test_append_jsonl_creates_parents(tmp_path):
    path = tmp_path / "deep" / "dir" / "rows.jsonl"
    append_jsonl({"id": "x"}, path)
    assert path.exists()


# ── partial_path / finalize_partial / load_partial_ids ────────────────────────

def test_partial_path_suffix(tmp_path):
    out = tmp_path / "cond.jsonl"
    pp = partial_path(out)
    assert pp.name == "cond.jsonl.partial"
    assert pp.parent == out.parent


def test_finalize_partial_renames(tmp_path):
    pp = tmp_path / "cond.jsonl.partial"
    final = tmp_path / "cond.jsonl"
    pp.write_text('{"id":"x"}\n')
    finalize_partial(pp, final)
    assert final.exists()
    assert not pp.exists()


def test_load_partial_ids_missing_file(tmp_path):
    pp = tmp_path / "nonexistent.partial"
    assert load_partial_ids(pp) == set()


def test_load_partial_ids_reads_ids(tmp_path):
    pp = tmp_path / "cond.jsonl.partial"
    rows = [{"id": f"r{i}", "output": "x"} for i in range(4)]
    pp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ids = load_partial_ids(pp)
    assert ids == {"r0", "r1", "r2", "r3"}


def test_load_partial_ids_skips_malformed(tmp_path):
    pp = tmp_path / "cond.jsonl.partial"
    pp.write_text('{"id":"good"}\nNOT_JSON\n{"id":"also_good"}\n')
    ids = load_partial_ids(pp)
    assert "good" in ids
    assert "also_good" in ids
    assert len(ids) == 2


def test_load_partial_ids_skips_empty_id(tmp_path):
    pp = tmp_path / "cond.jsonl.partial"
    pp.write_text('{"id":""}\n{"id":"real"}\n')
    ids = load_partial_ids(pp)
    assert ids == {"real"}


# ── checkpoint_dir / find_hf_resume_checkpoint ────────────────────────────────

def test_checkpoint_dir_path(tmp_checkpoints_root):
    d = checkpoint_dir("qwen3-8b", "banking77", "lora-500")
    assert d == tmp_checkpoints_root / "qwen3-8b" / "banking77" / "lora-500"


def test_find_hf_resume_checkpoint_no_dir(tmp_checkpoints_root):
    result = find_hf_resume_checkpoint("qwen3-8b", "banking77", "lora-500")
    assert result is None


def test_find_hf_resume_checkpoint_empty_dir(tmp_checkpoints_root):
    ckpt_dir = checkpoint_dir("qwen3-8b", "banking77", "lora-500")
    ckpt_dir.mkdir(parents=True)
    assert find_hf_resume_checkpoint("qwen3-8b", "banking77", "lora-500") is None


def test_find_hf_resume_checkpoint_returns_latest(tmp_checkpoints_root):
    ckpt_dir = checkpoint_dir("qwen3-8b", "banking77", "lora-500")
    for n in [1, 5, 3]:
        (ckpt_dir / f"checkpoint-{n}").mkdir(parents=True)
    result = find_hf_resume_checkpoint("qwen3-8b", "banking77", "lora-500")
    assert result is not None
    assert result.name == "checkpoint-5"


# ── save_train_state / load_train_state ───────────────────────────────────────

def test_save_load_train_state_roundtrip(tmp_checkpoints_root):
    save_train_state("qwen3-8b", "fpb", "lora-500", {"status": "complete", "eval_loss": 0.42})
    state = load_train_state("qwen3-8b", "fpb", "lora-500")
    assert state is not None
    assert state["status"] == "complete"
    assert state["eval_loss"] == pytest.approx(0.42)
    assert "saved_at" in state


def test_load_train_state_missing(tmp_checkpoints_root):
    assert load_train_state("qwen3-8b", "fpb", "lora-500") is None


def test_save_train_state_adds_timestamp(tmp_checkpoints_root):
    t_before = time.time()
    save_train_state("qwen3-8b", "fpb", "lora-500", {"status": "in_progress"})
    state = load_train_state("qwen3-8b", "fpb", "lora-500")
    assert state["saved_at"] >= t_before


# ── _Tee ───────────────────────────────────────────────────────────────────────

from checkpoint_utils import _Tee
from io import StringIO
from unittest.mock import MagicMock


class TestTee:
    def test_writes_to_all_streams(self):
        a, b = StringIO(), StringIO()
        tee = _Tee(a, b)
        tee.write("hello\n")
        assert a.getvalue() == "hello\n"
        assert b.getvalue() == "hello\n"

    def test_skips_closed_stream(self):
        live = StringIO()
        dead = StringIO()
        dead.close()
        tee = _Tee(dead, live)
        tee.write("hi\n")
        assert live.getvalue() == "hi\n"

    def test_tolerates_value_error_on_write(self):
        bad = MagicMock()
        bad.closed = False
        bad.write.side_effect = ValueError("I/O operation on closed file")
        good = StringIO()
        tee = _Tee(bad, good)
        tee.write("msg\n")  # must not raise
        assert good.getvalue() == "msg\n"

    def test_tolerates_os_error_on_write(self):
        bad = MagicMock()
        bad.closed = False
        bad.write.side_effect = OSError("broken pipe")
        good = StringIO()
        tee = _Tee(bad, good)
        tee.write("msg\n")  # must not raise
        assert good.getvalue() == "msg\n"

    def test_flush_only_on_newline(self):
        s = MagicMock()
        s.closed = False
        tee = _Tee(s)
        tee.write("no newline here")
        s.flush.assert_not_called()
        tee.write("has\nnewline")
        s.flush.assert_called()

    def test_flush_skips_closed_stream(self):
        dead = StringIO()
        dead.close()
        live = MagicMock()
        live.closed = False
        tee = _Tee(dead, live)
        tee.flush()  # must not raise
        live.flush.assert_called_once()

    def test_tolerates_value_error_on_flush(self):
        bad = MagicMock()
        bad.closed = False
        bad.flush.side_effect = ValueError("closed")
        tee = _Tee(bad)
        tee.flush()  # must not raise

    def test_isatty_returns_false(self):
        tee = _Tee(StringIO())
        assert tee.isatty() is False

    def test_training_log_tees_to_file(self, tmp_path):
        import sys
        from checkpoint_utils import training_log
        ckpt_dir = tmp_path / "ckpt"
        stdout_before = sys.stdout
        with training_log(ckpt_dir) as log_path:
            print("logged line")
        assert log_path.exists()
        assert "logged line" in log_path.read_text()
        assert sys.stdout is stdout_before
