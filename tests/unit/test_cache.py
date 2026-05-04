"""Unit tests for pipeline.cache."""
import json
import pytest
from pathlib import Path

from pipeline.cache import (
    dict_hash,
    file_content_hash,
    inputs_changed,
    read_stored_hash,
    rows_sha,
    training_inputs_hash,
)

pytestmark = pytest.mark.unit


class TestFileContentHash:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")
        assert file_content_hash(f) == file_content_hash(f)

    def test_changes_with_content(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello")
        h1 = file_content_hash(f)
        f.write_bytes(b"world")
        assert file_content_hash(f) != h1

    def test_returns_16_hex_chars(self, tmp_path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"x")
        h = file_content_hash(f)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestDictHash:
    def test_deterministic(self):
        d = {"a": 1, "b": [2, 3]}
        assert dict_hash(d) == dict_hash(d)

    def test_key_order_irrelevant(self):
        assert dict_hash({"a": 1, "b": 2}) == dict_hash({"b": 2, "a": 1})

    def test_value_change_changes_hash(self):
        assert dict_hash({"a": 1}) != dict_hash({"a": 2})

    def test_returns_16_hex_chars(self):
        h = dict_hash({"x": "y"})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestTrainingInputsHash:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "train.jsonl"
        f.write_bytes(b'{"messages":[]}')
        params = {"epochs": 3, "smoke_test": False}
        assert training_inputs_hash(f, params) == training_inputs_hash(f, params)

    def test_changes_with_data(self, tmp_path):
        f = tmp_path / "train.jsonl"
        params = {"epochs": 3}
        f.write_bytes(b"row A")
        h1 = training_inputs_hash(f, params)
        f.write_bytes(b"row B")
        assert training_inputs_hash(f, params) != h1

    def test_changes_with_hyperparams(self, tmp_path):
        f = tmp_path / "train.jsonl"
        f.write_bytes(b"data")
        h1 = training_inputs_hash(f, {"epochs": 3})
        h2 = training_inputs_hash(f, {"epochs": 5})
        assert h1 != h2


class TestReadStoredHash:
    def test_returns_none_when_file_missing(self, tmp_path):
        assert read_stored_hash(tmp_path / "meta.json") is None

    def test_returns_none_when_key_absent(self, tmp_path):
        f = tmp_path / "meta.json"
        f.write_text(json.dumps({"model_id": "x"}))
        assert read_stored_hash(f) is None

    def test_reads_stored_hash(self, tmp_path):
        f = tmp_path / "meta.json"
        f.write_text(json.dumps({"input_hash": "abc123"}))
        assert read_stored_hash(f) == "abc123"


class TestRowsSha:
    def test_deterministic(self):
        rows = [{"messages": [{"role": "user", "content": "hi"}]}]
        assert rows_sha(rows) == rows_sha(rows)

    def test_changes_with_content(self):
        r1 = [{"messages": [{"role": "user", "content": "a"}]}]
        r2 = [{"messages": [{"role": "user", "content": "b"}]}]
        assert rows_sha(r1) != rows_sha(r2)

    def test_empty_list(self):
        h = rows_sha([])
        assert len(h) == 16

    def test_matches_write_jsonl_bytes(self, tmp_path):
        import json
        rows = [{"id": "x", "val": 1}, {"id": "y", "val": 2}]
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        assert rows_sha(rows) == file_content_hash(f)


class TestInputsChanged:
    def test_false_when_file_missing(self, tmp_path):
        assert inputs_changed("any_hash", tmp_path / "meta.json") is False

    def test_false_when_key_absent(self, tmp_path):
        f = tmp_path / "meta.json"
        f.write_text(json.dumps({"model_id": "x"}))
        assert inputs_changed("any_hash", f) is False

    def test_false_when_hashes_match(self, tmp_path):
        f = tmp_path / "meta.json"
        f.write_text(json.dumps({"input_hash": "abc123"}))
        assert inputs_changed("abc123", f) is False

    def test_true_when_hashes_differ(self, tmp_path):
        f = tmp_path / "meta.json"
        f.write_text(json.dumps({"input_hash": "old_hash"}))
        assert inputs_changed("new_hash", f) is True
