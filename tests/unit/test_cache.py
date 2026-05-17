"""Unit tests for pipeline.cache."""
import json
import pytest
from pathlib import Path

from pipeline.cache import (
    code_closure_hash,
    dict_hash,
    file_content_hash,
    inputs_changed,
    read_stored_hash,
    record_fingerprint,
    reuse_is_valid,
    rows_sha,
    training_inputs_hash,
    tree_hash,
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


class TestCodeClosureHash:
    def _scripts(self, tmp_path):
        """a.py imports b.py; c.py is unrelated."""
        (tmp_path / "a.py").write_text("import b\nimport os\n")
        (tmp_path / "b.py").write_text("x = 1\n")
        (tmp_path / "c.py").write_text("y = 1\n")
        return tmp_path / "a.py"

    def test_deterministic(self, tmp_path):
        entry = self._scripts(tmp_path)
        assert code_closure_hash(entry) == code_closure_hash(entry)

    def test_changes_when_imported_module_changes(self, tmp_path):
        entry = self._scripts(tmp_path)
        h1 = code_closure_hash(entry)
        (tmp_path / "b.py").write_text("x = 2\n")
        assert code_closure_hash(entry) != h1

    def test_stable_when_unrelated_module_changes(self, tmp_path):
        entry = self._scripts(tmp_path)
        h1 = code_closure_hash(entry)
        (tmp_path / "c.py").write_text("y = 999\n")
        assert code_closure_hash(entry) == h1

    def test_changes_when_entry_changes(self, tmp_path):
        entry = self._scripts(tmp_path)
        h1 = code_closure_hash(entry)
        entry.write_text("import b\n# edited\n")
        assert code_closure_hash(entry) != h1

    def test_follows_transitive_imports(self, tmp_path):
        (tmp_path / "a.py").write_text("import b\n")
        (tmp_path / "b.py").write_text("import d\n")
        (tmp_path / "d.py").write_text("z = 1\n")
        entry = tmp_path / "a.py"
        h1 = code_closure_hash(entry)
        (tmp_path / "d.py").write_text("z = 2\n")
        assert code_closure_hash(entry) != h1

    def test_resolves_package_submodule(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "sub.py").write_text("v = 1\n")
        (tmp_path / "a.py").write_text("from pkg import sub\n")
        entry = tmp_path / "a.py"
        h1 = code_closure_hash(entry)
        (pkg / "sub.py").write_text("v = 2\n")
        assert code_closure_hash(entry) != h1

    def test_ignores_stdlib_imports(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\nimport json\n")
        h = code_closure_hash(tmp_path / "a.py")
        assert len(h) == 16
        assert all(ch in "0123456789abcdef" for ch in h)


class TestTreeHash:
    def test_deterministic(self, tmp_path):
        (tmp_path / "f1.txt").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "f2.txt").write_text("b")
        assert tree_hash(tmp_path) == tree_hash(tmp_path)

    def test_changes_when_file_content_changes(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("a")
        h1 = tree_hash(tmp_path)
        f.write_text("b")
        assert tree_hash(tmp_path) != h1

    def test_changes_when_file_added(self, tmp_path):
        (tmp_path / "f1.txt").write_text("a")
        h1 = tree_hash(tmp_path)
        (tmp_path / "f2.txt").write_text("b")
        assert tree_hash(tmp_path) != h1

    def test_missing_dir_is_stable(self, tmp_path):
        missing = tmp_path / "nope"
        assert tree_hash(missing) == tree_hash(missing)
        assert len(tree_hash(missing)) == 16


def _pred_pair(tmp_path):
    """A completed prediction file and its (unused) partial-file path."""
    out = tmp_path / "pred.jsonl"
    out.write_text('{"id":1}\n')
    return out, tmp_path / "pred.jsonl.partial"


class TestReuseIsValid:
    def test_true_when_output_exists_and_no_sidecar(self, tmp_path):
        # Grandfathered: a pre-fingerprint output with no sidecar is reused.
        out, pp = _pred_pair(tmp_path)
        assert reuse_is_valid(out, pp, "fp123") is True

    def test_true_when_fingerprint_matches(self, tmp_path):
        out, pp = _pred_pair(tmp_path)
        record_fingerprint(out, "fp123")
        assert reuse_is_valid(out, pp, "fp123") is True

    def test_false_when_output_missing(self, tmp_path):
        out = tmp_path / "pred.jsonl"
        assert reuse_is_valid(out, tmp_path / "pred.jsonl.partial", "fp123") is False

    def test_false_and_discards_stale_on_mismatch(self, tmp_path):
        out, pp = _pred_pair(tmp_path)
        pp.write_text("partial\n")
        record_fingerprint(out, "old_fp")
        assert reuse_is_valid(out, pp, "new_fp") is False
        assert not out.exists()
        assert not pp.exists()
        assert not out.with_suffix(".meta.json").exists()


class TestRecordFingerprint:
    def test_writes_readable_sidecar(self, tmp_path):
        out = tmp_path / "pred.jsonl"
        record_fingerprint(out, "abc123")
        assert read_stored_hash(out.with_suffix(".meta.json")) == "abc123"

    def test_creates_parent_dir(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "pred.jsonl"
        record_fingerprint(out, "xyz")
        assert read_stored_hash(out.with_suffix(".meta.json")) == "xyz"
