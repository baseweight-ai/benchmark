"""Unit tests for pipeline.validation."""
import json

import pytest

from pipeline.validation import (
    InputValidationError,
    check_contamination,
    reject_test_path,
    require_dir,
    require_jsonl,
    validate_chat_row,
    validate_dataset,
)

pytestmark = pytest.mark.unit


# ── require_jsonl ──────────────────────────────────────────────────────────────

def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _chat_row(user="q", assistant="a"):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


class TestRequireJsonl:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(InputValidationError, match="not found"):
            require_jsonl(tmp_path / "missing.jsonl")

    def test_valid_file_returns_count(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"x": i} for i in range(10)])
        assert require_jsonl(p, min_rows=10) == 10

    def test_too_few_rows_raises(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"x": 1}])
        with pytest.raises(InputValidationError, match="Expected >= 5"):
            require_jsonl(p, min_rows=5)

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"ok": true}\nNOT JSON\n')
        with pytest.raises(InputValidationError, match="Invalid JSON on line 2"):
            require_jsonl(p, min_rows=1)

    def test_chat_format_check_passes(self, tmp_path):
        p = tmp_path / "train.jsonl"
        _write_jsonl(p, [_chat_row() for _ in range(3)])
        assert require_jsonl(p, min_rows=3, check_chat_format=True) == 3

    def test_chat_format_check_fails_on_bad_row(self, tmp_path):
        p = tmp_path / "train.jsonl"
        bad = {"messages": [{"role": "user", "content": "q"}]}  # no assistant
        _write_jsonl(p, [bad])
        with pytest.raises(InputValidationError, match="chat-format check"):
            require_jsonl(p, min_rows=1, check_chat_format=True)

    def test_require_assistant_false_accepts_user_last(self, tmp_path):
        p = tmp_path / "test.jsonl"
        test_row = {"messages": [{"role": "user", "content": "q"}]}
        _write_jsonl(p, [test_row])
        assert require_jsonl(p, min_rows=1, check_chat_format=True, require_assistant_completion=False) == 1

    def test_lazy_counts_large_file(self, tmp_path):
        p = tmp_path / "big.jsonl"
        n = 200
        _write_jsonl(p, [{"i": i} for i in range(n)])
        assert require_jsonl(p, min_rows=n, sample_size=5) == n

    def test_blank_lines_ignored(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('\n{"x":1}\n\n{"x":2}\n\n')
        assert require_jsonl(p, min_rows=2) == 2

    def test_sample_size_default_five(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [_chat_row() for _ in range(3)])
        # 3 rows < default sample_size=5, but min_rows=1 — should still pass
        assert require_jsonl(p, min_rows=1, check_chat_format=True) == 3


# ── require_dir ────────────────────────────────────────────────────────────────

class TestRequireDir:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(InputValidationError, match="not found"):
            require_dir(tmp_path / "nonexistent")

    def test_file_where_dir_expected_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(InputValidationError, match="Expected a directory"):
            require_dir(f)

    def test_empty_dir_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(InputValidationError, match="directory is empty"):
            require_dir(d)

    def test_nonempty_dir_returns_one(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "file.txt").write_text("x")
        assert require_dir(d) == 1

    def test_min_files_satisfied(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        for i in range(3):
            (d / f"f{i}.txt").write_text("x")
        assert require_dir(d, min_files=3) == 3

    def test_min_files_not_satisfied_raises(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "one.txt").write_text("x")
        with pytest.raises(InputValidationError, match="Expected >= 3"):
            require_dir(d, min_files=3)

    def test_desc_appears_in_error(self, tmp_path):
        with pytest.raises(InputValidationError, match="raw data for toy"):
            require_dir(tmp_path / "nonexistent", desc="raw data for toy")


def _row(*role_content_pairs):
    return {"messages": [{"role": r, "content": c} for r, c in role_content_pairs]}


class TestValidateChatRow:
    def test_valid_system_user_assistant(self):
        assert validate_chat_row(_row(("system", "sys"), ("user", "q"), ("assistant", "a"))) is None

    def test_valid_user_assistant_only(self):
        assert validate_chat_row(_row(("user", "q"), ("assistant", "a"))) is None

    def test_missing_messages_key(self):
        assert validate_chat_row({}) is not None

    def test_empty_messages_list(self):
        assert validate_chat_row({"messages": []}) is not None

    def test_invalid_role(self):
        assert validate_chat_row({"messages": [{"role": "bot", "content": "hi"}]}) is not None

    def test_empty_assistant_completion(self):
        assert validate_chat_row(_row(("user", "q"), ("assistant", "  "))) is not None

    def test_role_alternation_user_user(self):
        assert validate_chat_row(_row(("user", "q1"), ("user", "q2"), ("assistant", "a"))) is not None

    def test_ends_on_user_fails_by_default(self):
        assert validate_chat_row(_row(("user", "q"))) is not None

    def test_ends_on_user_allowed_when_flag_false(self):
        assert validate_chat_row(_row(("system", "s"), ("user", "q")), require_assistant_completion=False) is None

    def test_non_string_content(self):
        assert validate_chat_row({"messages": [{"role": "user", "content": 42}]}) is not None

    def test_multi_turn_valid(self):
        row = _row(("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2"))
        assert validate_chat_row(row) is None

    def test_multi_turn_alternation_violation(self):
        row = _row(("user", "q1"), ("assistant", "a1"), ("assistant", "a2"))
        assert validate_chat_row(row) is not None


class TestValidateDataset:
    def test_all_valid(self):
        rows = [_row(("user", "q1"), ("assistant", "a1")), _row(("user", "q2"), ("assistant", "a2"))]
        valid, invalid = validate_dataset(rows)
        assert len(valid) == 2 and len(invalid) == 0

    def test_filters_invalid(self):
        rows = [_row(("user", "q"), ("assistant", "a")), {"messages": []}]
        valid, invalid = validate_dataset(rows)
        assert len(valid) == 1
        assert "validation_error" in invalid[0]

    def test_empty_input(self):
        valid, invalid = validate_dataset([])
        assert valid == [] and invalid == []


class TestCheckContamination:
    @staticmethod
    def _train(text):
        return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": text}, {"role": "assistant", "content": "a"}]}

    @staticmethod
    def _test(text):
        return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": text}]}

    def test_no_contamination(self):
        assert check_contamination([self._train("train Q")], [self._test("test Q")]) == []

    def test_exact_match_detected(self):
        hits = check_contamination([self._train("same question")], [self._test("same question")])
        assert len(hits) == 1

    def test_case_insensitive(self):
        hits = check_contamination([self._train("The Question")], [self._test("the question")])
        assert len(hits) == 1

    def test_whitespace_normalized(self):
        hits = check_contamination([self._train("q  with  spaces")], [self._test("q with spaces")])
        assert len(hits) == 1

    def test_empty_prompts_ignored(self):
        assert check_contamination([{"messages": []}], [{"messages": []}]) == []

    def test_multiple_hits(self):
        train = [self._train("shared"), self._train("unique"), self._train("shared")]
        test = [self._test("shared")]
        hits = check_contamination(train, test)
        assert len(hits) == 2


# ── reject_test_path ───────────────────────────────────────────────────────────

class TestRejectTestPath:
    def test_blocks_test_jsonl(self, tmp_path):
        p = tmp_path / "test.jsonl"
        with pytest.raises(InputValidationError, match="test-split pattern"):
            reject_test_path(p)

    def test_blocks_test_labels(self, tmp_path):
        p = tmp_path / "test_labels.jsonl"
        with pytest.raises(InputValidationError, match="test-split pattern"):
            reject_test_path(p)

    def test_blocks_smoke_test(self, tmp_path):
        p = tmp_path / "smoke_test.jsonl"
        with pytest.raises(InputValidationError, match="test-split pattern"):
            reject_test_path(p)

    def test_blocks_test_full(self, tmp_path):
        p = tmp_path / "test_full.jsonl"
        with pytest.raises(InputValidationError, match="test-split pattern"):
            reject_test_path(p)

    def test_allows_train_jsonl(self, tmp_path):
        reject_test_path(tmp_path / "train.jsonl")  # must not raise

    def test_allows_smoke_train(self, tmp_path):
        reject_test_path(tmp_path / "smoke_train.jsonl")  # must not raise

    def test_allows_train_cap(self, tmp_path):
        reject_test_path(tmp_path / "train_cap500.jsonl")  # must not raise
