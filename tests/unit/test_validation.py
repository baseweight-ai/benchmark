"""Unit tests for pipeline.validation."""
import pytest

from pipeline.validation import check_contamination, validate_chat_row, validate_dataset

pytestmark = pytest.mark.unit


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
