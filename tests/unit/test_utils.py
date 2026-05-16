"""Unit tests for utils.py."""
import json
from collections import Counter

import pytest

from utils import (
    build_messages,
    is_chunked,
    load_jsonl,
    question_id,
    seed_sample_questions,
    write_jsonl,
)

pytestmark = pytest.mark.unit


def _write_jsonl_raw(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ── load_jsonl / write_jsonl ───────────────────────────────────────────────────

def test_load_jsonl_roundtrip(tmp_path):
    rows = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    path = tmp_path / "rows.jsonl"
    _write_jsonl_raw(rows, path)
    loaded = load_jsonl(path)
    assert loaded == rows


def test_write_jsonl_creates_file(tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl([{"x": 1}], path)
    assert path.exists()
    assert json.loads(path.read_text().strip())["x"] == 1


def test_write_jsonl_creates_parents(tmp_path):
    path = tmp_path / "a" / "b" / "out.jsonl"
    write_jsonl([{"x": 1}], path)
    assert path.exists()


def test_write_jsonl_preserves_unicode(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl([{"text": "こんにちは"}], path)
    loaded = load_jsonl(path)
    assert loaded[0]["text"] == "こんにちは"


def test_load_jsonl_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert load_jsonl(path) == []


# ── build_messages ─────────────────────────────────────────────────────────────

def _make_row(system="You classify.", user="Classify this.", assistant=None):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if assistant:
        msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


def test_build_messages_zero_shot_returns_base(toy_chat_rows):
    row = toy_chat_rows[0]
    msgs = build_messages(row, [], "zero-shot")
    assert msgs == row["messages"]


def test_build_messages_lora_same_as_zero_shot(toy_chat_rows):
    row = toy_chat_rows[0]
    assert build_messages(row, [], "lora-500") == build_messages(row, [], "zero-shot")


def test_build_messages_5shot_prepends_examples(toy_chat_rows):
    test_row = _make_row(user="Classify: final question")
    few_shot = [_make_row(user=f"Q{i}", assistant=f"A{i}") for i in range(5)]
    msgs = build_messages(test_row, few_shot, "5-shot")
    # Structure: [system, user0, asst0, ..., user4, asst4, final_user]
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == test_row["messages"][1]
    # 1 system + 5×(user+asst) + 1 final user = 12
    assert len(msgs) == 12


def test_build_messages_5shot_empty_few_shot(toy_chat_rows):
    row = toy_chat_rows[0]
    # Falls back to base when few_shot is empty
    assert build_messages(row, [], "5-shot") == row["messages"]


def test_build_messages_5shot_skips_incomplete_examples():
    test_row = _make_row()
    # Few-shot row with only system+user (no assistant)
    incomplete = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
    full = _make_row(user="Q", assistant="A")
    msgs = build_messages(test_row, [incomplete, full], "5-shot")
    # Only the full example contributes 2 turns
    assert len(msgs) == 1 + 2 + 1  # system + (user+asst) + final_user


# ── question_id / is_chunked ───────────────────────────────────────────────────

def test_question_id_strips_chunk_suffix():
    assert question_id("cuad_test_0007_chunk03") == "cuad_test_0007"


def test_question_id_leaves_unchunked_ids_unchanged():
    assert question_id("fpb_test_0042") == "fpb_test_0042"


def test_is_chunked_true_for_chunk_ids():
    assert is_chunked([{"id": "cuad_test_0000_chunk00"}, {"id": "cuad_test_0000_chunk01"}])


def test_is_chunked_false_for_plain_ids():
    assert not is_chunked([{"id": "fpb_test_0000"}, {"id": "fpb_test_0001"}])


# ── seed_sample_questions ──────────────────────────────────────────────────────

def _chunk_rows(n_questions: int, chunks_per_q: int) -> list[dict]:
    return [
        {"id": f"cuad_test_{q:04d}_chunk{c:02d}"}
        for q in range(n_questions)
        for c in range(chunks_per_q)
    ]


def test_seed_sample_questions_keeps_each_questions_windows_together():
    """The whole point: a sampled question keeps every one of its windows."""
    rows = _chunk_rows(n_questions=20, chunks_per_q=5)
    sampled = seed_sample_questions(rows, n_questions=8, seed=1)
    per_q = Counter(question_id(r["id"]) for r in sampled)
    assert len(per_q) == 8
    assert all(count == 5 for count in per_q.values())


def test_seed_sample_questions_deterministic():
    rows = _chunk_rows(15, 4)
    a = seed_sample_questions(rows, 6, seed=3)
    b = seed_sample_questions(rows, 6, seed=3)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_seed_sample_questions_different_seeds_differ():
    rows = _chunk_rows(30, 3)
    a = {question_id(r["id"]) for r in seed_sample_questions(rows, 10, seed=1)}
    b = {question_id(r["id"]) for r in seed_sample_questions(rows, 10, seed=2)}
    assert a != b


def test_seed_sample_questions_caps_at_available():
    rows = _chunk_rows(5, 3)
    sampled = seed_sample_questions(rows, n_questions=99, seed=1)
    assert len(sampled) == 15  # 5 questions x 3 windows — never more


def test_seed_sample_questions_unchunked_is_reproducible_row_sample():
    """One row per question → a plain, reproducible, distinct row subsample."""
    rows = [{"id": f"fpb_test_{i:04d}"} for i in range(50)]
    sampled = seed_sample_questions(rows, n_questions=20, seed=7)
    assert len(sampled) == 20
    assert len({r["id"] for r in sampled}) == 20
