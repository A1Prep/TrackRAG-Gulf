
"""Tests for the evaluation module."""

import json
import pytest

from trackrag.evaluation import evaluate, load_answer_key, load_generated_answers


ANSWER_KEY_LINES = [
    {"question_id": "Q001", "expected_answer": "Two pedestrians.", "answerable": True,
     "acceptable_track_ids": [33, 34], "acceptable_intervals": []},
    {"question_id": "Q002", "expected_answer": "The footage does not establish the destination.",
     "answerable": False, "acceptable_track_ids": [], "acceptable_intervals": []},
]

GENERATED_LINES = [
    {"question_id": "Q001", "answer_status": "answered",
     "answer": "Two pedestrians are visible.", "cited_evidence_ids": ["Q001_E1"]},
    {"question_id": "Q002", "answer_status": "insufficient_evidence",
     "answer": None, "cited_evidence_ids": []},
]


@pytest.fixture
def answer_key_file(tmp_path):
    path = tmp_path / "answer_key.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in ANSWER_KEY_LINES) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def generated_file(tmp_path):
    path = tmp_path / "generated.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in GENERATED_LINES) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_answer_key(answer_key_file):
    key = load_answer_key(answer_key_file)
    assert "Q001" in key
    assert key["Q001"]["answerable"] is True


def test_load_generated_answers(generated_file):
    gen = load_generated_answers(generated_file)
    assert "Q001" in gen
    assert gen["Q001"]["answer_status"] == "answered"


def test_correct_answerable_question(answer_key_file, generated_file):
    records = evaluate(generated_file, answer_key_file)
    q1 = next(r for r in records if r["question_id"] == "Q001")
    assert q1["outcome"] == "answered_correct"


def test_correct_unanswerable_question(answer_key_file, generated_file):
    records = evaluate(generated_file, answer_key_file)
    q2 = next(r for r in records if r["question_id"] == "Q002")
    assert q2["outcome"] == "answered_correct"


def test_error_outcome_passthrough(answer_key_file, tmp_path):
    gen_path = tmp_path / "gen.jsonl"
    gen_path.write_text(
        json.dumps({
            "question_id": "Q001",
            "answer_status": "error",
            "answer": None,
            "cited_evidence_ids": [],
        }) + "\n",
        encoding="utf-8",
    )
    records = evaluate(gen_path, answer_key_file)
    q1 = next(r for r in records if r["question_id"] == "Q001")
    assert q1["outcome"] == "error"


def test_missing_generated_answer_is_not_requested(answer_key_file, tmp_path):
    gen_path = tmp_path / "gen.jsonl"
    gen_path.write_text("", encoding="utf-8")
    records = evaluate(gen_path, answer_key_file)
    assert all(r["outcome"] == "not_requested" for r in records)


def test_model_answers_unanswerable_question_is_incorrect(answer_key_file, tmp_path):
    gen_path = tmp_path / "gen.jsonl"
    gen_path.write_text(
        json.dumps({
            "question_id": "Q002",
            "answer_status": "answered",
            "answer": "The pedestrians are going to the market.",
            "cited_evidence_ids": ["Q002_E1"],
        }) + "\n",
        encoding="utf-8",
    )
    records = evaluate(gen_path, answer_key_file)
    q2 = next(r for r in records if r["question_id"] == "Q002")
    assert q2["outcome"] == "answered_incorrect"
