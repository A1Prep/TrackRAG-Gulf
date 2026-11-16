
import json
import pytest

from trackrag.questions import (
    load_questions,
    validate_question,
    validate_questions,
    validate_evidence,
)


def make_question():
    return {
        "question_id": "Q001",
        "recording_id": "EMT_001",
        "query_frame": 200,
        "question": "Which bus entered from the left?",
    }


def make_evidence(end_frame=180):
    return [{
        "evidence_id": "E1",
        "interval": {
            "recording_id": "EMT_001",
            "source": "annotated",
            "track_id": 18,
            "start_frame": 100,
            "end_frame": end_frame,
            "rank": 1,
            "score": 0.95,
        },
        "images": [{
            "frame_id": 150,
            "bbox_xyxy": (10.0, 20.0, 100.0, 120.0),
            "image_path": "evidence/frame_150.jpg",
        }],
    }]


def test_valid_question():
    question = validate_question(make_question())
    assert question["question_id"] == "Q001"


def test_missing_field_is_rejected():
    question = make_question()
    del question["question"]

    with pytest.raises(ValueError, match="Missing fields"):
        validate_question(question)


def test_invalid_query_frame_is_rejected():
    question = make_question()
    question["query_frame"] = True

    with pytest.raises(ValueError, match="integer"):
        validate_question(question)


def test_duplicate_question_ids_are_rejected():
    questions = [make_question(), make_question()]

    with pytest.raises(ValueError, match="Duplicate"):
        validate_questions(questions)


def test_jsonl_file_loads(tmp_path):
    file_path = tmp_path / "questions.jsonl"

    questions = [
        make_question(),
        {
            "question_id": "Q002",
            "recording_id": "EMT_001",
            "query_frame": 300,
            "question": "Which vehicle appeared first?",
        },
    ]

    with file_path.open("w", encoding="utf-8") as file:
        for question in questions:
            file.write(json.dumps(question) + "\n")

    loaded = load_questions(file_path)

    assert len(loaded) == 2
    assert loaded[0]["question_id"] == "Q001"


def test_valid_evidence():
    validate_evidence(make_question(), make_evidence())


def test_future_evidence_is_rejected():
    with pytest.raises(ValueError, match="after query_frame"):
        validate_evidence(make_question(), make_evidence(end_frame=250))


def test_empty_question_text_is_rejected():
    question = make_question()
    question["question"] = ""

    with pytest.raises(ValueError, match="non-empty string"):
        validate_question(question)


def test_negative_query_frame_is_rejected():
    question = make_question()
    question["query_frame"] = -1

    with pytest.raises(ValueError, match="negative"):
        validate_question(question)


def test_malformed_jsonl_is_rejected(tmp_path):
    file_path = tmp_path / "questions.jsonl"
    file_path.write_text('{"question_id": invalid}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_questions(file_path)


def test_empty_jsonl_returns_empty_list(tmp_path):
    file_path = tmp_path / "questions.jsonl"
    file_path.write_text("", encoding="utf-8")

    assert load_questions(file_path) == []


def test_wrong_recording_evidence_is_rejected():
    evidence = make_evidence()
    evidence[0]["interval"]["recording_id"] = "EMT_002"

    with pytest.raises(ValueError, match="different recording"):
        validate_evidence(make_question(), evidence)


def test_duplicate_evidence_ids_are_rejected():
    evidence = make_evidence()
    evidence.append(make_evidence()[0])

    with pytest.raises(ValueError, match="Duplicate evidence ID"):
        validate_evidence(make_question(), evidence)


def test_image_outside_interval_is_rejected():
    evidence = make_evidence()
    evidence[0]["images"][0]["frame_id"] = 190

    with pytest.raises(ValueError, match="outside its evidence interval"):
        validate_evidence(make_question(), evidence)


def test_invalid_frame_interval_is_rejected():
    evidence = make_evidence()
    evidence[0]["interval"]["start_frame"] = 180
    evidence[0]["interval"]["end_frame"] = 100

    with pytest.raises(ValueError, match="Invalid frame interval"):
        validate_evidence(make_question(), evidence)
