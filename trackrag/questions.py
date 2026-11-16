
"""Load and validate reviewer questions and their evidence."""

import json
from pathlib import Path
from typing import cast

from trackrag.contracts import Question, Evidence


REQUIRED_FIELDS = {
    "question_id",
    "recording_id",
    "query_frame",
    "question",
}


def validate_question(record: object, line_number: int | None = None) -> Question:
    """Validate one question record."""

    location = f"Line {line_number}" if line_number else "Question"

    if not isinstance(record, dict):
        raise ValueError(f"{location}: Expected a JSON object.")

    missing = REQUIRED_FIELDS - record.keys()
    extra = record.keys() - REQUIRED_FIELDS

    if missing:
        raise ValueError(f"{location}: Missing fields: {sorted(missing)}")

    if extra:
        raise ValueError(f"{location}: Unexpected fields: {sorted(extra)}")

    for field in ("question_id", "recording_id", "question"):
        value = record[field]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{location}: '{field}' must be a non-empty string.")

    query_frame = record["query_frame"]

    if isinstance(query_frame, bool) or not isinstance(query_frame, int):
        raise ValueError(f"{location}: 'query_frame' must be an integer.")

    if query_frame < 0:
        raise ValueError(f"{location}: 'query_frame' cannot be negative.")

    return cast(Question, record)


def validate_questions(records: list[object]) -> list[Question]:
    """Validate a collection of questions and reject duplicate IDs."""

    validated = []
    seen_ids = set()

    for index, record in enumerate(records, start=1):
        question = validate_question(record, index)

        question_id = question["question_id"]

        if question_id in seen_ids:
            raise ValueError(f"Duplicate question ID: {question_id}")

        seen_ids.add(question_id)
        validated.append(question)

    return validated


def load_questions(path: str | Path) -> list[Question]:
    """Load questions from a JSONL file."""

    questions = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error.msg}"
                ) from error

            questions.append(record)

    return validate_questions(questions)


def validate_evidence(question: Question, evidence: list[Evidence]) -> None:
    """Reject evidence that does not belong to the question's time boundary."""

    seen_ids = set()

    for item in evidence:
        evidence_id = item["evidence_id"]
        interval = item["interval"]

        if evidence_id in seen_ids:
            raise ValueError(f"Duplicate evidence ID: {evidence_id}")

        seen_ids.add(evidence_id)

        if interval["recording_id"] != question["recording_id"]:
            raise ValueError(
                f"{evidence_id}: Evidence belongs to a different recording."
            )

        start = interval["start_frame"]
        end = interval["end_frame"]

        if start < 0 or end < start:
            raise ValueError(f"{evidence_id}: Invalid frame interval.")

        if end > question["query_frame"]:
            raise ValueError(
                f"{evidence_id}: Evidence interval ends after query_frame."
            )

        for image in item["images"]:
            frame_id = image["frame_id"]

            if frame_id < start or frame_id > end:
                raise ValueError(
                    f"{evidence_id}: Image frame {frame_id} "
                    "is outside its evidence interval."
                )

            if frame_id > question["query_frame"]:
                raise ValueError(
                    f"{evidence_id}: Image is dated after query_frame."
                )
