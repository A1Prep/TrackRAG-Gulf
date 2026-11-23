
"""Compare answer status and reference wording for the starter QA exercise.

Exact wording, optionally with a trailing “is/are visible”, can establish text
agreement. Additional accepted wordings may be supplied in acceptable_answers.
Other free text requires human review. Grounding and retrieval are evaluated by
the separate evaluator using the saved evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import NotRequired, TypedDict


class AnswerKeyEntry(TypedDict):
    question_id: str
    expected_answer: str
    answerable: bool
    acceptable_track_ids: list[int]
    acceptable_intervals: list[dict]
    acceptable_answers: NotRequired[list[str]]


class EvalRecord(TypedDict):
    question_id: str
    answer_status: str
    answer: str | None
    cited_evidence_ids: list[str]
    expected_answer: str
    answerable: bool
    outcome: str
    notes: str


def load_answer_key(path: str | Path) -> dict[str, AnswerKeyEntry]:
    result: dict[str, AnswerKeyEntry] = {}
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            identity = entry["question_id"]
            if identity in result:
                raise ValueError(f"Duplicate reference question ID: {identity}")
            alternatives = entry.get("acceptable_answers", [])
            if not isinstance(alternatives, list) or not all(isinstance(item, str) and item.strip() for item in alternatives):
                raise ValueError(f"Invalid acceptable_answers for {identity}")
            result[identity] = entry
    return result


def load_generated_answers(path: str | Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record.get("question"), dict):
                identity = record["question"]["question_id"]
                if record.get("question_id", identity) != identity:
                    raise ValueError("The nested and flat question IDs disagree")
                answer = record["answer"]
                record = {"question_id": identity, "answer_status": answer["answer_status"],
                          "answer": answer["answer"], "cited_evidence_ids": answer["cited_evidence_ids"]}
            identity = record["question_id"]
            if identity in result:
                raise ValueError(f"Duplicate generated question ID: {identity}")
            result[identity] = record
    return result


def _reference_words(text: str) -> list[str]:
    # Ignore sentence punctuation and one purely descriptive closing phrase.
    # Keep all nouns, counts, polarity, places, dates and relations.
    words = re.findall(r"\d+(?:[.,]\d+)*|\w+(?:['’]\w+)*|[^\w\s.,!?]", text.casefold())
    if len(words) > 2 and words[-2:] in (["is", "visible"], ["are", "visible"]):
        return words[:-2]
    return words


def _classify_outcome(
    generated: dict,
    key: AnswerKeyEntry,
) -> tuple[str, str]:
    """Return (outcome_label, notes)."""

    status = generated.get("answer_status")

    if status == "not_requested":
        return "not_requested", "answering was never attempted"

    if status == "error":
        return "error", "provider failure or citation mismatch"

    if status == "insufficient_evidence":
        if generated.get("cited_evidence_ids"):
            return "error", "abstention claims supporting citations"
        if not key["answerable"]:
            return "answered_correct", "model correctly identified unanswerable question"
        return "answered_incorrect", "model said insufficient evidence but question is answerable"

    if status != "answered":
        return "error", "unsupported answer status"
    text = generated.get("answer")
    citations = generated.get("cited_evidence_ids")
    if (not isinstance(text, str) or not text.strip() or not isinstance(citations, list) or not citations
            or not all(isinstance(item, str) and item.strip() for item in citations)
            or len(set(citations)) != len(citations)):
        return "error", "answered record has no usable text or evidence citations"

    if not key["answerable"]:
        return "answered_incorrect", "model answered a question declared unanswerable"

    actual = _reference_words(text)
    references = [key["expected_answer"], *key.get("acceptable_answers", [])]
    if actual and any(actual == _reference_words(reference) for reference in references):
        return "answered_correct", "wording matches an accepted reference"
    return "needs_review", "compare this free-text answer with the reference and cited evidence"


def evaluate(
    generated_path: str | Path,
    answer_key_path: str | Path,
) -> list[EvalRecord]:
    key = load_answer_key(answer_key_path)
    generated = load_generated_answers(generated_path)

    records: list[EvalRecord] = []

    for question_id, key_entry in key.items():
        gen = generated.get(
            question_id,
            {
                "question_id": question_id,
                "answer_status": "not_requested",
                "answer": None,
                "cited_evidence_ids": [],
            },
        )

        outcome, notes = _classify_outcome(gen, key_entry)

        records.append(
            EvalRecord(
                question_id=question_id,
                answer_status=gen.get("answer_status", "error"),
                answer=gen.get("answer"),
                cited_evidence_ids=gen.get("cited_evidence_ids", []),
                expected_answer=key_entry["expected_answer"],
                answerable=key_entry["answerable"],
                outcome=outcome,
                notes=notes,
            )
        )

    return records


def print_summary(records: list[EvalRecord]) -> None:
    from collections import Counter

    counts = Counter(r["outcome"] for r in records)
    total = len(records)

    print(f"\n{'='*60}")
    print(f"Evaluation summary  ({total} questions)")
    print(f"{'='*60}")

    for outcome, count in sorted(counts.items()):
        print(f"  {outcome:<30} {count:>3} / {total}")

    print()

    for record in records:
        status = record["answer_status"]
        print(f"  {record['question_id']} [{status}, {record['outcome']}] {record['notes']}")
        if record["answer"]:
            print(f"      model: {record['answer'][:120]}")
        print(f"      ref  : {record['expected_answer'][:120]}")
