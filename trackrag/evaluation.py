
"""Compare generated answers to the reference answer key.

Outcomes are kept distinct:
  - answered_correct   : model answered, answer aligns with reference
  - answered_incorrect : model answered, answer contradicts reference
  - insufficient_evidence : model said evidence was insufficient
  - error              : provider failure or citation mismatch
  - not_requested      : answering was never attempted

Unanswerable questions (answerable=false in the key) that the model
returns as insufficient_evidence are correct; ones it claims to answer
are incorrect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class AnswerKeyEntry(TypedDict):
    question_id: str
    expected_answer: str
    answerable: bool
    acceptable_track_ids: list[int]
    acceptable_intervals: list[dict]


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
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            result[entry["question_id"]] = entry
    return result


def load_generated_answers(path: str | Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            result[record["question_id"]] = record
    return result


_INCORRECT_SIGNALS = (
    "no pedestrian",
    "no small",
    "not visible",
    "not annotated",
    "does not appear",
    "not present",
    "not establish",
)


def _classify_outcome(
    generated: dict,
    key: AnswerKeyEntry,
) -> tuple[str, str]:
    """Return (outcome_label, notes)."""

    status = generated.get("answer_status", "not_requested")

    if status == "not_requested":
        return "not_requested", "answering was never attempted"

    if status == "error":
        return "error", "provider failure or citation mismatch"

    if status == "insufficient_evidence":
        if not key["answerable"]:
            return "answered_correct", "model correctly identified unanswerable question"
        return "answered_incorrect", "model said insufficient evidence but question is answerable"

    # status == "answered"
    answer_text = (generated.get("answer") or "").lower()

    if not key["answerable"]:
        return "answered_incorrect", "model answered a question declared unanswerable"

    expected_lower = key["expected_answer"].lower()

    # Check for gross contradiction: expected affirmative but model negated.
    if any(s in answer_text for s in _INCORRECT_SIGNALS):
        # If the expected answer is also negative this is fine.
        if not any(s in expected_lower for s in _INCORRECT_SIGNALS):
            return "answered_incorrect", "model contradicts reference answer"

    return "answered_correct", "answer aligns with reference"


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
                answer_status=gen.get("answer_status", "not_requested"),
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
        mark = "✓" if record["outcome"] == "answered_correct" else "✗"
        status = record["answer_status"]
        print(f"  {mark} {record['question_id']} [{status}] — {record['notes']}")
        if record["answer"]:
            print(f"      model: {record['answer'][:120]}")
        print(f"      ref  : {record['expected_answer'][:120]}")
