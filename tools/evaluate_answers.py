
"""Compare generated answers to the reference answer key and print a report."""

import json
from pathlib import Path

from trackrag.evaluation import evaluate, print_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANSWER_KEY = PROJECT_ROOT / "annotations" / "answer_key.jsonl"
GENERATED = PROJECT_ROOT / "outputs" / "generated_answers.jsonl"
EVAL_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation_results.jsonl"


def main():
    if not GENERATED.is_file():
        print(f"Generated answers not found: {GENERATED}")
        print("Run tools/run_questions.py first.")
        return

    records = evaluate(GENERATED, ANSWER_KEY)
    print_summary(records)

    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with EVAL_OUTPUT.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    print(f"\nEvaluation saved to: {EVAL_OUTPUT}")


if __name__ == "__main__":
    main()
