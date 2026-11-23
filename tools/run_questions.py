"""Run starter questions with the normal retrieval, masking and citation checks.

Use --data-root for the EMT folder containing images/ and annotations/. The
review directory keeps the source tracks and exact provider input receipts.
The additional flat export remains compatible with tools.evaluate_answers.
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from trackrag.__main__ import main as review_main
from trackrag.evaluation import load_generated_answers
from trackrag.questions import load_questions

DATA_ROOT = PROJECT_ROOT / "data" / "emt"
QUESTIONS_FILE = PROJECT_ROOT / "annotations" / "questions.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "generated_answers.jsonl"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="EMT root containing images and annotations")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE, help="Questions for one recording")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "starter-review",
                        help="New directory for the full review, processed images, evidence and receipts")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Flat question-ID/answer JSONL export")
    parser.add_argument("--source", choices=["annotated", "predicted"], default="annotated")
    parser.add_argument("--answerer", choices=["none", "gemini"], default="gemini")
    parser.add_argument("--retrieval-adapter", help="Optional retrieval implementation for the normal reviewer")
    args = parser.parse_args(argv)
    try:
        recordings = {q["recording_id"] for q in load_questions(args.questions)}
    except (OSError, ValueError) as error:
        print(f"Cannot load starter questions: {error}", file=sys.stderr)
        return 1
    if len(recordings) != 1:
        parser.error("The question file must contain questions for exactly one recording.")
    output = args.output.resolve()
    if output == args.questions.resolve() or output.is_relative_to(args.run_dir.resolve()):
        parser.error("Choose a separate answer export outside the review directory and the question file.")

    command = ["review", "--data-root", str(args.data_root), "--recording", recordings.pop(),
               "--questions", str(args.questions), "--source", args.source, "--answerer", args.answerer,
               "--out", str(args.run_dir)]
    if args.retrieval_adapter:
        command.extend(["--retrieval-adapter", args.retrieval_adapter])
    code = review_main(command)
    if code:
        return code

    records = load_generated_answers(args.run_dir / "results.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the convenient export only after the canonical review is complete.
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        staged = Path(work) / "answers.jsonl"
        with staged.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records.values():
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        staged.replace(output)
    errors = sum(record["answer_status"] == "error" for record in records.values())
    print(f"Saved {len(records)} answers to {output}. Operational errors: {errors}.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
