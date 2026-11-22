"""Compare a starter review or flat answer file with its reference key offline."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from trackrag.evaluation import evaluate, print_summary

ANSWER_KEY = PROJECT_ROOT / "annotations" / "answer_key.jsonl"
GENERATED = PROJECT_ROOT / "outputs" / "generated_answers.jsonl"
EVAL_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation_results.jsonl"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, default=GENERATED, help="Full review results.jsonl or flat answers")
    parser.add_argument("--answer-key", type=Path, default=ANSWER_KEY)
    parser.add_argument("--output", type=Path, default=EVAL_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.resolve() in {args.generated.resolve(), args.answer_key.resolve()}:
        parser.error("Choose a report path distinct from both input files.")
    try:
        records = evaluate(args.generated, args.answer_key)
    except (OSError, ValueError) as error:
        print(f"Cannot evaluate the supplied files: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print_summary(records)
    print(f"Saved {len(records)} evaluation records to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
