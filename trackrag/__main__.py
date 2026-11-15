"""Task 5's command-line entry point."""

import argparse
import sys
from importlib import import_module
from pathlib import Path


class AdapterUnavailable(RuntimeError):
    """An integration module or one of its dependencies is not installed."""


def _adapter(module_name, function_name, task):
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        raise AdapterUnavailable(
            f"Cannot load Task {task} adapter {module_name}.{function_name}: "
            f"{error}. Merge that task or install its dependencies before using this mode."
        ) from error
    function = getattr(module, function_name, None)
    if not callable(function):
        raise AdapterUnavailable(f"Task {task} must supply {module_name}.{function_name}()")
    return function


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find inspectable evidence in an EMT recording.")
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="Save ranked evidence images and optional answers")
    review.add_argument("--data-root", type=Path, required=True)
    review.add_argument("--recording", required=True)
    review.add_argument("--questions", type=Path, required=True, help="Task 3 question JSONL, without the answer key")
    review.add_argument("--source", choices=("annotated", "predicted"), required=True)
    review.add_argument("--answerer", choices=("none", "gemini"), default="none")
    review.add_argument("--out", type=Path, required=True, help="A new run directory for results.jsonl and images")
    args = parser.parse_args(argv)
    try:
        from .review import review_recording

        open_recording = _adapter("trackrag.emt", "open_recording", 2)
        load_questions = _adapter("trackrag.questions", "load_questions", 3)
        if args.source == "annotated":
            load_tracks = _adapter("trackrag.emt", "load_annotated_tracks", 2)
        else:
            load_tracks = _adapter("trackrag.vision", "predict_tracks", 1)
        build_index = _adapter("trackrag.retrieval", "build_index", 4)
        rank_evidence = _adapter("trackrag.retrieval", "rank_evidence", 4)
        answer_question = None
        if args.answerer == "gemini":
            answer_question = _adapter("trackrag.answering", "answer_question", 3)
        results = review_recording(
            open_recording(args.data_root, args.recording), load_questions(args.questions),
            source=args.source, load_tracks=load_tracks, build_index=build_index,
            rank_evidence=rank_evidence, answer_question=answer_question, output_dir=args.out,
        )
    except (AdapterUnavailable, ValueError, KeyError, OSError, ImportError) as error:
        print(f"Review failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved review results to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
