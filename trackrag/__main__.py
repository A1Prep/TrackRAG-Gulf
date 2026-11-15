"""Task 5's review and adapter-check commands."""

import argparse
import inspect
import json
import sys
from importlib import import_module
from pathlib import Path


class AdapterUnavailable(RuntimeError):
    """An integration module or one of its dependencies is not installed."""


def _adapter(module_name, function_name, task, adapter_module=None):
    module_name = adapter_module or module_name
    try:
        module = import_module(module_name)
    except Exception as error:
        raise AdapterUnavailable(
            f"Cannot load Task {task} adapter {module_name}.{function_name}: "
            f"{type(error).__name__}: {error}. Check that task and its dependencies."
        ) from error
    function = getattr(module, function_name, None)
    if not callable(function):
        raise AdapterUnavailable(f"Task {task} must supply {module_name}.{function_name}()")
    return function


def check_adapters(source, answerer, adapter_module=None):
    """Check callable signatures without opening footage or running any models."""
    requirements = [
        ("trackrag.emt", "open_recording", 2, (Path("."), "recording"), {}),
        ("trackrag.questions", "load_questions", 3, (Path("questions.jsonl"),), {}),
        ("trackrag.retrieval", "build_index", 4, (None, []), {}),
        ("trackrag.retrieval", "rank_evidence", 4, ({}, None), {"k": 3}),
    ]
    if source == "annotated":
        requirements.append(("trackrag.emt", "load_annotated_tracks", 2, (None,), {}))
    else:
        requirements.append(("trackrag.vision", "predict_tracks", 1, (None,), {}))
    if answerer == "gemini":
        requirements.append(("trackrag.answering", "answer_question", 3, ({}, [], Path(".")), {}))
    checks = []
    for module, name, task, positional, keyword in requirements:
        selected_module = adapter_module or module
        check = {"task": task, "function": f"{selected_module}.{name}", "status": "ready"}
        try:
            function = _adapter(module, name, task, adapter_module)
            inspect.signature(function).bind(*positional, **keyword)
        except (AdapterUnavailable, TypeError, ValueError) as error:
            check.update(status="unavailable", detail=str(error))
        checks.append(check)
    return {
        "source": source, "answerer": answerer, "adapter_module": adapter_module,
        "ready": all(check["status"] == "ready" for check in checks), "adapters": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find inspectable evidence in an EMT recording.")
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="Save ranked evidence images and optional answers")
    review.add_argument("--data-root", type=Path, required=True)
    review.add_argument("--recording", required=True)
    review.add_argument("--questions", type=Path, required=True, help="Task 3 question JSONL, without the answer key")
    review.add_argument("--source", choices=("annotated", "predicted"), required=True)
    review.add_argument("--answerer", choices=("none", "gemini"), default="none")
    review.add_argument("--out", type=Path, required=True, help="A new run directory for results.jsonl, tracks.jsonl and images")
    review.add_argument("--adapters", help="Optional module supplying development adapters")
    check = commands.add_parser("check-adapters", help="Report missing modules or incompatible function signatures")
    check.add_argument("--source", choices=("annotated", "predicted"), required=True)
    check.add_argument("--answerer", choices=("none", "gemini"), default="none")
    check.add_argument("--adapters", help="Optional module supplying development adapters")
    args = parser.parse_args(argv)
    if args.command == "check-adapters":
        report = check_adapters(args.source, args.answerer, args.adapters)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 1
    try:
        from .review import review_recording

        open_recording = _adapter("trackrag.emt", "open_recording", 2, args.adapters)
        load_questions = _adapter("trackrag.questions", "load_questions", 3, args.adapters)
        if args.source == "annotated":
            load_tracks = _adapter("trackrag.emt", "load_annotated_tracks", 2, args.adapters)
        else:
            load_tracks = _adapter("trackrag.vision", "predict_tracks", 1, args.adapters)
        build_index = _adapter("trackrag.retrieval", "build_index", 4, args.adapters)
        rank_evidence = _adapter("trackrag.retrieval", "rank_evidence", 4, args.adapters)
        answer_question = None
        if args.answerer == "gemini":
            answer_question = _adapter("trackrag.answering", "answer_question", 3, args.adapters)
        recording = open_recording(args.data_root, args.recording)
        if recording.recording_id != args.recording:
            raise ValueError("Recording loader returned a different recording than requested")
        results = review_recording(
            recording, load_questions(args.questions),
            source=args.source, load_tracks=load_tracks, build_index=build_index,
            rank_evidence=rank_evidence, answer_question=answer_question, output_dir=args.out,
            run_metadata={"adapter_module": args.adapters or "task_modules", "answerer": args.answerer},
        )
    except Exception as error:
        print(f"Review failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Saved review results to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
