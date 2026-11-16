"""Score a completed run against a separate answer key."""

import argparse
import json
import sys
from pathlib import Path

from .score import read_jsonl, score_results


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        from .compare import main as compare_main
        return compare_main(argv[1:])
    if argv and argv[0] == "review-template":
        from .review_template import main as template_main
        return template_main(argv[1:])
    parser = argparse.ArgumentParser(
        description="Score saved evidence retrieval without calling a model.",
        epilog="Other commands: python -m evaluation compare --help; python -m evaluation review-template --help",
    )
    parser.add_argument("--results", required=True, type=Path, help="results.jsonl from a review run")
    parser.add_argument("--gold", required=True, type=Path, help="Gold JSONL with acceptable_evidence alternatives")
    parser.add_argument("--temporal-iou", type=float, default=0.5, help="Inclusive frame-interval overlap, selected on development data")
    parser.add_argument("--box-iou", type=float, default=0.5, help="Mean box overlap for matching predicted IDs to labels")
    parser.add_argument("--predicted-tracks", type=Path, help="JSONL observations when scoring predicted results")
    parser.add_argument("--annotated-tracks", type=Path, help="JSONL observations when scoring predicted results")
    args = parser.parse_args(argv)
    try:
        scores = score_results(
            read_jsonl(args.results), read_jsonl(args.gold),
            temporal_iou=args.temporal_iou, box_iou=args.box_iou,
            predicted_tracks=read_jsonl(args.predicted_tracks) if args.predicted_tracks else (),
            annotated_tracks=read_jsonl(args.annotated_tracks) if args.annotated_tracks else (),
        )
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"Scoring failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(scores, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
