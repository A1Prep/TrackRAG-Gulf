"""Run Task 4 retrieval on synthetic bus/car observations, without image files.

From the repository root: python -m trackrag.retrieval_demo --help
The source option labels synthetic rows; it never runs a detector or loads labels.
"""

import argparse
from collections.abc import Iterator, Sequence
import json
import sys

from .contracts import Frame, Question, RankedInterval, TrackObservation, TrackSource
from .retrieval import RetrievalConfig, build_index, rank_evidence


class _DemoRecording:
    """Metadata-only Recording implementation for this demo, not a data loader."""

    recording_id = "demo"

    def __init__(self) -> None:
        self.frames: dict[int, Frame] = {
            frame_id: {
                "recording_id": self.recording_id,
                "frame_id": frame_id,
                "timestamp_s": frame_id / 10,  # Synthetic 10 fps timeline only.
                "width": 100,
                "height": 80,
                "image_path": f"synthetic/frame-{frame_id:04d}.jpg",
            }
            for frame_id in range(90, 131)
        }

    def iter_frames(self) -> Iterator[Frame]:
        yield from self.frames.values()

    def get_frame(self, frame_id: int) -> Frame:
        return self.frames[frame_id]


def _demo_tracks(source: TrackSource) -> list[TrackObservation]:
    """A bus moves inward from the left edge; a car moves left within the image."""
    tracks: list[TrackObservation] = []
    for frame_id in range(90, 131):
        elapsed = min(frame_id - 90, 20)
        bus_left = float(2 * elapsed)
        car_left = float(70 - elapsed)
        for track_id, class_name, left, width in (
            (7, "Bus", bus_left, 10.0), (42, "Car", car_left, 20.0),
        ):
            tracks.append({
                "recording_id": "demo", "frame_id": frame_id,
                "track_id": track_id, "class_name": class_name,
                "bbox_xyxy": (left, 20.0, left + width, 40.0), "source": source,
            })
    return tracks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search synthetic bus/car tracks using Task 4's real retrieval functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--question", default="Which bus entered the scene?", help="Search question")
    parser.add_argument("--query-frame", type=int, default=100, help="Latest permitted frame ID")
    parser.add_argument("--k", type=int, choices=range(4), default=3, help="Maximum results")
    parser.add_argument("--source", choices=("annotated", "predicted"), default="annotated",
                        help="Source label for synthetic observations; no model is run")
    parser.add_argument("--format", choices=("table", "json"), default="table", help="Output format")
    defaults = RetrievalConfig()
    for name, value_type, help_text in (
        ("window_frames", int, "Interval length in original frame-number units"),
        ("stride_frames", int, "How far window starts advance"),
        ("max_gap_frames", int, "Allowed missing frame numbers before splitting a track"),
        ("class_weight", float, "Weight of vehicle-class text matches"),
        ("location_weight", float, "Weight of image-location text matches"),
        ("movement_weight", float, "Weight of image-motion text matches"),
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=value_type,
                            default=getattr(defaults, name), help=help_text)
    return parser


def _print_table(intervals: list[RankedInterval]) -> None:
    cells = [["Rank", "Recording", "Source", "Track", "Start", "End", "Score"]]
    for row in intervals:
        cells.append([
            str(row["rank"]), row["recording_id"], row["source"], str(row["track_id"]),
            str(row["start_frame"]), str(row["end_frame"]), f"{row['score']:.4f}",
        ])
    widths = [max(len(cell) for cell in column) for column in zip(*cells)]
    for row in cells:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    if not intervals:
        print("No matching intervals.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = RetrievalConfig(
            window_frames=args.window_frames, stride_frames=args.stride_frames,
            max_gap_frames=args.max_gap_frames, class_weight=args.class_weight,
            location_weight=args.location_weight, movement_weight=args.movement_weight,
        )
        question: Question = {
            "question_id": "demo-question", "recording_id": "demo",
            "query_frame": args.query_frame, "question": args.question,
        }
        # Deliberately index through frame 130 even when querying at frame 100.
        index = build_index(_DemoRecording(), _demo_tracks(args.source), config=config)
        intervals = rank_evidence(question, index, k=args.k)
    except ValueError as error:
        parser.error(str(error))

    print("Synthetic demo: bus 7 and car 42, frames 90-130; image files are not included.",
          file=sys.stderr)
    if args.format == "json":
        print(json.dumps(intervals, indent=2, allow_nan=False))
    else:
        _print_table(intervals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
