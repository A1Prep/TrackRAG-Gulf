"""Task 4's retrieval interface and baseline configuration.

This first implementation step defines configuration, index storage and function
signatures. Window generation and ranking are not implemented yet; both entry
points raise NotImplementedError rather than report a successful empty search.

The intended baseline uses short track windows described by class, image location
and image-space movement. It needs no detector, language model or GPU. Defaults
are starting values, not settings validated against real recordings.

Temporal policy for the implementation:
* Measure window length and stride in original frame-number units, not seconds
  or positions in an observation list. Do not assume a recording's frame rate.
* Anchor windows at the first observation in each continuous track segment.
  A window starting at s covers [s, s + window_frames - 1], inclusively.
* Split a segment when current_frame - previous_frame - 1 exceeds max_gap_frames.
  Tune this allowance to the recording's sampling cadence. Never invent boxes
  for missing observations. Keep short and partial windows that contain data.
* Return the first and last observed frames within each effective window.
* Before constructing effective windows or computing descriptions and scores,
  restrict observations to frame_id <= question['query_frame']. Rebuild crossing
  windows from that prefix. Future observations must not affect even the scores.

The planned text baseline combines class, location and movement match scores
with the configured weights. No full-recording text statistics may affect an
earlier query. Ties use ascending (start_frame, end_frame, track_id), after sorting
by descending score. Ranking details and their tests belong to later steps.
"""

from collections.abc import Iterable
from dataclasses import dataclass
import math

from .contracts import Question, RankedInterval, Recording, TrackObservation, TrackSource


@dataclass(frozen=True)
class RetrievalConfig:
    """Validated settings shared by index construction and question ranking.

    Windows default to 30 frame-number units and advance by 10. Up to five
    missing frame numbers between consecutive observations stay in one segment.
    For example, frames 10 and 16 may share a segment; frames 10 and 17 may not.
    Stride cannot exceed window length, so the schedule cannot skip frame ranges.

    Feature weights default to class:location:movement = 2:1:1. A zero weight
    disables that feature; at least one weight must be positive. All weights and
    their sum must be finite, so a later weighted score can be normalized safely.
    The configuration is immutable; create another instance to change settings.
    """

    window_frames: int = 30
    stride_frames: int = 10
    max_gap_frames: int = 5
    class_weight: float = 2.0
    location_weight: float = 1.0
    movement_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("window_frames", "stride_frames", "max_gap_frames"):
            value = getattr(self, name)
            minimum = 0 if name == "max_gap_frames" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.stride_frames > self.window_frames:
            raise ValueError("stride_frames must not exceed window_frames")

        weights = (self.class_weight, self.location_weight, self.movement_weight)
        for name, value in zip(
            ("class_weight", "location_weight", "movement_weight"), weights,
        ):
            try:
                valid = (
                    type(value) in (int, float) and math.isfinite(value) and value >= 0
                )
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be a finite, nonnegative number")
        weight_sum = sum(float(weight) for weight in weights)
        if not math.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("Feature weights must have a finite, positive sum")


@dataclass(frozen=True)
class RetrievalIndex:
    """Index storage for one recording and one observation source.

    The builder will own a snapshot of the observations and retain frame metadata
    access through recording. Raw observations are kept so query-specific windows
    and descriptions can be derived without using future frames. Candidate storage
    will be added with window generation. source is None only for an empty index.

    The frozen wrapper prevents field reassignment; it does not freeze the shared
    dictionary record types. Consumers must not mutate the stored observations.
    """

    recording: Recording
    source: TrackSource | None
    observations: tuple[TrackObservation, ...]
    config: RetrievalConfig


def build_index(
    recording: Recording,
    tracks: Iterable[TrackObservation],
    *,
    config: RetrievalConfig | None = None,
) -> RetrievalIndex:
    """Build an index for one recording and source (not implemented yet).

    The two positional arguments preserve Task 5's expected interface. An optional
    keyword-only config overrides RetrievalConfig defaults. tracks may be a list
    or a one-pass iterable. The implementation will snapshot and validate records,
    group histories and generate short intervals without changing caller inputs.

    Mixed recordings, mixed sources and duplicate observations will be rejected.
    An empty iterable will produce an empty index with source=None. Annotated and
    predicted tracks will use this same function, in separate indexes.
    """
    raise NotImplementedError("Task 4 index construction is not implemented yet")


def rank_evidence(
    question: Question,
    index: RetrievalIndex,
    k: int = 3,
) -> list[RankedInterval]:
    """Return at most k intervals for the question (not implemented yet).

    The implementation will accept integer k from 0 through 3 (excluding bool).
    k=0, an empty index, or no eligible observations will return an empty list.
    Invalid k or a question for another recording will raise ValueError.

    Only the four public Question fields are retrieval inputs; gold answers and
    acceptable evidence are not. Use observations at or before query_frame for
    candidate construction, descriptions and scoring, even with a complete index.
    Repeated calls, including out-of-order query times, must not change the index.

    Returned records will preserve recording/source/track identity and original,
    inclusive observed frame bounds, with finite scores and ranks 1 through n.
    Task 5 owns evidence image export; Task 3 owns answering.
    """
    raise NotImplementedError("Task 4 evidence ranking is not implemented yet")
