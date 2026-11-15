"""Task 4's retrieval interface and baseline configuration.

The index builder validates, snapshots and groups track observations, then divides
histories into short sliding windows. Ranking is not implemented yet;
rank_evidence raises NotImplementedError rather than report a successful search.

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
  Skip windows with no observations and retain only one copy of an exact
  (track identity, first observed frame, last observed frame) duplicate.
* Before constructing effective windows or computing descriptions and scores,
  restrict observations to frame_id <= question['query_frame']. Rebuild crossing
  windows from that prefix. Future observations must not affect even the scores.

The planned text baseline combines class, location and movement match scores
with the configured weights. No full-recording text statistics may affect an
earlier query. Ties use ascending (start_frame, end_frame, track_id), after sorting
by descending score. Ranking details and their tests belong to later steps.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from .contracts import Question, RankedInterval, Recording, TrackObservation, TrackSource


TrackKey = tuple[str, TrackSource, int]  # recording_id, source, track_id


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
class TrackWindow:
    """An unranked candidate containing observations of one road user.

    start_frame and end_frame are inclusive original IDs of actual observations,
    which may be narrower than the configured window span. No missing boxes are
    interpolated. observations is nonempty and ordered by frame_id; its records
    share the index's owned snapshots and must not be mutated by consumers.

    This is an internal retrieval record, without a score or rank. Task 5 will
    receive RankedInterval records only after question-specific ranking exists.
    """

    recording_id: str
    source: TrackSource
    track_id: int
    start_frame: int
    end_frame: int
    observations: tuple[TrackObservation, ...]


@dataclass(frozen=True)
class RetrievalIndex:
    """Index storage for one recording and one observation source.

    The builder owns a snapshot of the observations and retains frame metadata
    access through recording. observations is ordered by (frame_id, track_id).
    histories maps full track identities to tuples ordered by frame_id, with keys
    in sorted order. Its records are the same snapshots as in observations.

    windows contains unranked candidates for the complete supplied histories,
    ordered by full track identity, then start/end frame. Raw histories remain
    available to rebuild candidates at a question's cutoff before computing any
    descriptions or scores. The stored full-recording windows are not yet safe
    query-specific results. source is None only for an empty index.

    The wrapper and history mapping prevent structural edits; they do not freeze
    the shared dictionary record types or recording. Consumers must not mutate
    stored observations. Caller-owned observation dicts and boxes are not shared.
    """

    recording: Recording
    source: TrackSource | None
    observations: tuple[TrackObservation, ...]
    config: RetrievalConfig
    histories: Mapping[TrackKey, tuple[TrackObservation, ...]]
    windows: tuple[TrackWindow, ...]


def _snapshot_observation(
    observation: TrackObservation, recording_id: str,
) -> TrackObservation:
    """Validate the six shared fields and copy a row before advancing its iterator."""
    if not isinstance(observation, Mapping):
        raise ValueError("Each track observation must be a mapping")
    missing = TrackObservation.__required_keys__ - observation.keys()
    if missing:
        raise ValueError(f"Track observation is missing fields: {', '.join(sorted(missing))}")
    if observation["recording_id"] != recording_id:
        raise ValueError("Observation recording_id does not match the recording")
    for field in ("frame_id", "track_id"):
        value = observation[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    class_name = observation["class_name"]
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name must be a nonempty string")
    source = observation["source"]
    if source not in ("annotated", "predicted"):
        raise ValueError("source must be 'annotated' or 'predicted'")

    box = observation["bbox_xyxy"]
    if (
        not isinstance(box, (tuple, list)) or len(box) != 4
        or any(type(value) not in (int, float) for value in box)
    ):
        raise ValueError("bbox_xyxy must contain four finite numbers")
    try:
        left, top, right, bottom = (float(value) for value in box)
    except OverflowError as error:
        raise ValueError("bbox_xyxy must contain four finite numbers") from error
    copied_box = (left, top, right, bottom)
    if not all(math.isfinite(value) for value in copied_box):
        raise ValueError("bbox_xyxy must contain four finite numbers")
    if not (0 <= left < right and 0 <= top < bottom):
        raise ValueError("bbox_xyxy must have nonnegative coordinates and positive area")

    # Reconstruct the shared record rather than retaining mutable caller objects
    # or copying optional upstream fields that retrieval does not consume.
    return {
        "recording_id": recording_id, "frame_id": observation["frame_id"],
        "track_id": observation["track_id"], "class_name": class_name,
        "bbox_xyxy": copied_box, "source": source,
    }


def _frame_dimensions(recording: Recording, frame_id: int) -> tuple[int, int]:
    """Resolve original dimensions without reading image pixels."""
    try:
        frame = recording.get_frame(frame_id)
    except KeyError as error:
        raise ValueError(f"Observation references missing frame {frame_id}") from error
    if not isinstance(frame, Mapping):
        raise ValueError(f"Lookup for frame {frame_id} must return frame metadata")
    if (
        frame.get("recording_id") != recording.recording_id
        or type(frame.get("frame_id")) is not int
        or frame.get("frame_id") != frame_id
    ):
        raise ValueError("Frame lookup returned a different recording or frame")
    for dimension in ("width", "height"):
        value = frame.get(dimension)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Frame {frame_id} {dimension} must be a positive integer")
    return frame["width"], frame["height"]


def _segment_windows(
    segment: tuple[TrackObservation, ...], config: RetrievalConfig,
) -> Iterator[TrackWindow]:
    """Slide over a nonempty, ordered segment using original frame-number units."""
    first = segment[0]
    window_start = first["frame_id"]
    last_frame = segment[-1]["frame_id"]
    left = right = 0
    previous_bounds: tuple[int, int] | None = None
    while window_start <= last_frame:
        window_end = window_start + config.window_frames - 1
        while left < len(segment) and segment[left]["frame_id"] < window_start:
            left += 1
        right = max(right, left)
        while right < len(segment) and segment[right]["frame_id"] <= window_end:
            right += 1
        if left == right:
            # Jump over empty spans while preserving the segment's stride grid.
            # stride <= window length guarantees we cannot skip the next sighting.
            distance = segment[left]["frame_id"] - window_end
            steps = (distance + config.stride_frames - 1) // config.stride_frames
            window_start += steps * config.stride_frames
            continue
        bounds = (segment[left]["frame_id"], segment[right - 1]["frame_id"])
        # Both pointers only advance, so identical observed ranges are consecutive.
        if bounds != previous_bounds:
            yield TrackWindow(
                recording_id=first["recording_id"], source=first["source"],
                track_id=first["track_id"], start_frame=bounds[0], end_frame=bounds[1],
                observations=segment[left:right],
            )
            previous_bounds = bounds
        window_start += config.stride_frames


def _track_windows(
    history: tuple[TrackObservation, ...], config: RetrievalConfig,
) -> Iterator[TrackWindow]:
    """Split a validated, ordered history at gaps and restart the window grid."""
    segment_start = 0
    for position in range(1, len(history) + 1):
        if position == len(history) or (
            history[position]["frame_id"] - history[position - 1]["frame_id"] - 1
            > config.max_gap_frames
        ):
            yield from _segment_windows(history[segment_start:position], config)
            segment_start = position


def build_index(
    recording: Recording,
    tracks: Iterable[TrackObservation],
    *,
    config: RetrievalConfig | None = None,
) -> RetrievalIndex:
    """Validate, group and window observations for one recording and source.

    The two positional arguments preserve Task 5's expected interface. An optional
    keyword-only config overrides RetrievalConfig defaults. tracks may be a list
    or a one-pass iterable. Each row is copied immediately, so a generator may
    reuse a mutable dict. Only the six shared observation fields are retained;
    class labels are preserved per frame and boxes become tuples of floats.

    Invalid records, mixed recordings/sources, duplicate track-frame pairs, missing
    frames and boxes outside original frame dimensions raise ValueError. Boxes are
    never clipped. Metadata is looked up once per referenced frame; images are not
    opened. Caller-owned observations are not modified.

    An empty iterable produces an empty index with source=None and no windows.
    Annotated and predicted tracks use this same function, in separate indexes.
    Histories split when missing frame numbers exceed config.max_gap_frames.
    Within each segment, windows start at its first observation and advance by
    config.stride_frames, retaining short and partial tails. Empty windows and
    exact duplicate observed intervals are omitted. Bounds refer to actual frames.

    These candidates cover the full input; a later query must rebuild windows
    from histories truncated at query_frame before deriving descriptions/scores.
    """
    if config is None:
        config = RetrievalConfig()
    elif not isinstance(config, RetrievalConfig):
        raise TypeError("config must be a RetrievalConfig instance or None")
    recording_id = getattr(recording, "recording_id", None)
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError("Recording recording_id must be a nonempty string")
    try:
        rows = iter(tracks)
    except TypeError as error:
        raise TypeError("tracks must be an iterable of track observations") from error

    source: TrackSource | None = None
    grouped: dict[TrackKey, dict[int, TrackObservation]] = {}
    dimensions: dict[int, tuple[int, int]] = {}
    for row in rows:
        copied = _snapshot_observation(row, recording_id)
        if source is None:
            source = copied["source"]
        elif copied["source"] != source:
            raise ValueError("Track observations mix sources; build one index per source")
        key = (recording_id, copied["source"], copied["track_id"])
        history = grouped.setdefault(key, {})
        frame_id = copied["frame_id"]
        if frame_id in history:
            raise ValueError(f"Duplicate observation for track {copied['track_id']} in frame {frame_id}")
        if frame_id not in dimensions:
            dimensions[frame_id] = _frame_dimensions(recording, frame_id)
        width, height = dimensions[frame_id]
        if copied["bbox_xyxy"][2] > width or copied["bbox_xyxy"][3] > height:
            raise ValueError(f"bbox_xyxy for track {copied['track_id']} is outside frame {frame_id}")
        history[frame_id] = copied

    histories = {
        key: tuple(history[frame_id] for frame_id in sorted(history))
        for key, history in sorted(grouped.items())
    }
    observations = tuple(sorted(
        (row for history in histories.values() for row in history),
        key=lambda row: (row["frame_id"], row["track_id"]),
    ))
    windows = tuple(
        window for history in histories.values() for window in _track_windows(history, config)
    )
    return RetrievalIndex(
        recording=recording, source=source, observations=observations,
        config=config, histories=MappingProxyType(histories), windows=windows,
    )


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
