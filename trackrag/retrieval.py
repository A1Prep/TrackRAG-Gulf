"""Task 4's causal track-window index and deterministic text-search baseline.

The index builder validates, snapshots and groups track observations, then divides
histories into short sliding windows. candidate_windows rebuilds those windows
from observations at or before a question's cutoff. rank_evidence matches text
features from those permitted windows and returns up to three ranked intervals.

The baseline uses short track windows described by class, image location
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

Text policy:
* Describe each window using its most frequent normalized class label (lexical
  tie break), mean normalized box center in image thirds, and observed motion.
  Horizontal thirds are left/center/right; vertical thirds are top/middle/bottom.
  A coordinate exactly at 1/3 or 2/3 belongs to the middle or final third.
* With two or more observations, a center range exceeding 2% of width or height
  means moving; otherwise stationary. Net displacement exceeding that threshold
  gives leftward/rightward/upward/downward. A single observation has unknown
  motion. A returning trajectory is moving even with zero net displacement.
* Entering/exiting are image-space proxies: the first and last boxes cross a 5%
  edge margin, with center displacement over 2% inward/outward at that same edge.
  First appearance, disappearance and tracking gaps alone imply no event.
  Motion and edge comparisons allow 1e-12 normalized units for floating rounding.
* Lowercase/tokenize labels and questions, normalize common class plurals and a
  fixed motion/location vocabulary. Direction phrases such as 'moving left' are
  motion terms, not position terms. These rules apply to every question.
* Class match C is the fraction of dominant-label tokens found in the question.
  Location L and motion M each measure the fraction of requested terms matched
  in their own feature group; an unrequested group scores zero. The final score
  is (class_weight*C + location_weight*L + movement_weight*M) / sum(weights).
  This fixed weighted overlap is in [0, 1], not a probability or confidence.
* Omit zero-score windows. Sort by descending score, then ascending
  (start_frame, end_frame, track_id). Keep overlapping but distinct intervals;
  do not force different vehicles. No corpus statistics or future data are used.

This English lexical baseline does not interpret negation, intent, physical
speed, colors or visual appearance. Entry/exit cues and numeric defaults are
heuristics that require evaluation on reviewed questions and real tracks.
"""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import math
import re
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
    their sum must be finite, so the weighted score can be normalized safely.
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
    interpolated. observations is nonempty and ordered by frame_id. Windows stored
    in an index share its owned snapshots and must not be mutated by consumers.
    candidate_windows instead returns copies isolated from the stored index.

    This is an internal retrieval record, without a score or rank. Task 5 receives
    RankedInterval records from rank_evidence.
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
    available for candidate_windows to rebuild candidates at a question's cutoff
    before computing any descriptions or scores. The stored full-recording windows
    are not query-specific results. source is None only for an empty index.

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

    These candidates cover the full input; use candidate_windows for a question's
    permitted history before deriving descriptions or scores.
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


def _validate_question(question: Question, index: RetrievalIndex) -> int:
    """Validate public question fields and return the inclusive frame cutoff."""
    if not isinstance(question, Mapping):
        raise ValueError("question must be a mapping with the four public Question fields")
    required = Question.__required_keys__
    missing = required - question.keys()
    if missing:
        raise ValueError(f"Question is missing fields: {', '.join(sorted(missing))}")
    if question.keys() - required:
        raise ValueError("Question must contain only the four public fields, without gold labels")
    for field in ("question_id", "recording_id", "question"):
        value = question[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string")
    if question["recording_id"] != index.recording.recording_id:
        raise ValueError("Question recording_id does not match the indexed recording")
    query_frame = question["query_frame"]
    if type(query_frame) is not int or query_frame < 0:
        raise ValueError("query_frame must be a nonnegative integer")
    return query_frame


def candidate_windows(question: Question, index: RetrievalIndex) -> tuple[TrackWindow, ...]:
    """Return all unranked windows using only observations up to query_frame.

    Validate the question even for an empty index. The cutoff is a temporal bound;
    it need not coincide with an observed or available frame. The reviewer checks
    whether its selected query frame can be opened. This function needs no image
    or frame lookups and also works before the first observation or in a gap.

    Truncate each ordered history BEFORE splitting gaps and generating windows.
    Do not clip or filter index.windows: a crossing window's observations and
    partial tails must be rebuilt, with duplicate removal applied to the prefix.
    Future-only tracks produce no candidates. Output order follows track identity
    and window bounds, not relevance; scoring and top-k selection are separate.

    Each eligible observation is copied once per call (boxes are immutable tuples).
    The question and index are not modified, and calls in any chronological order
    are independent. Changes to returned records cannot alter subsequent queries.
    Descriptions and scoring consume only these permitted candidates.
    """
    query_frame = _validate_question(question, index)
    windows: list[TrackWindow] = []
    for history in index.histories.values():
        eligible: list[TrackObservation] = []
        for row in history:
            if row["frame_id"] > query_frame:
                break
            eligible.append(row.copy())
        windows.extend(_track_windows(tuple(eligible), index.config))
    return tuple(windows)


_CLASS_ALIASES = {
    "buses": "bus", "cars": "car", "trucks": "truck", "vans": "van",
    "bicycles": "bicycle", "bikes": "bicycle", "bike": "bicycle",
    "motorcycles": "motorcycle", "motorbikes": "motorcycle", "motorbike": "motorcycle",
    "pedestrians": "pedestrian", "people": "person", "persons": "person",
    "vehicles": "vehicle", "automobiles": "car", "automobile": "car",
}
_LOCATION_ALIASES = {
    "left": "left", "right": "right", "center": "center", "centre": "center",
    "top": "top", "upper": "top", "middle": "middle",
    "bottom": "bottom", "lower": "bottom",
}
_MOTION_VERBS = frozenset((
    "move", "moves", "moving", "moved", "heading", "going",
    "travel", "travels", "traveling", "travelling", "drive", "drives", "driving",
))
_DIRECTIONS = {
    "left": "leftward", "right": "rightward", "up": "upward", "down": "downward",
}
_DIRECTION_PHRASES = {
    **_DIRECTIONS,
    **{word: direction for direction in _DIRECTIONS.values()
       for word in (direction, direction + "s")},
}
_MOTION_ALIASES = {
    **{word: "moving" for word in _MOTION_VERBS},
    **{word: "stationary" for word in (
        "stationary", "stopped", "stopping", "stop", "parked", "standing", "still",
    )},
    **{word: "entering" for word in ("enter", "enters", "entered", "entering")},
    **{word: "exiting" for word in (
        "exit", "exits", "exited", "exiting", "leave", "leaves", "leaving",
    )},
    **{word: direction for direction in _DIRECTIONS.values()
       for word in (direction, direction + "s")},
    "up": "upward", "down": "downward",
}
_MOVEMENT_THRESHOLD = 0.02
_EDGE_MARGIN = 0.05


@dataclass(frozen=True)
class _TextFeatures:
    """Canonical keyword descriptions, kept separate to distinguish place/motion."""

    class_terms: frozenset[str]
    location_terms: frozenset[str]
    movement_terms: frozenset[str]


def _tokens(text: str) -> list[str]:
    """Normalize case, punctuation and shared class vocabulary without a model."""
    return [_CLASS_ALIASES.get(word, word)
            for word in re.findall(r"[^\W_]+", text.casefold())]


def _question_features(text: str) -> _TextFeatures:
    """Parse generic direction phrases before standalone position/motion words."""
    tokens = _tokens(text)
    classes: set[str] = set()
    locations: set[str] = set()
    movements: set[str] = set()
    position = 0
    while position < len(tokens):
        word = tokens[position]
        # 'moving [to/toward/towards] [the] left', and analogous directions.
        if word in _MOTION_VERBS:
            end = position + 1
            if end < len(tokens) and tokens[end] in ("to", "toward", "towards"):
                end += 1
            if end < len(tokens) and tokens[end] == "the":
                end += 1
            if end < len(tokens) and tokens[end] in _DIRECTION_PHRASES:
                movements.add(_DIRECTION_PHRASES[tokens[end]])
                position = end + 1
                continue
        # Also recognize adjectives such as 'left-moving bus'.
        if (word in _DIRECTION_PHRASES and position + 1 < len(tokens)
                and tokens[position + 1] in _MOTION_VERBS):
            movements.add(_DIRECTION_PHRASES[word])
            position += 2
            continue
        if word in _LOCATION_ALIASES:
            locations.add(_LOCATION_ALIASES[word])
        elif word in _MOTION_ALIASES:
            movements.add(_MOTION_ALIASES[word])
        else:
            classes.add(word)
        position += 1
    return _TextFeatures(frozenset(classes), frozenset(locations), frozenset(movements))


def _above_threshold(value: float, threshold: float) -> bool:
    """Keep exact pixel boundaries stable after normalized floating arithmetic."""
    return value > threshold and not math.isclose(value, threshold, rel_tol=0, abs_tol=1e-12)


def significant_image_displacement(delta: float) -> bool:
    """Use the same image-fraction tolerance for retrieval and cited motion evidence."""
    return _above_threshold(abs(delta), _MOVEMENT_THRESHOLD)


def _describe_window(
    window: TrackWindow, recording: Recording, dimensions: dict[int, tuple[int, int]],
) -> _TextFeatures:
    """Derive text only from this permitted window; share metadata lookups per call."""
    labels: Counter[str] = Counter()
    boxes: list[tuple[float, float, float, float]] = []
    for row in window.observations:
        labels[" ".join(_tokens(row["class_name"]))] += 1
        frame_id = row["frame_id"]
        if frame_id not in dimensions:
            dimensions[frame_id] = _frame_dimensions(recording, frame_id)
        width, height = dimensions[frame_id]
        left, top, right, bottom = row["bbox_xyxy"]
        boxes.append((left / width, top / height, right / width, bottom / height))
    label = min(labels, key=lambda name: (-labels[name], name))
    xs = [(left + right) / 2 for left, _, right, _ in boxes]
    ys = [(top + bottom) / 2 for _, top, _, bottom in boxes]
    mean_x, mean_y = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)

    def third(value: float) -> int:
        return 0 if value < 1 / 3 else 1 if value < 2 / 3 else 2

    locations = frozenset((
        ("left", "center", "right")[third(mean_x)],
        ("top", "middle", "bottom")[third(mean_y)],
    ))
    movements: set[str] = set()
    if len(boxes) >= 2:
        dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
        if _above_threshold(max(max(xs) - min(xs), max(ys) - min(ys)), _MOVEMENT_THRESHOLD):
            movements.add("moving")
        else:
            movements.add("stationary")
        for delta, negative, positive in ((dx, "leftward", "rightward"),
                                          (dy, "upward", "downward")):
            if significant_image_displacement(delta):
                movements.add(positive if delta > 0 else negative)
        first, last = boxes[0], boxes[-1]
        first_edges = (first[0], first[1], 1 - first[2], 1 - first[3])
        last_edges = (last[0], last[1], 1 - last[2], 1 - last[3])
        for before, after, inward in zip(first_edges, last_edges, (dx, dy, -dx, -dy)):
            before_inside = _above_threshold(before, _EDGE_MARGIN)
            after_inside = _above_threshold(after, _EDGE_MARGIN)
            if (not before_inside and after_inside
                    and _above_threshold(inward, _MOVEMENT_THRESHOLD)):
                movements.add("entering")
            if (before_inside and not after_inside
                    and _above_threshold(-inward, _MOVEMENT_THRESHOLD)):
                movements.add("exiting")
    return _TextFeatures(frozenset(label.split()), locations, frozenset(movements))


def _text_score(query: _TextFeatures, description: _TextFeatures, config: RetrievalConfig) -> float:
    """Fixed weighted overlap; no learned parameters or recording-wide statistics."""
    def coverage(requested: frozenset[str], observed: frozenset[str]) -> float:
        return len(requested & observed) / len(requested) if requested else 0.0

    scores = (
        coverage(description.class_terms, query.class_terms),
        coverage(query.location_terms, description.location_terms),
        coverage(query.movement_terms, description.movement_terms),
    )
    weights = (config.class_weight, config.location_weight, config.movement_weight)
    total = sum(float(weight) for weight in weights)
    return sum(float(weight) * score for weight, score in zip(weights, scores)) / total


def rank_evidence(
    question: Question,
    index: RetrievalIndex,
    k: int = 3,
) -> list[RankedInterval]:
    """Return up to k intervals ranked by class, location and image-motion text.

    Accept integer k from 0 through 3 (excluding bool). Validate the question and
    rebuild candidates using candidate_windows before scoring. k=0, an empty
    index, no eligible observations or no positive text match returns []. Invalid k
    or invalid questions raise ValueError, including when k=0 or the index is empty.

    Only the four public Question fields are retrieval inputs; gold answers and
    acceptable evidence are not. Use observations at or before query_frame for
    candidate construction, descriptions and scoring, even with a complete index.
    Repeated calls, including out-of-order query times, must not change the index.

    Descriptions and weighted overlap follow the module's documented text policy.
    Ties use ascending (start_frame, end_frame, track_id). Results preserve
    recording/source/track identity and original inclusive observed frame bounds,
    with finite scores in [0, 1] and ranks 1 through n. Metadata is looked up at most
    once per eligible frame per call; image pixels are never opened. Only positive
    scores are returned, so fewer than k results is normal. A class-only exact
    match scores 0.5 under default weights, regardless of other indexed vehicles.
    Task 5 owns evidence image export; Task 3 owns answering.
    """
    if type(k) is not int or not 0 <= k <= 3:
        raise ValueError("k must be an integer from 0 through 3")
    windows = candidate_windows(question, index)
    if k == 0 or not windows:
        return []
    query = _question_features(question["question"])
    dimensions: dict[int, tuple[int, int]] = {}
    scored: list[tuple[float, TrackWindow]] = []
    for window in windows:
        description = _describe_window(window, index.recording, dimensions)
        score = _text_score(query, description, index.config)
        if score > 0:
            scored.append((score, window))
    scored.sort(key=lambda item: (
        -item[0], item[1].start_frame, item[1].end_frame, item[1].track_id,
    ))
    return [
        {
            "recording_id": window.recording_id, "source": window.source,
            "track_id": window.track_id, "start_frame": window.start_frame,
            "end_frame": window.end_frame, "rank": rank, "score": score,
        }
        for rank, (score, window) in enumerate(scored[:k], start=1)
    ]
