"""Task 4's versioned, model-neutral evidence pool for future learned reranking.

This module does not replace the lexical baseline or call a model. It retains all
permitted local windows, adds whole observed histories and exact scene frames,
and supplies grounded facts plus original-image references. Candidate generation
does not use gold labels, a lexical shortlist or an arbitrary top-k limit.

Scenes cover the query frame, explicitly named past frames, and the earliest
permitted observation frame (which is NOT necessarily the recording's beginning).
Whole histories can span gaps; observations, missing ranges and continuity breaks
remain explicit. First observation is not proof of entry, and missing observations
do not prove visual absence. Displacement/range describe image-space geometry,
not physical motion, speed, intent or uninterrupted visibility.

Images use the reviewer's first/middle/last plus explicitly requested observed
frames. References are not visual embeddings, captions, OCR or processed images.
A later visual model must actually read appropriate pixels; text-only models
cannot infer color or sign text from these metadata fields.
"""

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math
from typing import Literal, TypedDict

from .contracts import Box, Frame, Question, RankedInterval, TrackObservation, TrackSource
from .evidence_time import requested_frame_ids
from .retrieval import RetrievalIndex, candidate_windows


SCHEMA_VERSION = "trackrag.candidates.v1"
CandidateKind = Literal["track_window", "track_history", "scene"]


class CandidateImage(Frame):
    """Original frame metadata plus the target box; scenes have no target box."""

    bbox_xyxy: Box | None


class ObservedFeatures(TypedDict):
    observed_frame_ids: list[int]
    observation_count: int
    class_counts: dict[str, int]
    first_box_xyxy: Box
    last_box_xyxy: Box
    first_center_normalized: tuple[float, float]
    last_center_normalized: tuple[float, float]
    center_displacement_normalized: tuple[float, float] | None
    center_range_normalized: tuple[float, float]
    missing_observation_ranges: list[tuple[int, int]]
    continuity_breaks: list[tuple[int, int]]


class TrackHistory(TypedDict):
    history_id: str
    recording_id: str
    source: TrackSource
    track_id: int
    observations: list[TrackObservation]
    features: ObservedFeatures


class EvidenceCandidate(TypedDict):
    candidate_id: str
    kind: CandidateKind
    recording_id: str
    source: TrackSource
    track_id: int | None
    start_frame: int
    end_frame: int
    history_id: str | None
    features: ObservedFeatures | None
    images: list[CandidateImage]


class CandidatePolicy(TypedDict):
    window_frames: int
    stride_frames: int
    max_gap_frames: int
    scenes: str
    images: str


class CandidateBatch(TypedDict):
    schema_version: str
    question: Question
    source: TrackSource
    policy: CandidatePolicy
    histories: list[TrackHistory]
    candidates: list[EvidenceCandidate]
    unavailable_scene_frames: list[int]
    excluded_future_frame_requests: list[int]
    metadata_sha256: str


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _identifier(prefix: str, *identity: object) -> str:
    return prefix + sha256(_canonical(identity).encode("utf-8")).hexdigest()


class _Frames:
    """Cache only requested legal frames, including missing exact scene lookups."""

    def __init__(self, index: RetrievalIndex, cutoff: int):
        self.recording = index.recording
        self.cutoff = cutoff
        self.cache: dict[int, Frame] = {}
        self.missing: set[int] = set()

    def get(self, frame_id: int) -> Frame:
        if type(frame_id) is not int or not 0 <= frame_id <= self.cutoff:
            raise ValueError("Candidate frame must be at or before query_frame")
        if frame_id in self.missing:
            raise KeyError(frame_id)
        if frame_id in self.cache:
            return self.cache[frame_id]
        try:
            frame = self.recording.get_frame(frame_id)
        except KeyError:
            self.missing.add(frame_id)
            raise
        if not isinstance(frame, Mapping) or Frame.__required_keys__ - frame.keys():
            raise ValueError(f"Frame {frame_id} is missing shared frame metadata")
        if (frame["recording_id"] != self.recording.recording_id
                or type(frame["frame_id"]) is not int or frame["frame_id"] != frame_id):
            raise ValueError("Frame lookup returned a different recording or frame")
        if any(type(frame[name]) is not int or frame[name] <= 0 for name in ("width", "height")):
            raise ValueError("Frame dimensions must be positive integers")
        timestamp = frame["timestamp_s"]
        try:
            valid_time = type(timestamp) in (int, float) and math.isfinite(timestamp) and timestamp >= 0
        except OverflowError:
            valid_time = False
        if not valid_time:
            raise ValueError("Frame timestamp_s must be finite and nonnegative")
        if not isinstance(frame["image_path"], str) or not frame["image_path"].strip():
            raise ValueError("Frame image_path must be a nonempty string")
        # Copy only the public fields, without sharing the reader's mutable dict.
        snapshot: Frame = {
            "recording_id": frame["recording_id"], "frame_id": frame_id,
            "timestamp_s": float(timestamp), "width": frame["width"],
            "height": frame["height"], "image_path": frame["image_path"],
        }
        self.cache[frame_id] = snapshot
        return snapshot

    def image(self, frame_id: int, box: Box | None) -> CandidateImage:
        frame = self.get(frame_id)
        return CandidateImage(**frame, bbox_xyxy=box)


def _features(rows: list[TrackObservation], frames: _Frames, max_gap: int) -> ObservedFeatures:
    centers: list[tuple[float, float]] = []
    for row in rows:
        try:
            frame = frames.get(row["frame_id"])
        except KeyError as error:
            raise ValueError(f"Observed frame {row['frame_id']} is unavailable") from error
        left, top, right, bottom = row["bbox_xyxy"]
        width, height = frame["width"], frame["height"]
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("Observation box is outside the current original frame dimensions")
        centers.append(((left / width + right / width) / 2, (top / height + bottom / height) / 2))
    gaps = [(a["frame_id"] + 1, b["frame_id"] - 1) for a, b in zip(rows, rows[1:])
            if b["frame_id"] > a["frame_id"] + 1]
    return {
        "observed_frame_ids": [row["frame_id"] for row in rows],
        "observation_count": len(rows),
        "class_counts": dict(sorted(Counter(row["class_name"] for row in rows).items())),
        "first_box_xyxy": rows[0]["bbox_xyxy"], "last_box_xyxy": rows[-1]["bbox_xyxy"],
        "first_center_normalized": centers[0], "last_center_normalized": centers[-1],
        "center_displacement_normalized": (
            (centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1]) if len(rows) > 1 else None
        ),
        "center_range_normalized": (
            max(point[0] for point in centers) - min(point[0] for point in centers),
            max(point[1] for point in centers) - min(point[1] for point in centers),
        ),
        "missing_observation_ranges": gaps,
        "continuity_breaks": [(start, end) for start, end in gaps if end - start + 1 > max_gap],
    }


def _tracked_candidate(
    kind: CandidateKind, rows: list[TrackObservation], history_id: str,
    features: ObservedFeatures, requested: set[int], frames: _Frames,
) -> EvidenceCandidate:
    first, last = rows[0], rows[-1]
    selected = {rows[position]["frame_id"] for position in (0, (len(rows) - 1) // 2, len(rows) - 1)}
    selected.update(row["frame_id"] for row in rows if row["frame_id"] in requested)
    return {
        "candidate_id": _identifier("C-", first["recording_id"], first["source"], kind,
                                    first["track_id"], first["frame_id"], last["frame_id"]),
        "kind": kind, "recording_id": first["recording_id"], "source": first["source"],
        "track_id": first["track_id"], "start_frame": first["frame_id"], "end_frame": last["frame_id"],
        "history_id": history_id, "features": features,
        "images": [frames.image(row["frame_id"], row["bbox_xyxy"]) for row in rows if row["frame_id"] in selected],
    }


def build_candidate_batch(
    question: Question, index: RetrievalIndex, *, source: TrackSource,
) -> CandidateBatch:
    """Build a complete query-bounded pool, independently of any ranking model.

    Source is an explicit run condition, including for empty/future-only indexes;
    only eligible observations validate it. It does not establish that objects
    are visually absent or that a detector was actually run. No complete-recording
    frame scan or future frame lookup is performed.

    Local windows reuse the baseline's causal window policy. Their observations
    collectively cover the permitted histories. Identical history/window interval
    bounds are emitted once, as a history; different time intervals stay distinct.
    Scene candidates are present independently of lexical matching, with no score
    or claim of relevance. Unavailable exact scenes are omitted and reported;
    an unavailable observed track frame is an error. No nearest frame is substituted.

    Frame references match the reviewer's representative/requested-frame policy.
    The shared parser extracts explicitly written numbers, including range
    endpoints; it does not expand 'frames 1 through 100' to one hundred images.

    Records are fresh JSON-compatible snapshots. Stable IDs identify candidate
    identity, not its changing content. metadata_sha256 binds this versioned batch
    (question, policies, observations, features and frame metadata) while excluding
    image_path strings for portability. It is NOT a hash of image bytes; future
    training manifests must also bind the actual image content and reviewed labels.
    """
    if source not in ("annotated", "predicted"):
        raise ValueError("source must be 'annotated' or 'predicted'")
    windows = candidate_windows(question, index)  # Validates the public question even when empty.
    cutoff = question["query_frame"]
    frames = _Frames(index, cutoff)
    requested = set(requested_frame_ids(question["question"]))
    allowed = {frame for frame in requested if frame <= cutoff}
    grouped: dict[int, dict[int, TrackObservation]] = {}
    for window in windows:
        for row in window.observations:
            if row["source"] != source:
                raise ValueError("Eligible observations do not match the requested source")
            grouped.setdefault(row["track_id"], {})[row["frame_id"]] = row.copy()

    histories: list[TrackHistory] = []
    candidates: list[EvidenceCandidate] = []
    history_ids: dict[int, str] = {}
    emitted: set[tuple[int, int, int]] = set()
    for track_id, observations in sorted(grouped.items()):
        rows = [observations[frame] for frame in sorted(observations)]
        history_id = _identifier("H-", question["recording_id"], source, track_id)
        history_ids[track_id] = history_id
        features = _features(rows, frames, index.config.max_gap_frames)
        histories.append({
            "history_id": history_id, "recording_id": question["recording_id"], "source": source,
            "track_id": track_id, "observations": rows, "features": features,
        })
        candidates.append(_tracked_candidate("track_history", rows, history_id, deepcopy(features), allowed, frames))
        emitted.add((track_id, rows[0]["frame_id"], rows[-1]["frame_id"]))
    for window in windows:
        identity = (window.track_id, window.start_frame, window.end_frame)
        if identity in emitted:
            continue
        emitted.add(identity)
        rows = list(window.observations)
        candidates.append(_tracked_candidate(
            "track_window", rows, history_ids[window.track_id],
            _features(rows, frames, index.config.max_gap_frames), allowed, frames,
        ))

    scene_ids = allowed | {cutoff}
    if histories:
        scene_ids.add(min(history["observations"][0]["frame_id"] for history in histories))
    unavailable: list[int] = []
    for frame_id in sorted(scene_ids):
        try:
            image = frames.image(frame_id, None)
        except KeyError:
            unavailable.append(frame_id)
            continue
        candidates.append({
            "candidate_id": _identifier("C-", question["recording_id"], source, "scene", None, frame_id, frame_id),
            "kind": "scene", "recording_id": question["recording_id"], "source": source,
            "track_id": None, "start_frame": frame_id, "end_frame": frame_id,
            "history_id": None, "features": None, "images": [image],
        })
    if len({candidate["candidate_id"] for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate identifier collision")
    batch: CandidateBatch = {
        "schema_version": SCHEMA_VERSION, "question": question.copy(), "source": source,
        "policy": {"window_frames": index.config.window_frames, "stride_frames": index.config.stride_frames,
                   "max_gap_frames": index.config.max_gap_frames,
                   "scenes": "query, named past frames, earliest permitted observation",
                   "images": "first, middle floor((n-1)/2), last, explicitly requested observed frames"},
        "histories": histories, "candidates": candidates, "unavailable_scene_frames": unavailable,
        "excluded_future_frame_requests": sorted(requested - allowed), "metadata_sha256": "",
    }
    portable = deepcopy(batch)
    del portable["metadata_sha256"]
    for candidate in portable["candidates"]:
        for image in candidate["images"]:
            del image["image_path"]
    batch["metadata_sha256"] = sha256(_canonical(portable).encode("utf-8")).hexdigest()
    return batch


def candidate_to_interval(candidate: EvidenceCandidate, *, rank: int, score: float) -> RankedInterval:
    """Project a generated candidate into Task 5's existing output fields.

    The later ranker supplies its finite relevance score and final rank; this
    helper does not compute relevance, sort candidates or choose evidence. Scores
    may be logits, so no probability range is imposed. Callers must select only
    candidates in the current batch. Extra internal features never enter output.
    """
    if type(rank) is not int or not 1 <= rank <= 3:
        raise ValueError("rank must be an integer from 1 through 3")
    try:
        valid = type(score) in (int, float) and math.isfinite(score)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("score must be finite")
    return {
        "recording_id": candidate["recording_id"], "source": candidate["source"],
        "track_id": candidate["track_id"], "start_frame": candidate["start_frame"],
        "end_frame": candidate["end_frame"], "rank": rank, "score": float(score),
    }
