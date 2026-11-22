"""Shared records for the TrackRAG-Gulf tasks. Each TypedDict is an ordinary dict at runtime.

Frame numbers and boxes refer to the original image. Intervals include both end frames.
A track ID is meaningful only within a recording and track source.
"""

from collections.abc import Iterator
from typing import Literal, Protocol, TypedDict


TrackSource = Literal["annotated", "predicted"]
Box = tuple[float, float, float, float]  # left, top, right, bottom in original pixels


class Frame(TypedDict):
    recording_id: str
    frame_id: int
    timestamp_s: float
    width: int
    height: int
    image_path: str


class Recording(Protocol):
    """Task 2 supplies this object. Other tasks can implement these methods in tests."""

    recording_id: str

    def iter_frames(self) -> Iterator[Frame]:
        """Yield original frames in ascending frame_id order."""
        ...

    def get_frame(self, frame_id: int) -> Frame:
        """Return an original frame. Raise KeyError if the frame is missing."""
        ...


class TrackObservation(TypedDict):
    """One road user seen in one frame. Frames may have zero observations."""

    recording_id: str
    frame_id: int
    track_id: int
    class_name: str
    bbox_xyxy: Box
    source: TrackSource


class Question(TypedDict):
    question_id: str
    recording_id: str
    query_frame: int  # Latest frame the reviewer is allowed to use.
    question: str


class RankedInterval(TypedDict):
    """Task 4 returns these in order. A null track_id denotes scene evidence."""

    recording_id: str
    source: TrackSource
    track_id: int | None
    start_frame: int
    end_frame: int
    rank: int
    score: float


class EvidenceImage(TypedDict):
    """Task 5 writes the marked image and records its path relative to the run folder."""

    frame_id: int
    bbox_xyxy: Box | None  # Scene images have no highlighted road user.
    image_path: str


class Evidence(TypedDict):
    evidence_id: str  # Unique within this question, for example E1 or E2.
    interval: RankedInterval
    images: list[EvidenceImage]


class Answer(TypedDict):
    """A hosted-model error and a valid finding of insufficient evidence are distinct."""

    answer_status: Literal["not_requested", "answered", "insufficient_evidence", "error"]
    answer: str | None
    cited_evidence_ids: list[str]


class ReviewResult(TypedDict):
    question: Question
    source: TrackSource
    evidence: list[Evidence]
    answer: Answer
