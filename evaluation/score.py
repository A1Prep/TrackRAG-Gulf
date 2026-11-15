"""Score retrieval independently of the answer generator.

Each gold JSONL row has question_id, recording_id, query_frame, answerable and
acceptable_evidence, a list of {track_id, start_frame, end_frame} alternatives.
One matching alternative is sufficient for this retrieval metric. The scorer does
not determine whether generated answer text is factually supported.

Frame intervals include both endpoints. Predicted IDs are matched using the mean
box IoU across the union of observed frames in the retrieved interval. A frame
missing from either track contributes zero. These are pilot evaluation rules;
choose thresholds on development recordings before scoring the held-out set.
"""

import json
import math
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Invalid JSON at {path}, line {number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}, line {number}")
            rows.append(row)
    return rows


def _interval(start, end):
    if type(start) is not int or type(end) is not int or start < 0 or start > end:
        raise ValueError("Frame intervals require nonnegative integers with start <= end")
    return start, end


def _temporal_iou(left, right):
    start_a, end_a = _interval(left["start_frame"], left["end_frame"])
    start_b, end_b = _interval(right["start_frame"], right["end_frame"])
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    union = end_a - start_a + end_b - start_b + 2 - overlap
    return overlap / union


def _box_iou(left, right):
    for box in (left, right):
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("Track boxes require four finite coordinates")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Track boxes require positive area")
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection)


def _track_boxes(observations, expected_source):
    tracks = {}
    for observation in observations:
        if observation["source"] != expected_source:
            raise ValueError(f"Expected {expected_source} observations")
        key = (observation["recording_id"], observation["track_id"])
        frames = tracks.setdefault(key, {})
        frame_id = observation["frame_id"]
        if frame_id in frames:
            raise ValueError("A track has duplicate observations in one frame")
        box = observation["bbox_xyxy"]
        _box_iou(box, box)
        frames[frame_id] = box
    return tracks


def _mean_track_iou(candidate, gold, predicted_boxes, annotated_boxes):
    recording_id = candidate["recording_id"]
    predicted = predicted_boxes.get((recording_id, candidate["track_id"]), {})
    annotated = annotated_boxes.get((recording_id, gold["track_id"]), {})
    frame_ids = {
        frame_id for frame_id in predicted.keys() | annotated.keys()
        if candidate["start_frame"] <= frame_id <= candidate["end_frame"]
    }
    if not frame_ids:
        return 0.0
    return sum(
        _box_iou(predicted[frame_id], annotated[frame_id])
        for frame_id in frame_ids if frame_id in predicted and frame_id in annotated
    ) / len(frame_ids)


def score_results(
    results: Iterable[dict],
    gold: Iterable[dict],
    *,
    temporal_iou: float = 0.5,
    box_iou: float = 0.5,
    predicted_tracks: Iterable[dict] = (),
    annotated_tracks: Iterable[dict] = (),
) -> dict:
    """Count questions whose first or first three results contain acceptable evidence."""
    if not 0 < temporal_iou <= 1 or not 0 < box_iou <= 1:
        raise ValueError("Overlap thresholds must be greater than 0 and at most 1")
    by_question = {}
    for result in results:
        question_id = result["question"]["question_id"]
        if question_id in by_question:
            raise ValueError(f"Duplicate result for question {question_id}")
        by_question[question_id] = result
    predicted_boxes = _track_boxes(predicted_tracks, "predicted")
    annotated_boxes = _track_boxes(annotated_tracks, "annotated")
    counts = {"answerable": 0, "unanswerable": 0, "top1_correct": 0, "top3_correct": 0, "missing_results": 0}
    rows = []
    seen = set()
    for reference in gold:
        question_id = reference["question_id"]
        if not isinstance(question_id, str) or not question_id.strip() or question_id in seen:
            raise ValueError("Gold question IDs must be nonempty and unique")
        seen.add(question_id)
        answerable = reference["answerable"]
        if type(answerable) is not bool:
            raise ValueError(f"Question {question_id} requires a boolean answerable flag")
        query_frame = reference["query_frame"]
        _interval(query_frame, query_frame)
        alternatives = reference["acceptable_evidence"]
        if not isinstance(alternatives, list) or bool(alternatives) != answerable:
            raise ValueError("Answerable questions need evidence; unanswerable questions use an empty list")
        for alternative in alternatives:
            _interval(alternative["start_frame"], alternative["end_frame"])
            if alternative["end_frame"] > query_frame:
                raise ValueError(f"Gold evidence uses future frames for {question_id}")
        result = by_question.get(question_id)
        matches = []
        status = "missing"
        if result is None:
            counts["missing_results"] += 1
        else:
            question = result["question"]
            if (question["recording_id"], question["query_frame"]) != (
                reference["recording_id"], query_frame,
            ):
                raise ValueError(f"Result and gold recording/query time disagree for {question_id}")
            status = result["answer"]["answer_status"]
            source = result["source"]
            if source not in {"annotated", "predicted"}:
                raise ValueError("Unknown track source in results")
            if len(result["evidence"]) > 3:
                raise ValueError("Expected at most three evidence intervals")
            for rank, item in enumerate(result["evidence"], start=1):
                candidate = item["interval"]
                if candidate["rank"] != rank or candidate["source"] != source:
                    raise ValueError("Evidence ranks or sources are inconsistent")
                if candidate["recording_id"] != question["recording_id"] or candidate["end_frame"] > query_frame:
                    raise ValueError("Evidence belongs to another recording or uses future frames")
                _interval(candidate["start_frame"], candidate["end_frame"])
                if source == "predicted" and (not predicted_boxes or not annotated_boxes):
                    raise ValueError("Scoring predicted IDs requires predicted and annotated track observations")
                for alternative in alternatives:
                    if _temporal_iou(candidate, alternative) < temporal_iou:
                        continue
                    if source == "annotated":
                        same_track = candidate["track_id"] == alternative["track_id"]
                    else:
                        same_track = _mean_track_iou(candidate, alternative, predicted_boxes, annotated_boxes) >= box_iou
                    if same_track:
                        matches.append(rank)
                        break
        top1 = 1 in matches
        top3 = bool(matches)
        if answerable:
            counts["answerable"] += 1
            counts["top1_correct"] += top1
            counts["top3_correct"] += top3
        else:
            counts["unanswerable"] += 1
        rows.append({
            "question_id": question_id, "answerable": answerable,
            "matching_ranks": matches,
            "top1_correct": top1 if answerable else None,
            "top3_correct": top3 if answerable else None,
            "answer_status": status,
        })
    if not seen:
        raise ValueError("Gold file contains no questions")
    unknown = set(by_question) - seen
    if unknown:
        raise ValueError(f"Results contain questions absent from the answer key: {sorted(unknown)}")
    denominator = counts["answerable"]
    return {
        "temporal_iou_threshold": temporal_iou,
        "box_iou_threshold": box_iou,
        "counts": counts,
        "top1_accuracy": counts["top1_correct"] / denominator if denominator else None,
        "top3_recall": counts["top3_correct"] / denominator if denominator else None,
        "questions": rows,
    }
