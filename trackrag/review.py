"""Run evidence review with the adapters supplied by Tasks 1–4."""

import json
import math
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .contracts import (
    Answer, Evidence, Question, RankedInterval, Recording, ReviewResult,
    TrackObservation, TrackSource,
)
from .evidence_time import requested_frame_ids


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_questions(recording: Recording, questions: Sequence[Question]) -> None:
    seen = set()
    for question in questions:
        if set(question) != {"question_id", "recording_id", "query_frame", "question"}:
            raise ValueError("Question inputs must contain only the four public fields, without gold labels")
        question_id = question["question_id"]
        if not isinstance(question_id, str) or not question_id.strip() or question_id in seen:
            raise ValueError("Question IDs must be nonempty, unique strings")
        seen.add(question_id)
        if question["recording_id"] != recording.recording_id:
            raise ValueError(f"Question {question_id} belongs to a different recording")
        _integer(question["query_frame"], "query_frame")
        if not isinstance(question["question"], str) or not question["question"].strip():
            raise ValueError(f"Question {question_id} has no question text")
        # Fail before running expensive adapters if the selected moment is unavailable.
        frame = recording.get_frame(question["query_frame"])
        if (frame["recording_id"], frame["frame_id"]) != (
            recording.recording_id, question["query_frame"],
        ):
            raise ValueError(f"Frame lookup returned a different frame for {question_id}")


def _observations_by_track(
    recording: Recording, tracks: Sequence[TrackObservation], source: TrackSource,
) -> dict[int, dict[int, TrackObservation]]:
    grouped: dict[int, dict[int, TrackObservation]] = {}
    for observation in tracks:
        if observation["recording_id"] != recording.recording_id or observation["source"] != source:
            raise ValueError("Track observations mix recordings or sources")
        track_id = _integer(observation["track_id"], "track_id")
        frame_id = _integer(observation["frame_id"], "frame_id")
        frames = grouped.setdefault(track_id, {})
        if frame_id in frames:
            raise ValueError(f"Track {track_id} has two boxes in frame {frame_id}")
        frames[frame_id] = observation
    return grouped


def _validate_interval(
    interval: RankedInterval, question: Question, source: TrackSource, rank: int,
) -> None:
    if interval["recording_id"] != question["recording_id"] or interval["source"] != source:
        raise ValueError("Retrieved evidence belongs to a different recording or source")
    start = _integer(interval["start_frame"], "start_frame")
    end = _integer(interval["end_frame"], "end_frame")
    if interval["track_id"] is not None:
        _integer(interval["track_id"], "track_id")
    if start > end or end > question["query_frame"]:
        raise ValueError("Retrieved evidence has a reversed interval or uses future frames")
    if type(interval["rank"]) is not int or interval["rank"] != rank:
        raise ValueError("Evidence ranks must start at 1 and follow their returned order")
    if isinstance(interval["score"], bool) or not math.isfinite(interval["score"]):
        raise ValueError("Evidence score must be finite")


def _resolve_evidence(
    recording: Recording,
    intervals: Sequence[RankedInterval],
    question: Question,
    source: TrackSource,
    tracks: dict[int, dict[int, TrackObservation]],
    run_dir: Path,
) -> list[Evidence]:
    if len(intervals) > 3:
        raise ValueError("Retriever returned more than the requested three results")
    evidence: list[Evidence] = []
    for rank, interval in enumerate(intervals, start=1):
        _validate_interval(interval, question, source, rank)
        observations = tracks.get(interval["track_id"], {})
        if interval["track_id"] is None:
            # Scene references are exact snapshots. A missing frame is never substituted.
            if interval["start_frame"] != interval["end_frame"]:
                raise ValueError("Scene evidence must reference one exact frame")
            available = [interval["start_frame"]]
        else:
            available = sorted(
                frame_id for frame_id in observations
                if interval["start_frame"] <= frame_id <= interval["end_frame"]
            )
        if not available:
            raise ValueError("Retrieved interval has no observed box for its track")
        evidence_id = f"E{rank}"
        images = []
        # Show the beginning, middle and end of the observations, including short tracks.
        selected_frames={available[position] for position in {0,(len(available)-1)//2,len(available)-1}}
        # Exact requested observations matter for discontiguous questions such
        # as presence in four named frames. An interval alone cannot prove them.
        selected_frames.update(set(requested_frame_ids(question['question']))&set(available))
        for frame_id in sorted(selected_frames):
            frame = recording.get_frame(frame_id)
            if (frame["recording_id"], frame["frame_id"]) != (
                question["recording_id"], frame_id,
            ):
                raise ValueError("Frame lookup returned a different recording or frame")
            box = observations[frame_id]["bbox_xyxy"] if observations else None
            if box is not None and (len(box) != 4 or not all(math.isfinite(value) for value in box)):
                raise ValueError("A box must have four finite coordinates")
            if box is not None:
                left, top, right, bottom = box
                if not (0 <= left < right <= frame["width"] and 0 <= top < bottom <= frame["height"]):
                    raise ValueError(f"Box for track {interval['track_id']} is outside frame {frame_id}")
            track_name = "scene" if interval["track_id"] is None else f"track-{interval['track_id']}"
            relative_path = Path("evidence") / track_name / f"{frame_id}.png"
            destination = run_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with Image.open(frame["image_path"]) as original:
                    if original.size != (frame["width"], frame["height"]):
                        raise ValueError(f"Image dimensions disagree with frame metadata for {frame_id}")
                    marked = original.convert("RGB")
                    if box is not None:
                        ImageDraw.Draw(marked).rectangle(box, outline=(255, 0, 0), width=2)
                    marked.save(destination)
            images.append({
                "frame_id": frame_id,
                "bbox_xyxy": box,
                "image_path": relative_path.as_posix(),
            })
        evidence.append({"evidence_id": evidence_id, "interval": dict(interval), "images": images})
    return evidence


def _answer(
    question: Question,
    evidence: list[Evidence],
    run_dir: Path,
    answer_question: Callable[[Question, list[Evidence], Path], Answer] | None,
) -> Answer:
    if answer_question is None:
        return {"answer_status": "not_requested", "answer": None, "cited_evidence_ids": []}
    if not evidence:
        explanation = None
        record = run_dir / "run.json"
        if record.is_file():
            clock = json.loads(record.read_bytes()).get("playback_scope", {}).get(question["question_id"])
            if clock and clock["status"] == "outside_review_window":
                explanation = "That time is outside the footage available from the start through the paused frame."
            elif clock and clock["status"] == "unavailable":
                explanation = "There is no original frame at that exact time in the available recording."
            elif clock and clock["status"] == "conflicting_frame_and_time":
                explanation = "The stated frame and time refer to different moments. Which one should I check?"
        return {"answer_status": "insufficient_evidence", "answer": explanation, "cited_evidence_ids": []}
    valid_ids = {item["evidence_id"] for item in evidence}
    try:
        answer = answer_question(deepcopy(question), deepcopy(evidence), run_dir)
        status = answer["answer_status"]
        citations = answer["cited_evidence_ids"]
        if status not in {"answered", "insufficient_evidence", "error"}:
            raise ValueError("Answerer returned an unsupported status")
        if (not isinstance(citations, list) or any(not isinstance(citation,str) or citation not in valid_ids for citation in citations)
                or len(set(citations))!=len(citations)):
            raise ValueError("Answerer cited evidence that was not supplied")
        if answer["answer"] is not None and not isinstance(answer["answer"], str):
            raise ValueError("Answer text must be a string or null")
        if status == "answered" and (not citations or not answer["answer"] or not answer["answer"].strip()):
            raise ValueError("An answered result requires text and a valid citation")
        if status in {"insufficient_evidence", "error"} and citations:
            raise ValueError("An unanswered result cannot claim supporting citations")
        return {key: answer[key] for key in ("answer_status", "answer", "cited_evidence_ids")}
    except Exception as error:
        # A failed answer must stay visible in the run, including later questions.
        return {
            "answer_status": "error", "answer": f"{type(error).__name__}: {error}",
            "cited_evidence_ids": [],
        }


def review_recording(
    recording: Recording,
    questions: Iterable[Question],
    *,
    source: TrackSource,
    load_tracks: Callable[[Recording], Iterable[TrackObservation]],
    build_index: Callable[[Recording, Sequence[TrackObservation]], Any],
    rank_evidence: Callable[..., Iterable[RankedInterval]],
    output_dir: Path,
    answer_question: Callable[[Question, list[Evidence], Path], Answer] | None = None,
    run_metadata: dict | None = None,
    image_guard=None,
) -> Path:
    """Publish one run after all evidence files and JSON records have been written.

    Callers supply the selected adapters. Paths stored in results are relative to the
    completed run directory. Use a fresh output directory for each comparison.
    """
    if source not in {"annotated", "predicted"}:
        raise ValueError("Track source must be annotated or predicted")
    output_dir = Path(output_dir).absolute()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}. Choose a new run directory.")
    questions = deepcopy(list(questions))
    if not questions:
        raise ValueError("The question file contains no questions")
    _validate_questions(recording, questions)
    observations = deepcopy(list(load_tracks(recording)))
    by_track = _observations_by_track(recording, observations, source)
    index = build_index(recording, deepcopy(observations))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trackrag-", dir=output_dir.parent) as temporary:
        run_dir = Path(temporary) / "run"
        run_dir.mkdir()
        metadata = {
            "recording_id": recording.recording_id, "source": source,
            "first_frame_id": next(recording.iter_frames())["frame_id"],
            "recording_adapter": f"{type(recording).__module__}.{type(recording).__name__}",
            "question_count": len(questions), "observation_count": len(observations),
            "configuration": run_metadata or {},
        }
        if "trackrag.retrieval_temporal" in (metadata["configuration"].get("retrieval"),
                                            metadata["configuration"].get("retrieval_adapter")):
            from trackrag.playback_time import playback_scope
            from trackrag.retrieval_temporal import rank_evidence as rank_temporal
            if rank_evidence is rank_temporal:
                rank_evidence = partial(rank_evidence, observation_source=source)
            clocks = {}
            for q in questions:
                clock = playback_scope(q["question"], recording, q["query_frame"])
                if clock:
                    clocks[q["question_id"]] = clock
            if clocks:
                metadata["playback_scope"] = clocks
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (run_dir / "tracks.jsonl").open("w", encoding="utf-8", newline="\n") as exported:
            for observation in observations:
                exported.write(json.dumps(observation, allow_nan=False) + "\n")
        with (run_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as results:
            for question in questions:
                intervals = deepcopy(list(rank_evidence(deepcopy(question), index, k=3)))
                with image_guard if image_guard is not None else nullcontext():
                    evidence = _resolve_evidence(
                        recording, intervals, question, source, by_track, run_dir,
                    )
                result: ReviewResult = {
                    "question": dict(question), "source": source, "evidence": evidence,
                    "answer": _answer(question, evidence, run_dir, answer_question),
                }
                results.write(json.dumps(result, allow_nan=False) + "\n")
        # Publish the images and their references together. Incomplete runs are removed.
        # Native Windows scanners may briefly lock the completed directory.
        # Never retry unrelated errors or overwrite a concurrent destination.
        for attempt,delay in enumerate((0,0.05,0.15,0.3)):
            if delay:
                time.sleep(delay)
            if output_dir.exists():
                raise FileExistsError('Output was created during review; choose a fresh directory')
            try:
                run_dir.rename(output_dir)
                break
            except PermissionError as error:
                if getattr(error,'winerror',None) not in {5,32,33} or attempt==3:
                    raise
    return output_dir / "results.jsonl"
