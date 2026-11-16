
"""Answer reviewer questions using privacy-processed visual evidence."""

from pathlib import Path
from typing import Protocol
import traceback

from trackrag.contracts import Answer, Evidence, Question
from trackrag.privacy import (
    anonymize_image,
    get_pedestrian_boxes,
)


class AnswerModel(Protocol):
    """Interface for a replaceable answer-generation model."""

    def generate(self, question: str, evidence: list[dict]) -> dict:
        ...


def make_answer(
    status: str,
    answer: str | None = None,
    cited_evidence_ids: list[str] | None = None,
) -> Answer:
    return {
        "answer_status": status,
        "answer": answer,
        "cited_evidence_ids": cited_evidence_ids or [],
    }


def prepare_evidence(
    evidence: list[Evidence],
    run_dir: str | Path,
) -> list[dict]:
    """Create privacy-processed evidence using EMT pedestrian annotations."""

    run_path = Path(run_dir).resolve()
    privacy_dir = run_path / "privacy_processed"
    annotation_dir = run_path / "EMT" / "annotations" / "test"

    prepared = []

    for item in evidence:
        images = []

        recording_id = item["interval"]["recording_id"]

        annotation_path = (
            annotation_dir / f"{recording_id}.txt"
        )

        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"EMT annotation file not found: {annotation_path}"
            )

        for image in item["images"]:
            original_path = (
                run_path / image["image_path"]
            ).resolve()

            if not original_path.is_relative_to(run_path):
                raise ValueError(
                    "Evidence path escapes run directory."
                )

            if not original_path.is_file():
                raise FileNotFoundError(
                    f"Evidence image not found: {original_path}"
                )

            # Find every annotated pedestrian in this frame.
            pedestrian_boxes = get_pedestrian_boxes(
                annotation_path,
                frame_id=image["frame_id"],
            )

            output_path = (
                privacy_dir
                / item["evidence_id"]
                / original_path.name
            )

            privacy_result = anonymize_image(
                original_path,
                output_path,
                pedestrian_boxes=pedestrian_boxes,
            )

            images.append({
                "frame_id": image["frame_id"],
                "bbox_xyxy": image["bbox_xyxy"],
                "image_path": str(
                    privacy_result["output_path"]
                ),
            })

        prepared.append({
            "evidence_id": item["evidence_id"],
            "interval": item["interval"],
            "images": images,
        })

    return prepared


def answer_question(
    question: Question,
    evidence: list[Evidence],
    run_dir: str | Path,
    model: AnswerModel,
) -> Answer:
    """Generate an answer and validate its evidence citations."""

    if not evidence:
        return make_answer("insufficient_evidence")

    try:
        prepared_evidence = prepare_evidence(
            evidence,
            run_dir,
        )

        result = model.generate(
            question=question["question"],
            evidence=prepared_evidence,
        )

        answer_text = result.get("answer")
        citations = result.get("cited_evidence_ids")

        if not isinstance(answer_text, str) or not answer_text.strip():
            return make_answer("error")

        if not isinstance(citations, list):
            return make_answer("error")

        if not all(isinstance(item, str) for item in citations):
            return make_answer("error")

        valid_ids = {
            item["evidence_id"] for item in evidence
        }

        if not citations or not set(citations).issubset(valid_ids):
            return make_answer("error")

        return make_answer(
            "answered",
            answer_text,
            citations,
        )

    except Exception as error:
        print(
            f"\nAnswer generation failed: "
            f"{type(error).__name__}: {error}"
        )
        traceback.print_exc()

        return make_answer("error")
