
"""Answer reviewer questions using privacy-processed visual evidence."""

from pathlib import Path
from typing import Protocol
import traceback

from trackrag.contracts import Answer, Evidence, Question


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
    """Validate evidence paths and return them unchanged for the model."""

    run_path = Path(run_dir).resolve()
    prepared = []

    for item in evidence:
        images = []

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

            images.append({
                "frame_id": image["frame_id"],
                "bbox_xyxy": image["bbox_xyxy"],
                "image_path": str(original_path),
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
    model: AnswerModel | None = None,
) -> Answer:
    """Generate an answer and validate its evidence citations."""

    if not evidence:
        return make_answer("insufficient_evidence")

    if model is None:
        from trackrag.model_client import GeminiAnswerModel
        model = GeminiAnswerModel()

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

        # Model explicitly declared it cannot answer from the evidence.
        UNSUPPORTED_PHRASES = (
            "does not establish",
            "cannot determine",
            "not visible",
            "insufficient",
            "unable to determine",
            "not establish",
            "cannot be determined",
        )
        lower_answer = answer_text.lower()
        if not citations and any(p in lower_answer for p in UNSUPPORTED_PHRASES):
            return make_answer("insufficient_evidence", answer_text, [])

        valid_ids = {
            item["evidence_id"] for item in evidence
        }

        # Citations are present but reference unknown evidence IDs.
        if citations and not set(citations).issubset(valid_ids):
            print(
                f"\nCitation mismatch: model cited {set(citations) - valid_ids} "
                "which are not in the supplied evidence."
            )
            return make_answer("error")

        # No citations at all and answer is not an explicit insufficient-evidence reply.
        if not citations:
            print("\nUnsupported answer: model answered without citing any evidence.")
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
