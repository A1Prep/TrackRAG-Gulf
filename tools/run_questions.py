
"""Run the Issue #3 starter questions through Gemini."""

import json
from pathlib import Path

from trackrag.answering import answer_question
from trackrag.model_client import GeminiAnswerModel
from trackrag.contracts import Evidence
from trackrag.questions import load_questions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "data" / "emt"

QUESTIONS_FILE = PROJECT_ROOT / "annotations" / "questions.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "generated_answers.jsonl"


# Evidence frames selected from the EMT annotations.
# Each question gets only its relevant evidence frames.
EVIDENCE_MAP = {
    "EMT_204647_Q001": [205],
    "EMT_204647_Q002": [411],
    "EMT_204647_Q003": [205],
    "EMT_204647_Q004": [205, 230, 260, 294],
    "EMT_204647_Q005": [411, 435],
    "EMT_204647_Q006": [205, 230, 260, 294],
    "EMT_204647_Q007": [1581, 1600],
    "EMT_204647_Q008": [],
    "EMT_204647_Q009": [],
}

TRACK_MAP = {
    "EMT_204647_Q001": [33, 34],
    "EMT_204647_Q002": [33, 37, 39, 40, 41],
    "EMT_204647_Q003": [35],
    "EMT_204647_Q004": [33],
    "EMT_204647_Q005": [37, 39, 40, 41],
    "EMT_204647_Q006": [35],
    "EMT_204647_Q007": [86],
}


def build_evidence(question):
    """Build evidence records from the selected EMT frames."""

    question_id = question["question_id"]
    recording_id = question["recording_id"]

    frames = EVIDENCE_MAP[question_id]
    tracks = TRACK_MAP.get(question_id, [])

    if not frames:
        return []

    evidence = []

    for index, frame_id in enumerate(frames, start=1):
        image_path = (
            RUN_DIR
            / "selected_frames"
            / recording_id
            / f"{frame_id:05d}.jpg"
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing frame image: {image_path}"
            )

        evidence_id = f"{question_id}_E{index}"

        evidence.append({
            "evidence_id": evidence_id,
            "interval": {
                "recording_id": recording_id,
                "source": "annotated",
                "track_id": tracks[0] if tracks else 0,
                "start_frame": frame_id,
                "end_frame": frame_id,
                "rank": index,
                "score": 1.0,
            },
            "images": [
                {
                    "frame_id": frame_id,
                    "bbox_xyxy": (0, 0, 0, 0),
                    "image_path": str(
                        image_path.relative_to(RUN_DIR)
                    ),
                }
            ],
        })

    return evidence


def main():
    questions = load_questions(QUESTIONS_FILE)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = GeminiAnswerModel()

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        for question in questions:
            question_id = question["question_id"]

            print(f"\nProcessing {question_id}...")

            evidence = build_evidence(question)

            result = answer_question(
                question=question,
                evidence=evidence,
                run_dir=RUN_DIR,
                model=model,
            )

            record = {
                "question_id": question_id,
                **result,
            }

            output.write(
                json.dumps(record) + "\n"
            )
            output.flush()

            print(record)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
