
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

import pytest


from trackrag.answering import answer_question

def create_test_annotation(run_dir):
    """Create a minimal EMT annotation file for unit tests."""

    annotation_dir = (
        run_dir / "EMT" / "annotations" / "test"
    )
    annotation_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = annotation_dir / "video_204647.txt"

    # 17 columns:
    # frame, track, class, ..., x1, y1, x2, y2, ...
    annotation_file.write_text(
        "205 33 pedestrian 0 0 0 100 100 200 300 "
        "0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )

    return annotation_file


@pytest.fixture
def sample_question():
    return {
        "question_id": "Q001",
        "recording_id": "video_204647",
        "query_frame": 205,
        "question": "How many pedestrians are visible?",
    }


@pytest.fixture
def sample_evidence(tmp_path):
    # Create a dummy image file for testing.
    image_path = tmp_path / "frame_00205.jpg"
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    return [
        {
            "evidence_id": "E1",
            "interval": {
                "recording_id": "video_204647",
                "source": "annotated",
                "track_id": 33,
                "start_frame": 205,
                "end_frame": 205,
                "rank": 1,
                "score": 0.95,
            },
            "images": [
                {
                    "frame_id": 205,
                    "bbox_xyxy": (100, 100, 200, 300),
                    "image_path": "frame_00205.jpg",
                }
            ],
        }
    ]


def test_no_evidence_returns_insufficient_evidence(
    sample_question, tmp_path
):
    model = Mock()

    result = answer_question(
        sample_question,
        [],
        tmp_path,
        model,
    )

    assert result == {
        "answer_status": "insufficient_evidence",
        "answer": None,
        "cited_evidence_ids": [],
    }

    # The model must not be called.
    model.generate.assert_not_called()


def test_valid_evidence_and_citation_are_accepted(
    sample_question, sample_evidence, tmp_path
):
    model = Mock()
    model.generate.return_value = {
        "answer": "One pedestrian is visible.",
        "cited_evidence_ids": ["E1"],
    }

    create_test_annotation(tmp_path)

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result == {
        "answer_status": "answered",
        "answer": "One pedestrian is visible.",
        "cited_evidence_ids": ["E1"],
    }

    model.generate.assert_called_once()


def test_invented_evidence_id_is_rejected(
    sample_question, sample_evidence, tmp_path
):
    model = Mock()
    model.generate.return_value = {
        "answer": "Two pedestrians are visible.",
        "cited_evidence_ids": ["E9"],
    }

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result == {
        "answer_status": "error",
        "answer": None,
        "cited_evidence_ids": [],
    }


def test_model_timeout_returns_error(
    sample_question, sample_evidence, tmp_path
):
    model = Mock()
    model.generate.side_effect = TimeoutError("Model request timed out")

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result == {
        "answer_status": "error",
        "answer": None,
        "cited_evidence_ids": [],
    }


def test_missing_image_returns_error(
    sample_question, tmp_path
):
    model = Mock()

    evidence = [
        {
            "evidence_id": "E1",
            "interval": {
                "recording_id": "video_204647",
                "source": "annotated",
                "track_id": 33,
                "start_frame": 205,
                "end_frame": 205,
                "rank": 1,
                "score": 0.95,
            },
            "images": [
                {
                    "frame_id": 205,
                    "bbox_xyxy": (100, 100, 200, 300),
                    "image_path": "missing.jpg",
                }
            ],
        }
    ]

    result = answer_question(
        sample_question,
        evidence,
        tmp_path,
        model,
    )

    assert result["answer_status"] == "error"
    model.generate.assert_not_called()


def test_model_response_without_citations_is_rejected(
    sample_question, sample_evidence, tmp_path
):
    model = Mock()
    model.generate.return_value = {
        "answer": "A pedestrian is visible.",
        "cited_evidence_ids": [],
    }

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result["answer_status"] == "error"
    assert result["cited_evidence_ids"] == []


def test_evidence_path_cannot_escape_run_directory(
    sample_question, tmp_path
):
    model = Mock()

    evidence = [
        {
            "evidence_id": "E1",
            "interval": {
                "recording_id": "video_204647",
                "source": "annotated",
                "track_id": 33,
                "start_frame": 205,
                "end_frame": 205,
                "rank": 1,
                "score": 0.95,
            },
            "images": [
                {
                    "frame_id": 205,
                    "bbox_xyxy": (100, 100, 200, 300),
                    "image_path": "../outside.jpg",
                }
            ],
        }
    ]

    result = answer_question(
        sample_question,
        evidence,
        tmp_path,
        model,
    )

    assert result["answer_status"] == "error"
    model.generate.assert_not_called()

def test_model_insufficient_evidence_reply_is_distinct(
    sample_question, sample_evidence, tmp_path
):
    """Model explicitly says evidence doesn't establish the answer."""
    model = Mock()
    model.generate.return_value = {
        "answer": "The footage does not establish the pedestrians' destination.",
        "cited_evidence_ids": [],
    }

    create_test_annotation(tmp_path)

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    # Must be insufficient_evidence, not error.
    assert result["answer_status"] == "insufficient_evidence"
    assert result["answer"] is not None


def test_citation_mismatch_is_distinct_from_provider_error(
    sample_question, sample_evidence, tmp_path
):
    """Hallucinated evidence ID must yield error, not answered."""
    model = Mock()
    model.generate.return_value = {
        "answer": "Two pedestrians.",
        "cited_evidence_ids": ["E_NONEXISTENT"],
    }

    create_test_annotation(tmp_path)

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result["answer_status"] == "error"


def test_model_receives_processed_image_not_original(
    sample_question, sample_evidence, tmp_path
):
    model = Mock()
    model.generate.return_value = {
        "answer": "One pedestrian is visible.",
        "cited_evidence_ids": ["E1"],
    }

    original_path = (
        tmp_path / sample_evidence[0]["images"][0]["image_path"]
    ).resolve()

    create_test_annotation(tmp_path)

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result["answer_status"] == "answered"

    model_evidence = model.generate.call_args.kwargs["evidence"]
    processed_path = model_evidence[0]["images"][0]["image_path"]

    assert Path(processed_path).is_file()
    assert Path(processed_path) != original_path
    assert "privacy_processed" in str(processed_path)