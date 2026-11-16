
from trackrag.answering import answer_question


def test_model_timeout_returns_error(monkeypatch, tmp_path):
    """A model timeout must return error, not insufficient_evidence."""

    question = {
        "question_id": "TIMEOUT_TEST",
        "recording_id": "video_204647",
        "query_frame": 205,
        "question": "What is visible?",
    }

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
            "images": [],
        }
    ]

    # Avoid filesystem/image processing in this unit test.
    monkeypatch.setattr(
        "trackrag.answering.prepare_evidence",
        lambda evidence, run_dir: evidence,
    )

    class TimeoutModel:
        def generate(self, question, evidence):
            raise TimeoutError("Simulated network timeout")

    result = answer_question(
        question=question,
        evidence=evidence,
        run_dir=tmp_path,
        model=TimeoutModel(),
    )

    assert result["answer_status"] == "error"
    assert result["answer"] is None
    assert result["cited_evidence_ids"] == []
