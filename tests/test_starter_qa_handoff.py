"""Exercise the starter CLI with generated frames, synthetic text and no network."""

from hashlib import sha256
import json
from pathlib import Path

from PIL import Image
import pytest

from tools import evaluate_answers, run_questions
from trackrag.evaluation import evaluate, load_generated_answers


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def scene(tmp_path):
    images = tmp_path / "images" / "train" / "fixture"
    annotations = tmp_path / "annotations" / "train"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    for frame in [1, 2]:
        Image.new("RGB", (80, 60), (80, 120, 160)).save(images / f"{frame:05d}.jpg")
    (annotations / "fixture.txt").write_text(
        "1 7 Bus 0 0 0 6 8 60 50 0 0 0 0 0 0 0\n"
        "2 7 Bus 0 0 0 7 8 61 50 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    records = [{"question_id": identity, "recording_id": "fixture", "query_frame": 2,
                "question": "Which bus is visible?"} for identity in ["first", "second", "third", "fourth"]]
    write_jsonl(questions, records)
    return questions


def test_starter_cli_uses_real_review_and_retains_failed_call(tmp_path, monkeypatch):
    questions = scene(tmp_path)
    seen = []

    class SoftwareModel:
        def generate(self, question, evidence):
            assert set(question) == {"question", "question_id", "query_frame", "recording_id"}
            processed = Path(evidence[0]["images"][0]["image_path"])
            assert "hosted" in processed.parts
            assert processed.is_file()
            seen.append((question["question_id"], sha256(processed.read_bytes()).hexdigest()))
            if question["question_id"] == "second":
                raise RuntimeError("software provider timeout")
            if question["question_id"] == "third":
                return {"answer_status": "insufficient_evidence", "answer": "The views do not establish this.",
                        "cited_evidence_ids": []}
            if question["question_id"] == "fourth":
                return {"answer_status": "answered", "answer": "A bus is visible.", "cited_evidence_ids": ["E999"]}
            return {"answer_status": "answered", "answer": "A bus is visible.",
                    "cited_evidence_ids": [evidence[0]["evidence_id"]]}

    monkeypatch.setattr("trackrag.model_client.GeminiAnswerModel", SoftwareModel)
    directory = tmp_path / "review"
    exported = tmp_path / "answers.jsonl"
    code = run_questions.main(["--data-root", str(tmp_path), "--questions", str(questions),
                               "--run-dir", str(directory), "--output", str(exported)])
    assert code == 1
    assert [identity for identity, _ in seen] == ["first", "second", "third", "fourth"]
    original = load_generated_answers(directory / "results.jsonl")
    assert load_generated_answers(exported) == original
    assert original["first"]["answer_status"] == "answered"
    assert original["second"]["answer_status"] == "error"
    assert "software provider timeout" in original["second"]["answer"]
    assert original["third"]["answer_status"] == "insufficient_evidence"
    assert original["fourth"]["answer_status"] == "error"
    assert "unknown" in original["fourth"]["answer"]
    for identity, digest in seen:
        folder = directory / "hosted" / sha256(identity.encode()).hexdigest()
        receipt = json.loads((folder / "receipt.json").read_bytes())
        assert receipt["sent"][0]["sha256"] == digest
        assert sha256((folder / receipt["sent"][0]["image"]).read_bytes()).hexdigest() == digest
        if identity == "second":
            assert receipt["request_status"] == "error"
        if identity == "fourth":
            assert receipt["request_status"] == "completed"
            assert receipt["response"]["cited_evidence_ids"] == ["E999"]

    reference = tmp_path / "key.jsonl"
    write_jsonl(reference, [{"question_id": q, "expected_answer": "A bus.", "answerable": q != "third",
                             "acceptable_track_ids": [7], "acceptable_intervals": []} for q in original])
    report = tmp_path / "evaluation.jsonl"
    assert evaluate_answers.main(["--generated", str(directory / "results.jsonl"), "--answer-key", str(reference),
                                  "--output", str(report)]) == 0
    outcomes = [json.loads(row)["outcome"] for row in report.read_bytes().splitlines()]
    assert outcomes == ["answered_correct", "error", "answered_correct", "error"]


@pytest.mark.parametrize(
    "answer,expected",
    [("Two pedestrians are visible.", "answered_correct"), ("a pair on foot", "answered_correct"),
     ("Nine pedestrians are visible.", "needs_review"), ("Two parked cars.", "needs_review"),
     ("Two pedestrians are not visible.", "needs_review"), ("Two pedestrians outside the requested frame.", "needs_review")],
)
def test_starter_free_text_requires_positive_reference_support(answer, expected, tmp_path):
    key = tmp_path / "reference.jsonl"
    generated = tmp_path / "answers.jsonl"
    write_jsonl(key, [{"question_id": "q", "answerable": True, "expected_answer": "Two pedestrians.",
                       "acceptable_answers": ["A pair on foot."], "acceptable_intervals": [],
                       "acceptable_track_ids": []}])
    write_jsonl(generated, [{"question_id": "q", "answer_status": "answered", "answer": answer,
                             "cited_evidence_ids": ["E1"]}])
    assert evaluate(generated, key)[0]["outcome"] == expected


def test_report_reader_rejects_duplicate_or_conflicting_identities(tmp_path):
    path = tmp_path / "results.jsonl"
    answer = {"answer_status": "answered", "answer": "One bus.", "cited_evidence_ids": ["E1"]}
    write_jsonl(path, [{"question_id": "other", "question": {"question_id": "q"}, "answer": answer}])
    with pytest.raises(ValueError, match="disagree"):
        load_generated_answers(path)
    write_jsonl(path, [{"question_id": "q", **answer}, {"question": {"question_id": "q"}, "answer": answer}])
    with pytest.raises(ValueError, match="Duplicate"):
        load_generated_answers(path)
