"""Transport-free checks for exact QA reuse and post-response recovery."""
import json
from hashlib import sha256
from types import SimpleNamespace

from PIL import Image
import pytest

from evaluation.run_audit import _verified_grounding_abstention
from tools.qa_request_fingerprints import preflight, verify_delivery
from tools.qa_temporal_regression import resume_attempt
from trackrag.answering import answer_question
from trackrag.model_client import GeminiAnswerModel


def inputs(root):
    (root / "tracks.jsonl").write_text(json.dumps({
        "recording_id": "synthetic", "source": "predicted", "frame_id": 1,
        "track_id": 7, "class_name": "Car", "bbox_xyxy": [2, 3, 21, 17],
    }) + "\n", encoding="utf8")
    Image.new("RGB", (40, 30), "grey").save(root / "source.png")
    question = {"question_id": "test-1", "recording_id": "synthetic", "query_frame": 1,
                "question": "What colour is the visible car?"}
    evidence = [{"evidence_id": "E1", "interval": {
        "recording_id": "synthetic", "source": "predicted", "track_id": 7,
        "rank": 1, "score": 0.5, "start_frame": 1, "end_frame": 1,
    }, "images": [{"frame_id": 1, "bbox_xyxy": [2, 3, 21, 17], "image_path": "source.png"}]}]
    return {"question": question, "evidence": evidence}


def test_offline_digest_covers_public_question_and_processed_pixels(tmp_path):
    row = inputs(tmp_path)
    first = preflight(row, tmp_path, row["question"], "synthetic-model")
    assert first["provider_calls"] == 0
    rewritten = {**row, "question": {**row["question"], "question": "Which colour is the car?"}}
    changed = preflight(rewritten, tmp_path, rewritten["question"], "synthetic-model")
    assert changed["images"][0]["sha256"] == first["images"][0]["sha256"]
    assert changed["request_arguments_sha256"] != first["request_arguments_sha256"]
    Image.new("RGB", (40, 30), "navy").save(tmp_path / "source.png")
    new_pixels = preflight(row, tmp_path, row["question"], "synthetic-model")
    assert new_pixels["images"][0]["sha256"] != first["images"][0]["sha256"]
    assert new_pixels["request_arguments_sha256"] != first["request_arguments_sha256"]


def test_completed_response_can_be_recovered_without_a_new_transport(tmp_path):
    row = inputs(tmp_path)
    q = row["question"]
    frozen = preflight(row, tmp_path, q, "synthetic-model")
    answer = {"answer_status": "answered", "answer": "Grey.", "cited_evidence_ids": ["E1"]}
    lower_transport = SimpleNamespace(interactions=SimpleNamespace(
        create=lambda **fields: SimpleNamespace(output_text=json.dumps(answer), status="completed", model="synthetic-model")))
    row["answer"] = answer_question(q, row["evidence"], tmp_path, GeminiAnswerModel("synthetic-model",client=lower_transport))
    host = tmp_path / "hosted" / sha256(q["question_id"].encode()).hexdigest()
    original = (host / "receipt.json").read_bytes()
    archive = tmp_path / "archived.json"
    recovered = resume_attempt({"state":"started","input_sha256":frozen["request_arguments_sha256"]},
                               archive, host, q, row, frozen, tmp_path)
    assert recovered["answer"] == answer
    assert archive.read_bytes() == original
    assert recovered["receipt_sha256"] == sha256(original).hexdigest()
    (host / "evidence-1-frame-1.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="submitted image differs"):
        verify_delivery(frozen,row,tmp_path)


def test_unknown_transmission_is_never_repeated_by_recovery(tmp_path):
    row = inputs(tmp_path)
    frozen = preflight(row, tmp_path, row["question"], "synthetic-model")
    with pytest.raises(ValueError,match="unknown service status"):
        resume_attempt({"state":"started","input_sha256":frozen["request_arguments_sha256"]},
                       tmp_path/"missing.json",tmp_path,row["question"],row,frozen,tmp_path)


@pytest.mark.parametrize(("question_text", "configuration", "policy"), [
    ("What colour is the car near the end?", {"retrieval": "trackrag.retrieval_temporal"}, "recomputed_temporal_abstention"),
    ("Which class has annotated track 7?", {"retrieval": "trackrag.retrieval"}, "recomputed_source_abstention"),
])
def test_audit_reconstructs_a_genuine_veto_and_rejects_a_forged_one(tmp_path,question_text,configuration,policy):
    row = inputs(tmp_path)
    q = {**row["question"], "query_frame": 2, "question": question_text}
    (tmp_path / "run.json").write_text(json.dumps({"first_frame_id": 1, "configuration": configuration}), encoding="utf8")
    raw = {"answer_status": "answered", "answer": "The car is black.", "cited_evidence_ids": ["E1"]}
    transport = SimpleNamespace(interactions=SimpleNamespace(create=lambda **_: SimpleNamespace(
        output_text=json.dumps(raw), status="completed", model="synthetic-model")))
    delivered = answer_question(q, row["evidence"], tmp_path, GeminiAnswerModel("synthetic-model",client=transport))
    directory = tmp_path / "hosted" / sha256(q["question_id"].encode()).hexdigest()
    receipt = json.loads((directory / "receipt.json").read_bytes())
    assert receipt["response"] == raw and delivered["answer_status"] == "insufficient_evidence"
    assert _verified_grounding_abstention(tmp_path,q,row["evidence"],delivered,receipt,raw) == policy
    assert not _verified_grounding_abstention(tmp_path,q,row["evidence"],{**delivered,"answer":"Invented."},receipt,raw)
    receipt["sent"][0]["sha256"] = "0" * 64
    assert not _verified_grounding_abstention(tmp_path,q,row["evidence"],delivered,receipt,raw)
