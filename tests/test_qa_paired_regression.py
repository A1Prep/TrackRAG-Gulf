"""Transport-free checks for exact QA reuse and post-response recovery."""
import json
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest

from evaluation.run_audit import SOURCE_NOT_AVAILABLE, _verified_grounding_abstention, audit_run
from tools.qa_request_fingerprints import preflight, verify_delivery
from tools import qa_temporal_regression as paired
from tools.qa_temporal_regression import dispatch_mode, resume_attempt
from trackrag.answering import answer_question, unavailable_observation_source
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
    # Historical source-mismatch requests reached the transport, then received
    # a local veto. Reconstruct that old dispatch order to exercise the archive reader.
    source_reason = unavailable_observation_source(q["question"], row["evidence"])
    with ExitStack() as historical:
        if source_reason:
            historical.enter_context(patch("trackrag.answering.unavailable_observation_source", side_effect=[None, source_reason]))
            historical.enter_context(patch("tools.qa_request_fingerprints.unavailable_observation_source", return_value=None))
            def original_inventory(_question, evidence):
                return [{"evidence_id": e["evidence_id"], "interval": e["interval"],
                         "images": [{k: im[k] for k in ("frame_id", "bbox_xyxy", "observations", "privacy")}
                                    for im in e["images"]]} for e in evidence]
            historical.enter_context(patch("trackrag.model_client.evidence_inventory", original_inventory))
            frozen = preflight({**row, "question": q}, tmp_path, q, "synthetic-model")
        delivered = answer_question(q, row["evidence"], tmp_path, GeminiAnswerModel("synthetic-model",client=transport))
    directory = tmp_path / "hosted" / sha256(q["question_id"].encode()).hexdigest()
    receipt = json.loads((directory / "receipt.json").read_bytes())
    if source_reason:
        receipt.pop("evidence_inventory_version")
        (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf8")
        saved = {**row, "question":q, "answer":delivered}
        assert verify_delivery(frozen, saved, tmp_path)["provider_called"] is True
    assert receipt["response"] == raw and delivered["answer_status"] == "insufficient_evidence"
    assert _verified_grounding_abstention(tmp_path,q,row["evidence"],delivered,receipt,raw) == policy
    assert not _verified_grounding_abstention(tmp_path,q,row["evidence"],{**delivered,"answer":"Invented."},receipt,raw)
    receipt["sent"][0]["sha256"] = "0" * 64
    assert not _verified_grounding_abstention(tmp_path,q,row["evidence"],delivered,receipt,raw)


def test_local_source_preflight_dispatch_verification_and_forged_receipts(tmp_path):
    row = inputs(tmp_path)
    q = row["question"]
    q["question"] = "Which class is annotated as track 7?"
    frozen = preflight(row, tmp_path, q, "synthetic-model")
    assert frozen["no_provider_reason"] == SOURCE_NOT_AVAILABLE
    assert frozen["request_arguments_sha256"] is None and frozen["images"] == []
    hosted = {"request_arguments_sha256": "a" * 64}
    empty = {"request_arguments_sha256": None}
    assert dispatch_mode(hosted, frozen) == dispatch_mode(empty, frozen) == "local_source_abstention"
    assert dispatch_mode(frozen, empty) == "changed_empty"
    assert dispatch_mode(frozen, hosted) == "changed_live"
    assert dispatch_mode(hosted, hosted) == dispatch_mode(empty, empty) == "unchanged_prior"
    row["answer"] = answer_question(q, row["evidence"], tmp_path)
    path = tmp_path / "hosted" / sha256(q["question_id"].encode()).hexdigest() / "receipt.json"
    receipt = json.loads(path.read_bytes())
    delivery = verify_delivery(frozen, row, tmp_path)
    assert delivery["provider_called"] is False and delivery["receipt_sha256"] == sha256(path.read_bytes()).hexdigest()
    for edits in ({"public_question": {**q, "question": "Other wording"}}, {"query_frame": True},
                  {"sent": [{"frame_id": 1}]}, {"response": {"answer_status": "answered"}},
                  {"reason": "provider_error"}, {"request_status": "completed"}):
        path.write_text(json.dumps({**receipt, **edits}), encoding="utf8")
        with pytest.raises(ValueError, match="source-abstention"):
            verify_delivery(frozen, row, tmp_path)
    path.write_text(json.dumps(receipt), encoding="utf8")
    for edits in ({"no_provider_reason": "error"}, {"images": [{"frame_id": 1}]}, {"request_arguments_sha256": "x"}):
        with pytest.raises(ValueError, match="fingerprint"):
            verify_delivery({**frozen, **edits}, row, tmp_path)
    row["answer"]["answer"] = "Invented explanation."
    with pytest.raises(ValueError, match="source-abstention"):
        verify_delivery(frozen, row, tmp_path)


def test_versioned_geometry_veto_requires_the_complete_derived_inventory(tmp_path):
    row = inputs(tmp_path)
    q = {**row["question"], "query_frame": 3, "question": "Does the car move left or right near the end?"}
    first = json.loads((tmp_path / "tracks.jsonl").read_bytes())
    second = {**first, "frame_id": 2, "bbox_xyxy": [10, 3, 29, 17]}
    with (tmp_path / "tracks.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(second) + "\n")
    item = row["evidence"][0]
    item["interval"]["end_frame"] = 2
    item["images"].append({**item["images"][0], "frame_id": 2, "bbox_xyxy": second["bbox_xyxy"]})
    (tmp_path / "run.json").write_text(json.dumps({"first_frame_id": 1,
        "configuration": {"retrieval": "trackrag.retrieval_temporal"}}), encoding="utf8")
    raw = {"answer_status": "answered", "answer": "The car moves right.", "cited_evidence_ids": ["E1"]}
    transport = SimpleNamespace(interactions=SimpleNamespace(create=lambda **_: SimpleNamespace(
        output_text=json.dumps(raw), status="completed", model="synthetic-model")))
    delivered = answer_question(q, row["evidence"], tmp_path, GeminiAnswerModel("synthetic-model", client=transport))
    receipt = json.loads((tmp_path / "hosted" / sha256(q["question_id"].encode()).hexdigest() / "receipt.json").read_bytes())
    assert receipt["evidence_inventory_version"] == 2
    assert receipt["evidence_inventory"][0]["observed_horizontal_motion"]["image_direction"] == "right"
    assert _verified_grounding_abstention(tmp_path,q,row["evidence"],delivered,receipt,raw) == "recomputed_temporal_abstention"
    altered = deepcopy(receipt)
    altered["evidence_inventory"][0]["observed_horizontal_motion"]["image_direction"] = "left"
    assert not _verified_grounding_abstention(tmp_path,q,row["evidence"],delivered,altered,raw)
    altered.pop("evidence_inventory_version")
    assert not _verified_grounding_abstention(tmp_path,q,row["evidence"],delivered,altered,raw)
    with pytest.raises(ValueError, match="inventory version"):
        _verified_grounding_abstention(tmp_path,q,row["evidence"],delivered,{**receipt, "evidence_inventory_version": 3},raw)


def test_paired_collect_resumes_a_local_source_receipt_without_a_provider_attempt(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    src = prepared / "synthetic"
    src.mkdir(parents=True)
    row = inputs(src)
    q = row["question"]
    q.update(question_id="temporal-12", question="Which class is annotated as track 7?")
    (src / "evidence").mkdir()
    (src / "source.png").rename(src / "evidence/source.png")
    row["evidence"][0]["images"][0]["image_path"] = "evidence/source.png"
    frozen = preflight(row, src, q, paired.MODEL)
    row.update(source="predicted", answer={"answer_status":"not_requested", "answer":None, "cited_evidence_ids":[]})
    (src / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf8")
    paired.write(src / "run.json", {"recording_id":"synthetic", "source":"predicted", "configuration":{},
                                    "first_frame_id":1, "question_count":1, "observation_count":1})
    group = {f"{name}_sha256": paired.file_sha(src / file) for name, file in (
        ("run","run.json"), ("tracks","tracks.jsonl"), ("results","results.jsonl"))}
    baseline = tmp_path / "previous.json"
    paired.write(baseline, {})
    paired.write(prepared / "new-images.json", [])
    paired.write(prepared / "selection.json", {"source_sha256":paired.model_source(), "model":paired.MODEL,
        "fingerprints_source_sha256":paired.file_sha(baseline), "new_images_sha256":paired.file_sha(prepared/"new-images.json"),
        "groups":{"synthetic":group}, "rows":{q["question_id"]:{"input":frozen, "dispatch":"local_source_abstention"}}})
    args = SimpleNamespace(prepared=prepared, out=tmp_path/"collected", selection_sha256=paired.file_sha(prepared/"selection.json"),
                           fingerprints=baseline, previous_run=[], spacing=0)
    monkeypatch.setattr(paired, "RECORDINGS", {"synthetic":"unused"})
    monkeypatch.setattr(paired, "previous", lambda *_: ({}, {q["question_id"]:(row, src)}))
    paired.collect(args)
    paired.collect(args)
    complete = json.loads((args.out/"complete.json").read_bytes())
    assert complete["rows"] == 1 and complete["provider_attempts"] == 0
    result = json.loads((args.out/"rows/temporal-12.json").read_bytes())
    assert result["answer"]["answer_status"] == "insufficient_evidence"
    assert not result["delivery"]["provider_called"] and not result["attempts"]
    output = args.out / "synthetic"
    assert audit_run(output, {q["question_id"]:q})["provenance"]["provider_receipts"][0]["delivered_via"] == "local_source_abstention"
