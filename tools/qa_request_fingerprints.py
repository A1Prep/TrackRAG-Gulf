"""Compute the normal QA client's exact model-input hashes offline and compare existing receipts."""
from hashlib import sha256
import json
from pathlib import Path

from evaluation.run_audit import SOURCE_NOT_AVAILABLE, receipt_inventory_version, verify_source_no_call
from trackrag.answering import prepare_evidence, unavailable_observation_source
from trackrag.model_client import GeminiAnswerModel
from trackrag.questions import validate_evidence


class _Capture:
    """Stop at the SDK transport boundary, after the real client builds its normal request."""
    def __init__(self):
        self.interactions = self
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        assert request["store"] is False
        raise _NotSent


class _NotSent(Exception):
    pass


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def processed_images(prepared, question):
    images = []
    for entry in prepared:
        for image in entry["images"]:
            info, path = image["privacy"], Path(image["image_path"])
            if not (info["policy"].startswith("opaque") and image["frame_id"] <= question["query_frame"]
                    and entry["interval"]["source"] == "predicted"):
                raise ValueError("Evidence lacks predicted observations, privacy masking or a valid cutoff")
            images.append({"evidence_id": entry["evidence_id"], "frame_id": image["frame_id"],
                           "sha256": sha256(path.read_bytes()).hexdigest(), "local_preview": str(path),
                           "people_regions_masked": info["people_regions_masked"],
                           "lower_vehicle_regions_masked": info["lower_vehicle_regions_masked"]})
    return images


def preflight(row, run_dir, original_question, model_name):
    """Fingerprint an actual retrieved row, without a provider request or substitute model response."""
    question, evidence = row["question"], row["evidence"]
    if json.dumps(question).encode() != json.dumps(original_question).encode():
        raise ValueError("Public question fields changed from the original bank")
    identity = question["question_id"]
    validate_evidence(question, evidence)
    record = {"question_id": identity, "recording_id": question["recording_id"], "query_frame": question["query_frame"],
              "public_question_submitted_utf8_sha256": sha256(json.dumps(question).encode()).hexdigest(),
              "evidence_intervals_sha256": digest([e["interval"] for e in evidence]), "provider_calls": 0,
              "model": model_name, "request_arguments_sha256": None, "images": []}
    if not evidence:
        return record
    if unavailable_observation_source(question["question"], evidence):
        record["no_provider_reason"] = SOURCE_NOT_AVAILABLE
        return record
    prepared = prepare_evidence(evidence, run_dir, question)
    transport = _Capture()
    client = GeminiAnswerModel(model_name=model_name, client=transport)
    try:
        client.generate(question=question, evidence=prepared)
    except _NotSent:
        pass
    if transport.calls != 1:
        raise ValueError("The real client failed before constructing its request")
    record["request_arguments_sha256"] = client.last_receipt["request_arguments_sha256"]
    record["images"] = processed_images(prepared, question)
    return record


def verify_delivery(frozen, row, directory):
    """Bind a real response (including a failed attempt or empty retrieval) to the offline fingerprint."""
    question, evidence, answer = row["question"], row["evidence"], row["answer"]
    receipt = directory / "hosted" / sha256(question["question_id"].encode()).hexdigest() / "receipt.json"
    if (frozen["public_question_submitted_utf8_sha256"] != sha256(json.dumps(question).encode()).hexdigest() or
            frozen["evidence_intervals_sha256"] != digest([e["interval"] for e in evidence])):
        raise ValueError("Delivered question or retrieved intervals changed")
    if not evidence:
        if (receipt.exists() or frozen["request_arguments_sha256"] is not None or frozen["images"]
                or "no_provider_reason" in frozen
                or answer != {"answer_status": "insufficient_evidence","answer": None,"cited_evidence_ids": []}):
            raise ValueError("A zero-evidence case has unexpected answer or provider receipt")
        return {"provider_called": False, "receipt_sha256": None}
    raw = receipt.read_bytes()
    provider = json.loads(raw)
    receipt_inventory_version(provider)
    if "no_provider_reason" in frozen:
        if (frozen["no_provider_reason"] != SOURCE_NOT_AVAILABLE or frozen["request_arguments_sha256"] is not None
                or frozen["images"] or frozen.get("provider_calls") != 0):
            raise ValueError("Invalid local source-abstention fingerprint")
        verify_source_no_call(question, evidence, answer, provider)
        return {"provider_called": False, "receipt_sha256": sha256(raw).hexdigest(),
                "request_status": "not_requested", "reason": SOURCE_NOT_AVAILABLE}
    seen = [(i["evidence_id"], i["frame_id"], i["sha256"]) for i in provider["sent"]]
    expected = [(i["evidence_id"], i["frame_id"], i["sha256"]) for i in frozen["images"]]
    if (seen != expected or provider.get("model") != frozen["model"] or provider.get("cached") is not False
            or provider.get("store") is not False or provider["request_arguments_sha256"] != frozen["request_arguments_sha256"]
            or question != provider["public_question"]):
        raise ValueError("Provider receipt does not match the frozen model, full question and processed evidence")
    for item in provider["sent"]:
        if sha256((receipt.parent / item["image"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Archived submitted image differs from the provider receipt")
    return {"provider_called": True, "receipt_sha256": sha256(raw).hexdigest(),
            "request_status": provider["request_status"], "http_status": provider.get("http_status"),
            "reported_model": provider.get("reported_model"), "error_type": provider.get("error_type")}
