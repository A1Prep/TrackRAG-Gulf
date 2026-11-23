import hashlib

import pytest

from tools import footage_acceptance
from tools.task2_review_package import original_bytes


def test_review_manifest_preserves_signed_csv_and_rejects_other_changes():
    signed = b"case,choice\r\nexample,yes\r\n"
    receipt = {"bytes": len(signed), "sha256": hashlib.sha256(signed).hexdigest()}
    checkout = signed.replace(b"\r\n", b"\n")
    assert original_bytes("reviewer.csv", checkout, receipt) == signed
    assert original_bytes("reviewer.csv", signed, receipt) == signed
    with pytest.raises(ValueError):
        original_bytes("reviewer.csv", checkout.replace(b"yes", b"no"), receipt)
    with pytest.raises(ValueError):
        original_bytes("frame.png", checkout, receipt)


def test_footage_acceptance_opens_citations_and_rejects_future_frames(monkeypatch):
    import json
    q = {"question_id": "q", "question": "What colour is the bus?", "recording_id": "r", "query_frame": 5}
    data = {"question": {**q, "question_id": "server-id"}, "vision": "validated_saved_predictions",
            "hosted_requested": True,
            "answer": {"answer_status": "answered", "answer": "yellow", "cited_evidence_ids": ["E1"]},
            "evidence": [{"id": "E1", "interval": {"end_frame": 5},
                          "images": [{"frame_id": 5, "url": "/image"}]}]}
    image = b"\xff\xd8\xffsample"
    monkeypatch.setattr(footage_acceptance, "fetch",
                        lambda _base, path, body=None: image if path == "/image" else json.dumps(data).encode())
    result = footage_acceptance.review("local", "scene", q, True)
    assert result["evidence"]["E1"][0]["sha256"] == hashlib.sha256(image).hexdigest()
    data["evidence"][0]["images"][0]["frame_id"] = 6
    with pytest.raises(ValueError, match="after the question frame"):
        footage_acceptance.review("local", "scene", q, True)
