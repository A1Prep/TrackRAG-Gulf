"""Inference CLI wiring checks with an explicitly substituted model."""

import hashlib
import json
from pathlib import Path

import pytest

from tools import dropoff_predict
from trackrag.dropoff.contracts import error_outcome
from trackrag.dropoff.model import GroundingConfig


def request(identity="request-1"):
    return {
        "request_id": identity, "text": "Stop next to the car", "input_mode": "typed", "dialogue": [],
        "scene": {"frame": {"recording_id": "recording", "frame_id": 10, "timestamp_s": 0.9,
                            "width": 160, "height": 90, "image_path": "image.png"},
                  "source": "predicted", "tracks": []},
    }


def test_export_does_not_construct_model_or_include_gold(tmp_path, monkeypatch):
    row = {"case_id": "request-1", "request": request(),
           "target": {"destination_point_xy": [30, 50]}, "source": {"image_sha256": "a" * 64}}
    monkeypatch.setattr(dropoff_predict, "load_cases", lambda *args, **kwargs: [row])
    def never_model(*args, **kwargs):
        raise AssertionError("export must not construct a model")
    monkeypatch.setattr(dropoff_predict, "GroundingModel", never_model)
    source = tmp_path / "source.jsonl"
    source.write_text('{"reference":"PRIVATE_REFERENCE"}\n', encoding="utf-8")
    output = tmp_path / "requests.jsonl"
    manifest = dropoff_predict.export_requests(source, "train", output)
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert set(exported) == set(row["request"])
    assert "PRIVATE_REFERENCE" not in output.read_text(encoding="utf-8")
    assert "target" not in exported and manifest["gold_fields_exported"] is False
    assert manifest["image_sha256"] == {"request-1": "a" * 64}
    with pytest.raises(FileExistsError):
        dropoff_predict.export_requests(source, "train", output)


def test_prediction_preserves_errors_and_semantic_abstentions_separately(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = [request("supported-abstention"), request("missing-checkpoint")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    class Substitute:
        def __init__(self, config):
            self.last_receipt = {}
        def interpret(self, row):
            self.last_receipt = {"request_id": row["request_id"], "backend_kind": "injected_test_backend"}
            if row["request_id"] == "missing-checkpoint":
                return error_outcome("checkpoint_unavailable", "Fixture checkpoint absent", row["request_id"])
            return {
                "ok": True, "request_id": row["request_id"],
                "result": {"status": "insufficient_visual_evidence", "destination_point_xy": None,
                           "destination_box_xyxy": None, "clarification_question": None},
                "provenance": {"backend_kind": "injected_test_backend"},
            }
    out = tmp_path / "run"
    summary = dropoff_predict.predict(path, GroundingConfig(), out, adapter_factory=Substitute)
    outcomes = [json.loads(line) for line in (out / "model-outcomes.jsonl").read_text().splitlines()]
    assert summary["requested_rows"] == 2
    assert summary["semantic_results"] == 1 and summary["operational_errors"] == 1
    assert summary["model_quality_measured"] is False
    assert outcomes[0]["result"]["status"] == "insufficient_visual_evidence"
    assert outcomes[1]["error"]["code"] == "checkpoint_unavailable"
    assert "result" not in outcomes[1]


@pytest.mark.parametrize("mutation", ("gold", "duplicate_id", "duplicate_json"))
def test_invalid_request_file_rejected_before_model_construction(tmp_path, mutation):
    row = request()
    if mutation == "gold":
        row["target"] = {"destination_point_xy": [30, 50]}
    payload = json.dumps(row) + "\n"
    if mutation == "duplicate_id":
        payload += json.dumps(row) + "\n"
    elif mutation == "duplicate_json":
        payload = payload.replace('"text": "Stop next to the car"', '"text": "one", "text": "two"')
    path = tmp_path / "requests.jsonl"
    path.write_text(payload, encoding="utf-8")
    out = tmp_path / "run"
    def never_model(config):
        raise AssertionError("invalid input reached model construction")
    with pytest.raises(ValueError):
        dropoff_predict.predict(path, GroundingConfig(), out, adapter_factory=never_model)
    assert not out.exists()


def test_request_image_paths_resolve_relative_to_input(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(request()) + "\n", encoding="utf-8")
    loaded = dropoff_predict.read_requests(path)
    assert Path(loaded[0]["scene"]["frame"]["image_path"]) == (tmp_path / "image.png").resolve()


@pytest.mark.parametrize("number", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_provenance_rejected_before_output_or_model(tmp_path, number):
    row = request()
    row["scene"]["provenance"] = {"invalid": number}
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "run"
    def never_model(config):
        raise AssertionError("nonfinite request reached model")
    with pytest.raises(ValueError):
        dropoff_predict.predict(path, GroundingConfig(), out, adapter_factory=never_model)
    assert not out.exists()


def test_request_digest_binds_consumed_snapshot_not_later_file(tmp_path):
    path = tmp_path / "requests.jsonl"
    original = (json.dumps(request()) + "\n").encode()
    path.write_bytes(original)
    class MutatingSubstitute:
        def __init__(self, config):
            self.last_receipt = {"kind": "injected_test_backend"}
        def interpret(self, row):
            path.write_text(json.dumps(request("different-request")) + "\n", encoding="utf-8")
            return error_outcome("checkpoint_unavailable", "Fixture only", row["request_id"])
    out = tmp_path / "run"
    summary = dropoff_predict.predict(path, GroundingConfig(), out, adapter_factory=MutatingSubstitute)
    assert summary["requests_sha256"] == hashlib.sha256(original).hexdigest()
    assert (out / "input-snapshot.jsonl").read_bytes() == original
    assert hashlib.sha256(path.read_bytes()).hexdigest() != summary["requests_sha256"]


def test_export_digest_binds_case_snapshot_not_later_file(tmp_path, monkeypatch):
    path = tmp_path / "cases.jsonl"
    original = b'{"fixture":"original"}\n'
    path.write_bytes(original)
    def loader(source, *, allowed_partitions, payload):
        assert payload == original
        path.write_text('{"fixture":"changed"}\n', encoding="utf-8")
        return [{"case_id": "request-1", "request": request(), "source": {"image_sha256": "a" * 64}}]
    monkeypatch.setattr(dropoff_predict, "load_cases", loader)
    output = tmp_path / "requests.jsonl"
    summary = dropoff_predict.export_requests(path, "train", output)
    assert summary["source_cases_sha256"] == hashlib.sha256(original).hexdigest()
    assert json.loads(output.read_text())["scene"]["provenance"]["image_sha256"] == "a" * 64


def test_prediction_closes_owned_adapter_once_after_success_and_uses_atomic_receipts(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = [request("first"), request("second")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    events = []
    class Substitute:
        def __init__(self, config): self.last_receipt = {"wrong": "legacy shared receipt"}
        def interpret(self, row): raise AssertionError("Atomic interpretation must be selected")
        def interpret_with_receipt(self, row):
            events.append(row["request_id"])
            return error_outcome("checkpoint_unavailable", "Fixture only", row["request_id"]), {
                "request_id": row["request_id"], "atomic": True}
        def close(self): events.append("closed")
    output = tmp_path / "run"
    summary = dropoff_predict.predict(path, GroundingConfig(), output, adapter_factory=Substitute)
    receipts = [json.loads(line) for line in (output / "receipts.jsonl").read_text().splitlines()]
    assert events == ["first", "second", "closed"]
    assert summary["requested_rows"] == 2
    assert receipts == [{"request_id": "first", "atomic": True}, {"request_id": "second", "atomic": True}]


@pytest.mark.parametrize("atomic", [False, True])
def test_prediction_closes_once_on_unexpected_error_without_masking_original(tmp_path, atomic):
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(request()) + "\n", encoding="utf-8")
    original = RuntimeError("Unexpected interpretation failure")
    events = []
    class Substitute:
        def __init__(self, config): self.last_receipt = {}
        def interpret(self, row):
            events.append("interpret")
            raise original
        if atomic:
            def interpret_with_receipt(self, row):
                events.append("atomic")
                raise original
        def close(self):
            events.append("closed")
            raise OSError("Secondary cleanup failure")
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Unexpected interpretation failure") as caught:
        dropoff_predict.predict(path, GroundingConfig(), output, adapter_factory=Substitute)
    assert caught.value is original
    assert events == ["atomic" if atomic else "interpret", "closed"]
    assert not (output / "run.json").exists()
