from copy import deepcopy
from hashlib import sha256
import json

from PIL import Image
import pytest

from tools.dropoff_passenger_handoff import batch, bound_outcome, session
from trackrag.dropoff.data import file_sha256, inference_input_sha256
from trackrag.dropoff.passenger import digest, unresolved


def jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf8")


@pytest.fixture
def input_output(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (40, 30), "grey").save(path)
    image_sha = file_sha256(path)
    frame = {"recording_id": "nonfinal-test", "frame_id": 2, "timestamp_s": 0.1, "width": 40,
             "height": 30, "image_path": str(path)}
    scene = {"source": "predicted", "frame": frame, "tracks": [], "provenance": {"image_sha256": image_sha}}
    request = {"request_id": "software-fixture", "text": "Let me out at the green bus", "input_mode": "typed",
               "dialogue": [], "scene": scene}
    messages = [{"role": "user", "content": request["text"]}]
    resolver = {"model_id": "fixture", "revision": "fixture", "adapter": None, "request_id": request["request_id"],
                "image_sha256": image_sha, "request_sha256": digest(request), "raw_text": "MISSING",
                "messages": messages, "messages_sha256": sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                "backend_details": {"model_id": "fixture", "revision": "fixture", "runtime": "streamed"}}
    provenance = {"method": "qwen_landmark_region", "geometry_sha256": "a"*64, "request_sha256": digest(request),
                  "resolver": resolver, "image": {"sha256": image_sha},
                  "inference_input_sha256": inference_input_sha256(request, image_sha),
                  "resolver_execution": "saved_decisions"}
    outcome = {"request_id": request["request_id"], "ok": True, "result": unresolved(), "provenance": provenance}
    return request, outcome


def test_raw_replay_and_model_stage_are_bound(tmp_path, input_output):
    request, outcome = input_output
    req, raw, preds = (tmp_path / name for name in ("requests.jsonl", "raw.jsonl", "outcomes.jsonl"))
    jsonl(req, [request])
    jsonl(raw, [outcome["provenance"]["resolver"]])
    jsonl(preds, [outcome])
    settings = {"requested": 1, "requests_sha256": file_sha256(req), "resolver_execution": "saved_decisions",
                "reference_access": False, "geometry_sha256": "a"*64, "saved_decisions_sha256": file_sha256(raw),
                "config": {"model_id": "fixture", "revision": "fixture", "adapter_dir": None}}
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    (tmp_path / "run.json").write_text(json.dumps(dict(settings, errors=0, outcomes_sha256=file_sha256(preds))))
    receipt = batch(tmp_path, req, raw)
    assert receipt["native_task1_condition"] is False
    assert receipt["events"][0]["phase"] == "initial_model"
    assert receipt["events"][0]["raw_saved_decision_line_sha256"] == sha256(raw.read_bytes().splitlines()[0]).hexdigest()
    raw.write_bytes(raw.read_bytes().replace(b"MISSING", b"MATCH"))
    with pytest.raises(ValueError, match="exact saved raw"):
        batch(tmp_path, req, raw)


def test_manual_or_changed_source_cannot_be_a_model_attempt(input_output):
    request, outcome = input_output
    altered = deepcopy(outcome)
    altered["provenance"]["method"] = "explicit_passenger_region"
    altered["provenance"]["independent_model_proposal"] = False
    with pytest.raises(ValueError, match="assisted"):
        bound_outcome(altered, request)
    with pytest.raises(ValueError,match="Passenger mark"):
        bound_outcome(altered, request, passenger_selected=True)
    altered = deepcopy(outcome)
    altered["provenance"]["resolver"]["messages"][0]["content"] = "Changed"
    with pytest.raises(ValueError, match="prompt"):
        bound_outcome(altered, request)
    diagnostic=deepcopy(outcome)
    diagnostic["provenance"]["injected"]=True
    with pytest.raises(ValueError,match="injected"):
        bound_outcome(diagnostic,request)


def test_failed_submission_survives_even_when_the_model_never_received_text(tmp_path, input_output):
    req, _ = input_output
    scene, request_id = req["scene"], req["request_id"]
    view = {"recording_id": "nonfinal-test", "frame_id": 2, "image_sha256": scene["provenance"]["image_sha256"],
            "request_id": None, "proposal_id": None, "result": None, "selected_by": None, "state": "idle"}
    opened = {"action": "opened", "detail": {"scene": scene}, "view": view, "timestamp": "software-test"}
    invalid = dict(req, text="")
    failed = {"action": "submitted", "detail": {"request": invalid, "outcome": None, "speech": None},
              "view": dict(view, state="error", request_id=request_id, error={"code": "invalid_request"}),
              "timestamp": "software-test"}
    file = tmp_path / "session.jsonl"
    jsonl(file, [opened, failed])
    exported = session(file)
    assert exported["events"][1]["outcome"]["adapter_outcome_missing"]
    assert exported["events"][1]["outcome"]["error"]["code"] == "invalid_request"
    confirmed = {"action": "confirmed", "detail": {"proposal_id": "wrong", "selected_by": "model"},
                 "view": dict(view, state="confirmed", request_id=request_id), "timestamp": "software-test"}
    jsonl(file, [opened, failed, confirmed])
    with pytest.raises(ValueError, match="Confirmation"):
        session(file)
