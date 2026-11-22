"""Shared handoff checks using illustrative records, not real grounding accuracy."""

from copy import deepcopy
import json

import pytest
from PIL import Image

from trackrag.dropoff.contracts import (
    error_outcome, proposal_fingerprint, validate_interaction_state,
    validate_outcome, validate_proposal, validate_request,
)
from trackrag.dropoff.formatting import build_messages
from trackrag.dropoff.model import GroundingModel


@pytest.fixture
def instruction_request():
    return {
        "session_id": "passenger-session-1", "request_id": "instruction-1",
        "scene": {
            "frame": {"recording_id": "example-recording", "frame_id": 150,
                      "timestamp_s": 6.0, "width": 1280, "height": 720,
                      "image_path": "original-frame.png"},
            "source": "predicted",
            "tracks": [{"recording_id": "example-recording", "frame_id": 150,
                        "track_id": 7, "source": "predicted", "class_name": "Car",
                        "bbox_xyxy": [780, 450, 910, 610]}],
            "provenance": {"condition": "illustrative_fixture", "image_sha256": "a" * 64},
        },
        "text": "Drop me off beside the white car.", "input_mode": "typed", "dialogue": [],
    }


def success(request, status="proposed"):
    return {
        "ok": True, "request_id": request["request_id"],
        "result": {
            "status": status,
            "destination_point_xy": [620, 650] if status == "proposed" else None,
            "destination_box_xyxy": [560, 610, 700, 700] if status == "proposed" else None,
            "clarification_question": "The car on the left or the right?"
            if status == "needs_clarification" else None,
            "anchor_boxes_xyxy": [[780, 450, 910, 610]],
        },
        "provenance": {"backend_kind": "injected_test_backend", "model_id": "fixture-only"},
    }


def proposal(request):
    outcome = validate_outcome(success(request), request)
    frame = request["scene"]["frame"]
    return {
        "proposal_id": "app-proposal-1", "request_id": request["request_id"],
        "scene": {"recording_id": frame["recording_id"], "frame_id": frame["frame_id"],
                  "source": request["scene"]["source"]},
        "result": outcome["result"], "provenance": outcome["provenance"],
    }


def state(request, status="pending"):
    current = proposal(request) if status in ("proposed", "confirmed") else None
    confirmation = None
    if status == "confirmed":
        confirmation = {
            "proposal_id": current["proposal_id"], "request_id": current["request_id"],
            "scene": deepcopy(current["scene"]),
            "proposal_sha256": proposal_fingerprint(current, request),
        }
    return {
        "session_id": request["session_id"], "request": deepcopy(request), "status": status,
        "pending_clarification": "Which white car?" if status == "needs_clarification" else None,
        "proposal": current, "confirmation": confirmation, "transcript_corrections": [],
    }


def test_white_car_handoff_round_trips_as_json_without_changing_caller_records(instruction_request):
    request = instruction_request
    original = deepcopy(request)
    outcome = validate_outcome(success(request), request)
    assert outcome["result"]["destination_box_xyxy"] != request["scene"]["tracks"][0]["bbox_xyxy"]
    assert "proposal_id" not in outcome["result"]
    checked = validate_interaction_state(json.loads(json.dumps(state(request, "confirmed"))))
    checked["request"]["scene"]["tracks"].clear()
    checked["proposal"]["result"]["destination_point_xy"][0] = 1
    assert request == original


@pytest.mark.parametrize("status", ["proposed", "needs_clarification", "insufficient_visual_evidence"])
def test_semantic_results_are_successful_calls_even_when_unresolved(instruction_request, status):
    request = instruction_request
    outcome = validate_outcome(success(request, status), request)
    assert outcome["ok"] is True
    if status != "proposed":
        assert outcome["result"]["destination_point_xy"] is None
        with pytest.raises(ValueError, match="requires a proposed"):
            validate_proposal({**proposal(request), "result": outcome["result"]}, request)


@pytest.mark.parametrize("code", [
    "checkpoint_unavailable", "artifact_invalid", "timeout", "resource_unavailable",
    "invalid_model_output", "dependency_unavailable", "generation_failed",
])
def test_operational_errors_remain_errors_with_no_result_or_proposal(instruction_request, code):
    request = instruction_request
    outcome = validate_outcome(error_outcome(code, "Illustrative failure", request["request_id"]), request)
    assert outcome["ok"] is False
    assert outcome["error"]["code"] == code
    assert "result" not in outcome
    assert "proposal" not in outcome
    with pytest.raises(ValueError):
        validate_outcome({**outcome, "result": success(request)["result"]}, request)


@pytest.mark.parametrize("mutation", [
    "wrong_request", "not_boolean", "both_payloads", "missing_box", "outside_image",
    "outside_region", "nonfinite", "negative_box", "model_confirmation", "confidence",
    "bad_provenance", "nonfinite_provenance", "clarification_with_destination",
])
def test_untrusted_outcomes_are_rejected_before_the_app_can_draw_them(instruction_request, mutation):
    request = instruction_request
    outcome = success(request)
    result = outcome["result"]
    if mutation == "wrong_request":
        outcome["request_id"] = "old-request"
    elif mutation == "not_boolean":
        outcome["ok"] = 1
    elif mutation == "both_payloads":
        outcome["error"] = {"code": "timeout", "message": "failed"}
    elif mutation == "missing_box":
        result.pop("destination_box_xyxy")
    elif mutation == "outside_image":
        result["destination_point_xy"] = [1280, 650]
    elif mutation == "outside_region":
        result["destination_point_xy"] = [1, 1]
    elif mutation == "nonfinite":
        result["destination_point_xy"][0] = float("nan")
    elif mutation == "negative_box":
        result["destination_box_xyxy"][0] = -1
    elif mutation == "model_confirmation":
        result["confirmation"] = {"proposal_id": "model-invented"}
    elif mutation == "confidence":
        result["confidence"] = 0.99
    elif mutation == "bad_provenance":
        outcome["provenance"] = []
    elif mutation == "nonfinite_provenance":
        outcome["provenance"]["elapsed_s"] = float("inf")
    else:
        result["status"] = "needs_clarification"
        result["clarification_question"] = "Which car?"
    with pytest.raises(ValueError):
        validate_outcome(outcome, request)


@pytest.mark.parametrize("status", [
    "pending", "needs_clarification", "insufficient_visual_evidence",
    "proposed", "confirmed", "cancelled", "error",
])
def test_submitted_instruction_states_have_a_shared_serializable_format(instruction_request, status):
    request = instruction_request
    assert validate_interaction_state(state(request, status))["status"] == status


@pytest.mark.parametrize("mutation", [
    "proposal_id", "request_id", "recording", "frame", "source", "point", "region",
    "corrected_text", "image_path", "image_hash", "dialogue", "session",
])
def test_old_confirmation_cannot_authorize_a_changed_proposal_or_input(instruction_request, mutation):
    request = instruction_request
    current = state(request, "confirmed")
    offered = current["proposal"]
    if mutation == "proposal_id":
        offered["proposal_id"] = "app-proposal-2"
    elif mutation == "request_id":
        current["request"]["request_id"] = "instruction-2"
    elif mutation in ("recording", "frame", "source"):
        key, value = {"recording": ("recording_id", "foreign"), "frame": ("frame_id", 151),
                      "source": ("source", "annotated")}[mutation]
        offered["scene"][key] = value
    elif mutation == "point":
        offered["result"]["destination_point_xy"] = [630, 650]
    elif mutation == "region":
        offered["result"]["destination_box_xyxy"] = [550, 610, 700, 700]
    elif mutation == "corrected_text":
        current["request"]["text"] = "Beside the black car."
    elif mutation == "image_path":
        current["request"]["scene"]["frame"]["image_path"] = "different-image.png"
    elif mutation == "image_hash":
        current["request"]["scene"]["provenance"]["image_sha256"] = "b" * 64
    elif mutation == "dialogue":
        current["request"]["dialogue"] = [{"role": "user", "text": "The other one.",
                                          "recording_id": "example-recording", "frame_id": 150}]
    else:
        current["session_id"] = "another-session"
    with pytest.raises(ValueError):
        validate_interaction_state(current)


@pytest.mark.parametrize("status", ["pending", "needs_clarification", "error", "cancelled"])
def test_unresolved_failed_or_cancelled_states_cannot_keep_a_live_proposal(instruction_request, status):
    request = instruction_request
    current = state(request, status)
    current["proposal"] = proposal(request)
    with pytest.raises(ValueError, match="current proposal"):
        validate_interaction_state(current)


def test_corrected_voice_text_preserves_asr_and_uses_the_typed_model_interface(instruction_request):
    request = instruction_request
    request.update(input_mode="voice", original_transcript="Drop me off beside the wide car.")
    current = state(request)
    current["transcript_corrections"] = [{
        "request_id": request["request_id"], "original_transcript": request["original_transcript"],
        "corrected_text": request["text"],
    }]
    checked = validate_interaction_state(current)
    assert checked["request"]["original_transcript"] != checked["request"]["text"]
    messages = build_messages(request)
    assert messages[-1]["content"] == request["text"]
    assert request["original_transcript"] not in json.dumps(messages)
    current["transcript_corrections"][0]["corrected_text"] = "The wrong correction"
    with pytest.raises(ValueError, match="correction differs"):
        validate_interaction_state(current)


def test_voice_state_requires_retaining_the_original_asr_transcript(instruction_request):
    request = instruction_request
    request["input_mode"] = "voice"
    with pytest.raises(ValueError, match="original transcript"):
        validate_interaction_state(state(request))


def test_legacy_batch_requests_remain_valid_and_session_metadata_is_not_prompt_text(instruction_request):
    request = instruction_request
    legacy = deepcopy(request)
    legacy.pop("session_id")
    assert validate_request(legacy) == legacy
    assert build_messages(request) == build_messages(legacy)
    request["session_id"] = ""
    with pytest.raises(ValueError, match="session_id"):
        validate_request(request)


def test_existing_task5_adapter_output_satisfies_the_app_boundary(instruction_request, tmp_path):
    request = instruction_request
    path = tmp_path / "fixture.png"
    Image.new("RGB", (1280, 720), "grey").save(path)
    request["scene"]["frame"]["image_path"] = str(path)
    request["scene"].pop("provenance")

    class Backend:
        def generate(self, messages, *, config):
            return json.dumps({
                "status": "proposed", "destination_point_xy": [500, 900],
                "destination_box_xyxy": [450, 850, 550, 950], "clarification_question": None,
            })

    model = GroundingModel(backend=Backend())
    outcome = validate_outcome(model.interpret(request), request)
    assert outcome["provenance"]["backend_kind"] == "injected_test_backend"
    assert outcome["result"]["destination_point_xy"] == [640, 648]
    assert "proposal_id" not in outcome
