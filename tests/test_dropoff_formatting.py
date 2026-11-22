"""Interpretation schema checks; no model outputs are repaired or fabricated."""

import hashlib
import json
import re

import pytest

from trackrag.dropoff.formatting import (DESTINATION_ONLY, INTERPRET, SYSTEM, build_messages,
                                        build_training_messages, parse_model_output)


def request():
    return {"request_id": "format-fixture", "text": "Stop near the entrance", "input_mode": "typed",
        "dialogue": [], "scene": {"source": "predicted", "tracks": [], "frame": {
            "recording_id": "format-recording", "frame_id": 3, "timestamp_s": 0.3,
            "width": 200, "height": 100, "image_path": "frame.png"}}}


def test_primary_destination_prompt_bytes_remain_frozen():
    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == "28e5c33e489b65450ff4f53ec37a643644f560f6d7824ed87b3d9e9095fc902d"
    assert hashlib.sha256(DESTINATION_ONLY.encode()).hexdigest() == "479385265f3247930689b3cbd364ba6a329dd808a0836c38a89d546854dbb3d7"
    assert hashlib.sha256((SYSTEM + DESTINATION_ONLY).encode()).hexdigest() == "35b73cf053bab91a2dfafbf10af2f996b5f7ac3d1d966b1e8b99b5c95f170e70"


def test_interpret_prompt_requires_complete_mutually_exclusive_structures_without_numeric_examples():
    for key in ("status", "destination_point_xy", "destination_box_xyxy", "clarification_question"):
        assert f'"{key}"' in INTERPRET
    assert "Never omit a required key" in INTERPRET
    assert "mutually exclusive" in INTERPRET
    assert '"destination_point_xy": null' in INTERPRET
    assert '"destination_box_xyxy": null' in INTERPRET
    assert '"clarification_question": null' in INTERPRET
    assert "Never combine a clarification question with destination coordinates" in INTERPRET
    assert not re.search(r"\d", INTERPRET)
    assert not re.search(r"\[\s*[-+]?\d", INTERPRET)


def test_only_the_interpretation_system_instruction_differs_between_prompt_modes():
    interpretation = build_messages(request(), task="interpret")
    destination = build_messages(request(), task="destination_only")
    assert interpretation[0]["content"] == SYSTEM + INTERPRET
    assert destination[0]["content"] == SYSTEM + DESTINATION_ONLY
    assert interpretation[1:] == destination[1:]


def test_public_partial_training_keeps_original_prompt_and_only_available_targets():
    case = {"request": request(), "supervised_fields": ["destination_point_xy", "destination_box_xyxy"],
        "target": {"destination_point_xy": [60, 60], "destination_box_xyxy": [50, 50, 70, 70]}}
    messages = build_training_messages(case)
    assert messages[0]["content"] == SYSTEM + DESTINATION_ONLY
    assert set(json.loads(messages[-1]["content"])) == {"destination_point_xy", "destination_box_xyxy"}


@pytest.mark.parametrize("status,point,box,question", [
    ("proposed", [400, 600], [300, 500, 500, 700], None),
    ("needs_clarification", None, None, "Which visible landmark do you mean?"),
    ("insufficient_visual_evidence", None, None, None),
])
def test_complete_exclusive_statuses_parse_without_changing_semantics(status, point, box, question):
    parsed = parse_model_output(json.dumps({"status": status, "destination_point_xy": point,
        "destination_box_xyxy": box, "clarification_question": question}), 200, 100)
    assert parsed["status"] == status and parsed["clarification_question"] == question
    if status == "proposed":
        assert parsed["destination_point_xy"] == [80, 60]
    else:
        assert parsed["destination_point_xy"] is None and parsed["destination_box_xyxy"] is None


@pytest.mark.parametrize("raw", [
    {"status": "needs_clarification", "destination_box_xyxy": [100, 200, 300, 400],
     "clarification_question": "Which visible landmark?"},
    {"status": "needs_clarification", "destination_point_xy": None,
     "destination_box_xyxy": [100, 200, 300, 400], "clarification_question": "Which visible landmark?"},
    {"status": "proposed", "destination_point_xy": None,
     "destination_box_xyxy": [100, 200, 300, 400], "clarification_question": None},
    {"status": "proposed", "destination_point_xy": [200, 300],
     "destination_box_xyxy": [100, 200, 300, 400], "clarification_question": "Which visible landmark?"},
])
def test_schema_violations_remain_errors_after_prompt_repair(raw):
    original = json.loads(json.dumps(raw))
    with pytest.raises(ValueError):
        parse_model_output(json.dumps(raw), 200, 100)
    assert raw == original
