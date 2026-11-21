"""Software-only boundary tests; no model invocation or task-accuracy claim."""
from copy import deepcopy

import pytest
from PIL import Image

from tools.dropoff_candidate_choice import parse_choice, PROMPT, validate_public_requests

OPTIONS = [{"candidate_id":"R01", "point_original_px":[20.5,30.5], "bbox_original_px":[20,30,22,32]}]
FRAME = {"width":100,"height":100}


def test_choice_returns_only_two_geometry_fields_without_app_semantics():
    selected, geometry = parse_choice('{"candidate_id":"R01"}', OPTIONS, FRAME)
    assert selected == "R01"
    assert geometry == {"destination_point_xy":[20.5,30.5], "destination_box_xyxy":[20,30,22,32]}
    assert "status" not in geometry
    assert "clarification_question" not in geometry


@pytest.mark.parametrize("text", [
    '{"candidate_id":"R99","candidate_id":"R01"}',
    '{"candidate_id":NaN}', '{"candidate_id":Infinity}',
    '{"candidate_id":"R99"}', '{"candidate_id":null}', '{"candidate_id":true}',
    '{"candidate_id":"R01","status":"proposed"}', '{}', '[]',
    'R01', '{"candidate_id":"R01"} explanation',
])
def test_invalid_choice_never_gets_fallback_or_semantic_defaults(text):
    with pytest.raises(ValueError): parse_choice(text, OPTIONS, FRAME)


def test_non_text_and_large_text_rejected():
    for text in ({"candidate_id":"R01"}, " "*4097):
        with pytest.raises(ValueError): parse_choice(text, OPTIONS, FRAME)


def test_frozen_prompt_declares_forced_choice_and_no_status_task():
    assert "destination-only annotation task" in PROMPT
    assert "forced-choice diagnostic" in PROMPT
    assert "Do not return status" in PROMPT


@pytest.fixture
def six_requests(tmp_path):
    image = tmp_path / "reference.png"
    Image.new("RGB", (100, 100), "gray").save(image)
    rows = [{"request_id":f"regseg-fixture-{i}", "scene":{"source":"predicted", "tracks":[],
             "frame":{"recording_id":f"regseg-fixture-{i}", "frame_id":0, "timestamp_s":0.0,
                      "width":100,"height":100,"image_path":str(image)}},
             "text":"Drop me beside the white truck.","input_mode":"typed","dialogue":[]} for i in range(6)]
    return rows, {"public_case_ids":[r["request_id"] for r in rows]}


def test_six_fixed_requests_preserve_frozen_protocol_order(six_requests):
    rows, protocol = six_requests
    assert validate_public_requests(list(reversed(rows)), protocol, rows) == rows


def test_wrong_instruction_is_not_a_valid_choice_diagnostic_input(six_requests):
    rows, protocol = six_requests
    changed = deepcopy(rows)
    changed[0]["text"] = "Drop me beside the black bus."
    with pytest.raises(ValueError, match="instruction/context"):
        validate_public_requests(changed, protocol, rows)


def test_replacement_image_rejected_even_when_scene_and_instruction_match(six_requests, tmp_path):
    rows, protocol = six_requests
    changed = deepcopy(rows)
    replacement = tmp_path / "other.png"
    Image.new("RGB", (100, 100), "black").save(replacement)
    changed[0]["scene"]["frame"]["image_path"] = str(replacement)
    with pytest.raises(ValueError, match="image differs"):
        validate_public_requests(changed, protocol, rows)


def test_emt_and_incomplete_cohorts_rejected(six_requests):
    rows, protocol = six_requests
    with pytest.raises(ValueError, match="Exactly"):
        validate_public_requests(rows[:-1], protocol, rows)
    changed = deepcopy(rows)
    changed[0]["scene"]["frame"]["recording_id"] = "video_122233"
    with pytest.raises(ValueError, match="EMT and final"):
        validate_public_requests(changed, protocol, rows)
