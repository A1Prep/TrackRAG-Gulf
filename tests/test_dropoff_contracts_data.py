"""Software checks with generated fixtures; not real-model accuracy evidence."""

from copy import deepcopy
import json

import pytest
from PIL import Image, ImageDraw

from trackrag.dropoff.contracts import error_outcome, validate_request
from trackrag.dropoff.data import (
    file_sha256, inference_input_sha256, load_cases, supervision_sha256, validate_disjoint_groups,
)
from trackrag.dropoff.formatting import build_messages, build_training_messages, parse_model_output


@pytest.fixture
def case(tmp_path):
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (160, 90), "grey").save(image)
    target_mask = Image.new("L", (160, 90))
    draw = ImageDraw.Draw(target_mask)
    draw.rectangle((20, 40, 119, 79), fill=255)
    draw.rectangle((60, 50, 79, 69), fill=0)  # Non-convex target: centre lies in a hole.
    target_mask.save(mask)
    return {
        "schema_version": 1, "case_id": "fixture-1", "scene_group": "fixture-scene",
        "partition": "train", "source": {
            "dataset": "generated-test-fixture", "source_split": "train",
            "image_sha256": file_sha256(image), "mask_sha256": file_sha256(mask),
            "mask_path": "mask.png",
        },
        "request": {
            "request_id": "fixture-1",
            "scene": {
                "frame": {"recording_id": "fixture-recording", "frame_id": 10, "timestamp_s": 0.9,
                          "width": 160, "height": 90, "image_path": "image.png"},
                "source": "predicted", "tracks": [],
            },
            "text": "Stop next to the car", "input_mode": "typed", "dialogue": [],
        },
        "target": {"destination_point_xy": [30.5, 60.5],
                   "destination_box_xyxy": [20, 40, 120, 80]},
        "supervised_fields": ["destination_point_xy", "destination_box_xyxy"],
        "review": {"status": "public_annotation"},
    }


def write_case(tmp_path, case):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", ("gold", "future", "foreign", "stale", "duplicate"))
def test_request_rejects_invalid_evidence_boundaries(case, kind):
    request = case["request"]
    row = {"recording_id": "fixture-recording", "frame_id": 10, "track_id": 2,
           "class_name": "Car", "bbox_xyxy": [10, 20, 40, 50], "source": "predicted"}
    if kind == "gold":
        request["target"] = {"destination_point_xy": [30, 60]}
    elif kind == "future":
        row["frame_id"] = 11
        request["scene"]["tracks"] = [row]
    elif kind == "foreign":
        row["recording_id"] = "another-recording"
        request["scene"]["tracks"] = [row]
    elif kind == "stale":
        request["dialogue"] = [{"role": "user", "text": "That one", "recording_id": "fixture-recording",
                                "frame_id": 9}]
    else:
        request["scene"]["tracks"] = [row, deepcopy(row)]
    with pytest.raises(ValueError):
        validate_request(request)


def test_partial_labels_do_not_become_fabricated_status_or_anchor_targets(case, tmp_path):
    loaded = load_cases(write_case(tmp_path, case))[0]
    messages = build_training_messages(loaded)
    target = json.loads(messages[-1]["content"])
    assert set(target) == {"destination_point_xy", "destination_box_xyxy"}
    assert "destination-only annotation task" in messages[0]["content"]
    request_before = deepcopy(loaded["request"])
    loaded["source"]["private_reference"] = "SECRET_REFERENCE"
    assert "SECRET_REFERENCE" not in json.dumps(build_messages(loaded["request"]))
    assert loaded["request"] == request_before


def test_normalized_to_original_geometry_round_trip(case, tmp_path):
    loaded = load_cases(write_case(tmp_path, case))[0]
    raw = json.loads(build_training_messages(loaded)[-1]["content"])
    raw.update(status="proposed", clarification_question=None)
    parsed = parse_model_output(json.dumps(raw), 160, 90)
    assert parsed["destination_point_xy"] == pytest.approx(case["target"]["destination_point_xy"], abs=1e-5)
    assert parsed["destination_box_xyxy"] == pytest.approx(case["target"]["destination_box_xyxy"], abs=1e-5)


@pytest.mark.parametrize("raw", (
    '{"status":"proposed","status":"needs_clarification"}',
    '{"status":[],"destination_point_xy":null,"destination_box_xyxy":null,"clarification_question":null}',
    '{"status":"proposed","destination_point_xy":[1000,500],"destination_box_xyxy":[0,0,1000,1000],"clarification_question":null}',
    '{"status":"needs_clarification","destination_point_xy":[10,10],"destination_box_xyxy":[0,0,20,20],"clarification_question":"Which car?"}',
    '{"status":"insufficient_visual_evidence","destination_point_xy":null,"destination_box_xyxy":null,"clarification_question":null,"confidence":0.99}',
))
def test_invalid_model_output_is_not_repaired_into_a_proposal(raw):
    with pytest.raises(ValueError):
        parse_model_output(raw, 160, 90)


def test_operational_failure_is_not_semantic_abstention():
    result = error_outcome("checkpoint_unavailable", "No checkpoint available", "request-1")
    assert result["ok"] is False
    assert "result" not in result and "status" not in result
    assert result["error"]["code"] == "checkpoint_unavailable"


@pytest.mark.parametrize("mutation", ("test", "pending", "image_hash", "mask_hole", "mask_box", "test_source"))
def test_training_data_rejects_unusable_or_leaking_cases(case, tmp_path, mutation):
    if mutation == "test":
        case["partition"] = "test"
    elif mutation == "pending":
        case["review"]["status"] = "pending"
    elif mutation == "image_hash":
        case["source"]["image_sha256"] = "0" * 64
    elif mutation == "mask_hole":
        case["target"]["destination_point_xy"] = [70.5, 60.5]
    elif mutation == "mask_box":
        case["target"]["destination_box_xyxy"] = [19, 40, 120, 80]
    else:
        case["source"]["source_split"] = "test"
    with pytest.raises(ValueError):
        load_cases(write_case(tmp_path, case))


def test_duplicate_json_and_excluded_rows_are_not_silently_filtered(case, tmp_path):
    path = write_case(tmp_path, case)
    contents = path.read_text(encoding="utf-8")
    path.write_text(contents.replace('"partition": "train"', '"partition": "test", "partition": "train"'),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_cases(path)


@pytest.mark.parametrize("source", (
    {"dataset": "Talk2Car-RegSeg", "source_split": "val", "scene_group_verified": True},
    {"dataset": "Talk2Car-RegSeg", "source_split": "train", "scene_group_verified": False},
    {"dataset": "Talk2Car-RegSeg", "source_split": "train", "scene_group_verified": True,
     "concise_test_overlap": True},
    {"dataset": "another-source", "source_split": "test"},
))
def test_checkpoint_selection_cannot_consume_reserved_sources(case, tmp_path, source):
    case["partition"] = "development"
    case["source"].update(source)
    with pytest.raises(ValueError):
        load_cases(write_case(tmp_path, case), allowed_partitions={"development"})


def test_disjoint_group_guard_includes_identical_image_bytes(case, tmp_path):
    training = load_cases(write_case(tmp_path, case))
    development = deepcopy(training)
    development[0]["case_id"] = "different-id"
    development[0]["scene_group"] = "different-group"
    development[0]["partition"] = "development"
    with pytest.raises(ValueError, match="identical image"):
        validate_disjoint_groups(training, development)


@pytest.mark.parametrize("field", ("image_sha256", "mask_sha256"))
def test_declared_hash_cannot_be_omitted(case, tmp_path, field):
    case["source"].pop(field)
    with pytest.raises(ValueError, match="SHA256"):
        load_cases(write_case(tmp_path, case))


def test_human_review_binds_instruction_as_well_as_image_and_labels(case, tmp_path):
    case["review"] = {
        "status": "human_reviewed", "reviewers": ["synthetic-test-reviewer"],
        "reviewed_at": "2026-11-20T00:00:00+00:00",
        "image_sha256": case["source"]["image_sha256"],
        "supervision_sha256": supervision_sha256(case["target"], case["supervised_fields"],
                                                case["request"]["scene"]["frame"]),
        "input_sha256": inference_input_sha256(case["request"], case["source"]["image_sha256"]),
    }
    load_cases(write_case(tmp_path, case))
    case["request"]["text"] = "Stop at the opposite location instead."
    with pytest.raises(ValueError, match="instruction and scene"):
        load_cases(write_case(tmp_path, case))
