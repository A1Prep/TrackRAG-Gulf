"""Scene/input boundary checks with generated images, not grounding quality."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from trackrag.dropoff.inputs import build_instruction_request


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "original.png"
    Image.new("RGB", (160, 90), "grey").save(path)
    frame = {
        "recording_id": "scene-a", "frame_id": 10, "timestamp_s": 0.9,
        "width": 160, "height": 90, "image_path": str(path),
    }

    class Reader:
        recording_id = "scene-a"

        def get_frame(self, frame_id):
            if frame_id != 10:
                raise KeyError(frame_id)
            return frame

    row = {
        "recording_id": "scene-a", "frame_id": 10, "track_id": 2,
        "class_name": "Car", "bbox_xyxy": [10, 20, 40, 50], "source": "predicted",
    }
    return Reader(), frame, row


def build(scene, observations=None, **overrides):
    reader, _, row = scene
    arguments = {"frame_id": 10, "request_id": "request-a", "text": "Drop me beside that car"}
    arguments.update(overrides)
    return build_instruction_request(reader, [row] if observations is None else observations, **arguments)


def test_exact_image_and_current_prediction_are_bound_without_annotations(scene):
    request = build(scene)
    image_path = Path(request["scene"]["frame"]["image_path"])
    assert request["scene"]["provenance"]["image_sha256"] == sha256(image_path.read_bytes()).hexdigest()
    assert request["scene"]["tracks"] == [scene[2]]
    assert request["scene"]["source"] == "predicted"
    assert set(request) == {"request_id", "scene", "text", "input_mode", "dialogue"}
    assert not (image_path.parent / "annotations").exists()


def test_temporal_selection_never_opens_or_exposes_past_or_future_images(scene):
    reader, _, row = scene
    history = [{**row, "frame_id": 9}, row, {**row, "frame_id": 11}]
    with patch.object(reader, "get_frame", wraps=reader.get_frame) as lookup:
        request = build(scene, history)
    lookup.assert_called_once_with(10)
    assert [item["frame_id"] for item in request["scene"]["tracks"]] == [10]
    assert len(history) == 3


def test_empty_predicted_detections_remain_empty(scene):
    request = build(scene, [])
    assert request["scene"]["tracks"] == []
    assert "target" not in request
    assert "result" not in request


def test_voice_transcript_and_dialogue_are_preserved_without_mutation(scene):
    dialogue = [{"role": "assistant", "text": "Which car?", "recording_id": "scene-a", "frame_id": 10}]
    provenance = {"model": {"condition": "saved_finetuned_predictions"}}
    originals = deepcopy((scene[1], scene[2], dialogue, provenance))
    request = build(scene, input_mode="voice", text="Beside the red car",
                    original_transcript="Beside the read car", dialogue=dialogue, provenance=provenance)
    assert request["original_transcript"] == "Beside the read car"
    assert request["text"] == "Beside the red car"
    assert request["dialogue"] == dialogue
    request["scene"]["tracks"][0]["bbox_xyxy"][0] = 12
    request["scene"]["provenance"]["model"]["condition"] = "changed-return-value"
    request["dialogue"][0]["text"] = "changed-return-value"
    assert (scene[1], scene[2], dialogue, provenance) == originals


@pytest.mark.parametrize("field,value", [
    ("recording_id", "another-recording"), ("frame_id", 11), ("frame_id", True),
    ("width", 159), ("height", 91), ("timestamp_s", float("nan")),
])
def test_reader_identity_dimensions_and_timestamp_are_enforced(scene, field, value):
    scene[1][field] = value
    with pytest.raises(ValueError):
        build(scene)


@pytest.mark.parametrize("frame_id", [True, -1, 10.0])
def test_invalid_frame_request_is_rejected_before_reader_lookup(scene, frame_id):
    with patch.object(scene[0], "get_frame", wraps=scene[0].get_frame) as lookup:
        with pytest.raises(ValueError):
            build(scene, frame_id=frame_id)
    lookup.assert_not_called()


def test_missing_original_frame_remains_an_explicit_reader_error(scene):
    with pytest.raises(KeyError):
        build(scene, frame_id=9)


@pytest.mark.parametrize("frame_id", [9, 10, 11])
@pytest.mark.parametrize("field,value", [
    ("recording_id", "foreign"), ("source", "annotated"), ("track_id", True),
    ("track_id", -1), ("frame_id", -1), ("class_name", ""),
    ("bbox_xyxy", [0, 0, 200, 80]), ("bbox_xyxy", [0, 0, float("inf"), 80]),
    ("bbox_xyxy", [30, 20, 20, 40]),
])
def test_bad_observations_do_not_disappear_during_temporal_selection(scene, frame_id, field, value):
    row = {**scene[2], "frame_id": frame_id, field: value}
    with pytest.raises(ValueError):
        build(scene, [row])


def test_duplicate_current_track_identity_is_rejected(scene):
    with pytest.raises(ValueError, match="duplicate observation"):
        build(scene, [scene[2], deepcopy(scene[2])])


def test_unknown_gold_fields_in_frame_or_observation_are_rejected(scene):
    scene[2]["destination_point_xy"] = [20, 30]
    with pytest.raises(ValueError, match="unknown fields"):
        build(scene)
    scene[2].pop("destination_point_xy")
    scene[1]["target"] = [20, 30]
    with pytest.raises(ValueError, match="unknown"):
        build(scene)


@pytest.mark.parametrize("expected", ["0" * 64, "ABC", None, 1])
def test_expected_image_binding_is_not_silently_replaced(scene, expected):
    with pytest.raises(ValueError, match="SHA256|sha256"):
        build(scene, provenance={"image_sha256": expected})


def test_valid_expected_image_binding_and_other_provenance_are_preserved(scene):
    digest = sha256(Path(scene[1]["image_path"]).read_bytes()).hexdigest()
    provenance = {"image_sha256": digest, "vision_identity_hash": "1" * 64}
    assert build(scene, provenance=provenance)["scene"]["provenance"] == provenance


@pytest.mark.parametrize("kind", ["missing", "invalid", "empty", "truncated"])
def test_unavailable_or_bad_image_is_not_replaced_with_another_frame(scene, kind):
    image = Path(scene[1]["image_path"])
    if kind == "missing":
        image.unlink()
    elif kind == "invalid":
        image.write_bytes(b"not an image")
    elif kind == "empty":
        image.write_bytes(b"")
    else:
        image.write_bytes(image.read_bytes()[:40])
    with pytest.raises((OSError, ValueError)):
        build(scene)


def test_image_byte_and_pixel_budgets_are_enforced(scene):
    with patch("trackrag.dropoff.inputs.MAX_IMAGE_BYTES", 10):
        with pytest.raises(ValueError, match="byte budget"):
            build(scene)
    with patch("trackrag.dropoff.inputs.MAX_ORIGINAL_PIXELS", 100):
        with pytest.raises(ValueError, match="pixel budget"):
            build(scene)


def test_bounded_generator_cannot_supply_unlimited_observations(scene):
    def observations():
        while True:
            yield {**scene[2], "frame_id": 9}
    with patch("trackrag.dropoff.inputs.MAX_OBSERVATIONS", 3):
        with pytest.raises(ValueError, match="row budget"):
            build(scene, observations())


def test_current_scene_budget_does_not_allow_unbounded_prompt(scene):
    rows = [{**scene[2], "track_id": i} for i in range(513)]
    with pytest.raises(ValueError, match="512-observation"):
        build(scene, rows)


@pytest.mark.parametrize("overrides", [
    {"text": ""}, {"request_id": ""}, {"input_mode": "microphone"},
    {"dialogue": [{"role": "user", "text": "That one", "recording_id": "scene-a", "frame_id": 9}]},
    {"provenance": {"nonfinite": float("nan")}},
])
def test_existing_instruction_and_dialogue_contracts_are_kept(scene, overrides):
    with pytest.raises(ValueError):
        build(scene, **overrides)
