"""Private-input-free checks for aggregate comparisons of model, landmark and ground coordinates."""

from io import BytesIO
import json
import tarfile

from PIL import Image
import pytest

from tools.passenger_reviewed_geometry_audit import (
    compare_geometry,
    object_comparison,
    original_object_boxes,
)


def case(case_id, number, size, point, box, landmark_ids=()):
    width, height = size
    return {
        "case_id": case_id,
        "request": {"scene": {"frame": {"width": width, "height": height}}},
        "target": {
            "status": "proposed",
            "destination_point_xy": point,
            "destination_box_xyxy": box,
        },
        "source": {
            "recording_id": "unit-test",
            "frame_id": number,
            "landmarks": [{"track_id": i} for i in landmark_ids],
        },
    }


def test_normalized_proposals_convert_independently_at_two_original_resolutions():
    first = case("fixture-one", 1, (200, 100), (32, 34), (20, 30, 40, 40))
    second = case("fixture-two", 2, (400, 300), (60, 106), (40, 90, 80, 120))
    targets = {c["case_id"]: c for c in (first, second)}
    # The first predicted point hits. The second has the right height but exceeds the signed x bound.
    predictions = [
        ("fixture-one", [150, 350], [100, 300, 200, 400], [30, 35], [20, 30, 40, 40]),
        ("fixture-two", [300, 350], [275, 320, 325, 380], [120, 105], [110, 96, 130, 114]),
    ]
    outcomes = []
    receipts = []
    for ident, norm_point, norm_box, px_point, px_box in predictions:
        raw = {
            "status": "proposed",
            "destination_point_xy": norm_point,
            "destination_box_xyxy": norm_box,
        }
        outcomes.append(
            {
                "ok": True,
                "request_id": ident,
                "result": {**raw, "destination_point_xy": px_point, "destination_box_xyxy": px_box},
            }
        )
        receipts.append({"request_id": ident, "raw_text": "```json\n" + json.dumps(raw) + "\n```"})
    result = compare_geometry(targets, outcomes, receipts)
    assert (
        result["hits_in_original_pixels"]
        == result["hits_independently_in_normalized_raw_model_space"]
        == 1
    )
    assert result["largest_reconstructed_coordinate_difference_original_pixels"] == 0
    assert result["miss_axis_among_returned_points_on_eligible_proposals"] == {"horizontal_only": 1}
    outcomes[1]["result"]["destination_point_xy"] = [
        60,
        105,
    ]  # Simulate a width-conversion error in a saved result.
    with pytest.raises(ValueError, match="Pixel and normalized containment disagree"):
        compare_geometry(targets, outcomes, receipts)


def test_project_emt_reader_preserves_distinct_landmark_and_stopping_region(tmp_path):
    image_dir = tmp_path / "original/images/test/unit-test"
    image_dir.mkdir(parents=True)
    for frame in (1, 2):
        Image.new("RGB", (200, 100)).save(image_dir / f"{frame:05}.jpg")
    archive = tmp_path / "software-fixture.tar.gz"
    # The three intermediate EMT columns are unrelated to the last four xyxy columns.
    lines = b"1 5 Car 351 435 513 20 30 60 70\n2 5 Car 351 435 513 40 30 90 70\n1 7 AV 0 0 0 2 2 197 98\n"
    with tarfile.open(archive, "w:gz") as z:
        member = tarfile.TarInfo("EMT/annotations/test/unit-test.txt")
        member.size = len(lines)
        z.addfile(member, BytesIO(lines))
    objects = original_object_boxes(
        {("unit-test", 1), ("unit-test", 2)}, archive.read_bytes(), tmp_path / "original"
    )
    assert objects[("unit-test", 1)] == {5: (20.0, 30.0, 60.0, 70.0)}
    assert objects[("unit-test", 2)] == {5: (40.0, 30.0, 90.0, 70.0)}
    beside = case("fixture-beside", 1, (200, 100), (70, 75), (65, 72, 75, 78), [5])
    on_vehicle = case("fixture-on-vehicle", 2, (200, 100), (50, 50), (40, 30, 90, 70), [5])
    result = object_comparison([beside, on_vehicle], objects)
    assert result["listed_ids_found_in_official_source"] == 2
    assert result["signed_rectangle_exactly_equal_to_a_listed_landmark_rectangle"] == 1
    assert result["eligible_frames_with_a_listed_landmark_box_iou_at_least_half"] == 1
    assert result["signed_point_inside_a_matching_listed_landmark_rectangle"] == 1
