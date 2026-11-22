"""Generated geometry/semantic fixtures; no pretrained parking accuracy claims."""
from copy import deepcopy
import json

import numpy as np
import pytest

from trackrag.dropoff import parking_surface as surface


def request():
    return {"request_id": "software", "input_mode": "typed", "dialogue": [],
            "text": "Drop me between the two cars", "scene": {
                "frame": {"recording_id": "fixture", "frame_id": 2, "timestamp_s": .2,
                          "width": 512, "height": 288, "image_path": "software.png"},
                "source": "predicted", "provenance": {"image_sha256": "a" * 64}, "tracks": [
                    {"recording_id": "fixture", "frame_id": 2, "track_id": i, "class_name": "Car",
                     "source": "predicted", "bbox_xyxy": box} for i, box in
                    ((1, [80, 100, 140, 180]), (2, [240, 100, 300, 180]))]}}


def map_for(ground=10):
    classes = np.full((288, 512), 27, dtype=np.uint8)
    classes[145:190, 140:240] = ground
    return classes


def test_primary_patches_are_all_semantic_parking_original_pixels_and_serializable():
    req, classes = request(), map_for()
    options = surface.options_for(req, classes, [1, 2])
    assert options and len(options) <= 4
    json.dumps(options, allow_nan=False)
    for option in options:
        assert option["surface"] == "parking" and not option["requires_visual_review"]
        left, top, right, bottom = option["region_cell_bounds_xyxy"]
        assert right - left >= 5 and bottom - top >= 5
        assert (classes[top:bottom, left:right] == 10).all()
        assert option["region_box_xyxy"] == [left, top, right, bottom]
        x, y = option["point"]
        assert left <= x < right and top <= y < bottom
        assert option["region_class_counts"] == {"10": (right - left) * (bottom - top)}
        assert option["tracks"] == [1, 2]
        assert surface.validate_option(req, classes, [1, 2], option)


def test_established_supported_center_matches_the_unchanged_research_control():
    from tools.passenger_parking_candidates import options_for as stock
    req, classes = request(), map_for()
    normal = surface.options_for(req, classes, [1, 2])
    reference = stock(req, classes, "parking", [1, 2])
    assert [v["point"] for v in normal] == [v["point"] for v in reference]
    assert all(v["center_source"] == "established_parking_component" for v in normal)


def test_optional_second_board_is_distinct_parking_only_and_requires_review():
    req, classes = request(), map_for()
    first = surface.options_for(req, classes, [1, 2])
    second = surface.options_for(req, classes, [1, 2], prefer_established=False)
    assert second
    for candidate in second:
        assert candidate["surface"] == "parking" and candidate["requires_visual_review"]
        assert not candidate["evidence"]["prefer_established"]
        assert all(np.linalg.norm(np.array(candidate["point"]) - row["point"]) >= 24 for row in first)
        assert surface.validate_option(req, classes, [1, 2], candidate, prefer_established=False)
        with pytest.raises(ValueError):
            surface.validate_option(req, classes, [1, 2], candidate)
    with pytest.raises(ValueError, match="parking-only"):
        surface.options_for(req, classes, [1, 2], prefer_established=False, allow_road_extension=True)


def test_original_pixel_scaling_keeps_exact_supported_cell_bounds():
    req, classes = request(), map_for()
    original = surface.options_for(req, classes, [1, 2])
    req["scene"]["frame"].update(width=1024, height=576)
    for track in req["scene"]["tracks"]:
        track["bbox_xyxy"] = [2 * v for v in track["bbox_xyxy"]]
    scaled = surface.options_for(req, classes, [1, 2])
    assert len(scaled) == len(original)
    for first, second in zip(original, scaled):
        assert second["point"] == [2 * v for v in first["point"]]
        assert second["region_box_xyxy"] == [2 * v for v in first["region_box_xyxy"]]
        assert second["region_cell_bounds_xyxy"] == first["region_cell_bounds_xyxy"]


def test_current_objects_and_tiny_support_cannot_produce_confirmation_patch():
    req, classes = request(), map_for()
    req["scene"]["tracks"].append({**req["scene"]["tracks"][0], "track_id": 3,
                                    "class_name": "Pedestrian", "bbox_xyxy": [140, 145, 240, 190]})
    assert surface.options_for(req, classes, [1, 2]) == []
    thin = np.full((288, 512), 27, dtype=np.uint8)
    thin[170:173, 145:235] = 10
    assert surface.options_for(request(), thin, [1, 2]) == []


def test_intervening_vehicle_breaks_adjacency_even_with_semantic_parking():
    req = request()
    req["scene"]["tracks"].append({**req["scene"]["tracks"][0], "track_id": 3,
                                    "bbox_xyxy": [170, 110, 210, 180]})
    assert surface.options_for(req, map_for(), [1, 2]) == []


def test_perspective_uses_local_scale_instead_of_hard_three_point_five_ratio():
    req = request()
    req["scene"]["tracks"][0]["bbox_xyxy"] = [30, 60, 130, 260]
    req["scene"]["tracks"][1]["bbox_xyxy"] = [225, 110, 265, 150]
    classes = np.full((288, 512), 27, dtype=np.uint8)
    classes[145:220, 140:220] = 10
    assert surface.options_for(req, classes, [1, 2])


def fallback_map():
    classes = map_for(13)
    classes[140:144, 145:235] = 2
    classes[146:186, 185:188] = 24
    return classes


def test_fallback_requires_measured_roadside_and_separate_nonparallel_marking():
    req = request()
    assert surface.options_for(req, fallback_map(), [1, 2]) == []
    options = surface.options_for(req, fallback_map(), [1, 2], allow_road_extension=True)
    assert options
    for option in options:
        assert option["surface"] == "roadside_road_visual_review"
        assert option["requires_visual_review"] is True
        assert [w["kind"] for w in option["surface_witnesses"]] == ["semantic_curb_or_sidewalk", "nonparallel_general_marking"]
        assert set(option["region_class_counts"]) <= {"10", "13", "24"}


def test_parallel_boundary_requires_close_aligned_curb_and_contact_band_then_visual_review():
    classes = map_for(13)
    classes[158:162, 145:235] = 2
    classes[165:168, 145:235] = 24
    choices = surface.options_for(request(), classes, [1, 2], allow_road_extension=True)
    assert choices
    assert all(o["requires_visual_review"] for o in choices)
    assert all(o["surface_witnesses"][-1]["kind"] == "parallel_marking_along_curb_contact_band" for o in choices)
    classes[classes == 2] = 13
    assert surface.options_for(request(), classes, [1, 2], allow_road_extension=True) == []


def test_occupied_semantic_witness_cannot_license_an_otherwise_free_road_patch():
    req, classes = request(), map_for(13)
    classes[140:144, 145:235] = 2
    classes[166:169, 170:173] = 10
    req["scene"]["tracks"].append({**req["scene"]["tracks"][0], "track_id": 3,
                                    "class_name": "Pedestrian", "bbox_xyxy": [169, 164, 174, 170]})
    assert surface.options_for(req, classes, [1, 2], allow_road_extension=True) == []


@pytest.mark.parametrize("change", ["plain_road", "no_curb", "parallel_lane", "crosswalk", "sidewalk"])
def test_lane_or_sidewalk_evidence_cannot_be_promoted_to_fallback(change):
    classes = fallback_map()
    if change == "plain_road":
        classes = map_for(13)
    elif change == "no_curb":
        classes[classes == 2] = 27
    elif change == "parallel_lane":
        classes[classes == 24] = 13
        classes[168:171, 145:235] = 24
    elif change == "crosswalk":
        classes[160:165, 145:230] = 23
    else:
        classes[classes == 13] = 15
    assert surface.options_for(request(), classes, [1, 2], allow_road_extension=True) == []


@pytest.mark.parametrize("change", ["past", "future", "gold", "unknown", "duplicate", "boolean", "missing_sha", "aspect"])
def test_stale_or_invalid_inputs_fail_before_any_geometry_is_offered(change):
    req, classes, ids = request(), map_for(), [1, 2]
    if change in ("past", "future"):
        req["scene"]["tracks"][0]["frame_id"] = 1 if change == "past" else 3
    elif change == "gold":
        req["target"] = {"point": [180, 170]}
    elif change == "unknown":
        ids = [1, 99]
    elif change == "duplicate":
        ids = [1, 1]
    elif change == "boolean":
        ids = [True, 2]
    elif change == "missing_sha":
        req["scene"]["provenance"].pop("image_sha256")
    else:
        classes = np.zeros((200, 200), dtype=np.uint8)
    with pytest.raises(ValueError):
        surface.options_for(req, classes, ids)


@pytest.mark.parametrize("change", ["point", "box", "pixel", "wording", "track"])
def test_confirmation_replay_rejects_changed_point_surface_identity_or_wording(change):
    req, classes = request(), map_for()
    option = deepcopy(surface.options_for(req, classes, [1, 2])[0])
    if change == "point":
        option["point"][0] += 1
    elif change == "box":
        option["region_box_xyxy"][0] -= 1
    elif change == "pixel":
        classes[0, 0] = 0
    elif change == "wording":
        req["text"] += " near me"
    else:
        req["scene"]["tracks"][0]["bbox_xyxy"][0] += 1
    with pytest.raises(ValueError, match="differs"):
        surface.validate_option(req, classes, [1, 2], option)
