import numpy as np
import pytest

from tools.dropoff_ground_candidates import build_candidates


def test_patches_are_wholly_predicted_road_and_exclude_vehicle_boxes():
    classes = np.full((120, 180), 2, dtype=np.uint8)
    classes[35:115, 5:175] = 0
    classes[60:90, 40:65] = 13  # semantic car hole
    probability = np.ones(classes.shape, dtype=np.float32)
    candidates, ground = build_candidates(classes, probability, [[100, 40, 140, 100]])
    assert candidates
    assert len(candidates) <= 36
    assert not ground[40:100, 100:140].any()
    for row in candidates:
        x1, y1, x2, y2 = row["bbox_original_px"]
        assert ground[y1:y2, x1:x2].all()
        x, y = row["point_original_px"]
        assert x1 <= x < x2 and y1 <= y < y2
        assert ground[int(y), int(x)]


def test_no_ground_returns_no_candidates_without_fallback():
    candidates, ground = build_candidates(np.ones((40, 50)), np.ones((40, 50)))
    assert candidates == []
    assert not ground.any()


def test_prediction_confidence_removes_road_without_replacing_it():
    candidates, ground = build_candidates(np.zeros((60, 60)), np.zeros((60, 60)))
    assert candidates == []
    assert not ground.any()


def test_candidate_ids_and_points_are_deterministic_and_original_pixel_scale():
    classes = np.zeros((120, 180), dtype=np.uint8)
    first, _ = build_candidates(classes, np.ones(classes.shape))
    second, _ = build_candidates(classes, np.ones(classes.shape))
    assert first == second
    assert len({row["candidate_id"] for row in first}) == 36
    for row in first:
        x, y = row["point_original_px"]
        assert row["point_normalized"] == [x / 180, y / 120]


@pytest.mark.parametrize("probability", [np.full((2, 2), np.nan), np.full((2, 2), 2)])
def test_invalid_probability_rejected(probability):
    with pytest.raises(ValueError):
        build_candidates(np.zeros((2, 2)), probability)


def test_reversed_exclusion_rejected():
    with pytest.raises(ValueError):
        build_candidates(np.zeros((60, 60)), np.ones((60, 60)), [[20, 20, 10, 30]])


def test_wholly_outside_boxes_do_not_remove_opposite_edge_pixels():
    classes = np.zeros((60, 60))
    baseline, mask = build_candidates(classes, np.ones((60, 60)))
    candidates, excluded = build_candidates(classes, np.ones((60, 60)), [[-20, 0, -5, 60], [0, -20, 60, -5]])
    assert candidates == baseline
    assert np.array_equal(mask, excluded)
