"""Controlled software photographs for measured roadside ground and its confounders."""

from hashlib import sha256
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from trackrag.dropoff import vehicle_curb_ground
from trackrag.dropoff import vehicle_rear_boundaries as geometry


def scene_with_edge(*, kind="dashes", mirrored=False, parking_class=True, optical=True):
    h, w = 230, 360
    labels = np.full((h, w), 13, np.uint8)
    rgb = np.full((h, w, 3), (60, 65, 70), np.uint8)
    for y in range(h):
        road_edge = round(192 + 0.86 * (y - 136))
        curb = round(251 + 1.21 * (y - 136))
        if road_edge < 4 or curb > w - 2:
            continue
        if parking_class:
            start = road_edge + 4 if kind == "dashes" else max(road_edge + 4, curb - 15)
            labels[y, start:curb] = 10
        labels[y, curb : curb + 4] = 2
        labels[y, curb + 4 :] = 15
        if optical:
            rgb[y, curb:] = (168, 157, 142)
        if kind == "paving":
            rgb[y, road_edge:curb] = (128, 119, 105)
            rgb[y, road_edge : road_edge + 2] = (81, 80, 76)
        if kind == "dashes":
            labels[y, road_edge : road_edge + 3] = 24
            if y % 15 < 8:
                rgb[y, road_edge : road_edge + 3] = (230, 231, 223)
    box = [200, 88, 244, 136]
    labels[88:136, 200:244] = 55
    rgb[88:136, 200:244] = (76, 81, 96)
    if mirrored:
        labels = np.fliplr(labels).copy()
        rgb = np.fliplr(rgb).copy()
        box = [w - box[2], box[1], w - box[0], box[3]]
    original = Image.fromarray(rgb)
    encoded = BytesIO()
    original.save(encoded, format="PNG")
    scene = {
        "source": "predicted",
        "frame": {"recording_id": "software-only", "frame_id": 3, "width": w, "height": h},
        "tracks": [{"track_id": 7, "frame_id": 3, "class_name": "car", "bbox_xyxy": box[:]}],
        "provenance": {"image_sha256": sha256(encoded.getvalue()).hexdigest()},
    }
    return original, labels, scene, {"track_id": 7, "class": "car", "box": list(map(float, box))}


@pytest.mark.parametrize("kind", ["dashes", "paving"])
@pytest.mark.parametrize("mirrored", [False, True])
def test_visible_ground_follows_its_own_boundary(kind, mirrored):
    photo, labels, scene, spec = scene_with_edge(kind=kind, mirrored=mirrored)
    options = geometry.candidates(photo, labels, scene, spec, facing="REAR")
    assert len(options) == 1
    outcome = options[0]
    x, y = outcome["point"]
    assert y > spec["box"][3] and labels[int(y), int(x)] in (10, 13)
    evidence = outcome["evidence"]
    assert evidence["curb_side"] == ("left" if mirrored else "right")
    assert evidence["coverage"] == "interior_point_only" and evidence["current_tracks_clear"]
    assert (
        evidence["physical_boundary"]
        == {
            "paving": "optical_paving_seam_and_observed_kerb",
            "dashes": "repeated_white_paint_and_observed_kerb",
        }[kind]
    )
    data = BytesIO()
    photo.save(data, format="PNG")
    assert geometry.candidates(data.getvalue(), labels, scene, spec, facing="REAR") == options
    assert not geometry.candidates(data.getvalue() + b"changed", labels, scene, spec, facing="REAR")


@pytest.mark.parametrize("kind", ["dashes", "paving"])
@pytest.mark.parametrize(
    "problem",
    [
        "front",
        "unknown",
        "wrong_box",
        "wrong_frame",
        "occupied",
        "no_kerb",
        "unsupported_lane",
        "no_optical_edge",
        "bicycle_lane",
    ],
)
def test_through_lane_or_mismatched_source_never_supplies_a_spot(kind, problem):
    photo, labels, scene, spec = scene_with_edge(kind=kind, parking_class=problem != "unsupported_lane")
    facing = "REAR"
    if problem in ("front", "unknown"):
        facing = problem.upper()
    elif problem == "wrong_box":
        spec["box"][0] -= 3
    elif problem == "wrong_frame":
        scene["tracks"][0]["frame_id"] -= 1
    elif problem == "occupied":
        scene["tracks"].append(
            {"frame_id": 3, "track_id": 8, "class_name": "truck", "bbox_xyxy": [185, 137, 340, 206]}
        )
    elif problem == "no_kerb":
        labels[np.isin(labels, (2, 15))] = 13
    elif problem == "no_optical_edge":
        photo = Image.new("RGB", photo.size, (107, 107, 107))
    elif problem == "bicycle_lane":
        labels[138:190, 185:325] = 8
    assert not geometry.candidates(photo, labels, scene, spec, facing=facing)


def test_continuous_white_line_is_insufficient_to_claim_dashed_parking():
    photo, classes, scene, spec = scene_with_edge()
    pixels = np.array(photo)
    pixels[classes == 24] = (235, 235, 224)
    result = geometry.candidates(Image.fromarray(pixels), classes, scene, spec, facing="REAR")
    assert all(
        o["evidence"]["physical_boundary"] != "repeated_white_paint_and_observed_kerb" for o in result
    )


def split_parking_strip(*, mirrored=False):
    photo, classes, scene, _ = scene_with_edge(mirrored=mirrored)
    pixels = np.array(photo)
    pixels[classes == 24] = (230, 231, 223)
    # Make the lane-side half of the strip read as road, including its midpoint.
    for row in range(137, 194):
        lane = round(192 + 0.86 * (row - 136))
        curb = round(251 + 1.21 * (row - 136))
        division = round(lane + 0.56 * (curb - lane))
        stop = classes.shape[1] - division
        cells = classes[row, stop:] if mirrored else classes[row, :division]
        cells[cells == 10] = 13
    spec, = vehicle_curb_ground.describe(scene, classes)
    return Image.fromarray(pixels), classes, scene, spec


@pytest.mark.parametrize("mirrored", [False, True])
def test_marked_strip_uses_a_parking_patch_away_from_the_lane(mirrored):
    photo, classes, scene, spec = split_parking_strip(mirrored=mirrored)
    midpoint_x, midpoint_y = map(int, spec["ground"]["point"])
    assert classes[midpoint_y, midpoint_x] == 13
    proposal, = geometry.marked_parking_point(photo, classes, scene, spec, facing="REAR")
    x, y = map(int, proposal["point"])
    assert y > spec["box"][3]
    assert np.all(classes[y - 2 : y + 3, x - 2 : x + 3] == 10)
    assert proposal["evidence"]["coverage"] == "interior_point_only"
    assert proposal["evidence"]["curb_side"] == ("left" if mirrored else "right")


@pytest.mark.parametrize("missing", ["visible_boundaries", "parking", "clear_ground", "rear"])
def test_marked_strip_requires_current_clear_parking_and_visible_boundaries(missing):
    photo, classes, scene, spec = split_parking_strip()
    facing = "REAR"
    if missing == "visible_boundaries":
        photo = Image.new("RGB", photo.size, (65, 65, 65))
    elif missing == "parking":
        classes[classes == 10] = 13
    elif missing == "clear_ground":
        scene["tracks"].append(
            {"frame_id": 3, "track_id": 8, "class_name": "truck", "bbox_xyxy": [185, 137, 340, 206]}
        )
    else:
        facing = "FRONT"
    assert geometry.marked_parking_point(photo, classes, scene, spec, facing=facing) == []


@pytest.mark.parametrize("flat_segments", [False, True])
def test_both_opencv_line_segment_shapes_replay_identical_ground(monkeypatch, flat_segments):
    import cv2

    photo, labels, scene, spec = scene_with_edge(kind="paving")
    expected = geometry.candidates(photo, labels, scene, spec, facing="REAR")
    assert (
        expected and expected[0]["evidence"]["physical_boundary"] == "optical_paving_seam_and_observed_kerb"
    )
    factory = cv2.createLineSegmentDetector

    class Detector:
        def __init__(self, *args):
            self.native = factory(*args)

        def detect(self, gray):
            lines, *others = self.native.detect(gray)
            if lines is not None:
                shape = (-1, 4) if flat_segments else (-1, 1, 4)
                lines = np.asarray(lines).reshape(shape)
            return (lines, *others)

    monkeypatch.setattr(cv2, "createLineSegmentDetector", Detector)
    assert geometry.candidates(photo, labels, scene, spec, facing="REAR") == expected
