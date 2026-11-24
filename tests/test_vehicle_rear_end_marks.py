"""Tests of the visible parking-end junctions and the short clear ground before a bonnet."""

import cv2
import numpy as np
from PIL import Image
import pytest

from trackrag.dropoff import vehicle_rear_stall as stalls


def short_parking_row(
    *, mirrored=False, marks=3, no_kerb=False, plain_dashes=False, shift=0, object_behind=False, hood=True
):
    w, h = 830, 500
    labels = np.full((h, w), 13, np.uint8)
    rgb = np.full((h, w, 3), (60, 63, 65), np.uint8)
    cx, bt, bw, bh = 540, 335, 180, 110
    lane = cx - 0.38 * bw
    for y in range(h):
        edge = round(cx + 0.56 * bw + 3.1 * (y - bt))
        if not no_kerb:
            if edge < w:
                labels[y, max(0, edge) : min(w, edge + 8)] = 2
                labels[y, max(0, edge + 8) :] = 15
                rgb[y, max(0, edge) : min(w, edge + 8)] = (176, 172, 157)
                rgb[y, max(0, edge + 8) :] = (145, 130, 119)
    for index, dy in enumerate((-0.73, -0.55, -0.29)[:marks]):
        y = round(bt + dy * bh)
        x = round(lane + 1.65 * (y - bt)) + (shift if index == 1 else 0)
        span = (21, 27, 36)[index]
        cv2.line(rgb, (x, y), (x + span, y + 1), (192, 191, 183), 2, cv2.LINE_AA)
        if not plain_dashes:
            half = (4, 6, 8)[index]
            cv2.line(
                rgb,
                (x - half, y - round(0.55 * half)),
                (x + half, y + round(0.55 * half)),
                (228, 229, 223),
                3,
                cv2.LINE_AA,
            )
    box = [cx - bw // 2, bt - bh, cx + bw // 2, bt]
    labels[box[1] : box[3], box[0] : box[2]] = 55
    rgb[box[1] : box[3], box[0] : box[2]] = (103, 109, 122)
    if hood:
        for x in range(w):
            edge = round(bt + 0.21 * bh + 0.06 * (x - cx))
            labels[edge:, x] = 64
            rgb[edge:, x] = (139, 150, 165)
    if mirrored:
        labels = np.fliplr(labels).copy()
        rgb = np.fliplr(rgb).copy()
        box = [w - box[2], box[1], w - box[0], box[3]]
    tracks = [{"track_id": 2, "frame_id": 4, "class_name": "car", "bbox_xyxy": box[:]}]
    if object_behind:
        tracks.append(
            {"track_id": 3, "frame_id": 4, "class_name": "car", "bbox_xyxy": [490, 337, 750, 390]}
        )
    scene = {
        "source": "predicted",
        "provenance": {"image_sha256": "b" * 64},
        "tracks": tracks,
        "frame": {"width": w, "height": h, "recording_id": "software-only", "frame_id": 4},
    }
    return (
        Image.fromarray(rgb),
        labels,
        scene,
        {"track_id": 2, "class": "car", "box": list(map(float, box))},
    )


@pytest.mark.parametrize("mirrored", [False, True])
def test_three_actual_t_markers_establish_a_short_visible_ground_point(mirrored):
    img, labels, scene, spec = short_parking_row(mirrored=mirrored)
    result = stalls.candidates(img, labels, scene, spec, facing="REAR")
    assert len(result) == 1
    point = result[0]["point"]
    assert labels[int(point[1]), int(point[0])] == 13 and point[1] > spec["box"][3]
    evidence = result[0]["evidence"]
    assert (
        evidence["physical_boundary"] == "three_optical_parking_end_marks_and_observed_kerb_before_bonnet"
    )
    assert len(evidence["source_end_marks_xy"]) == 3
    assert evidence["curb_side"] == ("left" if mirrored else "right")


@pytest.mark.parametrize(
    "edit",
    [
        {"marks": 2},
        {"no_kerb": True},
        {"plain_dashes": True},
        {"shift": 40},
        {"object_behind": True},
        {"hood": False},
    ],
)
def test_a_short_hood_region_requires_independent_observed_kerb_and_repeated_parking_ends(edit):
    img, labels, scene, spec = short_parking_row(**edit)
    assert not stalls.candidates(img, labels, scene, spec, facing="REAR")


@pytest.mark.parametrize("rgb_colour", [(239, 175, 24), (215, 47, 35)])
def test_bright_nonwhite_junction_markings_do_not_certify_parking(rgb_colour):
    image, labels, scene, spec = short_parking_row()
    pixels = np.array(image)
    paint = (labels == 13) & (pixels[:, :, 0] > 105)
    pixels[paint] = rgb_colour
    assert not stalls.candidates(Image.fromarray(pixels), labels, scene, spec, facing="REAR")
