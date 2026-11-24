"""Association and clearance checks for a photographed, transversely marked parking bay."""

from hashlib import sha256
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest

from trackrag.dropoff import vehicle_rear_stall as stalls


def marked_source(mirrored=False):
    w, h = 600, 400
    labels = np.full((h, w), 13, np.uint8)
    image = Image.new("RGB", (w, h), (64, 69, 72))
    draw = ImageDraw.Draw(image)
    dividers = [
        ((150, 204), (275, 198)),
        ((188, 233), (318, 230)),
        ((230, 270), (423, 258)),
        ((280, 312), (580, 290)),
    ]
    mouth = [a for a, b in dividers]
    curb = [b for a, b in dividers]
    full = np.int32([*mouth, *reversed(curb)])
    cv2.fillConvexPoly(labels, full, 10)
    cv2.polylines(labels, [np.int32(curb)], False, 2, 5)
    draw.polygon([tuple(v) for v in full], fill=(124, 123, 117))
    for a, b in dividers:
        draw.line([a, b], fill=(240, 237, 228), width=4)
    box = [182, 154, 294, 227]
    labels[box[1] : box[3], box[0] : box[2]] = 55
    draw.rectangle(tuple(box), fill=(51, 56, 64))
    if mirrored:
        labels = np.fliplr(labels).copy()
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        box = [w - box[2], box[1], w - box[0], box[3]]
        dividers = [((w - 1 - a[0], a[1]), (w - 1 - b[0], b[1])) for a, b in dividers]
    data = BytesIO()
    image.save(data, format="PNG")
    sc = {
        "source": "predicted",
        "provenance": {"image_sha256": sha256(data.getvalue()).hexdigest()},
        "frame": {"recording_id": "software-only", "frame_id": 2, "width": w, "height": h},
        "tracks": [{"track_id": 1, "frame_id": 2, "class_name": "car", "bbox_xyxy": box[:]}],
    }
    selected = {"box": list(map(float, box)), "track_id": 1, "class": "car"}
    return image, labels, sc, selected, [(np.array(a), np.array(b)) for a, b in dividers[1:3]]


@pytest.mark.parametrize("mirrored", [False, True])
def test_uses_measured_cross_bars_and_renders_only_clear_original_pixels(mirrored):
    im, classes, sc, spec, _ = marked_source(mirrored)
    locations = stalls.candidates(im, classes, sc, spec, facing="REAR")
    assert len(locations) == 1
    one = locations[0]
    x, y = one["point"]
    assert y > spec["box"][3] and classes[int(y), int(x)] == 10
    assert one["evidence"]["curb_side"] == ("left" if mirrored else "right")
    assert one["evidence"]["coverage"] == "interior_point_only"
    assert len(one["evidence"]["source_dividers_xy"]) == 2
    buf = BytesIO()
    im.save(buf, format="PNG")
    assert stalls.candidates(buf.getvalue(), classes, sc, spec, facing="REAR") == locations
    assert not stalls.candidates(buf.getvalue() + b"x", classes, sc, spec, facing="REAR")


@pytest.mark.parametrize(
    "failure",
    [
        "ahead",
        "side_aspect",
        "wrong_row",
        "unrelated_behind",
        "only_one",
        "parallel_identical",
        "wrong_box",
        "travel_lane",
        "obstruction",
        "across_kerb",
        "sidewalk",
        "source_frame",
    ],
)
def test_measured_paint_on_the_wrong_ground_or_relative_to_the_wrong_vehicle_is_rejected(
    monkeypatch, failure
):
    im, classes, sc, spec, borders = marked_source()
    facing = "REAR"
    if failure == "ahead":
        borders = [(a - [0, 77], b - [0, 77]) for a, b in borders]
    elif failure == "side_aspect":
        facing = "SIDE"
    elif failure == "wrong_row":
        borders = [(a + [180, 0], b + [180, 0]) for a, b in borders]
    elif failure == "unrelated_behind":
        borders = [(a + [0, 65], b + [0, 65]) for a, b in borders]
    elif failure == "only_one":
        borders = borders[:1]
    elif failure == "parallel_identical":
        borders = [borders[0], borders[0]]
    elif failure == "wrong_box":
        spec["box"][0] += 13
    elif failure == "obstruction":
        sc["tracks"].append(
            {"frame_id": 2, "track_id": 5, "class_name": "car", "bbox_xyxy": [223, 233, 355, 268]}
        )
    elif failure == "across_kerb":
        classes[:, 160:166] = 2
        for y in range(233, 270):
            x = round(188 + (y - 233) * 1.14)
            classes[y, x - 6 : x + 3] = 2
        classes[226:233, 190:405] = 2
        classes[268:278, 130:600] = 2
        classes[231:267, 400:] = 2
    elif failure == "travel_lane":
        classes[classes == 10] = 13
    elif failure == "sidewalk":
        classes[classes == 10] = 15
    elif failure == "source_frame":
        sc["tracks"][0]["frame_id"] = 1
    monkeypatch.setattr(stalls, "_borders", lambda *_: borders)
    assert not stalls.candidates(im, classes, sc, spec, facing=facing)


def test_two_yellow_crossbars_are_not_evidence_for_a_white_marked_bay():
    image, labels, scene, spec, _ = marked_source()
    pixels = np.array(image)
    paint = np.min(pixels, axis=2) > 185
    pixels[paint] = (238, 177, 18)
    assert not stalls.candidates(Image.fromarray(pixels), labels, scene, spec, facing="REAR")
