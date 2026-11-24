"""Software fixtures for a single-car, visually separated parking apron."""

from hashlib import sha256
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from trackrag.dropoff import vehicle_rear_apron as geometry


def apron():
    h, w = 180, 300
    lab = np.full((h, w), 13, dtype=np.uint8)
    rgb = np.full((h, w, 3), (51, 55, 59), dtype=np.uint8)
    # Genuine off-lane frontage apron with a curb at the building and a material break at the road.
    lab[69:74, :] = 2
    lab[:69, :] = 15
    lab[74:125, :] = 10
    rgb[:74, :] = (201, 181, 159)
    rgb[74:125, :] = (142, 127, 105)
    b = [123, 43, 167, 93]
    lab[b[1] : b[3], b[0] : b[2]] = 55
    rgb[b[1] : b[3], b[0] : b[2]] = (79, 95, 109)
    scene = {
        "frame": {"width": w, "height": h, "frame_id": 4, "recording_id": "software-only"},
        "source": "predicted",
        "provenance": {"image_sha256": "a" * 64},
        "tracks": [{"frame_id": 4, "track_id": 2, "class_name": "car", "bbox_xyxy": list(b)}],
    }
    image = Image.fromarray(rgb)
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    scene["provenance"]["image_sha256"] = sha256(encoded.getvalue()).hexdigest()
    return image, lab, scene, {"track_id": 2, "class": "car", "box": list(map(float, b))}


def test_alone_genuine_parking_center_does_not_use_a_ground_inference():
    rgb, labels, scene, anchor = apron()
    options = geometry.candidates(rgb, labels, scene, anchor, facing="REAR")
    assert len(options) == 1
    opt = options[0]
    x, y = opt["point"]
    assert anchor["box"][0] <= x <= anchor["box"][2] and anchor["box"][3] < y < 125
    assert opt["evidence"]["kerb_rays"] or opt["evidence"]["flank_kerb"]
    assert opt["evidence"]["road_arc"] >= 3
    assert opt["evidence"]["point_surface"] == 10 and opt["evidence"]["parking_connection"] >= 0.90
    b = BytesIO()
    rgb.save(b, format="PNG")
    assert options == geometry.candidates(b.getvalue(), labels, scene, anchor, facing="REAR")
    assert not geometry.candidates(b.getvalue() + b"changed", labels, scene, anchor, facing="REAR")


@pytest.mark.parametrize(
    "neg",
    [
        "front",
        "side",
        "unknown",
        "wrong_box",
        "stale",
        "lane",
        "disconnected",
        "aisle",
        "no_road_edge",
        "unseen_boundary",
        "occluded",
        "chain_over_parking",
        "straight_rail",
        "cycle_lane",
        "sidewalk",
        "moving_anchor",
        "travel_arrow",
    ],
)
def test_an_unscored_identity_ground_cannot_bypass_geometry(neg):
    rgb, labels, scene, anchor = apron()
    facing = "REAR"
    if neg in ("front", "side", "unknown"):
        facing = neg.upper()
    elif neg == "wrong_box":
        anchor["box"][0] += 1
    elif neg == "stale":
        scene["tracks"][0]["frame_id"] += 1
    elif neg == "lane":
        labels[labels == 10] = 13
    elif neg == "disconnected":
        labels[123:129, :] = 2
    elif neg == "aisle":
        labels[(labels == 15) | (labels == 2)] = 13
    elif neg == "no_road_edge":
        labels[125:, :] = 15
    elif neg == "unseen_boundary":
        rgb = Image.new("RGB", rgb.size, (100, 100, 100))
    elif neg == "occluded":
        scene["tracks"].append(
            {"frame_id": 4, "class_name": "truck", "track_id": 7, "bbox_xyxy": [125, 94, 169, 127]}
        )
    elif neg == "chain_over_parking":
        from PIL import ImageDraw

        draw = ImageDraw.Draw(rgb)
        draw.line([(120, 100), (142, 109), (154, 109), (171, 100)], fill=(10, 10, 10), width=2)
    elif neg == "straight_rail":
        from PIL import ImageDraw

        ImageDraw.Draw(rgb).line([(120, 105), (173, 105)], fill=(10, 10, 10), width=2)
    elif neg == "cycle_lane":
        labels[96:129, :] = 8
    elif neg == "sidewalk":
        labels[96:129, :] = 15
    elif neg == "moving_anchor":
        scene["tracks"].append(
            {"frame_id": 3, "class_name": "car", "track_id": 2, "bbox_xyxy": [100, 43, 144, 93]}
        )
    elif neg == "travel_arrow":
        from PIL import ImageDraw

        corners = [(170, 96), (177, 107), (173, 107), (173, 121), (168, 121), (168, 107), (163, 107)]
        arrow = Image.new("L", rgb.size)
        ImageDraw.Draw(arrow).polygon(corners, fill=1)
        labels[np.asarray(arrow, dtype=bool)] = 24
        ImageDraw.Draw(rgb).polygon(corners, fill=(247, 247, 247))
    assert not geometry.candidates(rgb, labels, scene, anchor, facing=facing)
