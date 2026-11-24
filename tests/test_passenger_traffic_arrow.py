import cv2
import numpy as np
import pytest
from PIL import Image

from trackrag.dropoff import traffic_arrow, vehicle_rear_ground as rear
from test_vehicle_rear_ground import parked_row


# Project a tip and head with a narrower shaft along the perspective of this
# synthetic road. The arrow lies beyond the original proposed curbside point.
FORWARD = [(262,169),(280,178),(274,178),(292,197),(284,198),(266,179),(258,184)]
TURN = [(268,168),(252,177),(268,185),(267,180),(273,179),
        (291,198),(298,197),(277,174),(266,174)]


def painted_scene(shape, *, offset=0, mirrored=False, pixels=True, semantic=True):
    im, classes, scene, anchor = parked_row()
    rgb = np.asarray(im).copy()
    polygon = np.int32(shape) + [offset, 0]
    if pixels:
        cv2.fillPoly(rgb, [polygon.astype(np.int32)], (231, 230, 225))
    if semantic:
        cv2.fillPoly(classes, [polygon.astype(np.int32)], 24)
    if mirrored:
        _, _, scene, anchor = parked_row(mirrored=True)
        rgb, classes = np.fliplr(rgb).copy(), np.fliplr(classes).copy()
    return rgb, classes, scene, anchor


def measured(rgb, classes, scene, anchor):
    return rear.candidates(Image.fromarray(rgb), classes, scene, anchor, facing="REAR")


@pytest.mark.parametrize("shape", [FORWARD, TURN])
@pytest.mark.parametrize("mirrored", [False, True])
def test_visible_travel_arrow_vetoes_an_otherwise_plausible_curb_strip(shape, mirrored):
    args = painted_scene(shape, mirrored=mirrored)
    assert measured(*args) == []


@pytest.mark.parametrize("shape,offset", [
    (FORWARD, -90),  # Valid arrow in the adjacent traffic lane.
    ([(250,184),(304,183),(304,185),(250,186)], 0),  # A transverse stall stripe.
    ([(261,181),(288,181),(288,183),(278,183),(278,194),(274,194),(274,183),(261,183)], 0),
])
def test_other_lane_and_parking_dividers_remain_eligible(shape, offset):
    assert measured(*painted_scene(shape, offset=offset))


def test_replayed_map_without_visible_arrow_or_unregistered_white_texture_cannot_veto():
    rgb, classes, _, anchor = painted_scene(FORWARD, pixels=False)
    sections = [rear.Section(y, round(198+.9*(y-136)), round(244+1.32*(y-136)), "paint", True)
                for y in range(139, 166)]
    # A native frame can be larger than its registered map.
    scaled_box = [3 * x for x in anchor["box"]]
    assert not traffic_arrow.in_measured_strip(rgb, classes, sections, scaled_box, (1020, 630), 1)
    rgb, classes, _, _ = painted_scene(FORWARD, semantic=False)
    assert not traffic_arrow.in_measured_strip(rgb, classes, sections, scaled_box, (1020, 630), 1)
    rgb, classes, _, _ = painted_scene(FORWARD)
    assert traffic_arrow.in_measured_strip(rgb, classes, sections, scaled_box, (1020, 630), 1)
