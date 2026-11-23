import numpy as np
import pytest
from PIL import Image

from trackrag.dropoff import curb_recess


def scene(*, flanks=((110, 142, 201, 238), (365, 142, 456, 238)), scale=1):
    image = Image.new("RGB", (600 * scale, 400 * scale))
    classes = np.full((400, 600), 17, dtype="uint8")
    classes[157:190] = 15
    classes[190:198] = 2
    classes[198:251] = 10
    classes[251:] = 13
    vehicles = []
    for track, (x0, y0, x1, y1) in enumerate(flanks, 10):
        classes[y0:y1, x0:x1] = 55
        box = [n * scale for n in (x0, y0, x1, y1)]
        vehicles.append([track, "Car", box])
    return image, classes, {"size": list(image.size), "current_vehicles": vehicles,
                            "current_boxes": [row[2] for row in vehicles]}


def test_repeated_observed_edges_and_both_parked_flanks_bind_an_interior_point():
    image, classes, context = scene()
    [offer] = curb_recess.candidates(image, classes, context)
    x, y = offer["point"]
    assert 240 < x < 325 and 206 < y < 241 and classes[int(y), int(x)] == 10
    assert "polygon" not in offer
    assert offer["evidence"]["extent"] == "interior_point_only"
    assert offer["evidence"]["flanking_track_ids"] == [10, 11]
    assert offer["evidence"]["observed_road_sections"] >= 25
    assert offer["evidence"]["both_vehicle_contacts_verified"]
    for (x0, y0), (x1, y1) in offer["evidence"]["kerb_and_aisle_sections"]:
        assert x0 == x1 and classes[int(y0) - 1, int(x0)] == 2
        assert classes[int(y1) + 1, int(x1)] == 13
    for px, py in offer["support_patch"]:
        assert classes[round(py), round(px)] == 10


@pytest.mark.parametrize("replacement", [13, 15, 24, 26, 35])
def test_no_point_on_travel_lane_sidewalk_marking_gravel_or_shoulder(replacement):
    image, classes, context = scene()
    classes[classes == 10] = replacement
    assert curb_recess.candidates(image, classes, context) == []


def test_curb_repeated_along_gap_is_required_even_when_a_road_exists():
    image, classes, context = scene()
    classes[classes == 2] = 15
    assert curb_recess.candidates(image, classes, context) == []


def test_unseen_or_divided_access_cannot_be_inferred_from_a_signed_parking_patch():
    image, classes, context = scene()
    classes[251:270] = 2
    assert curb_recess.candidates(image, classes, context) == []
    classes[251:270] = 64
    assert curb_recess.candidates(image, classes, context) == []


def test_signed_bonnet_occlusion_away_from_the_offered_point_permits_partial_evidence():
    image, classes, context = scene()
    classes[251:, 320:] = 64
    [offer] = curb_recess.candidates(image, classes, context)
    assert offer["point"][0] < 320
    assert offer["evidence"]["observed_road_sections"] < offer["evidence"]["observed_sections"]


def test_signed_detections_both_belong_to_this_same_parking_strip():
    image, classes, context = scene()
    classes[235:260, 365:457] = 13
    assert curb_recess.candidates(image, classes, context) == []


def test_sub_car_width_frontal_interval_rejected_even_if_its_point_is_fully_open():
    # Independent reviewer reproduced a false offer on approximately this 43-pixel
    # interval between two face-on cars, each 110 pixels wide at the same depth.
    image, labels, context = scene(flanks=((100, 141, 210, 238), (253, 141, 363, 238)))
    assert curb_recess.candidates(image, labels, context) == []


@pytest.mark.parametrize("blocker", [
    [255, 125, 270, 237],       # Person standing in the parking strip.
    [5, 236, 395, 320],         # Passing vehicle crossing the strip, centroid outside it.
    [5, 252, 395, 319],         # Aisle entrance hidden behind a bus while the patch is visible.
])
def test_current_obstacle_can_occlude_the_gap_or_its_entrance(blocker):
    image, classes, context = scene()
    context["current_boxes"].append(blocker)
    assert curb_recess.candidates(image, classes, context) == []


def test_pedestrian_behind_kerb_is_distinct_from_an_occupied_gap():
    image, classes, context = scene()
    context["current_boxes"].append([250, 120, 264, 192])
    assert len(curb_recess.candidates(image, classes, context)) == 1


def test_identities_current_boxes_and_original_dimensions_are_all_required():
    image, classes, context = scene()
    no_roles = {k: v for k, v in context.items() if k != "current_vehicles"}
    assert curb_recess.candidates(image, classes, no_roles) == []
    wrong_role = {**context, "current_vehicles": [[10, "Person", context["current_boxes"][0]],
                                                 context["current_vehicles"][1]]}
    assert curb_recess.candidates(image, classes, wrong_role) == []
    assert curb_recess.candidates(image, classes, {**context, "current_boxes": []}) == []
    assert curb_recess.candidates(image, classes, {**context, "size": [900, 800]}) == []


def test_point_and_patch_respect_original_resolution_and_current_box_order():
    small = curb_recess.candidates(*scene())[0]
    im, cls, c = scene(scale=2)
    c["current_vehicles"].reverse()
    c["current_boxes"].reverse()
    [large] = curb_recess.candidates(im, cls, c)
    # Half-pixel centres are retained on nearest-neighbour maps of different sizes.
    assert np.allclose(np.array(small["point"]) * 2 + .5, large["point"])
    assert np.allclose(np.array(small["support_patch"]) * 2 + .5, large["support_patch"])
    assert large["evidence"]["flanking_track_ids"] == small["evidence"]["flanking_track_ids"]
