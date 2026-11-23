import json

import cv2
import numpy as np
from PIL import Image

from trackrag.dropoff import painted_bays as bays


def stall_row(*, semantic_paint=True, curb=True, boxes=()):
    width, height = 360, 220
    classes = np.full((height, width), 13, dtype="uint8")
    rgb = np.full((height, width, 3), 90, dtype="uint8")
    # A visible row of two angled bays beside the sidewalk. The surrounding aisle is road.
    cv2.fillConvexPoly(classes, np.int32([[64, 173], [109, 64], [249, 64], [252, 190]]), 10)
    if curb:
        classes[:63] = 15
        classes[63:66] = 2
    for a, b in [((69, 171), (116, 67)), ((157, 180), (178, 67)), ((245, 188), (240, 67))]:
        cv2.line(rgb, a, b, (235, 232, 223), 3, cv2.LINE_8)
        if semantic_paint:
            cv2.line(classes, a, b, 24, 3, cv2.LINE_8)
    return Image.fromarray(rgb), classes, {"size": [width, height], "current_boxes": list(boxes)}


def matches(options, classes, p, scale=1, half_size=1):
    frame = {"width": classes.shape[1] * scale, "height": classes.shape[0] * scale}
    x, y = p
    mark = [x - half_size, y - half_size, x + half_size, y + half_size]
    return bays.matching_options(options, mark, p, frame, classes)


def test_observed_trapezoids_interiors_and_shared_paint_are_distinct():
    im, classes, ctx = stall_row()
    options = bays.candidates(im, classes, ctx)
    assert len(options) == 2
    assert json.loads(json.dumps(options)) == options
    for option in options:
        polygon = np.float32(option["polygon"])
        left, top, right, bottom = option["box"]
        assert 0 < cv2.contourArea(polygon) < .99 * (right - left) * (bottom - top)
        assert cv2.pointPolygonTest(polygon, tuple(option["point"]), True) > 8
        assert len(option["evidence"]["painted_boundaries"]) == 2
        assert option["evidence"]["visible_parking_cells"] > 30
        assert matches(options, classes, option["point"]) == [option]
    assert matches(options, classes, [167.5, 123.5]) == options
    frame = {"width": im.width, "height": im.height}
    assert bays.match_mark(options, [166.5, 122.5, 168.5, 124.5], [167.5, 123.5], frame, classes) is None
    assert not matches(options, classes, [73, 76])  # in the axis-aligned extent, outside the angled space
    assert not matches(options, classes, options[0]["point"], half_size=12)


def test_coordinates_scale_to_original_and_no_instruction_or_scene_identity_is_needed():
    im, classes, ctx = stall_row()
    first = bays.candidates(im, classes, ctx)
    enlarged = im.resize((720, 440), Image.Resampling.NEAREST)
    enlarged_ctx = {"size": list(enlarged.size), "current_boxes": [], "words": "unrelated sender"}
    second = bays.candidates(enlarged, classes, enlarged_ctx)
    assert len(first) == len(second) == 2
    for a, b in zip(first, second):
        assert np.allclose(np.array(a["polygon"]) * 2, b["polygon"], atol=.002)
    assert matches(second, classes, [335, 247], scale=2) == second


def test_visible_paint_can_recover_a_semantically_unlabelled_divider():
    im, classes, ctx = stall_row(semantic_paint=False)
    options = bays.candidates(im, classes, ctx)
    assert len(options) == 2
    assert all(option["evidence"]["visible_parking_cells"] for option in options)


def test_no_automatic_outline_from_class_map_alone_one_separator_or_no_curb():
    im, classes, ctx = stall_row()
    assert not bays.candidates(Image.new("RGB", im.size, "#626262"), classes, ctx)
    im, classes, ctx = stall_row(curb=False)
    assert not bays.candidates(im, classes, ctx)
    im, classes, ctx = stall_row()
    rgb = np.array(im)
    rgb[:, 150:] = 90
    classes[(np.indices(classes.shape)[1] >= 150) & (classes == 24)] = 10
    assert not bays.candidates(Image.fromarray(rgb), classes, ctx)


def test_occupied_stall_and_unseen_other_side_never_become_unambiguous_points():
    im, classes, ctx = stall_row(boxes=[[100, 82, 152, 165]])
    options = bays.candidates(im, classes, ctx)
    assert len(options) == 1 and options[0]["point"][0] > 170
    assert not matches(options, classes, [125, 121])
    assert not matches(options, classes, [240, 70])  # outer stripe has no verified closed side here
    ctx["current_boxes"].append([182, 83, 230, 167])
    assert not bays.candidates(im, classes, ctx)


def test_small_semantic_obstruction_inside_stall_survives_absent_detector_box():
    im, classes, ctx = stall_row()
    options = bays.candidates(im, classes, ctx)
    x, y = np.rint(options[0]["point"]).astype(int)
    classes[y + 6:y + 10, x - 10:x - 7] = 19
    remaining = bays.candidates(im, classes, ctx)
    assert len(remaining) == 1 and remaining[0]["point"] == options[1]["point"]


def test_single_side_of_observed_paint_is_only_available_as_an_explicit_suggestion():
    im, classes, ctx = stall_row()
    first = bays.candidates(im, classes, ctx)[:1]
    center = [167.5, 123.5]
    mark = [166.5, 122.5, 168.5, 124.5]
    frame = {"width":im.width, "height":im.height}
    assert bays.match_mark(first, mark, center, frame, classes) is None
    assert not bays.matching_options(first, mark, center, frame, classes)
    assert bays.matching_options(first, mark, center, frame, classes, offer_visible_side=True) == first
    assert not bays.matching_options(first, [46, 193, 48, 195], [47,194], frame, classes, offer_visible_side=True)
    assert not bays.matching_options(first, [142, 100, 193, 147], center, frame, classes, offer_visible_side=True)
    classes[122:126, 166:169] = 55
    assert not bays.matching_options(first, mark, center, frame, classes, offer_visible_side=True)


def test_road_lines_rectangular_zebra_and_parking_outside_the_polygon_are_rejected():
    im, classes, ctx = stall_row()
    classes[classes == 10] = 13
    classes[70:90, 15:45] = 10
    assert not bays.candidates(im, classes, ctx)

    im, classes, ctx = stall_row()
    classes[112:121, 65:260] = 23
    assert not bays.candidates(im, classes, ctx)
    road = np.full_like(classes, 13)
    road[:45] = 15
    road[45:49] = 2
    road[50:112, 20:340] = 10
    pixels = np.full((*road.shape, 3), 78, dtype="uint8")
    for y in (55, 73, 91):
        cv2.line(road, (25, y), (335, y), 24, 3)
        cv2.line(pixels, (25, y), (335, y), (240, 240, 230), 3)
    assert not bays.candidates(Image.fromarray(pixels), road, ctx)  # longitudinal through-lane boundaries
    road[50:] = 10
    pixels[:] = 78
    for x in range(75, 273, 24):
        cv2.line(road, (x, 52), (x, 170), 24, 10)
        cv2.line(pixels, (x, 52), (x, 170), (240, 240, 230), 10)
    assert not bays.candidates(Image.fromarray(pixels), road, ctx)


def test_faint_observed_kerb_facing_parking_row_can_recover_an_unclassified_interior():
    im, classes, ctx = stall_row()
    classes[classes == 10] = 13
    rgb = np.asarray(im).copy()
    for x, y in [(69, 171), (157, 180), (245, 188)]:
        cv2.line(classes, (x - 8, y - 1), (x + 8, y + 1), 24, 3)
        cv2.line(rgb, (x - 8, y - 1), (x + 8, y + 1), (235, 232, 223), 3)
    # Retain modest real optical contrast while the signed surface model labels the asphalt road.
    rgb = np.rint(55. + (rgb.astype(float) - 90.) * .18).clip(0, 255).astype("uint8")
    options = bays.candidates(Image.fromarray(rgb), classes, ctx)
    assert len(options) == 2
    for o in options:
        evidence = o["evidence"]
        assert evidence["recipe"] == bays.ROW_RECIPE and evidence["visible_parking_cells"] == 0
        assert evidence["supporting_row_dividers"] and evidence["painted_aisle_endcaps"]
        assert evidence["curb_facing_terminations"] == 2
        assert matches(options, classes, o["point"]) == [o]
    # The location of a road pixel without the physical row cannot establish a parking stall.
    classes[classes == 24] = 13
    assert not bays.candidates(Image.fromarray(rgb), classes, ctx)
