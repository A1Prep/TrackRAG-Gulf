import numpy as np
import pytest
from PIL import Image

from trackrag.dropoff import gravel_pullin


def frontage(*, soil=(182, 168, 147), detail=True, road_rgb=(95, 96, 99), scale=1):
    height, width = 400, 600
    labels = np.full((height, width), 17, dtype="uint8")
    labels[162:278] = 26
    labels[278:] = 13
    rng = np.random.default_rng(432)
    rgb = np.full((height, width, 3), [170, 164, 160], dtype="int16")
    rgb[278:] = road_rgb
    rgb[162:278] = soil
    if detail:
        # Vary lightness across the surface to provide resolved grains; green tests
        # receive the same grain distribution and must still abstain.
        rgb[162:278] += rng.integers(-12, 13, (116, width, 1), dtype="int16")
    boxes = [(70, 142, 140, 215), (250, 127, 335, 226), (391, 111, 485, 237)]
    for x0, y0, x1, y1 in boxes:
        labels[y0:y1, x0:x1] = 55
        rgb[y0:y1, x0:x1] = [50, 65, 80]
    im = Image.fromarray(np.clip(rgb, 0, 255).astype("uint8"))
    if scale != 1:
        im = im.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    native = [[v * scale for v in box] for box in boxes]
    return im, labels, {"size": list(im.size), "current_boxes": native,
                        "current_vehicles": [[i+1, "Car", b] for i, b in enumerate(native)]}


def test_visible_patch_between_observed_vehicles_needs_a_third_parked_row_member():
    im, labels, c = frontage()
    [opt] = gravel_pullin.candidates(im, labels, c)
    assert 140 < opt["point"][0] < 250
    assert 190 < opt["point"][1] < 256
    assert opt["evidence"]["flanking_track_ids"] == [1, 2]
    assert opt["evidence"]["additional_row_track_id"] == 3
    assert opt["evidence"]["extent"] == "interior_point_only" and "polygon" not in opt
    assert opt["evidence"]["observed_road_sections"] >= 10
    assert opt["evidence"]["terrain_to_road_color_distance"] > 15
    assert all(labels[round(y), round(x)] == 26 for x, y in opt["support_patch"])


def test_gray_granular_ground_can_qualify_independently_of_a_sand_colour():
    im, labels, c = frontage(soil=(176, 176, 173))
    assert len(gravel_pullin.candidates(im, labels, c)) == 1


@pytest.mark.parametrize("variant", ["highway", "median", "barrier", "paint", "bonnet", "no_terrain"])
def test_surface_and_adjacent_buildings_cannot_be_inferred_across_barriers(variant):
    im, labels, c = frontage()
    if variant == "highway":
        labels[labels == 17] = 27
    elif variant == "median":
        labels[142:162] = 13  # Another traffic lane between this island and the distant buildings.
    elif variant == "barrier":
        labels[277:283] = 2
    elif variant == "paint":
        labels[277:283] = 24
    elif variant == "bonnet":
        labels[278:300] = 64
    else:
        labels[labels == 26] = 13
    assert gravel_pullin.candidates(im, labels, c) == []


@pytest.mark.parametrize("visual", ["grass", "smooth_ground", "same_as_road"])
def test_plain_terrain_class_is_insufficient_even_with_parked_cars(visual):
    opts = {"grass": {"soil": (82, 154, 54)}, "smooth_ground": {"detail": False},
            "same_as_road": {"soil": (95, 96, 99)}}[visual]
    im, labels, c = frontage(**opts)
    assert gravel_pullin.candidates(im, labels, c) == []


def test_two_vehicles_or_a_duplicate_identity_never_establish_a_parking_row():
    im, labels, c = frontage()
    c["current_vehicles"].pop()
    assert gravel_pullin.candidates(im, labels, c) == []
    c["current_vehicles"].append([4, "Car", c["current_boxes"][0]])
    assert gravel_pullin.candidates(im, labels, c) == []


def test_the_third_vehicle_must_have_a_current_tyre_contact_on_the_same_unpaved_strip():
    im, labels, c = frontage()
    labels[236:267, 390:486] = 13
    assert gravel_pullin.candidates(im, labels, c) == []


def test_a_wide_tyre_window_cannot_look_across_a_kerb_under_all_three_vehicles():
    im, _, _ = frontage()
    labels = np.full((400, 600), 17, dtype="uint8")
    labels[190:280] = 26
    labels[280:] = 13
    boxes = [[40, 105, 120, 195], [225, 105, 305, 195], [340, 105, 420, 195]]
    for l, t, r, b in boxes:
        labels[t:b, l:r] = 55
    c = {"size": list(im.size), "current_boxes": boxes,
         "current_vehicles": [[i, "Car", box] for i, box in enumerate(boxes)]}
    assert len(gravel_pullin.candidates(im, labels, c)) == 1
    labels[185:205] = 2  # Terrain starts ten pixels beyond the lower edge of each box.
    assert gravel_pullin.candidates(im, labels, c) == []


@pytest.mark.parametrize("box", [[181, 173, 205, 238], [10, 280, 275, 326]])
def test_an_obstacle_on_the_patch_or_over_the_entrance_blocks_its_entire_candidate(box):
    im, labels, c = frontage()
    c["current_boxes"].append(box)
    assert gravel_pullin.candidates(im, labels, c) == []


def test_building_side_object_is_distinct_from_one_in_the_access_strip():
    im, labels, c = frontage()
    c["current_boxes"].append([165, 108, 184, 165])
    assert len(gravel_pullin.candidates(im, labels, c)) == 1


def test_stale_vehicles_and_wrong_image_dimensions_fail_closed():
    im, labels, c = frontage()
    assert gravel_pullin.candidates(im, labels, {**c, "current_boxes": []}) == []
    assert gravel_pullin.candidates(im, labels, {**c, "size": [1920, 1080]}) == []


def test_original_pixel_scale_is_preserved_for_pin_and_local_patch():
    first = gravel_pullin.candidates(*frontage())[0]
    large = gravel_pullin.candidates(*frontage(scale=2))[0]
    assert np.allclose(np.asarray(first["point"]) * 2 + .5, large["point"])
    assert np.allclose(np.asarray(first["support_patch"]) * 2 + .5, large["support_patch"])
