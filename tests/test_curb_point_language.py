from copy import deepcopy

import pytest

from trackrag.dropoff.curb_point_language import match_points


SIZE = (1000,600)


def option(point, near, far, frontage, roadside):
    return {"point":point, "descriptions":{"nearer_vehicle":near,"farther_vehicle":far,
                                         "frontage":frontage,"roadside":roadside}}


def scenes():
    left = option([206,434],"WHITE FLATBED TRUCK","a small red hatchback",
                  "Cedar Furniture Store","palm trees | a green waste bin")
    right = option([751,451],"a black sedan","charcoal SUV",
                   "Cedar Market","an iron fence")
    return left,right


@pytest.mark.parametrize("text", [
    "Could you drop me off outside Cedar Furniture in the gap between the white truck and the little red hatchback?",
    "I'd like to be dropped off in the open space between the red hatch and the white flatbed, by the furniture store.",
    "Let us out on the left-hand side of the road, next to the trees.",
    "The spot by the green trash can, next to the palms, would work for us.",
    "Drop me outside the furniture store by the white lorry.",
    "The gap right between the white and red vehicles works for me.",
])
def test_all_supported_attributes_apply_to_one_original_point(text):
    left,right = scenes()
    result = match_points(text,[right,left],SIZE)
    assert result==([left],[])
    assert result[0][0] is left


def test_distinct_vehicle_slots_and_same_option_for_every_clause():
    left,right = scenes()
    assert match_points("Drop me between the two dark cars on the right",[left,right],SIZE)==([right],[])
    assert match_points("Drop me in the spot by a white flatbed and outside Cedar Market",[left,right],SIZE)==([],[])
    one_dark = deepcopy(right)
    one_dark["descriptions"]["farther_vehicle"] = "NONE"
    assert match_points("Drop me between two dark cars",[one_dark],SIZE)==([],[])


@pytest.mark.parametrize("text", [
    "Drop me opposite Cedar Furniture.",
    "Drop me on the left of the red hatchback.",
    "Please stop behind the black sedan.",
    "Drop me in the gap by the fence but not by the black sedan.",
    "Drop me at a spot near the trees without a bin.",
    "Drop me in the space by the bin, past the palms.",
    "Any legally available spot by Cedar Furniture.",
    "Drop me in the marked bay near the fence.",
    "The nearest spot by the trees.",
    "Drop me by the green bin on the left and on the right.",
    "Drop me outside Cedar Storehouse.",
    "Drop me between the red vehicle with a roof rack and the white truck.",
    "The gap outside Cedar Market with a loading dock.",
    "Please use the blue striped curb space.",
    "Please stop in front of the white truck.",
    "Is that a white truck?",
])
def test_unknown_or_contradictory_restrictions_never_become_a_proposal(text):
    assert match_points(text,scenes(),SIZE)==([],[])


def test_explicit_category_or_colour_mismatch_prevents_any_offer():
    left,right = scenes()
    coarse = deepcopy(left)
    coarse["descriptions"]["farther_vehicle"] = "a red car"
    words = "Drop me in the gap between the white truck and the small red hatchback outside Cedar Furniture."
    assert match_points(words,[coarse],SIZE)==([],[coarse])
    coarse["descriptions"]["farther_vehicle"] = "red sedan"
    assert match_points(words,[coarse],SIZE)==([],[])
    coarse["descriptions"]["farther_vehicle"] = "large red hatchback"
    assert match_points(words,[coarse],SIZE)==([],[])
    coarse["descriptions"]["farther_vehicle"] = "red hatchback"
    assert match_points(words,[coarse],SIZE)==([],[coarse])
    coarse["descriptions"]["farther_vehicle"] = "small blue hatchback"
    assert match_points(words,[coarse],SIZE)==([],[])
    assert match_points("Drop me in the gap between two black cars",[right],SIZE)==([],[])


def test_second_word_in_a_named_place_matters_even_when_it_is_also_a_parking_noun():
    original = option([102,230],"red hatchback","white pickup","Ace Point shop","trees")
    assert match_points("Drop me in the gap outside Ace Spot",[original],SIZE)==([],[])
    assert match_points("Drop me outside Ace Point",[original],SIZE)==([original],[])
    original["descriptions"]["frontage"] = "Ace is maybe visible"
    assert match_points("Drop me outside Ace",[original],SIZE)==([],[])


def test_raw_roadside_landmark_cannot_establish_a_business_name_or_colour_from_elsewhere():
    left,_ = scenes()
    assert match_points("Drop me outside the green trees",[left],SIZE)==([],[])
    assert match_points("Drop me by the green trees",[left],SIZE)==([],[])
    assert match_points("Drop me by the green bin",[left],SIZE)==([left],[])
    assert match_points("Drop me by the red bin",[left],SIZE)==([],[])


def test_geometry_bounds_stable_originals_and_duplicate_words():
    left,right = scenes()
    center = deepcopy(left)
    center["point"] = [500,300]
    query = "Drop me in a spot on the left by the trees"
    assert match_points(query,[None,left,deepcopy(left),right,center],{"width":1000,"height":600})==([left],[center])
    assert match_points("The open curbside space",[left,right],SIZE)==([left,right],[])
    assert match_points(query,[{**left,"point":[1000,600]},{**left,"point":[True,4]}],SIZE)==([],[])
    assert match_points(query,[left],(0,600))==([],[])


def test_ground_material_is_independently_required():
    left,_ = scenes()
    words = "The open gravel pull-in by the green rubbish bin"
    assert match_points(words,[left],SIZE)==([],[])
    left["descriptions"]["roadside"] += " | gravel"
    assert match_points(words,[left],SIZE)==([left],[])
    assert match_points(words + " would work. Can you drop me there?",[left],SIZE)==([left],[])


def test_modifiers_stay_with_the_landmark_that_owns_them():
    left,_ = scenes()
    left["descriptions"]["roadside"] = "Palm trees and a large green bin"
    assert match_points("The gap by the green trees",[left],SIZE)==([],[])
    assert match_points("The gap by the green bin and the trees",[left],SIZE)==([left],[])
    left["descriptions"]["frontage"] = "Oak Cafe; Ash Cafe"
    assert match_points("Drop me outside Oak and Ash",[left],SIZE)==([],[])
    left["descriptions"]["frontage"] = "Oak and Ash Cafe"
    assert match_points("Drop me outside Oak and Ash",[left],SIZE)==([left],[])


def test_same_row_does_not_prove_no_paint_or_that_vehicles_are_parked():
    left,_ = scenes()
    words = "Please use the unmarked space by the green bin"
    assert match_points(words,[left],SIZE)==([],[left])
    left["descriptions"]["roadside"] += " | unmarked pavement"
    assert match_points(words,[left],SIZE)==([left],[])
    left["evidence"] = {"painted_boundaries":[[1,2]]}
    assert match_points(words,[left],SIZE)==([],[])
    parking = "Drop me in the gap between the parked hatchback and the white flatbed."
    assert match_points(parking,[left],SIZE)==([],[left])
    left["descriptions"]["farther_vehicle"] += " parked"
    assert match_points(parking,[left],SIZE)==([left],[])


@pytest.mark.parametrize("raw",["a green bin beneath a red sign","a red sign on a green bin",
                                "the green bin and sign painted red","a red marker attached to a green bin"])
def test_one_landmark_cannot_borrow_another_landmarks_colour(raw):
    left,_ = scenes()
    left["descriptions"]["roadside"] = raw
    assert match_points("Drop me by the red bin",[left],SIZE)==([],[])


def test_a_vehicle_sticker_cannot_establish_body_colour():
    left,_ = scenes()
    for text in ("a sedan with a red advertising sticker","a red sticker sedan",
                 "a sedan in red advertising","a sedan beneath a red sign"):
        left["descriptions"]["farther_vehicle"] = text
        assert match_points("Drop me by the red sedan",[left],SIZE)==([],[])
    left["descriptions"]["farther_vehicle"] = "a sedan painted red"
    assert match_points("Drop me by the red sedan",[left],SIZE)==([left],[])


@pytest.mark.parametrize("raw",["Lighthouse across the road from the point",
                                "Lighthouse, on the other side of the street","Lighthouse across from the mark"])
def test_explicitly_separated_frontage_cannot_ground_beside(raw):
    left,_ = scenes()
    left["descriptions"]["frontage"] = raw
    assert match_points("Drop me outside Lighthouse",[left],SIZE)==([],[])


def test_fee_conditions_and_adjacent_colour_do_not_establish_a_business():
    left,_ = scenes()
    assert match_points("Find me a parking space for free",[left],SIZE)==([],[])
    assert match_points("Could you let me park for free by the trees?",[left],SIZE)==([],[])
    left["descriptions"]["frontage"] = "the market entrance below the red sign"
    assert match_points("Stop outside Red Market",[left],SIZE)==([],[])
    left["descriptions"]["frontage"] = "Red Market shop"
    assert match_points("Stop outside Red Market",[left],SIZE)==([left],[])


def test_recycling_and_dumpster_are_narrower_than_an_unspecified_bin():
    left,_ = scenes()
    assert match_points("The gap by the green recycling bin",[left],SIZE)==([],[])
    assert match_points("The gap by the green dumpster",[left],SIZE)==([],[])
    left["descriptions"]["roadside"] = "a green recycling bin"
    assert match_points("The gap by the green bin",[left],SIZE)==([left],[])
