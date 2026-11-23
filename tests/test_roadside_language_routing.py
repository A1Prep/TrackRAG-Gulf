"""Exercise signed candidate descriptions at the normal roadside and legacy measured-stall matchers."""
from copy import deepcopy

import pytest

from trackrag.dropoff import bay_intent, curb_point_language, roadside_context
from trackrag.dropoff.passenger import digest


SCENE = {"source":"predicted","frame":{"recording_id":"fixture","frame_id":4,"width":1000,"height":600},
         "tracks":[]}


def candidate(kind="stall",x=800):
    box = [x-80,350,x+80,430]
    return {"kind":kind,"point":[x,380],"box":box,
            "polygon":[[box[0],350],[box[2],350],[box[2],430],[box[0],430]],
            "evidence":{"recipe":"qualified_fixture","coverage":"full_stall","orientation":"parallel"},
            "descriptions":{"nearer_vehicle":"white van","farther_vehicle":"brown SUV",
                            "frontage":"Market entrance below a red sign",
                            "roadside":"a green bin beneath a red sign | trees | low fence"}}


def roadside(words, options, **kwargs):
    return roadside_context.match({"text":words,"scene":SCENE},options,**kwargs)


def early(words, candidate, **kwargs):
    d = candidate["descriptions"]
    # Old, region-aligned bay captions have just a set of labelled phrases and an optional track pair.
    context = {digest(candidate["polygon"]):{"frontage":d["frontage"]+" | "+d["roadside"],"pair":[]}}
    return bay_intent.match(bay_intent.destination_words(words),[candidate],SCENE,context,**kwargs)


@pytest.mark.parametrize("words",[
    "Drop me anywhere.",
    "I do not want a spot by that green bin.",
    "Please stop in the middle of the traffic lane next to the trees.",
    "Drop me in the through lane by the green bin.",
    "Drop me by the green bin, but in the middle of the street.",
    "Stop in the active travel lane by the fence.",
    "Just drop me in the live traffic lane, next to the market.",
    "Drop me by the red bin.",
    "Stop outside Red Market.",
    "Find me a parking space for free by the trees.",
    "What colour is the parking bay by the green bin?",
    "Drop me at an open spot next to a violet submarine.",
    "Drop me in a painted parking bay by the red bin.",
])
def test_strict_route_cannot_turn_off_task_or_contradicted_claim_into_any_selectable_option(words):
    opts = [candidate(),candidate("recess",200),candidate("unpaved_pullin",470),candidate("parking_point",640)]
    assert roadside(words,opts)==([],False)
    assert early(words,opts[0],strict_words=words)==([],[])


def test_archive_reproduces_earlier_decision_and_current_evidence_still_supports_an_honest_instruction():
    opt = candidate()
    wrong = "Drop me by the red bin."
    assert roadside(wrong,[opt],strict=False)==([opt],True)
    assert early(wrong,opt)==([opt],[])
    assert roadside(wrong,[opt])==([],False)
    assert early(wrong,opt,strict_words=wrong)==([],[])
    words = "Use the empty marked parking bay by the green bin."
    assert roadside(words,[opt])==([opt],True)
    assert early(words,opt,strict_words=words)==([opt],[])
    assert roadside("Drop me by the white van.",[opt])==([opt],True)
    assert not bay_intent.permitted_destination("You can drop me anywhere.")
    assert bay_intent.permitted_destination("You can drop me anywhere near the white van.")


def test_distinct_same_language_targets_are_retained_including_unpaved_and_only_overlaps_are_merged():
    left, right = candidate("recess",130),candidate("parking_point",680)
    unpaved = candidate("unpaved_pullin",405)
    duplicate = deepcopy(left)
    duplicate.update(kind="parking_point",point=[133,382])
    words = "Drop me by the white van."
    shown, strict = roadside(words,[left,right,duplicate,unpaved])
    assert strict and len(shown)==3
    assert {tuple(o["point"]) for o in shown}=={(130,380),(405,380),(680,380)}
    assert roadside(words,[left,right],strict=False)==([left],True)


def test_observed_median_supports_divider_but_unobserved_height_can_only_prompt_a_choice():
    opt = candidate(x=220)
    words = "Please use one of the empty marked bays along the low divider on our left."
    opt["descriptions"]["roadside"] = "a concrete median"
    assert roadside(words,[opt])==([opt],False)
    opt["descriptions"]["roadside"] = "a high median"
    assert roadside(words,[opt])==([],False)
    opt["descriptions"]["roadside"] = "a low traffic island"
    assert roadside(words,[opt])==([opt],True)
    opt["descriptions"]["roadside"] = "curb and pavement"
    assert roadside(words,[opt])==([opt],False)
    opt["evidence"]["coverage"] = "visible_portion"
    assert roadside(words,[opt])==([],False)
    assert curb_point_language.match_points("What time is the restaurant open?",[opt],(1000,600))==([],[])


def test_measured_parking_does_not_strip_an_action_word_from_a_business_name():
    opt = candidate()
    words = "Drop me in a marked bay outside Lighthouse Spot."
    opt["descriptions"]["frontage"] = "Lighthouse Park"
    assert roadside(words,[opt])==([],False)
    assert early(words,opt,strict_words=words)==([],[])
    opt["descriptions"]["frontage"] = "Lighthouse Spot"
    assert roadside(words,[opt])==([opt],True)
    assert early(words,opt,strict_words=words)==([opt],[])


def test_production_numpy_geometry_is_eligible_but_unobserved_size_requires_a_choice():
    import numpy as np

    opt = candidate("recess")
    opt["point"] = [np.float64(755.5),np.float64(366.5)]
    opt["descriptions"].update(nearer_vehicle="blue sport utility vehicle",farther_vehicle="grey hatchback",
                               frontage="Corner Pet Supply")
    words = "I'd like the open spot between the blue SUV and the little grey car outside Corner Pet Supply."
    assert roadside(words,[opt])==([opt],False)
    assert roadside(words,[opt],strict=False)==([],False)
    opt["descriptions"]["farther_vehicle"] = "small grey hatchback"
    assert roadside(words,[opt])==([opt],True)
    opt["point"][0] = np.bool_(True)
    assert roadside(words,[opt])==([],False)
