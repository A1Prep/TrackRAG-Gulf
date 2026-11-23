from copy import deepcopy

import pytest

from trackrag.dropoff import curb_point_language, roadside_context


SCENE={"source":"predicted","frame":{"width":1000,"height":600,"frame_id":4},"tracks":[]}
OPT={"kind":"recess","point":[800,400],"box":[750,360,850,440],
     "polygon":[[750,360],[850,360],[850,440],[750,440]],"evidence":{"recipe":"verified_fixture"},
     "descriptions":{"nearer_vehicle":"gold sedan","farther_vehicle":"blue van",
                     "frontage":"Timber Workshop","roadside":"blue bin | gravel"}}


def ground(words, conversational=True):
    return roadside_context.match({"text":words,"scene":SCENE},[OPT],conversational=conversational)


@pytest.mark.parametrize("words",[
    "Timber Workshop is my destination. I'll get off between the gold sedan and the blue van.",
    "My stop is Timber Workshop. We would like to hop out between the blue van and the gold sedan.",
    "Can you see Timber Workshop? I'd like to get off in the opening between the blue van and the gold sedan there.",
    "See the blue van and the gold sedan? Can you stop in the gap between them?",
    "The blue van and the gold sedan on our right have some room between them. That's where we want to get out.",
    "There is an opening flanked by the gold sedan and the blue van. Could you drop us off there?",
    "The gap between the gold sedan on one side and the blue van on the other is where I'd like to get out.",
    "See the blue bin? The patch of gravel beside it is where we want to get off.",
    "Could you pull into the gravel area beside the blue waste bin and let us out there?",
    "I need to get off in that gap between the blue van and the gold sedan.",
])
def test_grounded_dialogue_preserves_explicit_objects_positions_and_properties(words):
    assert curb_point_language.indexed_reference(words)
    assert ground(words)==([OPT],True)


@pytest.mark.parametrize("words",[
    "Pine Workshop is my stop. I'll get out between the blue van and the gold sedan.",
    "See Timber Workshop? I'll get out between the red van and the gold sedan there.",
    "See a red bin? The gravel patch beside it is where I want to get out.",
    "The blue van and the gold sedan on our left have some room between them. That's where I want to get out.",
    "See Timber Workshop? I'd like to get out at the cinema instead.",
    "I don't want the blue van and the gold sedan. Please let me out there.",
    "See the blue bin? Let me out behind it.",
    "Drop me by Timber Workshop on the other side.",
    "Timber Workshop is my stop. Let me off between the blue van and the gold sedan if it is legal.",
    "The speed sign is blue. That's where I'd like to stop.",
    "I saw a van. What's the registration plate?",
])
def test_bad_or_unresolved_refs_never_offer_the_same_place(words):
    assert ground(words)==([],False)


def test_clarify_missing_size_and_replay_pre_dialogue_language_policy():
    words="Timber Workshop is my stop. I'll get out between that blue van and the small gold sedan."
    assert ground(words)==([OPT],False)
    assert ground(words,conversational=False)==([],False)
    assert not curb_point_language.indexed_reference("Could you drop me near the public library?")
    assert not curb_point_language.indexed_reference("Drop me in the middle of the road next to the blue van.")


def test_opposed_sides_require_two_separate_vehicle_crops_whose_outer_edges_enclose_the_point():
    item=deepcopy(OPT)
    item["vehicle_crops"]=[[54,[400,240,600,500]],[91,[840,260,960,480]]]
    def scoped(text):
        return roadside_context.match({"text":text,"scene":SCENE},[item],conversational=True,expanded=True)
    request="At Timber Workshop, leave me in the opening to the right of the gold sedan and left of the blue van."
    assert scoped(request)==([item],True)
    assert scoped(request.replace("gold sedan","little gold sedan"))==([item],False)
    assert scoped(request.replace("gold sedan","brown sedan"))==([],False)
    assert scoped(request.replace("Timber Workshop","City Bakery"))==([],False)
    assert scoped(request.replace("right of the gold sedan and left of the blue van",
                                  "left of the gold sedan and right of the blue van"))==([],False)
    assert scoped(request.replace("right of the gold sedan","left of the gold sedan"))==([],False)
    assert not curb_point_language.indexed_reference(request)
    assert curb_point_language.indexed_reference(request,expanded=True)
    crops=item.pop("vehicle_crops")
    assert scoped(request)==([],False)
    item.update(kind="stall")
    assert scoped(request.replace("opening","parking bay"))==([],False)
    item["vehicle_crops"]=[crops[0],[54,crops[1][1]]]
    assert scoped(request)==([],False)
    item["vehicle_crops"]=crops
    item["point"]=[600,400]
    assert scoped(request)==([],False)


def test_room_is_a_destination_when_another_clause_explicitly_points_back_to_it():
    request="There’s room outside Timber Workshop between the gold sedan and the blue van. Could you let me out there?"
    def route(text,expanded=True):
        return roadside_context.match({"text":text,"scene":SCENE},[OPT],conversational=True,expanded=expanded)
    assert route(request)==([OPT],True)
    assert route(request,expanded=False)==([],False)
    assert route(request.replace("blue van","red van"))==([],False)
    assert route(request.replace("Timber Workshop","City Bakery"))==([],False)
    assert route(request.replace("Could you let me out there?","Drive past it."))==([],False)
