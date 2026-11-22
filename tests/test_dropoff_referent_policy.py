"""The single-vehicle guard can request clarification, never assign or confirm geometry."""
import pytest

from trackrag.dropoff.referent_policy import clarify_reference


def candidate(identity,x,*,kind="Car",color="red", top=30,span=16):
    return {"track_id":identity,"bbox_xyxy":[x,top,x+span,top+span],"score":1/(identity+1),
            "class_name":kind,"category":{"Car":"car","Bus":"bus"}.get(kind,"other"),
            "colour_fractions":{color:.6},"central_colour_fractions":{color:.6}}


def ask(text, *rows):
    return clarify_reference(text,{"alternatives":list(rows)},(100,100))


def test_bare_and_competing_objects_are_never_grounded_by_guard():
    a,b=candidate(1,8),candidate(2,75)
    x=ask("Let me out beside that car",a,b)
    assert x["track_ids"]==[1,2] and "farther left" in x["question"]
    assert ask("Could you drop me off right here?",a,b)["reason"]=="unpointed_deictic"
    assert ask("beside the car",a) is None
    assert "destination_point_xy" not in x and "destination_box_xyxy" not in x


def test_requested_side_colour_and_ordinal_that_distinguish():
    a,b=candidate(5,8,color="black"),candidate(2,60,color="red")
    assert ask("The black car, please",a,b) is None
    assert ask("the car on my right",a,b) is None
    assert ask("the leftmost car",a,b) is None
    assert ask("Stop to the left of the car",a,b)["candidate_count"]==2
    red2=candidate(3,80,color="red")
    assert set(ask("Beside the red car on the right",a,b,red2)["track_ids"])=={2,3}
    assert ask("By the rightmost car",a,b,red2) is None


def test_no_duplicate_no_coarse_truck_and_real_options():
    a=candidate(2,5)
    assert ask("the car", a,dict(a,track_id=6)) is None
    truckish=candidate(7,66,kind="Large_vehicle")
    assert ask("that truck",a,truckish) is None
    assert ask("that vehicle",a,truckish)["candidate_count"]==2
    assert ask("the vehicle",a,candidate(9,70,span=1)) is None
    d,e=candidate(7,70,top=10),candidate(6,70,top=74)
    assert "higher in the picture" in ask("that car",d,e)["question"]


@pytest.mark.parametrize("text",["the white bus shelter","between the cars","not by the car",
                                 "the car beside the building","the nearest car","the second car",
                                 "the car with its lights on","the damaged car","the turquoise car",
                                 "the car displaying a badge","the blue car or the red car",
                                 "the garage next to the car","the vehicle in front of the crossing"])
def test_do_not_override_visual_reasoning_or_two_references(text):
    assert ask(text,candidate(1,8),candidate(2,70)) is None


def test_correction_uses_new_noun_and_real_geometry():
    a,b=candidate(1,8),candidate(2,75)
    assert ask("By the car. Actually, the shelter please.",a,b) is None
    assert ask("By the shelter. I mean the car.",a,b)["candidate_count"]==2
    assert ask("by the car. The one on my right, please.",a,b) is None


def test_a_vehicle_subtype_mentioned_with_a_second_reference_is_not_the_second_object():
    a,b=candidate(1,7,color="yellow"),candidate(2,75,color="yellow")
    for text in ("Behind the black SUV to the right of the yellow car",
                 "The sedan beside the yellow car", "The pickup just behind that car"):
        assert ask(text,a,b) is None


def test_coarse_labels_and_weak_colour_evidence_ask_without_inventing_types():
    large=candidate(5,75,kind="Large_vehicle",color="black")
    medium=candidate(6,8,kind="Medium_vehicle",color="gray")
    r=ask("Stop behind the truck",large,medium)
    assert r["requested_category"]=="truck" and r["category"]=="vehicle"
    assert "truck" not in r["question"]
    dark_a, dark_b = candidate(1,6,color="black"),candidate(2,46,color="gray")
    r=ask("That white vehicle, please.",dark_a,dark_b)
    assert r["reason"]=="multiple_current_objects_colour_unresolved" and "white" not in r["question"]
    highlight=dict(dark_a,colour_fractions={"white":.24},central_colour_fractions={"white":.22})
    assert ask("That white vehicle, please.",highlight,dark_b)["reason"]==r["reason"]


def test_deictic_broad_setting_and_precise_landmark_relation():
    a,b=candidate(1,9),candidate(2,66)
    assert ask("That car under the bridge",a,b)["reason"]=="demonstrative_and_unlocalized_broad_setting"
    assert ask("That car under the bridge",a) is None
    assert ask("The car under the red awning",a,b) is None
