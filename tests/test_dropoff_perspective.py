from types import SimpleNamespace

import pytest

from trackrag.dropoff.perspective import PassengerViewpoint, clarify_reference
from trackrag.dropoff.referent_policy import clarify_reference as previous


def cars():
    def car(identity, left):
        return {"track_id": identity, "category": "car", "class_name": "Car",
                "bbox_xyxy": [left, 30, left + 15, 56], "score": 0.48}
    return {"alternatives": [car(81, 16), car(53, 73)]}


@pytest.mark.parametrize("text", [
    "Please stop beside the vehicle just up ahead.",
    "Let me out past that car immediately in front of me.",
    "Pull over next to that car right in front of our taxi, please.",
    "Can we stop by the car straight ahead of you?",
])
def test_speaker_relative_language_asks_about_two_observed_alternatives(text):
    advice = clarify_reference(text, cars(), (100, 100))
    assert advice["track_ids"] == [81, 53]
    assert advice["question"].endswith("?")
    assert clarify_reference(text, {"alternatives": cars()["alternatives"][:1]}, (100, 100)) is None


@pytest.mark.parametrize("text", [
    "Let me out by the car directly in front of the shop.",
    "Next to the car straight ahead of the cyclist, please.",
    "I mean the rightmost car just ahead of us.",
    "Behind the car just up ahead with its hazard lights on.",
    "Not the car beside me; the car in front of the white van.",
    "In the gap between that car straight ahead and the bus.",
    "Beside the car straight ahead of my friend.",
])
def test_context_can_already_identify_a_specific_landmark(text):
    assert clarify_reference(text, cars(), (100, 100)) is None


def test_existing_single_reference_decision_keeps_its_identity():
    text = "Before that car, please."
    assert clarify_reference(text, cars(), (100, 100)) == previous(text, cars(), (100, 100))


def test_interface_preserves_original_proposal_and_user_selection():
    result = {"status": "proposed", "destination_point_xy": [34, 66],
              "destination_box_xyxy": [30, 60, 50, 70], "clarification_question": None}
    original = {"ok": True, "result": result, "provenance": {"referent_ranking": cars()}}
    request = {"text": "Next to that car straight ahead of us.", "dialogue": [],
               "scene": {"frame": {"width": 100, "height": 100}}}
    manual = object()
    model = PassengerViewpoint(SimpleNamespace(interpret=lambda r: original,
                                               passenger_area=lambda r, box: manual))
    checked = model.interpret(request)
    assert checked["result"]["status"] == "needs_clarification"
    assert checked["result"]["destination_point_xy"] is None
    assert checked["provenance"]["viewpoint_clarification"]["original_proposal"] == result
    assert original["result"] == result and "viewpoint_clarification" not in original["provenance"]
    assert model.passenger_area(request, [30, 60, 50, 70]) is manual
    for preserved in ({"ok": False, "error": {"code": "timeout"}},
                      {"ok": True, "result": {"status": "needs_clarification"}},
                      {"ok": True, "result": {"status": "insufficient_visual_evidence"}},
                      {"ok": True, "result": result, "provenance": {}}):
        assert PassengerViewpoint.review(request, preserved) is preserved
