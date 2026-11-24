"""A written or reviewed spoken homophone needs passenger consent before grounding a car."""
from copy import deepcopy

import pytest

from tests.test_passenger_between_road import roadside
from trackrag.dropoff import manual_mark, parking_clarification, vehicle_ground_reference, vehicle_wording, whole_bay
from trackrag.dropoff.passenger_pair_choice import PassengerSession
from trackrag.dropoff.place_evidence import PlaceDialogueGrounder


ORIGINAL = "Could you drop me behind the white card?"
CORRECTED = "Could you drop me behind the white car?"


@pytest.mark.parametrize(("words", "expected"), [
    (ORIGINAL, CORRECTED),
    ("Stop beside a parked card.", "Stop beside a parked car."),
    ("Please let me out next to that blue card on the right.", "Please let me out next to that blue car on the right."),
])
def test_standalone_vehicle_like_noun_keeps_original_placement_and_offsets(words, expected):
    proposal = vehicle_wording.suggestion(words)
    assert proposal.corrected == expected
    assert words[slice(*proposal.word_span)] == "card"
    assert words[slice(*proposal.subject_span)] in proposal.question


@pytest.mark.parametrize("words", [
    "Could you drop me behind the white card reader?", "Drop me beside the red card shop.",
    "Stop behind the blue credit card.", "Give me the white card.", "Drop me behind the white cart.",
    "Drop me beside the card.", "Drop me behind the red and white cards.", "Don't stop behind the white card.",
    "Drop me behind the red card and the van.", "Stop in the traffic lane behind the white card.",
    "Could you drop the card behind the white card?", "Stop in the bay before the sign behind the blue card.",
])
def test_literal_or_compound_requests_do_not_get_a_vehicle_suggestion(words):
    assert vehicle_wording.suggestion(words) is None


def test_repeat_after_explicit_mark_preserves_original_frame_word_and_decline(roadside):
    req = {**roadside[0], "text":ORIGINAL}
    frame = req["scene"]["frame"]
    def turn(role, words):
        return {"role":role, "text":words, "frame_id":frame["frame_id"], "recording_id":frame["recording_id"]}
    question = vehicle_wording.suggestion(ORIGINAL).question
    dialogue = [turn("user", ORIGINAL), turn("assistant", manual_mark.MARK_PROMPT), turn("assistant", question)]
    def answer(words, history=dialogue):
        return vehicle_wording.reply({**req, "dialogue":history, "text":words})
    assert answer("Yes") == answer("Yes, the car.") == ("accepted", CORRECTED)
    assert answer("No, I meant the blue SUV.") == ("accepted", "Could you drop me behind the blue suv?")
    assert answer("No") == ("declined", None)
    assert answer("No, the card reader") == ("declined", "card reader")
    for words, colour, kind, field, value in [
        ("No, behind the blue SUV outside the pharmacy","blue","suv","place","pharmacy"),
        ("No, I meant the grey hatchback","gray","car","appearance","hatchback"),
        ("No, the truck beside the shops",None,"truck","place","shops"),
    ]:
        request = {**req, "text":words, "dialogue":dialogue}
        assert answer(words)[0] == "accepted"
        ref = vehicle_ground_reference.direct_reference(request)
        assert (ref["relation"],ref["kind"],ref["color"]) == ("behind",kind,colour)
        assert any(q["field"] == field and q["value"] == value for q in ref["qualifiers"])
    complex_words = "No, the truck near the shops with the purple crate."
    unresolved = parking_clarification.clarify({**req, "text":complex_words, "dialogue":dialogue})
    assert "the truck near the shops with the purple crate" in unresolved["result"]["clarification_question"]
    assert unresolved["result"]["destination_point_xy"] is None
    numbered = "No, the truck outside Building 2"
    assert answer(numbered) == ("replacement_unclear", "the truck outside Building 2")
    for uncertain in ("Car #2", "Yes, #2", "The car?", "Correct?", "Yes 2", "Yes + #5"):
        assert answer(uncertain) == ("unclear", None)
        request = {**req,"text":uncertain,"dialogue":dialogue}
        assert vehicle_ground_reference.direct_reference(request) is None
        assert parking_clarification.clarify(request)["provenance"]["reference_evidence"]["reason"] == "vehicle_word_undetermined"
    assert answer("Stop beside the black SUV.") is None
    assert answer("Yes", dialogue + [turn("user","No"), turn("assistant","Where is the landmark?")]) is None
    assert answer("Yes", [dialogue[0],turn("assistant","Can you identify the vehicle?"),dialogue[-1]]) is None
    wrong_photo = deepcopy(dialogue)
    wrong_photo[0]["frame_id"] += 1
    assert answer("Yes", wrong_photo) is None


@pytest.mark.parametrize("first_manual", [False, True])
def test_real_session_marks_cannot_approve_words_and_actual_reply_checks_ground(roadside, tmp_path, first_manual):
    request, base, *_ = roadside
    model = PlaceDialogueGrounder(base, object())
    session = PassengerSession(request["scene"], model, audit_dir=tmp_path/"verbatim")
    initial = session.submit(ORIGINAL, mark_on_photo=first_manual)
    assert bool(initial.get("word_confirmation")) is (not first_manual)
    box = [100,108,104,112]
    marked = session.select_area(box)
    assert marked["word_confirmation"] and marked["proposal_id"] is None
    assert marked["turns"][-1]["text"] == vehicle_wording.suggestion(ORIGINAL).question
    assert "beside" not in marked["result"]["clarification_question"]
    manual = whole_bay.select_mark(session._request, box)
    assert manual["provenance"]["reason"] == "vehicle_word_check_at_mark"
    assert not whole_bay.verify_current(session._request, manual, requested_box=box)["offlane_mask_replayed"]
    with pytest.raises(ValueError):
        whole_bay.verify(session._request, manual, requested_box=[105,108,109,112])
    changed = deepcopy(manual)
    changed["result"]["clarification_question"] = "Is it beside the vehicle?"
    with pytest.raises(ValueError):
        whole_bay.verify(session._request, changed, requested_box=box)
    approved = session.submit("yes")
    assert approved["state"] == "needs_clarification" and "word_confirmation" not in approved
    assert not approved["proposal_id"] and "verified view" in approved["result"]["clarification_question"]
    assert approved["turns"][0]["text"] == ORIGINAL
    assert vehicle_ground_reference.direct_reference(session._request)["color"] == "white"


def test_recognized_words_edited_transcript_and_new_turn_are_separate(roadside, tmp_path):
    request, base, *_ = roadside
    req = {**request, "text":ORIGINAL, "input_mode":"voice", "original_transcript":ORIGINAL}
    direct = parking_clarification.clarify(req)
    assert direct["provenance"]["reference_evidence"]["suggested_text"] == CORRECTED
    assert parking_clarification.verify(req,direct)["native_model_calls"] == 0
    assert parking_clarification.clarify({**req, "text":CORRECTED}) is None
    session = PassengerSession(req["scene"], PlaceDialogueGrounder(base,object()), audit_dir=tmp_path/"decline")
    assert session.submit(ORIGINAL)["word_confirmation"]
    declined = session.submit("No, I meant the card shop.")
    assert not declined.get("word_confirmation") and not declined["proposal_id"]
    assert "card shop" in declined["result"]["clarification_question"]
    assert vehicle_wording.reply(session._submission_input("yes",None,None,"a-new-turn")) is None
    assert vehicle_wording.reply(session._submission_input("Stop behind the red van.",None,None,"b-new-turn")) is None
