import pytest

from trackrag import footage_proximity as proximity
from trackrag.answering import validate_answer


@pytest.mark.parametrize("words,place", [
    ("Is the truck near the building?", "the building"),
    ("Is that pickup parked close to the tower?", "the tower"),
    ("How far is the blue sedan from that shop?", "that shop"),
    ("How far away is the truck from the mall?", "the mall"),
    ("Can you check whether the bus is actually next to the gate?", "the gate"),
])
def test_unmeasured_distance_preserves_referents_in_optional_visual_reframe(words,place):
    match = proximity.physical_question(words)
    assert match and proximity.can_reframe(match) and match["place"] == place
    assert match["subject"] in proximity.image_question(match)


@pytest.mark.parametrize("words", [
    "What colour is the truck near the building?", "Is the truck near the end of the clip passing a building?",
    "Does it look like the truck and building overlap in the picture?", "Is the truck left of the shop in this image?",
    "In this image, is the bus near the building?", "How many vehicles are near the building?",
])
def test_appearance_time_counts_and_explicit_image_comparison_keep_their_own_routes(words):
    assert proximity.physical_question(words) is None


def test_no_visual_rewrite_that_would_discard_another_condition_or_time():
    for words in ("Is the bus near the gate at frame 1?", "At frame 1, was the truck close to the building?", "Is the bus near the gate and facing left?"):
        match = proximity.physical_question(words)
        assert match and not proximity.can_reframe(match)


def test_physical_claim_is_withheld_even_if_model_cites_a_real_frame():
    evidence = [{"evidence_id":"E1","interval":{"track_id":None},
                 "images":[{"frame_id":8,"observations":[]}]}]
    def check(words,answer):
        return validate_answer({"answer_status":"answered","answer":answer,"cited_evidence_ids":["E1"]}, evidence,
                               {"question":words,"query_frame":8})
    wrong = "Yes, the truck is near the buildings visible in the distance and background skyline."
    for words in ("Is the truck near the building?", "Does it look like the truck and building overlap in the picture?"):
        result = check(words,wrong)
        assert result["answer_status"] == "insufficient_evidence" and not result["cited_evidence_ids"]
    visible = check("Does it look like the truck and building overlap in the picture?",
                    "The vehicle is below the building in the image and their outlines do not overlap.")
    assert visible["answer_status"] == "answered" and visible["cited_evidence_ids"] == ["E1"]
    visual_words = "In this frame, describe the image-plane positions of the truck and the building."
    assert check(visual_words,"The truck is on the left; they are physically close.")["answer_status"] == "insufficient_evidence"
    assert check(visual_words,"The truck is on the left and parked within a few steps of the building in real life.")["answer_status"] == "insufficient_evidence"
    assert check(visual_words,"The truck is below the building, but physical distance cannot be measured in metres.")["answer_status"] == "answered"
    assert check("At frame 1, was the truck close to the building?","They were about two metres away.")["answer_status"] == "insufficient_evidence"
