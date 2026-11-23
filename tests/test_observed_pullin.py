from copy import deepcopy

import pytest

from trackrag.dropoff import observed_pullin


def location():
    return {"point":[350,300],"vehicle_crops":[[5,[30,160,275,340]],[7,[435,170,675,330]]],
            "descriptions":{"nearer_vehicle":"silver SUV","farther_vehicle":"blue sedan", "frontage":"Federal Exchange",
                            "roadside":"dark curb"}, "rear_vehicle":"NONE"}


def test_words_bind_to_correct_vehicle_and_sign():
    option=location()
    good="Pull into the empty space just to the left of the blue sedan outside the exchange, please."
    assert observed_pullin._relative(good,[option],[800,500])==([option],[])
    for text in (good.replace("left","right"),good.replace("blue","yellow"),good.replace("exchange","library"),
                 good.replace("space","wheelchair space")):
        assert observed_pullin._relative(text,[option],[800,500])==([],[])
    wrong_tracks=deepcopy(option)
    wrong_tracks["vehicle_crops"][1][1]=[50,170,200,330]
    assert observed_pullin._relative(good,[wrong_tracks],[800,500])==([],[])


def test_rear_and_qualifiers_are_checked_for_the_same_vehicle():
    option=location()
    words="Drop me behind the blue sedan on the right"
    assert observed_pullin._relative(words,[option],[800,500])==([],[option])
    option["rear_vehicle"]="blue sedan"
    assert observed_pullin._relative(words,[option],[800,500])==([option],[])
    assert observed_pullin._relative(words.replace("blue","orange"),[option],[800,500])==([],[])
    assert observed_pullin._relative(words.replace("right","left"),[option],[800,500])==([],[])
    assert observed_pullin._relative(words.replace("blue sedan","blue advertising sedan"),[option],[800,500])==([],[option])


@pytest.mark.parametrize("text",["behind the post opposite the blue sedan","drop me behind the car or van",
                                 "drop me behind the blue sedan outside the bakery next to the station"])
def test_irreconcilable_instructions_have_no_target(text):
    assert observed_pullin._relative(text,[location()],[800,500])==([],[])
