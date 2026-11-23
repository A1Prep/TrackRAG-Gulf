from copy import deepcopy

import pytest

from trackrag.dropoff import bay_clearance, bay_intent, painted_bays, surface_stalls


def patch(left,top,right,bottom, *, measured=False):
    return {"box":[left,top,right,bottom], "point":[(left+right)/2,(top+bottom)/2],
            "polygon":[[left,top],[right,top],[right,bottom],[left,bottom]],
            "evidence":{"recipe":painted_bays.RECIPE if measured else surface_stalls.RECIPE}}


def road():
    return {"frame":{"height":600,"width":1000,"frame_id":5}, "tracks":[
        {"class_name":"Car", "frame_id":5, "bbox_xyxy":[260,320,460,450]},
        {"class_name":"Car", "frame_id":5, "bbox_xyxy":[570,315,766,443]}]}


@pytest.mark.parametrize("bounds", [(275,451,444,468), (480,431,527,457), (622,450,743,469)])
def test_strips_beneath_or_between_parked_vehicles_do_not_supply_a_bay(bounds):
    assert not bay_clearance.vehicle_sized(patch(*bounds), road())


def test_neighbouring_vehicle_sized_space_and_polygon_measured_from_paint_survive():
    assert bay_clearance.vehicle_sized(patch(70,395,246,492), road())
    assert bay_clearance.vehicle_sized(patch(620,448,743,469,measured=True), road())


def test_stale_or_different_depth_vehicle_cannot_veto_a_distant_bay():
    scene = deepcopy(road())
    for t in scene["tracks"]:
        t["frame_id"] -= 1
    assert bay_clearance.vehicle_sized(patch(275,451,444,468), scene)
    assert bay_clearance.vehicle_sized(patch(275,262,344,284), road())


def test_explicit_space_between_vehicles_uses_visible_parking_geometry():
    words = "I’d like to get out in the space between the silver sedan and the dark SUV on the right."
    intent = bay_intent.parse(words)
    assert intent and intent.vehicle_relation == "between" and intent.side == "right"
    assert [v.kind for v in intent.vehicles] == ["sedan", "suv"]
    assert bay_intent.parse("Is the space between the two cars bigger now?") is None
    assert bay_intent.parse("Drop me between the two cars.") is None
