from copy import deepcopy

import pytest
import numpy as np

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


def test_raised_divider_splits_parking_from_the_road_with_an_observed_entrance_restoring_it():
    classes = np.full((120,180),13,np.uint8)
    classes[:,110:] = 10
    classes[:,106:110] = 2
    f = {"width":180,"height":120}
    far = patch(125,62,160,91)
    near = patch(56,65,86,95)
    divided = bay_clearance.visible_road_component(classes)
    assert bay_clearance.on_visible_road(near,divided,f)
    assert not bay_clearance.on_visible_road(far,divided,f)
    classes[82:96,106:110] = 13
    assert bay_clearance.on_visible_road(far,bay_clearance.visible_road_component(classes),f)


def test_missing_view_of_the_ego_road_is_inconclusive():
    classes = np.full((90,140),10,np.uint8)
    assert bay_clearance.visible_road_component(classes) is None


def test_large_dashboard_does_not_hide_a_visible_road_above_it():
    classes = np.full((180,280),64,np.uint8)
    classes[88:110,:133]=13
    classes[84:110,135:]=10
    classes[84:110,133:135]=2
    road = bay_clearance.visible_road_component(classes)
    assert road is not None and road[101,99] and not road[101,190]


def test_replay_uses_the_road_seed_recorded_for_each_policy():
    from trackrag.dropoff import whole_bay
    classes = np.full((180,280),64,np.uint8)
    classes[88:110,:133]=13
    classes[84:110,135:]=10
    classes[84:110,133:135]=2
    assert whole_bay._road_for_policy(classes,whole_bay.VEHICLE_POLICY) is None
    assert whole_bay._road_for_policy(classes,whole_bay.CONNECTED_POLICY) is None
    current = whole_bay._road_for_policy(classes,whole_bay.CLEARANCE_POLICY)
    assert current[101,99] and not current[101,190]
    classes[110:170,:133]=13
    historic = whole_bay._road_for_policy(classes,whole_bay.CONNECTED_POLICY)
    assert historic[101,99] and not historic[101,190]
    assert whole_bay._road_for_policy(classes,whole_bay.VEHICLE_POLICY) is None
    with pytest.raises(ValueError):
        whole_bay._road_for_policy(classes,"unknown")
