from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from trackrag.dropoff import vehicle_rear_ground as rear


def parked_row(*, mirrored=False, boundary="paint"):
    height, width = 210, 340
    classes = np.full((height,width), 13, np.uint8)
    photo = np.full((height,width,3), (56,59,63), np.uint8)
    for y in range(height):
        inner = round(198 + .9*(y-136))
        outer = round(244 + 1.32*(y-136))
        if outer <= inner:
            continue
        if boundary == "material":
            classes[y,inner:outer] = 10
            photo[y,inner:outer] = (126,106,95)
        else:
            classes[y,inner:inner+3] = 24
            photo[y,inner:inner+3] = (220,222,220) if boundary == "paint" else (195,153,17)
        classes[y,outer:outer+4] = 2
        classes[y,outer+4:] = 15
        photo[y,outer:] = (155,145,145)
    raw = [(1,[202,88,238,136]),(2,[171,70,198,104])]
    if mirrored:
        classes, photo = np.fliplr(classes).copy(), np.fliplr(photo).copy()
        raw = [(i,[width-x1,y0,width-x0,y1]) for i,(x0,y0,x1,y1) in raw]
    tracks = []
    for ident,box in raw:
        x0,y0,x1,y1 = box
        classes[y0:y1,x0:x1] = 55
        photo[y0:y1,x0:x1] = (85,90,92)
        tracks.append({"track_id":ident,"class_name":"car","frame_id":20,"bbox_xyxy":box})
    scene={"frame":{"width":width,"height":height,"recording_id":"development-fixture","frame_id":20},
           "source":"predicted","provenance":{"image_sha256":"a"*64},"tracks":tracks}
    anchor={"track_id":1,"class":"car","box":list(map(float,raw[0][1]))}
    return Image.fromarray(photo), classes, scene, anchor


@pytest.mark.parametrize("mirror",[True,False])
@pytest.mark.parametrize("boundary,surface", [("paint",13),("material",10)])
def test_ground_on_both_sides_can_be_proved_by_paint_or_a_distinct_parking_material(mirror,boundary,surface):
    rgb, classes, scene, anchor = parked_row(mirrored=mirror,boundary=boundary)
    result = rear.candidates(rgb,classes,scene,anchor,facing="REAR")
    [point] = result
    evidence = point["evidence"]
    assert evidence["coverage"] == "interior_point_only" and evidence["aligned_neighbour_id"] == 2
    assert evidence["curb_side"] == ("left" if mirror else "right")
    assert evidence["point_surface_class"] == surface
    assert evidence["divider_observations"]["paint" if boundary=="paint" else "parking_material"] > 8
    x,y=point["point"]
    assert y>anchor["box"][3] and classes[int(y),int(x)] == surface
    buffer=BytesIO(); rgb.save(buffer,format="PNG")
    assert rear.candidates(buffer.getvalue(),classes,scene,anchor,facing="REAR") == result
    first, last = evidence['bounds_map_y_lane_curb'][0], evidence['bounds_map_y_lane_curb'][-1]
    assert last[0] > first[0] + 7
    y, lane, kerb = first
    assert rear.contains_measured_point(point, [(lane+kerb)/2+.5,y+.5],scene['frame'])
    assert not rear.contains_measured_point(point,[min(lane,kerb)-2,y+.5],scene['frame'])
    assert not rear.contains_measured_point(point,[(lane+kerb)/2+.5,last[0]+9],scene['frame'])


@pytest.mark.parametrize("facing", ["FRONT","SIDE","UNCLEAR",None])
def test_side_and_front_do_not_establish_visible_ground_behind_the_car(facing):
    rgb, classes, scene, anchor = parked_row()
    assert not rear.candidates(rgb,classes,scene,anchor,facing=facing)


@pytest.mark.parametrize("change", ["no_kerb","unpainted_road","isolated","stale","duplicate","traffic",
                                    "pedestrian","foreign","far_side","yellow","no_optical_paint","flat_parking"])
def test_curb_queue_road_aisle_blockage_and_unrelated_detections_cannot_supply_an_option(change):
    boundary = "material" if change=="flat_parking" else "yellow" if change=="yellow" else "paint"
    rgb, classes, scene, anchor = parked_row(boundary=boundary)
    if change=="no_kerb":
        classes[np.isin(classes,(2,15))]=13
    elif change=="unpainted_road":
        classes[classes==24]=13
    elif change=="isolated":
        scene["tracks"].pop()
    elif change=="stale":
        scene["tracks"][1]["frame_id"]-=1
    elif change=="duplicate":
        scene["tracks"][1]["bbox_xyxy"]=list(scene["tracks"][0]["bbox_xyxy"])
    elif change=="traffic":
        scene["tracks"].append({"track_id":3,"class_name":"truck","frame_id":20,"bbox_xyxy":[194,138,291,190]})
    elif change=="pedestrian":
        classes[138:161,195:285]=23
    elif change=="foreign":
        anchor=deepcopy(anchor);anchor["box"][0]+=1
    elif change=="far_side":
        scene["tracks"][1]["bbox_xyxy"]=[25,70,50,104]
    elif change in {"no_optical_paint","flat_parking"}:
        rgb=Image.new("RGB",rgb.size,(85,85,85))
    assert rear.candidates(rgb,classes,scene,anchor,facing="REAR") == []
