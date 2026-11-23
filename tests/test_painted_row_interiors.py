import cv2
import numpy as np
import pytest
from PIL import Image

from trackrag.dropoff import painted_row_interiors as parking


def scene(fault=None, *, mirrored=False):
    h,w=390,600
    labels=np.full((h,w),13,np.uint8)
    labels[:160]=27
    labels[172:350,250:]=10
    rgb=np.full((h,w,3),[68,69,71],np.uint8)
    rgb[labels==10]=[70,72,74]
    rgb[:160]=[132,143,155]
    for y in (238,272,306):
        if fault=="two_stripes" and y==238:
            continue
        cv2.line(labels,(266,y),(598,y+7),24,3)
        cv2.line(rgb,(266,y),(598,y+7),(191,194,190),4,cv2.LINE_AA)
    boxes=[[335,161,497,223]]
    if fault=="no_nearby_vehicle":
        boxes=[]
    if fault=="lane":
        labels[labels==10]=13
    elif fault=="crosswalk":
        labels[labels==24]=23
    elif fault=="blank":
        rgb[:]=71
    elif fault in ("closed","walk_gate","vehicle_gate","optical_only_kerb"):
        obstacle=np.zeros((h,w),bool)
        obstacle[:,245:252]=True
        if fault=="walk_gate":
            obstacle[285:290]=False
        if fault=="vehicle_gate":
            obstacle[232:326]=False
        if fault!="optical_only_kerb":
            labels[obstacle]=5
        rgb[obstacle]=[201,203,190]
    elif fault=="occupied_lower":
        boxes.append([325,280,449,310])
    elif fault=="occupied_both":
        boxes.append([300,236,450,317])
    elif fault=="current_blocking_truck":
        boxes.append([222,142,259,360])
    elif fault=="hood":
        labels[225:335,260:]=1
    for x0,y0,x1,y1 in boxes:
        rgb[y0:y1,x0:x1]=[64,75,95]
    if mirrored:
        labels=np.fliplr(labels).copy()
        rgb=np.fliplr(rgb).copy()
        boxes=[[w-1-x1,y0,w-1-x0,y1] for x0,y0,x1,y1 in boxes]
    im=Image.fromarray(rgb)
    frame={"frame_id":1,"width":w,"height":h}
    tracks=[{"frame_id":1,"track_id":i,"class_name":"Car","bbox_xyxy":b} for i,b in enumerate(boxes,1)]
    return im,labels,{"frame":frame,"tracks":tracks}


def test_three_real_markings_generate_two_empty_interiors_on_either_side():
    im,labels,s=scene()
    options=parking.candidates(im,labels,s)
    assert len(options)==2
    assert {o["evidence"]["coverage"] for o in options}=={"interior_point_only"}
    assert min(o["point"][1] for o in options)>238
    assert max(o["point"][1] for o in options)<310
    assert all(o["evidence"]["aisle_direction"]==-1 for o in options)
    assert len(parking.matched(im,labels,s,[419,272,423,278]))==2
    assert not parking.matched(im,labels,s,[419,247,423,253])
    assert not parking.matched(im,labels,s,[119,272,123,278])
    mirrored=parking.candidates(*scene(mirrored=True))
    assert len(mirrored)==2
    assert all(o["evidence"]["aisle_direction"]==1 for o in mirrored)


@pytest.mark.parametrize("fault",["two_stripes","blank","lane","crosswalk","no_nearby_vehicle","closed",
                                 "walk_gate","optical_only_kerb","current_blocking_truck","occupied_both","hood"])
def test_missing_geometry_or_blocked_entrance_cannot_yield_a_destination(fault):
    im,labels,s=scene(fault)
    assert not parking.candidates(im,labels,s)
    assert not parking.matched(im,labels,s,[419,272,423,278])


def test_a_vehicle_opening_remains_usable_and_one_blocked_side_is_not_offered():
    assert len(parking.candidates(*scene("vehicle_gate")))==2
    candidates=parking.candidates(*scene("occupied_lower"))
    assert len(candidates)==1
    assert candidates[0]["point"][1]<272
