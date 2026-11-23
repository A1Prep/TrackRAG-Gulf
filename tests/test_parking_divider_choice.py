import cv2
import numpy as np
import pytest
from PIL import Image

from trackrag.dropoff import parking_divider_choice


def stripe(fault=None):
    h,w=270,460
    labels=np.full((h,w),15,np.uint8)
    labels[101:215]=10
    labels[215:]=13
    rgb=np.full((h,w,3),92,np.uint8)
    rgb[:101]=[161,147,132]
    line=((71,146),(408,173))
    cv2.line(labels,*line,24,3)
    cv2.line(rgb,*line,(218,218,218),4,cv2.LINE_AA)
    f={"frame_id":3,"width":w,"height":h}
    tracks=[]
    options=[]
    for n,p in enumerate(([207.,139.],[228.,188.])):
        options.append({"point":p,"polygon":[p],"box":[*p,*p],"evidence":{"recipe":"signed_current_point"}})
    if fault=="one_point":
        options.pop()
    elif fault=="no_visible_paint":
        rgb[:]=92
    elif fault=="no_signed_paint":
        labels[labels==24]=10
    elif fault=="through_lane":
        labels[165:215]=13
    elif fault=="zebra":
        labels[labels==24]=23
    elif fault=="another_divider_to_point":
        cv2.line(labels,(50,171),(390,182),24,3)
    elif fault=="wall_across":
        labels[206:221]=6
    elif fault in ("walk_gate","vehicle_gate"):
        labels[204:211]=5
        width=3 if fault=="walk_gate" else 33
        labels[204:211,262-width//2:263+width//2]=10
    elif fault=="parked_vehicle_between":
        tracks.append({"track_id":2,"frame_id":3,"bbox_xyxy":[215,165,233,178]})
    elif fault=="tapped_vehicle":
        tracks.append({"track_id":5,"frame_id":3,"bbox_xyxy":[205,145,235,161]})
    elif fault=="unknown_surface_on_one_side":
        labels[166:178]=15
    return Image.fromarray(rgb),labels,{"frame":f,"tracks":tracks},[216,155,224,161],options


def test_real_paint_requires_two_current_independent_interiors_and_explicit_choice():
    matches=parking_divider_choice.adjacent_points(*stripe())
    assert matches is not None
    options,measure=matches
    assert [p["point"] for p in options]==[[207.,139.],[228.,188.]]
    assert all(p["evidence"]["coverage"]=="interior_point_only" for p in options)
    assert len(measure["centerline_original_xy"])==2
    assert min(measure["parking_fraction_per_side"])>.9
    assert parking_divider_choice.adjacent_points(*stripe("vehicle_gate")) is not None


@pytest.mark.parametrize("fault",["one_point","no_visible_paint","no_signed_paint","through_lane","zebra",
                                 "another_divider_to_point","wall_across","walk_gate","parked_vehicle_between",
                                 "tapped_vehicle","unknown_surface_on_one_side"])
def test_no_choice_without_source_corroboration_or_an_open_path_to_both_points(fault):
    assert parking_divider_choice.adjacent_points(*stripe(fault)) is None
