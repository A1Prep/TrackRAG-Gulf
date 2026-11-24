import numpy as np
import pytest
from PIL import Image

from trackrag.dropoff import row_extension


def street(*, broader=False, fault=None):
    """A raised kerb and separate light shoulder widen toward the camera."""
    h, w = 400, 700
    yy,xx = np.indices((h,w))
    d = np.maximum(0, yy - 170)
    paint = 275 + .88*d
    kerb = 275 + (1.95 if broader else 1.44)*d
    stripe = (yy >= 170) & (xx >= paint) & (xx < paint + 5)
    strip = (yy >= 170) & (xx >= paint+5) & (xx < kerb)
    lip = (yy >= 170) & (xx >= kerb) & (xx < kerb + 8)
    outside = (yy >= 170) & (xx >= kerb + 8)
    labels = np.full((h,w),13,np.uint8)
    labels[:168] = 17
    labels[outside],labels[lip],labels[stripe] = 15,2,24
    rgb = np.full((h,w,3),78,np.uint8)
    rgb[strip],rgb[stripe],rgb[outside] = 146,221,117
    rgb[lip] = 90
    rgb[lip & (xx < kerb + 4)] = 187
    far,near=[368,229,418,272],[386,243,452,295]
    if fault == "duplicate_vehicle":
        far=[390,235,443,289]
    boxes=[near,far]
    if fault == "one_vehicle":
        boxes = [near]
    if fault == "no_paint_in_photo":
        rgb[stripe]=78
    if fault == "same_asphalt":
        rgb[strip]=78
    if fault == "glare_only":
        rgb[strip]=78
        rgb[strip & (abs(xx-(paint*.45+kerb*.55))<4)]=225
    if fault == "no_kerb":
        labels[lip] = 15
    if fault == "guardrail":
        labels[outside & (xx<kerb+15)]=4
    if fault == "cycle_lane":
        labels[strip]=7
    if fault == "pedestrian_crossing":
        labels[strip & (yy>302)&(yy<313)]=23
    if fault == "planted_access_on_mislabelled_road":
        rgb[(yy>=275)&(xx>paint+7)&(xx<paint+15)] = [35,154,38]
    if fault == "no_frontage":
        labels[:168]=27
    for left,top,right,bottom in boxes:
        if fault != "boxes_on_empty_photo":
            labels[top:bottom+1,left:right+1]=55
            rgb[top:bottom+1,left:right+1]=67
    img=Image.fromarray(rgb).resize((2*w,2*h),Image.Resampling.NEAREST)
    native=[[float(2*v) for v in box] for box in boxes]
    context={"size":list(img.size),"current_boxes":native.copy(),
             "current_vehicles":[[i+1,"Car",b] for i,b in enumerate(native)]}
    if fault=="truck_across_entrance":
        context["current_boxes"].append([440,612,914,668])
        labels[306:335,220:457]=61
    return img,labels,context


def test_markup_is_an_interior_point_beyond_the_last_car_in_a_measured_row():
    im,mask,ctx=street()
    [o]=row_extension.candidates(im,mask,ctx)
    assert o["evidence"]["nearest_row_track_id"]==1
    assert o["evidence"]["supporting_row_track_ids"]==[1,2]
    assert o["evidence"]["signed_center_class"]==13
    assert o["evidence"]["extent"]=="interior_point_only"
    assert "flanking_track_ids" not in o["evidence"]
    x,y=o["point"]
    assert y>ctx["current_boxes"][0][3]
    x+=2
    mark=[x-1,y-1,x+1,y+1]
    assert row_extension.marked(im,mask,ctx,mark)["point"]==[x,y]
    assert row_extension.marked(im,mask,ctx,[x-70,y-30,x+70,y+30]) is None
    assert row_extension.marked(im,mask,ctx,[594,y-1,596,y+1]) is None


@pytest.mark.parametrize("fault",[
    "one_vehicle","duplicate_vehicle","no_paint_in_photo","same_asphalt","glare_only",
    "no_kerb","guardrail","cycle_lane","pedestrian_crossing","no_frontage",
    "boxes_on_empty_photo","truck_across_entrance","planted_access_on_mislabelled_road",
])
def test_a_stripe_or_a_pair_of_detections_does_not_establish_a_curbside_destination(fault):
    im,mask,ctx=street(fault=fault)
    assert not row_extension.candidates(im,mask,ctx)


def test_two_vehicles_in_a_carriageway_sized_outer_travel_lane_are_not_a_parked_shoulder():
    im,mask,ctx=street(broader=True)
    assert not row_extension.candidates(im,mask,ctx)


def test_equivalent_row_on_the_other_side_uses_the_same_pixel_evidence():
    im,mask,ctx=street()
    [o]=row_extension.candidates(im,mask,ctx)
    w=im.width
    def flip(box):
        l,t,r,b=box
        return [w-2-r,t,w-2-l,b]
    ctx={**ctx,"current_boxes":[flip(v) for v in ctx["current_boxes"]],
         "current_vehicles":[[tid,kind,flip(box)] for tid,kind,box in ctx["current_vehicles"]]}
    [mirrored]=row_extension.candidates(im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),mask[:,::-1],ctx)
    assert abs((w-1-o["point"][0])-mirrored["point"][0])<1
    assert mirrored["point"][1]==o["point"][1]


def test_native_local_planting_prevents_a_map_based_point():
    im,mask,ctx=street()
    [o]=row_extension.candidates(im,mask,ctx)
    x,y=map(round,o["point"])
    data=np.asarray(im).copy()
    data[y-12:y+13,x-12:x+13]=[40,181,40]
    modified=Image.fromarray(data)
    assert not row_extension.candidates(modified,mask,ctx)
    assert row_extension.marked(modified,mask,ctx,[x-1,y-1,x+1,y+1]) is None
