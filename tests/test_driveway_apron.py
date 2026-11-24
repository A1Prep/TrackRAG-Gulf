import numpy as np
import pytest
from PIL import Image

from trackrag.dropoff import driveway_apron


def driveway(fault=None):
    h,w=400,700
    y,x=np.indices((h,w))
    left=267-.33*(y-161)
    right=435+.27*(y-161)
    labels=np.full((h,w),15,np.uint8)
    labels[y<141]=17
    lane=y>=267
    labels[lane]=13
    approach=(y>=145)&(y<267)&(x>=left)&(x<=right)
    curb=(y>=145)&(y<265)&(((x>=left-5)&(x<left))|((x>right)&(x<=right+5)))
    lip=(y>=257)&(y<267)
    cut=lip & approach
    labels[curb | lip]=2
    labels[cut]=9
    rgb=np.full((h,w,3),[194,139,148],np.uint8)
    rgb[lane]=[82,83,87]
    rgb[approach]=[209,200,183]
    rgb[curb]=[130,131,126]
    rgb[lip & ~cut]=[138,138,127]
    rgb[:141]=[67,82,86]
    boxes=[[271,115,343,165],[365,110,435,159]]
    labels[165:186,277:344]=10
    if fault=="same_paving":
        rgb[(y>=141)&(y<256)]=[209,200,183]
    if fault=="different_sides":
        rgb[(x<left-5)&(y>=145)&(y<257)]=[72,110,145]
    if fault=="no_kerb_returns":
        labels[curb]=15
    if fault=="no_parking_at_vehicles":
        labels[labels==10]=15
    if fault=="one_vehicle":
        boxes=boxes[:1]
    if fault=="false_vehicle_tracks":
        draw_cars=False
    else:
        draw_cars=True
    if fault=="small_pedestrian_ramp":
        labels[cut]=2
        labels[cut&(x>341)&(x<356)]=9
        rgb[cut & ((x<=341)|(x>=356))]=[120,125,123]
    if fault=="no_lowered_kerb":
        labels[cut]=2
    if fault=="zebra_crossing":
        labels[267:271,280:410]=23
        rgb[267:271,280:410]=[242,240,235]
    if fault=="cars_beyond_barrier":
        labels[187:196,187:510]=2
        rgb[187:196,187:510]=[95,97,91]
    for l,t,r,b in boxes:
        if draw_cars:
            labels[t:b+1,l:r+1]=55
            rgb[t:b+1,l:r+1]=[76,87,118]
        if fault=="raised_kerb_immediately_under_wheels":
            labels[b+1:b+4,l-3:r+4]=2
            rgb[b+1:b+4,l-3:r+4]=[174,175,174]
    actual=boxes.copy()
    if fault=="closed_gate":
        actual.append([195,241,490,267])
    if fault=="small_gap_in_gate":
        actual.extend([[195,241,338,267],[354,241,490,267]])
    if fault=="occupied":
        actual.append([286,187,420,234])
    for l,t,r,b in actual[len(boxes):]:
        labels[t:b+1,l:r+1]=35
        rgb[t:b+1,l:r+1]=[78,78,79]
    native=lambda box:[2*float(a) for a in box]
    image=Image.fromarray(rgb).resize((w*2,h*2),Image.Resampling.NEAREST)
    return image,labels,{"size":list(image.size),"current_boxes":[native(b) for b in actual],
                         "current_vehicles":[[i+1,"Car",native(b)] for i,b in enumerate(boxes)]}


def test_passenger_can_point_inside_a_lowered_entrance_between_the_two_walkways():
    im,labels,context=driveway()
    [o]=driveway_apron.candidates(im,labels,context)
    x,y=o["point"]
    assert o["evidence"]["signed_center_class"]==15
    assert o["evidence"]["extent"]=="interior_point_only"
    assert set(o["evidence"]["supporting_entrance_track_ids"])=={1,2}
    assert o["evidence"]["cut_vehicle_clear_cells"]>10
    original=[x+2,y+2]
    mark=[x,y,x+4,y+4]
    assert driveway_apron.marked(im,labels,context,mark)["point"]==original
    assert not driveway_apron.marked(im,labels,context,[x-85,y-55,x+85,y+55])
    assert not driveway_apron.marked(im,labels,context,[507,627,513,633])
    assert not driveway_apron.marked(im,labels,context,[120,397,126,403])


@pytest.mark.parametrize("fault",[
    "no_lowered_kerb","no_kerb_returns","small_pedestrian_ramp","same_paving",
    "different_sides","no_parking_at_vehicles","one_vehicle","false_vehicle_tracks",
    "small_gap_in_gate","closed_gate","cars_beyond_barrier","zebra_crossing","raised_kerb_immediately_under_wheels",
])
def test_a_pedestrian_or_inaccessible_ramp_cannot_become_a_vehicle_entrance(fault):
    im,labels,ctx=driveway(fault)
    assert not driveway_apron.candidates(im,labels,ctx)
    assert not driveway_apron.marked(im,labels,ctx,[696,416,704,424])


def test_an_obstacle_on_the_intended_apron_point_is_withheld():
    im,labels,ctx=driveway("occupied")
    assert not driveway_apron.marked(im,labels,ctx,[696,416,704,424])
    l,t,r,b=ctx["current_boxes"][-1]
    for item in driveway_apron.candidates(im,labels,ctx):
        x,y=item["point"]
        assert not (l<=x<=r and t<=y<=b)


def test_a_left_and_right_mirrored_entrance_remains_a_destination():
    im,labels,ctx=driveway()
    [initial]=driveway_apron.candidates(im,labels,ctx)
    w=im.width
    def flipped(box):
        l,t,r,b=box
        return [w-2-r,t,w-2-l,b]
    ctx={**ctx,"current_boxes":[flipped(b) for b in ctx["current_boxes"]],
         "current_vehicles":[[i,n,flipped(b)] for i,n,b in ctx["current_vehicles"]]}
    [other]=driveway_apron.candidates(im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),labels[:,::-1],ctx)
    assert abs((w-1-initial["point"][0])-other["point"][0])<2.1
    assert abs(initial["point"][1]-other["point"][1])<2.1
