import numpy as np
import pytest
from PIL import Image

from trackrag.dropoff import paver_pullin as method


def source(kind="pavers", *, both=False):
    w,h=600,400
    yy,xx=np.indices((h,w))
    labels=np.full((h,w),13,dtype="uint8")
    rng=np.random.default_rng(9)
    if kind=="pavers":
        rear=.22*xx+169
        aisle=.42*xx+153
        labels[yy<rear-8]=17
        labels[(yy>=rear-8)&(yy<rear-3)]=15
        labels[(yy>=rear-3)&(yy<rear)]=2
        labels[abs(yy-aisle)<=1.4]=24
        brick=(yy>=rear)&(yy<aisle)
        boxes=[[160,147,235,240],[360,176,487,289]]
        trial=[600.,525.]
    else:
        labels[:230]=17
        labels[245:253,210:423]=2
        labels[253:285,230:420]=10
        labels[273:275,231:415]=24
        labels[285:293,280:424]=2
        brick=np.zeros(labels.shape,bool)
        boxes=[[257,180,392,245],[419,205,550,310]]
        trial=[648.,538.]
    # The native road has irregular grit. Only the parking row has an independently
    # repeating 2-D paving pattern. A third condition deliberately paves the road too.
    y,x=np.indices((2*h,2*w))
    ground=rng.normal(112,1.6,(2*h,2*w))
    joints=((y%8)<2)|(((x+(y//8%2)*5)%11)<2)
    texture=169-27*joints+rng.normal(0,.8,(2*h,2*w))
    pattern=np.repeat(np.repeat(brick,2,axis=0),2,axis=1)
    if both:pattern[:]=True
    luminance=np.where(pattern,texture,ground)
    color=np.stack([luminance]*3,axis=2).astype("uint8")
    mark=np.repeat(np.repeat(labels==24,2,axis=0),2,axis=1)
    color[mark]=205
    for l,t,r,b in boxes:labels[t:b,l:r]=55
    raw=[[float(2*n) for n in b] for b in boxes]
    context={"size":[2*w,2*h],"current_boxes":raw,
             "current_vehicles":[[i+1,"Car",box] for i,box in enumerate(raw)]}
    return Image.fromarray(color),labels,context,trial


def test_original_joints_adjacent_plain_asphalt_kerb_and_two_vehicle_contacts_support_a_point():
    im,m,c,_=source()
    [o]=method.candidates(im,m,c)
    assert o["evidence"]["surface_type"]=="kerbside_block_paving"
    assert o["evidence"]["flanking_track_ids"]==[1,2]
    assert o["evidence"]["joint_repeat_strength"]>o["evidence"]["neighbouring_road_repeat_strength"]
    assert o["evidence"]["signed_center_class"]==13
    assert "polygon" not in o
    x,y=o["point"]; box=[x-.5,y-.5,x+.5,y+.5]
    assert method.marked(im,m,c,box)["point"]==[x,y]


@pytest.mark.parametrize("fault",["both_paved","no_texture","no_aisle_mark","no_kerb","unbound_vehicle","occupied"])
def test_pavement_in_a_traffic_lane_without_independent_boundaries_is_not_a_parking_patch(fault):
    im,m,c,_=source(both=fault=="both_paved")
    if fault=="no_texture":im=Image.new("RGB",im.size,(110,110,110))
    if fault=="no_aisle_mark":m[m==24]=13
    if fault=="no_kerb":m[np.isin(m,(2,15))]=13
    if fault=="unbound_vehicle":c["current_vehicles"].pop()
    if fault=="occupied":c["current_boxes"].append([450,480,730,594])
    assert method.candidates(im,m,c)==[]
    assert method.marked(im,m,c,[598.,520.,602.,524.]) is None


def test_user_mark_must_stay_at_exact_indicated_location_and_respect_source_scale():
    im,m,c,_=source()
    assert method.marked(im,m,c,[50,680,52,682]) is None
    assert method.marked(im,m,c,[490,480,657,536]) is None
    assert method.marked(im,m,c,[float("nan"),0,1,2]) is None
    assert method.marked(im,m,c,["x",0,1,2]) is None
    assert method.candidates(im,m,{**c,"size":[800,600]})==[]


def test_signed_side_entry_can_show_an_intended_spot_despite_a_wheelstop_in_front():
    im,m,c,point=source("lateral")
    [o]=method.candidates(im,m,c)
    assert o["evidence"]["surface_type"]=="painted_side_entry"
    assert o["evidence"]["signed_center_class"]==10
    assert o["evidence"]["optically_corroborated_paint_cells"]>0
    x,y=point;v=method.marked(im,m,c,[x-1,y-1,x+1,y+1]);assert v["point"]==point


@pytest.mark.parametrize("fault",["sealed_mouth","single_cell_road","passing_bus","merely_paint","no_optical_paint","broad_mark"])
def test_a_bumper_strip_or_blocked_lateral_access_is_not_an_open_side_entry(fault):
    im,m,c,point=source("lateral")
    if fault=="sealed_mouth":m[245:290,210:230]=2
    if fault=="single_cell_road":m[245:290,208:229]=2
    if fault=="passing_bus":c["current_boxes"].append([350,470,450,620])
    if fault=="merely_paint":m[m==10]=13
    if fault=="no_optical_paint":im=Image.new("RGB",im.size,(115,115,115))
    if fault=="broad_mark":
        assert method.marked(im,m,c,[500,500,760,570]) is None
        return
    assert method.candidates(im,m,c)==[]


@pytest.mark.parametrize("kind",["pavers","lateral"])
def test_visible_bollard_slot_narrower_than_a_vehicle_does_not_create_an_entrance(kind):
    from PIL import ImageDraw
    im,m,c,_=source(kind)
    [o]=method.candidates(im,m,c)
    e=o["evidence"]
    lines=e.get("lateral_road_contacts_grid", e.get("aisle_mark_grid"))
    y=lines[1][1];x=min(row[0] for row in lines)-6
    middle=e["required_entrance_rows"]//2-2
    rectangles=[[x-4,y-65,x+4,y-middle-1],[x-4,y+middle+1,x+4,y+65]]
    for l,t,r,b in rectangles:
        m[t:b+1,l:r+1]=2
        native=[2*n for n in (l,t,r,b)]
        c["current_boxes"].append(native)
        ImageDraw.Draw(im).rectangle(native,fill=(143,147,142))
    px,py=o["point"]
    assert method.marked(im,m,c,[px-1,py-1,px+1,py+1]) is None
    assert method.candidates(im,m,c)==[]


@pytest.mark.parametrize("kind",["pavers","lateral"])
def test_a_green_planting_lattice_cannot_substitute_for_visible_parking_pavement(kind):
    im,m,c,_=source(kind)
    [o]=method.candidates(im,m,c)
    xx,yy=map(int,o["point"])
    arr=np.asarray(im).copy()
    y,x=np.indices(arr.shape[:2])
    patch=(abs(y-yy)<80)&(abs(x-xx)<190)
    seedlings=(((x%13)<5)&((y%9)<7))
    arr[patch]=[132,113,73];arr[patch&seedlings]=[51,147,47]
    img=Image.fromarray(arr)
    assert method.marked(img,m,c,[xx-1,yy-1,xx+1,yy+1]) is None
    assert method.candidates(img,m,c)==[]
