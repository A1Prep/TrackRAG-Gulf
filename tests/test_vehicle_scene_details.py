import cv2
import numpy as np
import pytest
from io import BytesIO
from PIL import Image

from trackrag.dropoff import vehicle_ground_reference as vehicles, vehicle_scene_details as details


def match(words, paint='black', landmark='bins', *, position=(150,67), extra=None):
    pixels = np.full((110,200), 13, np.uint8)
    x,y = position
    if landmark == 'bins':
        pixels[y:y+8,x:x+8] = 51
    if landmark in {'circle','triangle'}:
        if landmark == 'circle':
            cv2.circle(pixels, (x,y), 8, 50, -1)
        else:
            cv2.fillPoly(pixels,[np.int32([[x,y-10],[x-8,y+5],[x+8,y+5]])],50)
    first = {'track_id':2,'box':[114.,58.,148.,91.], 'class':'car'}
    specs = [first] + ([extra] if extra else [])
    scene = {'provenance':{'network_policy':'local_only'},'frame':{'width':200,'height':110}}
    req = {'text':words,'scene':scene,'dialogue':[]}
    return details.matching(req,vehicles.direct_reference(req),specs,[[paint,'pickup','REAR','NONE']]*len(specs),pixels)


@pytest.mark.parametrize('landmark,at,colour,found',[
    ('bins',(150,67),'black',True),('bins',(32,64),'black',False), ('none',(150,67),'black',False),
    ('bins',(150,67),'white',False), ('bins',(135,6),'black',False)])
def test_same_side_nearby_measured_bin_and_observed_dark_paint(landmark,at,colour,found):
    rows = match('Drop me behind the dark pickup by the bins on the right.',colour,landmark,position=at)
    assert (len(rows)==1) is found


def test_the_requested_shape_must_be_present_in_the_sign_observation():
    request='Please drop me behind the black pickup near the triangular road sign.'
    assert match(request,landmark='triangle')
    assert not match(request,landmark='circle')
    assert not match(request,landmark='none')
    assert not match('Please drop me behind the black pickup near the pharmacy.',landmark='triangle')


def test_stacked_information_plate_can_share_a_label_with_a_triangular_sign():
    colour = np.full((100,100,3),190,np.uint8)
    contour = np.int32([[[30,18]],[[58,18]],[[58,80]],[[30,80]]])
    def encoded():
        stream=BytesIO(); Image.fromarray(colour).save(stream,format='PNG');return stream.getvalue()
    assert not details._triangular_panel(encoded(),contour,(100,100))
    cv2.polylines(colour,[np.int32([[44,19],[31,40],[57,40]])],True,(190,24,42),3)
    assert details._triangular_panel(encoded(),contour,(100,100))


def test_kerb_and_row_require_measured_ground_details():
    [[_,_,kerb]]=match('Let me out behind the dark pickup along the right kerb.')
    assert not details.ground_confirms({'curb_side':'left'},kerb)
    assert details.ground_confirms({'curb_side':'right'},kerb)
    [[_,_,row]]=match('Let me out behind the black pickup in the row on the right.')
    assert not details.ground_confirms({'curb_side':'right'},row)
    assert details.ground_confirms({'aligned_neighbour_id':3},row)


def test_wall_at_car_depth_and_on_named_side_is_required():
    car = [119,55,155,89]
    frame = {'width':200,'height':110}
    mask = np.full((110,200),13,np.uint8)
    mask[65:88,:31] = 6
    mask[2:18,141:] = 6
    assert not details._near_landmark(mask,'wall',car,frame,'right',None)
    mask[62:90,154:178] = 6
    assert details._near_landmark(mask,'wall',car,frame,'right',None)
    assert not details._near_landmark(mask,'wall',car,frame,'left',None)


@pytest.mark.parametrize('rgb,surface,nearer,expected', [
    ((22,43,87),55,[158,65,191,102],True),
    ((135,169,242),55,[158,65,191,102],False),
    ((42,42,44),55,[158,65,191,102],False),
    ((22,43,87),13,[158,65,191,102],False),
    ((22,43,87),55,[28,65,61,102],False),
    ((22,43,87),55,[158,46,191,64],False),
])
def test_dark_blue_vehicle_farther_up_needs_body_paint_and_a_nearer_same_side_vehicle(rgb,surface,nearer,expected):
    mask = np.full((110,200),13,np.uint8)
    mask[52:78,118:152] = surface
    image = Image.new('RGB',(200,110),rgb)
    stream = BytesIO()
    image.save(stream,format='PNG')
    specs = [{'track_id':3,'box':[118,52,152,78]}, {'track_id':5,'box':nearer}]
    scene = {'provenance':{'network_policy':'local_only'},'frame':{'width':200,'height':110}}
    request = {'text':'Drop me behind the dark blue sedan further up the right-hand side.','scene':scene,'dialogue':[]}
    rows = details.matching(request,vehicles.direct_reference(request),specs,
                            [['blue','sedan','REAR','NONE'],['white','suv','REAR','NONE']],
                            mask,original=stream.getvalue())
    assert [s['track_id'] for s,_,_ in rows] == ([3] if expected else [])
