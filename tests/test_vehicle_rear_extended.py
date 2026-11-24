from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from trackrag.dropoff import vehicle_ground_reference as reference, vehicle_profile_ground as profile
from trackrag.dropoff import vehicle_rear_marked_strip as marked


def parking_profile(*, paint=True, neighbour=True):
    w, h = 500, 300
    labels = np.full((h,w), 13, np.uint8)
    labels[:,250:] = 10
    pixels = np.full((h,w,3), 60, np.uint8)
    if paint:
        labels[:,247:250] = 24
        pixels[:,247:250] = 200
    source = BytesIO()
    Image.fromarray(pixels).save(source,format='PNG')
    photo = source.getvalue()
    anchor = {'track_id':3,'class':'car','box':[320.,137.,444.,190.]}
    boxes = [(3,anchor['box'])]
    if neighbour:
        boxes.append((7,[259.,137.,305.,175.]))
    tracks = [{'track_id':key,'class_name':'car','bbox_xyxy':box,'frame_id':7,'recording_id':'test'} for key,box in boxes]
    scene = {'source':'predicted','frame':{'width':w,'height':h,'frame_id':7,'recording_id':'test'},
             'tracks':tracks,'provenance':{'image_sha256':sha256(photo).hexdigest()}}
    return photo,labels,scene,anchor


def test_reflected_tail_and_pictured_seam_land_past_the_bumper():
    photo,labels,scene,anchor = parking_profile()
    point, = profile.candidates(photo,labels,scene,anchor,facing='SIDE',tail='LEFT')
    x,y = point['point']
    assert 250 < x < anchor['box'][0] and y > anchor['box'][3]
    assert point['evidence']['nearby_parked_track_id'] == 7 and point['evidence']['painted_boundary_rows'] >= 8
    assert profile.candidates(photo,labels,scene,anchor,facing='SIDE',tail='RIGHT') == []
    assert profile.candidates(photo,labels,scene,anchor,facing='SIDE',tail='UNCLEAR') == []
    labels[190:206,285:314] = 13
    assert profile.candidates(photo,labels,scene,anchor,facing='SIDE',tail='LEFT') == []


@pytest.mark.parametrize(('paint','neighbour'), [(False,True), (True,False)])
def test_a_classifier_label_or_one_parked_vehicle_cannot_establish_an_apron(paint,neighbour):
    values = parking_profile(paint=paint,neighbour=neighbour)
    assert profile.candidates(*values,facing='SIDE',tail='LEFT') == []


@pytest.mark.parametrize('barrier', ['tyres','narrow_opening','continuous'])
def test_raised_kerb_at_the_tyres_or_short_pedestrian_gap_withholds(barrier):
    photo,labels,scene,anchor = parking_profile()
    if barrier=='tyres':
        labels[175:181,259:306]=2
    else:
        labels[:,244:252]=2
        if barrier=='narrow_opening':
            labels[188:202,244:247]=13
            labels[188:202,247:250]=24
            labels[188:202,250:252]=10
    assert profile.candidates(photo,labels,scene,anchor,facing='SIDE',tail='LEFT') == []


def test_vehicle_shade_and_landmark_clarifications_keep_the_requested_detail():
    ref = reference.direct_reference({'text':'Could you drop me behind the beige sedan?'})
    assert ref['color']=='beige' and ref['kind']=='sedan'
    assert reference.direct_reference({'text':'Drop me behind the beige or black sedan'}) is None
    spec = {'box':[50,10,115,80]}
    def crops(first,second):
        return {'calls':[{'reply':first},{'reply':second}]}
    neutral=crops('silver | sedan | REAR','gray | sedan | REAR')
    assert reference._possible_light_vehicle(ref,[spec],{'body':{'rows':[neutral]}},200)==spec['box']
    assert reference._possible_light_vehicle(ref,[spec,spec],{'body':{'rows':[neutral,neutral]}},200) is None
    dark=crops('black | sedan | REAR','gray | sedan | REAR')
    assert reference._possible_light_vehicle(ref,[spec],{'body':{'rows':[dark]}},200) is None
    shop=reference.direct_reference({'text':'Drop me behind the red sedan by the book shop'})
    assert reference._unread_shop(shop)=='book shop'
    assert reference._unread_shop({**shop,'color':None}) is None
    assert reference._unread_shop({**shop,'qualifiers':[{'field':'ground','value':'driveway'}]}) is None


@pytest.mark.parametrize(('obstruction','accepted'), [(None,True),('cross_fence',False),('walking_gap',False)])
def test_marked_row_requires_continuous_bumper_ground_and_an_open_street_edge(obstruction,accepted):
    labels = np.full((240,400),13,np.uint8)
    for y in range(120,211):
        lane = 226+round(.75*(y-145))
        labels[y,lane:lane+48] = 10
        labels[y,lane+48:] = 15
        if obstruction=='cross_fence' and y==152:
            labels[y,lane:lane+48]=5
        if obstruction=='walking_gap' and not 154<=y<=164:
            labels[y,lane-2:lane+2]=5
    view = SimpleNamespace(classes=labels,occupied=np.zeros_like(labels,bool),road=np.isin(labels,(10,13,24)),
                           rgb=np.zeros((*labels.shape,3),np.uint8),bumper=(250,145),vehicle_size=(40,35),
                           spec={'box':[230,110,270,145]},original=Image.new('RGB',(400,240)))
    measured=marked._parking_crossbar(view,np.array([264,160]),np.array([238,161]),np.array([286,161]))
    assert (measured is not None) == accepted
