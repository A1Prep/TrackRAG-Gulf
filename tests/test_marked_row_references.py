import cv2
import numpy as np
import pytest

from trackrag.dropoff import marked_row_references as row


@pytest.mark.parametrize('text',[
    'Use a parking space beside the pink wall.',  # No claim that the lines can be traced.
    'Use the marked space by the red wall and pharmacy.',
    'Book an empty marked bay near the pink wall.',
    'Use a free marked space by the wall.',
    'Use the marked space by the wall unless it is in the lane.',
    'Do not use the painted space near the wall.',
    'Use the painted parking on the left and on the right beside the green wall.',
    'Use the marked parking on our right, this side of the red SUV and the flatbed.',
    'Use a marked bay across the road by the gray wall.',
    'Use the reserved painted bay near the red wall.',
])
def test_whole_instruction_required(text):
    assert row.intent(text) is None


def test_apparent_side_and_relative_vehicle_are_independently_kept():
    spec=row.intent('Could I get out in one of the empty marked spaces on the right, this side of the red SUV?')
    assert spec['side']=='right' and spec['vehicle'][:2]==('suv',{'red'})
    assert row.intent('Use an empty painted bay closer to us than the blue truck.')['vehicle'][:2]==('truck',{'blue'})
    assert row.intent('Please use an empty painted parking bay near that green wall.')['colour']=='green'
    assert row.intent('Show me an empty painted parking space on our right.')=={'kind':'marked','side':'right'}
    assert row.intent('Use the marked parking reserved for an ambulance.') is None
    assert row.intent('Use a painted bay by the unknown landmark.') is None


def test_only_a_point_between_two_measured_edges_and_clear_parking_is_usable():
    surface=np.full((90,150),10,np.uint8);road=np.ones_like(surface,bool);occupied=np.zeros_like(road)
    lines=[np.array([[60.,30.],[36.,75.]]),np.array([[93.,32.],[97.,78.]])]
    assert row.bracketing_pair((67.,53.),lines,surface,road,occupied)
    assert not row.bracketing_pair((67.,78.),lines,surface,road,occupied)  # No continuation past the bonnet.
    assert not row.bracketing_pair((115.,53.),lines,surface,road,occupied)
    assert not row.bracketing_pair((67.,53.),lines,surface,~road,occupied)
    occupied[35:68,59:76]=True
    assert not row.bracketing_pair((67.,53.),lines,surface,road,occupied)
    occupied[:]=False;surface[30:80,50:95]=13
    assert not row.bracketing_pair((67.,53.),lines,surface,road,occupied)


def test_wall_colour_comes_from_coherent_wall_and_uncertainty_is_retained():
    surface=np.full((100,200),13,np.uint8);surface[20:60,120:185]=6
    rgb=np.full((100,200,3),[14,50,197],np.uint8);rgb[surface==6]=[238,211,192]
    option={'point':[150,75]}
    assert row.wall_colour(option,surface,rgb,(200,100),'pink')=='pale_warm'
    assert row.wall_colour(option,surface,rgb,(200,100),'blue') is None
    assert row.wall_colour({'point':[45,75]},surface,rgb,(200,100),'pink') is None


def test_pictured_vehicle_colour_cannot_come_from_small_lights_or_neighbor():
    scene={'frame':{'frame_id':2,'width':150,'height':100},'tracks':[
        {'frame_id':2,'track_id':1,'class_name':'car','bbox_xyxy':[87,34,142,67]}]}
    region=np.full((100,150),13,np.uint8);region[34:67,87:142]=55
    options=[{'point':[126,79],'descriptions':{'nearer_vehicle':'a red SUV','farther_vehicle':'a white sedan'}}]
    spec=row.intent('Use an empty marked bay on the right, this side of the red SUV.')
    colour=np.full((100,150,3),[145,141,140],np.uint8);colour[51:54,89:93]=[196,53,50]
    assert row.colour_vehicle(spec,options,scene,region,colour) is None
    colour[45:63,90:139]=[171,47,52]
    assert row.colour_vehicle(spec,options,scene,region,colour)[0]==1
    assert row.colour_vehicle({**spec,'side':'left'},options,scene,region,colour) is None
    options[0]['descriptions']['nearer_vehicle']='a blue SUV'
    assert row.colour_vehicle(spec,options,scene,region,colour) is None
