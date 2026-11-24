import cv2
import numpy as np
import pytest

from trackrag.dropoff import curb_alternatives as alternatives


@pytest.mark.parametrize('text',[
    'Use the striped travel lane beside the parking sign.',
    'Use a free parking space near the wall.',
    'Free parking by the wall please.',
    'Only use the lay-by if stopping there is legal.',
    'Use a parking space in the middle of the road by the sign.',
    'Use a space near the wall but not the fence.',
    'Use the space near the wall before the crossing.',
    'Please check if the parking sign is safe.',
    'Use a parking space near the marble mosque.',
    'Use the parking space on the left and on the right.',
    'Use a space across from the building.',
    'Use a parking space near the red and blue sign.',
])
def test_unknown_targets_and_conditions_do_not_disappear(text):
    assert alternatives.intent(text) is None


def test_generic_layby_is_distinct_from_an_unspecified_place_or_a_verb():
    assert alternatives.intent('Use the lay-by on the right.')['recess']
    assert alternatives.intent('Use a parking space.') is None
    assert not alternatives.intent('Pull into a parking space near the wall.')['recess']
    assert alternatives.intent('Use the pictured roadside recess outside the bank.') is None


def test_absolute_side_is_independent_of_direction_relative_to_a_vehicle():
    relative=alternatives.intent('Find open gravel to the left of the black SUV.')
    assert relative['direction']=='left' and relative['side'] is None
    absolute=alternatives.intent('Please use a parking space on the right by the wall and palm trees.')
    assert absolute['side']=='right'
    # Here ‘right’ merely emphasises how near the target is.
    assert alternatives.intent('Use a parking space right by the wall.')['side'] is None


def test_named_far_landmark_requires_an_exact_separately_observed_name():
    parsed=alternatives.intent('Use an empty space by the orange parking sign, opposite Town Square.')
    assert parsed['other']=='town square' and parsed['parking_sign'] and parsed['sign_color']=='orange'
    assert alternatives.observed_name('Town Square',['NONE','townsquare | building entrance'])
    assert not alternatives.observed_name('Town Square',['Town Office | Square House'])
    assert not alternatives.observed_name('Town Square',['perhaps Town Square'])


def test_sign_colour_must_come_from_the_pictured_sign():
    labels=np.full((90,90),13,np.uint8)
    labels[5:25,40:65]=50
    hsv=np.zeros((90,90,3),np.uint8)
    hsv[:]=[12,245,220]  # Orange persists across the background.
    hsv[5:25,40:65]=[120,220,220]  # The actual sign is blue.
    image=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    assert alternatives.coloured_signs(labels,image,'blue')
    assert not alternatives.coloured_signs(labels,image,'orange')
    assert alternatives.pictures_landmark('sign',labels,[53,65],[90,90])
    assert not alternatives.pictures_landmark('sign',labels,[13,65],[90,90])


def test_public_wide_and_close_footprint_must_contain_every_corner_of_point():
    def square(x0,x1):
        return {'polygon':[[x0,0],[x1,0],[x1,10],[x0,10]]}
    valid=square(12,18)
    edge=square(9,19)  # The centre sits inside the reviewed footprint but a corner falls outside.
    assert alternatives.enclosed_points([valid,edge],[square(10,30)])==[valid]
    assert alternatives.enclosed_points([valid],[])==[]


def test_existing_unmatched_gallery_request_requires_both_distinct_packets(monkeypatch):
    from trackrag.dropoff import curb_points, photo_clarification, surface_stalls
    request={'text':'Use a parking space by the wall.','scene':{'provenance':{'image_sha256':'actual-photo'}}}
    prior={'result':{'status':'needs_clarification'},'provenance':{'reason':'no_supported_bay_for_instruction'}}
    monkeypatch.setattr(surface_stalls,'get_packet',lambda key: {'review':key})
    monkeypatch.setattr(curb_points,'get_packet',lambda key: None)
    assert not photo_clarification.has_public_alternative(request,prior)
    monkeypatch.setattr(curb_points,'get_packet',lambda key: {'point':key})
    assert photo_clarification.has_public_alternative(request,prior)
    assert not photo_clarification.has_public_alternative({**request,'text':'Use the lane by the sign.'},prior)
    assert not photo_clarification.has_public_alternative(request,{**prior,'result':{'status':'proposed'}})
    prior['provenance']['reason']='separated_parking_surface'
    assert not photo_clarification.has_public_alternative(request,prior)


def test_relative_vehicle_direction_requires_all_plausible_neighbours_to_agree(monkeypatch):
    spec=alternatives.intent('Use the gravel to the right of that parked dark SUV.')
    option={'point':[39,70],'descriptions':{'nearer_vehicle':'silver car','farther_vehicle':'black SUV'}}
    scene={'frame':{'frame_id':5},'tracks':[]}
    tracks=[(1,'car',[10,30,20,60]),(2,'car',[30,35,42,60])]
    monkeypatch.setattr(alternatives.bay_anchors,'closest_tracks',lambda *_:tracks)
    assert alternatives.seen_vehicle(spec,option,scene)
    tracks.append((3,'car',[45,35,60,65]))
    assert not alternatives.seen_vehicle(spec,option,scene)
    option['descriptions']['farther_vehicle']='black SUV with a white logo'
    spec=alternatives.intent('Use the gravel to the right of that white SUV.')
    assert not alternatives.seen_vehicle(spec,option,scene)
