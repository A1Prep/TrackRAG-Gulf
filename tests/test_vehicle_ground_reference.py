import numpy as np
import pytest

from trackrag.dropoff import vehicle_curb_ground, vehicle_ground_reference as reference, vehicle_ground_view


@pytest.fixture
def scene():
    return {'frame':{'width':400, 'height':300, 'frame_id':1, 'recording_id':'study'},
            'tracks':[{'frame_id':1, 'track_id':2, 'class_name':'car', 'bbox_xyxy':[250,120,320,200]}],
            'provenance':{'image_sha256':'1'*64}}


@pytest.fixture
def pavement():
    labels = np.full((300,400), 13, dtype='uint8')
    labels[:, 331:] = 15
    labels[207:238, 246:249] = 24
    labels[207:238, 249:300] = 10
    return labels


def test_vehicle_remains_visible_when_road_or_obstacle_invalidates_the_ground(scene, pavement):
    measured = vehicle_curb_ground.describe(scene, pavement)
    assert len(measured) == 1 and measured[0]['ground']['basis'] == 'curb_and_lane_divider'
    no_paint = pavement.copy()
    no_paint[np.isin(no_paint,(10,24))] = 13
    assert vehicle_curb_ground.describe(scene, no_paint)[0]['ground'] is None
    scene['tracks'].append({'frame_id':1,'track_id':3,'class_name':'person','bbox_xyxy':[249,200,332,262]})
    assert vehicle_curb_ground.describe(scene, pavement)[0]['ground'] is None


def test_geometry_tolerates_transverse_line_without_snapping_to_lane_or_crossing(scene, pavement):
    spec = vehicle_curb_ground.describe(scene, pavement)[0]
    polygon = spec['ground']['polygon']
    x,y = spec['ground']['point']
    lower = max(p[1] for p in polygon)
    assert reference._accepts_mark(spec, [x,lower+1], scene['frame'], pavement)
    assert not reference._accepts_mark(spec, [x,lower+9], scene['frame'], pavement)
    assert not reference._accepts_mark(spec, [min(p[0] for p in polygon)-2,y], scene['frame'], pavement)
    pavement[int(y),int(x)] = 23
    assert not reference._accepts_mark(spec, [x,y], scene['frame'], pavement)


def test_the_same_vehicle_gives_distinct_rear_and_side_responses(scene,pavement):
    spec = vehicle_curb_ground.describe(scene,pavement)[0]
    words = {'kind':'car','color':'white','relation':'behind','side':None}
    args = ({'scene':scene}, words, [spec], [['white','suv','REAR','CURBSIDE']], pavement)
    boxes, ground, reason, _ = reference.decision(*args)
    assert boxes == [spec['box']] and ground and reason == reference.CHOICE_REASON
    words['relation'] = 'beside'
    boxes, ground, _, message = reference.decision(*args)
    assert boxes == [spec['box']] and ground == [] and 'beside' in message


def test_ambiguous_paint_does_not_disappear_when_only_one_car_has_a_patch(scene,pavement):
    spec = vehicle_curb_ground.describe(scene,pavement)[0]
    other = {**spec, 'box':[40,130,100,201], 'ground':None}
    reading = ['white','suv','REAR','CURBSIDE']
    args = ({'scene':scene}, {'kind':'car','color':'white','relation':'behind','side':None},
            [spec,other], [reading,['white','sedan','REAR','NONE']], pavement)
    boxes, ground, _, message = reference.decision(*args)
    assert len(boxes) == 2 and not ground and 'More than one' in message


@pytest.mark.parametrize(('orientation','ring'), [('FRONT','CURBSIDE'), ('SIDE','CURBSIDE'), ('REAR','LANE'), ('REAR','BLOCKED')])
def test_orientation_and_ground_obstruction_can_independently_withhold_point(scene,pavement,orientation,ring):
    options = vehicle_curb_ground.describe(scene,pavement)
    ref = {'kind':'car','color':'white','relation':'behind','side':None}
    boxes, ground, _, _ = reference.decision({'scene':scene},ref,options,[['white','suv',orientation,ring]],pavement)
    assert len(boxes) == 1 and not ground


def test_wrong_body_and_colour_are_not_substituted_for_destination(scene,pavement):
    options = vehicle_curb_ground.describe(scene,pavement)
    ref = {'kind':'car','color':'white','relation':'behind','side':None}
    _, ground, _, _ = reference.decision({'scene':scene},ref,options,[['cream','sedan','REAR','CURBSIDE']],pavement)
    assert not ground
    boxes, ground, _, message = reference.decision({'scene':scene},ref,options,[['white','van','REAR','CURBSIDE']],pavement)
    assert boxes and not ground and 'van' in message


def test_the_reader_never_invents_ground_from_an_incomplete_response():
    assert vehicle_ground_view.parse('1 | white | SUV | REAR | CURBSIDE\n2 | red | tank | REAR | CURBSIDE',2) == [
        ['white','suv','REAR','CURBSIDE'], ['unknown','unknown','UNCLEAR','UNCLEAR']]


def test_surface_point_and_reference_are_bound_to_passenger_words():
    actual = {'text':'Could you drop me behind the white car?'}
    assert reference.direct_reference(actual)['relation'] == 'behind'
    assert reference.direct_reference({'text':'Tell me the price behind the white car.'}) is None
    assert reference.direct_reference({'text':'Stop behind the white car or the red one.'}) is None
