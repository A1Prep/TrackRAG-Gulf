"""Exercise the passenger confirmation and replay using an isolated geometry fixture."""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest

from test_passenger_bay_evidence import bank, seal
from test_passenger_manual_offlane import provide_map
from test_vehicle_rear_ground import parked_row
from trackrag.dropoff import vehicle_curb_ground, vehicle_ground_reference, vehicle_ground_view
from trackrag.dropoff import vehicle_rear_ground, whole_bay
from trackrag.dropoff.interaction import PassengerSession


def neutral_readings(bank, colour='white', facing='REAR', target_track=4):
    specs = vehicle_curb_ground.describe(bank.scene, bank.classes)
    view = vehicle_ground_view
    body = {'source': bank.scene['provenance']['image_sha256'], 'recipe': view.RECIPE,
            'model': vehicle_ground_reference.MODEL, 'revision': vehicle_ground_reference.REVISION,
            'prompt': view.checksum(view.GUIDE), 'spec': view.checksum(specs), 'count': len(specs),
            'view': sha256(view.render(bank.image, specs)).hexdigest(),
            'answers': [[colour, 'sedan', facing, 'NONE'] if spec['track_id'] == target_track
                        else ['black', 'van', 'SIDE', 'NONE'] for spec in specs]}
    return seal(body, bank.key)


def measured_patch():
    return {'point': [50, 136], 'box': [47, 133, 53, 139],
            'polygon': [[47, 133], [53, 133], [53, 139], [47, 139]],
            'evidence': {'recipe': vehicle_rear_ground.RECIPE, 'coverage': 'interior_point_only',
                         'curb_side': 'left', 'anchor_id': 4}}


QUESTION = 'Could you drop me behind the white car?'


def test_first_reply_renders_the_destination_and_requires_separate_confirmation(bank, monkeypatch, tmp_path):
    patch = measured_patch()
    calls = []
    def measure(original, classes, scene, anchor, *, facing):
        assert anchor['track_id'] == 4 and facing == 'REAR'
        calls.append(anchor['box'])
        return [patch]
    monkeypatch.setattr(vehicle_rear_ground, 'candidates', measure)
    packet = neutral_readings(bank)
    def answer(req):
        return vehicle_ground_reference.outcome(req, bank.photo.read_bytes(), bank.surface, packet, measure_rear=True)
    req = {**bank.req, 'text': QUESTION}
    before = vehicle_ground_reference.outcome(req, bank.photo.read_bytes(), bank.surface, packet)
    assert before['result']['destination_point_xy'] is None
    response = answer(req)
    assert whole_bay.verify(req, response)['decision_reproduced']
    altered = deepcopy(response)
    altered['result']['destination_point_xy'][0] += 6
    with pytest.raises(ValueError):
        whole_bay.verify(req, altered)

    session = PassengerSession(bank.scene, SimpleNamespace(interpret=answer), audit_dir=tmp_path/'sessions')
    view = session.submit(QUESTION)
    assert view['state'] == 'proposed' and view['rear_vehicle'] == 'white sedan'
    assert view['result']['destination_point_xy'] == patch['point']
    assert view['result']['anchor_boxes_xyxy'] == [calls[0]]
    assert not view.get('bay_choices')
    assert session.confirm(view['proposal_id'])['state'] == 'confirmed'


def test_registered_first_request_reaches_the_actual_pavement_measurement(bank, tmp_path, monkeypatch):
    im, classes, scene, anchor = parked_row()
    photo = tmp_path/'occupied-row.png'
    im.save(photo)
    sha = sha256(photo.read_bytes()).hexdigest()
    scene['provenance']['image_sha256'] = sha
    scene['frame'].update(image_path=str(photo), timestamp_s=2.0)
    for track in scene['tracks']:
        track.update(recording_id=scene['frame']['recording_id'], source='predicted')
    surface = provide_map(monkeypatch, scene, classes)
    example = SimpleNamespace(scene=scene, classes=classes, image=im, key=bank.key)
    packet = neutral_readings(example, target_track=anchor['track_id'])
    monkeypatch.setattr(vehicle_ground_reference, 'read_photo_packet', lambda *args: packet)
    model = SimpleNamespace(interpret=vehicle_ground_reference.available)
    session = PassengerSession(scene, model, audit_dir=tmp_path/'journal')
    view = session.submit(QUESTION)
    assert view['state'] == 'proposed' and not view.get('bay_choices')
    x, y = view['result']['destination_point_xy']
    assert y > anchor['box'][3] and classes[round(y),round(x)] == 13
    assert view['result']['anchor_boxes_xyxy'] == [anchor['box']]
    assert session.confirm(view['proposal_id'])['state'] == 'confirmed'


@pytest.mark.parametrize('colour,facing,words', [('black', 'REAR', QUESTION), ('white', 'FRONT', QUESTION),
    ('white', 'SIDE', QUESTION), ('white', 'REAR', 'Could you drop me beside the white car?')])
def test_unmatched_or_wrong_relation_cannot_acquire_an_automatic_rear_point(bank, monkeypatch, colour, facing, words):
    def unexpected(*args, **kwargs):
        pytest.fail('Rear ground was predicted for the wrong vehicle or relation.')
    monkeypatch.setattr(vehicle_rear_ground, 'candidates', unexpected)
    req = {**bank.req, 'text': words}
    response = vehicle_ground_reference.outcome(req, bank.photo.read_bytes(), bank.surface,
        neutral_readings(bank, colour, facing), measure_rear=True, load_side_packet=False)
    assert response['result']['destination_point_xy'] is None
    assert not response['provenance'].get('automatic_rear_ground')


def test_a_passenger_lane_tap_cannot_trigger_the_automatic_rear_fallback(bank, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('A manual mark was moved to an unrelated automatic destination.')
    monkeypatch.setattr(vehicle_rear_ground, 'candidates', unexpected)
    req = {**bank.req, 'text': QUESTION}
    assert bank.classes[202, 180] == 13
    response = vehicle_ground_reference.outcome(req, bank.photo.read_bytes(), bank.surface, neutral_readings(bank),
        [178, 200, 182, 204], measure_rear=True)
    assert response['result']['destination_point_xy'] is None
    assert not response['provenance']['bay_choices']
