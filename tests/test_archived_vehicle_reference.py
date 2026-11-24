"""Confirm identity from a previous visual reading without treating the old ground as a destination."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from trackrag.dropoff import archived_vehicle_reference as archive
from trackrag.dropoff import manual_surfaces, vehicle_ground_reference as reference, vehicle_identity_reply, whole_bay
from trackrag.dropoff.bay_packets import read_photo_packet


BOX = [10, 20, 65, 70]
OTHER = [130, 22, 180, 72]
REF = {'relation':'beside', 'kind':'pickup', 'color':'black', 'side':None}
READING = (4, BOX, 'truck', 'black')


def possible(ref=REF, readings=(READING,), boards=()):
    return archive.possible_vehicle(ref, readings,
        [{'track_id':track, 'box':box} for track, box, _, _ in boards],
        [[paint, body, 'SIDE', 'NONE'] for _, _, paint, body in boards], 200)


def test_coarse_truck_remains_only_a_question():
    box, question = possible()
    assert box == BOX
    assert 'black truck left of centre' in question and 'couldn’t verify that it is a pickup' in question
    assert 'haven’t verified an open space beside it' in question
    assert possible({**REF, 'kind':'truck'}) == (BOX, question.replace(' I couldn’t verify that it is a pickup.', ''))
    assert possible({**REF, 'side':'left'})


@pytest.mark.parametrize('changed', [
    {'color':'red'}, {'color':None}, {'kind':'suv'}, {'side':'right'},
    {'qualified':True}, {'unsupported':True}, {'ground_side':'left'}, {'relation':'front'},
])
def test_colour_type_side_and_unsupported_details_never_transfer(changed):
    assert possible({**REF, **changed}) is None


def test_competing_or_conflicting_vehicles_are_withheld():
    assert possible(readings=[READING, (5, OTHER, 'truck', 'black')]) is None
    assert possible(boards=[(5, OTHER, 'black', 'truck')]) is None
    assert possible(boards=[(5, OTHER, 'unknown', 'unknown')]) is None
    assert possible(boards=[(4, BOX, 'black', 'suv')]) is None
    assert possible(boards=[(4, BOX, 'white', 'truck')]) is None
    assert possible(boards=[(4, OTHER, 'black', 'truck')]) is None
    assert possible(readings=[(4, BOX, 'truck', 'unknown')]) is None
    assert possible(boards=[(5, OTHER, 'black', 'suv')]) is not None


def test_all_archived_packets_are_authenticated_in_the_pinned_public_release():
    from trackrag.dropoff.parking_evidence import _authenticate
    assert sha256(archive.BANK.read_bytes()).hexdigest() == archive.BANK_SHA256
    with ZipFile(archive.BANK) as bank:
        assert len(bank.namelist()) == 12
        for name in bank.namelist():
            packet = archive.get_packet(Path(name).stem)
            _authenticate(packet)
            assert packet['body']['image_sha256'] == Path(name).stem
            assert packet['body']['models'] == archive.single_vehicle.MODELS


@pytest.fixture(scope='module')
def real_source():
    root = Path(__file__).resolve().parents[1] / 'data/uae-parking-study'
    bank = root / 'passenger-requests.jsonl'
    if not bank.exists():
        pytest.skip('Public original photo bundle is optional on a fresh checkout.')
    for line in bank.read_bytes().splitlines():
        scene = json.loads(line)['scene']
        frame = scene['frame']
        if (frame['recording_id'], frame['frame_id']) == ('kartaview_5124809', 287):
            break
    else:
        pytest.skip('The public pickup original has not been installed.')
    frame['image_path'] = str((root / frame['image_path']).resolve())
    if not Path(frame['image_path']).is_file():
        pytest.skip('The public pickup original has not been installed.')
    key = scene['provenance']['image_sha256']
    request = {'scene':scene, 'text':'Could you drop me next to that black pickup truck?',
               'request_id':'public-vehicle-test', 'input_mode':'typed', 'dialogue':[]}
    return request, Path(frame['image_path']).read_bytes(), key


def test_real_crop_replays_and_yes_confirms_only_identity(real_source, monkeypatch):
    request, photo, key = real_source
    surface = manual_surfaces.get_packet(key)
    board = read_photo_packet(reference.BANK, key, 8192)
    old = reference.outcome(request, photo, surface, board)
    assert old['provenance']['reason'] == 'vehicle_reference_unmatched'
    assert whole_bay.verify(request, old)['decision_reproduced']
    outcome = reference.available(request)
    result = outcome['result']
    target = next(t['bbox_xyxy'] for t in request['scene']['tracks'] if t['track_id'] == 4)
    assert result['anchor_boxes_xyxy'] == [target]
    assert vehicle_identity_reply.is_question(request, outcome)
    assert result['destination_point_xy'] is None and result['destination_box_xyxy'] is None
    assert not outcome['provenance'].get('bay_choices')
    monkeypatch.setattr(archive, 'get_packet', lambda *_: pytest.fail('Replay must use the published decision receipt.'))
    assert whole_bay.verify_current(request, outcome)['decision_reproduced']
    location = {k: request['scene']['frame'][k] for k in ('recording_id','frame_id')}
    following = {**request, 'text':'Yes', 'request_id':'reply',
                 'dialogue':[{'role':'user', 'text':request['text'], **location},
                             {'role':'assistant', 'text':result['clarification_question'], **location}]}
    agreed = vehicle_identity_reply.answer(following, request, outcome)
    assert agreed['result']['anchor_boxes_xyxy'] == [target]
    assert agreed['result']['status'] == 'needs_clarification' and agreed['result']['destination_point_xy'] is None
    assert vehicle_identity_reply.verify_current(following, agreed)['vehicle_selected_by'] == 'passenger'
    declined = vehicle_identity_reply.answer({**following, 'text':'No'}, request, outcome)
    assert not declined['result'].get('anchor_boxes_xyxy') and declined['result']['destination_point_xy'] is None


@pytest.mark.parametrize('mutation', ['another_photo','box','future','tampered_signature','tampered_crop'])
def test_original_frame_track_and_rendered_crop_are_individually_bound(real_source, monkeypatch, mutation):
    request, photo, key = real_source
    request = deepcopy(request)
    packet = archive.get_packet(key)
    if mutation == 'another_photo':
        with ZipFile(archive.BANK) as bank:
            packet = archive.get_packet(next(Path(n).stem for n in bank.namelist() if Path(n).stem != key))
    elif mutation == 'box':
        request['scene']['tracks'][0]['bbox_xyxy'][0] += 2
    elif mutation == 'future':
        request['scene']['tracks'][0]['frame_id'] += 1
    elif mutation == 'tampered_signature':
        packet['signature'] = '0' * len(packet['signature'])
    elif mutation == 'tampered_crop':
        # Exercise the independent pixel/track verifier even if outer authentication were bypassed.
        monkeypatch.setattr(archive, '_authenticate', lambda _: None)
        packet['body']['calls'][0]['rendered_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        archive.clarify(request, photo, REF, [], [], packet)


def test_local_only_photo_never_consults_public_identity(real_source, monkeypatch):
    request, photo, key = real_source
    private = deepcopy(request)
    private['scene']['provenance']['network_policy'] = 'local_only'
    monkeypatch.setattr(archive, '_authenticate', lambda *_: pytest.fail('Public crops were accessed'))
    assert archive.clarify(private, photo, REF, [], [], archive.get_packet(key)) is None
