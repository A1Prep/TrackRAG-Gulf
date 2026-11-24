"""A fixture-signed single candidate can be confirmed as a vehicle, never as a destination."""
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from test_passenger_bay_evidence import bank, seal
from tools.dropoff_passenger_handoff import session as audit_session
from trackrag.dropoff import manual_offlane, vehicle_curb_ground, vehicle_ground_reference, vehicle_ground_view
from trackrag.dropoff import vehicle_identity_reply as identity
from trackrag.dropoff.interaction import PassengerSession
from trackrag.dropoff.passenger import unresolved


QUESTION = 'Could you drop me behind the dark sedan on the right?'
PICTURED = 'I can see a black sedan left of centre in the photo. Is that the vehicle you meant by “on the right”?'


@pytest.fixture
def interaction(bank, tmp_path):
    specs = vehicle_curb_ground.describe(bank.scene, bank.classes)
    assert {s['track_id'] for s in specs} == {4,7}
    body = {'source':bank.scene['provenance']['image_sha256'], 'recipe':vehicle_ground_view.RECIPE,
            'model':vehicle_ground_reference.MODEL, 'revision':vehicle_ground_reference.REVISION,
            'prompt':vehicle_ground_view.checksum(vehicle_ground_view.GUIDE), 'spec':vehicle_ground_view.checksum(specs),
            'view':sha256(vehicle_ground_view.render(bank.image, specs)).hexdigest(), 'count':len(specs),
            'answers':[['black','sedan','REAR','NONE'] if s['track_id'] == 4 else ['white','van','SIDE','NONE'] for s in specs]}
    packet = seal(body, bank.key)
    calls = []
    def interpret(req):
        calls.append(req)
        if req['text'] == QUESTION:
            return vehicle_ground_reference.outcome(req, bank.photo.read_bytes(), bank.surface, packet,
                                                    qualifier_packets={}, load_side_packet=False)
        return {'ok':True, 'request_id':req['request_id'], 'result':unresolved('Describe another landmark.'),
                'provenance':{'request_sha256':identity.digest(req),'image':{'sha256':body['source']}}}
    model = SimpleNamespace(interpret=interpret, passenger_area=manual_offlane.select, requires_verified_manual_surface=True)
    session = PassengerSession(bank.scene, model, audit_dir=tmp_path/'sessions')
    first = session.submit(QUESTION)
    assert first.get('reference_kind') == 'possible_vehicle', first
    assert first['result']['clarification_question'] == PICTURED
    return session, first, calls


@pytest.mark.parametrize('text',['Yes','Yes, please','That’s the one','Yes, that’s the sedan I meant','Yes, that vehicle'])
def test_yes_binds_the_verified_box_and_original_relation(interaction, text):
    session, first, calls = interaction
    reply = session.submit(text)
    assert len(calls) == 1 and reply['references_only'] and not reply.get('reference_kind')
    assert reply['result']['anchor_boxes_xyxy'] == first['result']['anchor_boxes_xyxy']
    assert 'behind it' in reply['result']['clarification_question']
    assert not reply['proposal_id'] and not reply['result']['destination_point_xy']
    assert [t['text'] for t in reply['turns'][-3:-1]] == [PICTURED,text]
    event = audit_session(session._journal)['events'][-1]['outcome']
    assert event['place_verification']['vehicle_selected_by'] == 'passenger'
    assert event['native_model_calls'] == 0 and not event['independent_model_proposal']
    assert not session.is_vehicle_identity_reply('yes')


@pytest.mark.parametrize('text',['No','No, another car','No, thanks','No, that’s not it','No, I meant the white SUV',
                                  'Yes, that’s the truck I meant'])
def test_decline_or_different_named_body_removes_anchor(interaction, text):
    session, _, calls = interaction
    reply = session.submit(text)
    assert len(calls) == 1 and reply['state'] == 'needs_clarification'
    assert not reply.get('references_only') and not reply['result'].get('anchor_boxes_xyxy') and not reply['proposal_id']
    assert not session.is_vehicle_identity_reply('Yes')
    fresh = session.submit('Drop me next to the white SUV.')
    assert len(calls) == 2 and fresh['turns'][-2]['text'] == 'Drop me next to the white SUV.'


@pytest.mark.parametrize('text',['Correct?', 'the sedan?', 'Yes, number two', 'Yes 2', 'Yes, the red car',
                                  'Yes, but the one on the left','No, stop on the crossing','Drop me behind the van'])
def test_unasked_changes_and_qualified_or_uncertain_replies_do_not_confirm(text):
    assert identity.reply_to(text, PICTURED) is None


def test_different_scene_intervening_exchange_and_replacement_cannot_reuse_question(interaction):
    session, first, _ = interaction
    prior, observed = session._vehicle_question
    req = session.prepare_automatic('Yes')
    assert identity.answer(req,prior,observed)
    for altered in ({**req, 'scene':{**req['scene'],'tracks':[]}},
                    {**req, 'dialogue':req['dialogue'][:-1]},
                    {**req, 'dialogue':[*req['dialogue'], *req['dialogue']]}):
        assert identity.answer(altered,prior,observed) is None
    assert not session.is_vehicle_identity_reply('Yes', replaces_request_id=first['request_id'])
    session.cancel()
    assert not session.is_vehicle_identity_reply('Yes')
    assert not session.prepare_automatic('Yes')['dialogue']


@pytest.mark.parametrize('changed',['box','actor','source','prior_id','relation','destination'])
def test_replay_rejects_forged_source_choice_actor_or_pin(interaction, changed):
    session, _, _ = interaction
    session.submit('Yes')
    rows = [json.loads(line) for line in session._journal.read_bytes().splitlines()]
    req, out = rows[-1]['detail']['request'], rows[-1]['detail']['outcome']
    p = out['provenance']
    if changed == 'box': out['result']['anchor_boxes_xyxy'][0][0] += 4
    if changed == 'actor': p['vehicle_selected_by'] = 'model'
    if changed == 'source': p['reference_outcome']['provenance']['vehicle_observation']['body']['answers'][0][0] = 'red'
    if changed == 'prior_id': p['reference_request']['request_id'] = 'different'
    if changed == 'relation': p['requested_relation'] = 'beside'
    if changed == 'destination': out['result']['destination_point_xy'] = [100,100]
    with pytest.raises(ValueError):
        identity.verify(req,out)


def test_an_unrelated_new_location_after_yes_needs_point_selection_and_manual_save(interaction, bank):
    session, _, _ = interaction
    session.submit('Yes')
    # One point on signed connected parking and one in the through lane.
    center = bank.options[0]['point']
    x,y = center
    box = [x-1,y-1,x+1,y+1]
    marked = session.select_area(box)
    assert marked['state'] == 'needs_clarification' and marked['proposal_id'] is None, marked
    choice, = marked['bay_choices']
    assert choice['center_xy'] == center and choice['extent'] == 'interior_point_only'
    assert 'couldn’t verify that it is behind' in marked['result']['clarification_question'].lower()
    assert marked['result']['destination_point_xy'] is None
    picked = session.select_bay(choice['id'])
    assert picked['selected_by'] == 'passenger' and picked['state'] == 'proposed'
    assert session.confirm(picked['proposal_id'])['state'] == 'confirmed'
    assert audit_session(session._journal)['events'][-1]['phase'] == 'confirmed_previous_selection'


def test_lane_and_occupied_marks_after_yes_keep_the_vehicle_but_offer_no_destination(interaction):
    session, first, _ = interaction
    session.submit('Yes')
    left,top,right,bottom = first['result']['anchor_boxes_xyxy'][0]
    for x,y in [(10,170), ((left+right)/2,(top+bottom)/2)]:
        state = session.select_area([x-1,y-1,x+1,y+1])
        assert state['state'] == 'needs_clarification' and state['proposal_id'] is None, state
        assert not state.get('bay_choices') and state['result']['anchor_boxes_xyxy'] == first['result']['anchor_boxes_xyxy']
        assert not state['result']['destination_point_xy']
