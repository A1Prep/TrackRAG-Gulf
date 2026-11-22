from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from trackrag.dropoff.landmarks import LandmarkResolver, resolve_image
from trackrag.dropoff.pair_resolution import (PairResolutionError, _digest, collect_pair_evidence,
    explicit_pair_quotes, replay_pair_decision)


EXAMPLE = 'Between the white SUV on our left and the large open-sided truck on our right, just behind them.'


@pytest.mark.parametrize('utterance,words', [
    (EXAMPLE, ['the white SUV on our left', 'the large open-sided truck on our right']),
    ('Between the blue and yellow sign and the narrow entrance please.', ['the blue and yellow sign','the narrow entrance']),
    ('I meant the bench. Actually, between the tree and the kiosk.', ['the tree','the kiosk']),
    ('Not between a car and a truck; instead, between the cafe and the shop.', ['the cafe','the shop']),
    ('Between the tree and the shop instead.', ['the tree','the shop']),
    ('Between the two black cars.', None), ('Between the white and red cars.', None),
    ('Between the top and bottom of the wall.', None), ('Between the bench and the bin or the bike.',None),
    ('Between a bench and a bike and a tree.',None), ('Instead of between a taxi and a van, by that cafe.',None),
    ('Do not drop me between the two doors and the tree.',None),
    ('Between the entrances and the sign; actually, by the stairs.',None), ('By the van instead.', None),
])
def test_extract_literal_current_spans(utterance, words):
    quotes = explicit_pair_quotes(utterance)
    if words is None:
        assert quotes is None
    else:
        assert [q['text'] for q in quotes] == words
        assert [utterance[q['start']:q['end']] for q in quotes] == words


@pytest.fixture
def setup_scene(tmp_path):
    path = tmp_path/'source.png'; Image.new('RGB',(100,80),'navy').save(path)
    frame = {'width':100,'height':80,'frame_id':11,'recording_id':'test-clip','timestamp_s':1.0,'image_path':str(path)}
    tracks = [{'recording_id':'test-clip','frame_id':11,'source':'predicted','track_id':idx,'class_name':'Test object',
               'bbox_xyxy':box} for idx,box in [(3,[4,20,20,60]),(7,[60,25,90,65])]]
    req = {'request_id':'pair-test','text':EXAMPLE,'scene':{'frame':frame, 'source':'predicted', 'tracks':tracks,
           'provenance':{'image_sha256':sha256(path.read_bytes()).hexdigest()}}, 'input_mode':'voice',
           'original_transcript':'between the white as you v and the truck',
           'dialogue':[{'role':'user','text':'Actually, use another place.','frame_id':11,'recording_id':'test-clip'}]}
    cfg = SimpleNamespace(max_image_bytes=1_000_000,max_original_pixels=10000,adapter_dir=None,model_id='fake-for-test',revision='test')
    return req,cfg


def collect(setup_scene, replies, text=None):
    req,cfg = setup_scene
    if text is not None: req = {**req, 'text':text}
    calls = []; replies = iter(replies)
    class Resolver:
        config = cfg
        def resolve(self, sub, image):
            calls.append((deepcopy(sub),id(image)))
            reply = next(replies)
            if isinstance(reply, Exception): raise reply
            # Use the real serialization, binding and parsing entry point around a substituted text backend.
            resolver = object.__new__(LandmarkResolver)
            resolver.config = cfg; resolver.adapter = None
            resolver.backend = SimpleNamespace(generate=lambda *a,**kw:SimpleNamespace(text=reply,details={}))
            return resolver.resolve(sub,image)
    image, binding = resolve_image(req,cfg)
    with image:
        evidence = collect_pair_evidence(req,image,Resolver())
    return req,cfg,binding,evidence,calls


def test_replay_reconstructs_both_sourced_boxes_without_generating_a_destination(setup_scene):
    req,cfg,binding,evidence,calls = collect(setup_scene,['LANDMARK L1', 'LANDMARK L2'])
    assert len(calls) == 2 and calls[0][1] == calls[1][1]
    for i,(sub,_) in enumerate(calls):
        quote = evidence['subqueries'][i]['quote']
        assert req['text'][quote['start']:quote['end']] == quote['text']
        assert sub['text'] == f"Beside {quote['text']}, please."
        assert sub['input_mode'] == 'typed' and 'original_transcript' not in sub
        assert sub['dialogue'] == [] and sub['scene'] == req['scene']
    actual = replay_pair_decision(req,binding,json.loads(json.dumps(evidence)),cfg)
    assert actual == {'kind':'landmark','track_ids':[3,7], 'anchor_track_ids':[3,7], 'boxes':[[4,20,20,60],[60,25,90,65]]}


@pytest.mark.parametrize('replies,kind', [
    (['LANDMARK L1','LANDMARK L1'],'ask'),
    (['OBJECT 100 100 500 500','OBJECT 490 490 800 800'],'ask'),
    (['LANDMARK L1 L2', 'LANDMARK L2'],'ask'),
    (['MISSING','LANDMARK L1'],'missing'), (['ASK Which sign?','LANDMARK L2'],'ask'),
    (['OBJECT 0 0 100 100','LANDMARK L2'],'object')])
def test_missing_ambiguity_overlap_and_mixed_sources(setup_scene,replies,kind):
    req,cfg,binding,evidence,_ = collect(setup_scene,replies)
    decision = replay_pair_decision(req,binding,evidence,cfg)
    assert decision['kind'] == kind
    if kind in ('ask','missing'): assert 'boxes' not in decision
    if kind == 'object':
        assert decision['anchor_track_ids'] == [None,7] and decision['track_ids'] == []
        assert decision['boxes'] == [[0,0,10,8],[60,25,90,65]]


def test_repeated_description_cannot_supply_two_identities_even_if_model_changes_its_mind(setup_scene):
    req,cfg,binding,evidence,_ = collect(setup_scene,['LANDMARK L1','LANDMARK L2'], 'Between a truck and the truck.')
    assert replay_pair_decision(req,binding,evidence,cfg)['kind'] == 'ask'


@pytest.mark.parametrize('reply,code', [(TimeoutError('time budget'), 'timeout'), ('LANDMARK L999', 'invalid_model_output')])
def test_one_operational_error_is_preserved_even_when_other_reference_is_missing(setup_scene,reply,code):
    req,cfg,binding,evidence,_ = collect(setup_scene,['MISSING',reply])
    with pytest.raises(PairResolutionError) as caught:
        replay_pair_decision(req,binding,json.loads(json.dumps(evidence)),cfg)
    assert caught.value.code == code


@pytest.mark.parametrize('change', ['quote','model','scene','raw','paired-order','outer-request','old-format'])
def test_serialized_subdecisions_cannot_be_swapped_or_rebound(setup_scene,change):
    req,cfg,binding,evidence,_=collect(setup_scene,['LANDMARK L1','LANDMARK L2'])
    second = evidence['subqueries'][1]
    if change == 'quote': second['quote']['text'] += ' elsewhere'
    if change == 'model': second['resolver']['model_id'] = 'other model'
    if change == 'scene': second['resolver']['image_sha256'] = '0'*64
    if change == 'raw': second['resolver']['raw_text'] = 'LANDMARK L1'
    if change == 'paired-order': evidence['subqueries'].reverse()
    if change == 'outer-request': req['text'] += ' instead'
    if change == 'old-format': evidence['version'] = 'other protocol'
    # Even a recomputed wrapper checksum cannot excuse a stale or inconsistent subresponse.
    evidence['receipt_sha256'] = _digest({k:v for k,v in evidence.items() if k!='receipt_sha256'})
    with pytest.raises(PairResolutionError) as caught: replay_pair_decision(req,binding,evidence,cfg)
    assert caught.value.code == 'artifact_invalid'


def test_changed_turn_and_wrong_original_do_not_reuse_old_pair(setup_scene):
    req,cfg = setup_scene
    prior = {'role':'user','text':EXAMPLE, 'recording_id':'test-clip','frame_id':11}
    current = {**req,'dialogue':[prior],'text':'Actually by the yellow bus instead.'}
    assert collect_pair_evidence(current,None,None) is None
    with Image.new('RGB',(100,80),'white') as other:
        with pytest.raises(PairResolutionError) as caught: collect_pair_evidence(req,other, SimpleNamespace(config=cfg))
    assert caught.value.code == 'artifact_invalid'
