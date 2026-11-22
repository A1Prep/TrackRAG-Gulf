"""Software-only photographs isolate references, missing gestures and independent model readings."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from tests.test_passenger_projected_pair import make, obs
from trackrag.dropoff import pair_dialogue, pair_geometry, projected_pair
from trackrag.dropoff.passenger import digest
from trackrag.dropoff.referent import ALIGN_NAMES, KIND, REGION_NAMES, ReferentRanker


@pytest.fixture
def deictic(tmp_path):
    boxes = [[5,20,25,50],[35,20,55,50],[65,20,85,50]]
    file = tmp_path/'traffic.png'
    image = Image.new('RGB',(100,80),'white')
    brush = ImageDraw.Draw(image)
    for box in boxes[:2]:
        brush.rectangle(box, fill='black')
    image.save(file)
    frame = {'recording_id':'fixture-clip','frame_id':8,'timestamp_s':.7,
             'width':100,'height':80,'image_path':str(file)}
    scene = {'frame':frame,'source':'predicted','tracks':[obs(i,b) for i,b in enumerate(boxes,1)],
             'provenance':{'image_sha256':sha256(file.read_bytes()).hexdigest()}}
    req = {'request_id':'deictic-fixture','input_mode':'typed','text':'Drop me between these two black cars.',
           'dialogue':[],'scene':scene}
    weights = tmp_path/'fixture-ranker.json'
    weights.write_text(json.dumps({'kind':KIND,'region_names':list(REGION_NAMES),'align_names':list(ALIGN_NAMES),
                       'vocabulary':[],'base':[0.]*len(REGION_NAMES),'aligned':[0.]*len(ALIGN_NAMES),'interactions':[]}))
    model = SimpleNamespace(referent_ranker=ReferentRanker.load(weights,expected_sha256=sha256(weights.read_bytes()).hexdigest()),
                            config=SimpleNamespace(max_image_bytes=100000,max_original_pixels=8000))
    return req, model


@pytest.mark.parametrize(('words','ids'), [
    ('Drop me between these two black cars.',None),
    ('Drop me between the two dark cars.',None),
    ('Please let us out between those 2 cars.',[1,2,3]),
    ('Park between those two red cars, please.',[]),
    ('Could you drop me off between those two white cars?',[3]),
    ('I’d like to get out in the gap between the two cars.',[1,2,3]),
    ('Could you pull over right in between the two cars?',[1,2,3]),
    ('I want to get out between those two black cars.',None),
    ('Can we stop between those two cars?',[1,2,3]),
    ('Those two cars there, could you drop me in the gap between them?',[1,2,3]),
    ('The gap between that pair of cars, thanks.',[1,2,3]),
    ('Please let me get out between the two cars up ahead.',[1,2,3]),
    ('What is between that pair of cars?',None),
    ('Can we drive between those two cars?',None),
    ('Please count the cars between those two cars.',None),
    ('Those two cars by the left median, stop in the gap between them.',None),
    ('ليس between those two cars.',None),
    ('Stop between the left car and the black SUV.',None),
    ('Do not drop me between those two black cars.',None),
    ('Between those two cars, actually beside the bus.',None),
    ('Behind the space between these two cars.',None),
    ('Drop me between those two cars on the left.',None),
])
def test_initial_pair_uses_current_pixels_and_scoped_text(deictic, words, ids):
    req, model = deictic
    req['text'] = words
    before = digest(req)
    result = pair_dialogue.unpointed_pair(model, req)
    assert digest(req)==before
    if ids is None:
        assert result is None
    else:
        assert result['result']['status']=='needs_clarification' and result['result']['destination_point_xy'] is None
        p = result['provenance']
        assert p['request_sha256']==before and p['native_model_calls']==0 and sorted(p['eligible_tracks'])==ids
        assert p['image']['sha256']==req['scene']['provenance']['image_sha256']
        assert p['ranker_sha256']==model.referent_ranker.sha256


@pytest.mark.parametrize('fault',['history','no-current','wrong-source','wrong-image','unloaded','missing-words','invalid'])
def test_unsupported_or_invalid_input_delegated_to_normal_model(deictic,fault):
    req, model = deictic
    req['text']='Between those two cars.'
    if fault=='history': req['dialogue']=[{'role':'user','text':'The two on the left.','recording_id':'fixture-clip','frame_id':8}]
    elif fault=='no-current':
        for t in req['scene']['tracks']: t['frame_id']=7
    elif fault=='wrong-source': req['scene']['source']='annotated'
    elif fault=='wrong-image': req['scene']['provenance']['image_sha256']='0'*64
    elif fault=='unloaded': model.referent_ranker=None
    elif fault=='missing-words': req['text']=None
    else: req=None
    assert pair_dialogue.unpointed_pair(model,req) is None


def test_fewer_than_two_visible_vehicles_offers_available_actions(deictic):
    req, model = deictic
    req['scene']['tracks']=req['scene']['tracks'][:1]
    answer=pair_dialogue.unpointed_pair(model,req)['result']
    assert answer['clarification_question']==pair_dialogue.FEW_VEHICLES


def test_verified_darkness_abstains_with_ambiguous_third_or_two_light_cars(deictic):
    req,model=deictic
    req["text"]="Drop me between the two dark cars."
    path=req["scene"]["frame"]["image_path"]
    for color, number in [("white",2),("black",3),("darkgray",3)]:
        from pathlib import Path
        Image.new("RGB",(100,80),color).save(path)
        req["scene"]["provenance"]["image_sha256"]=sha256(Path(path).read_bytes()).hexdigest()
        selected=deepcopy(req)
        selected["scene"]["tracks"]=selected["scene"]["tracks"][:number]
        out=pair_dialogue.unpointed_pair(model,selected)
        assert out["result"]["status"]=="needs_clarification"
        assert not out["provenance"]["dark_appearance"]["distinct_supported_pair"]


@pytest.mark.parametrize('case',['conflict','agrees','different-input','changed-raw','manual','fixture','outer-fixture',
                                'assisted','failed','changed-boxes','checked-crop'])
def test_proposal_review_requires_two_bound_disagreeing_readings(make,monkeypatch,case):
    _, _, model, _ = make()
    def estimate(boxes, size, text):
        point, region = pair_geometry.predict(model.geometry.pair_regression, boxes, size)
        return point, region, []
    model.geometry.predict_with_evidence = estimate
    def third(req):
        req['scene']['tracks'].append(obs(103,[240,68,268,100],'Bus'))
    req, raw, model, _ = make(whole='LANDMARK L1 L3' if case=='agrees' else 'LANDMARK L1 L2',
                              first='LANDMARK L1',second='LANDMARK L3',update=third)
    assert raw['ok'] and raw['result']['status']=='proposed'
    if case!='fixture':
        monkeypatch.setattr(pair_dialogue,'native_record',projected_pair.native_record)
    if case=='different-input': req['text']+=' Please.'
    elif case=='changed-raw': raw['provenance']['resolver']['raw_text']='LANDMARK L2 L3'
    elif case=='manual': raw['provenance']['method']='explicit_passenger_region'
    elif case=='outer-fixture': raw['injected']=True
    elif case=='assisted': raw['provenance']['independent_model_proposal']=False
    elif case=='failed': raw={'ok':False,'request_id':req['request_id'],'error':{'code':'resource_error'}}
    elif case=='changed-boxes': raw['result']['anchor_boxes_xyxy'][0]=[0,0,12,12]
    elif case=='checked-crop': raw['provenance']['visual_verification']={'raw_text':'MATCH'}
    before=deepcopy(raw)
    reviewed=pair_dialogue.review_pair_disagreement(model,req,raw)
    assert raw==before
    if case=='conflict':
        assert reviewed['result']['status']=='needs_clarification' and reviewed['result']['destination_point_xy'] is None
        evidence=reviewed['provenance']['pair_disagreement']
        assert evidence['whole_sentence_ids']==[51,92] and evidence['separate_description_ids']==[51,103]
        assert evidence['original_proposal']==raw['result']
    else:
        assert reviewed is raw
