from copy import deepcopy
from pathlib import Path
import json

import pytest

from trackrag.dropoff import vehicle_road_comparison as comparison


LEFT = {'track_id':4, 'box':[70,150,160,245]}
RIGHT = {'track_id':6, 'box':[720,90,810,165]}
GRAY = ['gray','sedan','REAR','CURBSIDE']
BLUE = ['blue','van','FRONT','NONE']


def assess(text, specs=(LEFT,RIGHT), answers=(GRAY,BLUE), observations=()):
    request = {'text':text, 'scene':{'frame':{'width':1000,'height':600}}, 'dialogue':[]}
    return comparison.clarification(request,specs,answers,observations)


@pytest.mark.parametrize('words, box, body, position, side', [
    ('Could you drop me behind the grey sedan further up the right-hand side?',LEFT,'gray sedan','left','right'),
    ('Let me out beside the blue van farther down the street on our left.',RIGHT,'blue van','right','left'),
    ('Stop behind the gray sedan farther ahead on the left side of the road.',LEFT,'gray sedan','left','left'),
])
def test_roadside_comparatives_name_only_a_visible_body(words, box, body, position, side):
    boxes, reason, question = assess(words)
    assert boxes == [box['box']] and reason == 'vehicle_side_uncertain'
    assert f'{body} {position} of centre in the photo.' in question
    assert f'{side}-hand side of the road' in question and 'can’t verify' in question
    assert 'haven’t verified a drop-off point' in question


@pytest.mark.parametrize('words', [
    'Drop me behind the grey sedan further up the right-hand side of the picture.',
    'Stop behind the rightmost grey sedan further up the right-hand side.',
    'Stop behind the gray Toyota sedan further up the left.',
    'Drop me behind the grey sedan with a green logo further along the right.',
    'Drop me behind the gray sedan further down the right compared to the red car.',
    'Stop behind the gray sedan further up the left and beside the black pickup.',
    'Drop me on its right, behind the grey sedan.',
])
def test_screen_conditions_and_other_requirements_are_never_discarded(words):
    assert assess(words) is None


def test_wrong_colour_or_body_cannot_receive_the_roadside_vehicle_question():
    for text in ('Drop me behind the red sedan further up the left.', 'Stop behind the grey bus farther ahead on the right.'):
        boxes, reason, _ = assess(text)
        assert boxes == [] and reason == 'vehicle_reference_unmatched'


def test_duplicate_unknown_and_contradictory_competitors_cannot_confirm_one_vehicle():
    words='Drop me behind the grey sedan further up the right-hand side.'
    options=([GRAY,GRAY], [GRAY,['unknown','sedan','SIDE','NONE']], [GRAY,['blue','sedan','SIDE','NONE']])
    for index, answers in enumerate(options):
        extra=[{**RIGHT, 'appearance':'gray sedan'}] if index==2 else []
        boxes, reason, _ = assess(words,answers=answers,observations=extra)
        assert boxes and reason=='vehicle_seen_space_withheld'


def test_real_original_first_turn_and_yes_are_identity_only_old_decision_replays(monkeypatch):
    from tools import passenger_release as release
    from trackrag.dropoff import local_surfaces, local_vehicle_observations as local, manual_surfaces
    from trackrag.dropoff import vehicle_ground_reference as grounder, vehicle_identity_reply, whole_bay

    root=Path(release.__file__).resolve().parents[1]
    gallery=root/release.RESERVE_PHOTO_BANK
    if not gallery.is_file():
        pytest.skip('The optional publicly sourced development photos have not been installed.')
    monkeypatch.setattr(local_surfaces,'_extra_sources',{})
    monkeypatch.setattr(local,'_registered',{})
    local_surfaces._sources.cache_clear()
    try:
        for kind, filename, digest in release.RESERVE_MODEL_FILES:
            register=local_surfaces.register if kind=='local_surface_index' else local.register
            register(root/release.RESERVE_PHOTO_RESULTS/filename,digest)
        scene=next(s for line in gallery.read_bytes().splitlines()
                   if (s:=json.loads(line)['scene'])['frame']['recording_id']=='kartaview_7422361')
        frame=scene['frame']
        original=gallery.parent/frame['image_path']
        frame['image_path']=str(original)
        photo=original.read_bytes()
        key=scene['provenance']['image_sha256']
        request={'scene':scene,'request_id':'road-relative-test','input_mode':'typed','dialogue':[],
                 'text':'Please let me out behind the grey sedan further up the right-hand side.'}
        old=grounder.outcome(request,photo,manual_surfaces.get_packet(key),local.get_packet(key),
                             measure_rear=True,extended_rear=True,road_comparisons=False)
        assert not old['result'].get('anchor_boxes_xyxy') and 'road_comparison_policy' not in old['provenance']
        assert whole_bay.verify(request,old)['decision_reproduced']
        offered=grounder.available(request,photo)
        target=next(t['bbox_xyxy'] for t in scene['tracks'] if t['track_id']==2)
        assert offered['result']['anchor_boxes_xyxy']==[target]
        assert vehicle_identity_reply.is_question(request,offered)
        assert whole_bay.verify_current(request,offered)['decision_reproduced']
        for text,expected in [('Yes, that vehicle',[target]),('No',[])]:
            where={'recording_id':frame['recording_id'],'frame_id':frame['frame_id']}
            follow={**request,'request_id':'reply','text':text,'dialogue':[
                {**where,'role':'user','text':request['text']},
                {**where,'role':'assistant','text':offered['result']['clarification_question']}]}
            replied=vehicle_identity_reply.answer(follow,request,offered)
            assert (replied['result'].get('anchor_boxes_xyxy') or [])==expected
            assert replied['result']['destination_point_xy'] is None
            assert vehicle_identity_reply.verify_current(follow,replied)['reply']==text.split(',')[0].lower()
        changed=deepcopy(offered)
        changed['result']['anchor_boxes_xyxy'][0][0]+=10
        with pytest.raises(ValueError):
            whole_bay.verify_current(request,changed)
    finally:
        local_surfaces._sources.cache_clear()
