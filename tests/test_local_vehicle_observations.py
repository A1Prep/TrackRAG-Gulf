"""Local release replay and ordinary abstention, using generated pixels and synthetic raw readings."""
from hashlib import sha256
from io import BytesIO
from copy import deepcopy
import json
import subprocess
import sys

from PIL import Image, ImageDraw
import pytest

from trackrag.dropoff import local_vehicle_observations as local
from trackrag.dropoff.parking_evidence import signed_bytes


def test_idle_crop_supervisor_leaves_image_and_model_libraries_for_child():
    script = """
import sys
loaded = set(sys.modules)
from tools import passenger_local_vehicle_views
heavy = {'torch', 'numpy', 'scipy', 'trackrag.passenger_app'}
assert heavy.isdisjoint(set(sys.modules) - loaded)
"""
    subprocess.run([sys.executable, '-c', script], check=True)


@pytest.mark.parametrize(('first','mirror','expected'),[
    ('LEFT','RIGHT','LEFT'),('RIGHT','LEFT','RIGHT'),('LEFT','LEFT','UNCLEAR'),
    ('UNCLEAR','RIGHT','UNCLEAR'),('LEFT or RIGHT','RIGHT','UNCLEAR'),('RIGHT','UNCLEAR','UNCLEAR')])
def test_tail_requires_opposite_single_word_measurements(first,mirror,expected):
    from trackrag.dropoff import local_vehicle_tail as tail
    assert tail.consensus([{'reply':first},{'reply':mirror}])==expected


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    im=Image.new('RGB',(100,80),'gray')
    ImageDraw.Draw(im).rectangle((15,20,72,64),fill='white')
    pixels=BytesIO();im.save(pixels,format='PNG')
    photo=pixels.getvalue();key=sha256(photo).hexdigest()
    spec={'track_id':2,'class':'car','box':[15,20,72,64],'ground':None}
    scene={'frame':{'recording_id':'test','frame_id':7,'width':100,'height':80},'tracks':[
        {'recording_id':'test','frame_id':7,'track_id':2,'class_name':'Car','bbox_xyxy':spec['box'],'source':'predicted'}],
        'source':'predicted','provenance':{'image_sha256':key,'network_policy':'local_only','vision_identity_sha256':local.TASK1_ID}}
    monkeypatch.setattr(local,'_registered',{})
    monkeypatch.setattr(local.local_surfaces,'matches_scene',lambda s:s['provenance']['image_sha256']==key)
    monkeypatch.setattr(local.local_surfaces,'get_packet',lambda h:{'signature':'a'*64} if h==key else None)
    model={'model_id':local.STREAMED_MODEL_ID,'revision':local.bay_landmarks.MODELS['base'],
           'adapter':local.bay_landmarks.MODELS['adapter'],'processor':local.bay_landmarks.MODELS['processor'],
           'runtime':'local_streamed','quantization':'nf4','weights':{**local.WEIGHT_SHA256,**local.METADATA_SHA256},
           'min_pixels':65536,'max_pixels':262144,'inference_resident_layers':2}
    task1={name:'b'*64 for name in ('raw','detections_sha256','tracks_sha256','vision_provenance_sha256','native_frame_sha256')}
    task1.update(vision_identity_sha256=local.TASK1_ID,checkpoint_sha256=local.TASK1_WEIGHTS,
                 current_tracks_sha256=local.digest(local.tracks(scene)),native_detection_count=1,current_track_count=1)
    reply='white | sedan | REAR'
    calls=[{'stage':stage,'view_sha256':sha256(local.render(im,spec['box'],stage)).hexdigest(),
            'reply':reply,'reply_sha256':sha256(reply.encode()).hexdigest(),'input_tokens':115,'output_tokens':8,'seconds':1.}
           for stage in local.VIEWS]
    body={'recipe':local.ORIGIN,'image_sha256':key,'recording_id':'test','frame_id':7,'size':[100,80],
          'task1':task1,'surface_sha256':'a'*64,'prompt_sha256':local.digest([local.GUIDE,local.VIEWS,local.MAX_TOKENS]),
          'model':model,'count':1,'rows':[{'anchor':local.anchor(spec),'calls':calls,'attributes':local.consensus(calls)}],'status':'complete'}
    def install(body):
        packet={'origin':local.ORIGIN,'body':body,'body_sha256':local.digest(body)}
        path=tmp_path/'model-evidence.json'
        path.write_bytes(signed_bytes({'origin':local.ORIGIN,'version':1,'sources':{key:packet}})+b'\n')
        return path,sha256(path.read_bytes()).hexdigest()
    return body,scene,photo,[spec],install


def test_original_crop_both_views_and_current_track_replay(evidence):
    body,scene,photo,specs,install=evidence
    assert local.get_packet(body['image_sha256']) is None
    local.register(*install(body))
    packet=local.get_packet(body['image_sha256'])
    assert local.verify(scene,photo,specs,packet)==[['white','sedan','REAR','NONE']]
    packet['body']['rows'][0]['attributes'][2]='FRONT'
    with pytest.raises(ValueError):local.verify(scene,photo,specs,packet)
    with pytest.raises(ValueError):local.register(*install(body))


@pytest.mark.parametrize('role',['eligible','control'])
def test_tail_registry_binds_both_native_images_and_keeps_controls_out_of_runtime(evidence,tmp_path,monkeypatch,role):
    import numpy as np
    from PIL import ImageOps
    from trackrag.dropoff import local_vehicle_tail as tail
    body,scene,photo,_,_=evidence
    key=body['image_sha256']; row=body['rows'][0]
    box=[4,20,94,60] if role=='eligible' else [15,20,72,64]
    row['anchor']['box']=box;scene['tracks'][0]['bbox_xyxy']=box
    body['task1']['current_tracks_sha256']=local.digest(local.tracks(scene))
    with Image.open(BytesIO(photo)) as native:
        for call in row['calls']:
            call['view_sha256']=sha256(local.render(native,box,call['stage'])).hexdigest()
            if role=='control':
                call['reply']='white | sedan | FRONT';call['reply_sha256']=sha256(call['reply'].encode()).hexdigest()
        row['attributes']=local.consensus(row['calls'])
        calls=[];images=[]
        for stage,answer in zip(tail.STAGES,('LEFT','RIGHT'),strict=True):
            view=tail.render(native,box,stage);images.append(Image.open(BytesIO(view)))
            calls.append({'stage':stage,'reply':answer,'reply_sha256':sha256(answer.encode()).hexdigest(),
                          'view_sha256':sha256(view).hexdigest(),'input_tokens':50,'output_tokens':2,'seconds':1.2})
        assert images[1].tobytes()==ImageOps.mirror(images[0]).tobytes()
    local.register(*expanded_pack(tmp_path,body));parent=local.get_packet(key)
    monkeypatch.setattr(tail,'_registered',{})
    monkeypatch.setattr(tail.manual_surfaces,'decode',lambda *args:np.zeros((40,50),dtype=np.uint8))
    measured={'source_sha256':key,'recording_id':'test','frame_id':7,'size':[100,80],
              'model':body['model'],'prompt_sha256':tail.PROMPT_SHA,'parent_body_sha256':parent['body_sha256'],
              'rows':[{'anchor':row['anchor'],'role':role,'calls':calls,'tail_edge':'LEFT'}]}
    def write_packet():
        packet={'origin':tail.ORIGIN,'body':measured,'body_sha256':local.digest(measured)}
        path=tmp_path/'tail.json';path.write_bytes(signed_bytes({'origin':tail.ORIGIN,'version':1,'sources':{key:packet}}))
        return path,sha256(path.read_bytes()).hexdigest()
    path,pin=write_packet()
    with pytest.raises(ValueError):tail.register(path,'0'*64)
    tail.register(path,pin);packet=tail.get_packet(key)
    assert tail.verify(scene,photo,packet)==({2:'LEFT'} if role=='eligible' else {})
    with pytest.raises(ValueError):tail.verify(scene,photo+b'other',packet)
    monkeypatch.setattr(tail,'_registered',{})
    measured['rows'][0]['calls'][1]['view_sha256']='c'*64
    tail.register(*write_packet())
    with pytest.raises(ValueError,match='reflected vehicle pixels'):
        tail.verify(scene,photo,tail.get_packet(key))
    monkeypatch.setattr(tail,'_registered',{})
    measured['parent_body_sha256']='9'*64
    with pytest.raises(ValueError,match='base crops'):tail.register(*write_packet())


@pytest.mark.parametrize('change',['guide','weights','detector','view','missing_view','source','raw','consensus','stage','count'])
def test_field_incompatibility_cannot_be_silently_accepted(evidence,change):
    body,scene,photo,specs,install=evidence
    r=body['rows'][0]
    if change=='guide': body['prompt_sha256']='f'*64
    if change=='weights':body['model']['weights']= {}
    if change=='detector':body['task1']['vision_identity_sha256']='c'*64
    if change=='view':r['calls'][0]['view_sha256']='d'*64
    if change=='missing_view':r['calls'].pop()
    if change=='source':body['image_sha256']='e'*64
    if change=='raw':r['calls'][0]['reply']='black | sedan | REAR'
    if change=='consensus':r['attributes'][0]='black'
    if change=='stage':r['calls'].reverse()
    if change=='count':body['count']=0
    with pytest.raises(ValueError):
        local.register(*install(body))
        local.verify(scene,photo,specs,local.get_packet(scene['provenance']['image_sha256']))


@pytest.mark.parametrize('change',['other_photo','relocated_frame','different_box','future_box','remote_scene','missing_surface'])
def test_bound_source_and_observation_never_transfer(evidence,change,monkeypatch):
    body,scene,photo,specs,install=evidence
    local.register(*install(body))
    packet=local.get_packet(body['image_sha256'])
    if change=='other_photo':photo+=b'second'
    if change=='relocated_frame':scene['frame']['frame_id']+=1
    if change=='different_box':specs[0]['box']=[1,4,10,24]
    if change=='future_box':scene['tracks'][0]['frame_id']+=1
    if change=='remote_scene':scene['provenance'].pop('network_policy')
    if change=='missing_surface':monkeypatch.setattr(local.local_surfaces,'get_packet',lambda _:None)
    with pytest.raises(ValueError):local.verify(scene,photo,specs,packet)


def test_disagreeing_or_malformed_crops_never_authorize_orientation_or_ground():
    def calls(a,b):
        return [{'reply':a},{'reply':b}]
    assert local.consensus(calls('black|sedan|REAR','black,sedan,FRONT'))==['black','sedan','UNCLEAR','NONE']
    assert local.consensus(calls('white|suv|REAR','black|suv|REAR'))==['unknown','suv','REAR','NONE']
    assert local.consensus(calls('black|sedan|REAR','black|sedan|REAR|parking'))==['unknown','unknown','UNCLEAR','NONE']


def test_empty_measured_source_and_missing_archive_keep_zero_readings(evidence):
    body,scene,photo,specs,install=evidence
    body.update(count=0,rows=[],status='no_eligible_current_vehicles')
    scene['tracks']=[]
    body['task1']['current_tracks_sha256']=local.digest([])
    body['task1']['current_track_count']=0
    location,pin=install(body)
    with pytest.raises(ValueError):local.register(location,'0'*64)
    local.register(location,pin)
    p=local.get_packet(body['image_sha256'])
    assert local.verify(scene,photo,[],p)==[]
    assert local.get_packet('9'*64) is None


def expanded_pack(tmp_path, body):
    body={**deepcopy(body),'recipe':local.ORIGIN_V2,'crop_policy':local.CROP_POLICY}
    packet={'origin':local.ORIGIN_V2,'body':body,'body_sha256':local.digest(body)}
    path=tmp_path/'expanded.json'
    path.write_bytes(signed_bytes({'origin':local.ORIGIN_V2,'version':2,'sources':{body['image_sha256']:packet}})+b'\n')
    return path,sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('new_first',[False,True])
def test_both_policies_verify_their_own_original_journal_after_restart(evidence,tmp_path,new_first):
    import numpy as np
    from tools.passenger_local_vehicle_views import reusable_rows
    from trackrag.dropoff import vehicle_curb_ground
    body,scene,photo,_,install=evidence
    v1,v2=install(body),expanded_pack(tmp_path,body)
    for archive in ((v2,v1) if new_first else (v1,v2)):
        local.register(*archive)
    classes=np.zeros((40,50),dtype=np.uint8)
    old=local.get_packet(body['image_sha256'],origin=local.ORIGIN)
    new=local.get_packet(body['image_sha256'])
    assert new['origin']==local.ORIGIN_V2 and old['origin']==local.ORIGIN
    assert local.specs_for(old,scene,classes)==vehicle_curb_ground.describe(scene,classes)
    for packet in (old,new):
        assert local.verify(scene,photo,local.specs_for(packet,scene,classes),packet)==[['white','sedan','REAR','NONE']]
    assert reusable_rows(scene,photo,classes,body['model'],body['task1'])[2]==body['rows'][0]
    with pytest.raises(ValueError,match='different model'):
        reusable_rows(scene,photo,classes,{**body['model'],'adapter':'different'},body['task1'])
    with pytest.raises(ValueError,match='native detector evidence'):
        reusable_rows(scene,photo,classes,body['model'],{**body['task1'],'native_frame_sha256':'0'*64})
    with pytest.raises(ValueError,match='original Task 1 preparation folder'):
        reusable_rows(scene,photo,classes,body['model'],{**body['task1'],'vision_provenance_sha256':'0'*64})
    with pytest.raises(ValueError):
        local.specs_for({'origin':'unknown'},scene,classes)


def test_expanded_release_cannot_overwrite_an_archived_observation(evidence,tmp_path):
    body,_,_,_,install=evidence
    local.register(*install(body))
    altered=deepcopy(body)
    altered['rows'][0]['calls'][0]['seconds']+=1
    with pytest.raises(ValueError,match='existing measured crop'):
        local.register(*expanded_pack(tmp_path,altered))
    assert local.get_packet(body['image_sha256'])['origin']==local.ORIGIN


def test_expanded_budget_spreads_legible_native_boxes_and_preserves_the_legacy_four():
    import numpy as np
    frame={'recording_id':'dense','frame_id':19,'width':1600,'height':900}
    boxes=[[x,600,x+120,690] for x in (20,155,290,425)]
    boxes += [[x,y,x+65,y+42] for y in (230,530) for x in (30,230,560,810,1110,1380)]
    boxes.append([1300,410,1345,448])
    tracks=[{'recording_id':'dense','frame_id':19,'track_id':i,'class_name':'Car','bbox_xyxy':box,'source':'predicted'}
            for i,box in enumerate(boxes)]
    scene={'frame':frame,'tracks':tracks,'provenance':{'image_sha256':'a'*64}}
    classes=np.zeros((45,80),dtype=np.uint8)
    v1=local.specs_for({'origin':local.ORIGIN},scene,classes)
    v2=local.specs_for({'origin':local.ORIGIN_V2},scene,classes)
    assert [row['track_id'] for row in v1]==[0,1,2,3]
    assert v2[:4]==v1 and len(v2)==8
    assert any(row['box'][0]>1000 for row in v2) and any(row['box'][1]<300 for row in v2)
    assert all(row['track_id']!=len(boxes)-1 for row in v2)
    assert v2==local.specs_for({'origin':local.ORIGIN_V2},{**scene,'tracks':list(reversed(tracks))},classes)


def test_source_only_inventory_rejects_substituted_detector_and_replaced_raw_boxes(tmp_path,monkeypatch):
    from tests.test_passenger_local_photos import FakeIoUTracker, OneCar, manifest, source
    from tools import passenger_local_photos as intake
    from tools.passenger_local_vehicle_views import task1
    from trackrag.passenger_app import load_scenes
    manifest_path=manifest(tmp_path,[source(tmp_path,'42',3)])
    config=intake.load_config(intake.ROOT/'configs/vision/finetuned.json',device='cpu')
    bank=intake.prepare(manifest_path,tmp_path/'bundle',config,detector_factory=OneCar,tracker_factory=FakeIoUTracker)
    scene=next(iter(load_scenes([bank]).values()))
    info=json.loads((bank.parent/'sources.json').read_bytes())
    monkeypatch.setattr(local.local_surfaces,'matches_scene',lambda _:True)
    with pytest.raises(ValueError,match='real detector'):
        task1(bank,scene,info)
    native=next((bank.parent/'predictions').glob('*.detections.jsonl'))
    altered=json.loads(native.read_bytes())
    altered['native'][0]['bbox_xyxy']=[1.,1.,12.,10.]
    native.write_bytes(signed_bytes(altered)+b'\n')
    with pytest.raises(ValueError,match='raw Task 1'):
        task1(bank,scene,info)


def test_separate_journal_export_registers_operator_pins_before_source_replay(tmp_path,monkeypatch):
    from tools import dropoff_passenger_handoff as handoff
    from trackrag.dropoff import local_8b_observations, local_vehicle_tail
    registrations=[]
    for name,module in [('map',local.local_surfaces),('curb',local_8b_observations),('vehicle',local),('tail',local_vehicle_tail)]:
        monkeypatch.setattr(module,'register',lambda path,pin,name=name:registrations.append((name,path,pin)))
    def replay(journal,**kwargs):
        assert registrations==[('map','index','a'*64),('curb','ground','b'*64),('vehicle','cars','c'*64),('tail','tails','d'*64)]
        return {'verified':True}
    out=tmp_path/'actual.json'
    monkeypatch.setattr(handoff,'session',replay)
    monkeypatch.setattr(sys,'argv',['handoff','session','--journal','original','--out',str(out),
        '--local-vehicle-observations','cars','c'*64,'--local-surface-index','index','a'*64,
        '--local-observations','ground','b'*64,'--local-vehicle-tail-observations','tails','d'*64])
    handoff.main()
    assert json.loads(out.read_bytes())=={'verified':True}


@pytest.mark.parametrize('wrapped', [False, True])
def test_saving_uses_current_vehicle_readings_but_a_past_journal_can_keep_its_original(wrapped, monkeypatch):
    from trackrag.dropoff import whole_bay
    old = {'origin':local.ORIGIN, 'body_sha256':'a'*64}
    current = {'origin':local.ORIGIN_V2, 'body_sha256':'b'*64}
    source = {'method':whole_bay.AUTO, 'clearance_policy':whole_bay.CLEARANCE_POLICY, 'vehicle_observation':old}
    result = {'provenance':source}
    if wrapped:
        result = {'provenance':{'method':whole_bay.CHOICE,'choice_source':result}}
    request = {'text':'Behind the white car','scene':{'provenance':{'image_sha256':'c'*64}}}
    monkeypatch.setattr(whole_bay,'verify',lambda *args,**kwargs:'replayed')
    for active in (None,current):
        monkeypatch.setattr(local,'get_packet',lambda _,value=active:value)
        with pytest.raises(ValueError,match='Check your location again'):
            whole_bay.verify_current(request,result)
        assert whole_bay.verify(request,result)=='replayed'
    monkeypatch.setattr(local,'get_packet',lambda _:old)
    assert whole_bay.verify_current(request,result)=='replayed'
