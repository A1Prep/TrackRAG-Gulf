"""Collect both neutral vehicle crops locally from Task 1 and Mask2Former, then freeze the raw replies."""
import argparse
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from time import monotonic

from PIL import Image

from trackrag.dropoff import bay_landmarks, local_surfaces, local_vehicle_observations as local, manual_surfaces, vehicle_curb_ground
from trackrag.dropoff.parking_evidence import signed_bytes
from trackrag.passenger_app import load_scenes


def write(path,value):
    with NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        stream.write(signed_bytes(value)+b'\n')
    try:
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def task1(bank,scene,info):
    frame=scene['frame']
    key=scene['provenance']['image_sha256']
    if (scene['source']!='predicted' or scene['provenance'].get('network_policy')!='local_only'
            or not local_surfaces.matches_scene(scene) or scene['provenance'].get('vision_identity_sha256')!=local.TASK1_ID):
        raise ValueError('Unregistered or nonlocal Task 1 photo.')
    rows=[r for r in info['sources'] if r['original_sha256']==key and r['recording_id']==frame['recording_id'] and r['frame_id']==frame['frame_id']]
    if len(rows)!=1 or info['vision_identity_sha256']!=local.TASK1_ID or info['checkpoint_sha256']!=local.TASK1_WEIGHTS:
        raise ValueError('A source row is missing or belongs to another detector.')
    row=rows[0]
    data={}
    for field,suffix in [('tracks_sha256','tracks.jsonl'),('detections_sha256','detections.jsonl'),
                         ('vision_provenance_sha256','provenance.json')]:
        raw=(bank.parent/'predictions'/(frame['recording_id']+'.'+suffix)).read_bytes()
        if sha256(raw).hexdigest()!=row[field]:
            raise ValueError('A raw Task 1 prediction artifact has changed.')
        data[suffix]=[json.loads(v) for v in raw.splitlines()] if suffix.endswith('jsonl') else json.loads(raw)
    active=local.tracks(scene)
    actual=[r for r in data['tracks.jsonl'] if r['recording_id']==frame['recording_id'] and r['frame_id']==frame['frame_id']]
    detected=[r for r in data['detections.jsonl'] if r['recording_id']==frame['recording_id'] and r['frame_id']==frame['frame_id']]
    run=data['provenance.json']
    if (actual!=active or len(detected)!=1 or not isinstance(detected[0].get('native'),list)
            or (detected[0]['width'],detected[0]['height'])!=(frame['width'],frame['height'])
            or run['identity_hash']!=local.TASK1_ID or run['detector']['kind']!='real'
            or run['detector']['checkpoint']['sha256']!=local.TASK1_WEIGHTS or run['tracker']['kind']!='ultralytics_botsort'
            or run['tracks_sha256']!=row['tracks_sha256']):
        raise ValueError('Current model boxes differ from the real detector and BoT-SORT outputs.')
    evidence={'vision_identity_sha256':local.TASK1_ID,'checkpoint_sha256':local.TASK1_WEIGHTS,
              **{field:row[field] for field in ('tracks_sha256','detections_sha256','vision_provenance_sha256')},
              'current_tracks_sha256':local.digest(active),'native_frame_sha256':local.digest(detected[0]),
              'native_detection_count':len(detected[0]['native']),'current_track_count':len(active)}
    return evidence,detected[0]


def inventory(bank,working,first=None):
    """Save full pre-VLM detections and tracker outputs, including sources/vehicles without a usable crop."""
    source=json.loads((bank.parent/'sources.json').read_bytes())
    if sha256(bank.read_bytes()).hexdigest()!=source['scene_bank_sha256']:
        raise ValueError('Scene bank differs from the source-only Task 1 intake.')
    scenes=sorted(load_scenes([bank]).values(),key=lambda s:s['provenance']['image_sha256'])
    if first is not None:
        scenes=scenes[:first]
    results,report=[],{}
    for scene in scenes:
        key=scene['provenance']['image_sha256'];frame=scene['frame']
        evidence,raw=task1(bank,scene,source)
        surface=local_surfaces.get_packet(key)
        classes=manual_surfaces.decode(surface,frame,key)
        specs=vehicle_curb_ground.describe(scene,classes)
        results.append((scene,evidence,surface,specs))
        report[key]={'recording_id':frame['recording_id'],'frame_id':frame['frame_id'],'size':[frame['width'],frame['height']],
                     'task1':evidence,'raw_detection_frame':raw,'current_tracks':local.tracks(scene),
                     'surface_sha256':surface['signature'],'eligible_vehicle_crops':[local.anchor(s) for s in specs]}
    working.mkdir(parents=True,exist_ok=True)
    path=working/'detector-inventory.json'
    report={'scene_bank_sha256':source['scene_bank_sha256'],'sources':report}
    if path.exists() and json.loads(path.read_bytes())!=report:
        raise ValueError('The working folder is already bound to another original, detector or surface inventory.')
    write(path,report)
    print(f'Verified original raw detections on {len(scenes)} sources; eligible vehicles per photo: {[len(r[3]) for r in results]}',flush=True)
    return results


def model_id(config,runtime):
    if not runtime.identity['verified_weights'] or not runtime.identity['verified_metadata']:
        raise ValueError('Vehicle preparation requires a verified local checkpoint.')
    return {'model_id':runtime.identity['model_id'],'revision':runtime.identity['revision'],
            'adapter':sha256((Path(config.adapter_dir)/'adapter_model.safetensors').read_bytes()).hexdigest(),
            'processor':bay_landmarks.MODELS['processor'],'quantization':config.quantization,'runtime':'local_streamed',
            'weights':{k:v['sha256'] for k,v in runtime.identity['files'].items()},
            'min_pixels':config.min_pixels,'max_pixels':config.max_pixels,'inference_resident_layers':config.inference_resident_layers}


def collect(bank,working,config,runtime,*,first=None,max_new_calls=None,output=None):
    bank=Path(bank).resolve();working=Path(working)
    calls=working/'calls';calls.mkdir(exist_ok=True,parents=True)
    inputs=inventory(bank,working,first)
    model=model_id(config,runtime)
    package={'version':1,'origin':local.ORIGIN,'sources':{}}
    new=0
    for scene,detection,surface,specs in inputs:
        frame=scene['frame'];key=scene['provenance']['image_sha256']
        rows=[]
        photo=Path(frame['image_path']).read_bytes()
        if sha256(photo).hexdigest()!=key:
            raise ValueError('The local original changed after detection.')
        with Image.open(BytesIO(photo)) as im:
            for spec in specs:
                anchor=local.anchor(spec); measured=[]
                for stage,word in local.VIEWS.items():
                    pixels=local.render(im,anchor['box'],stage);viewsha=sha256(pixels).hexdigest()
                    identity={'source':key,'anchor':anchor,'input':viewsha,'stage':stage,'guide':local.GUIDE,
                              'words':word,'output_limit':local.MAX_TOKENS,'model':model}
                    path=calls/(local.digest(identity)+'.json')
                    if path.exists():
                        cached=json.loads(path.read_bytes()); result=cached['result']
                        if (cached['input']!=identity or result['stage']!=stage or result['view_sha256']!=viewsha
                                or sha256(result['reply'].encode()).hexdigest()!=result['reply_sha256']):
                            raise ValueError('A recorded crop call failed its source, model or raw-output binding.')
                    else:
                        if max_new_calls and new>=max_new_calls:
                            return False
                        import torch
                        torch.cuda.empty_cache()
                        with Image.open(BytesIO(pixels)) as target:
                            msg=[{'role':'system','content':local.GUIDE},
                                 {'role':'user','content':[{'type':'image','image':target},{'type':'text','text':word}]}]
                            response=runtime.generate(msg,config=replace(config,max_new_tokens=local.MAX_TOKENS))
                        # Preserve malformed text for audit; parsing or cross-view disagreement never invents an attribute.
                        raw=response.text
                        result={'stage':stage,'view_sha256':viewsha,'reply':raw,'reply_sha256':sha256(raw.encode()).hexdigest(),
                                'input_tokens':response.details['input_tokens'],'output_tokens':response.details['output_tokens'],
                                'seconds':round(response.details['generation_elapsed_s'],3)}
                        write(path,{'input':identity,'result':result})
                        new+=1
                        print(f'{key[:10]} track {anchor["track_id"]} {stage} measured in {result["seconds"]} s',flush=True)
                    measured.append(result)
                rows.append({'anchor':anchor,'calls':measured,'attributes':local.consensus(measured)})
        body={'recipe':local.ORIGIN,'image_sha256':key,'recording_id':frame['recording_id'],'frame_id':frame['frame_id'],
              'size':[frame['width'],frame['height']],'task1':detection,'surface_sha256':surface['signature'],
              'prompt_sha256':local.digest([local.GUIDE,local.VIEWS,local.MAX_TOKENS]),'model':model,'count':len(rows),'rows':rows,
              'status':'complete' if rows else 'no_eligible_current_vehicles'}
        packet={'origin':local.ORIGIN,'body':body,'body_sha256':local.digest(body)}
        local._check_packet(key,packet)
        write(working/(key+'.json'),packet)
        package['sources'][key]=packet
    destination=Path(output) if output else working/'vehicle-observations.json'
    write(destination,package)
    print(f'{len(inputs)} sources. Vehicle archive SHA-256 {sha256(destination.read_bytes()).hexdigest()}',flush=True)
    return True


def main():
    import subprocess
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True,type=Path)
    p.add_argument('--working',required=True,type=Path)
    p.add_argument('--local-surface-index',nargs=2,action='append',metavar=('JSON','SHA256'))
    p.add_argument('--output',type=Path)
    p.add_argument('--inventory-only',action='store_true',help='Save native RF-DETR outputs, BoT-SORT tracks and eligible crops before VLM inference.')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.inventory_only:
        for index,pin in a.local_surface_index or []:
            local_surfaces.register(index,pin)
        inventory(a.bank.resolve(),a.working)
        return
    if not a.worker:
        while True:
            before=set((a.working/'calls').glob('*.json'))
            completed=subprocess.run([sys.executable,'-u','-m','tools.passenger_local_vehicle_views',*sys.argv[1:],'--worker'])
            if completed.returncode==0: return
            after=set((a.working/'calls').glob('*.json'))
            if completed.returncode not in (75,76) or after==before:
                raise SystemExit(completed.returncode)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
                      HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false')
    def offline(event,values):
        if event=='socket.connect':
            raise RuntimeError('Local vehicle measurement cannot connect to a network.')
    sys.addaudithook(offline)
    from trackrag.dropoff.model import GroundingConfig
    from trackrag.dropoff.streamed import StreamedRuntime
    import torch
    torch.set_num_threads(2)
    for index,pin in a.local_surface_index or []:
        local_surfaces.register(index,pin)
    root=Path(__file__).resolve().parents[1]
    config=GroundingConfig.from_mapping(json.loads((root/'models/passenger-feature/grounding.json').read_bytes()))
    config=replace(config,adapter_dir=str(root/'models/passenger-feature/adapter'),model_dir=str(root/config.model_dir),
                   cache_dir=str(root/config.cache_dir),max_time_s=60,inference_resident_layers=2)
    start=monotonic()
    runtime=StreamedRuntime(config,verify_weights=True)
    print(f'Verified offline 8B weights loaded in {monotonic()-start:.2f}s.',flush=True)
    try:
        if not collect(a.bank,a.working,config,runtime,max_new_calls=12,output=a.output):
            raise SystemExit(75)
    except torch.OutOfMemoryError as error:
        print(f'Incomplete local vehicle observations after GPU memory exhaustion: {error}',file=sys.stderr,flush=True)
        raise SystemExit(76)
    finally:
        runtime.close()


if __name__=='__main__':
    main()
