"""Measure apparent tail direction using identical original and mirrored, on-device vehicle crops."""
import argparse
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys


def inventory(args):
    from trackrag.dropoff import local_vehicle_tail as tail, local_vehicle_observations as vehicle, local_surfaces, manual_surfaces
    from trackrag.passenger_app import load_scenes
    from tools.passenger_local_vehicle_views import write

    local_surfaces.register(*args.surfaces); vehicle.register(*args.vehicles)
    cases=[]; controls=[]
    for scene in sorted(load_scenes([args.bank.resolve()]).values(),key=lambda s:s['provenance']['image_sha256']):
        key=scene['provenance']['image_sha256']; frame=scene['frame']
        parent=vehicle.get_packet(key,origin=vehicle.ORIGIN_V2)
        if not parent: raise ValueError('Missing local 8B source observation.')
        photo=Path(frame['image_path']).read_bytes()
        if sha256(photo).hexdigest()!=key: raise ValueError('Source photo changed.')
        classes=manual_surfaces.decode(local_surfaces.get_packet(key),frame,key)
        vehicle.verify(scene,photo,vehicle.specs_for(parent,scene,classes),parent)
        measured=[]; possible=[]
        for row in parent['body']['rows']:
            if tail.eligible(row): measured.append((row['anchor'],'eligible'))
            elif tail.control(row): possible.append((vehicle.digest([key,row['anchor']]),row['anchor']))
        if possible:
            order,anchor=min(possible,key=lambda p:p[0]); controls.append((order,measured,anchor))
        cases.append((scene,parent,measured))
    # At most two controls per release, on different photos, selected without passenger wording, points or gold.
    for _,rows,anchor in sorted(controls,key=lambda t:t[0])[:2]: rows.append((anchor,'control'))
    doc={'parent_archive_sha256':args.vehicles[1],'prompt_sha256':tail.PROMPT_SHA,'sources':{scene['provenance']['image_sha256']:
         [{'anchor':anchor,'role':role} for anchor,role in rows] for scene,_,rows in cases}}
    args.working.mkdir(exist_ok=True,parents=True); path=args.working/'neutral-inventory.json'
    if path.exists() and json.loads(path.read_bytes())!=doc: raise ValueError('The previously fixed inventory changed.')
    write(path,doc)
    print('Finite neutral inventory:',sum(r=='eligible' for _,_,rows in cases for _,r in rows),'eligible,',sum(r=='control' for _,_,rows in cases for _,r in rows),'controls.',flush=True)
    return cases


def collect(args, cases, runtime, config):
    from PIL import Image
    from trackrag.dropoff import local_vehicle_tail as tail, local_vehicle_observations as vehicle
    from tools.passenger_local_vehicle_views import model_id,write
    import torch

    model=model_id(config,runtime); new=0; sources={}; logs=args.working/'calls';logs.mkdir(exist_ok=True)
    for scene,parent,selected in cases:
        base=parent['body']; key=base['image_sha256']; rows=[]
        if model!=base['model']:raise ValueError('The parent measurements used a different local model.')
        with Image.open(scene['frame']['image_path']) as photo:
            for anchor,role in selected:
                calls=[]
                for stage in tail.STAGES:
                    pixels=tail.render(photo,anchor['box'],stage); ph=sha256(pixels).hexdigest()
                    identity={'source':key,'anchor':anchor,'parent':parent['body_sha256'],'model':model,'prompt_sha256':tail.PROMPT_SHA,'stage':stage,'pixels':ph}
                    path=logs/(vehicle.digest(identity)+'.json')
                    if path.exists():
                        rec=json.loads(path.read_bytes());call=rec['result']
                        if (rec['input']!=identity or call['view_sha256']!=ph or call['stage']!=stage or
                                sha256(call['reply'].encode()).hexdigest()!=call['reply_sha256']):raise ValueError('Cached tail call differs from its inputs or raw reply.')
                    else:
                        if new>=12:return False
                        torch.cuda.empty_cache()
                        with Image.open(BytesIO(pixels)) as image:
                            result=runtime.generate([{'role':'system','content':tail.GUIDE},
                                {'role':'user','content':[{'type':'image','image':image},{'type':'text','text':tail.QUESTION}]}],
                                config=replace(config,max_new_tokens=tail.TOKEN_LIMIT))
                        call={'stage':stage,'view_sha256':ph,'reply':result.text,'reply_sha256':sha256(result.text.encode()).hexdigest(),
                              'seconds':round(result.details['generation_elapsed_s'],3),'input_tokens':result.details['input_tokens'],'output_tokens':result.details['output_tokens']}
                        write(path,{'input':identity,'result':call});new+=1
                        print(f'{key[:10]} track {anchor["track_id"]} {role} {stage} {call["seconds"]}s',flush=True)
                    calls.append(call)
                rows.append({'anchor':anchor,'role':role,'calls':calls,'tail_edge':tail.consensus(calls)})
        body={'source_sha256':key,'parent_body_sha256':parent['body_sha256'],'model':model,'recording_id':base['recording_id'],
              'frame_id':base['frame_id'],'size':base['size'],'prompt_sha256':tail.PROMPT_SHA,'rows':rows}
        pack={'origin':tail.ORIGIN,'body':body,'body_sha256':vehicle.digest(body)};tail._check(key,pack);sources[key]=pack
    path=args.working/'vehicle-tail-observations.json';write(path,{'origin':tail.ORIGIN,'version':1,'sources':sources})
    print('Finished exact tail pack',sha256(path.read_bytes()).hexdigest(),flush=True)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--working',type=Path,required=True)
    parser.add_argument('--surfaces',nargs=2,required=True,metavar=('JSON','SHA256'))
    parser.add_argument('--vehicles',nargs=2,required=True,metavar=('JSON','SHA256'))
    parser.add_argument('--inventory-only',action='store_true');parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.inventory_only: inventory(args);return
    if not args.worker:
        while True:
            before=set((args.working/'calls').glob('*.json'))
            code=subprocess.run([sys.executable,'-u','-m','tools.passenger_local_vehicle_tail',*sys.argv[1:],'--worker']).returncode
            if code==0: return
            if code not in (75,76) or before==set((args.working/'calls').glob('*.json')):raise SystemExit(code)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
    def offline(event,values):
        if event=='socket.connect': raise RuntimeError('On-device vehicle tails cannot connect to the network.')
    sys.addaudithook(offline)
    cases=inventory(args)
    import torch
    from trackrag.dropoff.model import GroundingConfig
    from trackrag.dropoff.streamed import StreamedRuntime
    torch.set_num_threads(2); root=Path(__file__).resolve().parents[1]
    cfg=GroundingConfig.from_mapping(json.loads((root/'models/passenger-feature/grounding.json').read_bytes()))
    cfg=replace(cfg,adapter_dir=str(root/'models/passenger-feature/adapter'),model_dir=str(root/cfg.model_dir),cache_dir=str(root/cfg.cache_dir),max_time_s=60,inference_resident_layers=2)
    runtime=StreamedRuntime(cfg,verify_weights=True)
    try:
        if not collect(args,cases,runtime,cfg):raise SystemExit(75)
    except torch.OutOfMemoryError as exc:
        print('Tail worker ran out of device memory; keeping completed raw calls.',str(exc),file=sys.stderr,flush=True);os._exit(76)
    finally:runtime.close()


if __name__=='__main__':main()
