"""Actual RF-DETR Large 2026 / BoT-SORT development inference, without gold."""
import argparse
import importlib.metadata
import json
import subprocess
import time
from hashlib import md5
from pathlib import Path
from types import SimpleNamespace

from PIL import Image,ImageDraw
from evaluation.run_audit import fingerprint
from tools.emt_trial_adapters import open_recording
from tools.model_environment import configure_model_caches

ROAD_USERS={'person','bicycle','car','motorcycle','bus','truck'}


def save(path,data):
    temporary=Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def tracker_config(method):
    return SimpleNamespace(tracker_type='botsort',track_high_thresh=0.25,track_low_thresh=0.1,
        new_track_thresh=0.25,track_buffer=30,match_thresh=0.8,fuse_score=True,
        gmc_method=method,proximity_thresh=0.5,appearance_thresh=0.8,with_reid=False,model='auto')


def clip_box(box,width,height):
    left,top,right,bottom=map(float,box)
    bounded=[max(0,min(left,width)),max(0,min(top,height)),max(0,min(right,width)),max(0,min(bottom,height))]
    if bounded[0]>=bounded[2] or bounded[1]>=bounded[3]:
        return None
    return bounded


def run(args):
    if args.out.exists():
        raise FileExistsError('Choose a new prediction output directory')
    if md5(args.weights.read_bytes()).hexdigest()!='5cb72153541cbcb9aa6efa26222acc75':
        raise ValueError('Expected publisher-verified RF-DETR Large 2026 checkpoint')
    configure_model_caches(args.cache_root)
    import cv2
    import numpy as np
    import torch
    from rfdetr import RFDETRLarge
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.engine.results import Boxes
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True)
    model_started=time.perf_counter()
    model=RFDETRLarge(pretrain_weights=str(args.weights.resolve()),device=args.device)
    model_load_seconds=time.perf_counter()-model_started
    print(f'Loaded actual RF-DETR Large 2026 in {model_load_seconds:.2f}s',flush=True)
    summary={'version':1,'scope':'Pretrained development inference, no fine-tuning or held-out accuracy claim',
        'executed_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'adapter_file_sha256':fingerprint(__file__),
        'model':'RF-DETR Large 2026','checkpoint':{'path':str(args.weights),'sha256':fingerprint(args.weights)},
        'device':args.device,'threshold':args.threshold,'input_resolution':704,'threads':args.threads,
        'model_load_seconds':model_load_seconds,'reid_enabled':False,
        'packages':{name:importlib.metadata.version(name) for name in ('torch','torchvision','rfdetr','transformers','ultralytics','numpy')},
        'annotation_access':'Inference source is an image-only staged recording; no label loader is called.',
        'recordings':[]}
    for recording_id in args.recordings:
        recording=open_recording(args.data_root,recording_id)
        directory=args.out/recording_id
        directory.mkdir()
        frames=list(recording.iter_frames())
        if args.max_frames:
            frames=frames[:args.max_frames]
        print(f'INFERENCE {recording_id}: {len(frames)} original frames',flush=True)
        rows=[]
        labels={}
        timings=[]
        with (directory/'detections.jsonl').open('w',encoding='utf-8') as stream:
            for position,frame in enumerate(frames,1):
                started=time.perf_counter()
                with Image.open(frame['image_path']) as image:
                    rgb=image.convert('RGB')
                with torch.inference_mode():
                    detections=model.predict(rgb,threshold=args.threshold,include_source_image=False)
                elapsed=time.perf_counter()-started
                names=detections.data.get('class_name')
                if names is None:
                    raise ValueError('Native RF-DETR output omitted class-name mapping')
                output=[]
                for box,score,class_id,name in zip(detections.xyxy,detections.confidence,detections.class_id,names):
                    name=str(name)
                    class_id=int(class_id)
                    if class_id in labels and labels[class_id]!=name:
                        raise ValueError('Inconsistent native class mapping')
                    labels[class_id]=name
                    bounded=clip_box(box,frame['width'],frame['height'])
                    if bounded is None:
                        continue
                    output.append({'bbox_xyxy':bounded,'confidence':float(score),'class_id':class_id,'class_name':name})
                row={'recording_id':recording_id,'frame_id':frame['frame_id'],'width':frame['width'],'height':frame['height'],
                     'image_path':frame['image_path'],'image_sha256':fingerprint(frame['image_path']),
                     'inference_seconds':elapsed,'detections':output}
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
                rows.append(row)
                timings.append(elapsed)
                if position==1 or position%10==0 or position==len(frames):
                    print(f'{recording_id} detected {position}/{len(frames)}; last={elapsed:.3f}s, boxes={len(output)}',flush=True)
        trackers=[]
        for method in ('sparseOptFlow','none'):
            tracker=BOTSORT(tracker_config(method))
            target=directory/('botsort-gmc' if method!='none' else 'botsort-no-gmc')
            target.mkdir()
            observations=[]
            clipped=0
            started=time.perf_counter()
            last_frame=None
            last_image=None
            with (target/'predicted_tracks.jsonl').open('w',encoding='utf-8') as stream:
                for position,row in enumerate(rows,1):
                    image=cv2.imread(row['image_path'])
                    if image is None:
                        raise ValueError('Could not decode an inference source image')
                    if last_frame is not None:
                        for missing in range(last_frame+1,row['frame_id']):
                            # Advance through a missing frame without inventing
                            # pixels or exporting an observed box for that time.
                            tracker.update(Boxes(np.empty((0,6),dtype=np.float32),image.shape[:2]).numpy(),last_image)
                    selected=[d for d in row['detections'] if d['class_name'].lower() in ROAD_USERS]
                    data=np.asarray([d['bbox_xyxy']+[d['confidence'],d['class_id']] for d in selected],dtype=np.float32).reshape(-1,6)
                    tracked=tracker.update(Boxes(data,image.shape[:2]).numpy(),image)
                    marked=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                    draw=ImageDraw.Draw(marked)
                    for result in tracked:
                        raw_box=result[:4].tolist()
                        box=clip_box(raw_box,row['width'],row['height'])
                        if box is None:
                            continue
                        clipped+=box!=raw_box
                        track_id,class_id=int(result[4]),int(result[6])
                        observation={'recording_id':recording_id,'frame_id':row['frame_id'],'track_id':track_id,
                            'class_name':labels[class_id],'bbox_xyxy':box,'source':'predicted'}
                        stream.write(json.dumps(observation,allow_nan=False)+'\n')
                        observations.append(observation)
                        draw.rectangle(box,outline='red',width=2)
                        draw.text((box[0],max(0,box[1]-12)),f'{track_id} {labels[class_id]}',fill='red')
                    if position in {1,(len(rows)+1)//2,len(rows)}:
                        marked.save(target/f'frame-{row["frame_id"]}.jpg',quality=90)
                    last_frame,last_image=row['frame_id'],image
            info={'gmc_method':method,'configuration':vars(tracker_config(method)),
                'tracking_seconds':time.perf_counter()-started,'observations':len(observations),
                'identities':len({r['track_id'] for r in observations}),'clipped_boxes':clipped,
                'tracks_path':(target/'predicted_tracks.jsonl').as_posix(),
                'tracks_sha256':fingerprint(target/'predicted_tracks.jsonl')}
            trackers.append(info)
            save(target/'provenance.json',info)
            print(f'{recording_id} BoT-SORT {method}: {len(observations)} observations, {info["identities"]} identities',flush=True)
        summary['recordings'].append({'recording_id':recording_id,'frame_count':len(rows),
            'first_frame':rows[0]['frame_id'],'last_frame':rows[-1]['frame_id'],
            'inference_seconds':sum(timings),'mean_inference_seconds':sum(timings)/len(timings),
            'native_class_mapping':labels,'detections_path':(directory/'detections.jsonl').as_posix(),
            'detections_sha256':fingerprint(directory/'detections.jsonl'),'trackers':trackers})
        save(args.out/'summary.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--recordings',nargs='+',required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,default=Path('runs/model-cache'))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--threshold',type=float,default=0.05)
    parser.add_argument('--max-frames',type=int)
    args=parser.parse_args(argv)
    if args.threads<1 or not 0<=args.threshold<=1 or (args.max_frames is not None and args.max_frames<1):
        parser.error('Invalid inference limits')
    run(args)


if __name__=='__main__':
    main()
