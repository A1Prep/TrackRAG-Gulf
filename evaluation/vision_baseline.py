"""Evaluate saved real predictions after inference, on an explicit coarse label space."""
import argparse
import io
import json
import math
from contextlib import redirect_stdout
from hashlib import sha256
from pathlib import Path

from .run_audit import _box, fingerprint, integer
from .score import read_jsonl
from trackrag.emt import open_recording,load_annotated_tracks

EMT_VEHICLES={'car','bus','motorbike','small_motorised_vehicle','medium_vehicle','large_vehicle','emergency_vehicle'}
COCO_VEHICLES={'car','bus','motorcycle','truck'}
GT_FIELDS=('recording_id','frame_id','track_id','class_name','bbox_xyxy','source')


def canonical_hash(rows):
    return sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def checked_detections(info,recording):
    """Bind the declared evaluation span to unique original frames and pixels."""
    count=integer(info['frame_count'],'declared frame_count')
    first=integer(info['first_frame'],'declared first_frame')
    last=integer(info['last_frame'],'declared last_frame')
    if count<1 or first>last:
        raise ValueError('The declared inference span must contain frames')
    path=Path(info['detections_path'])
    if fingerprint(path)!=info['detections_sha256']:
        raise ValueError('Detection output changed after inference')
    rows=read_jsonl(path)
    frames=[frame for frame in recording.iter_frames() if first<=frame['frame_id']<=last]
    expected=[frame['frame_id'] for frame in frames]
    actual=[integer(row['frame_id'],'detection frame_id') for row in rows]
    if (len(rows)!=count or len(frames)!=count or expected!=sorted(set(expected))
            or actual!=expected or not actual or actual[0]!=first or actual[-1]!=last):
        raise ValueError('Detection frames must uniquely cover the declared available inference span')
    mapping=info['native_class_mapping']
    if not isinstance(mapping,dict) or any(not isinstance(value,str) or not value.strip() for value in mapping.values()):
        raise ValueError('Native class mapping requires nonempty class names')
    verified=[]
    for row,frame in zip(rows,frames):
        if row['recording_id']!=recording.recording_id:
            raise ValueError('Detection output belongs to a different recording')
        for dimension in ('width','height'):
            if type(row[dimension]) is not int or row[dimension]!=frame[dimension]:
                raise ValueError('Detection dimensions differ from the original frame')
        if row['image_sha256']!=fingerprint(frame['image_path']):
            raise ValueError('Evaluation image pixels differ from the saved inference input')
        if not isinstance(row['detections'],list):
            raise ValueError('Frame detections must be a list')
        for detection in row['detections']:
            class_id=integer(detection['class_id'],'native class_id')
            if mapping.get(str(class_id),mapping.get(class_id))!=detection['class_name']:
                raise ValueError('Detection class disagrees with its native class mapping')
            confidence=detection['confidence']
            if (isinstance(confidence,bool) or not isinstance(confidence,(int,float))
                    or not math.isfinite(confidence) or not 0<=confidence<=1):
                raise ValueError('Detection confidence must be finite and within zero and one')
            _box(detection['bbox_xyxy'])
            if detection['bbox_xyxy'][2]>frame['width'] or detection['bbox_xyxy'][3]>frame['height']:
                raise ValueError('Detection box exceeds original frame dimensions')
        verified.append({key:row[key] for key in ('recording_id','frame_id','width','height','image_sha256')})
    return rows,{frame['frame_id']:frame for frame in frames},canonical_hash(verified)


def checked_tracks(rows,recording_id,frames,native_names):
    seen=set()
    for row in rows:
        if row['recording_id']!=recording_id or row['source']!='predicted':
            raise ValueError('Mixed prediction identities/sources')
        frame=integer(row['frame_id'],'prediction frame_id')
        track=integer(row['track_id'],'prediction track_id')
        if frame not in frames:
            raise ValueError('Predicted track lies outside the evaluated inference frames')
        if (frame,track) in seen:
            raise ValueError('Duplicate predicted track/frame observation')
        seen.add((frame,track))
        if not isinstance(row['class_name'],str) or row['class_name'].lower() not in native_names:
            raise ValueError('Predicted track class is absent from the native detector mapping')
        _box(row['bbox_xyxy'])
        if row['bbox_xyxy'][2]>frames[frame]['width'] or row['bbox_xyxy'][3]>frames[frame]['height']:
            raise ValueError('Predicted track box exceeds original frame dimensions')
    return rows


def xywh(box):
    left,top,right,bottom=box
    return [left,top,right-left,bottom-top]


def evaluate(vision_directory,data_root):
    import numpy as np
    import motmetrics as mm
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    summary=json.loads((Path(vision_directory)/'summary.json').read_text())
    images=[]
    annotations=[]
    detections=[]
    tracking={}
    coverage=[]
    input_hashes=[]
    ground_truth_provenance=[]
    frames_total=0
    seen_recordings=set()
    if not summary['recordings']:
        raise ValueError('The inference summary contains no recordings')
    for recording_info in summary['recordings']:
        name=recording_info['recording_id']
        if not isinstance(name,str) or not name.strip() or name in seen_recordings:
            raise ValueError('Inference recordings must be nonempty and unique')
        seen_recordings.add(name)
        recording=open_recording(Path(data_root),name)
        rows,frames,verified_images_hash=checked_detections(recording_info,recording)
        gt=[row for row in load_annotated_tracks(recording) if row['frame_id'] in frames]
        by_frame={}
        for row in gt:
            by_frame.setdefault(row['frame_id'],[]).append(row)
        input_hashes.append({'path':recording_info['detections_path'],'sha256':recording_info['detections_sha256']})
        canonical_gt=[{key:row[key] for key in GT_FIELDS} for row in sorted(gt,key=lambda row:(row['frame_id'],row['track_id']))]
        provenance={'recording_id':name,'evaluated_frames':list(frames),
            'canonical_gt_fields':list(GT_FIELDS),'canonical_gt_rows_sha256':canonical_hash(canonical_gt),
            'gt_observation_count':len(canonical_gt),'verified_image_manifest_sha256':verified_images_hash}
        annotation_path=getattr(recording,'annotation_path',None)
        if annotation_path is not None and Path(annotation_path).is_file():
            annotation_file={'path':str(annotation_path),'sha256':fingerprint(annotation_path)}
            provenance['annotation_file']=annotation_file
            input_hashes.append(annotation_file)
        ground_truth_provenance.append(provenance)
        coverage.append({'recording_id':name,'ground_truth_classes':sorted({row['class_name'] for row in gt}),
                         'native_predicted_classes':recording_info['native_class_mapping'],
                         'unmapped_gt_classes':sorted({row['class_name'] for row in gt if row['class_name'].lower() not in EMT_VEHICLES})})
        for row in rows:
            frame=row['frame_id']
            image_id=len(images)+1
            images.append({'id':image_id,'width':row['width'],'height':row['height']})
            for truth in by_frame.get(frame,[]):
                if truth['class_name'].lower() not in EMT_VEHICLES:
                    continue
                box=xywh(truth['bbox_xyxy'])
                annotations.append({'id':len(annotations)+1,'image_id':image_id,'category_id':1,
                    'bbox':box,'area':box[2]*box[3],'iscrowd':0})
            for prediction in row['detections']:
                if prediction['class_name'].lower() in COCO_VEHICLES:
                    detections.append({'image_id':image_id,'category_id':1,'bbox':xywh(prediction['bbox_xyxy']),
                                       'score':prediction['confidence']})
        frames_total+=len(rows)
        seen_methods=set()
        for tracker_info in recording_info['trackers']:
            method=tracker_info['gmc_method']
            if not isinstance(method,str) or not method.strip() or method in seen_methods:
                raise ValueError('Tracker methods must be nonempty and unique within a recording')
            seen_methods.add(method)
            path=Path(tracker_info['tracks_path'])
            if fingerprint(path)!=tracker_info['tracks_sha256']:
                raise ValueError('Tracker outputs changed after inference')
            input_hashes.append({'path':path.as_posix(),'sha256':fingerprint(path)})
            predictions=checked_tracks(read_jsonl(path),name,frames,
                {value.lower() for value in recording_info['native_class_mapping'].values()})
            pred_frames={}
            for row in predictions:
                if row['class_name'].lower() in COCO_VEHICLES:
                    pred_frames.setdefault(row['frame_id'],[]).append(row)
            accumulator=mm.MOTAccumulator(auto_id=False)
            for frame in [row['frame_id'] for row in rows]:
                truth=[row for row in by_frame.get(frame,[]) if row['class_name'].lower() in EMT_VEHICLES]
                predicted=pred_frames.get(frame,[])
                distances=mm.distances.iou_matrix(np.asarray([xywh(row['bbox_xyxy']) for row in truth]).reshape(-1,4),
                    np.asarray([xywh(row['bbox_xyxy']) for row in predicted]).reshape(-1,4),max_iou=0.5)
                accumulator.update([row['track_id'] for row in truth],[row['track_id'] for row in predicted],distances,frameid=frame)
            tracking.setdefault(method,[]).append((name,accumulator))
    gt=COCO()
    gt.dataset={'info':{'description':'Explicit motor-vehicle class-agnostic development evaluation'},
        'images':images,'annotations':annotations,'categories':[{'id':1,'name':'motor_vehicle'}]}
    with redirect_stdout(io.StringIO()) as captured:
        gt.createIndex()
        if detections:
            predicted=gt.loadRes(detections)
        else:
            # pycocotools.loadRes indexes detections[0]; an empty prediction set
            # is valid and must score as misses instead of crashing.
            predicted=COCO()
            predicted.dataset={**gt.dataset,'annotations':[]}
            predicted.createIndex()
        evaluator=COCOeval(gt,predicted,'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names=['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
    track_results={}
    for method,pairs in tracking.items():
        result=mm.metrics.create().compute_many([pair[1] for pair in pairs],names=[pair[0] for pair in pairs],
            metrics=['num_frames','num_objects','num_predictions','num_matches','num_misses','num_false_positives',
                     'num_switches','mota','motp','idf1','idp','idr'],generate_overall=True)
        track_results[method]=json.loads(result.to_json(orient='index'))
    return {'version':2,'scope':'Declared-recording development motor-vehicle localization/tracking; no full EMT taxonomy or held-out accuracy claim.',
        'vision_summary_sha256':fingerprint(Path(vision_directory)/'summary.json'),'frame_count':frames_total,
        'gt_vehicle_boxes':len(annotations),'detected_vehicle_boxes_at_fixed_threshold':len(detections),
        'class_policy':{'gt_merged_to_one_vehicle_category':sorted(EMT_VEHICLES),'prediction_merged_to_one_vehicle_category':sorted(COCO_VEHICLES),
            'limitations':'COCO lacks EMT medium/large/emergency categories. This does not measure full EMT class accuracy. Motorbike/rider box conventions may differ.'},
        'class_coverage':coverage,'coco_metrics':{key:float(value) for key,value in zip(names,evaluator.stats)},
        'tracking_iou_threshold':0.5,'tracking_metrics':track_results,'coco_console':captured.getvalue(),
        'ground_truth_provenance':ground_truth_provenance,
        'metric_policy':{'detection_confidence_threshold':summary.get('threshold'),
            'coco_max_detections_per_image':list(evaluator.params.maxDets),
            'motp':'Mean matched-box distance 1-IoU; lower is better, not an IoU accuracy.'},
        'input_fingerprints':input_hashes}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-run',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('Choose a fresh metric output path')
    result=evaluate(args.vision_run,args.data_root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as output:
        output.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'frames':result['frame_count'],'coco_metrics':result['coco_metrics'],'tracking':result['tracking_metrics']},indent=2))


if __name__=='__main__':
    main()
