"""Real streamed 8B LoRA training and matched public destination evaluation.

Public destination supervision is a complete training stage, not a claim that
local clarification/status labels exist. Final EMT references never enter fit.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random
import time
import traceback

from tools.dropoff_train import (GroundingCollator, TrainConfig, adapter_parameter_hashes,
                                load_training_cases, package_versions, prepare_reload_probe,
                                sha256_file, write_json)
from trackrag.dropoff.model import GroundingConfig


def cohort(cases, count, seed=42):
    """Target-independent order, avoiding duplicate source scenes when possible."""
    if type(count) is not int or count<1 or type(seed) is not int or seed<0:
        raise ValueError('Cohort count must be positive and seed nonnegative')
    ordered=sorted(cases,key=lambda r:hashlib.sha256(f"{seed}:{r['case_id']}".encode()).hexdigest())
    selected=[];groups=set()
    for row in ordered:
        if row['scene_group'] not in groups:
            selected.append(row);groups.add(row['scene_group'])
            if len(selected)>=min(count,len(cases)):
                return selected
    for row in ordered:
        if row not in selected:
            selected.append(row)
            if len(selected)>=min(count,len(cases)):
                break
    return selected


def prepare_public_probe(case, directory):
    """Save a portable, annotation-free reload input from the public train split."""
    if (case.get('partition')!='train' or case.get('source',{}).get('dataset')!='Talk2Car-RegSeg'
            or case.get('review',{}).get('status')!='public_annotation'):
        raise ValueError('The public-stage portable probe requires an audited public training case')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    probe=prepare_reload_probe(case,directory)
    (directory/'requests.jsonl').write_bytes((json.dumps(probe['request'],sort_keys=True,allow_nan=False)+'\n').encode('utf-8'))
    return probe


def supervised_forward(runtime, collator, case):
    import torch
    batch=collator([case]);labels=batch.pop('labels').to(runtime.device)
    positions=(labels[0]!=-100).nonzero(as_tuple=False).flatten()
    if not len(positions) or int(positions.min())<=0:
        raise ValueError('Assistant-only target positions are invalid')
    if not (batch['input_ids']==runtime.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')).any():
        raise ValueError('Actual scene image tokens are missing')
    batch={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    out=runtime.model(**batch,use_cache=False,logits_to_keep=positions-1)
    loss=torch.nn.functional.cross_entropy(out.logits.float().reshape(-1,out.logits.shape[-1]),labels[0,positions])
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite supervised loss')
    return loss,int(len(positions))


def teacher_evaluate(runtime,collator,cases,*,disabled=False):
    import torch
    rows=[];total=0.0;tokens=0
    context=runtime.model.disable_adapter() if disabled else nullcontext()
    with runtime.execution_lock,context,torch.no_grad():
        runtime.model.eval()
        for case in cases:
            loss,count=supervised_forward(runtime,collator,case)
            value=float(loss);total+=value*count;tokens+=count
            rows.append({'case_id':case['case_id'],'nll':value,'target_tokens':count})
            del loss
    return {'kind':'teacher_forced_target_nll_not_spatial_accuracy','cases':len(rows),'tokens':tokens,
            'token_weighted_nll':total/tokens,'perplexity':math.exp(min(50,total/tokens)),'rows':rows}


def parse_destination(raw,width,height):
    from trackrag.dropoff.contracts import DESTINATION_FIELDS,validate_target
    from trackrag.dropoff.formatting import _unique_object,parse_model_output
    text=raw.strip()
    if text.startswith('```') and text.endswith('```'):
        lines=text.splitlines()
        if lines[0] not in ('```','```json'):
            raise ValueError('Unknown JSON fence')
        text='\n'.join(lines[1:-1])
    value=json.loads(text,object_pairs_hook=_unique_object)
    if not isinstance(value,dict):
        raise ValueError('Expected a JSON object')
    if set(value)!=set(DESTINATION_FIELDS):
        result=parse_model_output(text,width,height)
        if result['status']!='proposed':
            raise ValueError('No destination proposal; public masks cannot validate an unresolved status')
        return {key:result[key] for key in DESTINATION_FIELDS}
    value=validate_target(value,list(DESTINATION_FIELDS),{'width':1000,'height':1000})
    point=value['destination_point_xy'];box=value['destination_box_xyxy']
    result={'destination_point_xy':[point[0]*width/1000,point[1]*height/1000],
            'destination_box_xyxy':[box[0]*width/1000,box[1]*height/1000,box[2]*width/1000,box[3]*height/1000]}
    return validate_target(result,list(DESTINATION_FIELDS),{'width':width,'height':height})


def spatial_score(result,case):
    from PIL import Image
    frame=case['request']['scene']['frame'];w,h=frame['width'],frame['height']
    x,y=result['destination_point_xy']
    mask_payload=Path(case['source']['mask_path']).read_bytes()
    if len(mask_payload)>32*1024**2 or hashlib.sha256(mask_payload).hexdigest()!=case['source']['mask_sha256']:
        raise ValueError('Evaluation mask bytes changed')
    with Image.open(BytesIO(mask_payload)) as original:
        mask=original.convert('L');mw,mh=mask.size
        inside=mask.getpixel((min(mw-1,int(x*mw/w)),min(mh-1,int(y*mh/h))))>0
    a=result['destination_box_xyxy'];b=case['target']['destination_box_xyxy']
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    gt=case['target']['destination_point_xy']
    return {'point_in_mask':bool(inside),'coarse_box_iou':inter/union,
            'normalized_point_error':math.hypot((x-gt[0])/w,(y-gt[1])/h)}


def free_evaluate(runtime,cases,output,*,disabled=False):
    from trackrag.dropoff.formatting import build_messages
    from trackrag.dropoff.model import _image_snapshot,_materialize_image
    from trackrag.dropoff.data import inference_input_sha256
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() or output.with_suffix('.summary.json').exists():
        raise FileExistsError('Choose a new evaluation output path; earlier outcomes are retained')
    results=[]
    context=runtime.model.disable_adapter() if disabled and hasattr(runtime.model,'disable_adapter') else nullcontext()
    with runtime.execution_lock,context,output.open('w',encoding='utf-8') as stream:
        for case in cases:
            started=time.monotonic();row={'case_id':case['case_id'],'scene_group':case['scene_group'],'ok':False,
                'point_in_mask':False,'coarse_box_iou':0.0,'normalized_point_error':None}
            if '_contrast' in case:
                row['controlled_contrast']=deepcopy(case['_contrast'])
            image=None
            try:
                request=case['request'];frame=request['scene']['frame']
                image,record=_image_snapshot(frame,runtime.config)
                if record['sha256']!=case['source']['image_sha256']:
                    raise ValueError('Evaluation image changed')
                row['request_sha256']=inference_input_sha256(request,record['sha256'])
                context_input=deepcopy(request)
                context_input.pop('text');context_input.pop('request_id')
                context_input['scene']['frame'].pop('image_path')
                context_input['scene']['frame']['image_sha256']=record['sha256']
                context_input['scene'].pop('provenance',None)
                row['scene_context_sha256']=hashlib.sha256(json.dumps(context_input,sort_keys=True,allow_nan=False).encode()).hexdigest()
                prompt=build_messages(request,task='destination_only')
                fingerprint=deepcopy(prompt)
                for message in fingerprint:
                    if isinstance(message.get('content'),list):
                        for part in message['content']:
                            if part.get('type')=='image':part['image']=record['sha256']
                row['prompt_sha256']=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,allow_nan=False).encode()).hexdigest()
                messages=_materialize_image(prompt,image)
                response=runtime.generate(messages,config=runtime.config)
                row['raw_text']=response.text
                row['generation_details']=response.details
                parsed=parse_destination(response.text,frame['width'],frame['height'])
                row.update(ok=True,result=parsed,**spatial_score(parsed,case))
            except Exception as exc:
                row['error']={'type':type(exc).__name__,'message':str(exc)[:3000]}
            finally:
                if image is not None:image.close()
            row['elapsed_s']=time.monotonic()-started
            results.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            print(json.dumps({'stage':'free_generation','completed':len(results),'requested':len(cases),
                              'case_id':case['case_id'],'ok':row['ok'],'point_in_mask':row['point_in_mask']}),flush=True)
    n=len(results);valid=[r for r in results if r['ok']]
    summary={'kind':'free_generation_public_destination_development_not_final_EMT','requested_cases':n,
        'valid_geometry':len(valid),'coverage':len(valid)/n if n else 0,
        'point_in_mask_all_requests':sum(r['point_in_mask'] for r in results)/n if n else 0,
        'coarse_box_iou_all_requests':sum(r['coarse_box_iou'] for r in results)/n if n else 0,
        'mean_point_error_valid_only':sum(r['normalized_point_error'] for r in valid)/len(valid) if valid else None,
        'case_ids':[r['case_id'] for r in results],'failures_in_denominator':True,'gold_fields_sent_to_model':False,
        'results_sha256':sha256_file(output),'runtime':asdict(runtime.config)}
    write_json(output.with_suffix('.summary.json'),summary)
    return summary


def save_adapter(runtime,path,run_dir,progress):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    model=runtime.model
    model.peft_config['default'].base_model_name_or_path=runtime.config.model_id
    model.peft_config['default'].revision=runtime.config.revision
    model.save_pretrained(path,safe_serialization=True,save_embedding_layers=False)
    runtime.processor.save_pretrained(path)
    hashes=adapter_parameter_hashes(model)
    initial=getattr(runtime,'initial_adapter_hashes',None)
    changed=[name for name in hashes if initial is not None and initial.get(name)!=hashes[name]]
    manifest={'schema_version':1,'artifact_kind':'lora_adapter','base_model_id':runtime.config.model_id,
        'base_revision':runtime.config.revision,'files':{n:sha256_file(path/n) for n in ['adapter_config.json','adapter_model.safetensors']},
        'coordinate_system':'normalized_0_1000_image_edges','image_processing':{'min_pixels':runtime.config.min_pixels,'max_pixels':runtime.config.max_pixels},
        'training_stage':'public_destination_geometry_finetuning','optimizer_steps':progress['optimizer_steps'],
        'actual_training_examples_consumed':progress['examples_consumed'],'training_manifest_sha256':sha256_file(run_dir/'dataset_manifest.json'),
        'training_settings_sha256':sha256_file(run_dir/'training_settings.json'),'trainable_modules':runtime.lora_target_modules,
        'suffix_layers':runtime.config.suffix_layers,'adapter_parameter_hashes':hashes,
        'changed_adapter_tensors':len(changed),'verification':{'nonzero_updates_verified':bool(changed),'fresh_base_reload_verified':False},
        'public_region_labels_do_not_supervise_clarification':True}
    write_json(path/'grounding_adapter_manifest.json',manifest)
    return manifest


def controlled_contrast_cases(cases,protocol,source_sha256,protocol_sha256):
    """Hold public-image scene inputs fixed while source instructions differ."""
    if (protocol.get('kind')!='public_same_image_language_contrast'
            or protocol.get('development_source_sha256')!=source_sha256):
        raise ValueError('Contrast protocol differs from the frozen development source')
    indexed={r['case_id']:r for r in cases};chosen=[];seen=set()
    for group in protocol['groups']:
        signature=None
        for identity in group['case_ids']:
            if identity in seen or identity not in indexed:
                raise ValueError('Missing or repeated contrast source case')
            row=deepcopy(indexed[identity]);seen.add(identity)
            if row['source']['image_sha256']!=group['image_sha256']:
                raise ValueError('Contrast source image changed')
            request=row['request'];frame=request['scene']['frame']
            if request['scene']['tracks'] or request['dialogue']:
                raise ValueError('Public contrast expects image-only empty dialogue context')
            current=(frame['frame_id'],frame['timestamp_s'],frame['width'],frame['height'],
                     request['scene']['source'],request['input_mode'])
            if signature is not None and signature!=current:
                raise ValueError('Contrast scene metadata differs beyond its artificial recording identity')
            signature=current;original=frame['recording_id']
            frame['recording_id']=group['canonical_recording_id']
            row['_contrast']={'original_recording_id':original,'canonical_recording_id':group['canonical_recording_id'],
                              'image_sha256':group['image_sha256'],'protocol_sha256':protocol_sha256}
            chosen.append(row)
    if len(chosen)!=protocol['case_count']:
        raise ValueError('Contrast cohort count differs from protocol')
    return chosen


def train(args):
    import torch
    from trackrag.dropoff.streamed import StreamedRuntime
    config=GroundingConfig.from_mapping(json.loads(args.config.read_text()))
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.90,0)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    settings={'epochs':args.epochs,'learning_rate':args.learning_rate,'gradient_accumulation':args.accumulation,
        'seed':42,'lora_rank':8,'lora_alpha':16,'lora_dropout':0.0,'development_selection_count':args.dev_limit,
        'stage':'public_destination_geometry','run_ends_after_all_epochs':True,'model_configuration':asdict(config)}
    source_root=Path(__file__).resolve().parents[1]
    source_names=['tools/dropoff_fit_local.py','tools/dropoff_train.py','trackrag/dropoff/model.py',
        'trackrag/dropoff/streamed.py','trackrag/dropoff/streamed_weights.py','trackrag/dropoff/streamed_edges.py',
        'trackrag/dropoff/formatting.py','trackrag/dropoff/contracts.py','trackrag/dropoff/data.py']
    settings['source_sha256']={name:sha256_file(source_root/name) for name in source_names}
    for name in source_names:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((source_root/name).read_bytes())
    write_json(out/'training_settings.json',settings)
    loader_config=TrainConfig(max_train_cases=None,max_dev_cases=None,run_kind='pilot')
    cases,dev,manifest=load_training_cases(args.train,args.development,loader_config)
    if not dev:raise ValueError('A disjoint development partition is required')
    write_json(out/'dataset_manifest.json',manifest)
    prepare_public_probe(cases[0],out/'production-probe-input')
    selected=cohort(dev,args.dev_limit);generation_cases=cohort(dev,args.generation_limit)
    write_json(out/'evaluation_cohort.json',{'teacher_case_ids':[r['case_id'] for r in selected],
        'generation_case_ids':[r['case_id'] for r in generation_cases],'selection':'seeded_case_sha256_then_distinct_scenes','seed':42})
    torch.manual_seed(42)
    runtime=None;started=time.monotonic();history=[]
    progress={'status':'starting','optimizer_steps':0,'examples_consumed':0,'epochs_completed':0,'actual_training_case_ids':[],
              'requested_training_cases':len(cases),'development_pool_cases':len(dev),'local_interpretation_labels':0}
    def event(stage,**extra):
        progress.update(status=stage,elapsed_s=time.monotonic()-started,
            peak_pytorch_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0,
            peak_pytorch_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0,**extra)
        write_json(out/'progress.json',progress)
        print(json.dumps({k:v for k,v in progress.items() if k!='actual_training_case_ids'}),flush=True)
    try:
        event('loading_verified_8b')
        runtime=StreamedRuntime(config,training_lora={'rank':8,'alpha':16,'dropout':0.0})
        collator=GroundingCollator(runtime.processor,config.max_input_tokens)
        before=adapter_parameter_hashes(runtime.model)
        runtime.initial_adapter_hashes=before
        write_json(out/'runtime_provenance.json',runtime.provenance)
        event('pretrained_teacher_evaluation')
        baseline=teacher_evaluate(runtime,collator,selected,disabled=True)
        write_json(out/'pretrained_teacher_metrics.json',baseline)
        if args.preflight_generation:
            event('pretrained_free_generation_gate')
            gate=free_evaluate(runtime,generation_cases[:1],out/'pretrained-generation-gate.jsonl',disabled=True)
            if not gate['valid_geometry']:
                raise RuntimeError('Pretrained generation gate failed; inspect raw result before long training')
        parameters=[p for p in runtime.model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=.01)
        total_updates=math.ceil(len(cases)/args.accumulation)*args.epochs
        warmup=max(1,round(total_updates*.05))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1,(s+1)/warmup)*.5*(1+math.cos(math.pi*min(1,max(0,s-warmup)/max(1,total_updates-warmup)))))
        best=None
        with (out/'training-log.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(args.epochs):
                order=list(range(len(cases)));random.Random(42+epoch).shuffle(order)
                runtime.model.eval() # LoRA dropout is explicitly zero; frozen modules remain deterministic.
                for start in range(0,len(order),args.accumulation):
                    group=order[start:start+args.accumulation];optimizer.zero_grad(set_to_none=True)
                    values=[]
                    for index in group:
                        case=cases[index];tick=time.monotonic()
                        with runtime.execution_lock:
                            loss,count=supervised_forward(runtime,collator,case)
                            value=float(loss.detach());(loss/len(group)).backward()
                        values.append(value);progress['examples_consumed']+=1
                        progress['actual_training_case_ids'].append(case['case_id'])
                        record={'epoch':epoch+1,'case_id':case['case_id'],'loss':value,'target_tokens':count,'seconds':time.monotonic()-tick}
                        log.write(json.dumps(record)+'\n');log.flush();history.append(record);del loss
                    grads=[p.grad for p in parameters if p.grad is not None]
                    if len(grads)!=len(parameters) or not all(bool(torch.isfinite(g).all()) for g in grads):
                        raise RuntimeError('Missing or nonfinite trainable gradients')
                    norm=torch.nn.utils.clip_grad_norm_(parameters,1.0)
                    optimizer.step();scheduler.step();progress['optimizer_steps']+=1
                    if progress['optimizer_steps']%5==0:
                        event('training',epoch=epoch+1,epoch_examples=min(start+len(group),len(cases)),
                              recent_loss=sum(values)/len(values),gradient_norm=float(norm),learning_rate=scheduler.get_last_lr()[0])
                    if progress['optimizer_steps']%50==0:
                        save_adapter(runtime,out/'latest-adapter',out,progress)
                        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                            'epoch':epoch,'next_case_offset':start+len(group),'torch_rng_state':torch.get_rng_state()},out/'latest-optimizer.pt')
                        event('checkpoint_saved')
                optimizer.zero_grad(set_to_none=True)
                progress['epochs_completed']=epoch+1
                event('development_teacher_evaluation')
                metrics=teacher_evaluate(runtime,collator,selected)
                write_json(out/f'epoch-{epoch+1}-teacher-metrics.json',metrics)
                save_adapter(runtime,out/f'epoch-{epoch+1}-adapter',out,progress)
                if best is None or metrics['token_weighted_nll']<best['token_weighted_nll']:
                    best={'epoch':epoch+1,'token_weighted_nll':metrics['token_weighted_nll']}
                    save_adapter(runtime,out/'adapter',out,progress)
                event('epoch_completed',best_epoch=best['epoch'],development_nll=metrics['token_weighted_nll'])
        after=adapter_parameter_hashes(runtime.model)
        changed=[key for key in before if before[key]!=after[key]]
        if not changed:raise RuntimeError('No adapter tensor changed')
        result={'status':'public_destination_finetuning_completed','completed_at':datetime.now(timezone.utc).isoformat(),
            **progress,'status':'public_destination_finetuning_completed','selected_checkpoint':best,
            'pretrained_development_nll':baseline['token_weighted_nll'],'changed_adapter_tensors':len(changed),
            'adapter_hashes_initial':before,'last_epoch_adapter_hashes':after,'trainable_parameters':runtime.trainable_parameter_count,
            'training_examples_unique':len(set(progress['actual_training_case_ids'])),'epochs':args.epochs,
            'packages':package_versions(),'local_clarification_training_complete':False,
            'fresh_reload_pending':True,'free_spatial_comparison_pending':True}
        selected_manifest=json.loads((out/'adapter/grounding_adapter_manifest.json').read_text())
        result['selected_adapter_parameter_hashes']=selected_manifest['adapter_parameter_hashes']
        result['selected_adapter_files']=selected_manifest['files']
        write_json(out/'training_result.json',result)
        trained=replace(config,adapter_dir=str((out/'adapter').resolve()))
        write_json(out/'inference_config.json',asdict(trained));write_json(out/'pretrained_config.json',asdict(replace(config,adapter_dir=None)))
        event('training_stage_completed',selected_epoch=best['epoch'])
        return result
    except Exception as exc:
        write_json(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'progress':progress})
        event('failed');raise
    finally:
        if runtime is not None:
            runtime.close()


def evaluate(args):
    import torch
    from trackrag.dropoff.data import load_cases
    from trackrag.dropoff.streamed import StreamedRuntime
    config=GroundingConfig.from_mapping(json.loads(args.config.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists() or args.out.with_suffix('.summary.json').exists():
        raise FileExistsError('Choose a new evaluation output path')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.90,0)
    if args.development.stat().st_size>64*1024**2:
        raise ValueError('Development case file exceeds the snapshot budget')
    case_snapshot=args.development.read_bytes()
    source_sha=hashlib.sha256(case_snapshot).hexdigest()
    cases=load_cases(args.development,allowed_partitions={'development'},payload=case_snapshot)
    for row in cases:
        row['_training_image_sha256']=row['source']['image_sha256']
    if args.protocol:
        protocol=json.loads(args.protocol.read_text())
        chosen=controlled_contrast_cases(cases,protocol,source_sha,sha256_file(args.protocol))
    else:
        chosen=cohort(cases,args.generation_limit)
    runtime=StreamedRuntime(config)
    try:
        if config.adapter_dir:
            write_json(args.out.with_suffix('.adapter-hashes.json'),adapter_parameter_hashes(runtime.model))
        teacher=None
        if args.teacher_all:
            teacher=teacher_evaluate(runtime,GroundingCollator(runtime.processor,config.max_input_tokens),cases)
            write_json(args.out.with_suffix('.teacher-metrics.json'),teacher)
        result=free_evaluate(runtime,chosen,args.out)
        result['development_source_sha256']=source_sha
        result['contrast_protocol_sha256']=sha256_file(args.protocol) if args.protocol else None
        result['all_development_teacher_nll']=teacher['token_weighted_nll'] if teacher else None
        result['teacher_forced_cases']=teacher['cases'] if teacher else 0
        write_json(args.out.with_suffix('.summary.json'),result)
        print(json.dumps(result),flush=True)
        return result
    finally:runtime.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('train');fit.add_argument('--train',type=Path,required=True);fit.add_argument('--development',type=Path,required=True)
    fit.add_argument('--config',type=Path,required=True);fit.add_argument('--out',type=Path,required=True)
    fit.add_argument('--epochs',type=int,default=2);fit.add_argument('--learning-rate',type=float,default=1e-4)
    fit.add_argument('--accumulation',type=int,default=2);fit.add_argument('--dev-limit',type=int,default=32)
    fit.add_argument('--generation-limit',type=int,default=20);fit.add_argument('--preflight-generation',action='store_true')
    check=commands.add_parser('evaluate');check.add_argument('--config',type=Path,required=True);check.add_argument('--development',type=Path,required=True)
    check.add_argument('--out',type=Path,required=True);check.add_argument('--generation-limit',type=int,default=20)
    check.add_argument('--protocol',type=Path,help='Frozen same-image language contrast protocol')
    check.add_argument('--teacher-all',action='store_true',help='Also measure teacher-forced NLL on the complete development partition')
    args=parser.parse_args(argv)
    if args.command=='train':
        if not 1<=args.epochs<=5 or not 1<=args.accumulation<=8 or not 0<args.learning_rate<=.001 or args.dev_limit<1 or args.generation_limit<1:
            parser.error('Invalid training settings')
        train(args)
    else:evaluate(args)
    return 0

if __name__=='__main__':raise SystemExit(main())
