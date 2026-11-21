"""Optional teacher-label codec checks with synthetic masks, no model or CUDA."""

from copy import deepcopy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

from PIL import Image
import pytest

from tools import dropoff_fit_local as fit
from trackrag.dropoff.formatting import build_training_messages


def make_case(tmp_path, *, width=1600, height=900, point=(1018.5, 587.5),
              box=(1000, 580, 1040, 600), pixels=None):
    image_path, mask_path = tmp_path/'frame.png', tmp_path/'mask.png'
    Image.new('RGB', (width, height)).save(image_path)
    mask=Image.new('L',(width,height),255 if pixels is None else 0)
    for pixel in pixels or []:mask.putpixel(pixel,255)
    mask.save(mask_path)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    return {'case_id':'synthetic-one','partition':'train','scene_group':'synthetic-scene',
        'request':{'request_id':'synthetic-one','text':'Stop beside the entrance','input_mode':'typed','dialogue':[],
            'scene':{'source':'predicted','tracks':[],'frame':{'recording_id':'synthetic-recording',
                'frame_id':0,'timestamp_s':0,'width':width,'height':height,'image_path':str(image_path)}}},
        'source':{'dataset':'Talk2Car-RegSeg','image_sha256':sha(image_path),'mask_path':str(mask_path),
                  'mask_sha256':sha(mask_path),'mask_width':width,'mask_height':height},
        'review':{'status':'public_annotation'},'supervised_fields':['destination_point_xy','destination_box_xyxy'],
        'target':{'destination_point_xy':list(point),'destination_box_xyxy':list(box)}}


def encode(case, **kwargs):
    return fit.prepare_target_encoding([case],fit.INTEGER_TARGET_ENCODING,**kwargs)


def test_default_uses_original_builder_and_exact_float_sorted_serialization(tmp_path):
    case=make_case(tmp_path);before=deepcopy(case)
    builder,receipt=fit.prepare_target_encoding([case])
    expected=build_training_messages(case)[-1]['content']
    assert builder is None
    assert expected=='{"destination_box_xyxy":[625.0,644.444444,650.0,666.666667],"destination_point_xy":[636.5625,652.777778]}'
    assert receipt['cases'][case['case_id']]['transformed_target_sha256']==hashlib.sha256(expected.encode()).hexdigest()
    assert case==before


def test_integer_label_adjusts_to_valid_adjacent_mask_point_without_mutating_case(tmp_path):
    case=make_case(tmp_path,pixels=[(1018,587),(1017,587)]);before=deepcopy(case)
    builder,receipt=encode(case,source_bindings={'train_sha256':'a'*64})
    messages=builder(case);target=json.loads(messages[-1]['content'])
    assert target['destination_point_xy']==[636,653]
    assert target['destination_box_xyxy']==[625,644,650,667]
    assert list(target)==['destination_point_xy','destination_box_xyxy']
    assert 'destination_box_xyxy' not in messages[-1]['content'].split('destination_point_xy')[0]
    assert set(target)==set(case['supervised_fields']) and 'status' not in target
    assert receipt['mask_adjusted_cases']==1 and receipt['runtime_prediction_snapping'] is False
    assert receipt['original_labels_modified'] is False and case==before
    assert receipt['source_bindings']['train_sha256']=='a'*64
    assert receipt['cases'][case['case_id']]['mask_sha256']==case['source']['mask_sha256']
    assert receipt['cases'][case['case_id']]['transformed_target_sha256']==hashlib.sha256(messages[-1]['content'].encode()).hexdigest()


def test_adjacent_point_minimizes_original_pixel_distance_not_normalized_distance(tmp_path):
    case=make_case(tmp_path,point=(162.24,91.17),box=(150,80,180,110),
                   pixels=[(162,91),(161,91),(163,90)])
    builder,_=encode(case)
    assert json.loads(builder(case)[-1]['content'])['destination_point_xy']==[101,102]


def test_training_only_boundary_point_clips_to999_with_valid_outward_box(tmp_path):
    case=make_case(tmp_path,width=1000,height=1000,point=(999.9,999.9),box=(999,999,1000,1000))
    builder,receipt=encode(case);target=json.loads(builder(case)[-1]['content'])
    assert target=={'destination_point_xy':[999,999],'destination_box_xyxy':[999,999,1000,1000]}
    assert receipt['clipped_to_999_cases']==1
    with pytest.raises(ValueError):
        fit.parse_destination('{"destination_point_xy":[1000,999],"destination_box_xyxy":[999,999,1000,1000]}',1000,1000)


def test_teacher_label_mask_sampling_uses_native_mask_dimensions(tmp_path):
    case=make_case(tmp_path,width=200,height=100,point=(125,75),box=(100,50,150,100))
    mask=Image.new('L',(4,2),0);mask.putpixel((2,1),255)
    path=Path(case['source']['mask_path']);mask.save(path)
    case['source'].update(mask_width=4,mask_height=2,mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    builder,receipt=encode(case)
    assert json.loads(builder(case)[-1]['content'])['destination_point_xy']==[625,750]
    assert receipt['mask_adjusted_cases']==0


def test_unrepresentable_thin_mask_fails_preflight_instead_of_wrong_teacher_label(tmp_path):
    case=make_case(tmp_path,width=5000,height=100,point=(1.5,50.5),box=(1,50,2,51),pixels=[(1,50)])
    with pytest.raises(ValueError,match='No valid adjacent integer label'):
        encode(case)


def test_mask_mutation_is_rejected_before_label_transformation(tmp_path):
    case=make_case(tmp_path)
    Path(case['source']['mask_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='mask bytes changed'):encode(case)


@pytest.mark.parametrize('change',['target','input','mask_binding','image_binding','partition'])
def test_case_changes_after_preflight_fail_and_receipt_mutation_cannot_change_builder(tmp_path,change):
    case=make_case(tmp_path);builder,receipt=encode(case)
    receipt['cases'][case['case_id']]['input_sha256']='externally mutated receipt'
    assert builder(case)[-1]['role']=='assistant'
    changed=deepcopy(case)
    if change=='target':changed['target']['destination_point_xy'][0]+=.1
    elif change=='input':changed['request']['text']='A different instruction'
    elif change=='mask_binding':changed['source']['mask_sha256']='0'*64
    elif change=='image_binding':changed['source']['image_sha256']='0'*64
    else:changed['partition']='development'
    with pytest.raises(ValueError,match='changed after'):builder(changed)


@pytest.mark.parametrize('change',['status_label','review','dataset','test_partition'])
def test_optional_codec_never_invents_full_status_supervision(tmp_path,change):
    case=make_case(tmp_path)
    if change=='status_label':
        case['target'].update(status='proposed',clarification_question=None)
        case['supervised_fields']+=['status','clarification_question']
    elif change=='review':case['review']['status']='human_reviewed'
    elif change=='dataset':case['source']['dataset']='another-source'
    else:case['partition']='test'
    with pytest.raises(ValueError,match='public destination-only'):encode(case)


def test_cli_defaults_and_explicit_optional_teacher_encoding(monkeypatch):
    captured=[];monkeypatch.setattr(fit,'train',lambda args:captured.append(args))
    common=['train','--train','train.jsonl','--development','dev.jsonl','--config','base.json','--out','new-run']
    fit.main(common);assert captured[-1].target_encoding==fit.SOURCE_TARGET_ENCODING
    fit.main(common+['--target-encoding',fit.INTEGER_TARGET_ENCODING])
    assert captured[-1].target_encoding==fit.INTEGER_TARGET_ENCODING
    monkeypatch.setattr(fit,'evaluate',lambda args:captured.append(args))
    fit.main(['evaluate','--development','dev.jsonl','--config','base.json','--out','evaluation.jsonl',
              '--teacher-target-encoding',fit.INTEGER_TARGET_ENCODING])
    assert captured[-1].teacher_target_encoding==fit.INTEGER_TARGET_ENCODING


def test_teacher_nll_labels_actual_codec_without_claiming_spatial_accuracy(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setattr(fit,'supervised_forward',lambda *args:(.5,12))
    runtime=SimpleNamespace(execution_lock=threading.RLock(),model=SimpleNamespace(eval=lambda:None))
    metrics=fit.teacher_evaluate(runtime,object(),[{'case_id':'synthetic-one'}],target_encoding=fit.INTEGER_TARGET_ENCODING)
    assert metrics['target_encoding']==fit.INTEGER_TARGET_ENCODING
    assert metrics['nll_comparison_scope']=='within_encoding_not_cross_codec_accuracy'
    assert metrics['kind']=='teacher_forced_target_nll_not_spatial_accuracy'


def test_controlled_codec_rejects_adapter_continuation_before_any_cuda_call(tmp_path,monkeypatch):
    from dataclasses import asdict
    from trackrag.dropoff.model import GroundingConfig
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    path=tmp_path/'config.json'
    fit.write_json(path,asdict(GroundingConfig(adapter_dir='previous-adapter')))
    with pytest.raises(ValueError,match='fresh base adapter'):
        fit.train(SimpleNamespace(config=path,target_encoding=fit.INTEGER_TARGET_ENCODING))


def test_saved_adapter_binds_encoding_receipt_and_detects_receipt_mutation(tmp_path,monkeypatch):
    case=make_case(tmp_path);_,receipt=encode(case)
    fit.write_json(tmp_path/'target-encoding.json',receipt)
    receipt_sha=fit.sha256_file(tmp_path/'target-encoding.json')
    fit.write_json(tmp_path/'training_settings.json',{'target_encoding':fit.INTEGER_TARGET_ENCODING,
        'target_encoding_receipt_sha256':receipt_sha})
    fit.write_json(tmp_path/'dataset_manifest.json',{'synthetic_fixture':True})
    def save(path,**kwargs):
        for name in ['adapter_config.json','adapter_model.safetensors']:(path/name).write_bytes(b'fixture')
    model=SimpleNamespace(peft_config={'default':SimpleNamespace()},save_pretrained=save)
    runtime=SimpleNamespace(model=model,processor=SimpleNamespace(save_pretrained=lambda p:None),
        config=SimpleNamespace(model_id='mock8b',revision='mock-revision',min_pixels=65536,max_pixels=262144,suffix_layers=8),
        lora_target_modules='synthetic-module',initial_adapter_hashes={'lora':'0'*64})
    monkeypatch.setattr(fit,'adapter_parameter_hashes',lambda model:{'lora':'1'*64})
    progress={'optimizer_steps':1,'examples_consumed':1}
    manifest=fit.save_adapter(runtime,tmp_path/'adapter',tmp_path,progress)
    assert manifest['target_encoding']==fit.INTEGER_TARGET_ENCODING
    assert manifest['target_encoding_receipt_sha256']==receipt_sha
    (tmp_path/'target-encoding.json').write_text('changed')
    with pytest.raises(ValueError,match='receipt changed'):
        fit.save_adapter(runtime,tmp_path/'must-not-write',tmp_path,progress)
    assert not (tmp_path/'must-not-write').exists()
