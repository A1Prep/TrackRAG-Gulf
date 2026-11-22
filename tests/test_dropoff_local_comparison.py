"""Synthetic protocol/scoring checks; no model inference or quality claim."""
from copy import deepcopy
import hashlib
import json

import pytest

from tools.dropoff_fit_local import controlled_contrast_cases
from tools.dropoff_compare_local import compare


def source_cases():
    return [{'case_id':name,'source':{'image_sha256':'a'*64},'target':{'hidden_gold':index},
        'request':{'request_id':name,'scene':{'frame':{'recording_id':name,'frame_id':0,'timestamp_s':0,
            'width':200,'height':100,'image_path':name+'.jpg'},'source':'predicted','tracks':[]},
            'text':text,'input_mode':'typed','dialogue':[]}}
        for index,(name,text) in enumerate([('a','park by the left car'),('b','park by the right car')])]


def protocol():
    return {'kind':'public_same_image_language_contrast','development_source_sha256':'d'*64,
        'groups':[{'image_sha256':'a'*64,'case_ids':['a','b'],'canonical_recording_id':'shared'}],'case_count':2}


def test_controlled_contrast_changes_only_artificial_recording_identity_and_preserves_gold_separation():
    cases=source_cases();before=deepcopy(cases)
    rows=controlled_contrast_cases(cases,protocol(),'d'*64,'e'*64)
    assert cases==before
    assert [r['request']['scene']['frame']['recording_id'] for r in rows]==['shared','shared']
    assert [r['request']['text'] for r in rows]==['park by the left car','park by the right car']
    assert rows[0]['request']['scene']['frame']['frame_id']==0
    assert 'target' not in rows[0]['request']
    assert [r['_contrast']['original_recording_id'] for r in rows]==['a','b']


@pytest.mark.parametrize('mutation',['source','duplicate','missing','image','context','tracks','dialogue','count'])
def test_contrast_rejects_unmatched_or_uncontrolled_inputs(mutation):
    cases=source_cases();spec=protocol()
    if mutation=='source':spec['development_source_sha256']='f'*64
    elif mutation=='duplicate':spec['groups'][0]['case_ids']=['a','a']
    elif mutation=='missing':spec['groups'][0]['case_ids']=['a','absent']
    elif mutation=='image':cases[1]['source']['image_sha256']='f'*64
    elif mutation=='context':cases[1]['request']['scene']['frame']['frame_id']=10
    elif mutation=='tracks':cases[0]['request']['scene']['tracks']=[{'future':True}]
    elif mutation=='dialogue':cases[0]['request']['dialogue']=[{'role':'user','text':'different context'}]
    else:spec['case_count']=3
    with pytest.raises(ValueError):controlled_contrast_cases(cases,spec,'d'*64,'e'*64)


def result(case,*,ok=True,hit=True,point=(10,20)):
    return {'case_id':case,'ok':ok,'point_in_mask':hit if ok else False,
        'coarse_box_iou':.5 if ok else 0,'normalized_point_error':.1 if ok else None,
        'request_sha256':hashlib.sha256(case.encode()).hexdigest(),
        'prompt_sha256':hashlib.sha256(('prompt:'+case).encode()).hexdigest(),
        'scene_context_sha256':hashlib.sha256(b'same context').hexdigest(),
        **({'result':{'destination_point_xy':list(point)}} if ok else {'error':{'type':'TimeoutError'}})}


def files(tmp_path,base_rows,trained_rows):
    paths=[]
    for name,rows,adapter in [('pretrained',base_rows,None),('trained',trained_rows,'selected/adapter')]:
        path=tmp_path/(name+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        path.with_suffix('.summary.json').write_text(json.dumps({'runtime':{'model_id':'pinned','revision':'a'*40,
            'max_pixels':262144,'adapter_dir':adapter},'development_source_sha256':'d'*64,
            'results_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'requested_cases':len(rows),'case_ids':[r['case_id'] for r in rows]}))
        paths.append(path)
    return paths


def test_matched_comparison_preserves_failures_and_reports_regressions(tmp_path):
    a,b=files(tmp_path,[result('a'),result('b',ok=False)],
        [result('a',hit=False),result('b')])
    report=compare(a,b)
    assert report['pretrained']['requested']==report['trained']['requested']==2
    assert report['pretrained']['coverage']==.5 and report['trained']['coverage']==1
    assert report['pretrained']['point_accuracy_all_requests']==.5
    assert report['trained']['point_accuracy_all_requests']==.5
    assert report['improved_case_ids']==['b'] and report['regressed_case_ids']==['a']
    assert report['pretrained']['error_types']=={'TimeoutError':1}


@pytest.mark.parametrize('mutation',['ids','duplicate','request','prompt','config','missing_source','source','failed_credit','nonfinite_distance','stale_hash','stale_cohort'])
def test_comparison_rejects_mismatched_or_miscredited_results(tmp_path,mutation):
    first=[result('a'),result('b')];second=deepcopy(first)
    if mutation=='ids':second[1]['case_id']='c'
    elif mutation=='duplicate':second[1]=deepcopy(second[0])
    elif mutation=='request':second[0]['request_sha256']='f'*64
    elif mutation=='prompt':second[0]['prompt_sha256']='f'*64
    elif mutation=='failed_credit':second[0]['ok']=False
    elif mutation=='nonfinite_distance':second[0]['normalized_point_error']=float('nan')
    a,b=files(tmp_path,first,second)
    if mutation=='stale_hash':b.write_text(b.read_text()+'\n')
    if mutation=='stale_cohort':
        path=b.with_suffix('.summary.json');meta=json.loads(path.read_text());meta['case_ids']=['not-the-generated-case'];path.write_text(json.dumps(meta))
    if mutation in ('config','missing_source','source'):
        path=b.with_suffix('.summary.json');meta=json.loads(path.read_text())
        if mutation=='config':meta['runtime']['max_pixels']=65536
        elif mutation=='missing_source':meta.pop('development_source_sha256')
        else:meta['development_source_sha256']='f'*64
        path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):compare(a,b)


def test_same_image_numeric_changes_are_separate_from_reference_accuracy(tmp_path):
    a,b=files(tmp_path,[result('a'),result('b')],[result('a',point=(20,30)),result('b',point=(100,30))])
    spec=tmp_path/'protocol.json';spec.write_text(json.dumps(protocol()))
    bind_protocol((a,b),spec)
    report=compare(a,b,protocol=spec)['same_image_contrast']
    assert report['pretrained_all_reference_groups']==1
    assert report['groups'][0]['pretrained']['different_proposals_beyond_one_pixel'] is False
    assert report['groups'][0]['trained']['different_proposals_beyond_one_pixel'] is True
    assert report['groups'][0]['trained']['numerical_difference_alone_is_accuracy'] is False


def bind_protocol(paths,spec):
    for path in paths:
        summary=path.with_suffix('.summary.json');value=json.loads(summary.read_text())
        value['contrast_protocol_sha256']=hashlib.sha256(spec.read_bytes()).hexdigest()
        summary.write_text(json.dumps(value))


@pytest.mark.parametrize('mutation',['stale_protocol','reference_snapshot','duplicate_group','wrong_count'])
def test_comparison_rejects_changed_or_repeated_contrast_protocol(tmp_path,mutation):
    a,b=files(tmp_path,[result('a'),result('b')],[result('a'),result('b')])
    value=protocol()
    if mutation=='reference_snapshot':value['development_source_sha256']='f'*64
    elif mutation=='duplicate_group':value['groups'].append(deepcopy(value['groups'][0]))
    elif mutation=='wrong_count':value['case_count']=3
    spec=tmp_path/'protocol.json';spec.write_text(json.dumps(value))
    bind_protocol((a,b),spec)
    if mutation=='stale_protocol':spec.write_text(spec.read_text()+'\n')
    with pytest.raises(ValueError):compare(a,b,protocol=spec)
