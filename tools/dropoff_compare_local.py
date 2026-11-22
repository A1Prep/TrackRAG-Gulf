"""Compare frozen, matched pretrained/trained public grounding outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from tools.dropoff_train import sha256_file,write_json


def read_results(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    indexed={row['case_id']:row for row in rows}
    if not rows or len(indexed)!=len(rows):
        raise ValueError('Empty cohort or duplicate result identities')
    for row in rows:
        if type(row.get('ok')) is not bool or type(row.get('point_in_mask')) is not bool:
            raise ValueError('Result coverage/point fields must be explicit booleans')
        if not row['ok'] and row['point_in_mask']:
            raise ValueError('A failed result cannot receive point accuracy credit')
        value=row['coarse_box_iou']
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid IoU value')
        if not row['ok'] and value!=0:
            raise ValueError('A failed result cannot receive IoU credit')
        distance=row.get('normalized_point_error')
        if row['ok'] and (isinstance(distance,bool) or not isinstance(distance,(int,float))
                or not math.isfinite(distance) or not 0<=distance<=math.sqrt(2)):
            raise ValueError('Invalid normalized point distance')
    return rows,indexed


def metrics(rows):
    n=len(rows);valid=[r for r in rows if r['ok']]
    return {'requested':n,'valid_geometry':len(valid),'coverage':len(valid)/n,
        'point_hits':sum(r['point_in_mask'] for r in rows),
        'point_accuracy_all_requests':sum(r['point_in_mask'] for r in rows)/n,
        'coarse_box_iou_all_requests':sum(r['coarse_box_iou'] for r in rows)/n,
        'mean_normalized_point_error_valid_only':sum(r['normalized_point_error'] for r in valid)/len(valid) if valid else None,
        'error_types':dict(Counter(r.get('error',{}).get('type','unknown') for r in rows if not r['ok']))}


def compare(pretrained,trained,*,protocol=None):
    a,ai=read_results(pretrained);b,bi=read_results(trained)
    if set(ai)!=set(bi):
        raise ValueError('Requested case identities differ between model conditions')
    configs=[]
    for p,rows in [(pretrained,a),(trained,b)]:
        summary=json.loads(Path(p).with_suffix('.summary.json').read_text())
        if (summary.get('results_sha256')!=sha256_file(p)
                or summary.get('requested_cases')!=len(rows)
                or summary.get('case_ids')!=[row['case_id'] for row in rows]):
            raise ValueError('Result hash or cohort differs from its condition summary')
        configs.append(summary)
    ac=dict(configs[0]['runtime']);bc=dict(configs[1]['runtime'])
    baseline_adapter=ac.pop('adapter_dir',None);trained_adapter=bc.pop('adapter_dir',None)
    if ac!=bc or baseline_adapter is not None or not trained_adapter:
        raise ValueError('Conditions are not a matched pretrained-versus-adapter comparison')
    source_sha=configs[0].get('development_source_sha256')
    if (not isinstance(source_sha,str) or len(source_sha)!=64
            or any(c not in '0123456789abcdef' for c in source_sha)
            or source_sha!=configs[1].get('development_source_sha256')):
        raise ValueError('Reference data snapshots differ')
    for key in ai:
        if not ai[key].get('request_sha256') or ai[key]['request_sha256']!=bi[key].get('request_sha256'):
            raise ValueError('Model inputs differ or lack a request binding: '+key)
        if not ai[key].get('prompt_sha256') or ai[key]['prompt_sha256']!=bi[key].get('prompt_sha256'):
            raise ValueError('Rendered model prompts differ or lack a binding: '+key)
    result={'schema_version':1,'kind':'matched_public_development_comparison_not_final_EMT',
        'pretrained':metrics(a),'trained':metrics(b),'case_ids':sorted(ai),
        'pretrained_results_sha256':sha256_file(pretrained),'trained_results_sha256':sha256_file(trained),
        'development_source_sha256':configs[0].get('development_source_sha256'),
        'preprocessing_and_decoding_matched':True,'failures_retained_in_denominator':True,
        'improved_case_ids':[k for k in ai if not ai[k]['point_in_mask'] and bi[k]['point_in_mask']],
        'regressed_case_ids':[k for k in ai if ai[k]['point_in_mask'] and not bi[k]['point_in_mask']],
        'public_masks_score_clarification':False}
    if protocol:
        spec=json.loads(Path(protocol).read_text());groups=[]
        protocol_sha=sha256_file(protocol)
        if (any(summary.get('contrast_protocol_sha256')!=protocol_sha for summary in configs)
                or spec.get('development_source_sha256')!=source_sha):
            raise ValueError('Contrast protocol differs from the generated conditions or reference snapshot')
        identities=[key for group in spec['groups'] for key in group['case_ids']]
        expected=set(identities)
        if expected!=set(ai) or len(identities)!=len(expected) or spec.get('case_count')!=len(expected):
            raise ValueError('Contrast protocol cohort differs from results')
        for group in spec['groups']:
            ids=group['case_ids'];row={'image_sha256':group['image_sha256'],'case_ids':ids,
                'reference_masks_pairwise_disjoint':group.get('all_reference_masks_pairwise_disjoint')}
            for label,indexed in [('pretrained',ai),('trained',bi)]:
                contexts={indexed[k].get('scene_context_sha256') for k in ids}
                if len(contexts)!=1 or None in contexts:
                    raise ValueError('Contrast changed non-instruction context')
                points=[indexed[k]['result']['destination_point_xy'] for k in ids if indexed[k]['ok']]
                distinct=(len(points)==len(ids) and all(math.dist(points[i],points[j])>1
                          for i in range(len(points)) for j in range(i)))
                row[label]={'all_references_hit':all(indexed[k]['point_in_mask'] for k in ids),
                    'different_proposals_beyond_one_pixel':distinct,
                    'numerical_difference_alone_is_accuracy':False}
            groups.append(row)
        result['same_image_contrast']={'groups':groups,'group_count':len(groups),
            'pretrained_all_reference_groups':sum(g['pretrained']['all_references_hit'] for g in groups),
            'trained_all_reference_groups':sum(g['trained']['all_references_hit'] for g in groups),
            'protocol_sha256':protocol_sha,
            'accuracy_alone_establishes_language_sensitivity_for_overlapping_masks':False,
            'disjoint_reference_group_count':sum(g['reference_masks_pairwise_disjoint'] is True for g in groups),
            'pretrained_disjoint_all_reference_groups':sum(g['reference_masks_pairwise_disjoint'] is True and g['pretrained']['all_references_hit'] for g in groups),
            'trained_disjoint_all_reference_groups':sum(g['reference_masks_pairwise_disjoint'] is True and g['trained']['all_references_hit'] for g in groups)}
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained',type=Path,required=True);parser.add_argument('--trained',type=Path,required=True)
    parser.add_argument('--protocol',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():raise FileExistsError('Choose a new comparison output path')
    result=compare(args.pretrained,args.trained,protocol=args.protocol)
    args.out.parent.mkdir(parents=True,exist_ok=True);write_json(args.out,result)
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
