"""Score two frozen direct-point result sets against local reference areas."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def parse(text):
    if text.strip() == 'NONE':
        return 'abstained', None
    point = re.fullmatch(r'POINT\s+(\d{1,4})\s+(\d{1,4})',text.strip())
    if point and all(0 <= int(v) <= 1000 for v in point.groups()):
        return 'point', tuple(map(int,point.groups()))
    return 'malformed', None


def inside(point, polygon):
    import cv2
    import numpy as np

    return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32),point,False) >= 0


def score(folder):
    load = lambda name: json.loads((folder/name).read_bytes())
    inputs = load('neutral-inputs.json')['cases']
    reference = load('frozen-reference.json')['cases']
    arms = {name:load(file) for name,file in [('8B','qwen8-raw.json'),('35B','qwen35-raw.json')]}
    if any(len(arm['results']) != len(inputs) for arm in arms.values()) or len(reference) != len(inputs):
        raise ValueError('Each paired result must contain exactly one decision per reference')
    rows=[]
    totals={}
    for i, (item, gold) in enumerate(zip(inputs,reference)):
        if item['source'] != gold['source']:
            raise ValueError('Mismatched reference')
        w,h=item['native_size']
        row = dict(case=gold['id'],source=item['source'],request=item['words'],native_size=item['native_size'],
                   input_size=item['input_size'],source_sha256=item['original_sha256'],common_view_sha256=item['input_sha256'],
                   cohort='previously_preregistered' if gold['id'].startswith('P') else 'development',
                   reference=gold['decision'],reference_polygons=gold['polygons'],outputs={})
        for name, arm in arms.items():
            output=arm['results'][i]
            if any(output[field] != value for field,value in item.items()):
                raise ValueError('The arms did not receive identical inputs')
            decision, normalized=parse(output['raw'])
            point = [w*normalized[0]/1000,h*normalized[1]/1000] if normalized else None
            hit = point is not None and any(inside(tuple(point),poly) for poly in gold['polygons'])
            category = ('format_error' if decision=='malformed' else
                        'correct_abstention' if gold['decision']=='withhold' and point is None else
                        'false_point' if gold['decision']=='withhold' else
                        'missed_abstention' if point is None else 'inside' if hit else 'outside')
            ok = category in {'correct_abstention','inside'}
            detail=dict(raw_reply=output['raw'],normalized_xy=normalized,native_xy=point,score=category,
                        correct=ok,wall_seconds=output['seconds'],prompt_tokens=output['prompt_tokens'],
                        output_tokens=output['output_tokens'])
            if 'image_grid' in output:
                detail['image_grid_thw']=output['image_grid']
            row['outputs'][name]=detail
            count=totals.setdefault(row['cohort'],{}).setdefault(name,Counter())
            count['cases']+=1;count['correct']+=int(ok);count[category]+=1
        rows.append(row)
    sources={name:{k:v for k,v in arm.items() if k!='results'} for name,arm in arms.items()}
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/name for name in
                       ('qwen35-raw.json','qwen8-raw.json','protocol-freeze.json','runtime-amendment.json','frozen-reference.json')]}
    return {'scope':'Direct point-or-abstain comparison. No outline produced. No measurement of the end-to-end passenger app.',
            'summary':totals, 'models':sources, 'hashes':hashes, 'cases':rows}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('output', type=Path)
    args=parser.parse_args()
    result=score(args.folder)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['summary']))
