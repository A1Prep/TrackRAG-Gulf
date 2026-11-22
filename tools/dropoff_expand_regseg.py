"""Expand pinned public destination supervision, preserving existing split scenes.

This adds no status, ambiguity, anchor, dialogue, or local human-review labels.
The reserved concise-test scenes remain excluded by the shared source guard.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from tools import dropoff_bootstrap_regseg as source
from tools.dropoff_split_public import _export_row, _identity
from trackrag.dropoff.data import load_cases, validate_disjoint_groups


def historical_locks(path):
    path=Path(path)
    payload=path.read_bytes()
    if len(payload)>1024**2:
        raise ValueError('Historical split registry is too large')
    registry=json.loads(payload)
    if registry.get('schema_version')!=1 or registry.get('kind')!='historical_public_scene_assignments':
        raise ValueError('Invalid historical split registry')
    identities={};locked={}
    for case_id,row in registry['cases'].items():
        value=row['identity'];partition=row['partition']
        if partition not in ('train','development') or not isinstance(value,list) or len(value)!=6:
            raise ValueError('Invalid historical identity or partition')
        if value[0] in locked and locked[value[0]]!=partition:
            raise ValueError('Historical source scene appears in conflicting partitions')
        identities[case_id]=tuple(value);locked[value[0]]=partition
    if len(identities)!=registry['case_count']:
        raise ValueError('Historical case count mismatch')
    return locked,identities,{'registry_sha256':hashlib.sha256(payload).hexdigest(),
                              **registry['original_split_sha256']}


def expand(output, *, limit=625, prefix_mib=192, previous_splits=None, previous_locks=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new data directory')
    if type(limit) is not int or not 101 <= limit <= 1200:
        raise ValueError('Expanded public corpus must request 101..1200 cases')
    if type(prefix_mib) is not int or not 32 <= prefix_mib <= 320:
        raise ValueError('Image prefix must be 32..320 MiB')
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 600*1024**2:
        raise RuntimeError('At least 600 MiB free is required before data expansion')
    budget = source.BoundedSource(512*1024**2)
    print(json.dumps({'stage':'pinned_source_metadata','requested_cases':limit}),flush=True)
    metadata = json.loads(budget.small_file(source.METADATA_URL,source.METADATA_SIZE,source.METADATA_SHA256))
    excluded = source.excluded_source_ids(budget.small_file(source.TRAIN_REWRITE_URL,7211,source.TRAIN_REWRITE_SHA256))
    scene_metadata = json.loads(budget.small_file(source.SCENE_METADATA_URL,source.SCENE_METADATA_SIZE,source.SCENE_METADATA_SHA256))
    scenes, reserved = source.verified_scene_map(metadata,scene_metadata,excluded)
    tail=budget.fetch(source.ARCHIVE_URL,source.ARCHIVE_SIZE-65557,source.ARCHIVE_SIZE-1,total=source.ARCHIVE_SIZE)
    eocd=tail.rfind(b'PK\x05\x06')
    if eocd<313 or source.sha256(tail[eocd-313:eocd]) != source.ARCHIVE_DIRECTORY_SHA256:
        raise source.BootstrapError('Pinned archive central directory changed')
    directory=source.zip_directory(tail,source.ARCHIVE_SIZE)
    image_name,mask_name='Talk2Car-RefSeg/images.tar','Talk2Car-RefSeg/masks.tar'
    print(json.dumps({'stage':'bounded_image_download','prefix_mib':prefix_mib}),flush=True)
    image_tar,image_receipt=source.read_outer_member(budget,image_name,directory[image_name],
        prefix_bytes=prefix_mib*1024**2,max_uncompressed=(prefix_mib+64)*1024**2)
    images=source.tar_files(image_tar,r'imgs/img_train_(\d+)\.jpg',partial=True)
    del image_tar
    print(json.dumps({'stage':'source_images_read','images':len(images)}),flush=True)
    mask_tar,mask_receipt=source.read_outer_member(budget,mask_name,directory[mask_name])
    if mask_receipt['compressed_sha256'] != source.MASK_MEMBER_COMPRESSED_SHA256:
        raise source.BootstrapError('Pinned masks SHA256 mismatch')
    masks=source.tar_files(mask_tar,r'(?:masks|val_masks_new)/gt_img_ann_train_(\d+)\.png')
    del mask_tar
    provenance={'repository_revision':source.SOURCE_REVISION,'metadata_url':source.METADATA_URL,
        'metadata_sha256':source.METADATA_SHA256,'archive_url':source.ARCHIVE_URL,
        'archive_size':source.ARCHIVE_SIZE,'archive_full_sha256_verified':False,
        'concise_test_train_exclusions_sha256':source.TRAIN_REWRITE_SHA256,
        'scene_metadata_url':source.SCENE_METADATA_URL,'scene_metadata_sha256':source.SCENE_METADATA_SHA256,
        'excluded_concise_test_train_scene_count':len(reserved)}
    rejected=[]
    rows=source.build_cases(metadata,images,masks,excluded,output,limit=limit,source_receipt=provenance,
        scene_map=scenes,excluded_scenes=reserved,rejected_records=rejected)
    del images,masks,metadata,scene_metadata
    rows=load_cases(output/'cases.jsonl',allowed_partitions={'train'})
    locked={}
    previous_identities={}
    previous_files={}
    if previous_splits is None:
        registry=previous_locks or Path(__file__).resolve().parents[1]/'configs/dropoff/public-historical-split-locks.json'
        locked,previous_identities,previous_files=historical_locks(registry)
    if previous_splits:
        for name,partition in [('train.jsonl','train'),('development.jsonl','development')]:
            previous_file=Path(previous_splits)/name
            previous_files[name]=hashlib.sha256(previous_file.read_bytes()).hexdigest()
            for row in load_cases(previous_file,allowed_partitions={partition}):
                group=row['scene_group']
                if group in locked and locked[group]!=partition:
                    raise ValueError('Previous scene appears in conflicting partitions')
                locked[group]=partition
                previous_identities[row['case_id']]=_identity(row)
    identities={row['case_id']:_identity(row) for row in rows}
    if any(identities.get(key)!=value for key,value in previous_identities.items()):
        raise ValueError('A prior training/development case is missing or changed')
    train,dev=[],[]
    for row in rows:
        group=row['scene_group']
        partition=locked.get(group)
        if partition is None:
            partition='development' if int(hashlib.sha256(('42:'+group).encode()).hexdigest(),16)%5==0 else 'train'
        copy=deepcopy(row);copy['partition']=partition
        (dev if partition=='development' else train).append(copy)
    if not train or not dev:
        raise ValueError('Both independent partitions are required')
    validate_disjoint_groups(train,dev)
    assignments={row['scene_group']:row['partition'] for row in train+dev}
    if any(assignments.get(group)!=partition for group,partition in locked.items()):
        raise ValueError('Prior scene partition was not preserved')
    splits=output/'splits';splits.mkdir()
    files={}
    for name,partition,items in [('train.jsonl','train',train),('development.jsonl','development',dev)]:
        payload=''.join(json.dumps(_export_row(row,partition,splits),sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n' for row in sorted(items,key=lambda x:x['case_id'])).encode()
        load_cases(splits/name,allowed_partitions={partition},payload=payload)
        (splits/name).write_bytes(payload)
        files[name]={'sha256':hashlib.sha256(payload).hexdigest(),'rows':len(items)}
    receipt={'schema_version':1,'kind':'expanded_public_destination_training_corpus',
        'requested_cases':limit,'accepted_cases':len(rows),'rejected_records':rejected,
        'source_revision':source.SOURCE_REVISION,'source_scene_mapping_verified':True,
        'reserved_scene_count':len(reserved),'reserved_scenes_excluded':True,
        'previous_scene_partition_preserved':bool(previous_identities),
        'previous_case_identities_preserved':len(previous_identities),
        'previous_split_sha256':previous_files,
        'scene_counts':{'train':len({x['scene_group'] for x in train}),'development':len({x['scene_group'] for x in dev})},
        'files':files,'image_member':image_receipt,'mask_member':mask_receipt,
        'downloaded_bytes':budget.used_bytes,'downloads':budget.receipts,
        'supervised_fields':['destination_point_xy','destination_box_xyxy'],
        'local_human_reviewed_interpretation_cases':0,'final_test':False,
        'warning':'Public region supervision only. No status, clarification, anchor or local EMT labels are fabricated.'}
    # Retain the master as a correctly partitioned corpus; it must not silently
    # expose development rows to a train-only loader.
    master=''.join(json.dumps(_export_row(row,row['partition'],output),sort_keys=True,allow_nan=False)+'\n'
                   for row in sorted(train+dev,key=lambda x:x['case_id']))
    load_cases(output/'cases.jsonl',allowed_partitions={'train','development'},payload=master.encode())
    (output/'cases.jsonl').write_text(master,encoding='utf-8')
    receipt['master_corpus_partitioned']=True
    (output/'expansion-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (splits/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=625)
    parser.add_argument('--image-prefix-mib',type=int,default=192)
    parser.add_argument('--previous-splits',type=Path)
    parser.add_argument('--previous-locks',type=Path)
    args=parser.parse_args(argv)
    if args.previous_splits and args.previous_locks:
        parser.error('Select previous split files or the portable lock registry, not both')
    result=expand(args.out,limit=args.limit,prefix_mib=args.image_prefix_mib,previous_splits=args.previous_splits,previous_locks=args.previous_locks)
    print(json.dumps({k:result[k] for k in ['accepted_cases','files','scene_counts','downloaded_bytes']}),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
