"""Adapt a contributor key for retrieval diagnostics and ungraded QA review."""
import argparse
import json
from pathlib import Path

from .compare import write_bundle,_md
from .run_audit import fingerprint
from .score import read_jsonl,score_results
from trackrag.questions import load_questions
from trackrag.evidence_time import requested_frame_ids


def prepare_annotations(questions_path,key_path,output_dir,split_manifest=None):
    questions=load_questions(questions_path)
    references=read_jsonl(key_path)
    lookup={q['question_id']:q for q in questions}
    if not lookup:
        raise ValueError('At least one question is required')
    keys={}
    gold=[]
    review=[]
    for reference in references:
        if set(reference)!={'question_id','expected_answer','answerable','acceptable_track_ids','acceptable_intervals'}:
            raise ValueError('Contributor key must contain exactly the five documented fields')
        qid=reference['question_id']
        if qid in keys or qid not in lookup:
            raise ValueError('Key IDs must match questions exactly once')
        question=lookup[qid]
        answerable=reference['answerable']
        text=reference['expected_answer']
        tracks=reference['acceptable_track_ids']
        intervals=reference['acceptable_intervals']
        if type(answerable) is not bool or not isinstance(text,str) or not text.strip():
            raise ValueError('Key requires a boolean answerability flag and expected answer text')
        if (not isinstance(tracks,list) or any(type(track) is not int or track<0 for track in tracks)
                or len(set(tracks))!=len(tracks) or not isinstance(intervals,list)):
            raise ValueError('Key track IDs and intervals must be valid lists')
        if answerable!=(bool(tracks) and bool(intervals)) or (not answerable and (tracks or intervals)):
            raise ValueError('Answerable keys need tracks and intervals; unanswerable keys use empty lists')
        alternatives=[]
        if len(tracks)>1 and len(intervals)>1:
            raise ValueError('Multiple tracks and intervals require explicit pairing before conversion')
        for interval in intervals:
            if set(interval)!={'recording_id','start_frame','end_frame'} or interval['recording_id']!=question['recording_id']:
                raise ValueError('Key interval must match the public recording')
            start,end=interval['start_frame'],interval['end_frame']
            if type(start) is not int or type(end) is not int or not 0<=start<=end<=question['query_frame']:
                raise ValueError('Key interval is invalid or uses future frames')
            alternatives.extend({'track_id':track,'start_frame':start,'end_frame':end} for track in tracks)
        if len({(row['track_id'],row['start_frame'],row['end_frame']) for row in alternatives})!=len(alternatives):
            raise ValueError('Key has duplicate retrieval alternatives')
        keys[qid]=reference
        gold.append({'question_id':qid,'recording_id':question['recording_id'],'query_frame':question['query_frame'],
                     'answerable':answerable,'acceptable_evidence':alternatives})
        review.append({'question_id':qid,'question':question['question'],'recording_id':question['recording_id'],
            'query_frame':question['query_frame'],'contributor_expected_answer':text,'contributor_answerable':answerable,
            'contributor_track_ids':tracks,'contributor_intervals':intervals,
            'explicit_requested_frame_ids':requested_frame_ids(question['question']),
            'source_verified':False,'independent_review_status':'pending','reviewer':'','note':''})
    if set(keys)!=set(lookup):
        raise ValueError('Every question must have exactly one key row')
    gold.sort(key=lambda row:list(lookup).index(row['question_id']))
    review.sort(key=lambda row:list(lookup).index(row['question_id']))
    # Exercise the actual scorer schema. Missing results are not model scores.
    score_results([],gold)
    inputs=[{'path':str(path),'sha256':fingerprint(path)} for path in (questions_path,key_path)]
    partitions={}
    if split_manifest is not None:
        splits=json.loads(Path(split_manifest).read_text(encoding='utf-8-sig'))
        for question in questions:
            matches=[name for name,recordings in splits.items() if question['recording_id'] in recordings]
            if len(matches)!=1:
                raise ValueError('Recording must have one unique declared partition')
            partitions[question['recording_id']]=matches[0]
        inputs.append({'path':str(split_manifest),'sha256':fingerprint(split_manifest)})
    metadata={'version':1,'scope':'Candidate contributor labels, structurally checked; no visual or human validation.',
        'question_count':len(questions),'answerable_count':sum(row['answerable'] for row in gold),
        'unanswerable_count':sum(not row['answerable'] for row in gold),'inputs':inputs,
        'declared_recording_partitions':partitions,'source_verified':False,'human_validated':False,
        'retrieval_metric_scope':'Any listed track/interval alternative is sufficient for a retrieval hit. This is not complete answer correctness.',
        'required_qa_review':['Verify counts and complete identity sets; one retrieved member is insufficient.',
            'Verify every requested timestamp; temporal overlap alone does not prove all listed observations.',
            'Verify visible facts against original frames and annotation-specific facts against the pinned source labels.',
            'Check abstentions separately from provider errors. Keep gold outside every inference input.',
            'Do not use test-partition recordings/questions for fine-tuning. A split declaration alone does not certify an untouched evaluation set.']}
    worksheet=['# Candidate Task 3 question review','',metadata['scope'],'',metadata['retrieval_metric_scope'],'',
        'All verification fields start pending. Fill reviewer, evidence checked, disagreements and the corrected reference only after inspection.',
        'Keep the contributor key unchanged; this packet does not certify its answers or redefine count/set questions as any-target questions.','']
    for row in review:
        worksheet.extend([f"## {_md(row['question_id'])}",'',_md(row['question']),
            f"Recording: {_md(row['recording_id'])}; cutoff: {row['query_frame']}; requested frames: {row['explicit_requested_frame_ids']}.",
            f"Contributor tracks: {row['contributor_track_ids']}; expected answer: {_md(row['contributor_expected_answer'])}",
            'Review status: pending. No inspected source frames or independent sign-off are supplied by this export.',''])
    write_bundle(output_dir,{'retrieval-key.jsonl':''.join(json.dumps(row)+'\n' for row in gold),
        'qa-review.jsonl':''.join(json.dumps(row)+'\n' for row in review),
        'metadata.json':json.dumps(metadata,indent=2)+'\n','worksheet.md':'\n'.join(worksheet)+'\n'})
    return metadata


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--key',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split-manifest',type=Path)
    args=parser.parse_args(argv)
    try:
        report=prepare_annotations(args.questions,args.key,args.out,args.split_manifest)
    except (ValueError,KeyError,TypeError,OSError) as error:
        parser.error(str(error))
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
