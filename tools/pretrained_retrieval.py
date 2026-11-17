"""Run real, explicitly selected pretrained text retrieval; never load a gold key.

Annotated tracks isolate retrieval quality. The other two conditions consume
saved genuine RF-DETR/BoT-SORT outputs. They are not training results.
"""
import argparse
import json
import subprocess
import time
from hashlib import sha256
from pathlib import Path

from evaluation.run_audit import fingerprint
from evaluation.score import read_jsonl
from trackrag.emt import open_recording,load_annotated_tracks
from trackrag.retrieval import rank_evidence
from tools.local_embeddings import LocalEmbeddings
from tools.pretrained_candidates import (build_index,candidate_catalog,describe,
    tfidf_scores,ranked_intervals,reserve_current_scene)

MODEL_SHA='3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85'


def write_row(stream,row):
    stream.write(json.dumps(row,allow_nan=False)+'\n')
    stream.flush()


def run(args):
    if args.out.exists():
        raise FileExistsError('Choose a fresh result directory; the vector cache can be reused')
    questions=[row for path in args.questions for row in read_jsonl(path)]
    if len({q['question_id'] for q in questions})!=len(questions):
        raise ValueError('Duplicate public question IDs')
    recordings={name:open_recording(args.data_root,name) for name in sorted({q['recording_id'] for q in questions})}
    vision=json.loads((args.vision_run/'summary.json').read_text())
    by_name={row['recording_id']:row for row in vision['recordings']}
    args.out.mkdir(parents=True)
    client=LocalEmbeddings(args.server,MODEL_SHA,args.vector_cache,batch_size=args.batch_size) if 'qwen' in args.rankers else None
    started=time.perf_counter()
    counts=[]
    track_inputs=[]
    input_files=[*args.questions,args.vision_run/'summary.json']
    try:
        with (args.out/'rankings.jsonl').open('w',encoding='utf-8') as outputs, (args.out/'candidates.jsonl').open('w',encoding='utf-8') as inventories:
            for condition in args.conditions:
                indexes={}
                for name,recording in recordings.items():
                    if condition=='annotated':
                        tracks=load_annotated_tracks(recording)
                    else:
                        method='sparseOptFlow' if condition=='botsort-gmc' else 'none'
                        info=next(row for row in by_name[name]['trackers'] if row['gmc_method']==method)
                        path=Path(info['tracks_path'])
                        if fingerprint(path)!=info['tracks_sha256']:
                            raise ValueError('Saved genuine predictions changed after inference')
                        input_files.append(path)
                        tracks=read_jsonl(path)
                    indexes[name]=build_index(recording,tracks)
                    track_inputs.append({'condition':condition,'recording_id':name,'source':indexes[name].source,
                        'observation_count':len(tracks),'canonical_rows_sha256':sha256(json.dumps(tracks,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
                for q in questions:
                    index=indexes[q['recording_id']]
                    recording=recordings[q['recording_id']]
                    baseline=candidate_catalog(q,index,expanded=False)
                    expanded=candidate_catalog(q,index,expanded=True)
                    text={mode:[describe(c,recording,temporal=(mode=='temporal')) for c in expanded] for mode in ('temporal','static')}
                    # Query time is public metadata, identical across expanded rankers.
                    query=q['question']+f" Review cutoff: frame {q['query_frame']}."
                    inventory=[]
                    for pos,candidate in enumerate(expanded):
                        inventory.append({**{key:value for key,value in candidate.items() if key!='rows'},
                            'observed_frames':[r['frame_id'] for r in candidate['rows']],
                            'temporal_text':text['temporal'][pos],'static_text':text['static'][pos]})
                    write_row(inventories,{'condition':condition,'question':q,'candidates':inventory,
                        'published_pool_candidate_ids':[c['candidate_id'] for c in baseline]})
                    def emit(variant,raw,pool):
                        write_row(outputs,{'condition':condition,'variant':variant,'question':q,'source':index.source,
                            'raw':raw,'reserved_current_scene':reserve_current_scene(raw,pool)})
                    if 'lexical' in args.rankers:
                        emit('published-task4',rank_evidence(q,index),baseline)
                    for mode in ('temporal','static'):
                        if 'tfidf' in args.rankers:
                            emit('expanded-tfidf-'+mode,ranked_intervals(expanded,tfidf_scores(query,text[mode])),expanded)
                        if client:
                            scores=client.rank(query,text[mode]) if expanded else []
                            emit('expanded-qwen8b-'+mode,ranked_intervals(expanded,scores),expanded)
                    counts.append({'condition':condition,'question_id':q['question_id'],'published_candidates':len(baseline),'expanded_candidates':len(expanded)})
                    print(f"RETRIEVAL {condition} {q['question_id']}: {len(expanded)} bounded candidates",flush=True)
        summary={'version':1,'scope':'Development retrieval only; no generation, fine-tuning or held-out score.',
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'adapter_hashes':{name:fingerprint(Path(__file__).with_name(name)) for name in ('pretrained_retrieval.py','pretrained_candidates.py','local_embeddings.py')},
            'inputs':[{'path':str(path),'sha256':fingerprint(path)} for path in dict.fromkeys(input_files)],
            'conditions':args.conditions,'rankers':args.rankers,'question_count':len(questions),'candidate_counts':counts,'track_inputs':track_inputs,
            'seconds':time.perf_counter()-started,'model_sha256':MODEL_SHA if client else None,
            'runtime_id':client.runtime_id if client else None,'new_http_calls':client.http_calls if client else 0,
            'new_vectors':client.new_vectors if client else 0,'http_events':client.events if client else [],
            'ranking_sha256':fingerprint(args.out/'rankings.jsonl'),'candidate_sha256':fingerprint(args.out/'candidates.jsonl'),
            'query_policy':'Public question plus numeric cutoff, same for expanded TF-IDF and dense retrieval; fixed query instruction only for Qwen.',
            'gold_access':'No key or rubric is opened by this command.'}
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return summary
    finally:
        if client:
            client.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--vision-run',type=Path,required=True)
    parser.add_argument('--questions',type=Path,nargs='+',required=True)
    parser.add_argument('--conditions',nargs='+',choices=('annotated','botsort-gmc','botsort-no-gmc'),default=['annotated','botsort-gmc','botsort-no-gmc'])
    parser.add_argument('--rankers',nargs='+',choices=('lexical','tfidf','qwen'),default=['lexical','tfidf','qwen'])
    parser.add_argument('--server',default='http://127.0.0.1:8091')
    parser.add_argument('--vector-cache',type=Path,default=Path('runs/qwen8b-vectors.sqlite'))
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args(argv))


if __name__=='__main__':
    main()
