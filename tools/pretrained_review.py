"""Connect actual pretrained comparisons to the standard reviewer and Task 3.

Embeddings may be reused from a checkpoint/runtime/text-bound vector cache.
The run receipt distinguishes new HTTP inference from cached model outputs.
"""
import argparse
import json
from pathlib import Path

from evaluation.run_audit import fingerprint
from evaluation.score import read_jsonl
from trackrag.emt import open_recording,load_annotated_tracks
from trackrag.questions import load_questions
from trackrag.retrieval import rank_evidence as lexical_rank
from trackrag.review import review_recording
from tools.local_embeddings import LocalEmbeddings
from tools.pretrained_retrieval import MODEL_SHA
from tools.pretrained_candidates import (build_index,candidate_catalog,describe,
    ranked_intervals,reserve_current_scene,tfidf_scores)

VARIANTS=('published-task4','expanded-tfidf-temporal','expanded-tfidf-static',
          'expanded-qwen8b-temporal','expanded-qwen8b-static')


def select(question,index,recording,variant,policy,client=None):
    expanded=variant!='published-task4'
    pool=candidate_catalog(question,index,expanded=expanded)
    if not expanded:
        ranked=lexical_rank(question,index)
    else:
        text=[describe(c,recording,temporal=variant.endswith('-temporal')) for c in pool]
        query=question['question']+f" Review cutoff: frame {question['query_frame']}."
        if 'qwen8b' in variant:
            if client is None:
                raise ValueError('Actual Qwen embeddings or their bound cache are required')
            scores=client.rank(query,text) if pool else []
        else:
            scores=tfidf_scores(query,text)
        ranked=ranked_intervals(pool,scores)
    return reserve_current_scene(ranked,pool) if policy=='reserved_current_scene' else ranked


def run(args):
    recording=open_recording(args.data_root,args.recording)
    questions=[q for q in load_questions(args.questions) if q['recording_id']==args.recording]
    if not questions:
        raise ValueError('No questions for the selected recording')
    source='annotated' if args.condition=='annotated' else 'predicted'
    provenance={'condition':args.condition,'variant':args.variant,'policy':args.policy,
        'scope':'Explicit pretrained development adapter; genuine predictions or separately labeled annotation condition.',
        'questions_sha256':fingerprint(args.questions),'fine_tuned':False}
    if source=='annotated':
        tracks=load_annotated_tracks(recording)
    else:
        summary_path=args.vision_run/'summary.json'
        summary=json.loads(summary_path.read_text())
        rec=next(row for row in summary['recordings'] if row['recording_id']==args.recording)
        method='sparseOptFlow' if args.condition=='botsort-gmc' else 'none'
        tracker=next(row for row in rec['trackers'] if row['gmc_method']==method)
        path=Path(tracker['tracks_path'])
        if fingerprint(path)!=tracker['tracks_sha256']:
            raise ValueError('Saved predictions changed after model inference')
        tracks=read_jsonl(path)
        provenance.update(vision_summary_sha256=fingerprint(summary_path),tracks_sha256=fingerprint(path),
                          detector=summary['model'],detector_checkpoint_sha256=summary['checkpoint']['sha256'])
    client=LocalEmbeddings(args.server,MODEL_SHA,args.vector_cache) if 'qwen8b' in args.variant else None
    if client:
        provenance.update(embedding_checkpoint_sha256=MODEL_SHA,embedding_runtime=client.runtime_id,
                          cache_semantics='Reuse actual model vectors only when checkpoint/runtime/pooling/exact text agree.')
    answer=None
    if args.answerer=='gemini':
        from trackrag.answering import answer_question
        answer=answer_question
    try:
        path=review_recording(recording,questions,source=source,load_tracks=lambda rec:tracks,
            build_index=build_index,
            rank_evidence=lambda q,index,k=3:select(q,index,recording,args.variant,args.policy,client)[:k],
            answer_question=answer,output_dir=args.out,
            run_metadata={**provenance,'answerer':args.answerer,'adapter':'tools.pretrained_review'})
        receipt={'new_http_calls':client.http_calls if client else 0,'new_vectors':client.new_vectors if client else 0,
                 'http_events':client.events if client else [],'results_sha256':fingerprint(path),
                 'cached_vector_outputs_may_be_reused':bool(client),'model_metadata':provenance}
        (args.out/'pretrained-receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return path
    finally:
        if client:
            client.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--vision-run',type=Path,required=True)
    parser.add_argument('--recording',required=True)
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--condition',choices=('annotated','botsort-gmc','botsort-no-gmc'),required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--policy',choices=('raw','reserved_current_scene'),default='raw')
    parser.add_argument('--answerer',choices=('none','gemini'),default='none')
    parser.add_argument('--server',default='http://127.0.0.1:8091')
    parser.add_argument('--vector-cache',type=Path,default=Path('runs/qwen8b-vectors.sqlite'))
    parser.add_argument('--out',type=Path,required=True)
    print(run(parser.parse_args(argv)))


if __name__=='__main__':
    main()
