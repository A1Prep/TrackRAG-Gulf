"""Run a fresh, paced Gemini development cohort, preserving every attempt."""
import argparse
import io
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from hashlib import sha256
from importlib import import_module
from pathlib import Path

from evaluation.run_audit import audit_run, fingerprint
from evaluation.score import read_jsonl
from tools.review_report import write_report
from trackrag.emt import open_recording, load_annotated_tracks
from trackrag.questions import load_questions, validate_questions
from trackrag.review import review_recording


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--questions',type=Path,nargs='+',required=True)
    parser.add_argument('--adapters',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--include',nargs='+')
    parser.add_argument('--exclude',nargs='+',default=[])
    parser.add_argument('--min-interval',type=float,default=13.0)
    args=parser.parse_args(argv)
    if not math.isfinite(args.min_interval) or args.min_interval<0:
        parser.error('min-interval must be finite and nonnegative')
    if args.out.exists():
        parser.error('Output exists; use a fresh output directory')
    questions=validate_questions([q for path in args.questions for q in load_questions(path)])
    available={q['question_id'] for q in questions}
    if (set(args.include or [])|set(args.exclude))-available:
        parser.error('A selected question ID is absent from the input files')
    questions=[q for q in questions if (args.include is None or q['question_id'] in args.include) and q['question_id'] not in args.exclude]
    if not questions:
        parser.error('No questions remain')
    module=import_module(args.adapters)
    os.environ['TRACKRAG_TRIAL_MODEL']=args.model
    if not os.environ.get('GEMINI_API_KEY'):
        parser.error('GEMINI_API_KEY is required')
    args.out.mkdir(parents=True)
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        revision=None
    summary={'version':1,'scope':'Fresh annotated development cohort; no held-out accuracy claim',
             'executed_code_commit':revision,'adapter_module':args.adapters,'requested_model':args.model,
             'adapter_file_sha256':fingerprint(module.__file__),
             'question_inputs':[{'path':str(path.resolve()),'sha256':fingerprint(path)} for path in args.questions],
             'planned_questions':questions,'cases':[],'unattempted_question_ids':[q['question_id'] for q in questions]}
    def save():
        temp=args.out/'summary.tmp'
        temp.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temp.replace(args.out/'summary.json')
    save()
    real_open=urllib.request.urlopen
    last_http=None
    stop=False
    for position,question in enumerate(questions,1):
        case={'question':question,'attempts':[]}
        for attempt in (1,2):
            directory=args.out/f'q-{position:02d}'/f'attempt-{attempt:02d}'
            details={'events':[]}
            def observed_request(request,**kwargs):
                nonlocal last_http
                if last_http is not None:
                    time.sleep(max(0,args.min_interval-(time.monotonic()-last_http)))
                last_http=time.monotonic()
                body=json.loads(request.data)
                required=body.get('generationConfig',{}).get('responseSchema',{}).get('required',[])
                event={'role':'selection' if 'selected' in required else 'answer',
                       'request_body_sha256':sha256(request.data).hexdigest()}
                details['events'].append(event)
                try:
                    with real_open(request,**kwargs) as response:
                        payload=response.read()
                    raw=json.loads(payload)
                    event.update(request_status='completed',reported_model=raw.get('modelVersion'),response_id=raw.get('responseId'))
                    return io.BytesIO(payload)
                except urllib.error.HTTPError as error:
                    error_body=error.read(16384)
                    try:
                        provider=json.loads(error_body).get('error',{})
                    except (ValueError,OSError):
                        provider={}
                    event.update(request_status='error',http_status=error.code,provider_status=provider.get('status'),
                                 provider_message=str(provider.get('message',''))[:1800],quota_details=provider.get('details',[]))
                    # Let the actual provider client inspect RetryInfo/quota
                    # details too; observation must not consume its error body.
                    raise urllib.error.HTTPError(error.url,error.code,error.reason,error.headers,io.BytesIO(error_body)) from error
                except Exception as error:
                    event.update(request_status='error',error_type=type(error).__name__)
                    raise
            print(f"LIVE {position}/{len(questions)} attempt{attempt}: {question['question_id']}",flush=True)
            try:
                recording=open_recording(args.data_root,question['recording_id'])
                # This synchronous CLI records the real HTTP transport. It never
                # substitutes model responses and restores the transport afterward.
                urllib.request.urlopen=observed_request
                try:
                    review_recording(recording,[question],source='annotated',load_tracks=load_annotated_tracks,
                                     build_index=module.build_index,rank_evidence=module.rank_evidence,
                                     answer_question=module.answer_question,export_retrieval=getattr(module,'export_retrieval',None),
                                     output_dir=directory,run_metadata={'adapter_module':args.adapters,'answerer':'gemini','requested_model':args.model})
                finally:
                    urllib.request.urlopen=real_open
                audit_run(directory,{question['question_id']:question})
                write_report(directory)
                result=read_jsonl(directory/'results.jsonl')[0]
                answer=result['answer']
                receipts=list(directory.glob('hosted/*/receipt.json'))
                receipt=json.loads(receipts[0].read_text()) if receipts else {}
                fresh=not receipt.get('cached',False) and any(event['role']=='answer' and event['request_status']=='completed' for event in details['events'])
                item={'run_directory':directory.as_posix(),'answer':answer,'fresh_answer_response':fresh,'provider':details}
                (directory/'provider-details.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
            except Exception as error:
                item={'run_directory':directory.as_posix() if directory.exists() else None,
                      'answer':{'answer_status':'error','answer':f'{type(error).__name__}: {error}','cited_evidence_ids':[]},
                      'fresh_answer_response':False,'provider':details}
            case['attempts'].append(item)
            print(json.dumps({'question_id':question['question_id'],'answer':item['answer'],'fresh':item['fresh_answer_response']}),flush=True)
            if item['answer']['answer_status']!='error' and item['fresh_answer_response']:
                break
            daily='perday' in json.dumps(details).lower() or 'per day' in json.dumps(details).lower()
            if daily:
                stop=True
                summary['stop_reason']='Daily provider limit; unsubmitted questions remain explicit'
                break
        case['selected_attempt']=len(case['attempts'])
        summary['cases'].append(case)
        summary['unattempted_question_ids']=[q['question_id'] for q in questions[position:]]
        save()
        if stop:
            break
    summary['all_planned_cases_have_fresh_responses']=len(summary['cases'])==len(questions) and all(c['attempts'][-1]['fresh_answer_response'] and c['attempts'][-1]['answer']['answer_status']!='error' for c in summary['cases'])
    save()
    print('Saved fresh cohort:',args.out/'summary.json',flush=True)
    return 0 if summary['all_planned_cases_have_fresh_responses'] else 1


if __name__=='__main__':
    raise SystemExit(main())
