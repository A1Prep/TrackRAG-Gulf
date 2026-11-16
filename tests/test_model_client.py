"""Actual installed Google SDK, bounded local HTTP transport; no live requests."""
import base64
import json

import httpx
import pytest
from google import genai
from PIL import Image

from trackrag.model_client import GeminiAnswerModel
from trackrag.answering import answer_question


def setup_client(tmp_path,reply=None,status=200):
    image=tmp_path/'processed.png'
    Image.new('RGB',(16,12),(35,35,35)).save(image)
    question={'question_id':'q','recording_id':'rec','query_frame':10,'question':'Which annotated track is visible at frame 10?'}
    evidence=[{'evidence_id':'E1','interval':{'recording_id':'rec','source':'predicted','track_id':99,
        'start_frame':10,'end_frame':10,'rank':1,'score':1.0},'images':[{'frame_id':10,'bbox_xyxy':[1,1,8,8],
        'image_path':str(image),'observations':[{'track_id':99,'class_name':'Bus','source':'predicted','bbox_xyxy':[1,1,8,8]}],
        'privacy':{'original_size_wh':[16,12],'processed_size_wh':[16,12]}}]}]
    calls=[]
    def transport(request):
        calls.append(json.loads(request.content))
        if status!=200:
            return httpx.Response(status,json={'error':{'code':status,'message':'Simulated quota error','status':'RESOURCE_EXHAUSTED'}})
        text=reply if reply is not None else json.dumps({'answer_status':'answered','answer':'Predicted track 99.','cited_evidence_ids':['E1']})
        return httpx.Response(200,json={'id':'offline-response','status':'completed','model':'gemini-offline-test',
            'steps':[{'type':'model_output','content':[{'type':'text','text':text}]}]})
    http=httpx.Client(transport=httpx.MockTransport(transport))
    sdk=genai.Client(api_key='offline-test-key',http_options={'httpx_client':http,
        'retry_options':{'attempts':1,'http_status_codes':[500,502,503,504]}})
    return GeminiAnswerModel('gemini-offline-test',client=sdk),question,evidence,calls,http


def test_sdk_serializes_images_status_schema_source_and_query_time(tmp_path):
    model,question,evidence,calls,http=setup_client(tmp_path)
    try:
        answer=model.generate(question,evidence)
        assert answer['cited_evidence_ids']==['E1']
        assert len(calls)==1
        request=calls[0]
        assert request['store'] is False
        assert 'previous_interaction_id' not in request
        assert request['response_format']['schema']['required']==['answer_status','answer','cited_evidence_ids']
        contents=request['input'][0]['content']
        prompt=contents[0]['text']
        assert '"query_frame": 10' in prompt and '"track_id": 99' in prompt
        assert '"source": "predicted"' in prompt
        sent=next(item for item in contents if item['type']=='image')
        assert base64.b64decode(sent['data'])==(tmp_path/'processed.png').read_bytes()
        assert model.last_receipt['response_id']=='offline-response'
        assert model.last_receipt['reported_model']=='gemini-offline-test'
        assert model.last_receipt['request_status']=='completed'
        assert 'offline-test-key' not in json.dumps(model.last_receipt)
    finally:
        http.close()


def test_actual_sdk_transport_error_is_not_an_abstention(tmp_path):
    model,question,evidence,calls,http=setup_client(tmp_path,status=429)
    try:
        with pytest.raises(Exception):
            model.generate(question,evidence)
        assert len(calls)==1
        assert model.last_receipt['request_status']=='error'
        assert model.last_receipt['http_status']==429
    finally:
        http.close()


def test_invalid_json_is_a_validation_error_after_http_success(tmp_path):
    model,question,evidence,calls,http=setup_client(tmp_path,reply='not JSON')
    try:
        with pytest.raises(ValueError):
            model.generate(question,evidence)
        assert model.last_receipt['raw_text']=='not JSON'
        assert model.last_receipt['request_status']=='completed'
        assert len(calls)==1
    finally:
        http.close()


def test_nonfinite_provider_json_preserves_error_receipt_and_raw_text(tmp_path):
    raw='{"answer_status":"answered","answer":NaN,"cited_evidence_ids":["E1"]}'
    model,question,evidence,calls,http=setup_client(tmp_path,reply=raw)
    image=evidence[0]['images'][0]
    image['image_path']='processed.png'
    (tmp_path/'tracks.jsonl').write_text(json.dumps({'recording_id':'rec','frame_id':10,'track_id':99,
        'source':'predicted','class_name':'Bus','bbox_xyxy':[1,1,8,8]})+'\n')
    try:
        result=answer_question(question,evidence,tmp_path,model)
        assert result['answer_status']=='error'
        assert len(calls)==1
        receipt=json.loads(next(tmp_path.glob('hosted/*/receipt.json')).read_text())
        assert receipt['raw_text']==raw
        assert receipt['request_status']=='completed'
        assert receipt['error_type']=='ValueError'
        assert not receipt.get('response')
    finally:
        http.close()
