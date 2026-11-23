"""Actual installed Google SDK, bounded local HTTP transport; no live requests."""
import base64
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
        response_status = status[len(calls)-1] if isinstance(status, tuple) else status
        if response_status == "timeout":
            raise httpx.ReadTimeout("simulated response timeout", request=request)
        if response_status == "connection":
            raise httpx.ConnectError("simulated connection failure", request=request)
        if response_status == 503:
            return httpx.Response(503, json={'error': {'code': 'service_unavailable', 'message': 'Simulated high demand'}})
        if response_status != 200:
            return httpx.Response(response_status,json={'error':{'code':response_status,'message':'Simulated HTTP error'}})
        text=reply if reply is not None else json.dumps({'answer_status':'answered','answer':'Predicted track 99.','cited_evidence_ids':['E1']})
        return httpx.Response(200,json={'id':'offline-response','status':'completed','model':'gemini-offline-test',
            'steps':[{'type':'model_output','content':[{'type':'text','text':text}]}]})
    http=httpx.Client(transport=httpx.MockTransport(transport))
    installed_client = genai.Client
    def offline_client(*, api_key, http_options):
        return installed_client(api_key=api_key, http_options={**http_options, 'httpx_client': http})
    with patch.object(genai, 'Client', offline_client), patch.dict(os.environ, GEMINI_API_KEY='offline-test-key'):
        model = GeminiAnswerModel('gemini-offline-test')
    return model,question,evidence,calls,http


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
        assert len(model.last_receipt['interaction_attempts']) == 1
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
        assert model.last_receipt['interaction_attempts'][0]['request_status'] == 'completed'
    finally:
        http.close()


@pytest.mark.parametrize("statuses, final_state", [((503, 200), "answered"), ((503, 503), "error"), ((503, 429), "error")])
def test_actual_interactions_503_recovery_preserves_both_attempts(tmp_path, monkeypatch, statuses, final_state):
    delay = Mock()
    monkeypatch.setattr("trackrag.model_client.sleep", delay)
    model, question, evidence, calls, http = setup_client(tmp_path, status=statuses)
    question["question"] = "Which predicted track is visible at frame 10?"
    image = evidence[0]["images"][0]
    image["image_path"] = "processed.png"
    (tmp_path/"tracks.jsonl").write_text(json.dumps({"recording_id":"rec", "frame_id":10, "track_id":99,
        "source":"predicted", "class_name":"Bus", "bbox_xyxy":image["bbox_xyxy"]})+"\n", encoding="utf8")
    try:
        delivered = answer_question(question, evidence, tmp_path, model)
        assert delivered["answer_status"] == final_state
        assert len(calls) == 2 and calls[0] == calls[1]
        receipt = json.loads(next(tmp_path.glob("hosted/*/receipt.json")).read_bytes())
        first, second = receipt["interaction_attempts"]
        assert first == {"attempt":1, "request_status":"error", "http_status":503,
                         "error_type":"InternalServerError", "retry_delay_seconds":1}
        assert second["attempt"] == 2
        if final_state == "answered":
            assert second == {"attempt":2, "request_status":"completed", "provider_status":"completed"}
            assert receipt["request_status"] == "completed" and receipt["response_id"] == "offline-response"
            assert "http_status" not in receipt
        else:
            assert second["request_status"] == receipt["request_status"] == "error"
            assert second["http_status"] == receipt["http_status"] == statuses[-1]
        assert len(receipt["sent"]) == 1
        delay.assert_called_once_with(1)
    finally:
        http.close()


@pytest.mark.parametrize("status", [400, 401, 403, 408, 409, 500, 502, 504, "timeout", "connection"])
def test_other_actual_interactions_errors_make_one_attempt(tmp_path, status):
    model, question, evidence, calls, http = setup_client(tmp_path, status=status)
    try:
        with pytest.raises(Exception):
            model.generate(question,evidence)
        assert len(calls) == len(model.last_receipt["interaction_attempts"]) == 1
        assert model.last_receipt["request_status"] == "error"
    finally:
        http.close()


def test_local_paths_and_generic_503_exceptions_cannot_trigger_a_retry(tmp_path):
    model, question, evidence, calls, http = setup_client(tmp_path)
    try:
        error = RuntimeError("service unavailable 503")
        error.response = httpx.Response(503)
        create = Mock(side_effect=error)
        model.client = SimpleNamespace(interactions=SimpleNamespace(create=create))
        with pytest.raises(RuntimeError):
            model.generate(question,evidence)
        assert create.call_count == 1 and not calls
        evidence[0]["images"][0]["image_path"] = str(tmp_path/"absent.png")
        with pytest.raises(FileNotFoundError):
            model.generate(question,evidence)
        assert create.call_count == 1 and not model.last_receipt
    finally:
        http.close()


@pytest.mark.parametrize('constant',['NaN','1e999'])
def test_nonfinite_provider_json_preserves_error_receipt_and_raw_text(tmp_path,constant):
    raw='{"answer_status":"answered","answer":'+constant+',"cited_evidence_ids":["E1"]}'
    model,question,evidence,calls,http=setup_client(tmp_path,reply=raw)
    image=evidence[0]['images'][0]
    image['image_path']='processed.png'
    (tmp_path/'tracks.jsonl').write_text(json.dumps({'recording_id':'rec','frame_id':10,'track_id':99,
        'source':'predicted','class_name':'Bus','bbox_xyxy':[1,1,8,8]})+'\n')
    try:
        # Exercise the transport with the source actually supplied in this fixture.
        result=answer_question({**question, 'question': 'Which predicted track is visible at frame 10?'},evidence,tmp_path,model)
        assert result['answer_status']=='error'
        assert len(calls)==1
        receipt=json.loads(next(tmp_path.glob('hosted/*/receipt.json')).read_text())
        assert receipt['raw_text']==raw
        assert receipt['request_status']=='completed'
        assert receipt['error_type']=='ValueError'
        assert not receipt.get('response')
    finally:
        http.close()
