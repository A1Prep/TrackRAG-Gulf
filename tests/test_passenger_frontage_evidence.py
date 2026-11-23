"""Exercise the release's signed frontage replay and slow-service fallback with synthetic pixels."""
import base64
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from threading import Event
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tests.test_passenger_place_grounding import case, packet as place_packet
from trackrag.dropoff import frontage_evidence as frontage, frontage_verifier, parking_evidence, place_evidence, place_grounding
from trackrag.dropoff.passenger import digest


def signed(body, key):
    return {"body":body, "signature":key.sign(parking_evidence.signed_bytes(body)).hex()}


def trial(tmp_path, monkeypatch):
    req, photo, classes, key = case(tmp_path, monkeypatch)
    classes.fill(13)
    def prior(current):
        packet = place_packet(current, photo, classes, key)
        return place_evidence.build_outcome(current, photo, packet)
    ctx = frontage.context_for(req)
    word, image_sha = ctx["words"], ctx["image_sha256"]
    xy = [500, 500]
    out = BytesIO()
    Image.new("L", tuple(ctx["size"]), 10).save(out,format="PNG")
    pixels = out.getvalue()
    with Image.open(BytesIO(photo)) as im:
        calls = [{**place_grounding.call_identity(stage,guide,word,frontage_verifier.photo_view(im,coord)),
                  "response":reply} for stage,guide,coord,reply in (
                      ("scene",frontage_verifier.SCENE,None,"OPEN"),("surface",frontage_verifier.SURFACE,xy,"BAY"))]
    checks = signed({"recipe":"frontage_point_crop_check_v1","model":"trained_qwen3_vl_8b","words":word,
                     "image_sha256":image_sha,"point_1000":xy,"calls":calls,"class_map":{
                         "png":base64.b64encode(pixels).decode(),"sha256":sha256(pixels).hexdigest()}}, key)
    out = BytesIO()
    np.savez_compressed(out, source=image_sha, words=word, revision=frontage.confidence.SURFACE_REVISION,
                       queries=[*frontage.confidence.QUERIES, "furniture store"],
                       logits=np.zeros((8,96,144),dtype=np.float16))
    maps = out.getvalue()
    specialist = {"point_1000":xy,"image_sha256":image_sha,"words":word,"revision":frontage.confidence.POINT_REVISION,
                  "point_answer":'<points coords="1 1 500 500">patch</points>',"check":"YES"}
    combined = signed({"recipe":frontage.METHOD,"image_sha256":image_sha,"context_sha256":digest(ctx),
                       "input_sha256":frontage.model_input(ctx),"point":specialist,"checks":checks,
                       "map_npz":base64.b64encode(maps).decode(),"map_sha256":sha256(maps).hexdigest()}, key)
    # The release scoring weights are independently tested and exercised on real photos.
    monkeypatch.setattr(frontage.confidence,"predict",lambda features,_:{
        "accept":not features["veto"],"score":1.0,"reason":features["veto"]})
    public = SimpleNamespace(python="unused",root=tmp_path,allowed={(image_sha,"kartaview_42",2)})
    service = frontage.PublicFrontageGrounder(SimpleNamespace(interpret=prior),public,[],tmp_path/"cache")
    return req, photo, prior, combined, service


def test_signed_frontage_replays_photograph_prior_and_current_words(tmp_path, monkeypatch):
    req,photo,prior,combined,_ = trial(tmp_path,monkeypatch)
    output = frontage.build_outcome(req,photo,combined,prior(req))
    assert output["result"]["destination_point_xy"]==[48.0,32.0]
    assert frontage.verify(req,output)["language_calls"]>=4
    forged=deepcopy(output)
    forged["result"]["destination_point_xy"][0] += 1
    with pytest.raises(ValueError):
        frontage.verify(req,forged)
    for field in ("point","map_npz","checks"):
        forged = deepcopy(combined)
        forged["body"][field]=None
        with pytest.raises(ValueError):
            frontage.build_outcome(req,photo,forged,prior(req))
    new = {**req, "text":"At the hospital, please."}
    with pytest.raises(ValueError):
        frontage.build_outcome(new,photo,combined,prior(new))
    with pytest.raises(ValueError):
        frontage.build_outcome(req,photo,combined,output)


def test_slow_service_deduplicates_then_rebinds_to_each_request(tmp_path, monkeypatch):
    req,photo,prior,combined,service = trial(tmp_path,monkeypatch)
    entered,release = Event(), Event()
    calls=[]
    def run(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(returncode=0,stdout=json.dumps(combined).encode())
    monkeypatch.setattr(frontage.subprocess,"run",run)
    monkeypatch.setattr(frontage,"INTERACTIVE_WAIT_SECONDS",.02)
    try:
        assert service.interpret(req)==prior(req)
        assert entered.wait(2)
        other = {**req, "request_id":"another-session"}
        assert service.interpret(other)==prior(other)
        assert len(calls)==1
        with service._pending_lock:
            job = next(iter(service._pending.values()))
    finally:
        release.set()
    job.result(3)
    repeated = service.interpret(other)
    assert repeated["request_id"]==other["request_id"] and frontage.verify(other,repeated)["decision_reproduced"]
    assert len(calls)==1


def test_disk_failure_preserves_verified_result(tmp_path, monkeypatch):
    req,_,_,combined,service = trial(tmp_path,monkeypatch)
    service.cache.write_text("already a file",encoding="utf-8")
    monkeypatch.setattr(frontage.subprocess,"run",lambda *args,**kw:SimpleNamespace(returncode=0,stdout=json.dumps(combined).encode()))
    assert service.interpret(req)["result"]["status"]=="proposed"
