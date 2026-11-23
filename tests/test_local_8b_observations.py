from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json

import pytest
from PIL import Image

from trackrag.dropoff import bay_landmarks, curb_points, local_8b_observations as local
from trackrag.dropoff.parking_evidence import signed_bytes
from trackrag.dropoff.passenger import digest
from trackrag.dropoff.streamed import METADATA_SHA256, STREAMED_MODEL_ID, WEIGHT_SHA256


def measured(point_verdict, crop_verdict):
    image=Image.new("RGB",(150,90),"grey")
    buffer=BytesIO()
    image.save(buffer,"PNG")
    original=buffer.getvalue()
    key=sha256(original).hexdigest()
    scene={"source":"predicted","provenance":{"image_sha256":key,"network_policy":"local_only"},
           "frame":{"recording_id":"kartaview_123","frame_id":2,"width":150,"height":90},"tracks":[]}
    model={"base":bay_landmarks.MODELS["base"],"adapter":bay_landmarks.MODELS["adapter"],
           "processor":bay_landmarks.MODELS["processor"],"runtime":"local_streamed","quantization":"nf4",
           "weights":{"model_id":STREAMED_MODEL_ID,"revision":bay_landmarks.MODELS["base"],
                      "verified_weights":True,"verified_metadata":True,
                      "files":{name:{"sha256":value} for name,value in {**WEIGHT_SHA256,**METADATA_SHA256}.items()}}}
    shared={"image_sha256":key,"frame_id":2,"recording_id":"kartaview_123","size":[150,90],
            "source_context":digest(curb_points.source_context(scene)),"local_model":model}
    point=[76,70]
    points={**shared,"recipe":curb_points.RECIPE,"model":bay_landmarks.MODELS,
            "guides":digest([curb_points.GUIDES,curb_points.WORDS]),"rows":[
                {"point":point,"views":[sha256(curb_points.view(image,point,close=c)).hexdigest() for c in (False,True)],
                 "replies":[point_verdict,"white van | blue truck","shop entrance","trees"],"crop_verdict":crop_verdict}]}
    landmarks={**shared,"recipe":bay_landmarks.RECIPE,"models":bay_landmarks.MODELS,
            "prompt_sha256":sha256((bay_landmarks.GUIDE+"\n"+bay_landmarks.WORDS).encode()).hexdigest(),
            "bands":[{"view_sha256":sha256(bay_landmarks.band_image(image,i)).hexdigest(),"places":word}
                     for i,word in enumerate(["trees","shop entrance","NONE"])]}
    packets={kind:{"origin":local.ORIGIN,"body_sha256":sha256(signed_bytes(body)).hexdigest(),"body":body}
             for kind,body in [("points",points),("landmarks",landmarks)]}
    return image,original,scene,[{"point":point}],packets


def attach(monkeypatch,scene,packets):
    key=scene["provenance"]["image_sha256"]
    monkeypatch.setattr(local,"_database",lambda:{key:packets})
    def matches(query):
        return (query["provenance"]["image_sha256"]==key and query["frame"]==scene["frame"])
    monkeypatch.setattr(local.local_surfaces,"matches_scene",matches)


@pytest.mark.parametrize("whole,near,visible", [("OPEN","OPEN",True),("OPEN","LANE",False),
    ("OPEN",None,False),("OCCUPIED","OPEN",False)])
def test_only_two_open_source_views_unlock_a_local_point(monkeypatch,whole,near,visible):
    image,_,scene,options,packets=measured(whole,near)
    attach(monkeypatch,scene,packets)
    result=curb_points.observations(image,scene,options,packets["points"])
    assert bool(result)==visible
    if visible:
        assert result[0]["descriptions"]["farther_vehicle"]=="blue truck"


def test_receipt_cannot_change_original_model_or_actor_policy(monkeypatch):
    image,photo,scene,options,packets=measured("OPEN","OPEN")
    attach(monkeypatch,scene,packets)
    key=scene["provenance"]["image_sha256"]
    assert curb_points.get_packet(key)==packets["points"]
    assert bay_landmarks.verify(bay_landmarks.get_packet(key),photo,image.size)==["trees","shop entrance","NONE"]
    assert bay_landmarks.get_packet("a"*64) is None
    changed=deepcopy(packets["points"])
    changed["body"]["rows"][0]["replies"][2]="different storefront"
    changed["body_sha256"]=sha256(signed_bytes(changed["body"])).hexdigest()
    with pytest.raises(ValueError,match="recorded model response"):
        curb_points.observations(image,scene,options,changed)
    wrong=deepcopy(scene)
    wrong["frame"]["frame_id"]=92
    with pytest.raises(ValueError,match="registered current original"):
        curb_points.observations(image,wrong,options,packets["points"])
    monkeypatch.setattr(curb_points.manual_surfaces,"decode",lambda *args: object())
    monkeypatch.setattr(curb_points,"locations",lambda *args: options)
    assert len(curb_points.verified_options(scene,photo,{},packets["points"]))==1
    with pytest.raises(ValueError,match="registered current original"):
        curb_points.verified_options(wrong,photo,{},packets["points"])
    wrong=deepcopy(scene)
    wrong["provenance"].pop("network_policy")
    with pytest.raises(ValueError,match="registered current original"):
        curb_points.observations(image,wrong,options,packets["points"])
    changed=deepcopy(packets["landmarks"])
    changed["body"]["local_model"]["adapter"]="f"*64
    changed["body_sha256"]=sha256(signed_bytes(changed["body"])).hexdigest()
    attach(monkeypatch,scene,{**packets,"landmarks":changed})
    with pytest.raises(ValueError,match="different checkpoint"):
        bay_landmarks.verify(changed,photo,image.size)


def test_unpinned_local_bank_is_ignored(tmp_path,monkeypatch):
    complete=tmp_path/"changed.json"
    complete.write_text(json.dumps({"version":1,"origin":local.ORIGIN,"sources":{"key":{"points":{"body":"forged"}}}}))
    monkeypatch.setattr(local,"BANK",complete)
    local._database.cache_clear()
    try:
        assert local.get_packet("key","points") is None
    finally:
        local._database.cache_clear()
