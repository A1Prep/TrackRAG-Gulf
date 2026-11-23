"""Place-route regression tests. These software fixtures do not estimate model accuracy."""
import base64
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from types import SimpleNamespace
from threading import RLock

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import numpy as np
from PIL import Image
import pytest

from trackrag.dropoff import parking_evidence, place_evidence, place_grounding as place
from trackrag.dropoff.interaction import PassengerSession
from trackrag.dropoff.passenger import digest


def case(tmp_path, monkeypatch):
    path = tmp_path/"source.png"
    Image.new("RGB",(96,64),"navy").save(path)
    photo = path.read_bytes()
    frame = {"recording_id":"kartaview_42","frame_id":2,"timestamp_s":1.,"width":96,"height":64,"image_path":str(path)}
    sha = sha256(photo).hexdigest()
    scene = {"frame":frame,"source":"predicted","tracks":[],"provenance":{"image_sha256":sha}}
    request = {"request_id":"test1","scene":scene,"text":"Near the furniture store on the right.","dialogue":[],"input_mode":"typed"}
    classes = np.full((64,96),13,dtype=np.uint8)
    classes[40:60,65:91] = 10
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(parking_evidence,"PUBLIC_KEY",key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw).hex())
    return request,photo,classes,key


def packet(request,photo,classes,key,answers=("1","MATCH"),recipe=place.METHOD):
    _,image_sha,input_sha = parking_evidence.source_input(request, photo)
    ctx = place.current_context(request)
    encoded = BytesIO()
    Image.fromarray(classes).save(encoded,format="PNG")
    data=encoded.getvalue()
    responses=iter(answers)
    calls=[]
    def reply(kind,guide,pixels):
        answer=next(responses)
        calls.append({**place.call_identity(kind,guide,ctx["words"],pixels),"response":answer})
        return answer
    with Image.open(BytesIO(photo)) as im:
        place.apply_steps(ctx,classes,im,reply,recipe)
    body={"recipe":recipe,"models":place_evidence.MODELS,"context_sha256":digest(ctx),"image_sha256":image_sha,
          "input_sha256":input_sha,"calls":calls,"class_map":{"sha256":sha256(data).hexdigest(),"png":base64.b64encode(data).decode()}}
    return {"body":body,"signature":key.sign(parking_evidence.signed_bytes(body)).hex()}


@pytest.mark.parametrize(("words","history","active"),[
    ("Please stop at Al Wahda Mall.",[],True),
    ("Between the hotel and the pharmacy.",[],True),
    ("The one on the right.",["Drop me at the pharmacy."],True),
    ("Actually, behind the taxi.",["Drop me at the pharmacy."],False),
    ("By the car outside the shop.",[],False),
    ("In the bay between the taxi and the van.",[],False),
    ("There.",["At the pharmacy.","Actually behind that van."],False),
    ("Near the storefront.",["My other stop was at the office."],True),
    ("Can you drop me beside the public library?",[],True),
    ("Near the school bus.",[],False),
    ("Could you stop behind the vehicle by the museum?",[],False),
    ("Actually right here.",["At the library."],False),
    ("Drop me near the one on the left.",[],False),
    ("Right here.",[],False),
])
def test_visual_referent_routing(words,history,active):
    req={"text":words,"dialogue":[{"role":"user","text":s} for s in history]}
    result=place.place_reference(req)
    assert bool(result)==active
    if words=="Near the storefront.":
        assert result==words


def test_visible_parking_and_bumper_strip_are_required():
    classes=np.full((72,100),13,dtype=np.uint8)
    assert place.surface_options(classes,[100,72],[])==[]
    classes[40:50,43:56]=10
    assert place.surface_options(classes,[100,72],[])
    assert place.surface_options(classes,[100,72],[[38,15,60,40]])==[]


def test_empty_surface_is_a_segmentation_abstention(tmp_path,monkeypatch):
    req,photo,classes,key=case(tmp_path,monkeypatch)
    classes[:]=13
    signed=packet(req,photo,classes,key,())
    out=place_evidence.build_outcome(req,photo,signed)
    assert out["result"]["status"]=="needs_clarification"
    assert out["result"]["destination_point_xy"] is None
    audit=place_evidence.verify(req,out)
    assert audit["surface_model_calls"]==1 and audit["language_calls"]==0 and not audit["independent_model_proposal"]


@pytest.mark.parametrize("obstruction", ["ordinary road", "landscaped median", "traffic beyond curb", "parked vehicle"])
def test_curb_frontage_requires_a_visible_connection_to_the_landmark(tmp_path, monkeypatch, obstruction):
    from trackrag.dropoff import curb_place
    _,_,classes,_=case(tmp_path,monkeypatch)
    classes[:]=13
    if obstruction!="ordinary road":
        classes[:24]=17
        classes[24:29]=15
        classes[29:32]=2
        if obstruction=="landscaped median":
            classes[14:26]=30
        elif obstruction=="traffic beyond curb":
            classes[14:26]=13
    boxes=[[0,22,96,63]] if obstruction=="parked vehicle" else []
    assert curb_place.options(classes,[96,64],boxes)==[]


def test_verified_curb_pin_survives_confirmation_and_old_receipts_replay(tmp_path, monkeypatch):
    req,photo,classes,key=case(tmp_path,monkeypatch)
    classes[:]=13
    classes[:24]=17
    classes[24:29]=15
    classes[29:32]=2
    prior=packet(req,photo,classes,key,())
    assert place_evidence.build_outcome(req,photo,prior)["result"]["status"]=="needs_clarification"
    signed=packet(req,photo,classes,key,("BAY","1","APRON"),place.CURB_METHOD)
    outcome=place_evidence.build_outcome(req,photo,signed)
    assert [c["stage"] for c in signed["body"]["calls"]]==["curb_layout","curb_choose","curb_check"]
    x,y=outcome["result"]["destination_point_xy"]
    assert classes[int(y),int(x)]==13 and place_evidence.verify(req,outcome)["language_calls"]==3
    session=PassengerSession(req["scene"],object(),audit_dir=tmp_path/"logs")
    session._request=deepcopy(req)
    session._accept(outcome,"model")
    proposal=session.view()["proposal_id"]
    assert proposal and session.confirm(proposal)["state"]=="confirmed"
    switched=deepcopy(outcome)
    switched["provenance"]["method"]="different_model"
    with pytest.raises(ValueError,match="differs"):
        session._accept(switched,"model")


@pytest.mark.parametrize("replies,reason", [
    (("LANE",),"curb_layout_lane"),
    (("BAY","NONE"),"no_matching_curb_place"),
    (("APRON","1","DIVIDED"),"curb_point_divided"),
])
def test_rejected_curb_evidence_has_no_point(tmp_path,monkeypatch,replies,reason):
    req,photo,classes,key=case(tmp_path,monkeypatch)
    classes[:]=13
    classes[:24]=17
    classes[24:29]=15
    classes[29:32]=2
    signed=packet(req,photo,classes,key,replies,place.CURB_METHOD)
    outcome=place_evidence.build_outcome(req,photo,signed)
    assert outcome["result"]["status"]=="needs_clarification"
    assert outcome["result"]["destination_point_xy"] is None and outcome["provenance"]["reason"]==reason


@pytest.mark.parametrize("verdict",["MATCH","WRONG","BLOCKED","UNCLEAR"])
def test_signed_grounding_rejects_other_requests_and_edits(tmp_path,monkeypatch,verdict):
    req,photo,classes,key=case(tmp_path,monkeypatch)
    signed=packet(req,photo,classes,key,("1",verdict))
    out=place_evidence.build_outcome(req,photo,signed)
    assert place_evidence.verify(req,out)["language_calls"]==2
    assert (out["result"]["status"]=="proposed")== (verdict=="MATCH")
    if verdict=="MATCH":
        x,y=out["result"]["destination_point_xy"]
        assert classes[int(y),int(x)]==10
    stale=deepcopy(req); stale["request_id"]="another-invocation"
    with pytest.raises(ValueError,match="does not match"):
        place_evidence.build_outcome(stale,photo,signed)
    changed=deepcopy(signed);changed["body"]["calls"][0]["response"]="2"
    with pytest.raises(ValueError,match="signature"):
        place_evidence.build_outcome(req,photo,changed)
    altered=deepcopy(out);altered["result"]["status"]="confirmed"
    with pytest.raises(ValueError):
        place_evidence.verify(req,altered)


def test_only_listed_originals_can_use_the_remote_backend(tmp_path,monkeypatch):
    req,photo,_,_=case(tmp_path,monkeypatch)
    manifest=tmp_path/"photos.json"
    manifest.write_text(json.dumps({"photos":[{"photo_url":"https://kartaview.org/details/42/2/","sha256":sha256(photo).hexdigest()}]}))
    backend=place_evidence.PublicPlaceBackend([manifest])
    assert backend.eligible(req)
    other=deepcopy(req);other["scene"]["frame"]["recording_id"]="EMT_private"
    assert not backend.eligible(other)
    with pytest.raises(ValueError,match="selected public"):
        backend.interpret(other)
    candidate=place.current_context(req)
    assert set(candidate)=={"words","size","image_sha256","current_boxes"} and str(tmp_path) not in str(candidate)
    base=object()
    dialogue=SimpleNamespace(model=base,passenger_area="existing",interpret=lambda _: "local")
    wrapper=place_evidence.PlaceDialogueGrounder(dialogue,backend)
    assert wrapper.model is base and wrapper.passenger_area=="existing" and wrapper.interpret(other)=="local"


def test_confirmation_is_separate_and_invalid_evidence_withdraws_the_pin(tmp_path,monkeypatch):
    req,photo,classes,key=case(tmp_path,monkeypatch)
    signed=packet(req,photo,classes,key)
    good=place_evidence.build_outcome(req,photo,signed)
    session=PassengerSession(req["scene"],object(),audit_dir=tmp_path/"logs")
    session._request=deepcopy(req)
    with pytest.raises(ValueError,match="automatic"):
        session._accept(good,"passenger")
    fabricated=deepcopy(good);del fabricated["provenance"]["place_evidence"]
    with pytest.raises((ValueError,KeyError)):
        session._accept(fabricated,"model")
    session._accept(good,"model")
    pending=session.view()
    assert pending["state"]=="proposed" and pending["proposal_id"]
    session._place_offer["result"]["destination_point_xy"][0]+=1
    with pytest.raises(ValueError):
        session.confirm(pending["proposal_id"])
    assert session.view()["state"]=="error" and not session.view()["proposal_id"]


@pytest.mark.parametrize("error",[RuntimeError("CUDA out of memory"),MemoryError()])
def test_resource_failure_offers_manual_recovery_only_for_the_bound_original(tmp_path,monkeypatch,error):
    from trackrag.dropoff import passenger
    from trackrag.dropoff.passenger_pair_choice import landmark_error_guidance
    req,photo,_,_=case(tmp_path,monkeypatch)
    original=req["scene"]["frame"]["image_path"]
    def exhaust(*_):
        raise error
    grounder=passenger.PassengerGrounder.__new__(passenger.PassengerGrounder)
    grounder.config=None
    grounder.lock=RLock()
    grounder.resolver=SimpleNamespace(resolve=exhaust)
    monkeypatch.setattr(passenger,"resolve_image",lambda *_:(Image.open(original),{"sha256":sha256(photo).hexdigest()}))
    failure=grounder.interpret(req)
    assert failure["ok"] is False and failure["error"]["code"]=="resource_unavailable"
    assert "mark the spot" in landmark_error_guidance(req,failure)
    stale=deepcopy(req);stale["text"]="A different place"
    assert landmark_error_guidance(stale,failure) is None
    unbound=deepcopy(failure);unbound["provenance"].pop("image")
    assert landmark_error_guidance(req,unbound) is None
    Image.new("RGB",(96,64),"red").save(original)
    assert landmark_error_guidance(req,failure) is None
