"""Offline software checks of the real GPU protocol. Synthetic scenes carry no accuracy claim."""
import base64
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import numpy as np
from PIL import Image
import pytest

from tools.passenger_parking_study import QWEN_REVISION, SEG_REVISION
from trackrag.dropoff import parking_evidence as evidence
from trackrag.dropoff.parking_steps import steps
from trackrag.dropoff.parking_views import render
from trackrag.dropoff.passenger_parking import parking_instruction


def setup_case(folder, monkeypatch):
    photo=folder/"street.png"
    Image.new("RGB",(256,144),"gray").save(photo)
    image_bytes=photo.read_bytes()
    frame={"recording_id":"one","frame_id":12,"timestamp_s":1.2,"width":256,"height":144,"image_path":str(photo)}
    tracks=[{"recording_id":"one","frame_id":12,"source":"predicted","class_name":"Car","track_id":i,"bbox_xyxy":b}
            for i,b in [(2,[44,64,90,110]),(4,[134,65,180,110])]]
    request={"request_id":"synthetic","text":"Drop me off between the two cars on the right.","input_mode":"typed","dialogue":[],
             "scene":{"source":"predicted","frame":frame,"tracks":tracks,"provenance":{"image_sha256":sha256(image_bytes).hexdigest()}}}
    classes=np.full((144,256),13,dtype="uint8")
    classes[90:115,91:134]=10
    key=Ed25519PrivateKey.generate()
    monkeypatch.setattr(evidence,"PUBLIC_KEY",key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw).hex())
    return request,image_bytes,classes,key


ANCHORS='{"vehicles":[[172,444,352,764],[523,451,703,764]]}'


def make_packet(request, photo, classes, key, answers, recipe=evidence.RECIPE):
    _,image_hash,input_hash=evidence.source_input(request,photo)
    png=BytesIO()
    Image.fromarray(classes).save(png,format="PNG")
    data=png.getvalue()
    from trackrag.dropoff.parking_reference_board import steps as reviewed_steps
    algorithm = reviewed_steps if recipe == evidence.REVIEWED_RECIPE else steps
    machine=algorithm(evidence.current_only(evidence.portable_request(request)),classes)
    calls=[]
    try:
        stage=next(machine)
        for answer in answers:
            path=Path(request["scene"]["frame"]["image_path"])
            seen=render(path,request["scene"]["frame"],stage["view"],path.with_suffix(".jpg"))
            calls.append({"kind":stage["kind"],"system_sha256":sha256(stage["system"].encode()).hexdigest(),
                          "view_sha256":evidence.view_digest(stage["view"]),"rendered_sha256":sha256(seen.read_bytes()).hexdigest(),"response":answer})
            stage=machine.send(answer)
    except StopIteration:
        pass
    body={"recipe":recipe,"models":[QWEN_REVISION,SEG_REVISION],"input_sha256":input_hash,
          "image_sha256":image_hash,"words":parking_instruction(request),"calls":calls,
          "class_map":{"sha256":sha256(data).hexdigest(),"png":base64.b64encode(data).decode("ascii")}}
    return {"body":body,"signature":key.sign(evidence.signed_bytes(body)).hex()}


def test_authenticated_proposal_replays_original_parking_cells_and_passenger_words(tmp_path,monkeypatch):
    req,photo,classes,key=setup_case(tmp_path,monkeypatch)
    packet=make_packet(req,photo,classes,key,[ANCHORS,'{"choice":"A"}'])
    outcome=evidence.build_outcome(req,photo,packet)
    assert outcome["result"]["status"]=="proposed"
    x,y=outcome["result"]["destination_point_xy"]
    assert classes[int(y),int(x)]==10 and evidence.verify(req,outcome)["independent_model_proposal"]
    for field in ("destination_point_xy","destination_box_xyxy"):
        changed=deepcopy(outcome);changed["result"][field][0]+=0.5
        with pytest.raises(ValueError,match="differs"):
            evidence.verify(req,changed)
    for field in ("calls","recipe","class_map"):
        changed=deepcopy(packet);changed["body"][field]=None
        with pytest.raises(ValueError,match="signature"):
            evidence.build_outcome(req,photo,changed)
    other=deepcopy(req);other["text"]=req["text"]+" Please."
    with pytest.raises(ValueError,match="different"):
        evidence.build_outcome(other,photo,packet)
    with pytest.raises(ValueError,match="photograph"):
        evidence.build_outcome(req,photo+b"changed",packet)


def test_model_ask_and_invalid_reply_remain_distinct(tmp_path,monkeypatch):
    req,photo,classes,key=setup_case(tmp_path,monkeypatch)
    for raw,expected in [('{"choice":null}',True),('{"choice":"Z"}',False)]:
        packet=make_packet(req,photo,classes,key,[ANCHORS,raw])
        out=evidence.build_outcome(req,photo,packet)
        assert out["ok"] is expected
        assert evidence.verify(req,out)["independent_model_proposal"] is False
        if expected:
            assert out["result"]["status"]=="needs_clarification"
        else:
            assert out["error"]["code"]=="invalid_model_output"


def test_stale_observations_never_become_vehicle_choices(tmp_path,monkeypatch):
    req,photo,classes,key=setup_case(tmp_path,monkeypatch)
    req["scene"]["tracks"][0]["frame_id"]=11
    packet=make_packet(req,photo,classes,key,[])
    served=evidence.build_outcome(req,photo,packet)
    assert served["provenance"]["reason"]=="fewer_than_two_observed_vehicles"
    assert packet["body"]["calls"]==[]
    other=deepcopy(req);other["scene"]["tracks"][0]["frame_id"]=13
    with pytest.raises(ValueError,match="future"):
        evidence.build_outcome(other,photo,packet)


def test_signed_incomplete_or_extra_stage_rejected(tmp_path,monkeypatch):
    req,photo,classes,key=setup_case(tmp_path,monkeypatch)
    packet=make_packet(req,photo,classes,key,[ANCHORS,'{"choice":"A"}'])
    for calls in [packet["body"]["calls"][:1],packet["body"]["calls"]*2]:
        new=deepcopy(packet)
        new["body"]["calls"]=calls
        new["signature"]=key.sign(evidence.signed_bytes(new["body"])).hex()
        with pytest.raises(ValueError,match="Incomplete|Extra"):
            evidence.build_outcome(req,photo,new)
    changed=deepcopy(packet)
    changed["body"]["calls"][1]["rendered_sha256"]="0"*64
    changed["signature"]=key.sign(evidence.signed_bytes(changed["body"])).hex()
    with pytest.raises(ValueError,match="Model-visible"):
        evidence.build_outcome(req,photo,changed)


@pytest.mark.parametrize("agreed,visible,expected_calls,expected_state", [
    (False, True, ["anchors", "tracks"], "needs_clarification"),
    (True, False, ["anchors", "tracks", "choice", "check"], "needs_clarification"),
    (True, True, ["anchors", "tracks", "choice", "check"], "proposed"),
])
def test_road_labelled_gap_needs_matching_references_and_visible_opening(
        tmp_path, monkeypatch, agreed, visible, expected_calls, expected_state):
    from trackrag.dropoff import parking_surface

    req, photo, classes, key = setup_case(tmp_path, monkeypatch)
    tracks = req["scene"]["tracks"]
    candidate = {
        "choice": "A", "point": [110, 102], "region_box_xyxy": [103, 96, 118, 109],
        "tracks": [2, 4], "anchor_boxes": [t["bbox_xyxy"] for t in tracks],
        "requires_visual_review": True, "surface": "roadside_road_visual_review",
    }

    def only_fallback(_request, _classes, _pair, *, allow_road_extension=False, **kwargs):
        return [candidate] if allow_road_extension else []

    monkeypatch.setattr(parking_surface, "options_for", only_fallback)
    model_replies = [
        ANCHORS if agreed else '{"vehicles":[]}', '{"ids":[2,4]}', '{"choice":"A"}',
        json.dumps({"references_match": True, "visible_opening": visible}),
    ]
    packet = make_packet(req, photo, classes, key, model_replies)
    assert [c["kind"] for c in packet["body"]["calls"]] == expected_calls
    outcome = evidence.build_outcome(req, photo, packet)
    assert outcome["result"]["status"] == expected_state
    if expected_state == "proposed":
        assert outcome["result"]["destination_point_xy"] == candidate["point"]
    else:
        assert outcome["result"]["destination_point_xy"] is None


def test_current_and_earlier_genuine_protocols_replay_independently(tmp_path, monkeypatch):
    req, photo, classes, key = setup_case(tmp_path, monkeypatch)
    for recipe, selector in [(evidence.RECIPE, "choice"), (evidence.REVIEWED_RECIPE, "parking_components")]:
        packet = make_packet(req, photo, classes, key, [ANCHORS, '{"choice":"A"}'], recipe)
        assert [c["kind"] for c in packet["body"]["calls"]] == ["anchors", selector]
        outcome = evidence.build_outcome(req, photo, packet)
        assert outcome["result"]["status"] == "proposed"
        assert evidence.verify(req, outcome)["model_images_reproduced"]
    declined = make_packet(req, photo, classes, key, [ANCHORS, '{"choice":null}'], evidence.REVIEWED_RECIPE)
    assert evidence.build_outcome(req, photo, declined)["provenance"]["reason"] == "model_declined_parking_components"


def test_structure_at_vehicle_depth_asks_and_a_distant_facade_does_not(tmp_path, monkeypatch):
    req, photo, classes, key = setup_case(tmp_path, monkeypatch)
    classes[55:86, 94:132] = 17
    ok = make_packet(req, photo, classes, key, [ANCHORS, '{"choice":"A"}'], evidence.REVIEWED_RECIPE)
    assert evidence.build_outcome(req, photo, ok)["result"]["status"] == "proposed"
    classes[70:109, 101:125] = 17
    blocked = make_packet(req, photo, classes, key, [ANCHORS], evidence.REVIEWED_RECIPE)
    outcome = evidence.build_outcome(req, photo, blocked)
    assert outcome["provenance"]["reason"] == "structural_obstruction_in_requested_row"
    assert outcome["result"]["status"] == "needs_clarification"


def test_output_track_order_matches_original_boxes_after_ranked_board_reorders_pair(tmp_path, monkeypatch):
    req, photo, classes, key = setup_case(tmp_path, monkeypatch)
    reverse = json.dumps({"vehicles": list(reversed(json.loads(ANCHORS)["vehicles"]))})
    pkt = make_packet(req, photo, classes, key, [reverse, '{"choice":"A"}'], evidence.REVIEWED_RECIPE)
    outcome = evidence.build_outcome(req, photo, pkt)
    ids = outcome["provenance"]["track_ids"]
    by_id = {t["track_id"]:t["bbox_xyxy"] for t in req["scene"]["tracks"]}
    assert outcome["result"]["anchor_boxes_xyxy"] == [by_id[i] for i in ids]
