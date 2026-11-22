"""Software fixtures for the passenger clarification policy; no human or model accuracy claims."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from trackrag.dropoff.contracts import validate_target
from trackrag.dropoff.data import inference_input_sha256
from trackrag.dropoff.infrastructure import PassengerDialogueGrounder, review
from trackrag.dropoff.landmarks import landmark_messages, parse_landmark
from trackrag.dropoff.passenger import digest, proposed, unresolved
from trackrag.dropoff.referent import ALIGN_NAMES, KIND, REGION_NAMES, ReferentRanker
from trackrag.vision.classes import EMT_CLASSES


ASK = [
    "Stop by the guardrail on the left.", "I want the marked pedestrian crossing.",
    "Opposite that bus stop, please.", "Just before the traffic lights.",
    "Below the street lamp near that gate.", "At the corner by the car park entrance.",
    "Under the railway overpass.", "Next to the low wall and the bollard.",
    "Beside the kerb cut.", "By the red stop sign.",
]
KEEP = [
    "Just behind the coach under the flyover.", "Between the van and the fence.",
    "Beside the cyclist waiting at the crosswalk.", "By the bus at the bus stop.",
    "Opposite the entrance, beside the red taxi.", "Next to the Toyota by the guardrail.",
    "Behind the pickup with the roof rack.", "Please stop on the road ahead.",
    "There, on the left.", "By the light blue sedan.",
    "Drop me off behind the car with the sign on its roof.",
    "Beside the vehicle with the park insignia.", "Stop next to the woman by the lamp post.",
    "By the gazebo and the silvery Edsel.", "The red one beyond the overpass.",
    "Below the tree, not the lamp.", "Between the wall and that moving thing.",
    "Stop by the Volvo, I mean under the bridge.", "At that one, I mean under the overpass.",
]


def fixture(text="At the gate.", history=(), *, class_name="Car"):
    image_hash = "a" * 64
    frame = {"recording_id":"fixture", "frame_id":5, "timestamp_s":.5,
             "width":100, "height":80, "image_path":"unused-fixture.jpg"}
    box = [10,20,24,40]
    track = {"recording_id":"fixture","frame_id":5,"track_id":7,"class_name":class_name,
             "bbox_xyxy":box,"source":"predicted"}
    scene = {"frame":frame,"source":"predicted","tracks":[track],"provenance":{"image_sha256":image_hash}}
    req = {"request_id":"fixture-007","text":text,"scene":scene,"input_mode":"typed",
           "dialogue":[{"role":role,"text":turn,"recording_id":"fixture","frame_id":5} for role,turn in history]}
    _, index = landmark_messages(req)
    decision = parse_landmark("LANDMARK L1", index, frame)
    info = {"method":"qwen_landmark_region","request_sha256":digest(req),"image":{"sha256":image_hash},
            "selected_track_ids":[7],"resolver":{"image_sha256":image_hash,"request_sha256":digest(req),
            "decision":decision,"raw_text":"LANDMARK L1"}}
    return req, {"ok":True,"request_id":"fixture-007","result":proposed([25,57],[15,45,33,66],[box.copy()]),
                 "provenance":info}


@pytest.mark.parametrize("text", ASK)
def test_withdraws_contradiction_and_keeps_model_observation(text):
    req, outcome = fixture(text)
    initial = deepcopy(outcome)
    answer = review(req, outcome)
    assert outcome == initial
    target = answer["result"]
    validate_target(target, list(target), req["scene"]["frame"])
    assert target["status"] == "needs_clarification" and target["clarification_question"].endswith("?")
    assert target["destination_point_xy"] is target["destination_box_xyxy"] is None
    assert "anchor_boxes_xyxy" not in target
    audit = answer["provenance"]["infrastructure_review"]
    assert audit["original_result"] == initial["result"]
    assert audit["selected_observation"] == req["scene"]["tracks"][0]
    assert audit["image_sha256"] == initial["provenance"]["image"]["sha256"]
    assert audit["request_sha256"] == digest(req)
    assert all(text[q["start"]:q["end"]]==q["text"] for q in audit["passenger_reference_spans"])
    assert review(req, answer) is answer


@pytest.mark.parametrize("name", EMT_CLASSES)
def test_detector_taxonomy_is_road_users(name):
    req, out = fixture(class_name=name)
    assert review(req, out)["result"]["status"] == "needs_clarification"


@pytest.mark.parametrize("text", KEEP)
def test_mixed_unmodelled_or_indeterminate_reference_preserved(text):
    req, out = fixture(text)
    assert review(req, out) is out


@pytest.mark.parametrize(("earlier","latest","expected"), [
    ("The white car.", "The one under the bridge.", "proposed"),
    ("The white car.", "Actually, beside the bus shelter instead.", "needs_clarification"),
    ("The bus shelter.", "Actually, the white minivan.", "proposed"),
    ("Over there, please.", "At the bollard beside that entrance.", "needs_clarification"),
    ("The taxi.", "Beside the bridge.", "proposed"),
    ("The Volvo.", "Beside the bridge.", "proposed"),
    ("The dark one.", "By the traffic light.", "proposed"),
    ("There.", "The road on our side of the concrete divider, below the big blue exit sign.", "needs_clarification"),
])
def test_current_passenger_scope_and_clarification(earlier,latest,expected):
    req, out = fixture(latest,[("user",earlier),("assistant","Is that the bus ahead?")])
    assert review(req,out)["result"]["status"] == expected


@pytest.mark.parametrize("change", [
    "object", "asked", "missing", "operational", "multiple", "manual", "duplicate",
    "foreign_recording","past","future","wrong_request","wrong_image","wrong_resolver_image",
    "foreign_source","annotated","unknown_class","wrong_selection","wrong_resolver_decision",
    "wrong_raw_landmark","wrong_anchors","different_verification","pair","unrelated_output",
])
def test_requires_effective_single_bound_prediction(change):
    req, out = fixture()
    p = out["provenance"]
    d = p["resolver"]
    t = req["scene"]["tracks"][0]
    if change=="object": d["raw_text"]="OBJECT 10 20 30 40"; p["selected_track_ids"]=[]
    elif change=="asked": out["result"]=unresolved("Which one?")
    elif change=="missing": out["result"]=unresolved()
    elif change=="operational": out={"ok":False,"error":{"code":"backend_error"},"request_id":req["request_id"]}
    elif change=="multiple": p["selected_track_ids"]=[7,8]
    elif change=="manual": p["method"]="explicit_passenger_region"
    elif change=="duplicate": req["scene"]["tracks"].append(deepcopy(t))
    elif change=="foreign_recording": t["recording_id"]="other"
    elif change=="past": t["frame_id"]-=1
    elif change=="future": t["frame_id"]+=1
    elif change=="wrong_request": p["request_sha256"]="1"*64
    elif change=="wrong_image": p["image"]["sha256"]="1"*64
    elif change=="wrong_resolver_image": d["image_sha256"]="1"*64
    elif change=="foreign_source": t["source"]="annotated"
    elif change=="annotated": req["scene"]["source"]=t["source"]="annotated"
    elif change=="unknown_class": t["class_name"]="Traffic light"
    elif change=="wrong_selection": p["selected_track_ids"]=[19]
    elif change=="wrong_resolver_decision": d["decision"]["boxes"]=[[2,3,15,19]]
    elif change=="wrong_raw_landmark": d["raw_text"]="LANDMARK L4"
    elif change=="wrong_anchors": out["result"]["anchor_boxes_xyxy"]=[[2,3,15,19]]
    elif change=="different_verification": p["visual_verification"]={"raw_text":"LANDMARK L2"}
    elif change=="pair": p["pair_decision"]=deepcopy(d["decision"])
    elif change=="unrelated_output": out["request_id"]="different"
    if change in {"unknown_class","past","foreign_source","foreign_recording","future","duplicate","annotated"}:
        p["request_sha256"]=d["request_sha256"]=digest(req)
    assert review(req,out) is out


def test_inherited_viewpoint_and_passenger_manual_forwarding():
    req, out = fixture()
    class Model:
        def interpret(self, request):
            assert request is req
            return out
        def passenger_area(self, request, box):
            return (request, box)
    app = PassengerDialogueGrounder(Model())
    assert app.interpret(req)["result"]["status"]=="needs_clarification"
    assert app.passenger_area(req,[2,3,8,9])==(req,[2,3,8,9])


@pytest.mark.parametrize("fault", [
    None, "stale_box", "wrong_class", "wrong_observation", "fabricated_score", "wrong_checkpoint",
    "image_missing", "image_replaced", "foreign_input", "foreign_request", "wrong_case", "stale_history",
    "wrong_original_anchor", "wrong_original_identity", "genuine_object",
])
def test_viewpoint_requires_recomputed_ranking_from_bound_original(tmp_path, fault):
    # Real rank() with explicitly software-only coefficients, geometry and photo.
    req,out=fixture("Could you let me out alongside the vehicle straight ahead?")
    pixels=tmp_path/"controlled-frame.png"
    Image.new("RGB",(100,80),"gray").save(pixels)
    img_hash=sha256(pixels.read_bytes()).hexdigest();scene=req["scene"]
    scene["frame"]["image_path"]=str(pixels);scene["provenance"]["image_sha256"]=img_hash
    scene["tracks"].append({**scene["tracks"][0],"bbox_xyxy":[67,22,81,41],"track_id":19})
    ckpt=tmp_path/"software-fixture.json"
    ckpt.write_text(json.dumps({"kind":KIND,"region_names":list(REGION_NAMES),"align_names":list(ALIGN_NAMES),
        "vocabulary":[],"base":[0.0]*len(REGION_NAMES),"aligned":[0.0]*len(ALIGN_NAMES),"interactions":[]}))
    ranker=ReferentRanker.load(ckpt,expected_sha256=sha256(ckpt.read_bytes()).hexdigest())
    with Image.open(pixels) as image:ranking=ranker.rank(image,req["text"],scene["tracks"],frame=scene["frame"])
    pr=out["provenance"];pr["referent_ranking"]=ranking
    pr["request_sha256"]=pr["resolver"]["request_sha256"]=digest(req)
    pr["image"]["sha256"]=pr["resolver"]["image_sha256"]=img_hash
    pr["inference_input_sha256"]=inference_input_sha256(req,img_hash)
    if fault=="stale_box":ranking["alternatives"][0]["bbox_xyxy"]=[11,20,25,40]
    if fault=="wrong_class":ranking["alternatives"][0]["class_name"]="Bus"
    if fault=="wrong_observation":ranking["alternatives"][0]["track_id"]=132
    if fault=="fabricated_score":ranking["null_score"]=.9
    if fault=="wrong_checkpoint":ranking["checkpoint_sha256"]="1"*64
    if fault=="image_missing":pr["resolver"].pop("image_sha256")
    if fault=="image_replaced":Image.new("RGB",(100,80),"blue").save(pixels)
    if fault=="foreign_input":pr["inference_input_sha256"]="2"*64
    if fault=="foreign_request":pr["request_sha256"]="3"*64
    if fault=="wrong_case":out["request_id"]="elsewhere"
    if fault=="stale_history":req["dialogue"]=[{"role":"user","text":"The taxi.","recording_id":"different","frame_id":4}]
    if fault=="wrong_original_anchor":out["result"]["anchor_boxes_xyxy"]=[[21,14,36,53]]
    if fault=="wrong_original_identity":pr["selected_track_ids"]=[19]
    if fault=="genuine_object":
        pr["resolver"]["raw_text"]="OBJECT 320 420 610 750"
        _,candidates=landmark_messages(req)
        first=parse_landmark(pr["resolver"]["raw_text"],candidates,scene["frame"])
        pr["resolver"]["decision"]=first;pr["selected_track_ids"]=[]
        out["result"]["anchor_boxes_xyxy"]=deepcopy(first["boxes"])
    config=SimpleNamespace(max_image_bytes=1_000_000,max_original_pixels=100_000)
    wrapper=PassengerDialogueGrounder(SimpleNamespace(interpret=lambda request:out,referent_ranker=ranker,config=config))
    response=wrapper.interpret(req)
    if fault and fault!="genuine_object":
        assert response is out
    else:
        assert response["result"]["status"]=="needs_clarification"
        why=response["provenance"]["viewpoint_clarification"]
        assert why["original_proposal"]==out["result"]
        assert {r["track_id"] for r in why["source_binding"]["current_observations"]}=={7,19}
        assert why["source_binding"]["image_sha256"]==img_hash
        assert why["source_binding"]["checkpoint_sha256"]==ranker.sha256
        assert "viewpoint_clarification" not in out["provenance"]
