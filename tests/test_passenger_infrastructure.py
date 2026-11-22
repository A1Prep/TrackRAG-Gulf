"""Software fixtures for the passenger clarification policy; no human or model accuracy claims."""
from copy import deepcopy

import pytest

from trackrag.dropoff.contracts import validate_target
from trackrag.dropoff.infrastructure import PassengerDialogueGrounder, review
from trackrag.dropoff.landmarks import landmark_messages, parse_landmark
from trackrag.dropoff.passenger import digest, proposed, unresolved
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
