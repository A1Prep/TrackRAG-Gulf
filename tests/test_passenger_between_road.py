"""Test the production two-vehicle flow on generated pixels with no real passenger labels."""
from copy import deepcopy
from hashlib import sha256
import json
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from trackrag.dropoff import between_policy, between_road, pair_geometry, passenger_pair_choice, projected_pair
from trackrag.dropoff.anchor_geometry import AnchorGeometry
from trackrag.dropoff.infrastructure import PassengerDialogueGrounder
from trackrag.dropoff.landmarks import landmark_messages, parse_landmark
from trackrag.dropoff.passenger import PassengerGrounder, digest
from trackrag.dropoff.road_cache import mask_shape
from trackrag.dropoff.structured import LANDMARK_CONTRACT
from tests.test_passenger_projected_pair import obs


SIZE = (320, 180)
WORDS = "Please drop me between the two dark cars."


@pytest.mark.parametrize("text,proper",[
    ("There's a blue pickup and a cream van ahead. Could you let me out between those two?",True),
    ("I see a taxi and a minivan. The patch between them, please.",True),
    ("Between the nearer SUV, the one partially cut off by the photo, and the pale van on the corner.",True),
    ("Let me out between the compact car, which is nearest the camera, and the other sedan farther back.",True),
    ("Between the door, the one beside the silver car, and the blue delivery van.",False),
    ("I see a gate beside the car and a taxi. I need the area between those two.",False),
    ("The stop is between the cafe and the bus station.",False),
    ("I can see a van and a cab, but please do not stop between those two.",False),
    ("The road just behind the gap between the car and a van.",False),
    ("Between the car and the van and the other car.",False),
])
def test_prefatory_vehicles_and_appositives_cannot_bypass_the_road_check(text,proper):
    assert bool(between_road.requested({"text":text,"dialogue":[]})) is proper


@pytest.fixture
def roadside(tmp_path, monkeypatch):
    picture = tmp_path/"illustrative-road.png"
    tracks = [obs(21, [18,24,48,66]), obs(46, [64,82,98,130])]
    with Image.new("RGB", SIZE, "white") as im:
        paint = ImageDraw.Draw(im)
        for car in tracks:
            paint.rectangle(car["bbox_xyxy"], fill=(37,48,64))
        im.save(picture)
    past = obs(90, [40,70,63,115])
    past["frame_id"] = 7
    scene = {"frame":{"recording_id":"fixture-clip","frame_id":8,"timestamp_s":0.7,
                      "width":320,"height":180,"image_path":str(picture)},
             "source":"predicted","tracks":[*tracks,past],
             "provenance":{"image_sha256":sha256(picture.read_bytes()).hexdigest()}}
    req = {"request_id":"between-road-test","scene":scene,"text":WORDS,"input_mode":"typed","dialogue":[]}
    # Fit an actual tiny regression on generated shapes, keeping software provenance explicit.
    examples = [{"anchors":[[18+i%3,24,48+i%3,66],[64+i%5,82,98+i%5,130]],"size":SIZE,
                 "point":[55+i%3,95], "box":[42+i%3,78,66+i%3,111]} for i in range(35)]
    head = pair_geometry.fit(examples)
    model = object.__new__(PassengerGrounder)
    model.config = SimpleNamespace(model_id="test-weights", revision="test", adapter_dir=None,
                                   max_image_bytes=100000, max_original_pixels=SIZE[0]*SIZE[1])
    model.lock = threading.RLock()
    model.verify_landmarks = False
    model.referent_ranker = None
    model.geometry = SimpleNamespace(sha256=digest({"software":head}), pair_regression=head)
    monkeypatch.setattr(passenger_pair_choice,"GEOMETRY",model.geometry.sha256)
    shape = mask_shape(SIZE)
    mask = np.zeros(shape, dtype=bool)
    x = (np.arange(shape[1])+0.5)*SIZE[0]/shape[1]
    mask[:,(x >= 34) & (x < 303)] = True
    model.road = SimpleNamespace(mask=lambda im, fingerprint: mask)
    raw = ["LANDMARK L1 L2"]

    def verify_test_only(record, config):
        if record.get("software_fixture") is not True or config is not model.config:
            raise ValueError("Wrong software-only test evidence")
    monkeypatch.setattr(projected_pair,"native_record",verify_test_only)

    def resolve(request, image):
        messages, index = landmark_messages(request)
        return {"raw_text": raw[0], "decision":parse_landmark(raw[0],index,request["scene"]["frame"]),
                "messages":messages,"messages_sha256":sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest(),
                "image_sha256":scene["provenance"]["image_sha256"], "request_sha256":digest(request),
                "model_id":model.config.model_id, "revision":model.config.revision, "adapter":None,
                "contract":LANDMARK_CONTRACT, "software_fixture":True}

    model.resolver = SimpleNamespace(resolve=resolve, config=model.config)
    return req, model, raw, mask


def test_native_and_passenger_choices_use_same_mask_and_learned_ground_contacts(roadside,tmp_path):
    req, model, raw, mask = roadside
    session = passenger_pair_choice.PassengerSession(req["scene"],PassengerDialogueGrounder(model),audit_dir=tmp_path/"audit")
    native = session.submit(WORDS)
    assert native["state"]=="proposed" and native["selected_by"]=="model"
    manual = session.select_pair([21,46],"between")
    assert manual["state"]=="proposed" and manual["selected_by"]=="passenger_landmarks"
    assert native["result"]==manual["result"]
    with pytest.raises(ValueError):
        session.confirm(native["proposal_id"])
    assert session.confirm(manual["proposal_id"])["state"]=="confirmed"
    rows = [json.loads(line) for line in next((tmp_path/"audit").glob("*.jsonl")).read_bytes().splitlines()]
    auto, explicit = (r["detail"]["outcome"]["provenance"] for r in rows if r["action"] in {"submitted","passenger_selected_pair"})
    evidence = auto["between_road"]["road_geometry"]
    assert evidence==explicit["road_geometry"]
    assert evidence["side_of_visible_road"]=="left" and evidence["learned_estimator"]=="learned_horizontal_pair_regression"
    assert between_road.inside_contact_polygon(evidence["contact_polygon_xy"],evidence["box"])
    assert set(auto["between_road"]["resolved_track_ids"])=={21,46}


@pytest.mark.parametrize("words",[
    "There is a grey car and a pale van further along. Let me out between those two.",
    "Stop between the nearest SUV, the one partially outside the frame, and the other car further ahead.",
])
def test_active_session_checks_ground_for_prefaces_and_appositives(roadside,words,tmp_path):
    req,model,raw,mask=roadside
    session=passenger_pair_choice.PassengerSession(req["scene"],PassengerDialogueGrounder(model),audit_dir=tmp_path/"audit")
    assert session.submit(words)["state"]=="proposed"
    mask[:]=False
    second=passenger_pair_choice.PassengerSession(req["scene"],PassengerDialogueGrounder(model),audit_dir=tmp_path/"repeat").submit(words)
    assert second["state"]=="needs_clarification" and second["proposal_id"] is None


def test_unknown_two_vehicle_reading_requires_clarification_and_cannot_be_confirmed(roadside):
    req,model,raw,mask=roadside
    req["text"]="Somewhere between those two, please."
    out=PassengerDialogueGrounder(model).interpret(req)
    assert out["result"]["status"]=="needs_clarification"
    assert out["provenance"]["constraint"]=="unbound_vehicle_between_reference"
    fake={**out,"result":{**out["result"],"status":"proposed"}}
    with pytest.raises(ValueError,match="must be clarified"):
        between_road.validate_offer(model,req,fake,"model")


@pytest.mark.parametrize("words",["Between the gate and the lamp.","Between the car and the bus stop.",
                                  "Do not stop between those two."])
def test_unbound_vehicle_backstop_preserves_other_routes(roadside,words):
    req,model,raw,mask=roadside
    req["text"]=words
    assert not between_road.unbound_vehicle_selection(req,{"kind":"landmark","track_ids":[21,46]})


@pytest.mark.parametrize("change",["no-road","all-road","opposite-sides","mid-lane","side-by-side","blocked",
                                   "edge-occluded","only-one-vehicle-beside-edge"])
def test_absent_support_and_through_lane_hypotheses_do_not_receive_a_point(roadside,change):
    req, model, raw, mask = roadside
    tracks = req["scene"]["tracks"]
    if change=="no-road": mask[:] = False
    if change=="all-road": mask[:] = True
    if change=="opposite-sides": tracks[1]["bbox_xyxy"]=[205,80,245,132]
    if change=="mid-lane":
        for t in tracks[:2]:
            t["bbox_xyxy"][0]+=75
            t["bbox_xyxy"][2]+=75
    if change=="side-by-side": tracks[1]["bbox_xyxy"]=[64,25,98,66]
    if change=="blocked": tracks.append(obs(89,[28,65,98,131],"Pedestrian"))
    if change=="edge-occluded": tracks.append(obs(89,[12,65,34,139],"Pedestrian"))
    if change=="only-one-vehicle-beside-edge": tracks[1]["bbox_xyxy"]=[124,82,158,130]
    result=passenger_pair_choice.choose(model,req,[21,46],"between")
    assert result["ok"] and result["result"]["status"]=="needs_clarification"
    assert result["result"]["destination_point_xy"] is None


def test_separated_depths_with_overlapping_horizontal_ranges_use_released_companion_head(roadside):
    req,model,raw,mask=roadside
    a,b=req["scene"]["tracks"][:2]
    a["bbox_xyxy"]=[18,24,60,66]
    b["bbox_xyxy"]=[45,84,78,129]
    examples=[{"anchor":[18+i%3,24,78+i%2,129],"size":SIZE,"text":WORDS,
               "point":[47+i%2,77],"box":[38,70,58,84]} for i in range(24)]
    data=AnchorGeometry.fit(examples)
    data["pair_regression"]=model.geometry.pair_regression
    fitted=AnchorGeometry(data)
    fitted.sha256=model.geometry.sha256
    model.geometry=fitted
    out=passenger_pair_choice.choose(model,req,[21,46],"between")
    assert out["result"]["status"]=="proposed"
    patch=out["provenance"]["road_geometry"]
    assert patch["learned_estimator"]=="learned_anchor_envelope_and_retrieved_offsets"
    assert between_road.inside_contact_polygon(patch["contact_polygon_xy"],patch["box"])
    # Near-identical boxes claiming two tracks are not a certified vehicle pair.
    b["bbox_xyxy"]=[19,26,58,69]
    out=passenger_pair_choice.choose(model,req,[21,46],"between")
    assert out["provenance"]["withheld_reason"]=="selected_vehicles_substantially_occlude_each_other"


def test_dark_cars_request_cannot_promote_bus_as_second_car(roadside):
    req,model,raw,mask=roadside
    req["scene"]["tracks"][1]["class_name"]="Bus"
    out=PassengerDialogueGrounder(model).interpret(req)
    assert out["result"]["status"]=="needs_clarification"
    assert out["provenance"]["between_road"]["withheld_reason"]=="native_vehicle_class_disagrees_with_request"


def test_dark_pair_native_must_match_pixels_with_light_distractor(roadside):
    req,model,raw,mask=roadside
    req["scene"]["tracks"].append(obs(82,[225,30,262,69]))
    result=PassengerDialogueGrounder(model).interpret(req)
    assert result["result"]["status"]=="proposed"
    assert result["provenance"]["between_road"]["dark_appearance"]["distinct_supported_pair"]==[21,46]
    raw[0]="LANDMARK L1 L3"
    wrong=PassengerDialogueGrounder(model).interpret(req)
    assert wrong["result"]["status"]=="needs_clarification"
    assert wrong["provenance"]["between_road"]["withheld_reason"]=="requested_dark_pair_not_visually_supported"


def test_even_two_visible_light_cars_cannot_ground_a_dark_pair(roadside):
    req,model,raw,mask=roadside
    from pathlib import Path
    path=Path(req["scene"]["frame"]["image_path"])
    Image.new("RGB", SIZE, "white").save(path)
    req["scene"]["provenance"]["image_sha256"]=sha256(path.read_bytes()).hexdigest()
    result=PassengerDialogueGrounder(model).interpret(req)
    assert result["result"]["status"]=="needs_clarification"
    assert result["provenance"]["between_road"]["dark_appearance"]["possible_tracks"]==[]


def test_one_native_dark_car_does_not_silently_supply_second_identity(roadside,tmp_path):
    req, model, raw, mask = roadside
    raw[0]="LANDMARK L1"
    session=passenger_pair_choice.PassengerSession(req["scene"],PassengerDialogueGrounder(model),audit_dir=tmp_path/"audit")
    state=session.submit(WORDS)
    assert state["state"]=="needs_clarification" and state["proposal_id"] is None
    assert {i["track_id"] for i in state["pair_choices"]}=={21,46}
    with pytest.raises(ValueError):
        session.confirm("missing")
    invalid=session.select_pair([21,90],"between")
    assert invalid["state"]=="error" and invalid["proposal_id"] is None
    assert session.select_pair([21,46],"between")["state"]=="proposed"


def test_parked_condition_without_observed_state_requires_passenger(roadside):
    req, model, raw, mask=roadside
    req["text"]="Let me out between the two parked cars."
    result=PassengerDialogueGrounder(model).interpret(req)
    assert result["result"]["status"]=="needs_clarification"
    assert result["provenance"]["between_road"]["withheld_reason"]=="parking_state_unverified_in_still"


@pytest.mark.parametrize("change",["road","tracks","image","outer_point","box","witness"])
def test_proposal_revalidates_current_photo_patch_and_selected_tracks(roadside,change):
    req, model, raw, mask=roadside
    result=PassengerDialogueGrounder(model).interpret(req)
    assert result["result"]["status"]=="proposed"
    bad=deepcopy(result)
    if change=="road": mask[:] = False
    if change=="tracks": req["scene"]["tracks"][0]["bbox_xyxy"][2] += 1
    if change=="image": req["scene"]["provenance"]["image_sha256"]="0"*64
    if change=="outer_point": bad["result"]["destination_point_xy"]=[140,155]
    if change=="box": bad["result"]["destination_box_xyxy"]=[0,0,160,160]
    if change=="witness": bad["provenance"]["between_road"]["road_geometry"]["edge_witness_mask_cell_xy"][0]+=1
    with pytest.raises(ValueError):
        between_road.validate_offer(model,req,bad,"model")


@pytest.mark.parametrize("actor",["model","passenger_landmarks"])
def test_another_supported_cell_cannot_replace_the_actual_learned_winner(roadside,actor):
    req,model,raw,mask=roadside
    if actor=="model":
        out=PassengerDialogueGrounder(model).interpret(req)
        patch=out["provenance"]["between_road"]["road_geometry"]
    else:
        out=passenger_pair_choice.choose(model,req,[21,46],"between")
        patch=out["provenance"]["road_geometry"]
    x,y=patch["point"];left,top,right,bottom=patch["box"]
    # Move only inside the *same* admissible road patch, retaining its hashes, landmarks and eligibility.
    dx=SIZE[0]/mask.shape[1]
    another=[x-dx if x-dx>left else x+dx,y]
    assert left<another[0]<right and another!=patch["point"]
    out["result"]["destination_point_xy"]=another
    patch["point"]=another
    with pytest.raises(ValueError,match="fitted winner"):
        between_road.validate_offer(model,req,out,actor)


def test_changing_original_qwen_ids_cannot_reuse_a_legitimate_road_offer(roadside):
    req,model,raw,mask=roadside
    out=PassengerDialogueGrounder(model).interpret(req)
    record=out["provenance"]["resolver"]
    record["raw_text"]="LANDMARK L1"
    # Also change the serialized parsed decision, so only a genuine raw-to-offer replay detects this.
    record["decision"]=parse_landmark(record["raw_text"],landmark_messages(req)[1],req["scene"]["frame"])
    with pytest.raises(ValueError,match="raw native selection"):
        between_road.validate_offer(model,req,out,"model")


@pytest.mark.parametrize("actor",["model","passenger_landmarks"])
def test_opt_in_preserves_existing_lateral_success_and_binds_mode_before_confirmation(roadside,actor):
    req,model,raw,mask=roadside
    def outcome():
        return (PassengerDialogueGrounder(model).interpret(req) if actor=="model"
                else passenger_pair_choice.choose(model,req,[21,46],"between"))
    ordinary=outcome()
    model.between_road_mode=between_policy.EDGE_OR_BAY
    opted=outcome()
    assert ordinary["result"]==opted["result"] and opted["result"]["status"]=="proposed"
    receipt=opted["provenance"]["between_road"] if actor=="model" else opted["provenance"]
    assert receipt["between_road_mode"]==between_policy.EDGE_OR_BAY
    assert receipt["road_geometry"]["guard"]==between_road.VERSION
    between_road.validate_offer(model,req,opted,actor)
    model.between_road_mode=between_policy.EDGE
    with pytest.raises(ValueError,match="different roadside geometry configuration"):
        between_road.validate_offer(model,req,opted,actor)


def test_other_orientation_cannot_supply_an_unresolved_identity_or_overlapping_duplicate(roadside):
    req,model,raw,mask=roadside
    model.between_road_mode=between_policy.EDGE_OR_BAY
    raw[0]="LANDMARK L1"
    first=PassengerDialogueGrounder(model).interpret(req)
    assert first["provenance"]["between_road"]["withheld_reason"]=="two_distinct_current_vehicles_required"
    req["scene"]["tracks"][1]["bbox_xyxy"]=[19,25,49,67]
    second=passenger_pair_choice.choose(model,req,[21,46],"between")
    assert second["provenance"]["withheld_reason"]=="selected_vehicles_substantially_occlude_each_other"
    assert "bay_withheld_reason" not in second["provenance"]
    with pytest.raises(ValueError,match="Unknown between-road"):
        between_policy.check_mode("anything")
