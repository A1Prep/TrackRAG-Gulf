from copy import deepcopy
from hashlib import sha256

import cv2
import numpy as np
from PIL import Image
import pytest

from test_passenger_manual_offlane import provide_map
from trackrag.dropoff import tree_landmark, whole_bay
from trackrag.dropoff.interaction import PassengerSession
from trackrag.dropoff.place_evidence import PlaceDialogueGrounder, PublicPlaceBackend


@pytest.fixture
def trees(tmp_path, monkeypatch):
    classes = np.full((180,320), 27, np.uint8)
    classes[110:] = 13
    classes[133:153,245:] = 15
    # The bulk of this right-side crown is right of the image centre, although its box crosses it.
    cv2.ellipse(classes,(180,55),(21,34),0,0,360,30,-1)
    classes[65:132,176:183] = 30
    cv2.ellipse(classes,(55,65),(30,25),0,0,360,30,-1)
    classes[70:119,53:57] = 30
    path = tmp_path/"photo.png"
    Image.fromarray(classes).convert("RGB").save(path)
    raw = path.read_bytes()
    scene = {"source":"predicted","provenance":{"image_sha256":sha256(raw).hexdigest()},"tracks":[],
             "frame":{"image_path":str(path),"recording_id":"example","frame_id":4,"timestamp_s":0.0,
                      "width":320,"height":180}}
    packet = provide_map(monkeypatch, scene, classes)
    request = {"request_id":"check","input_mode":"typed","dialogue":[],"scene":scene,
               "text":"Could you drop me next to that tree?"}
    return request, raw, packet, classes


def test_generic_tree_asks_side_and_short_reply_requires_that_current_question(trees):
    request, raw, packet, _ = trees
    answer = tree_landmark.build(request,raw,packet)
    assert answer["provenance"]["tree_side_choices"] == ["left", "right"]
    assert len(answer["result"]["anchor_boxes_xyxy"]) == 2
    assert "tree_landmark_side" not in answer["provenance"]
    frame = {"recording_id":"example","frame_id":4}
    turns = [{"role":"user","text":request["text"],**frame},
             {"role":"assistant","text":answer["result"]["clarification_question"],**frame}]
    for reply in ("The tree on the right", "Right", "the one on the right, please"):
        clarified = {**request,"text":reply,"dialogue":turns}
        result = tree_landmark.build(clarified, raw, packet)
        assert result["provenance"]["tree_landmark_side"] == "right"
        assert len(result["result"]["anchor_boxes_xyxy"]) == 1
        assert result["result"]["destination_point_xy"] is None
        assert whole_bay.verify(clarified, result)["decision_reproduced"]
        assert tree_landmark.requested({**clarified, "dialogue":[]}) is None
        assert tree_landmark.requested({**clarified, "dialogue":[turns[0],{**turns[1],"frame_id":8}]}) is None
        assert tree_landmark.requested({**clarified, "dialogue":[turns[0],{**turns[1],"text":"Which car?"}]}) is None


@pytest.mark.parametrize(("box","message"), [
    ([177,120,182,125], "planted"), ([260,138,264,142], "may also serve a driveway"),
    ([145,150,150,155], "reads your mark as road")])
def test_original_mark_and_tree_replay_together_without_a_pin(trees, box, message):
    request, raw, packet, _ = trees
    req = {**request,"text":"Drop me beside the tree on the right."}
    out = tree_landmark.build(req, raw, packet, box=box)
    assert out["provenance"]["tree_landmark_side"] == "right"
    assert out["provenance"]["selected_box"] == box
    assert message in out["result"]["clarification_question"]
    assert out["result"]["destination_point_xy"] is None
    assert whole_bay.verify(req, out, requested_box=box)["decision_reproduced"]
    altered = deepcopy(out)
    altered["result"]["anchor_boxes_xyxy"][0][0] += 3
    with pytest.raises(ValueError):
        whole_bay.verify(req, altered, requested_box=box)
    with pytest.raises(ValueError):
        whole_bay.verify(req, out, requested_box=[10,10,15,15])


def test_crop_or_hedge_is_not_a_confirmed_individual_or_palm(trees, monkeypatch):
    request, raw, _, classes = trees
    m = np.where(classes == 30, 27, classes).astype("uint8")
    m[70:105,200:300] = 30  # a wide hedge with a level top and no lower trunk
    surface = provide_map(monkeypatch,request["scene"],m)
    out = tree_landmark.build({**request,"text":"Drop me by the tree on the right."},raw,surface)
    assert "tree_landmark_side" not in out["provenance"]
    assert "individual tree" in out["result"]["clarification_question"]
    m[0:110,205:300] = 30
    surface = provide_map(monkeypatch,request["scene"],m)
    out = tree_landmark.build({**request,"text":"Drop me next to the tall palm on the right, please."},raw,surface)
    assert "verify that it is a palm" in out["result"]["clarification_question"]
    assert not out["provenance"]["bay_choices"] and "tree_landmark_side" not in out["provenance"]


@pytest.mark.parametrize("words", [
    "Drop me before the tree.", "Don't stop by the tree on the right.", "Drop me by the red tree.",
    "Drop me next to the tree near the black car.", "Stop next to the tree behind the shops.",
    "Which tree is on the right?", "Drop me by the tree if it is accessible."])
def test_unsupported_tree_qualifiers_cannot_authorize_a_location(words):
    assert tree_landmark.requested({"text":words,"dialogue":[]}) is None


def test_same_frontage_requires_projected_canopy_kerb_footway_and_no_separating_road(trees):
    req, _, _, _ = trees
    frame = req["scene"]["frame"]
    m = np.full((180,320),17,np.uint8)
    m[10:63,188:269] = 30
    m[77:98] = 15
    m[98:100] = 2
    m[100:] = 10
    box = [188,10,269,63]
    assert tree_landmark._on_same_frontage(m,frame,[211,120],box,True)
    assert not tree_landmark._on_same_frontage(m,frame,[65,120],box,True)
    for label in (13, 3, 19, 55):
        blocked = m.copy()
        blocked[88:91] = label
        assert not tree_landmark._on_same_frontage(blocked,frame,[211,120],box,True)
    no_kerb = np.where(m == 2,15,m)
    assert not tree_landmark._on_same_frontage(no_kerb,frame,[211,120],box,True)


@pytest.mark.parametrize("authorized", [True, False])
def test_a_generic_parking_proposal_cannot_bypass_a_tree_request(trees, monkeypatch, authorized):
    req, _, _, _ = trees
    req = {**req, "text":"Can you stop by the tree on the right?"}
    backend = PublicPlaceBackend([])
    monkeypatch.setattr(backend,"allows_photo",lambda _:authorized)
    grounder = PlaceDialogueGrounder(object(), backend)
    monkeypatch.setattr(whole_bay, "explicit_request",lambda *args:pytest.fail("An unrelated bay was consulted"))
    out = grounder.interpret(req)
    assert out["result"]["destination_point_xy"] is None and not out["provenance"]["bay_choices"]
    assert whole_bay.verify_current(req,out)["decision_reproduced"]
    assert (out["provenance"]["reason"] == "tree_surface_evidence_unavailable") != authorized


def test_old_generic_pin_cannot_be_saved_for_a_tree_even_if_session_is_still_open(trees, tmp_path):
    req, _, _, _ = trees
    session = PassengerSession(req["scene"], object(), audit_dir=tmp_path/"session")
    old = {"result":{"status":"proposed","destination_point_xy":[200,150]},
           "provenance":{"method":whole_bay.AUTO,"clearance_policy":whole_bay.CLEARANCE_POLICY}}
    session._request, session._place_offer, session._result = req, old, old["result"]
    session._state, session._proposal, session._selected_by = "proposed", "old", "model"
    with pytest.raises(ValueError,match="Recheck the tree"):
        whole_bay.verify_current(req, old)
    with pytest.raises(ValueError,match="Recheck the tree"):
        session.confirm("old")
    assert session.view()["error"]["code"] == "tree_confirmation_blocked"
    assert session.view()["proposal_id"] is None


def test_only_exact_mark_inside_the_same_measured_frontage_can_gain_a_point(trees, monkeypatch):
    req, photo, _, labels = trees
    req = {**req,"text":"Drop me beside the tree on the right."}
    labels[137:156,170:200] = 10
    source = provide_map(monkeypatch, req["scene"], labels)
    measured = {"point":[182,145],"box":[177,140,189,151],
                "polygon":[[177,140],[189,140],[189,151],[177,151]],
                "evidence":{"recipe":tree_landmark.POLICY,"coverage":"interior_point_only"}}
    monkeypatch.setattr(tree_landmark,"_parking",lambda *args:[deepcopy(measured)])
    options = tree_landmark.build(req, photo, source)
    assert options["result"]["destination_point_xy"] is None
    only, = options["provenance"]["bay_choices"]
    selected = whole_bay.choice_outcome(req, options, only["id"])
    assert selected["result"]["anchor_boxes_xyxy"] == options["result"]["anchor_boxes_xyxy"]
    assert whole_bay.verify_current(req, selected)["decision_reproduced"]
    inside = [181,144,187,150]
    marked = tree_landmark.build(req, photo, source, box=inside)
    assert marked["result"]["destination_point_xy"] == [184,147]
    assert whole_bay.verify_current(req,marked,requested_box=inside)["decision_reproduced"]
    for invalid in ([174,143,178,148],[120,152,124,156]):
        withheld = tree_landmark.build(req, photo, source, box=invalid)
        assert not withheld["provenance"]["bay_choices"] and withheld["result"]["destination_point_xy"] is None
        assert withheld["provenance"]["tree_parking_available"] is True
        assert whole_bay.verify_current(req,withheld,requested_box=invalid)["decision_reproduced"]
