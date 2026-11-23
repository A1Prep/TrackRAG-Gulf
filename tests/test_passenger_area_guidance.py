"""Check signed vehicle context, targeted area clarification and cold replay of passenger marks."""
from copy import deepcopy
import json

import pytest

from tests.test_passenger_between_road import roadside
from tests.test_passenger_manual_offlane import provide_bay
from tools.dropoff_passenger_handoff import bound_outcome
from trackrag.dropoff import manual_mark
from trackrag.dropoff.infrastructure import PassengerDialogueGrounder
from trackrag.dropoff.landmarks import InvalidLandmarkIDs
from trackrag.dropoff.passenger_pair_choice import PassengerSession,pair_intent
from trackrag.dropoff.road_cache import save_mask


def test_two_matched_vehicles_remain_visible_for_an_area_question(roadside,tmp_path):
    req,model,raw,mask=roadside;mask[:]=False
    session=PassengerSession(req["scene"],PassengerDialogueGrounder(model),audit_dir=tmp_path/"app")
    native=session.submit(req["text"])
    assert native["state"]=="needs_clarification" and native["proposal_id"] is None
    hint=native["area_landmarks"]
    assert hint["source"]=="model" and hint["track_ids"]==[21,46] and hint["request_id"]==native["request_id"]
    assert hint["boxes_xyxy"]==[t["bbox_xyxy"] for t in req["scene"]["tracks"][:2]]
    chosen=session.select_pair([46,21],"between")
    assert chosen["area_landmarks"]["source"]=="passenger_landmarks" and chosen["area_landmarks"]["track_ids"]==[46,21]
    offroad=session.select_area([5,1,15,10]);assert offroad["state"]=="needs_clarification"
    assert offroad["area_landmarks"]==chosen["area_landmarks"]
    assert "area_landmarks" not in session.cancel()
    raw[0]="LANDMARK L1"
    unknown=session.submit(req["text"])
    assert unknown["state"]=="needs_clarification" and "area_landmarks" not in unknown


def test_stale_native_evidence_cannot_display_current_vehicle_outlines(roadside,tmp_path):
    req,model,raw,mask=roadside;mask[:]=False
    out=PassengerDialogueGrounder(model).interpret(req)
    app=PassengerSession(req["scene"],PassengerDialogueGrounder(model),audit_dir=tmp_path/"app");app._request=req
    assert app._landmarks_awaiting_area(model,out,"model")
    for field in ("image_sha256","request_sha256","geometry_sha256","current_observations_sha256"):
        altered=deepcopy(out);altered["provenance"]["between_road"][field]="0"*64
        assert app._landmarks_awaiting_area(model,altered,"model") is None
    altered=deepcopy(out);altered["provenance"]["resolver"]["raw_text"]="LANDMARK L1"
    assert app._landmarks_awaiting_area(model,altered,"model") is None


def test_crossover_head_in_appositive_offers_vehicle_selection_but_a_store_does_not():
    text="Please use the road between the silver car, the one nearest the camera, and the dark crossover farther up."
    assert pair_intent({"text":text,"dialogue":[]})
    assert not pair_intent({"text":text.replace('dark crossover farther up','shop beside the dark crossover'),"dialogue":[]})


def test_a_named_unmatched_pair_offers_a_direct_mark_without_fabricating_matched_outlines(roadside,tmp_path):
    req,model,raw,mask=roadside
    raw[0]='LANDMARK L1'
    app=PassengerSession(req['scene'],PassengerDialogueGrounder(model),audit_dir=tmp_path/'app')
    state=app.submit('Between the nearer car, the one cut off by the photo, and the cream crossover farther along, please.')
    assert state['state']=='needs_clarification' and state.get('unmatched_described_pair') is True
    assert 'area_landmarks' not in state and state['proposal_id'] is None


def archived_road_mark(req,model,box):
    """Produce the old road-only journal shape to keep historical exports verifiable."""
    from trackrag.dropoff.data import inference_input_sha256
    from trackrag.dropoff.landmarks import resolve_image
    from trackrag.dropoff.passenger import digest,proposed,unresolved
    image,binding=resolve_image(req,model.config)
    center=[(box[0]+box[2])/2,(box[1]+box[3])/2]
    with image:
        support,road=model._evidence(image,req,binding["sha256"],center,box)
    target=proposed(*support) if support else unresolved("The selected area has no visible road support. Can you outline a different patch?")
    return {"ok":True,"request_id":req["request_id"],"result":target,"provenance":{
        "method":"explicit_passenger_region","selected_box":box,"image":binding,"request_sha256":digest(req),
        "inference_input_sha256":inference_input_sha256(req,binding["sha256"]),"road_evidence":road,
        "independent_model_proposal":False}}


def test_archived_road_only_mark_replays_the_position_it_originally_offered(roadside,tmp_path):
    req,model,raw,mask=roadside
    # The rectangle's middle is on an observed vehicle, while its right edge covers visible road.
    box=[39,35,53,54];out=archived_road_mark(req,model,box)
    assert out['result']['status']=='proposed' and out['result']['destination_box_xyxy']==box
    point=out['result']['destination_point_xy'];assert 48<=point[0]<53 and 35<=point[1]<54
    cache=tmp_path/'mask';save_mask(cache,req['scene']['provenance']['image_sha256'],(320,180),mask)
    evidence=bound_outcome(out,req,passenger_selected=True,requested_box=box,road_cache=cache)['passenger_mark_verification']
    assert evidence['decision_reproduced'] and evidence['road_mask_replayed'] and evidence['independent_model_proposal'] is False
    assert not manual_mark.verify(req,out)['road_mask_replayed']
    obstructed=[25,32,39,54];missing=archived_road_mark(req,model,obstructed)
    assert missing['result']['status']=='needs_clarification' and manual_mark.verify(req,missing,requested_box=obstructed,road_cache=cache)['decision_reproduced']


@pytest.mark.parametrize('change',['requested','point','road_hash','obstacles','role','outside','missing_cache'])
def test_historical_manual_export_refuses_changed_actions_surfaces_and_locations(roadside,tmp_path,change):
    req,model,raw,mask=roadside
    box=[47,70,81,79];out=archived_road_mark(req,model,box)
    cache=tmp_path/'mask';save_mask(cache,req['scene']['provenance']['image_sha256'],(320,180),mask)
    if change=='requested': box=[30,30,39,44]
    if change=='point': out['result']['destination_point_xy'][0]+=1.25
    if change=='road_hash':out['provenance']['road_evidence']['road_mask_sha256']='1'*64
    if change=='obstacles':out['provenance']['road_evidence']['current_exclusion_boxes']=[]
    if change=='role':out['provenance']['independent_model_proposal']=True
    if change=='outside':out['result']['destination_point_xy']=[20,34]
    if change=='missing_cache':cache=tmp_path/'missing'
    with pytest.raises((ValueError,FileNotFoundError)):
        bound_outcome(out,req,passenger_selected=True,requested_box=box,road_cache=cache)


@pytest.mark.parametrize("words,code", [
    ("Drop me between the two nearest SUVs on the right.", "invalid_model_output"),
    ("Drop me between the grey sedan and the blue van.", "invalid_input_or_output"),
])
def test_malformed_vehicle_ids_retain_raw_errors_and_allow_source_bound_selection(roadside, tmp_path, words, code, monkeypatch):
    from tools.dropoff_passenger_handoff import session as export
    req, model, raw, mask = roadside
    provide_bay(monkeypatch,req["scene"],[42,66,88,84])
    resolve = model.resolver.resolve
    def resolver(request, image):
        raw[0] = "LANDMARK L2" if "Beside the blue van" in request["text"] else "LANDMARK L1"
        result = resolve(request, image)
        if not request["request_id"].startswith("pair-"):
            result.pop("decision")
            result.update(raw_text="LANDMARK L1 LANDMARK L2", parse_error=InvalidLandmarkIDs.message)
        return result
    model.resolver.resolve = resolver
    audit = tmp_path / "app"
    app = PassengerSession(req["scene"], PassengerDialogueGrounder(model), audit_dir=audit)
    state = app.submit(words)
    assert state["state"] == "error" and state["error"]["code"] == code
    assert state["proposal_id"] is None and state["result"] is None and "area_landmarks" not in state
    assert "Which two" in state["landmark_error_guidance"] and len(state["pair_choices"]) == 2
    with pytest.raises(ValueError):
        app.confirm(None)
    journal, = audit.glob("*.jsonl")
    original = json.loads(journal.read_bytes().splitlines()[-1])["detail"]["outcome"]
    assert original["ok"] is False and original["provenance"]["resolver"]["raw_text"] == "LANDMARK L1 LANDMARK L2"
    if code == "invalid_input_or_output":
        assert [r["resolver"]["raw_text"] for r in original["provenance"]["resolver"]["pair_resolution"]["subqueries"]] == [
            "LANDMARK L1", "LANDMARK L2"]
        assert original["provenance"]["rejected_landmark_stage"] == "whole_instruction_pair_consistency"
    chosen = app.select_pair([21, 46], "between")
    assert chosen["selected_by"] == "passenger_landmarks" and "landmark_error_guidance" not in chosen
    result = export(journal)
    assert result["events"][1]["outcome"]["ok"] is False
    assert result["events"][2]["phase"] == "passenger_selected_landmarks"
    state = app.submit(words)
    marked = app.select_area([47, 70, 81, 79])
    assert marked["state"] == "proposed" and marked["selected_by"] == "passenger"
    assert app.confirm(marked["proposal_id"])["state"] == "confirmed"


@pytest.mark.parametrize("mismatch", ["timeout", "integrity", "other_error", "mixed_request", "photo_digest", "request_digest", "method"])
def test_pair_recovery_never_turns_other_errors_or_requests_into_a_mark(roadside, tmp_path, mismatch):
    from trackrag.dropoff.contracts import error_outcome
    req, model, raw, mask = roadside
    words = "Between the van and the hospital gate" if mismatch == "mixed_request" else "Between the two cars on the left"
    def failed(request):
        code = {"timeout":"timeout", "integrity":"artifact_invalid"}.get(mismatch, "invalid_input_or_output")
        error = "Landmark adapter changed" if mismatch == "other_error" else InvalidLandmarkIDs.message
        source = {"method":"qwen_landmark_region"}
        if mismatch == "photo_digest": source["image"] = {"sha256":"0"*64}
        if mismatch == "request_digest": source["request_sha256"] = "0"*64
        if mismatch == "method": source["method"] = "unverified"
        return error_outcome(code, error, request["request_id"], source)
    model.interpret = failed
    app = PassengerSession(req["scene"], PassengerDialogueGrounder(model), audit_dir=tmp_path/"app")
    state = app.submit(words)
    assert state["state"] == "error" and "landmark_error_guidance" not in state
    with pytest.raises(ValueError):
        app.select_area([47, 70, 81, 79])
