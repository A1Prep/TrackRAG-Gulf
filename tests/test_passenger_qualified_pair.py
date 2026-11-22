"""Normal passenger-session checks on generated street pixels, predicted-shape fixtures and fitted geometry."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from tests.test_passenger_between_road import SIZE, roadside
from tests.test_passenger_pair_dialogue import deictic
from tests.test_passenger_projected_pair import obs
from trackrag.dropoff import between_policy, between_road, pair_dialogue, passenger_pair_choice, qualified_pair
from trackrag.dropoff.anchor_geometry import AnchorGeometry
from trackrag.dropoff.infrastructure import PassengerDialogueGrounder
from trackrag.dropoff.model import GroundingConfig
from trackrag.dropoff.road_cache import SIDEWALK_RECIPE_SHA256, read_mask, save_mask


@pytest.fixture
def parking_row(roadside, deictic, tmp_path, monkeypatch):
    req,model,raw,mask=roadside
    _,example_ranker=deictic
    model.referent_ranker=example_ranker.referent_ranker
    monkeypatch.setattr(qualified_pair,"GEOMETRY",model.geometry.sha256)
    monkeypatch.setattr(qualified_pair,"RANKER",model.referent_ranker.sha256)
    model.between_road_mode=between_policy.EDGE_OR_BAY
    image_path=Path(req["scene"]["frame"]["image_path"])
    def make(*,third="distant",color=(27,27,27),sidewalk=True):
        tracks=[obs(21,[35,40,96,94]),obs(46,[158,41,222,94])]
        if third=="distant": tracks.append(obs(82,[241,7,253,17]))
        if third=="nearby": tracks.append(obs(82,[255,42,307,94]))
        req["scene"]["tracks"]=tracks
        with Image.new("RGB",SIZE,(97,97,97)) as picture:
            paint=ImageDraw.Draw(picture)
            for item in tracks:
                paint.rectangle(item["bbox_xyxy"],fill=color if item["track_id"]==82 else (25,25,25))
            picture.save(image_path)
        photo=sha256(image_path.read_bytes()).hexdigest()
        req["scene"]["provenance"]["image_sha256"]=photo
        ys=(np.arange(mask.shape[0])+.5)*SIZE[1]/mask.shape[0]
        mask[:]=(ys>=90)[:,None]
        # Each generated scene variant uses its own new cache. Fixtures are never genuine sensor evaluation.
        cache=tmp_path/("walk-"+photo[:10]+("-yes" if sidewalk else "-no"))
        if sidewalk:
            save_mask(cache,photo,SIZE,~mask,recipe=SIDEWALK_RECIPE_SHA256)
        save_mask(cache,photo,SIZE,mask)
        model.road=SimpleNamespace(mask=lambda im,fingerprint:mask,cache=cache,image_hash=photo)
        return req,model,raw,mask
    return make


def test_facing_row_is_real_opt_in_for_language_and_explicit_choices(parking_row,tmp_path):
    req,model,raw,mask=parking_row(third=None)
    model.between_road_mode=between_policy.EDGE
    assert PassengerDialogueGrounder(model).interpret(req)["result"]["status"]=="needs_clarification"
    model.between_road_mode=between_policy.EDGE_OR_BAY
    session=passenger_pair_choice.PassengerSession(req["scene"],PassengerDialogueGrounder(model),audit_dir=tmp_path/"app")
    automatic=session.submit(req["text"])
    assert automatic["state"]=="proposed" and automatic["selected_by"]=="model"
    assisted=session.select_pair([21,46],"between")
    assert assisted["state"]=="proposed" and assisted["selected_by"]=="passenger_landmarks"
    assert assisted["result"]==automatic["result"]
    with pytest.raises(ValueError): session.confirm(automatic["proposal_id"])
    assert session.confirm(assisted["proposal_id"])["state"]=="confirmed"
    out=passenger_pair_choice.choose(model,req,[21,46],"between")
    p=out["provenance"]["road_geometry"]
    assert p["guard"]==between_policy.BAY_GUARD and p["witness_kinds"]==["independent_sidewalk_semantics"]
    assert p["lateral_withheld_reason"]=="indistinguishable_ground_contact_depth"
    damaged=deepcopy(out)
    damaged["provenance"]["road_geometry"]["sidewalk_mask_sha256"]="0"*64
    with pytest.raises(ValueError): between_road.validate_offer(model,req,damaged,"passenger_landmarks")


def test_unique_supported_gap_from_three_dark_vehicles_preserves_actual_initial_question(parking_row,tmp_path):
    req,model,raw,mask=parking_row()
    original=pair_dialogue.unpointed_pair(model,req)
    assert original["provenance"]["native_model_calls"]==0 and len(original["provenance"]["eligible_tracks"])==3
    out=PassengerDialogueGrounder(model).interpret(req)
    assert out["result"]["status"]=="proposed"
    info=out["provenance"]
    assert info["method"]==qualified_pair.METHOD and info["retained_native_track_ids"]==[]
    assert info["original_outcome"]==original and info["independent_native_proposal"] is False
    assert info["additional_native_calls"]==0 and info["selected_track_ids"]==[21,46]
    assert len(info["evaluated_pairs"])==3 and sum("road_guard" in p for p in info["evaluated_pairs"])==1
    qualified_pair.validate(model,req,out)
    for tweak in ("evaluated_pairs","selected_track_ids"):
        altered=deepcopy(out)
        altered["provenance"][tweak].pop()
        with pytest.raises(ValueError,match="did not reproduce"): qualified_pair.validate(model,req,altered)
    session=passenger_pair_choice.PassengerSession(req["scene"],PassengerDialogueGrounder(model),audit_dir=tmp_path/"app")
    offer=session.submit(req["text"])
    assert offer["selected_by"]==qualified_pair.ACTOR and offer["state"]=="proposed"
    assert "one visible road gap" in offer["turns"][-1]["text"]
    assert session.confirm(offer["proposal_id"])["state"]=="confirmed"


@pytest.mark.parametrize("colour",[(25,25,25),(126,126,126)])
def test_another_supported_pair_competes_even_if_its_appearance_is_uncertain(parking_row,colour):
    req,model,raw,mask=parking_row(third="nearby",color=colour)
    first=pair_dialogue.unpointed_pair(model,req)
    assert first["result"]["status"]=="needs_clarification"
    assert qualified_pair.search(model,req,first) is None
    assert PassengerDialogueGrounder(model).interpret(req)==first


def test_ordinary_road_without_independent_sidewalk_or_paint_keeps_clarification(parking_row):
    req,model,raw,mask=parking_row(sidewalk=False)
    assert PassengerDialogueGrounder(model).interpret(req)["result"]["status"]=="needs_clarification"
    choice=passenger_pair_choice.choose(model,req,[21,46],"between")
    assert choice["result"]["status"]=="needs_clarification"
    assert choice["provenance"]["bay_withheld_reason"]=="no_independent_roadside_evidence"


def test_single_native_identity_must_belong_to_the_independently_qualified_pair(parking_row):
    req,model,raw,mask=parking_row(third="distant",color=(242,242,242))
    raw[0]="LANDMARK L1"
    positive=PassengerDialogueGrounder(model).interpret(req)
    assert positive["provenance"]["method"]==qualified_pair.METHOD
    assert positive["provenance"]["retained_native_track_ids"]==[21]
    assert positive["provenance"]["original_outcome"]["provenance"]["resolver"]["raw_text"]=="LANDMARK L1"
    raw[0]="LANDMARK L3"
    unrelated=PassengerDialogueGrounder(model).interpret(req)
    assert unrelated["result"]["status"]=="needs_clarification"
    assert unrelated["provenance"]["method"]=="qwen_landmark_region"


def test_too_many_competitors_retires_whole_search_instead_of_truncating_tracks(parking_row):
    req,model,raw,mask=parking_row()
    req["text"]="Drop me between the two cars."
    for i in range(7): req["scene"]["tracks"].append(obs(200+i,[3+21*i,7,19+21*i,19]))
    first=pair_dialogue.unpointed_pair(model,req)
    assert first and qualified_pair.search(model,req,first) is None


def test_detector_car_category_does_not_establish_a_requested_suv_or_sedan(parking_row):
    req,model,raw,mask=parking_row()
    for noun in ("SUVs","sedans"):
        req["text"]=f"Drop me between the two {noun}."
        first=pair_dialogue.unpointed_pair(model,req)
        assert first and qualified_pair.search(model,req,first) is None


@pytest.mark.parametrize("prior",["initial_question","one_native_id"])
def test_cold_file_replay_includes_typed_mask_and_every_competitor(parking_row,tmp_path,monkeypatch,prior):
    color=(28,28,28) if prior=="initial_question" else (240,240,240)
    req,model,raw,mask=parking_row(color=color)
    geometry_path=tmp_path/"measured-geometry.json"
    data=AnchorGeometry.fit([{"anchor":[35+i,40,222,94],"point":[123+i,98],"box":[108,91,135,115],
                              "size":SIZE,"text":req["text"]} for i in range(10)])
    data["pair_regression"]=model.geometry.pair_regression
    geometry_path.write_text(json.dumps(data),encoding="utf-8")
    model.geometry=AnchorGeometry.load(geometry_path)
    monkeypatch.setattr(qualified_pair,"GEOMETRY",model.geometry.sha256)
    monkeypatch.setattr(passenger_pair_choice,"GEOMETRY",model.geometry.sha256)
    model.config=GroundingConfig();model.resolver.config=model.config
    if prior=="one_native_id": raw[0]="LANDMARK L1"
    offered=PassengerDialogueGrounder(model).interpret(req)
    assert offered["provenance"]["method"]==qualified_pair.METHOD
    cache=model.road.cache
    verification=qualified_pair.replay_files(req,offered,geometry_path,cache,tmp_path/"fixture-ranker.json",config=model.config)
    assert verification["all_candidate_pairs_recomputed"]==(3 if prior=="initial_question" else 1)
    assert verification["decision_reproduced"] and verification["additional_native_calls"]==0
    changed=deepcopy(offered)
    changed["provenance"]["road_geometry"]["sidewalk_mask_sha256"]="0"*64
    with pytest.raises(ValueError,match="did not reproduce"):
        qualified_pair.replay_files(req,changed,geometry_path,cache,tmp_path/"fixture-ranker.json",config=model.config)
    missing=tmp_path/"cache-without-sidewalk"
    save_mask(missing,req["scene"]["provenance"]["image_sha256"],SIZE,mask)
    with pytest.raises(ValueError,match="did not reproduce"):
        qualified_pair.replay_files(req,offered,geometry_path,missing,tmp_path/"fixture-ranker.json",config=model.config)


@pytest.mark.parametrize("failure,code",[("disk","optional_evidence_unavailable"),("stale","optional_evidence_invalid")])
def test_optional_evidence_errors_persist_separately_from_a_geometric_abstention(parking_row,failure,code):
    req,model,raw,mask=parking_row()
    initial=pair_dialogue.unpointed_pair(model,req)
    if failure=="disk":
        def unavailable(*args): raise FileNotFoundError("Synthetic missing segmentation cache")
        model.road.mask=unavailable
    else:
        model.road.image_hash="1"*64
    outcome=PassengerDialogueGrounder(model).interpret(req)
    assert outcome["result"]==initial["result"]
    error=outcome["provenance"]["qualified_pair_failure"]
    assert error["code"]==code and error["additional_native_calls"]==0
    assert qualified_pair.original_after_failure(outcome)==initial
    changed=deepcopy(outcome);changed["provenance"]["qualified_pair_failure"]["original_outcome_sha256"]="0"*64
    with pytest.raises(ValueError,match="preserve"): qualified_pair.original_after_failure(changed)


def test_two_sessions_hold_shared_model_lock_during_native_and_explicit_offer_validation(parking_row,tmp_path,monkeypatch):
    req,model,raw,mask=parking_row(third=None)
    first,second=deepcopy(req["scene"]),deepcopy(req["scene"])
    second_file=tmp_path/"same-size-other-frame.png"
    with Image.open(second["frame"]["image_path"]) as src:
        image=src.convert("RGB");image.putpixel((0,0),(225,225,225));image.save(second_file)
    second["frame"]["image_path"]=str(second_file)
    other_sha=sha256(second_file.read_bytes()).hexdigest()
    second["provenance"]["image_sha256"]=other_sha
    cache=model.road.cache
    for recipe in (False,True):
        if recipe: save_mask(cache,other_sha,SIZE,~mask,recipe=SIDEWALK_RECIPE_SHA256)
        else: save_mask(cache,other_sha,SIZE,mask)
    def read(image,fingerprint):
        current=read_mask(cache,fingerprint,image.size)
        assert current is not None
        model.road.image_hash=fingerprint
        return current
    model.road.mask=read
    actual_resolver=model.resolver.resolve
    def resolve(request,image):
        return {**actual_resolver(request,image),"image_sha256":request["scene"]["provenance"]["image_sha256"]}
    model.resolver.resolve=resolve
    observed=[]
    original_validation=between_road.validate_offer
    def checked(grounder,request,outcome,actor):
        # A different thread must *never* be able to change the currently selected road/sidewalk during validation.
        acquired=[]
        def probe():
            got=model.lock.acquire(blocking=False);acquired.append(got)
            if got: model.lock.release()
        contender=Thread(target=probe);contender.start();contender.join()
        assert acquired==[False]
        observed.append((actor,request["scene"]["provenance"]["image_sha256"]))
        return original_validation(grounder,request,outcome,actor)
    monkeypatch.setattr(between_road,"validate_offer",checked)
    def browser(scene):
        app=passenger_pair_choice.PassengerSession(scene,PassengerDialogueGrounder(model),audit_dir=tmp_path/"sessions")
        assert app.submit(req["text"])["state"]=="proposed"
        assert app.select_pair([21,46],"between")["state"]=="proposed"
    with ThreadPoolExecutor(max_workers=2) as clients:
        list(clients.map(browser,[first,second]))
    assert set(observed)=={(a,s["provenance"]["image_sha256"]) for a in ("model","passenger_landmarks") for s in (first,second)}


def test_a_contradictory_original_bound_sidewalk_is_an_evidence_error(parking_row,tmp_path):
    req,model,raw,mask=parking_row()
    model.road.cache=tmp_path/"conflicting-semantics"
    contradictory=~mask;contradictory[-1,-1]=True
    save_mask(model.road.cache,model.road.image_hash,SIZE,contradictory,recipe=SIDEWALK_RECIPE_SHA256)
    out=PassengerDialogueGrounder(model).interpret(req)
    assert out["result"]["status"]=="needs_clarification"
    failure=out["provenance"]["qualified_pair_failure"]
    assert failure["code"]=="optional_evidence_invalid" and "contradictory_semantic_surfaces" in failure["message"]


@pytest.mark.parametrize("reason",["unsupported_sidewalk_evidence","roadside_image_dimensions_differ","unknown_future_error"])
def test_a_lateral_winner_cannot_be_certified_when_a_competitor_fails_to_run(roadside,deictic,monkeypatch,reason):
    from trackrag.dropoff import curb_facing
    from trackrag.dropoff.projected_pair import Ineligible
    req,model,raw,mask=roadside
    req["text"]="Drop me between the two cars."
    req["scene"]["tracks"].append(obs(82,[225,45,269,90]))
    model.referent_ranker=deictic[1].referent_ranker
    monkeypatch.setattr(qualified_pair,"RANKER",model.referent_ranker.sha256)
    monkeypatch.setattr(qualified_pair,"GEOMETRY",model.geometry.sha256)
    model.between_road_mode=between_policy.EDGE_OR_BAY
    def abort(*args,**kwargs): raise Ineligible(reason)
    monkeypatch.setattr(curb_facing,"estimate",abort)
    confirmed_lateral=passenger_pair_choice.choose(model,req,[21,46],"between")
    assert confirmed_lateral["provenance"]["road_geometry"]["guard"]==between_road.VERSION
    outcome=PassengerDialogueGrounder(model).interpret(req)
    assert outcome["result"]["status"]=="needs_clarification"
    assert reason in outcome["provenance"]["qualified_pair_failure"]["message"]


@pytest.mark.parametrize("method,actor",[("explicit_passenger_landmark_pair","model"),
    ("explicit_passenger_region","model"),("passenger_requested_pointing","model"),
    ("qwen_landmark_region","passenger_landmarks"),("qwen_landmark_region","passenger"),
    ("deictic_pair_clarification","passenger"),("explicit_passenger_region",qualified_pair.ACTOR)])
def test_incorrect_actor_never_receives_a_confirmation_token(roadside,tmp_path,method,actor):
    req,model,raw,mask=roadside
    session=passenger_pair_choice.PassengerSession(req["scene"],PassengerDialogueGrounder(model),audit_dir=tmp_path/"app")
    session._request=req
    actual=PassengerDialogueGrounder(model).interpret(req)
    actual["provenance"]["method"]=method
    with pytest.raises(ValueError,match="source and actual decision method disagree"):
        session._accept(actual,actor)
    assert session.view()["proposal_id"] is None
