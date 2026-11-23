"""Synthetic source and role tests: native geometry, signed observations and marks stay distinct."""
from copy import deepcopy
from hashlib import sha256

from PIL import Image
import pytest

from trackrag.dropoff import driveway_apron, native_destination as native, row_extension, whole_bay
from test_driveway_apron import driveway
from test_passenger_manual_offlane import provide_map


def option(*, entrance=False):
    recipe=driveway_apron.RECIPE if entrance else row_extension.RECIPE
    shape={"point":[720,340],"box":[715,335,725,345],"support_patch":[[715,335],[725,335],[725,345],[715,345]],
           "evidence":{"recipe":recipe,"nearest_row_track_id":3}}
    return {"native_option":shape,"vehicle_crops":[[9,[470,170,553,235]],[3,[600,180,710,275]]],
            "descriptions":{"frontage":"Saint Anne hotel","roadside":"large refuse bin", "nearer_vehicle":"gray SUV",
                            "farther_vehicle":"white sedan"},"rear_vehicle":"white sedan", "evidence":{"status":"OPEN"}}


def assert_unmatched(answer):
    assert answer["provenance"]["reason"]=="native_reference_unmatched"
    assert answer["result"]["status"]=="needs_clarification" and answer["result"]["destination_point_xy"] is None
    assert not answer["provenance"]["bay_choices"] and not answer["result"].get("anchor_boxes_xyxy")


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    photo=tmp_path/"street.png"
    Image.new("RGB",(1000,520),"gray").save(photo)
    data=photo.read_bytes(); key=sha256(data).hexdigest()
    sc={"source":"predicted","frame":{"recording_id":"native-fixture","frame_id":8,"timestamp_s":0.0,
        "width":1000,"height":520,"image_path":str(photo)},"provenance":{"image_sha256":key},"tracks":[]}
    def req(words):
        return {"request_id":"request","scene":deepcopy(sc),"text":words,"input_mode":"typed","dialogue":[]}
    # Only the heavyweight model boundary is replaced here. Its actual signed bank
    # is checked separately on the public originals in the release experiment.
    obs={"body":{"sides":{"right":{"reviewed":[1]}}},"test":"fixture"}
    rows=[option()]
    def review(scene,photo_data,surface,signed):
        if signed!=obs or scene!=sc or data!=photo_data or surface!={"test":"map"}:
            raise ValueError("The input to the synthetic visual reader changed.")
        return deepcopy(rows)
    monkeypatch.setattr(native.visual_space,"verified_extension",review)
    return req,data,{"test":"map"},obs,rows


def test_named_vehicle_entrance_requires_the_visible_name(fixture):
    req,photo,surface,obs,rows=fixture
    rows[:]=[option(entrance=True)]
    words="Could you let us out in the St. Anne entrance driveway, off the main road?"
    answer=native.language_outcome(req(words),photo,surface,obs)
    assert answer["result"]["destination_point_xy"]==[720,340]
    assert answer["provenance"]["bay_outline"]["extent"]=="interior_point_only"
    assert whole_bay.verify_current(req(words),answer)["archived_native_inspections"]==1
    assert_unmatched(native.language_outcome(req(words.replace("St. Anne","Amber Palace")),photo,surface,obs))
    for misleading in ("St. Anne hotel across the road","Saint Anne or other hotel","Saint's Bistro; Anne Apartments"):
        rows[0]["descriptions"]["frontage"]=misleading
        assert_unmatched(native.language_outcome(req(words),photo,surface,obs))


def test_behind_nearest_car_requires_identity_direction_material_and_the_pictured_landmark(fixture):
    req,photo,surface,obs,rows=fixture
    words="Let me out on the paved shoulder behind the white sedan, near the big rubbish skip on the right."
    a=native.language_outcome(req(words),photo,surface,obs)
    assert a["result"]["status"]=="proposed" and a["provenance"]["independent_model_proposal"]
    assert whole_bay.verify_current(req(words),a)["decision_reproduced"]
    for changed in (words.replace("white","black"),words.replace("white sedan","gray SUV"),words.replace("right","left"),
                    words.replace("paved","gravel"),words.replace("rubbish skip","pharmacy")):
        assert_unmatched(native.language_outcome(req(changed),photo,surface,obs))
    rows[0]["rear_vehicle"]="NONE"
    partial=native.language_outcome(req(words),photo,surface,obs)
    assert partial["result"]["status"]=="needs_clarification" and len(partial["provenance"]["bay_choices"])==1
    assert partial["result"]["destination_point_xy"] is None and not partial["provenance"]["independent_model_proposal"]
    chosen=whole_bay.choice_outcome(req(words),partial,"bay-1")
    assert chosen["result"]["destination_point_xy"]==[720,340]
    assert not whole_bay.verify_current(req(words),chosen)["independent_model_proposal"]


@pytest.mark.parametrize("status",["BLOCKED","LANE","SEPARATED","UNCLEAR"])
def test_an_identified_vehicle_cannot_override_a_physical_veto_even_with_other_unseen_details(fixture,status):
    req,photo,surface,receipt,rows=fixture
    opt=rows[0];opt["evidence"]["status"]=status
    words="Drop me on the paved shoulder behind the white sedan near the orange pharmacy on the right"
    a=native.language_outcome(req(words),photo,surface,receipt)
    assert a["result"]["anchor_boxes_xyxy"]==[opt["vehicle_crops"][1][1]]
    assert a["provenance"]["reason"]=="vehicle_seen_space_withheld" and not a["provenance"]["bay_choices"]
    assert not a["provenance"]["bay_outline"] and a["result"]["destination_point_xy"] is None
    assert "pharmacy" not in a["result"]["clarification_question"]
    assert whole_bay.verify_current(req(words),a)["decision_reproduced"]
    assert_unmatched(native.language_outcome(req(words.replace("white sedan","gray SUV")),photo,surface,receipt))
    assert_unmatched(native.language_outcome(req(words.replace("right","left")),photo,surface,receipt))
    with pytest.raises(ValueError):
        whole_bay.choice_outcome(req(words),a,"bay-1")


@pytest.mark.parametrize("change",["image","tracks","words","board","map","point","actor","role"])
def test_replay_rejects_changed_context_observations_input_role_and_display(fixture,change):
    make,photo,surface,obs,_=fixture
    req=make("Drop me on the paved shoulder behind the white sedan on the right.")
    answer=native.language_outcome(req,photo,surface,obs)
    assert whole_bay.verify_current(req,answer)["decision_reproduced"]
    if change=="image":
        req["scene"]["provenance"]["image_sha256"]="0"*64
    elif change=="tracks":
        req["scene"]["tracks"]=[{"recording_id":"native-fixture","frame_id":8,"source":"predicted","class_name":"Car",
                                "bbox_xyxy":[2,5,16,23],"track_id":2}]
    elif change=="words":
        req["text"]=req["text"].replace("white","red")
    elif change=="board":
        answer=deepcopy(answer);answer["provenance"]["native_location_observations"]["test"]="other board"
    elif change=="map":
        answer=deepcopy(answer);answer["provenance"]["surface_evidence"]["test"]="other map"
    elif change=="point":
        answer["result"]["destination_point_xy"]=[500,499]
    elif change=="actor":
        answer["result"]["anchor_boxes_xyxy"]=[[2,5,16,23]]
    else:
        answer["provenance"]["method"]=whole_bay.MARK
    with pytest.raises(ValueError):
        whole_bay.verify_current(req,answer)


def test_exact_manual_mark_replays_against_the_original_surface_and_the_same_action(tmp_path,monkeypatch):
    original,classes,geometry_context=driveway()
    im=original.resize((classes.shape[1],classes.shape[0]),Image.Resampling.NEAREST)
    path=tmp_path/"synthetic-entrance.png"; im.save(path)
    data=path.read_bytes(); record="synthetic-driveway"; frame_id=11
    frame={"width":im.width,"height":im.height,"image_path":str(path),"frame_id":frame_id,"timestamp_s":2.2,
           "recording_id":record}
    tracks=[{"recording_id":record,"frame_id":frame_id,"track_id":key,"class_name":kind,"source":"predicted",
             "bbox_xyxy":[v/2 for v in box]} for key,kind,box in geometry_context["current_vehicles"]]
    scene={"source":"predicted","frame":frame,"provenance":{"image_sha256":sha256(data).hexdigest()},"tracks":tracks}
    req={"request_id":"original-mark","scene":scene,"text":"Drop us in the driveway","input_mode":"typed","dialogue":[]}
    surface=provide_map(monkeypatch,scene,classes)
    [opt]=driveway_apron.candidates(im,classes,native.visual_space.native_context(scene)); x,y=opt["point"]
    x+=2; box=[x-1,y-1,x+1,y+1]
    a=native.marked_outcome(req,data,surface,box)
    assert a["result"]["destination_point_xy"]==[x,y]
    assert a["provenance"]["bay_outline"]["passenger_tap_xy"]==[x,y]
    assert not a["provenance"]["independent_model_proposal"]
    assert whole_bay.verify_current(req,a,requested_box=box)["offlane_mask_replayed"]
    assert native.marked_outcome(req,data,surface,[x-1,y+90,x+1,y+92]) is None
    with pytest.raises(ValueError):
        whole_bay.verify_current(req,a,requested_box=[n+1 for n in box])
    with pytest.raises(ValueError):
        whole_bay.verify_current(req,a)
    tampered=deepcopy(a); tampered["provenance"]["native_mark_geometry"]["cut_cells"]+=1
    with pytest.raises(ValueError):
        whole_bay.verify_current(req,tampered,requested_box=box)
