"""Session, role and replay regressions over a generated road with two real painted dividers."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from test_painted_bays import stall_row
from test_passenger_manual_offlane import provide_map
from tools.dropoff_passenger_handoff import bound_outcome, session as export_journal
from trackrag.dropoff import manual_surfaces, whole_bay
from trackrag.dropoff.passenger import PassengerGrounder
from trackrag.dropoff.passenger_pair_choice import PassengerSession


def test_matching_measured_bays_take_priority_over_a_partial_pavement_patch():
    from trackrag.dropoff import painted_bays, surface_stalls

    def option(x, recipe):
        return {"point": [x+3, 3], "polygon": [[x,0],[x+6,0],[x+6,6],[x,6]], "evidence": {"recipe": recipe}}

    measured = option(20, painted_bays.RECIPE)
    patch = option(42, surface_stalls.RECIPE)
    assert whole_bay._prefer_measured([patch, measured]) == [measured]
    assert whole_bay._prefer_measured([patch]) == [patch]
    assert "extent" not in whole_bay._present([measured], None)[0]
    assert whole_bay._present([patch], None)[0]["extent"] == "visible_parking_surface"


def test_unsupported_anchor_packet_preserves_independent_region_review(monkeypatch, tmp_path):
    from trackrag.dropoff import bay_anchors, surface_stalls

    path = tmp_path / "source.bin"
    path.write_bytes(b"original")
    req = {"text": "Use a parking bay on the right.", "scene": {"frame": {"image_path": str(path)}, "provenance": {"image_sha256": "0"*64}}}
    monkeypatch.setattr(manual_surfaces, "get_packet", lambda key: "surface")
    monkeypatch.setattr(surface_stalls, "get_packet", lambda key: "region review")
    monkeypatch.setattr(bay_anchors, "get_packet", lambda key: "another vehicle set")
    seen = []

    def result(request, photo, surface, region, anchors, *, strict_words):
        assert strict_words
        seen.append((region, anchors))
        if anchors is not None:
            raise ValueError("Changed vehicles")
        return {"provenance": {"reason": "visible_bay_matches_stated_anchors"}}

    monkeypatch.setattr(whole_bay, "explicit_outcome", result)
    assert whole_bay.explicit_request(req)["provenance"]["reason"] == "visible_bay_matches_stated_anchors"
    assert seen == [("region review", "another vehicle set"), ("region review", None)]


@pytest.fixture
def road(tmp_path, monkeypatch):
    image, classes, _ = stall_row()
    path = tmp_path / "curb.png"
    image.save(path)
    frame = {"width":image.width, "height":image.height, "image_path":str(path), "frame_id":12,
             "recording_id":"synthetic-street", "timestamp_s":1.2}
    scene = {"source":"predicted","frame":frame,"provenance":{"image_sha256":sha256(path.read_bytes()).hexdigest()},
             "tracks":[]}
    packet = provide_map(monkeypatch, scene, classes)
    class Model:
        requires_verified_manual_surface = True
        passenger_area = PassengerGrounder.passenger_area

        def interpret(self, request):
            return whole_bay.automatic_outcome(request,path.read_bytes(),packet,None)
    return PassengerSession(scene,Model(),audit_dir=tmp_path/"journal"), packet


@pytest.mark.parametrize("origin", ["typed","mark"])
@pytest.mark.parametrize("choice", ["bay-1","bay-2"])
def test_choose_and_replay_both_sides_of_real_paint(road, origin, choice, monkeypatch):
    dialogue,_ = road
    options = dialogue.submit("Use the marked parking space.",mark_on_photo=origin=="mark")
    if origin=="mark":
        options=dialogue.select_area([166.5,122.5,168.5,124.5])
    assert options["state"]=="needs_clarification" and options["proposal_id"] is None, options.get("error")
    assert len(options["bay_choices"])==2 and not options.get("bay_outline")
    with pytest.raises(ValueError):
        dialogue.select_bay("bay-999")
    assert dialogue.view()==options
    expected=next(c for c in options["bay_choices"] if c["id"]==choice)
    selected=dialogue.select_bay(choice)
    assert selected["state"]=="proposed" and selected["selected_by"]=="passenger"
    assert selected["bay_outline"]==expected and not selected.get("bay_choices")
    assert selected["result"]["destination_point_xy"]==expected["center_xy"]
    assert expected["passenger_tap_xy"]==([167.5,123.5] if origin=="mark" else None)
    # Replay works on a new process without a working installation cache.
    monkeypatch.setattr(manual_surfaces,"get_packet",lambda _:None)
    whole_bay._cache.clear()
    saved=dialogue.confirm(selected["proposal_id"])
    assert saved["bay_outline"]==expected and saved["state"]=="confirmed"
    with pytest.raises(ValueError):
        dialogue.select_bay(choice)
    events=export_journal(dialogue._journal)["events"]
    assert events[-2]["phase"]=="passenger_selected_bay"
    assert events[-2]["outcome"]["passenger_mark_verification"]["decision_reproduced"]
    assert events[-2]["outcome"]["independent_model_proposal"] is False
    assert events[-1]["verified_manual_outcome_sha256"]


@pytest.mark.parametrize("change",["bay_id","origin","paint","map","display","role"])
def test_changed_bay_cannot_be_exported_as_selected(road,change):
    dialogue,_=road
    dialogue.submit("Use a parking bay.",mark_on_photo=True)
    dialogue.select_area([166.5,122.5,168.5,124.5])
    dialogue.select_bay("bay-2")
    rows=[json.loads(line) for line in dialogue._journal.read_bytes().splitlines()]
    picked=rows[-1]["detail"]
    source=picked["outcome"]["provenance"]
    if change=="bay_id": picked["bay_id"]="bay-1"
    if change=="origin": picked["prior_outcome_sha256"]="0"*64
    if change=="paint": source["choice_source"]["provenance"]["selected_box"]=[100,100,102,102]
    if change=="map": source["choice_source"]["provenance"]["surface_evidence"]["signature"]="0"*128
    if change=="display": rows[-1]["view"]["bay_outline"]["polygon_xy"][0][1]+=2
    if change=="role": rows[-1]["view"]["selected_by"]="model"
    alt=dialogue._journal.with_suffix(".changed")
    alt.write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    with pytest.raises(ValueError):
        export_journal(alt)


def test_center_road_never_gets_bay_and_interior_keeps_visible_footprint(road):
    dialogue,_=road
    dialogue.submit("Use this bay.",mark_on_photo=True)
    lane=dialogue.select_area([16,175,19,178])
    assert lane["state"]=="needs_clarification" and not lane.get("bay_choices") and not lane.get("bay_outline")
    assert lane["result"]["destination_point_xy"] is None and lane["proposal_id"] is None
    proposal=dialogue.select_area([126,126,128,128])
    assert proposal["state"]=="proposed" and len(proposal["bay_outline"]["polygon_xy"])>=4
    assert proposal["bay_outline"]["passenger_tap_xy"]==[127,127]
    assert proposal["result"]["destination_box_xyxy"] != [126,126,128,128]
    evt=export_journal(dialogue._journal)["events"][-1]["outcome"]
    assert evt["passenger_mark_verification"]["offlane_mask_replayed"]
    with pytest.raises(ValueError):
        bound_outcome(dialogue._manual_offer[1],dialogue._request,dialogue_policy=True)


def test_language_cannot_turn_a_missing_name_or_an_unspecified_side_into_a_bay(road):
    dialogue,packet=road
    dialogue.submit("Use a parking bay.")
    req=deepcopy(dialogue._request)
    _,_,_,options=whole_bay._layout(req,Path(req["scene"]["frame"]["image_path"]).read_bytes(),packet)
    one,other=options
    req["text"]="At the restaurant on the right."
    assert not whole_bay.language_candidates(req,options,["cafe","school","shop"])
    req["text"]="At the restaurant."
    assert whole_bay.language_candidates(req,options,["home","PIQUANT PICK restaurant","shop"])
    req["text"]="At the bay opposite the restaurant."
    assert not whole_bay.language_candidates(req,[one,other],["restaurant"]*3)
