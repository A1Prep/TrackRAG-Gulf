"""Control tests of the text-only clarification, its provenance and the separate resolved-model turn."""
from copy import deepcopy
import json

import pytest

from tests.test_passenger_between_road import roadside
from tests.test_passenger_parking_session import configured, later
from tools.dropoff_passenger_handoff import bound_outcome, session as export
from trackrag.dropoff import parking_clarification as repair
from trackrag.dropoff import passenger_parking as parking
from trackrag.dropoff.passenger_pair_choice import PassengerSession


UNCLEAR = "Could you show me the gap between the white box truck and the bike on the right?"
CLEAR = "Drop me off between the white box truck and the white van, please."


@pytest.mark.parametrize("words, reason", [
    ("Drop me on the zebra stripes right by the traffic island.", "crossing_target"),
    ("Let us out on that pedestrian crossing.", "crossing_target"),
    ("Can you let me out right behind that green-covered lorry, in its lane?", "lane_target"),
    ("Please stop in the middle of the road.", "lane_target"),
    ("Drop me in the traffic lane between these two cars.", "lane_target"),
])
def test_explicit_crossing_or_lane_asks_for_an_alternative_without_a_pin(roadside, words, reason):
    request = {**roadside[0], "text":words}
    result = repair.clarify(request)
    assert result["provenance"]["reference_evidence"]["reason"] == reason
    assert result["result"]["destination_point_xy"] is None
    assert repair.verify(request,result)["decision_reproduced"]


@pytest.mark.parametrize("words", ["Drop me near the zebra crossing.", "Stop after the marked pedestrian crossing.",
    "Drop me at the bay across the zebra crossing.", "Drop me beside the SUV that is in the traffic lane.",
    "Drop me just before the crossing, not on the zebra stripes.", "What is parked in the bus lane?"])
def test_hazard_can_be_an_orientation_landmark(words):
    assert repair.requested_hazard(words) is None


@pytest.mark.parametrize("words,reason", [
    (UNCLEAR,"cycle"),
    ("Show me the open spot between the e-bike by the curb and the white sedan.","cycle"),
    ("Drop me between the blue bicycle and the black taxi.","cycle"),
    ("Please stop between the grey Otria and the red SUV.","unclear_type"),
    ("On the right, mark the empty space between the taxi and the blue ZR-27.","unclear_type"),
    ("Drop me between the truck and the dark one.","unclear_type"),
    ("Let me out between the white van and...","missing_reference"),
])
def test_explicit_unresolved_type_or_incomplete_request(roadside, words, reason):
    req, *_ = roadside
    req = {**req, "text":words}
    out = repair.clarify(req)
    src = out["provenance"]
    assert out["result"]["status"] == "needs_clarification"
    assert out["result"]["destination_point_xy"] is out["result"]["destination_box_xyxy"] is None
    assert src["reference_evidence"]["reason"] == reason
    assert src["reference_evidence"]["wording"] == words
    for span in src["reference_evidence"]["literal_references"]:
        assert words[span["start"]:span["end"]] == span["text"]
    assert src["native_model_calls"] == 0 and src["independent_model_proposal"] is False
    assert repair.verify(req, out)["decision_reproduced"]


@pytest.mark.parametrize("words", [
    CLEAR, "Drop me off between the silver Nissan and the dark convertible.", "Between the van and the bicycle.",
    "Show me the bike lane between the sedan and the bicycle.", "Drop me beside the blue bicycle.",
    "Can you see a bicycle between the sedan and the red bike?", "Is the SUV beside the bicycle?",
    "Don't stop between the sedan and the bike.", "Drop me behind the gap between the sedan and the bicycle.",
    "Let me out between the sedan and the blue pharmacy.", "Stop between the orange car and the bike rack.",
    "Drop me between the taxi and the rider on the bicycle.", "Drop me between the bicycle and the tree.",
    "Drop me between the car and a red umbrella.", "The bicycle is parked between the taxi and the SUV.",
    "Drop me between the sedan and the white truck near the bikes.", "Stop between the gate and the bus stop.",
    "Let me out between the van and the Mr. Ford dealership.", "Drop me between the car and a blue motor bike.",
])
def test_other_destinations_questions_and_supported_types_stay_on_existing_routes(roadside, words):
    req, *_ = roadside
    assert repair.clarify({**req, "text":words}) is None


def test_repair_rejects_different_words_frames_source_and_omitted_claims(roadside):
    req, *_ = roadside
    req = {**req, "text":UNCLEAR}
    out = repair.clarify(req)
    for key in ("reference_evidence", "native_model_calls", "image", "inference_input_sha256"):
        bad=deepcopy(out)
        del bad["provenance"][key]
        with pytest.raises(ValueError): repair.verify(req,bad)
    bad=deepcopy(out); bad["result"]["destination_point_xy"]=[20,30]
    with pytest.raises(ValueError): repair.verify(req,bad)
    for kind in ("words","stale_track","photo"):
        other=deepcopy(req)
        if kind=="words": other["text"]+=" Please."
        elif kind=="stale_track": other["scene"]["tracks"][0]["frame_id"]-=1
        else: other["scene"]["provenance"]["image_sha256"]="0"*64
        with pytest.raises(ValueError): repair.verify(other,out)
    other=later(req, UNCLEAR); other["dialogue"][0]["frame_id"]-=1
    assert repair.references(other) is None
    annotated=deepcopy(req); annotated["scene"]["source"]="annotated"
    assert repair.clarify(annotated) is None
    with pytest.raises(ValueError):
        bound_outcome(out,req,dialogue_policy=True, passenger_selected=True)


def test_normal_clarification_then_resolved_model_turn_preserves_roles(configured, tmp_path):
    req, model, app, cloud = configured
    # The clarification is local even if the external model is disabled.
    off = PassengerSession(req["scene"],model,audit_dir=tmp_path/"disabled")
    assert off.submit(UNCLEAR)["selected_by"] == repair.ACTOR
    first = app.submit(UNCLEAR)
    assert first["selected_by"]==repair.ACTOR and first["state"]=="needs_clarification" and first["proposal_id"] is None
    assert not cloud.calls
    with pytest.raises(ValueError): app.confirm("0"*32)
    result=app.submit(CLEAR)
    assert result["selected_by"]==parking.ACTOR and len(cloud.calls)==1
    assert parking.parking_instruction(cloud.calls[0])==CLEAR
    assert cloud.calls[0]["dialogue"][0]["text"]==UNCLEAR
    assert app.confirm(result["proposal_id"])["state"]=="confirmed"
    events=export(app._journal)["events"]
    assert [e["phase"] for e in events]==["opened","reference_type_clarification","automatic_parking_proposal","confirmed_previous_selection"]
    assert events[1]["outcome"]["native_model_calls"]==0
    assert events[1]["outcome"]["independent_model_proposal"] is False
    records=[json.loads(line) for line in app._journal.read_bytes().splitlines()]
    records[1]["view"]["selected_by"]="parking_model"
    changed=tmp_path/"changed.jsonl"
    changed.write_text("\n".join(json.dumps(row) for row in records),encoding="utf8")
    with pytest.raises(ValueError,match="who selected"):
        export(changed)
