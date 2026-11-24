"""A nearby passenger destination must adjoin the identified current vehicle."""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from test_passenger_bay_evidence import bank, seal
from trackrag.dropoff import near_vehicle, painted_bays, place_evidence, vehicle_curb_ground
from trackrag.dropoff import vehicle_ground_reference as reference, vehicle_ground_view as view, whole_bay
from trackrag.dropoff.interaction import PassengerSession


WORDS = "Do you drop me off near the black car?"


@pytest.mark.parametrize("words,kind,colour,side", [
    (WORDS, "car", "black", None),
    ("Could you let me out by the silver sedan parked on the right?", "sedan", "silver", "right"),
    ("Can I get out near the grey SUV on the right?", "suv", "gray", "right"),
    ("Please pull up close to the white van on our left.", "van", "white", "left"),
    ("Near the bus", "bus", None, None),
])
def test_conversational_dropoffs_keep_type_paint_side_and_near_relation(words, kind, colour, side):
    ref = reference.direct_reference({"text": words})
    assert (ref["relation"], ref["kind"], ref["color"], ref["side"]) == ("near", kind, colour, side)


@pytest.mark.parametrize("words", [
    "Do not drop me near the black car.",
    "Drop me near either the black car or the van.",
    "Is the man by the black car?",
    "Drop me by the car wash.",
    "Let me out near the card reader.",
    "Pull up by the red sedan outside the school.",
    "Drop me near the SUV with its door open.",
    "Drop me near the blue and red car.",
    "Near the red car if parking is free.",
    "Stop in the traffic lane near the black car.",
])
def test_unrepresented_conditions_are_never_removed_to_make_a_near_vehicle_match(words):
    assert near_vehicle.reference(words) is None


def painted_stall():
    return {"point":[92,75], "box":[60,60,140,88],
            "polygon":[[60,60], [140,60], [140,88], [60,88]],
            "evidence":{"recipe":painted_bays.RECIPE,
                        "painted_boundaries":[[[60,60],[140,60]], [[60,88],[140,88]]]}}


def test_painted_divider_requires_immediate_opposite_tyre_and_an_unobstructed_parking_surface():
    bay = painted_stall()
    anchor = [60,15,110,58]
    car = {"frame_id":3, "track_id":1, "class_name":"car", "bbox_xyxy":anchor}
    scene = {"frame":{"frame_id":3, "width":200, "height":140}, "tracks":[car]}
    classes = np.full((140,200), 10, np.uint8)
    component = np.ones_like(classes, dtype=bool)
    assert near_vehicle._adjacent_to_divider(bay, anchor, scene, classes)
    assert near_vehicle._open_point(bay, scene, classes, component)
    for wrong_anchor in ([60,5,110,32], [140,15,190,58], [60,21,110,64]):
        assert not near_vehicle._adjacent_to_divider(bay, wrong_anchor, scene, classes)
    assert not near_vehicle._adjacent_to_divider({**bay, "evidence":{"recipe":"loose-surface"}}, anchor, scene, classes)
    for intervening in (13,15,23):
        blocked = classes.copy()
        blocked[65:70] = intervening
        assert not near_vehicle._adjacent_to_divider(bay, anchor, scene, blocked)
    object_in_front = {**car, "track_id":2, "bbox_xyxy":[57,63,143,80]}
    current = {**scene, "tracks":[car, object_in_front]}
    assert not near_vehicle._adjacent_to_divider(bay, anchor, current, classes)
    assert not near_vehicle._open_point(bay, current, classes, component)
    earlier = {**scene, "tracks":[car, {**object_in_front, "frame_id":2}]}
    assert near_vehicle._adjacent_to_divider(bay, anchor, earlier, classes)
    assert near_vehicle._open_point(bay, earlier, classes, component)
    component[75,92] = False
    assert not near_vehicle._open_point(bay, scene, classes, component)
    classes[75,92] = 13
    assert not near_vehicle._open_point(bay, scene, classes, np.ones_like(component))


def identity_packet(bank, second=("white","van")):
    specs = vehicle_curb_ground.describe(bank.scene, bank.classes)
    body = {"source":bank.scene["provenance"]["image_sha256"], "recipe":view.RECIPE,
            "model":reference.MODEL, "revision":reference.REVISION, "prompt":view.checksum(view.GUIDE),
            "spec":view.checksum(specs), "view":sha256(view.render(bank.image, specs)).hexdigest(),
            "count":len(specs), "answers":[["black","sedan","REAR","LANE"] if s["track_id"] == 4
                                         else [*second,"SIDE","LANE"] for s in specs]}
    return seal(body, bank.key)


@pytest.mark.parametrize("number", [1,2])
def test_session_keeps_unique_vehicle_outlined_through_explicit_choice_and_save(bank, tmp_path, monkeypatch, number):
    packet = identity_packet(bank)
    def parking(request, photo, surface, classes, spec):
        assert spec["track_id"] == 4
        return bank.options[:number]
    monkeypatch.setattr(near_vehicle, "parking", parking)
    def interpret(request):
        return reference.outcome(request, bank.photo.read_bytes(), bank.surface, packet, nearby_parking=True)
    session = PassengerSession(bank.scene, SimpleNamespace(interpret=interpret), audit_dir=tmp_path/"near")
    first = session.submit(WORDS)
    expected_anchor = [[42.,100.,60.,126.]]
    assert first["result"]["anchor_boxes_xyxy"] == expected_anchor
    if number == 2:
        assert first["state"]=="needs_clarification" and not first["proposal_id"] and first["vehicle_ground_choice"]
        assert len(first["bay_choices"])==2
        selected = session.select_bay(first["bay_choices"][1]["id"])
        assert selected["selected_by"]=="passenger"
    else:
        selected = first
        assert selected["selected_by"]=="model"
    assert selected["state"]=="proposed" and selected["near_vehicle_parking"]
    assert selected["result"]["anchor_boxes_xyxy"]==expected_anchor
    saved = session.confirm(selected["proposal_id"])
    assert saved["state"]=="confirmed" and saved["near_vehicle_parking"]
    assert saved["bay_outline"]==selected["bay_outline"] and saved["result"]["anchor_boxes_xyxy"]==expected_anchor
    session.cancel()
    assert "near_vehicle_parking" not in session.view()


@pytest.mark.parametrize("second,words,anchors", [
    (("white","van"), "Drop me by the blue sedan", 0),
    (("black","sedan"), WORDS, 2),
    (("unknown","suv"), WORDS, 2),
    (("black","unknown"), WORDS, 2),
])
def test_contradiction_or_ambiguous_vehicle_never_reaches_parking_search(bank, monkeypatch, second, words, anchors):
    packet = identity_packet(bank, second)
    def unexpected(*args):
        pytest.fail("Parking cannot be attached to an unresolved vehicle")
    monkeypatch.setattr(near_vehicle,"parking",unexpected)
    request = {**bank.req, "text":words}
    answer = reference.outcome(request, bank.photo.read_bytes(), bank.surface, packet, nearby_parking=True)
    assert answer["result"]["status"]=="needs_clarification"
    assert len(answer["result"].get("anchor_boxes_xyxy", []))==anchors and answer["result"]["destination_point_xy"] is None
    assert not answer["provenance"]["bay_choices"] and whole_bay.verify_current(request, answer)["decision_reproduced"]


def test_replay_binds_explicitly_selected_parking_and_vehicle_to_original_words_and_source(bank, monkeypatch):
    packet = identity_packet(bank)
    photo = bank.photo.read_bytes()
    request = {**bank.req, "text":WORDS}
    monkeypatch.setattr(near_vehicle, "parking", lambda *args: bank.options)
    assert reference.outcome(request,photo,bank.surface,packet) is None
    offered = reference.outcome(request,photo,bank.surface,packet,nearby_parking=True)
    selected = whole_bay.choice_outcome(request,offered,"bay-2")
    assert whole_bay.verify_current(request,selected)["decision_reproduced"]
    for field in ("destination_point_xy","anchor_boxes_xyxy"):
        tampered = deepcopy(selected)
        tampered["result"][field] = [12,12] if field=="destination_point_xy" else [[4,4,12,12]]
        with pytest.raises(ValueError):
            whole_bay.verify_current(request,tampered)
    for changed in ({**request,"text":"By the white van."},
                    {**request,"scene":{**bank.scene,"provenance":{"image_sha256":"1"*64}}}):
        with pytest.raises(ValueError):
            whole_bay.verify_current(changed,selected)


SIDE_WORDS = 'To the right side of it, please.'


def test_beside_reply_keeps_the_same_car_and_checks_only_the_requested_side(bank, tmp_path, monkeypatch):
    packet = identity_packet(bank)
    photo = bank.photo.read_bytes()
    left = {'point': [28, 131], 'box': [23, 128, 33, 134],
            'polygon': [[23, 128], [33, 128], [33, 134], [23, 134]],
            'evidence': {'recipe': painted_bays.RECIPE, 'coverage': 'interior_point_only'}}
    right = bank.options[0]
    assert right['point'][0] > 60
    captured = {}

    def parking(request, photo, surface, classes, spec):
        assert spec['track_id'] == 4
        return [left, right]

    def interpret(request):
        outcome = reference.outcome(request, photo, bank.surface, packet, side_parking=True)
        captured[request['text']] = (request, outcome)
        return outcome

    monkeypatch.setattr(near_vehicle, 'parking', parking)
    session = PassengerSession(bank.scene, SimpleNamespace(interpret=interpret), audit_dir=tmp_path/'side')
    initial = session.submit('Can you drop me next to the black car?')
    assert initial['state'] == 'needs_clarification'
    assert reference.SIDE_QUESTION in initial['result']['clarification_question']

    offered = session.submit(SIDE_WORDS)
    assert offered['state'] == 'proposed' and offered['near_vehicle_parking']
    assert offered['result']['anchor_boxes_xyxy'] == initial['result']['anchor_boxes_xyxy']
    assert offered['result']['destination_point_xy'] == right['point']
    assert session.confirm(offered['proposal_id'])['state'] == 'confirmed'

    request, outcome = captured[SIDE_WORDS]
    assert whole_bay.verify_current(request, outcome)['decision_reproduced']
    for change in ('direction', 'earlier_frame', 'other_prompt', 'condition', 'other_car'):
        edited = deepcopy(request)
        if change == 'direction':
            edited['text'] = 'To the left side of it, please.'
        elif change == 'earlier_frame':
            edited['dialogue'][-2]['frame_id'] -= 1
        elif change == 'other_prompt':
            edited['dialogue'][-1]['text'] = 'Is that the vehicle you meant?'
        elif change == 'condition':
            edited['dialogue'][-2]['text'] += ' Only if parking is free.'
        else:
            edited['dialogue'][-2]['text'] = 'Can you drop me next to the red car?'
        with pytest.raises(ValueError):
            whole_bay.verify_current(edited, outcome)


def test_beside_request_with_only_wrong_side_parking_keeps_the_reference_without_a_point(bank, monkeypatch):
    packet = identity_packet(bank)
    request = {**bank.req, 'text': 'Drop me to the right of the black car.'}
    monkeypatch.setattr(near_vehicle, 'parking', lambda *args: [{'point': [21, 130]}])
    result = reference.outcome(request, bank.photo.read_bytes(), bank.surface, packet, side_parking=True)
    assert result['result']['status'] == 'needs_clarification'
    assert result['result']['destination_point_xy'] is None
    assert result['result']['anchor_boxes_xyxy'] == [[42.,100.,60.,126.]]
    assert not result['provenance']['bay_choices']
    assert whole_bay.verify_current(request, result)['decision_reproduced']


def test_regular_registered_route_cannot_fall_through_to_slow_language_or_generic_bays(bank, monkeypatch):
    backend = object.__new__(place_evidence.PublicPlaceBackend)
    monkeypatch.setattr(backend,"allows_photo",lambda request: True)
    model = object.__new__(place_evidence.PlaceDialogueGrounder)
    model.backend = backend
    model.dialogue = SimpleNamespace(interpret=lambda _: pytest.fail("Avoid unrelated native language inference"))
    monkeypatch.setattr(model,"inspect_requested_bay",lambda _: pytest.fail("Avoid unrelated parking choices"))
    request = {**bank.req,"text":WORDS}
    monkeypatch.setattr(reference,"available",lambda _: None)
    assert model.uses_public_service(request)
    missing = model.interpret(request)
    assert missing["result"]["status"]=="needs_clarification" and not missing["result"]["destination_point_xy"]
    assert missing["provenance"]["method"]==reference.NO_READING


def test_nearby_row_alternative_needs_uninterrupted_parking_and_a_bounded_distance():
    anchor = [60,15,110,58]
    scene = {"frame":{"frame_id":3,"width":300,"height":150},"tracks":[]}
    classes = np.full((150,300),10,np.uint8)
    candidate = {**painted_stall(),"point":[125,82]}
    assert near_vehicle._continuous_parking_to_row(candidate,anchor,scene,classes)
    for barrier in (2, 13, 15, 23):
        interrupted = classes.copy()
        interrupted[64:70] = barrier
        assert not near_vehicle._continuous_parking_to_row(candidate,anchor,scene,interrupted)
    obstructed = {**scene,"tracks":[{"frame_id":3,"bbox_xyxy":[70,63,133,77]}]}
    assert not near_vehicle._continuous_parking_to_row(candidate,anchor,obstructed,classes)
    assert not near_vehicle._continuous_parking_to_row({**candidate,"point":[240,90]},anchor,scene,classes)


def test_side_and_neutral_colour_clarifications_never_substitute_an_opposite_or_competing_vehicle():
    reference_words = near_vehicle.reference("Stop by the grey SUV on the right.")
    right = {"track_id":1,"box":[67,20,85,42],"class":"car"}
    middle = {"track_id":2,"box":[39,20,64,45],"class":"car"}
    opposite = {"track_id":3,"box":[5,20,27,45],"class":"car"}
    silver = ["silver","suv","REAR","NONE"]
    gray = ["gray","suv","REAR","NONE"]
    assert "silver" in near_vehicle.conditional_vehicle([right],[silver],reference_words,100)[2]
    assert "centre" in near_vehicle.conditional_vehicle([middle],[gray],reference_words,100)[2]
    assert near_vehicle.conditional_vehicle([opposite],[gray],reference_words,100) is None
    assert near_vehicle.conditional_vehicle([middle],[silver],reference_words,100) is None
    assert near_vehicle.conditional_vehicle([right],[["blue",*silver[1:]]],reference_words,100) is None
    assert near_vehicle.conditional_vehicle([right],[["silver","sedan",*silver[2:]]],reference_words,100) is None
    assert near_vehicle.conditional_vehicle([middle,right],[gray,silver],reference_words,100) is None


def test_signed_neutral_paint_alternative_carries_qualification_through_explicit_selection_and_save(bank,tmp_path,monkeypatch):
    packet = identity_packet(bank,("silver","suv"))
    words = "Can I get out near the grey SUV on the right?"
    monkeypatch.setattr(near_vehicle,"parking",lambda *args: [])
    monkeypatch.setattr(near_vehicle,"optical_parking",lambda *args: (bank.options[:1],True))
    def interpret(request):
        return reference.outcome(request,bank.photo.read_bytes(),bank.surface,packet,nearby_parking=True)
    session = PassengerSession(bank.scene,SimpleNamespace(interpret=interpret),audit_dir=tmp_path/"conditional")
    offer = session.submit(words)
    assert offer["state"]=="needs_clarification" and not offer["proposal_id"]
    assert len(offer["result"]["anchor_boxes_xyxy"])==1 and len(offer["bay_choices"])==1
    detail=offer["near_vehicle_detail"]
    assert "silver" in detail and "grey" in detail and "immediately adjoining" in detail
    chosen=session.select_bay(offer["bay_choices"][0]["id"])
    assert chosen["near_vehicle_detail"]==detail and chosen["near_vehicle_parking"]
    assert session.confirm(chosen["proposal_id"])["near_vehicle_detail"]==detail


def test_no_verified_ground_still_allows_a_separate_explicit_vehicle_confirmation(bank,tmp_path,monkeypatch):
    packet = identity_packet(bank,("silver","suv"))
    def interpret(request):
        assert request["text"]!="Yes"
        return reference.outcome(request,bank.photo.read_bytes(),bank.surface,packet,nearby_parking=True)
    monkeypatch.setattr(near_vehicle,"parking",lambda *args: [])
    monkeypatch.setattr(near_vehicle,"optical_parking",lambda *args: ([],False))
    session=PassengerSession(bank.scene,SimpleNamespace(interpret=interpret),audit_dir=tmp_path/"no-ground")
    shown=session.submit("Drop me near the grey SUV on the right.")
    assert shown["reference_kind"]=="possible_vehicle"
    assert len(shown["result"]["anchor_boxes_xyxy"])==1 and not shown.get("bay_choices")
    confirmed=session.submit("Yes")
    assert confirmed["result"]["anchor_boxes_xyxy"]==shown["result"]["anchor_boxes_xyxy"]
    assert confirmed["references_only"] and not confirmed["proposal_id"] and not confirmed.get("bay_outline")
    assert "near it" in confirmed["result"]["clarification_question"]


def test_archived_near_receipt_replays_without_using_current_policy_or_new_approval(bank,monkeypatch):
    request = {**bank.req, "text":WORDS}
    photo = bank.photo.read_bytes()
    packet = identity_packet(bank,("unknown","suv"))
    monkeypatch.setattr(near_vehicle,"parking",lambda *args: bank.options[:1])
    old = reference.outcome(request,photo,bank.surface,packet,nearby_parking=near_vehicle.ORIGINAL_POLICY)
    assert old["result"]["status"]=="proposed" and whole_bay.verify(request,old)["decision_reproduced"]
    with pytest.raises(ValueError,match="again before saving"):
        whole_bay.verify_current(request,old)
    current = reference.outcome(request,photo,bank.surface,packet,nearby_parking=True)
    assert current["result"]["status"]=="needs_clarification" and not current["provenance"]["bay_choices"]


def test_native_parking_cache_rejects_replaced_scene_surface_and_words(bank):
    photo = bank.photo.read_bytes()
    request = {**bank.req, "text":WORDS}
    original = identity_packet(bank)
    whole_bay._cache.clear()
    cold = reference.outcome(request,photo,bank.surface,original,nearby_parking=True)
    assert cold == reference.outcome(request,photo,bank.surface,original,nearby_parking=True)
    changed = deepcopy(request)
    changed["scene"]["tracks"][1]["bbox_xyxy"][0] += 1
    with pytest.raises(ValueError):
        reference.outcome(changed,photo,bank.surface,original,nearby_parking=True)
    corrupted = deepcopy(bank.surface)
    corrupted["signature"] = "0"*128
    with pytest.raises(ValueError):
        reference.outcome(request,photo,corrupted,original,nearby_parking=True)
    unrelated = {**request, "text":"Pull over between the two cars."}
    assert reference.outcome(unrelated,photo,bank.surface,original,nearby_parking=True) is None
    assert reference.direct_reference({**request, "text":"Drop me behind the black car."})["relation"]=="behind"
