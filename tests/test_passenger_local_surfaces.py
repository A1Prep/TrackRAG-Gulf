from copy import deepcopy
import json

import pytest

from trackrag.dropoff import local_surfaces, manual_surfaces
from trackrag.dropoff.parking_evidence import ModalParkingBackend
from trackrag.dropoff.photo_clarification import canopy_boxes, vegetation_reference, visible_category
from trackrag.dropoff.passenger_parking_session import ParkingPassengerSession
from trackrag.dropoff.passenger_pair_choice import PassengerSession
from trackrag.dropoff.place_evidence import PublicPlaceBackend


def registered_scene():
    key, row = next(iter(local_surfaces._sources().items()))
    width, height = row["size"]
    return {"source":"predicted", "provenance":{"image_sha256":key,"network_policy":"local_only"},
            "frame":{"width":width,"height":height,"recording_id":row["recording_id"],"frame_id":row["frame_id"]}}


def test_locally_generated_surfaces_are_bound_to_original_frame_source_and_model():
    index=json.loads(local_surfaces.BANK.with_suffix(".json").read_bytes())
    assert len(index["sources"]) == 6
    for key,row in index["sources"].items():
        frame={"width":row["size"][0],"height":row["size"][1],
               "recording_id":row["recording_id"],"frame_id":row["frame_id"]}
        packet=manual_surfaces.get_packet(key)
        assert packet["origin"] == "local_release_v1"
        assert manual_surfaces.decode(packet,frame,key).max() <= 64
        broken=deepcopy(packet)
        broken["body"]["class_map"]["sha256"]="0"*64
        with pytest.raises(ValueError):
            manual_surfaces.decode(broken,frame,key)
        with pytest.raises(ValueError):
            manual_surfaces.decode(packet,{**frame,"recording_id":"another_drive"},key)
        with pytest.raises(ValueError):
            manual_surfaces.decode(packet,frame,"f"*64)


def test_local_only_tag_prevents_sharing_even_when_an_operator_adds_the_same_public_entry(tmp_path):
    scene=registered_scene()
    key=scene["provenance"]["image_sha256"]
    path=tmp_path/"sources.json"
    path.write_text(json.dumps({"photos":[{"sha256":key,"photo_url":local_surfaces._sources()[key]["source_url"]}]}))
    backend=PublicPlaceBackend([path])
    assert not backend.allows_photo({"scene":scene})
    assert not backend.allows_photo({"scene":{**scene,"provenance":{"image_sha256":key}}})
    assert local_surfaces.local_only({"provenance":{"network_policy":"local_only","image_sha256":"e"*64}})
    assert not local_surfaces.matches_scene({**scene,"frame":{**scene["frame"],"frame_id":-1}})


def test_replacing_the_local_index_and_map_cannot_authorize_new_pixels(tmp_path,monkeypatch):
    scene=registered_scene()
    path=tmp_path/"surfaces.zip"
    forged=json.loads(local_surfaces.BANK.with_suffix(".json").read_bytes())
    forged["sources"][scene["provenance"]["image_sha256"]]["body_sha256"]="e"*64
    path.with_suffix(".json").write_text(json.dumps(forged))
    with monkeypatch.context() as changes:
        changes.setattr(local_surfaces,"BANK",path)
        local_surfaces._sources.cache_clear()
        assert not local_surfaces.matches_scene(scene)
        assert local_surfaces.local_only(scene)
    local_surfaces._sources.cache_clear()


def test_local_two_vehicle_request_never_dispatches_to_the_parking_service(monkeypatch):
    request={"scene":registered_scene(),"text":"Drop me between the black SUV and the white van."}
    local_result=object()
    monkeypatch.setattr(PassengerSession,"interpret_automatic",lambda self, req: local_result)
    session=object.__new__(ParkingPassengerSession)
    assert session.interpret_automatic(request) is local_result
    with pytest.raises(ValueError,match="local processing only"):
        ModalParkingBackend().interpret(request)


def test_reference_clarification_requires_all_named_details_on_the_requested_side():
    labels=["pharmacy","trees","home restaurant"]
    assert visible_category("Could you drop me next to the restaurant on the right?",labels)=="the restaurant"
    assert visible_category("Could you drop me next to the pharmacy on the right?",labels) is None
    assert visible_category("Drop me outside Moonlight Restaurant.",labels) is None
    assert visible_category("Drop me between the restaurant and the pharmacy.",labels) is None


def test_tall_greenery_can_be_identified_on_either_side_without_marking_the_road():
    import numpy as np
    labels=np.full((100,160),13,dtype="uint8")
    labels[15:60,5:42]=30
    labels[6:58,119:158]=30
    labels[78:86,54:94]=30
    frame={"width":1600,"height":1000}
    assert canopy_boxes(labels,frame,"left")==[("left",[50,150,420,600])]
    assert [s for s,_ in canopy_boxes(labels,frame,"either")]==["left","right"]
    assert vegetation_reference("Drop me next to the trees on the right.")=="right"
    assert vegetation_reference("Could you drop me behind the black car near the trees?") is None
    assert vegetation_reference("Drop me by the tall red tree beside the hotel.") is None
