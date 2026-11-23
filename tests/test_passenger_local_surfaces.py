from copy import deepcopy
import json

import pytest

from trackrag.dropoff import local_surfaces, manual_surfaces
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


def test_local_two_vehicle_request_never_dispatches_to_the_parking_service(monkeypatch):
    request={"scene":registered_scene(),"text":"Drop me between the black SUV and the white van."}
    local_result=object()
    monkeypatch.setattr(PassengerSession,"interpret_automatic",lambda self, req: local_result)
    session=object.__new__(ParkingPassengerSession)
    assert session.interpret_automatic(request) is local_result
