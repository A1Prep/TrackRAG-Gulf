from copy import deepcopy
from functools import partial
from hashlib import sha256
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image

from trackrag import passenger_app
from trackrag.dropoff import bay_anchors, bay_intent, bay_landmarks, parking_evidence, whole_bay
from trackrag.dropoff.frontage_evidence import PublicFrontageGrounder
from trackrag.dropoff.passenger import digest
from trackrag.dropoff.passenger_pair_choice import PassengerSession
from trackrag.dropoff.passenger_parking_session import ParkingPassengerSession
from trackrag.dropoff.place_evidence import PlaceDialogueGrounder, PublicPlaceBackend


@pytest.fixture
def catalog_setup(tmp_path,monkeypatch):
    image_path=tmp_path/"road.png"
    Image.new("RGB",(96,64),"#727270").save(image_path)
    sha=sha256(image_path.read_bytes()).hexdigest()
    frame={"width":96,"height":64,"image_path":str(image_path),"recording_id":"kartaview_49","frame_id":7,"timestamp_s":0.}
    scene={"source":"predicted","frame":frame,"provenance":{"image_sha256":sha},"tracks":[
        {"track_id":34,"class_name":"Car","frame_id":7,"bbox_xyxy":[8,17,38,41]}]}
    manifest=tmp_path/"public.json"
    manifest.write_text(json.dumps({"photos":[{"sha256":sha,"photo_url":"https://kartaview.org/details/49/7/"}]}))
    key=Ed25519PrivateKey.generate()
    monkeypatch.setattr(parking_evidence,"PUBLIC_KEY",key.public_key().public_bytes_raw().hex())
    body={"recipe":bay_anchors.RECIPE,"model":bay_landmarks.MODELS,"prompt":bay_anchors.prompt_digest(),
          "image_sha256":sha,"size":[96,64],"current_vehicles_sha256":digest(bay_anchors.vehicles(scene)),
          "rows":[{"shape":"a"*64,"view":"b"*64,"frontage":"sensitive fixture label","pair_reply":"NONE"}],"appearance":{}}
    def sign(body):
        return {"body":body,"signature":key.sign(parking_evidence.signed_bytes(body)).hex()}
    packet=sign(body)
    monkeypatch.setattr(bay_anchors,"get_packet",lambda _sha:packet)
    monkeypatch.setattr(whole_bay,"index_has_outline",lambda scene,photo,shapes: shapes == {"a"*64})
    backend=PublicPlaceBackend([manifest],python="unavailable-test-executable")
    grounder=PlaceDialogueGrounder(object(),backend)
    model=PublicFrontageGrounder(grounder,backend,[manifest],tmp_path/"frontage")
    different=deepcopy(scene);different["frame"]["recording_id"]="kartaview_44"
    # Deliberately reverse display order to check capability never influences sorting or keys.
    service=passenger_app.PassengerSite({"qualified":scene,"other":different},model,None,tmp_path/"logs",
                                partial(ParkingPassengerSession,parking_backend=None))
    return service,scene,packet,sign


def test_signed_outline_marker_preserves_all_frames_and_hides_private_details(catalog_setup):
    site,scene,_,_=catalog_setup
    frames=site.catalog()["scenes"]
    assert [row["key"] for row in frames]==["other","qualified"]
    assert "parking_areas_indexed" not in frames[0] and frames[1]["parking_areas_indexed"] is True
    assert not {"rows","image_path","sha256","shape","frontage","pair_reply"} & set(frames[1])
    assert "sensitive fixture label" not in json.dumps(frames)
    # New detections in a different frame do not alter the authenticated current observation.
    later={**scene["tracks"][0],"frame_id":8};scene["tracks"].append(later)
    assert site.catalog()["scenes"][1]["parking_areas_indexed"] is True
    scene["tracks"][0]["bbox_xyxy"][0]+=1
    assert "parking_areas_indexed" not in site.catalog()["scenes"][1]
    intent=bay_intent.parse("Find the marked parking bays.")
    assert intent and not intent.uncertain and not intent.blocked


def test_index_with_only_occupied_fragments_does_not_offer_shortcut(catalog_setup,monkeypatch):
    site, _, _, _ = catalog_setup
    monkeypatch.setattr(whole_bay, "index_has_outline", lambda scene, photo, shapes: False)
    assert all("parking_areas_indexed" not in row for row in site.catalog()["scenes"])


@pytest.mark.parametrize("field",["recipe","model","prompt","image_sha256","size","current_vehicles_sha256","rows","signature"])
def test_invalid_or_empty_receipt_cannot_advertise_parking(catalog_setup,monkeypatch,field):
    site,_,packet,sign=catalog_setup
    if field=="signature":
        invalid={**packet,"signature":"00"*64}
    else:
        body=deepcopy(packet["body"])
        body[field]=[] if field in {"rows","size"} else "unexpected"
        invalid=sign(body)
    monkeypatch.setattr(bay_anchors,"get_packet",lambda _sha:invalid)
    assert not any(s.get("parking_areas_indexed") for s in site.catalog()["scenes"])


def test_original_or_normal_grounder_is_required(catalog_setup,tmp_path):
    site,scene,_,_=catalog_setup
    unsupported=passenger_app.PassengerSite({"frame":scene},object(),None,tmp_path/"other",PassengerSession)
    assert not unsupported.catalog()["scenes"][0].get("parking_areas_indexed")
    disabled=passenger_app.PassengerSite({"frame":scene},site.model,None,tmp_path/"other",lambda *args:None)
    assert not disabled.catalog()["scenes"][0].get("parking_areas_indexed")
    Image.new("RGB",(96,64),"red").save(scene["frame"]["image_path"])
    assert not any(s.get("parking_areas_indexed") for s in site.catalog()["scenes"])
