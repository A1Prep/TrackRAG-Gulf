"""Check real cryptographic replay over small, explicitly synthetic parking maps."""
import base64
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image
import pytest

from trackrag.dropoff import manual_offlane, manual_surfaces, parking_evidence
from trackrag.dropoff.passenger import PassengerGrounder


def provide_map(monkeypatch, scene, classes):
    """Sign a synthetic map with a fixture-only key isolated to the manual-map reader."""
    frame = scene["frame"]
    data = BytesIO()
    Image.fromarray(classes.astype("uint8")).save(data,"PNG")
    payload = data.getvalue()
    key = Ed25519PrivateKey.generate()
    body = {"recipe":manual_surfaces.RECIPE,"model_revision":manual_surfaces.REVISION,
            "size":[frame["width"],frame["height"]],"image_sha256":scene["provenance"]["image_sha256"],
            "class_map":{"sha256":sha256(payload).hexdigest(),"png":base64.b64encode(payload).decode("ascii")}}
    packet = {"body":body,"signature":key.sign(parking_evidence.signed_bytes(body)).hex()}
    def authenticate(proof):
        from cryptography.exceptions import InvalidSignature
        try:
            key.public_key().verify(bytes.fromhex(proof["signature"]),parking_evidence.signed_bytes(proof["body"]))
        except InvalidSignature as exc:
            raise ValueError("Changed software-only surface signature") from exc
    monkeypatch.setattr(manual_surfaces,"_authenticate",authenticate)
    monkeypatch.setattr(manual_surfaces,"get_packet",lambda digest:packet if digest==body["image_sha256"] else None)
    return packet


def provide_bay(monkeypatch, scene, box):
    """Put one synthetic parking bay in an otherwise road-labelled, original-sized fixture."""
    frame=scene["frame"]
    shape=parking_evidence.map_shape(frame)
    assert shape==(frame["height"],frame["width"])
    classes=np.full(shape,13,dtype=np.uint8)
    x0,y0,x1,y1=box
    classes[y0:y1,x0:x1]=10
    return provide_map(monkeypatch,scene,classes)


@pytest.fixture
def marked_scene(tmp_path,monkeypatch):
    path = tmp_path/"original.png"
    Image.new("RGB",(320,180),"white").save(path)
    frame = {"width":320,"height":180,"recording_id":"test_scene","frame_id":9,"timestamp_s":.9,
             "image_path":str(path)}
    obs = {"recording_id":frame["recording_id"],"frame_id":9,"track_id":3,"class_name":"Car",
           "bbox_xyxy":[220,95,255,129],"source":"predicted"}
    past = {**obs,"track_id":4,"frame_id":8,"bbox_xyxy":[170,112,194,133]}
    scene = {"source":"predicted","frame":frame,"provenance":{"image_sha256":sha256(path.read_bytes()).hexdigest()},
             "tracks":[past,obs]}
    request = {"request_id":"manual-check","text":"By the shop.","input_mode":"typed","dialogue":[],"scene":scene}
    classes = np.full((180,320),13,dtype=np.uint8)
    classes[85:151,168:284] = 10
    classes[158,177] = 10  # An isolated mislabeled pixel in the through lane.
    classes[88:93,188:197] = 15
    classes[41:51,58:69] = 52
    packet = provide_map(monkeypatch,scene,classes)
    return request,packet


@pytest.mark.parametrize(("mark","reason"),[
    ([98,110,103,115],"through_lane"),
    ([224,101,232,110],"observed_object"),
    ([200,92,226,101],"observed_object"),  # The centre is free, but this box includes part of a car.
    ([149,120,180,131],"through_lane"),  # A prior road-only check could move the pin to its far edge.
    ([162,115,190,126],"mixed_area"),   # The centre is parking, part of the mark is a through lane.
    ([185,88,196,93],"footpath_access_unverified"),
    ([177.1,158.1,177.9,158.9],"unverified_surface"),
    ([60,43,65,48],"unverified_surface"),  # Vehicle pixels even without a Task 1 detection.
    ([65,140,172,150],"broad_area"),
])
def test_refuses_lane_obstructions_boundaries_and_imprecise_areas(marked_scene, mark, reason):
    request,_ = marked_scene
    out = PassengerGrounder.passenger_area(object(),request,mark)
    assert out["result"]["status"]=="needs_clarification" and out["result"]["destination_point_xy"] is None
    assert out["result"]["destination_box_xyxy"] is None and out["provenance"]["reason"]==reason
    checks = manual_offlane.verify(request,out,requested_box=mark)
    assert checks["offlane_mask_replayed"] and checks["decision_reproduced"]
    assert not checks["independent_model_proposal"]


@pytest.mark.parametrize("crossing_class", [8, 23])
def test_crossing_tap_explains_the_surface_and_replays_older_refusals(marked_scene, monkeypatch, crossing_class):
    request, _ = marked_scene
    classes = np.full((180,320), 13, dtype=np.uint8)
    classes[110:130,80:110] = crossing_class
    packet = provide_map(monkeypatch, request["scene"], classes)
    box = [88,113,96,120]
    response = manual_offlane.select(request, box)
    assert response["result"]["destination_point_xy"] is None
    assert "crossing" in response["result"]["clarification_question"]
    assert manual_offlane.verify(request,response)["decision_reproduced"]
    clean, original, mark, centre = manual_offlane.mark_input(request, box)
    old = manual_offlane._outcome(clean, original, mark, centre, packet, explain_crossing=False)
    assert old["provenance"]["reason"] == "unverified_surface"
    assert manual_offlane.verify(request,old)["decision_reproduced"]


def test_footpath_explains_unverified_vehicle_access_and_preserves_old_generic_receipt(marked_scene,monkeypatch):
    req,_=marked_scene
    classes=np.full((180,320),13,dtype=np.uint8)
    classes[90:130,80:155]=15
    packet=provide_map(monkeypatch,req["scene"],classes)
    box=[92,100,98,108]
    current=manual_offlane.select(req,box)
    assert current["result"]["destination_point_xy"] is None
    assert current["provenance"]["reason"]=="footpath_access_unverified"
    assert "footpath" in current["result"]["clarification_question"]
    assert manual_offlane.verify(req,current)["decision_reproduced"]
    clean,key,mark,center=manual_offlane.mark_input(req,box)
    old=manual_offlane._outcome(clean,key,mark,center,packet,explain_footpath=False)
    assert old["provenance"]["reason"]=="unverified_surface"
    assert manual_offlane.verify(req,old)["decision_reproduced"]


def test_exact_original_point_survives_fractional_coordinates_and_historical_tracks(marked_scene, monkeypatch):
    req,packet = marked_scene
    box=[182.25,119.25,187.75,123.75]
    out=manual_offlane.select(req,box)
    assert out["provenance"]["reason"]=="parking" and out["result"]["destination_point_xy"]==[185.,121.5]
    assert out["result"]["destination_box_xyxy"]==box and out["provenance"]["selected_box"]==box
    assert out["provenance"]["offlane_surface_packet"]==packet
    # The archived signed evidence is sufficient to replay, even if the index later disappears.
    monkeypatch.setattr(manual_surfaces,"get_packet",lambda _:None)
    assert manual_offlane.verify(req,out,requested_box=box)["offlane_mask_replayed"]
    missing = manual_offlane.select(req,box)
    assert missing["provenance"]["reason"]=="unavailable" and not missing["result"]["destination_point_xy"]
    assert not manual_offlane.verify(req,missing)["offlane_mask_replayed"]


@pytest.mark.parametrize("change",["point","words","box","submitted","source","signature","signed_map","role","tracks"])
def test_replay_refuses_unbound_or_relocated_marks(marked_scene,change):
    req,packet=marked_scene
    mark=[178,116,183,120]
    out=manual_offlane.select(req,mark)
    assert out["result"]["status"]=="proposed"
    q=deepcopy(req)
    if change=="point":out["result"]["destination_point_xy"][0]+=.5
    if change=="box":out["result"]["destination_box_xyxy"][1]+=1
    if change=="words":q["text"]="At a different shop."
    if change=="submitted":mark=[175,110,182,116]
    if change=="signature":out["provenance"]["offlane_surface_packet"]["signature"]="01"*64
    if change=="signed_map":out["provenance"]["offlane_surface_packet"]["body"]["class_map"]["sha256"]="0"*64
    if change=="role":out["provenance"]["independent_model_proposal"]=True
    if change=="tracks":q["scene"]["tracks"][1]["bbox_xyxy"]=mark
    if change=="source":Path(q["scene"]["frame"]["image_path"]).write_bytes(b"changed pixels")
    with pytest.raises((ValueError,OSError)):
        manual_offlane.verify(q,out,requested_box=mark)


@pytest.mark.parametrize("missing",["absent","corrupt","unreadable","other_photo"])
def test_unknown_or_invalid_map_has_no_offer(marked_scene,monkeypatch,missing):
    req,packet=marked_scene
    broken=deepcopy(packet)
    if missing=="corrupt":broken["signature"]="01"*64
    if missing=="other_photo":broken["body"]["image_sha256"]="0"*64
    def lookup(_):
        if missing=="unreadable":
            raise OSError("Unavailable test segmenter")
        return None if missing=="absent" else broken
    monkeypatch.setattr(manual_surfaces,"get_packet",lookup)
    out=manual_offlane.select(req,[178,116,183,120])
    assert out["provenance"]["reason"]=="unavailable" and out["provenance"]["offlane_surface_packet"] is None
    assert out["result"]["status"]=="needs_clarification" and out["result"]["destination_point_xy"] is None
