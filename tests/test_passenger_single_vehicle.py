"""Software fixtures for the single-vehicle surface, evidence and cache boundaries."""
import base64
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from threading import Event
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tests.test_passenger_place_grounding import case
from trackrag.dropoff import parking_evidence, single_vehicle as vehicle, single_vehicle_backend as transport
from trackrag.dropoff.passenger import digest
from trackrag.dropoff.single_vehicle_surface import behind_patch, beside_patches


@pytest.mark.parametrize(("text","expected"), [
    ("Could you drop me off behind the white car?", ("behind","car","white",None)),
    ("can u drop me behind the truck", ("behind","truck",None,None)),
    ("Could you drop me next to that black pickup truck?", ("beside","pickup","black",None)),
    ("Let me out alongside the grey pick-up on my right.", ("beside","pickup","gray","right")),
    ("Beside the leftmost sedan please.", ("beside","sedan",None,"left")),
    ("Between the two white cars.", None),
    ("Behind both trucks.", None),
    ("By the café behind the car.", None),
    ("Drop me at the bus stop.", None),
    ("Next to the bus shelter.", None),
    ("Behind the truck and the car.", None),
    ("Not behind the white car.", None),
])
def test_vehicle_relation(text, expected):
    actual = vehicle.request_reference({"text":text})
    result = tuple(actual[k] for k in ("relation","kind","color","side")) if actual else None
    assert result == expected
    if actual:
        assert not actual["unsupported"]


def test_visible_ground_beside_and_behind_cannot_be_interchanged():
    size = [128,96]
    anchor = [54,20,82,57]
    area = np.full((96,128),13,dtype=np.uint8)
    assert behind_patch(area,anchor,[anchor],size) is None
    assert beside_patches(area,anchor,[anchor],size) == []
    area[57:90,52:85] = 10
    rear = behind_patch(area,anchor,[anchor],size)
    assert rear is not None and rear["point"][1] > anchor[3]
    assert beside_patches(area,anchor,[anchor],size) == []
    assert behind_patch(area,anchor,[anchor,[48,56,91,91]],size) is None
    area[:] = 13
    area[45:66,28:54] = 10
    assert behind_patch(area,anchor,[anchor],size) is None
    side, = beside_patches(area,anchor,[anchor],size)
    assert side["side"] == "left" and side["box"][2] < anchor[0]
    assert beside_patches(area,anchor,[anchor,[25,40,55,70]],size) == []


def make_receipt(tmp_path, monkeypatch, *, answer="SEDAN WHITE", box=None):
    req, photo, classes, key = case(tmp_path, monkeypatch)
    req["text"] = "Could you drop me off behind the white car?"
    scene = req["scene"]
    f = scene["frame"]
    scene["tracks"].append({"recording_id":f["recording_id"],"frame_id":f["frame_id"],
                            "track_id":7,"class_name":"Car","bbox_xyxy":box or [44,11,68,37],"source":"predicted"})
    # An earlier tracked vehicle cannot supply an anchor or occupy the current patch.
    scene["tracks"].append({**scene["tracks"][0],"frame_id":1,"track_id":17,"bbox_xyxy":[30,1,95,63]})
    classes[:] = 13
    classes[37:63,41:74] = 10
    ctx = vehicle.context(req)
    assert [row[0] for row in ctx["tracks"]] == [7]
    def stub_rank(image, text, tracks, *, frame):
        return {"alternatives":[{"track_id":7,"score":.8}],"null_score":.1,"checkpoint_sha256":"software"}
    monkeypatch.setattr(vehicle,"ranker_from",lambda archive:SimpleNamespace(rank=stub_rank))
    options = vehicle.opportunities(ctx,classes)
    assert len(options) == 1
    with Image.open(BytesIO(photo)) as original:
        calls = vehicle.observation_calls(original,options,lambda guide,words,pixels:answer)
    raw = BytesIO()
    Image.fromarray(classes).save(raw,format="PNG")
    data = raw.getvalue()
    body = {"recipe":vehicle.RECIPE,"models":vehicle.MODELS,"image_sha256":ctx["image_sha256"],
            "context_sha256":digest(ctx),"class_map":{"png":base64.b64encode(data).decode(),"sha256":sha256(data).hexdigest()},
            "ranker_zlib":"software-only","calls":calls}
    return req, photo, {"body":body,"signature":key.sign(parking_evidence.signed_bytes(body)).hex()}, key


@pytest.mark.parametrize(("answer","expected"),[("SEDAN WHITE","proposed"),("SEDAN BLUE","needs_clarification"),("NONE","needs_clarification")])
def test_signed_appearance_surface_and_actor(tmp_path, monkeypatch, answer, expected):
    req, photo, packet, key = make_receipt(tmp_path,monkeypatch,answer=answer)
    out = vehicle.outcome(req,photo,packet)
    assert out["result"]["status"] == expected
    assert vehicle.verify(req,out)["independent_model_proposal"] is (expected == "proposed")
    changed = deepcopy(req)
    changed["text"] = "Drop me beside that white car."
    side = vehicle.outcome(changed,photo,packet)
    assert side["result"]["status"] == "needs_clarification" and side["result"]["destination_point_xy"] is None
    assert vehicle.verify(changed,side)["decision_reproduced"]
    with pytest.raises(ValueError):
        vehicle.verify(changed,out)
    bad = deepcopy(packet)
    bad["body"]["calls"][0]["track_id"] = 17
    # A legitimate signature from an incompatible crop/track still cannot replay.
    bad["signature"] = key.sign(parking_evidence.signed_bytes(bad["body"])).hex()
    with pytest.raises(ValueError,match="displayed anchor"):
        vehicle.outcome(req,photo,bad)


def test_broadside_vehicle_cannot_issue_a_rear_pin_even_if_the_trained_reader_agrees(tmp_path,monkeypatch):
    req, photo, packet, _ = make_receipt(tmp_path,monkeypatch,box=[26,18,75,37])
    assert packet["body"]["calls"][0]["response"] == "SEDAN WHITE"
    prior = vehicle.outcome(req,photo,packet,policy=vehicle.PREVIOUS_POLICY)
    assert prior["result"]["status"] == "proposed"
    assert vehicle.verify(req,prior)["independent_model_proposal"]
    with pytest.raises(ValueError, match="superseded"):
        vehicle.verify_current(req,prior)
    current = vehicle.outcome(req,photo,packet)
    assert current["provenance"]["reason"] == "rear_view_unclear"
    assert current["result"]["destination_point_xy"] is None
    assert not vehicle.verify_current(req,current)["independent_model_proposal"]
    assert not vehicle.visible_vehicle_end([0,18,30,37],[96,64])
    assert vehicle.visible_vehicle_end([44,11,68,37],[96,64])


def fake_backend(tmp_path, req):
    f = req["scene"]["frame"]
    allowed = {(req["scene"]["provenance"]["image_sha256"],f["recording_id"],f["frame_id"])}
    model = SimpleNamespace(allowed=allowed,root=tmp_path,python="test-interpreter")
    return transport.SingleVehicleBackend(model,cache=tmp_path/"cache")


def test_writable_disk_is_optional_and_the_original_is_rechecked(tmp_path, monkeypatch):
    req, photo, packet, _ = make_receipt(tmp_path,monkeypatch)
    backend = fake_backend(tmp_path,req)
    calls = []
    def remote(*args,**kwargs):
        calls.append(kwargs["input"])
        return SimpleNamespace(returncode=0,stdout=json.dumps(packet).encode())
    def readonly(*_):
        raise PermissionError("Read-only test storage")
    monkeypatch.setattr(transport.subprocess,"run",remote)
    monkeypatch.setattr(transport.os,"replace",readonly)
    assert backend.interpret(req)["result"]["status"] == "proposed"
    another = {**req,"text":"Behind the red car please."}
    assert backend.interpret(another)["result"]["status"] == "needs_clarification"
    assert len(calls) == 1 and not list(backend.cache.glob("*.json"))
    from pathlib import Path
    Path(req["scene"]["frame"]["image_path"]).write_bytes(b"edited original")
    with pytest.raises(ValueError,match="Source photograph"):
        backend.interpret(req)


def test_cold_request_is_actionable_while_one_shared_job_finishes(tmp_path, monkeypatch):
    req, photo, packet, _ = make_receipt(tmp_path,monkeypatch)
    backend = fake_backend(tmp_path,req)
    started, finish = Event(), Event()
    calls = []
    def delayed(*args,**kwargs):
        calls.append(1)
        started.set()
        assert finish.wait(5)
        return SimpleNamespace(returncode=0,stdout=json.dumps(packet).encode())
    monkeypatch.setattr(transport.subprocess,"run",delayed)
    monkeypatch.setattr(transport,"INTERACTIVE_WAIT_SECONDS",.02)
    try:
        waiting = backend.interpret(req)
        assert started.is_set() and waiting["provenance"]["operational_status"] == "pending"
        assert waiting["result"]["destination_point_xy"] is None and "mark" in waiting["result"]["clarification_question"]
        assert vehicle.verify(req,waiting)["language_calls"] == 0
        repeat = backend.interpret({**req,"text":"Behind that red car."})
        assert repeat["provenance"]["operational_status"] == "pending" and len(calls) == 1
        # A different cold original would be held independently or promptly marked busy.
        other = deepcopy(req)
        other["scene"]["frame"]["frame_id"] += 1
        f = other["scene"]["frame"]
        backend.places.allowed.add((other["scene"]["provenance"]["image_sha256"],f["recording_id"],f["frame_id"]))
        monkeypatch.setattr(transport,"MAX_PENDING",1)
        assert backend.interpret(other)["provenance"]["operational_status"] == "busy"
        pending, = backend._pending.values()
    finally:
        finish.set()
    pending.result(3)
    assert backend.interpret(req)["result"]["status"] == "proposed" and len(calls) == 1
    backend.places.allowed.clear()
    with pytest.raises(ValueError,match="outside the public"):
        backend.interpret(req)
