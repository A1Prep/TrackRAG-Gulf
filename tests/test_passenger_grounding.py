from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from trackrag.dropoff.anchor_geometry import AnchorGeometry
from trackrag.dropoff.interaction import PassengerSession
from trackrag.dropoff.landmark_check import evidence_board
from trackrag.dropoff.landmarks import landmark_messages, parse_landmark
from trackrag.dropoff.passenger import (PassengerGrounder, digest, instruction_context, pair_required,
                                       proposed, supported_region, unresolved)


@pytest.fixture
def request_row(tmp_path):
    im = tmp_path/"frame.png"
    Image.new("RGB",(100,100),"navy").save(im)
    sha = sha256(im.read_bytes()).hexdigest()
    frame = {"recording_id":"recording-a","frame_id":3,"width":100,"height":100,
             "timestamp_s":.2,"image_path":str(im)}
    def track(t,i,b):
        return {"recording_id":"recording-a","source":"predicted","class_name":"Bus",
                "frame_id":t,"track_id":i,"bbox_xyxy":b}
    scene = {"frame":frame,"source":"predicted","provenance":{"image_sha256":sha},
             "tracks":[track(3,717,[10,40,35,75]),track(2,727,[8,30,28,63]),track(3,752,[63,42,84,74])]}
    return {"request_id":"request","scene":scene,"text":"Let me out between the buses.",
            "input_mode":"typed","dialogue":[]}


def test_only_current_candidates_and_parse(request_row):
    messages, ids = landmark_messages(request_row)
    assert list(ids) == ["L1","L2"]
    assert [x["track_id"] for x in ids.values()] == [717,752]
    for raw in ("LANDMARK L1 L2", "L1 L2"):
        assert parse_landmark(raw,ids,request_row["scene"]["frame"])["track_ids"] == [717,752]
    for bad in ("LANDMARK L7","L1 L1","OBJECT 2 5 2 9","LANDMARK L1\nThis is the correct answer.",
                "AREA 53 41 79 50","MISSING\nL1","ASK "):
        with pytest.raises(ValueError):
            parse_landmark(bad,ids,request_row["scene"]["frame"])
    assert parse_landmark("OBJECT 200 300 400 900",ids,request_row["scene"]["frame"])["boxes"] == [[20,30,40,90]]
    request_row["scene"]["tracks"][0]["frame_id"] = 4
    with pytest.raises(ValueError,match="future"):
        landmark_messages(request_row)


def test_crop_board_keeps_no_target_or_future(request_row):
    _, ids = landmark_messages(request_row)
    with Image.open(request_row["scene"]["frame"]["image_path"]) as im:
        ims, alt = evidence_board(im,ids,parse_landmark("L1",ids,request_row["scene"]["frame"]))
        assert ims.size == (672,384)
        assert alt == ["L2"]
        ims.close()


def test_displacement_selects_only_supported_region():
    mask = np.ones((100,100),dtype=bool)
    a,b = [1,10,20,20],[75,10,90,20]
    assert supported_region(mask,(100,100),[50,89],[36,74,66,96],[a,b],between=[a,b]) is None
    point, box = supported_region(mask,(100,100),[49,19],[38,15,60,26],[a,b],between=[a,b])
    assert box[0] <= point[0] < box[2] and 20 <= point[0] < 75
    assert supported_region(mask,(100,100),[7,15],[4,11,9,19],[a,b]) is None
    mask[:] = False
    assert supported_region(mask,(100,100),[50,20],[38,12,66,28],[]) is None


def test_correction_and_followup_preserve_active_relation(request_row):
    request_row["dialogue"] = [{"role":"user","text":"Between the two motorbikes, please."}]
    request_row["text"] = "The two black ones on our right."
    assert pair_required(instruction_context(request_row))
    for phrase in ("Actually, beside the bus.", "Not between them. Beside the lamppost."):
        request_row["text"] = phrase
        assert not pair_required(instruction_context(request_row))


def test_relative_regression_roundtrip(tmp_path):
    data = [{"anchor":[10+i,15,35+i,50],"point":[16+i,58],"box":[12+i,51,28+i,67],
             "size":[100,80],"text":"beside the bus"} for i in range(10)]
    path = tmp_path/"head.json"
    path.write_text(json.dumps(AnchorGeometry.fit(data)))
    m = AnchorGeometry.load(path)
    pt, box = m.predict([[24,15,49,50]],(100,80),"beside the bus")
    assert box[0] < pt[0] < box[2] and box[1] < pt[1] < box[3]
    with pytest.raises(ValueError,match="SHA256"):
        AnchorGeometry.load(path,"2"*64)


class FixedModel:
    def interpret(self,r):
        result = unresolved("Which bus?") if r["text"] == "a bus" else proposed([55,72],[50,65,62,80])
        return {"ok":True,"request_id":r["request_id"],"result":result,
                "provenance":{"image":{"sha256":r["scene"]["provenance"]["image_sha256"]},"request_sha256":digest(r)}}

    def passenger_area(self,r,box):
        result = self.interpret(r)
        result["result"] = proposed([65,45],box)
        return result


def test_session_speech_correction_binding_and_confirmation(request_row,tmp_path):
    session = PassengerSession(request_row["scene"],FixedModel(),audit_dir=tmp_path/"receipt")
    v = session.submit("a bus")
    assert v["state"] == "needs_clarification"
    audio = b"test WAV binding supplied by this fixture"
    h = sha256(audio).hexdigest()
    d = tmp_path/"receipt"/"audio"
    d.mkdir()
    (d/(h+".wav")).write_bytes(audio)
    v = session.submit("the bus on our left",original_transcript="the bus on our lift",
                       speech={"audio_sha256":h,"transcript":"the bus on our lift"})
    assert v["state"] == "proposed"
    first = v["proposal_id"]
    replacement = session.select_area([60,40,72,50])
    assert replacement["selected_by"] == "passenger"
    with pytest.raises(ValueError):
        session.confirm(first)
    assert session.confirm(replacement["proposal_id"])["state"] == "confirmed"
    with pytest.raises(ValueError):
        session.confirm(replacement["proposal_id"])
    assert session.submit("again")["turns"][0]["text"] == "again"
    audit = [json.loads(l) for l in next((tmp_path/"receipt").glob("*.jsonl")).read_text().splitlines()]
    assert audit[2]["detail"]["request"]["original_transcript"] == "the bus on our lift"
    assert audit[2]["detail"]["request"]["text"] == "the bus on our left"


def test_error_or_changed_photo_cannot_confirm(request_row,tmp_path):
    session = PassengerSession(request_row["scene"],FixedModel(),audit_dir=tmp_path/"journal")
    token = session.submit("behind this bus")["proposal_id"]
    clone = session.scene
    clone["frame"]["recording_id"] = "foreign"
    assert session.scene["frame"]["recording_id"] == "recording-a"
    from pathlib import Path
    Path(clone["frame"]["image_path"]).unlink()
    with pytest.raises(ValueError,match="changed"):
        session.confirm(token)
    assert session.view()["state"] == "error" and session.view()["proposal_id"] is None


def test_persistence_failure_leaves_proposal_unconfirmed(request_row,tmp_path,monkeypatch):
    s = PassengerSession(request_row["scene"],FixedModel(),audit_dir=tmp_path)
    token = s.submit("behind the bus")["proposal_id"]
    def fail(action,detail):
        raise OSError("no space")
    monkeypatch.setattr(s,"_save",fail)
    with pytest.raises(OSError):
        s.confirm(token)
    assert s.view()["state"] == "proposed"


def test_error_envelope_for_invalid_payload():
    m = object.__new__(PassengerGrounder)
    import threading
    m.lock = threading.RLock()
    assert m.interpret(None)["ok"] is False
    assert m.passenger_area([],None)["ok"] is False
