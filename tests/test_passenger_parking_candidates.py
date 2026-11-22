"""Software checks for the visible-parking research comparator."""
import json
from hashlib import sha256

import numpy as np
import pytest
from PIL import Image

from tools import passenger_parking_candidates as parking
from tools import passenger_parking_anchors as anchors


def scene(tmp_path=None):
    photo = (tmp_path / "original.png") if tmp_path else "unused.png"
    tracks = [{"recording_id": "clip", "frame_id": 4, "source": "predicted", "class_name": "Car",
               "track_id": i, "bbox_xyxy": b} for i,b in [(1,[40,50,82,104]),(2,[132,52,173,106])]]
    req = {"request_id":"c1","text":"The opening between those two cars.","input_mode":"typed","dialogue":[],
           "scene":{"source":"predicted","frame":{"recording_id":"clip","frame_id":4,"timestamp_s":0.3,
              "width":220,"height":150,"image_path":str(photo)},"tracks":tracks,"provenance":{}}}
    labels = np.full((150,220),13,dtype=np.uint8)
    labels[88:110,86:126] = 10
    return req, labels


def test_only_parking_surface_and_original_pixel_scale():
    req, labels = scene()
    options = parking.options_for(req,labels,"parking")
    assert options and options[0]["tracks"] == [1,2]
    x,y = map(int, options[0]["point"])
    assert 86 <= x < 126 and 88 <= y < 110
    labels[:] = 13
    assert parking.options_for(req,labels,"parking") == []
    assert parking.options_for(req,labels,"all-road")
    assert parking.options_for(req,labels,"parking",[1,99]) == []


def test_vehicle_reply_requires_distinct_boxes_bound_to_current_tracks():
    request, _ = scene()
    boxes = [[182, 333, 373, 693], [600, 347, 786, 707]]
    resolve = lambda value: anchors.resolve_anchor_reply(request, value)
    real = resolve(json.dumps({"vehicles": boxes}))
    assert real["state"] == "matched" and real["track_ids"] == [1, 2]
    assert resolve(json.dumps({"vehicles": [boxes[0], boxes[0]]}))["state"] == "ask"
    assert resolve('{"vehicles":[]}')["state"] == "ask"
    assert resolve('{"vehicles":[],"vehicles":[]}')["state"] == "error"
    assert resolve('{"vehicles":[[NaN,0,1,1],[600,300,780,700]]}')["state"] == "error"
    request["scene"]["tracks"][0]["frame_id"] = 3
    with pytest.raises(ValueError, match="current-frame"):
        resolve(json.dumps({"vehicles": boxes}))


def test_anchor_receipts_require_the_same_image_words_and_prompt():
    request, _ = scene()
    request["scene"]["provenance"]["image_sha256"] = "a" * 64
    record = {"id": "c1", "source_sha": "a" * 64, "request": request["text"],
              "prompt_sha256": sha256(anchors.ANCHOR_PROMPT.encode()).hexdigest(),
              "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "raw": '{"vehicles":[]}'}
    assert anchors.read_anchor_records([request], json.dumps([record]))["c1"]["state"] == "ask"
    for field in ("source_sha", "request", "prompt_sha256", "revision"):
        with pytest.raises(ValueError, match="changed"):
            anchors.read_anchor_records([request], json.dumps([{**record, field: "different"}]))
    with pytest.raises(ValueError, match="duplicate"):
        anchors.read_anchor_records([request], '[{"id":"c1","id":"c1"}]')


def test_out_of_scope_noncurrent_or_scoring_inputs():
    req, labels = scene()
    req["text"]="Next to the bin please"
    assert parking.options_for(req,labels,"parking") == []
    req["scene"]["tracks"][0]["frame_id"]=3
    with pytest.raises(ValueError, match="current"):
        parking.options_for(req,labels,"parking")
    req,labels=scene()
    req["target"]={"destination_point_xy":[109,102]}
    with pytest.raises(ValueError):
        parking.options_for(req,labels,"parking")


def test_small_intervening_car_counts_even_outside_anchor_budget():
    req, labels = scene()
    template = req["scene"]["tracks"][0]
    for i in range(12):
        req["scene"]["tracks"].append({**template,"track_id":i+3,"bbox_xyxy":[1,1,37,48]})
    req["scene"]["tracks"].append({**template,"track_id":15,"bbox_xyxy":[98,67,115,104]})
    # 15 is the smallest box by area. Even when the two anchors are explicit,
    # an intervening vehicle still breaks adjacency.
    assert parking.options_for(req,labels,"parking",[1,2]) == []
    req["scene"]["tracks"].pop()
    assert parking.options_for(req,labels,"parking",[1,2])


def test_export_binds_original_and_actual_class_map(tmp_path):
    req,labels=scene(tmp_path)
    photo=tmp_path/"original.png"
    Image.new("RGB",(220,150),"gray").save(photo)
    sha=sha256(photo.read_bytes()).hexdigest()
    req["scene"]["provenance"]["image_sha256"]=sha
    path=tmp_path/"requests.jsonl";path.write_text(json.dumps(req)+"\n",encoding="utf8")
    maps=tmp_path/"maps";maps.mkdir();mask_file=maps/(sha+".png")
    Image.fromarray(labels).save(mask_file)
    receipt={"sha256":sha,"model":parking.MODEL,"revision":parking.REVISION,"size":[220,150],"mask_shape":[150,220],
             "labels":{"10":"Parking","13":"Road"},"file":mask_file.name,"mask_sha":sha256(mask_file.read_bytes()).hexdigest()}
    (maps/"receipt.json").write_text(json.dumps([receipt]),encoding="utf8")
    dest=tmp_path/"choices"
    rows=parking.export(path,maps,dest,"parking")
    assert len(rows)==1 and rows[0]["is_pair"] and len(rows[0]["options"])==1
    manifest=json.loads((dest/"manifest.json").read_bytes())
    assert manifest["choices_sha256"]==sha256((dest/"choices.json").read_bytes()).hexdigest()
    assert manifest["requests_sha256"]==sha256(path.read_bytes()).hexdigest()
    assert manifest["destination_labels_used"] is False
    mask_file.write_bytes(mask_file.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="class map changed"):
        parking.export(path,maps,tmp_path/"broken","parking")
    assert not (tmp_path/"broken").exists()
