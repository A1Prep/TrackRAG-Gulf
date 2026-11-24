import json
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from tools import passenger_local_photos as local
from trackrag.passenger_app import load_scenes
from trackrag.vision.detections import Detection
from tests.test_vision_pipeline import FakeIoUTracker


def source(tmp,rec,fid):
    path=tmp/f"{rec}-{fid}.jpg"
    Image.new("RGB",(112,72),(fid,32,int(rec)%245)).save(path)
    return {"recording_id":"kartaview_"+rec,"frame_id":fid,"width":112,"height":72,
            "source_url":f"https://kartaview.org/details/{rec}/{fid}/","attribution":"Example contributor",
            "license":"CC BY-SA 4.0","license_url":"https://creativecommons.org/licenses/by-sa/4.0/",
            "source_terms":"https://kartaview.org/terms","original_blur_status":"BLURRED",
            "processed_original_url":"https://cdn.kartaview.org/pr:sharp/sample",
            "sha256":sha256(path.read_bytes()).hexdigest(),"original_path":path.name}


class OneCar:
    kind="stub"

    def __init__(self,config):
        pass

    def detect(self,image):
        return [Detection((10.,10.,48.,42.),.9,"Car")]


def manifest(tmp,sources):
    file=tmp/"pictures.json"
    file.write_text(json.dumps({"sources":sources}),encoding="utf8")
    return file


def test_portable_source_only_bank_reaches_real_catalog_and_keeps_frame_coordinates(tmp_path):
    a,b,c=source(tmp_path,"42",3),source(tmp_path,"42",5),source(tmp_path,"73",1)
    f=manifest(tmp_path,[c,b,a])
    out=tmp_path/"bundle"
    config=local.load_config(local.ROOT/"configs/vision/finetuned.json",device="cpu")
    bank=local.prepare(f,out,config,detector_factory=OneCar,tracker_factory=FakeIoUTracker)
    rows=[json.loads(line) for line in bank.read_bytes().splitlines()]
    assert all(set(row)=={"scene"} for row in rows)
    catalog=load_scenes([bank])
    frames=[s["frame"] for s in catalog.values()]
    assert [(s["recording_id"],s["frame_id"]) for s in frames]==[("kartaview_42",3),("kartaview_42",5),("kartaview_73",1)]
    assert [s["tracks"][0]["track_id"] for s in catalog.values()]==[1,1,1]
    for scene in catalog.values():
        assert Path(scene["frame"]["image_path"]).parent==out/"images"
        assert scene["provenance"]["network_policy"]=="local_only"
        assert sha256(Path(scene["frame"]["image_path"]).read_bytes()).hexdigest()==scene["provenance"]["image_sha256"]
    info=json.loads((out/"sources.json").read_bytes())
    assert info["source_manifest_sha256"]==sha256(f.read_bytes()).hexdigest()
    assert info["scene_bank_sha256"]==sha256(bank.read_bytes()).hexdigest()
    assert all(s["observations"]==1 and s["tracks_sha256"] for s in info["sources"])
    with pytest.raises(FileExistsError):
        local.prepare(f,out,config,detector_factory=OneCar,tracker_factory=FakeIoUTracker)


@pytest.mark.parametrize("change",[
    {"expected":"point"},{"words":"my parking request"},{"regions":{}},{"sha256":"0"*64},
    {"width":500},{"source_url":"https://kartaview.org/details/12/7/"},{"original_blur_status":"FAILED"},
    {"processed_original_url":"https://example.invalid/unprocessed"},{"license":"unverified"},
    {"attribution":None},{"attribution":{"contributor":"unverified"}},
])
def test_mismatched_source_or_reference_data_never_enters_inference(tmp_path,change):
    raw=source(tmp_path,"42",3)
    f=manifest(tmp_path,[{**raw,**change}])
    with pytest.raises(ValueError):
        local.read_sources(f)


def test_failed_detector_leaves_no_accepted_partial_scene_or_prediction(tmp_path):
    f=manifest(tmp_path,[source(tmp_path,"42",3)])
    class Broken(OneCar):
        def detect(self,image):
            raise OSError("simulated input failure")
    config=local.load_config(local.ROOT/"configs/vision/finetuned.json",device="cpu")
    with pytest.raises(OSError):
        local.prepare(f,tmp_path/"partial",config,detector_factory=Broken,tracker_factory=FakeIoUTracker)
    assert not (tmp_path/"partial").exists()
