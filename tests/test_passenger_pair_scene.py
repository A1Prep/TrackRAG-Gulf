"""Local single-photo intake tests use a painted software fixture and synthetic track observations."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import pytest
from PIL import Image

from tools import passenger_pair_scene as scene
from trackrag.passenger_app import load_scenes


def make_source(tmp_path,monkeypatch):
    buf=BytesIO();Image.new("RGB",(100,80),"gray").save(buf,format="JPEG");pic=buf.getvalue()
    identity=scene.load_config(scene.ROOT/"configs/vision/finetuned.json").identity_hash()
    observations=[{"track_id":f,"recording_id":scene.RECORDING,"frame_id":f,"source":"predicted",
                   "class_name":"Car","bbox_xyxy":[10,12,40,38]} for f in (780,781,782)]
    raw=b"".join(json.dumps(obs).encode()+b"\n" for obs in observations)
    track_hash=sha256(raw).hexdigest()
    meta={"recording_id":scene.RECORDING,"frame_ids":[1,780,781,782],"first_frame_size":[100,80],
          "tracks_sha256":track_hash,"identity_hash":identity,"observations":len(observations)}
    prov=json.dumps(meta).encode()
    monkeypatch.setattr(scene,"load_run_manifest",lambda _:({"runs":{identity:{scene.RECORDING:{
        "tracks_sha256":track_hash,"provenance_sha256":sha256(prov).hexdigest()}}}},"software-run"))
    monkeypatch.setattr(scene,"load_frame_manifest",lambda _:({"recordings":{scene.RECORDING:{"781":sha256(pic).hexdigest()}}},"software-images"))
    def only_photo(repo,path):
        assert path==scene.PHOTO_IN_TASK2
        return pic
    monkeypatch.setattr(scene,"git_blob",only_photo)
    zip_path=tmp_path/"team-predictions.zip"
    with ZipFile(zip_path,"w") as archive:
        archive.writestr("nested/video_054907.tracks.jsonl",raw)
        archive.writestr("nested/video_054907.provenance.json",prov)
    return zip_path,pic,raw,prov


def test_portable_scene_only_contains_current_detector_rows_and_reloads(tmp_path,monkeypatch):
    zipped,_,_,_=make_source(tmp_path,monkeypatch)
    out=tmp_path/"private"
    receipt=scene.prepare(zipped,out)
    assert scene.prepare(zipped,out)==receipt and receipt["current_predicted_observations"]==1
    assert receipt["human_target_rows_loaded"]==0 and receipt["footage_prefix_expanded"] is False
    before=(out/"requests.jsonl").read_bytes();request=json.loads(before)
    assert request["scene"]["frame"]["image_path"]=="images/00781.jpg"
    assert request["scene"]["tracks"][0]["frame_id"]==781 and request["scene"]["tracks"][0]["source"]=="predicted"
    assert request["text"]=="Open this recorded scene." and "target" not in request
    moved=tmp_path/"moved";out.rename(moved)
    loaded=next(iter(load_scenes([moved/"requests.jsonl"]).values()))
    assert loaded["provenance"]["image_sha256"]==receipt["original_photo_sha256"] and str(moved) in loaded["frame"]["image_path"]


@pytest.mark.parametrize("fault",["foreign_photo","foreign_provenance","corrupt_track","duplicate_zip_member","conflicting_existing_bank"])
def test_does_not_publish_unbound_inputs(tmp_path,monkeypatch,fault):
    zipped,pic,tracks,prov=make_source(tmp_path,monkeypatch);out=tmp_path/"private"
    photo=tmp_path/"photo.jpg";photo.write_bytes(b"wrong" if fault=="foreign_photo" else pic)
    if fault=="conflicting_existing_bank":
        scene.prepare(zipped,out,image=photo)
        (out/"requests.jsonl").write_text("A previous bank that must be preserved")
    elif fault in {"foreign_provenance","corrupt_track","duplicate_zip_member"}:
        with ZipFile(zipped,"w") as archive:
            archive.writestr("first/video_054907.provenance.json",prov if fault!="foreign_provenance" else b"{}")
            archive.writestr("first/video_054907.tracks.jsonl",tracks if fault!="corrupt_track" else tracks+b"\n")
            if fault=="duplicate_zip_member":archive.writestr("second/video_054907.tracks.jsonl",tracks)
    with pytest.raises(ValueError):scene.prepare(zipped,out,image=photo)
    if fault=="conflicting_existing_bank":assert (out/"requests.jsonl").read_text()=="A previous bank that must be preserved"
    else:assert not out.exists()
