"""Prepare a local KartaView photo cohort with Task 1's released detector and tracker."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from PIL import Image

from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.inputs import _inspect_image
from trackrag.vision import predict_tracks
from trackrag.vision.__main__ import JsonlSink
from trackrag.vision.config import load_config
from trackrag.vision.detector import RFDETRDetector


ROOT=Path(__file__).resolve().parents[1]
REFERENCE_FIELDS={"text","words","request","target","expected","regions","lane","interior","painted_edge",
                  "tracks","annotations","labels","reference","reference_polygon","visual_reference"}


@dataclass
class PhotoSequence:
    recording_id: str
    frames: list

    def iter_frames(self):
        return iter(self.frames)


def read_sources(manifest, originals=None):
    """Read source identity and licence provenance only. No model is involved in verification."""
    rows=json.loads(manifest.read_text(encoding="utf-8-sig")).get("sources")
    if not isinstance(rows,list) or not rows:
        raise ValueError("A source-only manifest needs a nonempty sources array.")
    by_frame={}
    hashes=set()
    for raw in rows:
        if not isinstance(raw,dict) or set(raw)&REFERENCE_FIELDS:
            raise ValueError("Photo ingestion accepts source metadata only.")
        rec,frame=raw.get("recording_id"),raw.get("frame_id")
        location=urlsplit(raw.get("source_url",""))
        m=re.fullmatch(r"/details/([1-9]\d*)/(\d+)/?",location.path)
        if (location.scheme!="https" or location.netloc!="kartaview.org" or location.query or location.fragment
                or not m or rec!="kartaview_"+m[1] or type(frame) is not int or frame!=int(m[2])):
            raise ValueError("Photo URL, KartaView recording and original frame index must agree.")
        derivative=urlsplit(raw.get("processed_original_url",""))
        if (raw.get("original_blur_status")!="BLURRED" or derivative.scheme!="https"
                or derivative.netloc!="cdn.kartaview.org" or derivative.query or derivative.fragment
                or not re.fullmatch(r"/pr:sharp/[A-Za-z0-9_-]+",derivative.path)):
            raise ValueError("The source needs a privacy-processed original from KartaView.")
        if (raw.get("license")!="CC BY-SA 4.0" or raw.get("license_url")!="https://creativecommons.org/licenses/by-sa/4.0/"
                or raw.get("source_terms")!="https://kartaview.org/terms"
                or not isinstance(raw.get("attribution"),str) or not raw["attribution"].strip()):
            raise ValueError("The KartaView source needs a licence, source terms and contributor attribution.")
        width,height=raw.get("width"),raw.get("height")
        if any(type(v) is not int or v<1 for v in (width,height)):
            raise ValueError("Original image dimensions must be positive integers.")
        key=raw.get("sha256")
        if not isinstance(key,str) or not re.fullmatch("[a-f0-9]{64}",key):
            raise ValueError("Every photograph needs its original SHA-256.")
        if (rec,frame) in by_frame or key in hashes:
            raise ValueError("Repeated original bytes or recording/frame identity in source manifest.")
        when=raw.get("timestamp_s",0.0)
        if type(when) not in (int,float) or not math.isfinite(when) or when<0:
            raise ValueError("Supplied capture times must be finite, nonnegative seconds.")
        supplied=raw.get("original_path")
        if not isinstance(supplied,str) or not supplied.strip():
            raise ValueError("Every original must already exist on this computer.")
        path=((Path(originals)/(key+".jpg")) if originals else manifest.parent/Path(supplied)).resolve(strict=True)
        scene_frame={"recording_id":rec,"frame_id":frame,"timestamp_s":float(when),"width":width,"height":height,
                     "image_path":str(path)}
        if _inspect_image(scene_frame)[1]!=key:
            raise ValueError(f"Original image hash disagrees with the manifest for {rec}/{frame}.")
        with Image.open(path) as image:
            if image.format!="JPEG":
                raise ValueError("Expected an original processed KartaView JPEG.")
        attribution={k:raw[k] for k in ("source_url","processed_original_url","photo_id","title","attribution",
                                         "license","license_url","credit","contributor","original_blur_status",
                                         "source_terms") if k in raw}
        by_frame[rec,frame]=(scene_frame,key,attribution,"provided" if "timestamp_s" in raw else "still image")
        hashes.add(key)
    return by_frame


def write_json(path,value):
    with path.open("w",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)
        stream.write("\n")


def prepare(manifest,out,config,*,originals=None,detector_factory=RFDETRDetector,tracker_factory=None):
    """Save a complete portable bundle atomically; keep the original photo bytes and dimensions."""
    manifest=Path(manifest).resolve(strict=True)
    out=Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    metadata=read_sources(manifest,originals)
    out.parent.mkdir(parents=True,exist_ok=True)
    detector=detector_factory(config)
    with TemporaryDirectory(dir=out.parent,prefix="."+out.name+"-") as td:
        bundle=Path(td)/"bundle"
        images=bundle/"images"
        predictions=bundle/"predictions"
        images.mkdir(parents=True)
        predictions.mkdir()
        grouped=defaultdict(list)
        for (rec,_),(current,key,_,_) in sorted(metadata.items()):
            dest=images/(key+".jpg")
            # Preserve the verified JPEG without recompression; same-volume cohorts share its bytes.
            try:
                os.link(current["image_path"],dest)
            except OSError:
                shutil.copyfile(current["image_path"],dest)
            frame={**current,"image_path":str(dest)}
            if _inspect_image(frame)[1]!=key:
                raise ValueError("Original photograph changed while the local bundle was prepared.")
            grouped[rec].append(frame)
        sources=[]
        bank=bundle/"passenger-requests.jsonl"
        with bank.open("w",encoding="utf8",newline="\n") as rows:
            for rec,frames in sorted(grouped.items()):
                sink=JsonlSink(predictions,rec)
                try:
                    tracks=predict_tracks(PhotoSequence(rec,frames),config=config,detector=detector,
                                          tracker_factory=tracker_factory,sink=sink)
                finally:
                    sink.close()
                track_file=predictions/(rec+".tracks.jsonl")
                raw_file=predictions/(rec+".detections.jsonl")
                provenance=predictions/(rec+".provenance.json")
                vision=json.loads(provenance.read_bytes())
                digests={"tracks_sha256":sha256(track_file.read_bytes()).hexdigest(),
                         "detections_sha256":sha256(raw_file.read_bytes()).hexdigest(),
                         "vision_provenance_sha256":sha256(provenance.read_bytes()).hexdigest()}
                if vision["identity_hash"]!=config.identity_hash() or vision["tracks_sha256"]!=digests["tracks_sha256"]:
                    raise RuntimeError("Task 1 output identity and artifact digests disagree.")
                for frame in frames:
                    fid=frame["frame_id"]
                    _,key,attribution,time_basis=metadata[rec,fid]
                    if _inspect_image(frame)[1]!=key:
                        raise ValueError("Original photograph changed while its detections were computed.")
                    current=[r for r in tracks if r["frame_id"]==fid]
                    portable={**frame,"image_path":"images/"+key+".jpg"}
                    scene={"frame":portable,"tracks":current,"source":"predicted","provenance":{
                        "mode":"current_frame_task1","image_sha256":key,"detector_run":True,
                        "network_policy":"local_only",
                        "vision_identity_sha256":config.identity_hash()}}
                    # The catalogue reads only this scene. No passenger text or gold target is exported.
                    validate_request({"scene":scene,"text":"Open frame","request_id":"validate","input_mode":"typed","dialogue":[]})
                    rows.write(json.dumps({"scene":scene},ensure_ascii=False,allow_nan=False)+"\n")
                    sources.append({**attribution,"recording_id":rec,"frame_id":fid,"original_sha256":key,
                                    "size":[frame["width"],frame["height"]],"relative_image":portable["image_path"],
                                    "timestamp_basis":time_basis,"observations":len(current),**digests})
        write_json(bundle/"sources.json",{"sources":sources,"vision_identity":config.identity(),
                   "vision_identity_sha256":config.identity_hash(),"checkpoint_sha256":config.detector.checkpoint_sha256,
                   "source_manifest_sha256":sha256(manifest.read_bytes()).hexdigest(),
                   "scene_bank_sha256":sha256(bank.read_bytes()).hexdigest(),
                   "time_policy":"A still image without a capture timestamp uses 0.0 seconds of clip-relative time.",
                   "scope":"Local inference. Original source-sized pixel boxes, online tracking within each provided KartaView sequence. A track from a one-image sequence is a per-image observation."})
        bundle.rename(out)
    return out/"passenger-requests.jsonl"


def main():
    if sys.argv[1:2]==["install"]:
        from tools.passenger_local_photo_install import install_main
        return install_main(sys.argv[2:])
    cmd=argparse.ArgumentParser(description=__doc__,epilog="Use the install subcommand for the released predictions without running the detector.")
    cmd.add_argument("--manifest",type=Path,required=True,help="JSON containing source metadata and local paths; no passenger annotations.")
    cmd.add_argument("--originals",type=Path,help="Read source JPEGs from this directory using their SHA-256 filenames.")
    cmd.add_argument("--out",type=Path,required=True)
    cmd.add_argument("--config",type=Path,default=ROOT/"configs/vision/finetuned.json")
    cmd.add_argument("--weights-dir",type=Path,default=ROOT/"models/task1")
    cmd.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    cmd.add_argument("--cpu-threads",type=int,default=2)
    args=cmd.parse_args()
    if args.cpu_threads<1:
        cmd.error("--cpu-threads must be at least 1.")
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",YOLO_AUTOINSTALL="false")
    if args.device=="cpu":
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    def deny_network(event,values):
        if event=="socket.connect":
            raise RuntimeError("This photo ingestion runs offline with local model weights.")
    sys.addaudithook(deny_network)
    import torch
    torch.set_num_threads(args.cpu_threads)
    config=load_config(args.config,weights_dir=args.weights_dir,device=args.device)
    target=prepare(args.manifest,args.out,config,originals=args.originals)
    print(target,flush=True)


if __name__=="__main__":
    main()
