"""Prepare input-only, SHA-bound sparse final frames and detector context. Never run a model or read targets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from PIL import Image

from tools.dropoff_predict import read_requests
from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.data import file_sha256, inference_input_sha256
from trackrag.dropoff.scenes import _load_artifact, _check_run_manifest, scene_sha256, RUN_SAVED
from trackrag.vision.config import load_config


ROOT=Path(__file__).resolve().parents[1]
TASK2="2d96dcb6afdffb46c5d94d299768fe6a92acc094"
PACKAGE="Task2_56_Case_Human_Review_Corrected"
TASK1="709e969260866a5476d8c199609c59de580589f7"
RELEASE="4418cec01b03c6f797cadb06c9405843a30988e2"
TEST_ARCHIVE_SHA="d2e22df9070cac1b89c052ae97297297592d4c6715d7db55ef54c0c9297c5ddc"
FINAL={"video_122233","video_125233","video_155425","video_160325","video_204647"}
FIELDS={"partition":True,"case_id":True,"request":True,
        "source":{"image_sha256":True},"review":{"input_sha256":True,"image_sha256":True}}
DECODER=json.JSONDecoder()


def require(ok,message):
    if not ok:
        raise ValueError(message)


def git(revision,name):
    return subprocess.check_output(["git","show",revision+":"+name],cwd=ROOT)


def space(s,p):
    while p<len(s) and s[p].isspace():
        p+=1
    return p


def skip_value(s,p):
    """Skip bytes of an unselected value without decoding strings, coordinates or other supervision."""
    if s[p]=='"':
        p+=1
        while p<len(s):
            if s[p]=="\\":
                p+=2
            elif s[p]=='"':
                return p+1
            else:
                p+=1
        raise ValueError("Unterminated JSON string")
    if s[p] in "{[":
        expected=["}" if s[p]=="{" else "]"]
        p+=1
        while p<len(s) and expected:
            ch=s[p]
            if ch=='"':
                p=skip_value(s,p)
                continue
            if ch in "{[":
                expected.append("}" if ch=="{" else "]")
            elif ch in "}]":
                require(ch==expected.pop(),"Malformed source JSON container")
            p+=1
        require(not expected,"Incomplete source JSON")
        return p
    end=p
    while end<len(s) and s[end] not in " ,}\r\n\t]":
        end+=1
    require(end>p,"Missing JSON value")
    return end


def project(s, fields=FIELDS, start=0):
    p=space(s,start)
    require(s[p]=="{","Expected JSON object")
    out={}
    p=space(s,p+1)
    while s[p]!="}":
        key,p=DECODER.raw_decode(s,p)
        require(isinstance(key,str) and key not in out,"Duplicate projected key")
        p=space(s,p)
        require(s[p]==":","Missing separator")
        p=space(s,p+1)
        spec=fields.get(key)
        if spec is True:
            out[key],p=DECODER.raw_decode(s,p)
        elif isinstance(spec,dict):
            out[key],p=project(s,spec,p)
        elif spec=="presence":
            out[key]=not s.startswith("null",p)
            p=skip_value(s,p)
        else:
            p=skip_value(s,p)
        p=space(s,p)
        if s[p]=="}":
            break
        require(s[p]==",","Malformed object")
        p=space(s,p+1)
    require(all(k in out for k,v in fields.items() if v!="presence"),"Required input metadata is absent")
    return out,p+1


def write(path,obj):
    with path.open("x",encoding="utf8",newline="\n") as f:
        f.write(json.dumps(obj,sort_keys=True,allow_nan=False,indent=2)+"\n")


def lines(path,rows):
    with path.open("x",encoding="utf8",newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")


def official_archive_receipt():
    url="https://huggingface.co/api/datasets/KuAvLab/EMT/tree/"+RELEASE+"?expand=true"
    with urlopen(Request(url),timeout=18) as response:
        tree=json.load(response)
    item=next(r for r in tree if r["path"]=="test_images.tar.gz")
    require(item["lfs"]["oid"]==TEST_ARCHIVE_SHA and item["size"]==8720080037,"Different original image archive")
    archive="https://huggingface.co/datasets/KuAvLab/EMT/resolve/"+RELEASE+"/test_images.tar.gz"
    req=Request(archive+"?trackrag_final_source_range=0",headers={"Range":"bytes=0-31","Accept-Encoding":"identity"})
    with urlopen(req,timeout=18) as response:
        require(response.status==206 and response.headers.get("Content-Range")=="bytes 0-31/8720080037","Exact source range refused")
        prefix=response.read(33)
    require(len(prefix)==32 and prefix[:2]==bytes.fromhex("1f8b"),"Unexpected archive prefix")
    return {"dataset_tree":url,"original_archive":archive,"archive_bytes":item["size"],"archive_lfs_sha256":TEST_ARCHIVE_SHA,
            "http_status":206,"range_bytes":"0-31","payload_sha256":sha256(prefix).hexdigest(),"downloaded_archive_bytes":32,
            "archive_full_downloaded_or_verified":False}


def prepare(args):
    require(not args.out.exists(),"Choose a fresh input inventory folder")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".final-input-",dir=args.out.parent) as transient:
        staged=Path(transient)/"ready"
        result=assemble(args,staged)
        staged.rename(args.out)
    return result


def assemble(args,out):
    frame_meta=json.loads(git(TASK1,"experiments/vision/emt_frame_sha256_manifest.json"))
    run_meta=json.loads(git(TASK1,"experiments/vision/shared_prediction_runs.json"))
    split=json.loads(git(TASK2,PACKAGE+"/split_audit.json"))
    m=json.loads(git(TASK2,PACKAGE+"/manifest.json"))
    require(set(split["recording_ids"]["test"])==FINAL and split["recording_disjoint"] and split["evaluation_recordings_reserved"],
            "Different corrected final scene split")
    current=m["current_approved_file"]
    require(current=="review_cases.jsonl" and m["fully_approved_cases"]==50 and m["reconfirmation_required_cases"]==6,
            "Unexpected approved package boundaries")
    data=git(TASK2,PACKAGE+"/"+current)
    require(sha256(data).hexdigest()==m["files"][current]["sha256"],"Approved Git blob differs from the package manifest")
    eligible=[]
    for raw in data.splitlines():
        line=raw.decode("utf8")
        row,end=project(line)
        require(not line[end:].strip(),"Unexpected trailing data")
        if row["partition"]=="test":
            eligible.append(row)
    require(len(eligible)==44 and len({r["case_id"] for r in eligible})==44
            and not ({r["case_id"] for r in eligible}&set(m["reconfirmation_case_ids"])),"Incorrect approved test membership")
    archive=official_archive_receipt() if args.verify_archive else None
    artifacts={}
    for condition in ("finetuned","pretrained"):
        cfg=load_config(ROOT/"configs/vision"/(condition+".json"))
        for rec in sorted(FINAL):
            prov,prov_digest,by_frame,frames,size,track_digest=_load_artifact(args.predictions/("predictions-"+condition),rec)
            require(prov["identity_hash"]==cfg.identity_hash() and prov["detector_kind"]!="stub","Incorrect saved detector")
            context={"track_source":"predicted","run_kind":RUN_SAVED,"detector_kind":prov["detector_kind"],
                "identity_hash":cfg.identity_hash(),"checkpoint_sha256":cfg.detector.checkpoint_sha256,
                "tracks_sha256":track_digest,"provenance_sha256":prov_digest,
                "replayed_from_detections_sha256":(prov.get("replayed_from") or {}).get("detections_sha256"),"history_frames":0}
            _check_run_manifest(run_meta,context,rec)
            require(list(frames)==sorted(int(k) for k in frame_meta["recordings"][rec]),"Detector frame IDs differ from pinned image inventory")
            artifacts[(condition,rec)]=(context,by_frame,frames,size)
    dest=out/"source"
    dest.mkdir(parents=True)
    target_frames=[];requests=[];originals=[];bindings=[];current_rows=[];max_ids=defaultdict(int);counts=Counter()
    for observed in eligible:
        old=validate_request(observed["request"])
        case=observed["case_id"];frame=old["scene"]["frame"]
        rec,fid=frame["recording_id"],frame["frame_id"]
        pic=f"images/test/{rec}/{fid:05d}.jpg"
        require(rec in FINAL and old["request_id"]==case and frame["image_path"].replace("\\","/").endswith(pic),
                "Official case and image identity disagree")
        image_sha=frame_meta["recordings"][rec][str(fid)]
        image=git(TASK2,PACKAGE+"/"+pic)
        require(image_sha==sha256(image).hexdigest()==m["files"][pic]["sha256"]==observed["source"]["image_sha256"]
                ==observed["review"]["image_sha256"],"Original image hash differs from the four pinned sources")
        with Image.open(BytesIO(image)) as im:
            width,height=im.size
            im.verify()
        require((width,height)==(frame["width"],frame["height"]),"Reviewed pixels have other dimensions")
        private_path=dest/pic
        private_path.parent.mkdir(parents=True,exist_ok=True)
        private_path.write_bytes(image)
        original_binding=inference_input_sha256(old,image_sha)
        require(original_binding==observed["review"]["input_sha256"],"Reviewed input binding does not verify")
        old["scene"]["frame"]["image_path"]="source/"+pic
        originals.append(old)
        detailed={}
        for condition in ("finetuned","pretrained"):
            context,rows,frames,size=artifacts[(condition,rec)]
            require(fid in frames and (width,height)==tuple(size),"Missing source prediction frame or wrong detector geometry")
            obs=sorted(rows.get(fid,[]),key=lambda r:r["track_id"])
            require(all(r["source"]=="predicted" and r["recording_id"]==rec and r["frame_id"]==fid for r in obs),
                    "Noncurrent or foreign observations")
            timestamp=(fid-min(frames))/10.
            fields={"recording_id":rec,"frame_id":fid,"timestamp_s":timestamp,
                    "width":width,"height":height,"image_sha256":image_sha,"observations":obs,"history":[]}
            canonical=scene_sha256({"model_input":fields,"provenance":context})
            detailed[condition]={"current_observations":len(obs),"scene_sha256_from_original_hashes":canonical,
                                 "tracks_sha256":context["tracks_sha256"],"vision_identity":context["identity_hash"]}
            if condition=="finetuned":
                new=deepcopy(old)
                new["scene"]["source"]="predicted"
                new["scene"]["frame"]["timestamp_s"]=timestamp
                new["scene"]["tracks"]=obs
                new["scene"]["provenance"]={"image_sha256":image_sha,"vision_identity_hash":context["identity_hash"],
                    "tracks_sha256":context["tracks_sha256"],"prediction_provenance_sha256":context["provenance_sha256"],
                    "source_frame_hash_commit":TASK1,"context_policy":"original_current_frame_saved_predictions_only",
                    "source_kind":"sparse_verified_originals_and_complete_source_id_index"}
                new=validate_request(new)
                new_binding=inference_input_sha256(new,image_sha)
                requests.append(new)
                current_rows.append({"case_id":case,"source_frame_hash":image_sha,"model_input":fields,"provenance":context,
                                     "expected_task1_scene_sha256":canonical})
        changes={"track_observations":old["scene"]["tracks"]!=new["scene"]["tracks"],
                 "timestamp":frame["timestamp_s"]!=new["scene"]["frame"]["timestamp_s"],
                 "source_kind":old["scene"]["source"]!=new["scene"]["source"]}
        counts.update({k:v for k,v in changes.items()})
        counts["different_model_visible_input"]+=new_binding!=original_binding
        bindings.append({"case_id":case,"recording_id":rec,"frame_id":fid,"original_git_image":PACKAGE+"/"+pic,
            "image_sha256":image_sha,"width":width,"height":height,"reviewed_input_sha256":original_binding,
            "saved_detector_input_sha256":new_binding,"reviewed_observation_count":len(old["scene"]["tracks"]),
            "reviewed_timestamp_s":frame["timestamp_s"],"reconstructed_timestamp_s":new["scene"]["frame"]["timestamp_s"],
            "input_changes":changes,"saved_detector_conditions":detailed})
        target_frames.append({"recording_id":rec,"frame_id":fid})
        max_ids[rec]=max(max_ids[rec],fid)
    lines(out/"requests-with-saved-tracks.jsonl",requests)
    lines(out/"requests-as-human-reviewed.jsonl",originals)
    lines(out/"input-bindings.jsonl",bindings);lines(out/"task1-scene-ingredients.jsonl",current_rows);write(out/"frames.json",target_frames)
    for name in ("requests-with-saved-tracks.jsonl","requests-as-human-reviewed.jsonl"):
        restored=read_requests(out/name)
        require(len(restored)==44 and all(file_sha256(r["scene"]["frame"]["image_path"])==b["image_sha256"]
                for r,b in zip(restored,bindings,strict=True)),"Normal inference reader failed to bind an original image")
    prefixes={}
    for rec,maximum in sorted(max_ids.items()):
        prefix={int(k) for k in frame_meta["recordings"][rec] if int(k)<=maximum}
        available={int(Path(n).stem) for n in m["files"] if n.startswith("images/test/"+rec+"/")} & prefix
        prefixes[rec]={"confirmed":sum(f["recording_id"]==rec for f in target_frames),"last_confirmed":maximum,
            "upstream_required_prefix_count":len(prefix),"in_corrected_package":len(available),
            "missing_from_corrected_package":len(prefix-available)}
    receipt={"kind":"input_only_unscored_shared_detector_sparse_source_handoff","images_publicly_redistributed":False,
        "final_supervision_values_decoded":False,"human_reviewer_names_read":False,"model_inference_performed":False,
        "task2_commit":TASK2,"task1_commit":TASK1,"approved_case_blob_sha256":sha256(data).hexdigest(),
        "confirmed_final_inputs":44,"pending_excluded":len(m["reconfirmation_case_ids"]),"saved_detectors_quantified":["finetuned","pretrained"],
        "model_visible_changes":dict(counts),"per_recording":prefixes,"archive_receipt":archive,
        "context_policy":"No past/future observations. Full temporal origin uses the pinned complete source frame ID inventory and Task 1's default release rate of 10 fps.",
        "formal_scoring":"No final model run before protocol freeze. Sparse ingredients give expected upstream scene hashes; the original canonical prefix check still requires actual image filenames through the query. Reviewed bindings are kept separate from any changed input.",
        "files":{n:file_sha256(out/n) for n in ("requests-with-saved-tracks.jsonl","requests-as-human-reviewed.jsonl","input-bindings.jsonl","task1-scene-ingredients.jsonl","frames.json")}}
    write(out/"receipt.json",receipt)
    print(json.dumps({k:receipt[k] for k in ("confirmed_final_inputs","pending_excluded","model_visible_changes","files")},indent=2))
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,default=ROOT/"data/task1-shared-20261119")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--verify-archive",action="store_true",help="Check 32 bytes and the LFS metadata of the original public test-image archive")
    prepare(parser.parse_args())
