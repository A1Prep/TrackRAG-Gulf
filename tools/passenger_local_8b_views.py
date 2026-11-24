"""Read neutral curb-point evidence on registered photos with the laptop's verified 8B adapter."""
import argparse
import gc
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

from PIL import Image

from tools.passenger_release import ROOT
from trackrag.dropoff import bay_landmarks, curb_points, local_surfaces, manual_surfaces
from trackrag.dropoff.parking_evidence import signed_bytes
from trackrag.dropoff.passenger import digest
from trackrag.passenger_app import load_scenes


def atomic_json(path, value):
    temporary = path.with_suffix(".writing")
    temporary.write_bytes(signed_bytes(value))
    os.replace(temporary, path)


def sha(data):
    return sha256(data).hexdigest()


def call_input(model, key, stage, guide, words, view, limit):
    return {"model":model,"stage":stage,"guide":sha(guide.encode()),"words":sha(words.encode()),
            "view":sha(view),"photo":key,"limit":limit}


def publish(bank, working, destination=None):
    """Package only complete measured observations and require an exact source/hash match."""
    from trackrag.dropoff import local_8b_observations as local

    entries={}
    for scene in load_scenes([bank]).values():
        key=scene["provenance"]["image_sha256"]
        if scene["provenance"].get("network_policy")!="local_only" or not local_surfaces.matches_scene(scene):
            raise ValueError("A local-only source or original changed before publication.")
        measured=json.loads((working/(key+".json")).read_bytes())
        points,landmarks=measured["points"],measured["landmarks"]
        def check_call(model,stage,guide,words,view,limit,claimed):
            identity=call_input(model,key,stage,guide,words,view,limit)
            receipt=json.loads((working/"calls"/(digest(identity)+".json")).read_bytes())
            if (receipt["input"]!=identity or receipt["reply"]!=claimed
                    or receipt["reply_sha256"]!=sha(claimed.encode()) or not 1<=receipt["output_tokens"]<=limit):
                raise ValueError("The model observation differs from the source pixels, prompt or raw response.")
            return {"input_sha256":digest(identity),"response_sha256":receipt["reply_sha256"],
                    "input_tokens":receipt["input_tokens"],"output_tokens":receipt["output_tokens"],
                    "elapsed_seconds":receipt["seconds"]}
        with Image.open(scene["frame"]["image_path"]) as im:
            classes=manual_surfaces.decode(manual_surfaces.get_packet(key),scene["frame"],key)
            options=curb_points.locations(scene,classes)
            for row,option in zip(points["rows"],options,strict=True):
                point=option["point"]
                if row["point"]!=point:
                    raise ValueError("Curb coordinates differ from the registered segmentation and tracks.")
                full,near=[curb_points.view(im,point,close=value) for value in (False,True)]
                if row["views"]!=[sha(full),sha(near)]:
                    raise ValueError("A neutral curb view has changed.")
                actual=[check_call(points["local_model"],"current_full",curb_points.CHECK,curb_points.WORDS[0],full,6,row["replies"][0])]
                if row["replies"][0].strip().upper().strip(".")=="OPEN":
                    actual.append(check_call(points["local_model"],"current_crop",curb_points.CHECK,curb_points.WORDS[0],near,6,row["crop_verdict"]))
                    if row["crop_verdict"].strip().upper().strip(".")=="OPEN":
                        for stage,guide,words,reply in zip(("neighbours","frontage","roadside"),curb_points.GUIDES[1:],
                                                           curb_points.WORDS[1:],row["replies"][1:],strict=True):
                            actual.append(check_call(points["local_model"],stage,guide,words,near,23,reply))
                elif row["crop_verdict"] is not None:
                    raise ValueError("An unexpected follow-up crop appears on a closed point.")
                if len(actual)<5 and any(row["replies"][1:]):
                    raise ValueError("A closed or unverified curb point has a location description.")
                row["calls"]=actual
            if len(landmarks["bands"])!=len(bay_landmarks.BANDS):
                raise ValueError("Missing original frontage band observations.")
            for i,row in enumerate(landmarks["bands"]):
                view=bay_landmarks.band_image(im,i)
                if sha(view)!=row["view_sha256"]:
                    raise ValueError("Frontage observation belongs to a different crop.")
                row["call"]=check_call(landmarks["local_model"],"landmark_band",bay_landmarks.GUIDE,bay_landmarks.WORDS,view,23,row["places"])
        entry={}
        for kind in ("points","landmarks"):
            body=measured[kind]
            if (body["image_sha256"]!=key or body["recording_id"]!=scene["frame"]["recording_id"]
                    or body["frame_id"]!=scene["frame"]["frame_id"] or body["source_context"]!=digest(curb_points.source_context(scene))):
                raise ValueError("Measured response belongs to a different original or tracks.")
            entry[kind]={"origin":local.ORIGIN,"body_sha256":sha(signed_bytes(body)),"body":body}
        entries[key]=entry
    output={"version":1,"origin":local.ORIGIN,"scope":"Local 8B neutral visual observations on provider-blurred public KartaView photographs.",
            "attribution":"© Grab and KartaView Contributors. Derived observations shared under CC BY-SA 4.0.",
            "license_url":"https://creativecommons.org/licenses/by-sa/4.0/","sources":entries}
    destination=destination or local.BANK
    atomic_json(destination,output)
    print(f"Published {len(entries)} sources. Release SHA-256 {sha(destination.read_bytes())}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=ROOT/"data/uae-parking-local/passenger-requests.jsonl")
    parser.add_argument("--working", type=Path, default=ROOT/"runs/submission-field-20261124/local-8b-views")
    parser.add_argument("--local-surface-index",nargs=2,metavar=("JSON","SHA256"),
                        help="Use another locally generated and source-verified surface bank")
    parser.add_argument("--output",type=Path,help="Destination JSON for --release")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--release", action="store_true",help="Package completed local measurements without starting the model.")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output and not args.release:
        parser.error("--output is used with --release.")
    if args.local_surface_index:
        local_surfaces.register(*args.local_surface_index)
        if args.release and not args.output:
            parser.error("An additional photo bank needs its own --output file.")
    if args.release:
        return publish(args.bank,args.working,args.output)
    if not (args.worker or args.smoke_only):
        # On a 4 GB Windows display GPU, release the graphics context between small batches.
        # Every observation has an input-bound receipt before restarting a worker.
        flags=[sys.executable,"-u","-m","tools.passenger_local_8b_views","--bank",str(args.bank),
               "--working",str(args.working),"--worker"]
        if args.local_surface_index:
            flags.extend(["--local-surface-index",*args.local_surface_index])
        while True:
            before=len(list((args.working/"calls").glob("*.json")))
            result=subprocess.run(flags)
            if result.returncode==0:
                return
            after=len(list((args.working/"calls").glob("*.json")))
            if after<=before or result.returncode not in (75,76):
                raise SystemExit(result.returncode)
            print(f"Releasing the laptop GPU after {after-before} recorded responses.",flush=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
    def offline(event, values):
        if event=="socket.connect":
            raise RuntimeError("Offline inference cannot connect to a socket.")
    sys.addaudithook(offline)
    from trackrag.dropoff.model import GroundingConfig
    from trackrag.dropoff.streamed import StreamedRuntime
    import torch
    torch.set_num_threads(2)
    work = args.working
    work.mkdir(parents=True,exist_ok=True)
    calls = work / "calls"
    calls.mkdir(exist_ok=True)
    config = GroundingConfig.from_mapping(json.loads((ROOT/"models/passenger-feature/grounding.json").read_bytes()))
    config = replace(config,adapter_dir=str(ROOT/"models/passenger-feature/adapter"),model_dir=str(ROOT/config.model_dir),
                     cache_dir=str(ROOT/config.cache_dir),max_time_s=60,inference_resident_layers=2)
    sources = []
    for scene in load_scenes([args.bank]).values():
        key = scene["provenance"]["image_sha256"]
        frame = scene["frame"]
        if scene["provenance"].get("network_policy")!="local_only" or not local_surfaces.matches_scene(scene):
            raise ValueError("An original is not in the published local-only registration.")
        packet = manual_surfaces.get_packet(key)
        classes = manual_surfaces.decode(packet,frame,key)
        options = curb_points.locations(scene,classes)
        sources.append((scene,[o["point"] for o in options]))
    sources.sort(key=lambda pair: (not bool(pair[1]),pair[0]["provenance"]["image_sha256"]))
    print("Native locations per original:",[len(o) for _,o in sources],flush=True)
    start = monotonic()
    runtime = StreamedRuntime(config,verify_weights=True)
    print(f"Loaded verified 8B base and trained adapter in {monotonic()-start:.2f}s.",flush=True)
    model = {"base":config.revision,"adapter":sha((Path(config.adapter_dir)/"adapter_model.safetensors").read_bytes()),
             "processor":bay_landmarks.MODELS["processor"],"quantization":config.quantization,"runtime":"local_streamed",
             "min_pixels":config.min_pixels,"max_pixels":config.max_pixels,"inference_resident_layers":config.inference_resident_layers,
             "weights":{k:v for k,v in runtime.identity.items() if k!="path"}}
    new_calls=0
    def read(key,stage,guide,words,view,limit):
        nonlocal new_calls
        identifier=call_input(model,key,stage,guide,words,view,limit)
        name=digest(identifier)
        path=calls/(name+".json")
        if path.exists():
            saved=json.loads(path.read_bytes())
            if saved["input"]==identifier and saved.get("reply_sha256")==sha(saved["reply"].encode()):
                return saved
        with Image.open(BytesIO(view)) as original:
            photo=original.convert("RGB")
        try:
            gc.collect()
            torch.cuda.empty_cache()
            msg=[{"role":"system","content":guide},{"role":"user","content":[{"type":"image","image":photo},
                                                                          {"type":"text","text":words}]}]
            reply=runtime.generate(msg,config=replace(config,max_new_tokens=limit))
        finally:
            photo.close()
        answer=bay_landmarks.public_phrase(reply.text.strip())
        details=reply.details
        saved={"input":identifier,"reply":answer,"reply_sha256":sha(answer.encode()),
               "input_tokens":details["input_tokens"],"output_tokens":details["output_tokens"],
               "seconds":round(details["generation_elapsed_s"],3)}
        atomic_json(path,saved)
        print(f"{key[:8]} {stage} {saved['seconds']}s {answer!r}",flush=True)
        new_calls+=1
        if args.worker and new_calls>=12:
            raise SystemExit(75)
        return saved
    def yes(value):
        return value.strip().upper().strip(".")=="OPEN"
    def read_bands(key,im):
        rows=[]
        for i in range(len(bay_landmarks.BANDS)):
            rendered=bay_landmarks.band_image(im,i)
            reply=read(key,"landmark_band",bay_landmarks.GUIDE,bay_landmarks.WORDS,rendered,23)
            rows.append({"view_sha256":sha(rendered),"places":reply["reply"]})
        return rows
    try:
        # A scene without a proposed point can still have a visible storefront or road landmark.
        if not args.smoke_only:
            for scene,_ in sources:
                key=scene["provenance"]["image_sha256"]
                with Image.open(scene["frame"]["image_path"]) as original:
                    bands=read_bands(key,original)
                atomic_json(work/(key+".landmarks.json"),bands)
        for scene,points in sources:
            frame=scene["frame"]; key=scene["provenance"]["image_sha256"]
            with Image.open(frame["image_path"]) as loaded:
                im=loaded.convert("RGB")
            context=digest(curb_points.source_context(scene))
            rows=[]
            for point in points:
                full=curb_points.view(im,point,close=False)
                near=curb_points.view(im,point,close=True)
                verdict=read(key,"current_full",curb_points.CHECK,curb_points.WORDS[0],full,6)
                if args.smoke_only:
                    return
                close = read(key,"current_crop",curb_points.CHECK,curb_points.WORDS[0],near,6) if yes(verdict["reply"]) else None
                phrases=["","",""]
                if close and yes(close["reply"]):
                    phrases=[read(key,stage,guide,word,near,23)["reply"]
                             for stage,guide,word in zip(("neighbours","frontage","roadside"),curb_points.GUIDES[1:],curb_points.WORDS[1:],strict=True)]
                rows.append({"point":point,"views":[sha(full),sha(near)],"replies":[verdict["reply"],*phrases],
                             "crop_verdict":close["reply"] if close else None})
            bands=read_bands(key,im)
            proof={"recording_id":frame["recording_id"],"frame_id":frame["frame_id"],"size":list(im.size),
                   "image_sha256":key,"source_context":context,"local_model":model}
            points_body={**proof,"recipe":curb_points.RECIPE,"model":bay_landmarks.MODELS,
                         "guides":digest([curb_points.GUIDES,curb_points.WORDS]),"rows":rows}
            bands_body={**proof,"recipe":bay_landmarks.RECIPE,"models":bay_landmarks.MODELS,
                        "prompt_sha256":sha((bay_landmarks.GUIDE+"\n"+bay_landmarks.WORDS).encode()),"bands":bands}
            atomic_json(work/(key+".json"),{"points":points_body,"landmarks":bands_body})
            im.close()
        if not args.smoke_only:
            # A uniform synthetic image contains no traffic or visible parking. Its reply is
            # reported as a diagnostic only and cannot enter a passenger source or proposal.
            blank=Image.new("RGB",(640,360),(127,127,127))
            raw=BytesIO()
            blank.save(raw,format="PNG")
            marker=curb_points.view(blank,[320,230],close=False)
            control=read(sha(raw.getvalue()),"uniform_control",curb_points.CHECK,curb_points.WORDS[0],marker,6)
            atomic_json(work/"uniform-control.json",control)
    except torch.OutOfMemoryError as exc:
        print(f"Local CUDA device ran out of staging memory: {exc}",flush=True)
        raise SystemExit(76)
    finally:
        runtime.close()


if __name__=="__main__":
    main()
