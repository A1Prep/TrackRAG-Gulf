"""Inspect only visible public bay outlines, their frontage and their current adjacent vehicles."""
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

import modal
from tools.passenger_bay_landmarks import image
from tools.passenger_place_modal import TrainedPlaceReader

app=modal.App("trackrag-public-bay-surroundings")
if modal.is_local():
    from tools.passenger_release import ROOT
    image=image.add_local_file(ROOT/"tools/passenger_bay_landmarks.py","/root/tools/passenger_bay_landmarks.py")


@app.cls(image=image,gpu="A100-40GB",cpu=4,memory=24000,timeout=300,startup_timeout=300,min_containers=0,max_containers=1,
         scaledown_window=10,volumes={"/base":modal.Volume.from_name("trackrag-qwen8b-public-weights").read_only()},
         secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
class BayContext(TrainedPlaceReader):
    @modal.enter()
    def load(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        self.key=Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ.pop("PARKING_SIGNING_KEY"),validate=True))
        self.known={r["sha256"] for f in Path("/sources").glob("*.json") for r in json.loads(f.read_bytes())["photos"]}
        self.qwen=None

    @modal.method()
    def inspect(self, photo, options, vehicles):
        from PIL import Image
        from trackrag.dropoff import bay_anchors as a,bay_landmarks
        from trackrag.dropoff.passenger import digest
        from trackrag.dropoff.parking_evidence import signed_bytes
        key=sha256(photo).hexdigest()
        if key not in self.known or not 1<=len(options)<=18:raise ValueError("Unknown public bay input")
        im=Image.open(BytesIO(photo)).convert("RGB");rows=[];appearance={}
        for opt in options:
            local=a.closest_tracks(opt,vehicles);overlay=a.view(im,opt,local)
            front=bay_landmarks.public_phrase(self.read(a.FRONT,a.FRONT_WORDS,overlay))
            pair=self.read(a.PAIR,a.PAIR_WORDS,overlay) if len(local)>=2 else "NONE"
            for idx in a.pair_indices(pair,len(local)):
                tid,_,box=local[idx]
                if str(tid) in appearance:continue
                crop=a.crop_car(im,box);reply=self.read(a.APPEARANCE,a.APPEARANCE_WORDS,crop)
                appearance[str(tid)]={"view":sha256(crop).hexdigest(),"reply":reply[:55]}
            rows.append({"shape":digest(opt["polygon"]),"view":sha256(overlay).hexdigest(),"frontage":front,
                         "pair_reply":pair[:60]})
        body={"recipe":a.RECIPE,"model":bay_landmarks.MODELS,"prompt":a.prompt_digest(),"image_sha256":key,
              "size":list(im.size),"current_vehicles_sha256":digest(vehicles),"rows":rows,"appearance":appearance}
        return {"body":body,"signature":self.key.sign(signed_bytes(body)).hex()}


@app.local_entrypoint()
def main():
    from PIL import Image
    from trackrag.dropoff import bay_anchors as a,manual_surfaces,surface_stalls,whole_bay
    from trackrag.dropoff.contracts import validate_request
    from trackrag.dropoff.parking_evidence import signed_bytes
    from tools.passenger_release import ROOT,PUBLIC_PHOTO_LISTS
    bank=ROOT/"data/uae-parking-study";cache=ROOT/"runs/model-cache/bay-anchors-v1";cache.mkdir(exist_ok=True,parents=True)
    keys={p["sha256"] for f in PUBLIC_PHOTO_LISTS for p in json.loads((ROOT/"experiments/dropoff"/f).read_bytes())["photos"]}
    seen=set();records={};worker=BayContext()
    for line in (bank/"passenger-requests.jsonl").read_bytes().splitlines():
        q=validate_request(json.loads(line));scene=q["scene"];f=scene["frame"];key=scene["provenance"]["image_sha256"]
        if key in seen or key not in keys:continue
        seen.add(key);photo=(bank/(key+".jpg")).read_bytes()
        if sha256(photo).hexdigest()!=key:raise ValueError("Original changed")
        im=Image.open(BytesIO(photo)).convert("RGB");s=manual_surfaces.get_packet(key)
        if s is None:continue
        context={"size":list(im.size),"image_sha256":key,
                 "current_boxes":[t["bbox_xyxy"] for t in scene["tracks"] if t["frame_id"]==f["frame_id"]]}
        opts=whole_bay.visible_stalls(im,manual_surfaces.decode(s,f,key),context,s,surface_stalls.get_packet(key))
        if not opts:continue
        path = cache / (key + ".json")
        try:
            packet = json.loads(path.read_bytes())
            reusable = packet["body"]["recipe"] == a.RECIPE and len(a.observations(im, opts, scene, packet)) == len(opts)
        except (OSError, TypeError, KeyError, ValueError):
            reusable = False
        if not reusable:
            packet=worker.inspect.remote(photo,[{"point":o["point"],"polygon":o["polygon"]} for o in opts],a.vehicles(scene))
        if len(a.observations(im,opts,scene,packet))!=len(opts):raise ValueError("Candidate prompts failed replay.")
        data=signed_bytes(packet);tmp=path.with_suffix(".tmp");tmp.write_bytes(data);os.replace(tmp,path);records[key]=data
        print(f"Inspected {len(seen)}/{len(keys)} original photos; {len(opts)} regions.",flush=True)
    target=a.BANK.with_suffix(".tmp")
    with ZipFile(target,"w",ZIP_DEFLATED) as archive:
        for key,data in sorted(records.items()):archive.writestr(key+".json",data)
    os.replace(target,a.BANK);print(f"Finished {len(seen)} originals, {len(records)} with bay context.",flush=True)
