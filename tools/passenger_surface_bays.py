"""Run the trained 8B visual reviewer on source-derived public parking regions."""
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

app=modal.App("trackrag-visible-parking-regions")
if modal.is_local():
    from tools.passenger_release import ROOT
    image=image.add_local_file(ROOT/"tools/passenger_bay_landmarks.py","/root/tools/passenger_bay_landmarks.py")


@app.cls(image=image,gpu="A100-40GB",cpu=4,memory=24000,min_containers=0,max_containers=1,scaledown_window=10,
         timeout=210,startup_timeout=300,volumes={"/base":modal.Volume.from_name("trackrag-qwen8b-public-weights").read_only()},
         secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
class PublicRegions(TrainedPlaceReader):
    @modal.enter()
    def load(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        self.key=Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ.pop("PARKING_SIGNING_KEY"),validate=True))
        self.known={r["sha256"] for p in Path("/sources").glob("*.json") for r in json.loads(p.read_bytes())["photos"]}
        self.qwen=None

    @modal.method()
    def check(self, photo, surface, context, options):
        from PIL import Image
        from trackrag.dropoff import surface_stalls as stalls,manual_surfaces as surfaces,bay_landmarks
        from trackrag.dropoff.parking_evidence import signed_bytes
        from trackrag.dropoff.passenger import digest
        sha=sha256(photo).hexdigest()
        if (sha not in self.known or context["image_sha256"]!=sha or not isinstance(options,list)
                or not 1<=len(options)<=8):
            raise ValueError("Expected verifiable candidates on a published original.")
        im=Image.open(BytesIO(photo)).convert("RGB")
        w,h=im.size
        surfaces.decode(surface,{"width":w,"height":h},sha)
        if context["size"]!=[w,h]:
            raise ValueError("Original dimensions changed.")
        rows=[]
        for option in options:
            replies=[]
            for close in (False,True):
                pixels=stalls.review_view(im,option,close)
                reply=self.read(stalls.CHECK,stalls.WORDS,pixels)
                replies.append({"view_sha256":sha256(pixels).hexdigest(),"reply":reply[:45]})
                if reply.strip().rstrip(".!")!="BAY": break
            rows.append({"shape_sha256":digest(option),"reviews":replies})
        body={"recipe":stalls.RECIPE,"models":bay_landmarks.MODELS,"image_sha256":sha,"size":[w,h],
              "context_sha256":digest(context),"surface_signature":surface["signature"],
              "prompt_sha256":sha256((stalls.CHECK+"\n"+stalls.WORDS).encode()).hexdigest(),"options":rows}
        return {"body":body,"signature":self.key.sign(signed_bytes(body)).hex()}


@app.local_entrypoint()
def main():
    from PIL import Image
    from trackrag.dropoff import manual_surfaces,surface_stalls as stalls
    from trackrag.dropoff.parking_evidence import signed_bytes
    from tools.passenger_release import ROOT,PUBLIC_PHOTO_LISTS
    bank=ROOT/"data/uae-parking-study"
    allowed={p["sha256"] for f in PUBLIC_PHOTO_LISTS for p in json.loads((ROOT/"experiments/dropoff"/f).read_bytes())["photos"]}
    cache=ROOT/"runs/model-cache/parking-stall-reviews-v1";cache.mkdir(exist_ok=True,parents=True)
    records={};seen=set();worker=PublicRegions()
    for line in (bank/"passenger-requests.jsonl").read_bytes().splitlines():
        q=json.loads(line);scene=q["scene"];f=scene["frame"];key=scene["provenance"]["image_sha256"]
        if key not in allowed or key in seen:continue
        seen.add(key)
        photo=(bank/(key+".jpg")).read_bytes()
        if sha256(photo).hexdigest()!=key:raise ValueError("Incorrect original")
        im=Image.open(BytesIO(photo)).convert("RGB");s=manual_surfaces.get_packet(key)
        if s is None:continue
        context={"size":[im.width,im.height],"image_sha256":key,
                 "current_boxes":[t["bbox_xyxy"] for t in scene["tracks"] if t["frame_id"]==f["frame_id"]]}
        classes=manual_surfaces.decode(s,f,key)
        options=stalls.proposals(im,classes,context)
        if not options:continue
        dest=cache/(key+".json")
        try:
            packet=json.loads(dest.read_bytes())
            stalls.approved(im,classes,context,s,packet)
        except (OSError, TypeError, KeyError, ValueError):
            packet=worker.check.remote(photo,s,context,options)
            stalls.approved(im,classes,context,s,packet)
        raw=signed_bytes(packet)
        temp=dest.with_suffix(".tmp");temp.write_bytes(raw);os.replace(temp,dest)
        records[key]=raw
        print(f"Prepared {len(seen)}/{len(allowed)} public photos, {len(options)} candidate regions on this photo.",flush=True)
    target=ROOT/"experiments/dropoff/passenger-stall-reviews-v1.zip"
    temp=target.with_suffix(".tmp")
    with ZipFile(temp,"w",ZIP_DEFLATED) as bundle:
        for key,data in sorted(records.items()):bundle.writestr(key+".json",data)
    os.replace(temp,target)
    print(f"Checked {len(seen)} photos. {len(records)} have reviewable regions.",flush=True)
