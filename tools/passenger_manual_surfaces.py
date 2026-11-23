"""Prepare photo-bound Mapillary parking classes for manual review of the published UAE photo bank."""
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import modal
from tools.passenger_place_modal import image

app=modal.App("trackrag-public-manual-parking-surface-study")
if modal.is_local():
    from tools.passenger_release import ROOT,PUBLIC_PHOTO_LISTS
    image=image.add_local_file(ROOT/"tools/passenger_place_modal.py","/root/tools/passenger_place_modal.py")
    for name in PUBLIC_PHOTO_LISTS:
        image=image.add_local_file(ROOT/"experiments/dropoff"/name,"/sources/"+name)


@app.cls(image=image,gpu="A10G",cpu=4,memory=10000,timeout=180,startup_timeout=240,
         min_containers=0,max_containers=1,scaledown_window=15,retries=0,
         volumes={"/parking":modal.Volume.from_name("trackrag-parking-vistas-weights").read_only()},
         secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
class PublicSurfaces:
    @modal.enter()
    def load(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from tools.passenger_parking_study import ParkingSegmenter, SEG_REVISION
        self.known={p["sha256"] for f in Path("/sources").glob("*.json") for p in json.loads(f.read_bytes())["photos"]}
        self.model=ParkingSegmenter(Path("/parking")/SEG_REVISION)
        self.signer=Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ.pop("PARKING_SIGNING_KEY"),validate=True))

    @modal.method()
    def segment(self, photos):
        from PIL import Image
        from trackrag.dropoff.manual_surfaces import RECIPE,REVISION
        from trackrag.dropoff.parking_evidence import MAX_IMAGE_BYTES,MAX_IMAGE_PIXELS,map_shape,signed_bytes
        if not isinstance(photos,list) or not 1<=len(photos)<=6:
            raise ValueError("Use up to six published originals per batch")
        packets=[]
        for photo in photos:
            if not isinstance(photo,bytes) or not 0<len(photo)<=MAX_IMAGE_BYTES:
                raise ValueError("Invalid source size")
            key=sha256(photo).hexdigest()
            if key not in self.known:
                raise ValueError("This image is outside the published public photo bank")
            with Image.open(BytesIO(photo)) as original:
                if original.width*original.height>MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the supported dimensions")
                frame={"width":original.width,"height":original.height}
                classes,_=self.model.predict(original.convert("RGB"),map_shape(frame))
            buf=BytesIO()
            Image.fromarray(classes).save(buf,format="PNG")
            data=buf.getvalue()
            body={"recipe":RECIPE,"model_revision":REVISION,"image_sha256":key,"size":[frame["width"],frame["height"]],
                  "class_map":{"sha256":sha256(data).hexdigest(),"png":base64.b64encode(data).decode("ascii")}}
            packets.append({"body":body,"signature":self.signer.sign(signed_bytes(body)).hex()})
        return packets


@app.local_entrypoint()
def main():
    from tools.passenger_release import ROOT,PUBLIC_PHOTO_LISTS
    from trackrag.dropoff import manual_surfaces as maps
    from trackrag.dropoff.parking_evidence import signed_bytes
    photos=[p for name in PUBLIC_PHOTO_LISTS for p in json.loads((ROOT/"experiments/dropoff"/name).read_bytes())["photos"]]
    folder=ROOT/"data/uae-parking-study"
    keys=sorted({p["sha256"] for p in photos})
    missing=[]
    known={}
    for key in keys:
        photo=folder/(key+".jpg")
        if sha256(photo.read_bytes()).hexdigest()!=key:
            raise ValueError("A selected public original differs from its source manifest")
        prior=maps.get_packet(key)
        if prior:
            width,height=prior["body"]["size"]
            maps.decode(prior,{"width":width,"height":height},key)
            known[key]=prior
        else:
            missing.append(key)
    worker=PublicSurfaces()
    for i in range(0,len(missing),6):
        batch=missing[i:i+6]
        packets=worker.segment.remote([(folder/(key+".jpg")).read_bytes() for key in batch])
        for key,packet in zip(batch,packets,strict=True):
            if packet["body"]["image_sha256"]!=key:
                raise ValueError("The service returned a different original")
            maps.save(packet)
            known[key]=packet
        print(f"{len(known)}/{len(keys)} public originals have verified semantic maps",flush=True)
    if set(known)!=set(keys):
        raise ValueError("Missing source photographs")
    partial=maps.BANK.with_suffix(".zip.tmp")
    with ZipFile(partial,"w",ZIP_DEFLATED) as archive:
        for key in keys:
            archive.writestr(key+".json",signed_bytes(known[key]))
    os.replace(partial,maps.BANK)
    print(f"Saved {len(keys)} signed maps, {maps.BANK.stat().st_size} bytes",flush=True)
