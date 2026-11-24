"""Read prominent vehicle identities independently of curb geometry on registered public photos."""
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import modal
from tools import passenger_qwen_point_protocol as protocol
from tools.passenger_roadside35 import image
from tools.passenger_qwen35_download import MODEL_FOLDER,REPO,REVISION,VOLUME

app=modal.App("trackrag-public-vehicle-reference")
if modal.is_local():
    source=Path(__file__).resolve().parents[1]/"trackrag/dropoff/vehicle_ground_view.py"
    image=image.add_local_file(source,"/root/trackrag/dropoff/vehicle_ground_view.py")


@app.function(image=image,gpu="H100",cpu=4,memory=55000,max_containers=1,timeout=1100,retries=0,
              single_use_containers=True,volumes={"/weights":modal.Volume.from_name(VOLUME).read_only()},
              secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
def inspect(rows):
    import subprocess
    import tempfile
    from PIL import Image
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from trackrag.dropoff import vehicle_ground_view as view
    known=set(protocol.read_allowlist("/trial-public").values())
    if not 0<len(rows)<=len(known):raise ValueError("Unexpected public photo set")
    signer=Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ.pop("PARKING_SIGNING_KEY"),validate=True))
    folder=Path("/weights")/MODEL_FOLDER
    manifest=json.loads((folder/"trial-download-verification.json").read_bytes())
    if (manifest["repository"],manifest["revision"],manifest["shards"])!=(REPO,REVISION,42):
        raise ValueError("Unexpected vehicle reader checkpoint")
    boards=[]; bodies={}
    for row in rows:
        original=row["photo"]; key=sha256(original).hexdigest()
        specs=row["specs"]
        if key not in known or key in bodies or not 1<=len(specs)<=4:raise ValueError("Unexpected source or vehicle count")
        with Image.open(BytesIO(original)) as im:
            montage=view.render(im,specs)
        boards.append(base64.b64encode(montage).decode("ascii"))
        bodies[key]={"recipe":view.RECIPE,"source":key,"model":REPO,"revision":REVISION,"spec":view.checksum(specs),
                     "view":sha256(montage).hexdigest(),"prompt":view.checksum(view.GUIDE),"count":len(specs)}
    with tempfile.TemporaryDirectory() as directory:
        work=Path(directory);inputs=work/"inputs.json";out=work/"replies.json";guide=work/"guide"
        inputs.write_text(json.dumps(boards));guide.write_text(view.GUIDE)
        subprocess.run(["/usr/bin/python3.12","-u","/root/tools/passenger_roadside35_native.py",
                        str(inputs),str(out),str(folder),str(guide)],check=True,timeout=980)
        replies=json.loads(out.read_bytes())
    results={}
    for (key,body),reply in zip(bodies.items(),replies,strict=True):
        body["answers"]=view.parse(reply,body["count"])
        data=json.dumps(body,ensure_ascii=True,allow_nan=False,sort_keys=True,separators=(",",":")).encode()
        results[key]={"body":body,"signature":signer.sign(data).hex()}
    return results


@app.local_entrypoint()
def main():
    import cv2
    from trackrag.dropoff import manual_surfaces, vehicle_curb_ground, vehicle_ground_view as view
    from trackrag.dropoff.parking_evidence import _authenticate
    from tools.passenger_release import ROOT,PUBLIC_PHOTO_LISTS
    cv2.setNumThreads(2)
    bank=ROOT/"data/uae-parking-study"
    allowed={p["sha256"] for f in PUBLIC_PHOTO_LISTS for p in json.loads((ROOT/"experiments/dropoff"/f).read_bytes())["photos"]}
    jobs=[];specs_by_key={}
    for line in (bank/"passenger-requests.jsonl").read_bytes().splitlines():
        scene=json.loads(line)["scene"];key=scene["provenance"]["image_sha256"]
        if key not in allowed or scene["provenance"].get("network_policy")=="local_only":continue
        packet=manual_surfaces.get_packet(key)
        if not packet:continue
        specs=json.loads(json.dumps(vehicle_curb_ground.describe(scene,manual_surfaces.decode(packet,scene["frame"],key)),
                                    allow_nan=False))
        if not specs:continue
        photo=(bank/(key+".jpg")).read_bytes()
        if sha256(photo).hexdigest()!=key or key in specs_by_key:raise ValueError("Unexpected source photo")
        jobs.append({"photo":photo,"specs":specs});specs_by_key[key]=specs
    print(f"Reading {len(jobs)} original photographs and {sum(map(len,specs_by_key.values()))} vehicles.",flush=True)
    packets=inspect.remote(jobs)
    if packets.keys()!=specs_by_key.keys():raise ValueError("Vehicle source coverage differs")
    target=ROOT/"experiments/dropoff/passenger-vehicle-reference-views-v1.zip"
    temp=target.with_suffix(".pending")
    from PIL import Image
    with ZipFile(temp,"w",ZIP_DEFLATED) as archive:
        for key,packet in sorted(packets.items()):
            _authenticate(packet)
            specs=specs_by_key[key];body=packet["body"]
            with Image.open(bank/(key+".jpg")) as im: expected=sha256(view.render(im,specs)).hexdigest()
            if body["spec"]!=view.checksum(specs) or body["view"]!=expected or len(body["answers"])!=len(specs):
                raise ValueError("Vehicle observations do not match the photographed candidates")
            archive.writestr(key+".json",json.dumps(packet,separators=(",",":")))
    temp.replace(target)
    print(f"Verified {len(packets)} original vehicle inspections in {target.name}.",flush=True)
